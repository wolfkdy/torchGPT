import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F


@dataclass
class HParams:
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int


def linear_from_params(params):
    """Build nn.Linear from a {"w": ..., "b": ...} leaf of the params dict.

    GPT-2 checkpoints store linear weights in Conv1D layout (in_features,
    out_features), applied as x @ w + b. nn.Linear.weight is (out, in), so
    copy the transpose.
    """
    # pop so the checkpoint copy is freed as soon as this function returns;
    # otherwise params + model coexist and peak memory is 2x the weights
    w, b = params.pop("w"), params.pop("b")
    linear = nn.Linear(w.shape[0], w.shape[1])
    with torch.no_grad():
        linear.weight.copy_(w.T)
        linear.bias.copy_(b)
    return linear


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.c_fc = linear_from_params(params["c_fc"])      # n_embd -> 4*n_embd
        self.gelu = nn.GELU(approximate="tanh")             # GPT-2 uses the tanh approximation
        self.c_proj = linear_from_params(params["c_proj"])  # 4*n_embd -> n_embd

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, params):
        super().__init__()
        self.weight = nn.Parameter(params["g"])  # gamma (scale)
        self.bias = nn.Parameter(params["b"])    # beta (offset)

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Attn(nn.Module):
    def __init__(self, params, n_head, mask):
        super().__init__()
        self.n_head = n_head
        self.c_attn = linear_from_params(params["c_attn"])  # [n_embd, 3*n_embd] wq|wk|wv
        self.c_proj = linear_from_params(params["c_proj"])  # [n_embd, n_embd] wo
        # buffer so it follows .to(device/dtype); persistent=False keeps it out of state_dict
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, attn_mask=None):
        B, L, n_embd = x.size()
        # attn_mask.shape  [B, L]
        hs = n_embd//self.n_head
        # [B, L, n_embd]@[n_embd, 3*n_embd] = [B, L, 3*n_embd]
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        q = q.reshape(B, L, self.n_head, hs).transpose(1, 2) # [B, n_head, L, hs]
        k = k.reshape(B, L, self.n_head, hs).transpose(1, 2) # [B, n_head, L, hs]
        v = v.reshape(B, L, self.n_head, hs).transpose(1, 2) # [B, n_head, L, hs]

        # [B, n_head, L, hs]@[B, n_head, hs, L] = [B, n_head, L, L]
        attn = q@k.transpose(-1, -2)
        # causal [L, L], and attn_mask [B, L] additionally hides pad keys:
        # reshape to [B, 1, 1, L] so it broadcasts over heads and queries
        mask = self.mask[:L, :L]
        if attn_mask is not None:
            mask = mask & attn_mask[:, None, None, :]
        # finfo.min instead of -inf: a pad query's row is fully masked, and
        # softmax over a row of -inf gives NaN, which then leaks everywhere
        attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn/math.sqrt(hs), dim=-1)

        # [B, n_head, L, L]@[B, nhead, L, hs] = [B, n_head, L, hs]
        o = attn@v

        o = o.transpose(1, 2).reshape(B, L, n_embd)
        return self.c_proj(o)

class Block(nn.Module):
    def __init__(self, params, n_head, mask):
        super().__init__()
        self.ln_1 = LayerNorm(params["ln_1"])
        self.attn = Attn(params["attn"], n_head, mask)
        self.ln_2 = LayerNorm(params["ln_2"])
        self.mlp = MLP(params["mlp"])

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, hparams, params):
        super().__init__()
        self.n_embd = hparams.n_embd
        self.n_head = hparams.n_head
        self.n_layer = hparams.n_layer
        self.wte = nn.Embedding.from_pretrained(params["wte"])  # (n_vocab, n_embd)
        self.wpe = nn.Embedding.from_pretrained(params["wpe"])  # (n_ctx, n_embd)
        mask = torch.tril(torch.ones(hparams.n_ctx, hparams.n_ctx, dtype=torch.bool))
        self.blocks = nn.ModuleList(
            Block(b, hparams.n_head, mask) for b in params["blocks"]
        )
        self.ln_f = LayerNorm(params["ln_f"])

    def forward(self, input_ids, attn_mask=None):
        B, L = input_ids.shape
        if attn_mask is None:
            pos = torch.arange(0, L, dtype=torch.long, device=input_ids.device)
        else:
            # left padding shifts real tokens right, so positions can't just be
            # arange: cumsum over the mask restarts each row at 0 on its first
            # real token; pad positions clamp to 0 (hidden by the mask anyway)
            # [B, L]
            pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)

        # B L n_embd
        pos_embd = self.wpe(pos)
        tok_embd = self.wte(input_ids)
        x = pos_embd + tok_embd

        for block in self.blocks:
            x = block(x, attn_mask)

        # [B L n_embd]
        x = self.ln_f(x)

        # only the last position's prediction is needed for generation,
        # so slice before the big vocab projection
        # [B, n_embd]@[n_embd, n_vocab] = [B, n_vocab]
        logits = x[:, -1, :]@self.wte.weight.transpose(-1, -2)
        return logits

def main(prompt: str | list[str], n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models",
         temperature: float = 0.8, top_k: int = 40):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    hparams = HParams(**hparams)

    gpt = GPT(hparams, params)

    if isinstance(prompt, str):
        prompt = [prompt]

    # encode the input strings using the BPE tokenizer
    ids_list = []
    for p in prompt:
        print("input prompt: ", p)
        ids = encoder.encode(p)
        # make sure we are not surpassing the max sequence length of our model
        assert len(ids) + n_tokens_to_generate < hparams.n_ctx
        ids_list.append(ids)

    # left-pad shorter prompts so every row's last position is a real token
    # and logits taken at -1 are valid for the whole batch; the pad id itself
    # is arbitrary (eos here) because attn_mask hides those positions anyway
    max_len = max(len(ids) for ids in ids_list)
    pad_id = encoder.encoder["<|endoftext|>"]
    n_pads = [max_len - len(ids) for ids in ids_list]
    input_ids = torch.tensor(
        [[pad_id] * n + ids for n, ids in zip(n_pads, ids_list)], dtype=torch.long)
    # [B, L] bool, True = real token, False = pad
    attn_mask = torch.tensor(
        [[False] * n + [True] * len(ids) for n, ids in zip(n_pads, ids_list)])

    with torch.no_grad():
        for _ in tqdm(range(n_tokens_to_generate), "generating"):
            # [B, n_vocab]
            logits = gpt(input_ids, attn_mask)
            # temperature < 1 sharpens the distribution, > 1 flattens it
            logits = logits / temperature
            # keep only the top_k most likely tokens; sampling from the
            # renormalized top slice cuts off the long tail of garbage
            # [B, top_k] values and their vocab ids
            topk_logits, topk_ids = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            # [B, 1] index into the top_k slice, mapped back to vocab ids
            sampled = torch.multinomial(probs, num_samples=1)
            next_ids = topk_ids.gather(-1, sampled)
            # generated tokens are real tokens in every row: grow ids and mask together
            input_ids = torch.cat([input_ids, next_ids], dim=1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones_like(next_ids, dtype=torch.bool)], dim=1)

    for n, ids in zip(n_pads, input_ids.tolist()):
        print("output: ", encoder.decode(ids[n:]))
