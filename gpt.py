import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
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
    w, b = params["w"], params["b"]
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
        # TODO: x -> c_fc -> gelu -> c_proj
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
    def __init__(self, params, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_attn = linear_from_params(params["c_attn"])  # [n_embd, 3*n_embd] wq|wk|wv
        self.c_proj = linear_from_params(params["c_proj"])  # [n_embd, n_embd] wo

    def forward(self, x):
        B, L, n_embd = x.size()
        hs = n_embd//self.n_head
        # [B, L, n_embd]@[n_embd, 3*n_embd] = [B, L, 3*n_embd]
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        q = q.reshape(B, L, self.n_head, hs).transpose(1, 2) # [B, n_head, L, hs]
        k = k.reshape(B, L, self.n_head, hs).transpose(1, 2) # [B, n_head, L, hs]
        v = v.reshape(B, L, self.n_head, hs).transpose(1, 2) # [B, n_head, L, hs]

        # [B, n_head, L, hs]@[B, n_head, hs, L] = [B, n_head, L, L]
        attn = q@k.transpose(-1, -2)
        attn = F.softmax(attn/math.sqrt(hs), dim=-1)

        # [B, n_head, L, L]@[B, nhead, L, hs] = [B, n_head, L, hs]
        o = attn@v

        o = o.transpose(1, 2).reshape(B, L, n_embd)
        return self.c_proj(o)

class Block(nn.Module):
    def __init__(self, params, n_head):
        super().__init__()
        self.ln_1 = LayerNorm(params["ln_1"])
        self.attn = Attn(params["attn"], n_head)
        self.ln_2 = LayerNorm(params["ln_2"])
        self.mlp = MLP(params["mlp"])

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
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
        self.blocks = nn.ModuleList(
            Block(b, hparams.n_head) for b in params["blocks"]
        )
        self.ln_f = LayerNorm(params["ln_f"])

    def forward(self, input_ids):
        # TODO: input_ids is (B, T) of token ids
        #   x = token embeddings + position embeddings (positions 0..T-1)
        #   run through self.blocks, then self.ln_f
        #   logits = x @ self.wte.weight.T  (no output head: weights tied with wte)
        raise NotImplementedError


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    hparams = HParams(**hparams)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams.n_ctx

