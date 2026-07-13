"""Load GPT-2 weights into PyTorch tensors, from HuggingFace safetensors files.

Same interface as picoGPT's utils.load_encoder_hparams_and_params, except the
params dict holds torch.Tensor leaves instead of numpy arrays.

A safetensors file is self-describing:

  [8 bytes]  header length N (little-endian uint64)
  [N bytes]  JSON header: {tensor name: {dtype, shape, data_offsets [begin, end]}}
  [rest]     the raw tensor bytes, offsets are relative to the end of the header

so unlike a TF checkpoint (see utils_tf.py), parsing it needs no format
archaeology -- just json + byte slicing.
"""

import json
import os
import struct

import numpy as np
import torch

from encoder import get_encoder

# HF hosts the converted weights of OpenAI's original TF checkpoints
_HF_REPOS = {
    "124M": "gpt2",
    "355M": "gpt2-medium",
    "774M": "gpt2-large",
    "1558M": "gpt2-xl",
}

_DTYPES = {"F32": "<f4"}  # GPT-2 weights are all float32


# ---------------------------------------------------------------------------
# safetensors parsing
# ---------------------------------------------------------------------------


def read_safetensors(path):
    """Read a .safetensors file into {tensor name: numpy array}."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)

    data_start = 8 + header_len
    tensors = {}
    for name, info in header.items():
        dtype = _DTYPES[info["dtype"]]
        begin, end = info["data_offsets"]
        array = np.fromfile(
            path, dtype=dtype, count=(end - begin) // 4, offset=data_start + begin
        )
        tensors[name] = array.reshape(info["shape"])
    return tensors


# ---------------------------------------------------------------------------
# GPT-2 specific part: download + reshape into picoGPT's nested dict layout
# ---------------------------------------------------------------------------


def download_gpt2_files(model_size, model_dir):
    import requests
    from tqdm import tqdm

    base = f"https://huggingface.co/openai-community/{_HF_REPOS[model_size]}/resolve/main"
    # local name -> name in the HF repo (tokenizer files keep picoGPT's names,
    # so the same models/ dir works for both picoGPT and torchGPT)
    files = {
        "model.safetensors": "model.safetensors",
        "config.json": "config.json",
        "encoder.json": "vocab.json",
        "vocab.bpe": "merges.txt",
    }
    for local_name, remote_name in files.items():
        local_path = os.path.join(model_dir, local_name)
        if os.path.isfile(local_path):
            continue
        r = requests.get(f"{base}/{remote_name}", stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            file_size = int(r.headers["content-length"])
            with tqdm(
                ncols=100,
                desc="Fetching " + local_name,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))


def load_hparams(model_dir):
    """Map HF config.json keys to picoGPT's hparams names."""
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    return {
        "n_vocab": config["vocab_size"],
        "n_ctx": config.get("n_ctx", config["n_positions"]),
        "n_embd": config["n_embd"],
        "n_head": config["n_head"],
        "n_layer": config["n_layer"],
    }


def load_gpt2_params_from_safetensors(path, hparams):
    """Nest HF's flat tensor names into picoGPT's dict layout, as torch tensors.

    h.5.attn.c_attn.weight -> params["blocks"][5]["attn"]["c_attn"]["w"]
    h.5.ln_1.weight        -> params["blocks"][5]["ln_1"]["g"]

    Returns a nested dict of torch.float32 tensors. Shapes below are for
    124M (n_embd=768); for other sizes substitute that model's hparams:

        {
          "wte": (50257, 768),            # token embedding [n_vocab, n_embd]
          "wpe": (1024, 768),             # position embedding [n_ctx, n_embd]
          "blocks": [                     # n_layer items, one per transformer block
            {
              "ln_1": {"g": (768,), "b": (768,)},        # LN before attention
              "attn": {
                "c_attn": {"w": (768, 2304), "b": (2304,)},  # fused qkv proj [n_embd, 3*n_embd]
                "c_proj": {"w": (768, 768),  "b": (768,)},   # attn output proj
              },
              "ln_2": {"g": (768,), "b": (768,)},        # LN before mlp
              "mlp": {
                "c_fc":   {"w": (768, 3072), "b": (3072,)},  # up proj [n_embd, 4*n_embd]
                "c_proj": {"w": (3072, 768), "b": (768,)},   # down proj
              },
            },
            ...
          ],
          "ln_f": {"g": (768,), "b": (768,)},            # final LN, before logits
        }

    LayerNorm leaves are g/b (gamma=scale, beta=offset), applied as g * x + b.
    There is no separate output head: logits = final_hidden @ wte.T (weight tying).

    Note: linear weights keep the Conv1D layout, i.e. shape (in_features,
    out_features) -- computed as x @ w + b. nn.Linear.weight is the transpose.
    """

    def leaf_name(module, param):  # HF param name -> picoGPT leaf name
        if param == "bias":
            return "b"
        return "g" if module.startswith("ln") else "w"

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, array in read_safetensors(path).items():
        name = name.removeprefix("transformer.")
        if name.endswith((".attn.bias", ".attn.masked_bias")) or name == "lm_head.weight":
            continue  # causal-mask buffers / tied output head, not weights

        tensor = torch.from_numpy(array)
        parts = name.split(".")
        if parts[0] in ("wte", "wpe"):  # wte.weight / wpe.weight
            params[parts[0]] = tensor
        elif parts[0] == "ln_f":  # ln_f.weight / ln_f.bias
            params.setdefault("ln_f", {})[leaf_name("ln_f", parts[1])] = tensor
        elif parts[0] == "h":  # h.{i}.{module...}.{weight|bias}
            block, modules, param = params["blocks"][int(parts[1])], parts[2:-1], parts[-1]
            for m in modules[:-1]:
                block = block.setdefault(m, {})
            block.setdefault(modules[-1], {})[leaf_name(modules[-1], param)] = tensor
        else:
            raise ValueError(f"unexpected tensor name: {name}")
    return params


def load_encoder_hparams_and_params(model_size, models_dir):
    assert model_size in _HF_REPOS

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)
    download_gpt2_files(model_size, model_dir)  # skips files already present

    encoder = get_encoder(model_size, models_dir)
    hparams = load_hparams(model_dir)
    params = load_gpt2_params_from_safetensors(
        os.path.join(model_dir, "model.safetensors"), hparams
    )

    return encoder, hparams, params
