"""Load HF-sharded, trained Flash W8A8 tensors without requantization."""
import json
import torch
from safetensors.torch import load_file
from .int8 import Int8Linear
from .weights import local_weights, sha256


def load_transformer(folder, device="cuda", gemm="triton"):
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel
    folder = local_weights(folder)
    checksums = json.loads((folder / "checksums.json").read_text())
    for name in ("config.json", "quantization_config.json", "diffusion_pytorch_model.safetensors.index.json"):
        if sha256(folder / name) != checksums[name]["sha256"]:
            raise ValueError("Flash metadata checksum mismatch: " + name)
    config = json.loads((folder / "config.json").read_text())
    quant = json.loads((folder / "quantization_config.json").read_text())
    index = json.loads((folder / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    remaining = set(model.state_dict())
    pending = {}
    for filename in sorted(set(index.values())):
        path = folder / filename
        if path.stat().st_size != checksums[filename]["bytes"] or sha256(path) != checksums[filename]["sha256"]:
            raise ValueError("Flash shard checksum mismatch: " + filename)
        tensors = load_file(str(path), device=device)
        if set(tensors) != {k for k, v in index.items() if v == filename}:
            raise ValueError("Flash shard index mismatch")
        pending.update(tensors)
    for name in quant["quantized_modules"]:
        weight = pending.pop(name + ".weight_int8")
        scale = pending.pop(name + ".weight_scale")
        bias = pending.pop(name + ".bias", None)
        layer = Int8Linear.from_packed(weight, scale, bias, gemm)
        parent, _, key = name.rpartition(".")
        setattr(model.get_submodule(parent), key, layer)
        remaining.difference_update({name + ".weight", name + ".bias"})
    if set(pending) != remaining:
        raise ValueError("Incomplete Flash model: " + str(set(pending) ^ remaining))
    result = model.load_state_dict(pending, strict=False, assign=True)
    if result.unexpected_keys:
        raise ValueError(result.unexpected_keys)
    del pending, tensors
    # Concatenate integer rows and their original scales, as in native inference.
    for block in model.transformer_blocks:
        attn = block.attn
        parts = [attn.to_q, attn.to_k, attn.to_v]
        bias = None if all(p.bias is None for p in parts) else torch.cat([
            p.bias if p.bias is not None else p.weight.new_zeros(p.out_features) for p in parts])
        attn.to_qkv = Int8Linear.from_packed(torch.cat([p.weight_int8.t() for p in parts]),
            torch.cat([p.weight_scale.t() for p in parts]), bias, gemm)
        attn.to_q = attn.to_k = attn.to_v = None
        attn.fused_projections = True
    return model.to(device).eval().requires_grad_(False)
