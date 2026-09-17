#!/usr/bin/env python3
"""Build the W8A8 INT8 Standard checkpoint that ComfyUI loads.

Mirrors `script/sample/standard/int8/ref2v.sh` (the release's `--precision int8`
path). That path is *runtime* quantization: `model/int8.py:quantize_transformer`
wraps the loaded transformer in place with `protected_edges=1`, so blocks 1..48
carry `Int8Linear` while block 0 and the last block keep their BF16 projections,
and the token refiner, adaLN and the IO projections are never quantized either.
Each `Int8Linear` stores per-output-channel weight scales and quantizes
activations per token at run time, accumulating in INT32.

This script reproduces that module set and that contract, and emits it in
ComfyUI's on-disk form: `weight` (int8 `[out, in]`), `weight_scale` (fp32
`[out, 1]`) and a `comfy_quant` blob holding `{"format": "int8_tensorwise"}`.
Everything else is carried over untouched from the BF16 ComfyUI checkpoint, so
the two files differ only in the projections the release quantizes.

Known deviation from the release's runtime path: `model/int8.py` lifts each
weight to FP32 before computing the scale, this script repeats the arithmetic in
BF16 so the exported rows follow the checkpoint's own precision. The scales
therefore differ by up to 3.9e-3 relative (~5-6% of the int8 entries land one
unit apart in a spot check of the qkv projections), which is a deliberate
choice -- the artifact shipped alongside these workflows was produced this way,
and the difference is far below the trajectory shift that W8A8 causes anyway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_DIR = Path(os.environ.get("LYNNREAL_DIFFUSION_MODELS", "models/diffusion_models"))
DEFAULT_SRC = DEFAULT_DIR / "lynnreal_omni_standard_bf16.safetensors"
DEFAULT_DST = DEFAULT_DIR / "lynnreal_omni_standard_int8.safetensors"

# The four projections `quantize_transformer` replaces in every quantized block.
QUANT_TAILS = ("attn.qkv_proj.weight", "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")
COMFY_QUANT = json.dumps({"format": "int8_tensorwise"})
SCALE_EPS = 1e-8


def block_index(key: str) -> int | None:
    if not key.startswith("blocks."):
        return None
    head, _, _ = key.partition(".")
    rest = key[len(head) + 1 :]
    index, _, _ = rest.partition(".")
    return int(index) if index.isdigit() else None


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Row-wise int8 quantization with the release's contract, in BF16.

    Same shape (per-output-channel scale = max|row| / 127, round to nearest-even,
    clamp to +-127) as `model/int8.py:Int8Linear`, but the arithmetic runs in the
    checkpoint's dtype instead of the release's FP32 lift -- see the module
    docstring for the measured consequence.
    """
    scale = weight.abs().amax(1, keepdim=True).clamp_min_(SCALE_EPS) / 127
    quantized = (weight / scale).round().clamp(-127, 127).to(torch.int8)
    error = (quantized.float() * scale.float() - weight.float()).abs().max().item()
    return quantized.contiguous(), scale.float().contiguous(), error


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                return digest.hexdigest()
            digest.update(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--protected-edges", type=int, default=1)
    parser.add_argument("--stats", type=Path, default=None, help="where to write the JSON report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dst.exists() and not args.overwrite:
        raise SystemExit("refusing to overwrite {}".format(args.dst))
    if not args.src.exists():
        raise SystemExit("missing source {}".format(args.src))

    started = time.perf_counter()
    converted: dict[str, torch.Tensor] = {}
    quantized_targets: list[str] = []
    errors: list[float] = []

    with safe_open(str(args.src), framework="pt", device="cpu") as source:
        keys = list(source.keys())
        blocks = sorted({index for key in keys if (index := block_index(key)) is not None})
        if not blocks:
            raise SystemExit("no transformer blocks found in {}".format(args.src))
        last = blocks[-1]
        quantized_blocks = [index for index in blocks if args.protected_edges <= index <= last - args.protected_edges]
        expected = len(quantized_blocks) * len(QUANT_TAILS)
        print(
            "source {} tensors, {} blocks; quantizing {} of them with protected edges {}".format(
                len(keys), len(blocks), len(quantized_blocks), args.protected_edges
            ),
            flush=True,
        )

        wanted = {"blocks.{}.{}".format(index, tail) for index in quantized_blocks for tail in QUANT_TAILS}
        found = {key for key in keys if key in wanted}
        if found != wanted:
            raise SystemExit("checkpoint is missing {}".format(sorted(wanted - found)[:5]))

        for number, key in enumerate(keys, 1):
            tensor = source.get_tensor(key)
            if key in wanted:
                if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
                    raise SystemExit("{} is {} {} , expected a 2-D BF16 weight".format(key, tensor.dtype, tuple(tensor.shape)))
                weight, scale, error = quantize(tensor)
                prefix = key[: -len(".weight")]
                converted[prefix + ".weight"] = weight
                converted[prefix + ".weight_scale"] = scale
                # One blob per layer: safetensors rejects dicts whose entries share storage.
                converted[prefix + ".comfy_quant"] = torch.tensor(
                    list(COMFY_QUANT.encode("utf-8")), dtype=torch.uint8
                )
                quantized_targets.append(prefix)
                errors.append(error)
            else:
                converted[key] = tensor
            del tensor
            if number % 40 == 0 or number == len(keys):
                print(
                    "  [{:3d}/{:3d}] {} tensors staged, {:.0f}s".format(
                        number, len(keys), len(converted), time.perf_counter() - started
                    ),
                    flush=True,
                )

    if len(quantized_targets) != expected:
        raise SystemExit("quantized {} layers, expected {}".format(len(quantized_targets), expected))

    errors.sort()
    stats = {
        "source": str(args.src),
        "destination": str(args.dst),
        "protected_edges": args.protected_edges,
        "quantized_blocks": [quantized_blocks[0], quantized_blocks[-1]],
        "quantized_blocks_count": len(quantized_blocks),
        "quantized_layers": len(quantized_targets),
        "quantized_layer_names": sorted(quantized_targets),
        "max_abs_dequantization_error": errors[-1],
        "median_abs_dequantization_error": statistics.median(errors),
        "tensors_total": len(converted),
        "payload_bytes": sum(t.numel() * t.element_size() for t in converted.values()),
    }

    print(
        "payload {:.2f} GB | max |x - dequant(q(x))| {:.5f} | median {:.6f}".format(
            stats["payload_bytes"] / 1e9, stats["max_abs_dequantization_error"], stats["median_abs_dequantization_error"]
        ),
        flush=True,
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    print("writing {}".format(args.dst), flush=True)
    save_file(converted, str(args.dst), metadata={"format": "pt"})
    stats["file_bytes"] = args.dst.stat().st_size
    stats["sha256"] = sha256(args.dst)
    stats["elapsed_seconds"] = time.perf_counter() - started
    print("wrote {:.2f} GB in {:.0f}s".format(stats["file_bytes"] / 1e9, stats["elapsed_seconds"]), flush=True)

    # Reload what was written and confirm the contract ComfyUI will read.
    with safe_open(str(args.dst), framework="pt", device="cpu") as check:
        check_keys = set(check.keys())
        if check_keys != set(converted):
            raise SystemExit("round-trip key mismatch")
        for target in quantized_targets[:1] + quantized_targets[-1:]:
            weight = check.get_slice(target + ".weight")
            scale = check.get_slice(target + ".weight_scale")
            blob = bytes(check.get_tensor(target + ".comfy_quant").tolist()).decode("utf-8")
            if weight.get_dtype() != "I8" or scale.get_dtype() != "F32":
                raise SystemExit("{} has dtypes {} / {}".format(target, weight.get_dtype(), scale.get_dtype()))
            if scale.get_shape() != (weight.get_shape()[0], 1):
                raise SystemExit("{} scale shape {}".format(target, scale.get_shape()))
            if blob != COMFY_QUANT:
                raise SystemExit("{} comfy_quant blob is {!r}".format(target, blob))
        for index in (quantized_blocks[0] - 1, quantized_blocks[-1] + 1):
            key = "blocks.{}.attn.qkv_proj.weight".format(index)
            if check.get_slice(key).get_dtype() != "BF16":
                raise SystemExit("protected block {} was quantized".format(index))
    print(
        "round-trip verified: {} tensors, {} int8 layers, protected blocks".format(
            len(check_keys), len(quantized_targets)
        ),
        flush=True,
    )

    if args.stats is not None:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(stats, indent=2) + "\n")
        print("stats written to {}".format(args.stats), flush=True)
    print("DONE {}".format(args.dst), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
