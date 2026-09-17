"""Take comfy-kitchen's per-shape INT8 autotune out of the request path.

``comfy_kitchen/backends/triton/quantization.py`` autotunes both of its INT8 matmul kernels
with ``key=['m', 'n', 'k']`` and without ``cache_results``. ``m`` is the packed row count --
text tokens plus video rows plus audio rows -- so editing the prompt changes it, and the next
request re-benchmarks six configs per projection before its first denoising step can run. On
the H3 shapes that is 1.0-2.1 s per projection (tools/probe_autotune.py), and the Flash graphs
reach six distinct ``(m, n, k)`` keys, which is where the whole stall comes from: a 5 s ref2v
run spends 13.9 s in step 1 and 2.6 s in each of the other two, for identical rows, and a
prompt edit costs ~10 s on a 10 s clip.

The benchmark is also pointless here. Measuring the six configs on the four H3 projections at
row counts from 11k to 109k (tools/probe_configs.py) shows the autotuner's own first config --
``block_m=128, block_n=256, block_k=64, warps=8, stages=3`` -- is **twice as fast as the
runner-up everywhere**, and it is what the autotuner returns for all four projections anyway.
Pinning it makes Triton take the single config directly, so ``Autotuner.run`` never calls
``do_bench``; measured against a pristine ``TRITON_CACHE_DIR``, the first call at any packed
row count then costs exactly one kernel launch -- this Triton does not re-specialise the
kernels per ``m``, so a new prompt needs no recompile either.

The pin is only applied on the device it was measured on (Hopper, ``sm_90``). Other
architectures keep comfy-kitchen's autotuner, because a tile size that wins by 2x on H100 need
not win on a part with different shared-memory and SM counts. ``LYNNREAL_INT8_PIN=force`` pins
anyway, ``LYNNREAL_INT8_PIN=0`` never pins, and no pin is applied when the CUDA backend serves
``int8_linear`` (it has no autotuner to begin with).
"""

from __future__ import annotations

import logging
import os

import torch

# The measured winner on the H3 projections, and comfy-kitchen's own first config.
BEST_CONFIG = {"block_m": 128, "block_n": 256, "block_k": 64}

_INSTALLED = False
_STATS = {"calls": 0}


def enabled() -> str:
    """``force``, ``auto`` (Hopper only) or ``off``."""
    return os.environ.get("LYNNREAL_INT8_PIN", "auto").lower()


def install() -> None:
    """Pin the Triton INT8 config list to the measured winner."""
    global _INSTALLED
    mode = enabled()
    if _INSTALLED:
        return
    if mode == "0" or mode == "off":
        logging.info("LynnReal: comfy-kitchen's INT8 autotuner left in charge "
                     "(LYNNREAL_INT8_PIN=0).")
        _INSTALLED = True
        return
    if mode == "auto" and not _measured_here():
        logging.info("LynnReal: comfy-kitchen's INT8 autotuner left in charge; the pinned "
                     "config was measured on Hopper (sm_90) and this device is %s. Set "
                     "LYNNREAL_INT8_PIN=force to pin anyway.", _device_name())
        _INSTALLED = True
        return
    _INSTALLED = True

    try:
        import comfy_kitchen as ck
    except Exception as error:  # pragma: no cover - depends on the install
        logging.info("LynnReal: comfy-kitchen unavailable (%s); INT8 autotune untouched.",
                     error)
        return

    if not _pin(ck):
        return
    _wrap(ck)


def _measured_here() -> bool:
    try:
        return torch.cuda.get_device_capability()[0] == 9
    except Exception:  # pragma: no cover - no usable CUDA device
        return False


def _device_name() -> str:
    try:
        return "{} (sm_{}{})".format(torch.cuda.get_device_name(),
                                     *_device_capability())
    except Exception:  # pragma: no cover - diagnostics only
        return "unknown"


def _device_capability() -> tuple:
    major, minor = torch.cuda.get_device_capability()
    return major, minor


def _pin(ck) -> bool:
    """Replace both matmul kernels' config lists with the measured winner."""
    try:
        from comfy_kitchen.backends.triton import quantization as triton_quant

        impl = ck.registry.get_implementation("int8_linear")
    except Exception as error:
        logging.info("LynnReal: comfy-kitchen INT8 autotune left alone (%s).", error)
        return False

    serving = getattr(impl, "__module__", "")
    if serving != triton_quant.__name__:
        logging.info("LynnReal: comfy-kitchen INT8 autotune left alone; int8_linear is served "
                     "by %s, whose kernels are not autotuned per shape.", serving)
        return False

    kernels = [("int8_linear", triton_quant._int8_matmul_dequant_kernel),
               ("int8_linear_per_row", triton_quant._int8_matmul_dequant_per_row_kernel)]
    chosen = []
    for name, kernel in kernels:
        matches = [config for config in kernel.configs
                   if all(config.kwargs.get(key) == value for key, value in BEST_CONFIG.items())]
        if not matches:
            logging.warning("LynnReal: comfy-kitchen's %s kernel no longer offers %s; leaving "
                            "the per-shape autotune in place.", name, BEST_CONFIG)
            return False
        chosen.append((name, kernel, matches[:1]))

    for name, kernel, config in chosen:
        kernel.configs = config
    logging.info("LynnReal: pinned comfy-kitchen's INT8 %s to %s (the autotuner's own winner, "
                 "2x the runner-up on the H3 shapes). Editing a prompt no longer re-tunes the "
                 "INT8 GEMMs.", ", ".join(name for name, _, _ in chosen),
                 ", ".join("%s=%s" % item for item in sorted(BEST_CONFIG.items())))
    return True


def _wrap(ck) -> None:
    """Count the calls so the bench line can show the pin is live."""
    original = ck.int8_linear

    def int8_linear(x, weight, weight_scale, bias=None, out_dtype=None, **kwargs):
        _STATS["calls"] += 1
        if out_dtype is None:
            return original(x, weight, weight_scale, bias=bias, **kwargs)
        return original(x, weight, weight_scale, bias=bias, out_dtype=out_dtype, **kwargs)

    int8_linear.__name__ = getattr(original, "__name__", "int8_linear")
    int8_linear.__doc__ = getattr(original, "__doc__", None)
    ck.int8_linear = int8_linear


def stats() -> dict:
    """Drain the padding counters (they show up in the bench line)."""
    out = dict(_STATS)
    for key in _STATS:
        _STATS[key] = 0
    return out
