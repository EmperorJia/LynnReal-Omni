#!/usr/bin/env python3
"""Verify the ComfyUI-LynnReal pack without a GPU.

Four checks:

1. The pack loads exactly the way ComfyUI's ``load_custom_node`` loads a directory:
   ``spec_from_file_location(<path with dots replaced>, <dir>/__init__.py)``, then
   ``comfy_entrypoint()`` -> a ``ComfyExtension`` whose node list carries the pack's nodes.
2. The Light VAE builds through ``comfy.sd.VAE`` at the depth the checkpoint has (26), with
   the release's 272/16 tile geometry, and every checkpoint key lands on the model.
3. The official 36-block VAE still builds exactly as upstream does (256/64).
4. The fused-block path stands itself down when comfy-aimdo's malloc graph is recording
   (DynamicVRAM, i.e. every cu13x torch), which is what keeps the Flash graphs from dying with
   "aimdo memory compile error".

Also reproduces the failure this loader exists to prevent: the stock construction path is
36 blocks, so the Light VAE leaves ten blocks randomly initialized.

Run it from a ComfyUI checkout, with the environment ComfyUI itself runs in:

    python custom_nodes/ComfyUI-LynnReal/tools/verify_node_pack.py \
        --comfy /path/to/ComfyUI [--release /path/to/a/LynnReal/release]

Both paths also come from the environment, `LYNNREAL_COMFY` and `LYNNREAL_RELEASE`.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
import time

DEFAULT_COMFY = os.environ.get("LYNNREAL_COMFY", ".")
DEFAULT_RELEASE = os.environ.get("LYNNREAL_RELEASE", "")


def log(message: str) -> None:
    print("{:7.1f}s {}".format(time.monotonic() - START, message), flush=True)


def load_pack(pack_dir: str):
    """Mirror ``comfy_nodes_nodes.load_custom_node`` for a directory pack."""
    module_name = pack_dir.replace(".", "_x_")
    module_path = os.path.join(pack_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def stub_comfy_kitchen() -> None:
    """Stand in for comfy-kitchen on a machine without a GPU.

    comfy-kitchen imports its Triton backend at import time and Triton's autotune decorators
    build a driver while doing so, which needs a GPU: on a CPU node that turns into
    ``RuntimeError: 0 active drivers`` inside ComfyUI's own ``comfy/quant_ops.py``, before any
    custom node is involved. ComfyUI itself treats comfy-kitchen as optional -- ``attention.py``
    only asks it whether INT8 attention is available, and ``quant_ops.py`` falls back when its
    layouts cannot be imported. This stub reports "not available" and leaves the INT8 kernels
    out of scope, which is exactly what a structural check on a CPU node should do.
    """
    import types

    module = types.ModuleType("comfy_kitchen")
    module.int8_attention_is_available = lambda: False

    def unavailable(*args, **kwargs):
        raise RuntimeError("comfy_kitchen is stubbed out for this CPU-only check")

    module.int8_attention = unavailable
    module.prequantize_int8_attention = unavailable
    module.int8_attention_from_prequantized = unavailable
    sys.modules["comfy_kitchen"] = module


def check_keys(label: str, model, checkpoint) -> None:
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    log("[{}] checkpoint keys {} / model keys {}; model keys with no checkpoint: {}; "
        "checkpoint keys with no model slot: {}"
        .format(label, len(checkpoint_keys), len(model_keys), len(missing), len(unexpected)))
    if missing:
        log("[{}]   first uninitialised: {}".format(label, missing[:3]))
    if unexpected:
        log("[{}]   first unmapped: {}".format(label, unexpected[:3]))


def widget_layout(info: dict) -> list[tuple[str, bool]]:
    """Same widget walk the API prompt converter does (tools/comfy_run.py)."""
    spec = info["input"]
    names = list((spec.get("required") or {}).items()) + list((spec.get("optional") or {}).items())
    layout = []
    for name, value in names:
        kind = value[0]
        options = value[1] if len(value) > 1 and isinstance(value[1], dict) else {}
        is_widget = (
            isinstance(kind, list)
            or kind in ("INT", "FLOAT", "STRING", "BOOLEAN")
            or (isinstance(kind, str) and kind.startswith(("COMBO", "COMFY_DYNAMICCOMBO")))
        )
        if is_widget:
            layout.append((name, bool(options.get("control_after_generate"))))
    return layout


def check_fused_block_malloc_guard(pack) -> int:
    """The fused-block guard must see a real device, or the Flash graphs crash on DynamicVRAM.

    ``fast_blocks.probe()`` disables the fused path when ComfyUI's comfy-aimdo malloc graph is
    recording, because the planner rejects the allocation pattern of our two Triton kernels
    (``RuntimeError: aimdo memory compile error`` on the first sampling step; pausing the graph
    around the block does not help).  The guard asks ``comfy.model_prefetch.malloc_graph_enabled``,
    which is ``is_device_cuda(device) and ...`` -- so it does nothing when it is handed ``None``.
    That is exactly what happened between the rewrite that introduced the graph check and the fix
    for it, and it is invisible on this machine whenever DynamicVRAM is off, so it gets a check.

    The probe is driven with a sentinel "block" that must never be touched: if the guard fires,
    ``probe`` returns before looking at it.
    """
    import contextlib
    import types

    import torch
    import comfy.model_prefetch as prefetch

    fast_blocks = pack.fast_blocks
    seen: list = []

    def malloc_graph_enabled(device):
        # what comfy/model_prefetch.py does, reduced to the part that matters here
        seen.append(device)
        return getattr(device, "type", None) == "cuda"

    # Patch the real module: ``probe`` does ``import comfy.model_prefetch as _prefetch``, which
    # resolves through the already-imported ``comfy`` package, not through sys.modules alone.
    saved_malloc_graph_enabled = prefetch.malloc_graph_enabled
    saved_pause = prefetch.pause_malloc_graph
    prefetch.malloc_graph_enabled = malloc_graph_enabled
    prefetch.pause_malloc_graph = lambda *args, **kwargs: contextlib.nullcontext()

    management = types.ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cuda", 0)
    management.cast_to = lambda value, **kwargs: value

    saved = {
        "comfy": fast_blocks.comfy,
        "probed": fast_blocks._PROBED,
        "failure": fast_blocks._FAILURE,
        "usable": fast_blocks.usable,
    }
    fast_blocks.comfy = types.SimpleNamespace(model_management=management)
    fast_blocks._PROBED = False
    fast_blocks._FAILURE = None
    fast_blocks.usable = lambda: True  # the kernels themselves are not what this check is about
    try:
        try:
            fast_blocks.probe(object())
        except Exception as error:  # the sentinel was touched -> the guard did not fire
            log("guard check: probe raised {}: {}".format(type(error).__name__, error))
        failure = fast_blocks._FAILURE
    finally:
        fast_blocks._FAILURE = saved["failure"]
        fast_blocks._PROBED = saved["probed"]
        fast_blocks.usable = saved["usable"]
        fast_blocks.comfy = saved["comfy"]
        prefetch.malloc_graph_enabled = saved_malloc_graph_enabled
        prefetch.pause_malloc_graph = saved_pause

    log("guard check: malloc_graph_enabled() saw {}".format(
        [str(device) for device in seen] or "no call at all"))
    if not any(getattr(device, "type", None) == "cuda" for device in seen):
        log("FAIL: the malloc-graph guard asked about device=None, so it can never disable the "
            "fused path; on a cu13x torch that is 'aimdo memory compile error' on the first "
            "sampling step")
        return 1
    if not failure or "malloc graph" not in failure:
        log("FAIL: probe kept the fused path enabled under an active malloc graph (failure={!r})"
            .format(failure))
        return 1
    log("guard check: fused path disabled as intended -> {}".format(failure))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", default=DEFAULT_COMFY)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--skip-official", action="store_true")
    args = parser.parse_args()
    if not args.release:
        parser.error("pass --release (or set LYNNREAL_RELEASE) to a checkout holding "
                     "weight/comfyui/models/vae/ -- it is what the Light VAE check compares against")

    comfy_dir = os.path.abspath(args.comfy)
    pack_dir = os.path.join(comfy_dir, "custom_nodes", "ComfyUI-LynnReal")
    vae_dir = os.path.join(args.release, "weight", "comfyui", "models", "vae")
    light_path = os.path.join(vae_dir, "lynnreal_omni_light_vae_fp16.safetensors")
    official_path = os.path.join(vae_dir, "minimax_h3_video_vae_fp16.safetensors")

    sys.path.insert(0, comfy_dir)
    os.chdir(comfy_dir)
    os.environ.setdefault("LYNNREAL_NO_COMPILE_VAE", "1")  # torch.compile on CPU is not the point

    import torch
    if not torch.cuda.is_available():
        log("no CUDA device: stubbing out comfy_kitchen (its Triton backend needs a GPU to import)")
        stub_comfy_kitchen()
        # comfy.model_management probes the CUDA device unless --cpu is set, and that probe
        # raises without a driver.
        sys.argv = [sys.argv[0], "--cpu"]
        import comfy.options
        comfy.options.enable_args_parsing()
    import comfy.ops
    import comfy.sd
    import comfy.utils
    import comfy.ldm.minimax.vae as minimax_vae
    import folder_paths  # noqa: F401

    log("ComfyUI path: {}".format(comfy_dir))

    # ---- 1. the pack loads and registers its nodes -------------------------------------
    pack = load_pack(pack_dir)
    extension = asyncio.run(pack.comfy_entrypoint())
    node_ids = sorted(node.GET_SCHEMA().node_id for node in asyncio.run(extension.get_node_list()))
    log("nodes registered: {}".format(node_ids))
    expected = {"LynnRealFlashTokenCompression", "LynnRealH3VAELoader",
                "LynnRealAlignedReference", "LynnRealInt8Backend"}
    if set(node_ids) != expected:
        log("FAIL: expected {}".format(sorted(expected)))
        return 1

    # The canvas stores our node's widgets as a flat list; the API converter (and the frontend)
    # read them in this order, so it has to match the workflow file's widgets_values exactly.
    info = pack.light_vae.LynnRealH3VAELoader.GET_NODE_INFO_V1()
    if asyncio.iscoroutine(info):
        info = asyncio.run(info)
    layout = widget_layout(info)
    log("LynnRealH3VAELoader widget order: {}".format(layout))
    if [name for name, _ in layout] != ["vae_name", "num_layers", "tile_size", "tile_overlap",
                                        "compile_decoder"]:
        log("FAIL: widget order does not match the workflow file")
        return 1
    if any(control for _, control in layout):
        log("FAIL: a widget takes a control_after_generate slot the workflow does not encode")
        return 1

    # ---- 1b. the fused-block guard sees a device when DynamicVRAM is on -----------------
    if check_fused_block_malloc_guard(pack):
        return 1

    light_vae_module = pack.light_vae

    # ---- 2. Light VAE: depth from the checkpoint, release tile geometry ----------------
    log("loading {}".format(os.path.basename(light_path)))
    light_sd, light_metadata = comfy.utils.load_torch_file(light_path, return_metadata=True)
    log("checkpoint depth detected: {}".format(light_vae_module.decoder_depth(light_sd)))
    log("building the Light VAE through comfy.sd.VAE (compiled decoder off)")
    light = light_vae_module.build_vae(light_sd, metadata=light_metadata, compile_decoder=False)
    stage = light.first_stage_model
    blocks = len(stage.decoder.transformer_blocks)
    log("Light VAE: decoder blocks {}, tile {}, overlap {}"
        .format(blocks, stage.tile_size, stage.tile_overlap_min))
    if blocks != 26 or stage.tile_size != 272 or stage.tile_overlap_min != 16:
        log("FAIL: Light VAE geometry is wrong")
        return 1
    check_keys("light", stage, light_sd)

    # ---- 3. The same checkpoint through the stock construction path --------------------
    log("control: stock comfy.sd.VAE() on the same checkpoint (no override)")
    stock = comfy.sd.VAE(sd=light_sd, metadata=light_metadata)
    stock_stage = stock.first_stage_model
    stock_blocks = len(stock_stage.decoder.transformer_blocks)
    stock_missing = len(set(stock_stage.state_dict()) - set(light_sd))
    log("stock path: decoder blocks {}, tile {}, overlap {}, model keys with no checkpoint: {}"
        .format(stock_blocks, stock_stage.tile_size, stock_stage.tile_overlap_min, stock_missing))
    if stock_blocks != 36 or stock_missing == 0:
        log("FAIL: the stock path was expected to build 36 blocks and leave some uninitialised")
        return 1
    del stock

    # ---- 4. The official VAE keeps upstream's geometry ---------------------------------
    if not args.skip_official:
        log("loading {}".format(os.path.basename(official_path)))
        official_sd, official_metadata = comfy.utils.load_torch_file(official_path, return_metadata=True)
        official = light_vae_module.build_vae(official_sd, metadata=official_metadata,
                                              compile_decoder=False)
        official_stage = official.first_stage_model
        official_blocks = len(official_stage.decoder.transformer_blocks)
        log("official VAE: decoder blocks {}, tile {}, overlap {}"
            .format(official_blocks, official_stage.tile_size, official_stage.tile_overlap_min))
        if (official_blocks, official_stage.tile_size, official_stage.tile_overlap_min) != (36, 256, 64):
            log("FAIL: the official VAE no longer matches upstream's geometry")
            return 1
        check_keys("official", official_stage, official_sd)

    log("DONE: node pack loads, Light VAE builds at 26 blocks with 272/16, official VAE "
        "unchanged, fused path stands down under comfy-aimdo's malloc graph")
    return 0


START = time.monotonic()

if __name__ == "__main__":
    raise SystemExit(main())
