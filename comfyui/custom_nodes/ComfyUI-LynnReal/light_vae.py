"""Load MiniMax-H3 video VAEs at the depth the checkpoint actually has.

The release ships two video VAEs for the same latents:

* ``minimax_h3_video_vae_fp16.safetensors`` -- the official 36-block decoder.
* ``lynnreal_omni_light_vae_fp16.safetensors`` -- LynnReal's depth-distilled Light VAE
  (26 blocks), trained at its own tile geometry (``student_tile_size`` 272,
  ``student_long_axis_overlap`` 16) and decoded with ``--compile-vae`` in the release
  scripts.

ComfyUI builds the H3 video VAE with a hardcoded depth of 36. Its stock ``VAELoader`` loads
with ``strict=False``, so pointing it at the Light VAE only prints ``Missing VAE keys`` --
ten transformer blocks keep their random initialization and every frame comes out garbage.
This node builds the same ``comfy.sd.VAE`` object with the depth taken from the checkpoint,
applies the Light VAE's tile geometry and (optionally) compiles the decoder the way the
release does. Feed its VAE output into ``VAEDecode`` exactly like the stock loader.

Nothing in ``comfy/`` is modified: the depth and tile size are injected into the two
constructors for the duration of this node's own build call, and the official VAE keeps
upstream's behaviour untouched.
"""

import contextlib
import inspect
import logging
import math
import os
import threading
import types
import warnings

import torch
from typing_extensions import override

import comfy.ldm.minimax.vae as minimax_vae
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from . import attention_fa3, bench

OFFICIAL_LAYERS = 36
LIGHT_LAYERS = 26
OFFICIAL_TILE = (256, 64)
LIGHT_TILE = (272, 16)


def _tile_batch() -> int:
    """Tiles decoded per decoder call; 0 (the release default) batches a whole chunk."""
    try:
        return max(0, int(os.environ.get("LYNNREAL_VAE_TILE_BATCH", "0")))
    except ValueError:
        return 0


# `weight/light-vae/decode_config.json` -- the geometry the distilled decoder was trained at.
STUDENT_SHORT_TILE = 272
STUDENT_LONG_TILE = 208
STUDENT_SHORT_OVERLAP = 0
STUDENT_LONG_OVERLAP = 16


def _release_geometry(height, width):
    """Short/long axis tile sizes and overlaps, exactly as the release picks them."""
    if height < width:
        return (STUDENT_SHORT_TILE, STUDENT_LONG_TILE, STUDENT_SHORT_OVERLAP, STUDENT_LONG_OVERLAP)
    if height > width:
        return (STUDENT_LONG_TILE, STUDENT_SHORT_TILE, STUDENT_LONG_OVERLAP, STUDENT_SHORT_OVERLAP)
    overlap = max(STUDENT_SHORT_OVERLAP, STUDENT_LONG_OVERLAP)
    return (STUDENT_SHORT_TILE, STUDENT_SHORT_TILE, overlap, overlap)


def _release_split(length, size, overlap, ratio):
    """`model/light_vae.py::_split_tiles`: expand a tile by at most one overlap.

    The tiles tile the axis exactly (`size * count - sum(overlaps) == length`), which is what
    lets the decoder batch them without ragged shapes.
    """
    if length <= size + overlap:
        return [0], [length], []
    count = math.ceil(length / size)
    required = math.ceil((length + overlap * (count - 1)) / count / ratio) * ratio
    if required > size + overlap:
        raise ValueError("adaptive tile expansion exceeds one overlap")
    size = max(size, required)
    while size * count - overlap * (count - 1) < length:
        count += 1
    overlaps = [overlap] * (count - 1)
    remaining = size * count - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (count - 1)] += ratio
    starts = [0]
    for value in overlaps:
        starts.append(starts[-1] + size - value)
    return starts, [size] * count, overlaps


def _batched_adaptive_decode(core, z):
    """Decode one temporal chunk's spatial tiles in a single batched decoder call.

    The release's lightweight decoder sets ``tile_batch = 0`` ("batches all tiles"), which is
    what makes its decoder 1.7 s against ComfyUI's per-tile loop at 3.2 s for the same 5 s
    1344x768 clip: a 272x272x28 tile through 26 blocks is far too small to fill an H100 on its
    own. Tiles are independent (the decoder has no cross-tile attention), so batching them is
    exact and the stitch below is ComfyUI's own.
    """
    if not core.tiling or os.environ.get("LYNNREAL_NO_BATCH_VAE") == "1":
        return core.tiled_decode(z)

    height, width = z.shape[-2] * core.vae_ratio, z.shape[-1] * core.vae_ratio
    if os.environ.get("LYNNREAL_VAE_LAYOUT", "release") == "comfy":
        y_idx, y_len, y_overlap = core.split_tiles(height)
        x_idx, x_len, x_overlap = core.split_tiles(width)
    else:
        tile_h, tile_w, overlap_h, overlap_w = _release_geometry(height, width)
        y_idx, y_len, y_overlap = _release_split(height, tile_h, overlap_h, core.vae_ratio)
        x_idx, x_len, x_overlap = _release_split(width, tile_w, overlap_w, core.vae_ratio)

    tiles = []
    for i_pos, i_len in zip(y_idx, y_len):
        zi, zl = i_pos // core.vae_ratio, i_len // core.vae_ratio
        for j_pos, j_len in zip(x_idx, x_len):
            zj, zw = j_pos // core.vae_ratio, j_len // core.vae_ratio
            tiles.append(z[..., zi:zi + zl, zj:zj + zw])
    if len({tuple(tile.shape) for tile in tiles}) != 1:
        return core.tiled_decode(z)  # ragged geometry: keep ComfyUI's per-tile loop

    batch = _tile_batch() or len(tiles)
    parts = []
    debug = os.environ.get("LYNNREAL_VAE_DEBUG") == "1"
    if debug:
        import time as _time

        started = _time.perf_counter()
    for start in range(0, len(tiles), batch):
        packed = torch.cat(tiles[start:start + batch], dim=0)
        parts.extend(core._decode_pixels(packed).split(z.shape[0]))
    if debug:
        torch.cuda.synchronize()
        elapsed = _time.perf_counter() - started
        weight = next(core.decoder.parameters())
        logging.info("LynnReal: VAE chunk decode: %d tiles batched %d, in %s / weight %s, "
                     "out %s, %.0f ms", len(tiles), batch, tuple(tiles[0].shape), weight.dtype,
                     tuple(parts[0].shape), elapsed * 1000)
        _STATS["forward_ms"] = _STATS.get("forward_ms", 0.0) + elapsed * 1000
        _STATS["chunks"] = _STATS.get("chunks", 0) + 1
    parts = iter(parts)

    if debug:
        stitch_started = _time.perf_counter()
        torch.cuda.synchronize()
    canvas = None
    row_tails = []
    out_y = 0
    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
        new_tails = []
        left_tail = None
        out_x = 0
        for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
            tile = next(parts)
            if i < len(y_idx) - 1:
                new_tails.append(tile[..., -y_overlap[i]:, :].clone())
            next_left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
            if i > 0:
                tile = core.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = core.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
            left_tail = next_left_tail
            if i < len(y_idx) - 1:
                tile = tile[..., :-y_overlap[i], :]
            if j < len(x_idx) - 1:
                tile = tile[..., :, :-x_overlap[j]]
            if canvas is None:
                canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype,
                                     device=tile.device)
            canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
            out_x += tile.shape[-1]
        row_tails = new_tails
        out_y += tile.shape[-2]
    if debug:
        torch.cuda.synchronize()
        _STATS["stitch_ms"] = _STATS.get("stitch_ms", 0.0) + (_time.perf_counter() - stitch_started) * 1000
    return canvas


def _install_batched_decode(vae):
    """Point one VAE object's spatial decode at the batched implementation."""
    if os.environ.get("LYNNREAL_NO_BATCH_VAE") == "1":
        return False
    core = vae.first_stage_model
    core_any = getattr(core, "model", None)
    if core_any is not None and hasattr(core_any, "split_tiles"):
        core = core_any
    if not hasattr(core, "split_tiles"):
        return False
    core._adaptive_decode = types.MethodType(_batched_adaptive_decode, core)
    return True


@contextlib.contextmanager
def _device_intermediates(device):
    """Let ComfyUI's decoder stream into a device buffer instead of a CPU one."""
    original = comfy.model_management.intermediate_device
    comfy.model_management.intermediate_device = lambda: device
    try:
        yield
    finally:
        comfy.model_management.intermediate_device = original


def _install_device_decode(vae):
    """Keep the decoder's output buffer on the GPU for the lifetime of one decode.

    ComfyUI allocates the streaming output buffer with ``intermediate_device()``, which is the
    CPU unless the process runs with ``--gpu-only``. For a 5 s 1344x768 clip that copies ~2 GiB
    (six chunk writes of 347 MB) over PCIe inside the timed decode, which is most of the gap to
    the release's 1.7 s. The math is untouched: only the buffer's device changes.
    """
    if os.environ.get("LYNNREAL_NO_DEVICE_VAE") == "1":
        return False
    core = vae.first_stage_model
    original = core.decode

    def decode(self, z, output_buffer=None):
        global _DECODE_CALLS
        # Resolve the device per call: the loader runs before ComfyUI moves the VAE onto the
        # GPU. `comfy.sd.VAE.decode` allocates its streaming buffer from `output_device`, which
        # `comfy.sd.VAE.__init__` pinned to the CPU, and then hands it to us -- so decode into a
        # device buffer and pay one device->host copy at the end instead of one per chunk.
        device = next(self.parameters()).device
        host_buffer = output_buffer is not None and output_buffer.device != device
        if os.environ.get("LYNNREAL_VAE_DEBUG") == "1":
            logging.info("LynnReal: VAE decode call: latent %s, buffer %s, decoder %s",
                         z.device,
                         output_buffer.device if output_buffer is not None else None,
                         device)
        if os.environ.get("LYNNREAL_PROFILE") == "1":
            _DECODE_CALLS += 1
            if _DECODE_CALLS == 2:
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with _device_intermediates(device):
                        result = original(z, None if host_buffer else output_buffer)
                with open("/tmp/lynnreal_vae_profile.txt", "w", encoding="utf-8") as handle:
                    handle.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
                    handle.write("\n=== CPU ===\n")
                    handle.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
                logging.info("LynnReal: profiled one video decode -> /tmp/lynnreal_vae_profile.txt")
                return result
        if os.environ.get("LYNNREAL_VAE_DEBUG") != "1":
            with _device_intermediates(device):
                result = original(z, None if host_buffer else output_buffer)
            if host_buffer:
                output_buffer.copy_(result)
                result = output_buffer
            return result

        import time as _time

        forward_before = _STATS.get("forward_ms", 0.0)
        chunks_before = _STATS.get("chunks", 0)
        started = _time.perf_counter()
        with _device_intermediates(device):
            result = original(z, None if host_buffer else output_buffer)
        torch.cuda.synchronize()
        total = (_time.perf_counter() - started) * 1000
        forwards = _STATS.get("forward_ms", 0.0) - forward_before
        stitches = _STATS.get("stitch_ms", 0.0)
        logging.info(
            "LynnReal: VAE decode total %.0f ms | decoder forwards %.0f ms over %d chunks | "
            "tile stitching %.0f ms | temporal finalize %.0f ms", total, forwards,
            _STATS.get("chunks", 0) - chunks_before, stitches, total - forwards - stitches)
        _STATS["stitch_ms"] = 0.0
        if host_buffer:
            output_buffer.copy_(result)
            result = output_buffer
        return result

    core.decode = types.MethodType(decode, core)
    return True

# Set by the loader for the duration of one VAE construction. Per-thread so a second
# prompt on another thread can never see it.
_OVERRIDE = threading.local()
_STATS: dict = {}
_DECODE_CALLS = 0


def _pending() -> dict | None:
    return getattr(_OVERRIDE, "geometry", None)


@contextlib.contextmanager
def _geometry(num_layers: int, tile_size: int, tile_overlap_min: int):
    previous = _pending()
    _OVERRIDE.geometry = {
        "num_layers": num_layers,
        "tile_size": tile_size,
        "tile_overlap_min": tile_overlap_min,
    }
    try:
        yield
    finally:
        _OVERRIDE.geometry = previous


def _positional_index(function, name: str) -> int | None:
    """Index of ``name`` among the function's positional arguments (``self`` excluded)."""
    index = 0
    for parameter in list(inspect.signature(function).parameters.values())[1:]:
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            if parameter.name == name:
                return index
            index += 1
        else:
            break
    return None


def _can_inject(function, name: str, args) -> bool:
    """True when the caller has not already supplied ``name`` positionally."""
    index = _positional_index(function, name)
    return index is None or len(args) <= index


def _install_geometry_shim() -> None:
    """Thread the loader's depth/tile through ComfyUI's own constructors.

    ``MiniMaxH3VideoVAE.__init__`` already takes ``tile_size``/``tile_overlap_min`` upstream
    and ``ViT3DDecoder.__init__`` already takes ``num_layers``; only the wiring between them
    is missing, because ComfyUI's H3 branch calls both without those two arguments.
    """
    decoder_init = minimax_vae.ViT3DDecoder.__init__
    if getattr(decoder_init, "_lynnreal_shim", False):
        return

    def decoder_init_with_depth(self, *args, **kwargs):
        pending = _pending()
        if pending is not None and _can_inject(decoder_init, "num_layers", args):
            kwargs["num_layers"] = pending["num_layers"]
        return decoder_init(self, *args, **kwargs)

    decoder_init_with_depth._lynnreal_shim = True
    minimax_vae.ViT3DDecoder.__init__ = decoder_init_with_depth

    vae_init = minimax_vae.MiniMaxH3VideoVAE.__init__
    def vae_init_with_geometry(self, *args, **kwargs):
        pending = _pending()
        if pending is not None:
            # Both are keyword-only in practice; the guard keeps positional callers safe.
            for name in ("tile_size", "tile_overlap_min"):
                if _can_inject(vae_init, name, args):
                    kwargs[name] = pending[name]
        return vae_init(self, *args, **kwargs)

    vae_init_with_geometry._lynnreal_shim = True
    minimax_vae.MiniMaxH3VideoVAE.__init__ = vae_init_with_geometry


def _recoverable_compile_failure(error: BaseException) -> bool:
    """Compilation/device support failures are recoverable; corrupt CUDA state is not."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return False
    module = type(error).__module__
    return (module.startswith(("triton", "torch._dynamo", "torch._inductor"))
            or any(marker in str(error).lower() for marker in
                   ("not supported", "not implemented", "no kernel image", "invalid device function")))


def _compile_decoder(decoder) -> None:
    """The release's ``model.acceleration.compile_decoder``.

    ``fullgraph=True`` refuses a partial graph instead of silently mis-compiling it, and any
    recoverable failure swaps the eager function back in. Compiling the module instead of
    ``forward`` goes through ``Module.__call__`` and the in-place ops it wraps, which
    mis-compiled to constant frames when it was tried here.
    """
    native = decoder.forward
    compiled = torch.compile(native, fullgraph=True, dynamic=False)

    def forward(*args, **kwargs):
        try:
            return compiled(*args, **kwargs)
        except Exception as error:
            if not _recoverable_compile_failure(error):
                raise
            warnings.warn("Decoder compilation unavailable; using the eager decoder: {}".format(error),
                          RuntimeWarning)
            decoder.forward = native
            return native(*args, **kwargs)

    decoder.forward = forward


def decoder_depth(sd) -> int:
    """Number of ``decoder.transformer_blocks.N`` entries in the checkpoint."""
    indices = set()
    prefix = "decoder.transformer_blocks."
    for key in sd:
        if key.startswith(prefix):
            head = key[len(prefix):].split(".", 1)[0]
            if head.isdigit():
                indices.add(int(head))
    return max(indices) + 1 if indices else 0


def build_vae(sd, metadata=None, device=None, num_layers: int = 0, tile_size: int = 0,
              tile_overlap: int = 0, compile_decoder: bool = True):
    """``comfy.sd.VAE(sd=...)`` with the H3 decoder depth taken from the checkpoint."""
    if not any(key.startswith("decoder.transformer_blocks.") for key in sd):
        raise ValueError(
            "This is not a MiniMax-H3 video VAE checkpoint (no decoder.transformer_blocks.*). "
            "Use the stock VAELoader for the audio VAE.")

    detected = decoder_depth(sd)
    layers = num_layers or detected or OFFICIAL_LAYERS
    light = layers == LIGHT_LAYERS
    tile = tile_size or (LIGHT_TILE[0] if light else OFFICIAL_TILE[0])
    overlap = tile_overlap or (LIGHT_TILE[1] if light else OFFICIAL_TILE[1])

    _install_geometry_shim()
    with _geometry(layers, tile, overlap):
        vae = comfy.sd.VAE(sd=sd, metadata=metadata, device=device)
    vae.throw_exception_if_invalid()

    compiled = bool(compile_decoder) and light and os.environ.get("LYNNREAL_NO_COMPILE_VAE") != "1"
    if compiled:
        _compile_decoder(vae.first_stage_model.decoder)

    batched = _install_batched_decode(vae)
    device_buffer = _install_device_decode(vae)

    # The release runs FA3 in the decoder too; the VAE object is ours, so marking it here cannot
    # reach a stock-loaded VAE.
    bench.install()
    attention_fa3.probe_once()
    marked = attention_fa3.install_vae(minimax_vae, vae.first_stage_model)
    if marked:
        logging.info("LynnReal: video VAE attention on FlashAttention 3 (%d blocks).", marked)

    if batched:
        logging.info("LynnReal: video VAE decodes whole tile batches per call (LYNNREAL_VAE_TILE_BATCH=%d, "
                     "0 = all tiles), the release's tile_batch setting.", _tile_batch())
    if device_buffer:
        logging.info("LynnReal: video VAE streams into a device-side output buffer (no per-chunk "
                     "PCIe copies).")
    logging.info(
        "LynnReal: H3 video VAE loaded -- %d decoder blocks (checkpoint %d), tile %d/%d, "
        "decoder %s.", layers, detected, tile, overlap, "compiled" if compiled else "eager")
    return vae


def load_vae_patcher(vae_path, metadata=None, device=None, disable_dynamic=False):
    """Reload factory so multi-GPU deep clones rebuild this VAE the same way."""
    if metadata is None:
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    else:
        sd = comfy.utils.load_torch_file(vae_path)
    return build_vae(sd, metadata=metadata, device=device).patcher


class LynnRealH3VAELoader(io.ComfyNode):
    """Load an H3 video VAE at its own decoder depth (Light VAE aware)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LynnRealH3VAELoader",
            display_name="Load LynnReal H3 VAE (Light VAE aware)",
            description=(
                "Loads a MiniMax-H3 video VAE with the decoder depth read from the checkpoint, so "
                "the 26-block LynnReal Light VAE loads correctly (the stock VAELoader builds 36 "
                "blocks, warns about the missing keys and leaves ten blocks randomly initialized). "
                "Applies the Light VAE's 272/16 tile geometry and the release's compiled decoder. "
                "The official 36-block VAE loads exactly as upstream does."
            ),
            category="model/loaders",
            inputs=[
                io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae"),
                               tooltip="A MiniMax-H3 video VAE: the LynnReal Light VAE or the official one."),
                io.Int.Input("num_layers", default=0, min=0, max=256, optional=True,
                             tooltip="Decoder blocks. 0 reads the depth from the checkpoint."),
                io.Int.Input("tile_size", default=0, min=0, max=2048, optional=True,
                             tooltip="Spatial tile. 0 uses 272 for the Light VAE, 256 otherwise."),
                io.Int.Input("tile_overlap", default=0, min=0, max=1024, optional=True,
                             tooltip="Tile overlap. 0 uses 16 for the Light VAE, 64 otherwise."),
                io.Boolean.Input("compile_decoder", default=True, optional=True,
                                 tooltip="torch.compile the Light VAE decoder, as the release's "
                                         "--compile-vae does. Ignored for the official VAE."),
            ],
            outputs=[io.Vae.Output(display_name="vae")],
        )

    @classmethod
    def execute(cls, vae_name: str, num_layers: int = 0, tile_size: int = 0, tile_overlap: int = 0,
                compile_decoder: bool = True) -> io.NodeOutput:
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = build_vae(sd, metadata=metadata, num_layers=num_layers, tile_size=tile_size,
                        tile_overlap=tile_overlap, compile_decoder=compile_decoder)
        # Same reload factory the stock loader registers, so Select VAE Device / multi-GPU
        # deep clones rebuild this VAE through the Light-VAE-aware path.
        vae.patcher.cached_patcher_init = (load_vae_patcher, (vae_path, metadata, None))
        return io.NodeOutput(vae)


class LynnRealLightVAEExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LynnRealH3VAELoader]


async def comfy_entrypoint() -> LynnRealLightVAEExtension:
    return LynnRealLightVAEExtension()
