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
import os
import threading
import warnings

import torch
from typing_extensions import override

import comfy.ldm.minimax.vae as minimax_vae
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

OFFICIAL_LAYERS = 36
LIGHT_LAYERS = 26
OFFICIAL_TILE = (256, 64)
LIGHT_TILE = (272, 16)

# Set by the loader for the duration of one VAE construction. Per-thread so a second
# prompt on another thread can never see it.
_OVERRIDE = threading.local()


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
