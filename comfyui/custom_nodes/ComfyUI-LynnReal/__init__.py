"""LynnReal nodes for ComfyUI.

Three things the LynnReal release needs that stock ComfyUI does not do by itself:

* ``LynnRealFlashTokenCompression`` -- the Flash DiT's trained token compression.
* ``LynnRealH3VAELoader`` -- loads the distilled Light VAE (26 decoder blocks) and the
  release's decode recipe (272/16 tiles, compiled decoder) without touching ``comfy/``.
* ``LynnRealInt8Backend`` -- reports (and, on a cu126 torch, enables) the comfy-kitchen
  Triton backend so INT8 checkpoints run fast without a launcher flag.

Plus ``LynnRealAlignedReference`` for the pose/hand control workflows.

``attention_fa3`` injects the release's FlashAttention 3 path into the Flash graph only (the
Flash token-compression node hands each block its attention callable) and ``bench`` measures
the release's per-stage timings when ``LYNNREAL_TIMING=1``.

``int8_fast`` takes comfy-kitchen's per-shape INT8 autotune out of the request path, so that
editing a prompt no longer re-tunes the quantised GEMMs before the first denoising step.

Every module here is a plain custom node: nothing under ``comfy/`` is patched on disk.
"""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# ``runtime`` and ``backends`` install their defaults when they are imported.
from . import (aligned_reference, attention_fa3, backends, bench, fast_blocks, flash_compression,  # noqa: F401
               int8_fast, light_vae, lynnreal_kernels, runtime)

bench.install()
attention_fa3.install_rope_patch()
int8_fast.install()


class LynnRealExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            *await flash_compression.LynnRealFlashExtension().get_node_list(),
            *await light_vae.LynnRealLightVAEExtension().get_node_list(),
            *await aligned_reference.LynnRealAlignedReferenceExtension().get_node_list(),
            *await backends.LynnRealBackendsExtension().get_node_list(),
        ]


async def comfy_entrypoint() -> LynnRealExtension:
    return LynnRealExtension()
