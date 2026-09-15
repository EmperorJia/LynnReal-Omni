"""LynnReal nodes for ComfyUI.

Three things the LynnReal release needs that stock ComfyUI does not do by itself:

* ``LynnRealFlashTokenCompression`` -- the Flash DiT's trained token compression.
* ``LynnRealH3VAELoader`` -- loads the distilled Light VAE (26 decoder blocks) and the
  release's decode recipe (272/16 tiles, compiled decoder) without touching ``comfy/``.
* ``LynnRealInt8Backend`` -- reports (and, on a cu126 torch, enables) the comfy-kitchen
  Triton backend so INT8 checkpoints run fast without a launcher flag.

Plus ``LynnRealAlignedReference`` for the pose/hand control workflows.

Every module here is a plain custom node: nothing under ``comfy/`` is patched on disk.
"""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# ``runtime`` and ``backends`` install their defaults when they are imported.
from . import aligned_reference, backends, flash_compression, light_vae, runtime  # noqa: F401


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
