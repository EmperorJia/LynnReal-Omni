"""Make MiniMax-H3 ``ref_image_size=max`` match the official Ref2VA pipeline.

Stock ComfyUI currently caps the scale at 1.0, so a small reference image is never
upscaled.  Official Ref2VA always resolves an image to a 2048-pixel short edge and
then rounds each axis independently to a multiple of 32.  Patch the shared core node
at runtime so every LynnReal workflow using that node gets the same behavior without
forking six workflow graphs or modifying files under ``comfy/``.
"""

from __future__ import annotations

import logging

import nodes as comfy_nodes
from comfy_extras import nodes_minimax_h3

REFERENCE_SHORT_EDGE = 2048
CANVAS_MULTIPLE = 32
_INSTALLED_ATTR = "_lynnreal_official_max_reference_size"


def resolve_max_reference_size(width: int, height: int) -> tuple[int, int]:
    """Return ``(width, height)`` for official Ref2VA ``max`` image sizing."""
    if width <= 0 or height <= 0:
        raise ValueError("A reference image must have a positive size, got {}x{}."
                         .format(width, height))
    scale = REFERENCE_SHORT_EDGE / min(width, height)
    target_width = max(
        CANVAS_MULTIPLE,
        round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    target_height = max(
        CANVAS_MULTIPLE,
        round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    return target_width, target_height


def _prepare_max_references(ref_images):
    if not ref_images:
        return ref_images
    prepared = {}
    for name, image in ref_images.items():
        if image is None:
            prepared[name] = None
            continue
        source_height, source_width = map(int, image.shape[1:3])
        target_width, target_height = resolve_max_reference_size(
            source_width, source_height)
        if (source_width, source_height) == (target_width, target_height):
            resized = image[:1]
        else:
            resized = nodes_minimax_h3._resize(
                image[:1], target_width, target_height, "disabled")
        prepared[name] = resized
        logging.info(
            "LynnReal: Ref2VA max reference %s %dx%d -> %dx%d "
            "(2048px short edge, 32-aligned).",
            name, source_width, source_height, target_width, target_height,
        )
    return prepared


def _update_schema_tooltips(schema) -> None:
    size_tooltip = (
        "Reference image sizing. 'match' scales each ref (down only, keeping aspect) "
        "to the generation's pixel area; 'max' follows official Ref2VA and resizes "
        "every image to a 2048px short edge (upscaling included), then aligns both "
        "axes to 32. Reference tokens ride through every sampling step, so 'max' can "
        "be several times slower."
    )
    image_tooltip = (
        "Reference image. With ref_image_size='max', it is resized to a 2048px short "
        "edge (upscaling included) and both axes are aligned to 32."
    )
    for input_spec in schema.inputs:
        if input_spec.id == "ref_image_size":
            input_spec.tooltip = size_tooltip
        elif input_spec.id == "ref_images":
            template = getattr(input_spec, "template", None)
            if template is None:
                continue
            template.input.tooltip = image_tooltip
            for cached_input in template.cached_inputs.values():
                cached_input.tooltip = image_tooltip


def _install_on_class(node) -> bool:
    """Patch one V3 class (ComfyUI registers a clone of the source class)."""
    if node.__dict__.get(_INSTALLED_ATTR, False):
        return False

    original_execute = node.execute
    original_define_schema = node.define_schema

    @classmethod
    def execute(cls, clip, prompt, width, height, length, ref_image_size="match",
                vae=None, audio_vae=None, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None):
        if ref_image_size == "max":
            ref_images = _prepare_max_references(ref_images)
        return original_execute(
            clip=clip,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            ref_image_size=ref_image_size,
            vae=vae,
            audio_vae=audio_vae,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )

    @classmethod
    def define_schema(cls):
        schema = original_define_schema()
        _update_schema_tooltips(schema)
        return schema

    node.execute = execute
    node.define_schema = define_schema
    setattr(node, _INSTALLED_ATTR, True)
    return True


def install() -> bool:
    """Patch both ComfyUI's registered V3 clone and the source class once."""
    source = nodes_minimax_h3.MiniMaxH3ReferenceToVideo
    registered = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3ReferenceToVideo")
    targets = []
    for node in (registered, source):
        if node is not None and node not in targets:
            targets.append(node)
    installed = sum(_install_on_class(node) for node in targets)
    if not installed:
        return False
    logging.info(
        "LynnReal: MiniMaxH3ReferenceToVideo max sizing uses the official 2048px "
        "short edge (upscaling included, 32-aligned); patched %d V3 class(es).",
        installed,
    )
    return True
