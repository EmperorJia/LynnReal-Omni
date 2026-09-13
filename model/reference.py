"""Frame controls share target geometry while retaining their native Ref2VA time offset."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep
from diffusers.modular_pipelines.minimax_h3.packing_ref2va import reference_media_to_uint8, resample_reference_frames, prepare_reference_image


class SizedReferenceSetup(MiniMaxH3Ref2VASetupStep):
    """Resize image references explicitly for the selected encoder pipeline."""
    def __init__(self, short_edge, video_short_edge=None):
        super().__init__()
        if short_edge < 32 or short_edge % 32:
            raise ValueError('reference short edge must be a positive multiple of 32')
        self.short_edge = short_edge
        if video_short_edge is not None and (video_short_edge < 32 or video_short_edge % 32):
            raise ValueError('reference video short edge must be a positive multiple of 32')
        self.video_short_edge = video_short_edge

    @torch.no_grad()
    def __call__(self, components, state):
        components, state = super().__call__(components, state)
        block = self.get_block_state(state)
        for entry, reference in zip(block.references, state.get('prepared_references')):
            if reference.kind == 'video' and self.video_short_edge is not None:
                frames = resample_reference_frames(reference_media_to_uint8(entry.video), float(entry.fps))[:block.num_frames]
                scale = self.video_short_edge / min(frames.shape[1:3])
                height, width = (max(32, round(size*scale/32)*32) for size in frames.shape[1:3])
                reference.frames = np.stack([np.asarray(Image.fromarray(frame).resize((width,height), Image.Resampling.LANCZOS)) for frame in frames])
            if reference.kind != 'image':
                continue
            image = entry.image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(reference_media_to_uint8(image))
            image = ImageOps.exif_transpose(image).convert('RGB')
            scale = self.short_edge / min(image.size)
            width, height = (max(32, round(size*scale/32)*32) for size in image.size)
            reference.image = prepare_reference_image(image, height, width)
        return components, state

class AlignedReferenceSetup(MiniMaxH3Ref2VASetupStep):
    def __init__(self, indices):
        super().__init__()
        self.aligned_indices = tuple(indices)

    @torch.no_grad()
    def __call__(self, components, state):
        components, state = super().__call__(components, state)
        block = self.get_block_state(state)
        prepared_references = state.get("prepared_references")
        for index in self.aligned_indices:
            entry, prepared = block.references[index], prepared_references[index]
            if prepared.kind != "video":
                raise ValueError("frame-aligned controls must be videos")
            frames = resample_reference_frames(reference_media_to_uint8(entry.video), float(entry.fps))
            frames = frames[:block.num_frames]
            if not len(frames):
                raise ValueError("empty frame control")
            if frames.shape[1:3] != (block.height, block.width):
                tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float()
                tensor = F.interpolate(tensor, size=(block.height, block.width), mode="bilinear",
                                       align_corners=False, antialias=True)
                frames = tensor.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
            if len(frames) < block.num_frames:
                frames = np.concatenate((frames, np.repeat(frames[-1:], block.num_frames - len(frames), axis=0)))
            prepared.frames = np.ascontiguousarray(frames)
            prepared.frame_aligned_to_target = True
        return components, state

class AlignedReferenceLayout(MiniMaxH3Ref2VAPrepareLayoutStep):
    @torch.no_grad()
    def __call__(self, components, state):
        components, state = super().__call__(components, state)
        block = self.get_block_state(state)
        layout = state.get("layout")
        target = layout.video_indices[layout.num_condition_video_rows:]
        positions = layout.position_ids[target]
        cursor = 0
        offsets = []
        for reference in block.prepared_references:
            if reference.kind == "audio":
                continue
            count = reference.num_video_rows
            indices = layout.video_indices[cursor:cursor + count]
            cursor += count
            if not getattr(reference, "frame_aligned_to_target", False):
                continue
            if (reference.num_latent_frames, reference.latent_height, reference.latent_width) != (
                    block.num_latent_frames, block.latent_height, block.latent_width):
                raise RuntimeError("frame control and target latent geometry differ")
            other = layout.position_ids[indices]
            if other.shape != positions.shape or not torch.equal(other[:, 1:], positions[:, 1:]):
                raise RuntimeError("frame control and target spatial grids differ")
            offset = positions[:, 0] - other[:, 0]
            if not torch.allclose(offset, offset[:1].expand_as(offset), atol=1e-9, rtol=0) or float(offset[0]) <= 0:
                raise RuntimeError("frame control requires a constant positive target time offset")
            offsets.append(float(offset[0]))
        if cursor != layout.num_condition_video_rows:
            raise RuntimeError("reference row accounting mismatch")
        layout.frame_reference_rope_offsets = offsets
        return components, state
