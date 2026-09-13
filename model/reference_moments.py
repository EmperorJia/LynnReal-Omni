"""Transfer only original-frame latent channel statistics to the current video reference."""
import copy
import numpy as np
from PIL import Image
from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents, unpatchify_video_tokens
from .offload import StagedReferenceEncoder


class MomentReferenceEncoder(StagedReferenceEncoder):
    def __init__(self, first_image, match_variance=False):
        super().__init__()
        self.first_image = Image.open(first_image).convert('RGB')
        self.match_variance = match_variance
        self.style = None
        self.audit = {}

    def encode_references(self, components, references, device=None):
        if len(references) != 1 or references[0].kind != 'video' or references[0].has_audio:
            raise ValueError('moment transfer requires one silent current video reference')
        rows, audio = super().encode_references(components, references, device)
        reference = references[0]
        shape = (reference.num_latent_frames, reference.latent_height, reference.latent_width)
        current = unpatchify_video_tokens(rows, *shape, 24, components.patch_size)
        if self.style is None:
            style_reference = copy.copy(reference)
            h, w = reference.frames.shape[1:3]
            if self.first_image.size != (w, h):
                raise ValueError('original first image must match the reference canvas')
            # Static repetition is derived solely from the original first RGB;
            # it introduces no subsequent real/generated appearance frame.
            style_reference.frames = np.repeat(np.asarray(self.first_image)[None], len(reference.frames), axis=0)
            style_rows, _ = super().encode_references(components, [style_reference], device)
            self.style = unpatchify_video_tokens(style_rows, *shape, 24, components.patch_size)
        if self.style.shape != current.shape:
            raise ValueError('reference geometry changed during the comparison')
        axes = (2, 3, 4)
        source_mean = current.mean(axes, keepdim=True)
        target_mean = self.style.mean(axes, keepdim=True)
        scale = (self.style.std(axes, keepdim=True) / current.std(axes, keepdim=True).clamp_min(1e-6)).clamp(.8, 1.25)
        if not self.match_variance:
            scale = scale.new_ones(scale.shape)
        corrected = (current - source_mean) * scale + target_mean
        self.audit = dict(method='mean_std' if self.match_variance else 'mean_only',
                          statistics_axes='time,height,width; separate latent channels',
                          mean_offset=(target_mean-source_mean).flatten().tolist(),
                          scale=scale.flatten().tolist(),
                          normalized_latent_delta_rms=float((corrected-current).square().mean().sqrt()),
                          extra_reference_rows=0, style_from_original_first_only=True,
                          target_reference_shape=list(shape))
        return patchify_video_latents(corrected, components.patch_size), audio
