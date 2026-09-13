"""Reuse the decoder's VAE residency to encode the next exact RGB boundary."""
import hashlib

import torch
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3KeyframeVaeEncoderStep
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep
from .offload import vae_phase


def image_key(image):
    return image.size, image.mode, hashlib.sha256(image.tobytes()).hexdigest()


class BoundaryCache:
    def __init__(self, output_frames):
        self.output_frames = output_frames
        self.prefetch = False
        self.key = self.rows = None
        self.hits = self.encodes = 0
        self.tone_anchor = None
        self.boundary_frame = -1
        self.latent_index = None
        self.latent_feedbacks = 0

    def matches(self, images):
        return len(images) == 1 and self.rows is not None and image_key(images[0]) == self.key

    def take(self, images):
        if not self.matches(images):
            return None
        rows = self.rows
        self.key = self.rows = None
        self.hits += 1
        return rows


class CachedKeyframeEncoder(MiniMaxH3KeyframeVaeEncoderStep):
    def __init__(self, cache):
        super().__init__()
        self.cache = cache

    def encode_keyframes(self, components, images, device=None):
        rows = self.cache.take(images)
        if rows is not None:
            return rows
        return super().encode_keyframes(components, images, device)

    @torch.no_grad()
    def __call__(self, components, state):
        if self.cache.matches(self.get_block_state(state).keyframes):
            return super().__call__(components, state)
        with vae_phase(components):
            return super().__call__(components, state)


class PreencodeBoundaryDecoder(MiniMaxH3VideoDecodeStep):
    def __init__(self, cache):
        super().__init__()
        self.cache = cache

    @torch.no_grad()
    def __call__(self, components, state):
        with vae_phase(components):
            decode = components.vae.decode
            if self.cache.tone_anchor is not None:
                def anchored_decode(*args, **kwargs):
                    result = decode(*args, **kwargs)
                    return (self.cache.tone_anchor(result[0]), *result[1:])
                components.vae.decode = anchored_decode
            try:
                components, state = super().__call__(components, state)
            finally:
                components.vae.decode = decode
            if self.cache.prefetch:
                videos = state.get("videos")
                if state.get("output_type") != "pil" or len(videos) != 1 or len(videos[0]) != self.cache.output_frames:
                    raise ValueError("boundary reuse requires one complete native PIL video")
                image = videos[0][self.cache.boundary_frame]
                if self.cache.latent_index is None:
                    self.cache.rows = MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes(components, [image])
                    self.cache.encodes += 1
                else:
                    index = self.cache.latent_index
                    if index % 5 or self.cache.boundary_frame != index // 5 * 17:
                        raise ValueError('latent boundary must occupy the single-frame codec phase')
                    if not 0 <= index < state.get('num_latent_frames'):
                        raise ValueError('latent boundary lies outside the decoded target')
                    rows = state.get('latent_height') * state.get('latent_width') // 4
                    start = state.get('num_condition_video_rows') + index * rows
                    self.cache.rows = state.get('latents')[start:start+rows].detach().clone().cpu()
                    self.cache.latent_feedbacks += 1
                self.cache.key = image_key(image)
        return components, state


def configure_boundary_reuse(blocks, output_frames):
    if not {"vae_encoder", "reference_encoder"}.intersection(blocks.sub_blocks) or output_frames < 1:
        raise ValueError("boundary reuse requires native keyframe conditioning")
    cache = BoundaryCache(output_frames)
    from .native_appearance import AppearanceEncoder
    if "reference_encoder" in blocks.sub_blocks:
        from .native_head import InitialReferenceEncoder
        encoder = blocks.sub_blocks["reference_encoder"]
        if not isinstance(encoder, InitialReferenceEncoder):
            raise ValueError("reference boundary reuse requires an immutable first-frame reference")
        encoder.cache = cache
    elif isinstance(blocks.sub_blocks['vae_encoder'], AppearanceEncoder):
        blocks.sub_blocks['vae_encoder'].cache = cache
    else:
        blocks.sub_blocks["vae_encoder"] = CachedKeyframeEncoder(cache)
    blocks.sub_blocks["decode"].sub_blocks["video"] = PreencodeBoundaryDecoder(cache)
    return cache
