"""A cached initial-image latent placed before the current native keyframe."""
import hashlib
from contextlib import nullcontext

import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep, _set_layout_state,
)
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3KeyframeVaeEncoderStep
from diffusers.modular_pipelines.minimax_h3.packing import (
    build_packed_sequence, _ROPE_FRAME_RESCALE, keyframe_condition_noise,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
)
from .offload import StagedKeyframeEncoder, vae_phase


def appearance_noise(shape, patch_size, channels, generator, device):
    """An extra reference must not shift the baseline's target video/audio noise."""
    if not isinstance(generator, torch.Generator):
        raise ValueError("initial appearance requires a single seeded request generator")
    current = keyframe_condition_noise((shape,), patch_size, channels, generator=generator, device=device)
    auxiliary = torch.Generator(device=generator.device).manual_seed(generator.initial_seed() ^ 0x5DEECE66D)
    initial = keyframe_condition_noise((shape,), patch_size, channels, generator=auxiliary, device=device)
    return torch.cat((current, initial))


def place_initial_reference(layout, head_only_boundary=False):
    """Keep current-keyframe/target alignment and place appearance in the past."""
    count = layout.num_condition_video_rows
    if count <= 0 or (not head_only_boundary and count % 2):
        raise ValueError("initial appearance requires two equal native keyframe blocks")
    half = 0 if head_only_boundary else count // 2
    origin = float(layout.text_indices.numel())
    layout.position_ids[layout.video_indices, 0] += 4 * _ROPE_FRAME_RESCALE
    layout.position_ids[layout.audio_indices, 0] += 4 * _ROPE_FRAME_RESCALE
    layout.position_ids[layout.video_indices[half:count], 0] = origin
    return layout


class AppearanceLayout(MiniMaxH3PrepareLayoutStep):
    def __init__(self, head_only_boundary=False):
        super().__init__()
        self.head_only_boundary = head_only_boundary

    @torch.no_grad()
    def __call__(self, components, state):
        block = self.get_block_state(state)
        if block.keyframe_anchors != ("first", "last"):
            raise ValueError("supply the current boundary followed by the initial appearance")
        layout = build_packed_sequence(block.text_token_tags, block.num_latent_frames,
            block.latent_height, block.latent_width, block.num_audio_latents,
            components.patch_size, ("first",) if self.head_only_boundary else block.keyframe_anchors)
        _set_layout_state(block, place_initial_reference(layout, self.head_only_boundary), components._execution_device)
        self.set_block_state(state, block)
        return components, state


class AppearanceEncoder(StagedKeyframeEncoder):
    def __init__(self, head_only_boundary=False):
        super().__init__()
        self.key = self.initial_rows = None
        self.cache = None
        self.initial_encodes = 0
        self.head_only_boundary = head_only_boundary

    def encode_keyframes(self, components, images, device=None):
        if len(images) != 2:
            raise ValueError("initial appearance requires current and initial images")
        current = self.cache.take(images[:1]) if self.cache is not None else None
        if current is None:
            current = MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes(components, images[:1], device)
        self.current_rows = current
        key = (images[1].size, images[1].mode, hashlib.sha256(images[1].tobytes()).hexdigest())
        if self.key != key:
            self.initial_rows = MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes(components, images[1:], device)
            self.key = key
            self.initial_encodes += 1
        return torch.cat((current, self.initial_rows))

    @torch.no_grad()
    def __call__(self, components, state):
        block = self.get_block_state(state)
        from .native_boundary import image_key
        cached = (len(block.keyframes) == 2 and self.cache is not None
                  and self.cache.matches(block.keyframes[:1]) and self.key == image_key(block.keyframes[1]))
        with nullcontext() if cached else vae_phase(components):
            device = components._execution_device
            clean = self.encode_keyframes(components, block.keyframes, device)
            noise = appearance_noise((1, block.latent_height, block.latent_width),
                components.patch_size, components.vae_latent_channels, block.generator, device)
            block.condition_latents = components.scheduler.scale_noise(
                clean.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
            if self.head_only_boundary:
                # Preserve both RNG draws, but carry the current boundary only in target slot zero.
                block.condition_latents = block.condition_latents[len(self.current_rows):]
            self.set_block_state(state, block)
        return components, state


def configure_initial_appearance(blocks, head_only_boundary=False):
    if "vae_encoder" not in blocks.sub_blocks:
        raise ValueError("initial appearance requires native keyframe conditioning")
    encoder = AppearanceEncoder(head_only_boundary)
    blocks.sub_blocks["vae_encoder"] = encoder
    blocks.sub_blocks["prepare_layout"] = AppearanceLayout(head_only_boundary)
    return encoder
