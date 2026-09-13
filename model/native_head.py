"""Keep the current RGB boundary inside the native target's first latent slot."""
from contextlib import nullcontext
import torch
from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG
from .offload import StagedDenoise, StagedReferenceDenoise, StagedReferenceEncoder


def head_timestep_plan(plan, video_indices, condition_rows, head_rows):
    result = []
    for times, assignments in plan:
        row_times = times[assignments].clone()
        indices = video_indices.to(row_times.device)
        row_times[indices[condition_rows:condition_rows+head_rows]] = row_times[indices[0]].clone()
        result.append(torch.unique(row_times, sorted=True, return_inverse=True))
    return result


class FixedHead:
    def __init__(self, encoder, offload_blocks):
        super().__init__()
        self.encoder, self.offload_blocks = encoder, offload_blocks
        self.audit = {}
        self.history = None

    @torch.no_grad()
    def __call__(self, components, state):
        if self.history is not None:
            self.history.inject(state)
        video = state.get('latents')
        clean = self.encoder.current_rows.to(video.device, video.dtype)
        start, rows = state.get('num_condition_video_rows'), len(clean)
        if rows != state.get('latent_height') * state.get('latent_width') // 4:
            raise ValueError('fixed native head requires one complete image latent')
        augmentation = MINIMAX_H3_KEYFRAME_NOISE_AUG
        noisy = augmentation * clean + (1-augmentation) * video[start:start+rows].clone()
        state.set('row_timestep_plan', head_timestep_plan(state.get('row_timestep_plan'),
            state.get('video_indices'), start, rows))
        calls = 0

        def restore(module, args, kwargs):
            nonlocal calls
            kwargs['hidden_states'][:, start:start+rows].copy_(noisy[None])
            calls += 1

        transformer = components.transformer_ref if self.reference else components.transformer
        handle = transformer.register_forward_pre_hook(restore, with_kwargs=True)
        try:
            components, state = super().__call__(components, state)
        finally:
            handle.remove()
        state.get('latents')[start:start+rows].copy_(clean)
        self.audit = dict(fixed_head_rows=rows, fixed_head_forwards=calls,
                          endpoint_restored=True, extra_transformer_rows=0)
        if self.reference and isinstance(self.encoder, InitialReferenceEncoder):
            self.audit.update(head_source='causal_boundary_only',
                              appearance_reference_encodes=self.encoder.initial_encodes)
        elif self.reference:
            self.audit.update(head_reference_index=self.encoder.index,
                              head_reference_row_start=self.encoder.current_start)
        if calls != len(state.get('row_timestep_plan')):
            raise RuntimeError('fixed-head hook missed a denoiser evaluation')
        if self.history is not None:
            self.audit.update(self.history.audit)
            self.history.advance(state)
        return components, state


class FixedHeadDenoise(FixedHead, StagedDenoise):
    reference = False


class FixedReferenceHeadDenoise(FixedHead, StagedReferenceDenoise):
    reference = True


class BoundaryReferenceEncoder(StagedReferenceEncoder):
    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def encode_references(self, components, references, device=None):
        if not references or not -len(references) <= self.index < len(references) or references[self.index].kind != 'image':
            raise ValueError('the selected head reference must be the exact current boundary image')
        video, audio = super().encode_references(components, references, device)
        index = self.index % len(references)
        self.current_start = sum(r.num_video_rows for r in references[:index])
        self.current_rows = video[self.current_start:self.current_start+references[index].num_video_rows].clone()
        return video, audio


class InitialReferenceEncoder(StagedReferenceEncoder):
    """Only the immutable first image enters reference conditioning."""
    def __init__(self):
        super().__init__()
        self.boundary_image = self.initial_key = self.initial_rows = None
        self.initial_encodes = 0
        self.cache = None

    @torch.no_grad()
    def __call__(self, components, state):
        from .native_boundary import image_key
        from .offload import vae_phase
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VAReferenceEncoderStep
        references = self.get_block_state(state).prepared_references
        cached = (self.cache is not None and self.boundary_image is not None
                  and self.cache.matches([self.boundary_image]) and len(references) == 1
                  and self.initial_key == image_key(references[0].image))
        with nullcontext() if cached else vae_phase(components):
            return MiniMaxH3Ref2VAReferenceEncoderStep.__call__(self, components, state)

    def encode_references(self, components, references, device=None):
        from .native_boundary import image_key
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3KeyframeVaeEncoderStep
        if len(references) != 1 or references[0].kind != 'image' or self.boundary_image is None:
            raise ValueError('supply one original appearance image and a separate target boundary')
        reference = references[0]
        key = image_key(reference.image)
        if self.initial_key is None:
            self.initial_rows, _ = super().encode_references(components, references, device)
            self.initial_shape = (reference.num_latent_frames, reference.latent_height, reference.latent_width)
            self.initial_key = key
            self.initial_encodes += 1
        elif key != self.initial_key:
            raise ValueError('the initial appearance reference must remain immutable')
        reference.num_latent_frames, reference.latent_height, reference.latent_width = self.initial_shape
        current = self.cache.take([self.boundary_image]) if self.cache is not None else None
        self.current_rows = current if current is not None else MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes(
            components, [self.boundary_image], device)
        return self.initial_rows, None


def configure_fixed_head(blocks, reference_index=0, independent_boundary=False):
    if 'reference_encoder' in blocks.sub_blocks:
        encoder = InitialReferenceEncoder() if independent_boundary else BoundaryReferenceEncoder(reference_index)
        blocks.sub_blocks['reference_encoder'] = encoder
        blocks.sub_blocks['denoise'] = FixedReferenceHeadDenoise(encoder, blocks.sub_blocks['denoise'].offload_blocks)
        return
    encoder = blocks.sub_blocks['vae_encoder']
    from .native_appearance import AppearanceEncoder
    if not isinstance(encoder, AppearanceEncoder):
        raise ValueError('fixed native head requires the initial-appearance encoder')
    blocks.sub_blocks['denoise'] = FixedHeadDenoise(encoder, blocks.sub_blocks['denoise'].offload_blocks)
