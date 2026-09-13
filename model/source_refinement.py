"""Source-initialized Ref2VA editing, using clean native reference encodings."""
from contextlib import contextmanager
import math
import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLatentsStep
from .offload import vae_phase
from .refinement import RefineTimesteps


def mix_source(clean, noise, sigma):
    if clean.shape != noise.shape or not torch.isfinite(clean).all():
        raise ValueError('source and target latent geometry must agree and be finite')
    return clean.to(noise.device).float() * (1 - sigma) + noise.float() * sigma


class SourceLatents(MiniMaxH3PrepareLatentsStep):
    def __init__(self, captured, sigmas, audio_sigmas, record):
        super().__init__()
        self.captured, self.sigmas, self.audio_sigmas, self.record = captured, sigmas, audio_sigmas, record

    @torch.no_grad()
    def __call__(self, components, state):
        components, state = super().__call__(components, state)
        video, audio = state.get('latents'), state.get('audio_latents')
        nv, na = state.get('num_condition_video_rows', 0), state.get('num_condition_audio_rows', 0)
        shape = tuple(state.get(k) for k in ('num_latent_frames', 'latent_height', 'latent_width'))
        if self.captured['geometry'] != shape:
            raise ValueError('source encoding must match the target temporal phase and spatial grid')
        video[nv:] = mix_source(self.captured['video'], video[nv:], self.sigmas[0])
        # Silent source clips have no audio reference. Encode actual silence;
        # normalized zero or random noise must not be treated as clean audio.
        n = (len(audio) - na) // 2
        stride = math.prod(components.audio_vae.config.encoder_rates)
        waveform = torch.zeros((2, 1, n * stride), device=audio.device)
        with vae_phase(components):
            posterior = components.audio_vae.encode(waveform, return_dict=False)[0]
        clean = posterior.mode().float().transpose(1, 2)
        mean = torch.tensor(components.audio_vae.config.latents_mean, device=audio.device)
        std = torch.tensor(components.audio_vae.config.latents_std, device=audio.device)
        clean = ((clean - mean) / std).reshape(-1, components.audio_latent_channels)
        audio[na:] = mix_source(clean, audio[na:], self.audio_sigmas[0])
        self.record.update(source_geometry=list(shape), source_rows=len(video)-nv,
                           audio_rows=len(audio)-na, audio_initialization='encoded silence posterior mean',
                           clean_source_before_reference_augmentation=True)
        return components, state


@contextmanager
def source_refinement(pipeline, sigmas):
    """Temporarily replace initialization and schedules, preserving native compression."""
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32).tolist()
    if (not 2 <= len(sigmas) <= 5 or not all(math.isfinite(s) for s in sigmas)
            or not 0 < sigmas[0] <= 1 or sigmas[-1] != 0
            or not all(a > b for a, b in zip(sigmas, sigmas[1:]))):
        raise ValueError('expected one to four decreasing flow transitions ending at zero')
    # Reuse exact native FP32 values at trained points. Algebraically inverting
    # the rounded video shift otherwise moves audio times by up to 1.8e-7.
    video_grid, audio_grid = MiniMaxH3Scheduler(shift=12.), MiniMaxH3Scheduler(shift=3.)
    video_grid.set_timesteps(5); audio_grid.set_timesteps(5)
    native_audio = dict(zip(video_grid.sigmas.tolist(), audio_grid.sigmas.tolist()))
    mapped_audio = (torch.tensor(sigmas) / (4 - 3 * torch.tensor(sigmas))).tolist()
    audio_sigmas = [native_audio.get(v, a) for v, a in zip(sigmas, mapped_audio)]
    runtime = pipeline.pipe._blocks.sub_blocks
    encoder = runtime['reference_encoder']
    if 'encode_references' in encoder.__dict__:
        raise ValueError('source refinement cannot replace an existing encoder override')
    original_encode = encoder.encode_references
    original_prepare, original_times = runtime['prepare_latents'], runtime['set_timesteps']
    captured = {}
    record = dict(video_sigmas=sigmas, audio_sigmas=audio_sigmas, actual_expected_forwards=len(sigmas)-1)

    def encode(components, references, device=None):
        if not references or references[0].kind != 'video' or any(r.has_audio for r in references):
            raise ValueError('this probe requires a silent source video as the first reference')
        video, audio = original_encode(components, references, device)
        ref = references[0]
        captured.update(video=video[:ref.num_video_rows].clone(),
                        geometry=(ref.num_latent_frames, ref.latent_height, ref.latent_width))
        return video, audio

    encoder.encode_references = encode
    runtime['prepare_latents'] = SourceLatents(captured, sigmas, audio_sigmas, record)
    runtime['set_timesteps'] = RefineTimesteps(sigmas, audio_sigmas)
    try:
        yield record
    finally:
        del encoder.encode_references
        runtime['prepare_latents'], runtime['set_timesteps'] = original_prepare, original_times
