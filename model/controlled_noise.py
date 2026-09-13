"""Hold target video/audio noise fixed when comparing different reference layouts."""
from contextlib import contextmanager
import hashlib
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLatentsStep
from diffusers.modular_pipelines.minimax_h3.packing import keyframe_condition_noise


@contextmanager
def controlled_target_noise(seed):
    original = MiniMaxH3PrepareLatentsStep.prepare_latents
    record = dict(seed=seed, convention='native full-resolution source-video-only reference draw, then target video/audio', calls=0)

    def prepare(components, num_latent_frames, latent_height, latent_width, num_audio_latents,
                device, generator=None, latents=None, audio_latents=None):
        if latents is not None or audio_latents is not None:
            raise ValueError('controlled comparison requires freshly drawn target noise')
        target_rng = torch.Generator().manual_seed(seed)
        # Reproduce the existing one-video baseline exactly. Image reference
        # count/resolution must not advance this independent target generator.
        shape = (num_latent_frames, latent_height, latent_width)
        keyframe_condition_noise((shape,), components.patch_size, components.vae_latent_channels,
                                 generator=target_rng, device=torch.device('cpu'))
        video, audio = original(components, *shape, num_audio_latents, device, target_rng)
        record.update(calls=record['calls'] + 1, video_shape=list(video.shape), audio_shape=list(audio.shape),
                      video_sha256=hashlib.sha256(video.detach().float().cpu().numpy().tobytes()).hexdigest(),
                      audio_sha256=hashlib.sha256(audio.detach().float().cpu().numpy().tobytes()).hexdigest())
        return video, audio

    MiniMaxH3PrepareLatentsStep.prepare_latents = staticmethod(prepare)
    try:
        yield record
        if record['calls'] != 1:
            raise RuntimeError('expected exactly one target noise preparation per sample')
    finally:
        MiniMaxH3PrepareLatentsStep.prepare_latents = staticmethod(original)
