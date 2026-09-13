"""Spatial refinement with explicit video/audio flow schedules and exact NFE."""
import math
import torch
import torch.nn.functional as F
from .latent_resize import bislerp_spatial
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLatentsStep, MiniMaxH3SetTimestepsStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep
from diffusers.modular_pipelines.minimax_h3.packing import (
    build_row_timesteps, patchify_video_latents, unpatchify_video_tokens,
)


def refinement_sigmas(strength, steps, schedule="trained-tail"):
    """Keep the trained final transition; a uniform low-noise grid is opt-in."""
    if not 0 < strength < 1 or not 1 <= steps <= 4 or schedule not in {"linear", "trained-tail"}:
        raise ValueError("refinement requires strength in (0,1) and 1..4 evaluations")
    if schedule == "linear" or steps == 1:
        return torch.linspace(strength, 0., steps + 1).tolist()
    if schedule != "trained-tail" or strength <= 12/14:
        raise ValueError("multi-step trained-tail refinement must start above sigma=12/14")
    return torch.linspace(strength, 12/14, steps).tolist() + [0.]


def resize_spatial(latents, height, width, method="bilinear", match_std=False):
    if latents.ndim != 5 or min(height, width) < 1:
        raise ValueError("expected B,C,T,H,W normalized video latents")
    if latents.shape[-2:] == (height, width):
        return latents.float()
    b, c, t, h, w = latents.shape
    frames = latents.float().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    if method == "bislerp":
        frames = bislerp_spatial(frames, height, width)
    else:
        options = {} if method == "nearest-exact" else {"align_corners": False}
        frames = F.interpolate(frames, (height, width), mode=method, **options)
    result = frames.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4).contiguous()
    if match_std:
        # One scale per channel over the entire clip avoids framewise brightness pumping.
        dims = (2, 3, 4)
        src_std, src_mean = torch.std_mean(latents.float(), dim=dims, keepdim=True, correction=0)
        dst_std, dst_mean = torch.std_mean(result, dim=dims, keepdim=True, correction=0)
        result = (result - dst_mean) * (src_std / dst_std.clamp_min(1e-6)) + src_mean
    return result


def target_latents(state, pipeline):
    """Extract normalized target video and channel-major audio rows, on CPU."""
    video = state["latents"][state.get("num_condition_video_rows", 0):]
    video = unpatchify_video_tokens(video, state["num_latent_frames"], state["latent_height"],
                                   state["latent_width"], pipeline.vae_latent_channels, pipeline.patch_size)
    audio = state["audio_latents"][state.get("num_condition_audio_rows", 0):]
    return video.float().cpu(), audio.float().cpu()


class RefineLatents(MiniMaxH3PrepareLatentsStep):
    def __init__(self, video, audio, sigma, audio_sigma=0., method="bilinear", match_std=False):
        super().__init__()
        self.video, self.audio, self.sigma = video, audio, sigma
        self.audio_sigma, self.method, self.match_std = audio_sigma, method, match_std

    @torch.no_grad()
    def __call__(self, components, state):
        components, state = super().__call__(components, state)
        video, audio = state.get("latents"), state.get("audio_latents")
        nv, na = state.get("num_condition_video_rows", 0), state.get("num_condition_audio_rows", 0)
        if self.video.shape[:3] != (1, components.vae_latent_channels, state.get("num_latent_frames")):
            raise ValueError("refinement cannot change the temporal latent clock or channels")
        clean = resize_spatial(self.video, state.get("latent_height"), state.get("latent_width"),
                               self.method, self.match_std)
        clean = patchify_video_latents(clean, components.patch_size).to(video.device)
        if clean.shape != video[nv:].shape or self.audio.shape != audio[na:].shape:
            raise ValueError("first/second-pass target row shapes disagree")
        video[nv:] = clean * (1 - self.sigma) + video[nv:] * self.sigma
        audio[na:] = self.audio.to(audio.device) * (1 - self.audio_sigma) + audio[na:] * self.audio_sigma
        return components, state


class RefineTimesteps(MiniMaxH3SetTimestepsStep):
    def __init__(self, sigmas, audio_sigmas=None):
        super().__init__()
        self.sigmas = list(sigmas)
        self.audio_sigmas = audio_sigmas

    @torch.no_grad()
    def __call__(self, components, state):
        block = self.get_block_state(state)
        device = components._execution_device
        components.scheduler.set_timesteps(sigmas=self.sigmas, device=device)
        block.timesteps = components.scheduler.timesteps
        if self.audio_sigmas is None:
            block.audio_timesteps = torch.ones_like(block.timesteps)
        else:
            components.audio_scheduler.set_timesteps(sigmas=self.audio_sigmas, device=device)
            block.audio_timesteps = components.audio_scheduler.timesteps
        block.row_timestep_plan = [tuple(x.to(device) for x in build_row_timesteps(
            block.layout, float(t), float(a), max(float(t), 0.999), 1.0))
            for t, a in zip(block.timesteps, block.audio_timesteps)]
        self.set_block_state(state, block)
        return components, state


class VideoOnlyUpdate(MiniMaxH3LoopSchedulerStep):
    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        n = block_state.num_condition_video_rows
        block_state.latents[n:] = components.scheduler.step(
            block_state.noise_pred[0, n:].float(), t, block_state.latents[n:], return_dict=False)[0]
        return components, block_state


def configure_refinement(blocks, video, audio, sigmas, *, audio_mode="frozen",
                         resize_method="bilinear", match_std=False):
    if (len(sigmas) < 2 or not all(math.isfinite(x) for x in sigmas)
            or not 0 < sigmas[0] <= 1 or sigmas[-1] != 0
            or not all(a > b for a, b in zip(sigmas, sigmas[1:]))):
        raise ValueError("refinement sigmas must strictly decrease from (0,1] to zero")
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError("first-pass latents contain nonfinite values")
    if audio_mode not in {"frozen", "joint"} or resize_method not in {"bilinear", "bicubic", "nearest-exact", "bislerp"}:
        raise ValueError("unsupported refinement audio or spatial interpolation policy")
    # Use the scheduler's FP32 values for both noise construction and row times.
    # Otherwise Python 1-sigma and FP32 1-sigma can round differently before INT8 projection.
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32).tolist()
    if not all(a > b for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError("refinement sigma points collapse after FP32 conversion")
    # Invert shift=12, then apply shift=3: sigma_a = sigma_v / (4 - 3*sigma_v).
    audio_sigmas = (torch.tensor(sigmas) / (4 - 3*torch.tensor(sigmas))).tolist() if audio_mode == "joint" else None
    blocks.sub_blocks["prepare_latents"] = RefineLatents(video, audio, sigmas[0],
        audio_sigmas[0] if audio_sigmas else 0., resize_method, match_std)
    blocks.sub_blocks["set_timesteps"] = RefineTimesteps(sigmas, audio_sigmas)
    blocks.sub_blocks["denoise"].sub_blocks["update"] = (
        MiniMaxH3LoopSchedulerStep() if audio_mode == "joint" else VideoOnlyUpdate())
