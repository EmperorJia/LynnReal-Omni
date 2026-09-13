"""Continuation with the trained head-overlap layout and bounded history."""
from contextlib import contextmanager, nullcontext
import sys
import time
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED, MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD,
    build_row_timesteps, patchify_video_latents, unpatchify_video_tokens,
)
from diffusers.utils.torch_utils import randn_tensor
from .framepack import pack_history, continuation_layout, advance_history

class Stream:
    def __init__(self, pipeline, capacity=64, stabilize_exposure=False, history_posterior="sample", history_feedback="latent", vae_offload=False,
                 sink_frames=1, mid_stride=4, boundary_rgb8=False, recent_stride=2,
                 sink_spatial="shared", history_pooling="point", reuse_vae_phase=False,
                 boundary_posterior="sample", chunk_frames=17, decode_layout="continuous"):
        if pipeline.config.get("variant") != "standard" or capacity < 7:
            raise ValueError("streaming requires the standard model and capacity >= 7")
        self.pipeline = pipeline
        self.transformer = pipeline.transformer
        self.vae = pipeline.pipe.vae
        self.scheduler = pipeline.pipe.scheduler
        self.capacity = capacity
        if chunk_frames not in (16, 17, 33, 34, 50, 51) or decode_layout not in {"native", "continuous"}:
            raise ValueError("unsupported continuation length or decode layout")
        if chunk_frames != 17 and history_feedback != "latent":
            raise ValueError("non-native continuation lengths require latent history")
        self.chunk_frames, self.decode_layout = chunk_frames, decode_layout
        self.target_latents = 1 + (chunk_frames + 3) // 4
        if sink_frames not in (1, 2) or mid_stride < 4:
            raise ValueError("sink_frames must be 1 or 2 and mid_stride >= 4")
        self.sink_frames, self.mid_stride = sink_frames, mid_stride
        self.boundary_rgb8 = boundary_rgb8
        if boundary_posterior not in {"sample", "mode"}:
            raise ValueError("boundary posterior must be sample or mode")
        self.boundary_posterior = boundary_posterior
        self.recent_stride, self.sink_spatial = recent_stride, sink_spatial
        self.history_pooling = history_pooling
        if reuse_vae_phase and (not vae_offload or history_feedback != "latent"):
            raise ValueError("VAE phase reuse requires phase offload and latent history")
        self.reuse_vae_phase = reuse_vae_phase
        self._held_phase, self._phase_entries = None, 0
        self.patch_size = tuple(self.transformer.config.patch_size)
        self.device = self.transformer.device
        self.stabilize_exposure = stabilize_exposure
        self.vae_offload = vae_offload
        if history_posterior not in {"sample", "mode"}:
            raise ValueError("history_posterior must be sample or mode")
        self.history_posterior = history_posterior
        if history_feedback not in {"latent", "native-rgb"}:
            raise ValueError("history feedback must be latent or native-rgb")
        self.history_feedback = history_feedback
        # A native 17*n+5 pixel window encodes to 5*n+2 temporal latents.
        self.rgb_capacity = 17 * ((capacity - 2) // 5) + 5

    @contextmanager
    def _vae_phase(self, hold=False):
        from .offload import vae_phase
        if self._held_phase is None:
            phase = vae_phase(self.pipeline.pipe) if self.vae_offload else nullcontext()
            phase.__enter__()
            self._held_phase = phase
            self._phase_entries += 1
        try:
            yield
        except BaseException:
            self.close(*sys.exc_info())
            raise
        else:
            if not hold:
                self.close()

    def close(self, *exception):
        if self._held_phase is not None:
            phase, self._held_phase = self._held_phase, None
            phase.__exit__(*(exception or (None, None, None)))

    def _pixels(self, pixels, inverse=False):
        mean = pixels.new_tensor(MINIMAX_H3_PIXEL_MEAN).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor(MINIMAX_H3_PIXEL_STD).view(1, 3, 1, 1, 1)
        return (pixels * std + mean).clamp(0, 1) if inverse else (pixels - mean) / std

    def _latents(self, latents, inverse=False):
        mean = latents.new_tensor(self.vae.config.latents_mean).view(1, 24, 1, 1, 1)
        std = latents.new_tensor(self.vae.config.latents_std).view(1, 24, 1, 1, 1)
        return latents * std + mean if inverse else (latents - mean) / std

    @torch.inference_mode()
    def initialize(self, pixels, appearance=None):
        """pixels: normalized [0,1] RGB, shape [1,3,T,H,W], native 17n+5 frames."""
        if pixels.shape[0:2] != (1, 3) or pixels.shape[2] < 5 or (pixels.shape[2] - 5) % 17:
            raise ValueError("initial history must contain 17*n+5 RGB frames")
        pixels = pixels.to(self.device, torch.float32)
        with self._vae_phase():
            self.history = self._encode_history(pixels)
        if appearance is not None:
            if appearance.shape != pixels[:, :, :1].shape:
                raise ValueError("appearance reference must be one RGB frame on the source canvas")
            self.history[:, :, :1] = self._encode_boundary(appearance.to(self.device, torch.float32))
        if self.history.shape[2] <= self.sink_frames:
            raise ValueError("source must include history after the retained sinks")
        self.sink = self.history[:, :, :self.sink_frames].clone()
        if self.history_feedback == "native-rgb":
            self.rgb_history = (pixels[:, :, -self.rgb_capacity:] * 255).round().byte().cpu()
        if self.history.shape[2] > self.capacity:
            self.history = torch.cat((self.sink, self.history[:, :, -(self.capacity - self.sink_frames):]), 2)
        self.boundary = pixels[:, :, -1:].clone()

    def _encode_boundary(self, pixels):
        with self._vae_phase(), torch.autocast(self.device.type, enabled=False):
            moments = self.vae._encode_clip(self._pixels(pixels.float()))
        posterior = DiagonalGaussianDistribution(moments)
        anchor = posterior.mode() if self.boundary_posterior == "mode" else posterior.sample(
            generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
        if anchor.shape[2] != 1:
            raise RuntimeError("RGB boundary must encode to one latent")
        return self._latents(anchor.to(torch.float16).float())

    def _encode_history(self, pixels):
        with torch.autocast(self.device.type, dtype=torch.float16):
            moments = self.vae._encode(self._pixels(pixels))
        # Native history constructs and samples the posterior outside encoder autocast.
        posterior = DiagonalGaussianDistribution(moments)
        if self.history_posterior == "sample":
            latent = posterior.sample(generator=torch.Generator().manual_seed(
                MINIMAX_H3_KEYFRAME_ENCODE_SEED)).to(torch.float16).float()
        else:
            latent = posterior.mode().float()
        return self._latents(latent)

    def _update_history(self, output, future):
        if self.history_feedback == "latent":
            self.history = advance_history(self.history, future, self.capacity, self.sink_frames)
            return
        from .offload import vae_phase
        rgb = (output * 255).round().byte().cpu()
        self.rgb_history = torch.cat((self.rgb_history, rgb), 2)[:, :, -self.rgb_capacity:].contiguous()
        # Re-encode a contiguous delivered RGB window on the native codec clock.
        # Restore the original dense sink after encoding, never splice distant
        # pixels into one VAE clip. This adds codec work, not DiT evaluations.
        with vae_phase(self.pipeline.pipe):
            pixels = self.rgb_history.to(self.device, torch.float32) / 255
            self.history = self._encode_history(pixels)
            del pixels
        self.history[:, :, :self.sink_frames] = self.sink
        if self.history.shape[2] > self.capacity:
            raise RuntimeError("native RGB history exceeds the latent capacity")

    @torch.inference_mode()
    def step(self, conditioning, seed=0, steps=4):
        if steps < 1:
            raise ValueError("continuation requires a positive denoiser forward count")
        self.pipeline.events.clear()
        self.pipeline.decoder_events.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        phase_entries = self._phase_entries
        # An image-only FP32 encoder pass gives the exact trained temporal phase.
        anchor = self._encode_boundary(self.boundary)
        generator = torch.Generator().manual_seed(seed)
        augmentation = MINIMAX_H3_KEYFRAME_NOISE_AUG
        history_noise = randn_tensor(self.history.shape, generator=generator, device=self.device, dtype=torch.float32)
        condition, metadata = pack_history(augmentation * self.history + (1 - augmentation) * history_noise,
                                           self.patch_size, self.sink_frames, self.mid_stride,
                                           self.recent_stride, self.sink_spatial, self.history_pooling)
        height, width = self.history.shape[3:]
        layout = continuation_layout(conditioning["text_token_tags"], self.history.shape[2], height, width,
                                     metadata, self.patch_size, self.target_latents)
        noise = randn_tensor((1, 24, self.target_latents, height, width), generator=generator, device=self.device, dtype=torch.float32)
        video = patchify_video_latents(noise, self.patch_size)
        anchor_rows = patchify_video_latents(anchor, self.patch_size)
        prefix_rows = anchor_rows.shape[0]
        noisy_anchor = augmentation * anchor_rows + (1 - augmentation) * video[:prefix_rows]
        self.scheduler.set_timesteps(steps + 1, device=self.device)
        indices = {name: getattr(layout, name).to(self.device) for name in
                   ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")}
        empty_audio = video.new_empty(1, 0, 32)
        prompt = conditioning["prompt_embeds"].to(self.device)
        for timestep in self.scheduler.timesteps:
            video[:prefix_rows] = noisy_anchor
            condition_time = max(float(timestep), augmentation)
            times, assignments = build_row_timesteps(layout, float(timestep), 1.0, condition_time, 1.0)
            row_times = times[assignments]
            target_indices = layout.video_indices[layout.num_condition_video_rows:]
            row_times[target_indices[:prefix_rows]] = condition_time
            times, assignments = torch.unique(row_times, sorted=True, return_inverse=True)
            prediction, _ = self.transformer(hidden_states=torch.cat((condition, video))[None],
                audio_hidden_states=empty_audio, encoder_hidden_states=prompt,
                timestep=times.to(self.device), timestep_indices=assignments.to(self.device),
                **indices, return_dict=False)
            velocity = prediction[0, layout.num_condition_video_rows:].float()
            video = self.scheduler.step(velocity, timestep, video).prev_sample
        video[:prefix_rows] = anchor_rows
        latent = unpatchify_video_tokens(video, self.target_latents, height, width, 24, self.patch_size)
        # Keep blocks offloaded from this decode until the next boundary encode.
        with self._vae_phase(hold=self.reuse_vae_phase), torch.autocast(self.device.type, dtype=torch.float16):
            decoder_input = self._latents(latent, inverse=True)
            if self.decode_layout == "continuous":
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                # The overlap target was encoded as one continuous clip. Native
                # video decoding inserts another temporal reset after latent 5.
                decoded = self.vae._decode_clip(decoder_input)[:, :, self.vae.frame_pre_padding:]
                end.record()
                self.pipeline.decoder_events.append((start, end))
            else:
                decoded = self.vae.decode(decoder_input, return_dict=False)[0]
        # Decode the head and future jointly; trim exactly the repeated head pixel.
        output = self._pixels(decoded.float(), inverse=True)[:, :, 1:1 + self.chunk_frames]
        if output.shape[2] != self.chunk_frames or not output.isfinite().all():
            raise RuntimeError("continuation decode returned an invalid frame count or nonfinite pixels")
        correction = None
        if self.stabilize_exposure:
            from .exposure import stabilize_exposure
            self.raw_output = output
            output, correction = stabilize_exposure(output, self.boundary[:, :, 0])
        self._update_history(output, latent[:, :, 1:])
        self.boundary = output[:, :, -1:].clone()
        if self.boundary_rgb8:
            self.boundary = self.boundary.mul(255).round().div_(255)
        torch.cuda.synchronize()
        timing = {"chunk_ms": (time.perf_counter() - started) * 1000,
            "dit_forward_ms": [a.elapsed_time(b) for a, b in self.pipeline.events],
            "actual_dit_forwards": len(self.pipeline.events), "history_latents": self.history.shape[2],
            "history_capacity": self.capacity, "condition_rows": layout.num_condition_video_rows,
            "sink_frames": self.sink_frames, "mid_stride": self.mid_stride,
            "recent_stride": self.recent_stride, "history_pooling": self.history_pooling,
            "sink_spatial_policy": "complementary_checkerboards" if self.sink_frames == 2 and self.sink_spatial == "shared" else "dense",
            "boundary_rgb8": self.boundary_rgb8,
            "boundary_posterior": self.boundary_posterior,
            "history_feedback": self.history_feedback,
            "history_rgb_frames": self.rgb_history.shape[2] if self.history_feedback == "native-rgb" else None,
            "packed_rows": layout.sequence_length, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "output_frames": self.chunk_frames, "target_latents": self.target_latents,
            "decode_layout": self.decode_layout, "exposure_postprocessing": self.stabilize_exposure,
            "vae_phase_offload": self.vae_offload,
            "reuse_vae_phase": self.reuse_vae_phase,
            "vae_phase_entries": self._phase_entries - phase_entries,
            "exposure_correction": correction}
        timing["dit_ms"] = sum(timing["dit_forward_ms"])
        timing["video_decoder_ms"] = sum(a.elapsed_time(b) for a, b in self.pipeline.decoder_events)
        if len(self.pipeline.events) != steps:
            raise RuntimeError("incorrect stream denoiser forward count")
        return output, timing
