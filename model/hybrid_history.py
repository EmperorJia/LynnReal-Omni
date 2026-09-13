"""Add bounded FramePack history beside an immutable Ref2VA image reference."""
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED, MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD, MINIMAX_H3_VIDEO_TAG,
    _ROPE_FRAME_RESCALE, unpatchify_video_tokens,
)
from .framepack import advance_history, continuation_layout, pack_history
from .offload import vae_phase


class HybridHistory:
    def __init__(self, capacity=64, sink_frames=2, mid_stride=8):
        if capacity < 7 or sink_frames not in (1, 2) or mid_stride < 4:
            raise ValueError('invalid bounded history policy')
        self.capacity, self.sink_frames, self.mid_stride = capacity, sink_frames, mid_stride
        self.seed = 0
        self.audit = {}

    @torch.inference_mode()
    def initialize(self, components, pixels):
        """Encode the actual source window once, preserving native video phases."""
        if pixels.shape[:2] != (1, 3) or (pixels.shape[2] - 5) % 17:
            raise ValueError('source history must contain 17*n+5 RGB frames')
        device = components._execution_device
        pixels = pixels.to(device, torch.float32)
        mean = pixels.new_tensor(MINIMAX_H3_PIXEL_MEAN).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor(MINIMAX_H3_PIXEL_STD).view(1, 3, 1, 1, 1)
        with vae_phase(components), torch.autocast('cuda', dtype=torch.float16):
            moments = components.vae._encode((pixels - mean) / std)
        posterior = DiagonalGaussianDistribution(moments)
        latent = posterior.sample(generator=torch.Generator().manual_seed(
            MINIMAX_H3_KEYFRAME_ENCODE_SEED)).to(torch.float16).float().cpu()
        mean = latent.new_tensor(components.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latent.new_tensor(components.vae.config.latents_std).view(1, -1, 1, 1, 1)
        self.history = (latent - mean) / std
        if not self.sink_frames < self.history.shape[2] <= self.capacity:
            raise ValueError('source history must fit the capacity and include frames after the sinks')
        self.sinks = self.history[:, :, :self.sink_frames].clone()

    @torch.no_grad()
    def inject(self, state):
        """Preserve native reference rows/RNG; insert history before generated rows."""
        video = state.get('latents')
        condition = int(state.get('num_condition_video_rows'))
        vi = state.get('video_indices')
        ai = state.get('audio_indices')
        ac = int(state.get('num_condition_audio_rows'))
        if condition < 1:
            raise ValueError('hybrid history requires an existing image reference')
        # Independent history noise leaves existing reference and target draws unchanged.
        noise = torch.randn(self.history.shape, generator=torch.Generator().manual_seed(self.seed))
        aug = MINIMAX_H3_KEYFRAME_NOISE_AUG
        rows, metadata = pack_history(aug*self.history + (1-aug)*noise,
            sink_frames=self.sink_frames, mid_stride=self.mid_stride, sink_spatial='dense')
        count = len(rows)
        height, width = self.history.shape[3:]
        layout = continuation_layout(torch.empty(0, dtype=torch.long), self.history.shape[2],
                                     height, width, metadata, target_latents=1)
        positions = state.get('position_ids')
        insert = min(int(vi[condition]), int(ai[ac]) if len(ai) > ac else len(positions))
        origin = positions[vi[condition], 0].clone()
        history_positions = layout.position_ids[:count].to(positions.device)
        history_positions[:, 0] += origin
        target_positions = positions[insert:].clone()
        target_positions[:, 0] += 4 * _ROPE_FRAME_RESCALE
        state.set('position_ids', torch.cat((positions[:insert], history_positions, target_positions)))
        tags = state.get('token_tags')
        state.set('token_tags', torch.cat((tags[:insert], tags.new_full((count,), MINIMAX_H3_VIDEO_TAG), tags[insert:])))
        shift = lambda indices: indices + (indices >= insert).to(indices.dtype)*count
        state.set('video_indices', torch.cat((vi[:condition],
            torch.arange(insert, insert+count, device=vi.device), shift(vi[condition:]))))
        state.set('audio_indices', shift(ai))
        state.set('text_indices', shift(state.get('text_indices')))
        plans = []
        for times, assignments in state.get('row_timestep_plan'):
            per_row = times[assignments]
            history_times = per_row[vi[0]].expand(count)
            plans.append(torch.unique(torch.cat((per_row[:insert], history_times, per_row[insert:])),
                                      sorted=True, return_inverse=True))
        state.set('row_timestep_plan', plans)
        state.set('latents', torch.cat((video[:condition], rows.to(video.device, video.dtype), video[condition:])))
        state.set('num_condition_video_rows', condition+count)
        self.audit = dict(history_rows=count, extra_transformer_rows=count, image_reference_rows=condition,
            stored_history_latents=self.history.shape[2], history_capacity=self.capacity,
            sink_frames=self.sink_frames, sink_spatial='dense', recent_stride=2,
            mid_stride=self.mid_stride, history_feedback='generated_latents',
            latest_history_latent_omitted=True, history_noise_seed=self.seed,
            history_time_start=float(history_positions[:, 0].min()),
            history_time_end=float(history_positions[:, 0].max()),
            target_head_time=float(origin + 4*_ROPE_FRAME_RESCALE))

    @torch.no_grad()
    def advance(self, state):
        start = int(state.get('num_condition_video_rows'))
        target = unpatchify_video_tokens(state.get('latents')[start:], state.get('num_latent_frames'),
            state.get('latent_height'), state.get('latent_width'), 24, (1, 2, 2))
        self.history = advance_history(self.history, target[:, :, 1:].float().cpu(),
                                       self.capacity, self.sink_frames)
        if not torch.equal(self.history[:, :, :self.sink_frames], self.sinks):
            raise RuntimeError('immutable sink latents changed')
