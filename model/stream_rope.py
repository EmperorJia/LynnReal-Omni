"""Experimental full-resolution sinks and bounded temporal coordinates for H3.

Only temporal position IDs change. These are H3 adaptations of Deep Sink and
Block-Relativistic RoPE, not their causal KV-cache implementations.
"""
import math
import torch
from diffusers.modular_pipelines.minimax_h3.packing import (
    _ROPE_FRAME_RESCALE, _temporal_position_grid, patchify_video_latents,
)
from .framepack import advance_history, continuation_layout, spatial_indices


class SinkWindow:
    def __init__(self, sinks=4, rope='trained', horizon=32, mid_frames=0, mid_stride=2):
        if sinks < 1 or rope not in {'trained', 'translated', 'deep', 'relative'} or horizon < 20:
            raise ValueError('invalid sink/window configuration')
        self.sinks, self.rope, self.horizon = sinks, rope, horizon
        if mid_frames < 0 or mid_stride not in (2, 4):
            raise ValueError('invalid compressed middle-history configuration')
        self.mid_frames, self.mid_stride = mid_frames, mid_stride
        self.records = []

    def initialize(self, history, source_frames, chunk_frames):
        if history.shape[2] < self.sinks + 2:
            raise ValueError('bootstrap needs sinks, a recent latent and the replaced boundary')
        self.times = _temporal_position_grid(history.shape[2], 0.) / _ROPE_FRAME_RESCALE
        self.head = source_frames - 1
        self.chunk_frames = chunk_frames

    def pack(self, latents, patch_size=(1, 2, 2)):
        frames, height, width = latents.shape[2:]
        rows = patchify_video_latents(latents, patch_size).reshape(
            frames, -1, latents.shape[1] * math.prod(patch_size))
        indices = torch.arange(rows.shape[1], device=latents.device)
        # The newest latent is represented by the independently encoded RGB head.
        middle = list(range(max(self.sinks, frames-2-self.mid_frames), frames-2))
        selected = list(range(self.sinks)) + middle + [frames-2]
        if len(set(selected)) != len(selected):
            raise ValueError('sink/recent overlap')
        sparse = spatial_indices(height//patch_size[1], width//patch_size[2], self.mid_stride, latents.device)
        metadata = tuple(('history', frame, self.mid_stride if frame in middle else 1,
                          (sparse if frame in middle else indices).cpu()) for frame in selected)
        return torch.cat([rows[frame, ids.to(rows.device)] for _, frame, _, ids in metadata]), metadata

    def layout(self, text_tags, history_frames, height, width, metadata,
               patch_size=(1, 2, 2), target_latents=6):
        layout = continuation_layout(text_tags, history_frames, height, width,
                                     metadata, patch_size, target_latents)
        text = len(layout.text_indices)
        target_start = text + layout.num_condition_video_rows
        rows_per_frame = height * width // (patch_size[1] * patch_size[2])
        original_target = layout.position_ids[target_start:, 0].clone()
        selected = [frame for _, frame, _, _ in metadata]
        if self.rope == 'translated':
            layout.position_ids[text:, 0] += (self.horizon-4) * _ROPE_FRAME_RESCALE
        elif self.rope in {'deep', 'relative'}:
            times = self.times[selected].clone()
            if self.rope == 'deep':
                # Deep Forcing Eq. (5): align the final sink with the first tail
                # timestamp, preserving spacing inside the immutable sink block.
                times[:self.sinks] += times[self.sinks] - times[self.sinks-1]
            times = (times-self.head+self.horizon).clamp_min(0)
            cursor = text
            for t, (_, _, _, indices) in zip(times, metadata):
                layout.position_ids[cursor:cursor+len(indices), 0] = text+t*_ROPE_FRAME_RESCALE
                cursor += len(indices)
            offsets = _temporal_position_grid(target_latents, 0.)
            layout.position_ids[target_start:, 0] = (
                text+self.horizon*_ROPE_FRAME_RESCALE+offsets[:, None]
            ).expand(-1, rows_per_frame).flatten()
        target = layout.position_ids[target_start:, 0]
        if not torch.allclose(target-target[0], original_target-original_target[0]):
            raise RuntimeError('temporal remapping changed the prediction cadence')
        cursor, positions = text, []
        for frame, (_, _, _, indices) in zip(selected, metadata):
            positions.append(dict(history_index=frame, source_frame=float(self.times[frame]),
                                  time_after_text=float((layout.position_ids[cursor, 0]-text)/_ROPE_FRAME_RESCALE)))
            cursor += len(indices)
        self.records.append(dict(head_source_frame=self.head, history=positions,
            target_times_after_text=((target[::rows_per_frame]-text)/_ROPE_FRAME_RESCALE).tolist(),
            note='Ref is prepended later at its unchanged native image slot; audio follows target time.'))
        return layout

    def advance(self, history, future, capacity, sink_frames=1):
        count = future.shape[2]
        offsets = _temporal_position_grid(count+1, 0.)[1:] / _ROPE_FRAME_RESCALE
        times = torch.cat((self.times, self.head+offsets))
        if len(times) > capacity:
            times = torch.cat((times[:self.sinks], times[-(capacity-self.sinks):]))
        self.times = times
        self.head += self.chunk_frames
        return advance_history(history, future, capacity, self.sinks)
