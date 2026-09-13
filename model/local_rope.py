"""Bounded H3 temporal coordinates inspired by DeepForcing and Infinity-RoPE.

Recomputed bidirectional attention: this is not causal KV-cache rebasing.
"""
import torch
from diffusers.modular_pipelines.minimax_h3.packing import _ROPE_FRAME_RESCALE


def local_history_clock(source_times, sink_count, head, compact_tail=False, age_horizon=0):
    """Keep sink spacing, place the tail after it, preserve recent/head distance."""
    source = torch.as_tensor(source_times, dtype=torch.float64)
    if (source.ndim != 1 or not 1 <= sink_count < len(source)
            or not torch.isfinite(source).all() or not torch.all(source[1:] > source[:-1])
            or not 0 < head-source[-1] <= 32):
        raise ValueError('invalid chronological history or recent/head separation')
    sink = source[:sink_count]-source[0]
    if age_horizon:
        if age_horizon < 32:
            raise ValueError("age_horizon must be at least 32 RGB time units")
        target_head = min(float(head-source[0]), float(age_horizon))
        tail = target_head-(head-source[sink_count:])
        if tail[0] <= sink[-1]:
            raise ValueError("age horizon is too small for retained recent history")
        return torch.cat((sink, tail)), torch.as_tensor(target_head, dtype=torch.float64)
    tail = source[sink_count:]-source[sink_count]+sink[-1]+4
    if compact_tail:
        # Fit compressed middle frames before the same recent slot. Adding
        # history must not shift the recent frame or prediction head forward.
        distance = source[-1]-source[sink_count:]
        tail = sink[-1]+4-3*distance/distance[0].clamp_min(1)
    history = torch.cat((sink, tail))
    return history, history[-1]+head-source[-1]


def apply_local_clock(layout, metadata, source_times, sink_count, head, compact_tail=False, age_horizon=0):
    text = len(layout.text_indices)
    target_start = text+layout.num_condition_video_rows
    old_target = layout.position_ids[target_start:, 0].clone()
    times, target_head = local_history_clock(
        [source_times[frame] for _, frame, _, _ in metadata], sink_count, head, compact_tail, age_horizon)
    cursor = text
    for time, (_, _, _, indices) in zip(times, metadata):
        layout.position_ids[cursor:cursor+len(indices), 0] = text+time*_ROPE_FRAME_RESCALE
        cursor += len(indices)
    # Target cadence is the native H3 temporal grid; only its origin changes.
    layout.position_ids[target_start:, 0] = old_target-old_target[0]+text+target_head*_ROPE_FRAME_RESCALE
    return times, target_head
