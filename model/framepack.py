"""Head-overlap continuation with compressed history and a fixed boundary latent."""
import math
import torch
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_VIDEO_TAG, MiniMaxH3PackedSequence, _ROPE_FRAME_RESCALE,
    _spatial_position_grid, _temporal_position_grid, patchify_video_latents,
)

def history_plan(frames, sink_frames=1, mid_stride=4, recent_stride=2):
    """Retain initial anchors; omit the newest latent, replaced by an RGB head."""
    if sink_frames not in (1, 2) or frames <= sink_frames or mid_stride < 4 or recent_stride < 2:
        raise ValueError("history requires 1 or 2 sinks, a newer latent, and mid stride >= 4")
    newest = frames - 1
    plan = {frame: 1 for frame in range(sink_frames)}
    plan[newest] = 1
    for age in (1, 2):
        if newest - age >= 0:
            plan.setdefault(newest - age, recent_stride)
    end = newest - 3
    if end >= sink_frames:
        slots = min(8, end - sink_frames + 1)
        for slot in range(slots):
            plan.setdefault(sink_frames + round(slot * (end - sink_frames) / max(slots - 1, 1)), mid_stride)
    return tuple((frame, stride) for frame, stride in sorted(plan.items()) if frame != newest)

def spatial_indices(height, width, stride, device):
    if stride == 1:
        return torch.arange(height * width, device=device)
    ys = torch.arange(stride // 2, height, stride, device=device)
    xs = torch.arange(stride // 2, width, stride, device=device)
    if not ys.numel():
        ys = torch.tensor([height // 2], device=device)
    if not xs.numel():
        xs = torch.tensor([width // 2], device=device)
    return (ys[:, None] * width + xs[None, :]).flatten()

def pack_history(latents, patch_size=(1, 2, 2), sink_frames=1, mid_stride=4,
                 recent_stride=2, sink_spatial="shared", pooling="point"):
    if sink_spatial not in {"shared", "dense"} or pooling not in {"point", "mean"}:
        raise ValueError("unknown history spatial policy")
    frames, height, width = latents.shape[2:]
    rows = patchify_video_latents(latents, patch_size).reshape(frames, -1, latents.shape[1] * math.prod(patch_size))
    selected, metadata = [], []
    grid_h, grid_w = height // patch_size[1], width // patch_size[2]
    for frame, stride in history_plan(frames, sink_frames, mid_stride, recent_stride):
        indices = spatial_indices(height // patch_size[1], width // patch_size[2], stride, latents.device)
        if frame < sink_frames and sink_frames == 2 and sink_spatial == "shared":
            # Complementary checkerboards preserve two temporal anchors within
            # the original one-dense-frame sink budget, without extra DiT rows.
            indices = indices[((indices // grid_w + indices % grid_w) % 2) == frame]
        if pooling == "mean" and stride > 1:
            # Average patch vectors within each stride-sized spatial cell.
            # Keep the same sampled positions and row count as point compression.
            grid = rows[frame].reshape(grid_h, grid_w, -1).permute(2, 0, 1)[None]
            pooled = torch.nn.functional.avg_pool2d(grid, stride, stride, ceil_mode=True)
            selected.append(pooled[0, :, indices // grid_w // stride, indices % grid_w // stride].T)
        else:
            selected.append(rows[frame, indices])
        metadata.append(("history", frame, stride, indices.cpu()))
    packed = torch.cat(selected)
    dense_sinks = sink_spatial == "dense" and sink_frames == 2
    if len(packed) > (3 if dense_sinks else 2) * rows.shape[1]:
        raise ValueError("history exceeds its declared dense-latent token budget")
    baseline_rows = sum(grid_h * grid_w if stride == 1 else
                        max(1, (grid_h - stride // 2 + stride - 1) // stride) *
                        max(1, (grid_w - stride // 2 + stride - 1) // stride)
                        for _, stride in history_plan(frames))
    if not dense_sinks and len(packed) > baseline_rows:
        raise ValueError("history policy increases the original attention sequence")
    return packed, tuple(metadata)

def continuation_layout(text_tags, history_frames, height, width, metadata, patch_size=(1, 2, 2), target_latents=6):
    """Use the trained positive RoPE clock and [text | history | target] order."""
    tags = text_tags.cpu()
    text_rows = tags.numel()
    condition_rows = sum(indices.numel() for _, _, _, indices in metadata)
    patch_h, patch_w = patch_size[1:]
    rows_per_frame = (height // patch_h) * (width // patch_w)
    target_start = text_rows + condition_rows
    length = target_start + target_latents * rows_per_frame
    grid = torch.stack([x.flatten() for x in torch.meshgrid(
        _spatial_position_grid(height, patch_h, math.sqrt(height * width)),
        _spatial_position_grid(width, patch_w, math.sqrt(height * width)), indexing="ij")], -1)
    positions = torch.zeros(length, 3, dtype=torch.float64)
    positions[:text_rows, 0] = torch.arange(text_rows, dtype=torch.float64)
    reference_time = _temporal_position_grid(2, float(text_rows))
    cursor = text_rows
    for _, frame, _, indices in metadata:
        count = indices.numel()
        positions[cursor:cursor + count, 0] = (float(reference_time[0])
            + (float(reference_time[-1]) - float(reference_time[0])) * frame / max(history_frames - 2, 1))
        positions[cursor:cursor + count, 1:] = grid[indices]
        cursor += count
    target_time = _temporal_position_grid(target_latents, float(text_rows) + _ROPE_FRAME_RESCALE * 4)
    positions[target_start:, 0] = target_time[:, None].expand(-1, rows_per_frame).flatten()
    positions[target_start:, 1:] = grid.repeat(target_latents, 1)
    token_tags = torch.full((length,), MINIMAX_H3_VIDEO_TAG, dtype=torch.long)
    token_tags[:text_rows] = tags.long()
    return MiniMaxH3PackedSequence(sequence_length=length, position_ids=positions, token_tags=token_tags,
        video_indices=torch.arange(text_rows, length), audio_indices=torch.empty(0, dtype=torch.long),
        text_indices=torch.arange(text_rows), num_condition_video_rows=condition_rows, num_condition_audio_rows=0)

def advance_history(history, future, capacity, sink_frames=1):
    """Bound stored history while preserving the original sink and recent latents."""
    if not 1 <= sink_frames < capacity or future.shape[:2] != history.shape[:2] or future.shape[3:] != history.shape[3:]:
        raise ValueError("invalid history capacity or latent geometry")
    combined = torch.cat((history, future), dim=2)
    if combined.shape[2] > capacity:
        combined = torch.cat((combined[:, :, :sink_frames], combined[:, :, -(capacity - sink_frames):]), dim=2)
    return combined.detach().contiguous()
