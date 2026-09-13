"""17-frame V2V sampling with dense history endpoints and optional fixed image refs."""
import math
import time
import torch
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG, MINIMAX_H3_VIDEO_TAG, MiniMaxH3PackedSequence,
    _ROPE_FRAME_RESCALE, _spatial_position_grid, _temporal_position_grid,
    build_row_timesteps, patchify_video_latents, unpatchify_video_tokens,
)
from diffusers.utils.torch_utils import randn_tensor
from .framepack import spatial_indices
from .stream import Stream


def history_plan(frames):
    """The causal_5plus17 training plan: dense endpoints, two 2x and up to eight 4x anchors."""
    if frames < 1:
        raise ValueError('history must contain at least one latent frame')
    newest = frames - 1
    plan = {0: 1, newest: 1}
    for age in (1, 2):
        if newest >= age:
            plan.setdefault(newest - age, 2)
    end = newest - 3
    if end >= 1:
        slots = min(8, end)
        for slot in range(slots):
            plan.setdefault(1 + round(slot * (end - 1) / max(slots - 1, 1)), 4)
    return tuple(sorted(plan.items()))


def pack_history(latents, patch_size):
    frames, height, width = latents.shape[2:]
    rows = patchify_video_latents(latents, patch_size).reshape(frames, -1, 24 * math.prod(patch_size))
    metadata = tuple(('history', frame, stride, spatial_indices(
        height // patch_size[1], width // patch_size[2], stride, latents.device).cpu())
        for frame, stride in history_plan(frames))
    packed = torch.cat([rows[frame, indices.to(latents.device)] for _, frame, _, indices in metadata])
    if len(packed) > 3 * rows.shape[1]:
        raise RuntimeError('history exceeds three dense latent frames')
    return packed, metadata


def build_layout(tags, frames, height, width, metadata, reference_shapes=(), patch_size=(1, 2, 2)):
    """Native image slots, then the trained three-latent history clock and five-latent future."""
    tags = tags.detach().cpu()
    text_rows = tags.numel()
    ph, pw = patch_size[1:]

    def grid(h, w):
        return torch.stack([x.flatten() for x in torch.meshgrid(
            _spatial_position_grid(h, ph, math.sqrt(h*w)),
            _spatial_position_grid(w, pw, math.sqrt(h*w)), indexing='ij')], -1)

    grids = [grid(h, w) for h, w in reference_shapes]
    target_grid = grid(height, width)
    condition_rows = sum(len(g) for g in grids) + sum(len(i) for _, _, _, i in metadata)
    target_start = text_rows + condition_rows
    length = target_start + 5 * len(target_grid)
    positions = torch.zeros(length, 3, dtype=torch.float64)
    positions[:text_rows, 0] = torch.arange(text_rows, dtype=torch.float64)
    cursor = text_rows
    for index, ref_grid in enumerate(grids):
        positions[cursor:cursor+len(ref_grid), 0] = float(text_rows + index)
        positions[cursor:cursor+len(ref_grid), 1:] = ref_grid
        cursor += len(ref_grid)
    origin = float(text_rows + len(grids))
    history_times = _temporal_position_grid(3, origin)
    for _, frame, _, indices in metadata:
        count = len(indices)
        positions[cursor:cursor+count, 0] = (float(history_times[0])
            + (float(history_times[-1])-float(history_times[0])) * frame/max(frames-1, 1))
        positions[cursor:cursor+count, 1:] = target_grid[indices]
        cursor += count
    future_times = _temporal_position_grid(5, origin + _ROPE_FRAME_RESCALE * 9)
    positions[target_start:, 0] = future_times[:, None].expand(-1, len(target_grid)).flatten()
    positions[target_start:, 1:] = target_grid.repeat(5, 1)
    token_tags = torch.full((length,), MINIMAX_H3_VIDEO_TAG, dtype=torch.long)
    token_tags[:text_rows] = tags.long()
    return MiniMaxH3PackedSequence(sequence_length=length, position_ids=positions, token_tags=token_tags,
        video_indices=torch.arange(text_rows, length), audio_indices=torch.empty(0, dtype=torch.long),
        text_indices=torch.arange(text_rows), num_condition_video_rows=condition_rows, num_condition_audio_rows=0)


class CausalStream(Stream):
    def __init__(self, pipeline, capacity=32, stabilize_exposure=True):
        super().__init__(pipeline, capacity, stabilize_exposure=stabilize_exposure,
                         vae_offload=True, sink_frames=1, chunk_frames=17, decode_layout='native')
        self.target_latents = 5
        self.references = []

    @torch.inference_mode()
    def add_reference(self, pixels):
        """Encode each fixed RGB reference once, independently of the sampling RNG."""
        if len(self.references) >= 2:
            raise ValueError('only the original first frame and first-chunk endpoint are allowed')
        self.references.append(self._encode_boundary(pixels.to(self.device, torch.float32)))

    @torch.inference_mode()
    def step(self, conditioning, seed=0, steps=4):
        if steps != 4:
            raise ValueError('this standard stream uses exactly four DiT evaluations')
        self.pipeline.events.clear()
        self.pipeline.decoder_events.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(seed)
        aug = MINIMAX_H3_KEYFRAME_NOISE_AUG
        history = self.history.clone()
        tail = history[:, :, -2:]
        noise = randn_tensor(tail.shape, generator=generator, device=self.device, dtype=tail.dtype)
        history[:, :, -2:] = aug * tail + (1-aug) * noise
        rows, metadata = pack_history(history, self.patch_size)
        height, width = history.shape[3:]
        # Keep the training history -> target random draw order unchanged by refs.
        noise = randn_tensor((1, 24, 5, height, width), generator=generator,
                            device=self.device, dtype=torch.float32)
        video = patchify_video_latents(noise, self.patch_size)
        reference_rows = []
        ref_rng = torch.Generator().manual_seed(seed)
        for ref in self.references:
            clean = patchify_video_latents(ref, self.patch_size)
            noise = randn_tensor(clean.shape, generator=ref_rng, device=self.device, dtype=clean.dtype)
            reference_rows.append(aug * clean + (1-aug) * noise)
        condition = torch.cat([*reference_rows, rows])
        layout = build_layout(conditioning['text_token_tags'], history.shape[2], height, width,
            metadata, [r.shape[3:] for r in self.references], self.patch_size)
        indices = {name: getattr(layout, name).to(self.device) for name in
                   ('position_ids', 'token_tags', 'video_indices', 'audio_indices', 'text_indices')}
        self.scheduler.set_timesteps(steps+1, device=self.device)
        prompt = conditioning['prompt_embeds'].to(self.device)
        empty_audio = video.new_empty(1, 0, 32)
        for timestep in self.scheduler.timesteps:
            times, assignments = build_row_timesteps(layout, float(timestep), 1., max(float(timestep), aug), 1.)
            prediction, _ = self.transformer(hidden_states=torch.cat((condition, video))[None],
                audio_hidden_states=empty_audio, encoder_hidden_states=prompt,
                timestep=times.to(self.device), timestep_indices=assignments.to(self.device),
                **indices, return_dict=False)
            video = self.scheduler.step(prediction[0, len(condition):].float(), timestep, video).prev_sample
        latent = unpatchify_video_tokens(video, 5, height, width, 24, self.patch_size)
        with self._vae_phase(), torch.autocast(self.device.type, dtype=torch.float16):
            decoded = self.vae.decode(self._latents(latent, inverse=True).contiguous(), return_dict=False)[0]
        output = self._pixels(decoded.float(), inverse=True)
        if output.shape[2] != 17 or not output.isfinite().all():
            raise RuntimeError('five future latents must decode to 17 finite RGB frames')
        self.raw_output = output
        correction = None
        if self.stabilize_exposure:
            from .exposure import stabilize_exposure
            output, correction = stabilize_exposure(output, self.boundary[:, :, 0])
        # Match train_v2v.advance_stream_condition: rolling history, no RGB re-encoding.
        self.history = torch.cat((self.history, latent), dim=2)[:, :, -self.capacity:].detach().contiguous()
        self.boundary = output[:, :, -1:].clone()
        torch.cuda.synchronize()
        forwards = self.pipeline.events
        if len(forwards) != steps:
            raise RuntimeError('incorrect DiT evaluation count')
        return output, dict(chunk_ms=(time.perf_counter()-started)*1000,
            dit_ms=sum(a.elapsed_time(b) for a, b in forwards), actual_dit_forwards=len(forwards),
            video_decoder_ms=sum(a.elapsed_time(b) for a, b in self.pipeline.decoder_events),
            packed_rows=layout.sequence_length, history_rows=len(rows), reference_rows=[len(r) for r in reference_rows],
            history_plan=[(f, s, len(i)) for _, f, s, i in metadata], input_history_latents=history.shape[2],
            reference_count=len(self.references), history_capacity=self.capacity, target_latents=5, output_frames=17,
            attention='bidirectional', target_head=False, exposure_correction=correction,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
