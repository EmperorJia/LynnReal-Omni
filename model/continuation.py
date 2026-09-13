"""Joint video/audio continuation with the unified model's dense-prefix layout."""
import time
import torch
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG, MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD,
    align_num_frames, audio_latent_num_frames, video_latent_num_frames,
    build_packed_sequence, build_row_timesteps, keyframe_condition_noise,
    patchify_video_latents, unpatchify_video_tokens, unpack_audio_tokens,
)
from diffusers.utils.torch_utils import randn_tensor
from .offload import vae_phase, temporarily_on_cpu, staged_transformer


@torch.inference_mode()
def continue_video(pipeline, conditioning, pixels, frames, seed=0, steps=4,
                   offload_blocks=18, posterior_mode=True, latent_history=None, condition_time=0.999, retain_anchor=False, history_filter=0.0):
    """Consume [1,3,17*n+5,H,W] source RGB; return only new RGB and stereo audio.

    The prefix and future share the native continuous video rotary clock.
    Audio contains future rows only, offset by the prefix duration in RoPE.
    Decode video jointly before removing prefix pixels; audio needs no prefix trim.
    """
    if pixels.ndim != 5 or tuple(pixels.shape[:2]) != (1, 3):
        raise ValueError('source pixels must have shape [1,3,T,H,W]')
    prefix_frames = pixels.shape[2]
    if prefix_frames < 5 or align_num_frames(prefix_frames) != prefix_frames or min(frames, steps) < 1:
        raise ValueError('positive future length and native 17*n+5 source frames required')
    if pipeline.reference or pipeline.config['variant'] != 'standard':
        raise ValueError('dense continuation requires the standard unified transformer')
    if not 0 <= history_filter <= 1:
        raise ValueError('history filter must lie in [0, 1]')
    if not 0 < condition_time <= 1:
        raise ValueError('condition time must lie in (0, 1]')
    pipe, transformer = pipeline.pipe, pipeline.transformer
    device, patch = transformer.device, tuple(transformer.config.patch_size)
    pixels = pixels.to(device, torch.float32)
    pixel_mean = pixels.new_tensor(MINIMAX_H3_PIXEL_MEAN).view(1, 3, 1, 1, 1)
    pixel_std = pixels.new_tensor(MINIMAX_H3_PIXEL_STD).view(1, 3, 1, 1, 1)
    mean = pixels.new_tensor(pipe.vae.config.latents_mean).view(1, 24, 1, 1, 1)
    std = pixels.new_tensor(pipe.vae.config.latents_std).view(1, 24, 1, 1, 1)
    pipeline.events.clear(); pipeline.decoder_events.clear()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.perf_counter()
    cached_prefix = latent_history is not None and 'prefix' in latent_history
    if cached_prefix:
        prefix = latent_history['prefix'].to(device)
    else:
        with vae_phase(pipe), torch.autocast(device.type, dtype=torch.float16):
            posterior = pipe.vae.encode((pixels - pixel_mean) / pixel_std).latent_dist
        prefix = (posterior.mode() if posterior_mode else posterior.sample(
            generator=torch.Generator().manual_seed(42))).float()
        prefix = (prefix - mean) / std
    prefix_stats = {'mean': prefix.mean((0, 2, 3, 4)).tolist(),
                    'std': prefix.std((0, 2, 3, 4)).tolist(),
                    'abs_max': float(prefix.abs().max())}
    if cached_prefix and history_filter:
        import torch.nn.functional as F
        b, c, t, hh, ww = prefix.shape
        flat = prefix.permute(0, 2, 1, 3, 4).reshape(b*t, c, hh, ww)
        smooth = F.avg_pool2d(F.pad(flat, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
        smooth = smooth.reshape(b, t, c, hh, ww).permute(0, 2, 1, 3, 4)
        prefix = prefix.lerp(smooth, history_filter)
    combined_frames = align_num_frames(prefix_frames + frames)
    total_latents = video_latent_num_frames(combined_frames)
    p, h, w = prefix.shape[2:]
    if p != video_latent_num_frames(prefix_frames):
        raise RuntimeError('encoder temporal geometry differs from native prefix contract')
    clean_condition = patchify_video_latents(prefix, patch)
    anchor_rows = 0
    if retain_anchor:
        if latent_history is None:
            raise ValueError('a fixed anchor requires latent history')
        if 'sink' not in latent_history:
            latent_history['sink'] = prefix[:, :, :1].detach().clone()
        sink = patchify_video_latents(latent_history['sink'].to(device), patch)
        anchor_rows = len(sink)
        clean_condition = torch.cat((sink, clean_condition))
    generator = torch.Generator().manual_seed(seed)
    noise = keyframe_condition_noise(((p+int(retain_anchor), h, w),), patch, 24, generator=generator,
                                    device=device, dtype=clean_condition.dtype)
    condition = pipe.scheduler.scale_noise(clean_condition, condition_time, noise)
    video = patchify_video_latents(randn_tensor((1, 24, total_latents-p, h, w),
        generator=generator, device=device, dtype=torch.float32), patch)
    num_audio = audio_latent_num_frames(combined_frames - prefix_frames)
    audio = randn_tensor((2*num_audio, 32), generator=generator, device=device, dtype=torch.float32)
    layout = build_packed_sequence(conditioning['text_token_tags'], total_latents+int(retain_anchor), h, w, num_audio, patch)
    layout.num_condition_video_rows = len(condition)
    layout.position_ids[layout.audio_indices, 0] += audio_latent_num_frames(prefix_frames)
    indices = {k: getattr(layout, k).to(device) for k in
               ('token_tags', 'position_ids', 'video_indices', 'audio_indices', 'text_indices')}
    pipe.scheduler.set_timesteps(steps+1, device=device)
    pipe.audio_scheduler.set_timesteps(steps+1, device=device)
    prompt = conditioning['prompt_embeds'].to(device)
    with temporarily_on_cpu(pipe.vae.decoder), staged_transformer(pipe, offload_blocks):
        for i, timestep in enumerate(pipe.scheduler.timesteps):
            at = pipe.audio_scheduler.timesteps[i]
            times, assignments = build_row_timesteps(layout, float(timestep), float(at),
                max(float(timestep), condition_time), 1.0)
            v, a = transformer(hidden_states=torch.cat((condition, video))[None],
                audio_hidden_states=audio[None], encoder_hidden_states=prompt,
                timestep=times.to(device), timestep_indices=assignments.to(device),
                **indices, return_dict=False)
            video = pipe.scheduler.step(v[0, len(condition):].float(), timestep, video).prev_sample
            audio = pipe.audio_scheduler.step(a[0].float(), at, audio).prev_sample
            print(f'Continuation DiT: {i+1}/{steps}', flush=True)
    latent = unpatchify_video_tokens(torch.cat((clean_condition[anchor_rows:], video)), total_latents, h, w, 24, patch)
    if latent_history is not None:
        # Advancing by 17*n RGB frames preserves the five-latent codec phase.
        # Carry generated latents directly; repeated RGB encoding amplifies errors.
        if frames % 17:
            raise ValueError('latent feedback requires a multiple of 17 future frames')
        latent_history['prefix'] = latent[:, :, -p:].detach().contiguous()
    with vae_phase(pipe), torch.autocast(device.type, dtype=torch.float16):
        rgb = pipe.vae.decode(latent * std + mean, return_dict=False)[0]
    rgb = (rgb.float()*pixel_std+pixel_mean).clamp(0, 1)[:, :, prefix_frames:prefix_frames+frames]
    audio_latent = unpack_audio_tokens(audio, num_audio)
    amean = audio.new_tensor(pipe.audio_vae.config.latents_mean).view(1, 32, 1)
    astd = audio.new_tensor(pipe.audio_vae.config.latents_std).view(1, 32, 1)
    with vae_phase(pipe):
        waveform = pipe.audio_vae.decode(audio_latent*astd+amean, return_dict=False)[0]
    waveform = waveform.float().permute(1, 0, 2)
    requested_samples = round(frames*32000/24)
    # The trained 40-Hz audio grid rounds its length; a 24-fps duration may
    # exceed it by at most half an 800-sample hop. Pad only that codec tail.
    tail_padding = max(0, requested_samples-waveform.shape[-1])
    if tail_padding > 400:
        raise RuntimeError(f'audio decoder is unexpectedly short by {tail_padding} samples')
    waveform = torch.nn.functional.pad(waveform, (0, tail_padding))[..., :requested_samples]
    torch.cuda.synchronize()
    timing = {'generation_and_decode_ms': (time.perf_counter()-started)*1000,
        'actual_dit_forwards': len(pipeline.events),
        'dit_forward_ms': [a.elapsed_time(b) for a,b in pipeline.events],
        'video_decoder_ms': sum(a.elapsed_time(b) for a,b in pipeline.decoder_events),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'condition_video_rows': len(condition), 'target_audio_rows': len(audio),
        'history_filter': history_filter, 'prefix_stats': prefix_stats,
        'fixed_anchor_latents': int(retain_anchor), 'condition_time': condition_time, 'prefix_frames': prefix_frames, 'model_future_frames': combined_frames-prefix_frames,
        'prefix_posterior': 'mode' if posterior_mode else 'seed42_sample',
        'prefix_source': 'generated_latent_history' if cached_prefix else 'encoded_source_video',
        'audio_codec_tail_padding_samples': tail_padding}
    timing['dit_ms'] = sum(timing['dit_forward_ms'])
    if len(pipeline.events) != steps or rgb.shape[2] != frames or waveform.shape[-1] != round(frames*32000/24):
        raise RuntimeError(f'incorrect continuation geometry: NFE={len(pipeline.events)}, RGB={tuple(rgb.shape)}, audio={tuple(waveform.shape)}')
    if not torch.isfinite(rgb).all() or not torch.isfinite(waveform).all():
        raise RuntimeError('non-finite continuation output')
    return rgb, waveform, timing
