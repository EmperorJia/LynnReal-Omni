"""Formal head-tail V2V, with an image-only fixed head and optional appearance refs."""
import math
import time
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED, MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_VIDEO_TAG, _spatial_position_grid, build_row_timesteps,
    patchify_video_latents, unpatchify_video_tokens,
)
from diffusers.utils.torch_utils import randn_tensor
from .framepack import pack_history, continuation_layout, advance_history
from .stream import Stream


def prepend_references(layout, reference_shapes, patch_size=(1, 2, 2)):
    """Add native integer-time image slots before the unchanged formal history/target clock."""
    text = len(layout.text_indices)
    ph, pw = patch_size[1:]
    positions = []
    for index, (h, w) in enumerate(reference_shapes):
        grid = torch.stack([v.flatten() for v in torch.meshgrid(
            _spatial_position_grid(h, ph, math.sqrt(h*w)),
            _spatial_position_grid(w, pw, math.sqrt(h*w)), indexing='ij')], -1)
        positions.append(torch.cat((torch.full((len(grid), 1), float(text+index), dtype=torch.float64), grid), -1))
    if not positions:
        return layout
    refs = torch.cat(positions)
    tail = layout.position_ids[text:].clone()
    tail[:, 0] += len(reference_shapes)
    layout.position_ids = torch.cat((layout.position_ids[:text], refs, tail))
    layout.token_tags = torch.cat((layout.token_tags[:text],
        torch.full((len(refs),), MINIMAX_H3_VIDEO_TAG, dtype=torch.long), layout.token_tags[text:]))
    layout.sequence_length += len(refs)
    layout.video_indices = torch.arange(text, layout.sequence_length)
    layout.num_condition_video_rows += len(refs)
    return layout


class FormalStream(Stream):
    def __init__(self, pipeline, capacity=192, decoder='context', audit_decoder=False, history_feedback='latent', v2v_autocast=False, history_clock='trained', chunk_frames=17):
        super().__init__(pipeline, capacity=capacity, vae_offload=True, sink_frames=1,
                         chunk_frames=chunk_frames, decode_layout='continuous')
        if chunk_frames != 17 and history_feedback != 'latent':
            raise ValueError('native-tail feedback requires 17-frame chunks')
        self.prediction_frames = chunk_frames
        self.history_commit_latents = self.target_latents-1
        if chunk_frames in (33, 50):
            if decoder != 'native':
                raise ValueError('buffered native windows require the native decoder')
            # Deliver whole native RGB sections, excluding the known head.
            # Two genuinely predicted right-context latents support the last
            # section. They are never committed as already delivered history.
            self.target_latents = 5*((chunk_frames+1)//17)+2
            self.history_commit_latents = self.target_latents-3
            self.prediction_frames = chunk_frames+5
        if decoder not in {'context', 'continuous', 'native'}:
            raise ValueError('unknown decoder schedule')
        self.decoder, self.audit_decoder = decoder, audit_decoder
        self.references = []
        self.reference_time_offset = 0
        self.next_anchor = None
        if history_feedback not in {'latent','native-tail'}:
            raise ValueError('unknown history feedback mode')
        self.formal_feedback = history_feedback
        self.v2v_autocast = v2v_autocast
        if history_clock not in {'trained', 'recent'}:
            raise ValueError('unknown history clock')
        self.history_clock = history_clock

    @torch.inference_mode()
    def initialize(self, pixels, appearance=None):
        super().initialize(pixels, appearance)
        self.tail_rgb = pixels[:,:,-5:].to(self.device).clone()

    def _boundary_in_phase(self, pixels):
        with torch.autocast(self.device.type, enabled=False):
            moments = self.vae._encode_clip(self._pixels(pixels.float()))
        latent = DiagonalGaussianDistribution(moments).sample(
            generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
        return self._latents(latent.to(torch.float16).float())

    @torch.inference_mode()
    def add_reference(self, pixels):
        if len(self.references) == 2:
            raise ValueError('the two appearance references are immutable')
        self.references.append(self._encode_boundary(pixels.to(self.device, torch.float32)))

    @torch.inference_mode()
    def step(self, conditioning, seed=0, steps=4):
        if steps < 1:
            raise ValueError('denoising evaluations must be positive')
        self.pipeline.events.clear()
        self.pipeline.decoder_events.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        anchor = self.next_anchor if self.next_anchor is not None else self._encode_boundary(self.boundary)
        generator = torch.Generator().manual_seed(seed)
        aug = MINIMAX_H3_KEYFRAME_NOISE_AUG
        noise = randn_tensor(self.history.shape, generator=generator, device=self.device, dtype=torch.float32)
        history, metadata = pack_history(aug*self.history+(1-aug)*noise, self.patch_size)
        height, width = self.history.shape[3:]
        layout = continuation_layout(conditioning['text_token_tags'], self.history.shape[2],
                                     height, width, metadata, self.patch_size, target_latents=self.target_latents)
        if self.history_clock == 'recent':
            from diffusers.modular_pipelines.minimax_h3.packing import _ROPE_FRAME_RESCALE
            # Keep the last two native history latents four RGB frames apart.
            # Older anchors alone occupy a compact interval; target spacing is unchanged.
            text = len(layout.text_indices)
            newest = self.history.shape[2]-1
            cursor = text
            for _, frame, _, indices in metadata:
                age = newest-frame
                pixel_time = 12-4*age if age <= 2 else 3*frame/max(newest-3,1)
                layout.position_ids[cursor:cursor+len(indices),0] = text+_ROPE_FRAME_RESCALE*pixel_time
                cursor += len(indices)
            layout.position_ids[cursor:,0] += _ROPE_FRAME_RESCALE*8
        layout = prepend_references(layout, [r.shape[3:] for r in self.references], self.patch_size)
        if self.reference_time_offset:
            layout.position_ids[len(layout.text_indices):,0] += self.reference_time_offset
        # Ref draws use a separate RNG; history -> target draws match formal sampling.
        ref_rng = torch.Generator().manual_seed(seed)
        refs = []
        for ref in self.references:
            clean = patchify_video_latents(ref, self.patch_size)
            noise = randn_tensor(clean.shape, generator=ref_rng, device=self.device, dtype=clean.dtype)
            refs.append(aug*clean+(1-aug)*noise)
        condition = torch.cat([*refs, history])
        noise = randn_tensor((1,24,self.target_latents,height,width), generator=generator, device=self.device, dtype=torch.float32)
        video = patchify_video_latents(noise, self.patch_size)
        head = patchify_video_latents(anchor, self.patch_size)
        fixed = aug*head+(1-aug)*video[:len(head)]
        self.scheduler.set_timesteps(steps+1, device=self.device)
        indices = {name:getattr(layout,name).to(self.device) for name in
                   ('position_ids','token_tags','video_indices','audio_indices','text_indices')}
        prompt = conditioning['prompt_embeds'].to(self.device)
        for timestep in self.scheduler.timesteps:
            video[:len(head)] = fixed
            condition_time = max(float(timestep), aug)
            times, assignments = build_row_timesteps(layout, float(timestep), 1., condition_time, 1.)
            per_row = times[assignments]
            per_row[layout.video_indices[len(condition):len(condition)+len(head)]] = condition_time
            times, assignments = torch.unique(per_row, sorted=True, return_inverse=True)
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.v2v_autocast):
                prediction, _ = self.transformer(hidden_states=torch.cat((condition,video))[None],
                    audio_hidden_states=video.new_empty(1,0,32), encoder_hidden_states=prompt,
                    timestep=times.to(self.device), timestep_indices=assignments.to(self.device), **indices, return_dict=False)
            video = self.scheduler.step(prediction[0,len(condition):].float(), timestep, video).prev_sample
        video[:len(head)] = head
        latents = unpatchify_video_tokens(video,self.target_latents,height,width,24,self.patch_size)
        if not latents.isfinite().all():
            raise RuntimeError('nonfinite predicted latents')
        decoder_events = []
        with self._vae_phase():
            z = self._latents(latents,inverse=True).contiguous()
            outputs = {}
            modes = tuple(dict.fromkeys((self.decoder,'native','continuous'))) if self.audit_decoder else (self.decoder,)
            for mode in modes:
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.autocast(self.device.type,dtype=torch.float16):
                    if mode=='native':
                        decoded = self.vae.decode(z,return_dict=False)[0]
                    else:
                        # The decoder needs right context even though no future RGB is known.
                        # Keep a continuous output clock; native decode removes padding at
                        # every 17-frame section and skips positions inside this canvas.
                        canvas = (torch.cat((z,z[:,:,-1:].expand(-1,-1,self.vae.token_overlap,-1,-1)),2)
                                  if mode=='context' else z)
                        decoded = self.vae._decode_clip(canvas)[:,:,self.vae.frame_pre_padding:]
                end.record()
                outputs[mode] = self._pixels(decoded.float(),inverse=True)[:,:,1:1+self.chunk_frames]
                if mode==self.decoder:
                    decoder_events.append((start,end))
            output = outputs[self.decoder]
            if output.shape[2]!=self.chunk_frames or not output.isfinite().all():
                raise RuntimeError('head-tail decode delivered an invalid future window')
            # Prefetch the next exact image anchor during this same VAE phase.
            delivered = (output*255).round()/255
            self.boundary = delivered[:,:,-1:].clone()
            self.next_anchor = self._boundary_in_phase(self.boundary)
            if self.formal_feedback=='native-tail':
                # Refresh the trailing native 5+17 RGB window. Its seven
                # latents replace the previous two and add five new ones;
                # continuous target latents have a different tail phase.
                window = torch.cat((self.tail_rgb,delivered),2)
                encoded = self._encode_history(window)
                if encoded.shape[2]!=7:
                    raise RuntimeError('native 22-frame history must contain seven latents')
                self.history = advance_history(self.history[:,:,:-2],encoded,self.capacity,1)
                self.tail_rgb = delivered[:,:,-5:].clone()
        self.decoder_outputs = outputs
        if self.formal_feedback=='latent':
            self.history = advance_history(self.history,latents[:,:,1:1+self.history_commit_latents],self.capacity,1)
        torch.cuda.synchronize()
        events = self.pipeline.events
        if len(events)!=steps:
            raise RuntimeError('incorrect denoiser forward count')
        return output,dict(chunk_ms=(time.perf_counter()-started)*1000,
            dit_ms=sum(a.elapsed_time(b) for a,b in events),actual_dit_forwards=len(events),
            video_decoder_ms=sum(a.elapsed_time(b) for a,b in decoder_events),
            history_rows=len(history),reference_rows=[len(r) for r in refs],packed_rows=layout.sequence_length,
            target_latents=self.target_latents,fixed_head_latents=1,output_frames=self.chunk_frames,reference_count=len(refs),
            prediction_frames=self.prediction_frames,history_commit_latents=self.history_commit_latents,
            generated_decoder_context_latents=self.target_latents-1-self.history_commit_latents,
            decoder=self.decoder,decoder_audit=self.audit_decoder,boundary_prefetched=True,
            decoder_right_context=self.vae.token_overlap if self.decoder=='context' else 0,
            history_feedback=self.formal_feedback,
            history_clock=self.history_clock,
            reference_time_offset=self.reference_time_offset,
            v2v_autocast=self.v2v_autocast,
            exposure_postprocessing=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
