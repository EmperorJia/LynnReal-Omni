"""Native two-latent overlap with bounded sinks and optional latent refinement."""
import time
import torch
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG, _temporal_position_grid, _ROPE_FRAME_RESCALE,
    audio_latent_num_frames, build_row_timesteps, patchify_video_latents, unpatchify_video_tokens,
)
from diffusers.utils.torch_utils import randn_tensor
from .formal_stream import FormalStream
from .formal_audio_stream import add_audio_rows
from .framepack import advance_history
from .local_rope import apply_local_clock
from .stream_rope import SinkWindow
from .stream_second_pass import TwoPassScheduler


class NativeOverlapStream(FormalStream):
    def __init__(self, pipeline, local_rope=True, second_steps=0, second_sigmas=None, mid_frames=0, compact_mid_clock=False, noise_correlation=0., appearance_reference=None, appearance_strength=0., sink_frames=4, age_horizon=0, appearance_max_scale=None, coarse_anchor=0., rolling_history=False, joint_second_pass=False, second_appearance=True):
        if mid_frames not in (0, 2):
            raise ValueError('native overlap supports zero or two spatially compressed middle latents')
        if sink_frames not in (1,2,3,4):
            raise ValueError('sink_frames must be between one and four')
        # Shared native codec state requires seven stored latents even when
        # fewer immutable sinks are selected for transformer conditioning.
        self.storage_capacity=max(7,sink_frames+3+mid_frames)
        super().__init__(pipeline,capacity=self.storage_capacity,decoder='native',history_feedback='latent',
                         v2v_autocast=True,chunk_frames=17)
        self.age_horizon,self.appearance_max_scale=age_horizon,appearance_max_scale
        if not 0<=coarse_anchor<1 or (coarse_anchor and not second_steps):
            raise ValueError('coarse anchor requires a second pass and weight in [0,1)')
        self.coarse_anchor,self.rolling_history=coarse_anchor,rolling_history
        if joint_second_pass and not second_steps:
            raise ValueError('joint second pass requires additional denoising evaluations')
        self.joint_second_pass=joint_second_pass
        self.second_appearance=second_appearance
        self.sink_frames,self.target_latents=sink_frames,7
        self.prediction_frames,self.history_commit_latents=22,5
        self.local_rope,self.second_steps=local_rope,second_steps
        self.second_sigmas=second_sigmas
        self.noise_correlation=noise_correlation
        if compact_mid_clock and not local_rope:
            raise ValueError('compact middle clock requires local RoPE')
        self.compact_mid_clock=compact_mid_clock
        self.window=SinkWindow(sink_frames,'trained',mid_frames=mid_frames,mid_stride=2)
        self.previous_overlap=None
        self.records=[]
        if not 0 <= appearance_strength <= 1 or (appearance_strength and appearance_reference is None):
            raise ValueError('appearance constraint needs a reference and strength in [0,1]')
        self.appearance_reference=appearance_reference
        self.appearance_strength=appearance_strength

    @torch.inference_mode()
    def decode_window(self, normalized):
        with self._vae_phase(),torch.autocast('cuda',dtype=torch.float16):
            decoded=self.vae.decode(self._latents(normalized,inverse=True).contiguous(),return_dict=False)[0]
        if decoded.shape[2]!=22:
            raise RuntimeError('seven native latent frames must decode to 22 RGB frames')
        raw=decoded[:,:,:17]
        if self.previous_overlap is not None:
            # Same pre-clamp overlap operation as AutoencoderKLMiniMaxH3._decode.
            raw=self.vae._blend(self.previous_overlap,raw,self.vae.frame_overlap,dim=-3)
        self.previous_overlap=decoded[:,:,17:].clone()
        return self._pixels(raw.float(),inverse=True)

    @torch.inference_mode()
    def initialize_latents(self, latents):
        if latents.shape[:3]!=(1,24,7):raise ValueError('bootstrap must contain seven native video latents')
        self.history=latents.to(self.device,torch.float32)
        self.window.initialize(self.history,18,17)  # first undisplayed native section starts at RGB time17
        self.initial_latents=self.history.clone()
        pixels=self.decode_window(self.history)
        self.boundary=(pixels[:,:,-1:]*255).round()/255
        self.initial_sink=self.history[:,:,:self.sink_frames].clone()
        return pixels

    @torch.inference_mode()
    def step(self, conditioning, seed, trace):
        self.pipeline.events.clear();self.pipeline.decoder_events.clear()
        torch.cuda.synchronize();started=time.perf_counter()
        rng=torch.Generator().manual_seed(seed);aug=MINIMAX_H3_KEYFRAME_NOISE_AUG
        noisy_history=aug*self.history+(1-aug)*randn_tensor(self.history.shape,generator=rng,device=self.device,dtype=torch.float32)
        # The final two history latents occupy the two fixed target slots.
        condition,metadata=self.window.pack(noisy_history[:,:,:-1],self.patch_size)
        h,w=self.history.shape[-2:]
        layout=self.window.layout(conditioning['text_token_tags'],self.history.shape[2],h,w,
                                  metadata,self.patch_size,target_latents=7)
        if self.local_rope:
            times,head=apply_local_clock(layout,metadata,self.window.times,self.sink_frames,self.window.head,self.compact_mid_clock,self.age_horizon)
            for record,t in zip(self.window.records[-1]['history'],times):record['time_after_text']=float(t)
            target_start=len(layout.text_indices)+layout.num_condition_video_rows
            self.window.records[-1]['target_times_after_text']=((layout.position_ids[target_start::h*w//4,0]-len(layout.text_indices))/_ROPE_FRAME_RESCALE).tolist()
        head=patchify_video_latents(self.history[:,:,-2:],self.patch_size)
        noise=randn_tensor((1,24,7,h,w),generator=rng,device=self.device,dtype=torch.float32)
        video=patchify_video_latents(noise,self.patch_size)
        fixed=aug*head+(1-aug)*video[:len(head)]
        sched=TwoPassScheduler(12,self.second_steps,seed+3000000,head,trace,second_sigmas=self.second_sigmas,
                               initial_noise=video,noise_correlation=self.noise_correlation)
        if self.coarse_anchor:
            from .refinement_anchor import CoarseLatentAnchor
            sched.clean_constraint=CoarseLatentAnchor(7,h,w,patch_size=self.patch_size,
                weight=self.coarse_anchor,fixed_head_latents=2)
        total=4+self.second_steps
        sched.set_timesteps(total+1,self.device)
        # Match the native shift-12/video and shift-3/audio noise relationship.
        audio_grid=sched.second.sigmas/(4-3*sched.second.sigmas) if self.joint_second_pass else None
        audio_sched=TwoPassScheduler(3,self.second_steps,seed+4000000,
            frozen=not self.joint_second_pass,second_sigmas=audio_grid)
        audio_sched.set_timesteps(total+1,self.device)
        appearance=[]
        if self.appearance_strength:
            from .first_appearance import constrain_moments
            appearance=constrain_moments(sched,self.appearance_reference.to(self.device),
                self.appearance_strength,self.patch_size,fixed_head_latents=2,max_scale=self.appearance_max_scale,
                max_calls=None if self.second_appearance else 4)
        audio=randn_tensor((2*audio_latent_num_frames(22),32),
            generator=torch.Generator().manual_seed(seed+1000000),device=self.device,dtype=torch.float32)
        indices={name:getattr(layout,name).to(self.device) for name in
                 ('position_ids','token_tags','video_indices','audio_indices','text_indices')}
        prompt=conditioning['prompt_embeds'].to(self.device)
        for i,timestep in enumerate(sched.timesteps):
            video[:len(head)]=fixed
            ct=max(float(timestep),aug)
            ts,assignments=build_row_timesteps(layout,float(timestep),1.,ct,1.)
            rows=ts[assignments];rows[layout.video_indices[len(condition):len(condition)+len(head)]]=ct
            ts,assignments=torch.unique(rows,sorted=True,return_inverse=True)
            kwargs=dict(hidden_states=torch.cat((condition,video))[None],
                audio_hidden_states=video.new_empty(1,0,32),encoder_hidden_states=prompt,
                timestep=ts.to(self.device),timestep_indices=assignments.to(self.device),**indices,return_dict=False)
            at=audio_sched.timesteps[i]
            kwargs=add_audio_rows(kwargs,audio,float(at),h*w//4,target_latents=7)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred,ap=self.transformer(**kwargs)
            video=sched.step(pred[0,len(condition):].float(),timestep,video).prev_sample
            audio=audio_sched.step(ap[0].float(),at,audio).prev_sample
        video[:len(head)]=head
        latent=unpatchify_video_tokens(video,7,h,w,24,self.patch_size)
        if not torch.equal(latent[:,:,:2],self.history[:,:,-2:]) or not latent.isfinite().all():
            raise RuntimeError('overlap changed or prediction is nonfinite')
        pixels=self.decode_window(latent)
        future=latent[:,:,2:].clone()
        self.commit_future(future)
        self.boundary=(pixels[:,:,-1:]*255).round()/255
        torch.cuda.synchronize()
        if len(self.pipeline.events)!=total or sched.calls!=total or audio_sched.calls!=total:
            raise RuntimeError('denoiser forward count mismatch')
        record=dict(chunk=len(self.records)+1,actual_dit_forwards=total,first_pass_forwards=4,
            second_pass_forwards=self.second_steps,video_schedule=sched.record,audio_schedule=audio_sched.record,
            fixed_overlap_latents=2,new_latents=5,output_frames=17,right_context_rgb=5,
            dense_sink_latents=self.sink_frames,stored_latents=self.history.shape[2],history_rows=len(condition),
            mid_latents=sum(stride>1 for _,_,stride,_ in metadata),mid_stride=2,
            compact_mid_clock=self.compact_mid_clock,age_horizon=self.age_horizon,appearance_max_scale=self.appearance_max_scale,
            coarse_anchor=self.coarse_anchor,rolling_history=self.rolling_history,joint_second_pass=self.joint_second_pass,
            second_appearance=self.second_appearance,
            appearance_strength=self.appearance_strength,appearance_constraint=appearance,
            history_source_times=self.window.times.tolist(),next_head_source_frame=self.window.head,
            rgb_boundary_encodes=0,history_rgb_encodes=0,editor_forwards=0,
            chunk_ms=(time.perf_counter()-started)*1000,dit_ms=sum(a.elapsed_time(b) for a,b in self.pipeline.events))
        self.records.append(record)
        return pixels,future,record

    def commit_future(self, future):
        times=torch.cat((self.window.times,self.window.head+_temporal_position_grid(7,0.)[2:]/_ROPE_FRAME_RESCALE))
        keep=0 if self.rolling_history else self.sink_frames
        self.history=(torch.cat((self.history,future),2)[:,:,-self.storage_capacity:].detach().contiguous()
                      if self.rolling_history else advance_history(self.history,future,self.storage_capacity,keep))
        self.window.times=torch.cat((times[:keep],times[-(self.storage_capacity-keep):]))
        self.window.head+=17
        if (not self.rolling_history and not torch.equal(self.history[:,:,:self.sink_frames],self.initial_sink)) or self.history.shape[2]!=self.storage_capacity:
            raise RuntimeError('immutable sink or bounded storage violated')

    def reduce_sinks(self, sinks, latest_window):
        """Keep the initial anchor but restore recent history from the last full window."""
        if self.storage_capacity!=7 or not 1<=sinks<self.sink_frames or latest_window.shape!=self.initial_latents.shape:
            raise ValueError('sink reduction requires a complete seven-latent native window')
        latest_window=latest_window.to(self.device)
        if not torch.equal(latest_window[:,:,-2:],self.history[:,:,-2:]):
            raise ValueError('sink reduction cannot change the current overlap')
        times=self.window.head-17+_temporal_position_grid(7,0.)/_ROPE_FRAME_RESCALE
        self.history=torch.cat((self.initial_latents[:,:,:sinks],latest_window[:,:,-(7-sinks):]),2)
        self.window.times=torch.cat((self.window.times[:sinks],times[-(7-sinks):]))
        self.sink_frames=self.window.sinks=sinks
        self.initial_sink=self.initial_latents[:,:,:sinks].clone()

    @torch.inference_mode()
    def replay_chunk(self, future):
        """Restore a saved causal prefix without repeating transformer inference."""
        future=future.to(self.device)
        if future.shape[2]!=5:
            raise ValueError('saved native continuation must have five new latents')
        pixels=self.decode_window(torch.cat((self.history[:,:,-2:],future),2))
        self.commit_future(future)
        self.boundary=(pixels[:,:,-1:]*255).round()/255
        self.records.append(dict(replayed=True))
        return pixels
