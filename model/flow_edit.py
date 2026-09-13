"""Experimental H3 adaptation of FlowEdit's source-to-target velocity difference.

Video starts from its encoded source. Both velocity queries share each noise
draw and the same audio state. H3 uses increasing clean-time, so dt is positive.
This is a few-step video adaptation, not the paper's validated image settings.
"""
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.modular_pipelines.minimax_h3.packing import build_row_timesteps
from diffusers.modular_pipelines.minimax_h3.packing_ref2va import build_ref2va_packed_sequence
from .offload import StagedReferenceEncoder, StagedReferenceDenoise, temporarily_on_cpu


class SourceVideoEncoder(StagedReferenceEncoder):
    def encode_references(self, components, references, device=None):
        video, audio = super().encode_references(components,references,device)
        cursor, sources = 0, []
        for reference in references:
            if reference.kind == 'video':
                sources.append(video[cursor:cursor+reference.num_video_rows].clone())
            if reference.kind != 'audio':
                cursor += reference.num_video_rows
        if len(sources) != 1:
            raise ValueError('FlowEdit requires exactly one source video')
        self.source_rows = sources[0]
        return video, audio


class FlowEditDenoise(StagedReferenceDenoise):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.source_conditioning = None
        self.mode = 'difference'
        self.seed = 0
        self.audit = {}

    @torch.no_grad()
    def __call__(self, components, state):
        device = components._execution_device
        video = state.get('latents')
        audio = state.get('audio_latents')
        nv, na = state.get('num_condition_video_rows'), state.get('num_condition_audio_rows')
        source = self.encoder.source_rows.to(device=device,dtype=torch.float32)
        if source.shape != video[nv:].shape or na:
            raise ValueError('FlowEdit requires matching source/target geometry and silent references')
        edited = source.clone()
        conditioning = self.source_conditioning
        target_prompt = state.get('prompt_embeds').to(device)
        source_prompt = conditioning['prompt_embeds'].to(device)
        references = state.get('prepared_references')
        shape = [state.get(k) for k in ('num_latent_frames','latent_height','latent_width','num_audio_latents')]
        source_layout = build_ref2va_packed_sequence(conditioning['text_token_tags'],references,*shape,components.patch_size)
        target_layout = state.get('layout')
        if self.mode == 'identity':
            source_prompt, source_layout = target_prompt, target_layout
        layouts = [source_layout,target_layout]
        indices = [{name:getattr(layout,name).to(device) for name in
                    ('position_ids','token_tags','video_indices','audio_indices','text_indices')} for layout in layouts]
        components.scheduler.set_timesteps(5,device=device)
        components.audio_scheduler.set_timesteps(5,device=device)
        rng = torch.Generator().manual_seed(self.seed+3000000)
        records = []
        with temporarily_on_cpu(components.vae.decoder):
            for index, time in enumerate(components.scheduler.timesteps):
                sigma = components.scheduler.sigmas[index]
                sigma_next = components.scheduler.sigmas[index+1]
                audio_time = components.audio_scheduler.timesteps[index]
                noise = randn_tensor(source.shape,generator=rng,device=device,dtype=torch.float32)
                noisy_source = float(time)*source + sigma*noise
                noisy_target = noisy_source + (edited-source)

                def velocity(rows, prompt, branch):
                    times, assignments = build_row_timesteps(layouts[branch],float(time),float(audio_time),
                                                              max(float(time),.999),1.)
                    prediction, sound = components.transformer_ref(
                        hidden_states=torch.cat((video[:nv],rows))[None],audio_hidden_states=audio[None],
                        encoder_hidden_states=prompt,timestep=times.to(device),timestep_indices=assignments.to(device),
                        **indices[branch],return_dict=False)
                    return prediction[0,nv:].float(),sound[0,na:].float()

                target_v, audio_v = velocity(noisy_target,target_prompt,1)
                if self.mode == 'hybrid' and index == 3:
                    edited = noisy_target + sigma*target_v
                    forwards = 1
                    difference = None
                else:
                    source_v, _ = velocity(noisy_source,source_prompt,0)
                    difference = target_v-source_v
                    edited = edited+(sigma-sigma_next)*difference
                    forwards = 2
                audio[na:] = components.audio_scheduler.step(audio_v,audio_time,audio[na:]).prev_sample
                if not edited.isfinite().all():
                    raise RuntimeError('FlowEdit produced nonfinite video latents')
                records.append(dict(clean_time=float(time),sigma=float(sigma),next_sigma=float(sigma_next),
                    forwards=forwards,source_displacement_rms=float((edited-source).square().mean().sqrt()),
                    velocity_difference_rms=None if difference is None else float(difference.square().mean().sqrt())))
        if self.mode == 'identity' and not torch.equal(edited,source):
            raise RuntimeError('same-conditioning FlowEdit must preserve source latents exactly')
        video[nv:] = edited
        state.set('latents',video)
        state.set('audio_latents',audio)
        self.audit = dict(mode=self.mode,source_latents_exact_identity=self.mode=='identity',
                          forwards=sum(r['forwards'] for r in records),transitions=records,
                          video_clock='native four-transition scheduler',audio_clock='native four-transition scheduler',
                          shared_source_target_noise=True,same_audio_state_for_both_queries=True)
        return components,state


def configure_flow_edit(blocks):
    encoder = SourceVideoEncoder()
    blocks.sub_blocks['reference_encoder'] = encoder
    blocks.sub_blocks['denoise'] = FlowEditDenoise(encoder)
