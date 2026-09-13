"""Experimental low-frequency geometry constraint for native reference editing.

The source is the already generated clip, never unseen future input. This is
an H3 latent-space diagnostic inspired by ILVR, not an ILVR reproduction.
"""
import torch
import torch.nn.functional as F
from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents, unpatchify_video_tokens
from .flow_edit import SourceVideoEncoder
from .offload import StagedReferenceDenoise


class StructureDenoise(StagedReferenceDenoise):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.factor, self.strength = 4, 1.
        self.audit = {}

    @torch.no_grad()
    def __call__(self, components, state):
        shape = [state.get(k) for k in ('num_latent_frames','latent_height','latent_width')]
        patch = components.patch_size
        source = self.encoder.source_rows.to(components._execution_device)
        def unpack(rows):
            return unpatchify_video_tokens(rows,*shape,24,patch)
        def lowpass(z):
            b,c,t,h,w=z.shape
            x=z.permute(0,2,1,3,4).reshape(b*t,c,h,w)
            x=F.interpolate(F.interpolate(x,size=(max(1,h//self.factor),max(1,w//self.factor)),
                mode='area'),size=(h,w),mode='bilinear',align_corners=False)
            return x.reshape(b,t,c,h,w).permute(0,2,1,3,4)
        coarse_source=lowpass(unpack(source))
        dims=(2,3,4)
        source_mean=coarse_source.mean(dims,keepdim=True)
        source_std=coarse_source.std(dims,keepdim=True,unbiased=False).clamp_min(1e-6)
        original=components.scheduler.step
        records=[]
        def constrained(velocity,timestep,sample,return_dict=True):
            if sample.shape != source.shape:
                raise RuntimeError('structure guidance requires matching source and generated rows')
            sigma=1-float(timestep)
            clean=unpack(sample+sigma*velocity)
            coarse=lowpass(clean)
            # Keep target appearance statistics while anchoring source geometry.
            target_mean=coarse.mean(dims,keepdim=True)
            target_std=coarse.std(dims,keepdim=True,unbiased=False)
            anchor=(coarse_source-source_mean)*(target_std/source_std).clamp(.5,2.)+target_mean
            correction=self.strength*(anchor-coarse)
            delta=patchify_video_latents(correction,patch)
            records.append(dict(clean_time=float(timestep),correction_rms=float(delta.square().mean().sqrt())))
            return original(velocity+delta/max(sigma,1e-6),timestep,sample,return_dict)
        components.scheduler.step=constrained
        try:
            result=super().__call__(components,state)
        finally:
            components.scheduler.step=original
        self.audit=dict(forwards=len(records),spatial_downsample=self.factor,strength=self.strength,
                        appearance_moments_from_target=True,transitions=records)
        return result


def configure_structure_repair(blocks):
    encoder=SourceVideoEncoder()
    blocks.sub_blocks['reference_encoder']=encoder
    blocks.sub_blocks['denoise']=StructureDenoise(encoder)
