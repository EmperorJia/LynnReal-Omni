"""Optional latent appearance constraint for a short image-conditioned bootstrap."""
import torch
from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents,unpatchify_video_tokens


@torch.inference_mode()
def encode_static_reference(pipeline, image, frames=22):
    """Encode only the input image, repeated to provide native video context."""
    import numpy as np
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD
    from .offload import vae_phase

    with vae_phase(pipeline.pipe):
        vae = pipeline.pipe.vae
        mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device='cuda').view(1, 3, 1, 1, 1)
        std = torch.tensor(MINIMAX_H3_PIXEL_STD, device='cuda').view(1, 3, 1, 1, 1)
        pixels = torch.from_numpy(np.array(image.convert('RGB'), copy=True)).cuda()
        pixels = pixels.permute(2, 0, 1)[None, :, None].float() / 255
        with torch.autocast('cuda', dtype=torch.float16):
            moments = vae._encode(((pixels - mean) / std).expand(-1, -1, frames, -1, -1))
        # Match native posterior construction, seed and FP16 rounding.
        posterior = DiagonalGaussianDistribution(moments)
        latent = posterior.sample(generator=torch.Generator().manual_seed(42)).half().float()
        latent_mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, 24, 1, 1, 1)
        latent_std = torch.tensor(vae.config.latents_std, device='cuda').view(1, 24, 1, 1, 1)
        return ((latent - latent_mean) / latent_std).cpu().contiguous()


def constrain_moments(scheduler,reference,strength,patch_size=(1,2,2),fixed_head_latents=1,max_scale=None,max_calls=None):
    """Constrain predicted-clean spatial moments; preserve the four-step grid.

    Reference is the supplied image encoded with video padding, not future
    observations. This changes sampling and must be reported as an ablation.
    """
    if not 0 <= strength <= 1 or reference.ndim!=5 or reference.shape[0]!=1:
        raise ValueError('expected one latent video and strength in [0,1]')
    if not 0 <= fixed_head_latents < reference.shape[2]:
        raise ValueError('fixed head must leave at least one predicted latent')
    if max_scale is not None and max_scale <= 0:
        raise ValueError("max_scale must be positive")
    if max_calls is not None and max_calls<1:
        raise ValueError('max_calls must be positive')
    ref=reference.detach().float();shape=ref.shape
    std,mean=torch.std_mean(ref,dim=(-2,-1),keepdim=True,correction=0)
    original=scheduler.step;records=[]
    def step(prediction,timestep,sample,**kwargs):
        if strength==0 or (max_calls is not None and len(records)>=max_calls):
            return original(prediction,timestep,sample,**kwargs)
        sigma=1-torch.as_tensor(timestep,device=sample.device,dtype=torch.float32)
        if sigma<=0:raise ValueError('predicted-clean constraint requires a nonzero noise level')
        clean=unpatchify_video_tokens(sample+sigma*prediction,*shape[2:],shape[1],patch_size)
        current_std,current_mean=torch.std_mean(clean.float(),dim=(-2,-1),keepdim=True,correction=0)
        scale=std.to(clean)/current_std.clamp_min(1e-6)
        if max_scale is not None:
            scale=scale.clamp_max(max_scale)
        adjusted=(clean-current_mean)*(1+strength*(scale-1))+current_mean+strength*(mean.to(clean)-current_mean)
        # The independently fixed head keeps its own near-clean time and endpoint.
        adjusted[:,:,:fixed_head_latents]=clean[:,:,:fixed_head_latents]
        corrected=(patchify_video_latents(adjusted,patch_size)-sample)/sigma
        if not torch.isfinite(corrected).all():raise RuntimeError('non-finite appearance constraint')
        records.append(dict(step=len(records)+1,strength=strength,mean_abs_shift=float((mean.to(clean)-current_mean).abs().mean()),median_scale=float(scale.median()),max_scale=float(scale.max()),scale_limit=max_scale))
        return original(corrected,timestep,sample,**kwargs)
    scheduler.step=step
    return records
