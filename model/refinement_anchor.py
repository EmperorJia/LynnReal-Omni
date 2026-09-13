"""Soft spatial constraint for latent refinement; no temporal averaging or RGB edits."""
import torch
import torch.nn.functional as F
from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents, unpatchify_video_tokens


class CoarseLatentAnchor:
    def __init__(self, frames, height, width, channels=24, patch_size=(1,2,2), weight=.25, fixed_head_latents=1):
        if not 0 < weight < 1 or min(height,width) < 5:
            raise ValueError('a soft anchor requires weight in (0,1) and spatial extent >=5')
        self.shape = (frames,height,width,channels,patch_size)
        self.weight = weight
        if not 0<=fixed_head_latents<frames:
            raise ValueError('fixed head must leave predicted frames')
        self.head_rows = fixed_head_latents*height*width//(patch_size[1]*patch_size[2])

    def __call__(self, prediction, timestep, sample, reference):
        # H3 predicts data-ward velocity: x0 = xt + (1-t) * prediction.
        sigma = 1-torch.as_tensor(timestep,device=sample.device,dtype=sample.dtype)
        if not 0 < sigma <= 1:
            raise ValueError('invalid noisy refinement timestep')
        predicted_clean = sample + sigma*prediction
        residual = unpatchify_video_tokens(reference-predicted_clean,*self.shape)
        b,c,t,h,w = residual.shape
        spatial = residual.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        coarse = F.avg_pool2d(F.pad(spatial,(2,2,2,2),mode='reflect'),5,stride=1)
        coarse = coarse.reshape(b,t,c,h,w).permute(0,2,1,3,4).contiguous()
        correction = self.weight*patchify_video_latents(coarse,self.shape[-1])/sigma
        correction[:self.head_rows] = 0
        return prediction+correction
