"""Project visual FlowEdit updates onto fine texture and global channel shifts."""
import torch
import torch.nn.functional as F
from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents, unpatchify_video_tokens
from .visual_flow_edit import AppearanceEncoder, VisualFlowEdit


class AppearanceFlowEdit(VisualFlowEdit):
    def __init__(self, encoder):
        super().__init__(encoder)
        self.kernel = 3

    @torch.no_grad()
    def __call__(self, components, state):
        shape = [state.get(k) for k in ('num_latent_frames','latent_height','latent_width')]
        nv = state.get('num_condition_video_rows')
        target, calls, records = None, 0, []

        def project(module, args, kwargs, output):
            nonlocal target, calls
            prediction, sound = output
            source_branch = calls % 2 == 1
            calls += 1
            if not source_branch:
                target = prediction[0,nv:].float().clone()
                return output
            delta = target-prediction[0,nv:].float()
            z = unpatchify_video_tokens(delta,*shape,24,components.patch_size)
            low = F.avg_pool3d(z,(1,self.kernel,self.kernel),stride=1,
                              padding=(0,self.kernel//2,self.kernel//2),count_include_pad=False)
            projected = z-low+z.mean((2,3,4),keepdim=True)
            delta_new = patchify_video_latents(projected,components.patch_size)
            records.append(dict(raw_delta_rms=float(delta.square().mean().sqrt()),
                                projected_delta_rms=float(delta_new.square().mean().sqrt())))
            prediction = prediction.float().clone()
            prediction[0,nv:] = target-delta_new
            return prediction,sound

        handle = components.transformer_ref.register_forward_hook(project,with_kwargs=True)
        try:
            result = super().__call__(components,state)
        finally:
            handle.remove()
        if calls != 8 or len(records) != 4:
            raise RuntimeError('appearance projection requires four paired queries')
        self.audit.update(projection='spatial high frequencies plus global channel mean',
                          spatial_kernel=self.kernel,projected_updates=records)
        return result


def configure_appearance_flow(blocks):
    encoder = AppearanceEncoder()
    blocks.sub_blocks['reference_encoder'] = encoder
    blocks.sub_blocks['denoise'] = AppearanceFlowEdit(encoder)
