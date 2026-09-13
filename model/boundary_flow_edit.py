"""Keep the source head latent exact during paired background FlowEdit."""
import torch
from .background_flow45_edit import BackgroundEncoder
from .visual_flow_edit import VisualFlowEdit


class BoundaryFlowEdit(VisualFlowEdit):
    @torch.no_grad()
    def __call__(self, components, state):
        nv = state.get('num_condition_video_rows')
        ph, pw = components.patch_size[1:]
        count = state.get('latent_height') * state.get('latent_width') // (ph * pw)
        calls, target = 0, None

        def preserve_head(module, args, kwargs, output):
            nonlocal calls, target
            prediction, audio = output
            if calls % 2 == 0:
                target = prediction[:, nv:nv + count].clone()
            else:
                prediction = prediction.clone()
                prediction[:, nv:nv + count] = target
            calls += 1
            return prediction, audio

        handle = components.transformer_ref.register_forward_hook(preserve_head, with_kwargs=True)
        try:
            result = super().__call__(components, state)
        finally:
            handle.remove()
        head = state.get('latents')[nv:nv + count]
        source = self.encoder.source_rows[:count].to(head)
        if calls != 8 or not torch.equal(head, source):
            raise RuntimeError('paired editor did not preserve its source head latent')
        self.audit.update(source_head_latent_exact=True, preserved_head_rows=count,
                          head_preservation='zero source-target velocity difference on first latent')
        return result


def configure_boundary_flow(blocks):
    encoder = BackgroundEncoder()
    blocks.sub_blocks['reference_encoder'] = encoder
    blocks.sub_blocks['denoise'] = BoundaryFlowEdit(encoder)
