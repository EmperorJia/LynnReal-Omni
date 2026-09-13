"""Joint audio/video continuation with the formal V2V video layout."""
import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_AUDIO_TAG, audio_latent_num_frames
from diffusers.utils.torch_utils import randn_tensor
from .formal_stream import FormalStream


def add_audio_rows(kwargs, audio, audio_time, rows_per_frame, target_latents=6):
    """Keep native [text | history | audio | target] order and future audio clock."""
    positions = kwargs['position_ids']
    start = len(positions) - target_latents*rows_per_frame
    count, frames = len(audio), len(audio)//2
    extra = positions.new_zeros((count,3))
    extra[:,0] = positions[start+rows_per_frame,0] + torch.arange(frames,device=positions.device).repeat(2)
    extra[:frames,2] = positions[start:,2].min()
    extra[frames:,2] = positions[start:,2].max()
    tags = kwargs['token_tags']
    times = kwargs['timestep'][kwargs['timestep_indices']]
    times = torch.cat((times[:start],times.new_full((count,),audio_time),times[start:]))
    unique, assignments = torch.unique(times,sorted=True,return_inverse=True)
    video = kwargs['video_indices']
    return dict(kwargs, audio_hidden_states=audio[None],
        position_ids=torch.cat((positions[:start],extra,positions[start:])),
        token_tags=torch.cat((tags[:start],tags.new_full((count,),MINIMAX_H3_AUDIO_TAG),tags[start:])),
        video_indices=video+(video>=start)*count,
        audio_indices=torch.arange(start,start+count,device=video.device),
        timestep=unique,timestep_indices=assignments)


class AudioStream(FormalStream):
    @torch.inference_mode()
    def step(self, conditioning, seed=0, steps=4):
        scheduler = MiniMaxH3Scheduler(shift=3)
        scheduler.set_timesteps(steps+1,device=self.device)
        audio = randn_tensor((2*audio_latent_num_frames(self.prediction_frames),32),
            generator=torch.Generator().manual_seed(seed+1000000),device=self.device,dtype=torch.float32)
        original, index = self.transformer, 0
        rows_per_frame = self.history.shape[-2]*self.history.shape[-1]//4
        def forward(**kwargs):
            nonlocal audio,index
            timestep = scheduler.timesteps[index]
            arguments = add_audio_rows(kwargs,audio,float(timestep),rows_per_frame,self.target_latents)
            video_prediction,audio_prediction = original(**arguments)
            audio = scheduler.step(audio_prediction[0].float(),timestep,audio).prev_sample
            index += 1
            return video_prediction,audio_prediction
        self.transformer = forward
        try:
            output,timing = super().step(conditioning,seed,steps)
        finally:
            self.transformer = original
        timing.update(audio_rows=len(audio),audio_decoded=False,audio_seed=seed+1000000)
        return output,timing
