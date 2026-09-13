"""Two-stage flow sampling: native four-step generation, then explicit latent refinement.

The default second grid is a trained tail; custom grids are experimental.
This adapter preserves the first-pass RNG and scheduler. A frozen instance
holds its modality clean; otherwise refinement uses a separate Gaussian draw.
"""
from types import SimpleNamespace
import math
import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.utils.torch_utils import randn_tensor


class TwoPassScheduler:
    def __init__(self, shift, extra_steps, seed=0, head=None, trace=None, frozen=False, clean_constraint=None, second_sigmas=None, initial_noise=None, noise_correlation=0.):
        if not 0 <= extra_steps <= 4:
            raise ValueError('second pass must use zero to four evaluations')
        self.first = MiniMaxH3Scheduler(shift=shift)
        self.second = MiniMaxH3Scheduler(shift=shift)
        self.extra_steps, self.seed = extra_steps, seed
        self.head, self.trace, self.frozen = head, trace, frozen
        self.calls = 0
        self.clean_constraint = clean_constraint
        self.reference = None
        if not math.isfinite(noise_correlation) or not 0 <= noise_correlation <= 1:
            raise ValueError('noise correlation must be in [0,1]')
        if noise_correlation and (initial_noise is None or not extra_steps or frozen):
            raise ValueError('correlated video refinement requires first-pass noise and extra steps')
        self.noise_correlation = noise_correlation
        self.initial_noise = initial_noise.detach().clone() if noise_correlation else None
        if second_sigmas is not None:
            grid = torch.as_tensor(second_sigmas, dtype=torch.float32)
            if (grid.ndim != 1 or len(grid) != extra_steps+1 or not extra_steps
                    or not torch.isfinite(grid).all() or not 0 < grid[0] <= 1
                    or grid[-1] != 0 or not torch.all(grid[1:] < grid[:-1])):
                raise ValueError('second_sigmas must decrease from (0,1] to zero, with extra_steps transitions')
            self.second_sigmas = grid
        else:
            self.second_sigmas = None

    def set_timesteps(self, num_inference_steps, device=None):
        if num_inference_steps != 5 + self.extra_steps:
            raise ValueError('expected four first-pass evaluations plus the explicit second pass')
        self.first.set_timesteps(5, device=device)
        self.calls = 0
        if self.extra_steps:
            grid = self.first.sigmas[-(self.extra_steps+1):] if self.second_sigmas is None else self.second_sigmas
            self.second.set_timesteps(sigmas=grid, device=device)
            tail = torch.ones_like(self.second.timesteps) if self.frozen else self.second.timesteps
            self.timesteps = torch.cat((self.first.timesteps, tail))
        else:
            self.timesteps = self.first.timesteps
        self.record = dict(first_sigmas=self.first.sigmas.tolist(),
                           second_sigmas=self.second.sigmas.tolist() if self.extra_steps else [],
                           extra_forwards=self.extra_steps, frozen_second_pass=self.frozen,
                           custom_second_grid=self.second_sigmas is not None,
                           noise_correlation=self.noise_correlation,
                           renoise_seed=None if self.frozen else self.seed)

    def step(self, prediction, timestep, sample):
        index = self.calls
        if index >= len(self.timesteps) or not torch.equal(torch.as_tensor(timestep), self.timesteps[index]):
            raise RuntimeError('unexpected two-pass scheduler call')
        if index < 4:
            result = self.first.step(prediction, timestep, sample).prev_sample
            if index == 3 and not self.frozen:
                clean = result.clone()
                if self.head is not None:
                    clean[:len(self.head)] = self.head
                if self.clean_constraint is not None:
                    self.reference = clean
                tensors = {'first_pass_video_rows': clean.cpu().contiguous()}
                if self.extra_steps:
                    noise = randn_tensor(clean.shape, generator=torch.Generator().manual_seed(self.seed),
                                         device=clean.device, dtype=clean.dtype)
                    if self.noise_correlation:
                        initial = self.initial_noise.to(clean)
                        if initial.shape != clean.shape or not initial.isfinite().all():
                            raise ValueError('first-pass noise has invalid geometry or values')
                        tensors['second_pass_independent_noise'] = noise.cpu().contiguous()
                        tensors['first_pass_initial_noise'] = initial.cpu().contiguous()
                        rho = self.noise_correlation
                        noise = rho*initial + math.sqrt(1-rho*rho)*noise
                    tensors['second_pass_noise'] = noise.cpu().contiguous()
                    sigma = self.second.sigmas[0]
                    result = (1-sigma)*clean + sigma*noise
                if self.trace is not None:
                    from safetensors.torch import save_file
                    self.trace.parent.mkdir(parents=True, exist_ok=True)
                    save_file(tensors, str(self.trace))
        elif self.frozen:
            result = sample
        else:
            if self.clean_constraint is not None:
                prediction = self.clean_constraint(prediction, timestep, sample, self.reference)
            result = self.second.step(prediction, timestep, sample).prev_sample
        self.calls += 1
        return SimpleNamespace(prev_sample=result)
