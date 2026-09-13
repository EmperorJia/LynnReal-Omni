"""Optional global tone stabilization before RGB clipping and boundary feedback."""
import torch
from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD


class ToneAnchor:
    def __init__(self, image, smoothing=0.08, region=None):
        self.reference = torch.as_tensor(image.copy()).permute(2, 0, 1).float() / 255
        if region is not None:
            x0, y0, x1, y1 = region
            if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
                raise ValueError('tone region must be normalized x0,y0,x1,y1 coordinates')
        self.region = region
        self.smoothing = smoothing
        self.previous = None
        self.audit = {}

    @staticmethod
    def statistics(rgb):
        # C,T,H,W -> three channel-wise quantiles per frame. No spatial filtering.
        sample = rgb[..., ::16, ::16].flatten(-2).float()
        q = torch.tensor([0.1, 0.5, 0.9], device=rgb.device)
        return torch.quantile(sample, q, dim=-1)

    def background_fit(self, rgb):
        """Fit channel quantiles within a user-specified stationary region."""
        h, w = rgb.shape[-2:]
        x0, y0, x1, y1 = self.region
        crop = (..., slice(round(y0*h), round(y1*h), 8), slice(round(x0*w), round(x1*w), 8))
        x = rgb[crop].flatten(-2)
        y = self.reference.to(rgb.device)[:, None][crop].flatten(-2)
        if x.shape[-1] < 64:
            raise ValueError('tone region is too small for a stable color fit')
        # Pixel correspondence would mistake small geometry drift for lost
        # contrast. Quantiles retain the region's tone without that assumption.
        q = rgb.new_tensor([.05, .25, .5, .75, .95])
        x, y = torch.quantile(x, q, dim=-1), torch.quantile(y, q, dim=-1)
        mx, my = x.mean(0), y.mean(0)
        variance = (x-mx).square().mean(0)
        covariance = ((x-mx)*(y-my)).mean(0)
        reliable = variance > .0025
        gain = torch.where(reliable, (covariance/variance.clamp_min(1e-6)).clamp(.65, 1.4), 1.)
        offset = torch.where(reliable, (my-gain*mx).clamp(-.12, .12), 0.)
        return gain, offset

    @torch.no_grad()
    def __call__(self, normalized):
        if normalized.ndim != 5 or normalized.shape[0] != 1:
            raise ValueError('tone anchoring expects one B,C,T,H,W video')
        mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=normalized.device).view(1, 3, 1, 1, 1)
        std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=normalized.device).view(1, 3, 1, 1, 1)
        rgb = normalized.float() * std + mean
        target = self.statistics(self.reference.to(rgb.device)[:, None])
        observed = self.statistics(rgb[0])
        if self.region is None:
            gain = ((target[2]-target[0]) / (observed[2]-observed[0]).clamp_min(0.05)).clamp(0.65, 1.4)
            offset = (target[1]-gain*observed[1]).clamp(-0.12, 0.12)
        else:
            if rgb.shape[-2:] != self.reference.shape[-2:]:
                raise ValueError('stationary tone reference and output canvas must match')
            gain, offset = self.background_fit(rgb[0])
        desired = torch.stack((gain, offset), dim=-1).permute(1, 0, 2)
        previous = self.previous
        if previous is None:
            previous = torch.stack((torch.ones(3, device=rgb.device), torch.zeros(3, device=rgb.device)), dim=-1)
        values = []
        for current in desired:
            previous = previous.lerp(current, self.smoothing)
            values.append(previous)
        self.previous = previous.detach()
        coefficients = torch.stack(values).permute(1, 0, 2)
        corrected = rgb * coefficients[None, ..., 0, None, None] + coefficients[None, ..., 1, None, None]
        self.audit = dict(kind='global_rgb_affine_before_clipping',
                          estimator='stationary_region_quantiles' if self.region else 'global_quantiles',
                          region=self.region,
                          coefficients=coefficients.permute(1, 0, 2).cpu().tolist(),
                          source_quantiles=target[..., 0].cpu().tolist(),
                          input_quantiles=observed.permute(2, 0, 1).cpu().tolist())
        return (corrected-mean)/std
