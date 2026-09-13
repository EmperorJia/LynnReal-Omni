"""Spatial interpolation of channel vectors; temporal samples remain independent."""
import torch


def _slerp(a, b, fraction):
    """Interpolate direction on the sphere and magnitude linearly, along channel 1."""
    na, nb = a.norm(dim=1, keepdim=True), b.norm(dim=1, keepdim=True)
    ua, ub = a / na.clamp_min(1e-8), b / nb.clamp_min(1e-8)
    cosine = (ua * ub).sum(dim=1, keepdim=True).clamp(-1, 1)
    angle = cosine.clamp(-0.999999, 0.999999).acos()
    direction = (torch.sin((1 - fraction) * angle) * ua
                 + torch.sin(fraction * angle) * ub) / angle.sin()
    spherical = direction * torch.lerp(na, nb, fraction)
    # Antipodal directions have no unique arc; zero vectors have no direction.
    regular = (cosine.abs() < 0.9995) & (na > 1e-8) & (nb > 1e-8)
    result = torch.where(regular, spherical, torch.lerp(a, b, fraction))
    return torch.where(fraction == 0, a, torch.where(fraction == 1, b, result))


def bislerp_spatial(frames, height, width):
    """Separable width-then-height SLERP on N,C,H,W, with half-pixel coordinates."""
    for axis, size in ((3, width), (2, height)):
        old = frames.shape[axis]
        if old == size:
            continue
        coordinates = ((torch.arange(size, device=frames.device, dtype=torch.float32) + 0.5)
                       * (old / size) - 0.5).clamp(0, old - 1)
        lower = coordinates.floor().long()
        upper = (lower + 1).clamp_max(old - 1)
        shape = [1] * frames.ndim
        shape[axis] = size
        fraction = (coordinates - lower).reshape(shape)
        frames = _slerp(frames.index_select(axis, lower), frames.index_select(axis, upper), fraction)
    return frames
