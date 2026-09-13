"""Optional causal exposure correction from the original V2V inference path."""
import torch
import torch.nn.functional as F


@torch.inference_mode()
def stabilize_exposure(video, boundary):
    """Correct RGB [B,3,T,H,W] using the previous delivered [B,3,H,W] frame.

    This changes pixels and the next RGB boundary. It cannot repair geometry.
    Callers must retain the uncorrected decoder output separately.
    """
    reference = boundary
    reference_rgb = reference.mean(dim=(-2, -1), keepdim=True)
    frames, gains, offsets, local_strengths = [], [], [], []
    for frame in video.unbind(2):
        current_rgb = frame.mean(dim=(-2, -1), keepdim=True)
        desired = 0.995 * reference_rgb + 0.005 * current_rgb
        gain = (desired / current_rgb.clamp_min(1e-5)).clamp(0.78, 1.35)
        corrected = (frame * gain).clamp(0, 1)
        offset = (desired - corrected.mean(dim=(-2, -1), keepdim=True)).clamp(-0.03, 0.03)
        corrected = (corrected + offset).clamp(0, 1)
        delta = F.avg_pool2d(corrected, 17, 1, 8) - F.avg_pool2d(reference, 17, 1, 8)
        gate = ((delta.abs().amax(dim=1, keepdim=True) - 0.18) / 0.12).clamp(0, 1)
        local = gate * (delta.sign() * (delta.abs() - 0.14).clamp_min(0))
        corrected = (corrected - local).clamp(0, 1)
        frames.append(corrected)
        gains.append(gain)
        offsets.append(offset.abs().max())
        local_strengths.append(local.abs().max())
        reference = corrected
        reference_rgb = 0.995 * reference_rgb + 0.005 * corrected.mean(dim=(-2, -1), keepdim=True)
    gains, local_strengths = torch.stack(gains), torch.stack(local_strengths)
    return torch.stack(frames, dim=2), {
        "method": "original_v2v_causal_exposure_v1",
        "gain_range": [float(gains.min()), float(gains.max())],
        "offset_abs_max": float(torch.stack(offsets).max()),
        "local_corrected_frames": int((local_strengths > 0).sum()),
        "local_correction_abs_max": float(local_strengths.max()),
    }
