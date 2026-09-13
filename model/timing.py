"""Measure complete video-VAE decode calls, including tiled decoder execution."""
from functools import wraps
import torch


def measure_decoder(vae, events):
    decode = vae.decode

    @wraps(decode)
    def timed(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = decode(*args, **kwargs)
        end.record()
        events.append((start, end))
        return result

    vae.decode = timed
