"""Lossless chunk interchange for the experimental generate-then-edit loop."""
import json
from pathlib import Path
import subprocess
import time
import uuid


def editing_window(previous, generated, context_frames=1):
    """Keep 16 new frames at their original timestamps in a native 22-frame input."""
    import numpy as np
    if context_frames not in (1, 6) or len(previous) < context_frames or len(generated) != 16:
        raise ValueError('expected 16 new frames and either 1 or 6 preceding RGB frames')
    if previous.shape[1:] != generated.shape[1:]:
        raise ValueError('history and generated RGB geometry differ')
    padding = 6 - context_frames
    source = np.concatenate((previous[-context_frames:], generated,
                             np.repeat(generated[-1:], padding, axis=0)))
    return source, slice(context_frames, context_frames + 16)


def write_video(path, rgb, lossless=False):
    from imageio_ffmpeg import get_ffmpeg_exe
    height, width = rgb.shape[1:3]
    command = [get_ffmpeg_exe(), '-v', 'error', '-y', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '24',
               '-i', 'pipe:0', '-an', '-threads', '4', '-c:v',
               'libx264rgb' if lossless else 'libx264', '-crf', '0' if lossless else '18',
               '-pix_fmt', 'rgb24' if lossless else 'yuv420p', str(path)]
    subprocess.run(command, input=rgb.tobytes(), check=True, capture_output=True)


def edit_remote(queue, directory, first, source, boundary, prompt, seed, timeout=900):
    from .weights import sha256
    import numpy as np
    queue, directory = Path(queue), Path(directory)
    queue.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    request = dict(id=identity, output=str(directory.resolve()), prompt=prompt, seed=seed,
                   references=[dict(path=str(Path(p).resolve()), sha256=sha256(p))
                               for p in (first, source, boundary)])
    temporary = queue / f'{identity}.tmp'
    temporary.write_text(json.dumps(request))
    temporary.rename(queue / f'{identity}.request.json')
    result = queue / f'{identity}.result.json'
    started = time.monotonic()
    while not result.exists():
        if time.monotonic()-started > timeout:
            raise TimeoutError(f'chunk editor did not answer {identity}')
        time.sleep(.1)
    record = json.loads(result.read_text())
    if record.get('request') != request or record.get('error'):
        raise RuntimeError(f'chunk edit failed: {record.get("error", "request mismatch")}')
    pixels = np.load(directory/'edited_rgb.npy', allow_pickle=False)
    if sha256(directory/'edited_rgb.npy') != record['rgb_sha256']:
        raise RuntimeError('edited pixels changed')
    return pixels, record
