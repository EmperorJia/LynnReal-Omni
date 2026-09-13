"""Read a bounded RGB context from the beginning or end of an actual video."""
from collections import deque
import hashlib
from pathlib import Path
import subprocess
import tempfile
import time


def read_video_window(video, count, width, height, position="tail"):
    import numpy as np
    from imageio_ffmpeg import get_ffmpeg_exe
    if position not in {"head", "tail"} or min(count, width, height) < 1:
        raise ValueError("invalid video context window")
    command = [get_ffmpeg_exe(), "-v", "error", "-threads", "2", "-i", str(Path(video)),
               "-vf", f"fps=24,scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"]
    if position == "head":
        command += ["-frames:v", str(count)]
    command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    retained = deque(maxlen=count)
    frame_bytes, decoded = width * height * 3, 0
    started = time.perf_counter()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                data = process.stdout.read(frame_bytes)
                if not data:
                    break
                if len(data) != frame_bytes:
                    raise RuntimeError("truncated RGB frame from video decoder")
                retained.append(data)
                decoded += 1
        finally:
            process.stdout.close()
            status = process.wait()
        if status:
            errors.seek(0)
            raise RuntimeError(errors.read().decode(errors="replace"))
    if len(retained) != count:
        raise ValueError(f"input provides {decoded} frames at 24 fps; {count} are required")
    rgb = np.frombuffer(b"".join(retained), dtype=np.uint8).reshape(count, height, width, 3).copy()
    return rgb, dict(position=position, fps=24, frames=count,
                    first_resampled_frame=decoded-count, last_resampled_frame=decoded-1,
                    boundary_rgb_sha256=hashlib.sha256(retained[-1]).hexdigest(),
                    preprocessing_seconds=time.perf_counter()-started)
