"""Extract a 24-fps control window and its first frame from an existing video."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new directory for control.mp4, first.png and provenance")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--frames", type=int, default=124)
    a = p.parse_args()
    if a.start < 0 or a.frames < 1 or a.output.exists():
        p.error("start must be nonnegative, frames positive, and output new")
    probe = json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(a.input)]))
    stream = next(x for x in probe["streams"] if x["codec_type"] == "video")
    duration = float(stream.get("duration",probe["format"]["duration"]))
    if a.start + a.frames/24 > duration + 1e-3:
        p.error("requested window exceeds source; choose a shorter window")
    a.output.mkdir(parents=True)
    video = a.output/"control.mp4"
    pixel_format = "yuv444p" if stream["width"] % 2 or stream["height"] % 2 else "yuv420p"
    subprocess.run(["ffmpeg","-v","error","-ss",str(a.start),"-i",str(a.input),"-vf","fps=24",
        "-frames:v",str(a.frames),"-an","-c:v","libx264","-crf","16","-pix_fmt",pixel_format,str(video)],check=True)
    subprocess.run(["ffmpeg","-v","error","-i",str(video),"-frames:v","1",str(a.output/"first.png")],check=True)
    actual=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames",
        "-show_entries","stream=nb_read_frames,width,height,avg_frame_rate","-of","json",str(video)]))["streams"][0]
    if int(actual["nb_read_frames"]) != a.frames:
        raise RuntimeError("decoded frame count differs from request")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output/"source.json").write_text(json.dumps({"source":str(a.input.resolve()),"sha256":digest(a.input),
        "source_stream":stream,"start_seconds":a.start,"frames":a.frames,"fps":24,"control":actual,
        "control_sha256":digest(video),"first_frame_sha256":digest(a.output/"first.png"),
        "operation":"temporal extraction and re-encoding only; no scene re-rendering or spatial resize"},indent=2))


if __name__ == "__main__":
    main()
