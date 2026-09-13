"""Shared CLI for explicit INT8 and BF16 sampling launchers."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

RELEASE = Path(__file__).resolve().parents[2]


def frame_count(value):
    """Bare numbers / 's' mean seconds; 'f' means an exact output frame count."""
    try:
        count = float(value[:-1]) if value[-1:] in {"s", "f"} else float(value)
        if not math.isfinite(count) or count <= 0:
            raise ValueError
        frames = count if value.endswith("f") else count * 24
        if not frames.is_integer():
            raise ValueError
        return int(frames)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError("use seconds (5s or 5) or an integer frame count (120f), at 24 fps")


def resolution(value):
    presets = {"480p": (832, 480), "540p": (960, 540), "720p": (1280, 720),
               "768p": (1344, 768), "1080p": (1920, 1080)}
    if value.lower() in presets:
        return presets[value.lower()]
    if re.fullmatch(r"[0-9]+[xX][0-9]+", value):
        width, height = map(int, value.lower().split("x"))
        if min(width, height) > 0 and width % 2 == height % 2 == 0:
            return width, height
    raise argparse.ArgumentTypeError("use 768p or even WIDTHxHEIGHT, e.g. 1344x768")


def media(values):
    paths = []
    for value in values or []:
        for item in value.split(","):
            if not item.strip():
                raise ValueError("empty path in comma-separated reference list")
            path = Path(item.strip()).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"not a file: {path}")
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("standard", "flash"), required=True)
    parser.add_argument("--precision", choices=("bf16", "int8"), required=True)
    parser.add_argument("--task", choices=("t2v", "ti2v", "v2v", "ref2v"), required=True)
    parser.add_argument("--frame", type=frame_count, default="5s", help="duration: 5s (default), 5, or exact frames: 120f")
    parser.add_argument("--resolution", type=resolution, default="768p")
    parser.add_argument("--prompt", help="literal prompt; otherwise use test/<task>.txt")
    parser.add_argument("--image", help="first frame for TI2V; optional appearance image for V2V/Ref2V")
    parser.add_argument("--ref_image", action="append", help="reference images, comma-separated; option may repeat")
    parser.add_argument("--ref__video", "--ref_video", dest="ref_video", action="append",
                        help="reference videos, comma-separated; both spellings accepted")
    parser.add_argument("--weights", type=Path, help="override the task's default HF model directory")
    parser.add_argument("--attention-backend", choices=("auto", "native", "_flash_3", "flash", "_native_flash", "_native_cudnn"), default="auto")
    parser.add_argument("--vae-tiles", choices=("native", "adaptive"), default=None,
                        help="native preserves the default layout; adaptive uses the trained faster tile layout")
    parser.add_argument("--vae-attention-backend", choices=("native", "auto", "_flash_3", "flash", "_native_flash", "_native_cudnn"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=None, help="unmeasured warmup calls")
    parser.add_argument("--repeats", type=int, default=1, help="measured calls; save the final video")
    parser.add_argument("--name", help="optional output directory name (must not already exist)")
    parser.add_argument("--refine", action="store_true", help="Flash T2V: spatial latent upscale followed by video-only refinement")
    parser.add_argument("--refine-strength", type=float, help="starting noise: 0.94 for two steps, 12/14 for one")
    parser.add_argument("--refine-steps", type=int, default=2)
    parser.add_argument("--refine-first-height", type=int, choices=(544, 640, 768), default=544,
                        help="first-pass canvas height cap; larger values cost more compute")
    parser.add_argument("--refine-schedule", choices=("trained-tail", "linear"), default="trained-tail")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print command without loading models")
    parser.add_argument("--continuation", action="store_true", help="continue the final 22 source frames; output contains only new frames")
    args = parser.parse_args()
    if args.warmups is None:
        args.warmups = 2 if args.precision == 'int8' else 0
    if args.vae_tiles is None:
        args.vae_tiles = 'adaptive' if args.precision == 'int8' else 'native'
    if args.vae_attention_backend is None:
        args.vae_attention_backend = 'auto' if args.precision == 'int8' else 'native' 
    if args.refine_strength is None:
        args.refine_strength = 12/14 if args.refine_steps == 1 else 0.94
    if args.continuation and (args.task != "v2v" or args.precision != "bf16" or args.warmups or args.repeats != 1):
        parser.error("continuation requires standard BF16 V2V and one sampling call")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats must be positive")
    if args.variant == "flash" and args.task not in {"t2v", "ti2v"}:
        parser.error("Flash supports T2V and TI2V")
    if args.variant == "flash" and args.precision != "int8":
        parser.error("Flash uses its trained INT8 weights; standard uses bf16 or int8")
    if args.refine and (args.variant != "flash" or args.task != "t2v" or args.repeats != 1):
        parser.error("refinement requires one Flash T2V sample without benchmark repetitions")
    if not 0 < args.refine_strength < 1 or not 1 <= args.refine_steps <= 4:
        parser.error("refinement requires strength in (0,1) and 1..4 steps")
    if args.refine and args.refine_schedule == "trained-tail" and args.refine_steps > 1 and args.refine_strength <= 12/14:
        parser.error("multi-step trained-tail refinement requires strength > 12/14; use --refine-schedule linear for other grids")
    defaults = json.loads((RELEASE / "test/defaults.json").read_text())[args.task]
    try:
        custom_media = args.image is not None or args.ref_image is not None or args.ref_video is not None
        images = media(([args.image] if args.image else []) + (args.ref_image or []))
        videos = media(args.ref_video)
        if not custom_media:
            images = [(RELEASE / "test" / p).resolve(strict=True) for p in defaults.get("images", [])]
            videos = [(RELEASE / "test" / p).resolve(strict=True) for p in defaults.get("videos", [])]
            if args.continuation:
                images = []
        if args.continuation and images:
            raise ValueError("video continuation consumes the source video; use Ref2V for separate appearance images")
        if any(p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} for p in images):
            raise ValueError("--image/--ref_image require image files")
        if any(p.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"} for p in videos):
            raise ValueError("--ref__video requires video files")
        if args.task == "t2v" and (images or videos):
            raise ValueError("T2V does not accept reference media")
        if args.task == "ti2v" and (len(images) != 1 or videos):
            raise ValueError("TI2V requires exactly one first-frame image")
        if args.task == "v2v" and len(videos) != 1:
            raise ValueError("V2V requires exactly one control video")
        if args.task == "ref2v" and not (images or videos):
            raise ValueError("Ref2V requires at least one image or video")
        weights = (args.weights or RELEASE / "weight" / (
            "standard" if args.variant == "standard" else "flash")).resolve(strict=True)
        config = json.loads((weights / "inference_config.json").read_text())
        if args.variant == "flash" and weights != (RELEASE / "weight/flash").resolve():
            raise ValueError("Flash launchers use weight/flash")
        if config["variant"] != args.variant:
            raise ValueError("checkpoint variant does not match the launcher")
        transformer = None
        if args.variant == 'standard':
            sys.path.insert(0, str(RELEASE))
            from model.weights import standard_transformer
            transformer = standard_transformer(weights)
        prompt = args.prompt if args.prompt is not None else (RELEASE / "test" / defaults["prompt"]).read_text()
        if args.continuation and args.prompt is None:
            prompt = (RELEASE / "test/v2v_continuation.txt").read_text()
        if args.refine and args.prompt is None:
            prompt = (RELEASE / "test/flash_wuxia.txt").read_text()
            for value in (3.542, 5.500, 7.792):
                seconds = value * args.frame / 243
                prompt = prompt.replace(f"00:{value:06.3f}", f"{int(seconds // 60):02}:{seconds % 60:06.3f}")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        name = args.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{os.getpid()}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("--name may contain only letters, digits, underscores and hyphens")
        output = RELEASE / "output" / args.variant / args.precision / args.task / name
        if output.exists():
            raise ValueError(f"output already exists: {output}")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    width, height = args.resolution
    steps = 4 if args.variant == "standard" else int(config["steps"])
    mode = {"v2v": "video-edit", "ref2v": "reference"}.get(args.task, args.task)
    command = [sys.executable, "-u", str(RELEASE / "script/sample.py"), "--weights", str(weights),
        "--mode", mode, "--prompt-file", str(output / "prompt.txt"),
        "--height", str(height), "--width", str(width), "--frames", str(args.frame),
        "--steps", str(steps), "--seed", str(args.seed),
        "--warmups", str(args.warmups), "--repeats", str(args.repeats), "--fast",
        "--attention-backend", args.attention_backend,
        "--cache", str(RELEASE / "output/.cache"), "--output", str(output / "video.mp4")]
    if args.refine:
        command += ["--refine-strength", str(args.refine_strength), "--refine-steps", str(args.refine_steps),
                    "--refine-schedule", args.refine_schedule,
                    "--refine-first-height", str(args.refine_first_height)]
    for path in images + videos:
        command += ["--reference", str(path)]
    if args.task == "ti2v":
        command.append("--native-keyframes")
    if args.task == "v2v":
        command += ["--aligned-reference", str(len(images))]
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.precision == "int8":
        cache = RELEASE / 'output/.cache'
        os.environ.setdefault('TRITON_CACHE_DIR', str(cache/'triton'))
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(cache/'inductor'))
        command += ['--vae-tiles', args.vae_tiles, '--vae-attention', args.vae_attention_backend]
        command += ["--int8", "--int8-gemm", "triton",
                    "--light-vae", str(RELEASE / "weight/light-vae")]
        command.append("--compile-vae")
        if not args.dry_run and args.variant == "flash":
            import torch
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 64 * 2**30:
                # Same spatial tiles and blend order; bound temporary decoder memory.
                command += ["--vae-tile-batch", "1"]
    elif args.precision == "bf16":
        command.append("--vae-offload")
        # The BF16 reference route stages blocks on memory-limited devices.
        if not args.dry_run:
            import torch
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 100 * 2**30:
                if args.task in {"v2v", "ref2v"} and args.frame > 39:
                    command += ["--dit-offload-blocks", "32"]
    record = {"variant": args.variant, "precision": args.precision, "task": args.task, "weights": str(weights),
              "transformer": transformer,
              "frames": args.frame, "fps": 24, "seconds": args.frame / 24,
              "resolution": [width, height], "steps": steps, "seed": args.seed,
              "warmups": args.warmups, "repeats": args.repeats,
              "images": list(map(str, images)), "videos": list(map(str, videos)),
              "cuda_allocator": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
              "prompt": prompt, "output": str(output), "command": command}
    if args.continuation:
        command = [sys.executable, "-u", str(RELEASE / "script/continue_video.py"),
                   "--weights", str(weights), "--video", str(videos[0]), "--source-window", "tail",
                   "--prompt-file", str(output / "prompt.txt"), "--output", str(output / "video.mp4"),
                   "--cache", str(output / "cache"), "--frames", str(args.frame), "--steps", "4",
                   "--height", str(height), "--width", str(width), "--seed", str(args.seed),
                   "--dit-offload-blocks", "32"]
        record.update(mode="dense-continuation", source_window="tail", condition_frames=22,
                      output_contract="new frames after the source video", command=command)
    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.txt").write_text(prompt.strip() + "\n")
    (output / "request.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    with (output / "run.log").open("w", buffering=1) as log:
        header = json.dumps(record, ensure_ascii=False) + "\n"
        print(header, end="", flush=True); log.write(header)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True); log.write(line)
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate(); process.wait()
            raise
        log.write(f"\nexit_code={code}\n")
    if code:
        raise SystemExit(code)
    print(f"Saved video, log and metadata: {output}", flush=True)


if __name__ == "__main__":
    main()
