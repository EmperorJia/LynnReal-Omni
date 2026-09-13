"""Standard BF16: four-step bootstrap and prefix, then 4+2-step latent continuation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from run import RELEASE, frame_count, resolution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    fixtures = RELEASE / 'test/stream'
    p.add_argument('--image', type=Path, default=fixtures / 'first_chunk/street.png')
    prompts = p.add_mutually_exclusive_group()
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path)
    p.add_argument('--captions', type=Path, help='JSON list of continuation prompts, one per 17 new frames')
    p.add_argument('--frame', type=frame_count, default='5s', help='total duration, e.g. 5s or 120f, at 24 fps')
    p.add_argument('--resolution', type=resolution, default='768p')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--first-chunk-only', action='store_true', help='generate only the native 22-frame bootstrap')
    p.add_argument('--appearance-strength', type=float, default=1.)
    p.add_argument('--refresh-context', action=argparse.BooleanOptionalAction, default=None,
                   help='enable refreshed text, one sink and unanchored 4+2-step refinement after the stable prefix; enabled for long clips')
    p.add_argument('--conditioner-service', type=Path, help='optional Qwen worker queue; otherwise stage Qwen on the sampling GPU')
    p.add_argument('--audit-codec', action='store_true', help='also verify incremental decoding against a full decode')
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.resolution != (1344, 768):
        p.error('this stream configuration uses 768p (1344x768)')
    if not a.image.is_file():
        p.error('--image must be an existing 1344x768 first image')
    if not 0 <= a.appearance_strength <= 1:
        p.error('--appearance-strength must be in [0, 1]')
    if not a.first_chunk_only and a.frame < 18:
        p.error('stream duration must contain at least 18 frames')
    custom = a.prompt is not None or a.prompt_file is not None or a.image.resolve() != (fixtures / 'first_chunk/street.png').resolve()
    if custom and a.prompt is None and a.prompt_file is None:
        p.error('provide --prompt or --prompt-file for a custom image')
    if custom and not a.first_chunk_only and a.captions is None:
        p.error('provide --captions for your scene and its ongoing actions')
    prompt_file = a.prompt_file or fixtures / 'normal_motion/initial.txt'
    prompt = a.prompt if a.prompt is not None else prompt_file.read_text().strip()
    if not prompt:
        p.error('the first-chunk prompt must not be empty')
    refresh = not a.first_chunk_only and (a.frame > 136 if a.refresh_context is None else a.refresh_context)
    captions_file = a.captions or fixtures / ('normal_motion/continuations_30s.json' if a.frame > 136 else 'normal_motion/continuations.json')
    if not a.first_chunk_only:
        captions = json.loads(captions_file.read_text())
        needed = (a.frame - 17 + 16) // 17
        if not isinstance(captions, list) or len(captions) < needed or not all(isinstance(c, str) and c.strip() for c in captions):
            p.error(f'--captions requires at least {needed} nonempty strings; the bundled plans cover up to 30 seconds')
    output = (a.output or RELEASE / 'output/standard/bf16/stream' /
              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    bootstrap = output / 'first_chunk'
    commands = [[sys.executable, '-u', str(RELEASE / 'script/sample/first_chunk.py'),
                 '--image', str(a.image.resolve()), '--prompt', prompt, '--seed', str(a.seed),
                 '--appearance-strength', str(a.appearance_strength), '--output', str(bootstrap)]]
    final = bootstrap
    if not a.first_chunk_only:
        final = output / 'continuation'
        commands.append([sys.executable, '-u', str(RELEASE / 'tool/sample_native_overlap.py'),
                         '--bootstrap', str(bootstrap), '--captions', str(captions_file.resolve()),
                         '--frames', str(a.frame), '--seed', str(a.seed), '--local-rope',
                         '--sink-frames', '1' if refresh else '4', '--mid-frames', '0',
                         '--second-steps', '2' if refresh else '0', '--coarse-anchor', '0', '--second-appearance',
                         '--appearance-strength', str(a.appearance_strength),
                         '--audit-codec' if a.audit_codec else '--no-audit-codec', '--output', str(final)])
        if refresh:
            commands[-1].extend(['--recent-text-context', '--warmup-chunks', '7'])
        if a.conditioner_service:
            commands[-1].extend(['--conditioner-service', str(a.conditioner_service.resolve())])
    warmup = min(7, needed) if refresh else 0
    steps = 6 if refresh else 4
    total_forwards = 4 if a.first_chunk_only else 4 + warmup*4 + (needed-warmup)*steps
    plan = dict(precision='bf16', steps_per_chunk=steps, bootstrap_steps=4,
                warmup_chunks=warmup, warmup_steps=4, second_pass_steps=2 if refresh else 0,
                coarse_anchor=0., second_appearance=True, total_dit_forwards=total_forwards, fps=24,
                frames=22 if a.first_chunk_only else a.frame, refresh_context=refresh, output=str(output), commands=commands)
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run.json').write_text(json.dumps(plan, indent=2) + '\n')
    with (output / 'sample.log').open('w') as log:
        for command in commands:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            if code:
                raise SystemExit(code)
        os.link(final / 'video.mp4', output / 'video.mp4')
        sampling = '4-step bootstrap/prefix; 4+2-step continuation' if refresh else '4-step DiT'
        message = f'Completed: {output / "video.mp4"} ({sampling}, BF16, 24 fps)'
        print(message)
        log.write(message + '\n')


if __name__ == '__main__':
    main()
