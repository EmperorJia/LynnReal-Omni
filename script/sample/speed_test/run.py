"""Warm T2V latency sweep; conditioning, loading and file encoding excluded."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
PROMPT = ('integrated_multimodal_description: [Shot 1] A single continuous 0.92-second '
          'medium close-up of a red ceramic teapot. A thin stream of tea flows steadily '
          'from its spout into a white cup on a wooden table; the liquid level rises slightly. '
          'Fixed camera, natural real-time motion, soft morning window light, crisp ceramic '
          'glaze and subtle steam.\n\noverall_soundscape: Gentle tea pouring.\n\nnon_diegetic_music: N/A')


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log
    def write(self, value):
        self.original.write(value); self.log.write(value)
    def flush(self):
        self.original.flush(); self.log.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('standard', 'flash'), required=True)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--seed', type=int, default=77)
    p.add_argument('--resolution', choices=('540p', '768p'), default='540p')
    p.add_argument('--frames', type=int, help='defaults to 22 at 540p or 120 at 768p')
    p.add_argument('--allow-terminal-padding', action='store_true',
                   help='15s boundary experiment: permit 362 computed frames for 360 delivered frames')
    p.add_argument('--search', choices=('full', 'fast', 'fixed'), default='full',
                   help='fast: Triton sweep; fixed: previously measured FA3/adaptive/Triton configuration')
    p.add_argument('--name', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = p.parse_args()
    frames_requested = args.frames if args.frames is not None else (22 if args.resolution == '540p' else 120)
    width, height = (960, 540) if args.resolution == '540p' else (1344, 768)
    canvas_height = ((height + 31) // 32) * 32
    if frames_requested < 1:
        p.error('frames must be positive')
    if args.allow_terminal_padding and frames_requested != 360:
        p.error('--allow-terminal-padding is only for the 360-frame boundary experiment')
    prompt = PROMPT.replace('0.92-second', f'{frames_requested/24:.2f}-second')
    if args.warmups < 1 or args.repeats < 3 or Path(args.name).name != args.name:
        p.error('use >=1 warmup, >=3 repeats, and a single output directory name')
    out = ROOT / 'output/speed_test' / args.variant / args.name
    out.mkdir(parents=True, exist_ok=False)
    log = (out / 'run.log').open('w', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    import torch
    from model.output import encode_video
    from model.acceleration import fastest_available
    from model.pipeline import Pipeline, encode_conditioning
    if args.allow_terminal_padding:
        from diffusers.modular_pipelines.minimax_h3 import before_encoder
        before_encoder.MINIMAX_H3_MAX_DURATION = 362 / 24
        print('Explicit 15s boundary experiment: compute 362 frames, deliver the first 360.', flush=True)
    from model.int8 import Int8Linear
    from model.provenance import snapshot_sources, changed_sources
    accel = fastest_available()
    if args.search in ('fast', 'fixed') and not accel['fused']:
        raise RuntimeError('fast sweep requires supported Triton fusion')
    def hardware():
        command = ['nvidia-smi', '--query-gpu=uuid,name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu', '--format=csv,noheader']
        result = subprocess.run(command, capture_output=True, text=True)
        return {'hostname': socket.gethostname(), 'visible_devices': torch.cuda.device_count(),
                'nvidia_smi': result.stdout.strip(), 'error': result.stderr.strip()}
    hardware_start = hardware()
    print(json.dumps({'hardware': hardware_start}), flush=True)
    print(json.dumps({'acceleration': accel}), flush=True)
    weights = ROOT / 'weight' / args.variant
    cache = ROOT / 'output/.cache'
    source = snapshot_sources(cache)
    (out / 'prompt.txt').write_text(prompt + '\n')
    condition, conditioning = encode_conditioning(weights, prompt, [], canvas_height, width, frames_requested, cache)
    pipe = Pipeline(weights, int8=True, fused=accel['fused'], light_vae=ROOT/'weight/light-vae',
        compile_vae=accel['fused'], adaln_cache=accel['fused'], int8_gemm='triton' if accel['fused'] else 'torch',
        attention_backend=accel['attention_backend'], vae_attention=accel['attention_backend'])
    steps = 4 if args.variant == 'standard' else 3
    from model.attention import probe_backend
    backends = []
    for backend in dict.fromkeys([accel['attention_backend'], 'flash', 'native']):
        try:
            probe_backend(backend)
            backends.append(backend)
        except Exception as error:
            print(json.dumps({'skipped_backend': backend, 'reason': str(error)}), flush=True)
    if args.search == 'fixed':
        probe_backend('_flash_3')
        backends = ['_flash_3']
    gemms = (['triton'] if args.search in ('fast', 'fixed') else ['triton', 'torch']) if accel['fused'] else ['torch']
    rows = []
    for tiles in (('adaptive',) if args.search == 'fixed' else ('adaptive', 'native')):
        pipe.pipe.vae.tile_layout = tiles
        for attention in backends:
            pipe.transformer.set_attention_backend(attention)
            pipe.pipe.vae.core.set_attention_backend(attention)
            for gemm in gemms:
                for module in pipe.transformer.modules():
                    if isinstance(module, Int8Linear): module.gemm = gemm
                label = f'{tiles}_{attention}_{gemm}'
                measurements = []
                for i in range(args.warmups + args.repeats):
                    state, timing = pipe.generate(condition, [], canvas_height, width, frames_requested, args.seed, steps)
                    if i >= args.warmups: measurements.append(timing)
                    print(json.dumps({'configuration': label, 'iteration': i, 'warmup': i < args.warmups, **timing}), flush=True)
                median = lambda key: statistics.median(x[key] for x in measurements)
                row = {'configuration': label, 'attention': attention, 'gemm': gemm, 'tiles': tiles,
                    'dit_ms': median('dit_ms'), 'video_decoder_ms': median('video_decoder_ms'),
                    'dit_plus_decoder_ms': statistics.median(x['dit_ms']+x['video_decoder_ms'] for x in measurements),
                    'generation_and_decode_ms': median('generation_and_decode_ms'),
                    'peak_gib': max(x['peak_allocated_bytes'] for x in measurements)/2**30,
                    'measurements': measurements}
                top = (canvas_height-height)//2
                frames = [f.crop((0,top,width,top+height)) for f in state['videos'][0][:frames_requested]]
                if len(frames) != frames_requested:
                    raise RuntimeError('generated frame count differs from request')
                audio = state['audio'][0][..., :round(frames_requested * int(state['sampling_rate']) / 24)]
                encode_video(frames, fps=24, output_path=str(out/(label+'.mp4')), audio=audio, audio_sample_rate=state['sampling_rate'])
                rows.append(row)
                (out/'progress.json').write_text(json.dumps(rows, indent=2))
                print(json.dumps({'median': row}), flush=True)
    best = min(rows, key=lambda x:x['dit_plus_decoder_ms'])
    import shutil
    shutil.copyfile(out/(best['configuration']+'.mp4'), out/'video.mp4')
    report = {'variant':args.variant,'weights':str(weights),'transformer':pipe.transformer_record,'precision':'W8A8', 'steps':steps,
        'geometry':[width,height,frames_requested],'native_canvas':[width,canvas_height],'fps':24,'seed':args.seed,'prompt':prompt,
        'allow_terminal_padding':args.allow_terminal_padding,
        'warmups':args.warmups,'repeats':args.repeats,'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'hardware_start':hardware_start,
        'hardware_end':hardware(),'search':args.search,'acceleration':accel,'light_vae':pipe.vae_record,
        'conditioning':conditioning,'excluded':['conditioning','weight loading','file encoding'],
        'selection_metric':'median of per-call DiT + video decoder CUDA times',
        'best':best,'configurations':rows,'source':source,'source_changed_during_run':changed_sources(source)}
    (out/'latency.json').write_text(json.dumps(report,indent=2))
    print('Best measured configuration: '+json.dumps(best),flush=True)


if __name__ == '__main__':
    main()
