"""Generate native audiovisual continuation from a dense video prefix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--condition-frames', type=int, default=22)
    p.add_argument('--source-start-frame', type=int, default=0)
    p.add_argument('--source-window', choices=('explicit', 'tail'), default='explicit')
    p.add_argument('--frames', type=int, default=119)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--height', type=int, default=768)
    p.add_argument('--width', type=int, default=1344)
    p.add_argument('--dit-offload-blocks', type=int, default=18)
    p.add_argument('--posterior', choices=('mode','sample'), default='mode')
    args = p.parse_args()
    if args.output.exists() or args.output.suffix != '.mp4':
        p.error('select a new .mp4 output')
    if min(args.frames,args.steps,args.height,args.width) < 1 or args.source_start_frame < 0:
        p.error('positive geometry, steps and length; nonnegative source start required')
    if args.height % 2 or args.width % 2 or args.condition_frames < 5 or (args.condition_frames-5)%17:
        p.error('even output dimensions and 17*n+5 source frames required')
    width = ((args.width + 31) // 32) * 32
    height = ((args.height + 31) // 32) * 32
    import numpy as np
    import torch
    from PIL import Image
    from imageio_ffmpeg import get_ffmpeg_exe
    from diffusers.utils.export_utils import encode_video
    from model.pipeline import Pipeline, encode_conditioning
    from model.continuation import continue_video
    from model.provenance import snapshot_sources, changed_sources
    from model.weights import sha256
    source_record = snapshot_sources(args.cache)
    prompt = args.prompt_file.read_text().strip()
    if not prompt: p.error('prompt must not be empty')
    # Decode an actual temporal window; never replace the input with its last image.
    filters = (f'fps=24,trim=start_frame={args.source_start_frame}:end_frame={args.source_start_frame+args.condition_frames},'
               f'scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}')
    window = None
    if args.source_window == 'tail':
        from model.video_window import read_video_window
        source_rgb, window = read_video_window(args.video,args.condition_frames,width,height,'tail')
        raw = source_rgb.tobytes()
    else:
        raw = subprocess.check_output([get_ffmpeg_exe(),'-v','error','-threads','2','-i',str(args.video),
            '-vf',filters,'-frames:v',str(args.condition_frames),'-f','rawvideo','-pix_fmt','rgb24','-'])
    shape = (args.condition_frames,height,width,3)
    if len(raw) != np.prod(shape): raise ValueError('source video is shorter than the requested window')
    pixels = torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).reshape(shape).copy()).permute(3,0,1,2)[None].float()/255
    conditioning, encode_timing = encode_conditioning(args.weights,prompt,[],height,width,
                                                     args.frames,args.cache)
    pipe = Pipeline(args.weights,attention_backend='native')
    rgb, audio, timing = continue_video(pipe,conditioning,pixels,args.frames,args.seed,args.steps,
                                       args.dit_offload_blocks,args.posterior=='mode')
    left, top = (width-args.width)//2, (height-args.height)//2
    frames = [Image.fromarray(x).crop((left,top,left+args.width,top+args.height))
              for x in (rgb[0].permute(1,2,3,0)*255).round().byte().cpu().numpy()]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    encode_video(frames,fps=24,output_path=str(args.output),audio=audio[0],audio_sample_rate=32000)
    record = {'mode':'dense-continuation','weights':str(args.weights.resolve()),'prompt':prompt,
        'transformer':pipe.transformer_record,
        'source_video':{'path':str(args.video.resolve()),'sha256':sha256(args.video),
                        'start_frame':args.source_start_frame if window is None else None,
                        'window':window,'frames':args.condition_frames},
        'native_canvas':[width,height],'requested_geometry':[args.width,args.height,args.frames],
        'seed':args.seed,'steps':args.steps,'fps':24,'conditioning':encode_timing,
        'timings':[timing],
        'output':str(args.output.resolve()),'sha256':sha256(args.output),'source':source_record,
        'source_changed_during_run':changed_sources(source_record),'audio_sample_rate':32000,
        'audio_contract':'future-only stereo rows; prefix duration offset in RoPE; no prefix waveform trim',
        'device':torch.cuda.get_device_name(),'torch':torch.__version__}
    args.output.with_suffix('.json').write_text(json.dumps(record,indent=2))
    args.output.with_suffix('.txt').write_text(prompt+'\n')
    print(json.dumps(timing),flush=True)
    print(f"DiT ({args.steps} steps): {timing['dit_ms']/1000:.3f}s | video decoder: {timing['video_decoder_ms']/1000:.3f}s",flush=True)


if __name__ == '__main__':
    main()
