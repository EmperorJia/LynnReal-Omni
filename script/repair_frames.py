"""Restore each source frame independently with the shared four-step Standard DiT."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Tee:
    def __init__(self, original, log):self.original, self.log = original, log
    def write(self, value):
        self.original.write(value)
        return self.log.write(value)
    def flush(self):self.original.flush();self.log.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--guide-video', type=Path, help='optional preprocessed video with exactly the same frames and timestamps')
    p.add_argument('--appearance-reference', type=Path)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=ROOT/'weight/standard')
    p.add_argument('--indices', help='comma-separated zero-based source indices; default: every source frame')
    p.add_argument('--model-frames', type=int, default=22, help='internal image-edit clip length; one selected image is kept per source frame')
    p.add_argument('--image-frame', type=int, default=11)
    p.add_argument('--reference-short-edge', type=int, default=768)
    p.add_argument('--offload-blocks', type=int, default=0)
    p.add_argument('--seed', type=int, default=7, help='same seed for every independent source-frame edit')
    p.add_argument('--keep-clips', action='store_true', help='retain internal image-edit clips for pilot review')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    from model.weights import standard_transformer, sha256
    transformer = standard_transformer(a.weights)
    if a.output.exists():p.error('output exists; choose a new directory')
    if a.model_frames < 22 or not 0 <= a.image_frame < a.model_frames:p.error('the official video VAE requires an internal clip of at least 22 frames; select one frame inside it')
    if a.reference_short_edge < 32 or a.reference_short_edge % 32 or a.offload_blocks < 0:p.error('invalid reference size or offload count')
    for path in [a.video, a.guide_video, a.appearance_reference, a.prompt_file]:
        if path is not None and not path.is_file():p.error(f'missing input: {path}')
    prompt = a.prompt_file.read_text().strip()
    if not prompt:p.error('empty prompt')
    import av
    with av.open(str(a.video)) as reader:
        stream = reader.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.average_rate)
        timestamps = [float(f.time) for f in reader.decode(video=0)]
    if fps != 24 or width % 32 or height % 32:p.error('use a 24-fps source with dimensions divisible by 32')
    count = len(timestamps)
    try:indices = sorted(set(int(s) for s in a.indices.split(','))) if a.indices else list(range(count))
    except ValueError:p.error('--indices must contain comma-separated integers')
    if not indices or indices[0] < 0 or indices[-1] >= count:p.error('indices outside source video')
    identity = dict(source_video=str(a.video.resolve()), source_sha256=sha256(a.video),
        guide_video=str(a.guide_video.resolve()) if a.guide_video else None,
        guide_sha256=sha256(a.guide_video) if a.guide_video else None,
        appearance_reference=str(a.appearance_reference.resolve()) if a.appearance_reference else None,
        appearance_sha256=sha256(a.appearance_reference) if a.appearance_reference else None,
        prompt=prompt, source_frames=count, fps=24, native_canvas=[width,height],
        model_frames=a.model_frames, selected_image_frame=a.image_frame,
        seed=a.seed, seed_policy='same independent seed for every source frame',
        reference_short_edge=a.reference_short_edge, steps_per_frame=4)
    if a.dry_run:
        print(json.dumps(dict(identity=identity, indices=indices, transformer=transformer), indent=2));return
    a.output.mkdir(parents=True)
    log=(a.output/'sample.log').open('w',buffering=1)
    sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    frames_dir=a.output/'frames';frames_dir.mkdir()
    input_dir=a.output/'inputs';input_dir.mkdir()
    (a.output/'prompt.txt').write_text(prompt+'\n')
    from model.provenance import snapshot_sources, changed_sources
    source=snapshot_sources(a.output/'cache')
    wanted=set(indices);paths={};guide_times=[]
    with av.open(str(a.guide_video or a.video)) as reader:
        for i, frame in enumerate(reader.decode(video=0)):
            if frame.width != width or frame.height != height:raise ValueError('guide geometry differs from source')
            guide_times.append(float(frame.time))
            if i in wanted:
                path=input_dir/f'{i:06d}.png';frame.to_image().save(path);paths[i]=[path]
                if a.appearance_reference:paths[i].append(a.appearance_reference)
    if len(guide_times)!=count or any(abs(x-y)>1e-6 for x,y in zip(guide_times,timestamps)):
        raise ValueError('guide video must preserve every source frame and timestamp')
    import torch
    from model.pipeline import Pipeline, encode_conditioning, release_memory
    conditioning={};resident={};started=time.perf_counter()
    # Encode the assigned frames together, then release Qwen before loading DiT.
    for i in indices:
        conditioning[i]=encode_conditioning(a.weights,prompt,paths[i],height,width,a.model_frames,a.output/'cache',
            native_keyframes=False,resident=resident,resident_device='cuda',reference_image_short_edge=a.reference_short_edge)
        print(json.dumps(dict(phase='conditioning',frame=i,**conditioning[i][1])),flush=True)
    resident.clear();release_memory()
    pipe=Pipeline(a.weights,reference=True,native_keyframes=False,vae_offload=True,
        dit_offload_blocks=a.offload_blocks,attention_backend='native',reference_image_short_edge=a.reference_short_edge)
    records=[]
    for i in indices:
        state,timing=pipe.generate(conditioning[i][0],paths[i],height,width,a.model_frames,a.seed,4)
        frames=state['videos'][0]
        if len(frames)<a.model_frames:raise RuntimeError('internal image-edit clip is incomplete')
        image=frames[a.image_frame];destination=frames_dir/f'{i:06d}.png';image.save(destination)
        if a.keep_clips:
            from diffusers.utils.export_utils import encode_video
            clips=a.output/'internal_clips';clips.mkdir(exist_ok=True)
            encode_video(frames[:a.model_frames],fps=24,output_path=str(clips/f'{i:06d}.mp4'))
        row=dict(source_frame=i,source_timestamp=timestamps[i],output=str(destination.resolve()),
            output_sha256=sha256(destination),input_sha256=sha256(paths[i][0]),conditioning=conditioning[i][1],**timing)
        records.append(row)
        with (a.output/'measurements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(phase='repair',**row)),flush=True)
        del state,frames,image
    record=dict(mode='frame-repair',identity=identity,indices=indices,frames=len(records),fps=24,
        transformer=pipe.transformer_record,steps_per_frame=4,total_dit_forwards=sum(r['actual_dit_forwards'] for r in records),
        temporal_interpolation=False,generated_frame_feedback=False,audio=False,postprocessing=False,
        measurements=records,seconds=time.perf_counter()-started,source=source,source_changed_during_run=changed_sources(source))
    (a.output/'frames.json').write_text(json.dumps(record,indent=2)+'\n')
    if indices==list(range(count)):
        import subprocess
        from imageio_ffmpeg import get_ffmpeg_exe
        video=a.output/'video.mp4'
        subprocess.run([get_ffmpeg_exe(),'-v','error','-framerate','24','-i',str(frames_dir/'%06d.png'),
            '-frames:v',str(count),'-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',str(video)],check=True)
        record.update(output=str(video.resolve()),output_sha256=sha256(video))
        (a.output/'video.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
