"""Preserve native RGB rounding while transferring one byte per channel."""
from PIL import Image
import torch


def configure_video_output(processor):
    native = processor.postprocess_video

    def postprocess(video, output_type="np", **kwargs):
        if (output_type != "pil" or kwargs or processor.config.do_normalize
                or video.device.type != "cuda" or video.dtype != torch.float32):
            return native(video, output_type, **kwargs)
        # H3 already supplies clamped [0,1] RGB. Match NumPy's FP32 multiply
        # and ties-to-even rounding; only the new temporary is modified.
        pixels = video.mul(255).round_().to(torch.uint8)
        pixels = pixels.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()
        return [[Image.fromarray(frame) for frame in clip] for clip in pixels]

    processor.postprocess_video = postprocess


def encode_video(video, fps, output_path, audio=None, audio_sample_rate=None):
    """Lossless RGB H.264 with explicit video/audio clocks and stereo AAC."""
    from fractions import Fraction
    import av
    import numpy as np
    if not video:
        raise ValueError('video must contain at least one frame')
    if int(fps) != fps or fps <= 0:
        raise ValueError('fps must be a positive integer')
    with av.open(str(output_path), 'w') as container:
        stream = container.add_stream('libx264rgb', rate=int(fps))
        stream.width, stream.height = video[0].size
        stream.pix_fmt = 'rgb24'
        stream.time_base = stream.codec_context.time_base = Fraction(1, int(fps))
        stream.options = {'crf': '0', 'preset': 'fast'}
        audio_stream = None
        if audio is not None:
            if audio_sample_rate is None or int(audio_sample_rate) <= 0:
                raise ValueError('positive audio_sample_rate is required with audio')
            rate = int(audio_sample_rate)
            audio_stream = container.add_stream('aac', rate=rate)
            audio_stream.codec_context.sample_rate = rate
            audio_stream.codec_context.layout = 'stereo'
            audio_stream.time_base = audio_stream.codec_context.time_base = Fraction(1, rate)
        # Finalize both encoder clocks before writing the MP4 header.
        container.start_encoding()
        for i, image in enumerate(video):
            frame = av.VideoFrame.from_ndarray(np.asarray(image), format='rgb24')
            frame.pts = i
            frame.time_base = Fraction(1, int(fps))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            samples = audio
            if samples.ndim != 2:
                raise ValueError('audio must have two dimensions and two channels')
            if samples.shape[0] == 2 and samples.shape[1] != 2:
                samples = samples.T
            if samples.shape[1] != 2:
                raise ValueError('audio must contain two channels')
            # Preserve the upstream float -> packed int16 conversion exactly.
            if samples.dtype != torch.int16:
                samples = (samples.clamp(-1, 1) * 32767).to(torch.int16)
            frame = av.AudioFrame.from_ndarray(samples.contiguous().reshape(1, -1).cpu().numpy(),
                                                format='s16', layout='stereo')
            clock = Fraction(1, rate)
            frame.sample_rate, frame.time_base, frame.pts = rate, clock, 0
            resampler = av.AudioResampler(format=audio_stream.codec_context.format,
                                         layout='stereo', rate=rate)
            pts = 0
            for resampled in [*resampler.resample(frame), *resampler.resample(None)]:
                resampled.sample_rate, resampled.time_base, resampled.pts = rate, clock, pts
                pts += resampled.samples
                for packet in audio_stream.encode(resampled):
                    if not packet.time_base:packet.time_base = clock
                    container.mux(packet)
            for packet in audio_stream.encode():
                if not packet.time_base:packet.time_base = clock
                container.mux(packet)
