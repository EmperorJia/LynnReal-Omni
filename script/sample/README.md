# Sampling launchers

Standard: `standard/bf16/{t2v,ti2v,v2v,ref2v}.sh` or the separate
`standard/int8/` launchers. Flash: `flash/int8/{t2v,ti2v}.sh`.

Options: `--frame 5s` (or `120f`), `--resolution 768p`, `--prompt TEXT`,
`--image PATH`, `--ref_image A,B`, `--ref_video VIDEO` (`--ref__video` is an alias),
`--seed N`. Inputs default to the distinct cases in `test/defaults.json`.
Videos, request metadata, full logs and timing JSON are stored together in
`output/<variant>/<precision>/<task>/<run>/`. Use `--dry-run` to inspect the command.

Standard BF16 uses the official VAE and CPU staging when needed. Standard INT8
uses W8A8 projections and the lightweight VAE. Flash uses its packaged W8A8
projections (INT8 weights and INT8 activations), with BF16 sensitive layers. Timings must identify codec, precision, GPU and offload.

## BF16 video continuation

`standard/bf16/v2v.sh` continues the end of an input video. It encodes the final
22 frames at 24 fps as a temporal prefix and generates only new frames after
that prefix, with four DiT evaluations. It does not recolor the input timeline.
Use a prompt describing what happens next while retaining the original scene.

```bash
bash script/sample/standard/bf16/v2v.sh --ref_video input.mp4 --frame 5s \
  --prompt 'Continue the same shot: the rider finishes the turn and travels down the slope.'
```

Separate appearance images and benchmark repetitions are unsupported by this
continuation entry. For aligned video editing use `script/sample.py --mode video-edit`
with `--reference input.mp4 --aligned-reference 0` and the usual model/output options.
The separate INT8 V2V entry retains its existing aligned editing behavior.

## Flash refinement

```bash
bash script/sample/flash/int8/t2v_refined.sh --frame 5s --resolution 768p
```

The default is the training Wuxia prompt, with shot timestamps scaled to the
requested duration. The first pass uses three Flash evaluations at a smaller
canvas. Video latents are enlarged spatially with bilinear interpolation, then
refined with **two additional evaluations**, using sigma `0.94 -> 12/14 -> 0`.
The final transition matches the trained three-step sampler; the initial
strength balances detail recovery and preservation of the first pass. The
audio latent stays clean and is not updated. Both passes use
`weight/flash`, and the output directory contains `first_pass.mp4`, `video.mp4`,
`run.log`, normalized `first_latents.safetensors` / `final_latents.safetensors`,
and per-pass timing/step records. Cached-latent experiments and this launcher
use the same prompt trimming and FP32 noise grid. The default costs **3+2 = 5 DiT
forwards**. Use `--refine-steps 1` for the previous `12/14 -> 0` one-pass
refinement (four forwards in total). `--refine-strength` overrides the initial
noise. `--refine-schedule linear` retains the earlier uniform-grid experiment;
adding uniform low-noise steps produced texture artifacts in our Wuxia tests.
No pixel sharpening or frame interpolation is applied.

`--refine-first-height 640` or `768` increases the first-pass height cap from
the default `544`, while retaining the requested final resolution and 3+2
evaluations. Larger canvases cost more compute and change the initial noise
shape even with the same seed; they are experimental quality options.

The default was checked on five-second Wuxia, clay-bakery, and motorcycle clips
with fixed prompts, seeds, and first-pass latents. It improves visible material
detail in these examples but can still alter fine geometry. This is not a
guarantee of improvement for every prompt. The complete candidate comparisons
are under `output/flash/int8/refinement_v2/`.

The spatial two-pass design follows [H3 latent upscaling workflows](https://github.com/rockerBOO/h3-latent-upscaler).
The implementation uses H3 flow interpolation `(1-sigma)*clean + sigma*noise`
once, avoiding a second scaling of an already-noised latent. Temporal latents,
frame rate, and first-pass audio are preserved.

## Attention selection

All short-clip launchers accept `--attention-backend auto|native|_native_cudnn|_native_flash|flash|_flash_3`.
`flash` denotes FlashAttention 2. Auto probes FA3, FA2, cuDNN SDPA, PyTorch Flash SDPA, then
native SDPA; unsupported kernels fall back before generation. Probes use their
own random generator and check numerical error against FP32 math attention.
`LYNNREAL_ATTENTION=native` pins the auto probe to native SDPA. A changed backend
can produce different pixels with the same seed; use a fixed backend for exact
reproduction. The streaming launcher retains its explicitly selected native
attention path.

INT8 launchers share hardware selection and kernel caching through `run.py`.
INT8 defaults use the compiled lightweight decoder, adaptive tiles and automatic
decoder attention selection, matching the accelerated timing configuration.
Tile geometry and attention changes can alter reconstructed pixels; use
`--vae-tiles native --vae-attention-backend native` to retain the earlier decoder
configuration. BF16 defaults are unchanged. Two preparation calls precede measured
inference and are logged separately, including any compilation or tuning.
`--warmups 0` skips these calls but a first-use measurement can include preparation.
For five measured repetitions:

```bash
bash script/sample/standard/int8/t2v.sh --frame 22f --resolution 540p \
  --vae-tiles adaptive --vae-attention-backend auto --warmups 2 --repeats 5
bash script/sample/flash/int8/t2v.sh --frame 22f --resolution 540p \
  --vae-tiles adaptive --vae-attention-backend auto --warmups 2 --repeats 5
```

INT8 outputs use lossless RGB H.264 (no YUV420 conversion); audio retains the
upstream encoding format. Use a player that supports H.264 RGB/4:4:4. File
encoding is outside the reported generation timing. BF16 launchers retain their
existing encoding and inference settings.

On Hopper, grouped INT8 GEMM and residual/FFN-normalization fusion preserve the
existing INT8 rounding and accumulation rules. Other GPUs retain the original
GEMM ordering and separate residual operation. Device- and source-specific tile
selections, Triton binaries and Inductor artifacts are cached under `output/.cache`;
new shapes can still require compilation. Optional compiler or device-support
failures warn and use the native operation. Invalid inputs, insufficient memory,
and a failed CUDA context remain real errors rather than false successful runs.

Measured public T2V commands on H100 80GB, 540p/22 frames, seed 7, two preparation
calls and five measured repetitions: DiT plus video decoder medians are 855 ms
(Standard, FA3), 381 ms (Flash, FA3), 986 ms (Standard, FA2), and 420 ms (Flash,
FA2). Corresponding generation wall times are 940, 458, 1068, and 500 ms.
These are warm measurements, excluding loading, conditioning and file encoding;
they are not cold command-to-video times. Logs and lossless videos are under
`output/{standard,flash}/int8/t2v/final_public_h100_*_20260913/`.

### Standard BF16 stream

`bash script/sample/standard/bf16/stream.sh --frame 5s` runs the first image-conditioned chunk and latent continuation through one entry. Add `--first-chunk-only` to inspect the 22-frame bootstrap. The default is the 768p normal-motion case, with four-step bootstrap/prefix sampling and unanchored 4+2-step continuation after seven continuation chunks; see [the streaming guide](../../test/streaming.md) for custom inputs and captions.
