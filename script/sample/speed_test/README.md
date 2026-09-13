# Warm T2V speed

Run on separate idle GPUs, or sequentially on one GPU:

```bash
bash script/sample/speed_test/standard.sh
bash script/sample/speed_test/flash.sh
# The native 768p, five-second request:
bash script/sample/speed_test/standard.sh --resolution 768p --search fast
bash script/sample/speed_test/flash.sh --resolution 768p --search fast
```

Both use the same 22-frame, 24-fps pouring-tea prompt and seed. The native
960x544 canvas is center-cropped to 960x540. Standard uses four W8A8 denoiser
calls; Flash uses three calls with its trained W8A8 weights. Both load the
lightweight decoder from `weight/light-vae`. The script sweeps available
attention backends, Torch/Triton INT8 GEMM, and native/adaptive decoder tiles.
Each configuration has two warmups and five measured calls.

`--resolution 768p` defaults to 120 delivered frames at 24 fps on a
1344x768 canvas; native temporal padding is included in computation.
`--frames` overrides the frame count. `--search fast` searches attention
backends and decoder tiling with fused Triton INT8 GEMM; the default
`--search full` additionally measures Torch INT8 GEMM. No refinement pass
is used in either speed test.

For duration scaling with the previously selected configuration, use
`--resolution 768p --frames 240 --search fixed` (10 seconds) or
`--frames 360` (15 seconds). `fixed` requires FA3 and Triton fusion and
uses the compiled adaptive light decoder; it does not repeat the backend
sweep. These are single-clip generations, without streaming or interpolation.

Outputs are in `output/speed_test/<variant>/<run>/`: each configuration's video,
`video.mp4` for the fastest measured configuration, `run.log`, `prompt.txt`, and
`latency.json`. The report distinguishes CUDA DiT time, CUDA decoder time,
their per-call sum, and generation wall time including decoding/postprocessing.
Conditioning, loading, compilation warmup, and file encoding are excluded.
Adaptive tiling changes the decoder context, so it is named explicitly.

Verified on 2026-09-13 using eight NORMAL single-H100 jobs. Each selected
configuration has two independent GPU replications, two warmups and five
measurements per replication. All selected configurations use FA3, fused
Triton W8A8, and a compiled FP16 light decoder with adaptive tiles.
Numbers below are pooled medians of ten calls, in milliseconds.

| Input | Model | DiT | Decoder | DiT + decoder | Generate wall |
|---|---|---:|---:|---:|---:|
| 540p / 22 frames | standard (4 steps) | 711.8 | 116.3 | 829.3 | 919.1 |
| 540p / 22 frames | flash (3 steps) | 257.1 | 115.2 | 371.5 | 462.6 |
| 768p / 120 frames (5s) | standard (4 steps) | 18292.7 | 1723.4 | 20018.1 | 20514.1 |
| 768p / 120 frames (5s) | flash (3 steps) | 5473.9 | 1724.5 | 7197.7 | 7660.5 |

The sum column is the median of per-call sums, not a sum of component medians.
These are complete model measurements, not extrapolations from a standalone
decoder benchmark. The 768p Torch INT8 GEMM controls were slower than Triton.
Original videos and logs are under output/speed_test/<variant>/retest_20260913_*;
output/speed_test/retest_20260913_review/index.html links the selected videos.
The candidate search is bounded; these are the fastest measured configurations.

For exactly 15 seconds, add `--allow-terminal-padding`: native alignment
rounds 360 requested frames to 362, exceeding H3's default aligned-duration
limit. This explicit boundary experiment admits those two padding frames
and retains the first 360 frames at 24 fps. Their computation is included
in timing. It is recorded in `latency.json`, not a change to model weights.

## Component ablations

```bash
python script/sample/speed_test/ablation.py --variant standard --name components_standard
python script/sample/speed_test/ablation.py --variant flash --name components_flash
python script/sample/speed_test/structure_ablation.py --name structure_flash
```

Run each command on a single H100 in the release environment. The component
sequence changes one execution setting at a time on the same loaded model,
with two warmups and five timed calls per setting. Flash keeps its trained
INT8 weights throughout. The structural controls separately disable Flash
video-token compression and compare full/light decoder depth with identical
native tiles, batching, precision and input latents. They are controls without
retraining, so output quality is assessed separately from latency.

The 2026-09-13 paired component runs end at 843 ms (standard) and 377 ms
(Flash) for DiT + decoder; generation wall times are 959 and 479 ms.
The table-ready LaTeX is in paper/latency_long_20260913/latency_section.tex
relative to the parent paper directory. Original per-stage videos, logs,
latent comparisons and decoder controls are linked from
output/speed_test/ablation_20260913_review/index.html.
