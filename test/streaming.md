# Standard BF16 streaming

Run from the release root:

```bash
bash script/sample/standard/bf16/stream.sh --frame 5s
bash script/sample/standard/bf16/stream.sh --frame 30s
```

This single entry generates an image-conditioned first chunk and continues its latent history. Defaults: the bundled rainy-street image and normal-speed motion prompts, seed 7, 1344×768, 24 fps, four Standard BF16 DiT evaluations for the bootstrap and first seven continuation chunks, then four initial plus two refinement evaluations per chunk, and the official VAE. Weights come from `weight/standard` and its release-local codec configuration.

```bash
# Inspect only the first native 22-frame chunk.
bash script/sample/standard/bf16/stream.sh --first-chunk-only

# Supply a different scene and a continuation plan.
bash script/sample/standard/bf16/stream.sh \
  --image /path/to/first.png --prompt-file /path/to/initial.txt \
  --captions /path/to/continuations.json --frame 5s --seed 7 \
  --output output/my_stream
```

`--image` must be 1344×768. `--prompt` accepts literal first-chunk text; `--prompt-file` reads it from a file. Custom scenes require their own continuation captions, supplied as a JSON array of strings. `--frame 5`, `--frame 5s` and `--frame 120f` all request 120 output frames. `--resolution` currently accepts only `768p`/`1344x768` for this validated configuration. Use `LYNNREAL_PYTHON=/path/to/python` when the runtime interpreter differs from `python`.

The bootstrap produces seven latent frames and 22 RGB frames of native decoder context. Streaming delivers its first 17 RGB frames, retains the decoder's five-frame overlap, and then predicts five new latent frames per chunk while fixing the last two history latents. Each continuation delivers 17 additional RGB frames. The final chunk is trimmed only to the requested total length. This reproduces the official decoder's overlap operation; it does not change playback speed or interpolate frames. A five-second run performs four bootstrap forwards plus seven groups of four continuation forwards.

The first seven continuation chunks retain four dense sink latents, no middle-history anchors, and the first chunk's image-aware text context. For clips exceeding 136 frames, subsequent chunks retain one initial sink and encode the preceding delivered chunk's first and last frames into the text context. Starting with continuation chunk eight, a second denoising round adds independent Gaussian noise to the first-pass clean video latent and uses the native tail sigmas `[0.92307693, 0.8, 0]` for two additional DiT forwards. No spatial low-frequency anchor is applied. Audio stays at its first-pass clean endpoint during refinement. The first-image latent-moment constraint applies in both rounds. The two fixed overlap latents and decoder overlap remain unchanged during this switch. Recent stored latents are restored from the complete latest native window, rather than discarded when reducing the sinks. History remains bounded to seven latent frames. Local RoPE preserves native target cadence and overlap distances. The first-image latent-moment constraint remains active (`--appearance-strength 1`); this is part of sampling, not RGB postprocessing. No extra DiT reference-image tokens are added.

This is the adopted unanchored two-pass stream configuration. It preserves the established five-second prefix. A 30-second run uses 242 DiT forwards in total: four for the bootstrap, seven groups of four for the prefix, and 35 groups of six for the remaining chunks. It does not eliminate all local texture, face or vehicle-shape drift, or establish robustness across arbitrary scenes. `--no-refresh-context` restores the earlier fixed-context, four-step-only policy for comparisons; it also disables second-pass refinement.

Prompts describe ordinary movement during each short native prediction window, with a continuing gait phase. Avoid describing a whole multi-second action as if it should complete in every chunk, or prescribing tiny displacements that imply slow motion. The entry selects the bundled five- or thirty-second plan by duration. Other plans require enough captions: `ceil((frames - 17) / 17)`.

On one GPU, refreshed Qwen conditioning stages its BF16 layers through CPU memory while the DiT remains resident. This increases end-to-end latency in addition to the six DiT evaluations in each refined continuation chunk. With a second GPU, run the following service in another terminal, then pass its queue to the sampler:

```bash
CUDA_VISIBLE_DEVICES=1 python script/conditioner.py --stream-context \
  --weights weight/standard --queue output/stream_conditioner \
  --cache output/stream_conditioner/cache

CUDA_VISIBLE_DEVICES=0 bash script/sample/standard/bf16/stream.sh --frame 30s \
  --conditioner-service output/stream_conditioner
```

Use a fresh queue for a new service. Stop the service after sampling with `touch output/stream_conditioner/STOP`. Conditioning time is recorded separately from DiT and decoder time. The single-GPU and service paths produced identical latent and RGB outputs on the same continuation chunk.

Every run stores `video.mp4`, `sample.log`, and `run.json` together. The run plan distinguishes four-step bootstrap/prefix sampling from 4+2-step continuation and records the planned total forward count. Per-chunk metadata records the actual forward counts. `first_chunk/` contains its prompt, input image, raw RGB, latents, timings and metadata. `continuation/` contains per-chunk prompts, seeds, raw RGB, latent traces, timings and final `video.json`. `--audit-codec` adds a full-sequence decoder check against incremental output; it is a verification pass and does not change generated frames. `--dry-run` checks the arguments and prints the commands without loading models.

This entry starts from an image. For an actual input-video continuation, use the separate `bf16/v2v.sh --continuation` interface; it is not silently converted to an image initialization here.
