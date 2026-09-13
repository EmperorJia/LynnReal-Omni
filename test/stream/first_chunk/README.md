# First-chunk sampling

The first chunk and stream now share one entry:

```bash
bash script/sample/standard/bf16/stream.sh --first-chunk-only
```

Run from `lynnreal_release`. The default uses `street.png`, the normal-speed prompt in `../normal_motion/initial.txt`, and seed 7. The historical `street.txt` remains unchanged for reproducing the earlier slow-motion baseline. This produces 22 frames at 24 fps, 1344×768, using four Standard BF16 DiT forwards and the official decoder.

Use `--image`, `--prompt` or `--prompt-file`, `--seed`, `--appearance-strength`, and `--output` for another input. Source images must be 1344×768. Video and the complete log are placed in the output root; first-chunk RGB, latents, source image, prompt, timing and metadata are under `first_chunk/`.

The fixed target head and first-image latent-moment constraint remain part of sampling. Historical validation covered street, snowboard and rider inputs with seeds 7 and 19. This does not establish stability under intentional lighting changes. See [streaming instructions](../../streaming.md) for continuation, caption plans and decoding details.
