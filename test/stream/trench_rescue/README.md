# Trench rescue streaming case

Three people, a supported stretcher, coordinated hand grips and foot contact, wet timber/sandbag parallax and backward camera tracking. The supplied image is an extracted existing generated frame; only that image is an input to this new stream. No source video or later frame conditions the continuation. Prompts cover 42 native continuation windows at 24fps; native decoded overlap still delivers 17 new frames per chunk.

```bash
bash script/sample/standard/bf16/stream.sh --frame 30s --image test/stream/trench_rescue/first.png --prompt-file test/stream/trench_rescue/initial.txt --captions test/stream/trench_rescue/continuations_30s.json --seed 7
```

The current stream exports video only; the native soundscape text remains part of H3 conditioning.
