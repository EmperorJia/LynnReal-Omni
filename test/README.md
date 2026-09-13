# Default sampling inputs

`defaults.json` selects one prompt and the relevant media for each task. Both variants share T2V and TI2V inputs.

| Task | Prompt | Input |
| --- | --- | --- |
| T2V | `t2v.txt` | Clay bakery, using the existing comparison prompt unchanged |
| TI2V | `ti2v.txt` | `assets/times_square.png`, native first-frame conditioning |
| V2V | `v2v.txt` | `assets/snowboard_style.png` for appearance and `assets/snowboard.mp4` for aligned motion |
| Ref2V | `ref2v.txt` | `assets/rider.png` for person, clothing and motorcycle appearance |

The media are physical copies of existing project inputs; the snowboard game was not rendered again. Defaults are examples for exercising each interface, not quantitative quality benchmarks. To compare with earlier samples, explicitly use their original seeds, frame counts and backend as well as their prompts.

Sampling results and complete logs are saved in `output/`, as described in
[the launcher guide](../script/sample/README.md). The published inputs support
the sampling entry points; internal validation programs are kept separately.
