---
name: lynnreal-prompt
description: Write or rewrite prompts for LynnReal-Omni DiT generation, reference control, editing and streaming. Match the actual standard or Flash conditioning layout and evaluate changes with paired samples.
---

# LynnReal prompt writing

Produce a usable prompt for the selected LynnReal inference route. Preserve the requested story, reference identities, dialogue and style. Prompt changes can clarify an action; they cannot guarantee physical correctness or repair a wrong conditioning layout.

## Select the actual conditioning contract

Read the selected weight bundle's `inference_config.json` and the inference arguments before choosing a format. The task name alone is insufficient.

| Route | Prompt format | Conditioning meaning |
| --- | --- | --- |
| Standard T2V | Three sections | No supplied visual reference |
| Standard image reference (`unified_ref_native`) | Six sections | Ordered reference pictures; an opening-frame request is semantic |
| Standard native TI2V (`--native-keyframes`) | Three sections plus keyframe alignment instruction | Native first/last image latent layout; verify the explicit CLI flag |
| Standard reference video | Six sections | Video supplies the explicitly named appearance, motion or scene characteristics |
| Body/hand pose, mesh/game render, video editing | Six sections | Appearance image and/or frame-aligned control video; alignment must also be enabled in code |
| Standard image editing | Six sections | Camera-locked generated clip; the release selects a frame for PNG and retains the companion video |
| Flash T2V | Three sections | Three actual denoiser evaluations by default |
| Flash I2V / first-last images | Three sections plus keyframe alignment instruction | Native `fl2va` first/last keyframe layout |
| Standard native stream | Six sections for video-context conditioning; three sections for native current/initial image conditioning | Actual input video; name each image role explicitly. Four forwards per chunk; fixed original-image retention does not establish drift-free output |
| Standard formal stream (`stream_formal.sh`) | Plain 17-frame caption, or `--v2v-caption-plan` JSON list | Default pictures supply initial appearance/lighting; only call Picture 2 current when boundary images actually update. Each interval is 0.708 seconds at 24 fps |
| Standard compact stream | Three sections with the actual Picture labels | Sink/current-boundary or recent-boundary images supply text context; bounded latent history supplies motion. Explicit 16-frame continuous decoding is available |

The four-step standard path and three-step Flash path count transformer forwards. Prompt text does not change the solver schedule. Do not invent an audio reference, independent negative-prompt input, last-frame-only route or per-chunk prompt API that the selected sampler does not expose.

Reference picture/video labels are numbered within their media category. `--aligned-reference` uses the zero-based position in the complete ordered reference argument list. With `[appearance.png, pose.mp4]`, use `<Picture 1>`, `<Video 1>` and aligned index `1`.

## Draft the visible action

Inspect supplied references before describing them. Keep a short working record of subject identity, current positions, held objects, contact/support, camera, light and unfinished actions. Use that record to prevent contradictions; the record itself need not appear in the final prompt.

Describe the state that exists now, the action that changes it, and the visible consequence. For a handoff, identify who initially holds the object, how the receiver contacts it, when the giver releases it and who holds it afterward. For a collision, identify the approach, contact and response. For liquids or cloth, describe the actual material motion. Keep these details proportional to the requested event; do not replace the user's scene with an easier unrelated action.

Fit the action to the actual clip duration. A 22-frame 24-fps sample is about 0.92 seconds and rarely accommodates a complete multi-event story. A 124-frame clip is about 5.17 seconds. Preserve requested complex stories by allocating sufficient duration or deliberate shots, rather than silently deleting events. Frame timestamps use frame index divided by fps; video duration uses frame count divided by fps.

For aligned controls, describe the actual sampled time window. The release resamples video to 24 fps, uses its prefix up to the requested frame count, and pads a short control with its last frame. A full-source caption may describe later actions that are absent from a shorter sample; trim or rewrite that caption to the selected interval rather than compressing all source events into it. Record source fps, duration, sampling window and any terminal padding in the experiment notes.

When an appearance image hides an object part needed later, inspect an additional real view before inventing its construction in prose. If that view is supplied to the sampler, label its appearance role explicitly, update the aligned index for the new input order, and show the added input in comparisons. A second view is a conditioning change, not a prompt-only improvement.

Give each important subject a stable visual description or reference label. After a cut, carry forward the resulting object state, clothing, location and direction of motion. Keep intended cuts explicit; never treat a streaming chunk boundary as an intended cut. Use concrete camera motion and distinguish subject rotation from a camera orbit, and a zoom from a camera translation.

For pose or game control, state what the control provides and what the style image provides. Preserve control timing, trajectory, camera and object contacts. Skeleton colors, mesh wireframes and proxy materials should only appear when requested. Do not ask for a new camera path that contradicts the supplied frame-aligned render.

For edits, identify the requested change and the source features to retain. If an edit is weak, test an explicit visible attribute change at the same seed and settings before attributing the failure to the sampler. Clear-weather relighting is a different task from strict dehazing with unchanged illumination; report that distinction. A successful hull-color edit does not establish physical video repair. Fog removal changes atmospheric visibility; it does not imply replacing buildings or changing people. A stationary camera still allows moving subjects unless the user requests a still image. Explicit visible edit descriptions are candidates for testing; stronger wording is not automatically better.

When transferring an edited still's style to a video, translate its edit instructions into resulting materials, colors and lighting. Remove single-image commands such as “maintain this pose” or “keep the exact camera view”; the control video's evolving poses and camera define those quantities at each timestamp. Inspect the edited image for moved objects before using it as an appearance reference. A successful palette change can still introduce a geometric conflict.

For image-assisted video repair, name the visible error, the corrected state, and the permitted departure from source motion. An erroneous aligned control must not simultaneously be declared fully preserved. For object duplication, specify which physical instance survives and how its support or contact changes during the action. An edited airborne pose is a state reference, not a hard temporal keyframe unless the chosen route supports that constraint. Verify the entire repaired event, including occlusions and landing; a corrected still alone demonstrates image editing.

For formal streaming, write the **next prediction interval**, not the full video:
17 predicted frames at 24 fps span 0.708 seconds; the last chunk may be trimmed. A global instruction such
as “walk across the foreground and leave” can squeeze a several-second event
into every chunk. Describe an incremental action that fits this interval: part
of a walking stride, a small vehicle displacement, or the next phase of an
ongoing interaction. Preserve the user's intended action speed; do not turn
fast combat into slow motion. Keep the scene description separate from the
short action increment. Picture 2 must actually update if the caption calls it
the current boundary; frozen image-fused text does not represent a new boundary.
For `--v2v-caption-plan`, write one caption per continuation with its concrete time interval and a coherent incremental action. At 24 fps after a 22-frame bootstrap, chunk k starts at `(22 + 17*(k-1))/24` seconds and predicts 17 frames. Vary the ongoing action without repeatedly restarting a crossing or reintroducing actors. Verify that both caption hashes and actual conditioning embeddings change. The sampler caches by caption and prefetches the next condition. Test five seconds before extending the same plan to thirty.
Review the complete generated sequence for motion continuity and natural speed.

For streaming, continue the current boundary's pose, contacts and motion direction. The sink defines enduring appearance and scene identity, not a command to return to the opening pose. Avoid restarting a completed action on each chunk. If a long story needs changing prompts but the current sampler accepts only one prompt, implement and verify the required interface before claiming per-chunk control.

## Repair language for streaming

Inspect the end of the actual input video and retain the newest boundary's pose,
contact and camera direction. In `sink-boundary` mode, Picture 1 supplies initial
appearance and Picture 2 supplies current state. In `recent-boundaries` mode, both
pictures are recent states; do not describe Picture 1 as the original clean
appearance. In `boundary` mode there is only Picture 1. Never label a current
boundary as the desired final frame of the next chunk.

Separate what advances from what is repaired. State the continuing action first,
then name a visible defect and the desired material appearance. For example,
retain the car's current track position while restoring red paint and natural
overcast exposure. Preserve real tire tread and water spray when asking to remove
repeated grid marks or bright speckles. Avoid generic “more detail” or “sharper”
instructions on an already oversharpened boundary.

A repair instruction inside the existing prompt adds no denoiser or decoder
pass; its effectiveness must be checked on complete videos. A separate image
edit is an additional inference operation. Save the failed frame, its index,
the edited image and edit timing. If an edited image later replaces a stream
boundary, use the same image for text and VAE conditioning and measure the
resulting join. A repaired still alone is not a repaired 30-second video.

## Color and lighting stability

Separate enduring material colors from changing illumination. For a fixed-light
shot, name a stable light source and preserve the subject's clothing, material
palette and exposure throughout. Do not add a sunset transition, pulsing neon,
colored spotlights or iridescent materials unless requested. For an edit, name
the one color/material to change and the source attributes to retain; do not
simultaneously request unchanged colors and a global palette transfer.

Treat this as a prompt hypothesis, not a numerical repair. For unexpected color
changes, compare the same input, seed and sampler with only this wording changed.
Inspect pre-encoding RGB as well as the MP4: normalization, precision and color
conversion faults must be fixed in code. Do not hide a failed rollout with an
unreported color correction.

## Corrections motivated by observed samples

- Reference-video edits: use the native six-section Ref2VA format and put camera, action timing and geometry in the retention analysis. State that the edit is already visible at frame zero. In the lighthouse check, this combined rewrite removed the earlier close-up and camera reset; format and wording changed together, so it is not an isolated format ablation.
- Mesh rendering: let the source video determine every evolving viewpoint; the appearance image supplies materials only. Check the same timestamps before accepting a result. Removing the appearance image alone did not improve the village camera with the first prompt. A video-only six-section edit prompt did follow the source path; an image-conditioned six-section candidate still showed ghosting. Judge the actual paired sample, not the format label alone.

- Hand contact: make support and ownership explicit. In the clay bakery, leave flour already on the board and ask both palms to press and fold one dough mound. Do not combine sprinkling flour, catching flying dough and changing utensils in five seconds. This simplifies that case's problematic interaction, without claiming that prose guarantees correct fingers.
- Multiple subjects: name every subject's initial screen position, clothing and held object in the opening state. Request a framing wide enough to show all of them from frame zero. Inspect the first second: an eventual group shot does not satisfy this opening constraint.
- Image edits: describe the changed appearance as already present from the first frame. Preserve framing, pose, object count and background except the explicit edit. Review both the exported PNG and the full clip for delayed edits or camera movement.
- Pose controls: match the source aspect ratio and selected time window. A landscape source must not be stretched into a portrait target. Finger accuracy needs a source in which the hand is actually resolved; distant body shots do not demonstrate precise hand control.
- Demo comparisons: give different tasks distinct inputs and prompts. Retain the tested prompt, seed, input hashes and sampling manifest so a displayed result can be regenerated. Disclose prompt changes when comparing against an earlier sample.

These are targeted drafting rules based on observed failures, not validated guarantees of improvement. Verify each proposed correction with the selected model.

## Emit native prompt sections

Write descriptive prose in English while preserving user-supplied dialogue, lyrics and visible text in their original language. Keep a single shot unless the requested story benefits from a cut. Avoid padding the prompt to a fixed word count.

Three sections, in order:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

For standard I2V with `--native-keyframes`, or Flash I2V with `conditioning_abi: native_fl2va`, prepend the native instruction that Picture 1 is fully referenced at 0.00 seconds. With two keyframes, identify both their opening and ending roles and describe the path between them. Do not copy that keyframe contract into `unified_ref_native` reference generation; codec reconstruction also prevents a general pixel-exact input guarantee.

Six sections, in order:

```text
subject_definitions:
<Picture 1> ...
<Video 1> ...

summary:
[reference generation] ...

retention_analysis:
<Picture 1>: fully_preserved - ...
<Video 1>: attribute_transfer - ...

detailed_description:
[Shot 1] ...

overall_soundscape:
...

non_diegetic_music:
...
```

Include only labels for actual inputs or subjects derived from them. Retention markers describe the defined reference role: `fully_preserved`, `partially_preserved`, `attribute_transfer` or `weak_reference`. Use `[video editing]` for modifying a source video and `[keyframe completion]` when requesting a concrete image anchor. Do not claim copied source audio unless the pipeline actually copies it.

Describe environmental and action sounds in `overall_soundscape`; describe audience-only music separately. Use `N/A` for an intentionally absent layer. Streaming has no generated audio in the current release, so do not promise an audible result from its sound fields.

When the user requests only a prompt, return the prompt alone. When they request experiments, keep sampling settings and evaluation notes in a separate manifest. For route-specific examples, read [references/examples.md](references/examples.md).

## Evidence status

The section formats derive from H3's prompt guides. Layout and timing rules above follow the LynnReal release implementation. The causal-writing recommendations are hypotheses until paired sampling supports them. Existing development samples expose stream-boundary identity changes, ball-contact uncertainty and weakly expressed edits; none establishes a universally best prompt style.

### Selecting an edited still

A locked-camera prompt does not guarantee that editing is complete at frame zero.
Inspect the full generated companion clip. If the change appears later, select the desired index with `script/sample.py --image-frame N` when running
`--mode image-edit`; keep the original clip and its log. State that it is a
selected still. Never describe a corrected still as a repaired full video.
