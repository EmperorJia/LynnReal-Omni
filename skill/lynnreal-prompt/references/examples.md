# Adapted examples

These examples demonstrate the interface, not measured quality improvements. Adjust them to the actual inspected inputs and requested duration.

## Short T2V: show one phase of an action

For a 22-frame clip already in the middle of a pour:

```text
integrated_multimodal_description: [Shot 1] A fixed close-up shows amber tea already flowing from a tilted white porcelain teapot into a matching cup. Over this brief moment, the connected stream strikes the liquid surface, making small concentric ripples as the level rises slightly. The teapot remains tilted above the cup on the sunlit wooden table.

overall_soundscape: A continuous soft trickle of liquid and quiet room tone.

non_diegetic_music: N/A
```

A longer requested story can include lifting, tilting, pouring, stopping and setting down. Do not force all those stages into the short latency-test clip.

## Body control: appearance and contact are different constraints

For `[first_rgb.png, body_pose.mp4]`, the manifest aligns reference index 1. A body skeleton may constrain wrists but omit the ball's geometry and contact. Keep the uncertainty visible in evaluation.

```text
subject_definitions:
<Picture 1> defines the beach, the man's appearance and the single volleyball.
<Video 1> supplies the frame-aligned body motion.

summary:
[reference generation] The man from <Picture 1> turns on the beach following <Video 1>.

retention_analysis:
<Picture 1>: fully_preserved - preserve identity, clothing, volleyball appearance and beach layout.
<Video 1>: attribute_transfer - transfer the body trajectory and timing to the rendered man.

detailed_description:
[Shot 1] Start with the man and ball in the visible arrangement of <Picture 1>. Follow the torso, arm and wrist motion in <Video 1>. Where the hands pass the ball behind his hips, the receiving hand contacts it before the other releases it; the ball remains supported during the exchange. Preserve the camera and horizon established by the inputs. Render the man and scene in the photograph's style.

overall_soundscape:
Gentle surf and a coastal breeze.

non_diegetic_music:
N/A
```

Use the handoff sentence only if the inspected reference shows a handoff. Do not introduce it merely because the example contains one.

## Streaming: current boundary has temporal authority

For the current two-image conditioning implementation:

```text
integrated_multimodal_description: [Shot 1] Continue directly from the latest street view in <Picture 2>. The pedestrians keep their current walking directions and carry their umbrellas through the same wet intersection. The woman crossing the foreground continues out of view along her existing path; people behind her become visible through that opening. Preserve the street, building arrangement and persistent appearance established by <Picture 1>, while advancing the action from <Picture 2>.

overall_soundscape: N/A

non_diegetic_music: N/A
```

The walking direction and foreground woman must actually be visible in the current boundary. A fixed prompt reused across a long stream must not repeatedly introduce a person who has already left. The native continuation route omits audio regardless of these fields.

## Appearance repair within the existing compact-stream prompt

This is a proposed prompt for the inspected wet-track racing-car input, using
`sink-boundary` context. It has not been established as a general restoration rule.

```text
integrated_multimodal_description: Continue the red race car's motion from its current position and rear view in <Picture 2>. Its wheels remain on the wet racing line as the camera follows smoothly behind. Use <Picture 1> to retain the original red body panels, white sponsor patches, black tires and soft overcast daylight. Restore those material colors where the current image has washed-out highlights; keep the current vehicle geometry, road layout and camera direction. Preserve the real water spray and tire tread while rendering the road and barriers with natural texture, without repeated grid marks or isolated bright speckles. Advance the motion continuously in the same shot.

overall_soundscape: N/A

non_diegetic_music: N/A
```

Use a separate six-section image-edit prompt when actually editing a saved failed
frame. Identify the appearance reference and damaged target separately and record
the additional inference; do not label that result as untouched stream output.
