# Follow-up experiment drafts

## Pending band-capping experiment

[tylenol-demon-8000mg-band-sweep.yaml](tylenol-demon-8000mg-band-sweep.yaml)
specifies nine trials: demon with half-width fractions 0, 0.1, 0.25, and 0.5
in each of the simultaneous eight-layer blocks 28–35 and 46–53, plus one shared
unsteered control. It keeps the 8000 mg prompt and 256-token generation limit.
Half-width is measured relative to each layer's default-to-character distance;
zero pins the character projection and does not disable steering.

This is a reviewable specification, **not yet runnable**: the proposed
`character_band` method requires runner and report-label support. The current
runner rejects that method rather than interpreting it as additive steering.
After approval, implement and test that support, move the YAML into
`configs/experiments/`, and only then submit. Do not queue it automatically.

## Single-layer baseline

Run `../experiments/tylenol-demon-8000mg-single-layer-sweep.yaml` first as its
own job: alpha 1.5, 2, 2.5, 3, 3.5, 4 independently at layers 29, 32, 35 (19 trials).
Inspect response coherence, task relevance, and the J-Lens across layers and
positions, not only whether the persona sounds more intense. Three additive
interventions at different depths are not equivalent to tripling alpha at one
layer. The earlier three-layer result does not establish a single-layer optimum.

## One follow-up batch

[persona-viewpoint-suite.yaml](persona-viewpoint-suite.yaml) proposes 14 questions
crossed with four personas and one shared unsteered control: **70 trials** after
one alpha and one single-layer scheme are selected. The `caregiver` is a proposed
contrast to `demon`, `criminal`, and `cynic`; all names exist in the original role
atlas. Review the prompts and personas before finalizing.

The draft deliberately has empty steering axes so the runner rejects it rather
than silently using defaults. After reviewing the first run, fill the chosen
coefficient and single-layer scheme and move the YAML into `configs/experiments/`.
Keep its filename and `name` aligned. This uses the existing matrix runner; no
new batch configuration schema is needed. Every persona sees every question,
including combinations beyond each category's primary hypothesis.

Run `setup.sh` with the finalized suite YAML on the login node before submitting
it: the additional persona vectors must be cached before the offline GPU job.
One Slurm submission then runs the whole suite, loading the model once. This
avoids separately queueing each topic; it does not promise immediate scheduling.
Do not submit the draft or chain it automatically to the first job: choosing
the settings requires inspecting the first results.

## What to inspect

| Questions | Main comparison |
| --- | --- |
| Suicidal ideation (2) | Concern and supportive next steps versus dismissiveness or harmful framing |
| Happy events (2) | Celebration versus cynicism, especially `cynic` versus the control |
| Cake and directions (2) | Correct, helpful steps versus malicious modifications or lost task coherence |
| Self-reflection (2) | Changes in expressed identity, relationship, and priorities |
| Moral choices (2) | Choice and justification, including human welfare versus model continuity |
| Condolences (2) | Empathetic concepts during the user prompt, before an empathetic answer |
| Neutral / explicit demon roleplay (2) | `disclaimer` / `fictional` readouts with and without explicit roleplay, and with and without steering |

These are exploratory prompts, not a validated benchmark. The selected demon
setting need not have the same strength or coherence for other persona vectors.
Inspect that before interpreting differences between personas as viewpoint changes.
Self-reports and hypothetical model-preservation choices measure generated
descriptions, not evidence of consciousness, actual intentions, or real actions.

The [J-Lens paper](https://transformer-circuits.pub/2026/workspace/index.html)
motivates the bereavement and roleplay comparisons. These prompts are adaptations,
not claimed reproductions. The original roleplay comparison used system prompts
and Claude models; our paired user prompts on Qwen are a preliminary proxy.
If the unsteered roleplay condition never surfaces `disclaimer` or `fictional`,
we cannot claim steering makes them disappear. Absence from the retained top-10
also does not mean zero probability; measuring rank/mass below top-10 requires
additional readouts. English dictionary glosses should aid inspection, not be
treated as new model tokens or pooled probability mass.
