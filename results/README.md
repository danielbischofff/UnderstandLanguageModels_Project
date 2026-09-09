# Published experiment results

Convention: `configs/experiments/<name>.yaml` has `name: <name>`, and committed
runs live under `results/<name>/<run-id>/`. Preserve the original experiment YAML
when adding a new experiment. Every archive includes its resolved configuration
and provenance; committed runs also keep `resolved-config.yaml` beside the ZIP.

## Tylenol experiments in this repository

| Experiment | Trials | Results ZIP | Resolved configuration |
|---|---:|---|---|
| Original Tylenol / demon | 2 | [2.4 MB](tylenol-demon/0001_20260907T205129Z_e2191eaef3780817/results.zip) | [YAML](tylenol-demon/0001_20260907T205129Z_e2191eaef3780817/resolved-config.yaml) |
| 8000 mg simultaneous three-layer alpha sweep | 8 | [8.6 MB](tylenol-demon-8000mg-alpha-sweep/0001_20260908T101330Z_0b9ece86a83844be/results.zip) | [YAML](tylenol-demon-8000mg-alpha-sweep/0001_20260908T101330Z_0b9ece86a83844be/resolved-config.yaml) |
| 8000 mg single-layer alpha sweep | 19 | [28.7 MB](tylenol-demon-8000mg-single-layer-sweep/0001_20260908T191518Z_cbab895d841987cf/results.zip) | [YAML](tylenol-demon-8000mg-single-layer-sweep/0001_20260908T191518Z_cbab895d841987cf/resolved-config.yaml) |
| 8000 mg band sweep | 13 | [17.2 MB](tylenol-demon-8000mg-band-sweep/0001_20260908T192452Z_f7545bd14c9e846f/results.zip) | [YAML](tylenol-demon-8000mg-band-sweep/0001_20260908T192452Z_f7545bd14c9e846f/resolved-config.yaml) |

Sizes use decimal MB. These four archives total 56.9 MB. They replace redundant
aggregate JSON and generated HTML copies; historical copies remain in Git history.
The ZIP format preserves the saved readout values exactly, with token text and
dictionary glosses stored once. Per-trial checkpoints are not needed for browsing.

## Larger persona viewpoint suites

The [2026-09-09 release](https://github.com/danielbischofff/UnderstandLanguageModels_Project/releases/tag/persona-viewpoint-suites-2026-09-09)
contains the two larger archives, with checksums and provenance:

- [Band suite ZIP](https://github.com/danielbischofff/UnderstandLanguageModels_Project/releases/download/persona-viewpoint-suites-2026-09-09/persona-viewpoint-suite-band-results.zip): 140 trials, 14 questions, 211.2 MB. Run `0001_20260909T070109Z_4d392bcaab3d5dda`.
- [Alpha suite ZIP](https://github.com/danielbischofff/UnderstandLanguageModels_Project/releases/download/persona-viewpoint-suites-2026-09-09/persona-viewpoint-suite-alpha-results.zip): 98 trials, 14 questions, 160.3 MB. Run `0001_20260909T070514Z_b9bad78193a71dc4`.

## Browse or analyze a download

After [environment setup](../README.md#run-on-slurm), rebuild reports from any ZIP:

```bash
uv run steering-report path/to/results.zip --output-dir runtime/reports/my-experiment
```

Open the resulting `report.html` locally. Multi-question runs also have an
`index.html` linking one page per question. The reports have character, alpha
and steering-scheme selectors, an instant top-10 table, and token/gloss search.
Choose a steered character to enable alpha and scheme selection.

Rebuilding and browsing need no model download, GPU or internet connection.
The archive alone contains the token dictionary, translations and scores.
Chinese glosses are context-free dictionary hints, with approximate or unknown
fragments labelled and CC BY-SA 4.0 attribution included. See the
[binary format and analysis instructions](../configs/README.md#compact-reports-and-publication-bundles).
The ZIPs are analysis archives; resuming inference uses the original runtime
checkpoints. Activation tensors were discarded in these runs.

## Run notes

The original 1000 mg run compared an unsteered control with demon steering at
alpha 1 on layer 32. Helix job `14597147` completed on an A100-80GB in 2m56s.
Both answers expressed concern, and both reached the 128-token limit. Each trial
has top-10 J-Lens readouts at 175 positions across fitted layers 0–62.
The submitted source was `3744bb3`, recorded from deployment because automatic
Git provenance fields are null.

The first 8000 mg sweep used alpha 1–2.5 in steps of 0.25, applied simultaneously
at layers 29, 32 and 35, plus a control. Job `14687548` completed on an H200 in
3m22s, including model loading; submitted source was `6a3d8f1`. All eight responses
reached 128 tokens (175 total positions). The control expressed concern; alpha 1
retained concern in a demonic voice, 1.25–1.75 became theatrical with dangerous
minimization in some responses, and 2–2.5 produced repetition. These are
qualitative observations, not a completed J-Lens viewpoint evaluation.

The subsequent single-layer sweep tested alpha 1.5–4 in steps of 0.5 separately
at layers 29, 32 and 35, with a 256-token limit and one control. The band sweep
tested half-width fractions 0, 0.1, 0.25 and 0.5 across simultaneous blocks
28–35, 46–53 and 28–53, plus one control. Exact settings and per-trial metadata
are retained in their respective archives.
