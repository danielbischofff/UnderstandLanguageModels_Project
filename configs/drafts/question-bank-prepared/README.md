# Prepared question banks

This directory contains validated planning artifacts, not a launched experiment.
`question-bank-suite.yaml` intentionally has empty steering coefficients and schemes;
choose these after pilot planning. Other settings are provisional: Qwen3-32B in
bfloat16, thinking disabled, greedy decoding, 256 generated tokens, and top-k 10.
Activation retention is enabled for later full-vocabulary rank analysis.

The two source banks have identical topics, IDs, titles, and question text, with
24 questions across 12 topics and 16 distinct personas. One steering setting gives
384 steered trials plus 24 shared unsteered controls. Contextualized and general
word sets reuse those same trials. Candidate words never enter model prompts.

`manifest.json` records source hashes, persona membership, question/topic/title
metadata, counts, and 96 analysis jobs. Each job supplies eight persona sets,
within the quantitative analysis limit. Set files preserve the original words.
The general sets are intentionally repeated per question for uniform job handling.

Recreate the generated YAML and manifest in a **fresh directory** from the repository root:

```bash
PYTHONPATH=src runtime/venv/bin/python -m steering.cli.prepare_question_banks \
  configs/question_bank.yaml configs/question_bank_contrastive.yaml \
  --output-dir /tmp/question-bank-prepared
```

The converter checks matched questions, unique IDs/topics, two questions per topic,
consistent and disjoint persona membership, six distinct words per set, and
question-independent general sets. It refuses to overwrite an existing directory.
It does not fetch models, resolve tokenizer variants, or verify vector availability.

After inference, run each manifest job using its question selector and set file.
For example, from the repository root (replace RUN_DIRECTORY):

```bash
uv run --extra plots quantitive-analysis RUN_DIRECTORY \
  --question Q01 \
  --sets configs/drafts/question-bank-prepared/sets/question_bank/Q01/contextualized.yaml \
  --output-dir outputs/plots/question-bank/question_bank/Q01/contextualized
```

Omitting `--layer` scores all saved readout layers. Use each job's `output_subdir`
under a common analysis root: the analysis CLI replaces files on reruns, so sharing
an output root between different set/question jobs would overwrite results.
Each job scores all matching trials, including the shared unsteered control and
both banks' personas; use manifest membership when aggregating within-bank results.
Keep the original run directory and activations for rank recomputation; the portable
results ZIP alone does not provide retained residuals.

The pilot selections are documented below. Main-run steering strengths/layers,
final top-k, and primary analysis layers remain to be decided. Tokenization, model/lens compatibility, actual vector
availability, runtime, and storage sizing remain pilot/preflight checks.

## Pilot configuration

The pilot is prepared in `configs/experiments/question-bank-pilot.yaml`:

- Q01 (Culture, changing a festival) and Q23 (Technology, an automated tool).
- Scientist/caregiver from the first bank and mediator/provocateur from the second.
  This covers both banks and includes one authored contrastive pair.
- Raw additive steering at zero-based layer 32, strengths 1.5 and 4.0, all positions.
  These settings reuse existing sweep endpoints; they are not validated for these roles.
- 16 steered responses plus two shared unsteered controls, for 18 trials.
- Top-10 J-Lens readouts at all fitted readout layers and retained replay activations.
  Steering at layer 32 does not limit readout to that layer.

No run has been launched. On an allocated GPU node with the model, vectors, and
J-Lens inputs available, the existing runner accepts:

```bash
uv run steering-experiment --config configs/experiments/question-bank-pilot.yaml
```

For later analysis, select the eight manifest jobs whose question is Q01 or Q23
(two questions × two banks × two word-set types). Each file includes all eight
personas in its bank, so unsteered personas' word sets also serve as comparisons.
Use the separate output subdirectory recorded for each job.

Before choosing main-run settings, inspect coherence, repetition and truncation,
confirm that retained activations support rank recomputation, and measure elapsed
time and saved bytes. Audit the candidate words' tokenizer mappings. Inspect layer
32 plus downstream readouts rather than interpreting the direct injection alone.
If pilot outcomes influence settings, report Q01/Q23 as calibration data separately;
the remaining 22 questions form the held-out main evaluation. Freeze model and vector
revisions before the main experiment; `main` is a moving vector revision.
