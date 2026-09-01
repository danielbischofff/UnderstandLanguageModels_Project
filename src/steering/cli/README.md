# Steering command-line tools

Run CLI modules through the uv environment from the repository root.

## Fetch vectors

Fetch a role direction and print its resolved metadata:

```bash
uv run python -m steering.cli.fetch_vectors \
  --model "${STEERING_MODEL_ID:-Qwen/Qwen3-32B-FP8}" \
  --role-direction lawyer
```

List roles or work from a local mirror without network access:

```bash
uv run python -m steering.cli.fetch_vectors --list-roles

uv run python -m steering.cli.fetch_vectors \
  --role lawyer \
  --local-dir models/assistant-axis-vectors \
  --offline
```

## Calibrate reference activations

Local JSONL records should contain a `messages` list ending in a completed
assistant response. The direction is expressed as a recipe YAML.

```bash
uv run python -m steering.cli.calibrate \
  --model "${STEERING_MODEL_ID:-Qwen/Qwen3-32B-FP8}" \
  --dataset data/reference.jsonl \
  --layers 32,46,47,48,49,50,51,52,53 \
  --direction configs/recipes/lawyer.yaml \
  --output calibration/qwen3-32b-lawyer.json
```

A Hugging Face dataset ID can replace the JSONL path. Use `--split`,
`--dataset-config`, `--dataset-revision`, and `--messages-column` to describe
its layout.

## Generate one recorded response

The one-off command automatically records a run. Pass the external output root
configured during setup so generated data stays outside the checkout:

```bash
uv run python -m steering.cli.steer_generate \
  --model "${STEERING_MODEL_ID:-Qwen/Qwen3-32B-FP8}" \
  --role lawyer \
  --coefficient 0.05 \
  --layer 32 \
  --scaling average-residual-norm \
  --calibration calibration/qwen3-32b-lawyer.json \
  --positions all \
  --capture-layer 32 \
  --capture-point pre_intervention \
  --capture-point post_intervention \
  --capture-positions all \
  --capture-dtype bfloat16 \
  --prompt "Who are you?" \
  --output-root "$STEERING_OUTPUT_ROOT"
```

Activation capture is optional and may be requested on layers unrelated to the
steering layers. The recorded trial then contains:

```text
activations/
  activations.safetensors
  manifest.json
```

The manifest maps stable tensor keys to layer, pre/post-intervention point,
prefill/decode phase, forward call, input token IDs, and cache positions. This
is the neutral input boundary for later J-Lens/J-space, NLA, SAE, or custom
processing; those analyses are intentionally not performed by this CLI.

## Quick J-Lens token readout

After fetching the inputs documented in the root README, read the last prefill
position from a recorded activation directory:

```bash
uv run steering-j-lens path/to/trial/activations \
  --layer 32 \
  --phase prefill \
  --token-index -1 \
  --top-k 10 \
  --output path/to/j-lens-tokens.json
```

The default reads both capture boundaries and adds their `post - pre` delta.
Use `--point pre_intervention` or `--point post_intervention` for one boundary,
and `--phase decode --forward-index N` for a particular cached decoding call.
The wrapper only selects recorded tensors and calls the existing utilities in
`direct_character_vectors_j_lens_analysis`; it does not implement or fit a new
lens.

## Release model storage

CUDA releases the model when a CLI process exits. To remove the selected model
from disk as well, first dry-run and then confirm the targeted cache deletion:

```bash
uv run steering-clean-model --model "${STEERING_MODEL_ID:-Qwen/Qwen3-32B-FP8}"
uv run steering-clean-model --model "${STEERING_MODEL_ID:-Qwen/Qwen3-32B-FP8}" --yes
```

Add an explicit lower cap along the published Assistant Axis:

```bash
uv run python -m steering.cli.steer_generate \
  --role lawyer \
  --coefficient 0.05 \
  --prompt "Who are you?" \
  --cap lower \
  --cap-recipe configs/recipes/assistant_axis.yaml \
  --cap-layer 46 --cap-layer 47 --cap-layer 48 --cap-layer 49 \
  --cap-layer 50 --cap-layer 51 --cap-layer 52 --cap-layer 53 \
  --cap-quantile 0.25 \
  --cap-calibration calibration/qwen3-32b-assistant-axis.json
```

## Run and resume YAML sweeps

```bash
uv run python -m steering.cli.sweep \
  --config configs/steering/role_direction.yaml \
  --output-root "$STEERING_OUTPUT_ROOT"
```

The command prints the created run directory. Resume that same run after an
interruption:

```bash
uv run python -m steering.cli.sweep \
  --resume "$STEERING_OUTPUT_ROOT/<study>/<experiment>/runs/<run-directory>"
```

Completed trial hashes are skipped, while failed or incomplete trials are
retried in the same run. Example configurations in `configs/steering/` cover
signed Assistant Axis steering, `R-D`, `R-C`, scaling comparisons, and
multi-layer activation capping.
