# Steering command-line tools

Run CLI modules through the uv environment from the repository root.

## Fetch vectors

Fetch a role direction and print its resolved metadata:

```bash
uv run python -m steering.cli.fetch_vectors \
  --model Qwen/Qwen3-32B \
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
  --model Qwen/Qwen3-32B \
  --dataset data/reference.jsonl \
  --layers 32,46,47,48,49,50,51,52,53 \
  --direction configs/recipes/lawyer.yaml \
  --output calibration/qwen3-32b-lawyer.json
```

A Hugging Face dataset ID can replace the JSONL path. Use `--split`,
`--dataset-config`, `--dataset-revision`, and `--messages-column` to describe
its layout.

## Generate one recorded response

The one-off command automatically writes a run beneath `outputs/steering/`:

```bash
uv run python -m steering.cli.steer_generate \
  --model Qwen/Qwen3-32B \
  --role lawyer \
  --coefficient 0.05 \
  --layer 32 \
  --scaling average-residual-norm \
  --calibration calibration/qwen3-32b-lawyer.json \
  --positions all \
  --prompt "Who are you?"
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
  --config configs/steering/role_direction.yaml
```

The command prints the created run directory. Resume that same run after an
interruption:

```bash
uv run python -m steering.cli.sweep \
  --resume outputs/steering/<study>/<experiment>/runs/<run-directory>
```

Completed trial hashes are skipped, while failed or incomplete trials are
retried in the same run. Example configurations in `configs/steering/` cover
signed Assistant Axis steering, `R-D`, `R-C`, scaling comparisons, and
multi-layer activation capping.

