# UnderstandLanguageModels_Project
The project is associated with the lecture, entitled "Understanding Large Language Models", which is taught by Prof. Dr. Carsten Eickhoff and Prof. Dr. Michael Franke.


# Persona-vector steering experiments

`steering` package provides inspectable PyTorch interventions for
Hugging Face causal language models. The implementation preserves the model's Transformers forward pass, chat template, generation loop, and KV cache.

## Setup

Install the locked development environment:

```bash
uv sync --extra dev
```

For optional bitsandbytes 4/8-bit loading:

```bash
uv sync --extra dev --extra quantization
```

Qwen 3 support requires Transformers 4.51 or newer. Full Qwen/Qwen3-32B runs
default to bfloat16 and `device_map="auto"`; quantization is an explicit local
development option rather than the research default.

## Vector geometry

At layer `l`, `D` is the neutral/default-Assistant centroid, `R_r` is the
centroid for a fully elicited role, and `C` is the mean across elicited roles.
The public named directions are:

- `RoleDirection(r) = R_r - D`, positive toward the target role.
- `RoleSpecificDirection(r) = R_r - C`, the role-specific contrast.
- `PublishedAssistantAxis() = D - C`, positive toward the default Assistant.

`role_vectors/assistant.pt` is an elicited character role and is never used as
`D`; the default is always `default_vector.pt`. 

Arbitrary contrasts remain possible with `Difference`, `Mean`,
and `Sum`.

Vectors are downloaded selectively from
`lu-christina/assistant-axis-vectors`, loaded on CPU with safe tensor loading,
and validated against `num_hidden_layers` and `hidden_size` before inference.

```bash
uv run python -m steering.cli.fetch_vectors \
  --model Qwen/Qwen3-32B --role-direction lawyer

uv run python -m steering.cli.fetch_vectors \
  --model Qwen/Qwen3-32B --list-roles
```

Use `--offline` to prohibit network access and `--local-dir` for a local mirror.
Offline errors list missing artifacts rather than falling back silently.

## Python inference API

The common case defaults to the zero-based middle transformer block—layer 32
for Qwen3-32B—and modifies the transformer-block output after its attention and
MLP residual updates:

```python
from steering import InferenceRunner, RoleDirection, SteeringRequest

runner = InferenceRunner.from_pretrained()
request = SteeringRequest(
    vector=RoleDirection("lawyer"),
    coefficient=0.05,
    layers=None,
    scaling="average_residual_norm",
    calibration="calibration/qwen3-32b-lawyer.json",
    positions="all",
)
response = runner.generate(
    [{"role": "user", "content": "Who are you?"}],
    steering=request,
    seed=42,
    max_new_tokens=128,
)
print(response.text)
```

For multiple additions and caps, use a plan. Cap layers never receive an
implicit default:

```python
from steering import (
    CappingRequest,
    PublishedAssistantAxis,
    RoleDirection,
    SteeringPlan,
    SteeringRequest,
)

plan = SteeringPlan(
    additive=[
        SteeringRequest(RoleDirection("lawyer"), coefficient=0.05, layers=[32])
    ],
    caps=[
        CappingRequest(
            PublishedAssistantAxis(),
            bound="lower",
            quantile=0.25,
            layers=list(range(46, 54)),
            calibration="calibration/qwen3-32b-assistant-axis.json",
        )
    ],
    ordering="add_then_cap",
)
```

Scaling modes are:

- `raw`: `delta = coefficient * vector`.
- `unit`: coefficient is an absolute residual-space displacement.
- `average_residual_norm`: coefficient is a fraction of the calibrated mean
  residual norm.
- `projection_std`: coefficient is measured in calibrated standard deviations
  along the chosen unit direction.

Whole-activation norm preservation is disabled by default and recorded as
non-paper behavior when requested. A zero additive coefficient registers no
additive hook.

Position policies are `all`, `last`, and `generated_only`. `all` modifies every
prompt position in prefill and each cached decoding position. `generated_only`
uses cache state/cache positions to skip prefill; consequently the first token
is sampled from unmodified prefill logits and intervention starts during cached
decode. Python forward hooks may cause graph breaks, so `torch.compile` is not
supported or claimed by this implementation.

## Calibration

Calibration defaults to non-padding assistant-response tokens and streams
float64 Welford statistics rather than retaining activations. Quantiles use a
seeded bounded reservoir and metadata reports whether they are exact or
sampled.

```bash
uv run python -m steering.cli.calibrate \
  --model Qwen/Qwen3-32B \
  --dataset data/reference.jsonl \
  --layers 32,46,47,48,49,50,51,52,53 \
  --direction configs/recipes/lawyer.yaml \
  --output calibration/qwen3-32b-lawyer.json
```

Local JSONL records use an OpenAI-style `messages` list. Hugging Face dataset
IDs are also accepted with configurable split, configuration, revision, and
message column. Any corpus other than the paper's named LMSYS reference is
marked as a reproduction approximation.

## Recorded experiments and recovery

Run the committed YAML examples with:

```bash
uv run python -m steering.cli.sweep \
  --config configs/steering/role_direction.yaml
```

Experiments are stored beneath:

```text
outputs/steering/<study>/<experiment>/
  README.md
  experiment.yaml
  summary.csv
  runs/<counter>_<UTC-time>_<config-hash>/
    manifest.json
    resolved-config.yaml
    trials/<index>_<trial-hash>/
```

Each trial contains the request, response, token IDs, resolved model/vector and
calibration metadata, projections, timing, Git state, and status. Writes are
atomic. To continue an interrupted run in place:

```bash
uv run python -m steering.cli.sweep --resume <run-directory>
```

Completed deterministic trial hashes are skipped; interrupted or failed trials
are attempted again. A run directory permits one writer at a time.

## Repository structure

```text
container/     Reproducible container setup
configs/       Human-authored experiment configurations
calibration/   Local calibration records (not committed)
outputs/       Generated experiment records (not committed)
models/        Local model/vector caches (not committed)
slurm/         Cluster entry points
container/     Reproducible container setup
src/steering/  Steering, calibration, fetching, inference, and experiment code
tests/         Unit tests and opt-in Hugging Face/GPU integration tests
```

Run ordinary tests with `uv run pytest`. Set `RUN_HF_INTEGRATION=1` to run the
tiny public-model integration test. Qwen3-32B checks remain opt-in GPU smoke
tests.


Large datasets, model files, logs, and generated outputs should remain local or on the cluster and should not be committed to Git.
