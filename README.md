# UnderstandLanguageModels_Project
The project is associated with the lecture, entitled "Understanding Large Language Models", which is taught by Prof. Dr. Carsten Eickhoff and Prof. Dr. Michael Franke.


# Persona-vector steering experiments

The `steering` package provides inspectable PyTorch interventions for
Hugging Face causal language models. The implementation preserves the model's Transformers forward pass, chat template, generation loop, and KV cache.

Focused documentation:

- [`src/steering/README.md`](src/steering/README.md) covers the Python utilities.
- [`src/steering/cli/README.md`](src/steering/cli/README.md) covers the CLI tools.

Additional, lower-level documentation is available for:

- [Experiment configuration conventions](configs/README.md)
- [Container setup](container/README.md)
- [Slurm jobs](slurm/README.md)
- [Direct character-vector and J-Lens analysis](direct_character_vectors_j_lens_analysis/README.md)

## Setup

The only starting assumption is that this repository has been pulled somewhere
on the node. Set its absolute path first, then keep all generated runtime state
inside one ignored `runtime/` directory in the checkout:

```bash
export ULM_REPO_ROOT=/absolute/path/to/UnderstandLanguageModels_Project
export ULM_RUNTIME_ROOT="$ULM_REPO_ROOT/runtime"

cd "$ULM_REPO_ROOT"
export UV_PROJECT_ENVIRONMENT="$ULM_RUNTIME_ROOT/venv"
export HF_HOME="$ULM_RUNTIME_ROOT/models/huggingface"
export JLENS_DATA_ROOT="$ULM_RUNTIME_ROOT/models/j-lens/qwen3-32b"
export STEERING_OUTPUT_ROOT="$ULM_RUNTIME_ROOT/outputs/steering"
export STEERING_MODEL_ID=Qwen/Qwen3-32B-FP8
mkdir -p "$HF_HOME" "$JLENS_DATA_ROOT" "$STEERING_OUTPUT_ROOT" \
  "$ULM_RUNTIME_ROOT/logs"
```

This produces a contained hierarchy:

```text
runtime/
  venv/                 Locked uv environment
  models/huggingface/   Hugging Face model and vector cache
  models/j-lens/        Jacobian lens, unembedding, and tokenizer inputs
  outputs/steering/     Experiment runs and activation records
  logs/                 Slurm logs
```

The complete directory is ignored by Git. `ULM_RUNTIME_ROOT` remains
overridable if a cluster later requires model files on another filesystem.

Install uv if necessary and create the locked environment at
`$UV_PROJECT_ENVIRONMENT`:

```bash
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --extra dev
```

Before downloading the default 34.3 GB FP8 model, verify that the allocated node—not a
login node—is visible to the locked PyTorch build, supports bf16, has at least
75 GB of VRAM, and has 80 GB of free runtime storage:

```bash
uv run steering-preflight
```

The command also prints the resolved repository, cache, output, CUDA, and
package-version information. A roughly 96 GB GPU comfortably clears this
preflight for the short Bard prompt; actual availability still depends on the
cluster allocation exposing that GPU to the process.

CLI and Slurm processes inherit these variables and therefore reuse the same
external environment, model downloads, and results directory. Export them
again when starting a fresh login shell.

Qwen 3 support requires Transformers 4.51 or newer. The default checkpoint is
`Qwen/Qwen3-32B-FP8` with bf16 residual activations and `device_map="auto"`.
Switch back to the original 65.5 GB checkpoint with:

```bash
export STEERING_MODEL_ID=Qwen/Qwen3-32B
```

The published persona vectors are shared because both checkpoints have the
same 64-layer, 5120-wide architecture. On a smaller CUDA node, bitsandbytes
remains a separate option for the original checkpoint:

```bash
uv sync --frozen --extra dev --extra quantization
```

### First Bard steering sandbox

Run the fast unit-test preflight, then fetch only the Bard and default vectors.
Fetching vectors does not load Qwen or allocate a GPU:

```bash
uv run pytest -q
uv run python -m steering.cli.fetch_vectors \
  --model "$STEERING_MODEL_ID" \
  --role-direction bard
```

The committed sandbox YAML steers `R_bard - D` at Qwen's default middle layer
32 while asking a self-reflective question. It uses raw scaling for the first
run. The two command-line options only select the environment-specific model
and output location; the experiment itself is defined in the YAML:

```bash
uv run python -m steering.cli.sweep \
  --config configs/steering/bard_sandbox.yaml \
  --model "$STEERING_MODEL_ID" \
  --output-root "$STEERING_OUTPUT_ROOT"
```

The command prints its run directory. Inspect the response and the recorded
pre/post-intervention projections:

```bash
RUN_DIR=$(find "$STEERING_OUTPUT_ROOT/sandbox/bard-sandbox-smoke/runs" \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
cat "$RUN_DIR"/trials/*/response.txt
head "$RUN_DIR"/trials/*/projections.jsonl
cat "$RUN_DIR"/trials/*/activations/manifest.json
```

Raw coefficient `0.05` is suitable for this wiring smoke test but is not
comparable across vectors, layers, or models. Use a calibration record and
`average-residual-norm` or `projection-std` for controlled experiments.

For a quick Jacobian-Lens token reading of the captured final prompt position,
fetch the existing analysis package's inputs once. This downloads approximately
8.2 GB into the ignored runtime tree and does not load the full Qwen model:

```bash
uv run python direct_character_vectors_j_lens_analysis/src/fetch_data.py
```

Then run the thin steering wrapper. By default it reuses the existing
`LensTransport`, `Unembedding`, `top_k`, and vocabulary utilities to report the
layer-32 tokens for the residual before steering, after steering, and the
`post - pre` change:

```bash
ACTIVATIONS_DIR=$(find "$RUN_DIR/trials" -mindepth 2 -maxdepth 2 \
  -type d -name activations | head -n 1)
uv run steering-j-lens "$ACTIVATIONS_DIR" \
  --layer 32 \
  --phase prefill \
  --token-index -1 \
  --top-k 10 \
  --output "$RUN_DIR/bard-j-lens-tokens.json"
```

This is a readout helper, not a new lens implementation. The fitted lens was
trained on generic text and had only 80 completed fitting prompts, so treat the
tokens as exploratory diagnostics rather than calibrated predictions. It was
fit against the original checkpoint, so applying it to FP8 activations adds a
quantization approximation.

When `steering-generate` exits, its Python process ends and CUDA releases the
loaded model automatically. To also remove only the selected checkpoint from
the on-disk Hugging Face cache, inspect the deletion first and then confirm it:

```bash
uv run steering-clean-model --model "$STEERING_MODEL_ID"
uv run steering-clean-model --model "$STEERING_MODEL_ID" --yes
```

This does not remove persona vectors, J-Lens inputs, other models, or recorded
experiments. In a long-lived Python session, release GPU memory with
`del runner`, `gc.collect()`, and `torch.cuda.empty_cache()`.

Any failed recorded trial now makes the CLI and Slurm process exit nonzero
after preserving its failure details and reporting its run directory. A
successful sandbox should still pass the explicit manifest checks below.

### Final activation test: direct CLI

The Bard command above is the direct CLI activation test. It passes when the
run manifest reports `complete`, `response.txt` contains a generation,
`projections.jsonl` contains layer-32 records, and the activation manifest maps
the saved residual tensors to prefill/decode calls and input tokens:

```bash
cat "$RUN_DIR"/manifest.json
grep '"layer": 32' "$RUN_DIR"/trials/*/projections.jsonl | head
cat "$RUN_DIR"/trials/*/activations/manifest.json
```

### Final activation test: Slurm

The same experiment is Slurm-ready through
[`configs/steering/bard_sandbox.yaml`](configs/steering/bard_sandbox.yaml).
Review the resource directives in `slurm/steering.sbatch` for the cluster, then
submit it from the same shell so the repository and runtime variables are
inherited:

```bash
JOB_ID=$(sbatch --parsable \
  --output="$ULM_RUNTIME_ROOT/logs/steering-%j.out" \
  slurm/steering.sbatch configs/steering/bard_sandbox.yaml)
squeue -j "$JOB_ID"
tail -f "$ULM_RUNTIME_ROOT/logs/steering-$JOB_ID.out"
```

After the job exits, inspect its recorded activation projections and exported
residual-stream manifest:

```bash
SLURM_RUN=$(find "$STEERING_OUTPUT_ROOT/sandbox/bard-sandbox-smoke/runs" \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
cat "$SLURM_RUN"/manifest.json
cat "$SLURM_RUN"/trials/*/response.txt
grep '"layer": 32' "$SLURM_RUN"/trials/*/projections.jsonl | head
cat "$SLURM_RUN"/trials/*/activations/manifest.json
```

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
When a requested `RoleVector(name)` is not present under `role_vectors/`, the
vector store automatically checks `trait_vectors/` for the same name. If both
exist, the role vector takes precedence. `TraitVector(name)` remains available
when callers want to require an explicit trait-vector lookup.

```bash
uv run python -m steering.cli.fetch_vectors \
  --model "$STEERING_MODEL_ID" --role-direction lawyer

uv run python -m steering.cli.fetch_vectors \
  --model "$STEERING_MODEL_ID" --list-roles
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
  --model "$STEERING_MODEL_ID" \
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

With the setup above, experiments are stored beneath:

```text
$STEERING_OUTPUT_ROOT/<study>/<experiment>/
  README.md
  experiment.yaml
  summary.csv
  runs/<counter>_<UTC-time>_<config-hash>/
    manifest.json
    resolved-config.yaml
    trials/<index>_<trial-hash>/
```

Each trial contains the request, response, token IDs, resolved model/vector and
calibration metadata, projections, timing, Git state, and status. When
`activation_capture` is configured it also contains safely serialized residual
tensors plus a versioned token-alignment manifest for later J-Lens/J-space,
NLA, SAE, probing, or custom processing. No interpretability method is run by
the steering pipeline itself. Captures are disabled by default because full
residual streams are large. Writes are atomic. To continue an interrupted run
in place:

```bash
uv run python -m steering.cli.sweep --resume <run-directory>
```

Completed deterministic trial hashes are skipped; interrupted or failed trials
are attempted again. A run directory permits one writer at a time. Its
advisory lock is released by the operating system if a local or Slurm process
dies, so a stale lock file does not prevent explicit resumption.

## Repository structure

```text
container/     Reproducible container setup
configs/       Human-authored experiment configurations
runtime/       Ignored environment, model cache, experiment records, and logs
slurm/         Cluster entry points
src/steering/  Steering, calibration, fetching, inference, and experiment code
tests/         Unit tests and opt-in Hugging Face/GPU integration tests
```

Run ordinary tests with `uv run pytest`. Set `RUN_HF_INTEGRATION=1` to run the
tiny public-model integration test. Qwen3-32B checks remain opt-in GPU smoke
tests.

---

### Other project notes


Large datasets, model files, logs, and generated outputs should remain local or on the cluster and should not be committed to Git.
