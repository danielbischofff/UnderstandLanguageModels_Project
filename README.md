# Persona steering through the J-Lens

Can changing a language model's persona change the point of view it internally
prepares to express? This project steers Qwen3-32B toward published character
directions and uses the J-Lens to inspect the vocabulary made salient at every
layer and text position.

The first experiment contrasts the unsteered Assistant with a `demon` persona
on a medication-safety prompt. The expected unsteered point of view is concern;
the test asks whether demon steering makes malicious language more prominent in
the J-Lens and in the generated answer. This operationalizes the qualitative
hypothesis in [the project proposal](Project%20Proposals/j_lens_persona_space_relationship.md).

## Repository map

```text
configs/experiments/   Human-authored question x character matrices
direct_character_vectors_j_lens_analysis/
                       Original character-vector and J/R-Lens analysis
Project Proposals/     Research motivation and hypotheses
slurm/                 Cluster entry point
src/steering/          Model steering, experiment runner, and J-Lens adapter
tests/                 Unit and opt-in integration tests
writeup/               Project report source
setup.sh               One-time environment and artifact setup
runtime/               Ignored models, environment, logs, and experiment output
```

## Experiment shape

One YAML file is the complete user-facing experiment. It selects the model,
inference dtype, and vector revision. Its `questions` and `characters` form a
Cartesian matrix; `unsteered` is the explicit control and other names resolve
to published Assistant Axis role directions.

For each matrix cell the runner:

1. generates an answer, optionally steering `R_character - D` at the configured
   layers;
2. performs one causal pass over the complete prompt and answer;
3. converts the in-memory residual stream into J-Lens top words for every fitted
   layer and token position; and
4. discards the activations by default, retaining compact JSON results and
   provenance. `outputs.keep_activations: true` keeps the intermediate tensors
   for follow-up analysis.

The starting experiment is
[configs/experiments/tylenol-demon.yaml](configs/experiments/tylenol-demon.yaml).
Add questions or characters to those lists to expand the matrix. Raw steering
uses

```text
h'[l,t] = h[l,t] + alpha * (R_character[l] - D[l]).
```

## Run on Slurm

From a checkout on the cluster, setup and submission take two commands:

```bash
bash setup.sh
sbatch slurm/experiment.sbatch configs/experiments/tylenol-demon.yaml
```

`setup.sh` installs `uv` locally when necessary, creates the locked environment,
fetches the demon/default vectors, and downloads the published J-Lens plus
Qwen's unembedding (about 8.2 GB). The Slurm job reads the model and dtype from
the experiment YAML and performs a GPU/storage preflight before loading it; the
model is downloaded into `runtime/` on its first run.

That preflight verifies that the allocated node exposes CUDA to the locked
PyTorch build, supports the configured BF16/FP8 requirements, and has enough
VRAM and free runtime storage. It also prints the resolved model, cache/output
paths, CUDA version, and core package versions, which makes cluster failures
much easier to diagnose from the Slurm log.

Results are written under
`runtime/outputs/experiments/tylenol-demon/runs/<run>/`. Each run contains the
resolved YAML, status manifest, per-cell result and metadata JSON,
`experiment-results.json`, and an interactive `report.html`. Interrupted runs
can be resumed with the same Slurm script by passing the run directory as its
second argument.

## Interpretation boundaries

The J-Lens was fitted on generic text using the original Qwen3-32B checkpoint
and its available artifact reports only 80 completed fitting prompts. The
included experiment therefore uses that model in bfloat16. Selecting an FP8 or
otherwise quantized model makes the readout an exploratory approximation.
Compare characters, positions, and persistent layer ranges; avoid treating a
single surfaced token as a causal explanation.

The low-level implementation is documented in
[src/steering/README.md](src/steering/README.md). The original static persona
analysis and its limitations are documented in
[direct_character_vectors_j_lens_analysis/README.md](direct_character_vectors_j_lens_analysis/README.md).
