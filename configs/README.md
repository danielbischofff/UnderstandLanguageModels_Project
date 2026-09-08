# Experiments

Every YAML file in `configs/experiments/` defines one complete experiment:
model and dtype, questions × characters × coefficients × schemes, generation
settings, seed, and the number of J-Lens words retained.

## Naming convention

Use the same identifier for the YAML filename (without `.yaml`), its `name`
field, and the directory under `results/`. For example:

| Configuration | Experiment name / published results directory |
| --- | --- |
| `tylenol-demon.yaml` | `tylenol-demon` |
| `tylenol-demon-8000mg-alpha-sweep.yaml` | `tylenol-demon-8000mg-alpha-sweep` |

Keep the original YAML when defining a new experiment; copy it to a new name.
Publish each run under `results/<experiment-name>/<run-id>/`, retaining its
`resolved-config.yaml`. Runtime output uses the same experiment name under
`runtime/outputs/experiments/<experiment-name>/runs/<run-id>/`.

Use the reserved character `unsteered` as the control. All other character names
must exist in the published Assistant Axis role-vector dataset.

`model.id` and `model.dtype` are required. Optional model fields are `revision`,
`tokenizer_revision`, `quantization` (`4bit` or `8bit`), and
`enable_thinking`. `vectors.revision` selects the published vector revision.

J-Lens residuals are discarded after readout by default. Set
`outputs.keep_activations: true` to retain one safetensors file and alignment
manifest per completed matrix cell for later low-level inspection.

## Steering sweeps

In a separately named experiment YAML, use a `steering` section such as:

```yaml
steering:
  coefficients: [-1.0, 0.0, 1.0, 2.0]
  schemes:
    - name: middle
      layers: [32]
    - name: late-generated
      layers: [40, 41, 42]
      positions: generated_only
    - name: character-floor
      method: projection_cap
      bound: lower
      layers: [32]
```

Every question/steered-character pair runs every coefficient/scheme combination.
One question, one steered character, four coefficients and three schemes produce
12 steered trials plus one control if `unsteered` is included. The browser exposes
all four axes; alpha and scheme selectors are disabled for the shared control.
Failed cells remain selectable and display a missing-result message.

Scheme names must be unique. Layers are explicit zero-based lists (not ranges).
`method` defaults to `additive`; `positions` defaults to `all`. Other position
policies are `last` (last position in each forward call) and `generated_only`
(leaves prefill and first-token logits unchanged, then steers cached decoding).
The same scheme is used during generation and the post-intervention J-Lens pass.
Position-restricted schemes require cached generation (`use_cache: true`, the
model default); explicitly disabling it is rejected for these schemes.

For additive schemes, `scaling: raw` (default) applies `h' = h + alpha * v`,
where `v = R_character - D`. `scaling: unit` uses `v / ||v||` instead.
`preserve_original_activation_norm: true` optionally rescales the result to
the original activation norm; its default is false.

For `projection_cap`, alpha is a threshold in activation units, **not** an
additive multiplier. Let `u = v / ||v||` and `p = dot(h, u)`. A `lower` cap uses
`h' = h + max(0, alpha - p) * u`; an `upper` cap uses
`h' = h + min(0, alpha - p) * u`. Orthogonal components remain unchanged.
Alpha zero therefore need not be an unsteered control. `scaling` and norm
preservation do not apply to this method. The report labels these semantics.
This is the existing projection cap; it does not constrain full Euclidean
distance to the character centroid.

The original `coefficient: 1.0` / `layers: [32]` syntax still works, and existing
runs remain resumable. Do not combine singular `coefficient` with `coefficients`,
or top-level `layers` with `schemes`. Coefficients must be finite and unique.
Changing sweep axes requires a new run. The Slurm command is unchanged.

## Next experiments

[tylenol-demon-8000mg-single-layer-sweep.yaml](experiments/tylenol-demon-8000mg-single-layer-sweep.yaml)
tests alpha 2.5, 3, 3.5, and 4 separately at layers 29, 32, and 35, with a
256-token limit. That is 12 single-layer steered trials plus one control; the
earlier `center-three-layers` experiment instead intervened at all three layers
in each steered trial and remains intact.

The [follow-up suite draft](drafts/README.md) groups the later question families
into one matrix for one Slurm job. Its steering settings are deliberately unset
until the single-layer results have been inspected.
