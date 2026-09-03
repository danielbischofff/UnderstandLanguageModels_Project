# Experiments

Every YAML file in `configs/experiments/` defines one complete experiment:
model and dtype, questions × characters, steering dose/layers, generation
settings, seed, and the number of J-Lens words retained.

Use the reserved character `unsteered` as the control. All other character names
must exist in the published Assistant Axis role-vector dataset.

`model.id` and `model.dtype` are required. Optional model fields are `revision`,
`tokenizer_revision`, `quantization` (`4bit` or `8bit`), and
`enable_thinking`. `vectors.revision` selects the published vector revision.

J-Lens residuals are discarded after readout by default. Set
`outputs.keep_activations: true` to retain one safetensors file and alignment
manifest per completed matrix cell for later low-level inspection.
