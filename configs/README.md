# Experiments

Every YAML file in `configs/experiments/` defines one complete experiment:
questions × characters, steering dose/layers, generation settings, seed, and
the number of J-Lens words retained.

Use the reserved character `unsteered` as the control. All other character names
must exist in the published Assistant Axis role-vector dataset.
