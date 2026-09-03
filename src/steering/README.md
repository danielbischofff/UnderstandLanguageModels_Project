# Steering internals

The public workflow is the experiment YAML described in the repository README.
This package contains the lower-level pieces used by that runner:

- `InferenceRunner` loads a Hugging Face causal LM, generates a response, and
  performs a single teacher-forced prompt-and-response pass for analysis.
- `RoleDirection(role)` materializes `R_role - D`, from the default Assistant
  toward a published character centroid.
- `SteeringRequest` adds either the raw vector or its unit direction at selected
  transformer-block outputs.
- `SteeringContext` installs exception-safe hooks and can capture residuals in
  memory at explicit pre/post-intervention boundaries.
- `JLenseReader` immediately converts full-sequence residuals into top vocabulary
  words per layer and token position. Experiment runs discard activations unless
  `outputs.keep_activations` is enabled.

The default experiment uses raw steering:

```python
from steering import InferenceRunner, RoleDirection, SteeringRequest

runner = InferenceRunner.from_pretrained("Qwen/Qwen3-32B-FP8")
steering = SteeringRequest(
    RoleDirection("demon"), coefficient=1.0, layers=[32], scaling="raw"
)
response = runner.generate(
    [{"role": "user", "content": "Who are you?"}], steering=steering
)
```

`raw` applies `h' = h + alpha * (R_role - D)`. `unit` replaces the direction
with `(R_role - D) / ||R_role - D||`. Explicit-threshold capping remains
available in the Python API, but it is not part of the experiment YAML.

Position policies are `all`, `last`, and `generated_only`. The matrix runner
uses `all`, then reads the post-intervention residual at every fitted J-Lens
layer in one full-sequence causal pass so the final generated token is included.
For unsteered trials this is the original block output; for steered trials it
includes the injection and its downstream effects.
