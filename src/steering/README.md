# Core steering utilities

The `steering` package applies persona-vector interventions to Hugging Face
causal language models using PyTorch forward hooks. Hooks operate on
transformer-block on post-MLP residual updates.

## Vector recipes

Named directions preserve these signs:

- `RoleDirection(role)`: `R_role - D`, from the default Assistant toward a role.
- `RoleSpecificDirection(role)`: `R_role - C`,from the roleplaying centroid toward the role centroid (role axis idea similar to the Assistant axis setup).
- `PublishedAssistantAxis()`: `D - C`, toward the default Assistant.

Recipes can be composed without modifying source tensors:

```python
from steering import DefaultVector, Difference, Mean, RoleVector

direction = Difference(
    Mean(
        [RoleVector("teacher"), RoleVector("critic")],
        weights=[0.8, 0.2],
    ),
    DefaultVector(),
)
```

`role_vectors/assistant.pt` remains an elicited role vector and is never used
as the neutral `DefaultVector()`.

## Additive steering

`layers=None` selects the zero-based middle layer—layer 32 for Qwen3-32B.

```python
from steering import InferenceRunner, RoleDirection, SteeringRequest

runner = InferenceRunner.from_pretrained("Qwen/Qwen3-32B-FP8")
request = SteeringRequest(
    vector=RoleDirection("lawyer"),
    coefficient=0.05,
    layers=None,
    scaling="average_residual_norm",
    calibration="calibration/qwen3-32b-lawyer.json",
    positions="all",
)

response = runner.generate(
    [{"role": "user", "content": "How would you review this contract?"}],
    steering=request,
    seed=42,
    max_new_tokens=128,
)
print(response.text)
```

Scaling modes are `raw`, `unit`, `average_residual_norm`, and
`projection_std`. Position policies are `all`, `last`, and `generated_only`.

## Capping and combined plans

Capping is disabled by default and always requires explicit layers.

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
        SteeringRequest(
            vector=RoleDirection("lawyer"),
            coefficient=0.05,
            layers=[32],
        )
    ],
    caps=[
        CappingRequest(
            direction=PublishedAssistantAxis(),
            bound="lower",
            quantile=0.25,
            layers=list(range(46, 54)),
            calibration="calibration/qwen3-32b-assistant-axis.json",
        )
    ],
    ordering="add_then_cap",
)

conversation = [{"role": "user", "content": "Who are you?"}]
response = runner.generate(conversation, steering=plan)
```

Use `bound="lower"`, `"upper"`, or `"interval"`. Thresholds can be absolute,
per-layer mappings, or quantiles stored in compatible calibration metadata.

## Downstream activation exports

Full post-MLP residual streams can be captured independently of steering for
later J-Lens/J-space, NLA, SAE, probing, or other analysis. This package does
not implement those methods; it exports a framework-neutral ingestion
contract:

```python
from steering import ActivationCaptureConfig

response = runner.generate(
    [{"role": "user", "content": "Who are you?"}],
    steering=plan,
    activation_capture=ActivationCaptureConfig(
        layers=[32, 40],
        points=["pre_intervention", "post_intervention"],
        positions="all",
        dtype="bfloat16",
    ),
)

for event in response.activations:
    print(event.layer, event.point, event.phase, event.tensor.shape)
```

Captures are off by default because residual streams can be large. Recorded
experiments write `activations/activations.safetensors` and a versioned
`activations/manifest.json`. Each event includes its zero-based layer,
pre/post-intervention stage, prefill/decode phase, forward index, model input
token IDs, cache positions when exposed by Transformers, tensor shape, and
dtype. Tensor axes are `[batch, selected_sequence_position, hidden]`.

`positions="all"` retains full prefill and decode inputs, `"last"` retains the
last input position in each call, and `"generated_only"` retains cached decode
calls only. The first generated token is sampled from the last prefill
position; the manifest states this explicitly so consumers do not shift token
attribution accidentally. Use the pre/post pair to compare the same hook point
before and after all ordered additions and caps.

Downstream code can lazily load individual events without depending on the
experiment runner:

```python
from steering import ActivationArtifact

artifact = ActivationArtifact.open("path/to/trial/activations")
for event, tensor in artifact.events():
    consume(layer=event["layer"], token_ids=event["input_token_ids"], h=tensor)
```

For a lightweight token inspection, `steering-j-lens` passes selected events
to the existing repository J-Lens transport and unembedding utilities. See the
[CLI guide](cli/README.md#quick-j-lens-token-readout).

## Vector storage and calibration

Fetch and materialize vectors without loading a language model:

```python
from steering import RoleDirection, VectorStore

store = VectorStore(model_id="Qwen/Qwen3-32B-FP8", revision="main")
print(store.list_roles())

lawyer = store.materialize(RoleDirection("lawyer"))
print(lawyer.tensor.shape)  # [64, 5120]
print(lawyer.layer_norms[32])
```

Calibration streams assistant-response activations instead of retaining full
activation tensors:

```python
from steering import RoleDirection
from steering.calibration import CalibrationRunner

calibrator = CalibrationRunner(
    runner.model,
    runner.tokenizer,
    runner.vector_store,
    model_id=runner.model_id,
    enable_thinking=False,
)
conversations = [
    [
        {"role": "user", "content": "What perspective do you bring?"},
        {"role": "assistant", "content": "I analyze the question carefully."},
    ]
]
record = calibrator.run(
    conversations,
    layers=[32],
    directions=[RoleDirection("lawyer")],
    dataset={"id": "local-reference", "revision": "v1"},
    seed=42,
)
```

The default Qwen checkpoint is `Qwen/Qwen3-32B-FP8`; non-quantized residual
activations are handled in bfloat16 and thinking is disabled. Pass
`model_id="Qwen/Qwen3-32B"` for the original bf16 checkpoint. Optional
bitsandbytes loading uses `quantization="4bit"` or `"8bit"`. Python hooks are
not advertised as compatible with `torch.compile`.
