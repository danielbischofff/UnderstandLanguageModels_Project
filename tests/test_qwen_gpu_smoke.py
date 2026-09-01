import os

import pytest
import torch

from steering import InferenceRunner, RoleDirection, SteeringRequest


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN_SMOKE") != "1",
    reason="set RUN_QWEN_SMOKE=1 on a node with enough VRAM for Qwen3-32B",
)
def test_qwen3_32b_fp8_steering_smoke():
    runner = InferenceRunner.from_pretrained(
        "Qwen/Qwen3-32B-FP8", dtype=torch.bfloat16, device_map="auto"
    )
    conversation = [{"role": "user", "content": "Who are you?"}]
    baseline = runner.generate(conversation, seed=7, do_sample=False, max_new_tokens=8)
    zero = runner.generate(
        conversation,
        steering=SteeringRequest(RoleDirection("lawyer"), coefficient=0.0),
        seed=7,
        do_sample=False,
        max_new_tokens=8,
    )
    assert baseline.generated_token_ids == zero.generated_token_ids
    assert not any(
        block._forward_hooks
        for block in runner.model.model.layers  # type: ignore[attr-defined]
    )
