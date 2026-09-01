import os

import pytest
import torch

from steering.hooks import SteeringContext
from steering.interventions import ResolvedIntervention


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_HF_INTEGRATION") != "1",
    reason="set RUN_HF_INTEGRATION=1 to download the tiny public model",
)
def test_tiny_hf_model_logits_and_cleanup():
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    model_id = "hf-internal-testing/tiny-random-gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer("hello world", return_tensors="pt")
    with torch.inference_mode():
        baseline = model(**inputs).logits
    zero_context = SteeringContext(model, [])
    with torch.inference_mode(), zero_context:
        zero = model(**inputs).logits
    assert torch.equal(baseline, zero)
    width = model.config.n_embd
    operation = ResolvedIntervention(
        operation="add",
        layer=1,
        direction=torch.nn.functional.normalize(torch.ones(width), dim=0),
        delta=torch.ones(width) * 0.1,
        positions="all",
        recipe_hash="test",
    )
    context = SteeringContext(model, [operation])
    with torch.inference_mode(), context:
        steered = model(**inputs).logits
        set_seed(7)
        first = model.generate(**inputs, do_sample=True, max_new_tokens=3)
        set_seed(7)
        second = model.generate(**inputs, do_sample=True, max_new_tokens=3)
    assert not torch.equal(baseline, steered)
    assert torch.equal(first, second)
    assert not model.transformer.h[1]._forward_hooks
