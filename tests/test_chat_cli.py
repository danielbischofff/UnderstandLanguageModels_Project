import json

import pytest
import torch

from steering.capture import ActivationCaptureEvent
from steering.recipes import DefaultVector, PublishedAssistantAxis, RoleVector
from steering.vector_store import VectorStore
from steering_chat.cli import (
    completion_residuals,
    parse_turn,
    percentile_token_indices,
    plan_for_tags,
    tags_with_defaults,
    visible_token_ids,
)


def test_parse_turn_supports_signed_coefficients_and_preserves_message():
    turn = parse_turn("/5bard /-2default_vector What do you like to sing?")
    assert [(tag.name, tag.coefficient) for tag in turn.tags] == [
        ("bard", 5.0),
        ("default_vector", -2.0),
    ]
    assert turn.text == "What do you like to sing?"


def test_parse_turn_defaults_to_addition_and_reserves_stop():
    turn = parse_turn("/-bard hello")
    assert turn.tags[0].coefficient == -1.0
    assert parse_turn("/stop").stop
    with pytest.raises(ValueError, match="on its own"):
        parse_turn("/stop now")
    with pytest.raises(ValueError, match="after the steering tags"):
        parse_turn("/bard")


def test_tags_become_direct_raw_middle_layer_additions():
    turn = parse_turn("/bard /-default /2assistant_axis hello")
    plan = plan_for_tags(turn.tags, layer=32)
    assert plan is not None
    assert [type(item.vector) for item in plan.additive] == [
        RoleVector,
        DefaultVector,
        PublishedAssistantAxis,
    ]
    assert [item.coefficient for item in plan.additive] == [1.0, -1.0, 2.0]
    assert all(item.layers == (32,) for item in plan.additive)
    assert all(item.scaling == "raw" for item in plan.additive)


def test_neutral_mode_subtracts_default_vector_on_every_turn():
    turn = parse_turn("/bard hello")
    effective = tags_with_defaults(turn.tags, neutral=True)
    assert [(tag.name, tag.coefficient) for tag in effective] == [
        ("default_vector", -1.0),
        ("bard", 1.0),
    ]

    untagged_plan = plan_for_tags((), layer=32, neutral=True)
    assert untagged_plan is not None
    assert isinstance(untagged_plan.additive[0].vector, DefaultVector)
    assert untagged_plan.additive[0].coefficient == -1.0


def test_named_chat_tag_materializes_a_trait_when_no_role_exists(tmp_path):
    local_dir = tmp_path / "vectors"
    trait_path = local_dir / "qwen-3-32b/trait_vectors/evil.pt"
    trait_path.parent.mkdir(parents=True)
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 1
    torch.save(expected, trait_path)
    store = VectorStore(
        model_id="Qwen/Qwen3-32B-FP8",
        local_dir=local_dir,
        index_cache_dir=tmp_path / "indices",
        offline=True,
    )
    store.index_cache_dir.mkdir(parents=True)
    store._index_path.write_text(
        json.dumps(
            {
                "resolved_revision": "test",
                "files": ["qwen-3-32b/trait_vectors/evil.pt"],
            }
        )
    )
    turn = parse_turn("/2evil Be wicked")
    plan = plan_for_tags(turn.tags, layer=1)

    assert plan is not None
    materialized = store.materialize(plan.additive[0].vector)
    assert torch.equal(materialized.tensor, expected)
    assert materialized.artifacts[0].filename.endswith("trait_vectors/evil.pt")


def _event(index, token_id, value):
    return ActivationCaptureEvent(
        tensor=torch.tensor([[[value, 0.0]]]),
        layer=32,
        point="post_intervention",
        phase="decode" if index else "prefill",
        forward_index=index,
        input_token_ids=[[token_id]],
        cache_positions=[index],
        position_policy="last",
    )


def test_completion_residuals_align_visible_tokens_and_ignore_lookahead():
    events = [_event(0, 99, 0.0), _event(1, 10, 1.0), _event(2, 11, 2.0)]
    result = completion_residuals(events, [10, 11], layer=32)
    assert torch.equal(result, torch.tensor([[1.0, 0.0], [2.0, 0.0]]))


def test_percentile_positions_use_concrete_nearest_rank_tokens():
    assert percentile_token_indices(100) == (24, 49, 74)
    assert percentile_token_indices(4) == (0, 1, 2)
    assert percentile_token_indices(1) == (0, 0, 0)
    with pytest.raises(ValueError, match="positive"):
        percentile_token_indices(0)


def test_visible_tokens_remove_lookahead_and_trailing_specials():
    assert visible_token_ids([10, 11, 12], limit=2, special_ids=set()) == [10, 11]
    assert visible_token_ids([10, 2], limit=300, special_ids={2}) == [10]
