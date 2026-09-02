import pytest
import torch

from steering.capture import ActivationCaptureEvent
from steering.recipes import DefaultVector, PublishedAssistantAxis, RoleVector
from steering_chat.cli import (
    completion_residuals,
    parse_turn,
    plan_for_tags,
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


def test_visible_tokens_remove_lookahead_and_trailing_specials():
    assert visible_token_ids([10, 11, 12], limit=2, special_ids=set()) == [10, 11]
    assert visible_token_ids([10, 2], limit=300, special_ids={2}) == [10]
