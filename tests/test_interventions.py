import pytest
import torch

from steering.interventions import (
    CappingRequest,
    SteeringRequest,
    apply_addition,
    apply_cap,
)
from steering.recipes import RoleDirection


def test_addition_changes_only_direction():
    activations = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])
    delta = torch.tensor([2.0, 0.0])
    changed = apply_addition(activations, delta)
    assert torch.equal(changed[..., 1], activations[..., 1])
    assert torch.equal(changed[..., 0], activations[..., 0] + 2)
    assert torch.equal(activations, torch.tensor([[[3.0, 4.0], [1.0, 2.0]]]))


def test_last_position_and_norm_preservation():
    activations = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    changed = apply_addition(
        activations,
        torch.tensor([1.0, 0.0]),
        positions="last",
        preserve_original_activation_norm=True,
    )
    assert torch.equal(changed[:, 0], activations[:, 0])
    assert torch.allclose(changed[:, -1].norm(dim=-1), activations[:, -1].norm(dim=-1))


def test_lower_upper_and_interval_caps():
    activations = torch.tensor([[[3.0, 2.0], [-2.0, 4.0], [0.5, 1.0]]])
    direction = torch.tensor([1.0, 0.0])
    upper = apply_cap(activations, direction, bound="upper", threshold=1.0)
    lower = apply_cap(activations, direction, bound="lower", threshold=0.0)
    interval = apply_cap(activations, direction, bound="interval", threshold=(0.0, 1.0))
    assert torch.equal(upper[..., 0], torch.tensor([[1.0, -2.0, 0.5]]))
    assert torch.equal(lower[..., 0], torch.tensor([[3.0, 0.0, 0.5]]))
    assert torch.equal(interval[..., 0], torch.tensor([[1.0, 0.0, 0.5]]))
    assert torch.equal(interval[..., 1], activations[..., 1])


def test_capping_requires_explicit_layers():
    with pytest.raises(ValueError, match="layers"):
        CappingRequest(RoleDirection("lawyer"), "upper", None, threshold=1.0)  # type: ignore[arg-type]


def test_coefficients_must_be_finite():
    with pytest.raises(ValueError):
        SteeringRequest(RoleDirection("lawyer"), coefficient=float("inf"))
