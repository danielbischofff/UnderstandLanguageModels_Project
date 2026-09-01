from types import SimpleNamespace

import pytest
import torch

from steering.inference import InferenceRunner
from steering.interventions import CappingRequest, SteeringPlan, SteeringRequest
from steering.recipes import PublishedAssistantAxis, RoleDirection
from steering.vector_store import MaterializedVector


class Store:
    def materialize(self, recipe, model_config=None):
        tensor = torch.tensor([[1.0, 0.0]] * model_config.num_hidden_layers)
        return MaterializedVector(recipe, tensor, (), tuple([1.0] * len(tensor)))


class Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers, hidden_size=2)
        self.weight = torch.nn.Parameter(torch.ones(1))


def runner(layers=6):
    return InferenceRunner(
        Model(layers), object(), vector_store=Store(), model_id="test"
    )


@pytest.mark.parametrize(("layers", "expected"), [(6, 3), (5, 2)])
def test_middle_layer_default(layers, expected):
    resolved = runner(layers).resolve(
        SteeringRequest(RoleDirection("lawyer"), coefficient=1.0)
    )
    assert [item.layer for item in resolved] == [expected]


def test_zero_coefficient_produces_no_intervention():
    assert (
        runner().resolve(SteeringRequest(RoleDirection("lawyer"), coefficient=0.0))
        == []
    )


def test_raw_unit_and_per_layer_coefficients():
    resolved = runner().resolve(
        SteeringRequest(
            RoleDirection("lawyer"),
            coefficient={1: 2.0, 2: -1.0},
            layers=[1, 2],
            scaling="unit",
        )
    )
    assert torch.equal(resolved[0].delta, torch.tensor([2.0, 0.0]))
    assert torch.equal(resolved[1].delta, torch.tensor([-1.0, 0.0]))
    raw = runner().resolve(
        SteeringRequest(
            RoleDirection("lawyer"), coefficient=3.0, layers=[1], scaling="raw"
        )
    )
    assert torch.equal(raw[0].delta, torch.tensor([3.0, 0.0]))


def test_scaling_modes_and_calibration():
    calibration = {
        "layers": {
            "3": {
                "residual_norm": {"mean": 10.0},
                "directions": {
                    RoleDirection("lawyer").stable_hash(): {
                        "std": 2.0,
                        "quantiles": {"0.25": -1.0},
                    }
                },
            }
        }
    }
    request = SteeringRequest(
        RoleDirection("lawyer"),
        coefficient=0.5,
        scaling="average_residual_norm",
        calibration=calibration,
    )
    assert torch.equal(runner().resolve(request)[0].delta, torch.tensor([5.0, 0.0]))
    request = SteeringRequest(
        RoleDirection("lawyer"),
        coefficient=0.5,
        scaling="projection_std",
        calibration=calibration,
    )
    assert torch.equal(runner().resolve(request)[0].delta, torch.tensor([1.0, 0.0]))


def test_quantile_cap_and_add_then_cap_order():
    recipe = PublishedAssistantAxis()
    calibration = {
        "layers": {
            "4": {
                "residual_norm": {"mean": 1.0},
                "directions": {
                    recipe.stable_hash(): {"std": 1.0, "quantiles": {"0.25": -2.0}}
                },
            }
        }
    }
    plan = SteeringPlan(
        additive=[SteeringRequest(RoleDirection("lawyer"), layers=[4])],
        caps=[
            CappingRequest(recipe, "lower", [4], quantile=0.25, calibration=calibration)
        ],
    )
    resolved = runner().resolve(plan)
    assert [item.operation for item in resolved] == ["add", "cap"]
    assert resolved[1].threshold == -2.0


def test_interval_quantile_cap():
    recipe = PublishedAssistantAxis()
    calibration = {
        "layers": {
            "4": {
                "residual_norm": {"mean": 1.0},
                "directions": {
                    recipe.stable_hash(): {
                        "std": 1.0,
                        "quantiles": {"0.25": -2.0, "0.75": 3.0},
                    }
                },
            }
        }
    }
    request = CappingRequest(
        recipe,
        "interval",
        [4],
        quantile=(0.25, 0.75),
        calibration=calibration,
    )
    assert runner().resolve(request)[0].threshold == (-2.0, 3.0)
