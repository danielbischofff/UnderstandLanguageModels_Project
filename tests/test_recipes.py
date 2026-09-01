import json

import pytest
import torch

from steering.recipes import (
    DefaultVector,
    Difference,
    Mean,
    PublishedAssistantAxis,
    RoleDirection,
    RoleSpecificDirection,
    RoleVector,
    Sum,
    recipe_from_dict,
)
from steering.vector_store import VectorStore


class MemoryStore(VectorStore):
    def __init__(self):
        self.model_id = "test/model"
        self.values = {
            "default_vector.pt": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            "assistant_axis.pt": torch.tensor([[0.5, 0.5], [1.0, 1.0]]),
            "role_vectors/lawyer.pt": torch.tensor([[3.0, 2.0], [5.0, 4.0]]),
            "role_vectors/critic.pt": torch.tensor([[1.0, 4.0], [3.0, 6.0]]),
        }

    def list_roles(self):
        return ["critic", "lawyer"]

    def _load_artifact(self, name):
        from steering.vector_store import ArtifactMetadata

        tensor = self.values[name]
        return tensor, ArtifactMetadata(
            "repo",
            "main",
            "sha",
            name,
            name,
            "sum",
            str(tensor.dtype),
            list(tensor.shape),
            self.model_id,
        )


def test_named_direction_signs_are_exact():
    store = MemoryStore()
    default = store.values["default_vector.pt"]
    lawyer = store.values["role_vectors/lawyer.pt"]
    critic = store.values["role_vectors/critic.pt"]
    centroid = (lawyer + critic) / 2
    assert torch.equal(
        store.materialize(RoleDirection("lawyer")).tensor, lawyer - default
    )
    assert torch.equal(
        store.materialize(RoleSpecificDirection("lawyer")).tensor, lawyer - centroid
    )
    assert torch.equal(
        store.materialize(PublishedAssistantAxis()).tensor,
        store.values["assistant_axis.pt"],
    )


def test_weighted_mean_and_sum_differ():
    store = MemoryStore()
    recipes = [RoleVector("lawyer"), RoleVector("critic")]
    mean = store.materialize(Mean(recipes, weights=[3, 1])).tensor
    summed = store.materialize(Sum(recipes, weights=[3, 1])).tensor
    assert torch.allclose(summed, mean * 4)


def test_recipe_round_trip_and_no_generic_direction_export():
    recipe = Difference(
        Mean([RoleVector("lawyer"), RoleVector("critic")], weights=[0.8, 0.2]),
        DefaultVector(),
    )
    encoded = json.loads(recipe.canonical_json())
    assert recipe_from_dict(encoded) == recipe
    import steering

    assert not hasattr(steering, "GenericRoleplayDirection")


def test_recipe_validation():
    with pytest.raises(ValueError):
        Mean([])
    with pytest.raises(ValueError):
        Mean([DefaultVector()], weights=[float("nan")])
    with pytest.raises(ValueError):
        Mean([DefaultVector(), RoleVector("x")], weights=[1, -1])
    with pytest.raises(ValueError):
        Sum([], weights=[])
