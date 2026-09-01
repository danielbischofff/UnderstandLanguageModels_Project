"""Serializable vector recipes for persona steering.

The named directions in this module intentionally encode their sign.  In
particular, the published Assistant Axis points from the role centroid toward
the neutral/default Assistant centroid.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml


class VectorRecipe:
    """Base class for a small, serializable vector expression tree."""

    kind: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def stable_hash(self, length: int = 16) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:length]


@dataclass(frozen=True)
class DefaultVector(VectorRecipe):
    """The neutral/default-Assistant centroid D."""

    kind: ClassVar[str] = "default_vector"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind}


@dataclass(frozen=True)
class RoleVector(VectorRecipe):
    """The fully elicited role centroid R_r."""

    name: str
    kind: ClassVar[str] = "role_vector"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("role name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "name": self.name}


@dataclass(frozen=True)
class TraitVector(VectorRecipe):
    name: str
    kind: ClassVar[str] = "trait_vector"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("trait name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "name": self.name}


@dataclass(frozen=True)
class PublishedAssistantAxis(VectorRecipe):
    """The published Assistant Axis D - C."""

    kind: ClassVar[str] = "published_assistant_axis"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind}


@dataclass(frozen=True)
class _AllRoles(VectorRecipe):
    """Internal leaf resolving to all role artifacts at a repository revision."""

    kind: ClassVar[str] = "all_roles"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind}


def _validated_children(recipes: Iterable[VectorRecipe]) -> tuple[VectorRecipe, ...]:
    result = tuple(recipes)
    if not result:
        raise ValueError("recipe operands must not be empty")
    if not all(isinstance(item, VectorRecipe) for item in result):
        raise TypeError("all operands must be VectorRecipe instances")
    return result


def _validated_weights(
    weights: Sequence[float] | None, expected: int, *, mean: bool
) -> tuple[float, ...] | None:
    if weights is None:
        return None
    result = tuple(float(weight) for weight in weights)
    if len(result) != expected:
        raise ValueError(f"expected {expected} weights, got {len(result)}")
    if not all(math.isfinite(weight) for weight in result):
        raise ValueError("weights must be finite")
    if mean and math.isclose(sum(result), 0.0, abs_tol=1e-15):
        raise ValueError("weighted mean weights must not sum to zero")
    return result


@dataclass(frozen=True)
class Mean(VectorRecipe):
    """Arithmetic or weighted mean; weighted means normalize by sum(weights)."""

    recipes: tuple[VectorRecipe, ...]
    weights: tuple[float, ...] | None = None
    kind: ClassVar[str] = "mean"

    def __init__(
        self, recipes: Iterable[VectorRecipe], weights: Sequence[float] | None = None
    ) -> None:
        children = _validated_children(recipes)
        object.__setattr__(self, "recipes", children)
        object.__setattr__(
            self, "weights", _validated_weights(weights, len(children), mean=True)
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.kind,
            "recipes": [recipe.to_dict() for recipe in self.recipes],
        }
        if self.weights is not None:
            result["weights"] = list(self.weights)
        return result


@dataclass(frozen=True)
class Sum(VectorRecipe):
    """Arithmetic or weighted sum; weights are never normalized."""

    recipes: tuple[VectorRecipe, ...]
    weights: tuple[float, ...] | None = None
    kind: ClassVar[str] = "sum"

    def __init__(
        self, recipes: Iterable[VectorRecipe], weights: Sequence[float] | None = None
    ) -> None:
        children = _validated_children(recipes)
        object.__setattr__(self, "recipes", children)
        object.__setattr__(
            self, "weights", _validated_weights(weights, len(children), mean=False)
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.kind,
            "recipes": [recipe.to_dict() for recipe in self.recipes],
        }
        if self.weights is not None:
            result["weights"] = list(self.weights)
        return result


@dataclass(frozen=True)
class Difference(VectorRecipe):
    """The vector ``positive - negative``."""

    positive: VectorRecipe
    negative: VectorRecipe
    kind: ClassVar[str] = "difference"

    def __post_init__(self) -> None:
        if not isinstance(self.positive, VectorRecipe) or not isinstance(
            self.negative, VectorRecipe
        ):
            raise TypeError("difference operands must be VectorRecipe instances")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
        }


@dataclass(frozen=True)
class RoleDirection(VectorRecipe):
    """Direction from the neutral/default Assistant to role r: R_r - D."""

    name: str
    kind: ClassVar[str] = "role_direction"

    def __post_init__(self) -> None:
        RoleVector(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "name": self.name}

    def expanded(self) -> Difference:
        return Difference(RoleVector(self.name), DefaultVector())


@dataclass(frozen=True)
class RoleSpecificDirection(VectorRecipe):
    """Role-specific contrast against the role centroid: R_r - C."""

    name: str
    kind: ClassVar[str] = "role_specific_direction"

    def __post_init__(self) -> None:
        RoleVector(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "name": self.name}

    def expanded(self) -> Difference:
        return Difference(RoleVector(self.name), Mean([_AllRoles()]))


_LEAF_TYPES: dict[str, type[VectorRecipe]] = {
    DefaultVector.kind: DefaultVector,
    PublishedAssistantAxis.kind: PublishedAssistantAxis,
    _AllRoles.kind: _AllRoles,
}


def recipe_from_dict(value: Mapping[str, Any]) -> VectorRecipe:
    """Deserialize a recipe from a JSON/YAML-compatible mapping."""

    if not isinstance(value, Mapping):
        raise TypeError("recipe must be a mapping")
    kind = value.get("type")
    if kind in _LEAF_TYPES:
        return _LEAF_TYPES[str(kind)]()
    if kind == RoleVector.kind:
        return RoleVector(str(value["name"]))
    if kind == TraitVector.kind:
        return TraitVector(str(value["name"]))
    if kind == RoleDirection.kind:
        return RoleDirection(str(value["name"]))
    if kind == RoleSpecificDirection.kind:
        return RoleSpecificDirection(str(value["name"]))
    if kind in {Mean.kind, Sum.kind}:
        children = [recipe_from_dict(item) for item in value.get("recipes", [])]
        cls = Mean if kind == Mean.kind else Sum
        return cls(children, weights=value.get("weights"))
    if kind == Difference.kind:
        return Difference(
            recipe_from_dict(value["positive"]),
            recipe_from_dict(value["negative"]),
        )
    raise ValueError(f"unknown vector recipe type: {kind!r}")


def load_recipe(path: str | Path) -> VectorRecipe:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return recipe_from_dict(value)


def dump_recipe(recipe: VectorRecipe, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(recipe.to_dict(), handle, sort_keys=False)
