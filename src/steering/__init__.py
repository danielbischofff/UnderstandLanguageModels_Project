"""Research-oriented persona-vector steering for Hugging Face causal LMs."""

from .capture import ActivationArtifact, ActivationCaptureConfig, ActivationCaptureEvent
from .inference import GenerationResponse, InferenceRunner
from .interventions import CappingRequest, CharacterBandRequest, SteeringPlan, SteeringRequest
from .recipes import (
    DefaultVector,
    Difference,
    Mean,
    PublishedAssistantAxis,
    RoleDirection,
    RoleSpecificDirection,
    RoleVector,
    Sum,
    TraitVector,
    VectorRecipe,
    dump_recipe,
    load_recipe,
    recipe_from_dict,
)
from .vector_store import MaterializedVector, VectorStore

__all__ = [
    "ActivationArtifact",
    "ActivationCaptureConfig",
    "ActivationCaptureEvent",
    "CappingRequest",
    "CharacterBandRequest",
    "DefaultVector",
    "Difference",
    "GenerationResponse",
    "InferenceRunner",
    "MaterializedVector",
    "Mean",
    "PublishedAssistantAxis",
    "RoleDirection",
    "RoleSpecificDirection",
    "RoleVector",
    "SteeringPlan",
    "SteeringRequest",
    "Sum",
    "TraitVector",
    "VectorRecipe",
    "VectorStore",
    "dump_recipe",
    "load_recipe",
    "recipe_from_dict",
]
