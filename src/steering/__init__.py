"""Persona-vector steering; load inference dependencies only when requested."""

from importlib import import_module

_EXPORTS = {
    "capture": ("ActivationArtifact", "ActivationCaptureConfig", "ActivationCaptureEvent"),
    "inference": ("GenerationResponse", "InferenceRunner"),
    "interventions": ("CappingRequest", "CharacterBandRequest", "SteeringPlan", "SteeringRequest"),
    "recipes": (
        "DefaultVector", "Difference", "Mean", "PublishedAssistantAxis", "RoleDirection",
        "RoleSpecificDirection", "RoleVector", "Sum", "TraitVector", "VectorRecipe",
        "dump_recipe", "load_recipe", "recipe_from_dict",
    ),
    "vector_store": ("MaterializedVector", "VectorStore"),
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    # Keep the public API, without making CPU-only report tools import Torch
    # and Transformers (expensive on shared cluster filesystems).
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
