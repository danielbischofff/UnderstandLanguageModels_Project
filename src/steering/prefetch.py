"""Download or verify an experiment's model and persona-vector artifacts."""

from __future__ import annotations

from huggingface_hub import snapshot_download

from .experiments import UNSTEERED, ExperimentConfig
from .recipes import RoleDirection
from .vector_store import VectorStore


def _model_revisions(config: ExperimentConfig) -> tuple[str | None, ...]:
    model_revision = config.model.get("revision")
    tokenizer_revision = config.model.get("tokenizer_revision", model_revision)
    return tuple(dict.fromkeys((model_revision, tokenizer_revision)))


def prepare_experiment_artifacts(
    config: ExperimentConfig, *, local_files_only: bool
) -> list[str]:
    """Fetch or locally verify the model snapshots and persona vectors."""

    model_id = str(config.model["id"])
    snapshots: list[str] = []
    for revision in _model_revisions(config):
        snapshots.append(
            str(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            )
        )

    store = VectorStore(
        model_id=model_id,
        revision=config.vector_revision,
        offline=local_files_only,
    )
    for character in config.characters:
        if character != UNSTEERED:
            store.materialize(RoleDirection(character))
    return snapshots
