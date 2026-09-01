"""Fetch, validate, and materialize published persona-vector artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download

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
    _AllRoles,
)

DEFAULT_VECTOR_REPO = "lu-christina/assistant-axis-vectors"
MODEL_DIRECTORIES = {
    "Qwen/Qwen3-32B": "qwen-3-32b",
    "google/gemma-2-27b-it": "gemma-2-27b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
}


class VectorStoreError(RuntimeError):
    pass


class MissingArtifactsError(VectorStoreError):
    def __init__(self, filenames: Iterable[str], *, offline: bool) -> None:
        files = sorted(set(filenames))
        mode = "offline mode" if offline else "the configured vector source"
        super().__init__(f"Missing vector artifacts in {mode}: {', '.join(files)}")
        self.filenames = files


@dataclass(frozen=True)
class ArtifactMetadata:
    repository: str
    requested_revision: str
    resolved_revision: str | None
    filename: str
    local_path: str
    sha256: str
    dtype: str
    shape: list[int]
    model_id: str


@dataclass(frozen=True)
class MaterializedVector:
    recipe: VectorRecipe
    tensor: torch.Tensor
    artifacts: tuple[ArtifactMetadata, ...]
    layer_norms: tuple[float, ...]

    @property
    def recipe_hash(self) -> str:
        return self.recipe.stable_hash()

    def metadata(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe.to_dict(),
            "recipe_hash": self.recipe_hash,
            "shape": list(self.tensor.shape),
            "dtype": str(self.tensor.dtype),
            "layer_norms": list(self.layer_norms),
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_tensor(value: Any, path: Path) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        tensor = next(
            (
                value[key]
                for key in ("vector", "axis", "vectors", "tensor")
                if key in value and torch.is_tensor(value[key])
            ),
            None,
        )
        if tensor is None:
            raise VectorStoreError(
                f"Legacy tensor dictionary {path} has no supported tensor key"
            )
    else:
        raise VectorStoreError(
            f"Unsupported vector artifact format in {path}: {type(value).__name__}"
        )
    if tensor.ndim != 2:
        raise VectorStoreError(
            f"Vector artifact {path} must have shape [layers, hidden], got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise VectorStoreError(f"Vector artifact {path} contains non-finite values")
    return tensor.detach().to(device="cpu")


class VectorStore:
    """Materialize vector recipes without loading a language model or GPU."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-32B",
        *,
        repo_id: str = DEFAULT_VECTOR_REPO,
        revision: str = "main",
        dataset_model_dir: str | None = None,
        local_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        index_cache_dir: str | Path | None = None,
        offline: bool = False,
    ) -> None:
        self.model_id = model_id
        self.repo_id = repo_id
        self.revision = revision
        self.dataset_model_dir = dataset_model_dir or MODEL_DIRECTORIES.get(model_id)
        self.local_dir = Path(local_dir).expanduser() if local_dir else None
        self.cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None
        self.offline = offline
        default_index = Path.home() / ".cache" / "steering" / "indices"
        self.index_cache_dir = Path(index_cache_dir or default_index).expanduser()
        self._resolved_revision: str | None = None
        self._cached_index: list[str] | None = None

    def _model_dir(self) -> str:
        if self.dataset_model_dir is None:
            raise VectorStoreError(
                f"No vector directory mapping for {self.model_id!r}; pass dataset_model_dir explicitly before fetching or materializing vectors"
            )
        return self.dataset_model_dir

    @property
    def _index_path(self) -> Path:
        key = hashlib.sha256(
            f"{self.repo_id}:{self.revision}:{self._model_dir()}".encode()
        ).hexdigest()[:20]
        return self.index_cache_dir / f"{key}.json"

    def _remote_index(self) -> list[str]:
        if self.offline:
            if not self._index_path.exists():
                raise MissingArtifactsError(["<cached repository index>"], offline=True)
            return json.loads(self._index_path.read_text(encoding="utf-8"))["files"]
        api = HfApi()
        info = api.dataset_info(self.repo_id, revision=self.revision)
        self._resolved_revision = info.sha
        files = sorted(
            path
            for path in api.list_repo_files(
                self.repo_id, repo_type="dataset", revision=self.revision
            )
            if path.startswith(f"{self._model_dir()}/")
        )
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(
            json.dumps(
                {"resolved_revision": info.sha, "files": files},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._cached_index = files
        return files

    def _index(self) -> list[str]:
        if self._cached_index is not None:
            return self._cached_index
        if self.offline:
            value = (
                json.loads(self._index_path.read_text(encoding="utf-8"))
                if self._index_path.exists()
                else None
            )
            if value is None:
                if self.local_dir is not None:
                    model_root = self.local_dir / self._model_dir()
                    local_files = (
                        sorted(
                            str(path.relative_to(self.local_dir))
                            for path in model_root.rglob("*")
                            if path.is_file()
                        )
                        if model_root.exists()
                        else []
                    )
                    if local_files:
                        self._cached_index = local_files
                        return local_files
                return self._remote_index()
            self._resolved_revision = value.get("resolved_revision")
            self._cached_index = value["files"]
            return self._cached_index
        return self._remote_index()

    def list_roles(self) -> list[str]:
        prefix = f"{self._model_dir()}/role_vectors/"
        return sorted(
            Path(path).stem
            for path in self._index()
            if path.startswith(prefix) and path.endswith(".pt")
        )

    def list_traits(self) -> list[str]:
        prefix = f"{self._model_dir()}/trait_vectors/"
        return sorted(
            Path(path).stem
            for path in self._index()
            if path.startswith(prefix) and path.endswith(".pt")
        )

    def _local_candidate(self, filename: str) -> Path | None:
        if self.local_dir is None:
            return None
        candidate = self.local_dir / filename
        return candidate if candidate.exists() else None

    def fetch(self, relative_filename: str) -> Path:
        filename = f"{self._model_dir()}/{relative_filename}"
        local = self._local_candidate(filename)
        if local is not None:
            if self._resolved_revision is None and self._index_path.exists():
                index = json.loads(self._index_path.read_text(encoding="utf-8"))
                self._resolved_revision = index.get("resolved_revision")
            return local
        if not self.offline and self._resolved_revision is None:
            self._resolved_revision = (
                HfApi().dataset_info(self.repo_id, revision=self.revision).sha
            )
        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=filename,
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_dir=str(self.local_dir) if self.local_dir else None,
                local_files_only=self.offline,
            )
        except Exception as error:
            raise MissingArtifactsError([filename], offline=self.offline) from error
        result = Path(path)
        parts = result.parts
        if "snapshots" in parts:
            index = parts.index("snapshots")
            if index + 1 < len(parts):
                self._resolved_revision = parts[index + 1]
        return result

    def _load_artifact(
        self, relative_filename: str
    ) -> tuple[torch.Tensor, ArtifactMetadata]:
        path = self.fetch(relative_filename)
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch 2.0 compatibility
            value = torch.load(path, map_location="cpu")
        except Exception as error:
            raise VectorStoreError(
                f"Unable to safely load tensor artifact {path}: {error}"
            ) from error
        tensor = _extract_tensor(value, path)
        metadata = ArtifactMetadata(
            repository=self.repo_id,
            requested_revision=self.revision,
            resolved_revision=self._resolved_revision,
            filename=f"{self._model_dir()}/{relative_filename}",
            local_path=str(path),
            sha256=_sha256(path),
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            model_id=self.model_id,
        )
        return tensor, metadata

    @staticmethod
    def _combine(
        tensors: list[torch.Tensor], weights: tuple[float, ...] | None, *, mean: bool
    ) -> torch.Tensor:
        if not tensors:
            raise VectorStoreError("cannot combine an empty tensor collection")
        shape = tensors[0].shape
        if any(tensor.shape != shape for tensor in tensors[1:]):
            raise VectorStoreError("recipe source tensors have incompatible shapes")
        numeric_weights = weights or tuple(1.0 for _ in tensors)
        result = sum(
            (
                tensor.to(torch.float32) * weight
                for tensor, weight in zip(tensors, numeric_weights)
            ),
            torch.zeros_like(tensors[0], dtype=torch.float32),
        )
        if mean:
            result = result / sum(numeric_weights)
        return result

    def _materialize(
        self, recipe: VectorRecipe
    ) -> tuple[torch.Tensor, list[ArtifactMetadata]]:
        if isinstance(recipe, DefaultVector):
            tensor, metadata = self._load_artifact("default_vector.pt")
            return tensor, [metadata]
        if isinstance(recipe, RoleVector):
            roles = self.list_roles()
            if recipe.name not in roles:
                raise VectorStoreError(
                    f"Unknown role {recipe.name!r} for {self.model_id}; available roles: {', '.join(roles)}"
                )
            tensor, metadata = self._load_artifact(f"role_vectors/{recipe.name}.pt")
            return tensor, [metadata]
        if isinstance(recipe, TraitVector):
            traits = self.list_traits()
            if recipe.name not in traits:
                raise VectorStoreError(
                    f"Unknown trait {recipe.name!r} for {self.model_id}; available traits: {', '.join(traits)}"
                )
            tensor, metadata = self._load_artifact(f"trait_vectors/{recipe.name}.pt")
            return tensor, [metadata]
        if isinstance(recipe, PublishedAssistantAxis):
            tensor, metadata = self._load_artifact("assistant_axis.pt")
            return tensor, [metadata]
        if isinstance(recipe, RoleDirection):
            return self._materialize(recipe.expanded())
        if isinstance(recipe, RoleSpecificDirection):
            return self._materialize(recipe.expanded())
        if isinstance(recipe, _AllRoles):
            role_results = [
                self._materialize(RoleVector(role)) for role in self.list_roles()
            ]
            tensors = [item[0] for item in role_results]
            artifacts = [artifact for item in role_results for artifact in item[1]]
            return self._combine(tensors, None, mean=True), artifacts
        if isinstance(recipe, Difference):
            positive, positive_meta = self._materialize(recipe.positive)
            negative, negative_meta = self._materialize(recipe.negative)
            if positive.shape != negative.shape:
                raise VectorStoreError("difference operands have incompatible shapes")
            return positive.to(torch.float32) - negative.to(
                torch.float32
            ), positive_meta + negative_meta
        if isinstance(recipe, (Mean, Sum)):
            children = [self._materialize(child) for child in recipe.recipes]
            tensor = self._combine(
                [child[0] for child in children],
                recipe.weights,
                mean=isinstance(recipe, Mean),
            )
            return tensor, [artifact for child in children for artifact in child[1]]
        raise TypeError(f"unsupported recipe: {type(recipe).__name__}")

    def materialize(
        self,
        recipe: VectorRecipe,
        *,
        model_config: Any | None = None,
    ) -> MaterializedVector:
        tensor, artifacts = self._materialize(recipe)
        if model_config is not None:
            expected_layers = int(model_config.num_hidden_layers)
            expected_width = int(model_config.hidden_size)
            if tuple(tensor.shape) != (expected_layers, expected_width):
                raise VectorStoreError(
                    "Vector/model shape mismatch: "
                    f"vector={tuple(tensor.shape)}, model={(expected_layers, expected_width)}"
                )
        norms = tensor.to(torch.float32).norm(dim=-1)
        if not torch.isfinite(norms).all() or torch.any(norms == 0):
            zero_layers = torch.nonzero(norms == 0).flatten().tolist()
            raise VectorStoreError(
                f"materialized vector has zero/non-finite norms at layers {zero_layers}"
            )
        unique_artifacts = tuple(
            {artifact.filename: artifact for artifact in artifacts}.values()
        )
        return MaterializedVector(
            recipe=recipe,
            tensor=tensor,
            artifacts=unique_artifacts,
            layer_norms=tuple(float(value) for value in norms),
        )
