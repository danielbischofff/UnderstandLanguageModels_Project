"""Readers for tensor files and steering's capture format, without steering imports."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


def extract_tensor(value, path):
    if isinstance(value, dict):
        value = next(
            (
                value[k]
                for k in ("vector", "axis", "vectors", "tensor")
                if k in value and torch.is_tensor(value[k])
            ),
            None,
        )
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"{path} must contain a [layers, hidden] tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{path} contains non-finite values")
    return value.detach().cpu()


@dataclass(frozen=True)
class ActivationArtifact:
    directory: Path
    manifest: dict

    @classmethod
    def open(cls, path):
        path = Path(path)
        manifest_path = path if path.is_file() else path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported activation schema version")
        tensor_path = manifest_path.parent / manifest["tensor_file"]
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        return cls(manifest_path.parent, manifest)

    def load_tensor(self, key):
        with safe_open(
            self.directory / self.manifest["tensor_file"], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(key)
