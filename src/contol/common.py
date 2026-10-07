"""Shared inputs for the write-up controls: role space, trials, layer-32 residuals.

Residual slices are cached once per trial under ``outputs/contol-cache`` (ignored
by git), so every control reads the 6 GB of retained activations only once.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tsne-lens"))

from tsne_lens.inputs import load_trajectory  # noqa: E402
from tsne_lens.projection import RoleMap  # noqa: E402

MAP = ROOT / "outputs/plots/tsne/roles/role-map.pkl"
DEFAULT = ROOT / "models/assistant-axis-vectors/qwen-3-32b/default_vector.pt"
EXPERIMENTS = ROOT / "runtime/outputs/experiments/17roles-tylenol"
CACHE = ROOT / "outputs/contol-cache"
RESULTS = Path(__file__).resolve().parent / "results"
NEAR = 0.5  # the ribbon threshold: a token is "near" a role above this cosine
LAYER = 32
MODEL = "Qwen/Qwen3-32B"


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@dataclass
class Space:
    labels: list[str]
    vectors: np.ndarray  # raw role vectors, float64
    clusters: np.ndarray  # Ward-6 labels of the saved map
    default: np.ndarray  # D

    @property
    def unit(self) -> np.ndarray:
        return unit(self.vectors)

    @property
    def mean(self) -> np.ndarray:
        return self.vectors.mean(0)

    def index(self, role: str) -> int:
        return self.labels.index(role)

    def steering(self, role: str) -> np.ndarray:
        """c = R - D, the direction every steering scheme uses."""
        return self.vectors[self.index(role)] - self.default


def load_space() -> Space:
    role_map = RoleMap.load(MAP)
    default = torch.load(DEFAULT, map_location="cpu", weights_only=True)
    default = default[LAYER] if default.ndim == 2 else default
    return Space(
        list(role_map.labels),
        role_map.vectors.astype(np.float64),
        np.asarray(role_map.clusters),
        default.float().numpy().astype(np.float64),
    )


def runs() -> dict[str, Path]:
    """The two 17-role Tylenol sweeps, keyed by steering method."""
    found = {}
    for method in ("alpha", "band"):
        candidates = sorted((EXPERIMENTS / f"tylenol-8000mg-roles-{method}/runs").iterdir())
        found[method] = candidates[-1]
    return found


def trials(method: str, run: Path) -> list[dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    outcomes = json.loads((run / "judging/outcomes.json").read_text())["outcomes"]
    records = []
    for trial in sorted(manifest["trials"].values(), key=lambda t: t["index"]):
        if trial["status"] != "complete":
            continue
        steering = trial.get("steering") or {}
        trial_hash = trial["path"].rsplit("_", 1)[-1]
        records.append(
            dict(
                method=method if steering else "unsteered",
                run=method,
                character=trial["character"],
                layer=(steering.get("layers") or [None])[0],
                coefficient=steering.get("coefficient"),
                judge=int(outcomes[trial_hash]),
                hash=trial_hash,
                path=run / trial["path"],
            )
        )
    return records


def residuals(trial: dict) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Layer-32 post-intervention residuals and token texts per segment."""
    cache = CACHE / trial["run"] / f"{trial['hash']}.npz"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for segment in ("prompt", "generated"):
            x, rows = load_trajectory(
                trial["path"],
                layer=LAYER,
                model_id=MODEL,
                prompt_only=segment == "prompt",
                generated_only=segment == "generated",
            )
            arrays[segment] = x.astype(np.float32)
            arrays[f"{segment}_tokens"] = np.array([r["token"] for r in rows])
        np.savez(cache, **arrays)
    with np.load(cache) as data:
        return {
            segment: (data[segment].astype(np.float64), data[f"{segment}_tokens"].tolist())
            for segment in ("prompt", "generated")
        }


def content_mask(tokens: list[str]) -> np.ndarray:
    """Prompt tokens written by the user: between "user\\n" and the first <|im_end|>."""
    end = tokens.index("<|im_end|>")
    mask = np.zeros(len(tokens), dtype=bool)
    mask[3:end] = True
    return mask


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(rounded(payload), indent=1) + "\n")
    return path


def rounded(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value
