"""Compute contrastive timelines from saved activations or top-word archives."""
from __future__ import annotations

import csv
import json
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from steering.binary_results import decode_trial
from .contrastive import COSINE_MEASURES, MEASURES, RANK_MEASURES, ContrastiveProbe

SCHEMA_VERSION = 1

#: How a layer's numbers were obtained, most trustworthy first.
RECOMPUTED = "recomputed"
COSINE_ONLY = "cosine_from_saved_top_k"


@dataclass(frozen=True)
class SetSeries:
    """One contrastive idea's timeline at one layer of one trial."""

    id: str
    label: str
    color: str
    words: tuple[str, ...]
    measures: Mapping[str, tuple[float | None, ...]]

    def available(self, names: Sequence[str]) -> list[str]:
        return [name for name in names if any(v is not None for v in self.measures.get(name, ()))]


@dataclass(frozen=True)
class LayerSeries:
    layer: int
    source: str
    top_k: int
    sets: tuple[SetSeries, ...]


@dataclass(frozen=True)
class TrialSeries:
    """Everything one plot needs about one matrix cell."""

    key: str
    question: Mapping[str, Any]
    character: str
    steering: Mapping[str, Any] | None
    prompt_length: int
    tokens: tuple[Mapping[str, Any], ...]
    response: str
    layers: tuple[LayerSeries, ...]

    @property
    def coefficient(self) -> float | None:
        return None if self.steering is None else self.steering.get("coefficient")

    @property
    def scheme(self) -> str | None:
        return None if self.steering is None else self.steering.get("name")

    @property
    def label(self) -> str:
        if self.steering is None:
            return f"{self.character} (control)"
        parts = [self.character, f"alpha={self.coefficient:g}"]
        if self.scheme:
            parts.append(str(self.scheme))
        if self.steering.get("method") and self.steering["method"] != "additive":
            parts.append(str(self.steering["method"]))
        return " · ".join(parts)


def _trial_key(trial: Mapping[str, Any]) -> str:
    steering = trial.get("steering")
    parts = [str(trial["character"])]
    if steering:
        parts.append(f"a{float(steering.get('coefficient', 0.0)):g}")
        if steering.get("name"):
            parts.append(str(steering["name"]))
    return _slug("_".join(parts))


def _slug(value: str) -> str:
    kept = [character if character.isalnum() or character in "-_." else "-" for character in value]
    return "".join(kept).strip("-") or "trial"


@dataclass(frozen=True)
class LoadedTrial:
    """One trial's metadata, with its readout words loaded only on demand."""

    metadata: Mapping[str, Any]
    load: Callable[[], Mapping[str, Any]]
    activations: Path | None = None


@dataclass(frozen=True)
class RunArchive:
    experiment: Mapping[str, Any]
    run: str
    source: Path
    trials: tuple[LoadedTrial, ...]


def load_run(source: str | Path) -> RunArchive:
    """Read a run directory or its exported ``results.zip``."""

    path = Path(source)
    if path.suffix == ".zip":
        return _load_bundle(path)
    checkpoints = sorted((path / "trials").glob("*/result.json")) if path.is_dir() else []
    if not checkpoints:
        # A published run keeps only the compact archive; the per-trial
        # checkpoints of a live run additionally point at retained activations.
        if path.is_dir() and (path / "results.zip").is_file():
            return _load_bundle(path / "results.zip")
        raise FileNotFoundError(f"not an experiment run: {path}")
    if not (path / "resolved-config.yaml").is_file():
        raise FileNotFoundError(f"run has no resolved-config.yaml: {path}")
    config = yaml.safe_load((path / "resolved-config.yaml").read_text(encoding="utf-8"))
    trials = []
    for result in checkpoints:
        activations = result.parent / "activations" / "manifest.json"
        trials.append(
            LoadedTrial(
                metadata=_without_readout_words(json.loads(result.read_text(encoding="utf-8"))),
                load=lambda result=result: json.loads(result.read_text(encoding="utf-8")),
                activations=activations if activations.is_file() else None,
            )
        )
    return RunArchive(config, path.name, path, tuple(trials))


def _without_readout_words(trial: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a whole matrix's metadata resident without its top-k word lists."""

    lens = trial["j_lens"]
    return {
        **trial,
        "j_lens": {
            **lens,
            "layers": [
                {key: value for key, value in layer.items() if key != "positions"}
                for layer in lens["layers"]
            ],
        },
    }


def _load_bundle(path: Path) -> RunArchive:
    with zipfile.ZipFile(path) as archive:
        index = json.loads(archive.read("index.json"))
        if index["format"] != "steering-binary-v1":
            raise ValueError(f"unsupported results archive format: {path}")
        vocabulary = json.loads(archive.read("vocabulary.json"))
        trials = []
        for descriptor in index["trials"]:

            def load(descriptor=descriptor) -> Mapping[str, Any]:
                with zipfile.ZipFile(path) as handle:
                    return decode_trial(descriptor, handle.read(descriptor["payload"]), vocabulary)

            trials.append(LoadedTrial(metadata=descriptor["metadata"], load=load))
    config = index["document"]["experiment"]
    return RunArchive(config, path.parent.name, path, tuple(trials))


def _series_values(entry: Mapping[str, Any], length: int) -> dict[str, tuple[float | None, ...]]:
    values = {}
    for measure in MEASURES:
        if measure not in entry:
            continue
        column = list(entry[measure])
        if len(column) != length:
            raise ValueError(
                f"contrastive series for {entry.get('set')!r} has {len(column)} points "
                f"but the trial has {length} token positions"
            )
        values[measure] = tuple(column)
    return values


def measured_layers(
    trial: Mapping[str, Any], *, layers: Sequence[int] | None, set_ids: Sequence[str] | None
) -> list[LayerSeries]:
    """Convert computed measures into plot series."""

    contrastive = trial.get("j_lens", {}).get("contrastive")
    if not contrastive:
        return []
    definitions = {item["id"]: item for item in contrastive["sets"]}
    wanted = list(set_ids) if set_ids else list(definitions)
    if any(identifier not in definitions for identifier in wanted):
        return []
    length = len(trial["j_lens"]["tokens"])
    by_layer: dict[int, list[SetSeries]] = {}
    for entry in contrastive["series"]:
        layer = int(entry["layer"])
        if layers is not None and layer not in layers:
            continue
        if entry["set"] not in wanted:
            continue
        definition = definitions[entry["set"]]
        by_layer.setdefault(layer, []).append(
            SetSeries(
                id=definition["id"],
                label=definition.get("label") or definition["id"],
                color=definition.get("color", "#2a78d6"),
                words=tuple(token["word"] for token in definition.get("tokens", [])),
                measures=_series_values(entry, length),
            )
        )
    order = {identifier: index for index, identifier in enumerate(wanted)}
    return [
        LayerSeries(
            layer=layer,
            source=RECOMPUTED,
            top_k=int(contrastive.get("top_k", 0)),
            sets=tuple(sorted(sets, key=lambda item: order[item.id])),
        )
        for layer, sets in sorted(by_layer.items())
        if len(sets) == len(wanted)
    ]


def top_k_arrays(trial: Mapping[str, Any], layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Saved top words at one layer as ``[positions, k]`` ids and probabilities."""

    for entry in trial["j_lens"]["layers"]:
        if int(entry["layer"]) != layer:
            continue
        positions = entry["positions"]
        width = min(len(words) for words in positions)
        ids = np.asarray([[word["id"] for word in words[:width]] for words in positions], dtype=np.int64)
        probabilities = np.asarray(
            [[word["probability"] for word in words[:width]] for words in positions], dtype=np.float64
        )
        return ids, probabilities
    raise ValueError(f"the trial readout has no layer {layer}")


def recomputed_layers(
    trial: Mapping[str, Any],
    probe: ContrastiveProbe,
    *,
    artifact: ActivationArtifact | None = None,
    reader: Any | None = None,
) -> list[LayerSeries]:
    """Score sets the run never saw, exactly when possible and honestly when not."""

    if artifact is not None and reader is not None:
        # The retained residuals give the complete logit row again, so both
        # measures come back exactly, for the probed layers only.
        computed = reader.probe_artifact(artifact, probe)
        exact = measured_layers(
            {**trial, "j_lens": {**trial["j_lens"], "contrastive": computed}},
            layers=probe.layers,
            set_ids=probe.set_ids,
        )
        return [LayerSeries(item.layer, RECOMPUTED, probe.top_k, item.sets) for item in exact]
    layers = []
    for layer in probe.layers:
        ids, probabilities = top_k_arrays(trial, layer)
        cosine = probe.cosine_measures(ids, probabilities)
        layers.append(
            LayerSeries(
                layer=layer,
                source=COSINE_ONLY,
                top_k=min(probe.top_k, ids.shape[1]),
                sets=tuple(
                    SetSeries(
                        id=item.id,
                        label=item.definition.title,
                        color=item.color,
                        words=tuple(word.word for word in item.words),
                        measures={
                            name: tuple(
                                float(value) if np.isfinite(value) else None
                                for value in cosine[item.id][name]
                            )
                            for name in COSINE_MEASURES
                        },
                    )
                    for item in probe.sets
                ),
            )
        )
    return layers


def trial_series(
    loaded: LoadedTrial,
    *,
    layers: Sequence[int] | None = None,
    probe: ContrastiveProbe | None = None,
    reader: Any | None = None,
) -> TrialSeries | None:
    """Score the supplied sets independently of the experiment configuration."""

    metadata = loaded.metadata
    if probe is None:
        raise ValueError("contrastive analysis requires a probe with explicit word sets")
    trial = loaded.load()
    artifact = None
    if loaded.activations is not None and reader is not None:
        from steering.capture import ActivationArtifact
        artifact = ActivationArtifact.open(loaded.activations)
    series = recomputed_layers(trial, probe, artifact=artifact, reader=reader)
    if not series:
        return None
    tokens = tuple(metadata["j_lens"]["tokens"])
    return TrialSeries(
        key=_trial_key(metadata),
        question=metadata["question"],
        character=str(metadata["character"]),
        steering=metadata.get("steering"),
        prompt_length=int(metadata["j_lens"]["prompt_length"]),
        tokens=tokens,
        response=str(metadata.get("response", {}).get("text", "")),
        layers=tuple(sorted(series, key=lambda item: item.layer)),
    )


def run_series(
    archive: RunArchive,
    *,
    layers: Sequence[int] | None = None,
    probe: ContrastiveProbe | None = None,
    reader: Any | None = None,
    questions: Sequence[str] | None = None,
) -> list[TrialSeries]:
    result = []
    for loaded in archive.trials:
        if questions and loaded.metadata["question"]["id"] not in questions:
            continue
        series = trial_series(loaded, layers=layers, probe=probe, reader=reader)
        if series is not None:
            result.append(series)
    return result


def to_document(archive: RunArchive, series: Sequence[TrialSeries]) -> dict[str, Any]:
    """The complete numbers, in the same shape the plots consume."""

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment": archive.experiment.get("name"),
        "run": archive.run,
        "source": str(archive.source),
        "measures": {"rank": list(RANK_MEASURES), "cosine": list(COSINE_MEASURES)},
        "trials": [
            {
                "key": trial.key,
                "question": dict(trial.question),
                "character": trial.character,
                "steering": trial.steering,
                "prompt_length": trial.prompt_length,
                "tokens": [dict(token) for token in trial.tokens],
                "layers": [
                    {
                        "layer": layer.layer,
                        "source": layer.source,
                        "top_k": layer.top_k,
                        "sets": [
                            {
                                "id": item.id,
                                "label": item.label,
                                "color": item.color,
                                "words": list(item.words),
                                "measures": {
                                    name: list(values) for name, values in item.measures.items()
                                },
                            }
                            for item in layer.sets
                        ],
                    }
                    for layer in trial.layers
                ],
            }
            for trial in series
        ],
    }


def table_rows(archive: RunArchive, series: Sequence[TrialSeries]) -> Iterator[dict[str, Any]]:
    """One row per trial, layer, set, and token position."""

    for trial in series:
        for layer in trial.layers:
            for item in layer.sets:
                for position, token in enumerate(trial.tokens):
                    yield {
                        "experiment": archive.experiment.get("name"),
                        "run": archive.run,
                        "question": trial.question["id"],
                        "character": trial.character,
                        "coefficient": trial.coefficient,
                        "scheme": trial.scheme,
                        "layer": layer.layer,
                        "source": layer.source,
                        "set": item.id,
                        "position": position,
                        "segment": token.get("segment"),
                        "token": token.get("text"),
                        **{
                            name: item.measures.get(name, ())[position]
                            if position < len(item.measures.get(name, ()))
                            else None
                            for name in MEASURES
                        },
                    }


def write_table(path: Path, rows: Iterator[dict[str, Any]]) -> Path:
    """The table view that keeps the plots readable without their colours."""

    columns = [
        "experiment", "run", "question", "character", "coefficient", "scheme",
        "layer", "source", "set", "position", "segment", "token", *MEASURES,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path
