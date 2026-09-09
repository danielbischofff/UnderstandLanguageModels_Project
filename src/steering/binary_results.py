"""Versioned, lossless binary readouts with shared token text and offline glosses."""

from __future__ import annotations

import base64
import gzip
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np
import yaml

from .glossary import report_glossary


def encode_trial(trial: Mapping[str, Any], vocabulary: dict[str, str]) -> tuple[dict, bytes]:
    """Flatten ragged top-k lists into little-endian arrays, without rounding."""
    ids, logits, probabilities, offsets = [], [], [], [0]
    lens = trial["j_lens"]
    for layer in lens["layers"]:
        for words in layer["positions"]:
            for word in words:
                if set(word) != {"id", "text", "logit", "probability"}:
                    raise ValueError("unexpected readout fields; refusing to discard data")
                token_id = word["id"]
                if type(token_id) is not int or not 0 <= token_id < 2**32:
                    raise ValueError("token IDs must fit uint32")
                key = str(token_id)
                if key in vocabulary and vocabulary[key] != word["text"]:
                    raise ValueError(f"conflicting text for token ID {token_id}")
                vocabulary[key] = word["text"]
                ids.append(token_id)
                logits.append(word["logit"])
                probabilities.append(word["probability"])
            offsets.append(len(ids))
    if len(ids) >= 2**32:
        raise ValueError("trial exceeds uint32 indexing capacity")
    arrays = [np.asarray(ids, dtype="<u4"), np.asarray(offsets, dtype="<u4")]
    for values in (logits, probabilities):
        full = np.asarray(values, dtype="<f8")
        if not np.isfinite(full).all():
            raise ValueError("readout scores must be finite")
        with np.errstate(over="ignore", under="ignore"):
            small = full.astype("<f4")
        arrays.append(small if np.array_equal(small.astype("<f8"), full) else full)
    layout, chunks, offset = {}, [], 0
    for name, array in zip(("ids", "offsets", "logits", "probabilities"), arrays):
        raw = array.tobytes()
        layout[name] = {"dtype": array.dtype.str, "offset": offset, "count": len(array)}
        chunks.append(raw)
        offset += len(raw)
    metadata = {**trial, "j_lens": {**lens, "layers": [
        {k: v for k, v in layer.items() if k != "positions"} for layer in lens["layers"]
    ]}}
    descriptor = {"metadata": metadata, "arrays": layout,
                  "position_counts": [len(layer["positions"]) for layer in lens["layers"]]}
    return descriptor, gzip.compress(b"".join(chunks), compresslevel=6, mtime=0)


def decode_trial(descriptor: Mapping[str, Any], payload: bytes, vocabulary: Mapping[str, str]) -> dict:
    """Read one trial for analysis/tests; the browser uses the same array layout."""
    raw = gzip.decompress(payload)
    arrays = {name: np.frombuffer(raw, dtype=item["dtype"], count=item["count"], offset=item["offset"])
              for name, item in descriptor["arrays"].items()}
    offsets = arrays["offsets"]
    if len(offsets) != sum(descriptor["position_counts"]) + 1 or offsets[0] != 0 or offsets[-1] != len(arrays["ids"]) or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid binary readout offsets")
    metadata = descriptor["metadata"]
    layers, position = [], 0
    for layer, count in zip(metadata["j_lens"]["layers"], descriptor["position_counts"]):
        positions = []
        for _ in range(count):
            positions.append([{"id": int(arrays["ids"][i]), "text": vocabulary[str(arrays["ids"][i])],
                               "logit": float(arrays["logits"][i]), "probability": float(arrays["probabilities"][i])}
                              for i in range(int(offsets[position]), int(offsets[position + 1]))])
            position += 1
        layers.append({**layer, "positions": positions})
    return {**metadata, "j_lens": {**metadata["j_lens"], "layers": layers}}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_result_bundle(run: Path, target: Path) -> None:
    """Export canonical binary results; retain originals and archive provenance."""
    paths = sorted((run / "trials").glob("*/result.json"))
    if not paths:
        raise ValueError("no per-trial results to export")
    config = yaml.safe_load((run / "resolved-config.yaml").read_text(encoding="utf-8"))
    index = {"format": "steering-binary-v1", "document": {"schema_version": 1, "experiment": config}, "trials": []}
    vocabulary, texts = {}, set()
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for number, source in enumerate(paths):
            trial = json.loads(source.read_text(encoding="utf-8"))
            descriptor, payload = encode_trial(trial, vocabulary)
            filename = f"trials/{number:04d}.bin.gz"
            archive.writestr(filename, payload, compress_type=zipfile.ZIP_STORED)
            index["trials"].append({**descriptor, "payload": filename})
            texts.update(token["text"] for token in trial["j_lens"]["tokens"])
        texts.update(vocabulary.values())
        glossary = report_glossary({"trials": [{"j_lens": {"tokens": [{"text": text} for text in texts], "layers": []}}]})
        archive.writestr("index.json", _json(index))
        archive.writestr("vocabulary.json", _json(vocabulary))
        archive.writestr("glossary.json", _json(glossary))
        archive.writestr("glossary-attribution.md", files("steering").joinpath("data/README.md").read_text(encoding="utf-8"))
        for source in sorted(run.rglob("*")):
            if source.is_file() and source.resolve() != target.resolve() and source.suffix not in {".html", ".zip"} and source.name not in {"experiment-results.json", "result.json"} and not source.name.startswith("."):
                archive.write(source, "provenance/" + source.relative_to(run).as_posix())


def load_bundle_report(bundle: Path, question_id: str | None = None) -> tuple[dict, dict, dict]:
    """Prepare an offline report directly from a ZIP, without extracting files."""
    with zipfile.ZipFile(bundle) as archive:
        index = json.loads(archive.read("index.json"))
        if index["format"] != "steering-binary-v1":
            raise ValueError("unsupported binary results format")
        trials = []
        for descriptor in index["trials"]:
            meta = descriptor["metadata"]
            if question_id is not None and meta["question"]["id"] != question_id:
                continue
            trials.append({**{key: meta.get(key) for key in ("question", "character", "steering")},
                           "binary": {k: v for k, v in descriptor.items() if k != "payload"},
                           "payload_gzip": base64.b64encode(archive.read(descriptor["payload"])).decode("ascii")})
        vocabulary = json.loads(archive.read("vocabulary.json"))
        document = {**index["document"], "trials": trials, "vocabulary": vocabulary}
        if question_id is not None:
            document["experiment"] = {**document["experiment"], "questions": [
                q for q in document["experiment"]["questions"] if q["id"] == question_id
            ]}
        return document, vocabulary, json.loads(archive.read("glossary.json"))
