"""Read an experiment run's trials without depending on the inference package.

A run directory is the only input this package needs: `manifest.json` supplies the
trial hashes, character and steering settings, and each trial's `result.json`
supplies the generated text. Nothing here imports `steering`, so grading can run on
a downloaded run with no model, GPU or vector store present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class Trial:
    """One generated response plus the condition that produced it."""

    hash: str
    index: int
    character: str
    question_id: str
    question_text: str
    response: str
    steering: Mapping[str, Any]
    status: str

    @property
    def condition(self) -> str:
        if not self.steering:
            return f"{self.character} (no steering)"
        layers = ",".join(str(layer) for layer in self.steering.get("layers", []))
        return f"{self.character} @ layer {layers}, alpha {self.steering.get('coefficient')}"

    def as_record(self, *, reveal: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "trial": self.hash,
            "index": self.index,
            "question_id": self.question_id,
            "question": self.question_text,
            "response": self.response,
        }
        if reveal:
            record["character"] = self.character
            record["steering"] = dict(self.steering) if self.steering else None
        return record


def load_run(run: Path) -> dict[str, Any]:
    manifest = run / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"not a run directory (no manifest.json): {run}")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["_root"] = run
    return document


def iter_trials(run: Mapping[str, Any], *, question: str | None = None) -> Iterator[Trial]:
    """Yield completed trials in manifest order, reading one result at a time."""
    root: Path = run["_root"]
    entries = sorted(run["trials"].items(), key=lambda item: item[1].get("index", 0))
    for trial_hash, entry in entries:
        if question and entry.get("question") != question:
            continue
        if entry.get("status") != "complete":
            continue
        result_path = root / entry["path"] / "result.json"
        if not result_path.is_file():
            raise SystemExit(f"missing result for trial {trial_hash}: {result_path}")
        # result.json also carries the full J-Lens dump; only the text is wanted.
        result = json.loads(result_path.read_text(encoding="utf-8"))
        yield Trial(
            hash=trial_hash,
            index=entry.get("index", 0),
            character=entry.get("character", "unknown"),
            question_id=entry.get("question", ""),
            question_text=(result.get("question") or {}).get("text", ""),
            response=(result.get("response") or {}).get("text") or "",
            steering=entry.get("steering") or {},
            status=entry.get("status", ""),
        )
