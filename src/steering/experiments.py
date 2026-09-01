"""Question-by-character experiments with immediate J-Lens readout."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from socket import gethostname
from types import TracebackType
from typing import Any, Self

import yaml

from .inference import GenerationResponse, InferenceRunner
from .j_lens import JLenseReader
from .interventions import SteeringRequest
from .recipes import RoleDirection

UNSTEERED = "unsteered"


def canonical_hash(value: Any, length: int = 16) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _git_metadata() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _package_version() -> str | None:
    try:
        return version("understand-language-models-project")
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class Question:
    id: str
    text: str


@dataclass(frozen=True)
class ExperimentConfig:
    """The complete, human-authored definition of an experiment matrix."""

    name: str
    questions: tuple[Question, ...]
    characters: tuple[str, ...]
    coefficient: float
    steering_layers: tuple[int, ...]
    generation: Mapping[str, Any]
    seed: int
    top_k: int
    description: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExperimentConfig:
        allowed = {
            "name",
            "description",
            "questions",
            "characters",
            "steering",
            "generation",
            "seed",
            "j_lens",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown experiment fields: {sorted(unknown)}")
        name = str(value.get("name", "")).strip()
        if not name:
            raise ValueError("experiment name must be nonempty")

        raw_questions = value.get("questions")
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError("questions must be a nonempty list")
        questions: list[Question] = []
        for index, item in enumerate(raw_questions, 1):
            if isinstance(item, str):
                question = Question(f"question-{index:03d}", item.strip())
            elif isinstance(item, Mapping):
                item_unknown = set(item) - {"id", "text"}
                if item_unknown:
                    raise ValueError(
                        f"unknown question fields: {sorted(item_unknown)}"
                    )
                question = Question(
                    str(item.get("id", f"question-{index:03d}")).strip(),
                    str(item.get("text", "")).strip(),
                )
            else:
                raise TypeError("each question must be text or an {id, text} mapping")
            if not question.id or not question.text:
                raise ValueError("question id and text must be nonempty")
            questions.append(question)
        if len({question.id for question in questions}) != len(questions):
            raise ValueError("question ids must be unique")

        raw_characters = value.get("characters")
        if not isinstance(raw_characters, list) or not raw_characters:
            raise ValueError("characters must be a nonempty list")
        characters = tuple(str(item).strip().lower() for item in raw_characters)
        if any(not character for character in characters):
            raise ValueError("character names must be nonempty")
        if len(set(characters)) != len(characters):
            raise ValueError("characters must not contain duplicates")

        steering = value.get("steering", {})
        if not isinstance(steering, Mapping):
            raise TypeError("steering must be a mapping")
        steering_unknown = set(steering) - {"coefficient", "layers"}
        if steering_unknown:
            raise ValueError(f"unknown steering fields: {sorted(steering_unknown)}")
        coefficient = float(steering.get("coefficient", 1.0))
        layers = tuple(int(layer) for layer in steering.get("layers", [32]))
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("steering layers must be nonempty and unique")

        generation = value.get("generation", {"max_new_tokens": 128})
        if not isinstance(generation, Mapping):
            raise TypeError("generation must be a mapping")
        generation = dict(generation)
        generation.setdefault("max_new_tokens", 128)

        j_lens = value.get("j_lens", {})
        if not isinstance(j_lens, Mapping):
            raise TypeError("j_lens must be a mapping")
        j_lens_unknown = set(j_lens) - {"top_k"}
        if j_lens_unknown:
            raise ValueError(f"unknown j_lens fields: {sorted(j_lens_unknown)}")
        top_k = int(j_lens.get("top_k", 10))
        if top_k <= 0:
            raise ValueError("j_lens.top_k must be positive")

        return cls(
            name=name,
            description=(
                str(value["description"]).strip() if value.get("description") else None
            ),
            questions=tuple(questions),
            characters=characters,
            coefficient=coefficient,
            steering_layers=layers,
            generation=generation,
            seed=int(value.get("seed", 0)),
            top_k=top_k,
        )

    def trials(self) -> list[dict[str, Any]]:
        """Expand the question x character matrix in stable row-major order."""
        return [
            {
                "question": {"id": question.id, "text": question.text},
                "character": character,
                "steering": (
                    None
                    if character == UNSTEERED
                    else {
                        "coefficient": self.coefficient,
                        "layers": list(self.steering_layers),
                    }
                ),
                "generation": dict(self.generation),
                "seed": self.seed,
            }
            for question in self.questions
            for character in self.characters
        ]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            **({"description": self.description} if self.description else {}),
            "questions": [
                {"id": question.id, "text": question.text}
                for question in self.questions
            ],
            "characters": list(self.characters),
            "steering": {
                "coefficient": self.coefficient,
                "layers": list(self.steering_layers),
            },
            "generation": dict(self.generation),
            "seed": self.seed,
            "j_lens": {"top_k": self.top_k},
        }


class ExperimentRecorder:
    """Persist compact results and make interrupted matrices resumable."""

    def __init__(self, run_dir: Path, config: ExperimentConfig, *, create: bool) -> None:
        self.run_dir = run_dir
        self.config = config
        self.config_hash = canonical_hash(config.to_mapping())
        self.lock_path = run_dir / ".writer.lock"
        self._lock_handle: Any | None = None
        if create:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "trials").mkdir()
            self._write_manifest({"status": "running", "trials": {}})
            _atomic_text(
                run_dir / "resolved-config.yaml",
                yaml.safe_dump(config.to_mapping(), sort_keys=False),
            )
        elif self.manifest()["config_hash"] != self.config_hash:
            raise ValueError("resume configuration does not match the existing run")

    @classmethod
    def create(
        cls,
        config: ExperimentConfig | Mapping[str, Any],
        *,
        root: str | Path = "outputs/experiments",
    ) -> ExperimentRecorder:
        parsed = config if isinstance(config, ExperimentConfig) else ExperimentConfig.from_mapping(config)
        runs = Path(root) / parsed.name / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        counters = []
        for path in runs.iterdir():
            try:
                counters.append(int(path.name.split("_", 1)[0]))
            except (ValueError, IndexError):
                continue
        counter = max(counters, default=0) + 1
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        config_hash = canonical_hash(parsed.to_mapping())
        return cls(runs / f"{counter:04d}_{timestamp}_{config_hash}", parsed, create=True)

    @classmethod
    def resume(cls, run_dir: str | Path) -> ExperimentRecorder:
        path = Path(run_dir)
        value = yaml.safe_load((path / "resolved-config.yaml").read_text(encoding="utf-8"))
        return cls(path, ExperimentConfig.from_mapping(value), create=False)

    def __enter__(self) -> Self:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - local Windows development
            fcntl = None
        handle = self.lock_path.open("a+", encoding="utf-8")
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError(f"run already has a writer lock: {self.run_dir}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"host={gethostname()} pid={os.getpid()}\n")
        handle.flush()
        self._lock_handle = handle
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        manifest = self.manifest()
        failures = any(
            trial.get("status") != "complete"
            for trial in manifest.get("trials", {}).values()
        )
        manifest["status"] = (
            "interrupted" if exc_type is not None else
            "complete_with_failures" if failures else "complete"
        )
        if exc is not None:
            manifest["last_error"] = f"{type(exc).__name__}: {exc}"
        self._write_manifest(manifest)
        if self._lock_handle is not None:
            try:
                import fcntl
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover
                pass
            self._lock_handle.close()
            self._lock_handle = None
        return False

    def manifest(self) -> dict[str, Any]:
        return json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))

    def _write_manifest(self, value: Mapping[str, Any]) -> None:
        record = {
            "schema_version": 1,
            "config_hash": self.config_hash,
            "git": _git_metadata(),
            **dict(value),
        }
        _atomic_text(
            self.run_dir / "manifest.json",
            json.dumps(record, indent=2, sort_keys=True) + "\n",
        )

    def is_complete(self, trial_hash: str) -> bool:
        return self.manifest().get("trials", {}).get(trial_hash, {}).get("status") == "complete"

    def record_result(
        self,
        index: int,
        trial: Mapping[str, Any],
        response: GenerationResponse,
        j_lens: Mapping[str, Any],
    ) -> None:
        trial_hash = canonical_hash(trial)
        trial_dir = self.run_dir / "trials" / f"{index:05d}_{trial_hash}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": 1,
            "question": trial["question"],
            "character": trial["character"],
            "response": {
                "text": response.text,
                "generated_token_ids": response.generated_token_ids,
            },
            "j_lens": dict(j_lens),
        }
        _atomic_text(
            trial_dir / "result.json",
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        )
        _atomic_text(trial_dir / "response.txt", response.text)
        _atomic_text(
            trial_dir / "metadata.json",
            json.dumps(
                {
                    **response.metadata,
                    "git": _git_metadata(),
                    "package_version": _package_version(),
                    "experiment": {
                        "config_hash": self.config_hash,
                        "trial_hash": trial_hash,
                        "run": self.run_dir.name,
                    },
                    "activation_artifact": None,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ) + "\n",
        )
        _atomic_text(
            trial_dir / "status.json",
            json.dumps({"status": "complete", "trial_hash": trial_hash}, indent=2) + "\n",
        )
        manifest = self.manifest()
        manifest.setdefault("trials", {})[trial_hash] = {
            "index": index,
            "status": "complete",
            "question": trial["question"]["id"],
            "character": trial["character"],
            "path": str(trial_dir.relative_to(self.run_dir)),
        }
        self._write_manifest(manifest)

    def record_failure(self, index: int, trial: Mapping[str, Any], error: Exception) -> None:
        trial_hash = canonical_hash(trial)
        trial_dir = self.run_dir / "trials" / f"{index:05d}_{trial_hash}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        _atomic_text(
            trial_dir / "status.json",
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                indent=2,
            ) + "\n",
        )
        manifest = self.manifest()
        manifest.setdefault("trials", {})[trial_hash] = {
            "index": index,
            "status": "failed",
            "question": trial["question"]["id"],
            "character": trial["character"],
            "path": str(trial_dir.relative_to(self.run_dir)),
        }
        self._write_manifest(manifest)

    def write_outputs(self) -> None:
        manifest = self.manifest()
        rows = [
            {"trial_hash": key, **value}
            for key, value in sorted(manifest.get("trials", {}).items())
        ]
        if rows:
            path = self.run_dir / "summary.csv"
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=sorted({key for row in rows for key in row})
                )
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
        results = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((self.run_dir / "trials").glob("*/result.json"))
        ]
        _atomic_text(
            self.run_dir / "experiment-results.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment": self.config.to_mapping(),
                    "trials": results,
                },
                ensure_ascii=False,
            ) + "\n",
        )


def _steering_for_trial(trial: Mapping[str, Any]) -> SteeringRequest | None:
    character = str(trial["character"])
    if character == UNSTEERED:
        return None
    steering = trial["steering"]
    return SteeringRequest(
        RoleDirection(character),
        coefficient=float(steering["coefficient"]),
        layers=steering["layers"],
        scaling="raw",
        positions="all",
    )


def run_experiment(
    runner: InferenceRunner,
    recorder: ExperimentRecorder,
    lens: JLenseReader,
) -> None:
    """Run the matrix, retaining only generations and compact token readouts."""
    trials = recorder.config.trials()
    failures: list[tuple[int, str, Exception]] = []
    with recorder:
        for index, trial in enumerate(trials):
            trial_hash = canonical_hash(trial)
            if recorder.is_complete(trial_hash):
                continue
            try:
                started_at = datetime.now(UTC)
                started_clock = time.perf_counter()
                steering = _steering_for_trial(trial)
                conversation = [{"role": "user", "content": trial["question"]["text"]}]
                response = runner.generate(
                    conversation,
                    steering=steering,
                    seed=int(trial["seed"]),
                    collect_diagnostics=False,
                    **trial["generation"],
                )
                events = runner.capture_complete_sequence(
                    conversation,
                    response.generated_token_ids,
                    steering=steering,
                    layers=lens.source_layers,
                )
                prompt_length = len(response.metadata["tokens"]["prompt_ids"])
                readout = lens.read(events, prompt_length=prompt_length)
                response.metadata["timing"] = {
                    "started_at": started_at.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": time.perf_counter() - started_clock,
                }
                recorder.record_result(index, trial, response, readout)
                del events
            except Exception as error:  # noqa: BLE001 - persist and continue matrix
                recorder.record_failure(index, trial, error)
                failures.append((index, trial_hash, error))
        recorder.write_outputs()
    if failures:
        details = "; ".join(
            f"trial {index} ({trial_hash}): {type(error).__name__}: {error}"
            for index, trial_hash, error in failures
        )
        raise RuntimeError(
            f"{len(failures)} of {len(trials)} experiment trials failed in "
            f"{recorder.run_dir}; {details}"
        )
