"""Human-readable, resumable steering experiment orchestration."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import subprocess
import time
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import yaml

from .inference import GenerationResponse, InferenceRunner
from .interventions import CappingRequest, SteeringPlan, SteeringRequest
from .recipes import recipe_from_dict


def canonical_hash(value: Any, length: int = 16) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def _set_path(value: dict[str, Any], path: str, replacement: Any) -> None:
    parts = path.split(".")
    current = value
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = replacement


def expand_sweep(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deterministically expand dotted-path Cartesian sweep dimensions."""

    base = deepcopy(dict(config))
    dimensions = base.pop("sweep", {})
    if not dimensions:
        return [base]
    names = sorted(dimensions)
    choices = []
    for name in names:
        values = dimensions[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep dimension {name!r} must be a nonempty list")
        choices.append(values)
    trials = []
    for combination in itertools.product(*choices):
        trial = deepcopy(base)
        for name, value in zip(names, combination):
            _set_path(trial, name, value)
        trials.append(trial)
    return trials


def _threshold(value: Any) -> Any:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return tuple(float(item) for item in value)
    if isinstance(value, dict):
        return {int(key): _threshold(item) for key, item in value.items()}
    return value


def steering_from_config(value: Mapping[str, Any] | None) -> SteeringPlan | None:
    if not value:
        return None
    additions_value = value.get("additive")
    if additions_value is None and "vector" in value:
        additions_value = [value]
    additions = []
    for item in additions_value or []:
        additions.append(
            SteeringRequest(
                vector=recipe_from_dict(item["vector"]),
                coefficient=item.get("coefficient", 1.0),
                layers=item.get("layers"),
                scaling=item.get("scaling", "raw"),
                positions=item.get("positions", "all"),
                calibration=item.get("calibration"),
                preserve_original_activation_norm=item.get(
                    "preserve_original_activation_norm", False
                ),
            )
        )
    caps = []
    for item in value.get("caps", []):
        caps.append(
            CappingRequest(
                direction=recipe_from_dict(item["direction"]),
                bound=item["bound"],
                layers=item["layers"],
                threshold=_threshold(item.get("threshold")),
                quantile=_threshold(item.get("quantile")),
                calibration=item.get("calibration"),
                positions=item.get("positions", "all"),
            )
        )
    return SteeringPlan(additions, caps, value.get("ordering", "add_then_cap"))


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
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
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


class ExperimentRecorder:
    def __init__(
        self, run_dir: Path, config: Mapping[str, Any], *, create: bool
    ) -> None:
        self.run_dir = run_dir
        self.config = deepcopy(dict(config))
        self.config_hash = canonical_hash(self.config)
        self.lock_path = run_dir / ".writer.lock"
        if create:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "trials").mkdir()
            self._write_manifest({"status": "running", "trials": {}})
            _atomic_text(
                run_dir / "resolved-config.yaml",
                yaml.safe_dump(self.config, sort_keys=False),
            )
        else:
            manifest = self.manifest()
            if manifest["config_hash"] != self.config_hash:
                raise ValueError("resume configuration does not match the existing run")

    @classmethod
    def create(
        cls, config: Mapping[str, Any], *, root: str | Path = "outputs/steering"
    ) -> ExperimentRecorder:
        study = str(config.get("study", "default-study"))
        experiment = str(config.get("experiment", "steering-experiment"))
        experiment_dir = Path(root) / study / experiment
        runs = experiment_dir / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        existing = [path for path in runs.iterdir() if path.is_dir()]
        counter = (
            max((int(path.name.split("_", 1)[0]) for path in existing), default=0) + 1
        )
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        config_hash = canonical_hash(config)
        run_dir = runs / f"{counter:04d}_{timestamp}_{config_hash}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        readme = experiment_dir / "README.md"
        if not readme.exists():
            _atomic_text(
                readme,
                f"# {experiment}\n\nStudy: `{study}`\n\nRuns are immutable except when explicitly resumed.\n",
            )
        _atomic_text(
            experiment_dir / "experiment.yaml",
            yaml.safe_dump(dict(config), sort_keys=False),
        )
        return cls(run_dir, config, create=True)

    @classmethod
    def resume(cls, run_dir: str | Path) -> ExperimentRecorder:
        path = Path(run_dir)
        config = yaml.safe_load(
            (path / "resolved-config.yaml").read_text(encoding="utf-8")
        )
        return cls(path, config, create=False)

    def __enter__(self) -> Self:
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
        except FileExistsError as error:
            raise RuntimeError(
                f"run already has a writer lock: {self.run_dir}"
            ) from error
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        manifest = self.manifest()
        has_failures = any(
            trial.get("status") != "complete"
            for trial in manifest.get("trials", {}).values()
        )
        manifest["status"] = (
            "interrupted"
            if exc_type is not None
            else "complete_with_failures"
            if has_failures
            else "complete"
        )
        if exc is not None:
            manifest["last_error"] = f"{type(exc).__name__}: {exc}"
        self._write_manifest(manifest)
        self.lock_path.unlink(missing_ok=True)
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
        return (
            self.manifest().get("trials", {}).get(trial_hash, {}).get("status")
            == "complete"
        )

    def record_result(
        self,
        index: int,
        trial: Mapping[str, Any],
        response: GenerationResponse,
    ) -> None:
        trial_hash = canonical_hash(trial)
        trial_dir = self.run_dir / "trials" / f"{index:05d}_{trial_hash}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        _atomic_text(
            trial_dir / "request.json",
            json.dumps(trial, indent=2, sort_keys=True) + "\n",
        )
        _atomic_text(trial_dir / "response.txt", response.text)
        _atomic_text(
            trial_dir / "result.json",
            json.dumps(
                {
                    "text": response.text,
                    "generated_token_ids": response.generated_token_ids,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
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
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
        )
        _atomic_text(
            trial_dir / "projections.jsonl",
            "".join(
                json.dumps(item, sort_keys=True) + "\n" for item in response.projections
            ),
        )
        _atomic_text(
            trial_dir / "status.json",
            json.dumps({"status": "complete", "trial_hash": trial_hash}, indent=2)
            + "\n",
        )
        manifest = self.manifest()
        manifest.setdefault("trials", {})[trial_hash] = {
            "index": index,
            "status": "complete",
            "path": str(trial_dir.relative_to(self.run_dir)),
        }
        self._write_manifest(manifest)

    def record_failure(
        self, index: int, trial: Mapping[str, Any], error: Exception
    ) -> None:
        trial_hash = canonical_hash(trial)
        trial_dir = self.run_dir / "trials" / f"{index:05d}_{trial_hash}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        _atomic_text(
            trial_dir / "request.json",
            json.dumps(trial, indent=2, sort_keys=True) + "\n",
        )
        _atomic_text(
            trial_dir / "status.json",
            json.dumps(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                indent=2,
            )
            + "\n",
        )
        manifest = self.manifest()
        manifest.setdefault("trials", {})[trial_hash] = {
            "index": index,
            "status": "failed",
            "path": str(trial_dir.relative_to(self.run_dir)),
        }
        self._write_manifest(manifest)

    def write_summary(self) -> None:
        manifest = self.manifest()
        experiment_dir = self.run_dir.parent.parent
        rows = [
            {"run": self.run_dir.name, "trial_hash": key, **value}
            for key, value in sorted(manifest.get("trials", {}).items())
        ]
        if not rows:
            return
        path = experiment_dir / "summary.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=sorted({key for row in rows for key in row})
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def run_experiment(
    runner: InferenceRunner,
    recorder: ExperimentRecorder,
) -> None:
    trials = expand_sweep(recorder.config)
    with recorder:
        for index, trial in enumerate(trials):
            trial_hash = canonical_hash(trial)
            if recorder.is_complete(trial_hash):
                continue
            try:
                started_at = datetime.now(UTC)
                started_clock = time.perf_counter()
                steering = steering_from_config(trial.get("steering"))
                response = runner.generate(
                    trial["conversation"],
                    steering=steering,
                    seed=int(trial.get("seed", 0)),
                    collect_diagnostics=bool(trial.get("collect_diagnostics", True)),
                    **trial.get("generation", {}),
                )
                response.metadata["timing"] = {
                    "started_at": started_at.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": time.perf_counter() - started_clock,
                }
                recorder.record_result(index, trial, response)
            except Exception as error:  # noqa: BLE001 - persist trial failure and continue
                recorder.record_failure(index, trial, error)
        recorder.write_summary()
