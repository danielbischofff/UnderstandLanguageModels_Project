"""Question/character/alpha/scheme experiments with immediate J-Lens readout."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from socket import gethostname
from types import TracebackType
from typing import Any, Self

import yaml
from safetensors.torch import save_file

from .capture import ActivationCaptureConfig, ActivationCaptureEvent
from .inference import GenerationResponse, InferenceRunner
from .j_lens import JLenseReader
from .interventions import CappingRequest, CharacterBandRequest, SteeringRequest
from .recipes import RoleDirection
from .report import write_experiment_report, write_bundle_reports
from .binary_results import write_result_bundle

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
class SteeringScheme:
    """One named set of layer, position, and intervention choices."""

    name: str
    layers: tuple[int, ...]
    method: str = "additive"
    positions: str = "all"
    scaling: str = "raw"
    preserve_original_activation_norm: bool = False
    bound: str = "lower"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SteeringScheme:
        if not isinstance(value, Mapping):
            raise TypeError("each steering scheme must be a mapping")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown steering scheme fields: {sorted(unknown)}")
        name = value.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("steering scheme name must be nonempty")
        layers = value.get("layers", [32])
        if (
            not isinstance(layers, (list, tuple)) or not layers
            or any(type(layer) is not int or layer < 0 for layer in layers)
            or len(set(layers)) != len(layers)
        ):
            raise ValueError("steering layers must be nonempty, unique nonnegative integers")
        scheme = cls(**{**value, "name": name.strip(), "layers": tuple(layers)})
        if scheme.method not in {"additive", "projection_cap", "character_band"}:
            raise ValueError("scheme method must be additive, projection_cap, or character_band")
        if type(scheme.preserve_original_activation_norm) is not bool:
            raise TypeError("preserve_original_activation_norm must be a boolean")
        if scheme.method != "additive":
            if "scaling" in value or "preserve_original_activation_norm" in value:
                raise ValueError("scaling and norm preservation apply only to additive schemes")
        if scheme.method == "projection_cap":
            if scheme.bound not in {"lower", "upper"}:
                raise ValueError("projection_cap bound must be lower or upper")
        elif "bound" in value:
            raise ValueError("bound applies only to projection_cap schemes")
        scheme.request("validation", 0.0)  # Validate without materializing a vector.
        return scheme

    def to_mapping(self) -> dict[str, Any]:
        value = {**asdict(self), "layers": list(self.layers)}
        if self.method == "additive":
            del value["bound"]
        else:
            del value["scaling"]
            del value["preserve_original_activation_norm"]
            if self.method == "character_band":
                del value["bound"]
        return value

    def request(self, character: str, alpha: float) -> SteeringRequest | CappingRequest | CharacterBandRequest:
        if self.method == "character_band":
            return CharacterBandRequest(character, alpha, self.layers, self.positions)
        direction = RoleDirection(character)
        if self.method == "projection_cap":
            return CappingRequest(
                direction, bound=self.bound, layers=self.layers,
                threshold=alpha, positions=self.positions,
            )
        return SteeringRequest(
            direction, coefficient=alpha, layers=self.layers, scaling=self.scaling,
            positions=self.positions,
            preserve_original_activation_norm=self.preserve_original_activation_norm,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """The complete, human-authored definition of an experiment matrix."""

    name: str
    model: Mapping[str, Any]
    vector_revision: str
    questions: tuple[Question, ...]
    characters: tuple[str, ...]
    coefficients: tuple[float, ...]
    schemes: tuple[SteeringScheme, ...]
    generation: Mapping[str, Any]
    seed: int
    top_k: int
    keep_activations: bool
    activation_capture: ActivationCaptureConfig | None = None
    description: str | None = None
    legacy_steering: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExperimentConfig:
        allowed = {
            "name",
            "description",
            "model",
            "vectors",
            "questions",
            "characters",
            "steering",
            "generation",
            "seed",
            "j_lens",
            "outputs",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown experiment fields: {sorted(unknown)}")
        name = str(value.get("name", "")).strip()
        if not name:
            raise ValueError("experiment name must be nonempty")

        raw_model = value.get("model")
        if not isinstance(raw_model, Mapping):
            raise TypeError("model must be a mapping")
        model_unknown = set(raw_model) - {
            "id",
            "dtype",
            "revision",
            "tokenizer_revision",
            "quantization",
            "enable_thinking",
        }
        if model_unknown:
            raise ValueError(f"unknown model fields: {sorted(model_unknown)}")
        model_id = str(raw_model.get("id", "")).strip()
        if not model_id:
            raise ValueError("model.id must be nonempty")
        dtype = str(raw_model.get("dtype", "")).strip()
        if dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("model.dtype must be auto, float32, float16, or bfloat16")
        quantization = raw_model.get("quantization")
        if quantization not in {None, "4bit", "8bit"}:
            raise ValueError("model.quantization must be 4bit or 8bit")
        enable_thinking = raw_model.get("enable_thinking", False)
        if not isinstance(enable_thinking, bool):
            raise TypeError("model.enable_thinking must be a boolean")
        model: dict[str, Any] = {
            "id": model_id,
            "dtype": dtype,
            "enable_thinking": enable_thinking,
        }
        for key in ("revision", "tokenizer_revision"):
            if raw_model.get(key) is not None:
                model[key] = str(raw_model[key]).strip()
                if not model[key]:
                    raise ValueError(f"model.{key} must be nonempty when provided")
        if quantization is not None:
            model["quantization"] = quantization

        vectors = value.get("vectors", {})
        if not isinstance(vectors, Mapping):
            raise TypeError("vectors must be a mapping")
        vector_unknown = set(vectors) - {"revision"}
        if vector_unknown:
            raise ValueError(f"unknown vectors fields: {sorted(vector_unknown)}")
        vector_revision = str(vectors.get("revision", "main")).strip()
        if not vector_revision:
            raise ValueError("vectors.revision must be nonempty")

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
        steering_unknown = set(steering) - {"coefficient", "layers", "coefficients", "schemes"}
        if steering_unknown:
            raise ValueError(f"unknown steering fields: {sorted(steering_unknown)}")
        if "coefficient" in steering and "coefficients" in steering:
            raise ValueError("use coefficient or coefficients, not both")
        if "layers" in steering and "schemes" in steering:
            raise ValueError("put layers inside each scheme when using schemes")
        raw_coefficients = steering.get("coefficients", [steering.get("coefficient", 1.0)])
        if not isinstance(raw_coefficients, list) or not raw_coefficients:
            raise ValueError("steering.coefficients must be a nonempty list")
        if any(isinstance(item, bool) for item in raw_coefficients):
            raise ValueError("steering coefficients must be numbers, not booleans")
        coefficients = tuple(float(item) for item in raw_coefficients)
        if (
            not all(math.isfinite(item) for item in coefficients)
            or len(set(coefficients)) != len(coefficients)
        ):
            raise ValueError("steering coefficients must be finite and unique")
        raw_schemes = steering.get(
            "schemes", [{"name": "additive", "layers": steering.get("layers", [32])}]
        )
        if not isinstance(raw_schemes, list) or not raw_schemes:
            raise ValueError("steering.schemes must be a nonempty list")
        schemes = tuple(SteeringScheme.from_mapping(item) for item in raw_schemes)
        if any(scheme.method == "character_band" for scheme in schemes) and any(alpha < 0 for alpha in coefficients):
            raise ValueError("band half-width fractions must be nonnegative")
        if len({scheme.name for scheme in schemes}) != len(schemes):
            raise ValueError("steering scheme names must be unique")

        generation = value.get("generation", {"max_new_tokens": 128})
        if not isinstance(generation, Mapping):
            raise TypeError("generation must be a mapping")
        generation = dict(generation)
        generation.setdefault("max_new_tokens", 128)
        if generation.get("use_cache") is False and any(
            scheme.positions != "all" for scheme in schemes
        ):
            raise ValueError(
                "position-restricted schemes require generation.use_cache: true for faithful replay"
            )

        j_lens = value.get("j_lens", {})
        if not isinstance(j_lens, Mapping):
            raise TypeError("j_lens must be a mapping")
        j_lens_unknown = set(j_lens) - {"top_k"}
        if j_lens_unknown:
            raise ValueError(f"unknown j_lens fields: {sorted(j_lens_unknown)}")
        top_k = int(j_lens.get("top_k", 10))
        if top_k <= 0:
            raise ValueError("j_lens.top_k must be positive")

        outputs = value.get("outputs", {})
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        output_unknown = set(outputs) - {"keep_activations", "activation_capture"}
        if output_unknown:
            raise ValueError(f"unknown outputs fields: {sorted(output_unknown)}")
        keep_activations = outputs.get("keep_activations", False)
        if not isinstance(keep_activations, bool):
            raise TypeError("outputs.keep_activations must be a boolean")
        activation_capture = None
        if "activation_capture" in outputs:
            capture = outputs["activation_capture"]
            if not keep_activations:
                raise ValueError("activation_capture requires keep_activations: true")
            if not isinstance(capture, Mapping):
                raise TypeError("outputs.activation_capture must be a mapping")
            if set(capture) - {"layers", "points", "positions", "dtype"}:
                raise ValueError("unknown outputs.activation_capture fields")
            layers = capture.get("layers")
            if not isinstance(layers, list) or any(
                type(layer) is not int or layer < 0 for layer in layers
            ):
                raise ValueError("activation_capture.layers must be nonnegative integer indices")
            activation_capture = ActivationCaptureConfig(**capture)
            if generation.get("use_cache") is False:
                raise ValueError("generation activation capture requires use_cache: true")

        return cls(
            name=name,
            model=model,
            vector_revision=vector_revision,
            description=(
                str(value["description"]).strip() if value.get("description") else None
            ),
            questions=tuple(questions),
            characters=characters,
            coefficients=coefficients,
            schemes=schemes,
            legacy_steering=not ({"coefficients", "schemes"} & set(steering)),
            generation=generation,
            seed=int(value.get("seed", 0)),
            top_k=top_k,
            keep_activations=keep_activations,
            activation_capture=activation_capture,
        )

    def trials(self) -> list[dict[str, Any]]:
        """Expand all axes, sharing one unsteered control per question."""
        return [
            {
                "question": {"id": question.id, "text": question.text},
                "character": character,
                "steering": (
                    None
                    if character == UNSTEERED
                    else {
                        **(
                            {"layers": list(scheme.layers)} if self.legacy_steering
                            else scheme.to_mapping()
                        ),
                        "coefficient": alpha,
                    }
                ),
                "generation": dict(self.generation),
                "seed": self.seed,
            }
            for question in self.questions
            for character in self.characters
            for alpha, scheme in (
                [(None, None)] if character == UNSTEERED else
                [(alpha, scheme) for alpha in self.coefficients for scheme in self.schemes]
            )
        ]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            **({"description": self.description} if self.description else {}),
            "model": dict(self.model),
            "vectors": {"revision": self.vector_revision},
            "questions": [
                {"id": question.id, "text": question.text}
                for question in self.questions
            ],
            "characters": list(self.characters),
            "steering": (
                {"coefficient": self.coefficients[0], "layers": list(self.schemes[0].layers)}
                if self.legacy_steering else
                {"coefficients": list(self.coefficients),
                 "schemes": [scheme.to_mapping() for scheme in self.schemes]}
            ),
            "generation": dict(self.generation),
            "seed": self.seed,
            "j_lens": {"top_k": self.top_k},
            "outputs": {
                "keep_activations": self.keep_activations,
                **({"activation_capture": {
                    key: value for key, value in self.activation_capture.to_dict().items()
                    if key in {"layers", "points", "positions", "dtype"}
                }} if self.activation_capture is not None else {}),
            },
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
        activations: Sequence[ActivationCaptureEvent] = (),
    ) -> None:
        trial_hash = canonical_hash(trial)
        trial_dir = self.run_dir / "trials" / f"{index:05d}_{trial_hash}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": 1,
            "question": trial["question"],
            "character": trial["character"],
            "steering": trial["steering"],
            "response": {
                "text": response.text,
                "generated_token_ids": response.generated_token_ids,
            },
            "j_lens": dict(j_lens),
        }
        activation_artifact = self._record_activations(trial_dir, activations)
        _atomic_text(
            trial_dir / "result.json",
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
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
                    "activation_artifact": activation_artifact,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ) + "\n",
        )
        manifest = self.manifest()
        manifest.setdefault("trials", {})[trial_hash] = {
            "index": index,
            "status": "complete",
            "question": trial["question"]["id"],
            "character": trial["character"],
            "steering": trial["steering"],
            "path": str(trial_dir.relative_to(self.run_dir)),
        }
        self._write_manifest(manifest)

    def _record_activations(
        self, trial_dir: Path, events: Sequence[ActivationCaptureEvent]
    ) -> dict[str, Any] | None:
        if not self.config.keep_activations or not events:
            return None
        activation_dir = trial_dir / "activations"
        activation_dir.mkdir(parents=True, exist_ok=True)
        tensors = {
            f"event_{index:06d}": event.tensor.contiguous()
            for index, event in enumerate(events)
        }
        tensor_path = activation_dir / "activations.safetensors"
        temporary = tensor_path.with_name(f".{tensor_path.name}.{os.getpid()}.tmp")
        save_file(tensors, temporary)
        temporary.replace(tensor_path)
        manifest = {
            "schema_version": 1,
            "format": "safetensors",
            "tensor_file": tensor_path.name,
            "source": "generation" if self.config.activation_capture else "causal_replay",
            "coordinate_system": {
                "hook_point": "transformer_block_output",
                "residual_stream": "post_block_post_mlp",
                "layer_indexing": "zero_based",
                "tensor_axes": ["batch", "sequence_position", "hidden"],
            },
            "events": [
                event.manifest_entry(key) for key, event in zip(tensors, events)
            ],
        }
        _atomic_text(
            activation_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        return {
            "manifest": "activations/manifest.json",
            "tensors": "activations/activations.safetensors",
            "event_count": len(events),
        }

    def record_failure(self, index: int, trial: Mapping[str, Any], error: Exception) -> None:
        trial_hash = canonical_hash(trial)
        failure = f"{type(error).__name__}: {error}"
        manifest = self.manifest()
        manifest.setdefault("trials", {})[trial_hash] = {
            "index": index,
            "status": "failed",
            "error": failure,
            "question": trial["question"]["id"],
            "character": trial["character"],
            "steering": trial["steering"],
        }
        self._write_manifest(manifest)

    def write_outputs(self) -> None:
        if not any((self.run_dir / "trials").glob("*/result.json")):
            write_experiment_report(self.run_dir / "report.html", {
                "schema_version": 1, "experiment": self.config.to_mapping(), "trials": [],
            })
            return
        # Preserve per-trial checkpoints for resume, but avoid a second giant
        # JSON document or loading the entire matrix into memory for reporting.
        temporary = self.run_dir / ".results.zip.tmp"
        temporary.unlink(missing_ok=True)
        write_result_bundle(self.run_dir, temporary)
        bundle = self.run_dir / "results.zip"
        temporary.replace(bundle)
        write_bundle_reports(bundle, self.run_dir)


def _steering_for_trial(trial: Mapping[str, Any]) -> SteeringRequest | CappingRequest | CharacterBandRequest | None:
    character = str(trial["character"])
    if character == UNSTEERED:
        return None
    steering = trial["steering"]
    scheme = SteeringScheme.from_mapping({
        "name": "additive",
        **{key: value for key, value in steering.items() if key != "coefficient"},
    })
    return scheme.request(character, float(steering["coefficient"]))


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
                    **({"activation_capture": recorder.config.activation_capture}
                       if recorder.config.activation_capture is not None else {}),
                    **trial["generation"],
                )
                events = runner.capture_complete_sequence(
                    conversation,
                    response.generated_token_ids,
                    steering=steering,
                    layers=lens.source_layers,
                )
                prompt_length = len(response.metadata["tokens"]["prompt_ids"])
                readout = lens.read(
                    events,
                    prompt_length=prompt_length,
                    device_readout=runner.j_lens_top_k,
                )
                response.metadata["timing"] = {
                    "started_at": started_at.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": time.perf_counter() - started_clock,
                }
                retained = response.activations if recorder.config.activation_capture else events
                if recorder.config.activation_capture and not retained:
                    raise ValueError("generation produced no requested activation captures")
                recorder.record_result(index, trial, response, readout, retained)
                del events
                del retained, response
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
