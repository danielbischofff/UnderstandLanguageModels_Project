"""Interactive multi-turn chat with compact direct-steering tags."""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from steering.capture import ActivationCaptureConfig, ActivationCaptureEvent
from steering.inference import DEFAULT_MODEL_ID, InferenceRunner
from steering.interventions import SteeringPlan, SteeringRequest
from steering.j_lens import _analysis_imports
from steering.recipes import DefaultVector, PublishedAssistantAxis, RoleVector, VectorRecipe
from steering.vector_store import VectorStore


TAG_PATTERN = re.compile(
    r"^/(?P<negative>-)?(?P<coefficient>(?:\d+(?:\.\d*)?|\.\d+))?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_-]*)$"
)


@dataclass(frozen=True)
class SteeringTag:
    """One direct vector addition parsed from a leading slash tag."""

    name: str
    coefficient: float

    def display(self) -> str:
        return f"{self.coefficient:+g}*{self.name}"


@dataclass(frozen=True)
class ParsedTurn:
    text: str
    tags: tuple[SteeringTag, ...]
    stop: bool = False


@dataclass(frozen=True)
class LensToken:
    text: str
    probability: float


@dataclass(frozen=True)
class LensRow:
    label: str
    tokens: tuple[LensToken, ...]


@dataclass(frozen=True)
class TurnResult:
    text: str
    rows: tuple[LensRow, ...]
    layer: int
    visible_token_count: int


def parse_turn(value: str) -> ParsedTurn:
    """Parse consecutive steering tags at the beginning of a user message."""

    remaining = value.strip()
    tags: list[SteeringTag] = []
    while remaining.startswith("/"):
        parts = remaining.split(maxsplit=1)
        tag_text = parts[0]
        tail = parts[1] if len(parts) == 2 else ""
        if tag_text == "/stop":
            if tail.strip():
                raise ValueError("/stop must be used on its own")
            return ParsedTurn("", (), stop=True)
        match = TAG_PATTERN.fullmatch(tag_text)
        if match is None:
            raise ValueError(
                f"invalid steering tag {tag_text!r}; expected /bard, /-bard, "
                "or /-2default_vector"
            )
        magnitude = float(match.group("coefficient") or 1.0)
        coefficient = -magnitude if match.group("negative") else magnitude
        tags.append(SteeringTag(match.group("name"), coefficient))
        remaining = tail.lstrip()
    if not remaining:
        raise ValueError("enter a message after the steering tags")
    return ParsedTurn(remaining, tuple(tags))


def recipe_for_tag(name: str) -> VectorRecipe:
    """Resolve controls; other names use the store's role-or-trait lookup."""

    if name in {"default", "default_vector"}:
        return DefaultVector()
    if name == "assistant_axis":
        return PublishedAssistantAxis()
    return RoleVector(name)


def tags_with_defaults(
    tags: tuple[SteeringTag, ...], *, neutral: bool
) -> tuple[SteeringTag, ...]:
    default_tags = (SteeringTag("default_vector", -1.0),) if neutral else ()
    return (*default_tags, *tags)


def plan_for_tags(
    tags: tuple[SteeringTag, ...], layer: int, *, neutral: bool = False
) -> SteeringPlan | None:
    effective_tags = tags_with_defaults(tags, neutral=neutral)
    if not effective_tags:
        return None
    return SteeringPlan(
        additive=[
            SteeringRequest(
                vector=recipe_for_tag(tag.name),
                coefficient=tag.coefficient,
                layers=(layer,),
                scaling="raw",
                positions="all",
            )
            for tag in effective_tags
        ]
    )


def visible_token_ids(
    generated: list[int], *, limit: int, special_ids: set[int]
) -> list[int]:
    """Drop the hidden look-ahead token and trailing non-text control tokens."""

    visible = list(generated[:limit])
    while visible and visible[-1] in special_ids:
        visible.pop()
    return visible


def completion_residuals(
    events: list[ActivationCaptureEvent],
    token_ids: list[int],
    *,
    layer: int,
) -> torch.Tensor:
    """Align decode-call post-intervention residuals with visible output IDs."""

    candidates = [
        event
        for event in events
        if event.layer == layer
        and event.point == "post_intervention"
        and event.forward_index > 0
    ]
    rows: list[torch.Tensor] = []
    cursor = 0
    for event in candidates:
        if cursor == len(token_ids):
            break
        supplied = event.input_token_ids
        if supplied is not None and supplied[0][-1] != token_ids[cursor]:
            continue
        rows.append(event.tensor[0, -1].to(torch.float32))
        cursor += 1
    if cursor != len(token_ids):
        raise RuntimeError(
            "could not align generated tokens with captured decode activations "
            f"({cursor}/{len(token_ids)} found); cached generation is required"
        )
    if not rows:
        raise RuntimeError("the model produced no visible output tokens to read")
    return torch.stack(rows)


@contextmanager
def _source_imports(source_dir: Path):
    source = str(source_dir.resolve())
    sys.path.insert(0, source)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == source:
            sys.path.pop(0)


def ensure_j_lens_assets(
    *, analysis_source: Path, data_dir: Path, lens_path: Path
) -> None:
    """Fetch only missing tokenizer, unembedding, and fitted-lens inputs."""

    with _source_imports(analysis_source):
        fetch_data = importlib.import_module("fetch_data")
    previous_data = fetch_data.DATA
    fetch_data.DATA = data_dir
    try:
        if not (data_dir / "tokenizer.json").is_file():
            fetch_data.fetch_tokenizer()
        unembed_files = (
            data_dir / "unembed/meta.json",
            data_dir / "unembed/lm_head.weight.bf16.bin",
            data_dir / "unembed/model.norm.weight.bf16.bin",
        )
        if not all(path.is_file() for path in unembed_files):
            fetch_data.fetch_unembedding()
        if not lens_path.is_file():
            fetch_data.fetch_lens()
    finally:
        fetch_data.DATA = previous_data


class JLensReader:
    """Session-scoped adapter around the repository's existing J-Lens code."""

    def __init__(
        self,
        *,
        analysis_source: Path,
        data_dir: Path,
        lens_path: Path,
        top_k: int,
    ) -> None:
        with _analysis_imports(analysis_source) as (
            readout_module,
            unembed_module,
            analysis_module,
        ):
            self.lens = readout_module.LensTransport(lens_path)
            self.unembedding = unembed_module.Unembedding(data_dir)
            tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
            self.texts, _ = analysis_module.decode_vocab(
                tokenizer, self.unembedding.vocab_size
            )
            self._top_k = unembed_module.top_k
        self.top_k = top_k

    def read(self, residuals: torch.Tensor, *, layer: int) -> tuple[LensRow, ...]:
        if layer not in self.lens.source_layers:
            raise ValueError(
                f"J-Lens has no fitted Jacobian for layer {layer}; "
                f"available={self.lens.source_layers}"
            )
        values = residuals.numpy()
        matrix = np.stack((values.mean(axis=0), values[0], values[-1])).astype(
            np.float32, copy=False
        )
        transported = self.lens.transport(layer, matrix)
        logits = self.unembedding.logits(transported)
        ids, _, probabilities = self._top_k(logits, self.top_k)
        labels = ("average", "first", "last")
        return tuple(
            LensRow(
                label,
                tuple(
                    LensToken(self.texts[int(token_id)], float(probability))
                    for token_id, probability in zip(ids[index], probabilities[index])
                ),
            )
            for index, label in enumerate(labels)
        )


class ChatSession:
    def __init__(
        self,
        runner: InferenceRunner,
        *,
        reader: JLensReader | None,
        max_new_tokens: int,
        seed: int,
        neutral: bool = False,
    ) -> None:
        self.runner = runner
        self.reader = reader
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.neutral = neutral
        self.history: list[dict[str, str]] = []
        self.turn_index = 0
        self.layer = int(runner.model.config.num_hidden_layers) // 2

    def respond(self, turn: ParsedTurn) -> TurnResult:
        plan = plan_for_tags(turn.tags, self.layer, neutral=self.neutral)
        conversation = [*self.history, {"role": "user", "content": turn.text}]
        capture = (
            ActivationCaptureConfig(
                [self.layer], points=["post_intervention"], positions="last"
            )
            if self.reader is not None
            else None
        )
        # One look-ahead token ensures that the final visible token is supplied
        # to a decode call and therefore has an exact selected-layer activation.
        generation_limit = self.max_new_tokens + (1 if self.reader else 0)
        response = self.runner.generate(
            conversation,
            steering=plan,
            seed=self.seed + self.turn_index,
            activation_capture=capture,
            max_new_tokens=generation_limit,
        )
        ids = visible_token_ids(
            response.generated_token_ids,
            limit=self.max_new_tokens,
            special_ids=set(getattr(self.runner.tokenizer, "all_special_ids", ())),
        )
        text = self.runner.tokenizer.decode(ids, skip_special_tokens=True)
        rows: tuple[LensRow, ...] = ()
        if self.reader is not None and ids:
            residuals = completion_residuals(
                response.activations, ids, layer=self.layer
            )
            rows = self.reader.read(residuals, layer=self.layer)
        self.history = [
            *conversation,
            {"role": "assistant", "content": text},
        ]
        self.turn_index += 1
        return TurnResult(text, rows, self.layer, len(ids))


def _render_row(row: LensRow) -> str:
    tokens = ", ".join(
        f"{token.text!r} ({token.probability:.3g})" for token in row.tokens
    )
    return f"J-Lens {row.label}: {tokens}"


def _run_turn(session: ChatSession, turn: ParsedTurn) -> None:
    effective_tags = tags_with_defaults(turn.tags, neutral=session.neutral)
    setting = ", ".join(tag.display() for tag in effective_tags) or "unsteered"
    print(f"steering layer {session.layer}: {setting}", flush=True)
    result = session.respond(turn)
    print(f"\nassistant> {result.text}")
    for row in result.rows:
        print(_render_row(row))
    print()


def _dtype(value: str) -> torch.dtype | str:
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[3]
    analysis_root = repository / "direct_character_vectors_j_lens_analysis"
    runtime = Path(os.environ.get("ULM_RUNTIME_ROOT", repository / "runtime"))
    data_dir = Path(
        os.environ.get("JLENS_DATA_ROOT", runtime / "models/j-lens/qwen3-32b")
    )
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--model", default=os.environ.get("STEERING_MODEL_ID", DEFAULT_MODEL_ID)
    )
    result.add_argument("--model-revision")
    result.add_argument("--tokenizer-revision")
    result.add_argument("--device-map", default="auto")
    result.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    result.add_argument("--quantization", choices=["4bit", "8bit"])
    result.add_argument("--enable-thinking", action="store_true")
    result.add_argument("--trust-remote-code", action="store_true")
    result.add_argument("--vector-revision", default="main")
    result.add_argument("--vector-local-dir")
    result.add_argument("--vector-cache-dir")
    result.add_argument("--dataset-model-dir")
    result.add_argument("--offline", action="store_true")
    result.add_argument(
        "-neutral",
        "--neutral",
        action="store_true",
        help="subtract the default vector on every turn",
    )
    result.add_argument("--max-new-tokens", type=int, default=300)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--top-k", type=int, default=10)
    result.add_argument("--no-j-lens", action="store_true")
    result.add_argument("--analysis-root", type=Path, default=analysis_root)
    result.add_argument("--j-lens-data", type=Path, default=data_dir)
    result.add_argument("--lens", type=Path)
    result.add_argument(
        "initial",
        nargs=argparse.REMAINDER,
        help="optional initial tagged message, for example /bard Who are you?",
    )
    return result


def _validate_args(args: argparse.Namespace, parser_: argparse.ArgumentParser) -> None:
    if args.max_new_tokens <= 0:
        parser_.error("--max-new-tokens must be positive")
    if args.top_k <= 0:
        parser_.error("--top-k must be positive")
    if not args.no_j_lens and args.model not in {
        "Qwen/Qwen3-32B-FP8",
        "Qwen/Qwen3-32B",
    }:
        parser_.error("the bundled J-Lens supports only Qwen3-32B; use --no-j-lens")


def main() -> None:
    parser_ = parser()
    args = parser_.parse_args()
    _validate_args(args, parser_)
    lens_path = args.lens or (
        args.j_lens_data
        / "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
    )
    if not args.no_j_lens and args.lens is not None and not lens_path.is_file():
        parser_.error(f"custom J-Lens file does not exist: {lens_path}")
    reader: JLensReader | None = None
    runner: InferenceRunner | None = None
    session: ChatSession | None = None
    try:
        if not args.no_j_lens:
            ensure_j_lens_assets(
                analysis_source=args.analysis_root / "src",
                data_dir=args.j_lens_data,
                lens_path=lens_path,
            )
            reader = JLensReader(
                analysis_source=args.analysis_root / "src",
                data_dir=args.j_lens_data,
                lens_path=lens_path,
                top_k=args.top_k,
            )
        store = VectorStore(
            model_id=args.model,
            revision=args.vector_revision,
            dataset_model_dir=args.dataset_model_dir,
            local_dir=args.vector_local_dir,
            cache_dir=args.vector_cache_dir,
            offline=args.offline,
        )
        runner = InferenceRunner.from_pretrained(
            args.model,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            device_map=args.device_map,
            dtype=_dtype(args.dtype),
            quantization=args.quantization,
            enable_thinking=args.enable_thinking,
            trust_remote_code=args.trust_remote_code,
            vector_store=store,
        )
        session = ChatSession(
            runner,
            reader=reader,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            neutral=args.neutral,
        )
        print(
            "Enter leading tags such as /bard, /-bard, or /5bard. "
            "Tags apply to one turn; -neutral subtracts default_vector on every "
            "turn; /stop exits."
        )
        pending = " ".join(args.initial).strip()
        while True:
            try:
                raw = pending or input("you> ")
                pending = ""
                turn = parse_turn(raw)
                if turn.stop:
                    break
                _run_turn(session, turn)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            except (RuntimeError, ValueError) as error:
                print(f"error: {error}", file=sys.stderr)
    finally:
        # Dropping the runner releases the sharded model; empty_cache returns
        # allocator-held CUDA memory immediately for a subsequent process.
        session = None
        runner = None
        reader = None
        gc.collect()
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except RuntimeError:
                    continue
        print("Model unloaded; accelerator caches cleared.")


if __name__ == "__main__":
    main()
