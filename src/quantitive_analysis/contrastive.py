"""Manually curated contrastive word sets and their J-Lens association measures.

One set names one contrastive idea for a question ("care and concern", "malice",
"condolence"). At a chosen layer and token position the J-Lens produces a
full-vocabulary distribution; this module reduces that state to two independent
per-set numbers, so a generation becomes a timeline per idea:

* **Rank association** - where the set's own words sit in the full-vocabulary
  ranking (median, mean, minimum, mean reciprocal). Lower ranks mean the idea is
  closer to what the state is disposed to say. Ranks need the complete logit
  row, so they are computed where the lens readout happens, not from saved
  top-k words.
* **Cosine association** - how close the salient words the state actually
  surfaces (its top-k) are to the set in unembedding space, weighted by the
  saved probabilities. This needs only the retained top-k plus unembedding rows,
  so it can be recomputed for sets curated after a run.

Ranks are over the full vocabulary, matching the convention of the analysis
code: the word-like mask selects which tokens are *displayed*, not how the
ranking is computed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import yaml

SCHEMA_VERSION = 1

#: Categorical slots assigned in fixed order, never cycled (validated for
#: colour-vision deficiency on a light surface). More than eight contrastive
#: ideas in one plot is not readable; split the question instead.
PALETTE = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
MAX_SETS = len(PALETTE)

#: Association measures, grouped by what they need to be computed.
RANK_MEASURES = ("median_rank", "mean_rank", "min_rank", "mean_reciprocal_rank")
TOP_K_COSINE_MEASURES = ("centroid_cosine", "weighted_max_cosine", "top1_cosine", "max_cosine")
ACTIVATION_COSINE_MEASURES = ("activation_word_cosine",)
COSINE_MEASURES = TOP_K_COSINE_MEASURES + ACTIVATION_COSINE_MEASURES
MEASURES = RANK_MEASURES + COSINE_MEASURES

#: Measures a plot defaults to, one per family.
PRIMARY_COSINE = "centroid_cosine"
PRIMARY_RANK = "median_rank"

#: Casing/spacing forms tried, in order, when mapping a curated word to one
#: vocabulary token. Byte-level BPE gives a word a different token with and
#: without its leading space; the space form is the one a running text produces.
_VARIANT_TEMPLATES = (" {word}", "{word}", " {title}", "{title}")


class Tokenizer(Protocol):
    """The subset of ``tokenizers.Tokenizer`` used to resolve curated words."""

    def encode(self, sequence: str, add_special_tokens: bool = ...) -> Any: ...


#: Gathers raw unembedding rows for token ids as ``[n, d_model]`` float32.
EmbeddingRows = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class WordSpec:
    """One curated word and the surface forms allowed to represent it."""

    word: str
    variants: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: Any) -> WordSpec:
        if isinstance(value, str):
            word = value.strip()
            if not word:
                raise ValueError("contrastive words must be nonempty")
            return cls(word)
        if not isinstance(value, Mapping):
            raise TypeError("each word must be text or a {word, variants} mapping")
        unknown = set(value) - {"word", "variants"}
        if unknown:
            raise ValueError(f"unknown word fields: {sorted(unknown)}")
        word = str(value.get("word", "")).strip()
        if not word:
            raise ValueError("contrastive words must be nonempty")
        raw = value.get("variants", ())
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            raise TypeError("word variants must be a list of surface forms")
        variants = tuple(str(item) for item in raw)
        if any(not item.strip() for item in variants):
            raise ValueError("word variants must be nonempty")
        return cls(word, variants)

    def to_value(self) -> Any:
        if not self.variants:
            return self.word
        return {"word": self.word, "variants": list(self.variants)}

    def surface_forms(self) -> tuple[str, ...]:
        if self.variants:
            return self.variants
        forms = [
            template.format(word=self.word, title=self.word[:1].upper() + self.word[1:])
            for template in _VARIANT_TEMPLATES
        ]
        return tuple(dict.fromkeys(forms))


@dataclass(frozen=True)
class ContrastiveSet:
    """A named idea, defined by the words that would express it."""

    id: str
    words: tuple[WordSpec, ...]
    label: str | None = None
    description: str | None = None
    color: str | None = None
    source: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, source: str | None = None) -> ContrastiveSet:
        if not isinstance(value, Mapping):
            raise TypeError("a contrastive set must be a mapping")
        unknown = set(value) - {"id", "label", "description", "color", "words", "source"}
        if unknown:
            raise ValueError(f"unknown contrastive set fields: {sorted(unknown)}")
        identifier = str(value.get("id", "")).strip()
        if not identifier:
            raise ValueError("contrastive set id must be nonempty")
        raw_words = value.get("words")
        if not isinstance(raw_words, list) or not raw_words:
            raise ValueError(f"contrastive set {identifier!r} must list words")
        words = tuple(WordSpec.from_value(item) for item in raw_words)
        if len({word.word.casefold() for word in words}) != len(words):
            raise ValueError(f"contrastive set {identifier!r} repeats a word")
        color = value.get("color")
        if color is not None:
            color = str(color).strip()
            if not color:
                raise ValueError("contrastive set colour must be nonempty when given")
        return cls(
            id=identifier,
            words=words,
            label=str(value["label"]).strip() if value.get("label") else None,
            description=str(value["description"]).strip() if value.get("description") else None,
            color=color,
            source=str(value.get("source", source)) if value.get("source", source) else None,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the word definition alongside analysis results."""
        value: dict[str, Any] = {"id": self.id}
        for key in ("label", "description", "color", "source"):
            if getattr(self, key):
                value[key] = getattr(self, key)
        value["words"] = [word.to_value() for word in self.words]
        return value

    @property
    def title(self) -> str:
        return self.label or self.id


def load_set_file(path: str | Path) -> ContrastiveSet:
    """Read one curated set from YAML, recording where it came from."""

    target = Path(path)
    value = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"contrastive set file must contain a mapping: {target}")
    value = {**value}
    value.setdefault("id", target.stem)
    return ContrastiveSet.from_mapping(value, source=target.as_posix())


def resolve_set_entries(
    entries: Iterable[Any], *, search_dirs: Sequence[Path] = ()
) -> list[ContrastiveSet]:
    """Accept inline definitions, file paths, and bare names from a set directory."""

    resolved: list[ContrastiveSet] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            resolved.append(ContrastiveSet.from_mapping(entry))
            continue
        if not isinstance(entry, str) or not entry.strip():
            raise TypeError("each contrastive set must be a name, path, or mapping")
        name = entry.strip()
        candidates: list[Path] = []
        if name.endswith((".yaml", ".yml")) or "/" in name:
            candidates.append(Path(name))
        else:
            candidates.extend(
                directory / f"{name}{suffix}"
                for directory in search_dirs
                for suffix in (".yaml", ".yml")
            )
        for candidate in candidates:
            if candidate.is_file():
                value = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    resolved.extend(ContrastiveSet.from_mapping(item, source=candidate.as_posix()) for item in value)
                else:
                    resolved.append(load_set_file(candidate))
                break
        else:
            searched = ", ".join(str(item) for item in candidates) or "no search path"
            raise FileNotFoundError(f"unknown contrastive set {name!r}; tried {searched}")
    return resolved


def merge_set_library(*groups: Iterable[ContrastiveSet]) -> dict[str, ContrastiveSet]:
    """One definition per id, in first-seen order; conflicting reuse is an error."""

    library: dict[str, ContrastiveSet] = {}
    for group in groups:
        for item in group:
            existing = library.get(item.id)
            if existing is not None and existing.to_mapping() != item.to_mapping():
                raise ValueError(
                    f"conflicting definitions for contrastive set {item.id!r} "
                    f"({existing.source or 'inline'} and {item.source or 'inline'})"
                )
            library.setdefault(item.id, item)
    return library


@dataclass(frozen=True)
class ResolvedWord:
    """One curated word mapped onto the single vocabulary token that stands for it."""

    word: str
    token_id: int
    token_text: str
    surface_form: str
    complete: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "token_id": self.token_id,
            "token_text": self.token_text,
            "surface_form": self.surface_form,
            "complete": self.complete,
        }


def resolve_word(tokenizer: Tokenizer, spec: WordSpec) -> ResolvedWord | None:
    """Pick the token that represents a word: a whole-token form when one exists.

    Otherwise the first token of the preferred form stands in for it and is
    marked incomplete, because a longer word only ever surfaces through its
    first piece in a single-position readout.
    """

    fallback: ResolvedWord | None = None
    for form in spec.surface_forms():
        ids = list(tokenizer.encode(form, add_special_tokens=False).ids)
        if not ids:
            continue
        if len(ids) == 1:
            return ResolvedWord(spec.word, int(ids[0]), form, form, True)
        if fallback is None:
            fallback = ResolvedWord(spec.word, int(ids[0]), form, form, False)
    return fallback


@dataclass(frozen=True)
class ResolvedSet:
    """A curated set bound to vocabulary tokens."""

    definition: ContrastiveSet
    words: tuple[ResolvedWord, ...]
    color: str

    @property
    def id(self) -> str:
        return self.definition.id

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.definition.to_mapping(),
            "color": self.color,
            "tokens": [word.to_mapping() for word in self.words],
            "incomplete_words": [
                word.word for word in self.words if not word.complete
            ],
        }


def _round(values: np.ndarray, digits: int = 6) -> list[float | None]:
    """Compact JSON numbers; an undefined position is null, never NaN."""

    return [
        float(f"%.{digits}g" % float(value)) if np.isfinite(value) else None
        for value in np.asarray(values, dtype=np.float64)
    ]


class ContrastiveProbe:
    """Both association measures for one question's sets, at chosen layers.

    The probe owns the mapping from curated words to vocabulary tokens, so the
    lens readout only has to hand it ranks for :attr:`token_ids` plus the top-k
    it already computed.
    """

    def __init__(
        self,
        sets: Sequence[ContrastiveSet],
        *,
        tokenizer: Tokenizer,
        embedding_rows: EmbeddingRows,
        layers: Sequence[int],
        top_k: int,
    ) -> None:
        if not sets:
            raise ValueError("a contrastive probe needs at least one set")
        if len(sets) > MAX_SETS:
            raise ValueError(
                f"at most {MAX_SETS} contrastive sets can be distinguished in one "
                f"plot; got {len(sets)}"
            )
        if len({item.id for item in sets}) != len(sets):
            raise ValueError("contrastive set ids must be unique")
        layers = tuple(int(layer) for layer in layers)
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("contrastive layers must be nonempty and unique")
        if top_k <= 0:
            raise ValueError("contrastive top_k must be positive")
        self.layers = layers
        self.top_k = int(top_k)
        self._embedding_rows = embedding_rows

        resolved: list[ResolvedSet] = []
        token_ids: list[int] = []
        index: dict[int, int] = {}
        columns: list[np.ndarray] = []
        for slot, definition in enumerate(sets):
            words = []
            for spec in definition.words:
                word = resolve_word(tokenizer, spec)
                if word is None:
                    raise ValueError(
                        f"contrastive set {definition.id!r} cannot represent {spec.word!r} "
                        "with any vocabulary token"
                    )
                words.append(word)
            if not words:
                raise ValueError(f"contrastive set {definition.id!r} resolved to no tokens")
            resolved.append(
                ResolvedSet(definition, tuple(words), definition.color or PALETTE[slot])
            )
            positions = []
            for word in words:
                if word.token_id not in index:
                    index[word.token_id] = len(token_ids)
                    token_ids.append(word.token_id)
                positions.append(index[word.token_id])
            columns.append(np.asarray(positions, dtype=np.int64))
        self.sets = tuple(resolved)
        #: Vocabulary tokens whose ranks the readout must supply, in this order.
        self.token_ids = tuple(token_ids)
        self._columns = tuple(columns)

        rows = np.asarray(embedding_rows(np.asarray(self.token_ids, dtype=np.int64)), dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] != len(self.token_ids):
            raise ValueError("embedding_rows must return one row per token id")
        self._unit_words = _unit(rows)
        gram = self._unit_words @ self._unit_words.T
        self._centroid_norm = np.asarray(
            [
                float(np.sqrt(max(gram[np.ix_(column, column)].mean(), 0.0)))
                for column in self._columns
            ],
            dtype=np.float32,
        )
        self._dot_cache: dict[int, np.ndarray] = {
            token_id: self._unit_words[position] @ self._unit_words.T
            for token_id, position in index.items()
        }

    @property
    def set_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.sets)

    def describe(self) -> dict[str, Any]:
        """Provenance and word-to-token resolution, saved beside the measures."""

        return {
            "schema_version": SCHEMA_VERSION,
            "layers": list(self.layers),
            "top_k": self.top_k,
            "rank_scope": "full_vocabulary",
            "measures": {"rank": list(RANK_MEASURES), "cosine": list(COSINE_MEASURES)},
            "sets": [item.to_mapping() for item in self.sets],
        }

    def _dots(self, ids: np.ndarray, block: int = 512) -> np.ndarray:
        """Cosines between each given token and every curated word."""

        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        missing = [int(value) for value in dict.fromkeys(ids.tolist()) if int(value) not in self._dot_cache]
        for start in range(0, len(missing), block):
            chunk = np.asarray(missing[start : start + block], dtype=np.int64)
            rows = _unit(np.asarray(self._embedding_rows(chunk), dtype=np.float32))
            dots = rows @ self._unit_words.T
            for offset, token_id in enumerate(chunk.tolist()):
                self._dot_cache[int(token_id)] = dots[offset]
        return np.stack([self._dot_cache[int(value)] for value in ids])

    def cosine_measures(
        self, top_ids: np.ndarray, top_probabilities: np.ndarray
    ) -> dict[str, dict[str, np.ndarray]]:
        """Association of the surfaced top-k words to each set, per position."""

        top_ids = np.asarray(top_ids, dtype=np.int64)
        weights = np.asarray(top_probabilities, dtype=np.float64)
        if top_ids.shape != weights.shape or top_ids.ndim != 2:
            raise ValueError("top ids and probabilities must be matching [positions, k]")
        top_ids = top_ids[:, : self.top_k]
        weights = weights[:, : self.top_k]
        total = weights.sum(axis=1, keepdims=True)
        # Renormalized over the retained words: the measure describes what the
        # state surfaces, not how much mass those words hold.
        weights = np.divide(weights, total, out=np.full_like(weights, np.nan), where=total > 0)
        dots = self._dots(top_ids.reshape(-1)).reshape(*top_ids.shape, len(self.token_ids))
        result: dict[str, dict[str, np.ndarray]] = {}
        for item, column, norm in zip(self.sets, self._columns, self._centroid_norm):
            selected = dots[:, :, column]
            centroid = selected.mean(axis=2) / norm if norm > 0 else np.full(selected.shape[:2], np.nan)
            nearest = selected.max(axis=2)
            result[item.id] = {
                "centroid_cosine": (centroid * weights).sum(axis=1),
                "weighted_max_cosine": (nearest * weights).sum(axis=1),
                "top1_cosine": centroid[:, 0],
                "max_cosine": nearest.max(axis=1),
            }
        return result

    def activation_measures(self, normalized_activations: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """Mean pairwise cosine to set words after transport and final RMSNorm.

        Each word has equal weight; no top-k selection, probability weighting,
        or set-centroid normalization is applied. Zero vectors are undefined.
        """
        activations = np.asarray(normalized_activations, dtype=np.float32)
        if activations.ndim != 2 or activations.shape[1] != self._unit_words.shape[1]:
            raise ValueError("normalized activations must be [positions, d_model]")
        dots = _unit(activations) @ self._unit_words.T
        dots[np.linalg.norm(activations, axis=1) == 0, :] = np.nan
        dots[:, np.linalg.norm(self._unit_words, axis=1) == 0] = np.nan
        return {
            item.id: {"activation_word_cosine": dots[:, column].mean(axis=1)}
            for item, column in zip(self.sets, self._columns)
        }

    def rank_measures(self, ranks: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """Where each set's own words sit in the full-vocabulary ranking."""

        ranks = np.asarray(ranks, dtype=np.float64)
        if ranks.ndim != 2 or ranks.shape[1] != len(self.token_ids):
            raise ValueError(
                f"ranks must be [positions, {len(self.token_ids)}] aligned with token_ids"
            )
        result: dict[str, dict[str, np.ndarray]] = {}
        for item, column in zip(self.sets, self._columns):
            selected = ranks[:, column]
            result[item.id] = {
                "median_rank": np.median(selected, axis=1),
                "mean_rank": selected.mean(axis=1),
                "min_rank": selected.min(axis=1),
                "mean_reciprocal_rank": (1.0 / selected).mean(axis=1),
            }
        return result

    def measure_layer(
        self,
        layer: int,
        *,
        top_ids: np.ndarray,
        top_probabilities: np.ndarray,
        ranks: np.ndarray | None,
        normalized_activations: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Every available measure for one layer, as one series per set."""

        cosine = self.cosine_measures(top_ids, top_probabilities)
        rank = self.rank_measures(ranks) if ranks is not None else {}
        activation = self.activation_measures(normalized_activations) if normalized_activations is not None else {}
        series = []
        for item in self.sets:
            values = {**cosine[item.id], **rank.get(item.id, {}), **activation.get(item.id, {})}
            series.append(
                {
                    "layer": int(layer),
                    "set": item.id,
                    **{name: _round(values[name]) for name in MEASURES if name in values},
                }
            )
        return series


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)


def ranks_from_logits(logits: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    """One-based ranks of ``token_ids`` in each full-vocabulary logit row.

    Ties take the best available rank, so identical logits share a rank instead
    of ordering arbitrarily.
    """

    logits = np.atleast_2d(np.asarray(logits))
    ids = np.asarray(token_ids, dtype=np.int64)
    selected = logits[:, ids]
    ranks = np.empty(selected.shape, dtype=np.int64)
    for row in range(logits.shape[0]):
        ranks[row] = 1 + (logits[row][None, :] > selected[row][:, None]).sum(axis=1)
    return ranks
