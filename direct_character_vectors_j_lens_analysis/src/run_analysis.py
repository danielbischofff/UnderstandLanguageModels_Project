"""J-Lens readout of Qwen3-32B persona directions, layer by layer.

For every persona direction and every fitted layer this computes
``unembed(J_l @ v_l)`` and records the highest-scoring vocabulary tokens --
"what this direction, sitting at this layer, is disposed to make the model
say". Results are written as one JSON document for the viewer in ``web/``.

Directions
----------
The published role vectors are *raw mean activations* (the mean residual over
samples a judge scored as fully in-role), not difference vectors. A raw mean is
dominated by the layer-generic component every persona shares, so the headline
direction here is the contrast ``bard - default``: what is distinctively bard
rather than what is merely typical of the residual stream. The raw vector is
reported alongside it, plus three controls -- the default (Assistant) vector,
the published assistant axis, and a norm-matched random direction.

Modes
-----
``jlens`` applies the Jacobian transport; ``logit_lens`` skips it and unembeds
the direction directly. The comparison isolates what the lens contributes over
a plain logit-lens reading. Only ``jlens`` is limited to the 63 fitted layers.

Usage::

    python src/run_analysis.py [--top-k 20] [--out out/readout.json]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from fetch_data import fetch_role
from readout import LensTransport, load_vector
from unembed import Unembedding, top_k

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LENS_PATH = DATA / "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
ROLE = "bard"


@dataclass(frozen=True)
class Direction:
    """A per-layer direction ``[n_layers, d_model]`` with a note on its meaning."""

    vectors: np.ndarray
    label: str
    description: str
    primary: bool = False


def build_directions(role: str = ROLE, seed: int = 0) -> dict[str, Direction]:
    """Assemble the role direction, its contrast, and the control directions."""
    vectors = DATA / "vectors"
    role_vector = load_vector(fetch_role(role))
    default = load_vector(vectors / "default_vector.pt")
    axis = load_vector(vectors / "assistant_axis.pt")
    contrast = role_vector - default

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(contrast.shape).astype(np.float32)
    noise *= (np.linalg.norm(contrast, axis=1, keepdims=True)
              / np.linalg.norm(noise, axis=1, keepdims=True))

    return {
        f"{role}_minus_default": Direction(
            contrast, f"{role} − default",
            f"What is distinctively {role}: the role's mean activation with the "
            "default Assistant mean subtracted. This is the steering direction.",
            primary=True),
        role: Direction(
            role_vector, role,
            "The published role vector as-is: the mean residual activation "
            f"while the model plays a {role}, per layer."),
        "default": Direction(
            default, "default (Assistant)",
            "Control. Mean activation of the default Assistant persona."),
        "assistant_axis": Direction(
            axis, "assistant axis",
            "Control. The published axis, default_mean − role_mean, pointing "
            "from role-play toward the Assistant."),
        "random_matched": Direction(
            noise, "random (norm-matched)",
            f"Null control. Gaussian noise, seed {seed}, rescaled per layer to "
            f"match the norm of the {role} − default contrast."),
    }


def decode_vocab(tokenizer: Tokenizer, vocab_size: int) -> tuple[list[str], np.ndarray]:
    """Decode every token id, and mark the word-like ones.

    The mask mirrors ``jlens.vis._meaningful_token_mask``: non-empty, not a
    special token, and alphanumeric apart from interior hyphens/apostrophes.
    Masking only affects which tokens are *displayed*; ranks and probabilities
    are always computed over the full vocabulary.
    """
    texts: list[str] = []
    wordlike = np.zeros(vocab_size, dtype=bool)
    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            decoded = ""
        texts.append(decoded)
        stripped = decoded.strip()
        if not stripped or "<|" in stripped:
            continue
        if stripped.startswith("<") and stripped.endswith(">"):
            continue
        wordlike[token_id] = all(
            ch.isalnum() or (0 < pos < len(stripped) - 1 and ch in "'-’")
            for pos, ch in enumerate(stripped)
        )
    return texts, wordlike


def entropy(logits: np.ndarray, block: int = 64) -> np.ndarray:
    """Shannon entropy (nats) of each row's full-vocabulary softmax.

    Worked in row blocks: the logit matrix is a few hundred MB and the naive
    expression would trip over several temporaries of that size.
    """
    out = np.empty(logits.shape[0], dtype=np.float64)
    for lo in range(0, logits.shape[0], block):
        rows = logits[lo:lo + block].astype(np.float64)
        rows -= rows.max(axis=1, keepdims=True)
        weights = np.exp(rows)
        total = weights.sum(axis=1)
        out[lo:lo + block] = np.log(total) - (weights * rows).sum(axis=1) / total
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", default=ROLE, help="role vector to read out")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "readout.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    directions = build_directions(args.role, args.seed)
    names = list(directions)
    stacked = np.stack([directions[n].vectors for n in names])  # [n_dir, L, d]
    n_layers = stacked.shape[1]

    lens = LensTransport(LENS_PATH)
    unembedding = Unembedding(DATA)
    print(lens, unembedding, sep="\n", flush=True)

    # One row per (mode, direction, layer): the transported readouts first,
    # then the untransported logit-lens baseline over every layer.
    rows: list[tuple[str, str, int]] = []
    residuals: list[np.ndarray] = []
    for layer in lens.source_layers:
        residuals.append(lens.transport(layer, stacked[:, layer, :]))
        rows += [("jlens", name, layer) for name in names]
        if layer % 10 == 0:
            print(f"  transported layer {layer}/{lens.source_layers[-1]}", flush=True)
    for layer in range(n_layers):
        residuals.append(stacked[:, layer, :])
        rows += [("logit_lens", name, layer) for name in names]

    matrix = np.concatenate(residuals, axis=0)
    print(f"unembedding {matrix.shape[0]} readouts "
          f"({time.time() - started:.0f}s elapsed)", flush=True)
    logits = unembedding.logits(matrix)

    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    texts, wordlike = decode_vocab(tokenizer, unembedding.vocab_size)
    print(f"word-like tokens: {int(wordlike.sum())}/{unembedding.vocab_size}",
          flush=True)

    ids_all, logit_all, prob_all = top_k(logits, args.top_k)
    ids_word, logit_word, prob_word = top_k(logits, args.top_k, allowed=wordlike)
    entropies = entropy(logits)

    def pack(i: int, ids, scores, probs) -> list[dict]:
        return [{"id": int(t), "text": texts[int(t)],
                 "logit": round(float(s), 4), "prob": float(f"{p:.6g}")}
                for t, s, p in zip(ids[i], scores[i], probs[i])]

    readouts = [
        {"mode": mode, "direction": name, "layer": layer,
         "entropy": round(float(entropies[i]), 4),
         "top": pack(i, ids_all, logit_all, prob_all),
         "top_wordlike": pack(i, ids_word, logit_word, prob_word)}
        for i, (mode, name, layer) in enumerate(rows)
    ]

    document = {
        "meta": {
            "role": args.role,
            "model": "Qwen/Qwen3-32B",
            "n_layers": n_layers,
            "jlens_layers": lens.source_layers,
            "lens_prompts_fitted": lens.n_prompts,
            "lens_source": "neuronpedia/jacobian-lens "
                           "(qwen3-32b/jlens/Salesforce-wikitext)",
            "vector_source": "lu-christina/assistant-axis-vectors (qwen-3-32b)",
            "top_k": args.top_k,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "directions": {
            n: {"label": d.label, "description": d.description, "primary": d.primary}
            for n, d in directions.items()
        },
        "readouts": readouts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
