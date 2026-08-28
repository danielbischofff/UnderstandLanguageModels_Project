"""Read Qwen3-32B's refusal direction through the Jacobian lens, layer by layer.

Same readout as :mod:`run_analysis` -- ``unembed(J_l @ v_l)`` -- applied to a
refusal direction instead of a persona, so the trajectory can be compared
against the character work directly.

Where the direction comes from
------------------------------
``thienkhoi01/qwen3-refusal-direction`` publishes a difference-of-means refusal
direction for Qwen3-32B in the style of Arditi et al. (arXiv 2406.11717). The
upload needs some care, and this module documents what it actually contains:

* the README states dimension 6656; Qwen3-32B's residual stream is **5120**,
  which is what the tensor really is;
* the README describes a single vector, but the 21 MB file is a ``[5120]``
  *view* at storage offset ``46 * 5120`` into a ``[512, 5120]`` float64
  storage. ``torch.save`` keeps the whole backing storage, so the file
  silently carries the entire candidate grid;
* 512 = 8 positions x 64 layers. The axis order is recoverable from the norms:
  read as ``[8, 64, 5120]`` the per-layer norm rises with depth (0 -> 806,
  Spearman 0.99), the profile a residual stream has; read as ``[64, 8, 5120]``
  it is unordered (0.08). The published selection (``layer 46``, ``pos -8``) is
  then row 46 = position index 0, layer 46, exactly the saved view.

Provenance is third-party and unverified. The readout is itself the evidence:
a genuine refusal direction should verbalize refusal, and a mislabelled one
should not.

Sign convention is not documented either, so both ``+r`` and ``-r`` are read
out; only one of them is refusal. A norm-matched Gaussian direction is included
as the noise floor.

Usage::

    python src/refusal_readout.py [--top-k 25]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from readout import LensTransport
from run_analysis import DATA, LENS_PATH, ROOT, decode_vocab, entropy
from unembed import Unembedding, top_k

DIRECTION_FILE = DATA / "refusal_direction.pt"
N_POSITIONS, N_LAYERS, D_MODEL = 8, 64, 5120
#: Position index 0 is the ``-8`` of the metadata; indices run to the last token.
POSITION_LABELS = [f"-{N_POSITIONS - i}" for i in range(N_POSITIONS)]
SELECTED = {"layer": 46, "position": "-8"}


def load_grid(path: Path = DIRECTION_FILE) -> np.ndarray:
    """Recover the full ``[positions, layers, d_model]`` candidate grid.

    The saved object is only a view; the grid lives in its backing storage.
    """
    view = torch.load(path, map_location="cpu", weights_only=True)
    storage = view.untyped_storage()
    flat = torch.empty(0, dtype=torch.float64)
    flat.set_(storage, 0, (storage.nbytes() // 8,))
    grid = flat.reshape(N_POSITIONS, N_LAYERS, D_MODEL).numpy().astype(np.float32)

    row = view.storage_offset() // D_MODEL
    expected = grid[row // N_LAYERS, row % N_LAYERS]
    if not np.allclose(expected, view.numpy().astype(np.float32)):
        raise ValueError("grid does not reproduce the saved view; axis order wrong?")
    # Read correctly, the per-layer norm rises with depth (Spearman 0.99);
    # read transposed it is unordered (0.08). Rank correlation rather than
    # strict monotonicity: the dip around layers 30-39 is a real feature.
    norms = np.linalg.norm(grid, axis=2).mean(axis=0)
    rank = lambda a: np.argsort(np.argsort(a))
    if np.corrcoef(rank(np.arange(N_LAYERS)), rank(norms))[0, 1] < 0.9:
        raise ValueError("per-layer norms do not rise with depth; axis order wrong?")
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "refusal.json")
    args = parser.parse_args()

    started = time.time()
    grid = load_grid()
    lens = LensTransport(LENS_PATH)
    unembedding = Unembedding(DATA)
    print(lens, unembedding, f"grid {grid.shape}", sep="\n", flush=True)

    rng = np.random.default_rng(args.seed)
    reference = grid[0]
    noise = rng.standard_normal(reference.shape).astype(np.float32)
    noise *= (np.linalg.norm(reference, axis=1, keepdims=True)
              / np.linalg.norm(noise, axis=1, keepdims=True))

    series: dict[tuple[str, str], np.ndarray] = {}
    for i, label in enumerate(POSITION_LABELS):
        series[("refusal", label)] = grid[i]
        series[("negated", label)] = -grid[i]
    series[("random", SELECTED["position"])] = noise

    keys = list(series)
    rows, residuals = [], []
    for layer in lens.source_layers:
        residuals.append(lens.transport(layer, np.stack([series[k][layer] for k in keys])))
        rows += [(k[0], k[1], layer) for k in keys]
    matrix = np.concatenate(residuals, axis=0)
    print(f"unembedding {matrix.shape[0]} readouts", flush=True)
    logits = unembedding.logits(matrix)

    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    texts, wordlike = decode_vocab(tokenizer, unembedding.vocab_size)
    ids_all, logit_all, prob_all = top_k(logits, args.top_k)
    ids_word, logit_word, prob_word = top_k(logits, args.top_k, allowed=wordlike)
    entropies = entropy(logits)

    def pack(i, ids, scores, probs):
        return [{"id": int(t), "text": texts[int(t)], "logit": round(float(s), 4),
                 "prob": float(f"{p:.6g}")}
                for t, s, p in zip(ids[i], scores[i], probs[i])]

    document = {
        "meta": {
            "model": "Qwen/Qwen3-32B",
            "n_layers": N_LAYERS,
            "jlens_layers": lens.source_layers,
            "lens_prompts_fitted": lens.n_prompts,
            "direction_source": "thienkhoi01/qwen3-refusal-direction",
            "lens_source": "neuronpedia/jacobian-lens "
                           "(qwen3-32b/jlens/Salesforce-wikitext)",
            "selected": SELECTED,
            "positions": POSITION_LABELS,
            "top_k": args.top_k,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "series": {
            "refusal": {"label": "refusal (+r)",
                        "description": "The published difference-of-means direction, "
                                       "as given."},
            "negated": {"label": "negated (-r)",
                        "description": "The same direction with the sign flipped. The "
                                       "convention is undocumented, so only one of the "
                                       "two is refusal."},
            "random": {"label": "random (norm-matched)",
                       "description": "Noise floor: Gaussian, rescaled per layer to the "
                                      "refusal direction's norm."},
        },
        "readouts": [
            {"series": s, "position": p, "layer": l,
             "entropy": round(float(entropies[i]), 4),
             "top": pack(i, ids_all, logit_all, prob_all),
             "top_wordlike": pack(i, ids_word, logit_word, prob_word)}
            for i, (s, p, l) in enumerate(rows)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
