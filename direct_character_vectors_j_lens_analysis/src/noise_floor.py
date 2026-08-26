"""How much of a character's signature is signal? A noise floor for the atlas.

The two obvious quality metrics are both confounded, and in opposite directions:

* **Entropy** looks like confidence, but a readout that collapses onto a single
  junk attractor -- a newline, a backtick -- is also very low entropy. Sharp is
  not the same as meaningful.
* **Distinctiveness** (how rare a signature's tokens are across the cast) looks
  like specificity, but a direction that lands on its own private patch of
  tokenizer garbage scores maximally distinctive.

So this module adds a third that neither failure mode can fake:

* **Stability** -- the mean Jaccard overlap between the top-k sets of
  *consecutive* layers across the deep band. A real character signal persists
  over a stretch of layers; a degenerate collapse is layer-local, and noise is
  uncorrelated from one layer to the next.

and calibrates all three against directions that are noise by construction:
Gaussian vectors whose per-layer norm matches the median ``role - default``
contrast, pushed through the identical lens and unembedding.

Usage::

    python src/noise_floor.py [--n-random 24]
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st

import numpy as np
from tokenizers import Tokenizer

from readout import LensTransport, load_vector
from run_analysis import DATA, LENS_PATH, ROOT, decode_vocab, entropy
from run_sweep import ROW_CHUNK, SIGNATURE_BAND, load_cast
from unembed import Unembedding, top_k


def stability(rows: list[list[int]]) -> float:
    """Mean Jaccard overlap of top-k token sets between consecutive layers."""
    scores = []
    for a, b in zip(rows, rows[1:]):
        sa, sb = set(a), set(b)
        scores.append(len(sa & sb) / len(sa | sb))
    return float(st.mean(scores)) if scores else 0.0


def band_rows(entry: dict, layers: list[int], k: int) -> list[list[int]]:
    """The top-k id lists for the layers inside the signature band."""
    return [entry["i"][i * k:(i + 1) * k]
            for i, layer in enumerate(layers) if layer in SIGNATURE_BAND]


def random_floor(n: int, seed: int, k: int) -> tuple[list[float], list[float]]:
    """Run ``n`` norm-matched Gaussian directions through the whole pipeline."""
    roles = sorted(p.stem for p in (DATA / "vectors").glob("*.pt")
                   if p.stem not in {"default_vector", "assistant_axis"})
    cast = load_cast(roles)
    default = load_vector(DATA / "vectors" / "default_vector.pt")
    # Match the typical contrast magnitude, layer by layer.
    target = np.median(np.linalg.norm(cast - default, axis=2), axis=0)
    del cast

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, *default.shape)).astype(np.float32)
    noise *= (target[None, :, None]
              / np.linalg.norm(noise, axis=2, keepdims=True))

    lens = LensTransport(LENS_PATH)
    unembedding = Unembedding(DATA)
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    _texts, wordlike = decode_vocab(tokenizer, unembedding.vocab_size)

    fitted = [i for i, l in enumerate(lens.source_layers) if l in SIGNATURE_BAND]
    transported = np.empty((n, len(lens.source_layers), default.shape[1]), np.float32)
    for i, layer in enumerate(lens.source_layers):
        transported[:, i, :] = lens.transport(layer, noise[:, layer, :])

    flat = transported[:, fitted, :].reshape(-1, default.shape[1])
    ids = np.empty((flat.shape[0], k), np.int32)
    ents = np.empty(flat.shape[0], np.float32)
    for lo in range(0, flat.shape[0], ROW_CHUNK):
        hi = min(lo + ROW_CHUNK, flat.shape[0])
        logits = unembedding.logits(flat[lo:hi])
        ids[lo:hi] = top_k(logits, k, allowed=wordlike)[0]
        ents[lo:hi] = entropy(logits)
        del logits

    per = len(fitted)
    ids = ids.reshape(n, per, k)
    ents = ents.reshape(n, per)
    return ([float(e.min()) for e in ents],
            [stability([list(r) for r in ids[i]]) for i in range(n)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-random", type=int, default=24)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    atlas = json.loads((ROOT / "out" / "atlas.json").read_text(encoding="utf-8"))
    layers, k, vocab = atlas["layers"], atlas["meta"]["top_k"], atlas["vocab"]

    print(f"noise floor: {args.n_random} norm-matched random directions")
    floor_ent, floor_stab = random_floor(args.n_random, args.seed, k)
    fe, fs = st.median(floor_ent), st.median(floor_stab)
    print(f"  random: entropy median {fe:.2f} (min {min(floor_ent):.2f}) | "
          f"stability median {fs:.3f} (max {max(floor_stab):.3f})")
    ceiling = max(floor_stab)

    for ck in atlas["contrasts"]:
        data = atlas["data"][ck]
        roles = [r for r in atlas["roles"] if r in data]
        ent = {r: min(data[r]["h"][i] for i, l in enumerate(layers)
                      if l in SIGNATURE_BAND) for r in roles}
        stab = {r: stability(band_rows(data[r], layers, k)) for r in roles}
        clears = [r for r in roles if stab[r] > ceiling]
        print(f"\n===== {ck} =====")
        print(f"  cast stability: median {st.median(stab.values()):.3f}, "
              f"assistant {stab['assistant']:.3f}")
        print(f"  cast entropy:   median {st.median(ent.values()):.2f}, "
              f"assistant {ent['assistant']:.2f}")
        print(f"  above every random direction's stability: "
              f"{len(clears)}/{len(roles)} ({100 * len(clears) / len(roles):.0f}%)")
        ranked = sorted(roles, key=lambda r: -stab[r])
        show = lambda r: f"{r}({stab[r]:.2f})"
        print(f"  most stable:  {', '.join(show(r) for r in ranked[:6])}")
        print(f"  least stable: {', '.join(show(r) for r in ranked[-4:])}")
        print(f"  assistant rank by stability: {ranked.index('assistant') + 1}/{len(roles)}")


if __name__ == "__main__":
    main()
