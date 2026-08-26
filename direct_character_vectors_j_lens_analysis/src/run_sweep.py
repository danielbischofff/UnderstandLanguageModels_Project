"""Read every published character vector through the Jacobian lens.

Same readout as :mod:`run_analysis` -- ``unembed(J_l @ v_l)`` at each fitted
layer -- applied to all 275 roles at once, and written as a compact atlas for
the viewer in ``web/atlas_template.html``.

Two contrasts
-------------
``vs_default``
    ``role - default``. The steering direction: the character measured against
    the default Assistant. This is what :mod:`run_analysis` reports.
``vs_cast``
    ``role - cast_mean``, where ``cast_mean`` is the mean of all 275 role
    vectors. Subtracting the default removes what the character shares with the
    Assistant, but *not* what it shares with being in character at all. This
    second contrast removes that too, so what is left is specific to this
    character rather than generic to role-play.

Each contrast also carries one anchor direction, which is the other end of the
comparison rather than a character: ``cast_mean - default`` (role-play in
general, as against the Assistant) and ``default - cast_mean`` (the Assistant,
as against the cast). These are negatives of each other, but a lens readout is a
top-k over logits, so the two are not redundant.

Usage::

    python src/run_sweep.py [--top-k 8] [--out out/atlas.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from fetch_data import fetch_all_roles
from readout import LensTransport, load_vector
from run_analysis import DATA, LENS_PATH, ROOT, decode_vocab, entropy
from unembed import Unembedding, top_k

#: Rows unembedded at once. Each row costs ~600 KB of float32 logits, and the
#: masking/softmax steps need a few copies of that, so this bounds peak memory.
ROW_CHUNK = 512
#: Layers aggregated into each character's headline "signature".
SIGNATURE_BAND = range(46, 63)

CONTRASTS = {
    "vs_default": {
        "label": "vs default",
        "suffix": "- default",
        "description": "Measured against the default Assistant. The steering "
                       "direction, as in the single-role analysis.",
        "anchor": "[all characters]",
    },
    "vs_cast": {
        "label": "vs cast",
        "suffix": "- cast mean",
        "description": "Measured against the mean of all 275 characters. Removes "
                       "what is generic to being in character, leaving what is "
                       "specific to this one.",
        "anchor": "[default Assistant]",
    },
}


def load_cast(roles: list[str]) -> np.ndarray:
    """Stack every role vector into ``[n_roles, n_layers, d_model]`` float32."""
    first = load_vector(DATA / "vectors" / f"{roles[0]}.pt")
    cast = np.empty((len(roles), *first.shape), dtype=np.float32)
    cast[0] = first
    for i, role in enumerate(roles[1:], 1):
        cast[i] = load_vector(DATA / "vectors" / f"{role}.pt")
    return cast


def transport_all(lens: LensTransport, directions: np.ndarray) -> np.ndarray:
    """Transport ``[n, n_layers, d]`` directions to ``[n, n_fitted, d]``."""
    out = np.empty((directions.shape[0], len(lens.source_layers),
                    directions.shape[2]), dtype=np.float32)
    for i, layer in enumerate(lens.source_layers):
        out[:, i, :] = lens.transport(layer, directions[:, layer, :])
        if layer % 20 == 0:
            print(f"    layer {layer}/{lens.source_layers[-1]}", flush=True)
    return out


def read_out(unembedding: Unembedding, transported: np.ndarray, k: int,
             wordlike: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k word-like tokens and entropy for every (row, layer) readout.

    Returns ``(ids, probs, entropies)`` shaped ``[n, n_layers, k]``,
    ``[n, n_layers, k]`` and ``[n, n_layers]``.
    """
    n, n_layers, d_model = transported.shape
    flat = transported.reshape(n * n_layers, d_model)
    ids = np.empty((flat.shape[0], k), dtype=np.int32)
    probs = np.empty((flat.shape[0], k), dtype=np.float32)
    ents = np.empty(flat.shape[0], dtype=np.float32)

    for lo in range(0, flat.shape[0], ROW_CHUNK):
        hi = min(lo + ROW_CHUNK, flat.shape[0])
        logits = unembedding.logits(flat[lo:hi])
        chunk_ids, _scores, chunk_probs = top_k(logits, k, allowed=wordlike)
        ids[lo:hi], probs[lo:hi] = chunk_ids, chunk_probs
        ents[lo:hi] = entropy(logits)
        del logits
        print(f"    rows {hi}/{flat.shape[0]}", flush=True)

    return (ids.reshape(n, n_layers, k), probs.reshape(n, n_layers, k),
            ents.reshape(n, n_layers))


def signature(ids: np.ndarray, layers: list[int], k: int = 6) -> list[int]:
    """Rank-weighted vote for a character's headline tokens over the deep band."""
    votes: collections.Counter[int] = collections.Counter()
    for position, layer in enumerate(layers):
        if layer not in SIGNATURE_BAND:
            continue
        for rank, token_id in enumerate(ids[position]):
            votes[int(token_id)] += ids.shape[1] - rank
    return [t for t, _ in votes.most_common(k)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "atlas.json")
    args = parser.parse_args()

    started = time.time()
    roles = fetch_all_roles()
    cast = load_cast(roles)
    default = load_vector(DATA / "vectors" / "default_vector.pt")
    cast_mean = cast.mean(axis=0)

    lens = LensTransport(LENS_PATH)
    unembedding = Unembedding(DATA)
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    texts, wordlike = decode_vocab(tokenizer, unembedding.vocab_size)
    print(lens, unembedding, f"{len(roles)} roles", sep="\n", flush=True)

    baselines = {"vs_default": default, "vs_cast": cast_mean}
    anchors = {"vs_default": cast_mean, "vs_cast": default}
    data: dict[str, dict] = {}
    used: set[int] = set()

    for key, spec in CONTRASTS.items():
        print(f"\n[{key}] {spec['label']}", flush=True)
        # The anchor rides along as one extra row, contrasted the same way.
        directions = np.concatenate(
            [cast - baselines[key], (anchors[key] - baselines[key])[None]], axis=0)
        names = roles + [spec["anchor"]]

        print("  transporting...", flush=True)
        transported = transport_all(lens, directions)
        del directions
        print("  unembedding...", flush=True)
        ids, probs, ents = read_out(unembedding, transported, args.top_k, wordlike)
        del transported

        entries = {}
        for i, name in enumerate(names):
            used.update(int(t) for t in ids[i].ravel())
            entries[name] = {
                "i": [int(t) for t in ids[i].ravel()],
                "p": [float(f"{p:.3g}") for p in probs[i].ravel()],
                "h": [round(float(h), 3) for h in ents[i]],
                "sig": signature(ids[i], lens.source_layers),
            }
        data[key] = entries
        print(f"  done ({time.time() - started:.0f}s elapsed)", flush=True)

    document = {
        "meta": {
            "model": "Qwen/Qwen3-32B",
            "n_roles": len(roles),
            "lens_prompts_fitted": lens.n_prompts,
            "lens_source": "neuronpedia/jacobian-lens "
                           "(qwen3-32b/jlens/Salesforce-wikitext)",
            "vector_source": "lu-christina/assistant-axis-vectors (qwen-3-32b)",
            "top_k": args.top_k,
            "signature_band": [SIGNATURE_BAND.start, SIGNATURE_BAND.stop - 1],
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "layers": lens.source_layers,
        "contrasts": {k: {"label": v["label"], "description": v["description"],
                          "suffix": v["suffix"], "anchor": v["anchor"]}
                      for k, v in CONTRASTS.items()},
        "roles": roles,
        "vocab": {str(t): texts[t] for t in sorted(used)},
        "data": data,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document), encoding="utf-8")
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
