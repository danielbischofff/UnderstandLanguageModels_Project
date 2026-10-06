"""Top-k J-Lens readout of ``role - default`` at every layer, for a few roles.

The atlas (``run_sweep.py``) keeps only the top 8. The steering runs read out the
top 10 per cell, so ``token_trend_analysis`` needs the same depth on the vector
itself to decide which words are the steering vector's own readout. This is the
atlas ``vs_default`` readout -- same transport, same word-like mask -- for just
the steered roles, which takes a few minutes instead of half an hour.

Usage::

    python src/vector_topk.py [--roles demon criminal cynic] [--top-k 10]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from fetch_data import fetch_role
from readout import LensTransport, load_vector
from run_analysis import DATA, LENS_PATH, ROOT, decode_vocab
from run_sweep import read_out, transport_all
from unembed import Unembedding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roles", nargs="+", default=["demon", "criminal", "cynic"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "vs_default_top10.json")
    args = parser.parse_args()

    started = time.time()
    default = load_vector(DATA / "vectors" / "default_vector.pt")
    directions = np.stack([load_vector(fetch_role(r)) - default for r in args.roles])

    lens = LensTransport(LENS_PATH)
    unembedding = Unembedding(DATA)
    texts, wordlike = decode_vocab(Tokenizer.from_file(str(DATA / "tokenizer.json")), unembedding.vocab_size)

    print("transporting...", flush=True)
    transported = transport_all(lens, directions)
    print("unembedding...", flush=True)
    ids, _probs, _ents = read_out(unembedding, transported, args.top_k, wordlike)

    document = {
        "meta": {"contrast": "role - default", "top_k": args.top_k, "lens_prompts_fitted": lens.n_prompts,
                 "generated": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "layers": list(lens.source_layers),
        # ids[role][layer index] -> top-k token ids, best first
        "ids": {role: ids[i].tolist() for i, role in enumerate(args.roles)},
        "vocab": {str(t): texts[t] for t in sorted({int(t) for t in ids.ravel()})},
    }
    args.out.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
