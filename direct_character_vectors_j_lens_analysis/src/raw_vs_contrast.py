"""Does subtracting the default cause the rare-token attractor?

An open question, not a settled result. Many characters' readouts surface a
recurring cluster of rare web-crawl tokens -- adult/SEO fragments, German city
names, Turkish and Polish fragments. The cluster is real (mean pairwise cosine
+0.22 between its unembedding rows, against +0.05 for random tokens) and sits
opposite the common-English cluster (-0.52), but what decides *which* characters
get exposed to it is unresolved. Four hypotheses have been tested and rejected:

* outlier unembedding norms -- the flagged tokens sit at the median (p50), while
  common English sits at p10;
* the Jacobian's dominant output direction -- cosine +0.06 with the cluster, and
  its own top tokens are whitespace and punctuation;
* the Jacobian mapping anything there -- random vectors through ``J`` give +0.01;
* weak or near-default characters -- Spearman +0.02 against stability, and
  ``teacher`` / ``tutor`` / ``interpreter`` are the closest to ``default`` and
  entirely clean.

A single case (``analyst`` at layer 58) points at subtraction: raw reads out as
``mindset stakeholders outcomes behaviors``, while ``analyst - default`` gives
``erotici prox operational erotik``. This script tests that on all 275
characters instead of one, by running the cast through the lens twice -- raw and
contrasted -- and counting flagged tokens in the deep band.

Note that the cosine-to-cluster-centroid statistic is *not* a reliable proxy
here: raw ``analyst`` scores higher on it (+0.218) than the contrasted vector
(+0.110) while showing none of the tokens. Count what actually reaches top-k.

Only the signature band is unembedded, which is about a tenth of the work of a
full sweep.

Usage::

    python src/raw_vs_contrast.py [--pattern REGEX]
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
from tokenizers import Tokenizer

from readout import LensTransport, load_vector
from run_analysis import DATA, LENS_PATH, ROOT, decode_vocab
from run_sweep import ROW_CHUNK, SIGNATURE_BAND, load_cast
from unembed import Unembedding, top_k

#: The recurring cluster, by surface form.
FLAGGED = (r"erot|porn|sex|fetisch|eskort|escort|milf|creampie|cumshot|lesbi"
           r"|geile|nude|prostitu|shemale|strapon")
TOP_K = 8


def deep_readout(lens: LensTransport, unembedding: Unembedding,
                 directions: np.ndarray, wordlike: np.ndarray) -> np.ndarray:
    """Top-k word-like ids for every character across the signature band."""
    band = [l for l in lens.source_layers if l in SIGNATURE_BAND]
    n, _, d_model = directions.shape
    transported = np.empty((n, len(band), d_model), dtype=np.float32)
    for j, layer in enumerate(band):
        transported[:, j, :] = lens.transport(layer, directions[:, layer, :])

    flat = transported.reshape(-1, d_model)
    ids = np.empty((flat.shape[0], TOP_K), dtype=np.int32)
    for lo in range(0, flat.shape[0], ROW_CHUNK):
        hi = min(lo + ROW_CHUNK, flat.shape[0])
        ids[lo:hi] = top_k(unembedding.logits(flat[lo:hi]), TOP_K,
                           allowed=wordlike)[0]
    return ids.reshape(n, len(band), TOP_K)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pattern", default=FLAGGED, help="tokens to count")
    args = parser.parse_args()

    atlas = json.loads((ROOT / "out" / "atlas.json").read_text(encoding="utf-8"))
    roles = [r for r in atlas["roles"] if r in atlas["data"]["vs_default"]]

    lens = LensTransport(LENS_PATH)
    unembedding = Unembedding(DATA)
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    texts, wordlike = decode_vocab(tokenizer, unembedding.vocab_size)
    pattern = re.compile(args.pattern, re.I)
    flagged = np.array([bool(pattern.search(t)) for t in texts])

    cast = load_cast(roles)
    default = load_vector(DATA / "vectors" / "default_vector.pt")

    counts = {}
    for label, directions in (("raw", cast), ("minus default", cast - default)):
        ids = deep_readout(lens, unembedding, directions, wordlike)
        per = {r: int(flagged[ids[i]].sum()) for i, r in enumerate(roles)}
        hit = sum(1 for c in per.values() if c)
        slots = ids.size
        print(f"{label:15s} characters affected {hit}/{len(roles)} "
              f"({100 * hit / len(roles):.0f}%) | flagged slots "
              f"{sum(per.values())}/{slots} ({100 * sum(per.values()) / slots:.1f}%)")
        counts[label] = per

    raw, sub = counts["raw"], counts["minus default"]
    only_sub = [r for r in roles if not raw[r] and sub[r]]
    only_raw = [r for r in roles if raw[r] and not sub[r]]
    both = [r for r in roles if raw[r] and sub[r]]
    print(f"\nboth: {len(both)} | only after subtracting: {len(only_sub)} | "
          f"only raw: {len(only_raw)}")
    print(f"analyst -> raw {raw['analyst']}, subtracted {sub['analyst']}")
    print(f"only-after-subtraction examples: {', '.join(only_sub[:10])}")
    print(f"only-raw examples: {', '.join(only_raw[:10])}")


if __name__ == "__main__":
    main()
