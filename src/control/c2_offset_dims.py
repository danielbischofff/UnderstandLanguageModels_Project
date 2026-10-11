"""C2: is the nearest-role readout decided by the shared offset coordinate?

Dim 731 carries the first-token massive activation and a constant offset on
every token, and it holds 60-72% of each role vector's squared norm. The model
needs it (it acts as a bias, Sun et al. 2024), so this is not a claim that the
coordinate is unimportant. The question is narrower: does the lens readout, the
role cluster nearest to each token, follow that one shared coordinate or the
remaining 5117 dimensions where roles differ? We recompute the readout
(a) without dims 731/2994/1016 (the three massive dims of the first token) and
(b) after subtracting the role mean from roles and residuals, and measure how
often the nearest cluster and the steered-role statistics stay the same.

The J-Lens part asks the same of the J-Lens readout itself: the top-10 word-like
tokens of every role vector (its base role tokens) with and without the three
coordinates, against silencing three random coordinates. It needs the fitted
lens, unembedding and tokenizer under direct_character_vectors_j_lens_analysis/data
(fetch_data.py there downloads them).

    uv run --extra plots python src/control/c2_offset_dims.py [--parts residuals jlens]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from common import LAYER, RESULTS, ROOT, load_space, residuals, runs, save, trials, unit

MASSIVE = [731, 2994, 1016]


def readouts(x, roles, mean):
    keep = np.ones(x.shape[1], dtype=bool)
    keep[MASSIVE] = False
    return {
        "full": unit(x) @ unit(roles).T,
        "no_massive_dims": unit(x[:, keep]) @ unit(roles[:, keep]).T,
        "centered": unit(x - mean) @ unit(roles - mean).T,
    }


def residual_part(space) -> dict:
    out = {}
    for segment in ("prompt", "generated"):
        nearest = {k: [] for k in ("full", "no_massive_dims", "centered")}
        own = {k: [] for k in nearest}
        rank = {k: [] for k in nearest}
        for method, run in runs().items():
            for trial in trials(method, run):
                if trial["character"] == "unsteered" and method == "band":
                    continue
                x = residuals(trial)[segment][0]
                x = x[1:] if segment == "prompt" else x
                target = "assistant" if trial["character"] == "unsteered" else trial["character"]
                t = space.index(target)
                for name, cos in readouts(x, space.vectors, space.mean).items():
                    cl = space.clusters[cos.argmax(1)]
                    nearest[name].append(cl)
                    own[name].append((cl == space.clusters[t]).mean())
                    rank[name].append(np.median((cos > cos[:, [t]]).sum(1) + 1))
        full = np.concatenate(nearest["full"])
        out[segment] = {
            name: dict(
                same_nearest_cluster=float((np.concatenate(nearest[name]) == full).mean()),
                kappa=float(cohen_kappa_score(full, np.concatenate(nearest[name]))),
                cluster_counts=np.bincount(np.concatenate(nearest[name]), minlength=6).tolist(),
                own_spearman_vs_full=float(spearmanr(own["full"], own[name]).statistic),
                rank_spearman_vs_full=float(spearmanr(rank["full"], rank[name]).statistic),
            )
            for name in ("no_massive_dims", "centered")
        } | {"full_cluster_counts": np.bincount(full, minlength=6).tolist(), "tokens": len(full)}
    return out


JLENS = ROOT / "direct_character_vectors_j_lens_analysis"
LENS_FILE = JLENS / "data/qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
EXAMPLES = ("assistant", "demon", "toddler", "infant", "criminal")


def jlens_part(space, k: int = 10, draws: int = 20, seed: int = 0) -> dict:
    """Base role tokens (J-Lens top-k of each role vector) vs. the same with
    the three rogue coordinates silenced, and vs. three random ones silenced."""
    sys.path.insert(0, str(JLENS / "src"))
    from readout import LensTransport  # noqa: E402
    from run_analysis import decode_vocab  # noqa: E402
    from tokenizers import Tokenizer  # noqa: E402
    from unembed import Unembedding, top_k  # noqa: E402

    lens = LensTransport(LENS_FILE)
    unembedding = Unembedding(JLENS / "data")
    texts, wordlike = decode_vocab(Tokenizer.from_file(str(JLENS / "data/tokenizer.json")),
                                   unembedding.vocab_size)

    def tokens(vectors):
        logits = unembedding.logits(lens.transport(LAYER, vectors))
        return top_k(logits, k, allowed=wordlike)[0]

    def silenced(vectors, dims):
        out = vectors.copy()
        out[:, dims] = 0.0
        return out

    def compare(base, other):
        overlap = np.array([len(set(a) & set(b)) / k for a, b in zip(base, other)])
        return dict(overlap_mean=float(overlap.mean()), overlap_median=float(np.median(overlap)),
                    overlap_min=float(overlap.min()), same_top1=float((base[:, 0] == other[:, 0]).mean()))

    rng = np.random.default_rng(seed)
    others = np.setdiff1d(np.arange(space.vectors.shape[1]), MASSIVE)
    result = {}
    for name, vectors in (("role_vectors", space.vectors),
                          ("steering_vectors", space.vectors - space.default)):
        vectors = vectors.astype(np.float32)
        base = tokens(vectors)
        rogue = tokens(silenced(vectors, MASSIVE))
        null = [compare(base, tokens(silenced(vectors, rng.choice(others, 3, replace=False))))
                for _ in range(draws)]
        result[name] = dict(
            rogue_dims_silenced=compare(base, rogue),
            random_dims_silenced={key: float(np.mean([n[key] for n in null])) for key in null[0]},
            examples={role: dict(base=[texts[i].strip() for i in base[space.index(role)]],
                                 silenced=[texts[i].strip() for i in rogue[space.index(role)]])
                      for role in EXAMPLES},
        )
    return dict(layer=LAYER, k=k, wordlike_only=True, null_draws=draws, **result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["residuals", "jlens"],
                        choices=["residuals", "jlens"])
    args = parser.parse_args()
    space = load_space()
    path = RESULTS / "c2_offset_dims.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    if "residuals" in args.parts:
        out.update(residual_part(space))
    if "jlens" in args.parts:
        out["jlens"] = jlens_part(space)
    save("c2_offset_dims", out)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
