"""Fetch what the Qwen3.6-27B J-lens / R-lens comparison needs.

A side investigation: does the rare-token cluster that shows up in early and
middle layers come from the Jacobian lens itself, or is it in the representation?
R-lens (RelP) is the same estimator read through an LRP-modified backward graph,
and its claimed advantage is exactly early-layer faithfulness, so a matched J/R
pair on one direction is a direct test.

Fetched here:

* the two lenses from ``camilablank/workspace-lenses`` (3.3 GB each, matched
  pair: same target layer, skip, corpus and prompt count);
* Qwen3.6-27B's unembedding -- ``lm_head.weight`` is 2.54 GB at a vocabulary of
  248,320, range-requested out of its shard rather than pulling the 60 GB model;
* the tokenizer.

Layout mirrors the Qwen3-32B side, so :class:`unembed.Unembedding` works
unchanged when pointed at ``data/qwen3.6-27b``.

Usage::

    python src/fetch_qwen36.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fetch_data import DATA, _download, _url, safetensors_header

MODEL = "Qwen/Qwen3.6-27B"
LENS_REPO = "camilablank/workspace-lenses"
ROOT = DATA / "qwen3.6-27b"
#: Source tensor -> the name :class:`unembed.Unembedding` expects.
UNEMBED = {
    "lm_head.weight": "lm_head.weight",
    "model.language_model.norm.weight": "model.norm.weight",
}


def fetch_lenses() -> None:
    from huggingface_hub import hf_hub_download

    print("lenses (3.3 GB each):")
    for arm in ("j-lens", "r-lens"):
        path = hf_hub_download(LENS_REPO, f"qwen3.6-27b/{arm}/lens.pt", local_dir=DATA)
        print(f"  {arm}: {Path(path).stat().st_size / 1e9:.2f} GB")


def fetch_unembedding() -> None:
    """Range-fetch the unembedding out of its shard, as raw bf16."""
    print("unembedding:")
    index = json.loads(
        _download(_url(MODEL, "model.safetensors.index.json"),
                  ROOT / "index.json", quiet=True).read_text(encoding="utf-8")
    )["weight_map"]

    meta = {}
    for source, alias in UNEMBED.items():
        url = _url(MODEL, index[source])
        header, data_start = safetensors_header(url)
        entry = header[source]
        lo, hi = entry["data_offsets"]
        assert entry["dtype"] == "BF16", entry["dtype"]
        dest = ROOT / "unembed" / f"{alias}.bf16.bin"
        _download(url, dest, data_start + lo, data_start + hi - 1)
        meta[alias] = {"shape": entry["shape"], "dtype": "bfloat16",
                       "file": dest.name, "source_name": source}
    (ROOT / "unembed" / "meta.json").write_text(json.dumps(meta, indent=2))


def fetch_tokenizer() -> None:
    print("tokenizer:")
    _download(_url(MODEL, "tokenizer.json"), ROOT / "tokenizer.json")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    fetch_tokenizer()
    fetch_unembedding()
    fetch_lenses()
    print("\nready in", ROOT)


if __name__ == "__main__":
    main()
