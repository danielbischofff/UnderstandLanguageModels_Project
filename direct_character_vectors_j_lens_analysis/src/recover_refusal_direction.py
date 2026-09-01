"""Recover a refusal direction from an abliterated model, by weight diff.

No refusal direction is published for Qwen3.6-27B, but abliterated builds are.
Abliteration removes the refusal direction from the weights by projecting it
out, so for each modified matrix

    W_abliterated = W - r (r^T W),      r a unit vector in residual space

which makes ``D = W - W_abliterated = r (r^T W)`` **rank one**, with left
singular vector exactly ``r``. Recovering the direction is therefore a diff of
one matrix plus a power iteration -- about 63 MB per side per layer, rather
than the ~60 GB of either model.

Rank-one-ness is the check that it worked: if ``s1 / ||D||_F`` is not close to
1, the diff is not a projection and the direction is not recoverable this way.
Agreement between layers is the second check: abliteration applies one global
direction, so independently recovered layers should be collinear.

Sign is not determined by this procedure (``r`` and ``-r`` span the same
projection), so downstream readouts should look at both.

Usage::

    python src/recover_refusal_direction.py [--layers 11,31,51] [--proj o_proj]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fetch_data import DATA, _download, _url, safetensors_header
from unembed import bf16_to_float32

BASE = "Qwen/Qwen3.6-27B"
ABLITERATED = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
ROOT = DATA / "qwen3.6-27b"
CACHE = ROOT / "weights"
TENSOR = "model.language_model.layers.{layer}.self_attn.{proj}.weight"


def load_index(repo: str) -> dict:
    dest = CACHE / f"{repo.replace('/', '_')}.index.json"
    _download(_url(repo, "model.safetensors.index.json"), dest, quiet=True)
    return json.loads(dest.read_text(encoding="utf-8"))["weight_map"]


def fetch_matrix(repo: str, name: str, index: dict) -> np.ndarray:
    """Range-fetch one tensor to disk (resumable) and return it as float32."""
    url = _url(repo, index[name])
    header, data_start = safetensors_header(url)
    entry = header[name]
    lo, hi = entry["data_offsets"]
    assert entry["dtype"] == "BF16", entry["dtype"]
    dest = CACHE / f"{repo.replace('/', '_')}__{name}.bf16.bin"
    _download(url, dest, data_start + lo, data_start + hi - 1, quiet=True)
    return bf16_to_float32(np.fromfile(dest, dtype="<u2")).reshape(entry["shape"])


def top_singular(matrix: np.ndarray, iters: int = 80) -> tuple[np.ndarray, float]:
    """Leading left singular vector and value, by power iteration."""
    v = np.random.default_rng(0).standard_normal(matrix.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    u = matrix @ v
    for _ in range(iters):
        u = matrix @ v
        u /= np.linalg.norm(u)
        v = matrix.T @ u
        value = np.linalg.norm(v)
        v /= value
    return u, float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", default="11,31,51")
    parser.add_argument("--proj", default="o_proj", choices=("o_proj", "down_proj"))
    parser.add_argument("--out", type=Path, default=ROOT / "refusal_direction.npy")
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    index = load_index(BASE)

    found: dict[int, np.ndarray] = {}
    for layer in (int(x) for x in args.layers.split(",")):
        name = TENSOR.format(layer=layer, proj=args.proj)
        if name not in index:
            print(f"L{layer}: no {args.proj} (this layer uses linear attention)")
            continue
        diff = fetch_matrix(BASE, name, index) - fetch_matrix(ABLITERATED, name, index)
        frobenius = float(np.linalg.norm(diff))
        if frobenius == 0:
            print(f"L{layer}: identical -- abliteration did not touch {args.proj}")
            continue
        vector, value = top_singular(diff)
        print(f"L{layer}: ||D||_F={frobenius:.4f}  s1={value:.4f}  "
              f"s1/||D||_F={value / frobenius:.4f}  (1.00 => rank one)")
        found[layer] = vector

    if not found:
        raise SystemExit("nothing recovered; try --proj down_proj")

    layers = sorted(found)
    for a, b in zip(layers, layers[1:]):
        cosine = float(abs(found[a] @ found[b]))
        print(f"|cos| between L{a} and L{b}: {cosine:.4f}  (1.00 => one global direction)")

    direction = found[layers[len(layers) // 2]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, direction.astype(np.float32))
    print(f"wrote {args.out}  (d_model={direction.size}, from layer {layers[len(layers)//2]})")


if __name__ == "__main__":
    main()
