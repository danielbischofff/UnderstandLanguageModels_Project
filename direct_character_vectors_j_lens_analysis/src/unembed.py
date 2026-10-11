"""Qwen3-32B's unembedding, streamed from disk.

``jlens`` defines the lens readout as ``unembed(J_l @ h)``, where ``unembed``
is the model's final RMSNorm followed by ``lm_head`` (Qwen3 has no logit
softcapping, so nothing else applies). Only those two tensors are needed, so
this module works directly on the raw bf16 bytes fetched by
:mod:`fetch_data` -- ``lm_head`` is memory-mapped and converted a slice at a
time, keeping resident memory at a few hundred MB instead of 1.6 GB.

The whole readout is a single streaming pass over the vocabulary, so callers
should batch every vector they care about into one ``[n, d_model]`` matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Qwen3-32B ``rms_norm_eps`` (config.json).
RMS_EPS = 1e-6
#: Vocabulary rows converted per streaming step (~160 MB of fp32 at 8192).
VOCAB_CHUNK = 8192


def bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    """Widen raw bfloat16 (as ``uint16``) to float32.

    bfloat16 is the top 16 bits of a float32, so the widening is an exact
    left-shift -- no lookup table and no precision loss.
    """
    return (raw.astype(np.uint32) << 16).view(np.float32)


def rms_norm(h: np.ndarray, gain: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    """Qwen3's final RMSNorm over the last axis, in float32.

    Note this makes the readout invariant to the scale of ``h``: any positive
    rescaling of a steering direction leaves the resulting logits unchanged.
    """
    h = np.asarray(h, dtype=np.float32)
    inv = 1.0 / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps)
    return (h * inv) * gain


class Unembedding:
    """Memory-mapped ``lm_head`` plus the final RMSNorm gain."""

    def __init__(self, data_dir: str | Path) -> None:
        root = Path(data_dir) / "unembed"
        meta = json.loads((root / "meta.json").read_text())
        head = meta["lm_head.weight"]
        self.vocab_size, self.d_model = head["shape"]
        self._weight = np.memmap(
            root / head["file"], dtype="<u2", mode="r",
            shape=(self.vocab_size, self.d_model),
        )
        gain = meta["model.norm.weight"]
        self.gain = bf16_to_float32(
            np.fromfile(root / gain["file"], dtype="<u2")
        )
        assert self.gain.shape == (self.d_model,), self.gain.shape

    def __repr__(self) -> str:
        return f"Unembedding(vocab={self.vocab_size}, d_model={self.d_model})"

    def rows(self, token_ids: np.ndarray) -> np.ndarray:
        """The raw ``lm_head`` rows for ``token_ids`` as ``[n, d_model]`` float32.

        These are the vocabulary's own directions in model space, used for
        token-to-token similarity. No norm gain is applied: that belongs to the
        readout of an activation, not to a comparison between two tokens.
        """
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise IndexError(f"token id outside vocabulary of {self.vocab_size}")
        return bf16_to_float32(np.asarray(self._weight[ids]))

    def normalize(self, residuals: np.ndarray) -> np.ndarray:
        """Apply the final RMSNorm, including its learned per-dimension gain."""
        return rms_norm(np.atleast_2d(np.asarray(residuals, dtype=np.float32)), self.gain)

    def logits(self, residuals: np.ndarray, chunk: int = VOCAB_CHUNK) -> np.ndarray:
        """Map ``[n, d_model]`` residuals to ``[n, vocab_size]`` logits.

        Applies the final norm, then streams ``lm_head`` in row blocks so the
        1.6 GB matrix is never fully resident.
        """
        residuals = np.atleast_2d(np.asarray(residuals, dtype=np.float32))
        normed = self.normalize(residuals)
        out = np.empty((normed.shape[0], self.vocab_size), dtype=np.float32)
        for lo in range(0, self.vocab_size, chunk):
            hi = min(lo + chunk, self.vocab_size)
            block = bf16_to_float32(np.asarray(self._weight[lo:hi]))
            out[:, lo:hi] = normed @ block.T
        return out


def top_k(logits: np.ndarray, k: int, allowed: np.ndarray | None = None):
    """Top ``k`` entries per row of ``logits``.

    Args:
        logits: ``[n, vocab]`` scores.
        k: How many entries to return per row.
        allowed: Optional boolean vocabulary mask. Masked-out tokens are
            excluded from the ranking but still count towards the softmax, so
            the reported probabilities stay comparable across maskings.

    Returns:
        ``(ids, scores, probs)``, each ``[n, k]``.
    """
    probs = logits - logits.max(axis=1, keepdims=True)
    np.exp(probs, out=probs)
    probs /= probs.sum(axis=1, keepdims=True)  # in place; these arrays are large

    scores = logits if allowed is None else np.where(allowed, logits, -np.inf)
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, idx, 1).argsort(axis=1)[:, ::-1]
    ids = np.take_along_axis(idx, order, 1)
    return ids, np.take_along_axis(logits, ids, 1), np.take_along_axis(probs, ids, 1)
