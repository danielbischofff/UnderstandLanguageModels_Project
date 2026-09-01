"""Transport per-layer persona directions through the fitted Jacobian lens.

The Jacobian lens reads out what an internal activation is disposed to make the
model say::

    lens_l(h) = unembed( J_l @ h ),    J_l = E[ d h_final / d h_l ]

so a persona direction at layer ``l`` becomes a distribution over vocabulary
tokens. ``J_l`` is ``[d_model, d_model]``; at 5120 wide and fp32 that is 100 MB
per layer and 6.6 GB for the 63 fitted layers, so the file is memory-mapped and
touched one layer at a time.

The artifact published by Neuronpedia is a *fit checkpoint* (``jacobian_sum``
accumulated over ``n_done`` prompts), not the output of
``JacobianLens.save()`` -- ``jlens.JacobianLens.load`` rejects it. Both layouts
are handled here. Dividing the sum by the prompt count only rescales ``J_l``,
and the final RMSNorm makes the readout scale-invariant, so the token rankings
are unaffected either way; we divide anyway to keep absolute logits meaningful.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_vector(path: str | Path) -> np.ndarray:
    """Load a ``[n_layers, d_model]`` persona vector as float32.

    Handles both the bare tensors published for Qwen3-32B and the
    ``{"axis": ...}`` dicts that ``assistant_axis.save_axis`` writes.
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        obj = obj["axis"] if "axis" in obj else next(iter(obj.values()))
    vector = obj.float().numpy()
    if vector.ndim != 2:
        raise ValueError(f"{path}: expected [n_layers, d_model], got {vector.shape}")
    return vector


class LensTransport:
    """Memory-mapped per-layer Jacobians, applied to batches of directions."""

    def __init__(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if "jacobian_sum" in checkpoint:
            # A fit checkpoint: an accumulated *sum* over prompts, so it has to
            # be divided by the count to become the averaged Jacobian.
            self._jacobians = checkpoint["jacobian_sum"]
            self.n_prompts = int(checkpoint["n_done"])
            self._scale = 1.0 / self.n_prompts
        elif "J" in checkpoint:
            # JacobianLens.save() output: already the mean, so no rescaling.
            self._jacobians = checkpoint["J"]
            self.n_prompts = int(checkpoint["n_prompts"])
            self._scale = 1.0
        else:
            raise ValueError(f"unrecognised lens file; keys={sorted(checkpoint)}")

        # J is either {layer: [d, d]} or a stacked [n_source, d, d] tensor whose
        # rows follow `source_layers`; index by layer number either way.
        if isinstance(self._jacobians, torch.Tensor):
            self.source_layers = [int(l) for l in checkpoint["source_layers"]]
            self._row = {l: i for i, l in enumerate(self.source_layers)}
        else:
            self.source_layers = sorted(self._jacobians)
            self._row = None
        self.d_model = int(self._at(self.source_layers[0]).shape[0])
        self.provenance = checkpoint.get("provenance")

    def _at(self, layer: int) -> torch.Tensor:
        """The Jacobian for ``layer``, whichever layout the file uses."""
        return (self._jacobians[self._row[layer]] if self._row is not None
                else self._jacobians[layer])

    def __repr__(self) -> str:
        first, last = self.source_layers[0], self.source_layers[-1]
        return (f"LensTransport(d_model={self.d_model}, n_prompts={self.n_prompts}, "
                f"layers={first}..{last})")

    def transport(self, layer: int, directions: np.ndarray) -> np.ndarray:
        """Map ``[n, d_model]`` directions at ``layer`` into the final-layer basis.

        Computed as ``(J_l @ V.T).T`` so the 100 MB Jacobian is consumed in its
        native row-major order and never transposed into a copy.
        """
        vectors = np.atleast_2d(np.asarray(directions, dtype=np.float32))
        # Lenses ship as fp32 or fp16; promote, since numpy's fp16 matmul is
        # an order of magnitude slower and this is the inner loop.
        jacobian = self._at(layer).float().numpy()
        return (jacobian @ vectors.T).T * self._scale

    def matrix(self, layer: int) -> tuple[torch.Tensor, float]:
        """Return one memory-mapped Jacobian and its averaging scale."""
        if layer not in self.source_layers:
            raise KeyError(layer)
        return self._at(layer), self._scale
