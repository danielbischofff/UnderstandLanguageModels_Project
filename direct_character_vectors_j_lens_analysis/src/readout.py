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
        if "jacobian_sum" in checkpoint:      # fit checkpoint (what is published)
            self._jacobians = checkpoint["jacobian_sum"]
            self.n_prompts = int(checkpoint["n_done"])
        elif "J" in checkpoint:               # JacobianLens.save() output
            self._jacobians = checkpoint["J"]
            self.n_prompts = int(checkpoint["n_prompts"])
        else:
            raise ValueError(f"unrecognised lens file; keys={sorted(checkpoint)}")
        self.source_layers = sorted(self._jacobians)
        self.d_model = int(self._jacobians[self.source_layers[0]].shape[0])

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
        jacobian = self._jacobians[layer].numpy()
        return (jacobian @ vectors.T).T / self.n_prompts
