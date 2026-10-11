"""CPU analysis of retained residuals; no steered model is loaded."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from steering.j_lens import _analysis_imports
from steering.capture import ActivationArtifact
from .contrastive import ContrastiveProbe, ContrastiveSet, ranks_from_logits


class AnalysisReader:
    def __init__(self, *, analysis_source: Path, data_dir: Path, lens_path: Path, top_k_count=10):
        self.top_k_count = top_k_count
        with _analysis_imports(analysis_source) as (readout, unembed, analysis):
            self._lens = readout.LensTransport(lens_path)
            self._unembedding = unembed.Unembedding(data_dir)
            self._top_k = unembed.top_k
            self._tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
            _, self._wordlike = analysis.decode_vocab(self._tokenizer, self._unembedding.vocab_size)
        self.source_layers = tuple(int(layer) for layer in self._lens.source_layers)

    def probe(
        self,
        sets: Sequence[ContrastiveSet],
        *,
        layers: Sequence[int],
        top_k: int | None = None,
    ) -> ContrastiveProbe:
        """Bind curated contrastive sets to this lens's vocabulary and layers."""

        unfitted = sorted(set(int(layer) for layer in layers) - set(self.source_layers))
        if unfitted:
            raise ValueError(
                f"the J-Lens has no fitted Jacobian for contrastive layers {unfitted}; "
                f"available={list(self.source_layers)}"
            )
        return ContrastiveProbe(
            sets,
            tokenizer=self._tokenizer,
            embedding_rows=self._unembedding.rows,
            layers=layers,
            top_k=self.top_k_count if top_k is None else top_k,
        )

    def probe_artifact(self, artifact: ActivationArtifact, probe: ContrastiveProbe):
        events = {
            int(event["layer"]): event for event in artifact.manifest["events"]
            if event["point"] == "post_intervention" and event["phase"] == "prefill"
        }
        missing = sorted(set(probe.layers) - set(events))
        if missing:
            raise ValueError(f"retained activations have no full-sequence layers {missing}")
        series = []
        for layer in probe.layers:
            tensor = artifact.load_tensor(events[layer]["tensor_key"])
            matrix = tensor[0].float().numpy().astype(np.float32, copy=False)
            transported = self._lens.transport(layer, matrix)
            normalized = self._unembedding.normalize(transported)
            logits = self._unembedding.logits(transported)
            ids, _, probabilities = self._top_k(logits, probe.top_k, allowed=self._wordlike)
            ranks = ranks_from_logits(logits, probe.token_ids)
            del logits
            series.extend(probe.measure_layer(
                layer, top_ids=ids, top_probabilities=probabilities, ranks=ranks,
                normalized_activations=normalized
            ))
        return {**probe.describe(), "series": series}
