"""Read J-Lens top-k readouts from a steering-binary-v1 results ZIP into dense arrays.

The shared decoder (`steering.binary_results.decode_trial`) builds one Python dict
per readout entry, ~200k per trial; here the id array is instead reshaped straight
to ``[layers, positions, top_k]``. That is only valid because every (layer,
position) cell of these runs holds exactly ``top_k`` entries, which is checked.

Only prompt positions are kept (see ``metrics.py`` for why), which also keeps the
140-trial persona suites small in memory.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trial:
    question: str
    character: str          # "unsteered" for the control
    scheme: str | None      # steering scheme name, None for the control
    layers: list[int]       # intervention layers ([] for the control)
    coefficient: float | None   # alpha (additive) or beta (character band); None for the control
    ids: np.ndarray         # uint32 [layers, prompt positions, top_k], sorted by logit per cell
    prompt: list[str]       # prompt token texts


def load_trials(bundle: Path) -> tuple[list[Trial], dict[str, str], dict]:
    """Return every trial, the shared id -> token-text vocabulary and the glossary."""
    with zipfile.ZipFile(bundle) as archive:
        index = json.loads(archive.read("index.json"))
        if index["format"] != "steering-binary-v1":
            raise ValueError("unsupported binary results format")
        vocabulary = json.loads(archive.read("vocabulary.json"))
        glossary = json.loads(archive.read("glossary.json"))
        top_k = index["document"]["experiment"]["j_lens"]["top_k"]
        trials = []
        for descriptor in index["trials"]:
            meta = descriptor["metadata"]
            raw = gzip.decompress(archive.read(descriptor["payload"]))
            item = descriptor["arrays"]
            ids = np.frombuffer(raw, dtype=item["ids"]["dtype"], count=item["ids"]["count"],
                                offset=item["ids"]["offset"])
            offsets = np.frombuffer(raw, dtype=item["offsets"]["dtype"], count=item["offsets"]["count"],
                                    offset=item["offsets"]["offset"])
            counts = descriptor["position_counts"]
            if len(set(counts)) != 1 or not np.all(np.diff(offsets) == top_k):
                raise ValueError("readout is ragged; cannot reshape to a dense array")
            prompt_length = meta["j_lens"]["prompt_length"]
            steering = meta["steering"]
            trials.append(Trial(
                question=meta["question"]["id"],
                character=meta["character"],
                scheme=steering["name"] if steering else None,
                layers=list(steering["layers"]) if steering else [],
                coefficient=float(steering["coefficient"]) if steering else None,
                ids=ids.reshape(len(counts), counts[0], top_k)[:, :prompt_length].copy(),
                prompt=[token["text"] for token in meta["j_lens"]["tokens"][:prompt_length]],
            ))
    return trials, vocabulary, glossary
