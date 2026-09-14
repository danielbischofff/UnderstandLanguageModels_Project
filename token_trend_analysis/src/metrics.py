"""Per-word occupancy of the J-Lens readout grid, restricted to the region steering can reach.

For a trial whose first intervention layer is ``L`` the analysed grid is prompt
positions x layers ``L..62``. The prompt (chat template and question, up to the
newline before the answer) is the same text for a question at every steering
strength, so positions are aligned; response positions are not, because the
answer itself changes with steering.

Each cell holds a top-k list; a *word* is a group of token ids whose text is equal
after stripping whitespace and lower-casing (" warn", "warn" and " Warn" are one
word), so a cell counts at most once per word.

Three numbers per (trial, word):

- ``cells``    the raw count of (position, layer) cells containing the word — the
               requested metric;
- ``weighted`` the same sum with each cell weighted by ``1 / rank`` (rank 1..k of
               the word's best-ranked variant in that cell);
- ``spread``   the fraction of prompt positions where the word appears at *any*
               analysed layer, used to flag suspected baseline tokens.

Within a scheme the grid size is constant, so dividing by it (as the tables do)
only rescales counts to per mille of the grid.
"""

from __future__ import annotations

import numpy as np

from load import Trial


def word_key(text: str) -> str:
    """Group token variants that differ only in leading space or capitalisation."""
    stripped = text.strip().lower()
    return stripped if stripped else repr(text)   # keep "\n", " " etc. distinct and visible


def build_groups(vocabulary: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Map every token id to a word index; ids absent from the vocabulary map to -1."""
    keys = sorted({word_key(text) for text in vocabulary.values()})
    position = {key: i for i, key in enumerate(keys)}
    lookup = np.full(max(int(i) for i in vocabulary) + 1, -1, dtype=np.int64)
    for token_id, text in vocabulary.items():
        lookup[int(token_id)] = position[word_key(text)]
    return lookup, keys


def occupancy(trial: Trial, first_layer: int, lookup: np.ndarray, n_words: int) -> dict[str, np.ndarray]:
    """Cells, rank-weighted cells and position spread of every word for one trial."""
    grid = trial.ids[first_layer:]          # the loader keeps prompt positions only
    n_layers, n_positions, top_k = grid.shape
    words = lookup[grid.astype(np.int64)]
    cell = np.arange(n_layers * n_positions).reshape(n_layers, n_positions, 1)
    rank = np.broadcast_to(np.arange(1, top_k + 1), grid.shape)

    # Row-major flattening keeps each cell's entries in rank order, so np.unique's
    # first occurrence of a (cell, word) pair is that word's best rank in the cell.
    pair = (np.broadcast_to(cell, grid.shape) * n_words + words).ravel()
    _, first = np.unique(pair, return_index=True)
    cells = np.bincount(words.ravel()[first], minlength=n_words)
    weighted = np.bincount(words.ravel()[first], weights=1.0 / rank.ravel()[first], minlength=n_words)

    position = np.broadcast_to(np.arange(n_positions).reshape(1, n_positions, 1), grid.shape)
    _, first = np.unique((position * n_words + words).ravel(), return_index=True)
    spread = np.bincount(words.ravel()[first], minlength=n_words) / n_positions

    return {"cells": cells, "weighted": weighted, "spread": spread,
            "grid_size": n_layers * n_positions, "n_positions": n_positions, "n_layers": n_layers}

