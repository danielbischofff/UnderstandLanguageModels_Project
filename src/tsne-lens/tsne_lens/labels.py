"""Original-space representatives and compact label placement for both renderers."""

import numpy as np
from sklearn.metrics import pairwise_distances

PINNED_ROLES = {"assistant", "demon", "bard", "criminal"}


def representative_indices(vectors, clusters, metric="cosine", count=3):
    """Rank members by mean distance to their cluster (the first is its medoid)."""
    selected = []
    for cluster in sorted(set(clusters)):
        if cluster == -1:
            continue
        members = np.flatnonzero(np.asarray(clusters) == cluster)
        distances = pairwise_distances(vectors[members], metric=metric)
        order = np.argsort(distances.mean(axis=1), kind="stable")[:count]
        selected.extend(members[order].tolist())
    return selected


def labeled_indices(vectors, labels, clusters, metric="cosine"):
    selected = set(representative_indices(vectors, clusters, metric))
    selected.update(i for i, label in enumerate(labels) if label in PINNED_ROLES)
    return sorted(selected)


def label_offsets(xy, names, span, width=760, height=640):
    """Greedily separate text boxes; offsets are pixels, with positive y upward."""
    points = np.asarray(xy) / np.maximum(span, 1e-12) * [width, height]
    boxes, offsets = [], []
    directions = np.asarray(
        [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
    )
    for point, name in zip(points, names):
        half = np.asarray([max(12, len(name) * 3.2), 8.0])
        candidates = []
        for radius in (14, 28, 44, 64, 88, 116):
            for direction in directions:
                offset = direction * (half + radius)
                center = point + offset
                low, high = center - half - 2, center + half + 2
                overlap = sum(
                    np.prod(np.maximum(0, np.minimum(high, b) - np.maximum(low, a)))
                    for a, b in boxes
                )
                covered = np.sum(np.all((points > low) & (points < high), axis=1))
                score = overlap * 100 + covered * 80 + np.linalg.norm(offset)
                candidates.append((score, offset, low, high))
        _, offset, low, high = min(candidates, key=lambda item: item[0])
        offsets.append(offset)
        boxes.append((low, high))
    return offsets
