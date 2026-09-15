"""Cluster colours keyed to who is in a cluster, not to Ward's numbering.

Ward numbers its clusters by merge order, so the same group of roles can come out
as cluster 3 in one fit and cluster 0 in the next, and two plots of the same six
clusters then disagree about which one is orange. Every figure here instead sorts
the clusters by their *medoid* — the member closest to the rest of its own
cluster, the same role that opens the cluster's name — and hands out the palette
in that alphabetical order. The colour therefore follows the cluster's
membership: one refit, the roles map, the training GIF and the ribbons all give
the same six roles the same six colours.

The slot order is stable while a cluster keeps its medoid, which is what holds
for the Ward-6 cut of the 275 role vectors we work with. It is not preserved
across different cuts of the hierarchy: at Ward 5 two clusters merge, one medoid
disappears, and the slots below it shift up by one.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

# Categorical slots for the six Ward clusters. Validated all-pairs on the light
# chart surface: worst CVD dE 8.7, worst normal-vision dE 15.6, both above the
# gate. Two slots sit under 3:1 contrast, so any figure that leans on them also
# prints the cluster id.
CLUSTER_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#009E73",  # green
    "#9B5DE5",  # purple
    "#0072B2",  # blue
    "#D55E00",  # vermillion
]
# Past the sixth slot only the subcluster views are left, where a colour
# separates neighbours rather than being remembered between plots.
OVERFLOW_COLORS = [
    "#CC79A7",  # mulberry
    "#8C6D31",  # bronze
    "#17BECF",  # teal
    "#7F7F7F",  # grey
]
NOISE_COLOR = "#9CA3AF"  # cluster -1, where a clusterer emits one


def medoid_labels(vectors, labels, clusters, metric="cosine") -> dict[int, str]:
    """The most central member of each cluster, by mean in-cluster distance."""
    clusters = np.asarray(clusters)
    distances = pairwise_distances(np.asarray(vectors, dtype=np.float64), metric=metric)
    medoids = {}
    for cluster in sorted(set(clusters.tolist()) - {-1}):
        members = np.flatnonzero(clusters == cluster)
        central = members[np.argmin(distances[np.ix_(members, members)].mean(axis=1))]
        medoids[int(cluster)] = labels[int(central)]
    return medoids


def cluster_slots(vectors, labels, clusters, metric="cosine") -> dict[int, int]:
    """Map each cluster id to its fixed palette slot, ordered by medoid label."""
    medoids = medoid_labels(vectors, labels, clusters, metric)
    order = sorted(medoids, key=lambda cluster: medoids[cluster])
    return {cluster: slot for slot, cluster in enumerate(order)}


def slot_color(slot: int) -> str:
    if slot < len(CLUSTER_COLORS):
        return CLUSTER_COLORS[slot]
    return OVERFLOW_COLORS[(slot - len(CLUSTER_COLORS)) % len(OVERFLOW_COLORS)]


def cluster_colors(vectors, labels, clusters, metric="cosine") -> dict[int, str]:
    """Cluster id -> hex colour, fixed by membership rather than by id."""
    colors = {
        cluster: slot_color(slot)
        for cluster, slot in cluster_slots(vectors, labels, clusters, metric).items()
    }
    if -1 in set(np.asarray(clusters).tolist()):
        colors[-1] = NOISE_COLOR
    return colors


def map_colors(role_map, clusters=None) -> dict[int, str]:
    """The colour table for a fitted map, or for one of its finer Ward cuts."""
    return cluster_colors(
        role_map.vectors,
        role_map.labels,
        role_map.clusters if clusters is None else clusters,
        role_map.config["metric"],
    )


def soften(color: str, amount: float = 0.55) -> str:
    """Blend a slot colour toward white, for backdrops that must not shout."""
    value = color.lstrip("#")
    channels = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return "#" + "".join(
        f"{round(c + (255 - c) * amount):02X}" for c in channels
    )
