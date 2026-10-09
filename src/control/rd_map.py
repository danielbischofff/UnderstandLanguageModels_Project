"""Exploratory: the persona map and the tone trajectories on R - D instead of R.

Fits a second t-SNE map, with the final map's settings, on the 275 steering
vectors c = R - D (layer 32) and its own Ward-6 clusters, then projects the
residuals as x - D, so both sit in the same "relative to the default" frame.
Not used in the paper; outputs go to writeup/figures/tsne/rd/ for comparison.

    uv run --extra plots python src/control/rd_map.py
"""

from __future__ import annotations

import json

import figures as F
from common import load_space
from tsne_lens.projection import RoleMap

OUT = F.OUT / "rd"
CACHE = F.ROOT / "outputs/plots/tsne/roles-rd/role-map.pkl"


def fit_map(space) -> RoleMap:
    if CACHE.exists():
        return RoleMap.load(CACHE)
    final = RoleMap.load(F.TSNE / "roles/role-map.pkl").config
    role_map = RoleMap.fit(
        space.vectors - space.default, space.labels, layer=32, metric="cosine",
        perplexity=tuple(final["perplexities"]), dof=final["dof"], seed=final["seed"],
        n_clusters=6, subclusters=6, max_iter=final["max_iter"], tolerance=final["tolerance"],
        early_exaggeration=final["early_exaggeration"],
        early_exaggeration_iter=final["early_exaggeration_iter"],
        exaggeration_anneal_iter=final["exaggeration_anneal_iter"],
        exaggeration=final["exaggeration"], n_iter=final["n_iter"],
        relaxation_iter=final["relaxation_iter"], theta=final["theta"],
    )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    role_map.save(CACHE)
    return role_map


def main() -> None:
    space = load_space()
    role_map = fit_map(space)
    OUT.mkdir(parents=True, exist_ok=True)
    F.print_layout(F.make_figure(role_map, [])).write_image(OUT / "roles_map_rd.pdf")
    clusters = {r["label"]: r["cluster"] for r in role_map.role_records()}
    names = {clusters["infant"]: (f"{clusters['infant']} · " + role_map.cluster_names[clusters["infant"]].replace(", ", ",\n", 1), (-8, 14)),
             clusters["assistant"]: (f"{clusters['assistant']} · " + role_map.cluster_names[clusters["assistant"]].replace(", ", ",\n", 1), (20, -58))}
    F.trajectory_figure(role_map, F.tone_runs(role_map, offset=space.default),
                        out=OUT / "trajectory_rd.pdf", names=names)
    print(json.dumps({k: role_map.config["quality"][k] for k in
                      ("neighbor_recall", "trustworthiness", "global_distance_spearman",
                       "cluster_silhouette_original", "cluster_silhouette_2d")}, indent=1))
    print({c: n for c, n in role_map.cluster_names.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
