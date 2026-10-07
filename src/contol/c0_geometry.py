"""C0: role-space geometry behind the results text (R1, R3, R4).

- how parallel the role vectors and the steering vectors R - D are, and how much
  of R - D lies along the Assistant Axis (D - mean role, Lu et al. 2026);
- the shared offset coordinate dim 731: its share in role vectors, on single
  tokens, and after averaging tokens (why role vectors are dominated by it);
- the first-token attention sink: norm, concentration, nearest roles;
- within the unsteered prompt, whether the nearest cluster follows the
  alignment with the role mean.

    uv run --extra plots python src/contol/c0_geometry.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pointbiserialr

from common import load_space, residuals, runs, save, trials, unit


def pairwise(u):
    upper = np.triu_indices(len(u), 1)
    s = (u @ u.T)[upper]
    return dict(min=s.min(), median=np.median(s), max=s.max())


def share731(x):
    return x[..., 731] ** 2 / (x**2).sum(-1)


def main() -> None:
    space = load_space()
    V, labels, clusters = space.vectors, space.labels, space.clusters
    mu = space.mean
    axis = unit(space.default - mu)
    c = V - space.default
    along = unit(c) @ -axis
    removed = unit(c - np.outer(c @ axis, axis))

    unsteered = next(t for t in trials("alpha", runs()["alpha"]) if t["character"] == "unsteered")
    seg = residuals(unsteered)
    prompt, generated = seg["prompt"][0], seg["generated"][0]
    tokens = np.vstack([prompt[1:], generated])
    rng = np.random.default_rng(0)

    sink = prompt[0]
    top = np.argsort(-np.abs(sink))[:3]
    x = prompt[1:]
    nearest = clusters[(unit(x) @ space.unit.T).argmax(1)]

    result = dict(
        roles=dict(
            cos=pairwise(space.unit),
            cos_centered=pairwise(unit(V - mu)),
            norm=dict(min=np.linalg.norm(V, axis=1).min(), max=np.linalg.norm(V, axis=1).max()),
            least_aligned_with_mean={labels[i]: float(space.unit[i] @ unit(mu))
                                     for i in np.argsort(space.unit @ unit(mu))[:3]},
        ),
        steering_vectors=dict(
            cos=pairwise(unit(c)),
            cos_with_reversed_axis_median=np.median(along),
            share_of_norm2_along_axis_median=np.median(along**2),
            cos_after_removing_axis=pairwise(removed),
            axis_alignment_by_cluster={str(k): along[clusters == k].mean() for k in range(6)},
            norm_by_cluster={str(k): np.linalg.norm(c[clusters == k], axis=1).mean() for k in range(6)},
        ),
        dim731=dict(
            share_in_role_vectors_by_cluster={str(k): share731(V[clusters == k]).mean() for k in range(6)},
            share_in_default=share731(space.default),
            value_by_cluster={str(k): V[clusters == k, 731].mean() for k in range(6)},
            token_value_mean_sd=[tokens[:, 731].mean(), tokens[:, 731].std()],
            token_share_median=np.median(share731(tokens)),
            share_after_averaging={str(n): share731(generated[rng.choice(len(generated), n, replace=False)].mean(0))
                                   for n in (5, 20, len(generated))},
        ),
        sink=dict(
            norm=np.linalg.norm(sink),
            other_token_norm_range=[np.linalg.norm(tokens, axis=1).min(), np.linalg.norm(tokens, axis=1).max()],
            top_dims=top.tolist(),
            top1_share_of_norm2=sink[top[0]] ** 2 / (sink**2).sum(),
            cos_default=float(unit(sink) @ unit(space.default)),
            cos_assistant=float(unit(sink) @ space.unit[space.index("assistant")]),
        ),
        unsteered_prompt=dict(
            nearest_cluster_counts=np.bincount(nearest, minlength=6).tolist(),
            align_mu_cluster0=(unit(x) @ unit(mu))[nearest == 0].mean(),
            align_mu_cluster3=(unit(x) @ unit(mu))[nearest == 3].mean(),
            pointbiserial_cluster3_align_mu=dict(zip(("r", "p"), pointbiserialr(nearest == 3, unit(x) @ unit(mu)))),
        ),
    )
    path = save("c0_geometry", result)
    print(path.read_text()[:2500])


if __name__ == "__main__":
    main()
