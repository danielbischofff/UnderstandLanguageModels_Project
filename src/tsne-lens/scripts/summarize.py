import csv
import hashlib
import json
from pathlib import Path

from tsne_lens.projection import RoleMap

root = Path("outputs/plots/tsne/roles-exp")
base = RoleMap.load(Path("outputs/plots/tsne/roles-exp/baseline-role-map.pkl"))
trials = []
for line in ["recall", "joint"]:
    for p in (root / line / "trials").glob("*.json"):
        r = json.loads(p.read_text())
        r["path"] = str(p.relative_to(root))
        trials.append(r)
resumed = {r["settings"].get("resume") for r in trials if r["settings"].get("resume")}
terminal = [r for r in trials if r["path"][:-5] not in resumed]
recall = max(
    [r for r in terminal if r["line"] == "recall"],
    key=lambda r: r["quality"]["neighbor_recall"],
)
joint = max(
    [r for r in terminal if r["line"] == "joint" and r["joint_eligible"]],
    key=lambda r: r["joint_score"],
)
selected = {"recall": recall["name"], "joint": joint["name"]}
(root / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
flat = []
for r in trials:
    flat.append(
        {
            "line": r["line"],
            "trial": r["name"],
            "perplexities": str(r["settings"]["perplexities"]),
            "dof": r["settings"].get("dof", 1),
            "seed": r["settings"]["seed"],
            "early_exaggeration": r["settings"]["early_exaggeration"],
            "final_exaggeration": r["settings"]["final_exaggeration"],
            **r["quality"],
            "joint_score": r["joint_score"],
            "joint_eligible": r["joint_eligible"],
            "converged": r["convergence"]["converged"],
            "iterations": r["convergence"]["iterations"],
            "seconds": r["seconds"],
        }
    )
with (root / "trials.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(flat[0]))
    w.writeheader()
    w.writerows(flat)
pareto = [
    r
    for r in terminal
    if not any(
        s["quality"]["neighbor_recall"] >= r["quality"]["neighbor_recall"]
        and s["quality"]["cluster_silhouette_2d"]
        >= r["quality"]["cluster_silhouette_2d"]
        and (
            s["quality"]["neighbor_recall"] > r["quality"]["neighbor_recall"]
            or s["quality"]["cluster_silhouette_2d"]
            > r["quality"]["cluster_silhouette_2d"]
        )
        for s in terminal
    )
]
summary = {
    "trial_count": len(trials),
    "terminal_trial_count": len(terminal),
    "baseline_quality": base.config["quality"],
    "selected": {
        line: {k: v for k, v in r.items() if k not in ["history", "default_pca"]}
        for line, r in [("recall", recall), ("joint", joint)]
    },
    "pareto_trials": [r["path"] for r in pareto],
    "selection_rule": "Recall: maximum terminal-trial recall@10. Joint: geometric mean of relative recall and silhouette gains, both above baseline. Resumed ancestors are excluded. KL is not a ranking criterion.",
    "limits": "In-sample parameter selection on these 275 roles; not a held-out validation. Heavy-tailed winners may not meet the strict KL tolerance; see convergence status.",
    "source": "src/tsne-lens/scripts/research.py",
    "baseline_sha256": hashlib.sha256(
        Path("outputs/plots/tsne/roles-exp/baseline-role-map.pkl").read_bytes()
    ).hexdigest(),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps(
        {
            **selected,
            "n": len(trials),
            "metrics": {
                line: r["quality"] for line, r in [("recall", recall), ("joint", joint)]
            },
        },
        indent=2,
    )
)
