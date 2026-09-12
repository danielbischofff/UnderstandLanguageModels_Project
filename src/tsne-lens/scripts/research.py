"""Bounded, reproducible parameter trials; only selected trials get plots/GIFs.

Run with the project environment and --help. Each trial retains its settings,
metrics, scalar history and coordinate checkpoints, without frame PNGs.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from openTSNE import TSNEEmbedding, affinity, initialization
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from tsne_lens.optimization import exact_kl
from tsne_lens.plots import write_outputs
from tsne_lens.projection import RoleMap
from tsne_lens.quality import embedding_quality
from tsne_lens.training_plots import export_training

BASELINE = Path("outputs/plots/tsne/roles-exp/baseline-role-map.pkl")
ROOT = Path("outputs/plots/tsne/roles-exp")


@lru_cache(maxsize=1)
def data():
    m = RoleMap.load(BASELINE)
    d = pairwise_distances(m.vectors, metric="cosine")
    np.fill_diagonal(d, 0)
    return m, d


def affinities(vectors, ps):
    single = affinity.PerplexityBasedNN(
        vectors,
        perplexity=ps[0],
        k_neighbors=len(vectors) - 1,
        metric="cosine",
        method="exact",
        n_jobs=1,
        random_state=42,
    )
    if len(ps) == 1:
        return single
    return affinity.Multiscale(perplexities=ps, knn_index=single.knn_index, n_jobs=1)


def run_trial(task):
    line, spec = task
    started = time.monotonic()
    m, distances = data()
    settings = dict(
        perplexities=[15, 90],
        seed=42,
        dof=1,
        early_exaggeration=2,
        early_iter=250,
        ramp_iter=125,
        final_exaggeration=1,
        n_iter=1000,
        min_relax=1000,
        max_relax=4000,
        tolerance=1e-6,
        theta=0,
        metric="cosine",
        initialization="pca",
    )
    settings.update(spec)
    name = settings.pop("name")
    pca = PCA(n_components=2, random_state=settings["seed"])
    initial = pca.fit_transform(m.vectors)
    scale = initial[:, 0].std() / 1e-4
    initial /= scale
    default = (
        torch.load(
            m.config["default_vector_source"], weights_only=True, map_location="cpu"
        )[m.config["layer"]]
        .float()
        .numpy()
    )
    default_pca = (pca.transform(default[None])[0] / scale).tolist()
    initialization.jitter(initial, inplace=True, random_state=settings["seed"])
    resumed = settings.get("resume")
    if resumed:
        source = ROOT / resumed
        old = json.loads(Path(str(source) + ".json").read_text())
        saved_frames = np.load(Path(str(source) + ".npz"))["coordinates"]
        initial = saved_frames[-1].copy()
        default_pca = old["default_pca"]
    e = TSNEEmbedding(
        np.ascontiguousarray(initial),
        affinities(m.vectors, settings["perplexities"]),
        dof=settings["dof"],
        negative_gradient_method="bh",
        theta=0,
        n_jobs=1,
        random_state=settings["seed"],
    )
    history, frames = [], []
    iteration = 0
    if resumed:
        history = old["history"]
        frames = list(saved_frames)
        iteration = history[-1]["iteration"]

    def snapshot(phase, exaggeration):
        history.append(
            dict(
                iteration=iteration,
                phase=phase,
                perplexities=settings["perplexities"],
                exaggeration=exaggeration,
                kl=exact_kl(e),
            )
        )
        frames.append(np.asarray(e).copy())

    def optimize(steps, phase, exag, interval=250):
        nonlocal iteration
        for start in range(0, steps, interval):
            count = min(interval, steps - start)
            e.optimize(
                count,
                exaggeration=exag,
                learning_rate="auto",
                momentum=0.8,
                inplace=True,
            )
            iteration += count
            snapshot(phase, exag)

    if not resumed:
        snapshot("pca", 1)
        optimize(settings["early_iter"], "early", settings["early_exaggeration"], 25)
        for start in range(0, settings["ramp_iter"], 25):
            count = min(25, settings["ramp_iter"] - start)
            fraction = (start + count) / settings["ramp_iter"]
            exag = settings["early_exaggeration"] + fraction * (
                settings["final_exaggeration"] - settings["early_exaggeration"]
            )
            optimize(count, "exaggeration-anneal", exag)
    else:
        snapshot("resume-momentum-reset", settings["final_exaggeration"])
    optimize(settings["n_iter"], "fit", settings["final_exaggeration"])
    stable = elapsed = 0
    previous = history[-1]["kl"]
    converged = False
    while elapsed < settings["max_relax"]:
        optimize(250, "relax", settings["final_exaggeration"])
        elapsed += 250
        delta = abs(history[-1]["kl"] - previous)
        stable = stable + 1 if delta < settings["tolerance"] else 0
        previous = history[-1]["kl"]
        if elapsed >= settings["min_relax"] and stable >= 2:
            converged = True
            break
    q = embedding_quality(distances, e, m.clusters)
    b = m.config["quality"]
    joint = float(
        np.sqrt(
            q["neighbor_recall"]
            / b["neighbor_recall"]
            * max(q["cluster_silhouette_2d"], 0)
            / b["cluster_silhouette_2d"]
        )
    )
    result = dict(
        name=name,
        line=line,
        settings=settings,
        quality=q,
        joint_score=joint,
        joint_eligible=q["neighbor_recall"] >= b["neighbor_recall"]
        and q["cluster_silhouette_2d"] >= b["cluster_silhouette_2d"],
        convergence=dict(
            converged=converged,
            iterations=iteration,
            final_kl=history[-1]["kl"],
            last_checkpoint_change=delta,
            plateau_checks=stable,
            objective_exaggeration=settings["final_exaggeration"],
            note="KL plateau under the stated fixed exaggeration; not a claim of ordinary-KL optimality",
        ),
        default_pca=default_pca,
        seconds=time.monotonic() - started,
        history=history,
    )
    directory = ROOT / line / "trials"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        directory / f"{name}.npz",
        coordinates=np.stack(frames),
        labels=np.asarray(m.labels),
    )
    return {k: v for k, v in result.items() if k not in ("history", "default_pca")}


def select(line, name):
    m, _ = data()
    trial = ROOT / line / "trials" / name
    result = json.loads(Path(str(trial) + ".json").read_text())
    frames = np.load(Path(str(trial) + ".npz"))["coordinates"]
    s = result["settings"]
    e = TSNEEmbedding(
        frames[-1].copy(),
        affinities(m.vectors, s["perplexities"]),
        dof=s.get("dof", 1),
        negative_gradient_method="bh",
        theta=0,
        n_jobs=1,
    )
    config = {
        **m.config,
        "perplexities": s["perplexities"],
        "initial_perplexities": s["perplexities"],
        "perplexity": max(s["perplexities"]),
        "affinity": "multiscale" if len(s["perplexities"]) > 1 else "single",
        "seed": s["seed"],
        "dof": s.get("dof", 1),
        "early_exaggeration": s["early_exaggeration"],
        "early_exaggeration_iter": s["early_iter"],
        "exaggeration_anneal_iter": s["ramp_iter"],
        "exaggeration": s["final_exaggeration"],
        "final_exaggeration": s["final_exaggeration"],
        "max_iter": s["max_relax"],
        "n_iter": s["n_iter"],
        "relaxation_iter": s["min_relax"],
        "quality": result["quality"],
        "training_history": result["history"],
        "convergence": {
            **result["convergence"],
            "reference_pca": result["default_pca"],
            "final_perplexities": s["perplexities"],
            "neighbors": len(m.vectors) - 1,
        },
        "research": {
            "line": line,
            "trial": name,
            "selection": "recall@10"
            if line == "recall"
            else "geometric mean of relative recall/silhouette gains; both must improve",
        },
    }
    selected = replace(m, embedding=e, config=config)
    default = (
        torch.load(
            m.config["default_vector_source"], weights_only=True, map_location="cpu"
        )[m.config["layer"]]
        .float()
        .numpy()
    )
    selected.config["reference_points"] = selected.project(
        default[None], [{"label": "default_vector"}]
    )
    out = ROOT / line
    selected.save(out / "role-map.pkl")
    write_outputs(selected, [], out, formats=("html", "png"))
    export_training(
        e,
        m.labels,
        result["history"],
        frames,
        out / "training",
        clusters=m.clusters,
        default_vector=default,
        default_pca=result["default_pca"],
        vectors=m.vectors,
    )
    (out / "selected-trial.json").write_text(json.dumps(result, indent=2) + "\n")
    print("EXPORTED", line, name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", type=Path, help="JSON list of {line, settings} trials"
    )
    parser.add_argument("--select", nargs=2, metavar=("LINE", "TRIAL"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch:
        tasks = [(r["line"], r["settings"]) for r in json.loads(args.batch.read_text())]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_trial, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps(result), flush=True)
    if args.select:
        select(*args.select)
