"""Fit a role t-SNE map and overlay token residuals from local experiment files."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--roles",
        type=Path,
        help="role_vectors directory (*.pt) or NPZ vectors + labels",
    )
    source.add_argument(
        "--map", type=Path, help="previously fitted trusted role-map.pkl (no refit)"
    )
    parser.add_argument(
        "--layer", type=int, help="zero-based post-MLP block index (fit default: 32)"
    )
    parser.add_argument(
        "--model-id", help="model identity (fit default: Qwen/Qwen3-32B)"
    )
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument(
        "--perplexity",
        type=float,
        nargs="+",
        default=None,
        help="initial scale(s); omit for the tuned 90 to 30 annealing schedule",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-exaggeration", type=float, default=2)
    parser.add_argument("--early-exaggeration-iter", type=int, default=250)
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=1,
        help="attraction during the main optimization phase",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="iterations at each fixed-affinity stage",
    )
    parser.add_argument(
        "--relaxation-iter",
        type=int,
        default=1000,
        help="minimum final ordinary-KL iterations before accepting a plateau",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0,
        help="Barnes-Hut approximation (0 uses exact repulsion)",
    )
    parser.add_argument(
        "--anneal-perplexity",
        type=float,
        nargs="+",
        default=None,
        help="successively lower perplexities after each fitting stage",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="maximum final ordinary-KL iterations before reporting nonconvergence",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="absolute KL change per 250 iterations for two consecutive checks",
    )
    parser.add_argument(
        "--clusters", type=int, default=3, help="broad Ward hierarchy cut"
    )
    parser.add_argument(
        "--subclusters", type=int, default=6, help="finer Ward hierarchy cut"
    )
    parser.add_argument(
        "--training-gif",
        action="store_true",
        help="save checkpoint PNGs, KL history and GIF",
    )
    parser.add_argument(
        "--default-vector", type=Path, help="held-out reference .pt tensor to highlight"
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        default=[],
        help="repeat for each trial directory, activation manifest, or NPZ",
    )
    parser.add_argument(
        "--name", action="append", help="one display name per --trajectory, in order"
    )
    parser.add_argument(
        "--point",
        choices=["pre_intervention", "post_intervention"],
        default="post_intervention",
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument(
        "--generated-only", action="store_true", help="omit prompt positions"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--transform-perplexity", type=float, default=5)
    parser.add_argument(
        "--formats", nargs="+", choices=["html", "png"], default=["html"]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/plots/residual-tsne")
    )
    args = parser.parse_args()
    if args.map and (args.training_gif or args.default_vector):
        parser.error("--training-gif and --default-vector require fitting with --roles")
    if args.name is not None and len(args.name) != len(args.trajectory):
        parser.error("provide exactly one --name per --trajectory")
    if args.batch_index < 0:
        parser.error("--batch-index must be nonnegative")
    try:
        from ..tsne_lens.inputs import load_roles, load_trajectory
        from ..tsne_lens.plots import write_outputs
        from ..tsne_lens.projection import RoleMap
    except ImportError as error:
        parser.error(
            f"install plotting dependencies with `uv sync --extra plots`: {error}"
        )
    try:
        if args.map:
            role_map = RoleMap.load(args.map)
            if args.layer is not None and args.layer != role_map.config["layer"]:
                raise ValueError("--layer conflicts with saved map")
            if (
                args.model_id is not None
                and args.model_id != role_map.config["model_id"]
            ):
                raise ValueError("--model-id conflicts with saved map")
        else:
            layer = args.layer if args.layer is not None else 32
            model_id = args.model_id or "Qwen/Qwen3-32B"
            vectors, labels = load_roles(args.roles, layer, model_id)
            default_vector = None
            if args.default_vector:
                import torch

                from ..vector_store import _extract_tensor

                default_vector = (
                    _extract_tensor(
                        torch.load(
                            args.default_vector, map_location="cpu", weights_only=True
                        ),
                        args.default_vector,
                    )[layer]
                    .float()
                    .numpy()
                )
            role_map = RoleMap.fit(
                vectors,
                labels,
                layer=layer,
                model_id=model_id,
                metric=args.metric,
                perplexity=args.perplexity,
                seed=args.seed,
                n_clusters=args.clusters,
                subclusters=args.subclusters,
                anneal_perplexities=args.anneal_perplexity,
                max_iter=args.max_iter,
                tolerance=args.tolerance,
                training_output=args.output / "training" if args.training_gif else None,
                default_vector=default_vector,
                early_exaggeration=args.early_exaggeration,
                early_exaggeration_iter=args.early_exaggeration_iter,
                exaggeration=args.exaggeration,
                n_iter=args.n_iter,
                relaxation_iter=args.relaxation_iter,
                theta=args.theta,
            )
            role_map.config.update(
                source=str(args.roles.resolve()),
                versions={
                    p: version(p)
                    for p in ("openTSNE", "scikit-learn", "numpy", "plotly")
                },
            )
            if args.default_vector:
                role_map.config["default_vector_source"] = str(
                    args.default_vector.resolve()
                )
        trajectories = []
        for i, path in enumerate(args.trajectory):
            vectors, rows = load_trajectory(
                path,
                layer=role_map.config["layer"],
                model_id=role_map.config["model_id"],
                point=args.point,
                batch_index=args.batch_index,
                generated_only=args.generated_only,
            )
            points = role_map.project(
                vectors, rows, top_k=args.top_k, perplexity=args.transform_perplexity
            )
            trajectories.append(
                dict(
                    name=args.name[i] if args.name else path.stem,
                    source=str(path.resolve()),
                    points=points,
                    transform_perplexity=min(
                        args.transform_perplexity, role_map.config["perplexity"]
                    ),
                )
            )
        args.output.mkdir(parents=True, exist_ok=True)
        role_map.save(args.output / "role-map.pkl")
        if not role_map.config.get("convergence", {}).get("converged", True):
            print(
                "Warning: final KL did not plateau within --max-iter; see convergence metadata."
            )
        write_outputs(role_map, trajectories, args.output, args.formats)
    except (ValueError, FileNotFoundError, IndexError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(
        f"Wrote {len(role_map.labels)} roles and {len(trajectories)} trajectories to {args.output}"
    )


if __name__ == "__main__":
    main()
