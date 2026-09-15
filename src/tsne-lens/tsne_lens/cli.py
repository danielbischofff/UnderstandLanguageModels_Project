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
        help="fixed affinity scale(s); default: combined 15, 25 and 90",
    )
    parser.add_argument(
        "--dof",
        type=float,
        default=0.7,
        help="kernel tail parameter; values below 1 give heavier tails",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-exaggeration", type=float, default=2)
    parser.add_argument("--early-exaggeration-iter", type=int, default=250)
    parser.add_argument(
        "--exaggeration-anneal-iter",
        type=int,
        default=125,
        help="linear early-to-main exaggeration ramp in 25-iteration steps; 0 disables",
    )
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
        help="iterations of main optimization with fixed affinities",
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
        "--max-iter",
        type=int,
        default=2000,
        help="maximum final relaxation iterations (about 3,375 total by default)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="absolute KL change per 250 iterations for two consecutive checks",
    )
    parser.add_argument(
        "--clusters", type=int, default=6, help="broad Ward hierarchy cut"
    )
    parser.add_argument(
        "--subclusters", type=int, default=6, help="finer Ward hierarchy cut"
    )
    parser.add_argument(
        "--training-gif",
        action="store_true",
        help="save checkpoint coordinates, KL history and adaptive-scale GIF",
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
    parser.add_argument("--run", type=Path, help="experiment run directory; one plot per completed trial (requires --map)")
    parser.add_argument(
        "--point",
        choices=["pre_intervention", "post_intervention"],
        default="post_intervention",
    )
    parser.add_argument("--batch-index", type=int, default=0)
    segment = parser.add_mutually_exclusive_group()
    segment.add_argument(
        "--generated-only", action="store_true", help="omit prompt positions"
    )
    segment.add_argument("--prompt-only", action="store_true", help="only prompt input tokens, including chat-template tokens")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--transform-perplexity", type=float, default=5)
    parser.add_argument(
        "--formats", nargs="+", choices=["html", "png"], default=["html"]
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/plots/tsne/roles"))
    args = parser.parse_args()
    if args.run and (not args.map or args.trajectory or args.name):
        parser.error("--run requires --map and cannot be combined with --trajectory or --name")
    if args.map and (args.training_gif or args.default_vector):
        parser.error("--training-gif and --default-vector require fitting with --roles")
    if args.name is not None and len(args.name) != len(args.trajectory):
        parser.error("provide exactly one --name per --trajectory")
    if args.batch_index < 0:
        parser.error("--batch-index must be nonnegative")
    try:
        from .inputs import load_roles, load_trajectory
        from .plots import write_outputs
        from .projection import RoleMap
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

                from .inputs import extract_tensor

                default_vector = (
                    extract_tensor(
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
                dof=args.dof,
                seed=args.seed,
                n_clusters=args.clusters,
                subclusters=args.subclusters,
                max_iter=args.max_iter,
                tolerance=args.tolerance,
                training_output=args.output / "training" if args.training_gif else None,
                default_vector=default_vector,
                early_exaggeration=args.early_exaggeration,
                early_exaggeration_iter=args.early_exaggeration_iter,
                exaggeration_anneal_iter=args.exaggeration_anneal_iter,
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
        if args.run:
            from .batch import project_run

            project_run(
                role_map, args.run, args.output, formats=args.formats,
                point=args.point, batch_index=args.batch_index,
                generated_only=args.generated_only, prompt_only=args.prompt_only,
                top_k=args.top_k, perplexity=args.transform_perplexity,
            )
            print(f"Open {args.output / 'index.html'}")
            return
        trajectories = []
        for i, path in enumerate(args.trajectory):
            vectors, rows = load_trajectory(
                path,
                layer=role_map.config["layer"],
                model_id=role_map.config["model_id"],
                point=args.point,
                batch_index=args.batch_index,
                generated_only=args.generated_only,
                prompt_only=args.prompt_only,
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
