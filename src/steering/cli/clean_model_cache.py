"""Safely inspect or remove one model repository from the Hugging Face cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound

from ..inference import DEFAULT_MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=os.environ.get("STEERING_MODEL_ID", DEFAULT_MODEL_ID)
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face hub cache; defaults to $HF_HUB_CACHE or $HF_HOME/hub",
    )
    parser.add_argument(
        "--yes", action="store_true", help="execute deletion instead of a dry run"
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = Path(
            os.environ.get(
                "HF_HUB_CACHE",
                Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
                / "hub",
            )
        )
    try:
        cache = scan_cache_dir(cache_dir)
    except CacheNotFound:
        print(f"no Hugging Face cache found under {cache_dir}")
        return
    repositories = [
        repo
        for repo in cache.repos
        if repo.repo_type == "model" and repo.repo_id == args.model
    ]
    if not repositories:
        print(f"no cached model found for {args.model!r} under {cache_dir}")
        return
    revisions = [revision for repo in repositories for revision in repo.revisions]
    hashes = [revision.commit_hash for revision in revisions]
    strategy = cache.delete_revisions(*hashes)
    print(f"model: {args.model}")
    print(f"cache: {cache_dir}")
    print(f"revisions: {', '.join(hashes)}")
    print(f"space recoverable: {strategy.expected_freed_size_str}")
    if not args.yes:
        print("dry run only; repeat with --yes to delete this model")
        return
    strategy.execute()
    print("model cache deleted")
