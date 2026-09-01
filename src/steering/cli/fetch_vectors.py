from __future__ import annotations

import argparse
import json

from ..recipes import (
    DefaultVector,
    PublishedAssistantAxis,
    RoleDirection,
    RoleVector,
    TraitVector,
    load_recipe,
)
from ..vector_store import VectorStore


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Fetch and inspect persona-vector artifacts"
    )
    result.add_argument("--model", default="Qwen/Qwen3-32B")
    result.add_argument("--dataset-model-dir")
    result.add_argument("--revision", default="main")
    result.add_argument("--local-dir")
    result.add_argument("--cache-dir")
    result.add_argument("--offline", action="store_true")
    group = result.add_mutually_exclusive_group()
    group.add_argument("--role")
    group.add_argument("--role-direction")
    group.add_argument("--trait")
    group.add_argument("--default-vector", action="store_true")
    group.add_argument("--assistant-axis", action="store_true")
    group.add_argument("--capping-config", action="store_true")
    group.add_argument("--recipe")
    group.add_argument("--list-roles", action="store_true")
    group.add_argument("--list-traits", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    store = VectorStore(
        model_id=args.model,
        dataset_model_dir=args.dataset_model_dir,
        revision=args.revision,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        offline=args.offline,
    )
    if args.list_roles:
        print("\n".join(store.list_roles()))
        return
    if args.list_traits:
        print("\n".join(store.list_traits()))
        return
    if args.role:
        recipe = RoleVector(args.role)
    elif args.role_direction:
        recipe = RoleDirection(args.role_direction)
    elif args.trait:
        recipe = TraitVector(args.trait)
    elif args.default_vector:
        recipe = DefaultVector()
    elif args.assistant_axis:
        recipe = PublishedAssistantAxis()
    elif args.capping_config:
        print(store.fetch("capping_config.pt"))
        return
    elif args.recipe:
        recipe = load_recipe(args.recipe)
    else:
        parser().error("select an artifact/recipe or a list operation")
    materialized = store.materialize(recipe)
    print(json.dumps(materialized.metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
