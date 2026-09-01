"""Quick J-Lens token readout for a recorded steering activation artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..j_lens import quick_j_lens_tokens


def main() -> None:
    repository = Path(__file__).resolve().parents[3]
    analysis_root = repository / "direct_character_vectors_j_lens_analysis"
    runtime = Path(os.environ.get("ULM_RUNTIME_ROOT", repository / "runtime"))
    default_data = Path(
        os.environ.get("JLENS_DATA_ROOT", runtime / "models" / "j-lens" / "qwen3-32b")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "activation_artifact",
        type=Path,
        help="trial activations directory or its manifest.json",
    )
    parser.add_argument("--layer", type=int)
    parser.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--forward-index", type=int)
    parser.add_argument(
        "--token-index",
        type=int,
        default=-1,
        help="selected sequence position inside the captured event (default: last)",
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument(
        "--point",
        action="append",
        choices=["pre_intervention", "post_intervention"],
        help="readout boundary (repeatable; default: both plus post-minus-pre)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--analysis-root", type=Path, default=analysis_root)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--lens", type=Path)
    parser.add_argument("--output", type=Path, help="also write the JSON result")
    args = parser.parse_args()
    lens_path = args.lens or (
        args.data_dir / "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
    )
    try:
        result = quick_j_lens_tokens(
            args.activation_artifact,
            analysis_source=args.analysis_root / "src",
            data_dir=args.data_dir,
            lens_path=lens_path,
            layer=args.layer,
            phase=args.phase,
            forward_index=args.forward_index,
            token_index=args.token_index,
            batch_index=args.batch_index,
            points=args.point or ("pre_intervention", "post_intervention"),
            top_k_count=args.top_k,
        )
    except FileNotFoundError as error:
        parser.error(
            f"missing J-Lens input {error.filename or error}; fetch the analysis "
            "assets as documented in the First Bard steering sandbox"
        )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
