"""Pack each experiment's ``web.json`` into a compact script for the explorer page.

    uv run --no-project --with numpy python token_trend_analysis/src/build_web.py

Writes ``out/web/data/<experiment>.js`` and copies ``web/index.html`` beside them. The page (``web/index.html``) loads these
with ``<script>`` tags rather than fetch, because the artifact host's content policy
blocks fetch; each file registers itself on ``window.TREND_DATA``. Rows become
positional arrays (column names in ``ROW_COLUMNS``) to keep the persona suites small.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from experiments import EXPERIMENTS

OUT = Path(__file__).resolve().parents[1] / "out"
TRENDS = {"": 0, "fades": 1, "appears": 2}
FLAGS = {"": 0, "possible baseline": 1, "suspected baseline": 2, "broad, also unsteered": 3}
ROW_COLUMNS = ["word", "gloss", "variants", "trend", "flag", "inVector", "spreadSteered", "spreadUnsteered",
               "qFade", "qAppear", "log2fc", "agree", "onset", "rho", "permille"]


def question_text(prompt: list[str]) -> str:
    """The user message inside the chat template."""
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", "".join(prompt), re.S)
    return match.group(1).strip() if match else ""


def pack_unit(unit: dict) -> dict:
    """One analysis unit: a character over all schemes, or a character on one question."""
    q = lambda v: None if v is None else float(f"{v:.3g}")
    metrics = {}
    for metric, rows in unit["metrics"].items():
        metrics[metric] = [[r["word"], r["gloss"], [v for v in r["variants"] if v.strip().lower() != r["word"]][:4],
                            TRENDS[r["trend"]], FLAGS[r["flag"]], int(r["in_vector_readout"]),
                            r["spread_steered"], r["spread_unsteered"], q(r["q_fade"]), q(r["q_appear"]),
                            r["log2fc"], r["agree"], r["onset"], [round(v, 2) for v in r["rho"]],
                            [[round(v, 1) for v in s] for s in r["permille"]]]
                           for r in rows]
    return {
        "mode": unit["mode"], "labels": unit["labels"], "tested": unit["tested"],
        "question": unit.get("question"), "questionText": question_text(unit["prompt"]) if unit.get("prompt") else "",
        "replicates": [{"name": r["name"], "firstLayer": r["first_layer"], "grid": r["grid"],
                        "positions": r["positions"], "layers": r["layers"]} for r in unit["replicates"]],
        "baselineShare": [[round(v, 3) for v in s] for s in unit["baseline_share"]],
        "metrics": metrics,
    }


def pack(web: dict) -> dict:
    characters = [{"character": c["character"], "units": [pack_unit(u) for u in c["units"]]} for c in web["characters"]]
    return {"name": web["name"], "summary": web["summary"], "replicate": web["replicate"], "symbol": web["symbol"],
            "columns": ROW_COLUMNS, "characters": characters}


def main() -> None:
    target = OUT / "web" / "data"
    target.mkdir(parents=True, exist_ok=True)
    for experiment in EXPERIMENTS:
        web = json.loads((OUT / experiment.name / "web.json").read_text(encoding="utf-8"))
        body = json.dumps(pack(web), ensure_ascii=False, separators=(",", ":"))
        script = f"(window.TREND_DATA=window.TREND_DATA||{{}})[{json.dumps(experiment.name)}]={body};\n"
        (target / f"{experiment.name}.js").write_text(script, encoding="utf-8")
        print(experiment.name, f"{len(script.encode('utf-8')) / 1e6:.2f} MB")
    page = Path(__file__).resolve().parents[1] / "web" / "index.html"
    (OUT / "web" / "index.html").write_text(page.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
