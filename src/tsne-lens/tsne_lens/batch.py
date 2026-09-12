"""Project completed local experiment trials separately onto one fixed role map."""

import json
from html import escape
from pathlib import Path

from .inputs import load_trajectory
from .plots import write_outputs


def project_run(role_map, run: Path, output: Path, *, formats=("html",),
                point="post_intervention", batch_index=0, generated_only=False,
                prompt_only=False, top_k=5, perplexity=5):
    manifest = json.loads((run / "manifest.json").read_text())
    trials = sorted(manifest.get("trials", {}).values(), key=lambda t: t["index"])
    complete = [t for t in trials if t["status"] == "complete"]
    if not complete:
        raise ValueError(f"no completed trials in {run}")
    # Preflight every source before doing any projection; missing tensors must
    # never silently turn an experiment matrix into a partial comparison.
    sources = []
    for trial in complete:
        source = (run / trial["path"]).resolve()
        if not source.is_relative_to(run.resolve()):
            raise ValueError("trial path escapes the experiment directory")
        if not (source / "activations" / "manifest.json").is_file():
            raise ValueError(f"missing activations in {source}; results.zip does not contain tensors")
        sources.append(source)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for trial, source in zip(complete, sources):
        steering = trial.get("steering")
        name = str(trial["character"])
        if steering:
            name += f" · {steering.get('name', steering.get('method', 'additive'))} · {steering['coefficient']}"
        vectors, rows = load_trajectory(
            source, layer=role_map.config["layer"], model_id=role_map.config["model_id"],
            point=point, batch_index=batch_index, generated_only=generated_only,
            prompt_only=prompt_only,
        )
        points = role_map.project(vectors, rows, top_k=top_k, perplexity=perplexity)
        destination = output / source.name
        trajectory = dict(name=name, source=str(source), points=points,
                          transform_perplexity=min(perplexity, role_map.config["perplexity"]))
        write_outputs(role_map, [trajectory], destination, formats)
        entries.append(dict(name=name, directory=source.name, tokens=len(rows),
                            character=trial["character"], steering=steering))
        print(f"Projected {name}: {len(rows)} tokens → {destination}", flush=True)
    summary = dict(source=str(run.resolve()), layer=role_map.config["layer"], point=point,
                   segment="prompt" if prompt_only else "generated" if generated_only else "all",
                   skipped_incomplete=len(trials) - len(complete), trials=entries)
    (output / "index.json").write_text(json.dumps(summary, indent=2) + "\n")
    items = []
    for entry in entries:
        links = [f'<a href="{escape(entry["directory"])}/roles.{fmt}">{fmt.upper()}</a>'
                 for fmt in formats]
        items.append(f'<li>{escape(entry["name"])} ({entry["tokens"]} tokens): {" · ".join(links)}</li>')
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Residual trajectories</title>'
        '<h1>Residual trajectories</h1>'
        f'<p>Fixed layer {summary["layer"]} role map · {escape(point)} · {summary["segment"]} tokens. '
        f'Skipped {summary["skipped_incomplete"]} incomplete trials.</p><ul>'
        + "\n".join(items) + '</ul></html>', encoding="utf-8",
    )
    return summary
