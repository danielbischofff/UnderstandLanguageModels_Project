"""The experiments analysed, and how each is arranged into strength ladders.

A *ladder* is one series of trials ordered from weakest to strongest intervention,
always starting at the unsteered control for the same question. Trends are tested per
*replicate* and combined:

- Tylenol sweeps (``replicate="scheme"``): one question, several steering schemes
  (layers or layer blocks); each scheme's ladder is a replicate.
- Persona suites (``replicate="question"``): every question is analysed on its own
  as a single 3-4 point ladder, so there is nothing to replicate over; its trends are
  described (monotone, at least 2x) rather than tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from load import Trial

ROOT = Path(__file__).resolve().parents[2]
# The persona suites are GitHub release assets, not committed; see results/README.md.
MAIN_CHECKOUT = ROOT.parents[2] if ROOT.parent.name == "worktrees" else ROOT
SUITES = MAIN_CHECKOUT / "runtime/cluster-results/compact"


@dataclass(frozen=True)
class Experiment:
    name: str
    bundle: Path
    replicate: str        # "scheme" or "question" (see module docstring)
    symbol: str           # "α" (additive) or "β" (character band half-width)
    tighter_is_stronger: bool   # beta: a narrower band pins the projection harder
    min_agree: float      # fraction of schemes whose trend sign must match (unused per question)
    summary: str


EXPERIMENTS = [
    Experiment(
        "tylenol-demon-8000mg-single-layer-sweep",
        ROOT / "results/tylenol-demon-8000mg-single-layer-sweep/0001_20260908T191518Z_cbab895d841987cf/results.zip",
        "scheme", "α", False, 1.0,
        "Additive demon steering at one layer (29, 32 or 35), α 1.5–4.0, on the 8000 mg Tylenol prompt."),
    Experiment(
        "tylenol-demon-8000mg-band-sweep",
        ROOT / "results/tylenol-demon-8000mg-band-sweep/0001_20260908T192452Z_f7545bd14c9e846f/results.zip",
        "scheme", "β", True, 1.0,
        "Demon character-band capping (projection clamped to c ± β·s) on layers 28–35, 46–53 or 28–53, "
        "β 0.5 → 0, on the 8000 mg Tylenol prompt."),
    Experiment(
        "persona-viewpoint-suite-band",
        SUITES / "persona-viewpoint-suite-band/0001_20260909T070109Z_4d392bcaab3d5dda/results.zip",
        "question", "β", True, 1.0,
        "Character-band capping on layers 28–35 for demon, criminal and cynic, β 0.25 → 0; each of the 14 questions analysed separately."),
    Experiment(
        "persona-viewpoint-suite-alpha",
        SUITES / "persona-viewpoint-suite-alpha/0001_20260909T070514Z_b9bad78193a71dc4/results.zip",
        "question", "α", False, 1.0,
        "Additive steering at layer 32 for demon, criminal and cynic, α 1.5 and 4.0; each of the 14 questions analysed separately."),
]


@dataclass
class Ladder:
    replicate: str        # scheme name, or the question id for per-question ladders
    first_layer: int
    labels: list[str]     # "unsteered", then e.g. "β=0.5" ... "β=0", weakest to strongest
    trials: list[Trial]


def ladders(experiment: Experiment, trials: list[Trial], character: str) -> list[Ladder]:
    """One ladder per scheme (Tylenol) or per question (suites) for a steered character, control first."""
    controls = {t.question: t for t in trials if t.character == "unsteered"}
    steered = [t for t in trials if t.character == character]
    key = (lambda t: t.scheme) if experiment.replicate == "scheme" else (lambda t: t.question)
    out = []
    for replicate in dict.fromkeys(key(t) for t in steered):
        group = sorted((t for t in steered if key(t) == replicate),
                       key=lambda t: -t.coefficient if experiment.tighter_is_stronger else t.coefficient)
        out.append(Ladder(
            replicate=replicate,
            first_layer=min(group[0].layers),
            labels=["unsteered"] + [f"{experiment.symbol}={t.coefficient:g}" for t in group],
            trials=[controls[group[0].question]] + group,
        ))
    if len({tuple(l.labels) for l in out}) != 1:
        raise ValueError("replicates do not share one strength ladder")
    return out
