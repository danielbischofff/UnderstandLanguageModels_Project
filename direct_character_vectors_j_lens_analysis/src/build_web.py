"""Inline the readout into the viewer, producing a self-contained page.

``web/template.html`` carries the markup with a ``__DATA__`` placeholder; this
writes ``out/index.html`` with the JSON embedded, so the page can be opened
straight from disk (a ``fetch`` of a sibling file is blocked under ``file://``)
or published as-is.

Usage::

    python src/build_web.py [--data out/readout.json] [--out out/index.html]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = "__DATA__"


def build(template: Path, data: Path, out: Path) -> Path:
    page = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in page:
        raise ValueError(f"{template} has no {PLACEHOLDER} placeholder")
    # The JSON lives inside a <script> element, so no literal '<' may survive
    # in it -- a token containing "</script>" would otherwise close the tag.
    # < is valid JSON and parses back to '<'.
    payload = data.read_text(encoding="utf-8").replace("<", '\\u003c')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace(PLACEHOLDER, payload), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", type=Path, default=ROOT / "web" / "template.html")
    parser.add_argument("--data", type=Path, default=ROOT / "out" / "readout.json")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "index.html")
    args = parser.parse_args()

    out = build(args.template, args.data, args.out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
