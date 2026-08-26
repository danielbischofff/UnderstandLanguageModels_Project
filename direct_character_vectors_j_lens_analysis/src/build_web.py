"""Inline a readout into its viewer, producing a self-contained page.

Each template in ``web/`` carries the markup with a ``__DATA__`` placeholder;
this writes the corresponding page in ``out/`` with the JSON embedded, so it can
be opened straight from disk (a ``fetch`` of a sibling file is blocked under
``file://``) or published as-is.

Usage::

    python src/build_web.py                     # the single-character page
    python src/build_web.py --page atlas        # the whole-cast atlas
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = "__DATA__"
BACKSLASH = chr(92)

#: name -> (template, readout, output page)
PAGES = {
    "role": ("template.html", "readout.json", "index.html"),
    "atlas": ("atlas_template.html", "atlas.json", "atlas.html"),
}


def ascii_safe(page: str) -> str:
    """Escape non-ASCII so the page survives any assumed charset.

    A static file server that omits ``charset=utf-8`` renders UTF-8 bytes as
    mojibake, and a ``<meta charset>`` is only honoured if it lands inside the
    encoding pre-scan window. Escaping sidesteps both: HTML text takes numeric
    character references, and the behaviour script -- everything from the last
    ``<script>`` tag -- takes JavaScript escapes, which are valid inside string
    literals and regexes alike. Idempotent on an already-ASCII page.
    """
    cut = page.rindex("<script>")
    head, script = page[:cut], page[cut:]
    head = "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in head)
    script = "".join(
        c if ord(c) < 128 else BACKSLASH + "u%04x" % ord(c) for c in script
    )
    return head + script


def build(template: Path, data: Path, out: Path) -> Path:
    page = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in page:
        raise ValueError(f"{template} has no {PLACEHOLDER} placeholder")
    # The JSON sits inside a <script> element, so no literal '<' may survive in
    # it -- a token containing "</script>" would close the tag early. The
    # escape is valid JSON and parses back to '<'.
    payload = data.read_text(encoding="utf-8").replace("<", BACKSLASH + "u003c")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ascii_safe(page.replace(PLACEHOLDER, payload)), encoding="ascii")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--page", choices=sorted(PAGES), default="role")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    template, data, page = PAGES[args.page]
    out = build(
        args.template or ROOT / "web" / template,
        args.data or ROOT / "out" / data,
        args.out or ROOT / "out" / page,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
