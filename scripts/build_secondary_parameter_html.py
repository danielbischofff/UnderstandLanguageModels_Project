#!/usr/bin/env python3
"""Build offline violin comparisons for word-set contextualisation and token policy."""
import argparse
import json
from pathlib import Path

from plotly.offline import get_plotlyjs


def build(folder):
    rows = json.loads((folder / 'comparisons.json').read_text())
    template = Path(__file__).with_name('secondary_parameter_template.html').read_text()
    payload = json.dumps(rows, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    output = folder / 'secondary-parameter-evaluation.html'
    output.write_text(template.replace('/*PLOTLY*/', get_plotlyjs()).replace('/*DATA*/', payload))
    print(f'Wrote {output} ({output.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, default=Path('results/mini-batch-cluster-readout'))
    build(parser.parse_args().folder)
