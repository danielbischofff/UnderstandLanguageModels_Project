#!/usr/bin/env python3
"""Build the standalone paired-difference comparison across questions."""
import argparse
import json
from pathlib import Path
from plotly.offline import get_plotlyjs


def build(folder):
    source = json.loads((folder / 'question-aggregation.json').read_text())
    data = {'questions': source['questions'], 'rows': source['per_question']}
    template = Path(__file__).with_name('question_comparison_template.html').read_text()
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    output = folder / 'question-comparison.html'
    output.write_text(template.replace('/*PLOTLY*/', get_plotlyjs()).replace('/*DATA*/', payload))
    print(f'Wrote {output} ({output.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, default=Path('results/mini-batch-cluster-readout'))
    build(parser.parse_args().folder)
