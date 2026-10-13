#!/usr/bin/env python3
"""Equal-question paired aggregation of the completed cluster comparison scores."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

KEYS = ('character', 'bank', 'layer', 'scheme', 'beta', 'word_set_type', 'policy')
METRICS = {
    'mrr': ('mean_reciprocal_rank', 'mean_reciprocal_rank_delta'),
    'cosine': ('activation_word_cosine', 'activation_word_cosine_delta'),
    'mrr_margin': ('mean_reciprocal_rank_persona_margin', 'mean_reciprocal_rank_margin_delta'),
    'cosine_margin': ('activation_word_cosine_persona_margin', 'activation_word_cosine_margin_delta'),
}


def aggregate_metric(rows, value_field, delta_field, samples=10000, seed=20261007):
    paired = [r for r in rows if r['control_available'] and r.get(value_field) is not None
              and r.get(delta_field) is not None]
    if not paired:
        return dict(n=0, mean_delta=None, median_delta=None, mean_steered=None,
                    mean_unsteered=None, positive=0, negative=0, zero=0, ci95=None)
    deltas = np.array([r[delta_field] for r in paired], dtype=float)
    values = np.array([r[value_field] for r in paired], dtype=float)
    if not np.isfinite(deltas).all() or not np.isfinite(values).all():
        raise ValueError('Nonfinite paired scores')
    # Resample questions, not tokens; the paired difference keeps control together.
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(samples, len(paired)))
    ci = np.quantile(deltas[indices].mean(axis=1), [.025, .975]).tolist()
    return dict(n=len(paired), mean_delta=float(deltas.mean()), median_delta=float(np.median(deltas)),
                mean_steered=float(values.mean()), mean_unsteered=float((values - deltas).mean()),
                positive=int((deltas > 0).sum()), negative=int((deltas < 0).sum()), zero=int((deltas == 0).sum()),
                ci95=ci)


def build(source, run, output, samples=10000, seed=20261007, questions_file=None):
    rows = json.loads(source.read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in KEYS)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: r['question'])
        questions = [r['question'] for r in group]
        if len(set(questions)) != len(questions):
            raise ValueError(f'Duplicate paired question: {key}')
        summaries.append(dict(zip(KEYS, key), questions=questions,
                              metrics={name: aggregate_metric(group, *fields, samples, seed)
                                       for name, fields in METRICS.items()}))
    questions = (json.loads(questions_file.read_text()) if questions_file is not None else
                 json.loads((run / 'index.json').read_text())['document']['experiment']['questions'])
    result = {
        'method': {
            'question': 'Does steering increase J-Lens association with the intended persona word bank?',
            'scope': 'Association in the readout, not literal word occurrence in generated text.',
            'aggregation': 'Mean over retained words and answer positions within each answer; paired steering-minus-unsteered differences; equal mean over all 12 questions.',
            'specificity': 'Own-persona score minus the mean of seven rival personas in the same bank; paired difference of these margins.',
            'bootstrap': {'unit': 'whole paired question', 'samples': samples, 'seed': seed,
                          'interval': '95% percentile interval for the mean paired difference'},
            'interpretation': 'Exploratory intervals describe variability across the observed questions. They are not adjusted for multiple comparisons and do not establish generalization to all prompts.',
            'missing': 'Missing pairs omitted per metric, never replaced by zero; n is reported.',
            'positions': 'Final generated token omitted because its input activation was not cached; short answers contribute one question-level mean, just like long answers.',
            'baseline': 'Absolute unsteered values reconstructed from the completed paired score and delta; subtraction may affect final floating-point digits.',
            'calibration': 'All questions included, including Q01 and Q23.',
            'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        'questions': questions, 'conditions': summaries, 'per_question': rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    template = Path(__file__).with_name('mini_batch_aggregation_template.html').read_text()
    payload = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    html = output.with_suffix('.html')
    html.write_text(template.replace('/*PLOTLY*/', get_plotlyjs()).replace('/*DATA*/', payload))
    print(f'Created {output} and {html}: {len(summaries)} conditions, {len(rows)} paired question rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('results/mini-batch-cluster-readout/comparisons.json'))
    parser.add_argument('--run', type=Path, default=Path('outputs/mini_batch_exp_results'))
    parser.add_argument('--output', type=Path, default=Path('results/mini-batch-cluster-readout/question-aggregation.json'))
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--questions', type=Path, help='Standalone question metadata; avoids requiring the raw run index')
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error('--bootstrap-samples must be positive')
    build(args.source, args.run, args.output, samples=args.bootstrap_samples, questions_file=args.questions)
