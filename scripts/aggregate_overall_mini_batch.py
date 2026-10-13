#!/usr/bin/env python3
"""Equal-persona, equal-question effects with a shared-question bootstrap."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from aggregate_mini_batch import METRICS

KEYS = ('layer', 'scheme', 'beta', 'word_set_type', 'policy')


def summarize(rows, questions, personas, samples=10000, seed=20261007):
    lookup = {(r['question'], r['character']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate question/persona pair')
    if set(lookup) != {(q, p) for q in questions for p in personas}:
        raise ValueError('Incomplete question/persona grid; refusing to change weights')
    indices = np.random.default_rng(seed).integers(0, len(questions), (samples, len(questions)))
    result = {}
    for name, (value, delta) in METRICS.items():
        matrix = []
        values = []
        for q in questions:
            q_deltas, q_values = [], []
            for p in personas:
                row = lookup[q, p]
                if not row['control_available'] or row.get(delta) is None or row.get(value) is None:
                    raise ValueError(f'Missing paired metric: {name}, {q}, {p}')
                q_deltas.append(row[delta])
                q_values.append(row[value])
            matrix.append(q_deltas)
            values.append(q_values)
        matrix, values = np.array(matrix), np.array(values)
        if not np.isfinite(matrix).all() or not np.isfinite(values).all():
            raise ValueError('Nonfinite paired metric')
        persona_means = matrix.mean(axis=0)
        # Each sampled question retains all eight personas together.
        question_means = matrix.mean(axis=1)
        interval = np.quantile(question_means[indices].mean(axis=1), [.025, .975])
        result[name] = dict(mean_delta=float(persona_means.mean()), ci95=interval.tolist(),
                            mean_steered=float(values.mean()), mean_unsteered=float((values-matrix).mean()),
                            positive_personas=int((persona_means > 0).sum()),
                            negative_personas=int((persona_means < 0).sum()),
                            zero_personas=int((persona_means == 0).sum()),
                            positive_pairs=int((matrix > 0).sum()), n_pairs=int(matrix.size),
                            persona_mean_deltas=dict(zip(personas, persona_means.tolist())),
                            question_mean_deltas=dict(zip(questions, question_means.tolist())))
    return result


def build(source, output, samples=10000):
    data = json.loads(source.read_text())
    rows = data['per_question']
    questions = sorted(q['id'] for q in data['questions'])
    personas = sorted({r['character'] for r in rows})
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in KEYS)].append(row)
    conditions = [dict(zip(KEYS, key), metrics=summarize(group, questions, personas, samples))
                  for key, group in sorted(groups.items())]
    result = dict(method={
        'aggregation': 'Mean paired steered-minus-unsteered difference over questions within each persona, then equal mean across personas. Configurations are kept separate.',
        'bootstrap': {'unit': 'question ID shared across all personas', 'samples': samples, 'seed': 20261007,
                      'interval': '95% percentile interval; resample 12 question IDs with replacement and retain all eight persona pairs for each sampled question'},
        'scope': 'Descriptive association for the selected eight personas and twelve questions, not literal word occurrence. Persona-specific word banks differ; their scores receive equal weight.',
        'specificity': data['method']['specificity'],
        'limitations': 'Exploratory question-composition intervals for a small, deliberately selected set, not population-wide inference. Personas are fixed, not resampled. No multiple-comparison correction.',
        'missing': 'Require a complete finite paired grid for every metric; fail rather than silently alter weights.',
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
        questions=questions, personas=personas, conditions=conditions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    flat = []
    for condition in conditions:
        row = {k: condition[k] for k in KEYS}
        row.update(n_questions=len(questions), n_personas=len(personas))
        for metric, stats in condition['metrics'].items():
            for field in ('mean_delta', 'mean_steered', 'mean_unsteered', 'positive_personas', 'negative_personas', 'zero_personas', 'positive_pairs', 'n_pairs'):
                row[f'{metric}_{field}'] = stats[field]
            row[f'{metric}_ci95_low'], row[f'{metric}_ci95_high'] = stats['ci95']
        flat.append(row)
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    template = Path(__file__).with_name('overall_aggregation_template.html').read_text()
    template = template.replace('/*SORTING*/', Path(__file__).with_name('mini_batch_table_sort.js').read_text())
    payload = json.dumps(result, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    output.with_suffix('.html').write_text(template.replace('/*DATA*/', payload))
    print(f'Created JSON, CSV and HTML: {len(conditions)} configurations, {len(personas)} personas, {len(questions)} questions each')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('results/mini-batch-cluster-readout/question-aggregation.json'))
    parser.add_argument('--output', type=Path, default=Path('results/mini-batch-cluster-readout/overall-aggregation.json'))
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error('bootstrap-samples must be positive')
    build(args.source, args.output, args.bootstrap_samples)
