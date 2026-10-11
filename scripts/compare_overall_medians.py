#!/usr/bin/env python3
"""Experimental mean/median comparison with joint question resampling."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from aggregate_mini_batch import METRICS
from aggregate_overall_mini_batch import KEYS


def median_statistics(matrix, indices):
    sampled = matrix[indices]
    estimates = {
        'median_pairs': (float(np.median(matrix)), np.median(sampled, axis=(1, 2))),
        'median_persona_means': (float(np.median(matrix.mean(axis=0))), np.median(sampled.mean(axis=1), axis=1)),
    }
    return {name: dict(estimate=estimate, ci95=np.quantile(draws, [.025, .975]).tolist())
            for name, (estimate, draws) in estimates.items()}


def render_html(result, folder):
    template = Path(__file__).with_name('overall_aggregation_template.html').read_text()
    replacements = {
        '<title>Overall J-Lens effects</title>': '<title>Experimental mean versus median comparison</title>',
        '<h1>Overall effect across eight personas</h1>': '<h1>Experimental comparison: mean versus median</h1>',
        '<p>Each cell is the mean paired <b>steered − unsteered</b> difference: average all twelve questions within each persona, then give all eight personas equal weight. Each row is a separate configuration (96 persona–question pairs).</p>':
            '<p>The median is also an aggregation. Here you can compare three summaries of the same 96 paired steered-minus-unsteered differences per configuration. The original mean analysis remains available separately.</p><p><b>Median of 96 pairs:</b> the middle persona–question effect, giving every pair equal weight. This is less sensitive to extreme individual differences. <b>Median of 8 persona means:</b> first average twelve questions per persona, then take the median of the eight means. This reduces the influence of extreme personas but retains sensitivity to question outliers within each persona. <b>Original mean:</b> the equal-persona, equal-question average.</p>',
        '<div class="controls">': '<div class="controls"><label>Statistic<select id="statistic"><option value="median_pairs">Median of 96 paired differences</option><option value="median_persona_means">Median of 8 persona means</option><option value="mean">Original overall mean</option></select></label>',
        '<p class="muted">Cells show mean difference, exploratory 95% interval, and number of personas with a positive mean (out of eight). Green shading marks a positive overall mean; it does not imply statistical significance.</p>':
            '<p class="muted">Cells show the selected statistic, its exploratory 95% bootstrap interval, and the original mean for comparison. Green shading marks a positive estimate, not statistical significance. These are medians of paired differences, not differences between separate steered and unsteered medians.</p>',
        'Metric columns sort by the mean difference.': 'Metric columns sort by the selected statistic.',
        'this is a descriptive average for the selected personas': 'this is a descriptive summary for the selected personas',
        'overall-aggregation.csv': 'overall-median-comparison.csv',
        'overall-aggregation.json': 'overall-median-comparison.json',
        'JSON with persona and question means': 'JSON with all three estimators',
        'const DATA=/*DATA*/;': 'const SOURCE=/*DATA*/;\nconst DATA={...SOURCE,conditions:SOURCE.conditions};',
        "function render(){const rows=":
            "function render(){DATA.conditions=SOURCE.conditions.map(c=>({...c,metrics:Object.fromEntries(Object.entries(c.metrics).map(([name,s])=>[name,{mean_delta:s.estimators[el('statistic').value].estimate,ci95:s.estimators[el('statistic').value].ci95,original_mean:s.estimators.mean.estimate}]))}));const rows=",
        '<small>${s.positive_personas} / ${DATA.personas.length} positive personas</small>': '<small>Original mean: ${fmt(s.original_mean)}</small>',
        'render();\n</script>': "el('statistic').addEventListener('change',render);render();\n</script>",
    }
    for old, new in replacements.items():
        if old not in template:
            raise ValueError(f'Template changed; missing fragment: {old[:80]}')
        template = template.replace(old, new)
    template = template.replace('/*SORTING*/', Path(__file__).with_name('mini_batch_table_sort.js').read_text())
    payload = json.dumps(result, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (folder / 'overall-median-comparison.html').write_text(template.replace('/*DATA*/', payload))


def build(folder):
    source = folder / 'question-aggregation.json'
    data = json.loads(source.read_text())
    overall = json.loads((folder / 'overall-aggregation.json').read_text())
    questions, personas = overall['questions'], overall['personas']
    groups = defaultdict(list)
    for row in data['per_question']:
        groups[tuple(row[k] for k in KEYS)].append(row)
    indices = np.random.default_rng(20261007).integers(0, len(questions), (10000, len(questions)))
    conditions, csv_rows = [], []
    for condition in overall['conditions']:
        key = tuple(condition[k] for k in KEYS)
        rows = groups[key]
        lookup = {(r['question'], r['character']): r for r in rows}
        if len(lookup) != len(rows) or set(lookup) != {(q,p) for q in questions for p in personas}:
            raise ValueError('Require a complete unique question/persona grid')
        metrics = {}
        flat = {k: condition[k] for k in KEYS}
        for name, (_, delta) in METRICS.items():
            matrix = np.array([[lookup[q,p][delta] for p in personas] for q in questions], dtype=float)
            if not all(r['control_available'] for r in rows) or not np.isfinite(matrix).all():
                raise ValueError('Missing or nonfinite paired score')
            original = condition['metrics'][name]
            if not np.isclose(matrix.mean(), original['mean_delta'], rtol=1e-12, atol=1e-15):
                raise ValueError('Existing overall mean differs from source')
            estimators = median_statistics(matrix, indices)
            estimators['mean'] = dict(estimate=original['mean_delta'], ci95=original['ci95'])
            metrics[name] = dict(estimators=estimators, n_pairs=int(matrix.size))
            for estimator, stats in estimators.items():
                flat[f'{name}_{estimator}'] = stats['estimate']
                flat[f'{name}_{estimator}_ci_low'], flat[f'{name}_{estimator}_ci_high'] = stats['ci95']
        conditions.append(dict(zip(KEYS, key), metrics=metrics))
        csv_rows.append(flat)
    result = dict(questions=questions, personas=personas, conditions=conditions, method={
        'status': 'Experimental sensitivity analysis; original mean results retained.',
        'median_pairs': 'Median of the 96 answer-level paired differences, not a token-level median or a difference of separate medians.',
        'median_persona_means': 'Median across eight per-persona means of twelve paired question differences.',
        'mean': 'Original equal-persona, equal-question mean, with unchanged interval.',
        'bootstrap': dict(samples=10000, seed=20261007, unit='Twelve question IDs resampled jointly across all eight fixed personas; recalculate each median per resample'),
        'limitations': 'Exploratory percentile intervals with twelve deliberately selected questions and no multiple-comparison adjustment. Different estimators answer different questions; not selected for favorable results.',
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    (folder / 'overall-median-comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (folder / 'overall-median-comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    render_html(result, folder)
    print(f'Created experimental JSON, CSV and HTML: {len(conditions)} configurations, three estimators each')


if __name__ == '__main__':
    build(Path('results/mini-batch-cluster-readout'))
