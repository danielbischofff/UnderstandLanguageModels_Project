#!/usr/bin/env python3
"""Create alpha-only paired summaries and dashboards from fresh readout scores."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quantitive_analysis.local_summary import summarize as summarize_readouts
from aggregate_mini_batch import METRICS, aggregate_metric
from aggregate_overall_mini_batch import summarize
from compare_overall_medians import median_statistics, render_html as render_medians

KEYS = ('layer', 'scheme', 'alpha', 'word_set_type', 'policy')


def adapt_template(template, alphas):
    # The table's machinery is generic; adapt all coefficient identifiers and labels.
    template = template.replace('beta', 'alpha').replace('Band-width β', 'Steering strength α').replace('β', 'α')
    template = template.replace('Overall effect across eight personas', 'Alpha steering: overall paired effects')
    template = template.replace('α = 0 pins the persona projection; α = 0.25 allows a band around the target.',
                                'Additive steering applies h′ = h + αc independently at each selected layer. The combined scheme is not dose-matched to single-layer steering.')
    template = template.replace('α = 0 still pins the persona projection; α = 0.25 allows a band around that target.',
                                'Additive strength α scales the persona vector independently at each selected layer.')
    template = template.replace('α = 0 still applies steering.', 'Additive strength α scales the persona vector.')
    template = template.replace('<option value="0">0</option><option value="0.25">0.25</option>',
                                ''.join(f'<option value="{alpha}">{alpha}</option>' for alpha in alphas))
    template = template.replace('The final generated token has no cached input activation.',
                                'All recorded response positions are included; a final token is omitted only when its activations are unavailable.')
    return template


def html_table(result, template_path):
    template = adapt_template(template_path.read_text(),sorted({c['alpha'] for c in result['conditions']}))
    template = template.replace('/*SORTING*/', template_path.with_name('mini_batch_table_sort.js').read_text())
    payload = json.dumps(result, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    return template.replace('/*DATA*/', payload)


def build(run, readouts, output, refresh=True):
    index = json.loads((run / 'index.json').read_text())
    trials = [t['metadata'] for t in index['trials']]
    schemes = [t['steering'] for t in trials if t['character'] != 'unsteered']
    if not schemes or any(s.get('method') != 'additive' for s in schemes):
        raise ValueError('This command requires an additive alpha experiment')
    expected = {(i, layer) for i in range(len(trials)) for layer in (29,32,35,62)}
    docs = [json.loads(p.read_text()) for p in sorted(readouts.glob('*-layer-*.json'))]
    actual = {(d['trial_index'],d['layer']) for d in docs}
    if actual != expected or len(docs) != len(expected):
        raise ValueError(f'Incomplete or duplicate readouts: {len(actual & expected)} / {len(expected)} expected jobs present')
    for doc in docs:
        trial = trials[doc['trial_index']]
        if (doc['question'],doc['character'],doc['steering']) != (trial['question']['id'],trial['character'],trial['steering']):
            raise ValueError('Readout metadata differs from requested run')
    if refresh:
        summarize_readouts(readouts)
    rows = json.loads((readouts / 'comparisons.json').read_text())
    if any('alpha' not in row or 'beta' in row for row in rows):
        raise ValueError('Expected alpha-labelled comparisons; refresh readout summaries')
    questions = sorted(q['id'] for q in index['document']['experiment']['questions'])
    personas = sorted({r['character'] for r in rows})
    groups, per_persona = defaultdict(list), defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        groups[key].append(row)
        per_persona[(row['character'],row['bank'],*key)].append(row)
    conditions, median_conditions = [], []
    indices = np.random.default_rng(20261007).integers(0, len(questions), (10000,len(questions)))
    for key, group in sorted(groups.items()):
        metrics = summarize(group, questions, personas)
        conditions.append(dict(zip(KEYS,key),metrics=metrics))
        lookup = {(r['question'],r['character']):r for r in group}
        medians = {}
        for name,(_,delta) in METRICS.items():
            matrix = np.array([[lookup[q,p][delta] for p in personas] for q in questions])
            estimators=median_statistics(matrix, indices)
            estimators['mean']=dict(estimate=metrics[name]['mean_delta'],ci95=metrics[name]['ci95'])
            medians[name] = dict(estimators=estimators,n_pairs=int(matrix.size))
        median_conditions.append(dict(zip(KEYS,key),metrics=medians))
    method = dict(steering='additive', aggregation='Equal-question mean within each persona, then equal-persona mean.',
                  bootstrap='10,000 joint question-ID resamples; fixed personas; 95% percentile intervals.',
                  interpretation='Exploratory for selected questions and personas; no multiple-comparison correction. Readout association, not literal word mentions.',
                  source=str(run), index_sha256=hashlib.sha256((run/'index.json').read_bytes()).hexdigest())
    result = dict(method=method, questions=questions, personas=personas, conditions=conditions)
    output.mkdir(parents=True,exist_ok=True)
    (output/'comparisons.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    (output/'overall-aggregation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    median_result=dict(method=method,questions=questions,personas=personas,conditions=median_conditions)
    (output/'overall-median-comparison.json').write_text(json.dumps(median_result,indent=2,allow_nan=False)+'\n')
    median_flat=[]
    for condition in median_conditions:
        row={k:condition[k] for k in KEYS}
        for metric,stats in condition['metrics'].items():
            for estimator,summary in stats['estimators'].items():
                row[f'{metric}_{estimator}']=summary['estimate']
                row[f'{metric}_{estimator}_ci_low'],row[f'{metric}_{estimator}_ci_high']=summary['ci95']
        median_flat.append(row)
    with (output/'overall-median-comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(median_flat[0]));writer.writeheader();writer.writerows(median_flat)
    persona_results = []
    for key, group in sorted(per_persona.items()):
        group.sort(key=lambda r:r['question'])
        persona_results.append(dict(zip(('character','bank',*KEYS),key),metrics={name:aggregate_metric(group,*fields) for name,fields in METRICS.items()}))
    question_result=dict(method=method,questions=index['document']['experiment']['questions'],conditions=persona_results,per_question=rows)
    (output/'question-aggregation.json').write_text(json.dumps(question_result,indent=2,allow_nan=False)+'\n')
    flat = []
    for condition in conditions:
        row = {k:condition[k] for k in KEYS}
        for metric,stats in condition['metrics'].items():
            row[metric+'_mean_delta']=stats['mean_delta']
            row[metric+'_positive_personas']=stats['positive_personas']
            row[metric+'_ci_low'],row[metric+'_ci_high']=stats['ci95']
        flat.append(row)
    with (output/'overall-aggregation.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    template = Path(__file__).with_name('overall_aggregation_template.html')
    html = html_table(result,template)
    (output/'overall-aggregation.html').write_text(html)
    alphas=sorted({c['alpha'] for c in conditions})
    for stem,template_name,payload in [('question-aggregation','mini_batch_aggregation_template.html',question_result),
                                      ('question-comparison','question_comparison_template.html',dict(questions=question_result['questions'],rows=rows))]:
        template=adapt_template(Path(__file__).with_name(template_name).read_text(),alphas)
        template=template.replace('/*PLOTLY*/',get_plotlyjs()).replace('/*DATA*/',json.dumps(payload,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
        (output/(stem+'.html')).write_text(template)
    render_medians(median_result,output)
    median_page=output/'overall-median-comparison.html'
    median_page.write_text(adapt_template(median_page.read_text(),alphas))
    # A dedicated landing page avoids linking alpha results into the band dashboard.
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Alpha analysis overview</title><body style="font:17px system-ui;max-width:900px;margin:40px auto;padding:20px"><h1>Alpha steering analysis overview</h1><p>Completed additive steering readouts, separately paired with their own unsteered controls.</p><ul><li><a href="overall-aggregation.html">Sortable overall-effects table</a></li><li><a href="question-aggregation.html">Per-persona aggregation and specificity</a></li><li><a href="question-comparison.html">Compare questions</a></li><li><a href="overall-median-comparison.html">Experimental mean versus median</a></li></ul><p>The token-level page is built separately by the cluster job after these tables.</p><p><a href="overall-aggregation.csv">Download table CSV</a> · <a href="question-aggregation.json">Per-persona and per-question JSON</a></p><p>All questions are included. The intervals are exploratory given the limited question set.</p></body></html>')
    print(f'Built alpha-only analysis: {len(conditions)} configurations, {len(docs)} readout jobs. Open {output / "index.html"}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--readouts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.run,args.readouts,args.output)
