#!/usr/bin/env python3
"""Build a standalone token-level comparison page from saved exact readouts."""
import argparse
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs


def word_audit():
    rows = json.loads(Path('data/word_tokenization/word_tokenization.json').read_text())['rows']
    return {(r['question'], r['bank'], r['persona'], r['word_set_type'], r['word']): r for r in rows}


def word_details(question, bank, persona, kind, words, token_ids, audit):
    details = []
    for word, token_id in zip(words, token_ids):
        row = audit[(question, bank, persona, kind, word)]
        assert row['selected_token_id'] == token_id
        details.append({'word': word, 'id': token_id, 'token': row['selected_token_decoded'],
                        'complete': row['complete']})
    return details


def build(run, readouts, output, comparisons_path=None):
    index = json.loads((run / 'index.json').read_text())
    comparisons = json.loads((comparisons_path or Path('results/mini-batch-cluster-readout/comparisons.json')).read_text())
    banks = {r['character']: r['bank'] for r in comparisons}
    audit = word_audit()
    trials = []
    jobs = 0
    for i, descriptor in enumerate(index['trials']):
        meta = descriptor['metadata']
        prompt_length = meta['j_lens']['prompt_length']
        trial = {'question': meta['question']['id'], 'persona': meta['character'],
                 'steering': meta['steering'], 'text': meta['response']['text'],
                 'tokens': [t['text'] for t in meta['j_lens']['tokens'] if t['segment'] == 'response'],
                 'layers': {}}
        for layer in (29, 32, 35, 62):
            stem = readouts / f'{i:04d}-layer-{layer}'
            if not stem.with_suffix('.json').exists() or not stem.with_suffix('.npz').exists():
                raise ValueError(f'Missing readout: {stem}')
            doc = json.loads(stem.with_suffix('.json').read_text())
            assert doc['trial_index'] == i and doc['question'] == trial['question']
            sets = {}
            with np.load(stem.with_suffix('.npz')) as arrays:
                columns = {int(t): j for j, t in enumerate(arrays['token_ids'])}
                mask = arrays['positions'] >= prompt_length
                positions = (arrays['positions'][mask] - prompt_length).tolist()
                ranks = arrays['ranks'][mask]
                cosines = arrays['activation_word_cosine'][mask]
                for score in doc['scores']:
                    persona = score['persona']
                    if score['segment'] != 'response' or persona not in banks or score['bank'] != banks[persona]:
                        continue
                    if trial['persona'] != 'unsteered' and persona != trial['persona']:
                        continue
                    cols = [columns[t] for t in score['token_ids']]
                    if cols:
                        mrr = (1.0 / ranks[:, cols]).mean(axis=1)
                        cosine = cosines[:, cols].mean(axis=1)
                        for values, field in ((mrr, 'mean_reciprocal_rank'), (cosine, 'activation_word_cosine')):
                            assert np.isclose(values.mean(), score[field], rtol=1e-6, atol=1e-9, equal_nan=True), (stem, field)
                        compact = lambda values: [float(f'{v:.8g}') if np.isfinite(v) else None for v in values]
                        series = {'mrr': compact(mrr), 'cosine': compact(cosine)}
                    else:
                        series = {'mrr': [None] * len(positions), 'cosine': [None] * len(positions)}
                    sets['|'.join((persona, score['word_set_type'], score['policy']))] = {
                        **series, 'words': score['words'], 'bank': score['bank'],
                        'details': word_details(trial['question'], score['bank'], persona,
                                               score['word_set_type'], score['words'], score['token_ids'], audit)}
            trial['layers'][str(layer)] = {'positions': positions, 'sets': sets}
            jobs += 1
        trials.append(trial)
        if len(trials) % 33 == 0:
            print(f'Processed {len(trials)} / {len(index["trials"])} trials', flush=True)
    data = {'questions': index['document']['experiment']['questions'], 'personas': list(banks),
            'trials': trials, 'jobs': jobs}
    template = Path(__file__).with_name('mini_batch_token_template.html').read_text()
    steering = [t['steering'] for t in trials if t['steering'] is not None]
    if steering and all(s.get('method') == 'additive' for s in steering):
        from build_alpha_analysis import adapt_template
        template = adapt_template(template, sorted({s['coefficient'] for s in steering}))
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('/*PLOTLY*/', get_plotlyjs()).replace('/*DATA*/', payload))
    print(f'Wrote {output}: {len(trials)} trials, {jobs} layers, {output.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('outputs/mini_batch_exp_results'))
    parser.add_argument('--readouts', type=Path, default=Path('outputs/analysis/mini-batch-local-readout'))
    parser.add_argument('--output', type=Path, default=Path('results/mini-batch-cluster-readout/token-comparison.html'))
    parser.add_argument('--comparisons', type=Path, help='Comparison JSON for this experiment')
    args = parser.parse_args()
    build(args.run, args.readouts, args.output, args.comparisons)
