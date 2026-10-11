"""Summarize completed local readouts; partial coverage is always explicit."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

MEASURES = ('mean_reciprocal_rank', 'activation_word_cosine')


def coefficient_field(documents):
    methods = {(d.get('steering') or {}).get('method', 'character_band')
               for d in documents if d['character'] != 'unsteered'}
    if len(methods) > 1:
        raise ValueError('Mixed steering methods require separate output folders')
    return 'alpha' if methods == {'additive'} else 'beta'


def paired_comparisons(documents):
    coefficient = coefficient_field(documents)
    scores = []
    controls = {}
    for doc in documents:
        for score in doc['scores']:
            if score['segment'] != 'response':
                continue
            key = (doc['question'], doc['layer'], score['bank'], score['word_set_type'], score['policy'])
            if doc['character'] == 'unsteered':
                full_key = (*key, score['persona'])
                if full_key in controls:
                    raise ValueError('Duplicate control')
                controls[full_key] = score
            scores.append((doc, score, key))
    results = []
    for doc in documents:
        if doc['character'] == 'unsteered':
            continue
        groups = defaultdict(dict)
        for score in doc['scores']:
            if score['segment'] == 'response':
                groups[(score['bank'], score['word_set_type'], score['policy'])][score['persona']] = score
        for (bank, kind, policy), sets in groups.items():
            if doc['character'] not in sets:
                continue
            own = sets[doc['character']]
            base_key = (doc['question'], doc['layer'], bank, kind, policy)
            base = controls.get((*base_key, doc['character']))
            row = dict(question=doc['question'], topic=own['topic'], character=doc['character'],
                       layer=doc['layer'], scheme=doc['steering']['name'],
                       bank=bank, word_set_type=kind, policy=policy, at_token_limit=doc['at_token_limit'],
                       valid_positions=own['valid_positions'], retained_words=own['retained_count'],
                       unique_tokens=own['unique_token_count'], control_available=base is not None)
            row[coefficient] = doc['steering']['coefficient']
            for measure in MEASURES:
                value = own[measure]
                baseline = base[measure] if base else None
                rivals = [s[measure] for p, s in sets.items() if p != doc['character']]
                base_rivals = [controls.get((*base_key, p), {}).get(measure) for p in sets if p != doc['character']]
                margin = value - statistics.mean(rivals) if value is not None and len(rivals) == 7 and None not in rivals else None
                base_margin = baseline - statistics.mean(base_rivals) if baseline is not None and len(base_rivals) == 7 and None not in base_rivals else None
                row[measure] = value
                row[measure + '_delta'] = value - baseline if value is not None and baseline is not None else None
                row[measure + '_persona_margin'] = margin
                row[measure + '_margin_delta'] = margin - base_margin if margin is not None and base_margin is not None else None
            results.append(row)
    return results


def summarize(output):
    output = Path(output)
    documents = [json.loads(p.read_text()) for p in sorted(output.glob('*-layer-*.json'))]
    comparisons = paired_comparisons(documents)
    coefficient = coefficient_field(documents)
    keys = ['character', 'layer', 'scheme', coefficient, 'bank', 'word_set_type', 'policy']
    summaries = []
    for subset in ('all_questions', 'excluding_Q01_Q23'):
        groups = defaultdict(list)
        for row in comparisons:
            if subset == 'excluding_Q01_Q23' and row['question'] in {'Q01', 'Q23'}:
                continue
            groups[tuple(row[k] for k in keys)].append(row)
        for key, rows in sorted(groups.items()):
            item = dict(zip(keys, key), subset=subset, questions=len({r['question'] for r in rows}))
            for metric in MEASURES:
                for suffix in ('', '_delta', '_persona_margin', '_margin_delta'):
                    field = metric + suffix
                    topics = defaultdict(list)
                    for row in rows:
                        if row[field] is not None:
                            topics[row['topic']].append(row[field])
                    item[field] = statistics.mean(statistics.mean(v) for v in topics.values()) if topics else None
                    item[field + '_valid_questions'] = sum(len(v) for v in topics.values())
            summaries.append(item)
    from .local_readout import atomic_json
    atomic_json(output / 'comparisons.json', comparisons)
    atomic_json(output / 'topic-balanced-summary.json', summaries)
    lines = ['# Local J-Lens readout analysis', '',
             f'Completed trial/layer jobs: **{len(documents)} / 1584** (396 trials × 4 readout layers).',
             'Partial results must not be interpreted as a complete experiment comparison.', '',
             'Recorded answer positions only. Both metrics average words equally. MRR uses full-vocabulary ranks; ties receive the best rank. Word tokenization and all audited variants were revalidated against the pinned tokenizer.', '',
             'Main view below: complete-token contextualized sets. Each row is topic-balanced; n is valid paired questions. All twelve questions remain in the main analysis; Q01 and Q23 were functionality checks. JSON also includes general sets, proxy-inclusive sets, margins and an optional exclusion check.', '',
             f'| Persona | Scheme | {coefficient} | Layer | n | MRR Δ | Activation cosine Δ | MRR margin Δ | Cosine margin Δ |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return '—' if x is None else f'{x:.6g}'
    for row in summaries:
        if row['subset'] != 'all_questions' or row['policy'] != 'complete_only' or row['word_set_type'] != 'contextualized':
            continue
        lines.append(f"| {row['character']} | {row['scheme']} | {row[coefficient]} | {row['layer']} | {row['mean_reciprocal_rank_delta_valid_questions']} | " + ' | '.join(fmt(row[field]) for field in ('mean_reciprocal_rank_delta', 'activation_word_cosine_delta', 'mean_reciprocal_rank_margin_delta', 'activation_word_cosine_margin_delta')) + ' |')
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    return len(documents)


if __name__ == '__main__':
    import sys
    print('Summarized', summarize(sys.argv[1] if len(sys.argv) > 1 else 'outputs/analysis/mini-batch-local-readout'), 'jobs')
