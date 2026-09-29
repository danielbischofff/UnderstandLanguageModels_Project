"""Prepare audited bank sets, score saved runs, and summarize paired response scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml

from .contrastive import WordSpec, resolve_word
from tokenizers import Tokenizer

MEASURES = ('mean_reciprocal_rank', 'centroid_cosine')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(audit_path, config_path, destination):
    audit = json.loads(audit_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    model = config['model']
    if (model['id'] != audit['provenance']['model_id'] or
        model.get('tokenizer_revision', model.get('revision')) != audit['provenance']['tokenizer_revision']):
        raise ValueError('Config model/tokenizer revision does not match audit')
    for source in audit['banks']:
        if digest(source['path']) != source['sha256']:
            raise ValueError(f"Stale audit: {source['path']}")
    if digest(Path(__file__).with_name('contrastive.py')) != audit['resolver_sha256']:
        raise ValueError('Stale audit: resolver changed')
    tokenizer_path = Path(audit['analysis_tokenizer']['path'])
    if digest(tokenizer_path) != audit['analysis_tokenizer']['sha256']:
        raise ValueError('Analysis tokenizer changed since audit')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    questions = {q['id'] for q in config['questions']}
    groups = defaultdict(list)
    for row in audit['rows']:
        if row['question'] in questions:
            resolved = resolve_word(tokenizer, WordSpec(row['word']))
            if resolved is None or (resolved.token_id, resolved.complete, resolved.surface_form) != (
                    row['selected_token_id'], row['complete'], row['selected_surface_form']):
                raise ValueError(f"Invalid audited mapping: {row['word']}")
            groups[(row['bank'], row['question'], row['word_set_type'])].append(row)
    if {key[1] for key in groups} != questions:
        raise ValueError('Audit does not cover configured questions')
    if destination.exists():
        raise FileExistsError(f'Use a fresh output directory: {destination}')
    jobs, counts = [], []
    for (bank, question, kind), rows in sorted(groups.items()):
        for policy in ('complete_only', 'including_proxies'):
            sets = []
            for persona in sorted({r['persona'] for r in rows}):
                original = [r for r in rows if r['persona'] == persona]
                selected = [r for r in original if policy == 'including_proxies' or r['complete']]
                if not selected:
                    raise ValueError(f'No complete words: {bank}/{question}/{persona}/{kind}')
                identifier = f'{bank}-{question}-{persona}-{kind}'
                sets.append({'id': identifier, 'label': persona,
                             'words': [{'word': r['word'], 'variants': [r['selected_surface_form']]} for r in selected]})
                counts.append({'bank': bank, 'question': question, 'persona': persona,
                               'word_set_type': kind, 'policy': policy, 'original_count': len(original),
                               'retained_count': len(selected),
                               'unique_token_count': len({r['selected_token_id'] for r in selected}),
                               'excluded_words': [r['word'] for r in original if r not in selected]})
            if len(sets) > 8:
                raise ValueError('Analysis supports at most eight sets per job')
            subdir = f'{policy}/{bank}/{question}/{kind}'
            jobs.append({'bank': bank, 'question': question, 'topic': rows[0]['topic'],
                         'word_set_type': kind, 'policy': policy, 'subdir': subdir,
                         'sets': sets})
    destination.mkdir(parents=True)
    for job in jobs:
        path = destination / 'sets' / (job['subdir'] + '.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(job.pop('sets'), sort_keys=False))
        job['sets_path'] = str(path.resolve())
    manifest = {'schema_version': 1, 'config': str(config_path), 'config_sha256': digest(config_path),
                'audit': str(audit_path), 'audit_sha256': digest(audit_path),
                'analysis_tokenizer_sha256': audit['analysis_tokenizer']['sha256'],
                'jobs': jobs, 'set_counts': counts}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def mean(values):
    return sum(values) / len(values) if values else None


def response_scores(path):
    groups = defaultdict(list)
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['segment'] != 'response':
                continue
            key = tuple(row[k] for k in ('experiment', 'run', 'question', 'character', 'coefficient', 'scheme', 'layer', 'set'))
            groups[key].append(row)
    result = []
    for key, rows in groups.items():
        entry = dict(zip(('experiment', 'run', 'question', 'character', 'coefficient', 'scheme', 'layer', 'set'), key))
        entry['response_positions'] = len(rows)
        for measure in MEASURES:
            values = [float(r[measure]) for r in rows if r.get(measure) not in ('', None)]
            values = [v for v in values if math.isfinite(v)]
            entry[measure] = mean(values)
            entry[measure + '_positions'] = len(values)
        result.append(entry)
    return result


def compare(scores, labels):
    controls = {}
    conditions = defaultdict(dict)
    for row in scores:
        if row['character'] == 'unsteered':
            key = (row['experiment'], row['run'], row['question'], row['layer'], row['set'])
            if key in controls:
                raise ValueError(f'Ambiguous control: {key}')
            controls[key] = row
        else:
            key = tuple(row[k] for k in ('experiment', 'run', 'question', 'character', 'coefficient', 'scheme', 'layer'))
            conditions[key][row['set']] = row
    results = []
    for key, sets in conditions.items():
        own_id = next((s for s in sets if labels[s] == key[3]), None)
        # Each bank's comparison includes only steered personas belonging to that bank.
        if own_id is None:
            continue
        own = sets[own_id]
        base_key = (key[0], key[1], key[2], key[6])
        baseline = controls.get((*base_key, own_id))
        result = {k: v for k, v in own.items() if k != 'set'}
        result['control_available'] = baseline is not None
        for measure in MEASURES:
            score = own[measure]
            rivals = [r[measure] for s, r in sets.items() if s != own_id and r[measure] is not None]
            base = baseline[measure] if baseline else None
            rival_base = [controls[(*base_key, s)][measure] for s in sets
                          if s != own_id and (*base_key, s) in controls and controls[(*base_key, s)][measure] is not None]
            margin = score - mean(rivals) if score is not None and len(rivals) == len(sets)-1 and rivals else None
            base_margin = base - mean(rival_base) if base is not None and len(rival_base) == len(sets)-1 and rival_base else None
            result[measure + '_unsteered'] = base
            result[measure + '_delta'] = score - base if score is not None and base is not None else None
            result[measure + '_persona_margin'] = margin
            result[measure + '_margin_delta'] = margin - base_margin if margin is not None and base_margin is not None else None
        results.append(result)
    return results


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(manifest, root):
    all_scores, comparisons = [], []
    for job in manifest['jobs']:
        paths = list((root / 'metrics' / job['subdir']).glob('*/*/metrics.csv'))
        if not paths:
            raise ValueError(f"Missing metrics for {job['subdir']}")
        labels = {s['id']: s['label'] for s in yaml.safe_load(Path(job['sets_path']).read_text())}
        metadata = {k: job[k] for k in ('bank', 'topic', 'word_set_type', 'policy')}
        for path in paths:
            scores = response_scores(path)
            all_scores.extend({**metadata, **r} for r in scores)
            comparisons.extend({**metadata, **r} for r in compare(scores, labels))
    if not comparisons:
        raise ValueError('No intended-persona comparisons found')
    write_csv(root / 'response_scores.csv', all_scores)
    write_csv(root / 'comparisons.csv', comparisons)
    # Average questions within each topic, then give topics equal weight.
    axes = ('experiment', 'run', 'bank', 'word_set_type', 'policy', 'character', 'coefficient', 'scheme', 'layer')
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[tuple(row[k] for k in axes)].append(row)
    totals = []
    for key, rows in grouped.items():
        entry = dict(zip(axes, key))
        entry['questions'] = len({r['question'] for r in rows})
        entry['topics'] = len({r['topic'] for r in rows})
        for measure in MEASURES:
            for suffix in ('', '_delta', '_persona_margin', '_margin_delta'):
                field = measure + suffix
                by_topic = defaultdict(list)
                for row in rows:
                    if row[field] is not None:
                        by_topic[row['topic']].append(row[field])
                entry[field] = mean([mean(v) for v in by_topic.values()])
                entry[field + '_valid_questions'] = sum(len(v) for v in by_topic.values())
        totals.append(entry)
    write_csv(root / 'topic_balanced_summary.csv', totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audit', type=Path, default=Path('data/word_tokenization/word_tokenization.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--layer', action='append', type=int, default=[])
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()
    if not args.prepare_only and args.run is None:
        parser.error('--run is required unless --prepare-only is set')
    if not args.prepare_only:
        try:
            import plotly  # Check before expensive CPU readout.
        except ImportError:
            parser.error('Plotly is required; include runtime/analysis-deps in PYTHONPATH or install the plots extra')
        from .quant import load_run
        from steering.experiments import ExperimentConfig
        archive = load_run(args.run)
        expected = ExperimentConfig.from_mapping(yaml.safe_load(args.config.read_text()))
        actual = ExperimentConfig.from_mapping(archive.experiment)
        if actual.to_mapping() != expected.to_mapping():
            parser.error('Run configuration differs from --config; use its resolved-config.yaml')
        if len(archive.trials) != len(actual.trials()):
            parser.error('Run is incomplete; wait for all trials before paired analysis')
    manifest = prepare(args.audit, args.config, args.output_dir)
    commands = []
    for job in manifest['jobs']:
        command = [sys.executable, '-m', 'quantitive_analysis.cli', str(args.run or 'RUN_DIRECTORY'),
                   '--question', job['question'], '--sets', job['sets_path'],
                   '--output-dir', str(args.output_dir / 'metrics' / job['subdir']),
                   '--top-k', str(args.top_k), '--rank-measure', 'mean_reciprocal_rank',
                   '--cosine-measure', 'centroid_cosine']
        for layer in args.layer:
            command.extend(['--layer', str(layer)])
        commands.append(command)
    (args.output_dir / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    if args.prepare_only:
        print(f'Prepared {len(commands)} analysis jobs in {args.output_dir}')
        return
    # Explicitly verify the runtime tokenizer rather than rely on defaults or env overrides.
    audit = json.loads(args.audit.read_text())
    tokenizer_path = Path(audit['analysis_tokenizer']['path']).resolve()
    for command in commands:
        subprocess.run([*command, '--j-lens-data', str(tokenizer_path.parent)], check=True)
    summarize(manifest, args.output_dir)
    print(args.output_dir / 'topic_balanced_summary.csv')


if __name__ == '__main__':
    main()
