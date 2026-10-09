"""Resumable CPU MRR/activation-word-cosine readout of extracted binary runs.

Stores word-level ranks/cosines once per trial/layer, then averages the audited
sets under both token policies. No top-k approximation or model inference.
"""
from __future__ import annotations
import argparse
import fcntl
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from .contrastive import WordSpec, resolve_word


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def trial_key(t):
    s = t.get('steering') or {}
    q = t['question']
    return (q['id'] if isinstance(q, dict) else q, t['character'], s.get('name'), s.get('coefficient'))


def assemble_activations(directory, layer, token_ids):
    """Input-token alignment; reject overlap, gaps, and wrong event token IDs."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    events = [e for e in manifest['events'] if e['layer'] == layer and e['point'] == 'post_intervention']
    rows = {}
    with safe_open(directory / 'activations.safetensors', framework='pt', device='cpu') as f:
        for event in events:
            positions = event['cache_positions']
            tensor = f.get_tensor(event['tensor_key']).float()
            if tensor.shape[0] != 1 or tensor.shape[1] != len(positions):
                raise ValueError('Expected one sequence with a row per recorded position')
            if event['input_token_ids'] != [[token_ids[p] for p in positions]]:
                raise ValueError('Activation input token mismatch')
            for i, position in enumerate(positions):
                if position in rows:
                    raise ValueError('Duplicate activation position')
                rows[position] = tensor[0, i]
    positions = sorted(rows)
    if positions not in (list(range(len(token_ids) - 1)), list(range(len(token_ids)))):
        raise ValueError('Expected contiguous input activations through the penultimate or final token')
    return np.asarray(positions, dtype=np.int32), torch.stack([rows[p] for p in positions])


def ranks_for_targets(logits, ids):
    """One-based full-vocabulary ranks, best rank on ties, no top-k cutoff."""
    values = np.asarray(logits)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite logits')
    selected = values[:, ids]
    ordered = np.sort(values, axis=1)
    return np.stack([1 + len(row) - np.searchsorted(row, targets, side='right')
                     for row, targets in zip(ordered, selected)]).astype(np.int32)


def mean_word_cosines(normalized, word_rows):
    """Return per-word cosines; set averaging deliberately happens later."""
    x = np.asarray(normalized, dtype=np.float32)
    u = np.asarray(word_rows, dtype=np.float32)
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    un = np.linalg.norm(u, axis=1, keepdims=True)
    x = np.divide(x, xn, out=np.full_like(x, np.nan), where=xn > 0)
    u = np.divide(u, un, out=np.full_like(u, np.nan), where=un > 0)
    return x @ u.T


def validate_word_audit(audit_path, tokenizer_path, questions):
    """Revalidate mappings; stale source hash alone is not silently trusted."""
    from . import contrastive
    audit = json.loads(audit_path.read_text())
    for bank in audit['banks']:
        if digest(bank['path']) != bank['sha256']:
            raise ValueError(f"Bank changed: {bank['path']}")
    if digest(tokenizer_path) != audit['analysis_tokenizer']['sha256']:
        raise ValueError('Pinned tokenizer bytes differ from audited tokenizer')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = [r for r in audit['rows'] if r['question'] in questions]
    for r in rows:
        actual = resolve_word(tokenizer, WordSpec(r['word']))
        if actual is None or (actual.token_id, actual.surface_form, actual.complete) != (
                r['selected_token_id'], r['selected_surface_form'], r['complete']):
            raise ValueError(f"Audited mapping changed: {r['word']}")
        for variant in r['variants']:
            if tokenizer.encode(variant['surface_form'], add_special_tokens=False).ids != variant['token_ids']:
                raise ValueError(f"Audited variant changed: {r['word']}")
    if {r['question'] for r in rows} != questions:
        raise ValueError('Missing question word sets')
    validation = {'audit_sha256': digest(audit_path), 'tokenizer_sha256': digest(tokenizer_path),
                  'current_resolver_sha256': digest(contrastive.__file__),
                  'original_resolver_sha256': audit['resolver_sha256'],
                  'validated_word_occurrences': len(rows), 'all_selected_ids_and_variants_match': True}
    return rows, validation


def definitions(rows, question):
    groups = defaultdict(list)
    for row in rows:
        if row['question'] == question:
            groups[(row['bank'], row['persona'], row['word_set_type'], row['topic'])].append(row)
    result = []
    for (bank, persona, kind, topic), words in sorted(groups.items()):
        for policy in ('complete_only', 'including_proxies'):
            chosen = [w for w in words if policy == 'including_proxies' or w['complete']]
            if not chosen:
                raise ValueError('Empty word set after filtering')
            result.append(dict(bank=bank, persona=persona, word_set_type=kind, topic=topic,
                               policy=policy, original_count=len(words), retained_count=len(chosen),
                               unique_token_count=len({w['selected_token_id'] for w in chosen}),
                               token_ids=[w['selected_token_id'] for w in chosen], words=[w['word'] for w in chosen]))
    return result


def score_sets(sets, ids, ranks, cosines, positions, prompt_length):
    columns = {int(token): i for i, token in enumerate(ids)}
    result = []
    for definition in sets:
        cols = [columns[t] for t in definition['token_ids']]
        for segment, mask in [('prompt', positions < prompt_length), ('response', positions >= prompt_length)]:
            rank = ranks[mask][:, cols]
            cosine = cosines[mask][:, cols]
            result.append({**definition, 'segment': segment, 'valid_positions': int(mask.sum()),
                           'mean_reciprocal_rank': float(np.mean(1.0 / rank)) if rank.size else None,
                           'activation_word_cosine': float(np.mean(cosine)) if cosine.size and np.isfinite(cosine).all() else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/mini_batch_exp_results'))
    parser.add_argument('--data', type=Path, default=Path('runtime/models/j-lens/qwen3-32b'))
    parser.add_argument('--audit', type=Path, default=Path('data/word_tokenization/word_tokenization.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/analysis/mini-batch-local-readout'))
    parser.add_argument('--layers', nargs='+', type=int, default=[32, 29, 35, 62])
    parser.add_argument('--limit', type=int, help='Maximum trial/layer jobs, useful for a benchmark')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.readout.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / 'direct_character_vectors_j_lens_analysis/src'))
    from readout import LensTransport
    from unembed import Unembedding
    index = json.loads((args.run / 'index.json').read_text())
    config = index['document']['experiment']
    manifest = json.loads((args.run / 'provenance/manifest.json').read_text())
    paths = {trial_key(t): args.run / 'provenance' / t['path'] for t in manifest['trials'].values()}
    if len(paths) != len(manifest['trials']):
        raise ValueError('Duplicate provenance trial keys')
    audit_doc = json.loads(args.audit.read_text())
    if audit_doc['provenance']['model_id'] != config['model']['id'] or audit_doc['provenance']['tokenizer_revision'] != config['model']['tokenizer_revision']:
        raise ValueError('Run model/tokenizer does not match word audit')
    rows, validation = validate_word_audit(args.audit, args.data / 'tokenizer.json', {q['id'] for q in config['questions']})
    lens_path = args.data / 'qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt'
    lens = LensTransport(str(lens_path))
    if not set(args.layers) <= set(lens.source_layers):
        raise ValueError(f'Unfitted layers: {set(args.layers) - set(lens.source_layers)}')
    args.output.mkdir(parents=True, exist_ok=True)
    signature = {'index_sha256': digest(args.run / 'index.json'), 'audit': validation,
                 'assets': json.loads((args.data / 'download-provenance.json').read_text()),
                 'implementation_sha256': digest(__file__), 'alignment': 'input_token; include every recorded response position',
                 'rank_scope': 'full_vocabulary; ties take best rank', 'dtype': 'float32', 'lens_sha256': digest(lens_path),
                 'unembedding_sha256': digest(args.data / 'unembed/lm_head.weight.bf16.bin'),
                 'rmsnorm_sha256': digest(args.data / 'unembed/model.norm.weight.bf16.bin')}
    provenance_path = args.output / 'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != signature:
        raise ValueError('Output provenance changed; select a new output directory')
    atomic_json(provenance_path, signature)
    unembedding = Unembedding(args.data)
    print('Loading FP32 unembedding into memory (~3.1 GB)', flush=True)
    head = torch.from_numpy(unembedding.rows(np.arange(unembedding.vocab_size)))
    gain = torch.from_numpy(unembedding.gain)
    jobs = []
    for layer in args.layers:
        # Keep each question together, putting its control before interventions.
        ordered = sorted(enumerate(index['trials']), key=lambda p: (p[1]['metadata']['question']['id'], p[1]['metadata']['character'] != 'unsteered', p[0]))
        for number, descriptor in ordered:
            jobs.append((layer, number, descriptor['metadata']))
    done = 0
    started = time.time()
    for layer, number, trial in jobs:
        stem = f'{number:04d}-layer-{layer}'
        summary_path = args.output / (stem + '.json')
        if summary_path.exists():
            continue
        tick = time.time()
        q = trial['question']['id']
        sets = definitions(rows, q)
        ids = np.array(sorted({t for s in sets for t in s['token_ids']}), dtype=np.int64)
        tokens = [t['id'] for t in trial['j_lens']['tokens']]
        positions, residuals = assemble_activations(paths[trial_key(trial)] / 'activations', layer, tokens)
        matrix, scale = lens.matrix(layer)
        transported = (residuals @ matrix.float().T) * scale
        normalized = transported * torch.rsqrt(transported.square().mean(dim=1, keepdim=True) + 1e-6) * gain
        cosine = mean_word_cosines(normalized.numpy(), head[ids].numpy())
        ranks = []
        with torch.inference_mode():
            for start in range(0, len(normalized), args.batch_size):
                logits = normalized[start:start + args.batch_size] @ head.T
                ranks.append(ranks_for_targets(logits.numpy(), ids))
                del logits
        ranks = np.concatenate(ranks)
        target = args.output / (stem + '.npz')
        temporary = target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, positions=positions, token_ids=ids, ranks=ranks, activation_word_cosine=cosine)
        temporary.replace(target)
        summary = {'trial_index': number, 'question': q, 'character': trial['character'], 'steering': trial.get('steering'),
                   'layer': layer, 'generated_tokens': len(trial['response']['generated_token_ids']),
                   'at_token_limit': len(trial['response']['generated_token_ids']) == config['generation']['max_new_tokens'],
                   'missing_final_input_activation': len(positions) < len(tokens),
                   'scores': score_sets(sets, ids, ranks, cosine, positions, trial['j_lens']['prompt_length']),
                   'elapsed_seconds': time.time() - tick}
        atomic_json(summary_path, summary)
        done += 1
        print(f'{stem} {q} {trial["character"]}: {time.time()-tick:.1f}s; new jobs={done}', flush=True)
        completed = len(list(args.output.glob('*-layer-*.json')))
        atomic_json(args.output / 'progress.json', {'completed_jobs': completed, 'total_jobs': len(jobs),
                    'new_jobs_this_invocation': done, 'elapsed_seconds': time.time() - started,
                    'status': 'complete' if completed == len(jobs) else 'running'})
        if done == 2 or done % 33 == 0:
            from .local_summary import summarize
            summarize(args.output)
        if args.limit and done >= args.limit:
            break
    from .local_summary import summarize
    completed = summarize(args.output)
    atomic_json(args.output / 'progress.json', {'completed_jobs': completed, 'total_jobs': len(jobs),
                'new_jobs_this_invocation': done, 'elapsed_seconds': time.time() - started,
                'status': 'complete' if completed == len(jobs) else 'limited_run_finished'})


if __name__ == '__main__':
    main()
