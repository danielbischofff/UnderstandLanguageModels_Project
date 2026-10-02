"""Audit bank words with the same resolver used by quantitative analysis (CPU only)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
import yaml

from .contrastive import WordSpec, resolve_word
from steering.cli.prepare_question_banks import read_bank


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(banks, tokenizer_path, analysis_tokenizer_path, provenance):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    analysis_tokenizer = Tokenizer.from_file(str(analysis_tokenizer_path))
    # Cosine/rank scoring uses the tokenizer shipped with the J-Lens artifacts.
    # Refuse a mismatched vocabulary rather than silently mix token ID spaces.
    if tokenizer.get_vocab() != analysis_tokenizer.get_vocab():
        raise ValueError('Pinned model and analysis tokenizer vocabularies differ')
    rows, sources = [], []
    for path in banks:
        bank = read_bank(path)
        sources.append({'path': str(path), 'sha256': bank['sha256']})
        for qid, question in bank['questions'].items():
            for persona, sets in question['personas'].items():
                for kind, words in sets.items():
                    for word in words:
                        spec = WordSpec(word)
                        variants = []
                        for form in spec.surface_forms():
                            encoding = tokenizer.encode(form, add_special_tokens=False)
                            if encoding.ids != analysis_tokenizer.encode(form, add_special_tokens=False).ids:
                                raise ValueError(f'Analysis tokenizer differs for {form!r}')
                            variants.append({'surface_form': form, 'token_ids': encoding.ids,
                                             'token_pieces': encoding.tokens,
                                             'decoded_pieces': [tokenizer.decode([i], skip_special_tokens=False) for i in encoding.ids]})
                        resolved = resolve_word(tokenizer, spec)
                        if resolved is None:
                            raise ValueError(f'Unresolvable word: {word}')
                        rows.append({'bank': path.stem, 'question': qid, 'topic': question['topic'],
                                     'persona': persona, 'word_set_type': kind, 'word': word,
                                     'variants': variants, 'selected_surface_form': resolved.surface_form,
                                     'selected_token_id': resolved.token_id,
                                     'selected_token_piece': tokenizer.id_to_token(resolved.token_id),
                                     'selected_token_decoded': tokenizer.decode([resolved.token_id], skip_special_tokens=False),
                                     'complete': resolved.complete,
                                     'status': 'single_token' if resolved.complete else 'first_token_proxy'})
    by_token = defaultdict(set)
    for row in rows:
        by_token[row['selected_token_id']].add(row['word'])
    for row in rows:
        row['collision_words'] = sorted(by_token[row['selected_token_id']] - {row['word']})
    return {'schema_version': 1, 'provenance': provenance, 'banks': sources,
            'tokenizer': {'path': str(tokenizer_path), 'sha256': sha256(tokenizer_path)},
            'analysis_tokenizer': {'path': str(analysis_tokenizer_path), 'sha256': sha256(analysis_tokenizer_path)},
            'resolver_sha256': sha256(Path(__file__).with_name('contrastive.py')),
            'compatibility': 'identical vocabularies and identical encodings for every audited variant',
            'rows': rows}


def write_outputs(document, destination, summary_dir):
    destination.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = document['rows']
    (destination / 'word_tokenization.json').write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')
    with (destination / 'word_tokenization.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    unique = {r['word']: r for r in rows}
    proxies = sorted(w for w, r in unique.items() if not r['complete'])
    groups = defaultdict(list)
    for row in rows:
        groups[(row['bank'], row['persona'], row['word_set_type'])].append(row)
    lines = ['# Word tokenization audit', '',
             f"Model: `{document['provenance']['model_id']}`; tokenizer revision: `{document['provenance']['tokenizer_revision']}`.", '',
             f'{len(rows)} word occurrences; {len(unique)} distinct words; {len(proxies)} distinct first-token proxies.',
             f"{sum(not r['complete'] for r in rows)} occurrences use proxies; {sum(bool(r['collision_words']) for r in unique.values())} distinct words collide with another word's selected token.", '',
             'Model and J-Lens tokenizers have identical vocabularies and agree on all audited encodings.',
             'Counts below include repeated general-word occurrences across questions. Collisions exclude repeated occurrences of the same word.', '',
             '| Bank | Persona | Set type | Occurrences | Proxy occurrences | Minimum complete words per question |',
             '|---|---|---|---:|---:|---:|']
    for (bank, persona, kind), entries in sorted(groups.items()):
        per_question = defaultdict(int)
        for r in entries:
            per_question[r['question']] += int(r['complete'])
        lines.append(f"| {bank} | {persona} | {kind} | {len(entries)} | {sum(not r['complete'] for r in entries)} | {min(per_question.values())} |")
    lines += ['', '## First-token proxies', '', '| Word | Selected surface | Selected piece | Decoded piece | Colliding words |', '|---|---|---|---|---|']
    for word in proxies:
        r = unique[word]
        lines.append(f"| {word} | `{r['selected_surface_form']}` | `{r['selected_token_piece']}` | `{r['selected_token_decoded']}` | {', '.join(r['collision_words'])} |")
    lines += ['', 'The banks are unchanged. This audit does not filter words or change similarity scoring.',
              'The current analysis CLI still resolves words itself; consuming this mapping or applying complete-only filters requires a separate integration.',
              'A single-token representation can be a capitalization or no-space variant, not necessarily the usual token in running text.', '']
    (summary_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--banks', type=Path, nargs='+', required=True)
    parser.add_argument('--cache-root', type=Path, default=Path('runtime/models/huggingface/hub'))
    parser.add_argument('--analysis-tokenizer', type=Path, default=Path('runtime/models/j-lens/qwen3-32b/tokenizer.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/word_tokenization'))
    parser.add_argument('--summary-dir', type=Path, default=Path('outputs/analysis/word-tokenization'))
    args = parser.parse_args()
    model = yaml.safe_load(args.config.read_text())['model']
    revision = model.get('tokenizer_revision', model.get('revision'))
    if not revision or len(revision) != 40:
        parser.error('Use a pinned 40-character tokenizer revision')
    tokenizer_path = args.cache_root / ('models--' + model['id'].replace('/', '--')) / 'snapshots' / revision / 'tokenizer.json'
    document = audit(args.banks, tokenizer_path, args.analysis_tokenizer,
                     {'model_id': model['id'], 'tokenizer_revision': revision,
                      'config': str(args.config), 'config_sha256': sha256(args.config)})
    write_outputs(document, args.output_dir, args.summary_dir)
    print(f"Audited {len(document['rows'])} word occurrences. Summary: {args.summary_dir / 'summary.md'}")


if __name__ == '__main__':
    main()
