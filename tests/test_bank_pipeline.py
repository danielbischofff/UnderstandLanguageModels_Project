from pathlib import Path
import csv
import json

import pytest
import yaml

from quantitive_analysis.bank_pipeline import compare, response_scores, prepare, summarize


def row(character, set_id, score, question='Q01'):
    return dict(experiment='test', run='run1', question=question, character=character,
                coefficient='' if character == 'unsteered' else '1.5',
                scheme='' if character == 'unsteered' else 'layer-32', layer='32',
                set=set_id, mean_reciprocal_rank=score, centroid_cosine=score,
                response_positions=2)


def test_paired_scores_and_persona_specificity():
    scores = [row('unsteered', 'own', .2), row('unsteered', 'other', .3),
              row('scientist', 'own', .6), row('scientist', 'other', .4),
              row('caregiver', 'own', .1), row('caregiver', 'other', .7)]
    result = compare(scores, {'own': 'scientist', 'other': 'caregiver'})
    assert result[0]['mean_reciprocal_rank_delta'] == pytest.approx(.4)
    assert result[0]['mean_reciprocal_rank_persona_margin'] == pytest.approx(.2)
    assert result[0]['mean_reciprocal_rank_margin_delta'] == pytest.approx(.3)
    assert result[1]['centroid_cosine_delta'] == pytest.approx(.4)


def test_missing_values_are_not_zero_and_controls_are_unique():
    result = compare([row('scientist', 'own', None), row('scientist', 'other', .5)],
                     {'own': 'scientist', 'other': 'caregiver'})[0]
    assert result['centroid_cosine_delta'] is None
    assert result['mean_reciprocal_rank_persona_margin'] is None
    assert not result['control_available']
    with pytest.raises(ValueError, match='Ambiguous control'):
        compare([row('unsteered', 'own', .1)] * 2, {'own': 'scientist'})


def test_prompt_exclusion_and_finite_counts(tmp_path):
    path = tmp_path / 'metrics.csv'
    rows = [{**row('scientist', 'own', score), 'segment': segment}
            for score, segment in [(100, 'prompt'), (.2, 'response'), (.6, 'response'), (float('nan'), 'response')]]
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = response_scores(path)[0]
    assert result['centroid_cosine'] == pytest.approx(.4)
    assert result['centroid_cosine_positions'] == 2
    assert result['response_positions'] == 3


def test_topic_balancing(tmp_path):
    jobs = []
    for question, topic, value in [('Q01', 'A', .2), ('Q02', 'A', .4), ('Q03', 'B', .9)]:
        sets = tmp_path / f'{question}.yaml'
        sets.write_text(yaml.safe_dump([{'id': 'own', 'label': 'scientist'}, {'id': 'other', 'label': 'caregiver'}]))
        path = tmp_path / 'metrics' / question / 'test' / 'run1' / 'metrics.csv'
        path.parent.mkdir(parents=True)
        rows = [{**row(role, set_id, score, question), 'segment': 'response'}
                for role, set_id, score in [('unsteered', 'own', 0), ('unsteered', 'other', 0),
                                            ('scientist', 'own', value), ('scientist', 'other', 0)]]
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        jobs.append(dict(subdir=question, sets_path=str(sets), bank='bank', topic=topic,
                         word_set_type='contextualized', policy='complete_only'))
    summarize({'jobs': jobs}, tmp_path)
    result = next(csv.DictReader((tmp_path / 'topic_balanced_summary.csv').open()))
    assert float(result['centroid_cosine_delta']) == pytest.approx(.6)
    assert int(result['centroid_cosine_delta_valid_questions']) == 3


def test_preparation_filters_proxies_and_rejects_stale_sources(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from quantitive_analysis.bank_pipeline import digest
    from quantitive_analysis import contrastive
    from quantitive_analysis.contrastive import WordSpec, resolve_word
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'care': 1, 'two': 2, 'words': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    token_path = tmp_path / 'tokenizer.json'
    tokenizer.save(str(token_path))
    bank = tmp_path / 'bank.yaml'; bank.write_text('original source')
    revision = 'a' * 40
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'model': {'id': 'test', 'revision': revision}, 'questions': [{'id': 'Q01'}]}))
    rows = []
    for word in ['care', 'two words']:
        r = resolve_word(tokenizer, WordSpec(word))
        rows.append(dict(bank='bank', question='Q01', topic='topic', persona='caregiver',
                         word_set_type='contextualized', word=word, selected_token_id=r.token_id,
                         complete=r.complete, selected_surface_form=r.surface_form))
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'provenance': {'model_id': 'test', 'tokenizer_revision': revision},
                                'banks': [{'path': str(bank), 'sha256': digest(bank)}],
                                'resolver_sha256': digest(contrastive.__file__),
                                'analysis_tokenizer': {'path': str(token_path), 'sha256': digest(token_path)}, 'rows': rows}))
    manifest = prepare(audit, config, tmp_path / 'prepared')
    assert [r['retained_count'] for r in manifest['set_counts']] == [1, 2]
    selected = yaml.safe_load(Path(manifest['jobs'][0]['sets_path']).read_text())[0]['words']
    assert selected == [{'word': 'care', 'variants': [' care']}]
    bank.write_text('changed source')
    with pytest.raises(ValueError, match='Stale audit'):
        prepare(audit, config, tmp_path / 'stale')
    assert not (tmp_path / 'stale').exists()
