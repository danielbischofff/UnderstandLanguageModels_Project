import json
from pathlib import Path
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from quantitive_analysis.local_readout import assemble_activations, ranks_for_targets, mean_word_cosines, score_sets


def test_rank_matches_brute_force_with_ties():
    logits = np.array([[3., 2., 2., 1.], [0., 5., 1., 4.]])
    ids = np.array([1, 2, 3])
    expected = 1 + (logits[:, :, None] > logits[:, ids, None].transpose(0, 2, 1)).sum(axis=1)
    np.testing.assert_array_equal(ranks_for_targets(logits, ids), expected)
    with pytest.raises(ValueError, match='Nonfinite'):
        ranks_for_targets(np.array([[np.nan, 1.]]), [1])


def test_activation_cosine_is_mean_of_words_not_centroid():
    cosines = mean_word_cosines([[1., 0.]], [[1., 0.], [0., 1.]])
    assert cosines.mean() == pytest.approx(.5)
    assert cosines.mean() != pytest.approx(1 / np.sqrt(2))
    assert np.isnan(mean_word_cosines([[0., 0.]], [[1., 0.]])).all()


def test_cached_event_alignment_and_overlap_rejection(tmp_path):
    events = [dict(layer=32, point='post_intervention', phase='prefill', cache_positions=[0, 1],
                   input_token_ids=[[5, 6]], tensor_key='a'),
              dict(layer=32, point='post_intervention', phase='decode', cache_positions=[2],
                   input_token_ids=[[7]], tensor_key='b')]
    save_file({'a': torch.tensor([[[1., 2.], [3., 4.]]]), 'b': torch.tensor([[[5., 6.]]])},
              tmp_path / 'activations.safetensors')
    (tmp_path / 'manifest.json').write_text(json.dumps({'events': events}))
    positions, tensor = assemble_activations(tmp_path, 32, [5, 6, 7, 8])
    assert positions.tolist() == [0, 1, 2]
    assert tensor.tolist() == [[1., 2.], [3., 4.], [5., 6.]]
    events.append(events[1])
    (tmp_path / 'manifest.json').write_text(json.dumps({'events': events}))
    with pytest.raises(ValueError, match='Duplicate'):
        assemble_activations(tmp_path, 32, [5, 6, 7, 8])


def test_prompt_excluded_and_duplicate_word_ids_retain_word_weighting():
    sets = [{'token_ids': [10, 10, 20]}]
    result = score_sets(sets, np.array([10, 20]), np.array([[1, 1], [2, 4]]),
                        np.array([[1., 1.], [.6, 0.]]), np.array([0, 1]), 1)
    response = next(r for r in result if r['segment'] == 'response')
    assert response['valid_positions'] == 1
    assert response['mean_reciprocal_rank'] == pytest.approx((.5+.5+.25)/3)
    assert response['activation_word_cosine'] == pytest.approx(.4)


def test_margin_delta_uses_same_set_control_and_all_seven_rivals():
    from quantitive_analysis.local_summary import paired_comparisons
    def document(character, own, rival):
        return dict(character=character, question='Q01', layer=32, at_token_limit=False,
                    steering=None if character=='unsteered' else {'name':'layer-32','coefficient':0.},
                    scores=[dict(segment='response', persona=p, bank='bank', word_set_type='contextualized',
                                 policy='complete_only', topic='topic', valid_positions=2,
                                 retained_count=6, unique_token_count=6,
                                 mean_reciprocal_rank=own if p=='scientist' else rival,
                                 activation_word_cosine=own if p=='scientist' else rival)
                            for p in ['scientist']+[f'r{i}' for i in range(7)]])
    rows=paired_comparisons([document('unsteered', .1, .05),document('scientist', .4, .2)])
    assert len(rows)==1
    assert rows[0]['mean_reciprocal_rank_delta']==pytest.approx(.3)
    assert rows[0]['mean_reciprocal_rank_persona_margin']==pytest.approx(.2)
    assert rows[0]['mean_reciprocal_rank_margin_delta']==pytest.approx(.15)
    missing=paired_comparisons([document('scientist', .4, .2)])[0]
    assert missing['mean_reciprocal_rank_delta'] is None
    assert missing['mean_reciprocal_rank_margin_delta'] is None
