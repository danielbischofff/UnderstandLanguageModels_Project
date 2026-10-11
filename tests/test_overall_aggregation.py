import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from aggregate_overall_mini_batch import summarize
from aggregate_mini_batch import METRICS


def test_shared_question_bootstrap_preserves_cross_persona_cancellation():
    rows = []
    for q, magnitude in [('q1', 1), ('q2', 100), ('q3', 8)]:
        for persona, sign in [('a', 1), ('b', -1)]:
            row = dict(question=q, character=persona, control_available=True)
            for value, delta in METRICS.values():
                row[value] = 200 + sign*magnitude
                row[delta] = sign*magnitude
            rows.append(row)
    result = summarize(rows, ['q1', 'q2', 'q3'], ['a', 'b'])
    for stats in result.values():
        assert stats['mean_delta'] == 0
        assert stats['ci95'] == [0, 0]
        assert stats['positive_personas'] == 1
        assert stats['positive_pairs'] == 3
        assert stats['mean_unsteered'] == 200
    with pytest.raises(ValueError, match='Incomplete'):
        summarize(rows[:-1], ['q1', 'q2', 'q3'], ['a', 'b'])
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(rows + [rows[0]], ['q1', 'q2', 'q3'], ['a', 'b'])
    rows[0]['control_available'] = False
    with pytest.raises(ValueError, match='Missing'):
        summarize(rows, ['q1', 'q2', 'q3'], ['a', 'b'])


def test_all_configurations_match_existing_persona_means_and_question_bootstrap():
    folder = ROOT / 'results/mini-batch-cluster-readout'
    overall = json.loads((folder / 'overall-aggregation.json').read_text())
    existing = json.loads((folder / 'question-aggregation.json').read_text())
    assert len(overall['conditions']) == 64
    indices = np.random.default_rng(20261007).integers(0, 12, (10000, 12))
    for condition in overall['conditions']:
        personas = [c for c in existing['conditions'] if all(c[k] == condition[k] for k in ('layer','scheme','beta','word_set_type','policy'))]
        assert len(personas) == 8
        for metric, stats in condition['metrics'].items():
            means = [c['metrics'][metric]['mean_delta'] for c in personas]
            assert stats['mean_delta'] == pytest.approx(np.mean(means), abs=1e-15)
            assert stats['positive_personas'] == sum(x > 0 for x in means)
            assert stats['n_pairs'] == 96
            question_means = [stats['question_mean_deltas'][q] for q in overall['questions']]
            expected = np.quantile(np.array(question_means)[indices].mean(axis=1), [.025,.975])
            np.testing.assert_allclose(stats['ci95'], expected, atol=1e-15, rtol=0)
