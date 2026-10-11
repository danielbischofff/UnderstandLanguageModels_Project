import hashlib
import json
from pathlib import Path
import statistics

from scripts.aggregate_mini_batch import aggregate_metric, KEYS, METRICS


def test_equal_question_weight_and_paired_bootstrap():
    rows = [dict(control_available=True, value=11, delta=1, valid_positions=1),
            dict(control_available=True, value=13, delta=3, valid_positions=255)]
    result = aggregate_metric(rows, 'value', 'delta')
    assert result['mean_delta'] == 2  # Not a token-weighted mean near 3.
    assert result['mean_unsteered'] == 10
    assert result['mean_steered'] == 12
    assert result['median_delta'] == 2
    assert result['positive'] == 2
    assert result['ci95'] == [1, 3]
    assert aggregate_metric(rows, 'value', 'delta') == result


def test_missing_pairs_are_not_zero():
    rows = [dict(control_available=True, value=7, delta=-2),
            dict(control_available=False, value=100, delta=100),
            dict(control_available=True, value=None, delta=None)]
    result = aggregate_metric(rows, 'value', 'delta')
    assert result['n'] == 1
    assert result['negative'] == 1
    assert result['mean_delta'] == -2
    assert result['ci95'] == [-2, -2]
    assert aggregate_metric([], 'value', 'delta')['mean_delta'] is None


def test_published_aggregation_matches_source_pairs():
    folder = Path('results/mini-batch-cluster-readout')
    result = json.loads((folder / 'question-aggregation.json').read_text())
    source = folder / 'comparisons.json'
    rows = json.loads(source.read_text())
    assert result['method']['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result['per_question'] == rows
    assert len(result['conditions']) == 512
    assert len(result['questions']) == 12
    for condition in result['conditions']:
        group = [r for r in rows if all(r[k] == condition[k] for k in KEYS)]
        assert len(group) == 12
        assert {'Q01', 'Q23'} <= set(condition['questions'])
        for name, (value, delta) in METRICS.items():
            valid = [r for r in group if r['control_available'] and r[value] is not None and r[delta] is not None]
            score = condition['metrics'][name]
            assert score['n'] == len(valid)
            if valid:
                assert abs(score['mean_delta'] - statistics.mean(r[delta] for r in valid)) < 1e-12
                assert score['median_delta'] == statistics.median(r[delta] for r in valid)
                assert score['positive'] == sum(r[delta] > 0 for r in valid)
                assert score['negative'] == sum(r[delta] < 0 for r in valid)
                assert score['ci95'][0] <= score['ci95'][1]
