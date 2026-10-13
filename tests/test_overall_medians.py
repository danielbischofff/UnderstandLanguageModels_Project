from pathlib import Path
import sys
import json

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_overall_medians import median_statistics
from aggregate_mini_batch import METRICS
from aggregate_overall_mini_batch import KEYS


def test_medians_have_distinct_definitions_and_preserve_shared_questions():
    matrix = np.array([[0, 1, 2], [0, 1, 100]])
    indices = np.random.default_rng(9).integers(0, 2, (1000, 2))
    stats = median_statistics(matrix, indices)
    assert stats['median_pairs']['estimate'] == 1
    assert stats['median_persona_means']['estimate'] == 1
    opposite = np.array([[1,-1], [100,-100], [5,-5]])
    indices = np.random.default_rng(9).integers(0, 3, (1000, 3))
    for stat in median_statistics(opposite, indices).values():
        assert stat['estimate'] == 0
        assert stat['ci95'] == [0,0]
    # A question outlier remains influential when taking a median of persona means.
    distinct = median_statistics(np.array([[0,0,0], [100,0,100]]), np.array([[0,1]]))
    assert distinct['median_pairs']['estimate'] == 0
    assert distinct['median_persona_means']['estimate'] == 50


def test_all_experimental_estimates_match_source_and_original_mean():
    folder = ROOT / 'results/mini-batch-cluster-readout'
    experiment = json.loads((folder / 'overall-median-comparison.json').read_text())
    source = json.loads((folder / 'question-aggregation.json').read_text())
    original = json.loads((folder / 'overall-aggregation.json').read_text())
    assert len(experiment['conditions']) == 64
    for condition, old in zip(experiment['conditions'], original['conditions']):
        rows = [r for r in source['per_question'] if all(r[k] == condition[k] for k in KEYS)]
        lookup = {(r['question'],r['character']):r for r in rows}
        for metric,(_,delta) in METRICS.items():
            matrix = np.array([[lookup[q,p][delta] for p in experiment['personas']] for q in experiment['questions']])
            estimators=condition['metrics'][metric]['estimators']
            assert estimators['median_pairs']['estimate'] == float(np.median(matrix))
            assert estimators['median_persona_means']['estimate'] == float(np.median(matrix.mean(axis=0)))
            assert estimators['mean']['estimate'] == old['metrics'][metric]['mean_delta']
            assert estimators['mean']['ci95'] == old['metrics'][metric]['ci95']
