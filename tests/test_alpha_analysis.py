import json
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from quantitive_analysis.local_summary import paired_comparisons, coefficient_field
from quantitive_analysis.local_readout import assemble_activations
from build_alpha_analysis import build


def test_complete_and_decode_only_activation_alignment(tmp_path):
    for n in (2,3):
        event=dict(layer=32,point='post_intervention',cache_positions=list(range(n)),
                   input_token_ids=[[10,11,12][:n]],tensor_key='x')
        (tmp_path/'manifest.json').write_text(json.dumps({'events':[event]}))
        save_file({'x':torch.ones(1,n,4)},str(tmp_path/'activations.safetensors'))
        positions,_=assemble_activations(tmp_path,32,[10,11,12])
        assert positions.tolist()==list(range(n))
    event['cache_positions']=[0,2,3]
    (tmp_path/'manifest.json').write_text(json.dumps({'events':[event]}))
    with pytest.raises((ValueError,IndexError)):
        assemble_activations(tmp_path,32,[10,11,12])


def test_alpha_pipeline_keeps_coefficients_controls_and_reports_separate(tmp_path):
    run=tmp_path/'run';run.mkdir()
    readouts=tmp_path/'readouts';readouts.mkdir()
    output=tmp_path/'report'
    personas=[f'p{i}' for i in range(8)]
    questions=[dict(id=f'Q{i}',text=f'Question {i}') for i in range(2)]
    trials=[]
    docs=[]
    for question in questions:
        for persona in ['unsteered',*personas]:
            for alpha in ([None] if persona=='unsteered' else [1.5,4.0]):
                steering=None if alpha is None else dict(name='layer-32',method='additive',coefficient=alpha,layers=[32])
                trial=dict(question=question,character=persona,steering=steering)
                number=len(trials);trials.append(dict(metadata=trial))
                for layer in (29,32,35,62):
                    scores=[]
                    for p in personas:
                        value=.1+(0 if alpha is None else alpha*.01+(alpha*.001 if p==persona else 0))
                        scores.append(dict(segment='response',bank='test_bank',persona=p,word_set_type='contextualized',
                                           policy='complete_only',topic=question['id'],valid_positions=3,retained_count=6,unique_token_count=6,
                                           mean_reciprocal_rank=value,activation_word_cosine=value))
                    doc=dict(trial_index=number,question=question['id'],character=persona,steering=steering,layer=layer,
                             at_token_limit=False,scores=scores)
                    docs.append(doc)
                    (readouts/f'{number:04d}-layer-{layer}.json').write_text(json.dumps(doc))
    (run/'index.json').write_text(json.dumps(dict(document=dict(experiment=dict(questions=questions)),trials=trials)))
    assert coefficient_field(docs)=='alpha'
    pairs=paired_comparisons(docs)
    assert all('alpha' in row and 'beta' not in row for row in pairs)
    assert all(row['control_available'] for row in pairs)
    build(run,readouts,output)
    data=json.loads((output/'overall-aggregation.json').read_text())
    assert len(data['conditions'])==8
    for condition in data['conditions']:
        assert condition['metrics']['mrr']['mean_delta']==pytest.approx(.011*condition['alpha'])
        assert condition['metrics']['mrr_margin']['mean_delta']==pytest.approx(.001*condition['alpha'])
        assert condition['metrics']['mrr']['n_pairs']==16
    html=(output/'overall-aggregation.html').read_text()
    assert 'Steering strength α' in html and 'Band-width' not in html
    assert 'pins the persona' not in html and 'id="alpha"' in html
    for name in ['question-aggregation.html','question-comparison.html','overall-median-comparison.html']:
        page=(output/name).read_text()
        assert 'id="alpha"' in page and 'Band-width' not in page
        assert 'value="1.5"' in page or name=='overall-median-comparison.html'
        assert 'value="0.25"' not in page
    (readouts/'0000-layer-29.json').unlink()
    with pytest.raises(ValueError,match='Incomplete'):
        build(run,readouts,output)
