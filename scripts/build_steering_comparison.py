#!/usr/bin/env python3
"""Combine completed alpha and band results without pooling their controls."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from plotly.offline import get_plotlyjs

METRICS=('mrr','cosine')


def load_method(folder,coefficient):
    question=json.loads((folder/'question-aggregation.json').read_text())
    overall=json.loads((folder/'overall-aggregation.json').read_text())
    median=json.loads((folder/'overall-median-comparison.json').read_text())
    def conditions(rows):
        return [{**{k:v for k,v in row.items() if k!='metrics'},
                 'metrics':{m:row['metrics'][m] for m in METRICS}} for row in rows]
    keys=('question','topic','character','layer','scheme',coefficient,'bank','word_set_type','policy',
          'valid_positions','retained_words','control_available','mean_reciprocal_rank','mean_reciprocal_rank_delta',
          'activation_word_cosine','activation_word_cosine_delta')
    rows=[{k:row[k] for k in keys} for row in question['per_question']]
    return dict(conditions=conditions(question['conditions']),overall=conditions(overall['conditions']),
                medians=conditions(median['conditions']),rows=rows),question,overall


def extract_token_payload(path):
    text=path.read_text()
    start=text.rfind('const DATA=')+len('const DATA=')
    payload,length=json.JSONDecoder().raw_decode(text[start:])
    return payload,text[start:start+length]


def build(band,alpha,output):
    band_data,bq,bo=load_method(band,'beta')
    alpha_data,aq,ao=load_method(alpha,'alpha')
    if bq['questions']!=aq['questions'] or bo['personas']!=ao['personas']:
        raise ValueError('Methods must use the same questions and selected personas')
    result=dict(questions=bq['questions'],personas=bo['personas'],band=band_data,alpha=alpha_data,
                sources={method:{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                  for name in ('question-aggregation.json','overall-aggregation.json','overall-median-comparison.json')}
                         for method,folder in [('band',band),('alpha',alpha)]})
    assets=output.parent/'steering-comparison';assets.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(result,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    (assets/'comparison-data.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    template=Path(__file__).with_name('steering_comparison_template.html').read_text()
    output.write_text(template.replace('/*DATA*/',payload).replace('/*PLOTLY*/',get_plotlyjs()))
    # Keep the large token datasets separate: ordinary local script loading works offline.
    for method,folder in [('band',band),('alpha',alpha)]:
        tokens,raw=extract_token_payload(folder/'token-comparison.html')
        if tokens['questions']!=result['questions'] or set(tokens['personas'])!=set(result['personas']):
            raise ValueError('Token page metadata differs from summaries')
        (assets/f'token-{method}-data.js').write_text('window.STEERING_TOKENS=window.STEERING_TOKENS||{};window.STEERING_TOKENS.'+method+'='+raw+';\n')
    token_template=Path(__file__).with_name('steering_token_comparison_template.html').read_text()
    original=Path(__file__).with_name('mini_batch_token_template.html').read_text()
    body=original.split('<div id="question-text"',1)[1].split('</main><script>',1)[0]
    body='<div id="question-text"'+body
    body=body.replace('β = 0 still applies steering: it pins the persona projection. ','')
    code=original.rsplit('<script>',1)[1].split('</script>',1)[0]
    code=code.replace('const DATA=/*DATA*/;','')
    begin=code.index("function fill(id,values)")
    end=code.index('let current;',begin)
    code=code[:begin]+code[end:]
    code=code.replace("const el=id=>document.getElementById(id), selected=id=>el(id).value;",
                      "const el=id=>document.getElementById(prefix+id), selected=id=>document.getElementById(id==='beta'?coefficient:id).value;")
    code=code.replace('Plotly.react(metric,','Plotly.react(prefix+metric,')
    code=code.replace("rangemode:'tozero'","rangemode:'tozero',range:TOKEN_RANGES[metric]")
    code=code.replace("for(const id of ids)el(id).addEventListener('change',render);",'')
    code+='\nreturn {render,choice};'
    for method in ('band','alpha'):
        token_template=token_template.replace('/*BODY_'+method.upper()+'*/',re.sub(r'id="([^"]+)"',lambda m:'id="'+method+'-'+m[1]+'"',body))
    token_template=token_template.replace('/*RENDERER*/',code).replace('/*PLOTLY*/',get_plotlyjs())
    (assets/'token-comparison.html').write_text(token_template)
    # Preserve old entry points as routes to the new combined overview.
    for folder in (band,alpha):
        (folder/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../index.html"><title>Combined steering overview</title></head><body><a href="../index.html">Open alpha and band-capping comparison →</a></body></html>')
    print(f'Created {output}, combined token page and datasets. Both methods: 512 persona conditions, 64 overall configurations.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--band',type=Path,default=Path('results/mini-batch-cluster-readout'))
    parser.add_argument('--alpha',type=Path,default=Path('results/mini-alpha-analysis'))
    parser.add_argument('--output',type=Path,default=Path('results/index.html'))
    args=parser.parse_args()
    build(args.band,args.alpha,args.output)
