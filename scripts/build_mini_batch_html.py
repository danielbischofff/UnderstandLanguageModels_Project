#!/usr/bin/env python3
"""Build a dependency-free HTML overview from the completed readout summaries."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    comparisons = json.loads((args.input / "comparisons.json").read_text())
    topic_summary = json.loads((args.input / "topic-balanced-summary.json").read_text())
    summary = (args.input / "summary.md").read_text()
    payload = json.dumps(comparisons, ensure_ascii=False, separators=(",", ":"))
    topic_payload = json.dumps(topic_summary, ensure_ascii=False, separators=(",", ":"))
    summary_text = html.escape(summary)
    doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mini-batch J-Lens readout overview</title>
<style>
body{{font:14px system-ui,sans-serif;line-height:1.4;color:#18202a;background:#f6f7f9;margin:0}}
main{{max-width:1500px;margin:0 auto;padding:28px}} h1{{margin:0 0 6px}} h2{{margin-top:30px}}
.muted{{color:#5e6978}} .cards{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}}
.card{{background:white;border:1px solid #dfe3e8;border-radius:8px;padding:12px 16px;min-width:145px}}
.card b{{font-size:22px;display:block}} .controls{{background:white;border:1px solid #dfe3e8;border-radius:8px;padding:12px;display:flex;gap:12px;flex-wrap:wrap}}
label{{display:flex;flex-direction:column;gap:4px;font-size:12px;color:#5e6978}} select{{font:inherit;padding:5px}}
.table-wrap{{overflow:auto;background:white;border:1px solid #dfe3e8;border-radius:8px;margin-top:12px}}
table{{border-collapse:collapse;width:100%;white-space:nowrap}} th,td{{padding:7px 9px;border-bottom:1px solid #edf0f2;text-align:right}} th{{background:#eef2f5;position:sticky;top:0}} th:first-child,td:first-child{{text-align:left}}
pre{{background:#1d2530;color:#e9eef4;padding:15px;border-radius:8px;overflow:auto;max-height:360px}}
.pos{{color:#087443}} .neg{{color:#a52a2a}}
</style></head><body><main>
<h1>Mini-batch J-Lens readout overview</h1>
<p class="muted">Exact full-vocabulary MRR and activationWordCosine. Answer positions only; the final generated token has no cached input activation. Values are paired steering-minus-control deltas unless labelled margin delta.</p>
<div class="cards" id="cards"></div>
<div class="controls"><label>Subset<select id="subset"></select></label><label>Word policy<select id="policy"></select></label><label>Word-set type<select id="kind"></select></label><label>Layer<select id="layer"></select></label><label>Scheme<select id="scheme"></select></label><label>Beta<select id="beta"></select></label></div>
<h2>Topic-balanced persona deltas</h2><div class="table-wrap"><table><thead><tr><th>Persona</th><th>MRR Δ</th><th>Cosine Δ</th><th>MRR margin Δ</th><th>Cosine margin Δ</th><th>Valid questions</th></tr></thead><tbody id="summary"></tbody></table></div>
<h2>Per-question comparisons</h2><div class="table-wrap"><table><thead><tr><th>Question</th><th>Persona</th><th>Layer</th><th>Scheme</th><th>β</th><th>MRR Δ</th><th>Cosine Δ</th><th>MRR margin Δ</th><th>Cosine margin Δ</th><th>Positions</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Method summary</h2><pre>{summary_text}</pre>
<script>
const rows={payload}; const topic={topic_payload};
const fields=['subset','policy','kind','layer','scheme','beta'];
const by=(xs,k)=>[...new Set(xs.map(x=>x[k]).filter(x=>x!==null&&x!==undefined))].sort((a,b)=>String(a).localeCompare(String(b),undefined,{{numeric:true}}));
const val=x=>x==null?'—':Number(x).toPrecision(6); const cls=x=>x>0?'pos':x<0?'neg':'';
function fill(id,values){{const e=document.getElementById(id);e.innerHTML='';for(const v of values){{const o=document.createElement('option');o.value=v;o.textContent=v;e.append(o)}}}}
const subsets=by(topic,'subset'), policies=by(topic,'policy'), kinds=by(topic,'word_set_type'), layers=by(topic,'layer'), schemes=by(topic,'scheme'), betas=by(topic,'beta');
fill('subset',subsets);fill('policy',policies);fill('kind',kinds);fill('layer',layers);fill('scheme',schemes);fill('beta',betas);
function selected(id){{return document.getElementById(id).value}}
function match(x){{return x.subset===selected('subset')&&x.policy===selected('policy')&&x.word_set_type===selected('kind')&&String(x.layer)===selected('layer')&&x.scheme===selected('scheme')&&String(x.beta)===selected('beta')}}
function render(){{const t=topic.filter(match),r=rows.filter(x=>x.subset===selected('subset')&&x.policy===selected('policy')&&x.word_set_type===selected('kind')&&String(x.layer)===selected('layer')&&x.scheme===selected('scheme')&&String(x.beta)===selected('beta'));
document.getElementById('summary').innerHTML=t.map(x=>`<tr><td>${{x.character}}</td><td class="${{cls(x.mean_reciprocal_rank_delta)}}">${{val(x.mean_reciprocal_rank_delta)}}</td><td class="${{cls(x.activation_word_cosine_delta)}}">${{val(x.activation_word_cosine_delta)}}</td><td class="${{cls(x.mean_reciprocal_rank_margin_delta)}}">${{val(x.mean_reciprocal_rank_margin_delta)}}</td><td class="${{cls(x.activation_word_cosine_margin_delta)}}">${{val(x.activation_word_cosine_margin_delta)}}</td><td>${{x.mean_reciprocal_rank_delta_valid_questions??'—'}}</td></tr>`).join('');
document.getElementById('rows').innerHTML=r.map(x=>`<tr><td>${{x.question}}</td><td>${{x.character}}</td><td>${{x.layer}}</td><td>${{x.scheme}}</td><td>${{x.beta}}</td><td class="${{cls(x.mean_reciprocal_rank_delta)}}">${{val(x.mean_reciprocal_rank_delta)}}</td><td class="${{cls(x.activation_word_cosine_delta)}}">${{val(x.activation_word_cosine_delta)}}</td><td class="${{cls(x.mean_reciprocal_rank_margin_delta)}}">${{val(x.mean_reciprocal_rank_margin_delta)}}</td><td class="${{cls(x.activation_word_cosine_margin_delta)}}">${{val(x.activation_word_cosine_margin_delta)}}</td><td>${{x.valid_positions??'—'}}</td></tr>`).join('');}}
for(const id of fields)document.getElementById(id).addEventListener('change',render);render();
document.getElementById('cards').innerHTML=`<div class="card"><b>${{rows.length}}</b>comparison rows</div><div class="card"><b>${{topic.length}}</b>topic summaries</div><div class="card"><b>${{new Set(rows.map(x=>x.question)).size}}</b>questions</div><div class="card"><b>${{new Set(rows.map(x=>x.layer)).size}}</b>layers</div>`;
</script></main></body></html>'''
    args.output.write_text(doc)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
