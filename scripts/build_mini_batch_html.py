#!/usr/bin/env python3
"""Build a dependency-free HTML overview from the completed readout summaries."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    comparisons = json.loads((args.input / "comparisons.json").read_text())
    topic_summary = json.loads((args.input / "topic-balanced-summary.json").read_text())
    overall = json.loads((args.input / 'overall-aggregation.json').read_text())
    overall_payload = json.dumps(overall['conditions'], separators=(',', ':'))
    sort_js = Path(__file__).with_name('mini_batch_table_sort.js').read_text()
    summary = (args.input / "summary.md").read_text()
    summary = summary.replace('Q01 and Q23 are calibration questions.',
                              'Q01 and Q23 were used only for pipeline checks; all twelve questions remain in the main analysis.')
    controls = {}
    control_keys = ('question', 'topic', 'character', 'layer', 'bank', 'word_set_type', 'policy')
    for row in comparisons:
        if not row['control_available']:
            continue
        key = tuple(row[k] for k in control_keys)
        control = {k: row[k] for k in control_keys}
        for metric in ('mean_reciprocal_rank', 'activation_word_cosine'):
            value, delta = row[metric], row[metric + '_delta']
            control[metric] = value - delta if value is not None and delta is not None else None
        if key in controls:
            for metric in ('mean_reciprocal_rank', 'activation_word_cosine'):
                a, b = controls[key][metric], control[metric]
                if a != b and (a is None or b is None or not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-15)):
                    raise ValueError(f'Inconsistent unsteered baseline: {key}, {metric}')
        else:
            controls[key] = control
    unsteered = list(controls.values())
    (args.output.parent / 'unsteered.json').write_text(json.dumps(unsteered, indent=2) + '\n')
    control_payload = json.dumps(unsteered, ensure_ascii=False, separators=(",", ":"))
    payload = json.dumps(comparisons, ensure_ascii=False, separators=(",", ":"))
    topic_payload = json.dumps(topic_summary, ensure_ascii=False, separators=(",", ":"))
    summary_text = html.escape(summary)
    alpha_link = ('<p class="overview-footer"><a href="../mini-alpha-analysis/index.html"><b>Alpha experiment · completed →</b></a> Open the additive-steering readouts, question comparisons, and mean-versus-median tables.</p>'
                  if (args.input.parent / 'mini-alpha-analysis/index.html').exists() else
                  '<p class="overview-footer"><b>Later: alpha experiment.</b> The additive-steering configuration is available. Its quantitative results will be added when the completed readouts are available.</p>')
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
.table-wrap{{overflow:auto;max-height:85vh;background:white;border:1px solid #dfe3e8;border-radius:8px;margin-top:12px}}
table{{border-collapse:separate;border-spacing:0;width:100%;white-space:nowrap}} th,td{{padding:7px 9px;border-bottom:1px solid #edf0f2;text-align:right}} th{{background:#eef2f5;position:sticky;top:0;z-index:2;box-shadow:0 1px 0 #cbd5e1}} th:first-child,td:first-child{{text-align:left}}
pre{{background:#1d2530;color:#e9eef4;padding:15px;border-radius:8px;overflow:auto;max-height:360px}}
.pos{{color:#087443}} .neg{{color:#a52a2a}}
.analysis-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:18px 0}} .analysis-card{{display:block;background:white;border:1px solid #dfe3e8;border-radius:10px;padding:18px;text-decoration:none;color:inherit}} a.analysis-card:hover{{border-color:#2563eb}} .analysis-card strong{{font-size:18px;color:#2563eb}} .analysis-card p{{margin:8px 0 0}} .planned{{opacity:.7}} .downloads a{{margin-right:18px;color:#2563eb}}
.overview-hero{{background:#17385c;color:white;padding:28px;border-radius:14px;margin-bottom:26px}} .overview-hero h1{{font-size:36px;line-height:1.2}} .overview-hero p{{color:#e0eafa;max-width:950px}} .overview-hero .badge{{display:inline-block;background:#ffffff22;padding:5px 10px;border-radius:6px;margin-bottom:12px;font-weight:650}} .section-note{{color:#5e6978;margin-top:-12px}} .primary-card{{border:2px solid #2563eb;background:#eff6ff}} details{{margin-top:24px}} summary{{cursor:pointer;font-size:20px;font-weight:650;padding:14px;background:#e8eef5;border-radius:8px}} .overview-footer{{margin-top:32px;color:#5e6978}}
.sort-button{{font:inherit;font-weight:inherit;color:inherit;background:none;border:0;padding:0;cursor:pointer}} .sort-button:hover{{text-decoration:underline}} #overall-preview small{{display:block;color:#617085}}
</style></head><body><main>
<header class="overview-hero"><span class="badge">MAIN ANALYSIS OVERVIEW</span><h1>J-Lens evaluation</h1>
<p>12 questions · 8 personas · 2 band widths · 2 steering schemes · 4 readout layers</p><p>Start with the overall effect, inspect differences between personas and questions, then explore the token-level readout. Every analysis page has a link back here. All pages work locally without a server.</p></header>
<h2>1. Main results</h2><p class="section-note">Does steering increase persona association overall, and which personas support the effect?</p>
<div class="analysis-grid">
<a class="analysis-card primary-card" href="overall-aggregation.html"><strong>Start here: overall effect</strong><p>All 64 configurations in one table. Equal-weight average across eight personas and twelve questions, with MRR, cosine, specificity, bootstrap intervals, and positive-persona counts.</p></a>
<a class="analysis-card" href="question-aggregation.html"><strong>Aggregate across questions</strong><p>Equal-question mean and median effects, positive-question counts, specificity, and exploratory bootstrap intervals. Includes individual question results.</p></a>
<a class="analysis-card" href="overall-median-comparison.html"><strong>Experimental: mean versus median</strong><p>Compare the original mean with the median of persona–question differences or the median of persona means. All 64 configurations, with paired-question bootstrap intervals.</p></a>
</div>
<h3>Overall effects at a glance</h3><p class="muted">Eight personas × twelve questions; complete tokens only. Compare both steering schemes and band widths below. <a href="overall-aggregation.html">Open all 64 configurations →</a></p>
<div class="controls"><label>Readout layer<select id="preview-layer"><option>29</option><option selected>32</option><option>35</option><option>62</option></select></label><label>Word bank<select id="preview-kind"><option value="contextualized">Contextualized</option><option value="non_contextualized">General</option></select></label></div>
<p class="muted">Click a column heading to sort; click again to reverse. Metric columns sort by the mean paired difference. Cells retain the exploratory 95% bootstrap intervals.</p>
<div class="table-wrap"><table><thead><tr><th id="preview-head-scheme"><button class="sort-button" id="preview-sort-scheme">Steered layers</button></th><th id="preview-head-beta"><button class="sort-button" id="preview-sort-beta">β</button></th><th id="preview-head-mrr"><button class="sort-button" id="preview-sort-mrr">MRR Δ</button></th><th id="preview-head-cosine"><button class="sort-button" id="preview-sort-cosine">Cosine Δ</button></th><th id="preview-head-mrr_margin"><button class="sort-button" id="preview-sort-mrr_margin">MRR specificity Δ</button></th><th id="preview-head-cosine_margin"><button class="sort-button" id="preview-sort-cosine_margin">Cosine specificity Δ</button></th></tr></thead><tbody id="overall-preview"></tbody></table></div>
<h2>2. Understand the differences</h2><p class="section-note">Which questions drive the result, and how does association change during an answer?</p><div class="analysis-grid">
<a class="analysis-card" href="question-comparison.html"><strong>Compare questions</strong><p>MRR and cosine paired differences side by side for all twelve questions. Compare one persona or all personas, inspect the wording, and see which questions drive the result.</p></a>
<a class="analysis-card" href="secondary-parameter-evaluation.html"><strong>Secondary Parameter Evaluation</strong><p>Violin plots for contextualised versus non-contextualised word sets and token policies, comparing both MRR and cosine under fixed experimental settings.</p></a>
<a class="analysis-card" href="token-comparison.html"><strong>Inspect token trajectories</strong><p>Per-token MRR and cosine against the unsteered answer, with target words and token IDs, answer texts, and an all-question average view.</p></a>
</div><h2>3. Data and measurement checks</h2><p class="section-note">Inspect the underlying scores, metric definitions, and target-word tokenization.</p><div class="analysis-grid">
<a class="analysis-card" href="#summary-tables"><strong>Browse scores and baselines</strong><p>Persona summary tables, absolute unsteered scores, steered scores, and paired differences under each experimental condition.</p></a>
<a class="analysis-card" href="metrics-review.md"><strong>Metric and data audit</strong><p>Review the metric definitions and checks on the exported experiment data.</p></a>
<a class="analysis-card" href="word-tokenization-summary.md"><strong>Word-tokenization audit</strong><p>Review target-word resolution, complete tokens, and the treatment of multi-token word proxies.</p></a>
</div>
{alpha_link}
<p class="downloads"><b>Source data:</b> <a href="comparisons.json" download>Per-question comparisons</a><a href="question-aggregation.json" download>Aggregation and intervals</a><a href="unsteered.json" download>Unsteered scores</a><a href="topic-balanced-summary.json" download>Persona summaries</a></p>
<p class="muted">Paired difference means steered minus unsteered on the same question. Specificity compares the intended bank with seven rival personas in the same bank. The main analysis retains all twelve questions; Q01 and Q23 were used only for pipeline checks. Bootstrap intervals are exploratory given the small, deliberately selected question set. Readout association does not measure literal word mentions.</p>
<details id="summary-tables"><summary>Open detailed score tables and baselines</summary>
<p class="muted">Exact full-vocabulary MRR and activationWordCosine. Answer positions only; the final generated token has no cached input activation. Values are paired steering-minus-control deltas unless labelled margin delta.</p>
<div class="cards" id="cards"></div>
<div class="controls"><label>Subset<select id="subset"></select></label><label>Word policy<select id="policy"></select></label><label>Word-set type<select id="kind"></select></label><label>Layer<select id="layer"></select></label><label>Scheme<select id="scheme"></select></label><label>Beta<select id="beta"></select></label></div>
<h2>Topic-balanced persona deltas</h2><div class="table-wrap"><table><thead><tr><th>Persona</th><th>MRR Δ</th><th>Cosine Δ</th><th>MRR margin Δ</th><th>Cosine margin Δ</th><th>Valid questions</th></tr></thead><tbody id="summary"></tbody></table></div>
<h2>Unsteered results per question</h2><p class="muted">One unsteered answer per question, evaluated against each persona's word set. Baselines are reconstructed as steered score minus paired delta from the completed comparison JSON; floating-point rounding may affect the last digits. Scheme and beta do not apply to this table.</p><div class="table-wrap"><table><thead><tr><th>Question</th><th>Topic</th><th>Word-set persona</th><th>Bank</th><th>MRR</th><th>Activation cosine</th></tr></thead><tbody id="controls"></tbody></table></div>
<h2>Per-question steering results</h2><p class="muted" id="selection-count"></p><div class="table-wrap"><table><thead><tr><th>Question</th><th>Persona</th><th>Layer</th><th>Scheme</th><th>β</th><th>Steered MRR</th><th>Steered cosine</th><th>MRR Δ</th><th>Cosine Δ</th><th>MRR margin Δ</th><th>Cosine margin Δ</th><th>Positions</th></tr></thead><tbody id="rows"></tbody></table></div>
</details><details><summary>Open method summary</summary><pre>{summary_text}</pre></details>
<script>
{sort_js}
const overallConditions={overall_payload};
const sortPreview=createTableSorter('preview-',['scheme','beta','mrr','cosine','mrr_margin','cosine_margin'],renderOverallPreview);
function renderOverallPreview(){{const conditions=sortPreview(overallConditions.filter(r=>String(r.layer)===document.getElementById('preview-layer').value&&r.word_set_type===document.getElementById('preview-kind').value&&r.policy==='complete_only'));document.getElementById('overall-preview').innerHTML=conditions.map(r=>`<tr><td>${{r.scheme==='layer-32'?'32':'29 + 32 + 35'}}</td><td>${{r.beta}}</td>`+['mrr','cosine','mrr_margin','cosine_margin'].map(m=>{{const s=r.metrics[m];return `<td><b class="${{s.mean_delta>0?'pos':'neg'}}">${{Number(s.mean_delta).toPrecision(5)}}</b><small>95% interval [${{Number(s.ci95[0]).toPrecision(5)}}, ${{Number(s.ci95[1]).toPrecision(5)}}]</small><small>${{s.positive_personas}} / 8 positive personas</small></td>`}}).join('')+'</tr>').join('')}}
for(const id of ['preview-layer','preview-kind'])document.getElementById(id).addEventListener('change',renderOverallPreview);renderOverallPreview();
const rows={payload}; const topic={topic_payload}; const controls={control_payload};
const fields=['subset','policy','kind','layer','scheme','beta'];
const by=(xs,k)=>[...new Set(xs.map(x=>x[k]).filter(x=>x!==null&&x!==undefined))].sort((a,b)=>String(a).localeCompare(String(b),undefined,{{numeric:true}}));
const val=x=>x==null?'—':Number(x).toPrecision(6); const cls=x=>x>0?'pos':x<0?'neg':'';
function fill(id,values){{const e=document.getElementById(id);e.innerHTML='';for(const v of values){{const o=document.createElement('option');o.value=v;o.textContent=id==='subset'?(v==='all_questions'?'All 12 questions (main analysis)':'Optional exclusion check: omit Q01/Q23'):v;e.append(o)}}}}
const subsets=by(topic,'subset'), policies=by(topic,'policy'), kinds=by(topic,'word_set_type'), layers=by(topic,'layer'), schemes=by(topic,'scheme'), betas=by(topic,'beta');
fill('subset',subsets);fill('policy',policies);fill('kind',kinds);fill('layer',layers);fill('scheme',schemes);fill('beta',betas);
function selected(id){{return document.getElementById(id).value}}
function common(x){{return x.policy===selected('policy')&&x.word_set_type===selected('kind')&&String(x.layer)===selected('layer')}}
function inSubset(x){{return selected('subset')==='all_questions'||!['Q01','Q23'].includes(x.question)}}
function condition(x){{return common(x)&&x.scheme===selected('scheme')&&String(x.beta)===selected('beta')}}
function render(){{const t=topic.filter(x=>x.subset===selected('subset')&&condition(x)),r=rows.filter(x=>inSubset(x)&&condition(x)),c=controls.filter(x=>inSubset(x)&&common(x));
document.getElementById('controls').innerHTML=c.map(x=>`<tr><td>${{x.question}}</td><td>${{x.topic}}</td><td>${{x.character}}</td><td>${{x.bank}}</td><td>${{val(x.mean_reciprocal_rank)}}</td><td>${{val(x.activation_word_cosine)}}</td></tr>`).join('');
document.getElementById('selection-count').textContent=`${{new Set(r.map(x=>x.question)).size}} questions · ${{r.length}} steering rows · ${{c.length}} unsteered word-set scores`;
document.getElementById('summary').innerHTML=t.map(x=>`<tr><td>${{x.character}}</td><td class="${{cls(x.mean_reciprocal_rank_delta)}}">${{val(x.mean_reciprocal_rank_delta)}}</td><td class="${{cls(x.activation_word_cosine_delta)}}">${{val(x.activation_word_cosine_delta)}}</td><td class="${{cls(x.mean_reciprocal_rank_margin_delta)}}">${{val(x.mean_reciprocal_rank_margin_delta)}}</td><td class="${{cls(x.activation_word_cosine_margin_delta)}}">${{val(x.activation_word_cosine_margin_delta)}}</td><td>${{x.mean_reciprocal_rank_delta_valid_questions??'—'}}</td></tr>`).join('');
document.getElementById('rows').innerHTML=r.map(x=>`<tr><td>${{x.question}}</td><td>${{x.character}}</td><td>${{x.layer}}</td><td>${{x.scheme}}</td><td>${{x.beta}}</td><td>${{val(x.mean_reciprocal_rank)}}</td><td>${{val(x.activation_word_cosine)}}</td><td class="${{cls(x.mean_reciprocal_rank_delta)}}">${{val(x.mean_reciprocal_rank_delta)}}</td><td class="${{cls(x.activation_word_cosine_delta)}}">${{val(x.activation_word_cosine_delta)}}</td><td class="${{cls(x.mean_reciprocal_rank_margin_delta)}}">${{val(x.mean_reciprocal_rank_margin_delta)}}</td><td class="${{cls(x.activation_word_cosine_margin_delta)}}">${{val(x.activation_word_cosine_margin_delta)}}</td><td>${{x.valid_positions??'—'}}</td></tr>`).join('');}}
for(const id of fields)document.getElementById(id).addEventListener('change',render);render();
document.querySelector('a[href="#summary-tables"]').addEventListener('click',()=>{{document.getElementById('summary-tables').open=true}});
document.getElementById('cards').innerHTML=`<div class="card"><b>${{rows.length}}</b>comparison rows</div><div class="card"><b>${{topic.length}}</b>topic summaries</div><div class="card"><b>${{new Set(rows.map(x=>x.question)).size}}</b>questions</div><div class="card"><b>${{new Set(rows.map(x=>x.layer)).size}}</b>layers</div>`;
</script></main></body></html>'''
    args.output.write_text(doc)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
