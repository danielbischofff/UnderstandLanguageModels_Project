// Validate the generated page's selectors and plotted data without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('results/mini-batch-cluster-readout/token-comparison.html', 'utf8');
const comparisons = JSON.parse(fs.readFileSync('results/mini-batch-cluster-readout/comparisons.json', 'utf8'));
const elements = {};
function element() {
  return { value: '', textContent: '', options: [], handlers: {},
    append(option) { this.options.push(option); if (this.options.length === 1) this.value = String(option.value); },
    addEventListener(name, handler) { this.handlers[name] = handler; },
    on(name, handler) { this.handlers[name] = handler; } };
}
const document = {
  getElementById(id) { return elements[id] ??= element(); },
  createElement() { return { value: '', textContent: '' }; }
};
for (const [id, value] of Object.entries({beta:'0', scheme:'layer-32', layer:'32', kind:'contextualized', policy:'complete_only'})) {
  document.getElementById(id).value = value;
}
const plots = {};
const context = vm.createContext({document, Plotly: {react(id, traces) { plots[id] = traces; }}});
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
vm.runInContext(scripts.at(-1)[1], context);
assert.equal(elements.question.options.length, 13);
assert.equal(elements.persona.options.length, 8);
let checked = 0;
const readouts = new Map();
function rawScore(trialIndex, row) {
  const filename = `outputs/analysis/mini-batch-local-readout/${String(trialIndex).padStart(4,'0')}-layer-${row.layer}.json`;
  if (!readouts.has(filename)) readouts.set(filename, JSON.parse(fs.readFileSync(filename, 'utf8')));
  return readouts.get(filename).scores.find(s => s.segment === 'response' && s.persona === row.character &&
    s.bank === row.bank && s.policy === row.policy && s.word_set_type === row.word_set_type);
}
for (const row of comparisons) {
  for (const [id, value] of Object.entries({question:row.question, persona:row.character, beta:row.beta,
      scheme:row.scheme, layer:row.layer, kind:row.word_set_type, policy:row.policy})) elements[id].value = String(value);
  elements.question.handlers.change();
  const trialIndices = vm.runInContext('[DATA.trials.indexOf(current.control), DATA.trials.indexOf(current.steered)]', context);
  const details=vm.runInContext('current.readout.sets[current.key].details',context);
  const targets=rawScore(trialIndices[1],row);
  assert.deepEqual(Array.from(details,d=>d.word),targets.words);
  assert.deepEqual(Array.from(details,d=>d.id),targets.token_ids);
  assert(details.every(d=>typeof d.token==='string'));
  if(row.policy==='complete_only')assert(details.every(d=>d.complete));
  assert.equal((elements['word-details'].innerHTML.match(/<tr>/g)||[]).length,targets.words.length);
  for (const [metric, field] of [['mrr','mean_reciprocal_rank'], ['cosine','activation_word_cosine']]) {
    const [control, steered] = plots[metric].filter(t=>t.name==='Unsteered'||t.name==='Steered');
    for (const trace of [control, steered]) {
      assert.equal(trace.x.length, trace.y.length);
      assert.equal(trace.x.length, trace.customdata.length);
      assert(trace.x.every((x, i) => x === i + 1));
    }
    const mean = xs => xs.length && xs.every(x => x !== null) ? xs.reduce((a,b)=>a+b,0) / xs.length : null;
    for (const [actual, expected] of [[mean(steered.y), rawScore(trialIndices[1],row)[field]],
        [mean(control.y), rawScore(trialIndices[0],row)[field]]]) {
      if (expected === null) assert.equal(actual, null);
      else assert(Math.abs(actual - expected) <= Math.max(1e-9, Math.abs(expected)*1e-6), `${row.question} ${row.character} ${field}: ${actual} vs ${expected}`);
    }
  }
  assert(elements['control-answer'].textContent.length > 0);
  assert(elements['steered-answer'].textContent.length > 0);
  elements.mrr.handlers.plotly_hover({points:[{x:1}]});
  assert(elements.inspect.textContent.includes('Unsteered · token 1:'));
  assert(elements.inspect.textContent.includes('Steered · token 1:'));
  checked++;
}
assert.equal(checked, 6144);
console.log(`Passed ${checked} question/persona/steering/layer/word-set combinations, paired traces, output texts, and hover inspection.`);
const source = vm.runInContext('DATA', context);
const unique = new Map(comparisons.map(row => [[row.character,row.beta,row.scheme,row.layer,row.word_set_type,row.policy].join('|'),row]));
let averaged = 0, sawReducedCount = false;
for (const row of unique.values()) {
  for (const [id,value] of Object.entries({question:'all',persona:row.character,beta:row.beta,scheme:row.scheme,
      layer:row.layer,kind:row.word_set_type,policy:row.policy})) elements[id].value=String(value);
  elements.question.handlers.change();
  const key=[row.character,row.word_set_type,row.policy].join('|');
  const controls=source.trials.filter(t=>t.persona==='unsteered');
  const steered=source.trials.filter(t=>t.persona===row.character&&t.steering.name===row.scheme&&t.steering.coefficient===row.beta);
  assert.equal(controls.length,12);assert.equal(steered.length,12);
  assert(elements['word-details'].innerHTML.includes('<td style="padding:5px 12px 5px 0;border-bottom:1px solid #edf1f6">Q01</td>'));
  assert(elements['word-details'].innerHTML.includes('<td style="padding:5px 12px 5px 0;border-bottom:1px solid #edf1f6">Q23</td>'));
  for (const metric of ['mrr','cosine']) {
    const traces=plots[metric].filter(t=>t.name==='Unsteered'||t.name==='Steered');
    for (const [trace,trials] of [[traces[0],controls],[traces[1],steered]]) {
      for (let i=0;i<trace.x.length;i++) {
        const values=trials.map(t=>{const layer=t.layers[row.layer];const j=layer.positions.indexOf(trace.x[i]-1);return j<0?null:layer.sets[key][metric][j];}).filter(v=>v!==null&&Number.isFinite(v));
        const expected=values.length?values.reduce((a,b)=>a+b,0)/values.length:null;
        assert.equal(trace.customdata[i],values.length);
        if(expected===null)assert.equal(trace.y[i],null);
        else assert(Math.abs(trace.y[i]-expected)<1e-12);
        if(values.length>0&&values.length<12)sawReducedCount=true;
      }
    }
  }
  elements.cosine.handlers.plotly_hover({points:[{x:1}]});
  assert(elements.inspect.textContent.includes('Mean MRR'));
  assert(elements.inspect.textContent.includes('(n='));
  for(const id of ['control-answer','steered-answer']) {
    assert(elements[id].textContent.includes('Q01 ·'));
    assert(elements[id].textContent.includes('Q23 ·'));
  }
  averaged++;
}
assert.equal(averaged,512);assert(sawReducedCount);
console.log(`Passed ${averaged} all-question averages, including every plotted value, contribution counts, and shorter-answer handling.`);
const area = vm.runInContext("positiveAreaTrace({x:[1,2,3],y:[1,1,1]},{x:[1,2,3],y:[0,2,0]})",context);
assert.deepEqual(Array.from(area.x),[1.5,2,2,1.5,null,2,2.5,2.5,2,null]);
assert.deepEqual(Array.from(area.y),[1,1,2,1,null,1,1,1,2,null]);
assert.equal(area.fillcolor,'rgba(22,163,74,0.16)');
assert.equal(vm.runInContext("positiveAreaTrace({x:[1,2],y:[1,1]},{x:[1,2],y:[0,1]}).x.length",context),0);
assert.equal(vm.runInContext("positiveAreaTrace({x:[1,2],y:[1,null]},{x:[1,2],y:[2,2]}).x.length",context),0);
console.log('Passed positive-area crossing, transparency, and missing-score checks.');
