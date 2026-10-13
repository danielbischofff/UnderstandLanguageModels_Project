const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const folder='results/mini-batch-cluster-readout';
const html=fs.readFileSync(`${folder}/secondary-parameter-evaluation.html`,'utf8');
const rows=JSON.parse(fs.readFileSync(`${folder}/comparisons.json`,'utf8'));
function load(source=html){
 const elements={},plots={};
 const document={getElementById(id){return elements[id]??={value:'',firstChild:{textContent:''},options:[],handlers:{},append(o){this.options.push(o);if(this.options.length===1)this.value=String(o.value)},addEventListener(event,handler){this.handlers[event]=handler}}},createElement(){return {value:'',textContent:''}}};
 for(const [id,value] of Object.entries({score:'delta',policy:'complete_only',kind:'contextualized'}))document.getElementById(id).value=value;
 const script=[...source.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
 vm.runInContext(script,vm.createContext({document,Plotly:{react(id,traces,layout){plots[id]={traces,layout}}}}));
 return {elements,plots};
}
const {elements,plots}=load();
assert.equal(elements.persona.options.length,9);
let count=0;
for(const layer of [29,32,35,62])for(const scheme of ['layer-32','combined-29-32-35'])for(const strength of [0,.25])for(const persona of elements.persona.options.map(o=>o.value))for(const score of ['delta','absolute'])for(const policy of ['complete_only','including_proxies'])for(const kind of ['contextualized','non_contextualized']){
 for(const [id,value] of Object.entries({layer,scheme,strength,persona,score,policy,kind}))elements[id].value=String(value);
 elements.score.handlers.change();
 for(const [prefix,group,values] of [['context','word_set_type',['contextualized','non_contextualized']],['policy','policy',['complete_only','including_proxies']]])for(const [metric,base] of [['mrr','mean_reciprocal_rank'],['cosine','activation_word_cosine']]){
  const field=base+(score==='delta'?'_delta':''),plot=plots[`${prefix}-${metric}`];
  assert.equal(plot.traces.length,2);
  for(let i=0;i<2;i++){
   const expected=rows.filter(r=>r.layer===layer&&r.scheme===scheme&&r.beta===strength&&(persona==='all'||r.character===persona)&&r[group]===values[i]&&(prefix==='context'?r.policy===policy:r.word_set_type===kind));
   const trace=plot.traces[i];
   assert.equal(trace.type,'violin');assert.equal(trace.box.visible,true);
   assert.equal(trace.y.length,persona==='all'?96:12);
   assert.deepEqual(Array.from(trace.y),expected.map(r=>r[field]));
   assert.deepEqual(Array.from(trace.customdata,d=>Array.from(d)),expected.map(r=>[r.question,r.character,r.topic]));
  }
 }
 count++;
}
// Missing values must remove the same matched pair from both categories, not become zero.
const changed=structuredClone(rows),target=changed.find(r=>r.layer===32&&r.beta===0&&r.scheme==='layer-32'&&r.policy==='complete_only'&&r.word_set_type==='contextualized');
target.mean_reciprocal_rank_delta=null;
const injected=html.replace(/const DATA=[\s\S]*?;\nconst el=/,`const DATA=${JSON.stringify(changed)};\nconst el=`);
const missing=load(injected).plots;
assert.equal(missing['context-mrr'].traces[0].y.length,95);
assert.equal(missing['context-mrr'].traces[1].y.length,95);
assert.equal(missing['context-cosine'].traces[0].y.length,96);
assert(fs.readFileSync(`${folder}/index.html`,'utf8').includes('href="secondary-parameter-evaluation.html"'));
console.log(`Passed ${count} secondary-parameter selections, exact source scores, matched categories, missing values and overview link.`);
