const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('results/mini-batch-cluster-readout/question-aggregation.html','utf8');
const data=JSON.parse(fs.readFileSync('results/mini-batch-cluster-readout/question-aggregation.json','utf8'));
const elements={},plots={};
function element(){return {value:'',textContent:'',innerHTML:'',options:[],handlers:{},append(o){this.options.push(o);if(this.options.length===1)this.value=String(o.value)},addEventListener(event,handler){this.handlers[event]=handler}}}
const document={getElementById(id){return elements[id]??=element()},createElement(){return {value:'',textContent:''}}};
for(const [id,value] of Object.entries({metric:'mrr',beta:'0',scheme:'layer-32',layer:'32',kind:'contextualized',policy:'complete_only'}))document.getElementById(id).value=value;
const context=vm.createContext({document,Plotly:{react(id,traces){plots[id]=traces}}});
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];vm.runInContext(scripts.at(-1)[1],context);
assert.equal(elements.persona.options.length,8);let count=0;
for(const condition of data.conditions){
 for(const metric of ['mrr','cosine','mrr_margin','cosine_margin']){
  for(const [id,value] of Object.entries({metric,beta:condition.beta,scheme:condition.scheme,layer:condition.layer,kind:condition.word_set_type,policy:condition.policy,persona:condition.character}))elements[id].value=String(value);
  elements.metric.handlers.change();
  assert.equal((elements.summary.innerHTML.match(/<tr>/g)||[]).length,8);
  assert.equal((elements.questions.innerHTML.match(/<tr>/g)||[]).length,12);
  assert.equal(plots.deltas[0].x.length,12);assert.equal(plots.forest[0].x.length,8);
  const expected=condition.metrics[metric];
  assert(elements.cards.innerHTML.includes(String(expected.positive)+' / '+String(expected.n)));
  const i=plots.forest[0].y.indexOf(condition.character);assert.equal(plots.forest[0].x[i],expected.mean_delta);
  if(expected.ci95){assert.equal(plots.forest[0].error_x.array[i],expected.ci95[1]-expected.mean_delta);assert.equal(plots.forest[0].error_x.arrayminus[i],expected.mean_delta-expected.ci95[0]);}
  assert(!elements.cards.innerHTML.includes('NaN'));count++;
 }
}
assert.equal(count,2048);console.log(`Passed ${count} aggregation selections, tables, plots and uncertainty intervals.`);
