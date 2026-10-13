const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const folder='results/mini-batch-cluster-readout';
const html=fs.readFileSync(`${folder}/question-comparison.html`,'utf8');
const source=JSON.parse(fs.readFileSync(`${folder}/question-aggregation.json`,'utf8'));
const elements={},plots={};
function element(){return {value:'',innerHTML:'',textContent:'',options:[],handlers:{},append(o){this.options.push(o);if(this.options.length===1)this.value=String(o.value)},addEventListener(event,handler){this.handlers[event]=handler}}}
const document={getElementById(id){return elements[id]??=element()},createElement(){return {value:'',textContent:''}}};
for(const [id,value] of Object.entries({beta:'0',scheme:'layer-32',layer:'32',kind:'contextualized',policy:'complete_only',order:'id'}))document.getElementById(id).value=value;
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
vm.runInContext(scripts.at(-1)[1],vm.createContext({document,Plotly:{react(id,traces,layout){plots[id]={traces,layout}}}}));
assert.equal(elements.persona.options.length,9);
const settings=[...new Map(source.conditions.map(c=>{const s={beta:c.beta,scheme:c.scheme,layer:c.layer,kind:c.word_set_type,policy:c.policy};return [JSON.stringify(s),s]})).values()];
let count=0;
for(const setting of settings)for(const persona of elements.persona.options.map(o=>o.value))for(const order of ['id','mrr','cosine']){
 for(const [id,value] of Object.entries({...setting,persona,order}))elements[id].value=String(value);
 elements.order.handlers.change();
 const rows=source.per_question.filter(r=>r.beta===setting.beta&&r.scheme===setting.scheme&&r.layer===setting.layer&&r.word_set_type===setting.kind&&r.policy===setting.policy&&(persona==='all'||r.character===persona));
 assert.equal(rows.length,persona==='all'?96:12);
 assert.equal((elements.rows.innerHTML.match(/<tr>/g)||[]).length,rows.length);
 assert.equal((elements['question-list'].innerHTML.match(/class="question"/g)||[]).length,12);
 const expectedOrder=source.questions.map(q=>q.id);
 if(order!=='id'){
  const field=order==='mrr'?'mean_reciprocal_rank_delta':'activation_word_cosine_delta';
  const mean=q=>{const ys=rows.filter(r=>r.question===q&&r[field]!=null).map(r=>r[field]);return ys.length?ys.reduce((a,b)=>a+b,0)/ys.length:-Infinity};
  expectedOrder.sort((a,b)=>mean(b)-mean(a)||a.localeCompare(b));
 }
 for(const [metric,field] of [['mrr','mean_reciprocal_rank_delta'],['cosine','activation_word_cosine_delta']]){
  const plot=plots[metric];assert.equal(plot.traces.length,persona==='all'?8:1);
  assert.deepEqual(Array.from(plot.layout.xaxis.categoryarray),expectedOrder);
  assert.equal(plot.layout.shapes.length,persona==='all'?1:2);
  for(const trace of plot.traces){
   assert.equal(trace.x.length,12);
   for(let i=0;i<trace.x.length;i++){
    const row=rows.find(r=>r.character===trace.name&&r.question===trace.x[i]);
    assert.equal(trace.y[i],row[field]);
    assert.equal(trace.customdata[i][1],source.questions.find(q=>q.id===row.question).text);
    if(persona!=='all')assert.equal(trace.marker.color[i],row[field]>0?'rgba(22,163,74,.6)':row[field]<0?'rgba(177,60,54,.6)':'#94a3b8');
   }
  }
  if(persona!=='all')assert.equal(plot.layout.shapes[1].y0,rows.reduce((sum,r)=>sum+r[field],0)/rows.length);
 }
 assert(!elements.rows.innerHTML.includes('NaN'));count++;
}
assert.equal(count,1728);
const index=fs.readFileSync(`${folder}/index.html`,'utf8');
for(const match of index.matchAll(/href="([^"]+)"/g)){
 const href=match[1];if(href.startsWith('#')||/^https?:/.test(href))continue;
 assert(fs.existsSync(path.resolve(folder,href.split('#')[0])),`Missing overview link: ${href}`);
}
console.log(`Passed ${count} question comparisons, paired scores, shared ordering, means, colors and overview links.`);
