// Exercise the combined dashboard and token page against their saved source data.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
function harness(defaults){
 const nodes={},plots={};
 const document={getElementById(id){return nodes[id]??={value:'',options:[],handlers:{},innerHTML:'',textContent:'',append(o){this.options.push(o);if(this.options.length===1)this.value=String(o.value)},addEventListener(e,f){this.handlers[e]=f},on(e,f){this.handlers[e]=f}}},createElement(){return{}},querySelectorAll(){return[]},body:{classList:{toggle(){}}}};
 for(const [id,value]of Object.entries(defaults)){const node=document.getElementById(id);node.value=value;node.options=[{value}]}
 const context=vm.createContext({document,URLSearchParams,location:{search:''},window:{},Plotly:{react(id,traces,layout){plots[id]={traces,layout}},Plots:{resize(){}}}});
 return {nodes,plots,context};
}
function execute(path,h){const html=fs.readFileSync(path,'utf8');const code=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];vm.runInContext(code,h.context);return html}
const defaults={persona:'all',layer:'32',scheme:'layer-32',kind:'contextualized',policy:'complete_only',beta:'0.25',alpha:'1.5','plot-metric':'mrr','table-layer':'all','table-scheme':'all','table-kind':'all','table-policy':'all'};
const h=harness(defaults),html=execute('results/index.html',h),data=JSON.parse(fs.readFileSync('results/steering-comparison/comparison-data.json'));
assert(!html.toLowerCase().includes('specificity'));assert(!JSON.stringify(data).includes('mrr_margin'));
assert.equal((h.nodes.comparison.innerHTML.match(/class="pair-bars"/g)||[]).length,128);
assert.equal((h.nodes.comparison.innerHTML.match(/class="pair-bar"/g)||[]).length,128*96);
for(const method of ['band','alpha'])for(const row of vm.runInContext('tableConditions("'+method+'")',h.context))for(const metric of ['mrr','cosine']){
 h.context.rowToCheck={...row,metric};
 const stats=vm.runInContext('sampleStats(pairSamples("'+method+'",rowToCheck))',h.context);
 assert(Math.abs(stats.mean-row.metrics[metric].mean_delta)<1e-12);assert(stats.sd>=0);
}
let checks=0;
for(const beta of ['0','0.25'])for(const alpha of ['1.5','4'])for(const layer of ['29','32','35','62'])for(const scheme of ['layer-32','combined-29-32-35'])for(const kind of ['contextualized','non_contextualized'])for(const policy of ['complete_only','including_proxies'])for(const persona of ['all',...data.personas])for(const metric of ['mrr','cosine']){
 for(const [id,value]of Object.entries({beta,alpha,layer,scheme,kind,policy,persona,'plot-metric':metric}))h.nodes[id].value=value;
 h.nodes.persona.handlers.change();
 const rows=(h.nodes.comparison.innerHTML.match(/<tr(?: |>)/g)||[]).length;
 assert.equal(rows,64);
 assert.equal((h.nodes.comparison.innerHTML.match(/class="pair-bar"/g)||[]).length,128*(persona==='all'?96:12));
 for(const method of ['band','alpha']){
  const selected=data[method].conditions.filter(r=>String(r.layer)===layer&&r.scheme===scheme&&r.word_set_type===kind&&r.policy===policy&&String(r[method==='band'?'beta':'alpha'])===(method==='band'?beta:alpha)&&(persona==='all'||r.character===persona));
  const actual=h.plots[method+'-forest'].traces[0];
  assert.deepEqual(Array.from(actual.x),selected.map(r=>r.metrics[metric].mean_delta));
  assert.deepEqual(Array.from(actual.error_x.array),selected.map(r=>r.metrics[metric].ci95[1]-r.metrics[metric].mean_delta));
  assert.equal(h.plots[method+'-questions'].traces[0].x.length,12);
 }
 assert.deepEqual(Array.from(h.plots['band-forest'].layout.xaxis.range),Array.from(h.plots['alpha-forest'].layout.xaxis.range));checks++;
}
Object.entries(defaults).forEach(([id,value])=>h.nodes[id].value=value);
for(const key of ['band','alpha'])for(let click=0;click<2;click++){
 h.nodes['sort-'+key].handlers.click();
 const cells=[...h.nodes.comparison.innerHTML.matchAll(/<b>(.*?)<\/b>/g)].map(m=>Number(m[1]));
 const values=cells.filter((v,i)=>i%2===(key==='band'?0:1));
 assert(values.every((v,i)=>!i||(click===0?values[i-1]>=v:values[i-1]<=v)));
}
for(const [id,value]of Object.entries({'table-layer':'32','table-scheme':'layer-32','table-kind':'contextualized','table-policy':'complete_only'}))h.nodes[id].value=value;
h.nodes['table-layer'].handlers.change();assert.equal((h.nodes.comparison.innerHTML.match(/<tr>/g)||[]).length,2);
const t=harness({...defaults,persona:'activist'});
for(const method of ['band','alpha'])vm.runInContext(fs.readFileSync('results/steering-comparison/token-'+method+'-data.js','utf8'),t.context);
execute('results/steering-comparison/token-comparison.html',t);
assert.equal(t.nodes.question.options.length,13);
let tokens=0;
for(const persona of data.personas)for(const layer of ['29','32','35','62'])for(const question of ['Q01','Q23','all'])for(const [beta,alpha]of [['0','1.5'],['0.25','4']]){
 for(const [id,value]of Object.entries({persona,layer,question,beta,alpha}))t.nodes[id].value=value;
 t.nodes.question.handlers.change();
 for(const method of ['band','alpha'])for(const [metric,field,delta]of [['mrr','mean_reciprocal_rank','mean_reciprocal_rank_delta'],['cosine','activation_word_cosine','activation_word_cosine_delta']]){
  const traces=t.plots[method+'-'+metric].traces,base=traces[1],steered=traces[2];
  assert.equal(traces[0].fillcolor,'rgba(22,163,74,0.16)');
  assert.deepEqual(Array.from(t.plots['band-'+metric].layout.yaxis.range),Array.from(t.plots['alpha-'+metric].layout.yaxis.range));
  if(question!=='all'){
   const r=data[method].rows.find(r=>r.character===persona&&String(r.layer)===layer&&r.question===question&&r.scheme==='layer-32'&&r.word_set_type==='contextualized'&&r.policy==='complete_only'&&String(r[method==='band'?'beta':'alpha'])===(method==='band'?beta:alpha));
   const mean=values=>{const valid=values.filter(v=>v!==null&&Number.isFinite(v));return valid.length?valid.reduce((a,b)=>a+b,0)/valid.length:null};
   const actual=mean(steered.y),control=mean(base.y);
   if(r[field]===null)assert.equal(actual,null);
   else assert(Math.abs(actual-r[field])<(method==='band'&&metric==='mrr'?2e-7:1e-8),`${method} ${question} ${persona} ${layer} ${metric}: ${actual} != ${r[field]}`);
   if(r[delta]!==null)assert(Math.abs(control-(r[field]-r[delta]))<(method==='band'&&metric==='mrr'?2e-7:1e-8));
  }else assert(base.customdata.every(n=>n>=1&&n<=12));
  assert(t.nodes[method+'-control-answer'].textContent.length>0);assert(t.nodes[method+'-steered-answer'].textContent.length>0);
  assert(t.nodes[method+'-word-details'].innerHTML.includes('<tr>'));
 }
 tokens++;
}
console.log(`Passed ${checks} dashboard selections, sorting, and ${tokens} combined token selections with both independent baselines.`);
