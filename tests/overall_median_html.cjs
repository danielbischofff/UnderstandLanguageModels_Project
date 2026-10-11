const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const folder='results/mini-batch-cluster-readout';
const data=JSON.parse(fs.readFileSync(`${folder}/overall-median-comparison.json`));
const html=fs.readFileSync(`${folder}/overall-median-comparison.html`,'utf8'),nodes={};
const document={getElementById(id){return nodes[id]??={value:'',innerHTML:'',textContent:'',options:[],handlers:{},attributes:{},setAttribute(k,v){this.attributes[k]=v},append(o){this.options.push(o);if(this.options.length===1)this.value=o.value},addEventListener(ev,fn){this.handlers[ev]=fn}}},createElement(){return{}}};
document.getElementById('statistic').value='median_pairs';
vm.runInNewContext([...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1],{document});
const fields=['layer','scheme','beta','word_set_type','policy'],metrics=['mrr','cosine','mrr_margin','cosine_margin'];
assert.equal(nodes.layer.options.length,5);assert.equal(nodes.beta.options.length,3);
let count=0;
for(const estimator of ['median_pairs','median_persona_means','mean']){
 nodes.statistic.value=estimator;
 for(const condition of data.conditions){
  for(const key of fields)nodes[key].value=String(condition[key]);
  nodes.statistic.handlers.change();
  assert.equal((nodes.rows.innerHTML.match(/<tr>/g)||[]).length,1);
  for(const metric of metrics){
   const stats=condition.metrics[metric].estimators[estimator];
   assert(nodes.rows.innerHTML.includes(`<b>${Number(stats.estimate).toPrecision(5)}</b>`));
   assert(nodes.rows.innerHTML.includes(`95% interval [${Number(stats.ci95[0]).toPrecision(5)}, ${Number(stats.ci95[1]).toPrecision(5)}]`));
  }
  count++;
 }
 for(const key of fields)nodes[key].value='all';
 nodes['overall-sort-mrr'].handlers.click();
 let actual=[...nodes.rows.innerHTML.matchAll(/<b>(.*?)<\/b>/g)].filter((_,i)=>i%4===0).map(x=>Number(x[1]));
 // Toggle sort direction, then verify the order after changing estimator.
 assert(actual.every((v,i)=>i===0||v<=actual[i-1])||actual.every((v,i)=>i===0||v>=actual[i-1]));
}
assert.equal(count,192);assert(!html.includes('<svg'));console.log('Passed all 192 configuration/statistic selections and numeric sorting.');
