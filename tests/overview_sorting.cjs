const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const folder='results/mini-batch-cluster-readout',nodes={};
const document={getElementById(id){return nodes[id]??={value:'',innerHTML:'',textContent:'',options:[],listeners:{},attributes:{},setAttribute(key,value){this.attributes[key]=value},append(o){this.options.push(o);if(this.options.length===1)this.value=String(o.value)},addEventListener(ev,fn){this.listeners[ev]=fn}}},createElement(){return{}},querySelector(){return document.getElementById('table-link')}};
document.getElementById('preview-layer').value='32';document.getElementById('preview-kind').value='contextualized';
const html=fs.readFileSync(`${folder}/index.html`,'utf8');
vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],{document});
const source=JSON.parse(fs.readFileSync(`${folder}/overall-aggregation.json`)).conditions;
const metrics=['mrr','cosine','mrr_margin','cosine_margin'];
let count=0;
for(const key of ['scheme','beta',...metrics]){
 for(const layer of ['29','32','35','62'])for(const kind of ['contextualized','non_contextualized']){
  nodes['preview-layer'].value=layer;nodes['preview-kind'].value=kind;
  for(let click=0;click<2;click++){
   nodes['preview-sort-'+key].listeners.click();
   const direction=(metrics.includes(key)?-1:1)*(click===0?1:-1);
   const value=r=>r.metrics[key]?.mean_delta??r[key];
   const expected=source.filter(r=>String(r.layer)===layer&&r.word_set_type===kind&&r.policy==='complete_only').sort((a,b)=>direction*(typeof value(a)==='number'?value(a)-value(b):String(value(a)).localeCompare(String(value(b)),undefined,{numeric:true})));
   const preview=nodes['overall-preview'].innerHTML;
   assert.equal((preview.match(/<tr>/g)||[]).length,4);assert(!preview.includes('<svg'));
   assert.deepEqual([...preview.matchAll(/<b class="\w+">(.*?)<\/b>/g)].map(x=>x[1]),expected.flatMap(r=>metrics.map(m=>Number(r.metrics[m].mean_delta).toPrecision(5))));
   count++;
  }
 }
}
nodes['table-link'].listeners.click();assert(nodes['summary-tables'].open);
console.log(`Passed ${count} overview sorting checks across all eight previews; no mini plots remain.`);
