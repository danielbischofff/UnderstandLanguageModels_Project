const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const folder='results/mini-batch-cluster-readout';
const data=JSON.parse(fs.readFileSync(`${folder}/overall-aggregation.json`));
const html=fs.readFileSync(`${folder}/overall-aggregation.html`,'utf8');
const elements={};
const document={getElementById(id){return elements[id]??={value:'',innerHTML:'',textContent:'',options:[],handlers:{},attributes:{},setAttribute(key,value){this.attributes[key]=value},append(option){this.options.push(option);if(this.options.length===1)this.value=option.value},addEventListener(event,handler){this.handlers[event]=handler}}},createElement(){return {}}};
const script=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
vm.runInContext(script,vm.createContext({document}));
const fields=['layer','scheme','beta','word_set_type','policy'];
assert.equal((elements.rows.innerHTML.match(/<tr>/g)||[]).length,64);
let tested=0;
function check(i){
 if(i<fields.length){const field=fields[i];for(const option of elements[field].options){elements[field].value=option.value;check(i+1)}return}
 elements.layer.handlers.change();
 const expected=data.conditions.filter(r=>fields.every(f=>elements[f].value==='all'||String(r[f])===elements[f].value));
 assert.equal((elements.rows.innerHTML.match(/<tr>/g)||[]).length,expected.length);
 assert(!elements.rows.innerHTML.includes('<svg'));
 assert(elements.count.textContent.startsWith(`${expected.length} of 64`));
 for(const row of expected)for(const stats of Object.values(row.metrics)){
  assert(elements.rows.innerHTML.includes(`<b>${Number(stats.mean_delta).toPrecision(5)}</b>`));
  assert(elements.rows.innerHTML.includes(`${stats.positive_personas} / 8 positive personas`));
 }
 assert(!elements.rows.innerHTML.includes('NaN'));tested++;
}
check(0);assert.equal(tested,405);
for(const field of fields)elements[field].value='all';
for(const key of [...fields,'mrr','cosine','mrr_margin','cosine_margin'])for(let click=0;click<2;click++){
 elements['overall-sort-'+key].handlers.click();
 const direction=(['mrr','cosine','mrr_margin','cosine_margin'].includes(key)?-1:1)*(click===0?1:-1);
 const value=row=>row.metrics[key]?.mean_delta??row[key];
 const expected=[...data.conditions].sort((a,b)=>direction*(typeof value(a)==='number'?value(a)-value(b):String(value(a)).localeCompare(String(value(b)),undefined,{numeric:true})));
 const actual=[...elements.rows.innerHTML.matchAll(/<b>(.*?)<\/b>/g)].map(x=>x[1]);
 assert.deepEqual(actual,expected.flatMap(row=>['mrr','cosine','mrr_margin','cosine_margin'].map(m=>Number(row.metrics[m].mean_delta).toPrecision(5))));
 assert.equal(elements['overall-head-'+key].attributes['aria-sort'],direction===1?'ascending':'descending');
}
elements.layer.value='32';elements.layer.handlers.change();
assert.equal((elements.rows.innerHTML.match(/<tr>/g)||[]).length,16);
console.log(`Passed ${tested} filters, all nine columns in both sort directions, metric cells and removal of mini plots.`);
