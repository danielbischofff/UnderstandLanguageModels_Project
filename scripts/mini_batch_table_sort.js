function createTableSorter(prefix,keys,render){
 let selected=null,direction=1;
 const buttons=keys.map(key=>{const button=document.getElementById(prefix+'sort-'+key),label=button.textContent;
  button.addEventListener('click',()=>{direction=selected===key?-direction:(['mrr','cosine','mrr_margin','cosine_margin'].includes(key)?-1:1);selected=key;update();render()});
  return {key,button,label};
 });
 function update(){for(const {key,button,label} of buttons){const active=selected===key;button.textContent=label+(active?(direction===1?' ↑':' ↓'):' ↕');document.getElementById(prefix+'head-'+key).setAttribute('aria-sort',active?(direction===1?'ascending':'descending'):'none')}}
 update();
 return rows=>{
  if(selected===null)return rows;
  const value=row=>row.metrics&&Object.hasOwn(row.metrics,selected)?row.metrics[selected].mean_delta:row[selected];
  return rows.map((row,index)=>({row,index})).sort((a,b)=>{
   const x=value(a.row),y=value(b.row);
   if(x==null||y==null)return x==null&&y==null?a.index-b.index:x==null?1:-1;
   const comparison=typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y),undefined,{numeric:true});
   return direction*comparison||a.index-b.index;
  }).map(x=>x.row);
 };
}
