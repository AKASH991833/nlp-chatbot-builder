/* BotForge - visual intent builder + in-browser Naive Bayes classifier. */
(()=>{
const $=s=>document.querySelector(s);
const DEFAULT=[
 {name:'greeting',open:true,examples:['hello','hi there','hey','good morning','what\'s up','hi how are you','hello bot','hey there','good evening','hello hello','hi bot'],responses:['Hello! How can I help you today?','Hi there! What can I do for you?']},
 {name:'hours',examples:['what time do you open','when are you open','opening hours','are you open on sunday','what are your hours','when do you close','are you open now','what time do you close','opening times'],responses:['We are open Monday to Saturday, 9 AM to 8 PM.']},
 {name:'pricing',examples:['how much does it cost','what is the price','pricing plans','is there a free tier','how much is the subscription','cost of premium','what does premium cost','how much is it','price of the plan'],responses:['Plans start free. Pro is $12/month with unlimited usage.']},
 {name:'refund',examples:['i want a refund','how do i get my money back','refund policy','cancel and refund','return my payment','can i get a refund','money back please'],responses:['Refunds are available within 30 days, no questions asked.']},
 {name:'goodbye',examples:['bye','see you later','goodbye','talk later','thanks bye','have a nice day','bye bye','see you soon','goodbye for now'],responses:['Goodbye! Come back anytime.','See you soon!']},
];
let intents=JSON.parse(localStorage.getItem('botforge-intents')||'null')||DEFAULT;
let model=null,holdAcc=null;
const save=()=>localStorage.setItem('botforge-intents',JSON.stringify(intents));
const STOP=new Set('a an the is are was were i you he she it we they do does did to of in on for with and or but my your me what when where how who why can could would should'.split(' '));
const words=s=>s.toLowerCase().replace(/[^a-z0-9'\s]/g,' ').split(/\s+/).filter(t=>t&&!STOP.has(t));
const toks=s=>{const w=words(s),f=[...w];for(let i=0;i<w.length-1;i++)f.push(w[i]+'_'+w[i+1]);return f;};
// ---- Naive Bayes ----
function train(){
  const docs=[];
  intents.forEach(it=>it.examples.forEach(e=>docs.push({intent:it.name,t:toks(e)})));
  if(docs.length<2){model=null;return;}
  // holdout: last example of each intent with >=3 examples
  const hold=[],trainD=[];
  intents.forEach(it=>{const ex=it.examples.map(e=>({intent:it.name,t:toks(e)}));
    if(ex.length>=3){hold.push(ex[ex.length-1]);trainD.push(...ex.slice(0,-1));}else trainD.push(...ex);});
  const vocab=new Set();trainD.forEach(d=>d.t.forEach(w=>vocab.add(w)));
  const V=vocab.size,priors={},counts={},totals={};
  intents.forEach(it=>{counts[it.name]={};totals[it.name]=0;});
  trainD.forEach(d=>{priors[d.intent]=(priors[d.intent]||0)+1;d.t.forEach(w=>{counts[d.intent][w]=(counts[d.intent][w]||0)+1;totals[d.intent]++;});});
  Object.keys(priors).forEach(k=>priors[k]/=trainD.length);
  model={vocab,V,priors,counts,totals,names:intents.map(i=>i.name)};
  // holdout accuracy
  if(hold.length>=2){
    let ok=0;hold.forEach(h=>{if(classifyToks(h.t).name===h.intent)ok++;});
    holdAcc=ok/hold.length;
  } else holdAcc=null;
}
function classifyToks(t){
  if(!model)return{name:'(untrained)',conf:0};
  let best=null,bestScore=-1e18,scores={};
  model.names.forEach(n=>{
    let s=Math.log(model.priors[n]||1e-9);
    const denom=model.totals[n]+model.V;
    t.forEach(w=>{s+=Math.log(((model.counts[n][w]||0)+1)/denom);});
    scores[n]=s;
    if(s>bestScore){bestScore=s;best=n;}
  });
  const mx=Math.max(...Object.values(scores));
  const ex=Object.fromEntries(Object.entries(scores).map(([k,v])=>[k,Math.exp(v-mx)]));
  const sum=Object.values(ex).reduce((a,b)=>a+b,0);
  return{name:best,conf:ex[best]/sum};
}
const classify=s=>classifyToks(toks(s));
// ---- intents UI ----
function render(){
  const el=$('#intents');el.innerHTML='';
  intents.forEach((it,i)=>{
    const d=document.createElement('div');d.className='intent'+(it.open?' open':'');
    d.innerHTML=`<div class="intent-head"><b>#${it.name}</b>
      <span class="count">${it.examples.length} examples · ${it.responses.length} responses</span>
      <button class="btn mini ghost" data-a="rename">Rename</button>
      <button class="btn mini ghost danger" data-a="del">Delete</button></div>
    <div class="intent-body">
      <div style="font-size:12px;font-weight:800;letter-spacing:.04em;color:var(--muted)">EXAMPLE PHRASES</div>
      <div class="taglist" data-k="examples"></div>
      <div class="addrow"><input class="input" placeholder="Add example phrase..." data-k="examples"><button class="btn mini ghost">Add</button></div>
      <div style="font-size:12px;font-weight:800;letter-spacing:.04em;color:var(--muted);margin-top:12px">BOT RESPONSES</div>
      <div class="taglist" data-k="responses"></div>
      <div class="addrow"><input class="input" placeholder="Add bot response..." data-k="responses"><button class="btn mini ghost">Add</button></div>
    </div>`;
    d.querySelector('.intent-head').onclick=e=>{
      const a=e.target.dataset.a;
      if(a==='rename'){const n=prompt('Intent name:',it.name);if(n){it.name=n.toLowerCase().replace(/\s+/g,'-');save();render();stats();}return;}
      if(a==='del'){if(confirm('Delete intent #'+it.name+'?')){intents.splice(i,1);save();render();stats();}return;}
      it.open=!it.open;d.classList.toggle('open');
    };
    ['examples','responses'].forEach(k=>{
      const tl=d.querySelector(`.taglist[data-k="${k}"]`);
      it[k].forEach((x,j)=>{const t=document.createElement('span');t.className='tag';t.innerHTML=x+' <i title="Remove">×</i>';
        t.querySelector('i').onclick=()=>{it[k].splice(j,1);save();render();stats();};tl.append(t);});
      const row=d.querySelector(`.addrow:has(.input[data-k="${k}"])`);
      const inp=row.querySelector('.input');
      const add=()=>{const v=inp.value.trim();if(!v)return;it[k].push(v);save();render();stats();};
      row.querySelector('.btn').onclick=add;
      inp.onkeydown=e=>{if(e.key==='Enter')add();};
    });
    el.append(d);
  });
}
function stats(){
  $('#m-intents').textContent=intents.length;
  $('#m-examples').textContent=intents.reduce((a,i)=>a+i.examples.length,0);
  $('#m-vocab').textContent=model?model.V:0;
  $('#m-acc').textContent=holdAcc==null?'–':Math.round(holdAcc*100)+'%';
  $('#confMatrix').textContent=holdAcc==null?'Add 3+ examples per intent to see holdout accuracy.':'Holdout check: last example of each intent (3+ examples) predicted before training on it.';
}
// ---- chat ----
function bubble(text,who,conf){
  const c=$('#chat');const m=document.createElement('div');m.className='msg '+who;
  m.innerHTML=text+(who==='bot'&&conf!=null?`<span class="conf">confidence ${(conf*100).toFixed(0)}%</span>`:'');
  c.append(m);c.scrollTop=c.scrollHeight;
}
function reply(q){
  const r=classify(q);
  const it=intents.find(i=>i.name===r.name);
  if(!it||r.conf<0.35){bubble("I'm not sure about that one. Train me with more examples!",'bot',r.conf);return;}
  bubble(it.responses[Math.floor(Math.random()*it.responses.length)],'bot',r.conf);
}
function send(){const v=$('#chatIn').value.trim();if(!v)return;$('#chatIn').value='';bubble(v.replace(/</g,'&lt;'),'user');
  setTimeout(()=>reply(v),250);}
$('#send').onclick=send;
$('#chatIn').onkeydown=e=>{if(e.key==='Enter')send();};
// ---- actions ----
$('#addIntent').onclick=()=>{const n=prompt('New intent name:');if(!n)return;
  intents.push({name:n.toLowerCase().replace(/\s+/g,'-'),open:true,examples:[],responses:[]});save();render();stats();};
$('#train').onclick=()=>{train();stats();toast('Model trained');bubble('Training complete. Ask me anything!','bot');};
$('#exportJson').onclick=()=>{
  const blob=new Blob([JSON.stringify({intents},null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='botforge-bot.json';a.click();toast('Exported botforge-bot.json');};
$('#importFile').onchange=e=>{
  const f=e.target.files[0];if(!f)return;
  const rd=new FileReader();rd.onload=()=>{try{const j=JSON.parse(rd.result);
    if(!Array.isArray(j.intents))throw 0;intents=j.intents;save();train();render();stats();toast('Imported and trained');}catch(_){toast('Invalid bot JSON');}};
  rd.readAsText(f);e.target.value='';};
$('#exportHtml').onclick=()=>{
  train();
  const data=JSON.stringify({intents,model});
  const html=`<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>My BotForge Bot</title>
<style>body{font-family:system-ui;max-width:520px;margin:40px auto;padding:0 16px;background:#f7f8fa;color:#0f2145}h1{font-size:22px}#c{height:420px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;background:#fff;border:1px solid #e3e8ef;border-radius:14px;padding:14px}.m{max-width:85%;padding:9px 13px;border-radius:14px;font-size:14px}.u{align-self:flex-end;background:#7c3aed;color:#fff}.b{align-self:flex-start;background:#eef1f5}#r{display:flex;gap:8px;margin-top:10px}input{flex:1;padding:11px 13px;border-radius:10px;border:1px solid #e3e8ef;font-size:14px}button{padding:11px 18px;border-radius:10px;border:none;background:#7c3aed;color:#fff;font-weight:700;cursor:pointer}</style></head>
<body><h1>My BotForge Bot</h1><div id=c></div><div id=r><input id=i placeholder="Type a message..." autofocus><button onclick=go()>Send</button></div>
<script>const D=${data.replace(/<\//g,'<\\/')};
const STOP=new Set('a an the is are was were i you he she it we they do does did to of in on for with and or but my your me what when where how who why can could would should'.split(' '));
const toks=s=>s.toLowerCase().replace(/[^a-z0-9'\\s]/g,' ').split(/\\s+/).filter(t=>t&&!STOP.has(t));
function cls(q){const t=toks(q),m=D.model;let best=null,bs=-1e18,sc={};m.names.forEach(n=>{let s=Math.log(m.priors[n]||1e-9);const dn=m.totals[n]+m.V;t.forEach(w=>{s+=Math.log(((m.counts[n][w]||0)+1)/dn)});sc[n]=s;if(s>bs){bs=s;best=n}});
const mx=Math.max(...Object.values(sc));const ex=Object.fromEntries(Object.entries(sc).map(([k,v])=>[k,Math.exp(v-mx)]));const sm=Object.values(ex).reduce((a,b)=>a+b,0);return{name:best,conf:ex[best]/sm}}
function add(t,w){const d=document.createElement('div');d.className='m '+w;d.textContent=t;c.append(d);c.scrollTop=c.scrollHeight}
function go(){const v=i.value.trim();if(!v)return;i.value='';add(v,'u');setTimeout(()=>{const r=cls(v);const it=D.intents.find(x=>x.name===r.name);
add(!it||r.conf<.35?"I'm not sure about that one.":it.responses[Math.floor(Math.random()*it.responses.length)],'b')},200)}
i.onkeydown=e=>{if(e.key==='Enter')go()};add('Hi! Ask me anything.','b')<\/script></body></html>`;
  const blob=new Blob([html],{type:'text/html'});const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='my-bot.html';a.click();toast('Exported my-bot.html');};
train();render();stats();
bubble('Hi! I\'m the demo bot. Try "what time do you open" or "i want a refund" - then edit my intents and retrain.','bot');
})();
