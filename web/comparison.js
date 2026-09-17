'use strict';
const $=id=>document.getElementById(id);
const roles=['trained','jev','base'];
const names={trained:'训练后的 NanoJev',jev:'Jev API',base:'原始 Qwen'};
const kinds={trained:'已训练模型',jev:'实际 API',base:'未经项目微调'};
const directions={north:'↑ 北',east:'→ 东',south:'↓ 南',west:'← 西'};
const state={data:null,policy:'greedy',caseIndex:0,stepIndex:0,timer:null};
const make=(tag,cls,text)=>{const e=document.createElement(tag);e.className=cls??'';if(text!==undefined)e.textContent=text;return e;};
function selectedCase(){return state.data?.cases[state.caseIndex];}
function models(){return state.data?.models.filter(m=>m.policy===state.policy)??[];}
function episode(model){return model.episodes.find(e=>e.id===selectedCase().id);}
function duration(){return Math.max(0,...models().map(m=>episode(m).steps.length));}
function stop(){clearInterval(state.timer);state.timer=null;$('play').textContent='播放';}
function grid(environment,trace){
 const e=make('div','grid');e.style.setProperty('--size',environment.size);
 const walls=new Set(environment.walls.map(c=>c.join(','))),visited=new Set(trace.map(s=>s.state.position.join(',')));
 for(let r=0;r<environment.size;r++)for(let c=0;c<environment.size;c++){
  const key=[r,c].join(','),isAgent=environment.position.join(',')===key,isGoal=environment.goal.join(',')===key;
  const item=make('div','cell'+(walls.has(key)?' wall':'')+(visited.has(key)?' trail':'')+(isGoal?' goal':'')+(isAgent?' agent':''));
  item.append(make('span','coordinate',`${r},${c}`));if(isAgent||isGoal)item.append(make('span','mark',isAgent?'●':'◎'));e.append(item);
 }return e;
}
function outcome(ep){return ep.success===true||ep.outcome==='goal'?'到达目标':ep.outcome==='horizon_exhausted'?'达到步数上限':ep.outcome??'轨迹结束';}
function panel(model){
 const ep=episode(model),index=Math.min(state.stepIndex,ep.steps.length),transition=ep.steps[index],finished=index>=ep.steps.length;
 const environment=transition?.state??ep.final_state??ep.steps.at(-1)?.next_state??ep.initial_state;
 const p=make('article',`panel ${model.role}`);p.dataset.role=model.role;
 const heading=make('div','panel-heading'),title=make('div');title.append(make('h2','system-name',names[model.role]),make('p','system-detail',model.display_detail??model.name));heading.append(title,make('span','kind',kinds[model.role]));p.append(heading);
 const meta=make('div','step-meta');meta.append(make('span',`status${finished?ep.success?' success':' failure':''}`,finished?outcome(ep):'正在行动'),make('span','step-count',`已执行 ${index} 步`));p.append(meta,grid(environment,ep.steps.slice(0,index)));
 const action=make('div','action-line');action.append(make('span','',finished?'终局':'下一步实际动作'),make('strong','',finished?'—':directions[transition.action]??transition.action));p.append(action);
 const probabilities=make('div','probabilities');
 if(transition&&transition.probabilities&&Object.keys(transition.probabilities).length){
  const entries=Object.entries(transition.probabilities).sort((a,b)=>Object.keys(directions).indexOf(a[0])-Object.keys(directions).indexOf(b[0]));
  for(const [id,value] of entries){const item=make('div',`probability${id===transition.action?' selected':''}`);item.dataset.action=id;const line=make('div','row');line.append(make('span','',directions[id]??id),make('span','',`${(value*100).toFixed(1)}%`));const track=make('div','track'),fill=make('div','fill');fill.style.width=`${Math.max(0,Math.min(1,value))*100}%`;track.append(fill);item.append(line,track);probabilities.append(item);}
 }else probabilities.append(make('div','final-message',finished?'终局保持，不再生成动作。':'这一步没有记录概率，不补造分布。'));
 p.append(probabilities);
 let note=finished?'已结束的轨迹保持终局；其他系统继续按环境步播放。':transition.forced?'唯一合法动作，由环境直接执行。':state.policy==='sample'?'高亮实际抽中的动作；原分布不等于通关概率。':'高亮实际执行动作；原分布不等于通关概率。';
 if(transition?.probabilities){const total=Object.values(transition.probabilities).reduce((a,b)=>a+b,0);if(Math.abs(total-1)>1e-5)note+=` 原始和=${total.toFixed(3)}，条形未归一化。`;}
 p.append(make('p','prob-note',note));
 const provenance=make('p','summary-steps');provenance.append(make('span','',model.role==='jev'?'实际外部 API 轨迹':model.role==='base'?'冻结原始权重 · A–D 标签概率':'固定 v3_teacher_coords_multi_seed17'));p.append(provenance);
 return p;
}
function render(){
 if(!state.data)return;const c=selectedCase(),max=duration();state.stepIndex=Math.max(0,Math.min(state.stepIndex,max));
 $('caseBadge').textContent=`${c.split.toUpperCase()} ${c.split_index} / 2`;$('mapLabel').textContent=`${c.initial_state.size}×${c.initial_state.size} 网格 · 固定案例 ${state.caseIndex+1} / ${state.data.cases.length}`;
 $('controllerLabel').textContent=state.policy==='sample'?'按各自原始分布采样 · T=1':'各自原始分布的最大概率动作';$('globalStep').textContent=`环境步 ${state.stepIndex} / ${max}`;$('caseId').textContent=c.id;
 $('panels').replaceChildren(...roles.map(role=>panel(models().find(m=>m.role===role))));
 $('slider').max=String(max);$('slider').value=String(state.stepIndex);$('prev').disabled=state.stepIndex===0;$('next').disabled=state.stepIndex===max;
 window.nanojevComparison.snapshot={policy:state.policy,case_index:state.caseIndex,case_id:c.id,step_index:state.stepIndex,panels:models().map(m=>{const e=episode(m),i=Math.min(state.stepIndex,e.steps.length);return{role:m.role,step_index:i,finished:i===e.steps.length,environment:e.steps[i]?.state??e.final_state??e.steps.at(-1)?.next_state??e.initial_state,action:e.steps[i]?.action??null};})};
}
window.nanojevComparison={ready:false,setFrame(policy,caseIndex,stepIndex){stop();state.policy=policy;state.caseIndex=caseIndex;state.stepIndex=stepIndex;$('policySelect').value=policy;$('caseSelect').value=String(caseIndex);render();return this.snapshot;},duration(policy,caseIndex){const old=[state.policy,state.caseIndex];state.policy=policy;state.caseIndex=caseIndex;const n=duration();[state.policy,state.caseIndex]=old;return n;}};
$('policySelect').addEventListener('change',e=>{stop();state.policy=e.target.value;state.stepIndex=0;render();});$('caseSelect').addEventListener('change',e=>{stop();state.caseIndex=Number(e.target.value);state.stepIndex=0;render();});
$('prev').addEventListener('click',()=>{stop();state.stepIndex--;render();});$('next').addEventListener('click',()=>{stop();state.stepIndex++;render();});$('slider').addEventListener('input',e=>{stop();state.stepIndex=Number(e.target.value);render();});
$('play').addEventListener('click',()=>{if(state.timer)return stop();if(state.stepIndex===duration())state.stepIndex=0;$('play').textContent='暂停';render();state.timer=setInterval(()=>{state.stepIndex++;render();if(state.stepIndex===duration())stop();},500);});
(async()=>{try{const response=await fetch('./comparison_results.json',{cache:'no-store'});if(!response.ok)throw Error('尚无真实三方素材');const data=await response.json();if(!Array.isArray(data.cases)||!Array.isArray(data.models)||!data.cases.length)throw Error('素材schema无效');
 const policies=[...new Set(data.models.map(m=>m.policy))];for(const policy of policies){for(const role of roles){const matches=data.models.filter(m=>m.policy===policy&&m.role===role);if(matches.length!==1||data.cases.some(c=>!matches[0].episodes.some(e=>e.id===c.id)))throw Error('三方轨迹或固定地图缺失');}}
 state.data=data;state.policy=policies.includes('greedy')?'greedy':policies[0];$('policySelect').replaceChildren(...policies.map(p=>{const e=make('option','',p==='sample'?'概率采样 T=1':'最大概率 greedy');e.value=p;return e;}));$('caseSelect').replaceChildren(...data.cases.map((c,i)=>{const e=make('option','',`${c.split.toUpperCase()} ${c.split_index} · ${c.initial_state.size}×${c.initial_state.size}`);e.value=String(i);return e;}));for(const id of ['policySelect','caseSelect','play','slider'])$(id).disabled=false;render();window.nanojevComparison.ready=true;window.nanojevComparison.data=data;$('loadStatus').textContent='真实素材已载入；API/GPU耗时没有转换为播放速度。';
 }catch(error){$('loadStatus').textContent=error.message+'；不显示模拟轨迹。';$('loadStatus').classList.add('error');window.nanojevComparison.error=error.message;}})();
