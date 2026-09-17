'use strict';

// 无生成结果、无内嵌成功轨迹；所有模型输出均来自结果文件或实时服务。
const $ = id => document.getElementById(id);
const ui = { data: null, models: [], modelIndex: 0, episodeIndex: 0, stepIndex: 0, batchIndex: 0, timer: null, batches: [] };
const own = (obj, key) => obj != null && Object.hasOwn(obj, key);
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const json = value => JSON.stringify(value, null, 2);
const number = value => typeof value === 'number' && Number.isFinite(value);
const labels = {
  north: '↑ 北', south: '↓ 南', east: '→ 东', west: '← 西', stop: '停止', STOP: '停止',
  true: '是', false: '否', success_rate: '成功率', completion_rate: '到达率', win_rate: '胜率', loss_rate: '负率',
  draw_rate: '和局率', accuracy: '准确率', optimal_action_accuracy: '最优动作命中率',
  episodes: '对局数', episode_count: '对局数', mean_steps: '平均步数',
  mean_regret: '平均路径额外步数', mean_latency_ms: '平均推理延迟 · ms',
  p50_latency_ms: '推理延迟 p50 · ms', p95_latency_ms: '推理延迟 p95 · ms',
  optimal_action_rate: '最优动作率', preserved_nonlosing_initial_value_rate: '非必败初局保值率',
  preserved_minimax_value_rate: '初局 minimax 保值率', mean_path_efficiency: '平均路径效率',
  mean_p_optimal: '最优动作上的概率质量', actual_optimal_action_rate: '实际最优动作率',
  goal: '到达目标', horizon_exhausted: '达到步数上限', win: '胜', draw: '和', loss: '负',
  student: 'NanoJev', random: '随机策略', oracle: '求解器上界', minimax_opponent: '最优对手', forced_legal_action: '唯一合法动作',
  greedy: 'Greedy · 最大概率', sample: 'Sample · 按原分布采样',
  grid_navigation: '网格寻路', tic_tac_toe: '井字棋', test: 'TEST', ood: 'OOD', dev: 'DEV', train: 'TRAIN',
};
const defaultSources = [
  { title: 'TypeSafe · 官方 Doom 与 Wikiracing 演示', url: 'https://typesafe.ai/blog/introducing-system-one-models-and-jev' },
  { title: 'Diogo Almeida · Doom 发布帖', url: 'https://x.com/CompleteSkeptic/status/2099925687465570372' },
  { title: 'vinnylarouge / jevlike · 独立开源实验', url: 'https://github.com/vinnylarouge/jevlike' },
];

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = String(text);
  return item;
}
function option(value, label) { const item = node('option', '', label); item.value = String(value); return item; }
function display(value) {
  if (value === null || value === undefined) return '未记录';
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (number(value)) return Number(value.toPrecision(5)).toString();
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}
function actionName(action) {
  const value = record(action) ? action.id ?? action.action ?? action.choice ?? JSON.stringify(action) : action;
  if (typeof value === 'string' && /^cell_[1-9]$/.test(value)) return `第 ${value.slice(5)} 格`;
  return labels[value] ?? display(value);
}
function selectedModel() { return ui.models[ui.modelIndex] ?? null; }
function modelVersion(model) {
  if (String(model?.schema_version ?? '').startsWith('openjev-navigation-v3-')) return 'V3';
  if (record(model?.summary) && Object.keys(model.summary).some(key => key.includes('/'))) return 'V2';
  return '未标版本';
}
function modelLabel(model, index) {
  const controller = model.controller ?? model.policy;
  return `${modelVersion(model)} · ${model.name ?? model.id ?? `模型 ${index + 1}`} · ${labels[controller] ?? controller ?? '未标策略'}`;
}
function modelIdentity(model) { return model ? `${model.name ?? model.id}|${model.policy}|${model.checkpoint_sha256}` : null; }
function episodes() { return Array.isArray(selectedModel()?.episodes) ? selectedModel().episodes : []; }
function selectedEpisode() { return episodes()[ui.episodeIndex] ?? null; }
function steps() { return Array.isArray(selectedEpisode()?.steps) ? selectedEpisode().steps : []; }
function currentStep() { return steps()[ui.stepIndex] ?? null; }
function stopPlayback() { if (ui.timer !== null) clearInterval(ui.timer); ui.timer = null; $('playButton').textContent = '播放回放'; }
function notice(text) { $('notice').textContent = text; $('notice').hidden = !text; }
function empty(container, title, description) {
  const block = node('div', 'empty-state');
  block.append(node('span', 'empty-icon', '∷'), node('h3', '', title), node('p', '', description));
  container.replaceChildren(block);
}

function renderSources() {
  const supplied = Array.isArray(ui.data?.sources) ? ui.data.sources : [];
  const unique = new Map();
  for (const source of [...defaultSources, ...supplied]) {
    const value = typeof source === 'string' ? { url: source, title: source } : source;
    if (!record(value)) continue;
    try {
      const url = new URL(value.url ?? value.href);
      if (!['https:', 'http:'].includes(url.protocol)) continue;
      unique.set(url.href, { url: url.href, title: value.title ?? value.name ?? value.label ?? url.hostname });
    } catch { /* 无效链接不变成可执行 HTML。 */ }
  }
  $('sourceLinks').replaceChildren(...[...unique.values()].map(source => {
    const link = node('a', 'source-link');
    link.href = source.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
    link.append(node('span', '', source.title), node('span', '', '↗'));
    return link;
  }));
}

function summaryEntries(summary, prefix = '', depth = 0) {
  if (!record(summary)) return [];
  const values = [];
  for (const [key, value] of Object.entries(summary)) {
    const title = `${prefix}${labels[key] ?? key}`;
    if (record(value) && depth < 1) values.push(...summaryEntries(value, `${title} · `, depth + 1));
    else if (['number', 'string', 'boolean'].includes(typeof value)) values.push([title, value, key]);
  }
  return values;
}
function renderSummary() {
  const model = selectedModel();
  const navigationGroups = record(model?.summary) ? ['test', 'ood'].filter(key => record(model.summary[key]) && own(model.summary[key], 'completion_rate')) : [];
  const grouped = record(model?.summary) ? Object.entries(model.summary).filter(([key, value]) => key.includes('/') && record(value)) : [];
  const entries = navigationGroups.length ? navigationGroups.flatMap(split => {
    const values = model.summary[split], prefix = `${labels[split]} · ${display(values.episodes)} 局`;
    return ['completion_rate', 'mean_path_efficiency', 'mean_steps', 'mean_p_optimal']
      .map(metric => [`${prefix} · ${labels[metric]}`, values[metric], metric]);
  }) : grouped.length ? grouped.flatMap(([group, values]) => {
    const name = group.split('/').map(part => labels[part] ?? part).join(' · ');
    const metric = own(values, 'success_rate') ? 'success_rate' : own(values, 'preserved_nonlosing_initial_value_rate') ? 'preserved_nonlosing_initial_value_rate' : 'optimal_action_rate';
    return [[`${name} · ${labels[metric] ?? metric}`, values[metric], metric], [`${name} · 已记录对局`, values.episodes, 'episode_count']];
  }).slice(0, 8) : summaryEntries(model?.summary).slice(0, 8);
  if (model && !entries.length) entries.push(['已记录对局', episodes().length, 'episode_count']);
  $('summaryCards').replaceChildren(...entries.map(([title, value, key]) => {
    const card = node('div', 'summary-card');
    const formatted = number(value) && (/(?:rate|accuracy)$/.test(key) || ['mean_path_efficiency', 'mean_p_optimal'].includes(key)) && value >= 0 && value <= 1
      ? `${(value * 100).toFixed(1)}%` : display(value);
    card.append(node('span', 'metric-label', title), node('span', `metric-value${number(value) ? '' : ' text-value'}`, formatted));
    return card;
  }));
  $('checkpointHash').textContent = model?.checkpoint_sha256 ? `${model.checkpoint_sha256.slice(0, 16)}…` : model ? '未提供 / 基线' : '尚未载入';
  $('checkpointHash').title = model?.checkpoint_sha256 ?? '';
  const cohortHash = model?.cohort?.initial_states_sha256;
  const controller = model?.controller ?? model?.policy;
  $('cohortScope').textContent = !model ? '载入后显示当前评测集合与执行策略。'
    : `${modelVersion(model)} · ${labels[controller] ?? controller ?? '未标策略'} · ${episodes().length} 局完整回放${typeof cohortHash === 'string' ? ` · 评测集合 ${cohortHash.slice(0, 12)}` : ''}。V2 与 V3 地图集合不同，数字应在同一评测集合内比较。`;
  $('cohortScope').title = typeof cohortHash === 'string' ? cohortHash : '';
  $('rawSummary').textContent = model ? json({ name: model.name, policy: model.policy, checkpoint_sha256: model.checkpoint_sha256,
    summary: model.summary, cohort: model.cohort, execution: model.execution }) : '尚未载入';
}

function environmentState(value, depth = 0) {
  if (depth > 4) return value;
  if (typeof value === 'string') { try { return environmentState(JSON.parse(value), depth + 1); } catch { return value; } }
  if (!record(value)) return value;
  if (value.game && (value.board !== undefined || value.position !== undefined)) return value;
  const nested = value.environment_state ?? value.metadata?.environment_state ?? value.state;
  return nested === undefined ? value : environmentState(nested, depth + 1);
}
function visibleState() {
  const episode = selectedEpisode(), list = steps(), step = currentStep();
  if (step) return environmentState(step.state ?? (ui.stepIndex === 0 ? episode.initial_state : list[ui.stepIndex - 1]?.next_state));
  return environmentState(list.at(-1)?.next_state ?? episode?.final_state ?? episode?.initial_state);
}
function legend(items) {
  $('boardLegend').replaceChildren(...items.map(([className, label]) => {
    const item = node('span', 'legend-item'); item.append(node('span', `legend-swatch ${className}`), node('span', '', label)); return item;
  }));
  $('boardLegend').hidden = false;
}
function renderBoard(state) {
  const board = $('gameBoard');
  board.replaceChildren(); board.className = 'game-board'; $('boardLegend').hidden = true;
  if (!state) { board.classList.add('empty-board'); empty(board, '等待真实实验结果', '没有预设成功轨迹。结果文件就绪后可刷新载入。'); return; }
  const game = state?.game ?? selectedEpisode()?.game;
  if (game === 'grid_navigation' && Number.isInteger(state.size) && state.size >= 2 && state.size <= 32 && Array.isArray(state.position)) {
    const size = state.size;
    const walls = new Set((state.walls ?? []).map(cell => cell.join(',')));
    const trail = new Set(steps().slice(0, ui.stepIndex).map(step => environmentState(step.state)?.position?.join(',')).filter(Boolean));
    board.classList.add('grid-board'); board.style.setProperty('--board-size', size);
    board.setAttribute('role', 'img'); board.setAttribute('aria-label', `${size} 乘 ${size} 网格；位置 ${state.position.join(',')}；目标 ${state.goal?.join(',')}`);
    for (let row = 0; row < size; row++) for (let col = 0; col < size; col++) {
      const key = `${row},${col}`, classes = ['grid-cell'];
      if (walls.has(key)) classes.push('wall');
      if (trail.has(key)) classes.push('trail');
      if (state.goal?.join(',') === key) classes.push('goal');
      if (state.position.join(',') === key) classes.push('agent');
      const cell = node('div', classes.join(' '));
      cell.title = `行 ${row}，列 ${col}`;
      cell.append(node('span', 'coord', `${row},${col}`));
      if (classes.includes('agent')) cell.append(node('span', '', '●'));
      else if (classes.includes('goal')) cell.append(node('span', '', '◎'));
      board.append(cell);
    }
    legend([['agent', '当前位置'], ['goal', '目标'], ['wall', '墙']]);
  } else if (game === 'tic_tac_toe' && ((typeof state.board === 'string' && state.board.length === 9) || Array.isArray(state.board))) {
    const cells = Array.isArray(state.board) ? state.board.flat() : [...state.board];
    const lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
    const winning = lines.find(([a, b, c]) => ['X', 'O'].includes(cells[a]) && cells[a] === cells[b] && cells[b] === cells[c]);
    const turn = winning ? `${cells[winning[0]]} 已连成三格` : cells.every(value => value !== '.') ? '棋盘已满' : `轮到 ${state.player ?? '未记录'}`;
    board.classList.add('ttt-board'); board.setAttribute('role', 'img'); board.setAttribute('aria-label', `井字棋，${turn}；棋盘 ${cells.join('')}`);
    cells.slice(0, 9).forEach((value, index) => {
      const token = value === '.' || value == null ? '' : String(value);
      const chosen = currentStep()?.action === `cell_${index + 1}`;
      const cell = node('div', `ttt-cell ${token.toLowerCase()}${chosen ? ' chosen' : ''}`);
      if (winning?.includes(index)) cell.style.borderColor = 'var(--green)';
      cell.append(node('span', 'cell-index', index + 1), node('span', '', token)); board.append(cell);
    });
    legend([['agent', '黄色描边：本步选中格'], ['goal', turn]]);
  } else {
    board.classList.add('empty-board'); board.removeAttribute('role'); board.removeAttribute('aria-label');
    const pre = node('pre', 'state-preview', typeof state === 'string' ? state : json(state));
    pre.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;max-height:350px;overflow:auto;font-size:11px;max-width:100%;';
    board.append(pre);
  }
}

function probabilityEntries(value) {
  if (Array.isArray(value)) return value.map((item, i) => record(item)
    ? [String(item.id ?? item.candidate_id ?? item.label ?? i), item.probability ?? item.p ?? item.value]
    : [String(i), item]).filter(([, p]) => number(p));
  if (!record(value)) return [];
  return Object.entries(value).filter(([, p]) => number(p));
}
function answerProbabilities(answer) {
  if (!record(answer)) return null;
  if (answer.probabilities) return answer.probabilities;
  const p = answer.p_true ?? (answer.type === 'boolean' ? answer.probability ?? answer.value : undefined);
  return number(p) ? { false: 1 - p, true: p } : null;
}
function renderProbabilities(container, probabilities, selected) {
  const entries = probabilityEntries(probabilities);
  container.replaceChildren();
  if (!entries.length) { container.append(node('p', 'muted-copy', '此步骤没有记录概率向量。')); return null; }
  const sum = entries.reduce((total, [, p]) => total + p, 0);
  const subset = entries.slice(0, 255);
  for (const [id, p] of subset) {
    const row = node('div', `probability-row${String(selected) === id ? ' selected' : ''}`);
    row.dataset.candidateId = id;
    const label = node('div', 'probability-label');
    label.append(node('span', 'label', actionName(id)), node('span', 'value', `${(p * 100).toFixed(2)}%`));
    const track = node('div', 'probability-track'), fill = node('div', 'probability-fill');
    fill.style.width = `${Math.min(1, Math.max(0, p)) * 100}%`;
    track.append(fill); row.append(label, track); container.append(row);
  }
  if (entries.length > 255) container.append(node('p', 'fine-print', `仅显示前 255 / ${entries.length} 项，完整值见 JSON。`));
  return sum;
}
function actionDistribution(step) {
  if (!step) return null;
  if (probabilityEntries(step.probabilities).length) return step.probabilities;
  const answers = step.answers?.answers ?? step.answers;
  if (!record(answers)) return null;
  const choices = Object.values(answers).filter(answer => answer?.type === 'choice');
  const match = choices.find(answer => own(answer.probabilities, step.action)) ?? choices[0];
  return match?.probabilities ?? null;
}
function addFact(container, key, value) {
  const item = node('div'); item.append(node('dt', '', key), node('dd', '', display(value))); container.append(item);
}
function renderStep() {
  const episode = selectedEpisode(), list = steps(), step = currentStep(), final = Boolean(episode) && ui.stepIndex >= list.length;
  renderBoard(visibleState());
  $('gameName').textContent = labels[episode?.game ?? visibleState()?.game] ?? episode?.game ?? '等待游戏记录';
  $('splitLabel').textContent = episode ? `${labels[episode.split] ?? episode.split ?? '未标分区'} · 回放${episode.outcome ? ` · ${labels[episode.outcome] ?? display(episode.outcome)}` : ''}` : '回放';
  $('stepLabel').textContent = !episode ? '未载入' : final ? `终局 · 共 ${list.length} 步` : `行动前 · 第 ${ui.stepIndex + 1} / ${list.length} 步`;
  $('actionLabel').textContent = step ? `${step.actor ? `${labels[step.actor] ?? step.actor} · ` : ''}${actionName(step.action)}` : final ? '轨迹结束' : '—';
  const outcome = episode?.outcome ?? episode?.result;
  $('episodeOutcome').textContent = episode ? labels[outcome] ?? display(outcome) : '尚未载入';
  const mass = renderProbabilities($('actionProbabilities'), actionDistribution(step), step?.action);
  const warning = mass !== null && Math.abs(mass - 1) > 1e-5;
  const controller = step?.controller ?? selectedModel()?.controller ?? selectedModel()?.policy;
  $('probabilityNote').textContent = mass === null ? (final ? '终局没有下一步动作分布。' : '未记录概率时不补成 one-hot 或均匀分布。')
    : warning ? `原始概率和为 ${mass.toPrecision(6)}，展示未重新归一化。动作概率不等于最终胜率。`
      : controller === 'sample' ? '金色标记实际抽中的动作，不一定是 argmax。原始 T=1 分布不是最终胜率。' : '原始概率按记录展示。动作策略分布不等于最终胜率。';
  $('probabilityNote').classList.toggle('warning', warning);
  $('distributionType').textContent = step?.forced ? '强制 / 环境动作' : '原始记录';
  const facts = $('executionFacts'); facts.replaceChildren();
  addFact(facts, '记录类型', episode ? '已保存的回放' : '尚无执行记录');
  addFact(facts, '模型前向', step ? own(step, 'model_forward') ? step.model_forward : '未记录' : '—');
  addFact(facts, '强制动作', step ? own(step, 'forced') ? step.forced : '未记录' : '—');
  if (step && ['greedy', 'sample', 'random', 'oracle'].includes(controller)) addFact(facts, '执行策略', labels[controller]);
  if (step?.distribution_argmax !== undefined) addFact(facts, '分布 argmax', actionName(step.distribution_argmax));
  if (number(step?.sample_uniform_draw)) addFact(facts, '本步采样随机数', step.sample_uniform_draw);
  const execution = step?.execution ?? step?.model_forward;
  if (record(execution)) for (const key of ['forward_passes', 'precision', 'autoregressive_decode_steps', 'total_paths']) if (own(execution, key)) addFact(facts, key, execution[key]);
  const latency = step?.latency_ms ?? step?.model_latency_ms ?? step?.timing?.model_ms;
  if (number(latency)) addFact(facts, '单步延迟 · ms', latency);
  if (step?.actor) addFact(facts, '执行方', step.actor);
  $('rawStep').textContent = step ? json(step) : episode ? json({ final_state: visibleState(), outcome: episode.outcome, total_steps: list.length }) : '尚未载入';
  $('stepSlider').max = String(list.length); $('stepSlider').value = String(ui.stepIndex); $('stepSlider').disabled = !list.length;
  $('previousButton').disabled = !episode || ui.stepIndex <= 0;
  $('nextButton').disabled = !episode || ui.stepIndex >= list.length;
  $('playButton').disabled = !list.length;
  if (!ui.batches.length) renderParallel();
}
function renderEpisodes() {
  const list = episodes();
  $('episodeSelect').replaceChildren(...(list.length ? list.map((episode, i) => option(i, `${episode.id ?? `对局 ${i + 1}`} · ${labels[episode.game] ?? episode.game ?? '游戏'} · ${episode.split ?? '未标分区'}`)) : [option('', '该模型尚无游戏轨迹')]));
  $('episodeSelect').disabled = !list.length; $('episodeSelect').value = list.length ? String(ui.episodeIndex) : '';
  renderStep();
}

function batchStates(batch) {
  if (Array.isArray(batch?.states)) return batch.states;
  if (Array.isArray(batch?.result?.states)) return batch.result.states;
  if (Array.isArray(batch?.response?.states)) return batch.response.states;
  return [];
}
function renderAnswers(container, states) {
  container.replaceChildren();
  for (const [index, state] of states.entries()) {
    const card = node('article', 'card state-card');
    card.append(node('h3', '', state.id ?? state.state_id ?? `状态 ${index + 1}`));
    if (state.state !== undefined) card.append(node('pre', 'state-preview', typeof state.state === 'string' ? state.state : json(state.state)));
    const answers = state.answers?.answers ?? state.answers;
    if (!record(answers) || !Object.keys(answers).length) card.append(node('p', 'muted-copy', '此状态未记录分题答案。'));
    else for (const [id, answer] of Object.entries(answers)) {
      const block = node('section', 'question-block'), heading = node('div', 'question-title');
      heading.append(node('span', '', id), node('span', 'pill muted', answer?.type ?? '未标类型')); block.append(heading);
      const probabilities = node('div', 'probability-list');
      const selected = answer?.choice ?? (typeof answer?.value === 'string' ? answer.value : answer?.level);
      const mass = renderProbabilities(probabilities, answerProbabilities(answer), selected); block.append(probabilities);
      if (mass !== null && Math.abs(mass - 1) > 1e-5) block.append(node('p', 'fine-print warning', `原始和 ${mass.toPrecision(6)}；未归一化`));
      if (number(answer?.score)) block.append(node('p', 'question-summary', `Score 期望：${display(answer.score)}`));
      if (selected !== undefined) block.append(node('p', 'question-summary', `选择：${actionName(selected)}`));
      card.append(block);
    }
    container.append(card);
  }
}
function collectBatches() {
  const model = selectedModel();
  const candidates = model?.parallel_batches ?? ui.data?.parallel_batches ?? ui.data?.batch_examples ?? [];
  ui.batches = (Array.isArray(candidates) ? candidates : [candidates]).filter(batch => !batch.model || !model || batch.model === model.name || batch.model === model.id);
  ui.batchIndex = Math.min(ui.batchIndex, Math.max(0, ui.batches.length - 1));
  $('batchSelect').replaceChildren(...(ui.batches.length ? ui.batches.map((batch, index) => option(index, batch.id ?? `批次 ${index + 1}`)) : [option('', '当前步骤 · 同一状态多题')]));
  $('batchSelect').value = ui.batches.length ? String(ui.batchIndex) : ''; $('batchSelect').disabled = !ui.batches.length;
}
function renderParallel() {
  const batch = ui.batches[ui.batchIndex];
  if (batch) {
    const states = batchStates(batch), execution = batch.execution ?? batch.result?.execution ?? batch.response?.execution;
    const measured = record(execution) && own(execution, 'forward_passes');
    $('batchEvidence').textContent = `${states.length} 个状态来自同一保存批次。${measured ? `执行记录：${execution.forward_passes} 次 forward，${execution.precision ?? '未标精度'}。` : '没有 forward 次数证据，不据此宣称一次前向。'}${states.length < 2 ? '此记录不足以展示多状态同批。' : ''}`;
    renderAnswers($('parallelStates'), states); $('rawBatch').textContent = json(batch);
    if (!states.length) empty($('parallelStates'), '批次没有可读取的状态', '完整数据仍可在原始 JSON 中查看。');
  } else {
    const step = currentStep();
    $('batchEvidence').textContent = '这里展示当前回放步骤的同一状态多题答案。尚无多状态同批执行记录，不把不同步骤拼接成一个批次。';
    if (step?.answers) {
      renderAnswers($('parallelStates'), [{ id: selectedEpisode()?.id ?? '当前状态', state: step.state, answers: step.answers }]);
      $('rawBatch').textContent = json({ kind: 'single_state_step_not_multistate_batch', state: step.state, answers: step.answers });
    } else {
      empty($('parallelStates'), '等待并行推理记录', '结果文件可提供 parallel_batches，或在步骤中保存 answers。'); $('rawBatch').textContent = '没有批次记录';
    }
  }
}
function changeModel() {
  stopPlayback(); ui.episodeIndex = 0; ui.stepIndex = 0; ui.batchIndex = 0;
  renderSummary(); collectBatches(); renderEpisodes(); renderParallel();
}

async function loadData() {
  stopPlayback(); $('reloadButton').disabled = true; notice('');
  try {
    const response = await fetch('./demo_results.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(response.status === 404 ? 'missing' : `HTTP ${response.status}`);
    const data = await response.json();
    if (!record(data) || !Array.isArray(data.models)) throw new Error('schema');
    if (data.models.some(model => !record(model) || (model.episodes !== undefined && !Array.isArray(model.episodes)))) throw new Error('schema');
    const previousIdentity = modelIdentity(selectedModel());
    ui.data = data; ui.models = data.models;
    const previousIndex = ui.models.findIndex(model => modelIdentity(model) === previousIdentity);
    // 预登记的主组；不读取成绩来选默认模型或控制器。刷新时保留用户选择。
    const primaryIndex = ui.models.findIndex(model => [model.name, model.source_artifact].some(value => String(value ?? '').includes('v3_gold_coords_multi_seed17')) && (model.controller ?? model.policy) === 'greedy');
    ui.modelIndex = previousIndex >= 0 ? previousIndex : primaryIndex >= 0 ? primaryIndex : 0;
    $('modelSelect').replaceChildren(...(ui.models.length ? ui.models.map((model, i) => option(i, modelLabel(model, i))) : [option('', '结果文件尚无模型')]));
    $('modelSelect').disabled = !ui.models.length; $('modelSelect').value = String(ui.modelIndex);
    $('dataStatus').className = `status-chip${ui.models.length ? ' ready' : ''}`;
    $('dataStatus').lastElementChild.textContent = ui.models.length ? `已载入 ${ui.models.length} 个模型 / 基线` : '结果文件为空';
    $('artifactMeta').textContent = data.generated_at ? `结果生成于 ${data.generated_at}` : '数据：本地 demo_results.json';
    if (!ui.models.length) notice('结果文件尚无模型记录。页面不会填充模拟成功数据。');
    if (data.notes) notice(Array.isArray(data.notes) ? data.notes.join('\n') : String(data.notes));
    renderSources(); changeModel();
  } catch (error) {
    ui.data = null; ui.models = []; ui.modelIndex = 0;
    $('modelSelect').replaceChildren(option('', '等待真实实验数据')); $('modelSelect').disabled = true;
    $('dataStatus').className = 'status-chip error'; $('dataStatus').lastElementChild.textContent = '尚未载入实验记录';
    const message = location.protocol === 'file:' ? '请通过 HTTP 服务打开页面。浏览器不能从 file:// 页面读取结果 JSON。'
      : error.message === 'schema' ? '结果 JSON 不符合约定的 models / episodes 结构。请检查数据文件；未显示任何模拟结果。'
        : error.message === 'missing' ? '尚未找到 demo_results.json。请生成真实实验记录放入 web 目录，再点击“刷新记录”。'
          : '无法读取或解析 demo_results.json。请检查本地服务和结果文件；页面没有调用教师 API。';
    notice(message); renderSources(); changeModel();
  } finally { $('reloadButton').disabled = false; }
}

function setStep(value) { ui.stepIndex = Math.max(0, Math.min(steps().length, value)); renderStep(); }
function togglePlayback() {
  if (ui.timer !== null) { stopPlayback(); return; }
  if (!steps().length) return;
  if (ui.stepIndex >= steps().length) setStep(0);
  $('playButton').textContent = '暂停';
  ui.timer = setInterval(() => { setStep(ui.stepIndex + 1); if (ui.stepIndex >= steps().length) stopPlayback(); }, Number($('speedSelect').value));
}

$('reloadButton').addEventListener('click', loadData);
$('modelSelect').addEventListener('change', event => { ui.modelIndex = Number(event.target.value); changeModel(); });
$('episodeSelect').addEventListener('change', event => { stopPlayback(); ui.episodeIndex = Number(event.target.value); ui.stepIndex = 0; renderStep(); renderParallel(); });
$('previousButton').addEventListener('click', () => { stopPlayback(); setStep(ui.stepIndex - 1); });
$('nextButton').addEventListener('click', () => { stopPlayback(); setStep(ui.stepIndex + 1); });
$('stepSlider').addEventListener('input', event => { stopPlayback(); setStep(Number(event.target.value)); });
$('playButton').addEventListener('click', togglePlayback);
$('speedSelect').addEventListener('change', () => { if (ui.timer !== null) { stopPlayback(); togglePlayback(); } });
$('batchSelect').addEventListener('change', event => { ui.batchIndex = Number(event.target.value); renderParallel(); });
document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  stopPlayback();
  document.querySelectorAll('.tab').forEach(item => { const active = item === tab; item.classList.toggle('active', active); item.setAttribute('aria-pressed', String(active)); $(item.dataset.panel).hidden = !active; });
  if (tab.dataset.panel === 'parallelPanel') renderParallel();
}));
document.addEventListener('visibilitychange', () => { if (document.hidden) stopPlayback(); });

$('liveForm').addEventListener('submit', async event => {
  event.preventDefault();
  const status = $('liveStatus'); status.classList.remove('error');
  let request;
  try { request = JSON.parse($('liveInput').value); if (!record(request)) throw new Error('object_required'); }
  catch { status.textContent = '请求必须是有效 JSON 对象；尚未发送。'; status.classList.add('error'); return; }
  if (location.protocol === 'file:') { status.textContent = '请先启动 HTTP 模型服务，再使用实时接口。'; status.classList.add('error'); return; }
  $('evaluateButton').disabled = true; status.textContent = '正在等待本地模型推理…';
  $('liveAnswers').replaceChildren(node('p', 'muted-copy', '等待这次真实请求的结果…'));
  $('liveRaw').textContent = '本次请求尚无响应'; $('liveLatency').textContent = '进行中';
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 120000), started = performance.now();
  try {
    const response = await fetch('/api/evaluate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request), signal: controller.signal });
    if ([404, 405, 501].includes(response.status)) throw new Error('service_missing');
    const text = await response.text();
    let result; try { result = JSON.parse(text); } catch { throw new Error('non_json_response'); }
    $('liveRaw').textContent = json(result);
    if (!response.ok) throw new Error(`http_${response.status}`);
    const elapsed = performance.now() - started;
    $('liveLatency').textContent = `${elapsed.toFixed(0)} ms · 浏览器往返`;
    const states = Array.isArray(result.states) ? result.states : Array.isArray(result.result?.states) ? result.result.states
      : record(result.answers) ? [{ id: '响应', answers: result.answers }] : [];
    if (states.length) renderAnswers($('liveAnswers'), states);
    else $('liveAnswers').replaceChildren(node('p', 'muted-copy', '服务已返回 JSON；该响应结构未匹配图形视图，请查看完整响应。'));
    status.textContent = '已完成一次真实请求。往返时间包含网络和服务处理，不等于纯 GPU 计算时间。';
  } catch (error) {
    status.classList.add('error');
    status.textContent = error.message === 'service_missing' ? '当前服务器只提供静态页面。请启动实现 POST /api/evaluate 的模型推理服务。'
      : error.name === 'AbortError' ? '等待超过 120 秒，浏览器已取消请求；服务端是否仍在运行需查看服务日志。'
        : error.message === 'non_json_response' ? '服务没有返回 JSON。请确认请求连接的是模型推理服务。'
          : error.message.startsWith('http_') ? `服务返回 HTTP ${error.message.slice(5)}，详见响应 JSON。` : '实时请求未完成。请检查本地模型推理服务；没有使用预设答案。';
    $('liveLatency').textContent = '请求未完成';
  } finally { clearTimeout(timeout); $('evaluateButton').disabled = false; }
});

renderSources();
loadData();
