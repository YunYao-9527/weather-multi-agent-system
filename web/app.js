const tokenInput = document.getElementById('token');
const cityInput = document.getElementById('city');
const areaInput = document.getElementById('area');
const minProbInput = document.getElementById('minProb');
const truthStartInput = document.getElementById('truthStart');
const truthEndInput = document.getElementById('truthEnd');
const llmProviderInput = document.getElementById('llmProvider');
const llmModeInput = document.getElementById('llmMode');
const llmModelInput = document.getElementById('llmModel');
const llmApiKeyInput = document.getElementById('llmApiKey');

const healthStatus = document.getElementById('healthStatus');
const llmStatusBadge = document.getElementById('llmStatusBadge');
const llmSummary = document.getElementById('llmSummary');
const llmBox = document.getElementById('llmBox');
const mapStatus = document.getElementById('mapStatus');
const sourceSummary = document.getElementById('sourceSummary');
const sourceBox = document.getElementById('sourceBox');
const probSummary = document.getElementById('probSummary');
const readinessSummary = document.getElementById('readinessSummary');
const auditSummary = document.getElementById('auditSummary');
const auditBox = document.getElementById('auditBox');
const evalSummary = document.getElementById('evalSummary');
const evolveSummary = document.getElementById('evolveSummary');
const evalBox = document.getElementById('evalBox');
const weightBox = document.getElementById('weightBox');
const rawSummary = document.getElementById('rawSummary');
const rawBox = document.getElementById('rawBox');
const autoWindowSummary = document.getElementById('autoWindowSummary');
const objectSummary = document.getElementById('objectSummary');
const objectBox = document.getElementById('objectBox');
const policySummary = document.getElementById('policySummary');
const policyBox = document.getElementById('policyBox');
const familySummary = document.getElementById('familySummary');
const familyBox = document.getElementById('familyBox');
const traceSummary = document.getElementById('traceSummary');
const traceBox = document.getElementById('traceBox');
const providerSummary = document.getElementById('providerSummary');
const providerBox = document.getElementById('providerBox');
const registrySummary = document.getElementById('registrySummary');
const registryBox = document.getElementById('registryBox');
const truthSummary = document.getElementById('truthSummary');
const truthBox = document.getElementById('truthBox');
const objectsSummary = document.getElementById('objectsSummary');
const objectsBox = document.getElementById('objectsBox');
const objectTimeline = document.getElementById('objectTimeline');
const replayVerifyCard = document.getElementById('replayVerifyCard');
const familyChart = document.getElementById('familyChart');
const truthVersionLeft = document.getElementById('truthVersionLeft');
const truthVersionRight = document.getElementById('truthVersionRight');
const compareTruthVersionsBtn = document.getElementById('compareTruthVersions');
const truthCompareChart = document.getElementById('truthCompareChart');
const objectsMiniTimeline = document.getElementById('objectsMiniTimeline');
const runCompareLeft = document.getElementById('runCompareLeft');
const runCompareRight = document.getElementById('runCompareRight');
const bundleCompareLeft = document.getElementById('bundleCompareLeft');
const bundleCompareRight = document.getElementById('bundleCompareRight');
const compareRunsBtn = document.getElementById('compareRuns');
const compareBundlesBtn = document.getElementById('compareBundles');
const replayCompareSummary = document.getElementById('replayCompareSummary');
const replayCompareBox = document.getElementById('replayCompareBox');
const replayCompareChart = document.getElementById('replayCompareChart');
const versionDetailSummary = document.getElementById('versionDetailSummary');
const versionDetailBox = document.getElementById('versionDetailBox');
const evidenceCards = document.getElementById('evidenceCards');
const probBars = document.getElementById('probBars');
const readinessBars = document.getElementById('readinessBars');
const radarToggle = document.getElementById('radarToggle');

const stateStatus = document.getElementById('stateStatus');
const stateAction = document.getElementById('stateAction');
const stateReadiness = document.getElementById('stateReadiness');
const stateDataQuality = document.getElementById('stateDataQuality');
const stateProxy = document.getElementById('stateProxy');
const stateRunId = document.getElementById('stateRunId');

const btnLocate = document.getElementById('locateCity');
const btnLive = document.getElementById('runLive');
const btnEval = document.getElementById('runEval');
const btnEvolve = document.getElementById('runEvolve');
const btnAutoWindow = document.getElementById('runAutoWindow');
const btnBuildTruth = document.getElementById('buildTruth');
const btnRefreshGovernance = document.getElementById('refreshGovernance');
const btnRefreshLLM = document.getElementById('refreshLLM');
const btnApplyLLM = document.getElementById('applyLLM');

const busyBar = document.getElementById('busyBar');
const busyText = document.getElementById('busyText');

const HAZARD_LABELS = {
  short_rain: '短时强降水',
  wind: '雷暴大风',
  hail: '冰雹',
  tornado: '龙卷',
};

let map = null;
let marker = null;
let cityDebounceTimer = null;
let baseLayer = null;
let radarLayer = null;
let radarTemplate = '';
let lastTraceId = '';
let lastBundleId = '';
const AUTO_MIN_TOTAL_POSITIVE = 5;
const AUTO_MIN_TRAIN_SAMPLES = 20;
const AUTO_MIN_CALIBRATION_SAMPLES = 12;

function authHeader() {
  const token = tokenInput?.value?.trim?.() || '';
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function setBusy(isBusy, text = '处理中...') {
  if (busyBar) busyBar.classList.toggle('hidden', !isBusy);
  if (busyText) busyText.textContent = text;
  [btnLocate, btnLive, btnEval, btnEvolve, btnAutoWindow, btnBuildTruth, btnRefreshGovernance, btnRefreshLLM, btnApplyLLM].forEach((b) => {
    if (!b) return;
    b.disabled = isBusy;
  });
}

function setButtonLoading(button, isLoading) {
  if (!button) return;
  button.classList.toggle('loading', isLoading);
}

async function getJson(url) {
  const r = await fetch(url, { headers: authHeader() });
  const j = await r.json();
  if (!r.ok || j.code !== 'OK') {
    throw new Error(`${j.code || r.status}: ${j.message || 'request failed'}`);
  }
  return j;
}

async function postJson(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeader() },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok || j.code !== 'OK') {
    throw new Error(`${j.code || r.status}: ${j.message || 'request failed'}`);
  }
  return j;
}

async function postJsonFirstOk(urls, body) {
  const errors = [];
  for (const url of urls) {
    try {
      const j = await postJson(url, body);
      return { data: j, url };
    } catch (e) {
      errors.push(`${url}: ${String(e)}`);
    }
  }
  throw new Error(errors.join(' | '));
}

function asPct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

function initMap() {
  if (!document.getElementById('map') || typeof L === 'undefined') return;
  map = L.map('map', { zoomControl: true }).setView([39.14, 117.17], 8);
  baseLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap',
  }).addTo(map);
}

function applyRadarLayer() {
  if (!map) return;
  const on = Boolean(radarToggle?.checked);
  if (!on || !radarTemplate) {
    if (radarLayer) {
      map.removeLayer(radarLayer);
      radarLayer = null;
    }
    return;
  }
  if (radarLayer) {
    map.removeLayer(radarLayer);
    radarLayer = null;
  }
  radarLayer = L.tileLayer(radarTemplate, {
    opacity: 0.6,
    maxZoom: 18,
    attribution: 'RainViewer',
  }).addTo(map);
}

function updateMap(lat, lon, label) {
  if (!map) return;
  const p = [Number(lat), Number(lon)];
  if (!marker) {
    marker = L.marker(p).addTo(map);
  } else {
    marker.setLatLng(p);
  }
  marker.bindPopup(label || `${lat}, ${lon}`).openPopup();
  map.setView(p, 8);
}

async function checkHealth() {
  try {
    const j = await (await fetch('/api/v1/health')).json();
    if (healthStatus) healthStatus.textContent = `${j.data.status} | v${j.data.version}`;
  } catch (e) {
    if (healthStatus) healthStatus.textContent = 'health failed';
  }
}

function renderLLMRuntime(data) {
  if (llmProviderInput) llmProviderInput.value = data?.provider || 'openai';
  if (llmModeInput) llmModeInput.value = data?.effective_mode || 'off';
  if (llmModelInput && data?.model) llmModelInput.value = data.model;
  if (llmStatusBadge) llmStatusBadge.textContent = `${data?.effective_mode || 'off'} | ${data?.status || 'unknown'}`;
  if (llmSummary) {
    llmSummary.textContent = [
      `当前模式：${data?.effective_mode || 'off'}（configured=${data?.configured_mode || '-' }）`,
      `状态：${data?.status || '-'}，ready=${data?.ready ? 'yes' : 'no'}，key_present=${data?.key_present ? 'yes' : 'no'}`,
      `provider=${data?.provider || '-'}，model=${data?.model || '-'}`,
      `timeout=${Number(data?.timeout_sec || 0).toFixed(1)}s，max_tokens=${data?.max_output_tokens || '-'}，temperature=${Number(data?.temperature || 0).toFixed(2)}`,
      `说明：${data?.note || '-'}`,
    ].join('\n');
  }
  if (llmBox) llmBox.textContent = JSON.stringify(data || {}, null, 2);
}

async function refreshLLMRuntime() {
  if (!llmSummary && !llmBox && !llmStatusBadge) return;
  try {
    const j = await getJson('/api/v1/runtime/llm');
    renderLLMRuntime(j.data || {});
  } catch (e) {
    if (llmStatusBadge) llmStatusBadge.textContent = 'llm failed';
    if (llmSummary) llmSummary.textContent = `LLM 状态加载失败：${String(e)}`;
    if (llmBox) llmBox.textContent = String(e);
  }
}

async function applyLLMRuntime() {
  const provider = llmProviderInput?.value || 'openai';
  const mode = llmModeInput?.value || 'off';
  const model = llmModelInput?.value?.trim?.() || '';
  const apiKey = llmApiKeyInput?.value?.trim?.() || '';
  const body = {
    provider,
    mode,
    model,
  };
  if (apiKey) body.api_key = apiKey;
  const j = await postJson('/api/v1/runtime/llm', body);
  if (llmApiKeyInput) llmApiKeyInput.value = '';
  renderLLMRuntime(j.data || {});
}

async function locateCity() {
  const city = cityInput?.value?.trim?.() || '';
  if (!city) return;
  setButtonLoading(btnLocate, true);
  setBusy(true, '定位城市中...');
  try {
    const j = await getJson(`/api/v1/geo/city?city=${encodeURIComponent(city)}`);
    const d = j.data;
    updateMap(d.latitude, d.longitude, `${d.name} (${d.latitude.toFixed(3)}, ${d.longitude.toFixed(3)})`);
    if (mapStatus) mapStatus.textContent = `定位：${d.name} / ${d.country || '-'} / ${d.admin1 || '-'} / ${d.timezone || '-'}`;
  } catch (e) {
    if (mapStatus) mapStatus.textContent = `定位失败：${String(e)}`;
  } finally {
    setBusy(false);
    setButtonLoading(btnLocate, false);
  }
}

function renderBars(container, mapData, labels = {}) {
  if (!container) return;
  container.innerHTML = '';
  Object.entries(mapData || {}).forEach(([k, v]) => {
    const row = document.createElement('div');
    row.className = 'bar';
    const pct = Math.max(1, Number(v) * 100);
    row.innerHTML = `
      <div class="bar-head"><span>${labels[k] || k}</span><span>${(Number(v) * 100).toFixed(1)}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
    `;
    container.appendChild(row);
  });
}

function setSelectOptions(select, items, getValue, getLabel) {
  if (!select) return;
  const previous = select.value;
  select.innerHTML = '';
  (items || []).forEach((item, idx) => {
    const opt = document.createElement('option');
    opt.value = getValue(item);
    opt.textContent = getLabel(item, idx);
    select.appendChild(opt);
  });
  if (previous && [...select.options].some((o) => o.value === previous)) {
    select.value = previous;
  } else if (select.options.length > 0) {
    select.selectedIndex = 0;
  }
}

function renderMetricGrid(container, items) {
  if (!container) return;
  container.innerHTML = '';
  (items || []).forEach((item) => {
    const div = document.createElement('div');
    div.className = 'metric-card';
    div.innerHTML = `<div class="label">${item.label}</div><div class="value">${item.value}</div>`;
    container.appendChild(div);
  });
}

function renderTimeline(container, steps) {
  if (!container) return;
  container.innerHTML = '';
  (steps || []).forEach((step) => {
    const div = document.createElement('div');
    div.className = 'timeline-step';
    div.innerHTML = `
      <div class="step-title">${step.title}</div>
      <div class="step-value">${step.value}</div>
      <div class="meta">${step.meta || ''}</div>
    `;
    container.appendChild(div);
  });
}

function renderCompareStack(container, rows) {
  if (!container) return;
  container.innerHTML = '';
  (rows || []).forEach((row) => {
    const wrap = document.createElement('div');
    wrap.className = 'compare-row';
    const bars = (row.bars || []).map((bar) => {
      const left = Math.max(0, Math.min(100, Number(bar.leftPct || 0)));
      const right = Math.max(0, Math.min(100, Number(bar.rightPct || 0)));
      return `
        <div class="compare-bar">
          <span>${bar.label}</span>
          <div class="track">
            <div class="left" style="width:${left}%"></div>
            <div class="right" style="width:${right}%"></div>
          </div>
          <span class="delta">${bar.delta}</span>
        </div>
      `;
    }).join('');
    wrap.innerHTML = `
      <div class="compare-head"><span>${row.title}</span><span>${row.meta || ''}</span></div>
      <div class="compare-bars">${bars}</div>
    `;
    container.appendChild(wrap);
  });
}

function renderStateBand(data) {
  if (!stateStatus || !stateAction || !stateReadiness || !stateDataQuality || !stateProxy || !stateRunId) return;
  const d = data?.enhanced?.decision || {};
  const src = data?.source_meta || {};
  let status = '正常';
  if (d.requires_human_review) status = '需人工复核';
  else if (d.degraded_mode) status = '降级运行';

  stateStatus.textContent = status;
  stateAction.textContent = d.action || '-';
  stateReadiness.textContent = Number(d.evidence_readiness_score || d.confidence || 0).toFixed(3);
  stateDataQuality.textContent = Number(src.data_quality_score || 0).toFixed(3);
  stateProxy.textContent = asPct(d.proxy_dependency_ratio || 0);
  stateRunId.textContent = data?.run_id || '-';
}

function renderSource(data) {
  if (!sourceSummary || !sourceBox) return;
  const src = data?.source_meta || {};
  const area = data?.area_resolution || {};
  const models = String(src.models || '').split(',').filter(Boolean);
  radarTemplate = String(src.radar_tile_url_template || '');
  applyRadarLayer();
  sourceSummary.textContent = [
    `模式：${models.length ? models.join(' / ') : '-'}，雷达：${src.radar_source || '-'}`,
    `数据质量：${Number(src.data_quality_score || 0).toFixed(3)}，模式离散：${Number(src.model_spread_score || 0).toFixed(3)}`,
    `雷达时效：${Number(src.radar_age_minutes || 0).toFixed(1)} 分钟，覆盖状态：${area.status || '-'}`,
    `雷达图层：${radarTemplate ? `可用（${src.radar_visual_source || 'same-origin'}）` : '不可用'}`,
    `自动落区：${area.area_text || '-'}${area.reason ? `（${area.reason}）` : ''}`,
  ].join('\n');
  sourceBox.textContent = JSON.stringify({ source_meta: src, area_resolution: area }, null, 2);
}

function renderObjectAndPolicy(data) {
  const enhanced = data?.enhanced || {};
  const hazardObject = enhanced?.hazard_object || {};
  const decisionPacket = enhanced?.decision_packet || {};
  const fusion = enhanced?.fusion_result || {};
  const decision = enhanced?.decision || {};
  if (objectSummary && objectBox) {
    objectSummary.textContent = [
      `对象ID：${hazardObject.object_id || '-'}`,
      `灾种：${HAZARD_LABELS[hazardObject.hazard_type] || hazardObject.hazard_type || '-'}`,
      `状态：${hazardObject.lifecycle_state || '-'}`,
      `面积：${Number(hazardObject.area_km2 || 0).toFixed(1)} km²`,
      `持续性：${hazardObject.evidence_persistence || 0} 周期，最小持续满足=${hazardObject.min_duration_met ? '是' : '否'}`,
      `源稳定性：${Number(hazardObject.source_stability || 0).toFixed(3)}，运动稳定性：${Number(hazardObject.motion_stability || 0).toFixed(3)}`,
    ].join('\n');
    objectBox.textContent = JSON.stringify(hazardObject, null, 2);
  }
  renderTimeline(objectTimeline, [
    { title: 'Candidate', value: hazardObject.object_version ? `v${hazardObject.object_version}` : '-', meta: `start=${hazardObject.start_time || '-'}` },
    { title: 'Watch', value: `${hazardObject.evidence_persistence || 0} cycles`, meta: `state=${hazardObject.lifecycle_state || '-'}` },
    { title: 'Recommend', value: decisionPacket.action || '-', meta: `p=${Number(decisionPacket.effective_probability_used || 0).toFixed(3)}` },
    { title: 'Review', value: decisionPacket.requires_human_review ? 'manual' : 'pass', meta: `policy=${decisionPacket.policy_version || '-'}` },
  ]);
  if (policySummary && policyBox) {
    policySummary.textContent = [
      `Action：${decisionPacket.action || decision.action || '-'}`,
      `Policy：${decisionPacket.policy_version || decision.policy_version || '-'}`,
      `有效概率：${Number(decisionPacket.effective_probability_used || 0).toFixed(3)}`,
      `Issue Gate：${decisionPacket.issue_gate_passed ? '通过' : '未通过'}，Clear Gate：${decisionPacket.clear_gate_passed ? '通过' : '未通过'}`,
      `人工复核：${decisionPacket.requires_human_review ? '是' : '否'}`,
      `Veto：${(decisionPacket.veto_reasons || decision.veto_reasons || []).join(', ') || '无'}`,
    ].join('\n');
    policyBox.textContent = JSON.stringify({ decision_packet: decisionPacket, replay_bundle_id: enhanced?.replay_bundle?.bundle_id || data?.run_id }, null, 2);
  }
  renderMetricGrid(replayVerifyCard, [
    { label: 'Bundle', value: (enhanced?.replay_bundle?.bundle_id || lastBundleId || '-').slice(0, 16) },
    { label: 'Trace', value: (enhanced?.trace_id || lastTraceId || '-').slice(0, 16) },
    { label: 'Veto', value: String((decisionPacket.veto_reasons || []).length || 0) },
  ]);
  if (familySummary && familyBox) {
    familySummary.textContent = [
      `有效源等级：${fusion.effective_source_tier || decision.effective_source_tier || '-'}`,
      `独立家族数：${fusion.independent_family_count ?? '-'}`,
      `降级标志：${(fusion.degraded_mode_flags || decision.degraded_mode_flags || []).join(', ') || '无'}`,
      `家族贡献：${Object.keys(fusion.family_contribution || decision.family_contribution || {}).length} 组`,
    ].join('\n');
    familyBox.textContent = JSON.stringify({
      family_contribution: fusion.family_contribution || decision.family_contribution || {},
      p_release: fusion.p_release || decision.p_release || {},
      lower_confidence_bound: fusion.lower_confidence_bound || decision.lower_confidence_bound || {},
      veto_reasons: fusion.veto_reasons || decision.veto_reasons || [],
    }, null, 2);
  }
  const familyRows = Object.entries(fusion.family_contribution || decision.family_contribution || {}).map(([family, scores]) => ({
    title: family,
    meta: `tier=${fusion.effective_source_tier || decision.effective_source_tier || '-'}`,
    bars: Object.entries(scores || {}).map(([hazard, value]) => ({
      label: HAZARD_LABELS[hazard] || hazard,
      leftPct: Number(value || 0) * 100,
      rightPct: Number((fusion.p_release || decision.p_release || {})[hazard] || 0) * 100,
      delta: `${(Number(value || 0) * 100).toFixed(0)} / ${(Number((fusion.p_release || decision.p_release || {})[hazard] || 0) * 100).toFixed(0)}`,
    })),
  }));
  renderCompareStack(familyChart, familyRows);
}

async function loadTrace(requestId) {
  if (!traceSummary || !traceBox || !requestId) return;
  try {
    const j = await getJson(`/api/v1/audit/trace/${encodeURIComponent(requestId)}`);
    const data = j.data || {};
    traceSummary.textContent = [
      `request_id: ${requestId}`,
      `trace_id: ${data.trace_id || data.audit?.trace_id || '-'}`,
      `bundle_id: ${data.bundle_id || data.audit?.bundle_id || '-'}`,
      `policy_version: ${data.policy_version || data.audit?.policy_version || '-'}`,
    ].join('\n');
    traceBox.textContent = JSON.stringify(data, null, 2);
    lastTraceId = data.trace_id || data.audit?.trace_id || '';
    lastBundleId = data.bundle_id || data.audit?.bundle_id || '';
  } catch (e) {
    traceSummary.textContent = `追踪加载失败：${String(e)}`;
    traceBox.textContent = String(e);
  }
}

function renderProbAndReadiness(data) {
  if (!probSummary || !readinessSummary || !probBars || !readinessBars) return;
  const d = data?.enhanced?.decision || {};
  const probs = d.hazard_prob || {};
  const rawProbs = d.hazard_prob_raw || {};
  const probQualityFactor = Number(d.probability_quality_factor ?? d?.evidence_readiness_breakdown?.prob_quality_factor ?? 1);
  const readiness = d.evidence_readiness_breakdown || {};

  renderBars(probBars, probs, HAZARD_LABELS);
  renderBars(readinessBars, readiness, {
    evidence_strength: '证据强度',
    agreement_score: '一致性',
    data_quality_score: '数据质量',
    freshness_score: '时效',
    model_dispersion_score: '模式稳定性',
    conflict_control_score: '冲突可控性',
  });

  const top = Object.entries(probs).sort((a, b) => b[1] - a[1])[0] || ['-', 0];
  probSummary.textContent = [
    `主导灾种：${HAZARD_LABELS[top[0]] || top[0]} ${asPct(top[1])}`,
    Object.keys(rawProbs).length ? `原始主导概率（未折减）：${asPct(rawProbs[top[0]] || 0)}` : '原始主导概率（未折减）：-',
    `概率质量折减系数：${probQualityFactor.toFixed(3)}（冲突高/代理依赖高时会下调）`,
    `建议动作：${d.action || '-'}，内部等级：${d.level || '-'}`,
    `说明：这是辅助决策等级，不代表官方预警发布等级。`,
  ].join('\n');

  const conflictPenalty = Number(readiness.conflict_penalty ?? readiness.conflict_score ?? 0);
  const conflictControl = Number(readiness.conflict_control_score ?? (1 - conflictPenalty));
  readinessSummary.textContent = [
    `Readiness 总分：${Number(d.evidence_readiness_score || 0).toFixed(3)}`,
    `冲突项：${(d.conflicts || []).length} 条`,
    `冲突惩罚：${asPct(conflictPenalty)}（越低越好）`,
    `冲突可控性：${asPct(conflictControl)}（越高越好）`,
    `代理源依赖：${asPct(d.proxy_dependency_ratio || 0)}`,
    `人工复核：${d.requires_human_review ? '是' : '否'}`,
  ].join('\n');
}

function renderEvidence(data) {
  if (!evidenceCards) return;
  const d = data?.enhanced?.decision || {};
  evidenceCards.innerHTML = '';
  (d.rationale || []).forEach((e) => {
    const div = document.createElement('div');
    div.className = 'card';
    const proxy = e.proxy_source || e.supporting_features?.proxy_source;
    div.innerHTML = `
      <div class="meta">[${e.agent}] conf=${Number(e.confidence || 0).toFixed(2)} ${proxy ? ' | proxy' : ''}</div>
      <div class="claim">${e.claim}</div>
      <div class="meta">source=${(e.upstream_sources || []).join(',') || '-'}</div>
      <div class="badge-row">
        <span class="badge">family=${e.family || '-'}</span>
        <span class="badge">tier=${e.source_tier || '-'}</span>
        <span class="badge">fingerprint=${String(e.provider_fingerprint || '-').slice(0, 10)}</span>
      </div>
    `;
    evidenceCards.appendChild(div);
  });
}

async function loadAudit(runId) {
  if (!auditSummary || !auditBox) return;
  if (!runId) return;
  try {
    const j = await getJson(`/api/v1/audit/${encodeURIComponent(runId)}`);
    auditBox.textContent = JSON.stringify(j.data, null, 2);
    const d = j.data?.decision_snapshot || {};
    auditSummary.textContent = [
      `Run ID: ${runId}`,
      `Action: ${d.action || '-'}，Issue: ${d.issue ? '是' : '否'}，Level: ${d.level || '-'}`,
      `判定链：${(d.decision_trace || []).slice(0, 6).join(' | ')}`,
    ].join('\n');
  } catch (e) {
    auditSummary.textContent = `审计加载失败：${String(e)}`;
    auditBox.textContent = String(e);
  }
}

function renderLive(data, raw) {
  renderStateBand(data);
  renderSource(data);
  renderProbAndReadiness(data);
  renderEvidence(data);
  renderObjectAndPolicy(data);

  if (data?.source_meta?.lat && data?.source_meta?.lon) {
    updateMap(data.source_meta.lat, data.source_meta.lon, `${data.enhanced.observation.city} 实时位置`);
  }
  if (data?.area_resolution?.area_text) {
    if (mapStatus) mapStatus.textContent = `落区：${data.area_resolution.area_text} (${data.area_resolution.status || '-'})`;
  }

  if (rawBox) rawBox.textContent = JSON.stringify(raw, null, 2);
  if (rawSummary) {
    rawSummary.textContent = [
      `request_id: ${raw.request_id}`,
      `trace_id: ${raw.trace_id || '-'}`,
      `run_id: ${data.run_id || '-'}`,
      `decision: action=${data.enhanced.decision.action}, level=${data.enhanced.decision.level}, issue=${data.enhanced.decision.issue}`,
    ].join('\n');
  }
}

function renderEval(data) {
  if (!evalSummary || !evalBox) return;
  const imp = data.improvements || {};
  const h = data.enhanced?.hazards || {};
  const guide = data.metric_guide || {};
  const sv = data.statistical_validity || {};
  const warn = [];
  Object.entries(h).forEach(([k, v]) => {
    if (v?.no_positive_warning) warn.push(`${HAZARD_LABELS[k] || k}: ${v.no_positive_warning}`);
  });
  evalBox.textContent = JSON.stringify(data, null, 2);
  evalSummary.textContent = [
    `样本：${data.samples}（test=${data.split_manifest?.counts?.test || 0}）`,
    `真值覆盖(qualified)：${asPct(data.truth_labels?.qualified_coverage_ratio || 0)}`,
    `统计效力：${sv.event_metrics_reliable ? '有效' : '无效'}（正例总数=${sv.total_positive_labels ?? '-'}）`,
    `改进摘要：短时强降水 Brier=${(imp.hazards?.short_rain?.brier_reduction_pct ?? 0) * 100}% ，大风 Brier=${(imp.hazards?.wind?.brier_reduction_pct ?? 0) * 100}%`,
    sv.warning ? `说明：${sv.warning}` : (warn.length ? `说明：${warn.join(' | ')}` : '说明：指标可正常解释'),
    `指标释义：Brier=${guide.brier || '概率均方误差'}；F1=${guide.f1 || 'precision/recall综合指标'}`,
    `报告：${JSON.stringify(data.reports || {})}`,
    `Registry Run ID：${data.registry_run_id || '-'}`,
  ].join('\n');
}

function renderEvolve(data) {
  if (!evolveSummary || !weightBox) return;
  weightBox.textContent = JSON.stringify(data.learned, null, 2);
  evolveSummary.textContent = [
    `训练窗口：${data.learned?.trained_period?.start || '-'} ~ ${data.learned?.trained_period?.end || '-'}`,
    `合格样本(train/calib/test)：${data.learned?.qualified_counts?.train || 0}/${data.learned?.qualified_counts?.calibration || 0}/${data.learned?.qualified_counts?.test || 0}`,
    `回退原因：${data.learned?.fallback_reason || '无'}`,
    `Registry Run ID：${data.registry_run_id || '-'}`,
  ].join('\n');

  if (data.post_eval) {
    if (evalBox) evalBox.textContent = JSON.stringify(data.post_eval, null, 2);
    if (evalSummary) {
      evalSummary.textContent = [
        `后评估改进：${JSON.stringify(data.post_eval?.improvements || {})}`,
        `真值覆盖(qualified)：${asPct(data.post_eval?.truth_labels?.qualified_coverage_ratio || 0)}`,
        `报告：${JSON.stringify(data.post_eval?.reports || {})}`,
      ].join('\n');
    }
  }
}

function renderGovernance({ providers, providerHealth, registries, truthVersions, objects }) {
  if (providerSummary && providerBox) {
    providerSummary.textContent = [
      `Provider 数：${providers?.items?.length || 0}`,
      `最近运行：${providerHealth?.latest_run_id || '-'}`,
      `健康摘要：${JSON.stringify(providerHealth?.source_health?.warnings || [])}`,
    ].join('\n');
    providerBox.textContent = JSON.stringify({ providers, providerHealth }, null, 2);
  }
  if (registrySummary && registryBox) {
    registrySummary.textContent = [
      `models=${registries.models?.items?.length || 0}`,
      `policies=${registries.policies?.items?.length || 0}`,
      `features=${registries.features?.items?.length || 0}`,
      `truth=${registries.truth?.items?.length || 0}`,
    ].join('\n');
    registryBox.textContent = JSON.stringify(registries, null, 2);
  }
  if (truthSummary && truthBox) {
    const items = truthVersions?.items || [];
    const latest = items[0] || {};
    truthSummary.textContent = [
      `truth_version 数：${items.length}`,
      `最新版本：${latest.name || '-'}`,
      `更新时间：${latest.updated_at || '-'}`,
    ].join('\n');
    truthBox.textContent = JSON.stringify(truthVersions, null, 2);
  }
  if (objectsSummary && objectsBox) {
    const items = objects?.items || [];
    const top = items[0] || {};
    objectsSummary.textContent = [
      `活动对象：${items.length}`,
      `最新对象：${top.object_id || '-'}`,
      `状态：${top.lifecycle_state || '-'}`,
      `灾种：${HAZARD_LABELS[top.hazard_type] || top.hazard_type || '-'}`,
    ].join('\n');
    objectsBox.textContent = JSON.stringify(objects, null, 2);
  }
  renderTimeline(objectsMiniTimeline, (objects?.items || []).slice(0, 6).map((item) => ({
    title: item.object_id || '-',
    value: HAZARD_LABELS[item.hazard_type] || item.hazard_type || '-',
    meta: `${item.lifecycle_state || '-'} | ${Number(item.confidence || 0).toFixed(2)}`,
  })));
  const truthItems = truthVersions?.items || [];
  setSelectOptions(truthVersionLeft, truthItems, (x) => x.name, (x) => x.name);
  setSelectOptions(truthVersionRight, truthItems, (x) => x.name, (x, idx) => idx === 1 ? `${x.name} (compare)` : x.name);
}

async function refreshGovernance() {
  try {
    const [providers, providerHealth, models, policies, features, truth, truthVersions, objects, recentRuns, bundles] = await Promise.all([
      getJson('/api/v1/ingest/providers'),
      getJson('/api/v1/ingest/health'),
      getJson('/api/v1/registry/models'),
      getJson('/api/v1/registry/policies'),
      getJson('/api/v1/registry/features'),
      getJson('/api/v1/registry/truth'),
      getJson('/api/v1/truth/versions'),
      getJson('/api/v1/objects/active'),
      getJson('/api/v1/registry/predict/recent?limit=10'),
      getJson('/api/v1/replay/bundles?limit=10'),
    ]);
    setSelectOptions(runCompareLeft, recentRuns.data?.items || [], (x) => x.run_id, (x) => `${x.run_id} | ${x.action || '-'} | ${x.level || '-'}`);
    setSelectOptions(runCompareRight, recentRuns.data?.items || [], (x) => x.run_id, (x, idx) => `${x.run_id}${idx === 1 ? ' (compare)' : ''}`);
    setSelectOptions(bundleCompareLeft, bundles.data?.items || [], (x) => x.bundle_id, (x) => `${x.bundle_id} | ${x.action || '-'} | ${x.level || '-'}`);
    setSelectOptions(bundleCompareRight, bundles.data?.items || [], (x) => x.bundle_id, (x, idx) => `${x.bundle_id}${idx === 1 ? ' (compare)' : ''}`);
    renderGovernance({
      providers: providers.data,
      providerHealth: providerHealth.data,
      registries: {
        models: models.data,
        policies: policies.data,
        features: features.data,
        truth: truth.data,
      },
      truthVersions: truthVersions.data,
      objects: objects.data,
    });
  } catch (e) {
    if (registrySummary) registrySummary.textContent = `治理视图加载失败：${String(e)}`;
    if (registryBox) registryBox.textContent = String(e);
  }
}

async function compareTruthVersions() {
  if (!truthVersionLeft?.value || !truthVersionRight?.value) return;
  const detail = await Promise.all([
    getJson(`/api/v1/truth/versions/${encodeURIComponent(truthVersionLeft.value)}`),
    getJson(`/api/v1/truth/versions/${encodeURIComponent(truthVersionRight.value)}`),
    postJson('/api/v1/truth/compare', {
      left_truth_version: truthVersionLeft.value,
      right_truth_version: truthVersionRight.value,
    }),
  ]);
  const [leftDetail, rightDetail, compare] = detail.map((x) => x.data);
  if (truthSummary) {
    truthSummary.textContent = [
      `Truth A：${compare.left.truth_version}`,
      `Truth B：${compare.right.truth_version}`,
      `headline tier 一致：${compare.same_headline_tier ? '是' : '否'}`,
    ].join('\n');
  }
  renderCompareStack(truthCompareChart, [{
    title: 'Truth Version Delta',
    meta: `${compare.left.truth_version} -> ${compare.right.truth_version}`,
    bars: Object.entries(compare.delta || {}).map(([k, v]) => ({
      label: k,
      leftPct: Math.min(100, Number(v.left || 0)),
      rightPct: Math.min(100, Number(v.right || 0)),
      delta: `${v.delta >= 0 ? '+' : ''}${v.delta}`,
    })),
  }]);
  if (truthBox) truthBox.textContent = JSON.stringify(compare, null, 2);
  if (versionDetailSummary) {
    versionDetailSummary.textContent = [
      `Truth A layers: ${JSON.stringify(leftDetail.manifest?.record_counts || {})}`,
      `Truth B layers: ${JSON.stringify(rightDetail.manifest?.record_counts || {})}`,
    ].join('\n');
  }
  if (versionDetailBox) {
    versionDetailBox.textContent = JSON.stringify({ left: leftDetail, right: rightDetail }, null, 2);
  }
}

async function compareRecentRuns() {
  if (!runCompareLeft?.value || !runCompareRight?.value) return;
  const j = await postJson('/api/v1/replay/compare', {
    baseline_run_id: runCompareLeft.value,
    enhanced_run_id: runCompareRight.value,
  });
  const data = j.data || {};
  if (replayCompareSummary) {
    replayCompareSummary.textContent = [
      `Run A：${data.baseline_run_id}`,
      `Run B：${data.enhanced_run_id}`,
      `issue_changed=${data.issue_changed}，level_changed=${data.level_changed}`,
      `readiness_delta=${Number(data.readiness_delta || 0).toFixed(4)}`,
    ].join('\n');
  }
  renderCompareStack(replayCompareChart, [{
    title: 'Run Hazard Probability Delta',
    meta: 'baseline vs enhanced',
    bars: Object.entries(data.hazard_prob_diff || {}).map(([hazard, row]) => ({
      label: HAZARD_LABELS[hazard] || hazard,
      leftPct: Number(row.baseline || 0) * 100,
      rightPct: Number(row.enhanced || 0) * 100,
      delta: `${(Number(row.delta || 0) * 100).toFixed(1)}%`,
    })),
  }]);
  if (replayCompareBox) replayCompareBox.textContent = JSON.stringify(data, null, 2);
}

async function compareReplayBundles() {
  if (!bundleCompareLeft?.value || !bundleCompareRight?.value) return;
  const j = await postJson('/api/v1/replay/compare-bundles', {
    left_bundle_id: bundleCompareLeft.value,
    right_bundle_id: bundleCompareRight.value,
  });
  const data = j.data || {};
  const comparison = data.comparison || {};
  if (replayCompareSummary) {
    replayCompareSummary.textContent = [
      `Bundle A：${data.left_bundle_id}`,
      `Bundle B：${data.right_bundle_id}`,
      `deterministic=${comparison.deterministic}`,
      `tolerance=${comparison.tolerance}`,
    ].join('\n');
  }
  renderCompareStack(replayCompareChart, [{
    title: 'Replay Bundle Hazard Probability Delta',
    meta: `deterministic=${comparison.deterministic}`,
    bars: Object.entries(comparison.hazard_prob || {}).map(([hazard, row]) => ({
      label: HAZARD_LABELS[hazard] || hazard,
      leftPct: Number(row.left || 0) * 100,
      rightPct: Number(row.right || 0) * 100,
      delta: `${Number(row.delta || 0).toFixed(6)}`,
    })),
  }]);
  if (replayCompareBox) replayCompareBox.textContent = JSON.stringify(data, null, 2);
  if (versionDetailSummary) versionDetailSummary.textContent = `Replay bundle compare loaded: ${data.left_bundle_id} vs ${data.right_bundle_id}`;
  if (versionDetailBox) versionDetailBox.textContent = JSON.stringify({ left: data.left?.bundle, right: data.right?.bundle }, null, 2);
}

async function executeEvaluateWindow(startDate, endDate, contextText = '评估', options = {}) {
  const forceRebuildTruth = Boolean(options.forceRebuildTruth);
  const j = await postJson('/api/v1/evaluate/recent', {
    city: cityInput.value.trim(),
    days: 3,
    truth_policy: 'require',
    min_truth_coverage: 0.6,
    min_total_positive_labels: AUTO_MIN_TOTAL_POSITIVE,
    start_date: startDate,
    end_date: endDate,
    headline_tiers: ['gold', 'silver'],
    force_rebuild_truth: forceRebuildTruth,
  });
  renderEval(j.data);
  rawBox.textContent = JSON.stringify(j, null, 2);
  rawSummary.textContent = `${contextText}完成，request_id=${j.request_id}`;
  return j;
}

async function executeEvolveWindow(startDate, endDate, contextText = '演进', options = {}) {
  const forceRebuildTruth = Boolean(options.forceRebuildTruth);
  const j = await postJson('/api/v1/evolve/weights', {
    city: cityInput.value.trim(),
    days: 5,
    truth_policy: 'require',
    min_truth_coverage: 0.6,
    min_total_positive_labels: AUTO_MIN_TOTAL_POSITIVE,
    min_train_samples: AUTO_MIN_TRAIN_SAMPLES,
    min_calibration_samples: AUTO_MIN_CALIBRATION_SAMPLES,
    start_date: startDate,
    end_date: endDate,
    headline_tiers: ['gold', 'silver'],
    calibrator_method: 'histogram',
    force_rebuild_truth: forceRebuildTruth,
  });
  renderEvolve(j.data);
  rawBox.textContent = JSON.stringify(j, null, 2);
  rawSummary.textContent = `${contextText}完成，request_id=${j.request_id}`;
  return j;
}

async function scanBestWindows() {
  const searchStart = truthStartInput?.value;
  const searchEnd = truthEndInput?.value;
  if (!searchStart || !searchEnd) {
    throw new Error('请先填写真值开始和结束日期，作为历史搜索范围');
  }
  const payload = {
    city: cityInput?.value?.trim?.() || 'Tianjin',
    search_start: searchStart,
    search_end: searchEnd,
    window_days: 3,
    step_days: 1,
    min_truth_coverage: 0.6,
    min_total_positive_labels: AUTO_MIN_TOTAL_POSITIVE,
    min_train_positive_labels: 1,
    min_calibration_positive_labels: 1,
    headline_tiers: ['gold', 'silver'],
    top_k: 10,
    force_rebuild_truth: true,
  };
  const result = await postJsonFirstOk(
    ['/api/v1/windows/scan', '/api/v1/window/scan', '/api/v1/windows_scan', '/api/v1/scan/windows'],
    payload,
  );
  const j = result.data;
  return { ...j.data, _apiPathUsed: result.url };
}

function uniqueWindows(rows) {
  const out = [];
  const seen = new Set();
  (rows || []).forEach((r) => {
    const start = r?.start_date;
    const end = r?.end_date;
    if (!start || !end) return;
    const key = `${start}__${end}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(r);
  });
  return out;
}

async function tryExecuteAcrossWindows(candidates, executor) {
  const errors = [];
  for (const c of uniqueWindows(candidates)) {
    try {
      await executor(c.start_date, c.end_date);
      return { ok: true, window: c, errors };
    } catch (e) {
      errors.push({ start_date: c.start_date, end_date: c.end_date, error: String(e) });
    }
  }
  return { ok: false, window: null, errors };
}

if (btnLocate) {
  btnLocate.addEventListener('click', locateCity);
}
if (cityInput && btnLocate) {
  cityInput.addEventListener('input', () => {
    clearTimeout(cityDebounceTimer);
    cityDebounceTimer = setTimeout(() => locateCity(), 900);
  });
}

if (btnLive) btnLive.addEventListener('click', async () => {
  setButtonLoading(btnLive, true);
  setBusy(true, '实时推理中（模式/雷达/客观概率）...');
  try {
    const j = await postJson('/api/v1/forecast/live', {
      city: cityInput.value.trim(),
      area: areaInput.value.trim(),
      auto_area: true,
      min_issue_prob: Number(minProbInput.value || 0.55),
      save_run: true,
    });
    renderLive(j.data, j);
    await loadAudit(j.data.run_id);
    await loadTrace(j.request_id);
  } catch (e) {
    rawSummary.textContent = `请求失败：${String(e)}`;
    rawBox.textContent = String(e);
  } finally {
    setBusy(false);
    setButtonLoading(btnLive, false);
  }
});

if (btnEval) btnEval.addEventListener('click', async () => {
  setButtonLoading(btnEval, true);
  setBusy(true, '评估中（时间切分 + 业务指标 + CI）...');
  try {
    const hasTruthWindow = Boolean(truthStartInput.value && truthEndInput.value);
    if (!hasTruthWindow) {
      throw new Error('请先设置真值时间范围，或使用“智能选窗并执行”');
    }
    await executeEvaluateWindow(truthStartInput.value, truthEndInput.value, '评估', { forceRebuildTruth: false });
  } catch (e) {
    evalSummary.textContent = `评估失败：${String(e)}`;
    evalBox.textContent = String(e);
  } finally {
    setBusy(false);
    setButtonLoading(btnEval, false);
  }
});

if (btnEvolve) btnEvolve.addEventListener('click', async () => {
  setButtonLoading(btnEvolve, true);
  setBusy(true, '演进中（训练/校准/回退机制）...');
  try {
    const hasTruthWindow = Boolean(truthStartInput.value && truthEndInput.value);
    if (!hasTruthWindow) {
      throw new Error('请先设置真值时间范围，或使用“智能选窗并执行”');
    }
    await executeEvolveWindow(truthStartInput.value, truthEndInput.value, '演进', { forceRebuildTruth: false });
  } catch (e) {
    evolveSummary.textContent = `演进失败：${String(e)}`;
    weightBox.textContent = String(e);
  } finally {
    setBusy(false);
    setButtonLoading(btnEvolve, false);
  }
});

if (btnAutoWindow) btnAutoWindow.addEventListener('click', async () => {
  setButtonLoading(btnAutoWindow, true);
  setBusy(true, '智能选窗中（扫描历史窗口并自动执行评估/演进）...');
  try {
    const scan = await scanBestWindows();
    const summary = scan?.result?.summary || {};
    const bestEval = summary.best_window;
    const bestEvolve = summary.best_window_evolve || summary.best_window;
    const evalCandidates = uniqueWindows([bestEval, ...(scan?.result?.top_passed_windows || [])]);
    const evolveCandidates = uniqueWindows([bestEvolve, ...(scan?.result?.top_passed_evolve_windows || [])]);

    if (!evalCandidates.length && !evolveCandidates.length) {
      throw new Error('未找到可执行窗口，请扩大日期范围或降低门槛');
    }

    const evalRun = await tryExecuteAcrossWindows(evalCandidates, async (s, e) => {
      truthStartInput.value = s;
      truthEndInput.value = e;
      await executeEvaluateWindow(s, e, '自动选窗评估', { forceRebuildTruth: false });
    });

    const evolveRun = await tryExecuteAcrossWindows(evolveCandidates, async (s, e) => {
      truthStartInput.value = s;
      truthEndInput.value = e;
      await executeEvolveWindow(s, e, '自动选窗演进', { forceRebuildTruth: false });
    });

    if (!evalRun.ok && !evolveRun.ok) {
      const evalErr = evalRun.errors?.[0]?.error || '无可用评估窗口';
      const evolveErr = evolveRun.errors?.[0]?.error || '无可用演进窗口';
      throw new Error(`评估与演进均失败：评估=${evalErr}；演进=${evolveErr}`);
    }

    if (autoWindowSummary) autoWindowSummary.textContent = [
      `选窗完成：扫描=${summary.window_count_scanned || 0}，可评估=${summary.window_count_passed || 0}，可演进=${summary.window_count_passed_evolve || 0}`,
      `评估窗口：${evalRun.ok ? `${evalRun.window.start_date} ~ ${evalRun.window.end_date}` : '执行失败'}`,
      `演进窗口：${evolveRun.ok ? `${evolveRun.window.start_date} ~ ${evolveRun.window.end_date}` : '执行失败'}`,
      evalRun.ok ? '评估执行：成功' : `评估执行：失败（尝试${evalRun.errors.length}个候选）`,
      evolveRun.ok ? '演进执行：成功' : `演进执行：失败（尝试${evolveRun.errors.length}个候选）`,
      evalRun.errors.length ? `评估回退：前${evalRun.errors.length}个候选失败后已切换` : '评估回退：未触发',
      evolveRun.errors.length ? `演进回退：前${evolveRun.errors.length}个候选失败后已切换` : '演进回退：未触发',
      `扫描接口：${scan?._apiPathUsed || '/api/v1/windows/scan'}`,
      `扫描报告：${JSON.stringify(scan?.reports || {})}`,
    ].join('\n');
  } catch (e) {
    if (autoWindowSummary) autoWindowSummary.textContent = `智能选窗失败：${String(e)}`;
  } finally {
    setBusy(false);
    setButtonLoading(btnAutoWindow, false);
  }
});

if (btnBuildTruth) btnBuildTruth.addEventListener('click', async () => {
  setButtonLoading(btnBuildTruth, true);
  setBusy(true, '构建真值标签中（NOAA站点+冰雹报告）...');
  try {
    const j = await postJson('/api/v1/truth/build', {
      city: cityInput?.value?.trim?.() || 'Tianjin',
      start_date: truthStartInput?.value,
      end_date: truthEndInput?.value,
      force_rebuild: true,
    });
    const m = j.data?.meta || {};
    evalSummary.textContent = [
      `真值构建完成：${m.city_resolved || cityInput.value}`,
      `覆盖：${m.label_hours || 0} 小时，${asPct(m.label_coverage_ratio || 0)}`,
      `tier：${JSON.stringify(m.label_tiering?.tier_counts || {})}`,
      `truth_version：${j.data?.truth_version?.truth_version || '-'}`,
    ].join('\n');
    evalBox.textContent = JSON.stringify(j.data, null, 2);
    await refreshGovernance();
  } catch (e) {
    evalSummary.textContent = `真值构建失败：${String(e)}`;
    evalBox.textContent = String(e);
  } finally {
    setBusy(false);
    setButtonLoading(btnBuildTruth, false);
  }
});

if (btnRefreshGovernance) btnRefreshGovernance.addEventListener('click', async () => {
  setButtonLoading(btnRefreshGovernance, true);
  setBusy(true, '刷新治理视图中...');
  try {
    await refreshGovernance();
  } finally {
    setBusy(false);
    setButtonLoading(btnRefreshGovernance, false);
  }
});
if (compareTruthVersionsBtn) compareTruthVersionsBtn.addEventListener('click', async () => {
  setButtonLoading(compareTruthVersionsBtn, true);
  setBusy(true, '对比真值版本中...');
  try {
    await compareTruthVersions();
  } finally {
    setBusy(false);
    setButtonLoading(compareTruthVersionsBtn, false);
  }
});
if (compareRunsBtn) compareRunsBtn.addEventListener('click', async () => {
  setButtonLoading(compareRunsBtn, true);
  setBusy(true, '对比运行结果中...');
  try {
    await compareRecentRuns();
  } finally {
    setBusy(false);
    setButtonLoading(compareRunsBtn, false);
  }
});
if (compareBundlesBtn) compareBundlesBtn.addEventListener('click', async () => {
  setButtonLoading(compareBundlesBtn, true);
  setBusy(true, '对比回放包中...');
  try {
    await compareReplayBundles();
  } finally {
    setBusy(false);
    setButtonLoading(compareBundlesBtn, false);
  }
});
if (btnRefreshLLM) btnRefreshLLM.addEventListener('click', async () => {
  setButtonLoading(btnRefreshLLM, true);
  setBusy(true, '刷新 LLM 运行时状态中...');
  if (llmStatusBadge) llmStatusBadge.textContent = 'refreshing...';
  if (llmSummary) llmSummary.textContent = '正在刷新 LLM 状态，请稍候...';
  try {
    await refreshLLMRuntime();
  } finally {
    setBusy(false);
    setButtonLoading(btnRefreshLLM, false);
  }
});
if (btnApplyLLM) btnApplyLLM.addEventListener('click', async () => {
  setButtonLoading(btnApplyLLM, true);
  setBusy(true, '应用 LLM 运行时配置中...');
  if (llmStatusBadge) llmStatusBadge.textContent = 'applying...';
  if (llmSummary) llmSummary.textContent = '正在应用 LLM 配置，请稍候...';
  try {
    await applyLLMRuntime();
  } catch (e) {
    if (llmSummary) llmSummary.textContent = `LLM 配置失败：${String(e)}`;
    if (llmBox) llmBox.textContent = String(e);
  } finally {
    setBusy(false);
    setButtonLoading(btnApplyLLM, false);
  }
});

initMap();
checkHealth();
if (btnLocate && map) locateCity();
if (btnRefreshGovernance || providerSummary || registrySummary || truthSummary || objectsSummary) {
  refreshGovernance();
}
if (llmSummary || llmBox || llmStatusBadge) {
  refreshLLMRuntime();
}

if (radarToggle) {
  radarToggle.addEventListener('change', () => applyRadarLayer());
}
