
'use strict';
const D = window.SITE_DATA || { summary: { models: {}, counts: {} }, scores: [], duels: [], labels: [], golden_sets: [], leaderboards: [], about: '', generated_at: '' };

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = v => v == null ? '—' : (v * 100).toFixed(0) + '%';
const bar = v => v == null ? '—' : `<span class="bar"><div style="width:${(v*100).toFixed(0)}%"></div></span>`;
const labelCls = l => l === 'good' ? 'good' : l === 'bad' ? 'bad' : 'dim';
const scoreCls = s => s >= 4 ? 'good' : s <= 2 ? 'bad' : 'mid';

document.getElementById('generated').textContent = (D.generated_at || '').replace('T', ' ').replace(/\.\d+Z$/, '').replace('Z', ' UTC');

// ── Visual helpers (inline SVG / CSS bars; no dependencies) ──────────────
const PAL = ['#ffb547', '#3ddbd0', '#ff6b9a', '#a594ff', '#8be072', '#ff8a4c', '#5fb4ff', '#e6e06a'];
const ALLM = [...new Set([...Object.keys(D.summary.models || {}), ...(D.runs || []).map(r => r.model)].filter(Boolean))].sort();
const mc = m => PAL[Math.max(0, ALLM.indexOf(m)) % PAL.length];
const shortM = m => String(m).replace(/[-.]?(UD-)?Q\d_[A-Z0-9_]+$/i, '').replace(/^microsoft_/, '').replace(/^Meta-/, '');
const dot = c => `<i class="dot" style="background:${c}"></i>`;
const okRun = r => r.status === 'ok' && (r.n_recs ?? 0) > 0;
const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
function latestGood() {
  const m = {};
  for (const r of D.runs || []) { if (!okRun(r)) continue; const k = `${r.model}|${r.agent || ''}|${r.profile}`; if (!m[k] || r.run > m[k].run) m[k] = r; }
  return Object.values(m);
}
function perModel(rows, key) {   // mean of `key` per model → [{m, v}] sorted high→low
  const by = {};
  for (const r of rows) if (r[key] != null) (by[r.model] = by[r.model] || []).push(r[key]);
  return Object.entries(by).map(([m, v]) => ({ m, v: avg(v) })).sort((a, b) => b.v - a.v);
}
function hbars(items, fmt, lo = 0) {
  if (!items.length) return '<p class="empty">No data yet.</p>';
  const hi = Math.max(...items.map(i => i.v)), span = (hi - lo) || 1;
  return `<div class="hb">${items.map(i => { const c = i.c || mc(i.m);
    return `<div class="hr"><span class="hl" title="${esc(i.m)}">${dot(c)}${esc(i.label || shortM(i.m))}</span>
      <span class="ht"><span style="width:${Math.max(3, (i.v - lo) / span * 100).toFixed(1)}%;background:${c}"></span></span><b>${fmt(i.v)}</b></div>`; }).join('')}</div>`;
}
function plot(series, o) {   // series: [{m, pts:[{x,y,t}]}]; o: {line, xf, yf}
  const W = 640, H = 240, L = 46, R = 14, T = 12, B = 30, pts = series.flatMap(s => s.pts);
  if (!pts.length) return '<p class="empty">Not enough data yet.</p>';
  const rng = k => { let a = Math.min(...pts.map(p => p[k])), b = Math.max(...pts.map(p => p[k])); if (a === b) { a -= 1; b += 1; } const d = (b - a) * .08; return [a - d, b + d]; };
  const [x0, x1] = rng('x'), [y0, y1] = rng('y');
  const X = v => L + (v - x0) / (x1 - x0) * (W - L - R), Y = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  let g = '';
  for (let i = 0; i <= 4; i++) { const v = y0 + (y1 - y0) * i / 4, y = Y(v).toFixed(1);
    g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text x="${L - 6}" y="${+y + 3}" text-anchor="end">${(o.yf || (q => q.toFixed(2)))(v)}</text>`; }
  for (let i = 0; i <= 3; i++) { const v = x0 + (x1 - x0) * i / 3; g += `<text x="${X(v).toFixed(1)}" y="${H - 10}" text-anchor="middle">${o.xf(v)}</text>`; }
  const body = series.map(s => { const c = mc(s.m);
    const pl = o.line && s.pts.length > 1 ? `<polyline fill="none" stroke="${c}" stroke-width="2" points="${s.pts.map(p => X(p.x).toFixed(1) + ',' + Y(p.y).toFixed(1)).join(' ')}"/>` : '';
    return pl + s.pts.map(p => `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="${o.line ? 3 : 6}" fill="${c}" fill-opacity=".85"><title>${esc(s.m)}: ${esc(p.t)}</title></circle>`).join(''); }).join('');
  return `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img">${g}${body}</svg><div class="lg">${series.map(s => `<span>${dot(mc(s.m))}${esc(shortM(s.m))}</span>`).join('')}</div>`;
}
function scoreDist() {
  const by = {}, cols = ['', '#f06a6a', '#f2a05c', '#f2d35c', '#a3d86b', '#4fd18b'];
  for (const r of D.scores) if (typeof r.score === 'number' && r.score >= 1 && r.score <= 5) (by[r.model] = by[r.model] || [0, 0, 0, 0, 0, 0])[r.score]++;
  const ms = Object.keys(by).sort();
  if (!ms.length) return '<p class="empty">No scores yet.</p>';
  return `<div class="hb">${ms.map(m => { const d = by[m], n = d.reduce((a, b) => a + b, 0), a = (d.reduce((s, v, k) => s + v * k, 0) / n).toFixed(2);
    return `<div class="hr"><span class="hl">${dot(mc(m))}${esc(shortM(m))}</span><span class="ht">${[1, 2, 3, 4, 5].map(k =>
      `<span style="width:${d[k] / n * 100}%;background:${cols[k]}" title="${k} of 5: ${d[k]}"></span>`).join('')}</span><b>${a}</b></div>`; }).join('')}</div>
    <div class="lg"><span>red = 1 → green = 5, bar width = share of ratings</span></div>`;
}
function runCharts(rows) {
  const runs = [...new Set(D.runs.filter(okRun).map(r => r.run))].sort(), by = {};
  for (const r of D.runs) { if (!okRun(r) || r.quality == null) continue; ((by[r.model] = by[r.model] || {})[r.run] = by[r.model][r.run] || []).push(r.quality); }
  const trend = Object.entries(by).map(([m, o]) => ({ m, pts: Object.entries(o).map(([run, v]) => ({ x: runs.indexOf(run), y: avg(v), t: `${runFmt(run)} → ${avg(v).toFixed(3)}` })).sort((a, b) => a.x - b.x) }));
  const sc = {};
  for (const r of latestGood()) if (r.quality != null && r.wallclock_s != null) (sc[r.model] = sc[r.model] || []).push({ x: r.wallclock_s, y: r.quality, t: `${r.agent || ''} / ${r.profile}: quality ${r.quality.toFixed(3)} in ${r.wallclock_s.toFixed(0)}s` });
  return `<div class="grid2"><div class="panel"><h3>Quality across runs</h3>${plot(trend, { line: true, xf: v => runFmt(runs[Math.min(runs.length - 1, Math.max(0, Math.round(v)))]) })}</div>
    <div class="panel"><h3>Speed vs quality</h3><p class="dim">Latest good run per model × agent × profile. Up and left is better.</p>${plot(Object.entries(sc).map(([m, pts]) => ({ m, pts })), { xf: v => v.toFixed(0) + 's' })}</div></div>`;
}
function recCharts() {
  const gc = {}, tg = {};
  for (const r of D.recs) {
    const g = (r.self_genre || '').toLowerCase().trim(); if (g) gc[g] = (gc[g] || 0) + 1;
    (tg[r.model] = tg[r.model] || []).push(r.tagged ? 1 : 0);
  }
  const genres = Object.entries(gc).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([g, v]) => ({ m: g, label: g, v, c: '#a594ff' }));
  const cov = Object.entries(tg).map(([m, a]) => ({ m, v: avg(a) })).sort((a, b) => b.v - a.v);
  return `<div class="grid2"><div class="panel"><h3>Most-recommended genres (self-reported)</h3>${hbars(genres, v => v)}</div>
    <div class="panel"><h3>Tag coverage by model</h3><p class="dim">Share of recs whose artist already has a golden-set tag.</p>${hbars(cov, pct)}</div></div>`;
}

// ── Dashboard ────────────────────────────────────────────────────────────
function dashboard() {
  const c = D.summary.counts, M = Object.entries(D.summary.models), lg = latestGood();
  const elo = M.filter(([, s]) => s.elo).sort((a, b) => b[1].elo.elo - a[1].elo.elo);
  const auto = perModel(lg, 'quality');
  let lead = '', ladder = '', facts = '';
  if (elo.length) {
    const [m, s] = elo[0], lo = Math.min(...elo.map(e => e[1].elo.elo)) - 40;
    lead = [m, 'Elo leader']; facts = `<div><b>${s.elo.elo.toFixed(0)}</b><span>Elo</span></div><div><b>${pct(s.elo.win_rate)}</b><span>duel win-rate</span></div>` + (s.avg_score != null ? `<div><b>${s.avg_score}</b><span>avg score of 5</span></div>` : '');
    ladder = `<h3>Elo ladder</h3>${hbars(elo.map(([n, x]) => ({ m: n, v: x.elo.elo })), v => v.toFixed(0), lo)}`;
  } else if (auto.length) {
    lead = [auto[0].m, 'Top automated quality']; facts = `<div><b>${auto[0].v.toFixed(3)}</b><span>avg quality</span></div>`;
    ladder = `<h3>Automated quality</h3>${hbars(auto, v => v.toFixed(3))}`;
  }
  let h = lead ? `<div class="hero"><div><span class="dim">${esc(lead[1])}</span><div class="name">${esc(shortM(lead[0]))}</div><div class="facts">${facts}</div></div><div>${ladder}</div></div>`
    : '<p class="empty">No judgments yet — run <code>make label</code>, <code>make score</code>, <code>make duel</code> or <code>make eval</code>, then <code>make site</code>.</p>';
  // Isolated ELO ladders — model-only and agent-only, alongside the combined one in the hero above.
  const eloPanel = (title, dim, obj) => {
    const items = Object.entries(obj || {}).filter(([, s]) => s && s.elo != null)
      .map(([n, s]) => ({ m: n, v: s.elo, label: n })).sort((a, b) => b.v - a.v);
    if (!items.length) return `<div class="panel"><h3>${title} <span class="dim">${dim}</span></h3><p class="empty">No data yet.</p></div>`;
    const lo = Math.min(...items.map(i => i.v)) - 40;
    return `<div class="panel"><h3>${title} <span class="dim">${dim}</span></h3>${hbars(items, v => v.toFixed(0), lo)}</div>`;
  };
  h += `<div class="grid2">${eloPanel('Model Elo', 'same-agent duels only', D.summary.model_elo)}${eloPanel('Agent Elo', 'same-model duels only', D.summary.agent_elo)}</div>`;
  h += `<div class="strip">${[['labels', 'rec labels'], ['scores', 'response scores'], ['duels', 'duels'], ['golden_sets', 'golden sets'], ['runs', 'automated runs'], ['recs', 'recs in history']]
    .map(([k, l]) => `<div><b>${c[k] ?? 0}</b><span>${l}</span></div>`).join('')}</div>`;
  h += `<div class="grid2"><div class="panel"><h3>How people rated responses</h3>${scoreDist()}</div>
    <div class="panel"><h3>Recs judged good</h3><p class="dim">Good ÷ (good + bad) from your labels.</p>${hbars(M.filter(([, s]) => s.label_good_rate != null).map(([m, s]) => ({ m, v: s.label_good_rate })).sort((a, b) => b.v - a.v), pct)}</div>
    <div class="panel"><h3>Do the recs exist?</h3><p class="dim">Round-1 picks found on Spotify, before any retry.</p>${hbars(perModel(lg, 'raw_existence_rate'), pct)}</div>
    <div class="panel"><h3>Automated quality</h3><p class="dim">Mean across latest good runs.</p>${hbars(auto, v => v.toFixed(3))}</div></div>`;
  if (D.runs.length) h += runCharts();
  if (M.length) {
    h += `<h2>Every model, side by side</h2><table><tr><th>model</th><th class="num">labels</th><th class="num">label good-rate</th>
      <th class="num">scores</th><th class="num">avg score (1–5)</th><th class="num">elo</th><th class="num">win-rate</th></tr>`;
    for (const [m, s] of M.sort((a, b) => (b[1].elo?.elo ?? -1) - (a[1].elo?.elo ?? -1)))
      h += `<tr><td>${dot(mc(m))}${esc(m)}</td><td class="num">${s.labels}</td><td class="num">${bar(s.label_good_rate)} ${pct(s.label_good_rate)}</td>
        <td class="num">${s.scores}</td><td class="num ${s.avg_score == null ? '' : scoreCls(s.avg_score)}">${s.avg_score ?? '—'}</td>
        <td class="num">${s.elo ? s.elo.elo.toFixed(1) : '—'}</td><td class="num">${s.elo ? pct(s.elo.win_rate) : '—'}</td></tr>`;
    h += '</table>';
  }
  if (D.leaderboards.length) h += `<details><summary>Latest automated leaderboard (${esc(D.leaderboards[0].name)})</summary><div class="md">${D.leaderboards[0].html}</div></details>`;
  return h;
}

// ── Scores ───────────────────────────────────────────────────────────────
function scores() {
  if (!D.scores.length) return '<p class="empty">No scores yet — <code>make score PROFILE=… MODELS=…</code></p>';
  let h = '<div class="filters">model: <select id="f-score-model"><option value="">all</option></select></div>';
  const models = [...new Set(D.scores.map(r => r.model))].sort();
  h += `<table id="t-scores"><tr><th>when</th><th>model</th><th>profile</th><th class="num">score</th><th>label</th><th>note</th><th>recs</th></tr>`;
  for (const r of [...D.scores].sort((a, b) => (b.ts || '').localeCompare(a.ts || ''))) {
    h += `<tr data-model="${esc(r.model)}"><td>${esc((r.ts || '').slice(0, 16))}</td><td>${esc(r.model)}</td>
      <td>${esc(r.profile)}</td><td class="num ${scoreCls(r.score)}">${r.score}</td>
      <td class="${scoreCls(r.score)}">${esc(r.score_label)}</td><td class="dim">${esc(r.note)}</td>
      <td>${recsCell(r.recs)}</td></tr>`;
  }
  h += '</table>';
  return h;
}

function recsCell(recs) {
  if (!recs || !recs.length) return '';
  return `<details><summary>${recs.length} recs</summary><ul>${
    recs.map(r => `<li><b>${esc(r.artist)}</b> — ${esc(r.album)}${r.verified ? '' : ' <span class="dim">(unverified)</span>'}</li>`).join('')}</ul></details>`;
}

// ── Duels ────────────────────────────────────────────────────────────────
function eloTable(title, dimLabel, obj) {
  const rows = Object.entries(obj || {}).filter(([, s]) => s && s.elo != null)
    .sort((a, b) => b[1].elo - a[1].elo);
  if (!rows.length) return '';
  let h = `<h2>${title}</h2><table><tr><th>${dimLabel}</th><th class="num">elo</th><th class="num">win-rate</th><th class="num">w–l</th><th class="num">duels</th></tr>`;
  for (const [n, s] of rows)
    h += `<tr><td>${esc(n)}</td><td class="num">${s.elo.toFixed(1)}</td><td class="num">${pct(s.win_rate)}</td><td class="num">${s.wins}–${s.losses}</td><td class="num">${s.duels}</td></tr>`;
  return h + '</table>';
}

function duels() {
  let h = '';
  h += eloTable('Combined ladder <span class="dim">(model + agent — the raw duel verdict)</span>', 'model [agent]', D.summary.combined_elo);
  h += eloTable('Model ladder <span class="dim">(same-agent duels only — the verdict belongs to the model, never the agent)</span>', 'model', D.summary.model_elo);
  h += eloTable('Agent ladder <span class="dim">(same-model duels only — the verdict belongs to the agent, never the model)</span>', 'agent', D.summary.agent_elo);
  h += '<h2>History</h2>';
  if (!D.duels.length) return h + '<p class="empty">No duels yet — <code>make duel PROFILE=… MODELS=a,b</code></p>';
  h += '<table><tr><th>when</th><th>profile</th><th>ranking (best → worst)</th><th>note</th><th>responses</th></tr>';
  for (const r of [...D.duels].sort((a, b) => (b.ts || '').localeCompare(a.ts || ''))) {
    const ranking = (r.ranking || []).map((m, i) => `${'ABC'[i] || i+1}·${m}`).join(' &gt; ');
    const extra = r.unranked?.length ? ` <span class="dim">(unranked: ${esc(r.unranked.join(', '))})</span>` : '';
    const ties = r.ties?.length ? ` <span class="mid">tie: ${esc(r.ties.join(', '))}</span>` : '';
    const responses = r.recs_by_model ? `<details><summary>${Object.keys(r.recs_by_model).length} responses</summary>` +
      Object.entries(r.recs_by_model).map(([m, recs]) => `<p><b>${esc(m)}</b></p><ul>${
        (recs || []).map(x => `<li><b>${esc(x.artist)}</b> — ${esc(x.album)}</li>`).join('')}</ul>`).join('') + '</details>' : '';
    h += `<tr><td>${esc((r.ts || '').slice(0, 16))}</td><td>${esc(r.profile)}</td>
      <td>${esc(ranking)}${extra}${ties}</td><td class="dim">${esc(r.note)}</td><td>${responses}</td></tr>`;
  }
  return h + '</table>';
}

// ── Labels ───────────────────────────────────────────────────────────────
function labels() {
  if (!D.labels.length) return '<p class="empty">No labels yet — <code>make label PROFILE=…</code></p>';
  let h = '<div class="filters">model: <select id="f-label-model"><option value="">all</option></select> '
        + 'profile: <select id="f-label-profile"><option value="">all</option></select></div>';
  h += '<table id="t-labels"><tr><th>when</th><th>model</th><th>profile</th><th>artist</th><th>album</th><th>verdict</th><th>reason</th></tr>';
  for (const r of [...D.labels].sort((a, b) => (b.ts || '').localeCompare(a.ts || ''))) {
    h += `<tr data-model="${esc(r.model)}" data-profile="${esc(r.profile)}">
      <td>${esc((r.ts || '').slice(0, 16))}</td><td>${esc(r.model)}</td><td>${esc(r.profile)}</td>
      <td>${esc(r.artist)}</td><td class="dim">${esc(r.album)}</td>
      <td class="${labelCls(r.label)}">${esc(r.label)}</td><td class="dim">${esc(r.reason || r.note)}</td></tr>`;
  }
  return h + '</table>';
}

// ── Golden sets ──────────────────────────────────────────────────────────
function golden() {
  if (!D.golden_sets.length) return '<p class="empty">No golden sets — add <code>eval/golden/&lt;profile&gt;.json</code>.</p>';
  let h = '';
  for (const g of D.golden_sets) {
    h += `<h2>${esc(g.profile)}</h2><p class="dim">${esc(g.description || '')} <span class="pill">${esc(g.annotated_by || '')}</span></p>`;
    for (const side of ['good', 'bad']) {
      const entries = g[side] || [];
      if (!entries.length) continue;
      h += `<table><tr><th>${side === 'good' ? '✅ good' : '❌ bad'}</th><th>album</th><th>tier</th><th>pair</th><th>why</th></tr>`;
      for (const e of entries) {
        const d = typeof e === 'string' ? { artist: e } : e;
        h += `<tr><td class="${side === 'good' ? 'good' : 'bad'}">${esc(d.artist)}</td>
          <td class="dim">${esc(d.album || '')}</td><td>${esc(d.tier || 'expected')}</td>
          <td>${esc(d.pair || '')}</td><td class="dim">${esc(d.reason || d.note || '')}</td></tr>`;
      }
      h += '</table>';
    }
  }
  return h;
}

// ── Runs (automated evals, merged across every runs_*.tsv) ─────────────
let runSort = { key: 'run', dir: -1 };
const runFmt = r => (r || '').length >= 13
  ? `${r.slice(4,6)}-${r.slice(6,8)} ${r.slice(9,11)}:${r.slice(11,13)}` : (r || '—');

function runSortRows(rows) {
  const { key, dir } = runSort;
  const num = key !== 'run' && key !== 'model' && key !== 'agent' && key !== 'profile';
  return [...rows].sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;            // nulls always last
    if (bv == null) return -1;
    return (num ? av - bv : String(av).localeCompare(String(bv))) * dir;
  });
}

function runRow(r) {
  const r1 = (r.raw_candidates != null && r.raw_candidates > 0)
    ? `${r.round1_found ?? 0}/${r.raw_candidates} (${pct(r.raw_existence_rate)})` : '—';
  // a run that 'succeeded' but produced 0 recs is an infra hiccup, not a result
  const broken = r.status !== 'ok' || !r.n_recs;
  const st = r.status !== 'ok'
    ? `<span class="bad">✗ ${esc(r.status || 'failed')}</span>`
    : !r.n_recs ? '<span class="bad">✗ no recs (broken run)</span>'
    : (r.untrusted ? '<span class="bad">⚠️ UNTRUSTED</span>' : '<span class="good">ok</span>');
  const stale = r._stale ? ' <span class="dim">← newest run broken, showing last good one</span>' : '';
  return `<tr data-model="${esc(r.model || '')}" data-profile="${esc(r.profile || '')}"
    class="${broken ? 'failed ' : ''}${r.untrusted ? 'untrusted' : ''}">
    <td>${runFmt(r.run)}</td><td>${esc(r.model)}</td><td class="dim">${esc(r.agent || '—')}</td>
    <td>${esc(r.profile || '—')}</td><td class="num">${r.n_recs ?? '—'}</td>
    <td class="num">${r1}</td><td class="num">${r.retry_rescued ?? '—'}</td>
    <td class="num">${r.quality != null ? r.quality.toFixed(3) : '—'}</td>
    <td>${st}${stale}</td><td class="num dim">${r.wallclock_s != null ? r.wallclock_s.toFixed(0) : '—'}</td></tr>`;
}

function runHead(label, key, cls = '') {
  const arrow = runSort.key === key ? (runSort.dir === -1 ? ' ↓' : ' ↑') : '';
  return `<th class="sortable ${cls}" data-key="${key}">${label}${arrow}</th>`;
}

function runTable(rows) {
  return `<table id="t-runs"><tr>${runHead('when', 'run')}${runHead('model', 'model')}${runHead('agent', 'agent')}${runHead('profile', 'profile')}
    ${runHead('recs', 'n_recs', 'num')}${runHead('real r1 (Spotify)', 'raw_existence_rate', 'num')}
    ${runHead('rescued', 'retry_rescued', 'num')}${runHead('quality', 'quality', 'num')}
    <th>status</th>${runHead('s', 'wallclock_s', 'num')}</tr>
    ${rows.map(runRow).join('')}</table>`;
}

function runs() {
  if (!D.runs.length) return '<p class="empty">No automated runs yet — <code>make eval</code>.</p>';
  // Current standings = latest *valid* run per (model × agent × profile).
  // A valid run is status=ok AND produced recs — broken runs (llama-server
  // hiccups, empty responses) never stand in for a model, so a bad rerun
  // cannot pollute the picture. If a pairing has no valid run at all we
  // show its newest broken run so the problem is still visible.
  const valid = r => r.status === 'ok' && (r.n_recs ?? 0) > 0;
  const latestGood = {}, latestAny = {};
  for (const r of D.runs) {
    const k = `${r.model}‖${r.agent || ''}‖${r.profile}`;
    if (!latestAny[k] || (r.run || '') > (latestAny[k].run || '')) latestAny[k] = r;
    if (valid(r) && (!latestGood[k] || (r.run || '') > (latestGood[k].run || ''))) latestGood[k] = r;
  }
  const standings = Object.keys(latestAny).map(k => ({
    ...(latestGood[k] || latestAny[k]),
    _stale: !!(latestGood[k] && latestAny[k] !== latestGood[k]),
  }));
  let h = runCharts() + `<h2>Current standings — latest good run per model × agent × profile</h2>
    <p class="dim">Every <code>make eval</code> command lands here, so separate runs (needed for big models) merge into one table. Broken runs (0 recs) are skipped here — they're still visible in the history below.</p>`;
  h += runTable(runSortRows(standings));
  h += `<h2>All runs (history)</h2>
    <div class="filters">model: <select id="f-run-model"><option value="">all</option></select>
      profile: <select id="f-run-profile"><option value="">all</option></select></div>`;
  h += runTable(runSortRows(D.runs));
  h += `<p class="dim">Click a column header to sort. Dimmed rows are broken (failed or 0 recs); red rows are UNTRUSTED (≥75% of round-1 recs don't exist on Spotify).</p>`;
  return h;
}

// ── Leaderboards ─────────────────────────────────────────────────────────
function leaderboards() {
  if (!D.leaderboards.length) return '<p class="empty">No leaderboards yet — <code>make eval</code>.</p>';
  let h = '';
  for (const lb of D.leaderboards) h += `<h2>${esc(lb.name)}</h2><div class="md">${lb.html}</div>`;
  return h;
}

// ── Tabs + filters ───────────────────────────────────────────────────────
function about() {
  return D.about || '<p class="empty">No reference info.</p>';
}

// ── Recs (self-reported genre/era from the append-only run history) ─────
function recs() {
  if (!D.recs.length) return '<p class="empty">No rec history yet — <code>make eval</code> appends to <code>eval/output/runs_history.jsonl</code>.</p>';
  let h = '<p class="dim">Genre/era are the model\u2019s OWN read of each pick (self-reported) — displayed to keep untagged recs legible, never scored on. ' +
          '"tagged" = the artist already has a human-curated entry in a golden set\u2019s <code>tags</code>.</p>';
  h += recCharts();
  h += '<div class="filters">model: <select id="f-rec-model"><option value="">all</option></select> '
     + 'profile: <select id="f-rec-profile"><option value="">all</option></select> '
     + 'tags: <select id="f-rec-tag"><option value="">all</option><option value="untagged">untagged only</option><option value="tagged">tagged only</option></select></div>';
  h += '<table id="t-recs"><tr><th>when</th><th>model</th><th>profile</th><th>artist</th><th>album</th><th>genre (self)</th><th>era (self)</th><th>year</th><th>tag</th></tr>';
  for (const r of [...D.recs].reverse()) {
    const yr = r.release_year || '';
    h += `<tr data-model="${esc(r.model)}" data-profile="${esc(r.profile)}" data-tag="${r.tagged ? 'tagged' : 'untagged'}">
      <td>${runFmt(r.run)}</td><td>${esc(r.model)}</td><td class="dim">${esc(r.profile)}</td>
      <td><b>${esc(r.artist)}</b></td><td class="dim">${esc(r.album)}</td>
      <td>${r.self_genre ? esc(r.self_genre) : '<span class="dim">—</span>'}</td>
      <td>${r.self_era ? esc(r.self_era) : '<span class="dim">—</span>'}</td>
      <td class="num">${yr}</td>
      <td>${r.tagged ? '<span class="good">tagged</span>' : '<span class="mid">untagged</span>'}</td></tr>`;
  }
  h += '</table>';
  return h;
}

const renderers = { dashboard, runs, scores, duels, labels, golden, recs, leaderboards, about };
document.querySelectorAll('#tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#tabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    const id = btn.dataset.tab;
    const el = document.getElementById('tab-' + id);
    el.classList.add('active');
    el.innerHTML = renderers[id]();
    wireFilters(id);
  });
});
document.getElementById('tab-dashboard').innerHTML = dashboard();

// click-to-sort on the Runs tab
function wireSort(tab) {
  document.querySelectorAll(`#tab-${tab} th.sortable`).forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      if (runSort.key === key) runSort.dir *= -1;
      else runSort = { key, dir: -1 };
      const el = document.getElementById('tab-' + tab);
      el.innerHTML = renderers[tab]();
      wireFilters(tab);
    });
  });
}

function wireFilters(tab) {
  const specs = {
    scores: [{ sel: 'f-score-model', field: 'model', rows: () => D.scores }],
    labels: [{ sel: 'f-label-model', field: 'model', rows: () => D.labels },
             { sel: 'f-label-profile', field: 'profile', rows: () => D.labels }],
    runs:   [{ sel: 'f-run-model', field: 'model', rows: () => D.runs },
             { sel: 'f-run-profile', field: 'profile', rows: () => D.runs }],
    recs:   [{ sel: 'f-rec-model', field: 'model', rows: () => D.recs },
             { sel: 'f-rec-profile', field: 'profile', rows: () => D.recs },
             { sel: 'f-rec-tag', field: 'tag', rows: () => D.recs.map(r => ({ ...r, tag: r.tagged ? 'tagged' : 'untagged' })) }],
  }[tab];
  if (!specs) return;
  if (tab === 'runs') wireSort('runs');
  for (const s of specs) {
    const sel = document.getElementById(s.sel);
    if (!sel) continue;
    const values = [...new Set(s.rows().map(r => r[s.field]).filter(Boolean))].sort();
    for (const v of values) { const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); }
    sel.addEventListener('change', apply);
  }
  function apply() {
    const vals = {};
    for (const s of specs) vals[s.sel] = document.getElementById(s.sel)?.value || '';
    document.querySelectorAll(`#tab-${tab} tr[data-model]`).forEach(tr => {
      let show = true;
      for (const s of specs) if (vals[s.sel] && tr.dataset[s.field] !== vals[s.sel]) { show = false; break; }
      tr.style.display = show ? '' : 'none';
    });
  }
}
