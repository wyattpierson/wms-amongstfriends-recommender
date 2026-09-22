
'use strict';
const D = window.SITE_DATA || { summary: { models: {}, counts: {} }, scores: [], duels: [], labels: [], golden_sets: [], leaderboards: [], generated_at: '' };

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = v => v == null ? '—' : (v * 100).toFixed(0) + '%';
const bar = v => v == null ? '—' : `<span class="bar"><div style="width:${(v*100).toFixed(0)}%"></div></span>`;
const labelCls = l => l === 'good' ? 'good' : l === 'bad' ? 'bad' : 'dim';
const scoreCls = s => s >= 4 ? 'good' : s <= 2 ? 'bad' : 'mid';

document.getElementById('generated').textContent = (D.generated_at || '').replace('T', ' ').replace(/\.\d+Z$/, '').replace('Z', ' UTC');

// ── Dashboard ────────────────────────────────────────────────────────────
function dashboard() {
  const c = D.summary.counts, rows = Object.entries(D.summary.models);
  let h = `<h2>Corpus so far</h2><div class="cards">
    <div class="card"><div class="n">${c.labels}</div><div class="l">rec labels</div></div>
    <div class="card"><div class="n">${c.scores}</div><div class="l">response scores</div></div>
    <div class="card"><div class="n">${c.duels}</div><div class="l">duels</div></div>
    <div class="card"><div class="n">${c.golden_sets}</div><div class="l">golden sets</div></div></div>`;
  h += `<h2>Models</h2>`;
  if (!rows.length) h += `<p class="empty">No human judgments yet — run <code>make label</code>, <code>make score</code> or <code>make duel</code>, then <code>make site</code>.</p>`;
  else {
    h += `<table><tr><th>model</th><th class="num">labels</th><th class="num">label good-rate</th>
          <th class="num">scores</th><th class="num">avg score (1–5)</th><th class="num">elo</th><th class="num">win-rate</th></tr>`;
    for (const [m, s] of rows.sort((a, b) => (b[1].elo?.elo ?? -1) - (a[1].elo?.elo ?? -1))) {
      h += `<tr><td>${esc(m)}</td><td class="num">${s.labels}</td>
        <td class="num">${bar(s.label_good_rate)} ${pct(s.label_good_rate)}</td>
        <td class="num">${s.scores}</td>
        <td class="num ${s.avg_score == null ? '' : scoreCls(s.avg_score)}">${s.avg_score ?? '—'}</td>
        <td class="num">${s.elo ? s.elo.elo.toFixed(1) : '—'}</td>
        <td class="num">${s.elo ? pct(s.elo.win_rate) : '—'}</td></tr>`;
    }
    h += '</table>';
  }
  if (D.leaderboards.length) {
    h += `<h2>Latest automated leaderboard</h2><div class="md">${D.leaderboards[0].html}</div>`;
  }
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
function duels() {
  const elo = Object.entries(D.summary.models).filter(([, s]) => s.elo).map(([m, s]) => [m, s.elo]);
  let h = '';
  if (elo.length) {
    h += '<h2>Standings</h2><table><tr><th>model</th><th class="num">elo</th><th class="num">win-rate</th><th class="num">w–l</th><th class="num">duels</th></tr>';
    for (const [m, s] of elo.sort((a, b) => b[1].elo - a[1].elo))
      h += `<tr><td>${esc(m)}</td><td class="num">${s.elo.toFixed(1)}</td><td class="num">${pct(s.win_rate)}</td>
            <td class="num">${s.wins}–${s.losses}</td><td class="num">${s.duels}</td></tr>`;
    h += '</table>';
  }
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

// ── Leaderboards ─────────────────────────────────────────────────────────
function leaderboards() {
  if (!D.leaderboards.length) return '<p class="empty">No leaderboards yet — <code>make eval</code>.</p>';
  let h = '';
  for (const lb of D.leaderboards) h += `<h2>${esc(lb.name)}</h2><div class="md">${lb.html}</div>`;
  return h;
}

// ── Tabs + filters ───────────────────────────────────────────────────────
const renderers = { dashboard, scores, duels, labels, golden, leaderboards };
document.querySelectorAll('#tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#tabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    const id = btn.dataset.tab;
    const el = document.getElementById('tab-' + id);
    el.innerHTML = renderers[id]();
    wireFilters(id);
  });
});
document.getElementById('tab-dashboard').innerHTML = dashboard();

function wireFilters(tab) {
  const fill = (selId, values) => {
    const sel = document.getElementById(selId);
    if (!sel) return;
    for (const v of values) { const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); }
    sel.addEventListener('change', apply);
  };
  if (tab === 'scores') {
    fill('f-score-model', [...new Set(D.scores.map(r => r.model))].sort());
  }
  if (tab === 'labels') {
    fill('f-label-model', [...new Set(D.labels.map(r => r.model))].sort());
    fill('f-label-profile', [...new Set(D.labels.map(r => r.profile))].sort());
  }
  function apply() {
    const get = id => document.getElementById(id)?.value || '';
    document.querySelectorAll(`#tab-${tab} tr[data-model]`).forEach(tr => {
      const okM = !get('f-score-model') || tr.dataset.model === get('f-score-model');
      const okP = tab !== 'labels' || !get('f-label-profile') || tr.dataset.profile === get('f-label-profile');
      tr.style.display = (okM && okP) ? '' : 'none';
    });
  }
}
