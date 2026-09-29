#!/usr/bin/env python3
"""
Build a static site (site/) from your local judging data — for GitHub Pages.

Reads (never writes):
  eval/golden/labeled.jsonl   — per-rec good/bad/unsure labels
  eval/golden/scores.jsonl    — whole-response 1–5 ratings
  eval/golden/duels.jsonl     — model-vs-model rankings
  eval/golden/<profile>.json  — your golden sets (shown as editable answer keys)
  eval/output/leaderboard_*.md — the most recent automated leaderboards
  eval/output/runs_*.tsv       — every automated run (one row per model×profile),
                                  merged into the "Runs" tab: latest-per-model
                                  standings + full sortable history
  afrec agent registry         — agent tags + summaries for the "Models & agents"
                                  reference tab (model notes are hardcoded in MODEL_NOTES)

Writes a self-contained static site (no build step, no CDN, works offline):
  site/index.html, site/style.css, site/app.js, site/data.js
(leaderboards are rendered client-side from the markdown embedded in data.js)

Usage:
  python eval/site.py            # → regenerate site/
Then `git add site && git push` and GitHub Pages (source: main branch, /site
folder) serves it. Nothing here leaves your machine except the push.
"""

from __future__ import annotations

import csv
import datetime
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLDEN_DIR = HERE / "golden"
OUTPUT_DIR = HERE / "output"
SITE_DIR = ROOT / "site"
MAX_LEADERBOARDS = 10

# columns we coerce to numbers when parsing runs_*.tsv (older files may lack some)
_RUN_INTS = ("ok", "n_recs", "n_new_artists", "raw_candidates", "round1_found",
             "round1_missing", "retry_rescued", "untrusted", "golden_good", "golden_bad")
_RUN_FLOATS = ("raw_existence_rate", "final_existence_rate", "golden_score",
               "avg_confidence", "quality", "wallclock_s")


def load_runs() -> list[dict]:
    """Parse every eval/output/runs_*.tsv into one flat list of run rows.

    One row per (model × profile) per run; `run` is the timestamp from the
    filename. Older TSVs may lack columns (e.g. agent/untrusted) — missing
    values come back as None and the UI renders '—'.
    """
    out: list[dict] = []
    for f in sorted(OUTPUT_DIR.glob("runs_*.tsv")):
        run = f.stem.replace("runs_", "")
        try:
            with f.open() as fh:
                reader = csv.reader(fh, delimiter="\t")
                header = next(reader, None)
                if not header:
                    continue
                for row in reader:
                    d = dict(zip(header, row))
                    if not d.get("model"):
                        continue
                    d["run"] = run
                    for k in _RUN_INTS:
                        try:
                            d[k] = int(float(d[k])) if d.get(k) not in (None, "") else None
                        except (ValueError, TypeError):
                            d[k] = None
                    for k in _RUN_FLOATS:
                        try:
                            d[k] = float(d[k]) if d.get(k) not in (None, "") else None
                        except (ValueError, TypeError):
                            d[k] = None
                    out.append(d)
        except OSError:
            continue
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _load_golden_sets() -> list[dict]:
    sets = []
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        try:
            sets.append(json.loads(p.read_text()))
        except Exception:
            continue
    return sets


def load_data() -> dict:
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "labels": _load_jsonl(GOLDEN_DIR / "labeled.jsonl"),
        "scores": _load_jsonl(GOLDEN_DIR / "scores.jsonl"),
        "duels": _load_jsonl(GOLDEN_DIR / "duels.jsonl"),
        "golden_sets": _load_golden_sets(),
        "leaderboards": find_leaderboards(),
        "runs": load_runs(),
    }


def find_leaderboards() -> list[dict]:
    """Most recent leaderboard .md files, rendered to HTML."""
    files = sorted(OUTPUT_DIR.glob("leaderboard_*.md"), reverse=True)[:MAX_LEADERBOARDS]
    out = []
    for f in files:
        out.append({
            "name": f.stem,
            "html": md_to_html(f.read_text()),
        })
    return out


# ── Stats (mirror the definitions in eval/humaneval.py / eval/goldenset.py) ──

def elo_from_duels(duels: list[dict], k: float = 32.0, start: float = 1000.0) -> dict:
    """Same algorithm as humaneval.elo_from_duels (kept local so this script
    stays dependency-free)."""
    ratings: dict[str, float] = {}

    def ensure(m: str) -> None:
        ratings.setdefault(m, start)

    for d in duels:
        ranking = d.get("ranking") or []
        unranked = d.get("unranked") or []
        tied = set(d.get("ties") or [])
        for m in [*ranking, *unranked, *tied]:
            ensure(m)
        if set(ranking) == tied:          # full tie — no updates
            continue
        for i, better in enumerate(ranking):
            if better in tied:
                continue
            for worse in ranking[i + 1:] + unranked:
                if worse in tied or worse == better:
                    continue
                eb = 1.0 / (1 + 10 ** ((ratings[worse] - ratings[better]) / 400.0))
                ew = 1.0 / (1 + 10 ** ((ratings[better] - ratings[worse]) / 400.0))
                ratings[better] += k * (1.0 - eb)
                ratings[worse] += k * (0.0 - ew)

    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    n_duels: dict[str, int] = {}
    for d in duels:
        ranking = d.get("ranking") or []
        unranked = d.get("unranked") or []
        for m in set(ranking + unranked + (d.get("ties") or [])):
            n_duels[m] = n_duels.get(m, 0) + 1
        for i, better in enumerate(ranking):
            for worse in ranking[i + 1:] + unranked:
                wins[better] = wins.get(better, 0) + 1
                losses[worse] = losses.get(worse, 0) + 1

    out = {}
    for m in ratings:
        w, l = wins.get(m, 0), losses.get(m, 0)
        out[m] = {
            "elo": round(ratings[m], 1),
            "wins": w,
            "losses": l,
            "duels": n_duels.get(m, 0),
            "win_rate": round(w / (w + l), 3) if (w + l) else None,
        }
    return out


def summarize(data: dict) -> dict:
    labels = data["labels"]
    scores = data["scores"]

    def by_model(rows, key):
        acc: dict[str, list] = {}
        for r in rows:
            acc.setdefault(r.get("model") or "?", []).append(r)
        return acc

    label_models = by_model(labels, "label")
    score_models = by_model(scores, "score")
    elo = elo_from_duels(data["duels"])

    models = {}
    for m in set(label_models) | set(score_models) | set(elo):
        lr = label_models.get(m, [])
        good = sum(1 for r in lr if r.get("label") == "good")
        bad = sum(1 for r in lr if r.get("label") == "bad")
        decisive = good + bad
        sv = [r["score"] for r in score_models.get(m, []) if isinstance(r.get("score"), int)]
        models[m] = {
            "labels": len(lr),
            "label_good_rate": round(good / decisive, 3) if decisive else None,
            "scores": len(sv),
            "avg_score": round(sum(sv) / len(sv), 2) if sv else None,
            "elo": elo.get(m),
        }
    return {
        "models": models,
        "counts": {
            "labels": len(labels),
            "scores": len(scores),
            "duels": len(data["duels"]),
            "golden_sets": len(data["golden_sets"]),
            "runs": len(data["runs"]),
        },
    }


# ── Minimal markdown → HTML (enough for the leaderboard files) ───────────────

def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s


def md_to_html(md: str) -> str:
    out: list[str] = []
    table_buf: list[str] = []

    def flush_table():
        if not table_buf:
            return
        rows = [r for r in table_buf if not re.match(r"^\s*\|?[\s:|-]+\|?\s*$", r)]
        if rows:
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
            head, body = rows[0], rows[1:]
            out.append("<table><thead><tr>"
                       + "".join(f"<th>{_inline(c)}</th>" for c in cells(head))
                       + "</tr></thead><tbody>")
            for r in body:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells(r)) + "</tr>")
            out.append("</tbody></table>")
        table_buf.clear()

    for line in md.splitlines():
        if line.strip().startswith("|"):
            table_buf.append(line)
            continue
        flush_table()
        if line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.strip() in ("---", "***"):
            out.append("<hr>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    flush_table()
    return "\n".join(out)


# ── Page assembly ─────────────────────────────────────────────────────────────

INDEX = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wyatt's AI Music Eval</title>
<link rel="stylesheet" href="style.css?v=__TS__">
</head>
<body>
<header>
  <h1>🎧 Wyatt's AI Music Eval</h1>
  <p class="sub">human-judged model evaluations · generated <span id="generated"></span> · <a href="https://github.com/wyattpierson/wms-amongstfriends-recommender/tree/main/eval" target="_blank" rel="noopener">how this is generated (eval/)</a></p>
  <nav id="tabs">
    <button data-tab="dashboard" class="active">Dashboard</button>
    <button data-tab="runs">Runs (automated)</button>
    <button data-tab="scores">Scores (1–5)</button>
    <button data-tab="duels">Duels &amp; Elo</button>
    <button data-tab="labels">Labels</button>
    <button data-tab="golden">Golden sets</button>
    <button data-tab="leaderboards">Leaderboards</button>
    <button data-tab="about">Models &amp; agents</button>
  </nav>
</header>
<main>
  <section id="tab-dashboard" class="tab active"></section>
  <section id="tab-runs" class="tab"></section>
  <section id="tab-scores" class="tab"></section>
  <section id="tab-duels" class="tab"></section>
  <section id="tab-labels" class="tab"></section>
  <section id="tab-golden" class="tab"></section>
  <section id="tab-leaderboards" class="tab"></section>
  <section id="tab-about" class="tab"></section>
</main>
<script src="data.js?v=__TS__"></script>
<script src="app.js?v=__TS__"></script>
</body>
</html>
"""

CSS = """
:root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --text:#e6e9ef; --dim:#8b93a3;
        --good:#4ade80; --bad:#f87171; --mid:#fbbf24; --accent:#60a5fa; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header { padding:20px 24px 0; border-bottom:1px solid var(--line); background:var(--panel); }
h1 { margin:0; font-size:22px; }
.sub { color:var(--dim); margin:4px 0 12px; font-size:13px; }
.sub a { color:var(--dim); }
nav { display:flex; gap:6px; flex-wrap:wrap; }
nav button { background:none; border:1px solid var(--line); color:var(--dim);
             padding:6px 12px; border-radius:8px; cursor:pointer; font-size:14px; }
nav button.active { color:var(--text); border-color:var(--accent); background:#1c2433; }
main { padding:20px 24px 60px; max-width:1100px; margin:0 auto; }
.tab { display:none; }
.tab.active { display:block; }
h2 { font-size:17px; margin:24px 0 10px; }
h2:first-child { margin-top:0; }
table { border-collapse:collapse; width:100%; margin:10px 0 20px; font-size:14px; }
th, td { border:1px solid var(--line); padding:6px 10px; text-align:left; vertical-align:top; }
th { background:#1c2029; color:var(--dim); font-weight:600; white-space:nowrap; }
td.num, th.num { text-align:right; }
tr:hover td { background:#1a1f29; }
th.sortable { cursor:pointer; user-select:none; } th.sortable:hover { color:var(--accent); }
tr.failed td { opacity:0.55; } tr.untrusted td { background:#2a1a1e; }
.good { color:var(--good); } .bad { color:var(--bad); } .mid { color:var(--mid); }
.dim { color:var(--dim); }
.bar { background:#232936; border-radius:4px; height:10px; width:120px; display:inline-block; vertical-align:middle; }
.bar > div { height:10px; border-radius:4px; background:var(--accent); }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; background:#232936; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin:12px 0 20px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:140px; }
.card .n { font-size:26px; font-weight:700; }
.card .l { color:var(--dim); font-size:13px; }
select, input { background:#1c2029; color:var(--text); border:1px solid var(--line);
                border-radius:6px; padding:5px 8px; font-size:14px; }
.filters { margin:8px 0; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
details { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:8px 12px; margin:8px 0; }
summary { cursor:pointer; color:var(--accent); }
pre { background:#0c0e12; border:1px solid var(--line); border-radius:8px;
      padding:12px; overflow-x:auto; font-size:13px; }
.md h1 { font-size:20px; } .md h2 { font-size:17px; } .md p { margin:8px 0; color:var(--dim); }
.empty { color:var(--dim); font-style:italic; padding:16px 0; }
code { background:#232936; padding:1px 5px; border-radius:4px; font-size:13px; }
"""

# app.js renders the tabs from window.SITE_DATA (data.js). Keep it dependency-free.
APP_JS = r"""
'use strict';
const D = window.SITE_DATA || { summary: { models: {}, counts: {} }, scores: [], duels: [], labels: [], golden_sets: [], leaderboards: [], about: '', generated_at: '' };

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
    <div class="card"><div class="n">${c.golden_sets}</div><div class="l">golden sets</div></div>
    <div class="card"><div class="n">${c.runs}</div><div class="l">automated runs</div></div></div>`;
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
  let h = `<h2>Current standings — latest good run per model × agent × profile</h2>
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

const renderers = { dashboard, runs, scores, duels, labels, golden, leaderboards, about };
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
"""


# ── Models & agents reference tab ──────────────────────────────────────────
# One-line notes for the models in the local pool. Update when the pool changes;
# agent descriptions come live from the afrec registry (same as `make agents`).
MODEL_NOTES: dict[str, str] = {
    "Qwen3.8-27B-Q8_0": "the big model (27B, Q8) — the quality ceiling; needs its own memory budget, so it runs alone",
    "gemma-4-26B-A4B-it-UD-Q5_K_M": "mixture-of-experts: 26B total params but only ~4B active per token — big-model quality at small-model speed, can pair with one small model",
    "gemma-4-E4B-it-Q8_0": "the lightest Gemma 4 (E4B, Q8) — fast small model for triage runs and pairing",
    "Dolphin3.0-Llama3.1-8B.Q8_0": "Cognitive Computations' Dolphin 3.0 fine-tune of Llama 3.1 8B (Q8) — a less-censored, instruction-following spin on Meta's 8B",
    "Meta-Llama-3.1-8B-Instruct-Q8_0": "Meta's stock Llama 3.1 8B instruct (Q8) — the baseline that fine-tunes are compared against",
    "Mistral-7B-Instruct-v0.3-Q8_0": "Mistral AI's 7B instruct v0.3 (Q8) — small and fast",
    "microsoft_Phi-4-mini-instruct-Q8_0": "Microsoft's Phi-4-mini (~4B, Q8) — the smallest in the pool; punches above its weight, good for interactive sessions",
}


def build_about() -> str:
    h = '<h2>Models (local pool)</h2><table><tr><th>model</th><th>what it is</th></tr>'
    for name, note in MODEL_NOTES.items():
        h += (f'<tr><td><code>{html.escape(name)}</code></td>'
              f'<td class="dim">{html.escape(note)}</td></tr>')
    h += '</table>'
    try:
        import sys
        sys.path.insert(0, str(ROOT))            # for afrec.* (same as evaluate.py)
        from afrec.agents import list_agents, selected_agent
        sel = selected_agent().tag
        h += ('<h2>Agents (playbooks)</h2>'
              '<p class="dim">The agent is the playbook (prompts + call flow + retry policy); '
              'the model is the brain. Every scored run records both, so results never mix silently.</p>'
              '<table><tr><th>tag</th><th>what it does</th></tr>')
        for a in list_agents():
            mark = ' <span class="pill">selected</span>' if a.tag == sel else ''
            h += (f'<tr><td><code>{html.escape(a.tag)}</code>{mark}</td>'
                  f'<td class="dim">{html.escape(a.summary)}</td></tr>')
        h += '</table>'
    except Exception:
        h += '<p class="dim">(agent reference unavailable — run <code>make agents</code> locally)</p>'
    return h


def main() -> None:
    data = load_data()
    data["summary"] = summarize(data)
    data["about"] = build_about()
    # cache-bust: browsers cache file:// resources aggressively; a per-build
    # version string forces a fresh fetch of css/js/data on every rebuild
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "index.html").write_text(INDEX.replace("__TS__", ts))
    (SITE_DIR / "style.css").write_text(CSS)
    (SITE_DIR / "app.js").write_text(APP_JS)
    (SITE_DIR / "data.js").write_text(
        "// generated by eval/site.py — do not edit\n"
        "window.SITE_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n"
    )

    n_lb = len(data["leaderboards"])
    print(f"✅ site/ rebuilt — {data['summary']['counts']['labels']} labels, "
          f"{data['summary']['counts']['scores']} scores, "
          f"{data['summary']['counts']['duels']} duels, "
          f"{data['summary']['counts']['golden_sets']} golden sets, "
          f"{data['summary']['counts']['runs']} runs, {n_lb} leaderboards")
    print("   → open site/index.html, or `git add site && git push` for GitHub Pages")


if __name__ == "__main__":
    main()
