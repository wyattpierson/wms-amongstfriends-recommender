#!/usr/bin/env python3
"""
Build a static site (docs/) from your local judging data — for GitHub Pages.

GitHub Pages can only serve the repo root or the /docs folder from a branch,
so the site is generated into docs/ — set Pages source to "main branch, /docs".

Reads (never writes):
  eval/golden/labeled.jsonl   — per-rec good/bad/unsure labels
  eval/golden/scores.jsonl    — whole-response 1–5 ratings
  eval/golden/duels.jsonl     — model-vs-model rankings
  eval/golden/<profile>.json  — your golden sets (shown as editable answer keys)
  eval/output/leaderboard_*.md — the most recent automated leaderboards
  eval/output/runs_*.tsv       — every automated run (one row per model×profile),
                                  merged into the "Runs" tab: latest-per-model
                                  standings + full sortable history
  eval/output/runs_history.jsonl — per-rec append-only history (model's
                                  self-reported genre/era), behind the "Recs"
                                  tab. Self-reports are display-only — never
                                  scored on; they exist to make untagged recs
                                  legible and to feed tag harvesting.
  afrec agent registry         — agent tags + summaries for the "Models & agents"
                                  reference tab (model notes are hardcoded in MODEL_NOTES)

Writes a self-contained static site (no build step, no CDN, works offline):
  docs/index.html, docs/style.css, docs/app.js, docs/data.js
(leaderboards are rendered client-side from the markdown embedded in data.js)

Usage:
  python eval/site.py            # → regenerate docs/
Then `git add docs && git push` and GitHub Pages (source: main branch, /docs
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
SITE_DIR = ROOT / "docs"
MAX_LEADERBOARDS = 10
MAX_REC_ROWS = 2000   # cap the embedded recs-history (oldest rows dropped)

# columns we coerce to numbers when parsing runs_*.tsv (older files may lack some)
_RUN_INTS = ("ok", "n_recs", "n_new_artists", "raw_candidates", "round1_found",
             "round1_missing", "retry_rescued", "untrusted", "golden_good", "golden_bad",
             "just_hits", "just_total", "just_fixture_hits")
_RUN_FLOATS = ("raw_existence_rate", "final_existence_rate", "golden_score",
               "avg_confidence", "quality", "wallclock_s",
               "structure_score", "lane_share", "lane_balance", "tag_coverage")


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
    golden_sets = _load_golden_sets()
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "labels": _load_jsonl(GOLDEN_DIR / "labeled.jsonl"),
        "scores": _load_jsonl(GOLDEN_DIR / "scores.jsonl"),
        "duels": _load_jsonl(GOLDEN_DIR / "duels.jsonl"),
        "golden_sets": golden_sets,
        "leaderboards": find_leaderboards(),
        "runs": load_runs(),
        "recs": load_rec_rows(golden_sets),
    }


def load_rec_rows(golden_sets: list[dict]) -> list[dict]:
    """Flatten eval/output/runs_history.jsonl into one row per rec, newest last.

    Each row carries the model's SELF-REPORTED genre/era (display only — never
    scored on) plus a `tagged` flag: whether the artist appears in any golden
    set's `tags` section. That flag is the at-a-glance 'how much of what the
    models recommend is still untagged' signal the harvest step grows out of.
    """
    idx: set[str] = set()
    for g in golden_sets:
        for a in (g.get("tags") or {}):
            idx.add(a.lower().strip())
    rows = []
    for entry in _load_jsonl(OUTPUT_DIR / "runs_history.jsonl"):
        for rec in (entry.get("recs") or []):
            artist = (rec.get("artist") or "").strip()
            if not artist:
                continue
            rows.append({
                "ts": entry.get("ts", ""), "run": entry.get("run", ""),
                "model": entry.get("model", ""), "agent": entry.get("agent", ""),
                "profile": entry.get("profile", ""),
                "artist": artist,
                "album": rec.get("album") or "",
                "self_genre": rec.get("self_genre"),
                "self_era": rec.get("self_era"),
                "release_year": rec.get("release_year"),
                "confidence": rec.get("confidence"),
                "verified": rec.get("verified"),
                "tagged": artist.lower() in idx,
            })
    if len(rows) > MAX_REC_ROWS:
        rows = rows[-MAX_REC_ROWS:]
    return rows


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

def _model_of(label: str) -> str:
    """Model part of a 'model [agent]' label (or the bare label if no agent)."""
    return label.rsplit(" [", 1)[0] if " [" in label else label


def _agent_of(label: str) -> str:
    """Agent part of a 'model [agent]' label (or '' if the label has none)."""
    return label.rsplit(" [", 1)[1].rstrip("]") if " [" in label else ""


def _contestants(d: dict) -> list[str]:
    return list(dict.fromkeys(
        (d.get("ranking") or []) + (d.get("unranked") or []) + (d.get("ties") or [])))


def _all_same(duel: dict, dim) -> bool:
    """True when every contestant in the duel shares one value of `dim(label)` —
    i.e. the duel is a clean head-to-head on the OTHER dimension."""
    labels = _contestants(duel)
    return bool(labels) and len({dim(l) for l in labels}) == 1


def _elo_core(duels: list[dict], keyfn, k: float = 32.0, start: float = 1000.0) -> dict:
    """ELO algorithm (same as humaneval.elo_from_duels), keying each contestant
    by keyfn(label) so the same duel set can be scored three ways: identity for
    the combined model+agent ladder, _agent_of for agent-only (same-model
    duels), _model_of for model-only (same-agent duels). A duel is skipped if
    keyfn collapses two distinct contestants onto one key (ambiguous verdict)."""
    matches = []
    for d in duels:
        labels = _contestants(d)
        if len({keyfn(l) for l in labels}) != len(set(labels)):
            continue
        matches.append(d)

    ratings: dict[str, float] = {}

    def ensure(m: str) -> None:
        ratings.setdefault(m, start)

    for d in matches:
        ranking = [keyfn(l) for l in (d.get("ranking") or [])]
        unranked = [keyfn(l) for l in (d.get("unranked") or [])]
        tied = {keyfn(l) for l in (d.get("ties") or [])}
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
    for d in matches:
        ranking = [keyfn(l) for l in (d.get("ranking") or [])]
        unranked = [keyfn(l) for l in (d.get("unranked") or [])]
        tied = [keyfn(l) for l in (d.get("ties") or [])]
        for m in set(ranking + unranked + tied):
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


def elo_from_duels(duels: list[dict], k: float = 32.0, start: float = 1000.0) -> dict:
    """Combined model+agent ELO, keyed by the raw 'model [agent]' label."""
    return _elo_core(duels, keyfn=lambda l: l, k=k, start=start)


def agent_elo(duels: list[dict], k: float = 32.0, start: float = 1000.0) -> dict:
    """Agent-only ELO, from SAME-MODEL duels: the verdict belongs to the agent,
    never the model (a model must not be credited for beating itself)."""
    return _elo_core([d for d in duels if _all_same(d, _model_of)], keyfn=_agent_of, k=k, start=start)


def model_elo(duels: list[dict], k: float = 32.0, start: float = 1000.0) -> dict:
    """Model-only ELO, from SAME-AGENT duels: the verdict belongs to the model,
    never the agent. Mixed-agent duels are excluded."""
    return _elo_core([d for d in duels if _all_same(d, _agent_of)], keyfn=_model_of, k=k, start=start)


def agent_duel_stats(duels: list[dict]) -> dict:
    """Mirror of humaneval.agent_duel_stats (kept local — dependency-free):
    roll up SAME-MODEL duels per agent. In those duels the verdict belongs to
    the agent, never the model (a model must not be credited for beating
    itself). Mixed-model duels are model comparisons and are skipped."""
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    ties: dict[str, int] = {}
    seen: dict[str, int] = {}

    def _model_of(label: str) -> str:
        return label.rsplit(" [", 1)[0] if " [" in label else label

    for d in duels:
        agents = d.get("agents") or {}
        if not agents:
            continue
        if len({_model_of(l) for l in agents}) != 1:
            continue
        ranked = [agents[m] for m in (d.get("ranking") or []) if m in agents]
        unranked = [agents[m] for m in (d.get("unranked") or []) if m in agents]
        tied = [agents[m] for m in (d.get("ties") or []) if m in agents]
        contest = list(dict.fromkeys(ranked + unranked + tied))
        if len(contest) < 2:
            continue
        for a in contest:
            seen[a] = seen.get(a, 0) + 1
        if tied and not ranked:
            for a in contest:
                ties[a] = ties.get(a, 0) + 1
            continue
        for i, better in enumerate(ranked):
            for worse in ranked[i + 1:]:
                wins[better] = wins.get(better, 0) + 1
                losses[worse] = losses.get(worse, 0) + 1
        for m in unranked:
            for better in ranked:
                wins[better] = wins.get(better, 0) + 1
                losses[m] = losses.get(m, 0) + 1

    out = {}
    for a in seen:
        w, l, t = wins.get(a, 0), losses.get(a, 0), ties.get(a, 0)
        out[a] = {"wins": w, "losses": l, "ties": t, "duels": seen[a],
                  "win_rate": round(w / (w + l), 3) if (w + l) else None}
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
        "agent_duels": agent_duel_stats(data["duels"]),
        "agent_elo": agent_elo(data["duels"]),
        "model_elo": model_elo(data["duels"]),
        "combined_elo": elo_from_duels(data["duels"]),
        "counts": {
            "labels": len(labels),
            "scores": len(scores),
            "duels": len(data["duels"]),
            "golden_sets": len(data["golden_sets"]),
            "runs": len(data["runs"]),
            "recs": len(data.get("recs", [])),
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
  <h1>Wyatt's AI Music Eval</h1>
  <p class="sub">human-judged model evaluations · generated <span id="generated"></span> · <a href="https://github.com/wyattpierson/wms-amongstfriends-recommender/tree/main/eval" target="_blank" rel="noopener">how this is generated (eval/)</a></p>
  <nav id="tabs">
    <button data-tab="dashboard" class="active">Dashboard</button>
    <button data-tab="runs">Runs (automated)</button>
    <button data-tab="scores">Scores (1–5)</button>
    <button data-tab="duels">Duels &amp; Elo</button>
    <button data-tab="labels">Labels</button>
    <button data-tab="golden">Golden sets</button>
    <button data-tab="recs">Recs (self-reported)</button>
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
  <section id="tab-recs" class="tab"></section>
  <section id="tab-leaderboards" class="tab"></section>
  <section id="tab-about" class="tab"></section>
</main>
<script src="data.js?v=__TS__"></script>
<script src="app.js?v=__TS__"></script>
</body>
</html>
"""

CSS = """
:root { --bg:#14111f; --panel:#1d1930; --panel2:#251f3b; --line:#2f2949; --text:#efeaf7; --dim:#9a93b3;
        --good:#7be0a8; --bad:#ff7a90; --mid:#ffc15e; --accent:#ffb547;
        --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
header { padding:26px 28px 0; background:linear-gradient(#1d1830,var(--bg)); position:sticky; top:0; z-index:5; border-bottom:1px solid var(--line); }
h1 { margin:0; font:600 30px/1.1 var(--serif); letter-spacing:-.01em; }
.sub { color:var(--dim); margin:6px 0 14px; font-size:13px; }
.sub a { color:var(--dim); }
nav { display:flex; gap:2px; overflow-x:auto; }
nav button { background:none; border:0; border-bottom:2px solid transparent; color:var(--dim); padding:8px 12px; cursor:pointer; font:inherit; font-size:14px; white-space:nowrap; }
nav button:hover { color:var(--text); }
nav button.active { color:var(--accent); border-color:var(--accent); }
button:focus-visible, select:focus-visible, summary:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
main { padding:26px 28px 70px; max-width:1200px; margin:0 auto; }
.tab { display:none; overflow-x:auto; } .tab.active { display:block; }
h2 { font:600 20px var(--serif); margin:34px 0 12px; } h2:first-child { margin-top:0; }
h3 { font:600 15px var(--serif); margin:0 0 12px; }
table { border-collapse:collapse; width:100%; margin:10px 0 22px; font-size:14px; }
th, td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
th { color:var(--dim); font-weight:600; font-size:13px; white-space:nowrap; background:var(--panel); position:sticky; top:0; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
tr:hover td { background:#221d37; }
th.sortable { cursor:pointer; user-select:none; } th.sortable:hover { color:var(--accent); }
tr.failed td { opacity:.5; } tr.untrusted td { background:#33182a; }
.good { color:var(--good); } .bad { color:var(--bad); } .mid { color:var(--mid); } .dim { color:var(--dim); }
.bar { background:#2a2444; border-radius:4px; height:8px; width:90px; display:inline-block; vertical-align:middle; }
.bar > div { height:8px; border-radius:4px; background:var(--accent); }
.pill { display:inline-block; padding:1px 9px; border-radius:10px; font-size:12px; background:#2a2444; }
.cards { display:flex; gap:12px; flex-wrap:wrap; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px; }
select, input { background:var(--panel2); color:var(--text); border:1px solid var(--line); border-radius:6px; padding:5px 8px; font:inherit; font-size:14px; }
.filters { margin:8px 0 12px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; color:var(--dim); }
details { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px 12px; margin:8px 0; }
summary { cursor:pointer; color:var(--accent); }
pre { background:#100d19; border:1px solid var(--line); border-radius:8px; padding:12px; overflow-x:auto; font-size:13px; }
.md h1 { font-size:20px; } .md p { margin:8px 0; color:var(--dim); }
.empty { color:var(--dim); font-style:italic; padding:16px 0; }
code { background:#2a2444; padding:1px 5px; border-radius:4px; font-size:13px; }
/* hero: a record half-sunk into the panel */
.hero { display:grid; grid-template-columns:1.1fr 1fr; gap:36px; align-items:center; padding:34px; border:1px solid var(--line); border-radius:18px; overflow:hidden; position:relative;
        background:radial-gradient(circle at 0% 50%,var(--panel2),var(--panel)); }
.hero::before { content:""; position:absolute; right:-170px; top:50%; width:420px; height:420px; margin-top:-210px; border-radius:50%; opacity:.5; pointer-events:none;
        background:radial-gradient(circle,var(--accent) 0 5%,#15111f 5.5% 14%,transparent 14.5%),repeating-radial-gradient(circle,#191428 0 3px,#231d38 3px 6px); }
.hero > * { position:relative; }
.hero .name { font:600 clamp(30px,5vw,50px)/1.05 var(--serif); letter-spacing:-.02em; margin:6px 0 14px; }
.facts { display:flex; gap:26px; flex-wrap:wrap; } .facts b { display:block; font:600 24px var(--serif); } .facts span { color:var(--dim); font-size:13px; }
.strip { display:flex; flex-wrap:wrap; gap:8px 34px; margin:22px 2px 6px; } .strip b { font:600 26px var(--serif); margin-right:7px; } .strip span { color:var(--dim); font-size:13px; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:16px; margin:16px 0; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px 20px; }
.panel p.dim { font-size:13px; margin:-6px 0 12px; }
.hb { display:grid; gap:9px; }
.hr { display:grid; grid-template-columns:minmax(110px,1.1fr) 2fr 56px; gap:10px; align-items:center; font-size:13px; }
.hr b { text-align:right; font-variant-numeric:tabular-nums; }
.hl { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ht { display:flex; height:10px; background:#2a2444; border-radius:5px; overflow:hidden; } .ht > span { height:100%; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; }
.plot { width:100%; height:auto; } .plot line { stroke:var(--line); } .plot text { fill:var(--dim); font-size:10px; }
.lg { display:flex; flex-wrap:wrap; gap:4px 14px; font-size:12px; color:var(--dim); margin-top:6px; }
@media (max-width:760px) { .hero { grid-template-columns:1fr; padding:22px; } header, main { padding-left:16px; padding-right:16px; } }
@media (prefers-reduced-motion:no-preference) { .ht > span { transition:width .6s ease; } }
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
    print(f"✅ docs/ rebuilt — {data['summary']['counts']['labels']} labels, "
          f"{data['summary']['counts']['scores']} scores, "
          f"{data['summary']['counts']['duels']} duels, "
          f"{data['summary']['counts']['golden_sets']} golden sets, "
          f"{data['summary']['counts']['runs']} runs, {n_lb} leaderboards")
    print("   → open docs/index.html, or `git add docs && git push` for GitHub Pages (source: main branch, /docs)")


if __name__ == "__main__":
    main()
