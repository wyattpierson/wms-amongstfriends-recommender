"""Interactive human-judgment records: whole-response scores + model duels.

Complements goldenset.py (per-rec good/bad labels + static golden sets) with
two coarser, very human signals that a golden set can't capture:

1. RESPONSE SCORES  (eval/golden/scores.jsonl, append-only)
   A human listens to / reads a model's FULL set of recommendations for a
   profile and rates the whole response 1–5:
       1 = bad        2 = weak      3 = neutral      4 = good       5 = great
   Unlike a golden set, this rewards answers the model came up with on its own
   that were "great but we never planned for."
   → append_score(), load_scores(), score_stats()
   → calibration_stats() — does the model's self-claim track your taste?

2. MODEL DUELS      (eval/golden/duels.jsonl, append-only)
   Two or more models answer the same profile; a human ranks the responses.
   Records the winner/loser per pair, aggregated as win-rates + Elo ratings.
   → append_duel(), load_duels(), elo_from_duels()

Both logs live next to goldenset's labeled.jsonl in eval/golden/ — "all human
judgment lives in one place." Both are append-only JSONL so they're greppable,
version-controllable, and easy to inspect in a text editor.
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path

import goldenset as _gs                  # labeled.jsonl (per-rec verdicts) — same dir, no cycle

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"
SCORES_LOG = GOLDEN_DIR / "scores.jsonl"
DUELS_LOG  = GOLDEN_DIR / "duels.jsonl"

# 1–5 scale labels (1 = worst, 5 = best).
SCORE_LABELS = {1: "bad", 2: "weak", 3: "neutral", 4: "good", 5: "great"}

BUCKETS = (("bad", (1, 2)), ("neutral", (3, 3)), ("good", (4, 5)))

# ── Response scores (1–5 on a full set of recs) ───────────────────────────────

def append_score(
    *,
    profile: str,
    model: str,
    agent: str = "",     # which agent version generated the response (e.g. "artist-then-album@v1")
    score: int,          # 1..5
    note: str = "",
    recs: list | None = None,
    spotify_on: bool = False,
    base_url: str = "",
    temperature: float = 0.7,
    wallclock_s: float = 0.0,
    round: int = 1,
) -> Path:
    """Record one human rating of a model's full response. Appends JSONL."""
    if not (isinstance(score, int) and 1 <= score <= 5):
        raise ValueError(f"score must be an int 1..5, got {score!r}")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "kind": "score",
        "profile": profile,
        "model": (model or "").strip(),
        "agent": (agent or "").strip(),
        "score": score,
        "score_label": SCORE_LABELS[score],
        "note": (note or "").strip(),
        "round": round,
        "recs": recs or [],
        "spotify_on": bool(spotify_on),
        "base_url": base_url or "",
        "temperature": temperature,
        "wallclock_s": wallclock_s,
        "source": "taste.py",
    }
    with SCORES_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return SCORES_LOG


def load_scores() -> list[dict]:
    if not SCORES_LOG.exists():
        return []
    out = []
    for line in SCORES_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "score" and isinstance(rec.get("score"), int):
            out.append(rec)
    return out


def score_stats(model: str | None = None, profile: str | None = None,
                agent: str | None = None) -> dict:
    """
    Aggregate response scores.

    Buckets (for the bad/neutral/good view):
      bad     = scores 1–2
      neutral = score  3
      good    = scores 4–5
    avg is the mean of the 1–5 scale.
    """
    rows = load_scores()
    if model:
        rows = [r for r in rows if r.get("model") == model]
    if profile:
        rows = [r for r in rows if r.get("profile") == profile]
    if agent:
        rows = [r for r in rows if (r.get("agent") or "") == agent]
    vals = [r["score"] for r in rows]
    n = len(vals)
    return {
        "n": n,
        "avg": round(sum(vals) / n, 3) if n else None,
        "bad": sum(1 for v in vals if v <= 2),
        "neutral": sum(1 for v in vals if v == 3),
        "good": sum(1 for v in vals if v >= 4),
        "distribution": {s: sum(1 for v in vals if v == s) for s in range(1, 6)},
    }


def calibration_stats(model: str | None = None) -> dict:
    """
    How well does a model's self-reported confidence track YOUR taste?

    Per scored response: mean_conf = average rec confidence (0–1) the model
    claimed; target = (score - 1) / 4 so 1→0.0, 3→0.5, 5→1.0. Then:

      bias        mean_conf - target  (> 0 = overconfident relative to taste,
                                       < 0 = underconfident, ~0 = well calibrated)
      rmse        sqrt of mean (mean_conf - target)^2
      spearman    rank correlation of claim vs taste (n≥3). ~1 = its
                  confidence ordering matches your ordering; ~0/-1 = it
                  doesn't know when it's off.
      buckets     {"bad": (n, avg_conf), "neutral": ..., "good": ...} — the
                  red-flag table: if a model claims ~0.8 average on your
                  1–2 rated responses, its confidence is meaningless.

    Requires score records that carry recs with confidence values (taste.py
    score stores both by default). Returns {"n": 0} when there's no data.
    """
    rows = load_scores()
    if model:
        rows = [r for r in rows if r.get("model") == model]

    points = []   # (mean_conf, target)
    by_bucket = {name: [] for name, _ in BUCKETS}
    for r in rows:
        s = r.get("score")
        if not (isinstance(s, int) and 1 <= s <= 5):
            continue
        confs = [rec.get("confidence") for rec in (r.get("recs") or [])
                 if isinstance(rec, dict) and isinstance(rec.get("confidence"), (int, float))]
        if not confs:
            continue
        points.append((sum(confs) / len(confs), (s - 1) / 4.0))
        for name, (lo, hi) in BUCKETS:
            if lo <= s <= hi:
                by_bucket[name].append(sum(confs) / len(confs))

    n = len(points)
    if n == 0:
        return {"n": 0}

    mean_conf = sum(c for c, _ in points) / n
    mean_target = sum(t for _, t in points) / n
    bias = mean_conf - mean_target
    rmse = (sum((c - t) ** 2 for c, t in points) / n) ** 0.5
    out = {
        "n": n,
        "mean_conf": round(mean_conf, 3),
        "mean_taste": round(1 + mean_target * 4, 3),   # back on the 1–5 scale
        "bias": round(bias, 3),
        "rmse": round(rmse, 3),
        "verdict": ("well calibrated" if abs(bias) < 0.1 else
                    "overconfident" if bias > 0 else "underconfident"),
        "buckets": {name: (len(vals), round(sum(vals) / len(vals), 3) if vals else None)
                   for name, vals in by_bucket.items()},
    }

    if n >= 3:
        spear = _spearman([c for c, _ in points], [t for _, t in points])
        if spear is not None:
            out["spearman"] = round(spear, 3)
    return out


def _avg_rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0      # 1-based average rank over the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _avg_rank(xs), _avg_rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


# ── Model duels (pairwise/batch human preference) ─────────────────────────────

def append_duel(
    *,
    profile: str,
    ranking: list[str],     # model names, best → (as far as the human ordered)
    unranked: list[str] | None = None,   # models that lost (no relative order given)
    ties: list[str] | None = None,   # models the human called tied-for-winner
    agent: str = "",        # agent used by ALL contenders (duels are same-agent);
                            # leave "" when contenders used different agents —
                            # the ranking entries then carry their own identity
    agents: dict | None = None,  # per-contestant agent map {label: agent} (traceability)
    note: str = "",
    recs_by_model: dict | None = None,
    spotify_on: bool = False,
    base_url: str = "",
    temperature: float = 0.7,
) -> Path:
    """
    Record one human-ordered duel. `ranking` lists model names best→worse;
    it may be partial (e.g. just the winner) — every model in `unranked`
    then counts as a loss to each ranked model without implying any order
    among themselves. `ties` is a full tie (used with an empty ranking).
    Appends JSONL and returns the path.
    """
    named = list(dict.fromkeys([*ranking, *(unranked or []), *(ties or [])]))
    if len(named) < 2:
        raise ValueError(f"needs at least 2 models total, got {named!r}")
    if ranking and len(set(ranking)) != len(ranking):
        raise ValueError(f"models must appear at most once in ranking: {ranking!r}")
    if not ranking and not (ties or unranked):
        raise ValueError("needs a ranking or a tie")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "kind": "duel",
        "profile": profile,
        "agent": (agent or "").strip(),
        "agents": agents or {},
        "ranking": ranking,
        "unranked": unranked or [],
        "ties": ties or [],
        "note": (note or "").strip(),
        "recs_by_model": recs_by_model or {},
        "spotify_on": bool(spotify_on),
        "base_url": base_url or "",
        "temperature": temperature,
        "source": "taste.py",
    }
    with DUELS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return DUELS_LOG


def load_duels() -> list[dict]:
    if not DUELS_LOG.exists():
        return []
    out = []
    for line in DUELS_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "duel":
            out.append(rec)
    return out


def _elo_update(ratings: dict, k: float, better: str, worse: str) -> None:
    rb, rw = ratings[better], ratings[worse]
    eb = 1.0 / (1.0 + 10 ** ((rw - rb) / 400.0))
    ew = 1.0 / (1.0 + 10 ** ((rb - rw) / 400.0))
    ratings[better] += k * (1.0 - eb)
    ratings[worse]  += k * (0.0 - ew)


def elo_from_duels(k: float = 32.0, start: float = 1000.0) -> dict:
    """
    Compute Elo ratings from duel history. For each duel, every pair implied by
    the (partial) human ranking contributes one win: in a ranking [a, b, c]
    that's a>b, a>c, b>c; every `unranked` model loses to each ranked
    model. Ties contribute nothing.
    Returns {model: {elo, wins, losses, ties, duels, win_rate}}.
    """
    ratings: dict[str, float] = {}
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    duels_seen: dict[str, int] = {}
    ties: dict[str, int] = {}

    def _touch(m: str):
        ratings.setdefault(m, start)
        wins.setdefault(m, 0)
        losses.setdefault(m, 0)
        duels_seen.setdefault(m, 0)
        ties.setdefault(m, 0)

    # Duels recorded with an agent tag are keyed as "model [agent]" so a
    # playbook change starts a fresh Elo ladder instead of polluting the
    # model's history. Old records (no agent field) key as bare model name.
    for d in load_duels():
        ag = (d.get("agent") or "").strip()
        def _key(m: str) -> str:
            return f"{m} [{ag}]" if ag else m
        ranking = [_key(m) for m in d.get("ranking", []) if m]
        unranked = [_key(m) for m in (d.get("unranked", []) or [])]
        tied = [_key(m) for m in (d.get("ties", []) or [])]
        models = list(dict.fromkeys(ranking + unranked + tied))
        if len(models) < 2:
            continue
        for m in models:
            _touch(m)
            duels_seen[m] += 1
        if tied and not ranking:
            # all models tied → no Elo movement, just tie tallies
            for m in models:
                ties[m] += 1
            continue
        # Every pair implied by the explicit ordering counts as a win.
        for i, better in enumerate(ranking):
            for worse in ranking[i + 1:]:
                _elo_update(ratings, k, better, worse)
                wins[better] += 1
                losses[worse] += 1
        # Unranked losers lost to every ranked model (no order among them).
        for m in unranked:
            for better in ranking:
                _elo_update(ratings, k, better, m)
                wins[better] += 1
                losses[m] += 1

    out = {}
    for m, elo in ratings.items():
        w, l, t = wins[m], losses[m], ties[m]
        decisive = w + l
        out[m] = {
            "elo": round(elo, 1),
            "wins": w,
            "losses": l,
            "ties": t,
            "duels": duels_seen[m],
            "win_rate": round(w / decisive, 3) if decisive else None,
        }
    return out


def agent_duel_stats() -> dict:
    """
    Agent-level results from SAME-MODEL duels (duels.py `agents`): when two
    agents duel on one model, the verdict is an AGENT win/loss — the model
    must never be credited for beating itself. (elo_from_duels already keeps
    these out of the model's ladder by keying them "Model [agent]"; this rolls
    the same-model duels up per AGENT across models, which the per-model
    ladders can't show.)

    A duel counts as same-model when every contestant label (from the
    `agents` map) refers to one single model. Mixed-model duels are model
    comparisons and are skipped here.

    Returns {agent: {wins, losses, ties, duels, win_rate}}.
    """
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    ties: dict[str, int] = {}
    seen: dict[str, int] = {}

    def _model_of(label: str) -> str:
        # "Model [agent]" → "Model"; bare "Model" → "Model"
        return label.rsplit(" [", 1)[0] if " [" in label else label

    for d in load_duels():
        agents = d.get("agents") or {}
        if not agents:
            continue
        models = {_model_of(l) for l in agents}
        if len(models) != 1:
            continue                       # mixed-model duel → model comparison
        ranked = [agents[m] for m in d.get("ranking", []) if m in agents]
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
        out[a] = {
            "wins": w, "losses": l, "ties": t, "duels": seen[a],
            "win_rate": round(w / (w + l), 3) if (w + l) else None,
        }
    return out


def all_models() -> list[str]:
    """Every model name seen in scores or duels, sorted."""
    models = {r.get("model") for r in load_scores() if r.get("model")}
    for d in load_duels():
        models.update(d.get("ranking", []))
        models.update(d.get("unranked", []) or [])
        models.update(d.get("ties", []) or [])
    models.discard(None)
    return sorted(models)


# ── Pool-vs-item consistency ─────────────────────────────────────────────────

def pool_item_consistency(min_decisive: int = 3) -> list[dict]:
    """
    Cross-check the two human signals against each other: for every 1–5 pool
    score whose per-rec verdicts are known (labeled.jsonl), does the pool
    score agree with the items?

    For each score record, every rec is matched to its most-recent DECISIVE
    label (good/bad) in labeled.jsonl by (profile, artist) — and by album too
    when both sides name one. good_frac = good / (good + bad) over the recs
    that have a decisive label. Records with fewer than `min_decisive`
    decisive recs are excluded (not enough signal).

    A record is a MISMATCH when the pool says good (4–5) but less than 50% of
    the judged recs were good, or the pool says bad (1–2) but more than 50%
    were good. Neutral pools (3) never mismatch — they claim nothing.

    Returns one dict per eligible record, oldest first:
      {ts, profile, model, agent, pool, good, bad, good_frac, n_decisive, mismatch}
    """
    if not _gs.LABELED_LOG.exists():
        return []
    # Index: (profile, normalized artist) -> [rows, in file order]
    idx: dict[tuple, list[dict]] = {}
    for r in _gs.load_labeled():
        idx.setdefault((r.get("profile") or "",
                        _gs._normalize(r.get("artist", ""))), []).append(r)

    out = []
    for s in load_scores():
        recs = s.get("recs") or []
        if not recs:
            continue
        prof = s.get("profile") or ""
        good = bad = 0
        for rec in recs:
            al = _gs._normalize(rec.get("album") or "")
            label = None
            for row in idx.get((prof, _gs._normalize(rec.get("artist", ""))), []):
                lab = row.get("label")
                if lab not in ("good", "bad"):
                    continue
                if al:
                    lal = _gs._normalize(row.get("album") or "")
                    if lal and lal != al:
                        continue
                label = lab            # most recent decisive label wins
            if label == "good":
                good += 1
            elif label == "bad":
                bad += 1
        n = good + bad
        if n < min_decisive:
            continue
        frac = good / n
        pool = s.get("score")
        mismatch = (pool >= 4 and frac < 0.5) or (pool <= 2 and frac > 0.5)
        out.append({
            "ts": s.get("ts", ""), "profile": prof, "model": s.get("model", ""),
            "agent": s.get("agent", ""), "pool": pool,
            "good": good, "bad": bad, "good_frac": round(frac, 3),
            "n_decisive": n, "mismatch": mismatch,
        })
    return out
