"""Human-labeled "golden sets" + an interactive labeling corpus.

Two related things live here, kept separate on purpose:

1. STATIC GOLDEN SETS  (eval/golden/<profile>.json)
   A human says, for a given taste profile, which artists (and optionally
   albums) are clearly GOOD picks vs BAD picks. These are model-independent,
   so any model's output can be scored against the same yardstick and the
   numbers are directly comparable in the leaderboard.
   → score_recs_against_golden()

2. INTERACTIVE CORPUS  (eval/golden/labeled.jsonl, append-only)
   When you run `python eval/evaluate.py --interactive`, the model generates recs, you
   label what it actually produced as good/bad, and each verdict is appended
   here with the model + timestamp. Over time this becomes a growing,
   model-attributable record of "what humans thought of what this model
   suggested" — the thing you can later learn from.
   → append_label(), labeled_stats()

Schema is deliberately permissive so it's easy to hand-edit (see
eval/golden/README.md). Each entry is EITHER a bare string
   "Nas"
OR an object
   {"artist": "Nas", "album": "God's Son", "note": "deep cut, boom-bap",
    "tier": "expected" | "novel",   # optional, default "expected"
    "pair": "canon-vs-filler"}       # optional, contrast-pair name

Beyond the base good/bad score, two signals are computed from the extra
fields (all optional — old-style files score exactly as before):
  • PAIRS  — entries sharing a `pair` name form a contrast pair (one good-
    side entry + one bad-side entry that are near-identical except for the
    subtlety the listener splits on). A pair scores +1 if the engine recs
    the good side only, 0 if neither side, -1 if it recs the bad side.
  • NOVEL  — good entries with `tier: "novel"` are original moves for the
    listener (adjacent scene / cross-genre bridge). We track how many get
    hit: the 'originality' signal, separate from safe in-lane hits.

3. STRUCTURE (response-level, still model-independent) — three optional
top-level sections in the golden file, scored by score_response_structure():
  • tags                   — hand-annotated artist facts: genre levels
                             ({"coarse": "hiphop", "fine": "jazz hiphop"}) and
                             era ("2000s"). Recs the model produces are looked
                             up here; untagged recs are NEUTRAL (excluded from
                             denominators), and tag coverage is reported.
  • constraints            — named checks on the RESPONSE as a whole (genre
                             spread, era spread, in-lane share). The principle
                             is MIRROR, not diversity: the checks describe the
                             fixture's own composition. Each check has a
                             weight (default 1) — structure = weighted pass
                             rate. See README.
  • justification_keywords — a rec's `reason` containing any keyword scores a
                             'just' hit (informational only; never in quality).
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path

# Canonical name normalization for matching — shared by the evaluator and the
# golden scorer so a model's rec and a human's entry line up the same way.
def _normalize(s) -> str:
    return " ".join((str(s) if s is not None else "").lower().replace("&", "and").split())


# This file lives at eval/goldenset.py, so the golden DATA directory is a
# sibling subfolder: eval/golden/ (never name the module eval/golden.py — it
# would collide with the golden/ data folder at import time).
HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"
LABELED_LOG = GOLDEN_DIR / "labeled.jsonl"


# ── Normalization ─────────────────────────────────────────────────────────────

def _norm_entry(entry) -> dict:
    """Coerce a bare string or dict entry into a normalized dict.

    Fields: artist, album?, note?, tier ("expected"|"novel"), pair?
    """
    if isinstance(entry, str):
        return {"artist": entry.strip(), "note": "", "tier": "expected", "pair": ""}
    if isinstance(entry, dict):
        tier = str(entry.get("tier") or "expected").strip().lower()
        return {
            "artist": str(entry.get("artist") or "").strip(),
            "album": str(entry.get("album") or "").strip(),
            "note": str(entry.get("note") or entry.get("reason") or "").strip(),
            "tier": tier if tier in ("expected", "novel") else "expected",
            "pair": str(entry.get("pair") or "").strip(),
        }
    return {"artist": "", "note": "", "tier": "expected", "pair": ""}


def _norm_artists(entries) -> list[dict]:
    out = []
    for e in entries or []:
        d = _norm_entry(e)
        if d.get("artist"):
            out.append(d)
    return out


# ── Static golden sets ────────────────────────────────────────────────────────

def golden_path_for(profile_name: str) -> Path:
    return GOLDEN_DIR / f"{profile_name}.json"


def load_golden_for(profile_name: str) -> dict | None:
    """
    Load one static golden set by profile name. Returns a dict:
      {name, good:[{artist,album,note}], bad:[...], meta:{...},
       tags:{artist:{genres:{coarse,fine},era}}, constraints:[...],
       justification_keywords:[...]}
    or None if no golden file exists for that profile.
    """
    p = golden_path_for(profile_name)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    return {
        "name": data.get("profile", profile_name),
        "description": data.get("description", ""),
        "annotated_by": data.get("annotated_by", ""),
        "good": _norm_artists(data.get("good", [])),
        "bad": _norm_artists(data.get("bad", [])),
        "meta": {k: data.get(k) for k in ("annotated_at", "notes", "reviewed") if k in data},
        "tags": _norm_tags(data.get("tags", {})),
        "constraints": _norm_constraints(data.get("constraints", [])),
        "justification_keywords": [str(k).strip().lower()
                                   for k in (data.get("justification_keywords") or [])
                                   if str(k).strip()],
    }


# ── Structure: tags + response-level constraints ─────────────────────────────

_LEVELS = ("coarse", "fine")


def _norm_tags(raw) -> dict:
    """Normalize the per-artist tag table: {norm_artist: {genres, era, note}}.

    genres is permissive: a dict of levels (canonical), a bare string
    (→ {"coarse": s}), or a list (first → coarse, second → fine).
    """
    out = {}
    if not isinstance(raw, dict):
        return out
    for artist, t in raw.items():
        if not isinstance(t, dict):
            t = {"genres": t}
        g = t.get("genres", {})
        if isinstance(g, str):
            g = {"coarse": g}
        elif isinstance(g, (list, tuple)):
            g = {lvl: str(v) for lvl, v in zip(_LEVELS, g) if v}
        elif not isinstance(g, dict):
            g = {}
        genres = {lvl: str(g[lvl]).strip().lower()
                  for lvl in _LEVELS if str(g.get(lvl) or "").strip()}
        out[_normalize(artist)] = {
            "genres": genres,
            "era": str(t.get("era") or "").strip().lower(),
            "note": str(t.get("note") or "").strip(),
        }
    return out


def _norm_constraints(raw) -> list[dict]:
    """Normalize the response-level constraint list. Unknown check names are
    kept but flagged (scored as skipped, not failed)."""
    known = {"min_distinct_genres", "max_single_genre_share",
             "min_lane_share", "lane_balance", "min_distinct_decades"}
    out = []
    for c in raw or []:
        if not isinstance(c, dict):
            continue
        check = str(c.get("check") or "").strip().lower()
        if not check:
            continue
        try:
            weight = float(c.get("weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        out.append({
            "check": check,
            "known": check in known,
            "value": c.get("value"),
            "weight": max(0.0, weight),
            "level": str(c.get("level") or "").strip().lower() or None,
            "lane": str(c.get("lane") or "").strip().lower(),
            "area": str(c.get("area") or "").strip().lower(),
            "reason": str(c.get("reason") or "").strip(),
        })
    return out


def _decade_of(year) -> str | None:
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    if not (1900 <= y <= 2100):
        return None
    return f"{(y // 10) * 10}s"


def _rec_genre(rec, tags: dict, level: str) -> str | None:
    """The rec's genre at `level` from the tag table, or None (untagged)."""
    t = tags.get(_normalize(getattr(rec, "artist", "") or ""))
    if not t:
        return None
    return t["genres"].get(level)


def _rec_era(rec, tags: dict) -> str | None:
    """Decade for a rec: Spotify release_year first, hand tag second."""
    d = _decade_of(getattr(rec, "release_year", None))
    if d:
        return d
    t = tags.get(_normalize(getattr(rec, "artist", "") or ""))
    if t and t["era"]:
        return t["era"]
    return None


def _lane_credit(rec, tags: dict, lane: str, level: str, area: str = "") -> float | None:
    """Credit for one rec against min_lane_share.

    1.0  — tagged with `lane` at the tested level (or, for level=fine,
           tagged with `lane` at coarse)
    0.5  — tagged in the constraint's `area` (the coarse genre adjacent to
           the lane, e.g. lane "jazz hiphop" / area "hiphop")
    0.0  — tagged, but outside the area entirely
    None — untagged (neutral: excluded from the share)
    """
    t = tags.get(_normalize(getattr(rec, "artist", "") or ""))
    if not t or not t["genres"]:
        return None
    genres = t["genres"]
    if genres.get(level) == lane or (level == "fine" and genres.get("coarse") == lane):
        return 1.0
    if area and genres.get("coarse") == area:
        return 0.5
    return 0.0


def score_response_structure(recs, golden: dict, fixture_names: list[str] | None = None) -> dict:
    """
    Score the RESPONSE as a whole against the golden set's optional structure
    sections (tags / constraints / justification_keywords).

    - `fixture_names` (normalized artist/album names from the fixture's
      reviews) enables the "justification cites the fixture" signal.

    Returns:
      checks            — {name: {pass, observed, target, weight, reason}}
      points / total    — weighted passes / weighted scoreable checks
      score             — points / total (None if no scoreable checks)
      lane_share        — observed min_lane_share value, if that check exists
      lane_balance      — observed lane_balance value, if that check exists
      tagged / total_recs / tag_coverage
      just_hits / just_total / just_score     — keyword hits in rec `reason`s
      just_fixture_hits — recs whose reason names something from the fixture
    """
    # Normalize defensively: works on raw file dicts or on load_golden_for output.
    tags: dict = _norm_tags(golden.get("tags") or {})
    constraints: list[dict] = _norm_constraints(golden.get("constraints") or [])
    keywords: list[str] = [str(k).strip().lower()
                           for k in (golden.get("justification_keywords") or [])
                           if str(k).strip()]

    total_recs = len(recs)
    tagged = sum(1 for r in recs if _normalize(getattr(r, "artist", "") or "") in tags)

    checks: dict[str, dict] = {}
    used_names: set[str] = set()
    points = total = 0.0
    lane_share = None
    lane_balance_score = None

    for c in constraints:
        name = c["check"]
        i = 2
        while name in used_names:
            name = f"{c['check']}-{i}"
            i += 1
        used_names.add(name)
        entry = {"pass": None, "observed": None, "target": c["value"],
                 "weight": c["weight"], "reason": c["reason"]}

        kind = c["check"]
        graded = False
        if not c["known"]:
            entry["pass"] = None  # unknown check: skip, don't fail
        elif c["value"] is None and kind != "lane_balance":
            entry["pass"] = False  # known check, no target: can't score
        else:
            observed = None
            if kind == "min_distinct_genres":
                lvl = c["level"] or "coarse"
                genres = [_rec_genre(r, tags, lvl) for r in recs]
                genres = [g for g in genres if g]
                if genres:
                    observed = len(set(genres))
                    entry["pass"] = observed >= c["value"]
                else:
                    entry["pass"] = None  # nothing tagged: unmeasurable, not failed
            elif kind == "max_single_genre_share":
                lvl = c["level"] or "coarse"
                genres = [_rec_genre(r, tags, lvl) for r in recs]
                genres = [g for g in genres if g]
                if genres:
                    top = max((sum(1 for g in genres if g == x) for x in set(genres)))
                    observed = round(top / len(genres), 3)
                    entry["pass"] = observed <= c["value"]
                else:
                    entry["pass"] = None  # nothing tagged: unmeasurable, not failed
            elif kind == "min_lane_share":
                lvl = c["level"] or "fine"
                credits = [_lane_credit(r, tags, c["lane"], lvl, c.get("area") or "") for r in recs]
                credits = [x for x in credits if x is not None]
                if credits:
                    observed = round(sum(credits) / len(credits), 3)
                    entry["pass"] = observed >= c["value"]
                    if c["value"] is not None:
                        lane_share = observed
                else:
                    entry["pass"] = None  # nothing tagged: unmeasurable, not failed
            elif kind == "min_distinct_decades":
                eras = [_rec_era(r, tags) for r in recs]
                eras = [e for e in eras if e]
                if eras:
                    observed = len(set(eras))
                    entry["pass"] = observed >= c["value"]
                else:
                    entry["pass"] = None  # no known eras: unmeasurable, not failed
            elif kind == "lane_balance":
                # Per-rec NET lane alignment (graded, not binary): each TAGGED
                # rec scores +1 in the lane, +0.5 in the adjacent area, −1
                # tagged OUT of the area; untagged recs are ignored. The mean
                # (∈ [−1, +1]) is mapped to [0, 1] and earns PARTIAL credit —
                # an out-of-lane rec actively subtracts, which a share check
                # can't express.
                pts = []
                for r in recs:
                    credit = _lane_credit(r, tags, c["lane"], c["level"] or "fine",
                                          c.get("area") or "")
                    if credit is None:
                        continue
                    pts.append(1.0 if credit >= 1.0 else (0.5 if credit >= 0.5 else -1.0))
                if pts:
                    avg = sum(pts) / len(pts)                    # ∈ [−1, +1]
                    observed = round((avg + 1.0) / 2.0, 3)       # ∈ [0, 1]
                    thr = float(c["value"]) if c["value"] is not None else 0.5
                    entry["pass"] = observed >= thr
                    entry["detail"] = {
                        "in_lane": sum(1 for p in pts if p == 1.0),
                        "in_area": sum(1 for p in pts if p == 0.5),
                        "out": sum(1 for p in pts if p < 0),
                        "untagged": len(recs) - len(pts),
                    }
                    lane_balance_score = observed
                    graded = True
                else:
                    entry["pass"] = None  # nothing tagged: unmeasurable, not failed
                    entry["detail"] = {"untagged": len(recs)}
            entry["observed"] = observed

        if entry["pass"] is not None and c["weight"] > 0:
            total += c["weight"]
            if graded:
                points += c["weight"] * (entry["observed"] or 0.0)  # partial credit
            elif entry["pass"]:
                points += c["weight"]
        checks[name] = entry

    # Justification keywords (informational — never folded into score/quality).
    just_hits = just_fixture_hits = 0
    if total_recs:
        fx = [_normalize(n) for n in (fixture_names or []) if len(_normalize(n)) >= 3]
        for r in recs:
            reason = _normalize(getattr(r, "reason", "") or "")
            if reason and any(k in reason for k in keywords):
                just_hits += 1
            if reason and fx and any(n in reason for n in fx):
                just_fixture_hits += 1

    return {
        "checks": checks,
        "points": round(points, 3),
        "total": round(total, 3),
        "score": round(points / total, 3) if total else None,
        "lane_share": lane_share,
        "lane_balance": lane_balance_score,
        "tagged": tagged,
        "total_recs": total_recs,
        "tag_coverage": round(tagged / total_recs, 3) if total_recs else None,
        "just_hits": just_hits,
        "just_total": total_recs,
        "just_score": round(just_hits / total_recs, 3) if total_recs else None,
        "just_fixture_hits": just_fixture_hits,
    }


def score_recs_against_golden(recs, golden: dict) -> dict:
    """
    How well did a model's recs match a human's good/bad list?

    Matching is artist-level first (robust to album choice); if a specific
    (artist, album) pair is on a list, the pair wins (used to disambiguate an
    artist who has both a good and a bad album on the list).

    Returns:
      good_hits / bad_hits / score — the base score (see README)
      novel_hits / novel_total     — how many tier:"novel" good entries got hit
      expected_hits                — good hits on non-novel entries
      pairs                        — {name: {good_hit, bad_hit, score}} for each
                                      contrast pair; score is +1 good-side only,
                                      0 neither, -1 bad-side hit
      pair_points / pair_total     — sum of pair scores / number of pairs
      pair_score                   — pair_points / pair_total (or None)
      details                      — per-rec verdicts, for transparency
    """
    good_entries = [g for g in golden["good"] if g.get("artist")]
    bad_entries = [b for b in golden["bad"] if b.get("artist")]

    good_artists = {_normalize(g["artist"]) for g in good_entries}
    bad_artists = {_normalize(b["artist"]) for b in bad_entries}
    good_pairs = {(_normalize(g["artist"]), _normalize(g.get("album", "")))
                  for g in good_entries if g.get("album")}
    bad_pairs = {(_normalize(b["artist"]), _normalize(b.get("album", "")))
                 for b in bad_entries if b.get("album")}
    ambiguous = good_artists & bad_artists  # artist appears on both lists

    # Index entries so a hit can be attributed to a specific entry (for
    # tier/pair tracking). Key: (artist, album-or-None).
    good_by_key = {}
    for g in good_entries:
        good_by_key.setdefault((_normalize(g["artist"]), _normalize(g.get("album", "")) or None), g)
    bad_by_key = {}
    for b in bad_entries:
        bad_by_key.setdefault((_normalize(b["artist"]), _normalize(b.get("album", "")) or None), b)

    pair_state: dict[str, dict] = {}
    for e in (*good_entries, *bad_entries):
        if e.get("pair"):
            pair_state.setdefault(e["pair"], {"good_hit": False, "bad_hit": False})

    good_hits = bad_hits = novel_hits = 0
    details = []
    for r in recs:
        a = _normalize(getattr(r, "artist", "")) or ""
        al = _normalize(getattr(r, "album", "")) or ""
        verdict, entry = None, None

        # 1) exact (artist, album) pair is most specific
        if (a, al) in good_pairs:
            verdict, entry = "good", good_by_key[(a, al)]
        elif (a, al) in bad_pairs:
            verdict, entry = "bad", bad_by_key[(a, al)]
        # 2) fall back to artist-level, but skip artists that are both good & bad
        elif a in good_artists and a not in ambiguous:
            verdict, entry = "good", good_by_key.get((a, None))
        elif a in bad_artists and a not in ambiguous:
            verdict, entry = "bad", bad_by_key.get((a, None))

        if not verdict:
            continue
        if verdict == "good":
            good_hits += 1
            if entry and entry.get("tier") == "novel":
                novel_hits += 1
        else:
            bad_hits += 1
        if entry and entry.get("pair"):
            pair_state[entry["pair"]]["good_hit" if verdict == "good" else "bad_hit"] = True
        details.append({
            "artist": r.artist, "album": r.album, "verdict": verdict,
            "tier": entry.get("tier") if entry else None,
            "pair": entry.get("pair") if entry else None,
        })

    pairs = {}
    for name, st in pair_state.items():
        pairs[name] = {
            "good_hit": st["good_hit"],
            "bad_hit": st["bad_hit"],
            "score": -1 if st["bad_hit"] else (1 if st["good_hit"] else 0),
        }
    pair_points = sum(p["score"] for p in pairs.values())
    pair_total = len(pairs)
    novel_total = sum(1 for g in good_entries if g.get("tier") == "novel")

    n = len(recs) or 1
    score = round((good_hits - bad_hits) / n, 3)
    return {
        "good_hits": good_hits,
        "bad_hits": bad_hits,
        "score": max(-1.0, min(1.0, score)),
        "novel_hits": novel_hits,
        "novel_total": novel_total,
        "expected_hits": good_hits - novel_hits,
        "pairs": pairs,
        "pair_points": pair_points,
        "pair_total": pair_total,
        "pair_score": round(pair_points / pair_total, 3) if pair_total else None,
        "details": details,
    }


# ── Interactive corpus (append-only JSONL) ────────────────────────────────────

def append_label(
    *,
    profile: str,
    artist: str,
    album: str = "",
    label: str,                 # "good" | "bad" | "unknown"
    model: str = "",
    agent: str = "",            # engine playbook that generated the rec (e.g. "artist-then-album@v1")
    note: str = "",
    reason: str = "",
    source: str = "interactive",
    pool_score: int | None = None,
) -> Path:
    """
    Record one human verdict. label is "good"/"bad"/"unknown".
    Appends a JSON line to eval/golden/labeled.jsonl and returns its path.

    `source` tags where the label came from: "interactive" (evaluate.py
    --interactive), "pending-judge" (taste.py pending --judge), "score" /
    "duel" (live per-rec grading in taste.py), "grade" (offline backfill,
    taste.py grade).
    `pool_score` (1–5) is stored when the label was made while judging a whole
    scored/duelled pool — it enables the offline pool-vs-item consistency
    check (humaneval.pool_item_consistency).
    """
    if label not in ("good", "bad", "unknown"):
        raise ValueError(f"label must be good/bad/unknown, got {label!r}")
    if pool_score is not None and not (isinstance(pool_score, int)
                                       and 1 <= pool_score <= 5):
        raise ValueError(f"pool_score must be an int 1..5 or None, got {pool_score!r}")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "profile": profile,
        "artist": (artist or "").strip(),
        "album": (album or "").strip(),
        "label": label,
        "model": (model or "").strip(),
        "agent": (agent or "").strip(),
        "note": (note or "").strip(),
        "reason": (reason or "").strip(),
        "source": source,
    }
    if pool_score is not None:
        rec["pool_score"] = pool_score
    with LABELED_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return LABELED_LOG


def load_labeled() -> list[dict]:
    if not LABELED_LOG.exists():
        return []
    out = []
    for line in LABELED_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def labeled_stats(model: str | None = None, profile: str | None = None,
                  agent: str | None = None) -> dict:
    """
    Aggregate the interactive corpus.

    good_rate = good / (good + bad) over DECISIVE labels (ignores "unknown").
    When model/profile/agent are given, restrict to those; otherwise all.
    """
    rows = load_labeled()
    if model:
        rows = [r for r in rows if (r.get("model") or "") == model]
    if profile:
        rows = [r for r in rows if (r.get("profile") or "") == profile]
    if agent:
        rows = [r for r in rows if (r.get("agent") or "") == agent]

    good = sum(1 for r in rows if r.get("label") == "good")
    bad = sum(1 for r in rows if r.get("label") == "bad")
    unknown = sum(1 for r in rows if r.get("label") == "unknown")
    decisive = good + bad
    return {
        "good": good,
        "bad": bad,
        "unknown": unknown,
        "total": good + bad + unknown,
        "good_rate": round(good / decisive, 3) if decisive else None,
    }


def labeled_for(profile: str, artist: str, album: str = "") -> list[dict]:
    """
    Every labeled.jsonl row for this profile + artist (album must also match
    when given). Matching is case/whitespace-insensitive. Used by the grade
    flows to show — and default-skip — recs you already judged.
    """
    a = _normalize(artist)
    al = _normalize(album)
    out = []
    for r in load_labeled():
        if r.get("profile") != profile:
            continue
        if _normalize(r.get("artist", "")) != a:
            continue
        if al and _normalize(r.get("album", "")) != al:
            continue
        out.append(r)
    return out


def promotion_candidates(min_good: int = 2, min_bad: int = 2,
                         max_per_list: int = 8) -> dict[str, dict]:
    """
    Artists you have now decisively labeled at least `min_good` (good) OR
    `min_bad` (bad) times in labeled.jsonl but that are NOT yet in that
    profile's golden file — i.e. ready to be promoted into the golden set
    (`make harvest`).

    Returns {profile: {"good": [entries], "bad": [entries]}} where each entry
    is {"artist": str, "albums": [str], "good": int, "bad": int}, sorted by
    (good - bad) desc then total labels desc, truncated to max_per_list.
    An artist can appear in both lists (you've seen it both ways — review
    before harvesting). Profiles with no candidates are omitted.
    """
    if not LABELED_LOG.exists():
        return {}
    per: dict[tuple, dict] = {}
    for r in load_labeled():
        lab = r.get("label")
        if lab not in ("good", "bad"):
            continue
        prof = r.get("profile") or ""
        if not prof:
            continue
        key = (prof, _normalize(r.get("artist", "")))
        e = per.setdefault(key, {"artist": (r.get("artist") or "").strip(),
                                 "albums": [], "good": 0, "bad": 0})
        e[lab] += 1
        al = (r.get("album") or "").strip()
        if al and al not in e["albums"]:
            e["albums"].append(al)

    out: dict[str, dict] = {}
    for (prof, _), e in per.items():
        golden = load_golden_for(prof)
        if golden and any(_normalize(g.get("artist", "")) == _normalize(e["artist"])
                          for g in [*golden["good"], *golden["bad"]]):
            continue
        if e["good"] >= min_good:
            out.setdefault(prof, {"good": [], "bad": []})
            out[prof]["good"].append(e)
        if e["bad"] >= min_bad:
            out.setdefault(prof, {"good": [], "bad": []})
            out[prof]["bad"].append(e)
    for prof in out:
        for kind in ("good", "bad"):
            out[prof][kind].sort(key=lambda x: (-(x["good"] - x["bad"]),
                                                -(x["good"] + x["bad"])))
            out[prof][kind] = out[prof][kind][:max_per_list]
    return out
