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
      {name, good:[{artist,album,note}], bad:[...], meta:{...}}
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
) -> Path:
    """
    Record one human verdict. label is "good"/"bad"/"unknown".
    Appends a JSON line to eval/golden/labeled.jsonl and returns its path.
    """
    if label not in ("good", "bad", "unknown"):
        raise ValueError(f"label must be good/bad/unknown, got {label!r}")
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
