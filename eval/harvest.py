#!/usr/bin/env python3
"""
Tag harvesting (layer 2 of the untagged-rec plan) — turn the engine's output
into tags, with a human in the loop.

Why this exists
---------------
The golden sets' `tags` sections are how the structure score reads a rec's
genre/era — but only for artists someone has already tagged. Recs of untagged
artists score NEUTRAL, so as the engine recommends more and more artists, the
score would quietly stop measuring anything. This is the flywheel that keeps
it measuring:

    run evals  →  runs_history.jsonl (every rec + model self-report)
               →  this script: find the artists that keep coming back
               →  MusicBrainz lookup (offline-safe, sanitized)
               →  YOU review  →  paste the entry into eval/golden/<profile>.json
               →  next run, those recs are scored, not neutral

Safety model (read before running `lookup`)
-------------------------------------------
`lookup` calls MusicBrainz (free, no key). The response is NEVER passed to any
LLM. Only a strict whitelist of fields is kept, every string passes the same
sanitizer (allowed characters, length cap, no control characters), and the
result is a human-review artifact: printed JSON + a line appended to
eval/golden/tag_proposals.jsonl. If that contract ever changes, change it here
first.

Subcommands
-----------
  report [--min-count N] [--json PATH]
      OFFLINE. Aggregates eval/output/runs_history.jsonl: which artists recur,
      from which models/profiles, with which self-reported genre/era votes —
      and flags which are already tagged / in a good-bad list / already
      proposed. Sorted by frequency. This tells you who is worth looking up.

  lookup ARTIST [ARTIST ...] [--suggest] [--no-save]
      NETWORK (MusicBrainz). Search + detail (genres + community tags) per
      artist, strictly sanitized (see safety model). Prints JSON to stdout and
      appends to eval/golden/tag_proposals.jsonl (unless --no-save).
      --suggest also prints a paste-ready snippet for a golden file's `tags`.

      Usage:  python eval/harvest.py lookup "J Dilla" "Madlib" --suggest

Nothing here scores anything. Self-reported genre/era is consensus fuel, not
ground truth — a tag only becomes real when YOU commit it to a golden file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Load .env so --source lastfm works from a plain shell (LASTFM_API_KEY).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from afrec import lastfm as lastfm_mod                                  # noqa: E402
from afrec.sanitize import clean_label  # re-exported: the shared sanitizer   # noqa: E402

OUTPUT_DIR = HERE / "output"
GOLDEN_DIR = HERE / "golden"
HISTORY = OUTPUT_DIR / "runs_history.jsonl"
PROPOSALS = GOLDEN_DIR / "tag_proposals.jsonl"

MB_BASE = "https://musicbrainz.org/ws/2"
MB_USER_AGENT = "amongstfriends-recommender-eval/1.0 (offline tag-harvest tool; not a web app)"
MB_MIN_INTERVAL_S = 1.05   # keyless rate limit is ~1 req/sec
MB_TIMEOUT_S = 10

# ── Sanitizers (the ONLY path external text may take into output) ────────────
# clean_label now lives in afrec/sanitize.py (shared with the enrichment
# layer); it is imported at the top and re-exported for existing callers.
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def vote_key(s: str) -> str:
    """Fold a self-reported genre into a counting key so 'NeoSoul', 'neo-soul'
    and 'neo soul' vote for the same thing (display keeps the raw label)."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)   # camelCase → spaces
    return re.sub(r"[\s\-_]+", " ", s.lower()).strip()


def normalize_era(value) -> str | None:
    """Coerce an era read into the decade form the scorer uses ("2010s").
    Non-decade text is kept as a short clean label (e.g. "classic")."""
    s = (clean_label(value, maxlen=12) or "").lower()
    if re.fullmatch(r"\d{4}s?", s):    # "2010", "1998", "1979s" → floor to decade
        return s[:3] + "0s"
    if re.fullmatch(r"[\d ]+", s):     # digits that are not a decade: garbage
        return None
    return s or None


# ── Golden-set index (what's already known) ──────────────────────────────────

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


def load_golden_index() -> tuple[dict, dict]:
    """artist_key → {profile: tags-entry} and artist_key → ["good (prof)", ...]."""
    tags: dict[str, dict] = {}
    verdicts: dict[str, list[str]] = {}
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        if p.name in ("tag_proposals.jsonl", "labeled.jsonl", "scores.jsonl", "duels.jsonl"):
            continue
        try:
            g = json.loads(p.read_text())
        except Exception:
            continue
        prof = g.get("profile") or p.stem
        for a, t in (g.get("tags") or {}).items():
            tags.setdefault(a.lower().strip(), {})[prof] = t
        for side in ("good", "bad"):
            for e in g.get(side) or []:
                a = e["artist"] if isinstance(e, dict) else e
                if isinstance(a, str) and a.strip():
                    verdicts.setdefault(a.lower().strip(), []).append(f"{side} ({prof})")
    return tags, verdicts


# ── History (the cross-run memory written by eval/evaluate.py) ───────────────

def load_history() -> list[dict]:
    rows = []
    for entry in _load_jsonl(HISTORY):
        # Skip runs where the catalog verifier died mid-run (e.g. Spotify 429):
        # those recs are unverified passthroughs, not confirmed picks — feeding
        # them into tag proposals would pollute the human review queue.
        if entry.get("verification_error"):
            continue
        for rec in entry.get("recs") or []:
            rows.append({
                "ts": entry.get("ts", ""),
                "model": entry.get("model", ""),
                "profile": entry.get("profile", ""),
                "artist": (rec.get("artist") or "").strip(),
                "album": (rec.get("album") or "").strip(),
                "self_genre": rec.get("self_genre"),
                "self_era": rec.get("self_era"),
                "release_year": rec.get("release_year"),
            })
    return rows


def consensus(rows: list[dict]) -> dict:
    """Aggregate history by artist: counts, distinct models/profiles, and the
    models' self-reported genre/era votes. Self-reports are CONSENSUS FUEL,
    not ground truth — they are shown to help your judgment, never auto-kept."""
    agg: dict[str, dict] = {}
    for r in rows:
        k = r["artist"].lower()
        if not k:
            continue
        a = agg.setdefault(k, {
            "artist": r["artist"], "n": 0, "models": set(), "profiles": set(),
            "genres": Counter(), "eras": Counter(),
            "years": Counter(), "albums": Counter(),
        })
        a["n"] += 1
        if r["model"]:
            a["models"].add(r["model"])
        if r["profile"]:
            a["profiles"].add(r["profile"])
        g = clean_label(r.get("self_genre"))
        if g:
            a["genres"][vote_key(g)] += 1
        e = normalize_era(r.get("self_era"))
        if e:
            a["eras"][e] += 1
        if isinstance(r.get("release_year"), int):
            a["years"][r["release_year"]] += 1
        if r["album"]:
            a["albums"][r["album"]] += 1
    return agg


def _top(counter: Counter, n: int = 3) -> list[tuple[str, int]]:
    return counter.most_common(n)


def cmd_report(args) -> int:
    rows = load_history()
    if not rows:
        print(f"❌ No run history yet at {HISTORY}. Run `make eval` first "
              f"(it appends every rec there).")
        return 2
    agg = consensus(rows)
    tags_idx, verdicts = load_golden_index()
    proposed = {p.get("artist", "").lower() for p in _load_jsonl(PROPOSALS) if p.get("found")}

    def status_of(k: str) -> str:
        bits = []
        if k in tags_idx:
            bits.append("tagged: " + ",".join(sorted(tags_idx[k])))
        if k in verdicts:
            bits.append("; ".join(verdicts[k]))
        if k in proposed:
            bits.append("proposed")
        return "; ".join(bits) if bits else ""

    entries = []
    for k, a in agg.items():
        if a["n"] < args.min_count:
            continue
        entries.append({
            "artist": a["artist"],
            "n_recs": a["n"],
            "n_models": len(a["models"]),
            "models": sorted(a["models"]),
            "profiles": sorted(a["profiles"]),
            "self_genre_votes": _top(a["genres"]),
            "self_era_votes": _top(a["eras"]),
            "year_votes": [(str(y), c) for y, c in sorted(a["years"].items())],
            "albums": _top(a["albums"], 5),
            "status": status_of(k),
            "untagged": k not in tags_idx,
        })
    # Untagged first (that's the work), then by frequency.
    entries.sort(key=lambda e: (not e["untagged"], -e["n_recs"], e["artist"].lower()))

    if args.json:
        Path(args.json).write_text(json.dumps(entries, indent=2, ensure_ascii=False))
        print(f"📄 {len(entries)} artists → {args.json}")

    print(f"\n🌱 Tag harvest report — {len(rows)} recs in history, {len(entries)} artists "
          f"(≥{args.min_count} recs), untagged first\n")
    for e in entries:
        mark = " " if e["untagged"] else "✓"
        gens = ", ".join(f"{g}×{c}" for g, c in e["self_genre_votes"]) or "—"
        eras = ", ".join(f"{g}×{c}" for g, c in e["self_era_votes"]) or "—"
        years = ", ".join(f"{y}×{c}" for y, c in e["year_votes"]) or "—"
        print(f" {mark} {e['artist']:<28} {e['n_recs']:>2} recs, {e['n_models']} model(s)   "
              f"self-genre: {gens:<32} self-era: {eras:<14} years: {years}")
        if e["status"]:
            print(f"     → {e['status']}")
    print("\n   ✓ = already tagged in a golden set. Untagged rows are the candidates for")
    print(f"   `python eval/harvest.py lookup \"ARTIST\" --suggest`, then paste the result into")
    print(f"   eval/golden/<profile>.json `tags` (you are the commit step).")
    return 0


# ── MusicBrainz (network; strictly sanitized output) ─────────────────────────

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        return None


def _mb_get(path: str, params: dict) -> dict:
    url = f"{MB_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": MB_USER_AGENT,
                                               "Accept": "application/json"})
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(req, timeout=MB_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sanitize_mb(search: dict, detail: dict, fallback_name: str) -> dict:
    """Extract the WHITELIST of fields we keep from MusicBrainz, through the
    label sanitizer. Everything else in the API response is discarded —
    disambiguation text, URLs, life dates, types, free-form notes: never kept,
    never shown, never fed anywhere else.
    """
    top = (search or {}).get("artist") or {}
    mb_id = top.get("id") or (detail or {}).get("id") or ""
    if not _UUID.match(mb_id):
        mb_id = None
    genres = [clean_label(g.get("name")) for g in (detail or {}).get("genres") or []]
    genres = [g for g in genres if g][:12]
    raw_tags = (detail or {}).get("tags") or []
    raw_tags = sorted(raw_tags, key=lambda t: int(t.get("count") or 0), reverse=True)
    tags = [clean_label(t.get("name")) for t in raw_tags]
    tags = [t for t in tags if t][:12]
    score = top.get("score")
    return {
        "artist": clean_label(top.get("name")) or clean_label(fallback_name),
        "mb_id": mb_id,
        "search_score": score if isinstance(score, int) else None,
        "genres": genres,
        "tags": tags,
    }


def lookup_artist(name: str) -> dict:
    search = _mb_get("/artist", {"qs": f'artist:"{name}"', "fmt": "json", "limit": "3"})
    results = search.get("artists") or []
    if not results:
        return {"artist": clean_label(name), "found": False, "genres": [], "tags": []}
    top = results[0]
    time.sleep(MB_MIN_INTERVAL_S)
    try:
        detail = _mb_get(f"/artist/{top['id']}", {"inc": "genres+tags", "fmt": "json"})
    except Exception:
        detail = {}
    out = sanitize_mb({"artist": top}, detail, name)
    out["found"] = True
    return out


def lookup_artist_lastfm(name: str) -> dict:
    """Last.fm artist lookup, same output contract as lookup_artist.
    genres = top-3 tags (the coarse label), tags = top-12 by count.
    Coverage note: Last.fm is patchy on electronic artists — that's why
    --source auto still falls back to MusicBrainz."""
    info = lastfm_mod.artist_info(name, tag_limit=12)
    if not info:
        return {"artist": clean_label(name), "found": False, "genres": [], "tags": []}
    tags = [t for t, _c in info["tags"]]
    return {
        "artist": info["name"] or clean_label(name),
        "found": True,
        "genres": tags[:3],
        "tags": tags,
        "similar": info["similar"][:10],
        "tag_counts": info["tags"][:8],
    }


def merge_lookup(mb: dict, lf: dict) -> dict:
    """auto source: MusicBrainz genres + Last.fm tags into one proposal.
    Whatever one source misses, the other may supply; provenance kept per field."""
    out = {"artist": mb.get("artist") or lf.get("artist") or clean_label("?"), "found": False}
    genres = list(mb.get("genres") or [])
    for g in lf.get("genres") or []:
        if g.lower() not in {x.lower() for x in genres}:
            genres.append(g)
    tags = list(mb.get("tags") or [])
    for g in lf.get("tags") or []:
        if g.lower() not in {x.lower() for x in tags}:
            tags.append(g)
    out.update({
        "found": bool(mb.get("found") or lf.get("found")),
        "mb_id": mb.get("mb_id"),
        "genres": genres[:12],
        "tags": tags[:12],
        "similar": lf.get("similar", []),
        "tag_counts": lf.get("tag_counts", []),
        "mb_found": bool(mb.get("found")),
        "lastfm_found": bool(lf.get("found")),
    })
    return out


def cmd_lookup(args) -> int:
    names = args.artist
    results = []
    for i, name in enumerate(names):
        if i:
            time.sleep(MB_MIN_INTERVAL_S)
        try:
            if args.source == "lastfm":
                print(f"🔎 Last.fm: {name} ...", file=sys.stderr)
                out = lookup_artist_lastfm(name)
                out["source"] = "lastfm"
            elif args.source == "musicbrainz":
                print(f"🔎 MusicBrainz: {name} ...", file=sys.stderr)
                out = {**lookup_artist(name), "source": "musicbrainz"}
            else:  # auto: Last.fm first (cheap, 50/s), MusicBrainz fills gaps
                print(f"🔎 Last.fm + MusicBrainz: {name} ...", file=sys.stderr)
                lf = {"artist": clean_label(name), "found": False, "genres": [], "tags": []}
                if lastfm_mod.api_key():
                    try:
                        lf = lookup_artist_lastfm(name)
                    except lastfm_mod.LastfmError as e:
                        print(f"   ⚠️ Last.fm: {e} (falling back to MusicBrainz)", file=sys.stderr)
                else:
                    print("   (no LASTFM_API_KEY — MusicBrainz only)", file=sys.stderr)
                time.sleep(MB_MIN_INTERVAL_S)
                mb = lookup_artist(name)
                out = {**merge_lookup(mb, lf), "source": "merged"}
        except Exception as e:
            out = {"artist": clean_label(name), "found": False, "source": args.source, "error": str(e)[:200]}
            print(f"   ⚠️ {out['error']}", file=sys.stderr)
        results.append(out)
        print(json.dumps(out, indent=2, ensure_ascii=False))

    if not args.no_save:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with PROPOSALS.open("a", encoding="utf-8") as f:
            for out in results:
                f.write(json.dumps({"ts": ts, **out}, ensure_ascii=False) + "\n")
        print(f"\n📄 {len(results)} proposal(s) appended → {PROPOSALS}", file=sys.stderr)

    if args.suggest:
        rows = load_history()
        agg = consensus(rows)
        tags_idx, _ = load_golden_index()
        snippet = {}
        for out in results:
            a = out.get("artist")
            if not a or not out.get("found"):
                continue
            k = a.lower()
            if k in tags_idx:
                print(f"   (skip '{a}' — already tagged in a golden set)", file=sys.stderr)
                continue
            entry: dict = {}
            genres = out.get("genres") or []
            if genres:
                entry["genres"] = {"coarse": genres[0]}
            era = normalize_era(agg.get(k, {}).get("eras").most_common(1)[0][0]
                                if agg.get(k, {}).get("eras") else None)
            if era:
                entry["era"] = era
            snippet[a] = entry or {}
        if snippet:
            print('\nPaste-ready for eval/golden/<profile>.json "tags" (review before committing):')
            print(json.dumps(snippet, indent=2, ensure_ascii=False))
        else:
            print("(nothing new to suggest)", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Turn the engine's recurring output into golden-set tags (human-reviewed).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("report", help="OFFLINE: which artists recur across runs (untagged first)")
    p_rep.add_argument("--min-count", type=int, default=1, help="only artists recommended ≥ N times (default 1)")
    p_rep.add_argument("--json", metavar="PATH", help="also write the report to this path")
    p_rep.set_defaults(fn=cmd_report)

    p_lk = sub.add_parser("lookup", help="NETWORK: sanitized genre/tag proposal per artist")
    p_lk.add_argument("artist", nargs="+", help="artist name(s) to look up")
    p_lk.add_argument("--source", choices=("auto", "lastfm", "musicbrainz"), default="auto",
                      help="lastfm (tags/counts), musicbrainz (genres/tags), auto (both merged; default)")
    p_lk.add_argument("--suggest", action="store_true",
                      help="also print a paste-ready golden-file tags snippet")
    p_lk.add_argument("--no-save", action="store_true",
                      help="don't append to eval/golden/tag_proposals.jsonl")
    p_lk.set_defaults(fn=cmd_lookup)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
