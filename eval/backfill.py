#!/usr/bin/env python3
"""
Paced backfill of the enrichment cache (afrec/cache/enrichment.json).

The free Musicae tier is 10 calls/DAY (5 track IDs/call). This script is
the thing YOU run whenever to grow the cache — fixtures, golden sets, and
everything the engine has recommended so far — at a pace you control:

    python eval/backfill.py status                 # offline: what's cached, what's missing
    python eval/backfill.py                        # backfill: Last.fm (free) + up to budget Musicae calls
    python eval/backfill.py --budget 3 --delay 30  # be gentle: 3 Musicae calls, 30s apart
    python eval/backfill.py --dry-run              # print the plan, fire nothing
    python eval/backfill.py --lastfm-only          # tags/years only (no quota spent)
    python eval/backfill.py smoke                  # 1 Last.fm + 1 Musicae call: verify keys/shape

What it enriches (unique artist+album pairs, in this order):
    --sources fixtures    the 7 eval fixture pools (what agents actually see)
    --sources golden      the good/bad artists of the golden sets
    --sources history     every rec in eval/output/runs_history.jsonl
                          (skips verification_error runs, same rule as harvest)

Pace control:
    --budget N            max Musicae calls THIS run (default: $MUSICAE_CALLS_PER_DAY
                          or 10). Last.fm is 50/s — never budgeted here.
    --delay S             seconds to sleep between Musicae calls (default 1).
                          Run the script multiple times across days to
                          outlast the free tier; the cache makes each run
                          strictly additive (nothing is ever re-fetched).

Nothing here talks to an LLM; every external string is clean_label-sanitized.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# ── .env (SPOTIFY_*, LASTFM_API_KEY, MUSICAE_API_KEY, …) ─────────────────────
def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env)
    except ImportError:
        import os
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

from afrec import enrich, lastfm, musicae, spotify          # noqa: E402

FIXTURES = HERE / "fixtures"
GOLDEN = HERE / "golden"
HISTORY = HERE / "output" / "runs_history.jsonl"


def pairs_from_fixtures() -> list[tuple[str, str]]:
    out = []
    for p in sorted(FIXTURES.glob("*.json")):
        try:
            fx = json.loads(p.read_text())
        except Exception:
            continue
        for r in fx.get("reviews") or []:
            cat = (r.get("catalog") or {}).get("metadata") or {}
            artist = (cat.get("artistName") or r.get("artist") or "").strip()
            album = (cat.get("albumName") or r.get("album") or "").strip()
            if artist and album:
                out.append((artist, album))
    return out


def pairs_from_golden() -> list[tuple[str, str]]:
    """(artist, album-or-first-tag) — golden entries are artist-keyed; the
    Enricher's (artist, album) contract still works when album == artist
    (Last.fm artist tags + year carry the signal; Musicae needs a real
    album, so these mostly land in the tags half until you add albums)."""
    out = []
    for p in sorted(GOLDEN.glob("*.json")):
        if p.suffix != ".json":
            continue
        try:
            g = json.loads(p.read_text())
        except Exception:
            continue
        for side in ("good", "bad"):
            for e in g.get(side) or []:
                if not isinstance(e, dict):
                    continue
                artist = (e.get("artist") or "").strip()
                if not artist:
                    continue
                album = (e.get("album") or "").strip()
                out.append((artist, album or artist))
    return out


def pairs_from_history() -> list[tuple[str, str]]:
    out = []
    if not HISTORY.exists():
        return out
    for line in HISTORY.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("verification_error"):     # unverified passthroughs: don't enrich
            continue
        for rec in entry.get("recs") or []:
            artist = (rec.get("artist") or "").strip()
            album = (rec.get("album") or "").strip()
            if artist and album:
                out.append((artist, album))
    return out


SOURCES = {
    "fixtures": pairs_from_fixtures,
    "golden": pairs_from_golden,
    "history": pairs_from_history,
}


def collect(names: list[str]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out = []
    for name in names:
        for artist, album in SOURCES[name]():
            k = enrich.name_key(artist, album)
            if k in seen:
                continue
            seen.add(k)
            out.append((artist, album))
    return out


def _coverage(pairs: list[tuple[str, str]], cache: enrich.Cache) -> tuple[int, int, int]:
    tags_n = feat_n = 0
    for artist, album in pairs:
        e = cache.get(enrich.name_key(artist, album)) or {}
        if e.get("tags"):
            tags_n += 1
        tid = e.get("track_id")
        if tid and (cache.get(f"track:{tid}") or {}).get("features"):
            feat_n += 1
    return len(pairs), tags_n, feat_n


def cmd_status(args) -> int:
    cache = enrich.Cache()
    names = args.sources.split(",")
    pairs = collect(names)
    total, t, f = _coverage(pairs, cache)
    print(f"\n📦 Enrichment cache — {len(cache._data)} entries on disk\n")
    for name in names:
        ps = collect([name])
        n, tn, fn = _coverage(ps, cache)
        print(f"  {name:<9} {n:>4} album(s)   tags {tn:>3}/{n}   features {fn:>3}/{n}")
    n, tn, fn = _coverage(pairs, cache)
    print(f"  {'(deduped)':<9} {n:>4} album(s)   tags {tn:>3}/{n}   features {fn:>3}/{n}")
    missing = n - fn
    if missing:
        calls = (missing + musicae.BATCH_SIZE - 1) // musicae.BATCH_SIZE
        days = (calls + 9) // 10
        print(f"\n  → {missing} album(s) still need features = {calls} Musicae call(s) "
              f"≈ {days} day(s) at the free 10/day. Run `python eval/backfill.py` whenever.")
    return 0


def cmd_backfill(args) -> int:
    pairs = collect(args.sources.split(","))
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        print("❌ No (artist, album) pairs found in the selected sources.")
        return 2

    enricher = enrich.Enricher(
        budget=args.budget,
        delay=args.delay,
        cache_path=Path(args.cache) if args.cache else enrich.CACHE_PATH,
    )
    if not enricher.available:
        print("❌ No keys: set LASTFM_API_KEY and/or MUSICAE_API_KEY in .env.")
        return 2
    if args.lastfm_only:
        enricher._has_musicae = False     # tags/years only — zero Musicae quota
    if args.musicae_only:
        enricher._has_lastfm = False      # features only — zero Last.fm calls

    before = _coverage(pairs, enricher.cache)
    if args.dry_run:
        print(f"\n🔎 dry run — {before[0]} unique album(s) from [{args.sources}]")
        print(f"   already cached: tags {before[1]}, features {before[2]}")
        print(f"   Musicae budget this run: {enricher.budget} call(s), delay {args.delay}s")
        print("   (no calls fired; re-run without --dry-run to backfill)")
        return 0

    t0 = time.time()
    print(f"⏳ Backfilling {len(pairs)} album(s) — budget {enricher.budget} Musicae call(s), "
          f"{args.delay}s between calls (Ctrl-C safe: cache saves per batch)")
    enricher.fact_map(pairs)
    after = _coverage(pairs, enricher.cache)
    print(f"\n✅ Done in {time.time() - t0:.0f}s — features {before[2]}→{after[2]}/{after[0]}, "
          f"tags {after[1]}/{after[0]}, Musicae calls used this run: {enricher.calls_used}")
    if enricher.error:
        print(f"⚠️ enrichment latched mid-run: {enricher.error}")
        print("   (cache-only from there — re-run tomorrow; the cache makes runs additive)")
    return 0


def cmd_smoke(args) -> int:
    """1 Last.fm artist call + 1 track-ID resolve + 1 Musicae batch call.
    Verifies the keys and the response shapes against the live APIs."""
    ok = True
    print("1/3 Last.fm artist.getinfo (Disclosure)")
    try:
        info = lastfm.artist_info("Disclosure")
        if not info:
            raise lastfm.LastfmError("no such artist (unexpected)")
        print(f"   ✓ name={info['name']} tags={info['tags'][:3]} similar={info['similar'][:2]}")
    except lastfm.LastfmError as e:
        ok = False
        print(f"   ✗ {e}")

    print("2/3 Spotify track-ID resolve (Disclosure — Settle)")
    token = spotify.get_token()
    track_id = None
    if not token:
        ok = False
        print("   ✗ no Spotify token in .env (SPOTIFY_CLIENT_ID/SECRET)")
    else:
        try:
            track_id, year = spotify.resolve_track_id("Disclosure", "Settle", token)
            if track_id:
                print(f"   ✓ track_id={track_id} year={year}")
            else:
                ok = False
                print("   ✗ no confident track match")
        except spotify.SpotifyAPIError as e:
            ok = False
            print(f"   ✗ {e}")

    print("3/3 Musicae /v2/audio-analysis (1 track, 1 of the 5/batch)")
    if track_id:
        try:
            results = musicae.audio_analysis([track_id])
            if not results:
                ok = False
                print("   ✗ empty result (check the response shape vs _parse_track)")
            else:
                r = results[0]
                print(f"   ✓ name={r['name']} bpm={r['bpm']} key={r['key']}/{r['mode']} "
                      f"camelot={r['camelot']} mood={r['mood']} genres={r['genres']}")
                print(f"     features={r['features']}")
        except musicae.MusicaeError as e:
            ok = False
            print(f"   ✗ {e}")
            if e.status_code == 403:
                print("     → on the RapidAPI dashboard, SUBSCRIBE (free plan) the key to")
                print("       'DJ Track Audio Analysis API', then re-run this smoke test.")
    else:
        print("   — skipped (needs a track ID from step 2)")
    print("\n" + ("✅ smoke OK — keys wired, shapes match" if ok else "❌ smoke failed — see above"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Paced backfill of the enrichment cache.")
    sub = parser.add_subparsers(dest="cmd")

    def add_sources(p):
        p.add_argument("--sources", default="fixtures,golden,history",
                       help="comma list of: fixtures,golden,history (default all)")

    p_st = sub.add_parser("status", help="offline: cache coverage per source")
    add_sources(p_st)
    p_st.set_defaults(fn=cmd_status)

    p_bf = sub.add_parser("backfill", help="grow the cache (Last.fm free + budgeted Musicae)")
    add_sources(p_bf)
    p_bf.add_argument("--budget", type=int, default=None,
                      help="max Musicae calls this run (default $MUSICAE_CALLS_PER_DAY or 10)")
    p_bf.add_argument("--delay", type=float, default=1.0,
                      help="seconds between Musicae calls (default 1)")
    p_bf.add_argument("--max-pairs", type=int, default=None, help="cap total album pairs")
    p_bf.add_argument("--cache", metavar="PATH", help="alternate cache file (default afrec/cache/enrichment.json)")
    p_bf.add_argument("--dry-run", action="store_true", help="print the plan, fire nothing")
    p_bf.add_argument("--lastfm-only", action="store_true", help="tags/years only (spends no Musicae quota)")
    p_bf.add_argument("--musicae-only", action="store_true", help="features only (no Last.fm calls)")
    p_bf.set_defaults(fn=cmd_backfill)

    p_sm = sub.add_parser("smoke", help="1 Last.fm + 1 Spotify resolve + 1 Musicae call")
    p_sm.set_defaults(fn=cmd_smoke)

    args = parser.parse_args()
    if not args.cmd:
        args.cmd = "backfill"     # bare `backfill.py` = backfill
        args.sources = "fixtures,golden,history"
        args.budget = None
        args.delay = 1.0
        args.max_pairs = None
        args.cache = None
        args.dry_run = False
        args.lastfm_only = False
        args.musicae_only = False
        args.fn = cmd_backfill
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
