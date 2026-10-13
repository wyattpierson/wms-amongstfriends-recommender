"""afrec.enrich — the enrichment orchestrator.

Combines the two layers into per-album catalog facts:

    Last.fm  (identity):  year, tags            — cheap, 50/s, no budget here
    Spotify  (hop):       (artist, album) → track ID — same quota as verify()
    Musicae  (measurement): features/bpm/key/mood — FREE TIER 10 CALLS/DAY

The whole design rule (ENRICHMENT_DESIGN.md §2.4): **enrichment can only ever
make a run less informed, never fail it.** No keys → silently off. First
Musicae/Spotify failure → latch: one loud ⚠, everything after is cache-only.
Budget is pre-checked: `fact_map()` counts the calls it would fire and never
burns more than `budget` per session.

The disk cache (afrec/cache/enrichment.json, gitignored) is shared by
harvest/fit/evaluate/backfill. Audio features: no TTL (a recording doesn't
change). Tags: 30-day TTL. Track IDs are stored next to the name key so
resolution never happens twice.

Nothing here scores anything and nothing here talks to an LLM. External
strings enter only through clean_label (afrec/sanitize.py).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from . import lastfm, musicae, spotify
from .sanitize import clean_label

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "enrichment.json"
TAG_TTL_DAYS = 30
SOUND_LINE_MAX = 80   # the whole per-review catalog line, hard-capped

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def name_key(artist: str, album: str) -> str:
    return f"{artist.lower().strip()}|{album.lower().strip()}"


def key_name(key, mode) -> str | None:
    """(int 0-11, 'major'/'minor') → 'D major'; None when either is missing."""
    try:
        k = int(key)
    except (TypeError, ValueError):
        return None
    if not 0 <= k < 12:
        return None
    mode = (mode or "").lower().strip()
    if mode.startswith("maj") or mode == "M":
        return f"{NOTE_NAMES[k]} major"
    if mode.startswith("min") or mode == "m":
        return f"{NOTE_NAMES[k]} minor"
    return NOTE_NAMES[k]


def key_short(key, mode) -> str | None:
    """'Dmaj' / 'Dmin' — the compact form for the prompt line."""
    full = key_name(key, mode)
    if not full:
        return None
    parts = full.split(" ", 1)
    if len(parts) == 1:
        return parts[0]                     # bare note (mode unrecognized)
    return parts[0] + ("maj" if parts[1] == "major" else "min")


# ── Disk cache ────────────────────────────────────────────────────────────────

class Cache:
    """JSON dict on disk. Reads are fault-tolerant (corrupt → empty); writes
    are atomic (tmp + replace) so an interrupted backfill can't lose it."""

    def __init__(self, path: Path = CACHE_PATH):
        self.path = Path(path)
        self._data: dict = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data = loaded
            except (ValueError, OSError):
                self._data = {}

    def get(self, key: str) -> dict | None:
        v = self._data.get(key)
        return v if isinstance(v, dict) else None

    def put(self, key: str, value: dict) -> None:
        self._data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)


# ── Facts ─────────────────────────────────────────────────────────────────────

@dataclass
class CatalogFact:
    """Everything we know about one (artist, album). All strings sanitized."""
    year: int | None = None
    tags: list[str] = field(default_factory=list)        # sanitized, ≤3
    features: dict | None = None                          # 7 continuous, 0-1
    bpm: float | None = None
    key: str | None = None                                # e.g. "D major"
    mood: str | None = None                               # e.g. "tense"
    track_id: str | None = None
    key_int: int | None = None                            # 0-11 (raw, for key_short)
    mode: str | None = None                               # 'major'/'minor'

    def sound_line(self) -> str | None:
        """Compact measurement line, ≤ SOUND_LINE_MAX chars. None if we have
        no measurements (a tags-only fact still renders, via catalog_line)."""
        parts = []
        if self.features:
            f = self.features
            bits = []
            for label, k in (("E", "energy"), ("D", "danceability"), ("V", "valence")):
                if f.get(k) is not None:
                    bits.append(f"{label}.{round(f[k] * 10)}")
            if bits:
                parts.append(" ".join(bits))
        if self.bpm is not None:
            parts.append(f"{round(self.bpm)}bpm")
        if self.key_int is not None:
            ks = key_short(self.key_int, self.mode)
            if ks:
                parts.append(ks)
        if self.mood:
            parts.append(self.mood)
        if self.tags:
            parts.append(", ".join(self.tags[:2]))
        if not parts:
            return None
        line = " · ".join(parts)
        return line[:SOUND_LINE_MAX]

    def catalog_line(self) -> str | None:
        """The full per-review rendering: '(catalog: 1998 · E.7 D.6 V.5 …)'.
        None when there is nothing to say (the prompt then renders nothing)."""
        bits = []
        if self.year:
            bits.append(str(self.year))
        sound = self.sound_line()
        if sound:
            bits.append(sound)
        if not bits:
            return None
        line = f"(catalog: {' · '.join(bits)})"
        return line[:SOUND_LINE_MAX + 20]   # leave headroom for the prefix


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Enricher:
    """Budgeted, latching, cached access to catalog facts.

    budget = maximum Musicae calls THIS SESSION (default: $MUSICAE_CALLS_PER_DAY
    or 10 — the free tier). Spotify track-resolution runs on the same token
    as verification (its own quota; a 429 latches enrichment, not the run).
    delay = seconds to sleep between Musicae calls (pace the free tier).
    """

    def __init__(self, *, budget: int | None = None, spotify_token: str | None = None,
                 delay: float = 0.0, cache_path: Path = CACHE_PATH,
                 log: Callable | None = None):
        self.log = log or (lambda s: print(s, file=sys.stderr))
        try:
            env_budget = int(os.getenv("MUSICAE_CALLS_PER_DAY") or 10)
        except ValueError:
            env_budget = 10
        self.budget = int(budget) if budget is not None else env_budget
        self.delay = max(0.0, float(delay))
        self.cache = Cache(cache_path)
        self.calls_used = 0
        self.error: str | None = None      # latch: set on first hard failure
        self._latched_warned = False
        self._lastfm_warned = False
        # Spotify token: explicit > env > None (musicae side degrades to cache-only)
        if spotify_token is None:
            spotify_token = spotify.get_token()
        self.spotify_token = spotify_token or None
        self._has_lastfm = bool(lastfm.api_key())
        self._has_musicae = bool(musicae.api_key())
        if not self._has_lastfm and not self._has_musicae:
            self.log("[enrich] no enrichment keys set (LASTFM_API_KEY / MUSICAE_API_KEY) — enrichment OFF")

    @property
    def available(self) -> bool:
        return self._has_lastfm or self._has_musicae

    # ── internals ──

    def _latch(self, err: Exception) -> None:
        if self.error is None:
            self.error = str(err)
            self.log(f"⚠️ enrichment latched (cache-only from here): {self.error}")
        # subsequent failures: silent, no retries anywhere

    def _lastfm_facts(self, artist: str, album: str) -> tuple[list[str], int | None]:
        """(tags ≤3, year) — per-artist failure is non-fatal (Last.fm is cheap
        but coverage is patchy, esp. electronic; one loud ⚠ per failure type)."""
        if self.error is not None or not self._has_lastfm:
            return [], None
        tags: list[str] = []
        year = None
        try:
            info = lastfm.artist_info(artist)
            if info:
                tags = [t for t, _c in info["tags"]][:3]
        except Exception as e:   # LastfmError OR shape drift — neither may kill the run
            if not self._lastfm_warned:
                self._lastfm_warned = True
                self.log(f"⚠️ last.fm artist lookup failing ({e}) — tags omitted, features unaffected")
        try:
            alb = lastfm.album_info(artist, album)
            if alb and alb.get("year"):
                year = alb["year"]
        except Exception as e:   # ditto
            if not self._lastfm_warned:
                self._lastfm_warned = True
                self.log(f"⚠️ last.fm album lookup failing ({e}) — year omitted, features unaffected")
        return tags, year

    def _features_for(self, track_id: str) -> dict | None:
        """Cached-or-fetched parsed analysis for one track ID (1 batch call
        max). Returns None on miss/latch. Budget is checked by the caller."""
        entry = self.cache.get(f"track:{track_id}")
        if entry and entry.get("features"):
            return entry
        if self.calls_used >= self.budget:
            return None
        try:
            results = musicae.audio_analysis([track_id])
        except (musicae.MusicaeError, spotify.SpotifyAPIError) as e:
            self._latch(e)
            return None
        self.calls_used += 1
        if self.delay:
            time.sleep(self.delay)
        if not results:
            return None
        self.cache.put(f"track:{track_id}", results[0])
        return results[0]

    def _build_fact(self, entry: dict | None, tags: list[str], year: int | None,
                    track_id: str | None, analysis: dict | None) -> CatalogFact:
        entry = entry or {}
        analysis = analysis or {}
        return CatalogFact(
            year=year if year is not None else entry.get("year"),
            tags=(tags or entry.get("tags") or [])[:3],
            features=analysis.get("features"),
            bpm=analysis.get("bpm"),
            key=key_name(analysis.get("key"), analysis.get("mode")),
            mood=analysis.get("mood"),
            track_id=track_id or entry.get("track_id"),
            key_int=analysis.get("key") if isinstance(analysis.get("key"), int) else None,
            mode=analysis.get("mode"),
        )

    # ── public API ──

    def facts_for(self, artist: str, album: str) -> CatalogFact | None:
        """Facts for one pair, or None when we know nothing (and can't learn)."""
        return self.fact_map([(artist, album)]).get(name_key(artist, album))

    def fact_map(self, pairs: list[tuple[str, str]]) -> dict[str, CatalogFact]:
        """Bulk facts keyed by name_key(artist, album).

        Order of operations, per pair: cache → Last.fm (tags/year) →
        resolve track ID (Spotify, cached) → features (Musicae, batched 5
        IDs/call, budgeted). Pre-counts the Musicae calls it would fire and
        says so BEFORE firing; never burns more than self.budget.
        """
        pairs = [(a, b) for a, b in pairs if a and b]
        if not pairs or not self.available:
            return {}

        # ── pass 1: what does the cache already know? ──
        facts: dict[str, CatalogFact] = {}
        need_lf: list[tuple[str, str, str]] = []       # (key, artist, album): tags stale/missing
        need_resolve: list[tuple[str, str, str]] = []  # (key, artist, album): no track_id
        need_features: list[tuple[str, str]] = []      # (key, track_id): analysis missing

        for artist, album in pairs:
            k = name_key(artist, album)
            entry = self.cache.get(k)
            tags_fresh = bool(
                entry and entry.get("tags")
                and (date.today() - datetime.strptime(entry["tags_fetched"], "%Y-%m-%d").date()).days
                <= TAG_TTL_DAYS
            ) if entry and entry.get("tags_fetched") else False
            if entry and (tags_fresh or not self._has_lastfm):
                tags, year = entry.get("tags") or [], entry.get("year")
            else:
                tags, year = ([], None)
                need_lf.append((k, artist, album))
                if entry and entry.get("tags"):
                    tags, year = entry.get("tags"), entry.get("year")   # stale beats nothing
            track_id = (entry or {}).get("track_id")
            if not track_id:
                need_resolve.append((k, artist, album))
            analysis = self.cache.get(f"track:{track_id}") if track_id else None
            if track_id and not (analysis and analysis.get("features")):
                need_features.append((k, track_id))
            facts[k] = self._build_fact(entry, tags, year, track_id, analysis)

        # ── pass 2: Last.fm (cheap, no budget) ──
        for k, artist, album in need_lf:
            try:
                tags, year = self._lastfm_facts(artist, album)
            except Exception:
                tags, year = [], None
            if not (tags or year):
                continue
            e = dict(self.cache.get(k) or {})
            if tags:
                e["tags"] = tags
                e["tags_fetched"] = date.today().isoformat()
            if year:
                e["year"] = year
            self.cache.put(k, e)
            facts[k] = self._build_fact(e, e.get("tags"), e.get("year"), e.get("track_id"), None)

        # ── pass 3: Spotify track-ID resolution (only for cache-miss pairs) ──
        if need_resolve and self.spotify_token is None:
            if self._has_musicae:
                self.log("[enrich] no Spotify token — Musicae features unavailable "
                         "(needs a track ID); tags/year still work")
        resolved: list[tuple[str, str]] = []
        for k, artist, album in need_resolve:
            if self.error is not None:
                break
            try:
                tid, year = (spotify.resolve_track_id(artist, album, self.spotify_token)
                             if self.spotify_token else (None, None))
            except spotify.SpotifyAPIError as e:
                self._latch(e)
                break
            if tid:
                e = dict(self.cache.get(k) or {})
                e["track_id"] = tid
                if year and not e.get("year"):
                    e["year"] = year
                self.cache.put(k, e)
                facts[k] = self._build_fact(e, e.get("tags"), e.get("year"), tid,
                                            self.cache.get(f"track:{tid}"))
                resolved.append((k, tid))

        # ── pass 4: Musicae features (batched 5/call, budgeted, pre-counted) ──
        pending = [(k, tid) for k, tid in
                   ((k, tid) for k, tid in need_features)
                   if not (self.cache.get(f"track:{tid}") or {}).get("features")]
        pending += [(k, tid) for k, tid in resolved
                    if not (self.cache.get(f"track:{tid}") or {}).get("features")
                    and (k, tid) not in pending]
        if pending:
            calls = (len(pending) + musicae.BATCH_SIZE - 1) // musicae.BATCH_SIZE
            allowed = min(calls, max(0, self.budget - self.calls_used))
            if allowed < calls:
                self.log(f"[enrich] {calls} Musicae call(s) needed, budget allows {allowed} — "
                         f"{len(pending) - allowed * musicae.BATCH_SIZE} album(s) wait for the next day")
            if allowed:
                self.log(f"[enrich] firing {allowed} Musicae call(s) for {min(len(pending), allowed * musicae.BATCH_SIZE)} album(s) "
                         f"(session: {self.calls_used}/{self.budget} used before)")
            for c in range(allowed):
                if self.error is not None:
                    break
                batch = pending[c * musicae.BATCH_SIZE:(c + 1) * musicae.BATCH_SIZE]
                ids = [tid for _k, tid in batch]
                try:
                    results = musicae.audio_analysis(ids)
                except musicae.MusicaeError as e:
                    self._latch(e)
                    break
                self.calls_used += 1
                by_id = {r["track_id"]: r for r in results if r.get("track_id")}
                for k, tid in batch:
                    r = by_id.get(tid)
                    if not r:
                        continue
                    self.cache.put(f"track:{tid}", r)
                    e = dict(self.cache.get(k) or {})
                    facts[k] = self._build_fact(e, e.get("tags"), e.get("year"), tid, r)
                if self.delay and c < allowed - 1:
                    time.sleep(self.delay)

        self.cache.save()
        return facts
