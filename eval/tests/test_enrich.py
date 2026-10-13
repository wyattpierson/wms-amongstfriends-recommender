#!/usr/bin/env python3
"""Offline tests for the enrichment layer (afrec/enrich.py + friends).

No network: every API module is monkeypatched with fakes. Run via
`make test` or directly.
"""

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from afrec import enrich, lastfm, musicae, spotify
from afrec.sanitize import clean_label

PASS = 0


def ok(name):
    global PASS
    PASS += 1
    print(f"  ✓ {name}")


# ── sanitizer (moved to afrec, re-exported by harvest) ────────────────────────

def test_clean_label():
    assert clean_label("  melodic   house ") == "melodic house"
    assert clean_label("dance<script>") == "dance script"
    assert clean_label("neo\u2019soul") == "neo'soul"
    assert clean_label(None) is None
    assert clean_label("###") is None
    assert clean_label("x" * 100, maxlen=10) == "x" * 10
    ok("clean_label whitelist + caps")


def test_harvest_reexport():
    sys.path.insert(0, str(ROOT / "eval"))
    import harvest
    assert harvest.clean_label is clean_label
    ok("harvest re-exports the shared clean_label")


# ── key rendering ─────────────────────────────────────────────────────────────

def test_key_names():
    assert enrich.key_name(2, "major") == "D major"
    assert enrich.key_name(2, "minor") == "D minor"
    assert enrich.key_short(2, "major") == "Dmaj"
    assert enrich.key_short(9, "m") == "Amin"
    assert enrich.key_name(None, "major") is None
    assert enrich.key_name(99, "major") is None
    ok("key_name / key_short")


# ── catalog fact rendering ────────────────────────────────────────────────────

def _fact(**kw):
    base = dict(year=1998, tags=["melodic house", "dance"],
                features={"energy": 0.74, "danceability": 0.58, "valence": 0.47},
                bpm=122.6, mood="tense")
    base.update(kw)
    return enrich.CatalogFact(**base)


def test_sound_line():
    f = _fact(key_int=2, mode="major")
    line = f.sound_line()
    assert line is not None
    assert len(line) <= enrich.SOUND_LINE_MAX, line
    assert "E.7" in line and "D.6" in line and "V.5" in line
    assert "123bpm" in line and "Dmaj" in line and "tense" in line
    assert line.endswith("melodic house, dance")
    ok(f"sound_line compact + capped: {line!r}")


def test_catalog_line():
    f = _fact(key_int=2, mode="major")
    line = f.catalog_line()
    assert line.startswith("(catalog: 1998 ·")
    assert line.endswith(")")
    assert len(line) <= enrich.SOUND_LINE_MAX + 20
    assert _fact(features=None, bpm=None, mood=None, tags=[]).catalog_line() == "(catalog: 1998)"
    assert enrich.CatalogFact().catalog_line() is None
    assert _fact(features=None, bpm=None, mood=None, year=None).sound_line() == "melodic house, dance"
    ok("catalog_line variants")


# ── cache ─────────────────────────────────────────────────────────────────────

def test_cache():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cache" / "enrichment.json"
        c = enrich.Cache(p)
        assert c.get("a|b") is None
        c.put("a|b", {"year": 1998})
        c.save()
        assert enrich.Cache(p).get("a|b") == {"year": 1998}
        p.write_text("{not json")
        assert enrich.Cache(p).get("a|b") is None
    ok("cache round-trip + corrupt-file tolerance")


# ── enricher with fakes ───────────────────────────────────────────────────────

class FakeMusicae:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def __call__(self, track_ids):
        self.calls += 1
        if self.fail:
            raise musicae.MusicaeError("Musicae API 429 — daily quota exhausted?")
        return [
            {"track_id": tid, "name": "Fake Track", "bpm": 122.7,
             "key": 2, "mode": "major", "camelot": "5A", "mood": "tense",
             "features": {"energy": 0.7, "danceability": 0.6, "valence": 0.5},
             "genres": ["melodic house"]}
            for tid in track_ids
        ]


def _mk_enricher(tmp, budget=10, fake_musicae=None, year=1998,
                 tag_names=("melodic house", "dance")):
    fake_musicae = fake_musicae or FakeMusicae()
    enrich.lastfm.artist_info = lambda a, **kw: {"name": a,
                                                 "tags": [(n, 50 - i * 10) for i, n in enumerate(tag_names)],
                                                 "similar": []}
    enrich.lastfm.album_info = lambda a, b, **kw: {"name": b, "artist": a, "year": year, "tags": [], "mbid": None}
    enrich.lastfm.api_key = lambda: "test-key"
    enrich.spotify.resolve_track_id = lambda a, b, t: ("tid-" + a.lower().replace(" ", ""), year)
    enrich.musicae.audio_analysis = fake_musicae
    enrich.musicae.api_key = lambda: "test-key"
    return enrich.Enricher(budget=budget, spotify_token="fake", cache_path=Path(tmp) / "c.json"), fake_musicae


def test_full_flow_and_cache_hit():
    with tempfile.TemporaryDirectory() as d:
        e, fm = _mk_enricher(d)
        pairs = [("Artist One", "Album A"), ("Artist Two", "Album B")]
        facts = e.fact_map(pairs)
        assert len(facts) == 2
        f = facts[enrich.name_key("Artist One", "Album A")]
        assert f.year == 1998 and f.tags[0] == "melodic house"
        assert f.bpm == 122.7 and f.key == "D major"
        assert fm.calls == 1 and e.calls_used == 1        # 2 IDs batched into ONE call
        assert e.error is None
        # second run: everything cached → zero network (fresh counter proves it)
        fm_fresh = FakeMusicae()
        e2, _ = _mk_enricher(d, fake_musicae=fm_fresh)
        facts2 = e2.fact_map(pairs)
        assert facts2[enrich.name_key("Artist One", "Album A")].bpm == 122.7
        assert fm_fresh.calls == 0 and e2.calls_used == 0
        ok("full flow: 2 pairs → 1 batched call; cache hit → 0 calls")


def test_budget_precount():
    with tempfile.TemporaryDirectory() as d:
        e, fm = _mk_enricher(d, budget=1)
        pairs = [(f"Artist {i}", f"Album {i}") for i in range(12)]
        facts = e.fact_map(pairs)
        assert fm.calls == 1, "budget 1 must pre-stop at 1 call"
        assert e.calls_used == 1
        got_features = sum(1 for f in facts.values() if f.features)
        assert got_features == musicae.BATCH_SIZE, f"exactly 5 IDs served, got {got_features}"
        ok(f"budget pre-count: 12 pairs, budget 1 → 1 call, {got_features} albums served")


def test_latch_on_error():
    with tempfile.TemporaryDirectory() as d:
        e, fm = _mk_enricher(d, fake_musicae=FakeMusicae(fail=True))
        logs = []
        e.log = logs.append
        facts = e.fact_map([("A", "B")])
        assert e.error is not None and "429" in e.error
        assert sum(1 for l in logs if "latched" in l) == 1     # one loud ⚠
        # a NEW session may try once (quota resets daily) — and latches again on failure
        fm_fresh = FakeMusicae(fail=True)
        e2, _ = _mk_enricher(d, fake_musicae=fm_fresh)
        facts2 = e2.fact_map([("A", "B")])
        assert fm_fresh.calls == 1 and e2.error is not None
        f2 = facts2[enrich.name_key("A", "B")]
        assert f2.year == 1998 and f2.tags, "degraded facts (tags/year) still rendered"
        assert f2.features is None
        ok("latch: one loud ⚠, cache-only afterwards, degraded facts still render")


def test_no_keys_off():
    with tempfile.TemporaryDirectory() as d:
        enrich.lastfm.api_key = lambda: None
        enrich.musicae.api_key = lambda: None
        e = enrich.Enricher(budget=10, spotify_token="fake", cache_path=Path(d) / "c.json")
        assert not e.available
        assert e.fact_map([("A", "B")]) == {}
        ok("no keys → enrichment silently OFF")


def test_stale_tags_refetched():
    with tempfile.TemporaryDirectory() as d:
        e, fm = _mk_enricher(d, year=1998)
        e.fact_map([("A", "B")])
        # age the tags past TTL, drop the features cache to force a fresh pass
        entry = e.cache.get(enrich.name_key("A", "B"))
        entry["tags_fetched"] = (date.today() - timedelta(days=40)).isoformat()
        e.cache.save()
        e2, fm2 = _mk_enricher(d)
        enrich.lastfm.artist_info = lambda a, **kw: {"name": a, "tags": [("freshtag", 5)], "similar": []}
        e2.fact_map([("A", "B")])
        assert e2.cache.get(enrich.name_key("A", "B"))["tags"][0] == "freshtag"
        ok("tags 30-day TTL → refetch")


def test_spotify_resolve_failure_latches():
    with tempfile.TemporaryDirectory() as d:
        e, fm = _mk_enricher(d)
        enrich.spotify.resolve_track_id = lambda a, b, t: (_ for _ in ()).throw(
            spotify.SpotifyAPIError("Spotify API 429 — daily quota exhausted?", status_code=429))
        e.fact_map([("A", "B")])
        assert e.error is not None and "429" in e.error
        assert fm.calls == 0, "no Musicae call when IDs can't be resolved"
        ok("Spotify 429 on resolve → latch, zero Musicae spend")


def main():
    print("test_enrich: offline tests (no network)\n")
    for fn in (test_clean_label, test_harvest_reexport, test_key_names,
               test_sound_line, test_catalog_line, test_cache,
               test_full_flow_and_cache_hit, test_budget_precount,
               test_latch_on_error, test_no_keys_off,
               test_stale_tags_refetched, test_spotify_resolve_failure_latches):
        fn()
    print(f"\n{PASS}/12 passed")


if __name__ == "__main__":
    main()
