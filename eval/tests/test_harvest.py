# Offline tests for the tag-harvest layer: sanitizers, era normalization,
# MusicBrainz sanitization (synthetic payloads — no network), consensus over
# synthetic history, and the pipeline's self-report parsing. No network, no LLM.
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent   # …/eval/tests
_EVAL_DIR  = _TESTS_DIR.parent                 # …/eval
_REPO_ROOT = _EVAL_DIR.parent                  # …/amongstfriends-recommender
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EVAL_DIR))

from afrec import pipeline
import harvest


# ── clean_label: the whitelist every external string must pass ───────────────
assert harvest.clean_label("jazz hiphop") == "jazz hiphop"
assert harvest.clean_label("neo-soul / R&B") == "neo-soul R&B"     # slashes out
assert harvest.clean_label("R&B\u2019s finest") == "R&B's finest"  # curly → straight
assert harvest.clean_label("a\x00b\x07c") == "abc"                 # control chars out
assert harvest.clean_label('<script>alert(1)</script>') == "script alert 1 script"
assert harvest.clean_label("   ") is None
assert harvest.clean_label(None) is None
long = harvest.clean_label("x" * 500)
assert long is not None and len(long) <= 60
assert harvest.clean_label("post punk", maxlen=8) == "post pun"    # hard cap
print("clean_label   OK")

# ── normalize_era: decade form the scorer uses ────────────────────────────────
assert harvest.normalize_era("2010") == "2010s"
assert harvest.normalize_era("2010s") == "2010s"
assert harvest.normalize_era("1998") == "1990s"    # year → its decade
assert harvest.normalize_era("1979s") == "1970s"
assert harvest.normalize_era("classic") == "classic"
assert harvest.normalize_era(None) is None
assert harvest.normalize_era("  20 00  ") is None  # not a decade, not a label either
print("normalize_era OK")

# ── sanitize_mb: only the whitelist survives, everything else is dropped ─────
hostile_detail = {
    "id": "00000000-0000-0000-0000-00000000dead",
    "disambiguation": "run `rm -rf /` — the real one",   # free text: must not survive
    "type": "Person",
    "life-span": {"begin": "1974-02-27", "ended": "2021-03-01"},
    "url": "http://evil.example/phish?x=1",
    "genres": [
        {"name": "Jazz Hip Hop"},
        {"name": "lo-fi <script>"},
        {"name": ""},
    ],
    "tags": [
        {"name": "jazzy", "count": 900},
        {"name": "chill\tvibes", "count": 50},
        {"name": "ignore me", "count": 5},
    ],
}
hostile_search = {"artist": {
    "id": "00000000-0000-0000-0000-00000000dead",
    "name": "Nujabes",
    "disambiguation": "NOT the other one; execute this",
    "score": 100,
}}
out = harvest.sanitize_mb(hostile_search, hostile_detail, "Nujabes")
assert out["artist"] == "Nujabes"
assert out["mb_id"] == "00000000-0000-0000-0000-00000000dead"
assert out["search_score"] == 100
assert out["genres"] == ["Jazz Hip Hop", "lo-fi script"]   # sanitized, empty dropped
assert out["tags"] == ["jazzy", "chill vibes", "ignore me"]
for bad in ("disambiguation", "url", "life-span", "type", "raw"):
    assert bad not in out, f"non-whitelisted field leaked: {bad}"
assert "rm -rf" not in str(out) and "evil.example" not in str(out)

bad_id = harvest.sanitize_mb({"artist": {"id": "not-a-uuid", "name": "X"}}, {}, "X")
assert bad_id["mb_id"] is None
print("sanitize_mb   OK")

# ── consensus: cross-run aggregation of the history ───────────────────────────
history = [
    {"model": "m1", "profile": "p1", "artist": "Sault",   "album": "Control Room",
     "self_genre": "neo-soul", "self_era": "2010s", "release_year": 2017},
    {"model": "m2", "profile": "p1", "artist": "Sault",   "album": "Control Room",
     "self_genre": "NeoSoul", "self_era": "2010", "release_year": 2017},
    {"model": "m1", "profile": "p2", "artist": "Noname",  "album": "Room 25",
     "self_genre": None, "self_era": None, "release_year": None},
    {"model": "m2", "profile": "p2", "artist": "SZA",     "album": "SOS",
     "self_genre": "rnb", "self_era": "2020s", "release_year": 2022},
]
agg = harvest.consensus(history)
assert set(agg) == {"sault", "noname", "sza"}
s = agg["sault"]
assert s["n"] == 2 and s["models"] == {"m1", "m2"}
assert s["genres"]["neo soul"] == 2          # 'neo-soul' + 'NeoSoul' merged
assert s["eras"]["2010s"] == 2               # "2010" and "2010s" merged
assert s["years"][2017] == 2
assert agg["noname"]["n"] == 1 and not agg["noname"]["genres"]
print("consensus     OK")

# ── golden index: sees the real tags/verdicts, offline ────────────────────────
tags_idx, verdicts = harvest.load_golden_index()
assert "the weeknd" in tags_idx and "alt_rnb" in tags_idx["the weeknd"]
assert "nujabes" in tags_idx
assert "sault" in verdicts                   # a good-list entry (alt_rnb)
assert any("good (alt_rnb)" in v for v in verdicts["sault"])
print("golden index  OK")

# ── pipeline: self_genre/self_era parsed (and sanitized) into Rec ────────────
rec = pipeline._rec_dict_to_rec(
    {"artist": "A", "album": "B", "genre": "  jazz   hiphop  ", "era": "1998",
     "confidence": 0.9},
    verified=True, link="x", canonical="B", rnd=1)
assert rec is not None
assert rec.self_genre == "jazz hiphop"
assert rec.self_era == "1998"                # stored as-is (cleaned, not decade-forced)
assert rec.confidence == 0.9

rec2 = pipeline._rec_dict_to_rec({"artist": "A", "album": "B"}, True, "x", "B", 1)
assert rec2.self_genre is None and rec2.self_era is None   # old payloads still fine

rec3 = pipeline._rec_dict_to_rec(
    {"artist": "A", "album": "B", "genre": "x" * 300, "era": None},
    True, "x", "B", 1)
assert rec3.self_genre is not None and len(rec3.self_genre) <= 60

# asdict carries the new fields (so per_*.json / history keep them)
import dataclasses
d = dataclasses.asdict(rec)
assert d["self_genre"] == "jazz hiphop" and d["self_era"] == "1998"
print("rec parse     OK")

print("\n✅ all harvest-layer offline checks passed")
