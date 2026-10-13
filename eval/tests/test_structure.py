# Offline test for the response-structure scoring (tags / constraints /
# justification keywords) + release_year plumbing. No network, no LLM.
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent   # …/eval/tests
_EVAL_DIR  = _TESTS_DIR.parent                 # …/eval
_REPO_ROOT = _EVAL_DIR.parent                  # …/amongstfriends-recommender
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EVAL_DIR))

from afrec import pipeline
from afrec.llm import BaseLLM, GenerationMeta
import goldenset
import evaluate


def R(artist, album, reason="", year=None, verified=True):
    return pipeline.Rec(artist=artist, album=album, confidence=0.8,
                        reason=reason, link="x", verified=verified,
                        release_year=year)


# ── Tag normalization coercion ────────────────────────────────────────────────
tags = goldenset._norm_tags({
    "Nujabes": {"genres": {"coarse": "hiphop", "fine": "jazz hiphop"}, "era": "2000s"},
    "Q-Tip":   "jazz hiphop",                    # bare string → coarse
    "Madlib":  ["hiphop", "lo-fi"],              # list → [coarse, fine]
    "Post Malone": {"genres": {"coarse": "pop"}, "era": "2010s"},
    "Jack Harlow": {"genres": {"coarse": "pop"}, "era": "2020s"},
})
assert tags["nujabes"]["genres"] == {"coarse": "hiphop", "fine": "jazz hiphop"}
assert tags["q-tip"]["genres"] == {"coarse": "jazz hiphop"}
assert tags["madlib"]["genres"] == {"coarse": "hiphop", "fine": "lo-fi"}
print("tag coercion  OK")

# ── Lane credit: graduated ───────────────────────────────────────────────────
golden = {
    "tags": tags,
    "constraints": [
        {"check": "min_lane_share", "lane": "jazz hiphop", "area": "hiphop", "level": "fine",
         "value": 0.5, "weight": 2, "reason": "stay in the lane"},
        {"check": "min_distinct_decades", "value": 2, "weight": 1},
    ],
    "justification_keywords": ["boom-bap", "dig"],
}
recs = [
    R("Nujabes", "Modal Soul", "soft boom-bap", year=2005),   # in-lane: 1.0
    R("Q-Tip", "Amplified", "jazz chops in the Nujabes vein", year=1997),  # in-lane: 1.0, cites fixture
    R("Madlib", "Lodger", "lo-fi diggings", year=2008),        # coarse hiphop only: 0.5
    R("Kamasi", "The Composer", "jazz, not hiphop"),           # untagged → neutral
]
lane = [goldenset._lane_credit(r, tags, "jazz hiphop", "fine", "hiphop") for r in recs]
assert lane == [1.0, 1.0, 0.5, None], lane
print("lane credits  OK →", lane)

# ── Weighted structure score ──────────────────────────────────────────────────
st = goldenset.score_response_structure(recs, golden,
                                        fixture_names=["Nujabes", "Modal Soul", "Common", "Be"])
# lane: (1 + 1 + 0.5) / 3 = 0.833 ≥ 0.5 → pass (weight 2)
# decades: {2000s, 1990s} = 2 ≥ 2 → pass (weight 1)
assert st["points"] == 3.0 and st["total"] == 3.0 and st["score"] == 1.0, st
assert abs(st["lane_share"] - 0.833) < 0.01, st["lane_share"]
assert st["tagged"] == 3 and st["total_recs"] == 4
assert st["just_hits"] == 2          # "boom-bap" rec + "dig" rec
assert st["just_fixture_hits"] == 1  # only "Nujabes / Modal Soul" names a fixture artist+album pair
print("structure OK →", {k: st[k] for k in ("points", "total", "score", "lane_share", "tagged", "just_hits", "just_fixture_hits")})

# Failing shape: all one decade, out of lane
bad_recs = [
    R("Post Malone", "Beerbongs", "pop", year=2018),   # tagged pop: 0.0 lane credit
    R("Jack Harlow", "Whoomp", "pop", year=2019),      # tagged pop: 0.0 lane credit
    R("SZA", "Ctrl", "rnb", year=2019),                # untagged → neutral
]
st_bad = goldenset.score_response_structure(bad_recs, golden)
assert st_bad["checks"]["min_lane_share"]["pass"] is False
assert st_bad["checks"]["min_distinct_decades"]["pass"] is False
assert st_bad["score"] == 0.0
print("failing shape OK → score", st_bad["score"])

# Unknown check type: skipped, not counted
golden2 = {"tags": tags, "constraints": [
    {"check": "min_distinct_genres", "value": 1},
    {"check": "max_popular_share", "value": 0.5, "weight": 9},   # future check, not implemented
], "justification_keywords": []}
st2 = goldenset.score_response_structure(recs, golden2)
assert st2["total"] == 1.0, st2   # only the known check counts
print("unknown check skipped OK")

# ── lane_balance: per-rec in/out points (graded, partial credit) ──────────────
golden_lb = {"tags": tags, "constraints": [
    {"check": "lane_balance", "lane": "jazz hiphop", "area": "hiphop", "level": "fine",
     "value": 0.5, "weight": 2, "reason": "net lane alignment"},
], "justification_keywords": []}
# recs: Nujabes +1 (in-lane), Q-Tip +1 (in-lane), Madlib +0.5 (area), Kamasi untagged (ignored)
# → avg = (1 + 1 + 0.5) / 3 = 0.833 → lane_score = (0.833 + 1) / 2 = 0.917
st_lb = goldenset.score_response_structure(recs, golden_lb)
e = st_lb["checks"]["lane_balance"]
assert e["pass"] is True and abs(e["observed"] - 0.917) < 0.01, e
assert e["detail"] == {"in_lane": 2, "in_area": 1, "out": 0, "untagged": 1}, e["detail"]
assert st_lb["total"] == 2.0 and abs(st_lb["points"] - 2.0 * 0.917) < 0.02
assert abs(st_lb["lane_balance"] - 0.917) < 0.01
print("lane_balance OK →", e["observed"], e["detail"])

# out-of-lane recs SUBTRACT: add Post Malone (tagged pop → −1)
# → avg = (1 + 1 + 0.5 − 1) / 4 = 0.375 → lane_score = 0.688
lb_recs = recs + [R("Post Malone", "Beerbongs", "pop", year=2018)]
st_lb2 = goldenset.score_response_structure(lb_recs, golden_lb)
e2 = st_lb2["checks"]["lane_balance"]
assert abs(e2["observed"] - 0.688) < 0.01 and e2["detail"]["out"] == 1, e2
assert st_lb2["points"] < st_lb["points"]   # the pop rec actively hurt
# all-out: every tagged rec outside the area → 0.0, no credit
st_lb3 = goldenset.score_response_structure(bad_recs, golden_lb)
e3 = st_lb3["checks"]["lane_balance"]
assert e3["observed"] == 0.0 and e3["pass"] is False and st_lb3["points"] == 0.0, e3
print("lane_balance penalties OK →", e2["observed"], e3["observed"])

# unmeasurable ≠ failed: all-untagged recs → pass=None, skipped, score=None
golden_mdg = {"tags": tags, "constraints": [
    {"check": "lane_balance", "lane": "jazz hiphop", "area": "hiphop", "level": "fine",
     "weight": 2},
    {"check": "min_distinct_genres", "level": "coarse", "value": 2, "weight": 1},
    {"check": "max_single_genre_share", "level": "coarse", "value": 0.75, "weight": 1},
    {"check": "min_distinct_decades", "value": 2, "weight": 1},
], "justification_keywords": []}
untagged_recs = [R("Kamasi", "The Composer", "jazz", year=2012),
                 R("Somebody New", "An Album", "x", year=2020)]
st_un = goldenset.score_response_structure(untagged_recs, golden_mdg)
assert st_un["checks"]["lane_balance"]["pass"] is None
assert st_un["checks"]["min_distinct_genres"]["pass"] is None
assert st_un["checks"]["max_single_genre_share"]["pass"] is None
# decades IS measurable here (Spotify years on both recs): {2010s, 2020s} = 2 → pass
assert st_un["checks"]["min_distinct_decades"]["pass"] is True
assert st_un["total"] == 1.0 and st_un["score"] == 1.0, (st_un["total"], st_un["score"])
# no eras at all → decades unmeasurable too
noyear = [R("Kamasi", "The Composer", "jazz"), R("Somebody New", "An Album", "x")]
st_un2 = goldenset.score_response_structure(noyear, golden_mdg)
assert st_un2["checks"]["min_distinct_decades"]["pass"] is None
assert st_un2["score"] is None
print("unmeasurable OK → pass=None (not False), score=None when nothing is known")

# ── release_year through the pipeline (4-tuple verifier) ─────────────────────
import json as _json
class FakeLLM(BaseLLM):
    name = "fake"
    def generate(self, prompt, **kw):
        if "recommend exactly" in prompt:          # album-first agent (the default): one-shot list
            return _json.dumps([
                {"artist": "Nujabes", "album": "Modal Soul", "confidence": 0.9, "reason": "soft boom-bap"},
                {"artist": "Q-Tip", "album": "Amplified", "confidence": 0.8, "reason": "jazz chops"},
                {"artist": "Madlib", "album": "Lodger", "confidence": 0.7, "reason": "lo-fi digging"},
            ]), GenerationMeta(name="fake", latency_s=0.01)
        if "recommend ONE real album" in prompt:   # artist-then-album agent: call 2 is batched
            return _json.dumps([
                {"artist": "Nujabes", "album": "Modal Soul", "confidence": 0.9, "reason": "soft boom-bap"},
                {"artist": "Q-Tip", "album": "Amplified", "confidence": 0.8, "reason": "jazz chops"},
                {"artist": "Madlib", "album": "Lodger", "confidence": 0.7, "reason": "lo-fi digging"},
            ]), GenerationMeta(name="fake", latency_s=0.01)
        # retry
        return _json.dumps([{"artist": "Madlib", "album": "Y", "confidence": 0.7, "reason": "x"}]), GenerationMeta(name="fake", latency_s=0.01)

YEARS = {"Nujabes": 2005, "Q-Tip": 1997, "Madlib": 2008}
def v4(artist, album):
    return True, album, f"https://example/{artist}", YEARS.get(artist)
out4 = pipeline.run(evaluate.discover_profiles()[0]["reviews"], FakeLLM(), verifier=v4)
assert out4.ok
assert all(r.release_year is not None for r in out4.recs), [r.release_year for r in out4.recs]
print("4-tuple verifier OK →", [(r.artist, r.release_year) for r in out4.recs])

# 3-tuple verifier (old fakes/tests) still works → release_year stays None
def v3(artist, album):
    return True, album, f"https://example/{artist}"
out3 = pipeline.run(evaluate.discover_profiles()[0]["reviews"], FakeLLM(), verifier=v3)
assert out3.ok and all(r.release_year is None for r in out3.recs)
print("3-tuple verifier OK (back-compat)")

# ── score_outcome picks the new metrics up ────────────────────────────────────
prof = None
for p in evaluate.discover_profiles():
    if p["name"] == "jazzy_hiphop":
        prof = p
assert prof is not None, "jazzy_hiphop fixture expected"
outcome = pipeline.run(prof["reviews"], FakeLLM(), verifier=v4)
m = evaluate.score_outcome(prof, outcome, True)
assert m["structure_total"] is not None and m["structure_points"] is not None, m
assert m["just_total"] == m["n_recs"]
print("score_outcome OK →", {k: m[k] for k in ("structure_points", "structure_total", "structure_score", "lane_share", "just_hits", "just_fixture_hits")})

print("\n✅ test_structure: all checks passed")
