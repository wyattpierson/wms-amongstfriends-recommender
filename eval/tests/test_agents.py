"""
Tests for the agent registry + the pipeline facade. Fully offline.

Covers:
  - both built-in agents are registered (artist-then-album@v1, album-first@v1)
  - tag resolution: full tag, bare id (→ latest), retired alias (two-call)
  - every run is stamped with its agent tag (via registry AND via facade)
  - the facade dispatches to the selected agent by default
  - selection: AFREC_AGENT env var wins; unknown env var fails loudly
  - publishing a new version: bare id resolves to latest, old version intact
  - the reference table renders and marks the selected agent
"""

import json
import os
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_TESTS_DIR.parent))
os.environ["LLM_BASE_URL"] = "http://localhost:1"
os.environ.pop("AFREC_AGENT", None)

from dataclasses import asdict

from afrec import agents as afrec_agents
from afrec import pipeline
from afrec.llm import BaseLLM, GenerationMeta
import evaluate


class FakeLLM(BaseLLM):
    name = "fake-agent-test"

    def generate(self, prompt, *, temperature=0.7, num_predict=None, options=None, timeout=None):
        if "recommend exactly 8 real albums" in prompt:          # album-first one-shot
            return (json.dumps([
                {"artist": "a", "album": "A1", "confidence": 0.9, "reason": "r"},
                {"artist": "b", "album": "B1", "confidence": 0.8, "reason": "r"},
                {"artist": "c", "album": "C1", "confidence": 0.7, "reason": "r"},
                {"artist": "d", "album": "D1", "confidence": 0.7, "reason": "r"},
                {"artist": "e", "album": "E1", "confidence": 0.6, "reason": "r"},
                {"artist": "f", "album": "F1", "confidence": 0.6, "reason": "r"},
                {"artist": "g", "album": "G1", "confidence": 0.5, "reason": "r"},
                {"artist": "J Dilla", "album": "Donuts", "confidence": 0.9, "reason": "repeat"},
            ]), GenerationMeta(name=self.name, latency_s=0.01, prompt_tokens=5, completion_tokens=5))
        if "suggest 8" in prompt or "NOT already listened" in prompt:  # artist-then-album call 1
            return (json.dumps(["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
                                "I1", "J1", "J Dilla"]),
                    GenerationMeta(name=self.name, latency_s=0.01, prompt_tokens=5, completion_tokens=5))
        # album recall / retry
        return (json.dumps([
            {"artist": "A1", "album": "AA", "confidence": 0.9, "reason": "r"},
            {"artist": "B1", "album": "BB", "confidence": 0.8, "reason": "r"},
            {"artist": "C1", "album": "CC", "confidence": 0.7, "reason": "r"},
            {"artist": "D1", "album": "DD", "confidence": 0.7, "reason": "r"},
            {"artist": "E1", "album": "EE", "confidence": 0.6, "reason": "r"},
            {"artist": "F1", "album": "FF", "confidence": 0.6, "reason": "r"},
            {"artist": "G1", "album": "GG", "confidence": 0.5, "reason": "r"},
            {"artist": "H1", "album": "HH", "confidence": 0.5, "reason": "r"},
            {"artist": "I1", "album": "II", "confidence": 0.4, "reason": "r"},
            {"artist": "J1", "album": "JJ", "confidence": 0.4, "reason": "r"},
        ]), GenerationMeta(name=self.name, latency_s=0.01, prompt_tokens=5, completion_tokens=5))


PROF = next(p for p in evaluate.discover_profiles() if p["name"] == "jazzy_hiphop")
REVIEWS = PROF["reviews"]

passed = []
def ok(name, cond=True):
    passed.append(name)
    print(f"  ✅ {name}" if cond else f"  ❌ {name}")
    assert cond, name


# ── 1. registry basics ────────────────────────────────────────────────────────
tags = [a.tag for a in afrec_agents.list_agents()]
ok("artist-then-album@v1 registered", "artist-then-album@v1" in tags)
ok("album-first@v1 registered", "album-first@v1" in tags)
ok("default is artist-then-album@v1", afrec_agents.default_agent().tag == "artist-then-album@v1")

a = afrec_agents.get_agent("artist-then-album@v1")
b = afrec_agents.get_agent("album-first@v1")
ok("get_agent by full tag", a.id == "artist-then-album" and b.id == "album-first")
ok("bare id resolves to latest of that id",
   afrec_agents.get_agent("album-first").tag == "album-first@v1")
ok("agent has a summary", len(a.summary) > 10 and len(b.summary) > 10)
ok("agent details describe the knobs", "calls" in a.details and "calls" in b.details)

# retired alias: old two-call name still resolves
ok("alias two-call@v1 → artist-then-album@v1",
   afrec_agents.get_agent("two-call@v1").tag == "artist-then-album@v1")

try:
    afrec_agents.get_agent("bogus@v9")
    ok("unknown tag rejected", False)
except ValueError:
    ok("unknown tag rejected")

# ── 2. both agents run end-to-end offline and stamp their tag ─────────────────
out_a = afrec_agents.get_agent("artist-then-album").run(REVIEWS, FakeLLM())
ok("artist-then-album runs offline", out_a.ok and len(out_a.recs) >= 8)
ok("artist-then-album stamped", out_a.agent == "artist-then-album@v1")
ok("artist-then-album did two calls", [s.stage for s in out_a.stages][:2] == ["artists", "albums"])

out_b = afrec_agents.get_agent("album-first").run(REVIEWS, FakeLLM())
ok("album-first runs offline", out_b.ok and len(out_b.recs) >= 7)
ok("album-first stamped", out_b.agent == "album-first@v1")
ok("album-first did ONE generation call", [s.stage for s in out_b.stages][0] == "albums_first")
ok("album-first filtered the repeated artist",
   len(out_b.repeated_artist_violations) == 1 and out_b.repeated_artist_violations[0].lower() == "j dilla")

# ── 3. facade: pipeline.run dispatches to the selected agent ──────────────────
sel = afrec_agents.selected_agent()
ok("selection falls back to the default", sel.tag == "artist-then-album@v1")

out_sel = pipeline.run(REVIEWS, FakeLLM())
ok("facade with no agent → selected agent", out_sel.agent == "artist-then-album@v1")

out_fac = pipeline.run(REVIEWS, FakeLLM(), agent="album-first")
ok("facade with bare id → that agent", out_fac.agent == "album-first@v1")
out_fac2 = pipeline.run(REVIEWS, FakeLLM(), agent=afrec_agents.get_agent("album-first@v1"))
ok("facade accepts an Agent instance", out_fac2.agent == "album-first@v1")

# selection via env var
os.environ["AFREC_AGENT"] = "album-first@v1"
try:
    ok("AFREC_AGENT selects album-first", afrec_agents.selected_agent().tag == "album-first@v1")
    out_env = pipeline.run(REVIEWS, FakeLLM())
    ok("facade follows AFREC_AGENT", out_env.agent == "album-first@v1")
finally:
    os.environ.pop("AFREC_AGENT", None)

os.environ["AFREC_AGENT"] = "nope@v1"
try:
    afrec_agents.selected_agent()
    ok("bad AFREC_AGENT fails loudly", False)
except ValueError:
    ok("bad AFREC_AGENT fails loudly")
finally:
    os.environ.pop("AFREC_AGENT", None)

# ── 4. publishing a new version ───────────────────────────────────────────────
def run_v2(reviews, llm, **kw):
    o = pipeline.RunOutcome()
    o.ok = True
    o.recs = [pipeline.Rec(artist="V2", album="V2 Album", confidence=1.0, verified=True)]
    return o

v2 = afrec_agents._register(
    id="album-first", version="v2", name="album-first v2",
    summary="test v2", run=run_v2,
)
ok("new version registered", v2.tag == "album-first@v2")
ok("bare id now → v2 (latest registered)", afrec_agents.get_agent("album-first").tag == "album-first@v2")
ok("old version still addressable", afrec_agents.get_agent("album-first@v1").tag == "album-first@v1")
try:
    afrec_agents._register(id="album-first", version="v2", name="dup", summary="x", run=run_v2)
    ok("duplicate version rejected", False)
except ValueError:
    ok("duplicate version rejected")
o2 = pipeline.run(REVIEWS, FakeLLM(), agent="album-first")
ok("facade uses latest for bare id", o2.agent == "album-first@v2")

# ── 5. the reference table ────────────────────────────────────────────────────
tbl = afrec_agents.render_table()
ok("table lists both agents", "artist-then-album@v1" in tbl and "album-first@v1" in tbl)
ok("table marks the selected agent", "(selected" in tbl)
ok("table marks the default", "(selected, default)" in tbl)

print(f"\nAll {len(passed)} agent-registry checks passed.")
