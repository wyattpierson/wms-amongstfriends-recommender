"""
Tests for the agent registry + the pipeline facade. Fully offline.

Covers:
  - both built-in agents are registered (artist-then-album@v1, album-first@v1)
  - tag resolution: full tag, bare id (→ latest), retired alias (two-call)
  - every run is stamped with its agent tag (via registry AND via facade)
  - the facade dispatches to the selected agent by default
  - pool-map@v1: full 4-call flow offline (map → discovery → critic → role retry),
    critic rejects/replaces, recs carry role+lane, and graceful degradation
    when the map call fails (pinned to the full tag — bare id now means v2.0)
  - pool-map@v1.1: grounded map, existence-first discovery, narrow critic,
    role-retry with slack, artist dedupe, topup, suggested_artists, degradation
  - pool-map@v1.2: catalog FACTS in call 0 (fact line rendered, Spotify year
    lookups skipped), fallback to bare years without facts, registry injection
    (enricher → v1.2 gets facts, v1.1 never does — the A/B invariant),
    and mid-run latch degradation (enrichment_error stamped, run still ok)
  - pool-map@v2.0: coherence-driven plan in call 0 — the core/adjacent/stretch
    split is set from the pool's range (coherent→constricted to 0 stretch,
    diverse→exploratory); the ONLY delta vs v1.2, so an A/B isolates it
    (fact line rendered, downstream flow/metrics identical to v1.2)
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
        meta = GenerationMeta(name=self.name, latency_s=0.01, prompt_tokens=5, completion_tokens=5)
        if "POOL MAP" in prompt:                                  # pool-map call 0
            return (json.dumps({
                "reads": [
                    {"album": "J Dilla — Donuts", "genre": "hiphop", "era": "2000s", "energy": 5, "known": True},
                ],
                "lanes": [
                    {"name": "boom bap", "why": "loopy 90s/00s hiphop", "albums": ["J Dilla — Donuts"], "energy": 5, "strength": "strong"},
                    {"name": "downtempo soul", "why": "lazy beats and warm vocals", "albums": ["a — A1"], "energy": 3, "strength": "moderate"},
                ],
                "axes": {"genre_spread": "medium", "era_spread": "narrow", "energy_range": [3, 6], "mood": ["warm", "laid-back"]},
                "fingerprint": ["prefers dusty drum breaks over crisp 808s"],
                "attracts": ["warm vocals", "loose grooves"],
                "repels": ["auto-tuned pop"],
                "surprises": ["one post-rock 5-star"],
                "tension": "none",
                "threads": ["shared sample sources", "detroit scene"],
                "plan": {"core": 4, "adjacent": 3, "stretch": 1},
            }), meta)
        if ("the brief you must work from" in prompt
                or "No pool map was produced" in prompt):         # pool-map discovery
            return (json.dumps([
                {"artist": "PA", "album": "PA1", "confidence": 0.9, "reason": "r", "genre": "hiphop", "era": "1990s", "role": "core", "lane": "boom bap"},
                {"artist": "PB", "album": "PB1", "confidence": 0.9, "reason": "r", "genre": "hiphop", "era": "2000s", "role": "core", "lane": "boom bap"},
                {"artist": "PC", "album": "PC1", "confidence": 0.8, "reason": "r", "genre": "soul", "era": "1970s", "role": "core", "lane": "downtempo soul"},
                {"artist": "PD", "album": "PD1", "confidence": 0.8, "reason": "r", "genre": "jazz", "era": "1960s", "role": "adjacent", "lane": "detroit scene"},
                {"artist": "PE", "album": "PE1", "confidence": 0.8, "reason": "r", "genre": "soul", "era": "1980s", "role": "adjacent", "lane": "warm vocals"},
                {"artist": "PF", "album": "PF1", "confidence": 0.7, "reason": "r", "genre": "hiphop", "era": "2010s", "role": "adjacent", "lane": "shared sample sources"},
                {"artist": "PG", "album": "PG1", "confidence": 0.7, "reason": "r", "genre": "downtempo", "era": "2000s", "role": "stretch", "lane": "loose grooves"},
                {"artist": "Nujabes", "album": "Modal Soul", "confidence": 0.9, "reason": "repeat"},
            ]), meta)
        if "skeptical music critic" in prompt:                    # pool-map critic
            return (json.dumps({
                "verdicts": [
                    {"artist": "PB", "album": "PB1", "verdict": "reject", "reason": "generic boom bap pick, ignores the fingerprint"},
                ],
                "unfilled_roles": ["stretch"],
            }), meta)
        if "replacement albums" in prompt:                        # pool-map critic_retry
            return (json.dumps([
                {"artist": "PR", "album": "PR1", "confidence": 0.8, "reason": "r", "genre": "post-rock", "era": "2000s", "role": "stretch", "lane": "the post-rock surprise"},
                {"artist": "PS", "album": "PS1", "confidence": 0.8, "reason": "r", "genre": "soul", "era": "1970s", "role": "adjacent", "lane": "downtempo soul"},
            ]), meta)
        if "recommend exactly 8 real albums" in prompt:          # album-first one-shot
            return (json.dumps([
                {"artist": "a", "album": "A1", "confidence": 0.9, "reason": "r"},
                {"artist": "b", "album": "B1", "confidence": 0.8, "reason": "r"},
                {"artist": "c", "album": "C1", "confidence": 0.7, "reason": "r"},
                {"artist": "d", "album": "D1", "confidence": 0.7, "reason": "r"},
                {"artist": "e", "album": "E1", "confidence": 0.6, "reason": "r"},
                {"artist": "f", "album": "F1", "confidence": 0.6, "reason": "r"},
                {"artist": "g", "album": "G1", "confidence": 0.5, "reason": "r"},
                {"artist": "Nujabes", "album": "Modal Soul", "confidence": 0.9, "reason": "repeat"},
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
ok("pool-map@v1.1 registered", "pool-map@v1.1" in tags)
ok("pool-map@v1.2 registered", "pool-map@v1.2" in tags)
ok("pool-map@v2.0 registered", "pool-map@v2.0" in tags)
ok("bare pool-map resolves to v2.0 (latest)",
   afrec_agents.get_agent("pool-map").tag == "pool-map@v2.0")
ok("v1.1 still addressable by full tag", afrec_agents.get_agent("pool-map@v1.1").tag == "pool-map@v1.1")

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
   len(out_b.repeated_artist_violations) == 1 and out_b.repeated_artist_violations[0].lower() == "nujabes")

# ── 2b. pool-map: 4-call flow, critic, graceful degradation ─────────────────────
out_p = afrec_agents.get_agent("pool-map@v1").run(REVIEWS, FakeLLM())
ok("pool-map registered + runs offline", out_p.ok and len(out_p.recs) == 8)
ok("pool-map stamped", out_p.agent == "pool-map@v1")
ok("pool-map stage sequence",
   [s.stage for s in out_p.stages] == ["pool_map", "pool_discovery", "critic", "critic_retry"])
ok("pool map stored on the outcome",
   out_p.pool_map is not None and any(l.get("name") == "boom bap" for l in out_p.pool_map["lanes"]))
ok("critic rejected one rec", out_p.critic_rejected == 1)
ok("critic_retry rescued two replacements", out_p.critic_rescued == 2)
ok("rejected rec removed from final list",
   all(not (r.artist == "PB" and r.album == "PB1") for r in out_p.recs))
ok("recs carry role + lane", out_p.recs[0].role == "core" and out_p.recs[0].lane == "boom bap")
ok("role-targeted replacements carry roles",
   any(r.role == "stretch" and r.lane for r in out_p.recs))
ok("pool-map filtered the repeated artist", len(out_p.repeated_artist_violations) == 1)


class NoMapLLM(FakeLLM):
    """Map call returns garbage → the agent must degrade gracefully."""

    def generate(self, prompt, **kw):
        if "POOL MAP" in prompt:
            return ("I cannot produce a JSON object right now.",
                    GenerationMeta(name=self.name, latency_s=0.01, prompt_tokens=5, completion_tokens=5))
        return super().generate(prompt, **kw)


out_nm = afrec_agents.get_agent("pool-map@v1").run(REVIEWS, NoMapLLM())
ok("map failure degrades to unguided discovery", out_nm.ok and len(out_nm.recs) == 7)
ok("no pool map stored on degradation", out_nm.pool_map is None)
ok("critic skipped on degradation",
   [s.stage for s in out_nm.stages] == ["pool_map", "pool_discovery"])


class NukeCriticLLM(FakeLLM):
    """A miscalibrated critic that rejects EVERYTHING — the circuit breaker
    must ignore such a pass instead of emptying the rec list."""

    def generate(self, prompt, **kw):
        if "skeptical music critic" in prompt:
            meta = GenerationMeta(name=self.name, latency_s=0.01, prompt_tokens=5, completion_tokens=5)
            return (json.dumps({
                "verdicts": [
                    {"artist": a, "album": b, "verdict": "reject", "reason": "no"}
                    for a, b in [("PA", "PA1"), ("PB", "PB1"), ("PC", "PC1"), ("PD", "PD1"),
                                 ("PE", "PE1"), ("PF", "PF1"), ("PG", "PG1")]
                ],
                "unfilled_roles": ["core", "adjacent", "stretch"],
            }), meta)
        return super().generate(prompt, **kw)


out_nuke = afrec_agents.get_agent("pool-map@v1").run(REVIEWS, NukeCriticLLM())
ok("circuit breaker ignores a nuke critic", out_nuke.critic_rejected == 0)
ok("nuke-critic recs survive", any(r.artist == "PB" for r in out_nuke.recs))
ok("nuke-critic run still fills the list", out_nuke.ok and len(out_nuke.recs) == 8)


# ── 2c. pool-map v1.1: grounded map, existence-first, dedupe, topup, metrics ─
class FakeV11LLM(FakeLLM):
    """Routes the v1.1 markers; the map call falls through to FakeLLM
    (same JSON shape — v1.1 map only changes the prompt, not the schema).

    Discovery returns 9 entries to exercise every v1.1 backstop:
      PA twice (artist dedupe), PB + PD (critic rejects), Nujabes (reviewed).
    critic_retry returns ONE rescue (so the batch is still short), and the
    topup returns a duplicate artist + two new ones (dedupe-skip, fill to 8).
    """

    def generate(self, prompt, **kw):
        meta = GenerationMeta(name=self.name, latency_s=0.01, prompt_tokens=5, completion_tokens=5)
        if "RECOMMEND (v1.1)" in prompt:                          # discovery
            return (json.dumps([
                {"artist": "PA", "album": "PA1", "confidence": 0.9, "reason": "r", "role": "core", "lane": "boom bap"},
                {"artist": "PA", "album": "PA2", "confidence": 0.9, "reason": "r", "role": "core", "lane": "boom bap"},
                {"artist": "PB", "album": "PB1", "confidence": 0.9, "reason": "r", "role": "core", "lane": "boom bap"},
                {"artist": "PC", "album": "PC1", "confidence": 0.8, "reason": "r", "role": "core", "lane": "downtempo soul"},
                {"artist": "PD", "album": "PD1", "confidence": 0.8, "reason": "r", "role": "adjacent", "lane": "detroit scene"},
                {"artist": "PE", "album": "PE1", "confidence": 0.8, "reason": "r", "role": "adjacent", "lane": "warm vocals"},
                {"artist": "PF", "album": "PF1", "confidence": 0.7, "reason": "r", "role": "adjacent", "lane": "sample sources"},
                {"artist": "PG", "album": "PG1", "confidence": 0.7, "reason": "r", "role": "stretch", "lane": "loose grooves"},
                {"artist": "Nujabes", "album": "Modal Soul", "confidence": 0.9, "reason": "repeat"},
            ]), meta)
        if "FILL THE REST" in prompt:                             # topup
            return (json.dumps([
                {"artist": "PA", "album": "PA9", "confidence": 0.7, "reason": "r", "role": "core", "lane": "boom bap"},
                {"artist": "PT", "album": "PT1", "confidence": 0.8, "reason": "r", "role": "core", "lane": "boom bap"},
                {"artist": "PU", "album": "PU1", "confidence": 0.7, "reason": "r", "role": "core", "lane": "downtempo soul"},
            ]), meta)
        if "skeptical music critic" in prompt:                    # narrow critic
            return (json.dumps({
                "verdicts": [
                    {"artist": "PB", "album": "PB1", "verdict": "reject", "reason": "lane void — no lane or thread covers it"},
                    {"artist": "PD", "album": "PD1", "verdict": "reject", "reason": "repel collision — echoes review [#4]"},
                ],
                "unfilled_roles": ["stretch"],
            }), meta)
        if "replacement albums" in prompt:                        # critic_retry (with slack)
            return (json.dumps([
                {"artist": "PR", "album": "PR1", "confidence": 0.8, "reason": "r", "role": "stretch", "lane": "the post-rock surprise"},
            ]), meta)
        return super().generate(prompt, **kw)


class NoMapV11LLM(FakeV11LLM):
    def generate(self, prompt, **kw):
        if "POOL MAP" in prompt:
            return ("{no json today",
                    GenerationMeta(name=self.name, latency_s=0.01, prompt_tokens=5, completion_tokens=5))
        return super().generate(prompt, **kw)


logs = []
out_v11 = afrec_agents.get_agent("pool-map@v1.1").run(REVIEWS, FakeV11LLM(), log=logs.append)
ok("v1.1 runs offline and fills 8", out_v11.ok and len(out_v11.recs) == 8)
ok("v1.1 stamped", out_v11.agent == "pool-map@v1.1")
ok("v1.1 full stage sequence (incl. topup)",
   [s.stage for s in out_v11.stages] == ["pool_map", "pool_discovery", "critic", "critic_retry", "topup"])
ok("v1.1 map stored (grounded)",
   out_v11.pool_map is not None and any(l.get("name") == "boom bap" for l in out_v11.pool_map["lanes"]))
ok("v1.1 critic rejected two", out_v11.critic_rejected == 2)
ok("v1.1 critic_retry rescued one", out_v11.critic_rescued == 1)
ok("v1.1 rejected recs gone from the list",
   all(not ((r.artist, r.album) in {("PB", "PB1"), ("PD", "PD1")}) for r in out_v11.recs))
ok("v1.1 filtered the reviewed artist", len(out_v11.repeated_artist_violations) == 1)
ok("v1.1 deterministic artist dedupe fired",
   any("dedupe" in ln and "PA" in ln for ln in logs))
ok("v1.1 no duplicate artists in the final list",
   len({r.artist.lower() for r in out_v11.recs}) == len(out_v11.recs))
ok("v1.1 topup filled to 8", any(r.album == "PU1" for r in out_v11.recs))
ok("v1.1 topup skipped the duplicate artist",
   not any(r.album == "PA9" for r in out_v11.recs))
ok("v1.1 sets suggested_artists (the metrics bug)",
   out_v11.suggested_artists == [r.artist for r in out_v11.recs] and len(out_v11.suggested_artists) == 8)
ok("v1.1 recs carry role + lane", out_v11.recs[0].role == "core" and out_v11.recs[0].lane)

logs = []
out_v11d = afrec_agents.get_agent("pool-map@v1.1").run(REVIEWS, NoMapV11LLM(), log=logs.append)
ok("v1.1 degrades without a map (no critic, still tops up)",
   out_v11d.ok and [s.stage for s in out_v11d.stages] == ["pool_map", "pool_discovery", "topup"])
ok("v1.1 degraded run still fills 8", len(out_v11d.recs) == 8)
ok("v1.1 degraded run sets suggested_artists", len(out_v11d.suggested_artists) == 8)

# ── 2d. pool-map v1.2: catalog facts in call 0, fallback, injection, latch ────
from afrec.enrich import CatalogFact, name_key


REV0 = REVIEWS[0]
FACT = CatalogFact(year=1998, tags=["jazz", "hiphop"],
                   features={"energy": 0.7, "danceability": 0.6, "valence": 0.5},
                   bpm=88.0, key_int=2, mode="major")
FACTS = {name_key(REV0["artist"], REV0["album"]): FACT}


class RecordingV12LLM(FakeV11LLM):
    """FakeV11LLM + prompt recording; routes the v1.2 map header and keeps
    v1.1's discovery/critic/topup routes (v1.2 reuses those templates)."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return super().generate(prompt, **kw)


logs = []
llm12 = RecordingV12LLM()
out_v12 = afrec_agents.get_agent("pool-map@v1.2").run(REVIEWS, llm12,
                                                       catalog_facts=FACTS, log=logs.append)
ok("v1.2 runs offline with facts and fills 8", out_v12.ok and len(out_v12.recs) == 8)
ok("v1.2 stamped", out_v12.agent == "pool-map@v1.2")
ok("v1.2 map call renders the full fact line", FACT.catalog_line() in llm12.prompts[0])
ok("v1.2 map call header says v1.2", "POOL MAP (v1.2)" in llm12.prompts[0])
ok("v1.2 map call has the facts contract note", "catalog FACTS" in llm12.prompts[0])
ok("v1.2 skipped the Spotify year lookups",
   any("skipping Spotify year lookups" in ln for ln in logs))
ok("v1.2 discovery still uses the v1.1 template", "RECOMMEND (v1.1)" in llm12.prompts[1])
ok("v1.2 downstream flow is v1.1's", 
   [s.stage for s in out_v12.stages] == ["pool_map", "pool_discovery", "critic", "critic_retry", "topup"])


# Fallback: no facts → bare Spotify year lookups (v1.1's path), year in prompt
class YearVerifier:
    """Counts calls; reports release year 1990 for every album."""

    def __init__(self):
        self.calls = 0

    def __call__(self, artist, album):
        self.calls += 1
        return (artist, album, "seed", 1990)


logs = []
llm12b = RecordingV12LLM()
ver = YearVerifier()
out_v12b = afrec_agents.get_agent("pool-map@v1.2").run(
    REVIEWS, llm12b, verifier=ver, log=logs.append)
ok("v1.2 without facts still fills 8", out_v12b.ok and len(out_v12b.recs) == 8)
ok("v1.2 without facts does Spotify year lookups",
   any("catalog years for" in ln for ln in logs))
ok("v1.2 without facts renders bare years in the map",
   f"(catalog: 1990)" in llm12b.prompts[0])
ok("v1.2 without facts keeps the year-only note",
   "catalog: YYYY\" carry a release year" in llm12b.prompts[0])
ok("v1.2 without facts does NOT render the fact line",
   FACT.catalog_line() not in llm12b.prompts[0])


# Registry injection: an enricher is offered to the run, not the playbook.
class FakeEnricher:
    available = True
    error = None

    def fact_map(self, pairs):
        return dict(FACTS)


class LatchedEnricher(FakeEnricher):
    error = "musicae 429 quota"   # fact_map still serves the cache


logs = []
llm12c = RecordingV12LLM()
out_v12c = afrec_agents.get_agent("pool-map@v1.2").run(
    REVIEWS, llm12c, enricher=FakeEnricher(), log=logs.append)
ok("registry injects enricher facts into v1.2", FACT.catalog_line() in llm12c.prompts[0])
ok("enricher did not leak into the playbook kwargs", out_v12c.ok)

# The A/B invariant: the SAME enricher offered to v1.1 — v1.1's run() has no
# catalog_facts param, so the registry must NOT inject; v1.1 stays v1.1.
logs = []
llm11 = RecordingV12LLM()
out_v11inj = afrec_agents.get_agent("pool-map@v1.1").run(
    REVIEWS, llm11, enricher=FakeEnricher(), log=logs.append)
ok("enricher offered to v1.1 leaves its map prompt fact-free",
   FACT.catalog_line() not in llm11.prompts[0] and "POOL MAP (v1.1)" in llm11.prompts[0])
ok("v1.1 with enricher still stamps v1.1", out_v11inj.agent == "pool-map@v1.1")

# Mid-run latch: facts went cache-only — flag the run, never fail it.
logs = []
llm12d = RecordingV12LLM()
out_v12d = afrec_agents.get_agent("pool-map@v1.2").run(
    REVIEWS, llm12d, enricher=LatchedEnricher(), log=logs.append)
ok("latched enricher: run still ok and fills 8", out_v12d.ok and len(out_v12d.recs) == 8)
ok("latched enricher: enrichment_error stamped on the outcome",
   out_v12d.enrichment_error == "musicae 429 quota")
ok("clean enricher: no enrichment_error", out_v12c.enrichment_error is None)


# ── 2e. pool-map v2.0: coherence-driven plan in call 0 (rest = v1.2's) ───────
# The ONLY delta vs v1.2 is the map call's step-6 "plan" rule, so this section
# pins that the rule is present in v2.0's map prompt, absent from v1.2's, and
# that everything downstream is v1.2's (fact line, stages, fill-to-8).
logs = []
llm20 = RecordingV12LLM()
out_v20 = afrec_agents.get_agent("pool-map@v2.0").run(REVIEWS, llm20,
                                                       catalog_facts=FACTS, log=logs.append)
ok("v2.0 runs offline with facts and fills 8", out_v20.ok and len(out_v20.recs) == 8)
ok("v2.0 stamped", out_v20.agent == "pool-map@v2.0")
ok("v2.0 map call header says v2.0", "POOL MAP (v2.0)" in llm20.prompts[0])
ok("v2.0 map call renders the full fact line", FACT.catalog_line() in llm20.prompts[0])
ok("v2.0 map call carries the coherence rule", "SET THE SPLIT FROM THE RANGE" in llm20.prompts[0])
ok("v1.2 map call lacks the coherence rule (the v2.0-only delta)",
   "SET THE SPLIT FROM THE RANGE" not in llm12.prompts[0])
ok("v2.0 downstream flow is v1.2's (v1.1 templates)",
   [s.stage for s in out_v20.stages] == ["pool_map", "pool_discovery", "critic", "critic_retry", "topup"])

# A/B invariant, v2.0 side: the enricher's facts reach v2.0 (it accepts the
# catalog_facts kwarg, like v1.2); the coherence rule needs no facts to apply.
logs = []
llm20b = RecordingV12LLM()
out_v20b = afrec_agents.get_agent("pool-map@v2.0").run(REVIEWS, llm20b,
                                                        enricher=FakeEnricher(), log=logs.append)
ok("registry injects enricher facts into v2.0", FACT.catalog_line() in llm20b.prompts[0])
ok("v2.0 coherence rule present even with facts (plan step is facts-independent)",
   "SET THE SPLIT FROM THE RANGE" in llm20b.prompts[0] and out_v20b.ok)


# ── 3. facade: pipeline.run dispatches to the selected agent ──────────────────
# afrec.firebase re-loads .env on import (which sets AFREC_AGENT) — clear it again.
os.environ.pop("AFREC_AGENT", None)
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
