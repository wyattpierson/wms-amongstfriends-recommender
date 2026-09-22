# Offline integration test: runs the real pipeline + evaluator against a fake LLM.
import json, sys
from pathlib import Path

# Tests live at eval/tests/. Expose the repo root (for `afrec`) and the eval
# dir (for `evaluate` / `goldenset`) on sys.path.
_TESTS_DIR = Path(__file__).resolve().parent   # …/eval/tests
_EVAL_DIR  = _TESTS_DIR.parent                 # …/eval
_REPO_ROOT = _EVAL_DIR.parent                  # …/amongstfriends-recommender
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EVAL_DIR))

from afrec import pipeline
from afrec.llm import BaseLLM, GenerationMeta
import evaluate


class FakeLLM(BaseLLM):
    """Returns deterministic artist/album JSON so the full pipeline can run offline."""
    name = "fake"
    def __init__(self):
        self.calls = []
    def generate(self, prompt, **kw):
        self.calls.append(prompt)
        if "suggest 8" in prompt or "NOT already listened" in prompt:
            return '["Sufjan Stevens","James Blake","Grouper","Nicolas Jaar","Arca","Shamir","Carpenter Brut","Lido"]', GenerationMeta(name="fake", latency_s=0.01)
        if "recommend ONE real album" in prompt:
            return json.dumps([
                {"artist":"Sufjan Stevens","album":"Carrie & Lowell","confidence":0.9,"reason":"fits the mood"},
                {"artist":"James Blake","album":"overgrown","confidence":0.8,"reason":"fits"},
                {"artist":"Grouper","album":"Beeswax","confidence":0.7,"reason":"fits"},
                {"artist":"Nicolas Jaar","album":"Space Is Only Noise","confidence":0.6,"reason":"fits"},
                {"artist":"Arca","album":"Mulatta","confidence":0.7,"reason":"fits"},
                {"artist":"Shamir","album":"Nervous","confidence":0.6,"reason":"fits"},
                {"artist":"Carpenter Brut","album":"Surgery","confidence":0.8,"reason":"fits"},
                {"artist":"Lido","album":"The New World Order","confidence":0.7,"reason":"fits"},
            ]), GenerationMeta(name="fake", latency_s=0.01)
        # retry
        return json.dumps([{"artist":"Nicolas Jaar","album":"Cumbia","confidence":0.6,"reason":"fits"}]), GenerationMeta(name="fake", latency_s=0.01)

# Use one profile
prof = evaluate.discover_profiles()[0]
print("Using profile:", prof["name"])

llm = FakeLLM()
# Spotify off path
outcome = pipeline.run(prof["reviews"], llm, verifier=None)
m = evaluate.score_outcome(prof, outcome, False)
assert outcome.ok, "pipeline should produce recs"
assert len(outcome.recs) >= 6, f"expected several recs, got {len(outcome.recs)}"
assert m["quality"] > 0
print("no-spotify OK →", {k:m[k] for k in ("n_recs","n_new_artists","avg_confidence","instruction_score","quality") if k in m})

# Spotify-on path: stub verifier that 'finds' everything → verified share high
def fake_verifier(artist, album):
    return True, album, f"https://open.spotify.com/album/{artist}"
outcome2 = pipeline.run(prof["reviews"], llm, verifier=fake_verifier)
m2 = evaluate.score_outcome(prof, outcome2, True)
assert m2["verified"] == m2["n_recs"] and m2.get("verified_share") == 1.0
print("spotify-on  OK →", {k:m2[k] for k in ("n_recs","verified","verified_share","quality") if k in m2})

# Leaderboard render smoke test
class A: pass
a = A(); a.base_url="x"; a.temperature=0.7
res = [{"model":"fake","base_url":"x","spotify":True,"avg_quality":m2["quality"],
        "n_ok":1,"n_total":1,"total_wallclock_s":0.01,
        "per_profile":{prof["name"]:{
            "metrics":{**m2,"status":"ok","wallclock_s":0.01},
            "outcome":{}}}}]
lb = evaluate.render_leaderboard(res, a)
assert "# AmongstFriends engine leaderboard" in lb and "fake" in lb
print("\nLeaderboard:\n", lb, "\nOK")
