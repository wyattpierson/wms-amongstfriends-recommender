"""
Test that the harness correctly measures how much of the model's output is
actually REAL stuff that exists on Spotify — the validity signal.

This is the honest, pre-retry number: raw_existence_rate = round1_found / total_candidates.

Scenario (fully deterministic, no network, no Ollama):
  - Model proposes 8 albums: 4 that EXIST on Spotify, 4 that are hallucinated.
  - A verifier (simulating Spotify) confirms the 4 real ones, rejects the 4 fakes.
  - The pipeline retries; the model supplies 4 more replacements that all exist.

We assert the harness reports:
  * total_candidates == 8, round1_found == 4, round1_missing == 4
  * raw_existence_rate == 0.5   <- "50% of what the model produced is real" (the key metric)
  * retry_rescued == 4          <- retries fixed the 4 hallucinations
  * final recs == 8, ALL verified (final_existence_rate == 1.0)
  * quality is pulled DOWN by the 0.5 existence rate, not hidden by the retry
"""
import json, sys
from pathlib import Path

# Tests live at eval/tests/. Expose the repo root (for `afrec`) and the eval
# dir (for `evaluate`) on sys.path.
_TESTS_DIR = Path(__file__).resolve().parent   # …/eval/tests
_EVAL_DIR  = _TESTS_DIR.parent                 # …/eval
_REPO_ROOT = _EVAL_DIR.parent                  # …/amongstfriends-recommender
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_EVAL_DIR))

from afrec import pipeline
from afrec.llm import BaseLLM, GenerationMeta
import evaluate


# A fixed set of (artist, album) that "exist on Spotify".
REAL = {
    ("A1", "Real One"), ("A2", "Real Two"), ("A3", "Real Three"), ("A4", "Real Four"),
    ("A5", "Rescue Five"), ("A6", "Rescue Six"), ("A7", "Rescue Seven"), ("A8", "Rescue Eight"),
}


def spotify_like_verifier(artist, album):
    key = (artist.strip().upper(), album.strip().upper())
    if key in {(a.upper(), b.upper()) for (a, b) in REAL}:
        return True, album, f"https://open.spotify.com/album/{artist}/{album}"
    return False, None, None   # hallucinated — Spotify has no such album


def exists(artist, album):
    return (artist.strip().upper(), album.strip().upper()) in {(a.upper(), b.upper()) for (a, b) in REAL}


class HallucinatingLM(BaseLLM):
    name = "hallucinator"
    def generate(self, prompt, **kw):
        meta = GenerationMeta(name="hallucinator", latency_s=0.01)
        # Call 1: artist discovery
        if "suggest 8" in prompt or "NOT already listened" in prompt:
            return json.dumps(["A1","A2","A3","A4","A5","A6","A7","A8"]), meta
        # Call 3 (retry): only this prompt talks about "replacement"
        if "replacement" in prompt:
            # Rescue the 4 that failed, with albums that actually exist
            return json.dumps([
                {"artist":"A5","album":"Rescue Five","confidence":0.8,"reason":"exists"},
                {"artist":"A6","album":"Rescue Six","confidence":0.8,"reason":"exists"},
                {"artist":"A7","album":"Rescue Seven","confidence":0.8,"reason":"exists"},
                {"artist":"A8","album":"Rescue Eight","confidence":0.8,"reason":"exists"},
            ]), meta
        # Call 2: albums — 4 REAL + 4 HALLUCINATED
        return json.dumps([
            {"artist":"A1","album":"Real One","confidence":0.9,"reason":"r"},
            {"artist":"A2","album":"Real Two","confidence":0.9,"reason":"r"},
            {"artist":"A3","album":"Real Three","confidence":0.9,"reason":"r"},
            {"artist":"A4","album":"Real Four","confidence":0.9,"reason":"r"},
            {"artist":"A5","album":"Totally Fake Album","confidence":0.9,"reason":"sounds plausible"},
            {"artist":"A6","album":"Made Up LP","confidence":0.8,"reason":"hmm did they do that?"},
            {"artist":"A7","album":"Imaginary Tapes","confidence":0.8,"reason":"sure, existed?"},
            {"artist":"A8","album":"Nonexistent","confidence":0.7,"reason":"?"},
        ]), meta


def main():
    # Flat, already-parsed review (like evaluate.load_profile gives the pipeline):
    # none of A1..A8 appears here, so nothing gets pre-filtered as "already reviewed".
    reviews = [{"reviewId": "r1", "type": "album", "album": "Old Friends EP",
                "artist": "Some Old Artist", "rating": 4, "text": "I love the music already in my library.",
                "groupId": "g1"}]
    out = pipeline.run(reviews, HallucinatingLM(), verifier=spotify_like_verifier)

    # Raw pipeline accounting
    assert out.ok, "pipeline should still succeed after retries"
    assert out.total_candidates == 8, f"expected 8 round-1 candidates, got {out.total_candidates}"
    assert out.round1_found == 4, f"round1_found={out.round1_found} (expected 4 real on first pass)"
    assert out.round1_missing == 4, f"round1_missing={out.round1_missing} (expected 4 hallucinated)"
    assert out.retry_found == 4, f"retry_found={out.retry_found} (expected 4 rescued)"
    assert len(out.recs) == 8, f"final recs={len(out.recs)} (expected 8 after rescue)"
    assert all(r.verified for r in out.recs), "all final recs must be verified/real"

    # Evaluator metrics
    prof = {"name": "existence_probe", "description": "", "reviews": reviews}
    m = evaluate.score_outcome(prof, out, spotify_ons=True)

    assert m["raw_candidates"] == 8
    assert m["raw_existence_rate"] == 0.5, f"raw_existence_rate={m['raw_existence_rate']} (expected 0.5)"
    assert m["final_existence_rate"] == 1.0, f"final_existence_rate={m['final_existence_rate']} (expected 1.0)"
    assert m["retry_rescued"] == 4
    assert m["verified"] == 8 and m["n_recs"] == 8

    # The critical regression check: quality must reflect the 0.5 existence rate,
    # NOT be driven to ~1.0 by the post-retry verified share. It blends
    # 0.4*confidence + 0.6*existence.
    assert 0.5 <= m["quality"] <= 0.75, f"quality={m['quality']} not pulled down by low existence"
    # And it must be lower than it would be if we (wrongly) used verified_share (1.0):
    naive = round(0.4 * m["avg_confidence"] + 0.6 * 1.0, 3)
    assert m["quality"] < naive, f"quality {m['quality']} should be below masked value {naive}"

    print("EXISTS-ON-SPOTIFY VALIDITY: OK")
    print(json.dumps({
        "raw_existence_rate": m["raw_existence_rate"],
        "round1": f"{m['round1_found']}/{m['raw_candidates']} real on first pass",
        "hallucinated": m["round1_missing"],
        "retry_rescued": m["retry_rescued"],
        "final_existence_rate": m["final_existence_rate"],
        "quality": m["quality"],
        "quality_if_masked": naive,
    }, indent=2))


if __name__ == "__main__":
    main()
