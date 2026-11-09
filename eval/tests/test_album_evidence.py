"""Behavioral tests: exact-album boundary, budgets, rejection, delivery and analytics."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afrec import agents, pipeline
from afrec.agent_album_evidence import AlbumEvidence
from afrec.llm import BaseLLM, GenerationMeta
from recommend import select_delivery


class Model(BaseLLM):
    name = "fixture"

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        data = ([{"artist": "A", "album": "One", "confidence": .99},
                 {"artist": "B", "album": "Two", "confidence": .5}]
                if len(self.prompts) == 1 else {"verdicts": self.verdicts})
        return json.dumps(data), GenerationMeta(name=self.name, latency_s=.01)


def verdict(i, **kw):
    return {"id": i, "verdict": "keep", "confidence": .8, "review_ids": [1],
            "reason": "Specific album texture matches the review",
            "contrast": "no relevant negative evidence", "artist_stereotype_risk": False, **kw}


class AlbumTests(unittest.TestCase):
    reviews = [{"artist": "Baauer", "album": "U", "rating": 5,
                "text": "I like this album's restraint, not his older bass-heavy music."}]

    def run_agent(self, verdicts, **kwargs):
        llm = Model(verdicts)
        with patch.dict(os.environ, {"AFREC_ALBUM_LOOKUP_BUDGET": "0"}), patch.object(AlbumEvidence, "get", return_value=None):
            out = agents.get_agent("album-evidence@v1").run(self.reviews, llm, **kwargs)
        return out, llm

    def test_llama_schema_sent_on_both_stages(self):
        from afrec.llm import LlamaCPPLM
        from unittest.mock import Mock
        outputs = [json.dumps([{"artist": "A", "album": "One"}]),
                   json.dumps({"verdicts": [verdict(1)]})]
        responses = []
        for content in outputs:
            response = Mock()
            response.json.return_value = {"choices": [{"message": {"content": content}}]}
            responses.append(response)
        with patch("afrec.llm.requests.post", side_effect=responses) as post, patch.object(AlbumEvidence, "get", return_value=None):
            out = agents.get_agent("album-evidence@v1.1").run(self.reviews, LlamaCPPLM("test"))
        self.assertEqual(len(out.recs), 1)
        schemas = [c.kwargs["json"]["response_format"]["schema"] for c in post.call_args_list]
        self.assertEqual([s["type"] for s in schemas], ["array", "object"])
        prompt = post.call_args_list[0].kwargs["json"]["messages"][0]["content"]
        self.assertIn("discover NEW albums by OTHER artists", prompt)
        self.assertIn('exclusion list: ["Baauer"]', prompt)
        self.assertEqual(schemas[0]["items"]["properties"]["confidence"]["type"], "number")

    def test_prose_failure_reported_in_evaluation(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import evaluate
        class Prose(Model):
            def generate(self, prompt, **kwargs):
                return "1. Artist: A, Album: One, Confidence: High", GenerationMeta(name=self.name, latency_s=.01)
        with patch.object(evaluate, "make_llm", return_value=Prose([])):
            result = evaluate.evaluate_model("test", [{"name": "test", "reviews": self.reviews}],
                                            "unused", .7, False, agents.get_agent("album-evidence@v1"))
        metrics = result["per_profile"]["test"]["metrics"]
        self.assertEqual(metrics["status"], "error")
        self.assertIn("album_discovery: parse error", metrics["error"])

    def test_rank_survives_production_and_pool_serialization(self):
        out, _ = self.run_agent([verdict(2, confidence=.75), verdict(1, confidence=.98)],
                               verifier=lambda a, b: (True, b, "url"))
        self.assertEqual([r.album for r in out.recs], ["Two", "One"])
        self.assertEqual(select_delivery(out.recs, self.reviews, 1)[0].album, "Two")
        self.assertEqual(out.to_dict()["recs"][0]["delivery_rank"], 1)
        self.assertEqual((out.total_candidates, out.round1_found, out.verified_count), (2, 2, 2))

    def test_rejections_are_never_restored(self):
        out, llm = self.run_agent([verdict(1, artist_stereotype_risk=True), verdict(2, verdict="reject")])
        self.assertTrue(out.ok)
        self.assertEqual(out.recs, [])
        self.assertEqual(out.critic_rejected, 2)
        self.assertEqual(len(llm.prompts), 2)

    def test_invalid_citations_and_missing_verdicts_fail_closed(self):
        out, _ = self.run_agent([verdict(1, review_ids=[99])])
        self.assertEqual(out.recs, [])

    def test_legacy_enrichment_is_not_fetched_or_prompted(self):
        class ForbiddenEnricher:
            available = True
            error = None
            def fact_map(self, pairs):
                raise AssertionError("artist-derived legacy enrichment must not run")
        out, llm = self.run_agent([verdict(1)], enricher=ForbiddenEnricher(),
                                 catalog_facts={"Baauer|U": "TRAP_STEREOTYPE"})
        self.assertEqual(len(out.recs), 1)
        self.assertNotIn("TRAP_STEREOTYPE", " ".join(llm.prompts))
        self.assertIn(self.reviews[0]["text"], llm.prompts[1])

    def test_verifier_outage_is_not_successful_verification(self):
        def fail(a, b):
            raise RuntimeError("quota")
        out, _ = self.run_agent([verdict(1)], verifier=fail)
        self.assertIn("quota", out.verification_error)
        self.assertFalse(out.recs[0].verified)
        self.assertEqual(out.verified_count, 0)

    def test_album_cache_budget_miss_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch("afrec.lastfm.api_key", return_value="key"), patch("afrec.lastfm.album_info", return_value=None) as lookup:
            path = Path(tmp) / "cache.json"
            store = AlbumEvidence(1, path)
            self.assertIsNone(store.get("Baauer", "U"))
            self.assertIsNone(store.get("Baauer", "Other"))
            self.assertEqual(lookup.call_count, 1)
            self.assertIsNone(AlbumEvidence(1, path).get("Baauer", "U"))
            self.assertEqual(lookup.call_count, 1)  # cached miss across runs
            lookup.return_value = {"artist": "Baauer", "name": "Wrong album", "tags": [("trap", 2)]}
            self.assertIsNone(AlbumEvidence(1, path).get("Baauer", "New"))

    def test_error_stops_requests_and_zero_budget_never_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch("afrec.lastfm.api_key", return_value="key"), patch("afrec.lastfm.album_info", side_effect=RuntimeError("429")) as lookup:
            path = Path(tmp) / "cache.json"
            self.assertIsNone(AlbumEvidence(0, path).get("A", "a"))
            self.assertEqual(lookup.call_count, 0)
            store = AlbumEvidence(4, path)
            store.get("A", "a")
            store.get("B", "b")
            self.assertEqual(lookup.call_count, 1)
            self.assertIn("429", store.error)


if __name__ == "__main__":
    unittest.main()
