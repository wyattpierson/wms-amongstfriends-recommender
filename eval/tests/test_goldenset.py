#!/usr/bin/env python3
"""
Offline tests for the per-rec grading API in eval/goldenset.py:
  • append_label(pool_score=...)  — the pool-vs-item linkage (score/duel/grade
                                    flows store the 1–5 pool score with each
                                    per-rec verdict)
  • labeled_for()                 — look up prior verdicts (grade-flow hints
                                    + default-skip for already-judged recs)
  • promotion_candidates()        — artists with enough decisive labels who
                                    aren't in the golden file yet (harvest)

No Ollama, no network, no Spotify. Deterministic: runs in a temp dir, never
touches the real eval/golden/*.jsonl.

    python eval/tests/test_goldenset.py
"""
import json
import sys
import tempfile
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # eval/
sys.path.insert(0, str(HERE.parent.parent))   # repo root

import goldenset as gs                         # noqa: E402

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {detail}")


def _raises(fn) -> bool:
    try:
        fn()
    except (ValueError, AssertionError):
        return True
    return False


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="goldenset_test_"))
    try:
        tmp_golden = tmp / "golden"
        tmp_golden.mkdir()
        gs.GOLDEN_DIR = tmp_golden
        gs.LABELED_LOG = tmp_golden / "labeled.jsonl"

        print("─ promotion_candidates() on an empty corpus")
        check("no labels → no candidates", gs.promotion_candidates() == {})

        print("─ append_label(pool_score=...)")
        gs.append_label(profile="p1", artist="A One", album="Album A", label="good",
                        model="m", source="score", pool_score=4)
        gs.append_label(profile="p1", artist="A Two", label="bad",
                        model="m", source="score", pool_score=4)
        gs.append_label(profile="p1", artist="A Three", label="unknown", source="score")
        rows = gs.load_labeled()
        check("3 rows", len(rows) == 3, str(len(rows)))
        check("pool_score stored when given", rows[0].get("pool_score") == 4, str(rows[0]))
        check("pool_score omitted when None", "pool_score" not in rows[2], str(rows[2]))
        check("source tag stored", rows[0].get("source") == "score", str(rows[0]))
        check("bad pool_score rejected", _raises(lambda: gs.append_label(
            profile="p1", artist="X", label="good", pool_score=9)))
        check("non-int pool_score rejected", _raises(lambda: gs.append_label(
            profile="p1", artist="X", label="good", pool_score="4")))

        print("─ labeled_for()")
        hits = gs.labeled_for("p1", "a one")
        check("case-insensitive artist match",
              len(hits) == 1 and hits[0]["artist"] == "A One", str(hits))
        hits = gs.labeled_for("p1", "A One", "album a")
        check("album match (case-insensitive)",
              len(hits) == 1 and hits[0]["label"] == "good", str(hits))
        check("album mismatch → no hits", gs.labeled_for("p1", "A One", "Other") == [])
        check("different profile → no hits", gs.labeled_for("p2", "A One") == [])

        print("─ promotion_candidates()")
        # A One:   1 good              → under min_good (2) → no candidate
        # B Big:   2 good              → good candidate
        # C Small: 2 bad               → bad candidate
        # D Both:  2 good + 2 bad      → both lists
        # E Golden: 2 good, IN golden  → excluded
        for lab, artist, album in (
            ("good", "B Big", "LP One"), ("good", "B Big", ""),
            ("bad", "C Small", ""), ("bad", "C Small", ""),
            ("good", "D Both", ""), ("good", "D Both", ""),
            ("bad", "D Both", ""), ("bad", "D Both", ""),
            ("good", "E Golden", ""), ("good", "E Golden", ""),
        ):
            gs.append_label(profile="p1", artist=artist, label=lab, album=album)
        (tmp_golden / "p1.json").write_text(json.dumps({
            "profile": "p1", "description": "test",
            "good": ["E Golden"], "bad": [],
            "tags": {}, "constraints": [], "justification_keywords": [],
        }), encoding="utf-8")

        cands = gs.promotion_candidates()
        good_names = [e["artist"] for e in cands.get("p1", {}).get("good", [])]
        bad_names = [e["artist"] for e in cands.get("p1", {}).get("bad", [])]
        check("2-good artist is a good candidate", "B Big" in good_names, str(good_names))
        check("2-bad artist is a bad candidate", "C Small" in bad_names, str(bad_names))
        check("mixed artist appears in both lists",
              "D Both" in good_names and "D Both" in bad_names, str(cands.get("p1")))
        check("golden-listed artist excluded",
              "E Golden" not in good_names, str(good_names))
        check("1-label artist under threshold", "A One" not in good_names, str(good_names))
        b = next(e for e in cands["p1"]["good"] if e["artist"] == "B Big")
        check("label counts + album capture",
              b["good"] == 2 and b["bad"] == 0 and b["albums"] == ["LP One"], str(b))

        # Thresholds are configurable.
        cands2 = gs.promotion_candidates(min_good=3)
        check("min_good=3 drops B Big",
              all(e["artist"] != "B Big" for e in cands2.get("p1", {}).get("good", [])),
              str(cands2.get("p1")))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'═' * 40}\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
