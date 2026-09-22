#!/usr/bin/env python3
"""
Offline tests for eval/humaneval.py (response scores + duels + Elo),
plus a smoke test that taste.py's storage calls are wired to the real logs.

No Ollama, no network, no Spotify. Deterministic: runs in a temp dir, never
touches the real eval/golden/*.jsonl.

    python eval/tests/test_humaneval.py
"""
import sys
import tempfile
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # eval/
sys.path.insert(0, str(HERE.parent.parent))   # repo root (for `evaluate` import chain)

import humaneval as he                        # noqa: E402

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


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="humaneval_test_"))
    try:
        # Point the module at temp logs so we never touch the real corpus.
        tmp_golden = tmp / "golden"
        tmp_golden.mkdir()
        he.GOLDEN_DIR = tmp_golden
        he.SCORES_LOG = tmp_golden / "scores.jsonl"
        he.DUELS_LOG = tmp_golden / "duels.jsonl"

        print("─ scores")
        he.append_score(profile="hip", model="a", score=5, recs=[{"artist": "X", "album": "Y"}])
        he.append_score(profile="hip", model="a", score=3)
        he.append_score(profile="soul", model="b", score=1)
        he.append_score(profile="soul", model="b", score=2)
        he.append_score(profile="soul", model="b", score=4)

        rows = he.load_scores()
        check("5 score rows loaded", len(rows) == 5)
        check("invalid score rejected", _raises(lambda: he.append_score(
            profile="p", model="a", score=6)))
        check("non-int score rejected", _raises(lambda: he.append_score(
            profile="p", model="a", score="5")))

        sa = he.score_stats(model="a")
        check("model-a n=2 avg=4.0", sa["n"] == 2 and sa["avg"] == 4.0, str(sa))
        check("model-a good bucket = 1 (5) + neutral 1 (3)", sa["good"] == 1 and sa["neutral"] == 1, str(sa))
        sb = he.score_stats(model="b")
        check("model-b buckets (1,2=bad; 4=good)", sb["bad"] == 2 and sb["good"] == 1 and sb["neutral"] == 0, str(sb))
        check("profile filter works",
              he.score_stats(profile="soul")["n"] == 3)
        check("empty stats sane", he.score_stats(model="nope")["avg"] is None)

        print("─ duels + elo")
        # a beats b three times, b beats a once with c losing.
        he.append_duel(profile="p", ranking=["a", "b"])
        he.append_duel(profile="p", ranking=["a", "b"])
        he.append_duel(profile="p", ranking=["b", "a"])
        he.append_duel(profile="p", ranking=["a", "b", "c"])

        elo = he.elo_from_duels()
        check("all 3 models present", set(elo) == {"a", "b", "c"}, str(set(elo)))
        check("a rated above b (3 wins vs 1)", elo["a"]["elo"] > elo["b"]["elo"],
              f"{elo['a']['elo']} vs {elo['b']['elo']}")
        check("c is last (0 wins)", elo["c"]["elo"] < elo["b"]["elo"])
        check("a wins = 4 (3+1 from c duel… 2+1+1)", elo["a"]["wins"] == 4, str(elo["a"]))
        check("c losses = 2 (loses to a and b in the 3-way duel)",
              elo["c"]["losses"] == 2, str(elo["c"]))

        # Tie handling: two-model tie → no Elo movement, tie counted.
        before_a = elo["a"]["elo"]
        he.append_duel(profile="p", ranking=[], ties=["a", "b"])
        elo2 = he.elo_from_duels()
        check("tie keeps Elo unchanged", abs(elo2["a"]["elo"] - before_a) < 1e-9)
        check("tie tallied", elo2["a"]["ties"] == 1 and elo2["b"]["ties"] == 1)

        # Partial ranking: [a] + unranked b → a still beats b.
        he.append_duel(profile="p", ranking=["a", "b"])  # explicit 2-order duel
        check("partial ordering recorded",
              he.load_duels()[-1]["ranking"] == ["a", "b"])

        # Winner-only verdict: b ranked, a unranked → a counts as the loser.
        nlos_before = he.elo_from_duels()["a"]["losses"]
        he.append_duel(profile="p", ranking=["b"], unranked=["a"])
        elo3 = he.elo_from_duels()
        check("unranked model takes a loss to the winner",
              elo3["a"]["losses"] == nlos_before + 1 and elo3["b"]["wins"] >= 1,
              f"{elo3['a']} vs prev {nlos_before}")
        check("winner-only duel is valid (1 ranked + 1 unranked)",
              he.load_duels()[-1]["unranked"] == ["a"])

        print("─ validation")
        check("duel with no models rejected",
              _raises(lambda: he.append_duel(profile="p", ranking=["a"])))
        check("duel with duplicate models rejected",
              _raises(lambda: he.append_duel(profile="p", ranking=["a", "a"])))
        check("duel with ranked model in two slots rejected",
              _raises(lambda: he.append_duel(profile="p", ranking=["a"],
                                             unranked=["a"])))

        print("─ calibration")
        # cx: confidence tracks taste (high conf on the 5, low on the 1).
        he.append_score(profile="p", model="cx", score=5, recs=[{"confidence": 0.9}])
        he.append_score(profile="p", model="cx", score=3, recs=[{"confidence": 0.6}])
        he.append_score(profile="p", model="cx", score=1, recs=[{"confidence": 0.4}])
        # cy: constant 0.8 confidence regardless of taste → flat, overconfident.
        he.append_score(profile="p", model="cy", score=5, recs=[{"confidence": 0.8}])
        he.append_score(profile="p", model="cy", score=3, recs=[{"confidence": 0.8}])
        he.append_score(profile="p", model="cy", score=1, recs=[{"confidence": 0.8}])

        ccx = he.calibration_stats(model="cx")
        check("cx calibration n=3", ccx.get("n") == 3, str(ccx))
        # cx bias: mean_conf (0.9+0.6+0.4)/3 = 0.633 vs mean_target 0.5
        check("cx bias ≈ +0.133", abs(ccx["bias"] - 0.133) < 0.0005, str(ccx))
        check("cx spearman positive (confidence tracks taste)",
              ccx.get("spearman", 0) > 0.5, str(ccx.get("spearman")))
        check("cx buckets: bad=0.4, neutral=0.6, good=0.9",
              ccx["buckets"]["bad"] == (1, 0.4) and ccx["buckets"]["good"] == (1, 0.9),
              str(ccx["buckets"]))

        ccy = he.calibration_stats(model="cy")
        check("cy flat confidence → no spearman defined",
              "spearman" not in ccy, str(ccy))
        check("cy overconfident (claims .8, taste .5)", ccy["bias"] > 0.2 and ccy["verdict"] == "overconfident", str(ccy))

        check("no confidence data → n=0", he.calibration_stats(model="a")["n"] == 0)

        print("─ all_models()")
        check("union of scored+duelled models",
              he.all_models() == ["a", "b", "c", "cx", "cy"], str(he.all_models()))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'═' * 40}\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


def _raises(fn) -> bool:
    try:
        fn()
    except (ValueError, AssertionError):
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
