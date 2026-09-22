#!/usr/bin/env python3
"""
Evaluation harness for the AmongstFriends recommendation engine.

Runs the *same* pipeline that recommend.py uses (prompts → LLM → Spotify
verify → retry) against saved user-review profiles, across many models and/or
prompt revisions. No Firebase involved.

Two jobs it's good at:
  1. Compare models     — hold the engine fixed, sweep model names.
  2. Compare the engine — hold the model fixed, edit afrec/prompts.py (or
                          pipeline.py) and re-run against the same fixtures.

Outputs (in eval/output/, gitignored):
  - per_<model>.json        raw RunOutcome per profile
  - leaderboard_<ts>.md     ranked table across models
  - runs_<ts>.tsv           one row per (model, profile) for spreadsheets
  - (--export-csv PATH)     one row per individual rec for humans to review

Typical use (run from the repo root, or anywhere):
  # Compare three models on all profiles:
  python eval/evaluate.py --models llama3,qwen2.5,mistral

  # One model, one profile, while you iterate on afrec/prompts.py:
  python eval/evaluate.py --models llama3 --profile jazzy_hiphop

  # Use a non-default llama-server host (e.g. a remote box with more RAM):
  python eval/evaluate.py --base-url http://192.168.1.50:8092
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Paths & imports ───────────────────────────────────────────────────────
# This file is eval/evaluate.py. The ENGINE (afrec/) lives at the repo root;
# the harness siblings (goldenset.py) live next to this file. We expose both
# on sys.path so `afrec` and `goldenset` import cleanly from anywhere.
EVAL_DIR     = Path(__file__).resolve().parent     # …/amongstfriends-recommender/eval
REPO_ROOT    = EVAL_DIR.parent                     # …/amongstfriends-recommender
FIXTURES_DIR = EVAL_DIR / "fixtures"
OUTPUT_DIR   = EVAL_DIR / "output"

sys.path.insert(0, str(REPO_ROOT))                 # for `afrec.*`
sys.path.insert(0, str(EVAL_DIR))                  # for the sibling `goldenset` module

# Load .env (SPOTIFY_CLIENT_ID/SECRET, LLM_*, …) from the repo root. dotenv is
# NOT a hard dependency — if it's missing we parse the file ourselves.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    def _tiny_load_dotenv(path: Path) -> None:
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        except FileNotFoundError:
            pass
    _tiny_load_dotenv(REPO_ROOT / ".env")

import goldenset as golden_mod                     # eval/goldenset.py (same directory)
from afrec import pipeline, spotify               # the engine (repo root)
from afrec.llm import make_llm, LLMError, LLMConnectionError, LLMTimeoutError, LLMHTTPError
from afrec.reviews import parse_reviews


# ── Fixture loading ───────────────────────────────────────────────────────────

def load_profile(path: Path) -> dict:
    """Load one profile JSON and return {name, description, reviews} (normalized)."""
    data = json.loads(path.read_text())
    raw = data.get("reviews", []) if isinstance(data, dict) else data
    reviews, _skipped = parse_reviews(raw)
    name = (data.get("profile") or path.stem) if isinstance(data, dict) else path.stem
    description = (data.get("description") or "") if isinstance(data, dict) else ""
    return {"name": name, "description": description, "reviews": reviews, "source": str(path)}


def discover_profiles() -> list[dict]:
    profiles = []
    for p in sorted(FIXTURES_DIR.glob("*.json")):
        try:
            profiles.append(load_profile(p))
        except Exception as e:
            print(f"⚠️  Skipped fixture {p.name}: {e}", file=sys.stderr)
    return profiles


# ── Scoring ───────────────────────────────────────────────────────────────────
# Given a RunOutcome, score the *quality* independent of the model. Spotify
# verification already tells us which albums are real; here we add signal about
# whether the picks are likely to be good, and whether the engine followed
# instructions (new artists only, one per artist, etc.).


def score_outcome(profile: dict, outcome: pipeline.RunOutcome, spotify_ons: bool) -> dict:
    """
    Produce a comparable metrics dict for one (profile × model) run.
    """
    m: dict = {
        "profile": profile["name"],
        "ok": bool(outcome.ok),
        "n_recs": len(outcome.recs),
        "n_new_artists": len(outcome.suggested_artists),
        "repeated_artist_violations": len(outcome.repeated_artist_violations),
        "total_latency_s": round(outcome.total_latency_s, 2),
        "spotify_on": spotify_ons,
        "verified": 0,
        "unverified": 0,
    }

    recs = outcome.recs if outcome.ok else []

    # Golden set (human-labeled good/bad for this profile). This is the
    # model-independent, comparable taste-fit signal: how many of the engine's
    # picks match a GOOD human pick vs a BAD one. See eval/golden/README.md.
    gset = golden_mod.load_golden_for(profile["name"])
    m["golden_has"] = gset is not None
    if gset is not None:
        gs = golden_mod.score_recs_against_golden(recs, gset)
        m["golden_good"] = gs["good_hits"]
        m["golden_bad"] = gs["bad_hits"]
        m["golden_score"] = gs["score"]
        m["golden_pair_points"] = gs["pair_points"]
        m["golden_pair_total"] = gs["pair_total"]
        m["golden_pair_score"] = gs["pair_score"]
        m["golden_novel_hits"] = gs["novel_hits"]
        m["golden_novel_total"] = gs["novel_total"]
    else:
        m["golden_good"] = None
        m["golden_bad"] = None
        m["golden_score"] = None   # no golden set for this profile
        m["golden_pair_points"] = None
        m["golden_pair_total"] = None
        m["golden_pair_score"] = None
        m["golden_novel_hits"] = None
        m["golden_novel_total"] = None

    if not outcome.ok:
        m["quality"] = 0.0
        m["notes"] = "engine returned no usable recs"
        return m

    m["verified"] = sum(1 for r in recs if r.verified)
    m["unverified"] = sum(1 for r in recs if not r.verified)

    # ── Existence on Spotify (validity) ────────────────────────────────
    # The pipeline RETRIES non-existent albums, so the final recs are all real
    # by construction (verified_share would read ~100% and hide the model's
    # hallucinations). The honest signal is the ROUND-1 (pre-retry) pass rate:
    #   raw_existence_rate = round1_found / total_candidates
    # i.e. what fraction of the albums the model *produced* actually exist in
    # Spotify. Retry then rescues the misses; we report both.
    if spotify_ons:
        total_c = outcome.total_candidates or 0
        m["raw_candidates"] = total_c
        m["round1_found"] = outcome.round1_found
        m["round1_missing"] = outcome.round1_missing
        m["retry_rescued"] = outcome.retry_found
        m["retry_dropped"] = outcome.retry_missing
        m["raw_existence_rate"] = round(outcome.round1_found / total_c, 3) if total_c else None
        m["final_existence_rate"] = round(m["verified"] / m["n_recs"], 3) if m["n_recs"] else None
    else:
        m["raw_candidates"] = 0
        m["round1_found"] = 0
        m["round1_missing"] = 0
        m["retry_rescued"] = 0
        m["retry_dropped"] = 0
        m["raw_existence_rate"] = None      # not measured (no verifier)
        m["final_existence_rate"] = None

    # Coverage: how many of the call-1 artists got a real album attached.
    artist_set = {a.lower() for a in outcome.suggested_artists}
    attached = {r.artist.lower() for r in recs}
    m["artist_coverage"] = round(len(artist_set & attached) / len(artist_set), 3) if artist_set else 0.0

    # One-per-artist: flag duplicates.
    seen = set()
    dupes = 0
    for r in recs:
        k = r.artist.lower()
        if k in seen:
            dupes += 1
        seen.add(k)
    m["duplicate_artists"] = dupes

    # Quality heuristic: confidence mean + (if Spotify on) the verified share.
    conf = [r.confidence for r in recs] or [0.0]
    m["avg_confidence"] = round(sum(conf) / len(conf), 3)

    if spotify_ons and m["n_recs"]:
        verified_share = m["verified"] / m["n_recs"]
        m["verified_share"] = round(verified_share, 3)
        # Validity = the round-1 existence rate (pre-retry), NOT the post-retry
        # verified_share — the latter is ~1.0 by construction and hides which
        # albums the model hallucinated vs. which were rescued by retry.
        existence = m.get("raw_existence_rate")
        if existence is None and m["n_recs"]:
            existence = verified_share
        m["quality"] = round(0.4 * m["avg_confidence"] + 0.6 * existence, 3)
    else:
        # No Spotify ground-truth — use confidence + instruction-following.
        instruction = 1.0
        if m["duplicate_artists"]:
            instruction -= 0.25
        if m["repeated_artist_violations"]:
            instruction -= 0.25
        if m["artist_coverage"] < 1.0:
            instruction -= (1.0 - m["artist_coverage"]) * 0.5
        m["instruction_score"] = round(max(0.0, min(1.0, instruction)), 3)
        m["quality"] = round(0.3 * m["avg_confidence"] + 0.7 * m["instruction_score"], 3)

    return m


# ── Model sweep ───────────────────────────────────────────────────────────────

def evaluate_model(
    model: str,
    profiles: list[dict],
    base_url: str,
    temperature: float,
    use_spotify: bool,
) -> dict:
    llm = make_llm(model, base_url=base_url)
    spotify_token: str | None = None
    verifier = None
    if use_spotify:
        spotify_token = spotify.get_token()
        if spotify_token:
            verifier = spotify.verifier_for(spotify_token)

    per_profile = {}
    for prof in profiles:
        t0 = time.time()
        try:
            outcome = pipeline.run(prof["reviews"], llm, verifier=verifier)
            metrics = score_outcome(prof, outcome, bool(verifier))
            status = "ok"
            err = None
        except LLMConnectionError as e:
            outcome = pipeline.RunOutcome(ok=False)
            metrics = score_outcome(prof, outcome, False)
            status, err = "error", f"connection: {e}"
        except LLMTimeoutError as e:
            outcome = pipeline.RunOutcome(ok=False)
            metrics = score_outcome(prof, outcome, False)
            status, err = "error", f"timeout: {e}"
        except LLMHTTPError as e:
            outcome = pipeline.RunOutcome(ok=False)
            metrics = score_outcome(prof, outcome, False)
            status, err = "error", f"http {e.status_code}: {model} not available?"
        except LLMError as e:
            outcome = pipeline.RunOutcome(ok=False)
            metrics = score_outcome(prof, outcome, False)
            status, err = "error", str(e)
        except Exception as e:  # unexpected
            outcome = pipeline.RunOutcome(ok=False)
            metrics = score_outcome(prof, outcome, bool(verifier))
            status, err = "error", f"unexpected: {e!r}"

        metrics["status"] = status
        if err:
            metrics["error"] = err
        metrics["wallclock_s"] = round(time.time() - t0, 2)

        per_profile[prof["name"]] = {
            "metrics": metrics,
            "outcome": outcome.to_dict() if outcome else {},
        }

    good = [p["metrics"]["quality"] for p in per_profile.values() if p["metrics"].get("ok")]
    return {
        "model": model,
        "base_url": base_url,
        "spotify": bool(verifier),
        "avg_quality": round(sum(good) / len(good), 3) if good else 0.0,
        "n_ok": sum(1 for p in per_profile.values() if p["metrics"].get("ok")),
        "n_total": len(per_profile),
        "total_wallclock_s": round(sum(p["metrics"].get("wallclock_s", 0) for p in per_profile.values()), 2),
        "per_profile": per_profile,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def avg_golden(model_result: dict) -> float | None:
    """Average golden_score across the model's profiles (ignoring profiles with no golden set)."""
    vals = [b["metrics"].get("golden_score") for b in model_result["per_profile"].values()
            if b["metrics"].get("golden_score") is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def avg_pair(model_result: dict) -> float | None:
    """Average contrast-pair score across the model's profiles (each pair -1..+1)."""
    vals = [b["metrics"].get("golden_pair_score") for b in model_result["per_profile"].values()
            if b["metrics"].get("golden_pair_score") is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def novel_totals(model_result: dict) -> tuple[int, int]:
    """(novel_hits, novel_total) summed across the model's profiles."""
    hits = sum(b["metrics"].get("golden_novel_hits") or 0 for b in model_result["per_profile"].values())
    tot = sum(b["metrics"].get("golden_novel_total") or 0 for b in model_result["per_profile"].values())
    return hits, tot


def avg_real(model_result: dict) -> float | None:
    """Average ROUND-1 existence-on-Spotify rate (pre-retry), the hallucination signal."""
    vals = [b["metrics"].get("raw_existence_rate") for b in model_result["per_profile"].values()
            if b["metrics"].get("raw_existence_rate") is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def render_leaderboard(results: list[dict], args) -> str:
    rows = sorted(results, key=lambda r: (r["avg_quality"], r["n_ok"]), reverse=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"# AmongstFriends engine leaderboard — {now}\n")
    lines.append(f"Base URL: `{args.base_url}`   Spotify: {'on' if any(r['spotify'] for r in rows) else 'off'}   "
                 f"temperature: {args.temperature}   profiles: {len(rows[0]['per_profile']) if rows else 0}\n")
    lines.append("| rank | model | avg quality | avg real (r1, Spotify) | avg golden (humans) | pairs | novel | ok/total | avg wallclock (s) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, 1):
        ggs = avg_golden(r)
        gtxt = f"{ggs:+.3f}" if ggs is not None else "n/a"
        art = avg_real(r)
        atxt = f"{art:.0%}" if art is not None else "n/a"   # not measured when Spotify off
        aps = avg_pair(r)
        ptxt = f"{aps:+.2f}" if aps is not None else "n/a"
        nh, nt = novel_totals(r)
        ntxt = f"{nh}/{nt}" if nt else "n/a"
        lines.append(f"| {i} | {r['model']} | {r['avg_quality']:.3f} | {atxt} | {gtxt} | {ptxt} | {ntxt} | {r['n_ok']}/{r['n_total']} | {r['total_wallclock_s']:g} |")

    # Per-profile detail for each model
    for r in rows:
        lines.append(f"\n## {r['model']}\n")
        lines.append("| profile | ok | recs | new artists | real r1 | rescued | golden g/b | golden | pairs | novel | avg conf | quality | status |")
        lines.append("|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|")
        for name, block in r["per_profile"].items():
            m = block["metrics"]
            if m.get("golden_score") is None:
                gcol, ggb = "n/a", ""
                pcol = f"{m['golden_pair_points']:+d}/{m['golden_pair_total']}" if m.get("golden_pair_total") else "n/a"
            else:
                gcol = f"{m['golden_score']:+.3f}"
                ggb = f"{m['golden_good']}g / {m['golden_bad']}b"
                pcol = f"{m['golden_pair_points']:+d}/{m['golden_pair_total']}" if m.get("golden_pair_total") else "—"
            ncol = f"{m['golden_novel_hits']}/{m['golden_novel_total']}" if m.get("golden_novel_total") else "—"
            if m.get("raw_existence_rate") is None:
                r1col, resc = "—", "—"
            else:
                r1col = f"{m['round1_found'] or 0}/{m['raw_candidates'] or 0} ({m['raw_existence_rate']:.0%})"
                resc = str(m.get("retry_rescued", 0))
            lines.append(
                f"| {name} | {'✅' if m.get('ok') else '❌'} | {m['n_recs']} | {m['n_new_artists']} "
                f"| {r1col} | {resc} | {ggb} | {gcol} | {pcol} | {ncol} | {m.get('avg_confidence','')} | {m['quality']:.3f} | {m.get('status')} |"
            )
    return "\n".join(lines)


def write_tsv(results: list[dict], path: Path) -> None:
    import csv
    cols = ["model", "profile", "status", "ok", "n_recs", "n_new_artists",
            "raw_candidates", "round1_found", "round1_missing", "retry_rescued",
            "raw_existence_rate", "final_existence_rate",
            "golden_good", "golden_bad", "golden_score",
            "golden_pair_points", "golden_pair_total", "golden_pair_score",
            "golden_novel_hits", "golden_novel_total",
            "avg_confidence", "quality", "wallclock_s"]
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for r in results:
            for name, block in r["per_profile"].items():
                m = block["metrics"]
                w.writerow([
                    r["model"], name, m.get("status"), int(bool(m.get("ok"))),
                    m["n_recs"], m["n_new_artists"],
                    m.get("raw_candidates", ""), m.get("round1_found", ""),
                    m.get("round1_missing", ""), m.get("retry_rescued", ""),
                    (f"{m['raw_existence_rate']:.3f}" if m.get("raw_existence_rate") is not None else ""),
                    (f"{m['final_existence_rate']:.3f}" if m.get("final_existence_rate") is not None else ""),
                    m.get("golden_good", ""), m.get("golden_bad", ""),
                    (f"{m['golden_score']:.3f}" if m.get("golden_score") is not None else ""),
                    m.get("golden_pair_points", ""), m.get("golden_pair_total", ""),
                    (f"{m['golden_pair_score']:.3f}" if m.get("golden_pair_score") is not None else ""),
                    m.get("golden_novel_hits", ""), m.get("golden_novel_total", ""),
                    m.get("avg_confidence", ""),
                    f"{m['quality']:.3f}", m.get("wallclock_s", ""),
                ])


# ── CLI ───────────────────────────────────────────────────────────────────────

def _read_label() -> str:
    while True:
        try:
            ans = input("   good / bad / unsure   (g / b / u): ").strip().lower()
        except EOFError:
            return "unknown"
        if ans in ("g", "good", "yes", "y"):
            return "good"
        if ans in ("b", "bad", "no", "n"):
            return "bad"
        if ans in ("u", "unsure", "unknown", "maybe", ""):
            return "unknown"
        print("   (type g, b, or u)")


def run_interactive(profs: list[dict], model: str, base_url: str,
                    temperature: float, use_spotify: bool) -> int:
    """
    Generate recs for one profile, present them one by one, and record
    human good/bad verdicts into the cumulative labeled corpus (eval/golden/labeled.jsonl).
    This is how a golden set grows from real human judgment over time.
    """
    prof = profs[0]
    verifier = None
    if use_spotify:
        token = spotify.get_token()
        if token:
            verifier = spotify.verifier_for(token)
        else:
            print("   ⚠️  no Spotify creds found — running interactive WITHOUT verification",
                  file=sys.stderr)

    print(f"\n🎧 Interactive eval — profile: {prof['name']}   model: {model}")
    print(f"   Spotify: {'on' if verifier else 'off'}\n")

    try:
        outcome = pipeline.run(
            prof["reviews"], make_llm(model, base_url),
            verifier=verifier, temperature=temperature,
        )
    except Exception as e:
        print(f"❌ Generation failed: {e!r}", file=sys.stderr)
        return 1

    recs = outcome.recs
    if not recs:
        print("❌ The engine returned no recs to label (check the model / logs).")
        return 1

    print(f"   {len(recs)} recs. Label each (this is what we learn from):\n")
    labels: list[tuple[str, str]] = []
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r.artist} — {r.album}")
        if (r.reason or "").strip():
            print(f"   {(r.reason).strip()}")
        lab = _read_label()
        try:
            note = input("   note (enter to skip): ").strip()
        except EOFError:
            note = ""
        golden_mod.append_label(
            profile=prof["name"], artist=r.artist, album=r.album,
            label=lab, model=model, note=note, reason=(r.reason or "").strip(),
        )
        labels.append((r.artist, lab))
        print(f"   → {lab}\n")

    good = sum(1 for _, l in labels if l == "good")
    bad = sum(1 for _, l in labels if l == "bad")
    unsure = len(labels) - good - bad
    print("─" * 50)
    print(f"Saved {len(labels)} verdicts → {golden_mod.LABELED_LOG}")
    print(f"   this session: good={good}  bad={bad}  unsure={unsure}")
    stats = golden_mod.labeled_stats(model=model)
    rate = f"{stats['good_rate']:.0%}" if stats["good_rate"] is not None else "n/a"
    print(f"   corpus for '{model}': good_rate={rate}  ({stats['total']} labels across sessions)")
    print("\n   💡 Static leaderboard scores read eval/golden/<profile>.json.")
    print("      This session grew the per-model human-preference corpus instead.")
    return 0


def export_recs_csv(results: list[dict], path: Path) -> int:
    """
    One row per (model × profile × recommendation) for humans to eyeball in a
    spreadsheet. Each row carries the rec, whether it actually exists on
    Spotify, its link, and the static good/bad golden verdict — everything you
    need to judge each individual pick. Returns the number of rows written.
    """
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "profile", "artist", "album", "round", "confidence",
            "verified", "spotify_url", "canonical_album", "golden", "reason"]
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            model = r["model"]
            for pname, block in r["per_profile"].items():
                recs = (block.get("outcome") or {}).get("recs") or []

                # Per-rec static golden verdict (good/bad), matched by normalized pair.
                verdicts: dict = {}
                golden = golden_mod.load_golden_for(pname)
                if golden and recs:
                    details = golden_mod.score_recs_against_golden(recs, golden).get("details", [])
                    for d in details:
                        key = (golden_mod._normalize(d["artist"]), golden_mod._normalize(d["album"]))
                        verdicts[key] = d["verdict"]

                for rec in recs:
                    a = golden_mod._normalize(rec.get("artist", ""))
                    al = golden_mod._normalize(rec.get("album", ""))
                    g = verdicts.get((a, al), "")
                    w.writerow([
                        model, pname,
                        rec.get("artist", ""), rec.get("album", ""),
                        rec.get("round", 1), rec.get("confidence", ""),
                        "yes" if rec.get("verified") else "no",
                        rec.get("link", ""), rec.get("canonical_album") or "",
                        g, rec.get("reason", ""),
                    ])
                    rows += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the recommendation engine across models/profiles.")
    parser.add_argument("--models", default=os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3",
                        help="Comma-separated model names (default: $LLM_MODEL or llama3)")
    parser.add_argument("--profile", help="Filter to one fixture by name or stem (repeatable).")
    parser.add_argument("--base-url", default=os.getenv("LLM_URL") or os.getenv("OLLAMA_URL") or "http://localhost:8092",
                        help="LLM server base URL (default: $LLM_URL or localhost:8092)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-spotify", action="store_true",
                        help="Skip Spotify verification (faster, weaker ground-truth).")
    parser.add_argument("--interactive", action="store_true",
                        help="Generate one profile's recs, ask good/bad per rec, and save to the labeled corpus (needs --profile).")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--export-csv", metavar="PATH",
                        help="Also write one row per (model × profile × rec) to this CSV "
                             "(rec + exists-on-Spotify + link + good/bad verdict) for spreadsheet review.")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_profiles = discover_profiles()
    if args.profile:
        profs = [p for p in all_profiles if p["name"] == args.profile or p["name"].endswith(args.profile)]
        if not profs:
            print(f"❌ No profile matched '{args.profile}'. Available: "
                  + ", ".join(p["name"] for p in all_profiles), file=sys.stderr)
            return 2
    else:
        profs = all_profiles

    if not profs:
        print(f"❌ No fixtures found in {FIXTURES_DIR}", file=sys.stderr)
        return 2

    if args.interactive:
        if args.profile is None:
            print("❌ --interactive needs a --profile. Available: "
                  + ", ".join(p["name"] for p in all_profiles), file=sys.stderr)
            return 2
        if len(profs) > 1:
            print(f"   (multiple matched '{args.profile}'; labeling first: {profs[0]['name']})")
        return run_interactive(profs, models[0], args.base_url, args.temperature,
                               use_spotify=not args.no_spotify)

    print("\n🧪 Evaluation harness")
    print(f"   Models:    {', '.join(models)}")
    print(f"   Profiles:  {', '.join(p['name'] for p in profs)}")
    print(f"   Base URL:  {args.base_url}")
    print(f"   Spotify:   {'off' if args.no_spotify else 'on (if creds)'}")
    print("─" * 50)

    results = []
    for model in models:
        print(f"\n▶ Evaluating {model} ...")
        result = evaluate_model(model, profs, args.base_url, args.temperature,
                                use_spotify=not args.no_spotify)
        results.append(result)
        print(f"   {result['n_ok']}/{result['n_total']} profiles OK — avg quality {result['avg_quality']:.3f} "
              f"in {result['total_wallclock_s']}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for r in results:
        (out_dir / f"per_{r['model']}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False))

    leaderboard = render_leaderboard(results, args)
    lb_path = out_dir / f"leaderboard_{ts}.md"
    lb_path.write_text(leaderboard + "\n")
    tsv_path = out_dir / f"runs_{ts}.tsv"
    write_tsv(results, tsv_path)
    extra_files = [tsv_path]

    if args.export_csv:
        csv_path = Path(args.export_csv)
        n_rows = export_recs_csv(results, csv_path)
        print(f"   {n_rows} recs exported → {csv_path}")
        extra_files.append(csv_path)

    print("\n" + "─" * 50)
    print(leaderboard)
    all_files = [lb_path] + extra_files
    for fp in all_files:
        print(f"\n📄 Wrote:{' '*2}{fp}")
    print(f"\n   (per-model detail under {out_dir}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
