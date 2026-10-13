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

# Trust cutoff: if ≤25% of a model's round-1 recs are real (i.e. ≥75% are albums
# that don't exist on Spotify), the model/agent pairing is UNTRUSTED — its output
# can't be relied on, so it's not worth spending human evaluation time on it.
UNTRUSTED_REAL_R1_MAX = 0.25


def thinking_override(args) -> bool | None:
    """CLI override for per-model thinking (afrec/model_options.json).

    None = use the per-model file; True/False = force on/off for this run.
    """
    if args.thinking and args.no_thinking:
        raise SystemExit("❌ --thinking and --no-thinking are mutually exclusive.")
    if args.thinking:
        return True
    if args.no_thinking:
        return False
    return None

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
from afrec import pipeline, spotify, model_options as mopts   # the engine (repo root) + per-model LLM options
from afrec import agents as afrec_agents          # the versioned playbooks the engine runs as
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

    # Structure (response-level: genre/era spread vs the fixture's own
    # composition) + justification keyword hits. Kept OUT of golden/quality —
    # a separate yardstick, see eval/golden/README.md.
    m["structure_points"] = None
    m["structure_total"] = None
    m["structure_score"] = None
    m["lane_share"] = None
    m["lane_balance"] = None
    m["tag_coverage"] = None
    m["just_hits"] = None
    m["just_total"] = None
    m["just_fixture_hits"] = None
    if gset is not None and (gset.get("constraints") or gset.get("justification_keywords")):
        fixture_names = [rv.get("artist", "") for rv in profile.get("reviews", [])]
        fixture_names += [rv.get("album", "") for rv in profile.get("reviews", [])]
        st = golden_mod.score_response_structure(recs, gset, fixture_names)
        m["structure_points"] = st["points"] if st["total"] else None
        m["structure_total"] = st["total"] if st["total"] else None
        m["structure_score"] = st["score"]
        m["lane_share"] = st["lane_share"]
        m["lane_balance"] = st["lane_balance"]
        m["tag_coverage"] = st["tag_coverage"]
        m["just_hits"] = st["just_hits"]
        m["just_total"] = st["just_total"]
        m["just_fixture_hits"] = st["just_fixture_hits"]

    if not outcome.ok:
        m["quality"] = 0.0
        m["notes"] = "engine returned no usable recs"
        return m

    m["verified"] = sum(1 for r in recs if r.verified)
    m["unverified"] = sum(1 for r in recs if not r.verified)
    # The catalog verifier died mid-run (e.g. Spotify 429): the recs are
    # unverified passthroughs, NOT measured misses. Score them like a
    # no-verifier run and flag the row, instead of recording a fake 0.
    m["verification_error"] = outcome.verification_error
    # Enrichment latched mid-run (Musicae/Spotify hop failure): facts went
    # cache-only. The run is LESS informed, never failed — flag the row the
    # same way verification errors are flagged, without touching the score.
    m["enrichment_error"] = outcome.enrichment_error

    # ── Existence on Spotify (validity) ────────────────────────────────
    # The pipeline RETRIES non-existent albums, so the final recs are all real
    # by construction (verified_share would read ~100% and hide the model's
    # hallucinations). The honest signal is the ROUND-1 (pre-retry) pass rate:
    #   raw_existence_rate = round1_found / total_candidates
    # i.e. what fraction of the albums the model *produced* actually exist in
    # Spotify. Retry then rescues the misses; we report both.
    # NOT measured when the verifier died mid-run (verification_error) — the
    # "found" flags on that run are passthroughs, not Spotify confirmations.
    if spotify_ons and not outcome.verification_error:
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
        m["raw_existence_rate"] = None      # not measured (no verifier, or verifier died)
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

    if spotify_ons and not outcome.verification_error and m["n_recs"]:
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
    agent: "afrec_agents.Agent",
    thinking: bool | None = None,
    enricher=None,
) -> dict:
    # Per-model extra options (afrec/model_options.json), with the CLI
    # thinking override applied. LlamaCPPLM also merges the table itself;
    # passing it here explicitly lets the --thinking/--no-thinking override
    # win. See afrec/model_options.py for the rationale.
    extra_options = mopts.merged(model, thinking)
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
            outcome = agent.run(prof["reviews"], llm, verifier=verifier,
                                extra_options=extra_options, enricher=enricher)
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
        metrics["agent"] = agent.tag    # which playbook produced this, whatever happened

        per_profile[prof["name"]] = {
            "metrics": metrics,
            "outcome": outcome.to_dict() if outcome else {},
        }

    good = [p["metrics"]["quality"] for p in per_profile.values() if p["metrics"].get("ok")]
    return {
        "model": model,
        "agent": agent.tag,
        "base_url": base_url,
        "spotify": bool(verifier),
        "avg_quality": round(sum(good) / len(good), 3) if good else 0.0,
        "n_ok": sum(1 for p in per_profile.values() if p["metrics"].get("ok")),
        "n_total": len(per_profile),
        "total_wallclock_s": round(sum(p["metrics"].get("wallclock_s", 0) for p in per_profile.values()), 2),
        # Cohort label for the runs history: facts = an enricher was offered
        # (agents without a catalog_facts param simply don't use it),
        # years = bare Spotify year lookups only (v1.1 style), None = no
        # catalog anchoring at all (no verifier, no enricher).
        "enriched": "facts" if enricher is not None else ("years" if verifier is not None else None),
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


def avg_structure(model_result: dict) -> float | None:
    """Average weighted structure (constraint) score across the model's profiles."""
    vals = [b["metrics"].get("structure_score") for b in model_result["per_profile"].values()
            if b["metrics"].get("structure_score") is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _role_mix(block: dict) -> str:
    """core/adjacent/stretch counts of a profile's recs — the pool-map plan's
    fingerprint. Shows whether the coherence rule constrained a coherent pool
    (stretch→0) or let a diverse pool explore (stretch>0). '—' for agents whose
    recs carry no roles (artist-then-album, album-first, pool-map@v1)."""
    recs = (block.get("outcome") or {}).get("recs") or []
    c = sum(1 for r in recs if r.get("role") == "core")
    a = sum(1 for r in recs if r.get("role") == "adjacent")
    s = sum(1 for r in recs if r.get("role") == "stretch")
    return f"{c}/{a}/{s}" if (c + a + s) else "—"


def render_leaderboard(results: list[dict], args) -> str:
    rows = sorted(results, key=lambda r: (r["avg_quality"], r["n_ok"]), reverse=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"# AmongstFriends engine leaderboard — {now}\n")
    agent_txt = rows[0].get("agent", "?") if rows else "?"
    sp = 'on' if any(r['spotify'] for r in rows) else 'off'
    if any((b.get('metrics') or {}).get('verification_error')
           for r in rows for b in r['per_profile'].values()):
        sp += ' ⚠️ (catalog API errors — some runs unverified, real r1 not measured)'
    enriched = 'off' if all(r.get('enriched') is None for r in rows) else (rows[0].get('enriched') or '?')
    if any((b.get('metrics') or {}).get('enrichment_error')
           for r in rows for b in r['per_profile'].values()):
        enriched += ' ⚠️ (enrichment latched — some runs cache-only facts)'
    lines.append(f"Agent: `{agent_txt}`   "
                 f"Base URL: `{args.base_url}`   Spotify: {sp}   Enriched: {enriched}   "
                 f"temperature: {args.temperature}   profiles: {len(rows[0]['per_profile']) if rows else 0}\n")
    lines.append("| rank | model | agent | avg quality | avg real (r1, Spotify) | avg golden (humans) | pairs | novel | structure | ok/total | avg wallclock (s) |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    untrusted: list[str] = []
    for i, r in enumerate(rows, 1):
        ggs = avg_golden(r)
        gtxt = f"{ggs:+.3f}" if ggs is not None else "n/a"
        art = avg_real(r)
        atxt = f"{art:.0%}" if art is not None else "n/a"   # not measured when Spotify off
        untr = art is not None and art <= UNTRUSTED_REAL_R1_MAX
        if untr:
            untrusted.append(r["model"])
        aps = avg_pair(r)
        ptxt = f"{aps:+.2f}" if aps is not None else "n/a"
        nh, nt = novel_totals(r)
        ntxt = f"{nh}/{nt}" if nt else "n/a"
        st = avg_structure(r)
        stxt = f"{st:.2f}" if st is not None else "n/a"
        flag = " ⚠️ UNTRUSTED" if untr else ""
        lines.append(f"| {i} | {r['model']}{flag} | {r.get('agent', '?')} | {r['avg_quality']:.3f} | {atxt} | {gtxt} | {ptxt} | {ntxt} | {stxt} | {r['n_ok']}/{r['n_total']} | {r['total_wallclock_s']:g} |")
    if untrusted:
        lines.append("")
        lines.append(f"> ⚠️ **UNTRUSTED** — avg real r1 ≤ {UNTRUSTED_REAL_R1_MAX:.0%} (≥75% of round-1 recs "
                     f"don't exist on Spotify; the pairing can't be trusted, skip human eval on it): "
                     + ", ".join(untrusted))

    # Per-profile detail for each model
    for r in rows:
        lines.append(f"\n## {r['model']}\n")
        lines.append("| profile | ok | recs | new artists | role mix (c/a/s) | real r1 | rescued | golden g/b | golden | pairs | novel | structure | just | avg conf | quality | status |")
        lines.append("|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
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
            if m.get("structure_total"):
                scol = f"{m['structure_points']:g}/{m['structure_total']:g}"
                if m.get("tag_coverage") is not None and m["tag_coverage"] < 0.6:
                    scol += f" ({m['tag_coverage']:.0%} tagged ⚠️)"
            else:
                scol = "—"
            jcol = f"{m['just_hits']}/{m['just_total']}" if m.get("just_total") else "—"
            status = m.get("status") or ""
            if m.get("raw_existence_rate") is not None and m["raw_existence_rate"] <= UNTRUSTED_REAL_R1_MAX:
                status = (status + " ⚠️untrusted").strip()
            if m.get("verification_error"):
                status = (status + " ⚠️spotify").strip()
            rolecol = _role_mix(block)
            lines.append(
                f"| {name} | {'✅' if m.get('ok') else '❌'} | {m['n_recs']} | {m['n_new_artists']} | {rolecol} "
                f"| {r1col} | {resc} | {ggb} | {gcol} | {pcol} | {ncol} | {scol} | {jcol} | {m.get('avg_confidence','')} | {m['quality']:.3f} | {status} |"
            )

        # Recs with the model's self-reported genre/era — display only (never
        # scored on); this is how untagged recs stay legible. Year is Spotify's
        # release_date when available.
        for name, block in r["per_profile"].items():
            recs = (block.get("outcome") or {}).get("recs") or []
            if not recs:
                continue
            lines.append(f"\n**{name} — recs** (genre/era = the model's own read, not scored on)")
            for rec in recs:
                bits = []
                if rec.get("self_genre"):
                    bits.append(rec["self_genre"])
                if rec.get("self_era"):
                    bits.append(rec["self_era"])
                if rec.get("release_year"):
                    bits.append(str(rec["release_year"]))
                title = rec.get("canonical_album") or rec.get("album", "")
                tag = f" *( {' · '.join(bits)} )" if bits else ""
                role = rec.get("role")
                lines.append(f"- {('**' + role + '** ') if role else ''}{rec.get('artist','?')} — {title}{tag}")
    return "\n".join(lines)


def append_run_history(results: list[dict], out_dir: Path, ts: str) -> int:
    """
    Append one line per (model × profile) run to eval/output/runs_history.jsonl.

    The per_<model>.json files are OVERWRITTEN each run; this append-only file
    is the cross-run memory. It is the input to the tag-harvest step
    (eval/harvest.py): which artists keep coming back, from which models, with
    which self-reported genre/era — the consensus that becomes human-reviewed
    tag proposals. Self-reported genre/era is display/harvest fuel only; it is
    never scored on.
    """
    path = out_dir / "runs_history.jsonl"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            for name, block in r["per_profile"].items():
                outcome = block.get("outcome") or {}
                recs = [
                    {
                        "artist": rec.get("artist"),
                        "album": rec.get("album"),
                        "self_genre": rec.get("self_genre"),
                        "self_era": rec.get("self_era"),
                        "release_year": rec.get("release_year"),
                        "confidence": rec.get("confidence"),
                        "verified": rec.get("verified"),
                        "round": rec.get("round"),
                    }
                    for rec in (outcome.get("recs") or [])
                ]
                f.write(json.dumps({
                    "ts": now, "run": ts, "model": r["model"], "agent": r.get("agent", ""),
                    "profile": name, "status": (block.get("metrics") or {}).get("status"),
                    "verification_error": (block.get("metrics") or {}).get("verification_error"),
                    "enriched": r.get("enriched"),
                    "recs": recs,
                }, ensure_ascii=False) + "\n")
                n += 1
    return n


def write_tsv(results: list[dict], path: Path) -> None:
    import csv
    cols = ["model", "agent", "profile", "status", "verification_error", "enriched", "ok", "n_recs", "n_new_artists",
            "raw_candidates", "round1_found", "round1_missing", "retry_rescued",
            "raw_existence_rate", "final_existence_rate", "untrusted",
            "golden_good", "golden_bad", "golden_score",
            "golden_pair_points", "golden_pair_total", "golden_pair_score",
            "golden_novel_hits", "golden_novel_total",
            "structure_points", "structure_total", "structure_score",
            "lane_share", "lane_balance", "tag_coverage",
            "just_hits", "just_total", "just_fixture_hits",
            "avg_confidence", "quality", "wallclock_s"]
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for r in results:
            for name, block in r["per_profile"].items():
                m = block["metrics"]
                w.writerow([
                    r["model"], m.get("agent", r.get("agent", "")), name, m.get("status"), m.get("verification_error") or "",
                    r.get("enriched") or "",
                    int(bool(m.get("ok"))),
                    m["n_recs"], m["n_new_artists"],
                    m.get("raw_candidates", ""), m.get("round1_found", ""),
                    m.get("round1_missing", ""), m.get("retry_rescued", ""),
                    (f"{m['raw_existence_rate']:.3f}" if m.get("raw_existence_rate") is not None else ""),
                    (f"{m['final_existence_rate']:.3f}" if m.get("final_existence_rate") is not None else ""),
                    (str(int(m["raw_existence_rate"] <= UNTRUSTED_REAL_R1_MAX))
                     if m.get("raw_existence_rate") is not None else ""),
                    m.get("golden_good", ""), m.get("golden_bad", ""),
                    (f"{m['golden_score']:.3f}" if m.get("golden_score") is not None else ""),
                    m.get("golden_pair_points", ""), m.get("golden_pair_total", ""),
                    (f"{m['golden_pair_score']:.3f}" if m.get("golden_pair_score") is not None else ""),
                    m.get("golden_novel_hits", ""), m.get("golden_novel_total", ""),
                    m.get("structure_points", ""), m.get("structure_total", ""),
                    (f"{m['structure_score']:.3f}" if m.get("structure_score") is not None else ""),
                    (f"{m['lane_share']:.3f}" if m.get("lane_share") is not None else ""),
                    (f"{m['lane_balance']:.3f}" if m.get("lane_balance") is not None else ""),
                    (f"{m['tag_coverage']:.3f}" if m.get("tag_coverage") is not None else ""),
                    m.get("just_hits", ""), m.get("just_total", ""), m.get("just_fixture_hits", ""),
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
                    temperature: float, use_spotify: bool,
                    agent: "afrec_agents.Agent",
                    thinking: bool | None = None, enricher=None) -> int:
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

    print(f"\n🎧 Interactive eval — profile: {prof['name']}   model: {model}   agent: {agent.tag}")
    print(f"   Spotify: {'on' if verifier else 'off'}\n")

    try:
        outcome = agent.run(
            prof["reviews"], make_llm(model, base_url),
            verifier=verifier, temperature=temperature,
            extra_options=mopts.merged(model, thinking),
            enricher=enricher,
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
            label=lab, model=model, agent=agent.tag,
            note=note, reason=(r.reason or "").strip(),
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
    cols = ["model", "agent", "profile", "artist", "album", "round", "confidence",
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
                        model, r.get("agent", ""), pname,
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
    parser.add_argument("--agent", default=afrec_agents.selected_agent().tag,
                        help="Agent version — the engine's playbook (prompts/call flow). "
                             "Tag like 'artist-then-album@v1' or bare id 'album-first' (latest). "
                             "Defaults to the selected agent (AFREC_AGENT in .env). See: make agents")
    parser.add_argument("--profile", help="Filter to one fixture by name or stem (repeatable).")
    parser.add_argument("--base-url", default=os.getenv("LLM_URL") or os.getenv("OLLAMA_URL") or "http://localhost:8092",
                        help="LLM server base URL (default: $LLM_URL or localhost:8092)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Force thinking OFF for every model this run (overrides afrec/model_options.json). "
                             "Default: the per-model file — Qwen runs no-think automatically, other models are unaffected.")
    parser.add_argument("--thinking", action="store_true",
                        help="Force thinking ON for every model this run (overrides the per-model file) — "
                             "for quality spot-checks; ~20x slower on Qwen.")
    parser.add_argument("--enrich", action="store_true",
                        help="Offer the enrichment layer's catalog facts to agents that accept them "
                             "(pool-map@v1.2 and later): year + tags + sound measurements in the map call, "
                             "served from the shared cache with a per-session Musicae budget. "
                             "Needs LASTFM_API_KEY and/or MUSICAE_API_KEY; without keys it is a no-op "
                             "and the run is identical to the agent's un-enriched path.")
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

    try:
        agent = afrec_agents.get_agent(args.agent)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 2

    # Enrichment: one Enricher for the WHOLE eval session (its Musicae budget
    # and latch are per-session by design). None → agents run un-enriched,
    # byte-identical to pre-enrichment behavior.
    enricher = None
    if args.enrich:
        from afrec import enrich
        enricher = enrich.Enricher(log=lambda s: print(f"   {s}", file=sys.stderr))
        if not enricher.available:
            print("   ⚠️  --enrich but no enrichment keys (LASTFM_API_KEY / MUSICAE_API_KEY) — "
                  "facts unavailable; agents run their un-enriched path", file=sys.stderr)

    if args.interactive:
        if args.profile is None:
            print("❌ --interactive needs a --profile. Available: "
                  + ", ".join(p["name"] for p in all_profiles), file=sys.stderr)
            return 2
        if len(profs) > 1:
            print(f"   (multiple matched '{args.profile}'; labeling first: {profs[0]['name']})")
        return run_interactive(profs, models[0], args.base_url, args.temperature,
                               use_spotify=not args.no_spotify, agent=agent,
                               thinking=thinking_override(args), enricher=enricher)

    print("\n🧪 Evaluation harness")
    print(f"   Models:    {', '.join(models)}")
    print(f"   Agent:     {agent.tag} — {agent.summary}")
    print(f"   Profiles:  {', '.join(p['name'] for p in profs)}")
    print(f"   Base URL:  {args.base_url}")
    print(f"   Spotify:   {'off' if args.no_spotify else 'on (if creds)'}")
    if args.enrich:
        print(f"   Enriched:  {'on' if (enricher and enricher.available) else 'off (no keys)'}")
    print("─" * 50)

    results = []
    for model in models:
        print(f"\n▶ Evaluating {model} (agent {agent.tag}) ...")
        result = evaluate_model(model, profs, args.base_url, args.temperature,
                                use_spotify=not args.no_spotify, agent=agent,
                                thinking=thinking_override(args), enricher=enricher)
        results.append(result)
        print(f"   {result['n_ok']}/{result['n_total']} profiles OK — avg quality {result['avg_quality']:.3f} "
              f"in {result['total_wallclock_s']}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for r in results:
        (out_dir / f"per_{r['model']}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False))

    # Cross-run memory (append-only): feeds the tag-harvest step (eval/harvest.py).
    n_hist = append_run_history(results, out_dir, ts)

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
    print(f"   (🧾 {n_hist} rec-runs appended → {out_dir / 'runs_history.jsonl'} — feeds 'make harvest')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
