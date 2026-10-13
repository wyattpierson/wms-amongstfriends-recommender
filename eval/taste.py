#!/usr/bin/env python3
"""
Human-taste evaluation: score whole responses, or pit models against each other.

This is the complement to evaluate.py's automated metrics and goldenset.py's
per-rec labels. Here *you* are the judge, judging the way a golden set can't:
a model can come up with a great answer nobody planned for — as long as you
like it, it counts.

Subcommands:

  score   One model answers a profile; you rate the WHOLE response 1–5
          (1 bad · 2 weak · 3 neutral · 4 good · 5 great). Repeat with
          --rounds N for N independent samples.

  duel    Two or more models answer the same profile; you rank the responses
          (shown as randomized letters to fight position bias). Feeds a
          running win-rate + Elo table.

  stats   Show everything: per-model response scores, distribution, duel
          win-rates, Elo ratings, and confidence calibration (does the
          model's self-claim track YOUR ratings?).

  pending The ASYNC LISTENING QUEUE: every rec you marked 'unsure' (u) in an
          interactive session — the ones you couldn't judge without actually
          listening. Plain call lists the queue; `pending --judge` walks it
          after you've listened and appends final good/bad verdicts.

  grade   OFFLINE BACKFILL: walk saved score/duel records (each stored its
          full recs) and grade the individual recs good/bad/unsure. Recs you
          already decisively labeled default to skip (--all re-labels).

Records (append-only JSONL in eval/golden/):
  scores.jsonl  every 1–5 rating, with the full recs so you can revisit them
  duels.jsonl   every ranking, including each model's recs
  (per-rec grades from every flow append to labeled.jsonl — the same corpus
   golden sets are harvested from — tagged by `source`, and carrying the
   pool score when graded off a scored/duelled pool)

Typical use:
  # Rate one model's take on a profile (maybe 2 independent samples):
  python eval/taste.py score --profile jazzy_hiphop --models qwen2.5 --rounds 2

  # Three models, you pick the best:
  python eval/taste.py duel --profile full_spectrum --models llama3,qwen2.5,mistral

  # Where do things stand?
  python eval/taste.py stats
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

# This file lives in eval/; evaluate.py does the sys.path/dotenv bootstrapping,
# so importing it first is what makes `afrec.*` importable here.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate as ev                      # noqa: E402  (bootstrap + shared helpers)
import goldenset as gs                     # noqa: E402  (labeled.jsonl — per-rec human corpus)
import humaneval as he                     # noqa: E402  (score/duel storage + stats)
from afrec import model_options as mopts   # noqa: E402  (per-model LLM options, e.g. Qwen no-think)
from afrec import agents as afrec_agents   # noqa: E402  (versioned engine playbooks)
from afrec.llm import LLMError             # noqa: E402


# ── Small interactive helpers ────────────────────────────────────────────────

def _ask(prompt: str, allow_eof: bool = False) -> str:
    """input() that returns '' on EOF instead of raising."""
    try:
        return input(prompt).strip()
    except EOFError:
        if allow_eof:
            return ""
        raise SystemExit("\n(EOF on stdin — nothing saved)")


def _recs_brief(outcome) -> list[dict]:
    return [
        {
            "artist": r.artist,
            "album": r.album,
            "confidence": r.confidence,
            "verified": bool(r.verified),
            "link": r.link or "",
            "reason": (r.reason or "").strip(),
        }
        for r in outcome.recs
    ]


def _print_response(recs: list[dict], letter: str = "") -> None:
    tag = f"{letter}. " if letter else ""
    print(f"\n{'─' * 62}")
    print(f"{tag}({len(recs)} recommendations)\n")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r['artist']} — {r['album']}"
              + ("" if r.get("verified") else "   (⚠ unverified)"))
        if r.get("reason"):
            for line in r["reason"].splitlines():
                line = line.strip()
                if line:
                    print(f"     {line}")


def _generate(profile: dict, model: str, base_url: str, temperature: float,
              verifier, agent: "afrec_agents.Agent",
              enricher=None) -> "ev.pipeline.RunOutcome":
    return agent.run(
        profile["reviews"], ev.make_llm(model, base_url),
        verifier=verifier, temperature=temperature,
        extra_options=mopts.options_for(model),
        enricher=enricher,
    )


def _pick_profiles(args) -> list[dict]:
    all_profiles = ev.discover_profiles()
    if not all_profiles:
        raise SystemExit(f"❌ No fixtures found in {ev.FIXTURES_DIR}")
    if not args.profile:
        return all_profiles
    wanted = {p.strip() for p in args.profile if p.strip()}
    sel = [p for p in all_profiles
           if p["name"] in wanted or any(p["name"].endswith(w) for w in wanted)]
    if not sel:
        raise SystemExit("❌ No profile matched. Available: "
                         + ", ".join(p["name"] for p in all_profiles))
    return sel


# ── score ─────────────────────────────────────────────────────────────────────

def _read_score() -> int | None:
    """Read a 1–5 rating. g/b/u map to 5/1/3. Returns None on EOF (abandon)."""
    while True:
        ans = _ask("   Rate 1–5 (1 bad · 2 weak · 3 neutral · 4 good · 5 great; "
                   "or g / u / b): ").lower()
        if ans in ("g", "good", "great", "yes", "y"):
            return 5
        if ans in ("b", "bad", "no", "n"):
            return 1
        if ans in ("u", "unsure", "neutral", "meh", "middle"):
            return 3
        if ans == "":
            continue
        if ans in ("1", "2", "3", "4", "5"):
            return int(ans)
        print("   (type 1–5, or g / u / b)")


def _read_gbu(prompt: str = "   g=good  b=bad  u=unsure   (enter=skip, q to stop): ") -> str | None:
    """Read a per-rec verdict. Returns 'good'/'bad'/'unsure'/'skip', or None to stop."""
    while True:
        ans = _ask(prompt, allow_eof=True).lower()
        if ans in ("g", "good", "yes", "y"):
            return "good"
        if ans in ("b", "bad", "no", "n"):
            return "bad"
        if ans in ("u", "unsure", "meh"):
            return "unsure"
        if ans == "":
            return "skip"
        if ans in ("q", "quit", "stop"):
            return None
        print("   (g / b / u, enter to skip, q to stop)")


def grade_recs(profile: str, model: str, recs: list[dict], *, agent: str = "",
               source: str, pool_score: int | None = None,
               skip_labeled: bool = False) -> dict | None:
    """
    Walk one model's recs and ask a per-rec verdict (good / bad / unsure).
    Each verdict appends a row to labeled.jsonl — the same per-rec corpus the
    golden sets are harvested from — tagged source=`source` ("score" | "duel"
    | "grade") and, when known, the pool score (1–5) the recs came from, so
    pool-vs-item consistency can be checked offline (he.pool_item_consistency).
    'unsure' flows into the pending to-listen queue like any other u label.

    skip_labeled=True (offline `grade` backfill) shows a hint for recs you
    already decisively labeled in this profile and defaults them to skip —
    type g/b/u to re-label anyway.

    Returns {"good": n, "bad": n, "unsure": n, "skipped": n}, or None if the
    human stopped (q) before any verdict.
    """
    counts = {"good": 0, "bad": 0, "unsure": 0, "skipped": 0}
    for i, r in enumerate(recs, 1):
        title = f"{r.get('artist') or '?'} — {r.get('album') or '(artist)'}"
        if not r.get("verified"):
            title += "   (⚠ unverified)"
        print(f"\n  [{i}/{len(recs)}] {title}")
        reason_lines = [ln.strip() for ln in (r.get("reason") or "").splitlines() if ln.strip()]
        if reason_lines:
            print(f"      {reason_lines[0]}")
        if skip_labeled:
            prev = [p for p in gs.labeled_for(profile, r.get("artist") or "",
                                              r.get("album") or "")
                    if p.get("label") in ("good", "bad")]
            if prev:
                print(f"      (already {prev[-1]['label']} in labeled.jsonl — enter to skip, g/b/u to re-label)")
        verdict = _read_gbu()
        if verdict is None:
            break
        if verdict == "skip":
            counts["skipped"] += 1
            continue
        note = ""
        if verdict in ("good", "bad"):
            note = _ask("   note (optional, enter to skip): ", allow_eof=True)
        gs.append_label(
            profile=profile, artist=r.get("artist") or "", album=r.get("album") or "",
            label="unknown" if verdict == "unsure" else verdict,   # corpus stores 'unknown'
            model=model, agent=agent, note=note,
            reason=r.get("reason") or "", source=source, pool_score=pool_score,
        )
        counts[verdict] += 1
        print(f"      → {verdict} (saved)")
    if not any(counts[k] for k in ("good", "bad", "unsure")):
        return None
    return counts


def run_score(profile: dict, model: str, base_url: str, temperature: float,
              rounds: int, use_spotify: bool, agent: "afrec_agents.Agent") -> None:
    verifier = None
    if use_spotify:
        tok = ev.spotify.get_token()
        if tok:
            verifier = ev.spotify.verifier_for(tok)

    for rnd in range(1, rounds + 1):
        print(f"\n🎧 SCORE — profile: {profile['name']}   model: {model}   agent: {agent.tag}   "
              f"(round {rnd}/{rounds})   Spotify: {'on' if verifier else 'off'}")
        t0 = time.time()
        try:
            outcome = _generate(profile, model, base_url, temperature, verifier, agent)
        except LLMError as e:
            print(f"❌ Generation failed for {model}: {e}", file=sys.stderr)
            return
        except Exception as e:
            print(f"❌ Generation failed for {model}: {e!r}", file=sys.stderr)
            return

        recs = _recs_brief(outcome)
        if not recs:
            print("❌ The engine returned no recs to rate (check the model / logs).")
            return
        _print_response(recs)

        score = _read_score()
        note = _ask("   note (optional, enter to skip): ", allow_eof=True)

        he.append_score(
            profile=profile["name"], model=model, agent=agent.tag, score=score, note=note,
            recs=recs, spotify_on=bool(verifier), base_url=base_url,
            temperature=temperature, wallclock_s=round(time.time() - t0, 2),
            round=rnd,
        )
        print(f"   → {score} ({he.SCORE_LABELS[score]}, saved)")

        ans = _ask("\n   Grade the individual recs? (Y/n): ", allow_eof=True).lower()
        if ans in ("n", "no"):
            print("   (skipped per-rec grading)")
        else:
            c = grade_recs(profile["name"], model, recs, agent=agent.tag,
                           source="score", pool_score=score)
            if c:
                print(f"   per-rec: {c['good']} good · {c['bad']} bad · {c['unsure']} unsure · "
                      f"{c['skipped']} skipped")

        s = he.score_stats(model=model)
        dist = " ".join(f"{k}:{v}" for k, v in s["distribution"].items() if v)
        print(f"   corpus for '{model}': n={s['n']}  avg={s['avg'] or 'n/a'}  "
              f"dist 1-5 = [{dist}]")
        cal = he.calibration_stats(model=model)
        if cal.get("n", 0) >= 3:
            extra = f", corr {cal['spearman']:+.2f}" if "spearman" in cal else ""
            print(f"   calibration: claims {cal['mean_conf']:.2f} vs your "
                  f"{cal['mean_taste']:.2f}/5 → {cal['verdict']} (bias {cal['bias']:+.2f}{extra})")
    print(f"\n✅ {rounds} score(s) saved → {he.SCORES_LOG}")


# ── duel ──────────────────────────────────────────────────────────────────────

def run_duel(profile: dict, models: list[str], base_url: str, temperature: float,
             use_spotify: bool, agent: "afrec_agents.Agent") -> None:
    verifier = None
    if use_spotify:
        tok = ev.spotify.get_token()
        if tok:
            verifier = ev.spotify.verifier_for(tok)

    print(f"\n⚔️  DUEL — profile: {profile['name']}   contenders: {', '.join(models)}   agent: {agent.tag}")
    print(f"   Spotify: {'on' if verifier else 'off'}")

    responses: dict[str, list[dict]] = {}
    for model in models:
        print(f"\n   … generating for {model}")
        try:
            outcome = _generate(profile, model, base_url, temperature, verifier, agent)
        except LLMError as e:
            print(f"❌ {model} failed: {e} — can't duel without it.", file=sys.stderr)
            return
        except Exception as e:
            print(f"❌ {model} failed: {e!r} — can't duel without it.", file=sys.stderr)
            return
        recs = _recs_brief(outcome)
        if not recs:
            print(f"❌ {model} produced no recs — can't duel.", file=sys.stderr)
            return
        responses[model] = recs

    # Randomize which letter is which model (fight position bias).
    order = models[:]
    random.shuffle(order)
    letters = {order[i]: chr(65 + i) for i in range(len(order))}
    model_for_letter = {v: k for k, v in letters.items()}

    for model in order:
        _print_response(responses[model], letter=letters[model])
    print("─" * 62)

    letters_in_order = [letters[m] for m in order]
    valid = set(letters_in_order)
    ranking: list[str] = []
    ties: list[str] = []

    # Winner (letter, or "t" for a full tie).
    while True:
        ans = _ask("Which response is BEST?  (" + " / ".join(letters_in_order)
                   + "/ t for tie): ").upper()
        if ans == "T" and len(models) == 2:
            ties = models[:]
            break
        if ans in valid:
            ranking = [model_for_letter[ans]]
            break
        if len(models) == 2 and ans in ("T", ""):
            print("   (pick a letter, or t for a tie)")
            continue
        print(f"   (type one of {' / '.join(letters_in_order)})")

    # Rank the rest (optional — Enter stops early; partial rankings still count).
    remaining = [m for m in models if m not in ranking]
    while remaining:
        left = [letters[m] for m in remaining]
        ans = _ask(f"Next best?  ({' / '.join(left)}, enter to stop): ",
                   allow_eof=True).upper()
        if not ans or ans not in left:
            break
        picked = model_for_letter[ans]
        ranking.append(picked)
        remaining = [mm for mm in remaining if mm != picked]

    if ties:
        verdict = "tie: " + " / ".join(ties)
    else:
        verdict = " > ".join(ranking)
        if remaining:
            verdict += "  (unranked: " + " / ".join(remaining) + ")"

    note = _ask("note (optional, enter to skip): ", allow_eof=True)
    path = he.append_duel(
        profile=profile["name"], ranking=ranking, unranked=remaining,
        ties=ties, note=note, agent=agent.tag,
        recs_by_model=responses, spotify_on=bool(verifier), base_url=base_url,
        temperature=temperature,
    )
    print(f"\n✅ verdict → {verdict}   (saved to {path.name})")

    elo = he.elo_from_duels()
    if elo:
        print("\n   Standings (Elo):")
        for m, s in sorted(elo.items(), key=lambda kv: -kv[1]["elo"]):
            print(f"     {m:<24} elo {s['elo']:>6.1f}   "
                  f"w {s['wins']}/{s['wins'] + s['losses']}  duels {s['duels']}")

    # Per-rec grading (optional): what was good/bad about the individual recs
    # in the pools you just ranked? Feeds labeled.jsonl (source="duel").
    if ranking:
        winner = ranking[0]
        ans = _ask(f"\nGrade the individual recs? (w = winner [{letters[winner]}], "
                   f"letter = that pool, all, or n): ", allow_eof=True).lower()
        if ans in ("w", "winner"):
            groups = [winner]
        elif ans in letters.values():
            groups = [model_for_letter[ans]]
        elif ans in ("all", "a"):
            groups = order
        else:
            groups = []
        for m in groups:
            c = grade_recs(profile["name"], m, responses[m], agent=agent.tag,
                           source="duel")
            if c:
                print(f"   per-rec for {m}: {c['good']} good · {c['bad']} bad · "
                      f"{c['unsure']} unsure · {c['skipped']} skipped")


# ── stats ─────────────────────────────────────────────────────────────────────

def run_stats() -> int:
    models = he.all_models()
    scores = [r for r in he.load_scores()]
    duels = he.load_duels()
    if not models and not scores and not duels:
        print("No human-judgment records yet.\n"
              "\n  Score a response:  python eval/taste.py score --profile <name> --models <model>\n"
              "  Duel models:       python eval/taste.py duel  --profile <name> --models a,b\n")
        return 0

    print(f"\n📊 Human-taste statistics   ({len(scores)} response scores, "
          f"{len(duels)} duels)\n")

    # Group scores by (model, agent) — the agent is the engine's playbook,
    # and the same model can be judged under several versions over time.
    score_keys: list[tuple[str, str]] = []
    for r in scores:
        key = (r.get("model") or "?", (r.get("agent") or "").strip())
        if key not in score_keys:
            score_keys.append(key)
    score_rows = []
    for m, ag in score_keys:
        s = he.score_stats(model=m, agent=ag or None)
        if s["n"]:
            score_rows.append((m, ag, s))
    if score_rows:
        print("  Response scores (1–5)   [agent = engine playbook that generated the response]")
        print(f"    {'model / agent':<38} {'n':>3} {'avg':>5}   {'bad':>3} {'neut':>4} {'good':>4}   dist 1→5")
        for m, ag, s in sorted(score_rows, key=lambda kv: -(kv[2]["avg"] or 0)):
            label = f"{m}" + (f" [{ag}]" if ag else "")
            print(f"    {label:<38} {s['n']:>3} {s['avg']:>5.2f}   "
                  f"{s['bad']:>3} {s['neutral']:>4} {s['good']:>4}   "
                  f"[{s['distribution'][1]} {s['distribution'][2]} "
                  f"{s['distribution'][3]} {s['distribution'][4]} {s['distribution'][5]}]")
        print()

    elo = he.elo_from_duels()
    if elo:
        print("  Duel standings:")
        print(f"    {'model':<24} {'elo':>7} {'win%':>6} {'w-l':>6}  {'duels':>5} {'ties':>5}")
        for m, s in sorted(elo.items(), key=lambda kv: -kv[1]["elo"]):
            wr = f"{s['win_rate']:.0%}" if s["win_rate"] is not None else " n/a"
            print(f"    {m:<24} {s['elo']:>7.1f} {wr:>6} {str(s['wins']) + '-' + str(s['losses']):>6}  "
                  f"{s['duels']:>5} {s['ties']:>5}")
        print()

    agstats = he.agent_duel_stats()
    if agstats:
        print("  Agent ladders (same-model duels — the model never beats itself;")
        print("  these roll up agent-vs-agent verdicts across models):")
        print(f"    {'agent':<32} {'win%':>6} {'w-l':>7}  {'duels':>5} {'ties':>5}")
        for a, s in sorted(agstats.items(),
                           key=lambda kv: (-(kv[1]["win_rate"] or 0), -kv[1]["wins"])):
            wr = f"{s['win_rate']:.0%}" if s["win_rate"] is not None else " n/a"
            print(f"    {a:<32} {wr:>6} {str(s['wins']) + '-' + str(s['losses']):>7}  "
                  f"{s['duels']:>5} {s['ties']:>5}")
        print()

    cal_rows = []
    for m in models:
        c = he.calibration_stats(model=m)
        if c.get("n"):
            cal_rows.append((m, c))
    if cal_rows:
        print("  Confidence calibration (self-claimed confidence vs YOUR ratings):")
        print("    ⚠ small n = directional only; correlation needs ≥3 rated responses")
        print(f"    {'model':<24} {'n':>3} {'avg claim':>9} {'your avg':>8}  {'bias':>9}  {'read':<17} {'corr':>7}")
        for m, c in sorted(cal_rows, key=lambda kv: abs(kv[1]["bias"])):
            corr = f"{c['spearman']:+.2f}" if "spearman" in c else "   — "
            print(f"    {m:<24} {c['n']:>3} {c['mean_conf']:>9.2f} {c['mean_taste']:>8.2f}  "
                  f"{c['bias']:>+9.2f}  {c['verdict']:<17} {corr:>7}")

        # Red-flag table: what models claim when you rate bad vs good.
        agg: dict[str, list] = {name: [] for name, _ in he.BUCKETS}
        for r in he.load_scores():
            s = r.get("score")
            if not (isinstance(s, int) and 1 <= s <= 5):
                continue
            confs = [rec.get("confidence") for rec in (r.get("recs") or [])
                     if isinstance(rec, dict) and isinstance(rec.get("confidence"), (int, float))]
            if not confs:
                continue
            avg = sum(confs) / len(confs)
            for name, (lo, hi) in he.BUCKETS:
                if lo <= s <= hi:
                    agg[name].append(avg)
        if any(agg.values()):
            print("\n    Claim vs taste (avg confidence claimed on responses you rated…):")
            for name, (lo, hi) in he.BUCKETS:
                vals = agg[name]
                if vals:
                    flag = "   ← red flag: confident but you hated it" if name == "bad" and sum(vals) / len(vals) >= 0.7 else ""
                    print(f"      you said {name:<8} (r{lo}–{hi})  → model claimed {sum(vals) / len(vals):.2f} avg  ({len(vals)} responses){flag}")
        print()

    # Per-rec item verdicts (labeled.jsonl): what was good/bad about the
    # individual recs, from every source (interactive, pending-judge, score,
    # duel, grade) — the corpus the golden sets are harvested from.
    labeled = gs.load_labeled()
    ls = gs.labeled_stats()
    print("  ITEM VERDICTS  (per-rec good/bad — all sources)")
    if ls["total"]:
        gr = f"{ls['good_rate']:.0%}" if ls["good_rate"] is not None else "n/a"
        print(f"    n={ls['total']}  good_rate={gr}  ({ls['good']} good · {ls['bad']} bad · "
              f"{ls['unknown']} unsure)")
        by_src: dict[str, list[str]] = {}
        for r in labeled:
            by_src.setdefault(r.get("source") or "?", []).append(r.get("label"))
        for src in sorted(by_src):
            labs = by_src[src]
            g, b, u = labs.count("good"), labs.count("bad"), labs.count("unknown")
            dec = g + b
            rate = f"good_rate={g / dec:.0%}" if dec else "no decisive yet"
            print(f"      {src:<13} n={len(labs):<3} {rate}   ({g} good · {b} bad · {u} unsure)")
        # Pool-vs-item consistency: does your 1–5 pool score match the recs you judged?
        cons = he.pool_item_consistency()
        if cons:
            print("\n    POOL vs ITEM  (your 1–5 score vs the per-rec verdicts on those same recs)")
            agg: dict[str, list[dict]] = {}
            for c in cons:
                bucket = "bad" if c["pool"] <= 2 else "neutral" if c["pool"] == 3 else "good"
                agg.setdefault(bucket, []).append(c)
            for bucket in ("bad", "neutral", "good"):
                rows_ = agg.get(bucket)
                if not rows_:
                    continue
                mm = sum(1 for c in rows_ if c["mismatch"])
                print(f"      pool {bucket:<8} n={len(rows_):<3} mismatched={mm}")
            if any(c["mismatch"] for c in cons):
                print("      ⚠ mismatch = pool rated 4–5 but <50% of judged recs were good (or 1–2 but >50%)")
        # Golden-set promotion candidates: enough decisive labels, not yet in the file.
        promo = gs.promotion_candidates()
        if promo:
            print("\n    GOLDEN-SET PROMOTION CANDIDATES  (enough labels, not yet in the golden file)")
            for prof in sorted(promo):
                for kind in ("good", "bad"):
                    for e in promo[prof][kind]:
                        albums = f"  [{', '.join(e['albums'][:2])}]" if e["albums"] else ""
                        print(f"      {prof:<16} {kind:<5} {e['artist']:<32} "
                              f"({e['good']} good / {e['bad']} bad){albums}")
            print("      → `make harvest` folds your verdicts into the golden sets")
        print()
    else:
        print("    (no per-rec labels yet — judge recs via score/duel/grade, or evaluate.py --interactive)\n")

    pending_n = sum(1 for r in labeled if r.get("label") == "unknown")
    if pending_n:
        print(f"  Pending 'unsure' labels (to-listen queue): {pending_n} — see `taste.py pending`\n")
    print(f"\n  Records:  {he.SCORES_LOG}\n            {he.DUELS_LOG}\n")
    return 0


def run_pending(args) -> int:
    """
    The async listening queue. In an interactive session, recs you can't judge
    without actually listening get marked 'u' (unsure) and land in labeled.jsonl
    as label="unknown" — they don't count toward good_rate until resolved.

    Plain `pending`: list the queue (deduped to unique artist+album) so you can
    drop them in a Spotify playlist and listen whenever.
    `pending --judge`: walk the queue and append final good/bad verdicts.
    The corpus is append-only, so resolving just adds a new row; the old
    "unknown" row stays but is ignored by the stats.
    """
    rows = [r for r in gs.load_labeled() if r.get("label") == "unknown"]
    if not rows:
        print("No pending 'unsure' labels — the queue is empty.")
        print("(In an interactive session, press u on a rec you need to listen to first.)")
        return 0

    # Dedupe to unique artist+album for the listening queue (same rec across
    # sessions/models = one thing to listen to).
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = ((r.get("artist") or "").strip().lower(), (r.get("album") or "").strip().lower())
        e = seen.get(key)
        if e is None:
            e = dict(r)
            e["n"] = 0
            e["models"] = set()
            seen[key] = e
        e["n"] += 1
        e["models"].add(r.get("model") or "?")
    queue = sorted(seen.values(), key=lambda e: e.get("ts") or "")

    print(f"🎧 Pending — {len(queue)} unique rec(s) you marked 'unsure' "
          f"({len(rows)} labels total) — your to-listen queue:\n")
    for i, e in enumerate(queue, 1):
        models = ",".join(sorted(e["models"]))
        print(f"{i}. {e.get('artist')} — {e.get('album') or '?'}")
        print(f"   profile={e.get('profile')}  model={models}  agent={e.get('agent') or '?'}  "
              f"first seen {str(e.get('ts', '?'))[:10]}  seen×{e['n']}")
        if (e.get("note") or "").strip():
            print(f"   note: {e['note'].strip()}")
        if (e.get("reason") or "").strip():
            print(f"   model's reason: {e['reason'].strip()[:140]}")

    if not args.judge:
        print("\nQueue these up in Spotify, listen whenever, then run:")
        print("  python eval/taste.py pending --judge")
        return 0

    print("\n── judging (verdicts append to labeled.jsonl; the old 'unknown' rows stay but don't count) ──")
    resolved = 0
    for i, e in enumerate(queue, 1):
        print(f"{i}. {e.get('artist')} — {e.get('album') or '?'}")
        lab = None
        while lab is None:
            ans = _ask("   good / bad / skip (g / b / s): ", allow_eof=True).lower()
            if ans in ("g", "good"):
                lab = "good"
            elif ans in ("b", "bad"):
                lab = "bad"
            elif ans in ("s", "skip", ""):
                break
        if lab is None:
            print("   → skipped\n")
            continue
        note = _ask("   note (enter to skip): ", allow_eof=True)
        gs.append_label(
            profile=e.get("profile") or "", artist=e.get("artist") or "",
            album=e.get("album") or "", label=lab,
            model=e.get("model") or "", agent=e.get("agent") or "",
            note=f"[resolves unsure] {note}".strip(),
            reason=e.get("reason") or "", source="pending-judge",
        )
        resolved += 1
        print(f"   → {lab}\n")
    print(f"Resolved {resolved} of {len(queue)} pending recs. `taste.py stats` now counts them.")
    return 0


# ── grade (offline backfill) ──────────────────────────────────────────────────

def _grade_records(args) -> list[dict]:
    """Flatten saved score/duel records into gradeable units, newest first."""
    records = []
    if "score" in args.kinds:
        for r in he.load_scores():
            if not r.get("recs"):
                continue
            model = r.get("model") or ""
            if args.model and args.model.lower() not in model.lower():
                continue
            if args.profile and r.get("profile") not in args.profile:
                continue
            records.append({"ts": r.get("ts", ""), "kind": "score",
                            "profile": r.get("profile") or "", "model": model,
                            "agent": r.get("agent") or "", "pool": r.get("score"),
                            "recs": r["recs"]})
    if "duel" in args.kinds:
        for d in he.load_duels():
            if args.profile and d.get("profile") not in args.profile:
                continue
            for label, recs in (d.get("recs_by_model") or {}).items():
                if not recs:
                    continue
                if " [" in label:          # agent-duel label: "Model [agent@v1]"
                    model, agent = label.split(" [", 1)
                    agent = agent.rsplit("]", 1)[0]
                else:
                    model, agent = label, d.get("agent") or ""
                if args.model and args.model.lower() not in model.lower():
                    continue
                records.append({"ts": d.get("ts", ""), "kind": "duel",
                                "profile": d.get("profile") or "", "model": model,
                                "agent": agent, "pool": None, "recs": recs})
    records.sort(key=lambda r: r["ts"], reverse=True)
    return records[:args.limit]


def run_grade(args) -> int:
    """
    Offline backfill: walk saved score/duel records (each stored its full
    recs) and grade the individual recs good/bad/unsure into labeled.jsonl.
    Recs you already decisively labeled in that profile default to skip
    (type g/b/u to re-label); --all walks everything.
    """
    kinds = {k.strip().lower() for k in args.grade_from.split(",") if k.strip()}
    unknown = kinds - {"score", "duel"}
    if unknown:
        raise SystemExit(f"❌ unknown --from kind(s): {', '.join(sorted(unknown))} "
                         "(expected score and/or duel)")
    args.kinds = kinds
    records = _grade_records(args)
    if not records:
        print("No matching records in scores.jsonl / duels.jsonl (nothing to grade yet).")
        return 1
    print(f"\n🔎 GRADING {len(records)} saved record(s) — newest first "
          f"(already-judged recs default to skip; --all re-labels)")
    totals = {"good": 0, "bad": 0, "unsure": 0, "skipped": 0}
    for i, r in enumerate(records, 1):
        pool = f" · pool {r['pool']}/5" if r["pool"] else ""
        print(f"\n{'═' * 62}\n  record {i}/{len(records)} · {r['kind']} · {r['profile']} · "
              f"{r['model']} · {str(r['ts'])[:10]}{pool}")
        ans = _ask(f"  grade these {len(r['recs'])} recs? (enter = yes, s = skip, q = stop): ",
                   allow_eof=True).lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        c = grade_recs(r["profile"], r["model"], r["recs"], agent=r["agent"],
                       source="grade", pool_score=r["pool"],
                       skip_labeled=not args.all)
        if c:
            for k in totals:
                totals[k] += c[k]
    n = totals["good"] + totals["bad"]
    rate = f"   (good_rate over new decisive: {totals['good'] / n:.0%})" if n else ""
    print(f"\n✅ Done. {totals['good']} good · {totals['bad']} bad · "
          f"{totals['unsure']} unsure · {totals['skipped']} skipped{rate}")
    print("   standings: python eval/taste.py stats")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", action="append",
                        help="Fixture profile name or suffix; repeatable. "
                             "Default: all profiles.")
    common.add_argument("--models", default=os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3",
                        help="Comma-separated model names (default: $LLM_MODEL or llama3)")
    common.add_argument("--agent", default=afrec_agents.selected_agent().tag,
                        help="Agent version — the engine's playbook (prompts/call flow). "
                             "Tag like 'artist-then-album@v1' or bare id 'album-first' (latest). "
                             "Defaults to the selected agent (AFREC_AGENT in .env). See: make agents")
    common.add_argument("--base-url", default=os.getenv("LLM_URL") or os.getenv("OLLAMA_URL") or "http://localhost:8092")
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--no-spotify", action="store_true",
                        help="Skip Spotify verification (faster).")
    common.add_argument("--rounds", type=int, default=1,
                        help="score: independent samples to rate per model (default 1)")

    parser = argparse.ArgumentParser(
        description="Score model responses or duel them — you are the judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("score", parents=[common],
                   help="rate one model's full response 1–5")
    sub.add_parser("duel", parents=[common],
                   help="rank responses from two or more models")
    sub.add_parser("stats", parents=[common],
                   help="show score + duel standings")
    p_pending = sub.add_parser("pending",
                   help="list (or with --judge, re-judge) 'unsure' labels — the to-listen queue")
    p_pending.add_argument("--judge", action="store_true",
                           help="walk the queue and append final good/bad verdicts")
    p_grade = sub.add_parser("grade", parents=[common],
                   help="grade recs from saved score/duel records (offline backfill)")
    p_grade.add_argument("--from", dest="grade_from", default="score,duel",
                         help="record kinds to backfill from (default: score,duel)")
    p_grade.add_argument("--model", default=None,
                         help="only records whose model name contains this substring")
    p_grade.add_argument("--limit", type=int, default=10,
                         help="max records to walk (default 10)")
    p_grade.add_argument("--all", action="store_true",
                         help="don't default-skip recs already decisively labeled")
    args = parser.parse_args()

    if args.cmd in (None, "score", "duel", "grade") and not sys.stdin.isatty():
        print("⚠️  stdin is not a TTY — use a real terminal for interactive judging "
              "(or pipe answers, e.g. `printf '4\\n\\n' | ...`).", file=sys.stderr)

    if args.cmd == "stats":
        return run_stats()

    if args.cmd == "pending":   # reads labeled.jsonl only — no models/profiles needed
        return run_pending(args)

    if args.cmd == "grade":     # reads scores.jsonl/duels.jsonl only — no generation
        return run_grade(args)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        parser.error("--models must name at least one model")

    if args.cmd == "duel" and len(models) < 2:
        parser.error("duel needs at least two models, e.g. --models llama3,qwen2.5")
    if args.cmd == "score" and len(models) > 1:
        print("   (scoring each model in turn: " + ", ".join(models) + ")")

    profs = _pick_profiles(args)

    if args.cmd in ("score", "duel"):
        try:
            agent = afrec_agents.get_agent(args.agent)
        except ValueError as e:
            parser.error(str(e))

    for prof in profs:
        if args.cmd == "score":
            for model in models:
                run_score(prof, model, args.base_url, args.temperature,
                          args.rounds, use_spotify=not args.no_spotify, agent=agent)
        else:
            run_duel(prof, models, args.base_url, args.temperature,
                     use_spotify=not args.no_spotify, agent=agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
