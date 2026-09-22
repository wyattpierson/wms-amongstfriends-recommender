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

Records (append-only JSONL in eval/golden/, next to labeled.jsonl):
  scores.jsonl  every 1–5 rating, with the full recs so you can revisit them
  duels.jsonl   every ranking, including each model's recs

Typical use:
  # Rate one model's take on a profile (maybe 2 independent samples):
  python eval/taste.py score --profile jazzy_hiphop --models qwen2.5 --rounds 2

  # Three models, you pick the best:
  python eval/taste.py duel --profile wyatt --models llama3,qwen2.5,mistral

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
import humaneval as he                     # noqa: E402  (score/duel storage + stats)
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
              verifier) -> "ev.pipeline.RunOutcome":
    return ev.pipeline.run(
        profile["reviews"], ev.make_llm(model, base_url),
        verifier=verifier, temperature=temperature,
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


def run_score(profile: dict, model: str, base_url: str, temperature: float,
              rounds: int, use_spotify: bool) -> None:
    verifier = None
    if use_spotify:
        tok = ev.spotify.get_token()
        if tok:
            verifier = ev.spotify.verifier_for(tok)

    for rnd in range(1, rounds + 1):
        print(f"\n🎧 SCORE — profile: {profile['name']}   model: {model}   "
              f"(round {rnd}/{rounds})   Spotify: {'on' if verifier else 'off'}")
        t0 = time.time()
        try:
            outcome = _generate(profile, model, base_url, temperature, verifier)
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
            profile=profile["name"], model=model, score=score, note=note,
            recs=recs, spotify_on=bool(verifier), base_url=base_url,
            temperature=temperature, wallclock_s=round(time.time() - t0, 2),
            round=rnd,
        )
        print(f"   → {score} ({he.SCORE_LABELS[score]}, saved)")
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
             use_spotify: bool) -> None:
    verifier = None
    if use_spotify:
        tok = ev.spotify.get_token()
        if tok:
            verifier = ev.spotify.verifier_for(tok)

    print(f"\n⚔️  DUEL — profile: {profile['name']}   contenders: {', '.join(models)}")
    print(f"   Spotify: {'on' if verifier else 'off'}")

    responses: dict[str, list[dict]] = {}
    for model in models:
        print(f"\n   … generating for {model}")
        try:
            outcome = _generate(profile, model, base_url, temperature, verifier)
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
        ties=ties, note=note,
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

    score_rows = []
    for m in models:
        s = he.score_stats(model=m)
        if s["n"]:
            score_rows.append((m, s))
    if score_rows:
        print("  Response scores (1–5):")
        print(f"    {'model':<24} {'n':>3} {'avg':>5}   {'bad':>3} {'neut':>4} {'good':>4}   dist 1→5")
        for m, s in sorted(score_rows, key=lambda kv: -(kv[1]["avg"] or 0)):
            print(f"    {m:<24} {s['n']:>3} {s['avg']:>5.2f}   "
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

    print(f"\n  Records:  {he.SCORES_LOG}\n            {he.DUELS_LOG}\n")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", action="append",
                        help="Fixture profile name or suffix; repeatable. "
                             "Default: all profiles.")
    common.add_argument("--models", default=os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3",
                        help="Comma-separated model names (default: $LLM_MODEL or llama3)")
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
    args = parser.parse_args()

    if args.cmd in (None, "score", "duel") and not sys.stdin.isatty():
        print("⚠️  stdin is not a TTY — use a real terminal for interactive judging "
              "(or pipe answers, e.g. `printf '4\\n\\n' | ...`).", file=sys.stderr)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        parser.error("--models must name at least one model")

    if args.cmd == "stats":
        return run_stats()

    if args.cmd == "duel" and len(models) < 2:
        parser.error("duel needs at least two models, e.g. --models llama3,qwen2.5")
    if args.cmd == "score" and len(models) > 1:
        print("   (scoring each model in turn: " + ", ".join(models) + ")")

    profs = _pick_profiles(args)

    for prof in profs:
        if args.cmd == "score":
            for model in models:
                run_score(prof, model, args.base_url, args.temperature,
                          args.rounds, use_spotify=not args.no_spotify)
        else:
            run_duel(prof, models, args.base_url, args.temperature,
                     use_spotify=not args.no_spotify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
