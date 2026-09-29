#!/usr/bin/env python3
"""
Duel sessions — batch the interactive duels so ONE command runs you through a
whole matrix, with resume. `taste.py duel` is single-shot; this drives many
of them in one terminal session (you still judge every duel — letters are
randomized exactly like `taste.py duel`).

Subcommands:

  agents    same model, agent A vs agent B — "which playbook is better?"
            One model is loaded at a time, so this respects the one-big-model
            limit automatically. Default: the 4 medium models × 3 profiles.

  models    different models, each with its PREFERRED agent — "which model?"
            --contestants "Model:agent@v1,OtherModel" (omit :agent to use
            the selected agent). Mixed agents → bare-model Elo ladder;
            all-same agent → per-agent keyed ladder (like taste.py duel).

  status    show every session and its progress.

Progress is saved after EVERY duel (eval/output/duel_sessions/<session>.json),
so you can press q — or Ctrl-C — and re-run the same command to pick up
exactly where you left off. Delete the state file to start a session over.
Each finished duel appends to eval/golden/duels.jsonl (same as taste.py), so
`python eval/taste.py stats` shows the running Elo / win-rates throughout.

Typical use:

  # Phase 1: agent showdown — 4 medium models × 3 profiles = 12 duels
  python eval/duels.py agents

  # or explicit:
  python eval/duels.py agents \
      --models Dolphin3.0-Llama3.1-8B.Q8_0,Meta-Llama-3.1-8B-Instruct-Q8_0,Mistral-7B-Instruct-v0.3-Q8_0,microsoft_Phi-4-mini-instruct-Q8_0 \
      --profiles jazzy_hiphop,warm_groove,alt_rnb

  # Phase 2: model showdown, each with its preferred agent
  python eval/duels.py models \
      --contestants Qwen3.8-27B-Q8_0:album-first@v1,Mistral-7B-Instruct-v0.3-Q8_0:artist-then-album@v1 \
      --profiles jazzy_hiphop,warm_groove

  # peek at the queue without generating anything
  python eval/duels.py agents --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate as ev                      # noqa: E402  (bootstrap + shared helpers)
import humaneval as he                     # noqa: E402  (duel storage + Elo)
import taste as t                          # noqa: E402  (reuse generation + judging helpers)
from afrec import agents as afrec_agents   # noqa: E402
from afrec.llm import LLMError             # noqa: E402

SESSIONS_DIR = ev.OUTPUT_DIR / "duel_sessions"


def _ask_line(prompt: str) -> str | None:
    """input() that returns None on EOF — lets a dead stdin mean "quit"
    (taste._ask turns EOF into "", which would loop forever here)."""
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        return None

# Models that must not be loaded alongside another big model (64 GB box —
# see EVAL_PLAN.md "Hardware constraints"). gemma-4-26B-A4B is a MoE with 4B
# active and is treated as pairable, per the plan.
BIG_MODELS = {"Qwen3.8-27B-Q8_0"}

MEDIUM_MODELS = [
    "Dolphin3.0-Llama3.1-8B.Q8_0",
    "Meta-Llama-3.1-8B-Instruct-Q8_0",
    "Mistral-7B-Instruct-v0.3-Q8_0",
    "microsoft_Phi-4-mini-instruct-Q8_0",
]
DEFAULT_PROFILES = ["jazzy_hiphop", "warm_groove", "alt_rnb"]


@dataclass(frozen=True)
class Contestant:
    label: str   # identity in the duel record AND the Elo key
    model: str   # pool model name
    agent: str   # agent tag that generated the response

    def key(self) -> str:
        return (self.model, self.agent)


class MatchupError(RuntimeError):
    """A duel could not be completed (generation failed, no recs, …)."""


# ── Matchup building ─────────────────────────────────────────────────────────

def agents_matchups(models: list[str], profiles: list[dict],
                    agent_a: str, agent_b: str) -> list[tuple[dict, list[Contestant]]]:
    """Same model, two agents. Model-major order so the pool keeps the model
    loaded across that model's profiles (fast reloads)."""
    out = []
    for m in models:
        for p in profiles:
            out.append((p, [
                Contestant(f"{m} [{agent_a}]", m, agent_a),
                Contestant(f"{m} [{agent_b}]", m, agent_b),
            ]))
    return out


def models_matchups(contestants: list[Contestant],
                    profiles: list[dict]) -> list[tuple[dict, list[Contestant]]]:
    """Different models (each with its preferred agent) on each profile."""
    return [(p, list(contestants)) for p in profiles]


def parse_contestants(spec: str, default_agent: str) -> list[Contestant]:
    """Parse "Model:agent@v1,OtherModel" into Contestants (label = model)."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            model, agent = (x.strip() for x in part.split(":", 1))
        else:
            model, agent = part, default_agent
        try:
            agent = afrec_agents.get_agent(agent).tag
        except ValueError as e:
            raise SystemExit(f"❌ {e}")
        out.append(Contestant(model, model, agent))
    if len(out) < 2:
        raise SystemExit("❌ models needs at least 2 contestants, "
                         "e.g. --contestants modelA:agent@v1,modelB")
    return out


# ── Session state (resume) ───────────────────────────────────────────────────

def matchup_key(profile: dict, conts: list[Contestant]) -> str:
    return json.dumps([profile["name"]] + [c.label for c in conts],
                      ensure_ascii=False)


def session_path(name: str) -> Path:
    return SESSIONS_DIR / f"{name}.json"


def load_state(name: str) -> dict | None:
    p = session_path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        print(f"⚠️  {p} is corrupt — starting the session fresh.", file=sys.stderr)
        return None


def save_state(state: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    p = session_path(state["name"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(p)


def done_map(state: dict | None) -> dict[str, dict]:
    """{matchup_key: record} for everything already finished/skipped."""
    if not state:
        return {}
    return {m["key"]: m for m in state.get("matchups", [])
            if m.get("status") in ("done", "skipped")}


# ── One duel (generation + judging), reusing taste.py's mechanics ────────────

def run_matchup(profile: dict, conts: list[Contestant], base_url: str,
                temperature: float, use_spotify: bool) -> str:
    """Generate for each contestant, let the human rank, append to duels.jsonl.
    Returns the verdict string. Raises MatchupError on generation failure."""
    verifier = None
    if use_spotify:
        tok = ev.spotify.get_token()
        if tok:
            verifier = ev.spotify.verifier_for(tok)

    labels = [c.label for c in conts]
    print(f"\n⚔️  DUEL — profile: {profile['name']}   contenders: {'  vs  '.join(labels)}")
    print(f"   Spotify: {'on' if verifier else 'off'}")

    # Generate (dedupe identical model+agent, e.g. retries in one session).
    cache: dict[tuple, list[dict]] = {}
    responses: dict[str, list[dict]] = {}
    for c in conts:
        if c.key() not in cache:
            t0 = time.time()
            print(f"\n   … generating for {c.label} ({c.model} × {c.agent})")
            try:
                agent = afrec_agents.get_agent(c.agent)
                outcome = t._generate(profile, c.model, base_url, temperature,
                                      verifier, agent)
            except LLMError as e:
                raise MatchupError(f"{c.label} failed: {e}") from e
            except Exception as e:
                raise MatchupError(f"{c.label} failed: {e!r}") from e
            recs = t._recs_brief(outcome)
            if not recs:
                raise MatchupError(f"{c.label} produced no recs")
            cache[c.key()] = recs
            print(f"   … done in {time.time() - t0:.0f}s")
        responses[c.label] = cache[c.key()]

    # Randomized letters (fight position bias) — same as taste.run_duel.
    order = labels[:]
    random.shuffle(order)
    letters = {order[i]: chr(65 + i) for i in range(len(order))}
    model_for_letter = {v: k for k, v in letters.items()}
    for lab in order:
        t._print_response(responses[lab], letter=letters[lab])
    print("─" * 62)

    letters_in_order = [letters[lab] for lab in order]
    valid = set(letters_in_order)
    ranking: list[str] = []
    ties: list[str] = []

    while True:
        tie = " / t for tie" if len(labels) == 2 else ""
        ans = _ask_line("Which response is BEST?  (" + " / ".join(letters_in_order)
                        + tie + "): ")
        if ans is None:              # EOF (closed terminal) — quit, keep progress
            raise KeyboardInterrupt
        ans = ans.upper()
        if ans == "T" and len(labels) == 2:
            ties = labels[:]
            break
        if ans in valid:
            ranking = [model_for_letter[ans]]
            break
        if len(labels) == 2 and ans in ("T", ""):
            print("   (pick a letter, or t for a tie)")
            continue
        print(f"   (type one of {' / '.join(letters_in_order)})")

    remaining = [lab for lab in labels if lab not in ranking]
    while remaining:
        left = [letters[lab] for lab in remaining]
        ans = _ask_line(f"Next best?  ({' / '.join(left)}, enter to stop): ")
        if ans is None:              # EOF — stop ranking, keep the winner
            break
        ans = ans.upper()
        if not ans or ans not in left:
            break
        picked = model_for_letter[ans]
        ranking.append(picked)
        remaining = [lab for lab in remaining if lab != picked]

    if ties:
        verdict = "tie: " + " / ".join(ties)
    else:
        verdict = " > ".join(ranking)
        if remaining:
            verdict += "  (unranked: " + " / ".join(remaining) + ")"
    note = _ask_line("note (optional, enter to skip): ") or ""

    # Elo keying: if every contestant used the same agent, record it at the
    # record level (per-agent ladder, like taste.py duel). If agents differ
    # (agent duels / preferred-agent model duels), record none — the labels
    # carry the identity ("Model [agent]" or bare model).
    agents_used = {c.agent for c in conts}
    common_agent = next(iter(agents_used)) if len(agents_used) == 1 else ""
    he.append_duel(
        profile=profile["name"], ranking=ranking, unranked=remaining,
        ties=ties, agent=common_agent, note=note,
        agents={c.label: c.agent for c in conts},
        recs_by_model=responses, spotify_on=bool(verifier), base_url=base_url,
        temperature=temperature,
    )
    print(f"\n✅ verdict → {verdict}   (saved to {he.DUELS_LOG.name})")
    _print_elos()
    return verdict


def _print_elos() -> None:
    elo = he.elo_from_duels()
    if not elo:
        return
    print("\n   Standings (Elo):")
    for m, s in sorted(elo.items(), key=lambda kv: -kv[1]["elo"]):
        print(f"     {m:<52} elo {s['elo']:>6.1f}   "
              f"w {s['wins']}/{s['wins'] + s['losses']}  duels {s['duels']}")


# ── Session runner ───────────────────────────────────────────────────────────

def _big_model_guard(conts: list[Contestant]) -> bool:
    """True = proceed. Warns when a matchup would load 2+ big models."""
    bigs = sorted({c.model for c in conts if c.model in BIG_MODELS})
    if len(bigs) < 2:
        return True
    print(f"\n⚠️  This matchup loads {len(bigs)} big models at once ({', '.join(bigs)}) — "
          f"tight on 64 GB. See EVAL_PLAN.md hardware constraints.")
    ans = _ask_line("   enter = try it anyway, s = skip, q = quit: ")
    if ans is None:
        raise KeyboardInterrupt
    ans = ans.lower()
    if ans in ("s", "skip"):
        return False
    if ans in ("q", "quit", "exit", ""):
        raise KeyboardInterrupt
    return True


def _pause() -> str:
    while True:
        try:
            ans = input("\nNext duel — enter = go, s = skip this one, q = quit "
                        "(progress is saved): ")
        except EOFError:
            print()
            return "quit"
        ans = ans.lower()
        if ans in ("", "enter", "next", "n", "g", "go"):
            return "next"
        if ans in ("s", "skip"):
            return "skip"
        if ans in ("q", "quit", "exit"):
            return "quit"


def run_session(cmd: str, matchups: list[tuple[dict, list[Contestant]]], args) -> None:
    if not matchups:
        raise SystemExit("❌ empty matchup queue — check --models/--profiles")

    key = json.dumps({
        "cmd": cmd,
        "models": args.models if cmd == "agents" else args.contestants,
        "profiles": args.profiles,
        "agents": [args.agent_a, args.agent_b] if cmd == "agents" else None,
    }, ensure_ascii=False, sort_keys=True)
    name = args.session or f"{cmd}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"
    state = load_state(name) or {
        "name": name, "cmd": cmd, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "matchups": [],
    }
    finished = done_map(state)

    pending, skipped_before = [], 0
    for prof, conts in matchups:
        k = matchup_key(prof, conts)
        if k in finished:
            skipped_before += 1
        else:
            pending.append((k, prof, conts))

    print(f"\n🏟️  DUEL SESSION: {name}   ({cmd})")
    print(f"   state: {os.path.relpath(session_path(name))}")
    print(f"   plan: {len(matchups)} duels — {skipped_before} already done, "
          f"{len(pending)} to go\n")
    for i, (k, prof, conts) in enumerate(pending, 1):
        print(f"   {i:>2}. {prof['name']:<16} "
              + "  vs  ".join(c.label for c in conts))

    if args.dry_run:
        print("\n(dry run — nothing generated, nothing saved)")
        return

    if not pending:
        print("\n✅ Nothing left in this session — run `python eval/taste.py stats`.")
        return

    if skipped_before:
        print(f"\n(continuing — {skipped_before} finished duel(s) skipped)")

    skip_next = False
    for i, (k, prof, conts) in enumerate(pending, 1):
        if skip_next:
            skip_next = False
            print(f"\n⏭️  skipping: {prof['name']}  vs  "
                  + "  vs  ".join(c.label for c in conts))
            state["matchups"].append({
                "key": k, "profile": prof["name"],
                "contestants": [{"label": c.label, "model": c.model,
                                 "agent": c.agent} for c in conts],
                "status": "skipped", "verdict": "skipped at pause",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            save_state(state)
            continue
        print(f"\n{'═' * 62}\n   duel {i}/{len(pending)} of {len(matchups)} total")
        status = "done"
        verdict = ""
        try:
            if not _big_model_guard(conts):
                status = "skipped"
                verdict = "skipped (memory guard)"
            else:
                verdict = run_matchup(prof, conts, args.base_url,
                                      args.temperature,
                                      use_spotify=not args.no_spotify)
        except MatchupError as e:
            print(f"❌ {e}", file=sys.stderr)
            ans = _ask_line("   r = retry, s = skip, q = quit: ")
            if ans is None:
                raise KeyboardInterrupt
            ans = ans.lower()
            if ans in ("r", "retry", ""):
                continue          # leave pending → retry next loop iteration
            status = "skipped" if ans == "s" else "aborted"
            verdict = str(e)
            if ans in ("q", "quit"):
                break
        except KeyboardInterrupt:
            print("\n(interrupted — progress saved)")
            break

        state["matchups"] = [m for m in state["matchups"] if m["key"] != k]
        state["matchups"].append({
            "key": k, "profile": prof["name"],
            "contestants": [{"label": c.label, "model": c.model, "agent": c.agent}
                            for c in conts],
            "status": status, "verdict": verdict,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        save_state(state)

        if status == "done" and i < len(pending):
            choice = _pause()
            if choice == "quit":
                break
            if choice == "skip":
                skip_next = True

    done_n = sum(1 for m in state["matchups"] if m["status"] == "done")
    print(f"\n🏁 Session {name}: {done_n} done, "
          f"{len(matchups) - sum(1 for m in state['matchups'] if m['status'] == 'done')} "
          f"remaining in plan. Re-run the same command to continue.")
    print("   standings: python eval/taste.py stats")


# ── status ───────────────────────────────────────────────────────────────────

def run_status() -> None:
    if not SESSIONS_DIR.exists():
        print("No duel sessions yet.")
        return
    any_shown = False
    for p in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            s = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        ms = s.get("matchups", [])
        done = sum(1 for m in ms if m.get("status") == "done")
        skipped = sum(1 for m in ms if m.get("status") == "skipped")
        print(f"\n🏟️  {s.get('name')}  ({s.get('cmd')}, created {s.get('created', '?')})")
        print(f"   {done} done · {skipped} skipped · {len(ms) - done - skipped} in flight/aborted")
        for m in ms[-5:]:
            print(f"     [{m.get('status', '?'):<7}] {m.get('profile', '?'):<16} "
                  f"{m.get('verdict', '')}")
        any_shown = True
    if not any_shown:
        print("No duel sessions yet.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _profiles_from_arg(spec: str) -> list[dict]:
    names = [x.strip() for x in spec.split(",") if x.strip()]
    return t._pick_profiles(types.SimpleNamespace(profile=names))


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profiles", default=",".join(DEFAULT_PROFILES),
                        help="Comma-separated fixture profiles (default: "
                             + ",".join(DEFAULT_PROFILES) + ")")
    common.add_argument("--base-url",
                        default=os.getenv("LLM_URL") or os.getenv("OLLAMA_URL")
                        or "http://localhost:8092")
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--no-spotify", action="store_true",
                        help="Skip Spotify verification (faster).")
    common.add_argument("--session", default=None,
                        help="Session name for resume (default: auto from args).")
    common.add_argument("--dry-run", action="store_true",
                        help="Print the queue and exit — nothing is generated.")

    parser = argparse.ArgumentParser(
        description="Batch interactive duels with resume. See module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    p_agents = sub.add_parser("agents", parents=[common],
                              help="same model, two agents — which playbook wins?")
    p_agents.add_argument("--models", default=",".join(MEDIUM_MODELS),
                          help="Comma-separated models (default: the 4 medium models)")
    p_agents.add_argument("--agent-a", default="artist-then-album@v1")
    p_agents.add_argument("--agent-b", default="album-first@v1")

    p_models = sub.add_parser("models", parents=[common],
                              help="models vs models, each with its preferred agent")
    p_models.add_argument("--contestants", required=True,
                          help='e.g. "Qwen3.8-27B-Q8_0:album-first@v1,Mistral-7B-Instruct-v0.3-Q8_0" '
                               "(omit :agent to use the selected agent)")
    p_models.add_argument("--models", default="", help=argparse.SUPPRESS)  # for session key

    sub.add_parser("status", help="show sessions and progress")

    args = parser.parse_args()
    if args.cmd == "status":
        run_status()
        return 0
    if args.cmd is None:
        parser.print_help()
        return 1
    if not sys.stdin.isatty():
        print("⚠️  stdin is not a TTY — use a real terminal for interactive judging.",
              file=sys.stderr)

    profiles = _profiles_from_arg(args.profiles)
    selected = afrec_agents.selected_agent().tag

    if args.cmd == "agents":
        try:
            a = afrec_agents.get_agent(args.agent_a).tag
            b = afrec_agents.get_agent(args.agent_b).tag
        except ValueError as e:
            parser.error(str(e))
        if a == b:
            parser.error("--agent-a and --agent-b must differ")
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models:
            parser.error("--models must name at least one model")
        matchups = agents_matchups(models, profiles, a, b)
    else:
        contestants = parse_contestants(args.contestants, selected)
        matchups = models_matchups(contestants, profiles)

    run_session(args.cmd, matchups, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
