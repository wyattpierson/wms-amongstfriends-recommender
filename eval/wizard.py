#!/usr/bin/env python3
"""Guided command builder. Keep runner-specific rules in build_commands()."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # shared .env loading, fixture discovery and registry

ROOT = ev.REPO_ROOT
ACTIONS = [
    ("eval", "Automated leaderboard — score models against saved profiles"),
    ("fast", "Fast experiment — one profile, skip Spotify"),
    ("csv", "Export recommendations to CSV"),
    ("label", "Judge individual recommendations — good / bad / unsure"),
    ("score", "Rate whole responses 1–5"),
    ("duel", "Compare models — same agent, rank their responses"),
    ("agents", "Compare two agents — resumable duels on each model"),
    ("models", "Compare models with a separate agent for each — resumable"),
    ("status", "Show saved duel-session progress"),
    ("stats", "Show scores, win rates and Elo"),
    ("pending", "Review the unsure / to-listen queue"),
    ("grade", "Grade individual recs from saved responses (no generation)"),
    ("site", "Build the local results website"),
]
GENERATION = {"eval", "fast", "csv", "label", "score", "duel", "agents", "models"}


def ask(prompt: str, default: str = "") -> str:
    value = input(f"{prompt}{' [' + default + ']' if default else ''}: ").strip()
    if value.lower() == "q":
        raise KeyboardInterrupt
    return value or default


def choose(title, options, defaults=(), minimum=1, maximum=None):
    """Portable checkbox list: numbered toggles, Enter accepts, q quits."""
    if len(options) < minimum:
        raise ValueError(f"{title}: need at least {minimum} available choices")
    selected = set(defaults)
    while True:
        limit = f"exactly {minimum}" if maximum == minimum else f"at least {minimum}"
        print(f"\n{title} — choose {limit}.")
        for i, (key, label) in enumerate(options, 1):
            print(f"  {'[x]' if key in selected else '[ ]'} {i:2}. {label}")
        raw = ask("Numbers toggle; all / none; Enter accepts; q quits")
        if not raw:
            result = [key for key, _ in options if key in selected]
            if len(result) >= minimum and (maximum is None or len(result) <= maximum):
                return result
            print(f"Select {limit}{f' (maximum {maximum})' if maximum else ''}.")
        elif raw.lower() in {"all", "none"}:
            selected = {key for key, _ in options} if raw.lower() == "all" else set()
        else:
            try:
                indices = [int(n) - 1 for n in raw.replace(",", " ").split()]
                if any(i < 0 or i >= len(options) for i in indices):
                    raise ValueError
            except ValueError:
                print("Use the displayed numbers, separated by spaces or commas.")
                continue
            if maximum == 1 and len(indices) == 1:
                selected = {options[indices[0]][0]}
            else:
                for i in indices:
                    selected.symmetric_difference_update({options[i][0]})


def yes(prompt, default=False):
    while True:
        raw = ask(prompt + " (y/n)", "y" if default else "n").lower()
        if raw in {"y", "yes", "n", "no"}:
            return raw in {"y", "yes"}


def positive(prompt, default):
    while True:
        raw = ask(prompt, str(default))
        try:
            if int(raw) > 0:
                return int(raw)
        except ValueError:
            pass
        print("Enter a positive whole number.")


def discover_models(base_url, backend):
    """Read the server catalog only; never load a model for discovery."""
    endpoint = "/api/tags" if backend == "ollama" else "/v1/models"
    with urlopen(base_url.rstrip("/") + endpoint, timeout=3) as response:
        data = json.load(response)
    rows, field = (data["models"], "name") if backend == "ollama" else (data["data"], "id")
    return sorted({row[field] for row in rows if isinstance(row.get(field), str) and row[field]})


def build_commands(action, *, profiles=(), models=(), agents=(), base_url="",
                   no_spotify=False, enrich=False, rounds=1, csv="recs.csv", judge=False, all_profiles=False):
    """Translate selections into argv arrays; no shell interpolation.

    evaluate.py supports one profile or all profiles in a single run.
    taste.py accepts repeated --profile;
    duels.py accepts comma-separated --profiles.
    """
    def command(script, *args):
        return [sys.executable, str(ROOT / "eval" / script), *args]

    if action not in GENERATION:
        if action == "site":
            return [command("site.py")]
        if action == "status":
            return [command("duels.py", "status")]
        return [command("taste.py", action, *(["--judge"] if action == "pending" and judge else []))]
    if (not profiles and not all_profiles) or not models or not agents:
        raise ValueError("Generation needs profiles, models and agents")
    if all_profiles and action != "eval":
        raise ValueError("All-profile mode is only for automated eval")
    if action in {"eval", "fast", "label", "csv"} and not all_profiles and len(profiles) != 1:
        raise ValueError("This workflow needs exactly one profile")
    if action == "label" and len(models) != 1:
        raise ValueError("Labeling needs exactly one model")
    if action in {"duel", "models"} and len(models) < 2:
        raise ValueError("Model duels need at least two models")
    expected = 2 if action == "agents" else len(models) if action == "models" else 1
    if len(agents) != expected or (action == "agents" and len(set(agents)) != 2):
        raise ValueError(f"This workflow needs {expected} agent selection(s)")
    if action == "models" and any(":" in model for model in models):
        raise ValueError("The batched contestant syntax cannot represent model IDs containing ':'. Use the same-agent model duel instead.")
    common = ["--base-url", base_url]
    if no_spotify or action == "fast":
        common += ["--no-spotify"]
    if enrich:
        if action in {"score", "duel"}:
            raise ValueError("taste.py does not support --enrich")
        common += ["--enrich"]
    if action in {"agents", "models"}:
        common += ["--profiles", ",".join(profiles)]
        if action == "agents":
            common += ["--models", ",".join(models), "--agent-a", agents[0], "--agent-b", agents[1]]
        else:
            common += ["--contestants", ",".join(f"{m}:{a}" for m, a in zip(models, agents))]
        return [command("duels.py", action, *common)]
    common += ["--models", ",".join(models), "--agent", agents[0]]
    if action in {"score", "duel"}:
        common += ["--rounds", str(rounds)]
        for profile in profiles:
            common += ["--profile", profile]
        return [command("taste.py", action, *common)]
    if action == "label":
        common += ["--interactive"]
    if action == "csv":
        common += ["--export-csv", csv]
    if all_profiles:
        return [command("evaluate.py", *common)]
    return [command("evaluate.py", *common, "--profile", profiles[0])]


def plan():
    print("\nEval launcher — Enter keeps checked defaults. q / Ctrl-C cancels.\n")
    action = choose("What do you want to do?", ACTIONS, ["eval"], maximum=1)[0]
    if action not in GENERATION:
        return build_commands(action, judge=action == "pending" and yes("Judge pending recommendations now?"))
    fixtures = ev.discover_profiles()
    options = [(p["name"], f"{p['name']} — {len(p['reviews'])} reviews; {p['description']}") for p in fixtures]
    all_profiles = action == "eval" and choose(
        "Leaderboard scope", [("all", "All profiles — one combined leaderboard"),
                              ("one", "One profile")], ["all"], maximum=1)[0] == "all"
    single = action in {"eval", "fast", "label", "csv"}
    defaults = [p["name"] for p in fixtures]
    if single:
        defaults = ["jazzy_hiphop"] if "jazzy_hiphop" in defaults else defaults[:1]
    profiles = [] if all_profiles else choose("Profiles", options, defaults, maximum=1 if single else None)
    base_url = ask("Model server URL", os.getenv("LLM_URL") or os.getenv("OLLAMA_URL") or "http://localhost:8092")
    default_model = os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3"
    try:
        available = discover_models(base_url, os.getenv("LLM_BACKEND", "llamacpp").lower())
        print("Model choices read from the server catalog (availability is determined by the server).")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Could not read model catalog ({type(exc).__name__}); use the configured default or enter IDs manually.")
        available = []
    available = list(dict.fromkeys([default_model, *available]))
    manual = ask("Additional model IDs, comma-separated (optional)")
    available = list(dict.fromkeys([*available, *(m.strip() for m in manual.split(",") if m.strip())]))
    models = choose("Models", [(m, m + (" (configured default)" if m == default_model else "")) for m in available],
                    [default_model], minimum=2 if action in {"duel", "models"} else 1,
                    maximum=1 if action == "label" else None)
    registry = ev.afrec_agents
    selected = registry.selected_agent().tag
    options = [(a.tag, f"{a.tag} — {a.summary}" + (" (selected default)" if a.tag == selected else "")) for a in registry.list_agents()]
    if action == "models":
        agents = [choose(f"Agent for {m}", options, [selected], maximum=1)[0] for m in models]
    else:
        count = 2 if action == "agents" else 1
        agents = choose("Agents (playbooks)", options, [selected], minimum=count, maximum=count)
    no_spotify = action == "fast" or not yes("Verify albums with Spotify (uses configured credentials)?", True)
    enrich = action not in {"score", "duel"} and yes("Enable catalog enrichment for agents that support it?")
    rounds = positive("Rounds per profile/model", 1) if action in {"score", "duel"} else 1
    csv = ask("CSV output path (relative to repository)", "recs.csv") if action == "csv" else "recs.csv"
    if action in {"agents", "models"}:
        print("Batched duels save progress. Re-running the printed command resumes the session.")
    return build_commands(action, profiles=profiles, models=models, agents=agents, base_url=base_url,
                          no_spotify=no_spotify, enrich=enrich, rounds=rounds, csv=csv, all_profiles=all_profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Build and print commands without executing them")
    args = parser.parse_args()
    try:
        commands = plan()
        print(f"\nWorking directory: {ROOT}\nCommands ({len(commands)}):")
        print("cd " + shlex.quote(str(ROOT)))
        for command in commands:
            print(shlex.join(command))
        if args.dry_run or not yes("Run these commands now?"):
            print("Commands printed only; nothing executed.")
            return 0
        for command in commands:
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                print(f"Runner exited with status {result.returncode}; stopping the remaining commands.")
                return result.returncode
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 130
    except ValueError as exc:
        print(f"\nCannot build command: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
