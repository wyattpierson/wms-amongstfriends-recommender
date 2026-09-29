"""Agent: artist-then-album — the two-call baseline strategy.

Call 1: taste profile → 8–10 NEW artists (no albums yet).
Call 2: per artist → one REAL album each (single-artist recall is the most
        reliable kind of factual recall a language model does).
Retry : the Spotify-unverified albums get one reflection-based replacement call.

Rationale: asking for albums *per confirmed artist* avoids the classic
hallucination mode — plausible-sounding album titles glued to the wrong
artist. The cost is a second LLM call and a taste→artists→albums detour.
The A/B counterpart is `album-first` (afrec/agent_album_first.py).

This module only implements the strategy. Shared machinery (RunOutcome,
stage runner, verification, retry) lives in afrec.pipeline; registration
lives in afrec.agents.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Callable

from . import pipeline, prompts
from .agent_registry import _register, add_alias
from .llm import BaseLLM


def run_artist_then_album(
    reviews: list[dict],
    llm: BaseLLM,
    *,
    verifier: pipeline.Verifier | None = None,
    temperature: float = 0.7,
    num_predict_artists: int = 512,
    num_predict_albums: int = 1500,
    extra_options: dict | None = None,   # e.g. {"seed": 42}
    log: Callable[[str], None] | None = None,
) -> pipeline.RunOutcome:
    """Run the two-call artist→album flow. See module docstring."""
    log = log or (lambda _s: None)
    outcome = pipeline.RunOutcome()
    reviewed_artists = pipeline._new_artists(reviews)

    # ── Call 1: artist discovery ──────────────────────────────────────────────
    prompt1 = prompts.build_prompt_artists(reviews)
    s1 = pipeline._run_stage(llm, "artists", prompt1, expected=list,
                             num_predict=num_predict_artists,
                             temperature=temperature, extra_options=extra_options)
    outcome.stages.append(s1)
    outcome.total_latency_s += s1.latency_s or 0.0

    if not s1.ok:
        outcome.ok = False
        return outcome

    raw_list = s1.parsed if isinstance(s1.parsed, list) else []
    artists: list[str] = []
    for a in raw_list:
        if not isinstance(a, str) or not a.strip():
            continue
        name = a.strip()
        if name.lower() in reviewed_artists:
            outcome.repeated_artist_violations.append(name)
            continue
        if name.lower() in {x.lower() for x in artists}:
            continue
        artists.append(name)

    outcome.suggested_artists = artists
    log(f"  [call 1] {len(artists)} new artist(s), {len(outcome.repeated_artist_violations)} repeated-despite-instruction")
    if not artists:
        outcome.ok = False
        return outcome

    # ── Call 2: album recall ──────────────────────────────────────────────────
    prompt2 = prompts.build_prompt_albums(artists, reviews)
    s2 = pipeline._run_stage(llm, "albums", prompt2, expected=list,
                             num_predict=num_predict_albums,
                             temperature=temperature, extra_options=extra_options)
    outcome.stages.append(s2)
    outcome.total_latency_s += s2.latency_s or 0.0

    if not s2.ok:
        outcome.ok = False
        return outcome

    rec_dicts = s2.parsed if isinstance(s2.parsed, list) else []
    candidates = [d for d in rec_dicts if isinstance(d, dict) and (d.get("artist") or d.get("album"))]
    outcome.total_candidates = len(candidates)

    if not candidates:
        outcome.ok = False
        outcome.stages[-1].ok = False
        outcome.stages[-1].error = "no valid recommendation objects returned"
        return outcome

    # ── Verification + one reflection retry (shared machinery) ───────────────
    pipeline.verify_and_retry(
        reviews, llm, candidates, outcome,
        verifier=verifier, temperature=temperature,
        num_predict=num_predict_albums, extra_options=extra_options, log=log,
    )
    return outcome


_register(
    id="artist-then-album",
    version="v1",
    name="artist-then-album (two-call baseline)",
    summary=(
        "Two-stage: call 1 picks 8–10 new artists from the taste profile, call 2 recalls "
        "one real album per artist; failed Spotify checks get one reflection-based retry call."
    ),
    details={
        "calls": "2 + optional retry",
        "artist count": "8–10 new artists",
        "albums per artist": "1",
        "retry policy": "1 reflection call for unverified albums",
        "prompt family": "prompts.build_prompt_artists / build_prompt_albums / build_retry_prompt",
        "verification": "Spotify artist+album search (if creds)",
    },
    run=run_artist_then_album,
    default=True,
)

# This agent used to be called "two-call" — keep old records/tags resolvable.
add_alias("two-call@v1", "artist-then-album@v1")
add_alias("two-call", "artist-then-album@v1")
