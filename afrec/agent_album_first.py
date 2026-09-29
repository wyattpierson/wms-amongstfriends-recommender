"""Agent: album-first — one-shot, direct album proposal.

ONE call: taste profile → 8 complete, specific, REAL (artist, album) picks.
Then the shared Spotify verification + one reflection retry (afrec.pipeline).

Rationale (what we're testing against artist-then-album):
  - The taste signal lives in the *albums* (the reviews are album reviews),
    so proposing the record up front lets the model reason from moods, eras
    and production instead of detouring through an artist list it must then
    recall one album for.
  - Half the generation calls → faster, cheaper, less room for the two
    stages to drift out of agreement with each other.
  - Room for bolder, more specific picks (cross-genre bridges, shared
    producers) that an artist-list intermediate step tends to sand down.
Expected weakness: no "this artist is definitely real" anchor, so first-pass
hallucination may be HIGHER — that's exactly what the `real r1` metric is
there to catch. If `real r1` drops a lot while quality holds, the retry call
(or a second verification pass) is the lever to pull in a future version.
"""

from __future__ import annotations

from typing import Callable

from . import pipeline, prompts
from .agent_registry import _register
from .llm import BaseLLM

N_ALBUMS = 8


def run_album_first(
    reviews: list[dict],
    llm: BaseLLM,
    *,
    verifier: pipeline.Verifier | None = None,
    temperature: float = 0.7,
    num_predict_artists: int = 512,        # unused; kept for the common agent signature
    num_predict_albums: int = 1500,
    extra_options: dict | None = None,
    log: Callable[[str], None] | None = None,
) -> pipeline.RunOutcome:
    """Run the one-shot album-first flow. See module docstring."""
    log = log or (lambda _s: None)
    outcome = pipeline.RunOutcome()
    reviewed_artists = pipeline._new_artists(reviews)
    reviewed_albums = {
        (r["artist"].lower().strip(), r["album"].lower().strip()) for r in reviews
    }

    # ── The single generation call ────────────────────────────────────────────
    prompt = prompts.build_prompt_albums_first(reviews, n=N_ALBUMS)
    s1 = pipeline._run_stage(llm, "albums_first", prompt, expected=list,
                             num_predict=num_predict_albums,
                             temperature=temperature, extra_options=extra_options)
    outcome.stages.append(s1)
    outcome.total_latency_s += s1.latency_s or 0.0

    if not s1.ok:
        outcome.ok = False
        return outcome

    rec_dicts = s1.parsed if isinstance(s1.parsed, list) else []
    candidates: list[dict] = []
    seen: set[str] = set()
    for d in rec_dicts:
        if not isinstance(d, dict) or not (d.get("artist") or d.get("album")):
            continue
        artist = str(d.get("artist", "")).strip()
        album = str(d.get("album", "") or d.get("albumTitle", "")).strip()
        if not artist or not album:
            continue
        if artist.lower() in reviewed_artists or (artist.lower(), album.lower()) in reviewed_albums:
            outcome.repeated_artist_violations.append(artist)
            continue
        key = f"{artist.lower()}|{album.lower()}"
        if key in seen:
            continue
        seen.add(key)
        candidates.append(d)

    outcome.suggested_artists = [str(d.get("artist", "")).strip() for d in candidates]
    outcome.total_candidates = len(candidates)
    log(f"  [call 1] {len(candidates)} album pick(s), {len(outcome.repeated_artist_violations)} repeated-despite-instruction")

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
    id="album-first",
    version="v1",
    name="album-first (one-shot)",
    summary=(
        "One call: the model proposes 8 complete real (artist, album) picks directly from the "
        "taste profile — no artist-list detour — then one shared reflection retry for "
        "Spotify-unverifiable albums."
    ),
    details={
        "calls": "1 + optional retry",
        "picks": f"{N_ALBUMS} direct (artist, album) pairs",
        "retry policy": "1 reflection call for unverified albums (shared with artist-then-album)",
        "prompt family": "prompts.build_prompt_albums_first / build_retry_prompt",
        "verification": "Spotify artist+album search (if creds)",
        "hypothesis": "album-level taste reasoning beats the artists→albums detour; watch real r1",
    },
    run=run_album_first,
)
