"""Orchestration: artists call → albums call → (Spotify verify) → retry.

This is the "engine" the evaluator measures. It is deliberately free of
Firebase and of Ollama-specifics: it takes any `BaseLLM` and any optional
verifier callable, and produces structured results that can be scored,
logged, or compared across models and prompt revisions.

The pipeline never `sys.exit`s and never raises for *expected* failure modes
(parse errors, no artists, no albums) — it records them on the `RunOutcome`
so a batch evaluation can keep going after one bad response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from .llm import BaseLLM, LLMError, ResponseParseError
from .prompts import build_prompt_artists, build_prompt_albums, build_retry_prompt
from .reviews import ACTIVE_TYPE
from . import spotify as spotify_mod

# A verifier callable: (artist, album) -> (found: bool, canonical_album: str|None, url: str|None)
Verifier = Callable[[str, str], tuple[bool, str | None, str | None]]


@dataclass
class StageResult:
    """The result of one LLM stage (artists / albums / retry)."""
    stage: str
    ok: bool
    prompt: str = ""
    raw: str | None = None
    parsed: Any = None
    error: str | None = None
    model: str | None = None
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class Rec:
    """One final recommendation after verification/retry."""
    artist: str
    album: str
    confidence: float = 0.0
    reason: str = ""
    link: str = ""
    verified: bool = False       # True if Spotify confirmed it (or no verifier)
    canonical_album: str | None = None
    round: int = 1               # 1 = initial, 2 = retry replacement


@dataclass
class RunOutcome:
    """Everything about one (profile × model × run) that the eval scores."""
    ok: bool = False
    suggested_artists: list = field(default_factory=list)   # new artists from call 1 (filtered)
    repeated_artist_violations: list = field(default_factory=list)  # artists the model repeated despite instructions
    recs: list = field(default_factory=list)                  # list[Rec]
    stages: list = field(default_factory=list)                # list[StageResult]
    total_candidates: int = 0   # recommendation dicts produced by call 2
    round1_found: int = 0       # verified in the first pass
    round1_missing: int = 0     # failed verification in the first pass
    retry_found: int = 0        # rescued by the retry call
    retry_missing: int = 0      # failed verification even on retry (dropped)
    verified_count: int = 0
    failed_count: int = 0
    total_latency_s: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stages"] = [asdict(s) for s in self.stages]
        d["recs"] = [asdict(r) for r in self.recs]
        return d


def _new_artists(reviews: list[dict]) -> set:
    return {r["artist"].lower().strip() for r in reviews}


def _run_stage(
    llm: BaseLLM,
    stage: str,
    prompt: str,
    expected: type,
    num_predict: int,
    temperature: float,
    extra_options: dict | None = None,
) -> StageResult:
    t0 = time.perf_counter()
    try:
        parsed, raw, meta = llm.generate_json(
            prompt, expected=expected, num_predict=num_predict,
            temperature=temperature, options=extra_options,
        )
        return StageResult(
            stage=stage, ok=True, prompt=prompt, raw=raw, parsed=parsed,
            model=meta.name, latency_s=meta.latency_s,
            prompt_tokens=meta.prompt_tokens, completion_tokens=meta.completion_tokens,
        )
    except ResponseParseError as e:
        return StageResult(stage=stage, ok=False, prompt=prompt, raw=getattr(e, "raw", None), error=f"parse error: {e}")
    except LLMError as e:
        return StageResult(stage=stage, ok=False, prompt=prompt, error=str(e), model=getattr(llm, "name", None), latency_s=time.perf_counter() - t0)


def _rec_dict_to_rec(d: Any, verified: bool, link: str, canonical: str | None, rnd: int) -> Rec | None:
    if not isinstance(d, dict):
        return None
    artist = str(d.get("artist", "")).strip()
    album = str(d.get("album", "") or d.get("albumTitle", "")).strip()
    if not artist or not album:
        return None
    try:
        confidence = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return Rec(
        artist=artist,
        album=album,
        confidence=confidence,
        reason=str(d.get("reason", "")),
        link=link,
        verified=verified,
        canonical_album=canonical,
        round=rnd,
    )


def run(
    reviews: list[dict],
    llm: BaseLLM,
    *,
    verifier: Verifier | None = None,
    temperature: float = 0.7,
    num_predict_artists: int = 512,
    num_predict_albums: int = 1500,
    content_type: str = ACTIVE_TYPE,
    extra_options: dict | None = None,   # e.g. {"seed": 42} — passed through to the LLM backend
    log: Callable[[str], None] | None = None,
) -> RunOutcome:
    """
    Run the full two-call recommendation flow for one user's reviews.

    Args:
        reviews: normalized reviews (see afrec.reviews.parse_reviews)
        llm: any BaseLLM
        verifier: optional Spotify verifier; None = skip verification
        temperature / num_predict_*: LLM sampling params
        log: optional callback for human-readable progress

    Returns a RunOutcome. `ok` is True if at least one usable rec survived.
    """
    log = log or (lambda _s: None)
    outcome = RunOutcome()
    reviewed_artists = _new_artists(reviews)

    # ── Call 1: artist discovery ──────────────────────────────────────────────
    prompt1 = build_prompt_artists(reviews)
    s1 = _run_stage(llm, "artists", prompt1, expected=list, num_predict=num_predict_artists, temperature=temperature, extra_options=extra_options)
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
    prompt2 = build_prompt_albums(artists, reviews)
    s2 = _run_stage(llm, "albums", prompt2, expected=list, num_predict=num_predict_albums, temperature=temperature, extra_options=extra_options)
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

    # ── Verification (if a verifier is provided) ──────────────────────────────
    if verifier is None:
        # Unverified path: all candidates pass, link = search URL.
        for d in candidates:
            artist = str(d.get("artist", "")).strip()
            album = str(d.get("album", "")).strip()
            rec = _rec_dict_to_rec(d, verified=False, link=spotify_mod.search_url(artist, album), canonical=None, rnd=1)
            if rec:
                outcome.recs.append(rec)
        outcome.round1_found = len(outcome.recs)
        outcome.ok = bool(outcome.recs)
        return outcome

    def _verify_one(d: dict, rnd: int) -> tuple[Rec | None, bool]:
        artist = str(d.get("artist", "")).strip()
        album = str(d.get("album", "")).strip()
        if not artist or not album:
            return None, False
        try:
            found, canonical, url = verifier(artist, album)
        except Exception as e:
            log(f"  [verify] spotify error for {artist} — {album}: {e}")
            found, canonical, url = False, None, None
        canonical = canonical or album
        link = url or spotify_mod.search_url(artist, canonical)
        rec = _rec_dict_to_rec(d, verified=bool(found), link=link, canonical=canonical, rnd=rnd)
        return rec, bool(found)

    verified_recs: list[Rec] = []
    failed_dicts: list[dict] = []

    for d in candidates:
        rec, found = _verify_one(d, rnd=1)
        if rec is None:
            continue
        if found:
            verified_recs.append(rec)
        else:
            failed_dicts.append(d)

    outcome.round1_found = len(verified_recs)
    outcome.round1_missing = len(failed_dicts)
    outcome.verified_count = len(verified_recs)
    outcome.failed_count = len(failed_dicts)

    # ── Retry: reflection on the failures ─────────────────────────────────────
    if failed_dicts:
        needed = len(failed_dicts)
        prompt3 = build_retry_prompt(
            reviews,
            verified=[asdict(r) for r in verified_recs],
            failed=failed_dicts,
            needed=needed,
        )
        s3 = _run_stage(llm, "retry", prompt3, expected=list, num_predict=num_predict_albums, temperature=temperature, extra_options=extra_options)
        outcome.stages.append(s3)
        outcome.total_latency_s += s3.latency_s or 0.0

        if s3.ok and isinstance(s3.parsed, list):
            accepted_pairs = {(r.artist.lower(), r.album.lower()) for r in verified_recs}
            for d in s3.parsed:
                if not isinstance(d, dict):
                    continue
                rec, found = _verify_one(d, rnd=2)
                if rec is None or not found:
                    outcome.retry_missing += 1
                    continue
                key = (rec.artist.lower(), rec.album.lower())
                if key in accepted_pairs:
                    continue
                accepted_pairs.add(key)
                verified_recs.append(rec)
                outcome.retry_found += 1

    outcome.recs = verified_recs
    outcome.verified_count = len(verified_recs)
    outcome.ok = bool(verified_recs)
    return outcome
