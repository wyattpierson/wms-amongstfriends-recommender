"""Orchestration facade + shared machinery for the recommendation agents.

This module is the "engine" the evaluator measures, but it is deliberately a
thin FACADE: the actual strategy (which prompts, how many LLM calls, what
retry policy) lives in one module per agent:

    afrec/agent_artist_then_album.py   →  artist-then-album@v1 (default)
    afrec/agent_album_first.py         →  album-first@v1

`pipeline.run()` picks the agent — an explicit one, or the SELECTED one
(`AFREC_AGENT` in .env, else the registry default) — and dispatches.
Everything here is free of Firebase and of backend specifics: it takes any
`BaseLLM` and any optional verifier callable, and produces a structured
`RunOutcome` that can be scored, logged, or compared across models, agents
and prompt revisions.

Nothing in here `sys.exit`s or raises for *expected* failure modes (parse
errors, no artists, no albums) — those are recorded on the `RunOutcome` so a
batch evaluation can keep going after one bad response.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from .llm import BaseLLM, LLMError, ResponseParseError
from .reviews import ACTIVE_TYPE
from . import spotify as spotify_mod

# A verifier callable: (artist, album) -> (found: bool, canonical_album: str|None, url: str|None)
# Verifier returns (found, canonical, url) — or (found, canonical, url, release_year)
# when the source can supply it (Spotify can). Older 3-tuple verifiers (tests,
# fakes) keep working: the pipeline unpacks the 4th element defensively.
Verifier = Callable[[str, str], tuple]


class VerificationGuard:
    """Circuit breaker around the injected catalog verifier.

    Verifier failures are INFRASTRUCTURE failures, not model failures: a
    Spotify 429 (QUOTA_EXCEEDED) does not mean "the album doesn't exist".
    Before this guard, every verifier exception was treated as a genuine
    miss, so a single quota exhaustion silently dropped ALL candidates —
    an all-zero run indistinguishable from a model failure (2026-10-04:
    gemma's "0 recs" was half Spotify 429, masked as hallucination).

    Now: on the first verifier failure the guard LATCHES. For the rest of
    the run, verification is treated as UNAVAILABLE — candidates pass
    through and are accepted as-is (found=True so every agent call site
    keeps them, the same as its verifier=None path), and the registry
    stamps outcome.verification_error afterward so the eval can flag the
    run (real r1 → n/a, quality → confidence-only branch) instead of
    recording a fake 0. Recorded recs are flipped back to verified=False
    so the data stays honest.
    """

    def __init__(self, verifier: Verifier, log: Callable | None = None):
        self._verifier = verifier
        self._log = log or (lambda _s: None)
        self.error: str | None = None

    @property
    def unavailable(self) -> bool:
        return self.error is not None

    def _passthrough(self, artist: str, album: str) -> tuple:
        # "Accept as-is": found, no canonical name, plain search link. No
        # 4th element (no year) — same shape the older verifiers return.
        return (True, None, spotify_mod.search_url(artist, album))

    def __call__(self, artist: str, album: str) -> tuple:
        if self.unavailable:
            return self._passthrough(artist, album)
        try:
            return self._verifier(artist, album)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}".replace("\n", " ").strip()
            msg = (f"⚠ VERIFICATION UNAVAILABLE — {self.error}. "
                   f"Remaining picks pass through UNVERIFIED for this run "
                   f"(they are NOT confirmed on the catalog, and the eval will flag this run).")
            print(msg, file=sys.stderr)
            self._log(msg)
            return self._passthrough(artist, album)


@dataclass
class StageResult:
    """The result of one LLM stage (artists / albums / albums_first / retry)."""
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
    release_year: int | None = None   # from the catalog (Spotify release_date), when available
    # The model's OWN read of the pick (self-reported, display-only). These are
    # NEVER scored on — a model grading its own taste is circular. Their jobs:
    # (1) make untagged recs legible in the TSV/leaderboard/site, (2) feed the
    # cross-model consensus that the tag-harvest step (eval/harvest.py) turns
    # into human-reviewed tag proposals.
    self_genre: str | None = None
    self_era: str | None = None
    role: str | None = None      # pool-map agent: core / adjacent / stretch
    lane: str | None = None      # pool-map agent: lane/thread the rec sits in
    round: int = 1               # 1 = initial, 2 = retry replacement


@dataclass
class RunOutcome:
    """Everything about one (profile × model × agent × run) that the eval scores."""
    ok: bool = False
    agent: str = ""            # which agent version produced this (e.g. "artist-then-album@v1")
    suggested_artists: list = field(default_factory=list)   # artists surfaced by the agent (filtered)
    repeated_artist_violations: list = field(default_factory=list)  # already-reviewed artists the model suggested anyway
    pool_map: dict | None = None         # pool-map agent: Call 0 output (the brief)
    critic_rejected: int = 0             # pool-map agent: recs the critic rejected
    critic_rescued: int = 0              # pool-map agent: critic-retry recs that verified
    recs: list = field(default_factory=list)                  # list[Rec]
    stages: list = field(default_factory=list)                # list[StageResult]
    total_candidates: int = 0   # recommendation dicts produced by the agent's generation call(s)
    round1_found: int = 0       # verified in the first pass
    round1_missing: int = 0     # failed verification in the first pass
    retry_found: int = 0        # rescued by the retry call
    retry_missing: int = 0      # failed verification even on retry (dropped)
    verified_count: int = 0
    failed_count: int = 0
    verification_error: str | None = None   # set when the catalog verifier died mid-run (e.g. Spotify 429); recs are unverified passthroughs
    enrichment_error: str | None = None     # set when the enricher latched mid-run (quota/API); facts were cache-only from then on — the run is LESS informed, never failed
    total_latency_s: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stages"] = [asdict(s) for s in self.stages]
        d["recs"] = [asdict(r) for r in self.recs]
        return d


# ── Shared machinery used by every agent ─────────────────────────────────────

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
    last_parse_err: ResponseParseError | None = None
    for _attempt in (1, 2):
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
            # One retry: thinking models (Qwen3-style) occasionally finish with
            # the answer buried in reasoning_content and an EMPTY content field,
            # which is a sampling flake — a second draw usually recovers it.
            last_parse_err = e
            continue
        except LLMError as e:
            return StageResult(stage=stage, ok=False, prompt=prompt, error=str(e), model=getattr(llm, "name", None), latency_s=time.perf_counter() - t0)
    return StageResult(stage=stage, ok=False, prompt=prompt, raw=getattr(last_parse_err, "raw", None),
                       error=f"parse error (after 2 attempts): {last_parse_err}")


def _clean_self_report(value: Any) -> str | None:
    """Light sanitize for model self-reports (genre/era). Display + harvest input
    only, but keep it tight: strip, collapse whitespace, drop control chars,
    cap length. Returns None when nothing usable remains."""
    if value is None:
        return None
    s = str(value).strip()
    s = "".join(ch for ch in s if ch.isprintable() and ch != "\x7f")
    s = " ".join(s.split())
    s = s[:60].strip()
    return s or None


def _rec_dict_to_rec(d: Any, verified: bool, link: str, canonical: str | None, rnd: int,
                     release_year: int | None = None) -> Rec | None:
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
        release_year=release_year,
        self_genre=_clean_self_report(d.get("genre")),
        self_era=_clean_self_report(d.get("era")),
        role=_clean_self_report(d.get("role")),
        lane=_clean_self_report(d.get("lane")),
        round=rnd,
    )


def verify_and_retry(
    reviews: list[dict],
    llm: BaseLLM,
    candidates: list[dict],
    outcome: RunOutcome,
    *,
    verifier: Verifier | None,
    temperature: float,
    num_predict: int = 1500,
    extra_options: dict | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Rec]:
    """
    Shared post-generation step, used by every agent:

    1. Verify each candidate (if a verifier is provided; otherwise all pass
       unverified with a search URL as the link).
    2. For the failures, run ONE reflection retry call and keep the
       replacements that verify.

    Appends to `outcome` (stages, round1_*, retry_*, counts, latency) and
    returns the final rec list.
    """
    from . import prompts as _prompts  # local import: avoid import-cycle noise

    log = log or (lambda _s: None)

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
        return outcome.recs

    def _verify_one(d: dict, rnd: int) -> tuple[Rec | None, bool]:
        artist = str(d.get("artist", "")).strip()
        album = str(d.get("album", "")).strip()
        if not artist or not album:
            return None, False
        try:
            result = verifier(artist, album)
        except Exception as e:
            log(f"  [verify] spotify error for {artist} — {album}: {e}")
            result = (False, None, None)
        found, canonical, url = result[0], result[1], result[2]
        release_year = result[3] if len(result) > 3 else None
        canonical = canonical or album
        link = url or spotify_mod.search_url(artist, canonical)
        rec = _rec_dict_to_rec(d, verified=bool(found), link=link, canonical=canonical, rnd=rnd,
                               release_year=release_year)
        return rec, found

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
        prompt3 = _prompts.build_retry_prompt(
            reviews,
            verified=[asdict(r) for r in verified_recs],
            failed=failed_dicts,
            needed=needed,
        )
        s3 = _run_stage(llm, "retry", prompt3, expected=list, num_predict=num_predict,
                        temperature=temperature, extra_options=extra_options)
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
    return verified_recs


# ── The facade ───────────────────────────────────────────────────────────────

def run(
    reviews: list[dict],
    llm: BaseLLM,
    *,
    agent: "str | Any | None" = None,   # Agent instance, tag ("artist-then-album@v1"), bare id, or None → selected
    verifier: Verifier | None = None,
    temperature: float = 0.7,
    num_predict_artists: int = 512,
    num_predict_albums: int = 1500,
    content_type: str = ACTIVE_TYPE,     # accepted for backwards compatibility
    extra_options: dict | None = None,
    log: Callable[[str], None] | None = None,
) -> RunOutcome:
    """
    Run the recommendation for one user's reviews, via the given agent.

    `agent` may be an Agent instance, a full tag ("album-first@v1"), a bare
    id ("album-first" = latest version), or None — in which case the SELECTED
    agent is used (AFREC_AGENT in .env, else the registry default). The
    returned RunOutcome is stamped with the agent's tag.
    """
    from . import agents  # lazy: the agent modules import this one

    a = agents.resolve_agent(agent)
    return a.run(
        reviews, llm,
        verifier=verifier,
        temperature=temperature,
        num_predict_artists=num_predict_artists,
        num_predict_albums=num_predict_albums,
        extra_options=extra_options,
        log=log,
    )
