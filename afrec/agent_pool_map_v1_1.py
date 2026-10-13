"""Agent: pool-map v1.1 — the v1 flow with the Oct-2 eval findings fixed.

v1 (pool-map@v1) ran the 4-call flow on the Dolphin 8B across 7 fixture
profiles: quality 0.685, real r1 57%, structure 0.47, wallclock ~1035s —
worse than album-first@v1 on every axis except golden (+0.106 vs +0.020 on
small 3-good/2-bad seed sets, not trustworthy). The data pointed at four
specific failures; v1.1 fixes exactly those, keeping the same call skeleton
so the A/B stays honest (same flow, better-grounded prompts + deterministic
backstops):

1. EXISTENCE COLLAPSE (real r1 57% vs 83% for album-first).
   v1 forced "hit the plan's counts exactly" with a role and a lane per rec —
   on an 8B that pressure produces invented titles, and critic_retry was a
   THIRD constrained generation call. v1.1 discovery makes the counts GOALS,
   not quotas: a certain core pick beats an uncertain stretch.

2. THE CRITIC WAS DISABLED ON EVERY FIXTURE. All 7 pools are 3–11 albums,
   and v1's calibration ("small pool → be LENIENT", "reject at most 3",
   "on the fence → KEEP") plus the circuit breaker neutered it while it
   still cost a full call. The one case it mattered for (Thelonious Monk for
   a club profile) was a LANE VOID. v1.1 critic keeps exactly LANE VOID +
   REPEL COLLISION — narrow, judgeable, cite-a-numbered-review rules — and
   drops GENERIC / POOL SHAPE (an 8B cannot judge them reliably).

3. MAP OVERFITTING. v1's map invented lanes its tiny pool couldn't support
   (a jazz lane out of 6 dance reviews, which every downstream stage then
   followed). v1.1 map: lanes must rest on 2+ named pool albums, at most 3
   lanes, and small pools (<8 albums) get an explicit small-pool block
   (fewer lanes, no "strong" without 2+ 5-star known albums, core-heavy
   plan).

4. SHORT BATCHES, DUPLICATE ARTISTS, BROKEN METRICS. Critic rejections with
   failed rescues left 5–7 recs; the model sometimes suggested the same
   artist twice (Smino ×2); and outcome.suggested_artists was never set, so
   pool-map rows reported "new artists 0" / artist_coverage 0. v1.1 adds a
   deterministic artist dedupe, a final existence-first topup call when the
   batch is short, and sets suggested_artists.

The v1 module is untouched (published-version invariant): v1.1 reuses its
private helpers (_enrich_years, _usable_map, _critic_split, _verify_candidate,
_rec_brief, _map_summary) by import, and reuses build_prompt_pool_replace
for the role-targeted retry.
"""

import afrec.pipeline as pipeline
import afrec.prompts as prompts
from afrec.agent_pool_map import (
    _critic_split,
    _enrich_years,
    _map_summary,
    _rec_brief,
    _usable_map,
    _verify_candidate,
)
from afrec.agent_registry import _register

N_RECS = 8
ANALYSIS_TEMP = 0.3        # map + critic: stable analysis, not sampling
MAP_NUM_PREDICT = 2500     # the map is a big JSON object (reads + lanes + plan)
CRITIC_NUM_PREDICT = 1200
REPLACEMENT_SLACK = 2      # ask the role-retry for a few extra so misses don't leave holes
TOPUP_MAX = 6              # cap on one topup request


def _dedupe_artists(recs, log):
    """One rec per artist (keep the first). v1 let the same artist through
    twice (Smino ×2 in the underground_flo run)."""
    seen = set()
    out = []
    for r in recs:
        k = r.artist.lower().strip()
        if k in seen:
            log(f"  [dedupe] dropped second album by {r.artist} — {r.album}")
            continue
        seen.add(k)
        out.append(r)
    return out


def _topup(reviews, pmap, kept, outcome, llm, verifier, temperature,
           num_predict, extra_options, log):
    """Final backstop: one existence-first call to fill a short batch.
    Returns the (possibly longer) kept list."""
    needed = N_RECS - len(kept)
    if needed <= 0:
        return kept

    prompt = prompts.build_prompt_pool_topup(reviews, pmap, [_rec_brief(r) for r in kept], needed)
    s = pipeline._run_stage(
        llm, "topup", prompt, expected=list,
        num_predict=num_predict, temperature=temperature,
        extra_options=extra_options,
    )
    outcome.stages.append(s)
    outcome.total_latency_s += s.latency_s or 0.0
    if not (s.ok and isinstance(s.parsed, list)):
        log(f"  [topup] no usable topup (needed {needed}) — keeping the short list")
        return kept

    accepted = {(r.artist.lower().strip(), r.album.lower().strip()) for r in kept}
    known_artists = {r.artist.lower().strip() for r in kept}
    added = 0
    for d in s.parsed:
        if len(kept) >= N_RECS:
            break
        if not isinstance(d, dict):
            continue
        a = str(d.get("artist", "")).strip()
        b = str(d.get("album", "") or d.get("albumTitle", "")).strip()
        if not a or not b:
            continue
        if a.lower() in known_artists or (a.lower(), b.lower()) in accepted:
            continue
        rec = _verify_candidate(d, verifier, rnd=4)
        if rec is None:
            continue
        if verifier is not None and not rec.verified:
            outcome.retry_missing += 1
            log(f"  [topup] spotify not found: {rec.artist} — {rec.album}")
            continue
        kept.append(rec)
        accepted.add((rec.artist.lower(), rec.album.lower()))
        known_artists.add(rec.artist.lower())
        added += 1
    log(f"  [topup] added {added} (list now {len(kept)})")
    return kept


def run_pool_map_v1_1(reviews: list[dict], llm, *,
                      verifier=None, temperature: float = 0.7,
                      num_predict_artists: int = 512, num_predict_albums: int = 1500,
                      extra_options: dict | None = None,
                      log=None) -> pipeline.RunOutcome:
    """Pool-map v1.1: grounded map → existence-first discovery → narrow
    critic → role-targeted retry (with slack) → artist dedupe → topup."""
    log = log or (lambda _s: None)
    outcome = pipeline.RunOutcome()
    reviewed_artists = pipeline._new_artists(reviews)
    reviewed_albums = {(r["artist"].lower().strip(), r["album"].lower().strip()) for r in reviews}

    # ── 0a. deterministic catalog enrichment (only when a verifier exists) ──
    years: dict = {}
    if verifier is not None:
        years = _enrich_years(reviews, verifier, log)
        log(f"  [enrich] catalog years for {len(years)} of {len(reviewed_albums)} pool album(s)")

    # ── Call 0: grounded pool map (degrades gracefully) ─────────────────────
    prompt_map = prompts.build_prompt_pool_map_v1_1(reviews, catalog_years=years, n_recs=N_RECS)
    s_map = pipeline._run_stage(llm, "pool_map", prompt_map, expected=dict,
                                num_predict=MAP_NUM_PREDICT, temperature=ANALYSIS_TEMP,
                                extra_options=extra_options)
    outcome.stages.append(s_map)
    outcome.total_latency_s += s_map.latency_s or 0.0

    pmap = s_map.parsed if (s_map.ok and _usable_map(s_map.parsed)) else None
    if pmap is not None:
        outcome.pool_map = pmap
        log(f"  [call 0] pool map: {_map_summary(pmap)}")
    else:
        log("  [call 0] no usable pool map — degrading to unguided discovery")

    # ── Call 1: existence-first discovery ───────────────────────────────────
    prompt1 = prompts.build_prompt_pool_discovery_v1_1(reviews, pmap, n=N_RECS)
    s1 = pipeline._run_stage(llm, "pool_discovery", prompt1, expected=list,
                             num_predict=num_predict_albums, temperature=temperature,
                             extra_options=extra_options)
    outcome.stages.append(s1)
    outcome.total_latency_s += s1.latency_s or 0.0

    candidates = []
    if s1.ok and isinstance(s1.parsed, list):
        seen = set()
        for d in s1.parsed:
            if not isinstance(d, dict):
                continue
            a = str(d.get("artist", "")).strip()
            b = str(d.get("album", "") or d.get("albumTitle", "")).strip()
            if not a or not b:
                continue
            if a.lower() in reviewed_artists or (a.lower(), b.lower()) in reviewed_albums:
                outcome.repeated_artist_violations.append(a)
                log(f"  [discovery] filtered already-reviewed: {a} — {b}")
                continue
            k = (a.lower(), b.lower())
            if k in seen:
                continue
            seen.add(k)
            candidates.append(d)
    else:
        log(f"  [call 1] discovery failed: {s1.error}")
        return outcome

    outcome.total_candidates = len(candidates)
    log(f"  [discovery] {len(candidates)} unique candidates")

    # ── Spotify existence verification + shared reflection retry ────────────
    pipeline.verify_and_retry(reviews, llm, candidates, outcome,
                              verifier=verifier, temperature=temperature,
                              num_predict=num_predict_albums,
                              extra_options=extra_options, log=log)

    # ── Call 2 + 3: narrow critic vs the map, then role-targeted fill ───────
    if pmap is not None and outcome.recs:
        prompt_c = prompts.build_prompt_critic_v1_1(reviews, pmap, [_rec_brief(r) for r in outcome.recs])
        s_c = pipeline._run_stage(llm, "critic", prompt_c, expected=dict,
                                  num_predict=CRITIC_NUM_PREDICT, temperature=ANALYSIS_TEMP,
                                  extra_options=extra_options)
        outcome.stages.append(s_c)
        outcome.total_latency_s += s_c.latency_s or 0.0

        if s_c.ok and isinstance(s_c.parsed, dict):
            kept, rejected = _critic_split(outcome.recs, s_c.parsed)
            unfilled = [str(u).strip() for u in (s_c.parsed.get("unfilled_roles") or []) if str(u).strip()]
            if rejected and len(rejected) * 2 > len(outcome.recs):
                # Circuit breaker — unchanged from v1.
                log(f"  [critic] miscalibrated — rejected {len(rejected)}/{len(outcome.recs)}; ignoring verdicts, keeping all")
                kept, rejected, unfilled = list(outcome.recs), [], []
            outcome.critic_rejected = len(rejected)
            log(f"  [critic] kept {len(kept)}, rejected {len(rejected)}; unfilled roles: {unfilled or 'none'}")

            needed = N_RECS - len(kept)
            if needed > 0:
                # v1.1: ask with SLACK so Spotify misses don't leave holes;
                # the topup below catches whatever still falls short.
                ask = min(needed + REPLACEMENT_SLACK, TOPUP_MAX)
                prompt_r = prompts.build_prompt_pool_replace(
                    reviews, pmap,
                    [_rec_brief(r) for r in kept],
                    [(_rec_brief(r), reason) for r, reason in rejected],
                    unfilled,
                    ask,
                )
                s_r = pipeline._run_stage(llm, "critic_retry", prompt_r, expected=list,
                                          num_predict=num_predict_albums, temperature=temperature,
                                          extra_options=extra_options)
                outcome.stages.append(s_r)
                outcome.total_latency_s += s_r.latency_s or 0.0

                if s_r.ok and isinstance(s_r.parsed, list):
                    accepted = {(r.artist.lower().strip(), r.album.lower().strip()) for r in kept}
                    known_artists = {r.artist.lower().strip() for r in kept}
                    rescued = 0
                    for d in s_r.parsed:
                        if len(kept) >= N_RECS:
                            break
                        if not isinstance(d, dict):
                            continue
                        a = str(d.get("artist", "")).strip()
                        b = str(d.get("album", "") or d.get("albumTitle", "")).strip()
                        if not a or not b:
                            continue
                        if a.lower() in reviewed_artists or (a.lower(), b.lower()) in reviewed_albums:
                            outcome.repeated_artist_violations.append(a)
                            continue
                        if a.lower() in known_artists:
                            continue
                        rec = _verify_candidate(d, verifier, rnd=3)
                        if rec is None:
                            continue
                        if verifier is not None and not rec.verified:
                            outcome.retry_missing += 1
                            log(f"  [critic_retry] spotify not found: {rec.artist} — {rec.album}")
                            continue
                        k = (rec.artist.lower(), rec.album.lower())
                        if k in accepted:
                            continue
                        accepted.add(k)
                        known_artists.add(rec.artist.lower())
                        kept.append(rec)
                        rescued += 1
                    outcome.critic_rescued = rescued
                    log(f"  [critic_retry] rescued {rescued} replacement(s)")

            outcome.recs = kept
        else:
            log("  [critic] unusable verdicts — keeping all verified recs")

    # ── Backstops: artist dedupe, then topup if still short ─────────────────
    outcome.recs = _dedupe_artists(outcome.recs, log)
    outcome.recs = _topup(reviews, pmap, outcome.recs, outcome, llm, verifier,
                          temperature, num_predict_albums, extra_options, log)

    # ── Finalize (v1 never set suggested_artists — the "new artists 0" bug) ──
    outcome.suggested_artists = [r.artist for r in outcome.recs]
    outcome.verified_count = sum(1 for r in outcome.recs if r.verified)
    if not outcome.recs:
        outcome.ok = False
    return outcome


_register(
    id="pool-map", version="v1.1", name="pool-map v1.1 (grounded map, existence-first, narrow critic)",
    summary=(
        "v1 flow with the Oct-2 eval fixes: grounded pool map (lanes must rest on 2+ pool "
        "albums; small-pool block), existence-first discovery (plan counts are goals, not "
        "quotas), a narrow critic (LANE VOID + REPEL COLLISION only), role-retry with slack, "
        "deterministic artist dedupe, a topup call for short batches, and suggested_artists "
        "set so the new-artists metric is real"
    ),
    run=run_pool_map_v1_1,
    details={
        "calls": "0 pool_map (grounded) → 1 discovery (existence-first) → 2 critic (narrow) → 3 critic_retry (with slack) → topup only if short",
        "dedupe": "deterministic: one rec per artist (first wins)",
        "backstops": "circuit breaker (v1) + topup call + suggested_artists set",
        "prompt family": "prompts.build_prompt_pool_map_v1_1 / _pool_discovery_v1_1 / _critic_v1_1 / _pool_topup (+ v1's _pool_replace)",
    },
)
