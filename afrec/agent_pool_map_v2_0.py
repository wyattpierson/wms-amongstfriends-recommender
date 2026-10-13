"""Agent: pool-map v2.0 — v1.2 with a COHERENCE-DRIVEN plan.

v1.2 (pool-map@v1.2) anchored the map with the FULL enrichment-layer fact per
pool album (year + ≤3 tags + sound measurements). v2.0 keeps that and changes
exactly one thing: the map call's "plan" step no longer leaves the
core/adjacent/stretch split to habit. It is set from the pool's measured range
(axes: genre_spread / era_spread / energy_range) — the more coherent the input,
the harder the plan constrains to in-lane picks, down to 0 stretch; a genuinely
diverse pool may explore. That is the whole point:

    a coherent input  →  a constricted, in-lane output (don't branch out of politeness)
    a diverse input   →  an exploratory output (the user IS eclectic)

That is the ONLY delta. The call flow, discovery/critic/topup prompts,
backstops, and all metrics are v1.2's — the run body is v1.2's with the map
call pointed at build_prompt_map_v2_0, and _years_from_facts imported from the
v1.2 module — so an A/B against v1.2 isolates this single change:

    "does branching that scales with input coherence improve quality?"

How facts get here: identical to v1.2 — the registry injects an Enricher (or a
ready catalog_facts dict) and passes `catalog_facts` down to run() because this
one accepts the kwarg. No facts (no keys / empty cache / latch) → v1.2's
_enrich_years path. Enrichment can only make a run less informed, never fail it.

Downstream calls inherit the coherence-driven plan only through the map TEXT
(the discovery prompt embeds the map summary and "hits the plan's counts
exactly") — their templates are v1.1's, unchanged. A plan of 0 stretch simply
yields 0 stretch picks; no downstream code special-cases it.
"""

import afrec.pipeline as pipeline
import afrec.prompts as prompts
from afrec.agent_pool_map_v1_1 import (
    ANALYSIS_TEMP,
    CRITIC_NUM_PREDICT,
    MAP_NUM_PREDICT,
    N_RECS,
    REPLACEMENT_SLACK,
    TOPUP_MAX,
    _dedupe_artists,
    _topup,
)
from afrec.agent_pool_map import (
    _critic_split,
    _enrich_years,
    _map_summary,
    _rec_brief,
    _usable_map,
    _verify_candidate,
)
from afrec.agent_pool_map_v1_2 import _years_from_facts
from afrec.agent_registry import _register


def run_pool_map_v2_0(reviews: list[dict], llm, *,
                      verifier=None, temperature: float = 0.7,
                      num_predict_artists: int = 512, num_predict_albums: int = 1500,
                      extra_options: dict | None = None,
                      log=None,
                      catalog_facts: dict | None = None) -> pipeline.RunOutcome:
    """Pool-map v2.0: v1.2's flow with a coherence-driven plan in Call 0.

    catalog_facts: {name_key(artist, album): CatalogFact} from the
    enrichment layer, injected by the registry. None/empty → identical to v1.2
    (including its per-album Spotify year lookups).
    """
    log = log or (lambda _s: None)
    outcome = pipeline.RunOutcome()
    reviewed_artists = pipeline._new_artists(reviews)
    reviewed_albums = {(r["artist"].lower().strip(), r["album"].lower().strip()) for r in reviews}

    # ── 0a. facts win; bare Spotify year lookups are the fallback (v1.2) ────
    years: dict = {}
    if catalog_facts:
        years = _years_from_facts(catalog_facts)
        log(f"  [enrich] catalog facts for {len(catalog_facts)} pool album(s) "
            f"({len(years)} with year) — skipping Spotify year lookups")
    elif verifier is not None:
        years = _enrich_years(reviews, verifier, log)
        log(f"  [enrich] catalog years for {len(years)} of {len(reviewed_albums)} pool album(s)")

    # ── Call 0: THE DELTA — coherence-driven plan (v2.0 map prompt) ─────────
    prompt_map = prompts.build_prompt_pool_map_v2_0(
        reviews, catalog_years=years, n_recs=N_RECS,
        catalog_facts=catalog_facts, version_label="v2.0",
    )
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

    # ── Call 1: existence-first discovery (v1.1, unchanged) ─────────────────
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

    # ── Call 2 + 3: narrow critic vs the map, then role-targeted fill (v1.1) ─
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

    # ── Backstops: artist dedupe, then topup if still short (v1.1) ──────────
    outcome.recs = _dedupe_artists(outcome.recs, log)
    outcome.recs = _topup(reviews, pmap, outcome.recs, outcome, llm, verifier,
                          temperature, num_predict_albums, extra_options, log)

    # ── Finalize (v1.1) ──
    outcome.suggested_artists = [r.artist for r in outcome.recs]
    outcome.verified_count = sum(1 for r in outcome.recs if r.verified)
    if not outcome.recs:
        outcome.ok = False
    return outcome


_register(
    id="pool-map", version="v2.0", name="pool-map v2.0 (v1.2 + coherence-driven plan)",
    summary=(
        "v1.2 (catalog facts in the map) plus a plan that scales with input "
        "coherence: a coherent pool (genre+era narrow) is served mostly in-lane, "
        "down to 0 stretch; a diverse pool may explore. Only the map call's "
        "plan step changes; flow/critic/topup/metrics are v1.2's. Without an "
        "enricher it runs exactly as v1.1"
    ),
    run=run_pool_map_v2_0,
    details={
        "calls": "same as v1.2: 0 pool_map (FACTS + coherence plan) → 1 discovery → 2 critic → 3 critic_retry → topup only if short",
        "delta vs v1.2": "Call 0 only: the core/adjacent/stretch plan is set from axes (range) — coherent→constricted (stretch 0), diverse→exploratory — via build_prompt_pool_map_v2_0",
        "degrade": "no facts (no keys / empty cache / latch) → v1.2's _enrich_years path; enrichment can only make a run less informed, never fail it",
    },
)
