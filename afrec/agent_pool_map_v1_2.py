"""Agent: pool-map v1.2 — v1.1 with catalog FACTS in the map call.

v1.1 (pool-map@v1.1) anchored the pool map with a bare release year per pool
album, fetched with one Spotify verify() per album (_enrich_years — capped at
ENRICH_CAP). v1.2 renders the FULL enrichment-layer fact for each pool album
in Call 0 (the taste map): year + ≤3 listener tags + sound measurements
(E/D/V, bpm, key) via CatalogFact.catalog_line() — see afrec/enrich.py and
ENRICHMENT_DESIGN.md §4.2.

That is the ONLY delta. The call flow, discovery/critic/topup prompts,
backstops, and all metrics are v1.1's — imported from the v1.1 module so the
A/B against v1.1 stays honest (same playbook, richer map input):

  v1.1:  (catalog: 1998)
  v1.2:  (catalog: 1998 · E.7 D.6 V.5 · 123bpm · Dmaj · melodic house, dance)

How facts get here: the registry (afrec/agent_registry.py) accepts an
injected Enricher (or a ready catalog_facts dict) and passes `catalog_facts`
down — but only to agent versions whose run() accepts the kwarg (this one
does; v1.1's doesn't, so a published version runs byte-identically without
it). When no facts are available (no keys, empty cache, latch), v1.2 falls
back to v1.1's _enrich_years path — enrichment can only make a run less
informed, never fail it.

Year dedupe: when facts are present, years are read off the facts and NO
per-album Spotify verify() is spent on the pool — the _enrich_years calls
are skipped entirely.

Downstream calls inherit the facts only through the map TEXT (the discovery
prompt embeds the map summary) — their templates are v1.1's, unchanged.
"""

import afrec.pipeline as pipeline
import afrec.prompts as prompts
from afrec.agent_pool_map import _enrich_years
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
    _map_summary,
    _rec_brief,
    _usable_map,
    _verify_candidate,
)
from afrec.agent_registry import _register


def _years_from_facts(catalog_facts: dict) -> dict:
    """{name_key: CatalogFact} → v1-style {(artist, album): year}.

    The prompts layer's year fallback is tuple-keyed (the _enrich_years
    convention); facts are name_key-string-keyed. Reviews that HAVE a fact
    never reach the fallback (catalog_line wins), but consistent keys keep
    the two paths composable.
    """
    out = {}
    for k, f in catalog_facts.items():
        if f.year:
            a, _, b = k.partition("|")
            out[(a, b)] = f.year
    return out


def run_pool_map_v1_2(reviews: list[dict], llm, *,
                      verifier=None, temperature: float = 0.7,
                      num_predict_artists: int = 512, num_predict_albums: int = 1500,
                      extra_options: dict | None = None,
                      log=None,
                      catalog_facts: dict | None = None) -> pipeline.RunOutcome:
    """Pool-map v1.2: v1.1's flow with Call 0 rendered from catalog facts.

    catalog_facts: {name_key(artist, album): CatalogFact} from the
    enrichment layer, injected by the registry. None/empty → identical to
    v1.1 (including its per-album Spotify year lookups).
    """
    log = log or (lambda _s: None)
    outcome = pipeline.RunOutcome()
    reviewed_artists = pipeline._new_artists(reviews)
    reviewed_albums = {(r["artist"].lower().strip(), r["album"].lower().strip()) for r in reviews}

    # ── 0a. THE DELTA: facts win; bare Spotify year lookups are the fallback ──
    years: dict = {}
    if catalog_facts:
        years = _years_from_facts(catalog_facts)
        log(f"  [enrich] catalog facts for {len(catalog_facts)} pool album(s) "
            f"({len(years)} with year) — skipping Spotify year lookups")
    elif verifier is not None:
        years = _enrich_years(reviews, verifier, log)
        log(f"  [enrich] catalog years for {len(years)} of {len(reviewed_albums)} pool album(s)")

    # ── Call 0: grounded pool map with facts (degrades gracefully) ──────────
    prompt_map = prompts.build_prompt_pool_map_v1_1(
        reviews, catalog_years=years, n_recs=N_RECS,
        catalog_facts=catalog_facts, version_label="v1.2",
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
    id="pool-map", version="v1.2", name="pool-map v1.2 (v1.1 + catalog facts in the map)",
    summary=(
        "v1.1 with the map call enriched: each pool album's line carries the "
        "enrichment layer's catalog fact (year + tags + sound measurements) "
        "instead of a bare Spotify-looked-up year; downstream calls, backstops "
        "and metrics are v1.1's. Without an enricher it runs exactly as v1.1"
    ),
    run=run_pool_map_v1_2,
    details={
        "calls": "same as v1.1: 0 pool_map (FACTS) → 1 discovery → 2 critic → 3 critic_retry → topup only if short",
        "delta vs v1.1": "Call 0 only: catalog_facts rendered per pool album (CatalogFact.catalog_line); Spotify year lookups skipped when facts present",
        "degrade": "no facts (no keys / empty cache / latch) → v1.1's _enrich_years path; enrichment can only make a run less informed, never fail it",
    },
)
