"""Agent: pool-map v2.1 — v1.2's facts + an EVIDENCE-BASED SHIP BAR.

The v2.0 A/B verdict: the coherence-driven PLAN was the wrong lever. Its
axis tiering turned the plan into quotas (coherent → "6–7 core") and 8Bs
pad quotas: lane-stretched filler (Tory Lanez as "core R&B" for an alt-R&B
listener), cross-lane overflow, and invented titles (jazzy_hiphop r1
88%→62%, underground_flo 88%→43% on Dolphin). Blind duels: v1.2 won 4–0.

v2.1 keeps v2.0's insight — the pool's range should constrain branching —
but reframes it as a bar-height rule instead of a count: the more coherent
the pool, the more each non-core pick must cite the listener's own words;
a slot that can't clear the bar goes UNFILLED instead of padded. The list
is allowed to come out short: the product ships ONE album at a time, so
the top of a ranked, gated list is what matters, not the tail.

Deltas vs v1.2 (three prompt builders + gate behavior; same 4-call
skeleton, same backstop shapes, same metrics):

1. CALL 0 (map): build_prompt_pool_map_v2_1 — the plan is a TARGET the
   ship bar may shrink (4/3/1 start), with the anti-reflex rule and the
   confidence evidence tiers stated here so every downstream call
   calibrates the same way.
2. CALL 1 (discovery): build_prompt_pool_discovery_v2_1 — reflex picks
   ship only on an explicit pool cue (confidence capped at 0.7 until
   then); adjacent/stretch must cite the specific cue in the reason.
3. CALL 2 (critic): build_prompt_critic_v2_1 — adds DEFENSE FAILURE
   (fame/right-genre is not a defense), drops the v1.1 "reject at most 3"
   cap, and RANKS its keeps best-first. The agent reorders the kept recs
   into critic rank — delivery order for the single-album product.
4. GATE: critic_retry fires only when the kept list is below MIN_SHIP
   (4) and asks for CORE only (a coherent pool is not refilled with
   adjacent/stretch the critic just cut); the final topup tops up to
   MIN_SHIP, not N_RECS. The >50%-rejection circuit breaker from v1.1 is
   kept — it is the only thing standing between an 8B critic and an empty
   list.
5. DEDUPE: artist keys are diacritic-folded (Jhené Aiko == Jhene Aiko) —
   v1.1's exact-match key let the same artist through under two spellings.

Everything else (enrichment, verification, reflection retry, metrics) is
v1.2's, imported from the v1.1/v1.2 modules so the A/B against v1.2 stays
isolated: does a bar-based gate + ranked delivery beat a quota-based plan?
"""

import unicodedata

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
from afrec.agent_pool_map_v1_1 import (
    ANALYSIS_TEMP,
    CRITIC_NUM_PREDICT,
    MAP_NUM_PREDICT,
    N_RECS,
    REPLACEMENT_SLACK,
    TOPUP_MAX,
)
from afrec.agent_pool_map_v1_2 import _years_from_facts
from afrec.agent_registry import _register

MIN_SHIP = 4   # soft floor: critic_retry and topup exist to reach this, never to fill to 8


def _norm_artist(a: str) -> str:
    """Diacritic-folded, lowercased artist key (Jhené Aiko == Jhene Aiko)."""
    a = unicodedata.normalize("NFKD", a)
    a = "".join(c for c in a if not unicodedata.combining(c))
    return a.lower().strip()


def _dedupe_artists(recs, log):
    """One rec per artist (keep the first — the list is already in critic
    rank, so 'first' is the better-ranked pick). v2.1 folds diacritics in
    the key; v1.1's exact-match key let Jhené/Jhene Aiko through twice."""
    seen = set()
    out = []
    for r in recs:
        k = _norm_artist(r.artist)
        if k in seen:
            log(f"  [dedupe] dropped second album by {r.artist} — {r.album}")
            continue
        seen.add(k)
        out.append(r)
    return out


def _topup_to_floor(reviews, pmap, kept, outcome, llm, verifier, temperature,
                    num_predict, extra_options, log, floor: int = MIN_SHIP):
    """v2.1 topup: fill a short list to the FLOOR, not to N_RECS.

    v1.1's _topup treats 8 as an obligation and backfills whatever the model
    volunteers — exactly the padding the ship bar is meant to stop. This
    version asks only for (floor - len(kept)) more picks and stops there.
    (Mirrors _topup's body; the v1.1 module is left untouched.)
    """
    needed = floor - len(kept)
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
    known_artists = {_norm_artist(r.artist) for r in kept}
    added = 0
    for d in s.parsed:
        if len(kept) >= floor:
            break
        if not isinstance(d, dict):
            continue
        a = str(d.get("artist", "")).strip()
        b = str(d.get("album", "") or d.get("albumTitle", "")).strip()
        if not a or not b:
            continue
        if _norm_artist(a) in known_artists or (a.lower(), b.lower()) in accepted:
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
        known_artists.add(_norm_artist(rec.artist))
        added += 1
    log(f"  [topup] added {added} (list now {len(kept)}, floor {floor})")
    return kept


def _critic_rank(kept: list, parsed: dict) -> list:
    """Reorder kept recs into the critic's delivery rank (best first).

    The v2.1 critic lists its keeps in the order it would ship them; this
    makes the final list match. Recs with no (matchable) keep verdict — the
    ones _critic_split keeps by default — sort last, in original order.
    """
    rank = {}
    for i, v in enumerate(parsed.get("verdicts") or []):
        if not isinstance(v, dict):
            continue
        if str(v.get("verdict", "")).lower().strip() != "keep":
            continue
        key = (str(v.get("artist", "")).lower().strip(), str(v.get("album", "")).lower().strip())
        rank.setdefault(key, i)
    kept.sort(key=lambda r: rank.get((r.artist.lower().strip(), r.album.lower().strip()), len(rank)))
    return kept


def run_pool_map_v2_1(reviews: list[dict], llm, *,
                      verifier=None, temperature: float = 0.7,
                      num_predict_artists: int = 512, num_predict_albums: int = 1500,
                      extra_options: dict | None = None,
                      log=None,
                      catalog_facts: dict | None = None) -> pipeline.RunOutcome:
    """Pool-map v2.1: v1.2's flow with the ship-bar prompts and the gate.

    catalog_facts: {name_key(artist, album): CatalogFact} from the
    enrichment layer, injected by the registry. None/empty → v1.1-style
    per-album Spotify year lookups (same fallback as v1.2).
    """
    log = log or (lambda _s: None)
    outcome = pipeline.RunOutcome()
    reviewed_artists = pipeline._new_artists(reviews)
    reviewed_albums = {(r["artist"].lower().strip(), r["album"].lower().strip()) for r in reviews}

    # ── 0a. Facts win; bare Spotify year lookups are the fallback (v1.2) ───
    years: dict = {}
    if catalog_facts:
        years = _years_from_facts(catalog_facts)
        log(f"  [enrich] catalog facts for {len(catalog_facts)} pool album(s) "
            f"({len(years)} with year) — skipping Spotify year lookups")
    elif verifier is not None:
        years = _enrich_years(reviews, verifier, log)
        log(f"  [enrich] catalog years for {len(years)} of {len(reviewed_albums)} pool album(s)")

    # ── Call 0: pool map with the SHIP BAR (v2.1) ──────────────────────────
    prompt_map = prompts.build_prompt_pool_map_v2_1(
        reviews, catalog_years=years, n_recs=N_RECS,
        catalog_facts=catalog_facts, version_label="v2.1",
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

    # ── Call 1: existence-first discovery, evidence-tuned (v2.1) ───────────
    prompt1 = prompts.build_prompt_pool_discovery_v2_1(reviews, pmap, n=N_RECS)
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

    # ── Spotify existence verification + shared reflection retry ───────────
    pipeline.verify_and_retry(reviews, llm, candidates, outcome,
                              verifier=verifier, temperature=temperature,
                              num_predict=num_predict_albums,
                              extra_options=extra_options, log=log)

    # ── Call 2 + 3: defense-gate critic (ranked), core-only refill (v2.1) ──
    if pmap is not None and outcome.recs:
        prompt_c = prompts.build_prompt_critic_v2_1(reviews, pmap, [_rec_brief(r) for r in outcome.recs])
        s_c = pipeline._run_stage(llm, "critic", prompt_c, expected=dict,
                                  num_predict=CRITIC_NUM_PREDICT, temperature=ANALYSIS_TEMP,
                                  extra_options=extra_options)
        outcome.stages.append(s_c)
        outcome.total_latency_s += s_c.latency_s or 0.0

        if s_c.ok and isinstance(s_c.parsed, dict):
            kept, rejected = _critic_split(outcome.recs, s_c.parsed)
            if rejected and len(rejected) * 2 > len(outcome.recs):
                # Circuit breaker (unchanged from v1.1): an 8B critic that
                # rejects more than half is miscalibrated — keep everything
                # rather than ship an empty list.
                log(f"  [critic] miscalibrated — rejected {len(rejected)}/{len(outcome.recs)}; ignoring verdicts, keeping all")
                kept, rejected = list(outcome.recs), []
            else:
                kept = _critic_rank(kept, s_c.parsed)   # v2.1: delivery order
            outcome.critic_rejected = len(rejected)
            log(f"  [critic] kept {len(kept)}, rejected {len(rejected)}")
            if kept:
                log(f"  [critic] top pick: {kept[0].artist} — {kept[0].album}")

            # v2.1 gate: refill only a list below the floor, and ask for
            # CORE only — the ship bar, not the critic's role bookkeeping,
            # decides whether adjacent/stretch come back.
            needed = MIN_SHIP - len(kept)
            if needed > 0:
                ask = min(needed + REPLACEMENT_SLACK, TOPUP_MAX)
                prompt_r = prompts.build_prompt_pool_replace(
                    reviews, pmap,
                    [_rec_brief(r) for r in kept],
                    [(_rec_brief(r), reason) for r, reason in rejected],
                    ["core"],
                    ask,
                )
                s_r = pipeline._run_stage(llm, "critic_retry", prompt_r, expected=list,
                                          num_predict=num_predict_albums, temperature=temperature,
                                          extra_options=extra_options)
                outcome.stages.append(s_r)
                outcome.total_latency_s += s_r.latency_s or 0.0

                if s_r.ok and isinstance(s_r.parsed, list):
                    accepted = {(r.artist.lower().strip(), r.album.lower().strip()) for r in kept}
                    known_artists = {_norm_artist(r.artist) for r in kept}
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
                        if _norm_artist(a) in known_artists:
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
                        known_artists.add(_norm_artist(rec.artist))
                        kept.append(rec)
                        rescued += 1
                    outcome.critic_rescued = rescued
                    log(f"  [critic_retry] rescued {rescued} core replacement(s)")

            outcome.recs = kept
        else:
            log("  [critic] unusable verdicts — keeping all verified recs")

    # ── Backstops: diacritic-folded dedupe, then topup to the FLOOR (v2.1) ─
    outcome.recs = _dedupe_artists(outcome.recs, log)
    outcome.recs = _topup_to_floor(reviews, pmap, outcome.recs, outcome, llm, verifier,
                                   temperature, num_predict_albums, extra_options, log)

    # ── Finalize (v1.1) ──
    outcome.suggested_artists = [r.artist for r in outcome.recs]
    outcome.verified_count = sum(1 for r in outcome.recs if r.verified)
    if not outcome.recs:
        outcome.ok = False
    return outcome


_register(
    id="pool-map", version="v2.1", name="pool-map v2.1 (v1.2 + evidence-based ship bar)",
    summary=(
        "v1.2 (facts) with the v2.0 quota-tiers replaced by a bar-height rule: "
        "the more coherent the pool, the more each non-core pick must cite the "
        "listener's own words; unfilled beats padded. Critic adds a positive "
        "defense test and ranks its keeps; refills are core-only and only below "
        "the ship floor of 4. Ships fewer than 8 when nothing better clears the bar."
    ),
    run=run_pool_map_v2_1,
    details={
        "calls": "same as v1.2: 0 pool_map (SHIP BAR) → 1 discovery (evidence) → 2 critic (defense + rank) → 3 critic_retry (core-only, below floor) → topup (to floor)",
        "delta vs v1.2": "three prompt builders (map v2.1, discovery v2.1, critic v2.1) + gate: core-only refill below MIN_SHIP=4, topup to floor, critic-rank delivery order, diacritic-folded dedupe",
        "degrade": "no facts → v1.1-style Spotify year lookups; critic over-rejects (>50%) → circuit breaker keeps all (v1.1)",
    },
)
