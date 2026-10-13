"""Agent: pool-map — analyze the user's pool, then recommend INTO it.

Call 0   pool_map        : profile the user's OWN albums → lanes, range axes,
                           fingerprint/attracts/repels, surprises, threads, and
                           a plan budget (core/adjacent/stretch). Low
                           temperature. Degrades gracefully on failure —
                           discovery then runs unguided and the critic is
                           skipped.
Call 1   pool_discovery  : album-first, guided by the map. Each pick declares
                           the lane it sits in and the role it fills per the
                           plan. (No artist detour — the album IS the unit of
                           discovery.)
Verify   (shared)        : Spotify existence check + one reflection retry
                           (afrec.pipeline.verify_and_retry).
Call 2   critic          : a FRESH, adversarial pass over the verified recs —
                           argue each one does NOT belong to THIS user. Rejects
                           role costumes, repel collisions, generic picks, and
                           pool-shape breaks.
Call 3   critic_retry    : only if the critic rejected something — fill
                           exactly the unfilled roles.

Before Call 0, when a Spotify verifier is available, the agent does cheap
DETERMINISTIC lookups of the user's own albums (catalog years) and folds them
into the map prompt. Real profiles are sparse — often just album + artist +
rating — so the map's era axis should rest on catalog facts, not the model's
memory of the user's own library.

Hypothesis vs album-first / artist-then-album: the extra analysis calls buy
personalization — recs fit this specific person's lanes, range, and oddities
instead of a typical fan of their top genre. Cost: +2–4 calls and their
latency per profile.
"""

import afrec.llm as llm_mod
import afrec.pipeline as pipeline
import afrec.prompts as prompts
import afrec.spotify as spotify
from afrec.agent_registry import _register

N_RECS = 8
ANALYSIS_TEMP = 0.3        # map + critic: stable analysis, not sampling
MAP_NUM_PREDICT = 2500     # the map is a big JSON object (reads + lanes + plan)
CRITIC_NUM_PREDICT = 1200
ENRICH_CAP = 30            # max pool albums to Spotify-lookup for catalog years


def _enrich_years(reviews: list[dict], verifier, log) -> dict:
    """Deterministic Spotify lookups of the user's OWN albums →
    {(artist.lower(), album.lower()): release_year}. Never raises; a lookup
    failure just leaves that album unanchored in the map prompt."""
    years: dict = {}
    seen: set = set()
    for r in reviews:
        key = (r["artist"].lower().strip(), r["album"].lower().strip())
        if key in seen:
            continue
        if len(seen) >= ENRICH_CAP:
            break
        seen.add(key)
        try:
            result = verifier(r["artist"], r["album"])
        except Exception as e:
            log(f"  [enrich] spotify lookup error for {r['artist']} — {r['album']}: {e}")
            continue
        yr = result[3] if len(result) > 3 else None
        if yr:
            try:
                years[key] = int(yr)
            except (TypeError, ValueError):
                pass
    return years


def _usable_map(parsed) -> bool:
    """A map is usable if it has at least one named lane, or at least one of
    the analysis fields downstream prompts actually render."""
    if not isinstance(parsed, dict):
        return False
    lanes = parsed.get("lanes")
    if isinstance(lanes, list):
        if any(isinstance(l, dict) and str(l.get("name", "")).strip() for l in lanes):
            return True
    return any(
        bool(parsed.get(k)) for k in ("axes", "fingerprint", "attracts", "repels", "threads")
    )


def _map_summary(pmap: dict) -> str:
    lanes = pmap.get("lanes") or []
    lane_txt = ", ".join(
        f"{str(l.get('name', '?'))} [{l.get('strength', '?')}]"
        for l in lanes[:4] if isinstance(l, dict)
    ) or "none"
    plan = pmap.get("plan") or {}
    return (f"{len(lanes)} lane(s) ({lane_txt}); "
            f"plan core/adjacent/stretch = {plan.get('core', '?')}/{plan.get('adjacent', '?')}/{plan.get('stretch', '?')}")


def _rec_brief(r) -> dict:
    """Rec → the dict shape the critic/replace prompts expect."""
    return {
        "artist": r.artist,
        "album": r.album,
        "reason": r.reason,
        "role": r.role,
        "lane": r.lane,
        "self_genre": r.self_genre,
        "self_era": r.self_era,
    }


def _critic_split(recs: list, parsed: dict) -> tuple[list, list]:
    """Apply critic verdicts. A rec with no (matchable) verdict is KEPT — a
    half-finished critic pass shouldn't nuke the whole batch."""
    by_key = {}
    for v in parsed.get("verdicts") or []:
        if not isinstance(v, dict):
            continue
        key = (str(v.get("artist", "")).lower().strip(), str(v.get("album", "")).lower().strip())
        by_key[key] = (str(v.get("verdict", "")).lower().strip(), str(v.get("reason", "")).strip())
    kept, rejected = [], []
    for r in recs:
        key = (r.artist.lower().strip(), r.album.lower().strip())
        v = by_key.get(key)
        if v and v[0] == "reject":
            rejected.append((r, v[1]))
        else:
            kept.append(r)
    return kept, rejected


def _verify_candidate(d: dict, verifier, rnd: int):
    """One replacement dict → Rec (Spotify-verified when a verifier exists)."""
    artist = str(d.get("artist", "")).strip()
    album = str(d.get("album", "") or d.get("albumTitle", "")).strip()
    if not artist or not album:
        return None
    if verifier is None:
        return pipeline._rec_dict_to_rec(
            d, verified=False, link=spotify.search_url(artist, album),
            canonical=None, rnd=rnd,
        )
    try:
        result = verifier(artist, album)
    except Exception:
        result = (False, None, None)
    found, canonical, url = result[0], result[1], result[2]
    year = result[3] if len(result) > 3 else None
    canonical = canonical or album
    return pipeline._rec_dict_to_rec(
        d, verified=bool(found), link=url or spotify.search_url(artist, canonical),
        canonical=canonical, rnd=rnd, release_year=year,
    )


def run_pool_map(reviews: list[dict], llm: llm_mod.BaseLLM, *,
                 verifier=None, temperature: float = 0.7,
                 num_predict_artists: int = 512, num_predict_albums: int = 1500,
                 extra_options: dict | None = None,
                 log=None) -> pipeline.RunOutcome:
    log = log or (lambda _s: None)
    outcome = pipeline.RunOutcome()
    reviewed_artists = pipeline._new_artists(reviews)
    reviewed_albums = {(r["artist"].lower().strip(), r["album"].lower().strip()) for r in reviews}

    # ── 0a. deterministic catalog enrichment (only when a verifier exists) ──
    years: dict = {}
    if verifier is not None:
        years = _enrich_years(reviews, verifier, log)
        log(f"  [enrich] catalog years for {len(years)} of {len(reviewed_albums)} pool album(s)")

    # ── Call 0: pool map (analysis; degrades gracefully) ───────────────────
    prompt_map = prompts.build_prompt_pool_map(reviews, catalog_years=years, n_recs=N_RECS)
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
        log("  [call 0] pool map unavailable — degrading to unguided discovery")

    # ── Call 1: map-guided album discovery ─────────────────────────────────
    prompt1 = prompts.build_prompt_pool_discovery(reviews, pmap, n=N_RECS)
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

    outcome.total_candidates = len(candidates)
    log(f"  [discovery] {len(candidates)} unique candidates")

    # ── Spotify existence verification + shared reflection retry ───────────
    pipeline.verify_and_retry(reviews, llm, candidates, outcome,
                              verifier=verifier, temperature=temperature,
                              num_predict=num_predict_albums,
                              extra_options=extra_options, log=log)

    # ── Call 2 + 3: adversarial critic vs the map, then role-targeted fill ─
    if pmap is not None and outcome.recs:
        prompt_c = prompts.build_prompt_critic(reviews, pmap, [_rec_brief(r) for r in outcome.recs])
        s_c = pipeline._run_stage(llm, "critic", prompt_c, expected=dict,
                                  num_predict=CRITIC_NUM_PREDICT, temperature=ANALYSIS_TEMP,
                                  extra_options=extra_options)
        outcome.stages.append(s_c)
        outcome.total_latency_s += s_c.latency_s or 0.0

        if s_c.ok and isinstance(s_c.parsed, dict):
            kept, rejected = _critic_split(outcome.recs, s_c.parsed)
            unfilled = [str(u).strip() for u in (s_c.parsed.get("unfilled_roles") or []) if str(u).strip()]
            if rejected and len(rejected) * 2 > len(outcome.recs):
                # Circuit breaker: a critic pass that rejects a MAJORITY of an
                # already-Spotify-verified batch is miscalibrated (usually an
                # overfit map on a small pool). Ignore the verdicts; the shared
                # verification already did the existence work.
                log(f"  [critic] miscalibrated — rejected {len(rejected)}/{len(outcome.recs)}; ignoring verdicts, keeping all")
                kept, rejected, unfilled = list(outcome.recs), [], []
            outcome.critic_rejected = len(rejected)
            log(f"  [critic] kept {len(kept)}, rejected {len(rejected)}; unfilled roles: {unfilled or 'none'}")

            needed = N_RECS - len(kept)
            if needed > 0:
                prompt_r = prompts.build_prompt_pool_replace(
                    reviews, pmap,
                    [_rec_brief(r) for r in kept],
                    [(_rec_brief(r), reason) for r, reason in rejected],
                    unfilled,
                    needed,
                )
                s_r = pipeline._run_stage(llm, "critic_retry", prompt_r, expected=list,
                                          num_predict=num_predict_albums, temperature=temperature,
                                          extra_options=extra_options)
                outcome.stages.append(s_r)
                outcome.total_latency_s += s_r.latency_s or 0.0

                if s_r.ok and isinstance(s_r.parsed, list):
                    accepted = {(r.artist.lower().strip(), r.album.lower().strip()) for r in kept}
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
                        kept.append(rec)
                        rescued += 1
                    outcome.critic_rescued = rescued
                    log(f"  [critic_retry] rescued {rescued} replacement(s)")

            outcome.recs = kept
            outcome.verified_count = sum(1 for r in kept if r.verified)
            outcome.ok = bool(kept)
        else:
            log("  [critic] unusable verdicts — keeping all verified recs")

    if not outcome.recs:
        outcome.ok = False
    return outcome


_register(
    id="pool-map",
    version="v1",
    name="pool-map (analyze, then recommend into it)",
    summary=(
        "Analyze the user's own pool first (lanes, range, what they're uniquely drawn to), then "
        "recommend albums that fill the plan — with an adversarial critic pass against the map. "
        "Degrades gracefully to unguided discovery if the map call fails."
    ),
    run=run_pool_map,
    details={
        "calls": "4 (map → discovery → critic → role-targeted retry) + shared Spotify retry",
        "picks": f"{N_RECS} albums; each declares lane + role (core/adjacent/stretch) per the plan",
        "pool map": "low-temp JSON: reads (known/guessing), lanes, range axes, fingerprint, attracts/repels, surprises, tension, threads, plan budget",
        "enrichment": "deterministic Spotify lookups of the user's OWN albums → catalog years in the map prompt",
        "critic": "fresh adversarial pass on VERIFIED recs: rejects role costumes, repel collisions, generic picks, pool-shape breaks",
        "retry policy": "shared Spotify reflection retry, then role-targeted replacement only for critic rejects",
        "prompt family": "prompts.build_prompt_pool_map / _pool_discovery / _critic / _pool_replace",
        "verification": "Spotify artist+album search (if creds)",
        "hypothesis": "explicit pool analysis buys personalization — recs fit THIS person's lanes/range/oddities, not a typical fan of their top genre",
    },
)
