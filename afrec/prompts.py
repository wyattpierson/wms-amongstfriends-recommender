"""Prompt builders — the experimentable core of the engine.

This is the module you edit when experimenting with "changes to the engine".
The evaluator (eval/evaluate.py) calls these through the same code path production
uses, so anything you change here is exactly what gets measured.

Prompt families, one per agent strategy (see afrec/agents.py):

  artist-then-album (two-call; avoids hallucinated cross-artist combos):
    Call 1  — build_prompt_artists : taste profile → 8–10 new artists
    Call 2  — build_prompt_albums  : per-artist → one real album each
    Retry   — build_retry_prompt   : replace Spotify-unverifiable albums
  album-first (one-shot):
    Call 1  — build_prompt_albums_first : taste profile → 8 real albums directly
    Retry   — build_retry_prompt   : (shared) replace Spotify-unverifiable albums
  pool-map (analyze the pool, then recommend into it):
    Call 0  — build_prompt_pool_map      : the user's OWN albums → lanes, range
                                           axes, fingerprint/attracts/repels,
                                           threads, plan budget (JSON object)
    Call 1  — build_prompt_pool_discovery: album-first, guided by the map;
                                           each pick declares lane + role
    Verify  — (shared) Spotify existence check + reflection retry
    Call 2  — build_prompt_critic        : fresh adversarial pass — argue each
                                            verified rec does NOT belong
    Call 3  — build_prompt_pool_replace  : fill exactly the unfilled roles
                                           (only if the critic rejected picks)
"""

from __future__ import annotations

import random

MAX_REVIEWS_FOR_LLM = 20


def _format_reviews_for_prompt(
    reviews: list[dict],
    max_reviews: int = MAX_REVIEWS_FOR_LLM,
    catalog_years: dict | None = None,
    catalog_facts: dict | None = None,
) -> tuple[str, str, str]:
    """
    Shared helper for all prompt builders.
    Returns (review_block, reviewed_artists_str, reviewed_albums_str).
    Reviews are sorted by rating (best first) then shuffled so no single
    artist gets positional prominence.

    catalog_years: optional {(artist.lower(), album.lower()): year}, usually from
    deterministic Spotify lookups — appended to each line as a fact the model
    should trust over its own memory (sparse real profiles carry only
    album + artist + rating, so the era signal should come from the catalog,
    not the model's recall of the user's own library).
    """
    top = sorted(reviews, key=lambda r: r["rating"], reverse=True)[:max_reviews]
    random.shuffle(top)

    lines = []
    for r in top:
        line = f"  - {r['artist']} — {r['album']} ({r['rating']}/5)"
        if r.get("text"):
            line += f'\n    Note: "{r["text"]}"'
        cl = _catalog_suffix(r, catalog_years, catalog_facts)
        if cl:
            line += "  " + cl
        lines.append(line)

    review_block = "\n".join(lines)
    reviewed_artists = ", ".join(sorted({f'"{r["artist"]}"' for r in top}))
    reviewed_albums = ", ".join(f'"{r["artist"]} - {r["album"]}"' for r in top)
    return review_block, reviewed_artists, reviewed_albums


def _catalog_suffix(r: dict, catalog_years: dict | None, catalog_facts: dict | None) -> str | None:
    """The per-review catalog suffix (without the leading spaces), or None.

    catalog_facts is {name_key(artist, album): CatalogFact} from the
    enrichment layer (afrec/enrich.py); its catalog_line() already carries
    the year plus tags/sound, so a fact WINS over the bare-year dict (tuples
    keyed, from _enrich_years), which is the fallback for reviews the
    enricher knows nothing about. Both absent → the review line renders
    exactly as pre-enrichment.
    """
    if catalog_facts:
        from .enrich import name_key   # local: keep prompts.py import-light
        fact = catalog_facts.get(name_key(r["artist"], r["album"]))
        if fact is not None:
            cl = fact.catalog_line()
            if cl:
                return cl
    if catalog_years:
        yr = catalog_years.get((r["artist"].lower().strip(), r["album"].lower().strip()))
        if yr:
            return f"(catalog: {yr})"
    return None


# ── Call 1: artist discovery ──────────────────────────────────────────────────

def build_prompt_artists(reviews: list[dict]) -> str:
    """
    Call 1 — artist discovery.
    Deliberately does NOT ask for album titles yet: models are far more
    reliable at taste-matching on artists than at cross-artist album recall.
    """
    review_block, reviewed_artists, _ = _format_reviews_for_prompt(reviews)

    return f"""You are a music taste analyst for a social app called AmongstFriends.

Based on this user's album ratings, suggest 8–10 artists they have NOT already listened to that fit their taste profile.

User's ratings:
{review_block}

Rules:
- Suggest ONLY artists not already in the list above.
- Already reviewed artists (never suggest these): {reviewed_artists}
- Weight suggestions by the user's RATINGS, not by how famous an artist is. A 5-star obscure artist should influence this more than a 3-star famous one.
- Treat each artist in the history as one equal data point regardless of their fame.
- Prioritize discovery: include a mix of well-known and lesser-known artists. Do not default to only the most commercially successful options.
- Do not anchor on any single source artist — reflect the overall taste pattern.
- Vary the list across sub-genres and eras.

Respond ONLY with a valid JSON array of artist name strings. No markdown, no explanation.
["Artist One", "Artist Two", "Artist Three"]"""


# ── Call 2: album recall ──────────────────────────────────────────────────────

def build_prompt_albums(artists: list[str], reviews: list[dict]) -> str:
    """
    Call 2 — album recall.
    Given confirmed artists, ask for one specific real album each. Factual
    recall of a single artist's discography is far more reliable than
    cross-artist associations.
    """
    review_block, _, reviewed_albums = _format_reviews_for_prompt(reviews)
    artists_block = "\n".join(f"  - {a}" for a in artists)

    return f"""You are a music recommendation engine for a social app called AmongstFriends.

For each artist below, recommend ONE real album they have actually released that this user would love.

Artists to find albums for:
{artists_block}

User's taste profile for context:
{review_block}

Rules:
- You MUST only recommend albums genuinely released by that specific artist. Do not mix album titles across artists.
- If you are not 100% certain an artist released an album with a specific title, choose a different album you ARE certain about.
- Pick the album that best fits the user's taste, not necessarily the artist's most famous one.
- Do not suggest artists the user has already reviewed.
- Already reviewed albums (never include): {reviewed_albums}
- One entry per artist — do not skip any artist from the list above.
- Keep the reason to one or two sentences, written like a friend's tip.
- Confidence reflects how well this fits their taste (0.0–1.0).

- Also report, per entry: "genre" — your short coarse genre label for this pick (e.g. "jazz hiphop", "neo-soul", "dance"), and "era" — its decade (e.g. "2010s"). Your own read, kept short.

Respond ONLY with a valid JSON array with exactly one entry per artist. No markdown, no preamble.
[
  {{
    "artist": "Artist Name",
    "album": "Real Album Title",
    "confidence": 0.88,
    "reason": "Friend-style reason tied to their taste.",
    "genre": "coarse genre label",
    "era": "2010s"
  }}
]"""


# ── album-first agent: one-shot direct album proposal ─────────────────────────

def build_prompt_albums_first(reviews: list[dict], n: int = 8) -> str:
    """
    One-shot prompt for the `album-first` agent: ask for complete, specific,
    real albums directly — no artist-list detour.

    Hypothesis being A/B'd against artist-then-album: taste is expressed *in
    albums* (the reviews are album reviews), so proposing the record up front
    lets the model reason from the specific records the person rated — moods,
    eras, production — instead of detouring through an artist list it must
    then recall. Fewer calls, and room for bolder, more specific picks.
    Failure mode to watch: higher first-pass hallucination (no "this artist is
    definitely real" anchor) — exactly what the `real r1` metric measures.
    """
    review_block, reviewed_artists, reviewed_albums = _format_reviews_for_prompt(reviews)

    return f"""You are a music recommendation engine for a social app called AmongstFriends.

Based on this user's album ratings, recommend exactly {n} real albums they have NOT already listened to, in ONE single list.

User's ratings:
{review_block}

Rules:
- Each entry is a SPECIFIC album (artist + exact album title) that genuinely exists and is well-known enough to be found on Spotify. If you are not certain an album with that exact title exists, choose one you ARE certain about.
- Do not suggest artists the user has already reviewed. Never-suggest artists: {reviewed_artists}
- Never include albums already reviewed: {reviewed_albums}
- Reason from the taste in the ratings — moods, eras, production choices, lyrical themes — not just the surface genre. Weight by RATINGS: a 5-star obscure artist should matter more than a 3-star famous one.
- Mix it: some safe-but-great follow-ups, plus a few bold-but-adjacent picks (a neighboring scene, a shared producer, a cross-genre bridge) this person is likely to love but has probably never heard.
- Vary the eras and sub-genres — do not give {n} albums that all sound the same.
- Keep the reason to one or two sentences, written like a friend's tip.
- Confidence reflects how certain you are the album is real AND that it fits their taste (0.0–1.0).
- Also report, per entry: "genre" — your short coarse genre label for this pick (e.g. "jazz hiphop", "neo-soul", "dance"), and "era" — its decade (e.g. "2010s"). Your own read, kept short.

Respond ONLY with a valid JSON array of exactly {n} objects. No markdown, no preamble.
[
  {{
    "artist": "Artist Name",
    "album": "Exact Real Album Title",
    "confidence": 0.88,
    "reason": "Friend-style reason tied to their taste.",
    "genre": "coarse genre label",
    "era": "2010s"
  }}
]"""


# ── pool-map agent: the analysis + guided-generation prompt family ────────────

def build_prompt_pool_map(
    reviews: list[dict],
    catalog_years: dict | None = None,
    n_recs: int = 8,
    max_reviews: int = 30,
) -> str:
    """
    Call 0 of the pool-map agent — analyze the user's OWN albums before anything
    is recommended. Output is a JSON OBJECT (the "pool map") that the later
    calls work from, and that the eval can hold the final recs against.

    Built for sparse input: real profiles may carry only album + artist +
    rating, so the model externalizes its read of each album and flags
    honestly ("known") when it is actually inferring — so lanes aren't built
    on hallucinated facts about the user's own library. catalog_years
    (deterministic Spotify lookups) anchors the era axis in facts.
    """
    review_block, _, _ = _format_reviews_for_prompt(
        reviews, max_reviews=max_reviews, catalog_years=catalog_years
    )

    return f"""You are a music taste analyst for a social app called AmongstFriends.

STEP 1 of 2 — POOL MAP. Before anything is recommended, you analyze this user's own listening. Do NOT recommend any new music in this step — analysis only.

User's ratings:
{review_block}

(Albums marked "catalog: YYYY" carry a release year from the music catalog — trust that over your memory. All other facts about these albums come from your own knowledge.)

Small-pool rule: if the user has rated fewer than ~8 albums, be conservative — a tiny pool is early evidence, not a verdict. Mark lanes "weak" unless several highly-rated KNOWN albums support them, keep the era claims in "axes" modest, and derive "repels" ONLY from albums they actually rated low (if they have none, use []). Do not invent hard constraints out of what the small pool happens not to contain.

Produce the pool map, in this order:

1. "reads" — for EVERY album above, your best read: a coarse genre, the decade, and energy (1–10; 1 = slow/downtempo, 10 = peak danceable). Set "known": true ONLY if you are confident you actually know this specific album — if you are inferring from the artist or guessing, set "known": false. Honesty matters: lanes built on guessed reads are weak.

2. "lanes" — group the albums into 2–4 lanes. A lane is ANY recurring property tying 2+ of their albums together: a genre, an era, a mood, a production style, a scene or city, a shared producer, a tempo family, a lyrical theme. For each: a short name, one sentence on what ties it together, its albums, an energy estimate (1–10), and strength — "strong" (several and/or highly-rated albums, mostly known), "moderate", or "weak" (1–2 albums, or mostly guessed reads).

3. "axes" — the RANGE of the pool: "genre_spread" and "era_spread" ("narrow" | "medium" | "wide"), "energy_range" as [low, high] across their albums, and 2–4 "mood" words.

4. Uniqueness — what makes recommendations fit THIS person instead of a typical fan of their top genre:
   - "fingerprint": 2–4 things that distinguish them from a typical listener of their dominant lane.
   - "attracts": 2–4 things their highest-rated albums share (production, vocal style, tempo, lyrical tone).
   - "repels": 1–4 things their low-rated albums rule out. These become hard constraints later.
   - "surprises": the album(s) that are unexpected given the rest of the pool, and what each suggests about them.
   - "tension": the biggest contradiction inside the pool and how to handle it (lean into one side, or bridge deliberately — never average the two). Use "none" if there isn't one.

5. "threads" — 2–4 connective threads linking their albums (shared producers, scenes, cities, sounds, techniques). The raw material for adjacent-but-not-identical discoveries.

6. "plan" — the allocation for the {n_recs} recommendations that will be made in step 2:
   - "core": picks sitting inside one of their existing lanes
   - "adjacent": picks from a neighboring scene or thread
   - "stretch": bold-but-plausible leaps, each with a stated thread back to the pool
   The three numbers must add up to {n_recs}. Weight by RATINGS, not fame: a 5-star obscure album matters more than a 3-star famous one.

Respond ONLY with a valid JSON object. No markdown, no preamble.
{{
  "reads": [
    {{"album": "Artist — Album", "genre": "coarse genre", "era": "1990s", "energy": 4, "known": true}}
  ],
  "lanes": [
    {{"name": "short lane name", "why": "one sentence", "albums": ["Artist — Album"], "energy": 4, "strength": "strong"}}
  ],
  "axes": {{"genre_spread": "medium", "era_spread": "narrow", "energy_range": [2, 6], "mood": ["warm", "reflective"]}},
  "fingerprint": ["..."],
  "attracts": ["..."],
  "repels": ["..."],
  "surprises": ["..."],
  "tension": "none",
  "threads": ["..."],
  "plan": {{"core": 3, "adjacent": 3, "stretch": 2}}
}}"""


def _map_block(pmap: dict) -> str:
    """Compact rendering of a pool map for the downstream calls (discovery,
    critic, replace). Tolerates missing/partial fields — a degraded map still
    renders whatever it has."""
    lines: list[str] = []

    lanes = pmap.get("lanes")
    if isinstance(lanes, list):
        rendered = []
        for ln in lanes:
            if not isinstance(ln, dict):
                continue
            name = str(ln.get("name", "?")).strip()
            meta = []
            if str(ln.get("strength", "")).strip():
                meta.append(str(ln["strength"]).strip())
            if ln.get("energy") not in ("", None):
                meta.append(f"energy ~{ln['energy']}")
            why = str(ln.get("why", "")).strip()
            rendered.append(f"    - {name}" + (f" ({', '.join(meta)})" if meta else "") + (f": {why}" if why else ""))
        if rendered:
            lines.append("  Lanes (the groupings in their listening):")
            lines.extend(rendered)

    axes = pmap.get("axes")
    if isinstance(axes, dict):
        parts = []
        if axes.get("genre_spread"):
            parts.append(f"genre spread: {axes['genre_spread']}")
        if axes.get("era_spread"):
            parts.append(f"era spread: {axes['era_spread']}")
        er = axes.get("energy_range")
        if isinstance(er, (list, tuple)) and len(er) == 2:
            parts.append(f"energy {er[0]}–{er[1]}")
        mood = axes.get("mood")
        if isinstance(mood, list) and mood:
            parts.append("mood: " + ", ".join(str(m) for m in mood))
        if parts:
            lines.append("  Range: " + " · ".join(parts))

    for key, label in (
        ("fingerprint", "What makes them unique (fingerprint)"),
        ("attracts", "Drawn to"),
        ("repels", "Repels (HARD CONSTRAINTS — never recommend these)"),
        ("tension", "Tension"),
        ("threads", "Threads"),
    ):
        val = pmap.get(key)
        if isinstance(val, list) and val:
            lines.append(f"  {label}: " + "; ".join(str(v) for v in val))
        elif isinstance(val, str) and val.strip():
            lines.append(f"  {label}: {val.strip()}")

    plan = pmap.get("plan")
    if isinstance(plan, dict) and any(plan.values()):
        lines.append(
            f"  Plan: {plan.get('core', '?')} core (inside a lane) · "
            f"{plan.get('adjacent', '?')} adjacent (neighboring scene) · "
            f"{plan.get('stretch', '?')} stretch (bold, with a stated thread)"
        )
    return "\n".join(lines)


def build_prompt_pool_discovery(reviews: list[dict], pmap: dict | None = None, n: int = 8) -> str:
    """
    Call 1 of the pool-map agent — album-first generation guided by the pool
    map. Each pick declares the lane it sits in and the role it fills (core /
    adjacent / stretch) per the plan budget. With pmap=None (the map call
    failed) it degrades to plain unguided album-first discovery.
    """
    review_block, reviewed_artists, reviewed_albums = _format_reviews_for_prompt(reviews)

    if pmap:
        map_block = "Your pool map of this user — the brief you must work from:\n" + _map_block(pmap)
        plan_line = (
            "- Follow the PLAN in the pool map: each entry's \"role\" says whether it is core, "
            "adjacent, or stretch — hit the plan's counts exactly."
        )
    else:
        map_block = "(No pool map was produced for this user — reason from the ratings directly.)"
        plan_line = (
            "- Use each entry's \"role\": mostly picks that sit inside their strongest taste areas "
            "(core), a few from neighboring scenes (adjacent), and one or two bold-but-plausible "
            "stretches with a stated thread back to the pool (stretch)."
        )

    return f"""You are a music recommendation engine for a social app called AmongstFriends.

STEP 2 of 2 — RECOMMEND. Based on this user's album ratings, recommend {n} real albums they have NOT already reviewed.

{map_block}

User's ratings:
{review_block}

Rules:
- Each entry is a SPECIFIC album (artist + exact album title) that genuinely exists. If you are not certain an album with that exact title exists, choose one you ARE certain about.
- Never suggest artists the user has already reviewed: {reviewed_artists}
- Never include albums the user has already reviewed: {reviewed_albums}
{plan_line}
- "lane": name the pool-map lane it sits in (core), or the adjacent scene/thread it crosses from (adjacent/stretch).
- Treat "Repels" as hard constraints — nothing in the list may trip them.
- Let the fingerprint drive the picks: a recommendation that would only fit a *typical* fan of their top genre, not this specific person, is a failure.
- Weight by RATINGS, not fame: a 5-star obscure album matters more than a 3-star famous one.
- Keep the reason to one or two sentences, written like a friend's tip — and for adjacent/stretch picks, name the thread that connects it to the pool.
- Confidence reflects how certain you are the album is real AND that it fits this specific user (0.0–1.0).
- Also report per entry: "genre" (a short coarse label) and "era" (the decade, e.g. "2010s").

Respond ONLY with a valid JSON array of exactly {n} objects. No markdown, no preamble.
[
  {{
    "artist": "Artist Name",
    "album": "Exact Real Album Title",
    "confidence": 0.88,
    "reason": "Friend-style reason tied to their taste.",
    "genre": "coarse genre label",
    "era": "2010s",
    "role": "core",
    "lane": "lane or adjacent scene name"
  }}
]"""


def build_prompt_critic(reviews: list[dict], pmap: dict, recs: list[dict]) -> str:
    """
    Call 2 of the pool-map agent — the adversarial taste check. Runs on the
    Spotify-VERIFIED candidates, so it judges taste-fit, not existence.
    A fresh model call argues each rec does NOT belong to this specific user;
    a rec survives only if the best rejection argument fails. Also returns the
    plan roles that no surviving rec covers.

    recs: dicts with artist/album/reason/role/lane/self_genre/self_era.
    """
    review_block, _, _ = _format_reviews_for_prompt(reviews)
    lines = []
    for i, r in enumerate(recs, 1):
        bits = []
        if r.get("role"):
            bits.append(f"role: {r['role']}")
        if r.get("lane"):
            bits.append(f"lane: {r['lane']}")
        if r.get("self_genre"):
            bits.append(f"genre: {r['self_genre']}")
        if r.get("self_era"):
            bits.append(f"era: {r['self_era']}")
        line = f"  {i}. {r['artist']} — {r['album']}" + (f" ({'; '.join(bits)})" if bits else "")
        if r.get("reason"):
            line += f'  reason: "{r["reason"]}"'
        lines.append(line)
    recs_block = "\n".join(lines)

    return f"""You are a skeptical music critic for AmongstFriends. You did NOT make the recommendations below — a different analyst did. Your job is to try to REJECT each one.

The pool map of the user (the analyst's own brief):
{_map_block(pmap)}

User's ratings (low ratings are evidence):
{review_block}

Candidate recommendations:
{recs_block}

For each rec, steelman the argument that it does NOT belong in THIS user's pool. Reject it if any of these hold:
- ROLE COSTUME: its claimed role is not what it is — a "core" pick that is not actually inside that lane, or a "stretch" with no thread back to the pool.
- REPEL COLLISION: it trips anything in the "Repels" constraints, or echoes one of the user's low-rated albums.
- GENERIC: it is a typical-fan pick for their top genre that ignores the fingerprint — fine for someone else, wrong for this person.
- POOL SHAPE: adding it would break the shape of the resulting pool — the recs would all cluster in one decade, one energy band, or one genre, or the plan's roles would be badly unbalanced.

Calibration rules — the map is only as reliable as the pool behind it, and you are a quality filter, not a gatekeeper:
- If the user's pool is small (fewer than ~8 albums) or a lane is "weak", be LENIENT: weak evidence is not grounds to reject.
- An era or energy mismatch alone is NOT grounds to reject unless the rec's claimed role is "core" and the plan explicitly contradicts it.
- Reject only when your argument is strong AND a clearly better alternative exists for this person. If you are on the fence, KEEP.
- Reject AT MOST 3 recs. A pass that rejects most of the batch is miscalibrated.

Keep a rec ONLY if your best rejection argument genuinely fails — it fits THIS specific person.
Also report "unfilled_roles": which of the plan's roles (core / adjacent / stretch) no surviving rec covers. Use [] if every role is covered.

Respond ONLY with a valid JSON object. No markdown, no preamble.
{{
  "verdicts": [
    {{"artist": "Artist Name", "album": "Exact Album Title", "verdict": "keep", "reason": "one line"}}
  ],
  "unfilled_roles": ["stretch"]
}}"""


def build_prompt_pool_replace(
    reviews: list[dict],
    pmap: dict | None,
    kept: list[dict],
    rejected: list[tuple[dict, str]],
    unfilled_roles: list[str],
    needed: int,
) -> str:
    """
    Call 3 of the pool-map agent — the role-targeted retry. Runs only when the
    critic rejected picks (or the pool is otherwise short). Fills EXACTLY the
    missing roles instead of re-rolling the whole list, so the recs the critic
    already vouched for are not disturbed.
    """
    review_block, reviewed_artists, reviewed_albums = _format_reviews_for_prompt(reviews)

    if pmap:
        map_block = "The pool map of this user:\n" + _map_block(pmap)
    else:
        map_block = "(No pool map was produced for this user.)"

    kept_block = "\n".join(f"  - {r['artist']} — {r['album']} (role: {r.get('role') or '?'})" for r in kept) or "  (none)"
    rej_lines = []
    for r, reason in rejected:
        rej_lines.append(f"  - {r['artist']} — {r['album']} (was role: {r.get('role') or '?'}) — rejected: {reason or 'no reason given'}")
    rej_block = "\n".join(rej_lines) or "  (none — the pool is short for other reasons)"
    roles = ", ".join(unfilled_roles) if unfilled_roles else "whatever roles the plan still needs"

    return f"""You are a music recommendation engine for a social app called AmongstFriends.

A critic reviewed the previous batch of recommendations for this user and rejected some of them. Provide up to {needed} replacement albums. Prioritize filling these roles first: {roles}.

{map_block}

Recs that SURVIVED (do not repeat them, and do not re-suggest anything similar to them):
{kept_block}

Recs the critic REJECTED (do NOT re-suggest them, and do not just swap in a near-duplicate — fill the missing ROLE):
{rej_block}

User's ratings:
{review_block}

Rules:
- Each entry is a SPECIFIC album (artist + exact album title) that genuinely exists. If you are not certain an album with that exact title exists, choose one you ARE certain about — or return FEWER entries. A correct short list beats a long list of guesses; never invent, translate, or "clean up" a title.
- Never suggest artists the user has already reviewed: {reviewed_artists}
- Never include albums the user has already reviewed: {reviewed_albums}
- Each entry's "role" must be one of the missing roles above; "lane" names the lane or thread it crosses.
- Treat "Repels" as hard constraints. Let the fingerprint drive the picks.
- Keep the reason to one or two sentences, like a friend's tip; for adjacent/stretch picks, name the connecting thread.
- Also report per entry: "genre" and "era" (decade, e.g. "2010s").

Respond ONLY with a valid JSON array of at most {needed} objects. No markdown, no preamble.
[
  {{
    "artist": "Artist Name",
    "album": "Exact Real Album Title",
    "confidence": 0.85,
    "reason": "Friend-style reason tied to their taste.",
    "genre": "coarse genre label",
    "era": "2010s",
    "role": "stretch",
    "lane": "lane or thread name"
  }}
]"""


# ── Retry (post-verification reflection) ──────────────────────────────────────

def build_retry_prompt(
    reviews: list[dict],
    verified: list[dict],
    failed: list[dict],
    needed: int,
) -> str:
    """
    Correction prompt: tell the LLM what passed and what was hallucinated,
    ask for replacements only for the failed ones.
    """
    verified_block = "\n".join(
        f"  - {r['artist']} — {r['album']} ✅" for r in verified
    ) or "  (none)"

    failed_block = "\n".join(
        f"  - {r['artist']} — {r['album']} ❌ (not found on Spotify — this album does not exist)"
        for r in failed
    )

    already_reviewed = ", ".join(
        f'"{r["artist"]} - {r["album"]}"' for r in reviews[:MAX_REVIEWS_FOR_LLM]
    )

    already_accepted = ", ".join(
        f'"{r["artist"]} - {r["album"]}"' for r in verified
    )

    return f"""You are a music recommendation engine for the app AmongstFriends.

In your previous response, some of your album recommendations could not be verified on Spotify — they appear to be albums that do not actually exist. I need you to provide {needed} replacement(s).

Your previous recommendations that were VERIFIED and are fine (do not repeat these):
{verified_block}

Your previous recommendations that FAILED verification (these albums do not exist — do not suggest them again):
{failed_block}

Albums the user has already reviewed (do not suggest these either):
{already_reviewed}

Albums already accepted above (do not repeat):
{already_accepted}

Please suggest exactly {needed} replacement album(s) that:
- Definitely exist (you are certain they are real albums with these exact titles)
- The user would enjoy based on their taste profile
- Are not any album listed above

As before, also report per entry: "genre" (short coarse genre label) and "era" (its decade, e.g. "2010s").

Respond ONLY with a valid JSON array of exactly {needed} item(s). No markdown, no explanation.
[
  {{
    "album": "Exact Album Title",
    "artist": "Artist Name",
    "confidence": 0.88,
    "reason": "One or two sentences, friend-style recommendation.",
    "genre": "coarse genre label",
    "era": "2010s"
  }}
]"""


# ── Pool-map v1.1 family (grounded map · existence-first discovery · narrow critic · topup) ──

def _format_reviews_numbered(reviews, max_reviews=30, catalog_years: dict | None = None,
                             catalog_facts: dict | None = None) -> tuple[str, str, str]:
    """Numbered review block shared by the pool-map v1.1 family.

    Unlike _format_reviews_for_prompt (shuffled, unnumbered), this one is
    STABLE across all four calls of a single run: sorted by rating (best
    first), then reviewId, and numbered [#1..n]. That lets the critic cite a
    review number as evidence in a verdict, and lets lane "albums" reference
    the same ids the ratings carry. Deterministic ordering is fine here:
    these are analysis prompts, where best-first IS the signal.
    Returns (block, reviewed_artists, reviewed_albums) for the rule lines.
    """
    top = sorted(reviews, key=lambda r: (-r["rating"], str(r.get("reviewId", ""))))[:max_reviews]
    lines = []
    for i, r in enumerate(top, 1):
        line = f"  [{i}] {r['artist']} — {r['album']} ({r['rating']}/5)"
        if r.get("text"):
            line += f'\n      Note: "{r["text"]}"'
        cl = _catalog_suffix(r, catalog_years, catalog_facts)
        if cl:
            line += "  " + cl
        lines.append(line)
    artists = ", ".join(sorted({r["artist"] for r in reviews})) or "(none)"
    albums = ", ".join(f"{r['artist']} — {r['album']}" for r in reviews) or "(none)"
    return "\n".join(lines), artists, albums


def build_prompt_pool_map_v1_1(reviews, catalog_years: dict | None = None, n_recs=8, max_reviews: int = 30,
                               catalog_facts: dict | None = None, version_label: str = "v1.1") -> str:
    """Pool-map v1.1: the v1 map with GROUNDING rules.

    v1's map let an 8B invent lanes its tiny pool couldn't support (the
    club_kid run mapped a jazz lane out of 6 dance reviews and every
    downstream stage followed it). v1.1: lanes must rest on 2+ named pool
    albums, at most 3 lanes, and small pools (<8 albums) get an explicit
    small-pool block — fewer lanes, no 'strong' without 2+ 5-star known
    albums, core-heavy plan. Reviews are numbered so lanes and verdicts
    cite real ids.
    """
    review_block, _artists, _albums = _format_reviews_numbered(reviews, max_reviews, catalog_years, catalog_facts)
    catalog_note = (
        '(Albums marked "catalog: …" carry catalog FACTS for that album — release year, listener tags, '
        'and sound measurements (energy/danceability/valence, bpm, key). Trust them over your memory.)'
        if catalog_facts else
        '(Albums marked "catalog: YYYY" carry a release year from the music catalog — trust that over your memory.)'
    )
    n_pool = len({(r["artist"].lower().strip(), r["album"].lower().strip()) for r in reviews})
    if n_pool < 8:
        small_block = f"""SMALL-POOL RULE — this user has rated only {n_pool} albums. A tiny pool is EARLY EVIDENCE, not a verdict:
- Use at most 2 lanes, and mark a lane "strong" ONLY if 2+ five-star KNOWN albums support it.
- Keep "axes" claims modest ("narrow"/"medium", not "wide").
- Plan for {n_recs} recs with "core" at least half and "stretch" at most 1."""
        weight_rule = ""
    else:
        small_block = ""
        weight_rule = " Weight by RATINGS, not fame: a 5-star obscure album matters more than a 3-star famous one."

    return f"""You are a music taste analyst for a social app called AmongstFriends.

STEP 1 of 2 — POOL MAP ({version_label}). Before anything is recommended, you analyze this user's own listening. Do NOT recommend any new music in this step — analysis only.

User's ratings (numbered — the number is the review's id for citing):
{review_block}

{catalog_note}

GROUNDING RULES — the map is only as good as its evidence:
- A lane must rest on 2+ albums numbered above (1 only if the user has fewer than 5 albums total). If fewer than 2 albums share a property, it is a "thread", not a lane.
- Mark a read "known": true ONLY if you are confident you actually know that specific album. A lane built mostly on guessed reads is "weak".
- "repels" comes ONLY from albums they rated 3★ or below (use [] if they have none). Never derive a repel from what the pool does NOT contain.
{small_block}
Produce the pool map, in this order:

1. "reads" — for EVERY album above, your best read: a coarse genre, the decade, energy (1–10; 1 = slow/downtempo, 10 = peak danceable), and "known": true/false per the grounding rules.
2. "lanes" — at most 3 lanes. A lane is a recurring property tying its albums together: a genre, era, mood, production style, scene/city, shared producer, tempo family, or lyrical theme. For each: a short name, one sentence on what ties it together, its albums (the exact "Artist — Album" lines from above, each with its [number]), an energy estimate (1–10), and strength — "strong" (2+ highly-rated, mostly KNOWN albums), "moderate", or "weak" (1–2 albums or mostly guessed reads).
3. "axes" — the RANGE of the pool: "genre_spread" and "era_spread" (each "narrow"|"medium"|"wide"), "energy_range" [low, high], and 2–4 "mood" words.
4. Uniqueness:
   - "fingerprint": 2–4 things that distinguish them from a typical listener of their dominant lane.
   - "attracts": 2–4 things their highest-rated albums share.
   - "repels": per the grounding rules above — hard constraints for the next step.
   - "surprises": the album(s) unexpected given the rest of the pool, and what each suggests.
   - "tension": the biggest contradiction and how to handle it (lean one way or bridge deliberately — never average). "none" if there isn't one.
5. "threads" — 2–4 connective threads (shared producers, scenes, cities, sounds, techniques). Raw material for adjacent discoveries.
6. "plan" — the allocation for the {n_recs} recommendations of step 2: "core" (inside an existing lane), "adjacent" (neighboring scene/thread), "stretch" (bold-but-plausible leap with a stated thread back to the pool). The three numbers must add up to {n_recs}.{weight_rule}

Respond ONLY with a valid JSON object. No markdown, no preamble.
{{
  "reads": [{{"album": "Artist — Album", "genre": "...", "era": "2010s", "energy": 6, "known": true}}],
  "lanes": [{{"name": "...", "why": "...", "albums": ["Artist — Album (2)"], "energy": 6, "strength": "strong"}}],
  "axes": {{"genre_spread": "medium", "era_spread": "narrow", "energy_range": [4, 8], "mood": ["warm"]}},
  "fingerprint": ["..."],
  "attracts": ["..."],
  "repels": ["..."],
  "surprises": ["..."],
  "tension": "...",
  "threads": ["..."],
  "plan": {{"core": 4, "adjacent": 3, "stretch": 1}}
}}"""


def build_prompt_pool_map_v2_0(reviews, catalog_years: dict | None = None, n_recs=8, max_reviews: int = 30,
                               catalog_facts: dict | None = None, version_label: str = "v2.0") -> str:
    """Pool-map v2.0: the v1.1/v1.2 map with a COHERENCE-DRIVEN plan.

    v1.2 fed catalog FACTS into the map (year + tags + sound measurements).
    v2.0's ONLY further delta is step 6, the "plan": the core/adjacent/stretch
    split is no longer left to habit — it is set from "axes" (the pool's
    measured range). The more coherent the pool (genre AND era "narrow"), the
    harder the plan constrains to in-lane picks, down to 0 stretch; a genuinely
    diverse pool may explore. The small-pool rule is a ceiling on exploration
    the coherence rule may only tighten, never loosen. Everything else (grounding,
    small-pool block, downstream discovery/critic/topup) is v1.1/v1.2's, so an
    A/B against v1.2 isolates this one change: does branching that scales with
    input coherence improve quality?
    """
    review_block, _artists, _albums = _format_reviews_numbered(reviews, max_reviews, catalog_years, catalog_facts)
    catalog_note = (
        '(Albums marked "catalog: …" carry catalog FACTS for that album — release year, listener tags, '
        'and sound measurements (energy/danceability/valence, bpm, key). Trust them over your memory.)'
        if catalog_facts else
        '(Albums marked "catalog: YYYY" carry a release year from the music catalog — trust that over your memory.)'
    )
    n_pool = len({(r["artist"].lower().strip(), r["album"].lower().strip()) for r in reviews})
    if n_pool < 8:
        small_block = f"""SMALL-POOL RULE — this user has rated only {n_pool} albums. A tiny pool is EARLY EVIDENCE, not a verdict:
- Use at most 2 lanes, and mark a lane "strong" ONLY if 2+ five-star KNOWN albums support it.
- Keep "axes" claims modest ("narrow"/"medium", not "wide").
- Plan for {n_recs} recs with "core" at least half and "stretch" at most 1."""
        weight_rule = ""
    else:
        small_block = ""
        weight_rule = " Weight by RATINGS, not fame: a 5-star obscure album matters more than a 3-star famous one."

    plan_rule = f"""
   SET THE SPLIT FROM THE RANGE, not from habit — the more coherent the pool, the harder you constrain; the more diverse it is, the more you may explore. Read "axes" (step 3) first, then allocate:
   • COHERENT — genre_spread "narrow" AND era_spread "narrow" (energy_range tight): this is ONE thing they like. Stay inside it. ~3/4 core, ~1/4 adjacent, 0 stretch — EXCEPT a genuine outlier named in "surprises" or "tension" earns at most 1 stretch. At {n_recs} recs: 6–7 core, 1–2 adjacent, 0 stretch. A coherent pool served mostly in-lane is the CORRECT answer, not a cop-out — do not branch out of politeness.
   • IN BETWEEN — exactly one of genre_spread/era_spread is "medium", the other "narrow": core-leaning. ~1/2–2/3 core, ~1/4–1/3 adjacent, 0–1 stretch. At {n_recs} recs: 4–5 core, 2–3 adjacent, 0–1 stretch.
   • DIVERSE — genre_spread "wide" OR era_spread "wide" (energy_range wide): they are eclectic and the map has several strong lanes/threads. Explore: balanced-to-exploratory, roughly 1/3 each (stretch 1–3). At {n_recs} recs: 3 core, 2–3 adjacent, 2–3 stretch.
   (If the SMALL-POOL RULE above applies, it is a ceiling on exploration — never exceed its "stretch" limit, even for a diverse pool.)"""

    return f"""You are a music taste analyst for a social app called AmongstFriends.

STEP 1 of 2 — POOL MAP ({version_label}). Before anything is recommended, you analyze this user's own listening. Do NOT recommend any new music in this step — analysis only.

User's ratings (numbered — the number is the review's id for citing):
{review_block}

{catalog_note}

GROUNDING RULES — the map is only as good as its evidence:
- A lane must rest on 2+ albums numbered above (1 only if the user has fewer than 5 albums total). If fewer than 2 albums share a property, it is a "thread", not a lane.
- Mark a read "known": true ONLY if you are confident you actually know that specific album. A lane built mostly on guessed reads is "weak".
- "repels" comes ONLY from albums they rated 3★ or below (use [] if they have none). Never derive a repel from what the pool does NOT contain.
{small_block}
Produce the pool map, in this order:

1. "reads" — for EVERY album above, your best read: a coarse genre, the decade, energy (1–10; 1 = slow/downtempo, 10 = peak danceable), and "known": true/false per the grounding rules.
2. "lanes" — at most 3 lanes. A lane is a recurring property tying its albums together: a genre, era, mood, production style, scene/city, shared producer, tempo family, or lyrical theme. For each: a short name, one sentence on what ties it together, its albums (the exact "Artist — Album" lines from above, each with its [number]), an energy estimate (1–10), and strength — "strong" (2+ highly-rated, mostly KNOWN albums), "moderate", or "weak" (1–2 albums or mostly guessed reads).
3. "axes" — the RANGE of the pool: "genre_spread" and "era_spread" (each "narrow"|"medium"|"wide"), "energy_range" [low, high], and 2–4 "mood" words.
4. Uniqueness:
   - "fingerprint": 2–4 things that distinguish them from a typical listener of their dominant lane.
   - "attracts": 2–4 things their highest-rated albums share.
   - "repels": per the grounding rules above — hard constraints for the next step.
   - "surprises": the album(s) unexpected given the rest of the pool, and what each suggests.
   - "tension": the biggest contradiction and how to handle it (lean one way or bridge deliberately — never average). "none" if there isn't one.
5. "threads" — 2–4 connective threads (shared producers, scenes, cities, sounds, techniques). Raw material for adjacent discoveries.
6. "plan" — the allocation for the {n_recs} recommendations of step 2: "core" (inside an existing lane), "adjacent" (neighboring scene/thread), "stretch" (bold-but-plausible leap with a stated thread back to the pool). The three numbers must add up to {n_recs}.{weight_rule}{plan_rule}

Respond ONLY with a valid JSON object. No markdown, no preamble.
{{
  "reads": [{{"album": "Artist — Album", "genre": "...", "era": "2010s", "energy": 6, "known": true}}],
  "lanes": [{{"name": "...", "why": "...", "albums": ["Artist — Album (2)"], "energy": 6, "strength": "strong"}}],
  "axes": {{"genre_spread": "narrow", "era_spread": "narrow", "energy_range": [3, 5], "mood": ["warm"]}},
  "fingerprint": ["..."],
  "attracts": ["..."],
  "repels": ["..."],
  "surprises": ["..."],
  "tension": "...",
  "threads": ["..."],
  "plan": {{"core": 7, "adjacent": 1, "stretch": 0}}
}}"""


def build_prompt_pool_discovery_v1_1(reviews, pmap: dict | None, n: int, max_reviews: int = 30) -> str:
    """Pool-discovery v1.1: EXISTENCE-FIRST.

    v1 forced the model to "hit the plan's counts exactly" with a role and a
    lane per rec — on an 8B that pressure produced invented titles (real r1
    fell to 57% vs album-first's 83%). v1.1 makes the counts GOALS, not
    quotas: a certain core pick beats an uncertain stretch, and a stretch
    slot with no real album in it gets filled by the next-best certain pick.
    Also adds the anti-reflex rule: if the obvious pick is the one a typical
    fan would name first, look for the fingerprint's second choice.
    """
    map_block = _map_block(pmap) if pmap else "(No pool map was produced — work directly from the ratings below.)"
    review_block, artists, albums = _format_reviews_numbered(reviews, max_reviews)

    if pmap:
        plan_rule = ("- Follow the PLAN in the pool map as closely as existence allows: each entry's "
                     '"role" says core, adjacent, or stretch.')
        lane_rule = ('- "lane": name the pool-map lane it sits in (core), or the adjacent '
                     'scene/thread it crosses from (adjacent/stretch).')
    else:
        plan_rule = ('- Spread the list across the user\'s range: a few safe picks, a couple of '
                     'adjacent scenes, one bold-but-plausible stretch. Each entry\'s "role" says '
                     'core, adjacent, or stretch.')
        lane_rule = ('- "lane": the taste area it sits in or the thread that connects it '
                     '(e.g. "downtempo soul", "shared producer: Madlib").')

    return f"""You are a music recommendation engine for a social app called AmongstFriends.

STEP 2 of 2 — RECOMMEND (v1.1). Based on this user's album ratings, recommend {n} real albums they have NOT already reviewed.

{map_block}
User's ratings (numbered):
{review_block}

Rules:
- Each entry is a SPECIFIC album (artist + exact album title) that genuinely exists. If you are not certain an album with that exact title exists, choose one you ARE certain about.
- EXISTENCE FIRST: the plan's counts are GOALS, not quotas. A certain core pick beats an uncertain stretch — if you cannot name a real album for a stretch slot, fill that slot with the next-best pick you ARE certain exists. Never invent, translate, or "clean up" a title.
- Avoid the reflex pick: if the most obvious album for a lane is the one a TYPICAL fan of that lane would name first, ask whether the fingerprint points to a better second choice — and only take it if you are certain it exists.
- Never suggest artists the user has already reviewed: {artists}
- Never include albums the user has already reviewed: {albums}
{plan_rule}
{lane_rule}
- Treat "Repels" as hard constraints — nothing in the list may trip them.
- Weight by RATINGS, not fame: a 5-star obscure album matters more than a 3-star famous one.
- Keep the reason to one or two sentences, written like a friend's tip — and for adjacent/stretch picks, name the thread that connects it to the pool.
- Confidence reflects how certain you are the album is real AND that it fits this specific user (0.0–1.0).
- Also report per entry: "genre" (a short coarse label) and "era" (the decade, e.g. "2010s").

Respond ONLY with a valid JSON array of exactly {n} objects. No markdown, no preamble.
[
  {{"artist": "Artist Name", "album": "Exact Album Title", "confidence": 0.8, "reason": "one or two sentences, like a friend's tip", "genre": "downtempo soul", "era": "2010s", "role": "core", "lane": "downtempo soul"}}
]"""


def build_prompt_critic_v1_1(reviews, pmap: dict, recs: list[dict], max_reviews: int = 30) -> str:
    """Critic v1.1: NARROW, judgeable rules.

    v1's critic had four rejection rules (LANE COLLISION, ROLE COSTUME,
    POOL SHAPE, GENERIC) and, on the tiny eval pools, the leniency rules
    ("small pool → be LENIENT", "reject at most 3", "on the fence → KEEP")
    plus the circuit breaker effectively disabled it — while it still cost a
    full generation call. The one case it mattered for (Thelonious Monk for
    a club profile) was a LANE VOID: a rec in a scene the map has no lane or
    thread for. v1.1 keeps exactly that rule plus REPEL COLLISION (the rule
    the repels feature exists for); drops GENERIC and POOL SHAPE (an 8B
    cannot judge them reliably). Verdicts may cite the numbered review that
    is the evidence.
    """
    map_block = _map_block(pmap)
    review_block, _artists, _albums = _format_reviews_numbered(reviews, max_reviews)
    recs_block = "\n".join(
        f"  {r['artist']} — {r['album']} (role: {r.get('role', '?')}, lane: {r.get('lane', '?')}, reason: {r['reason']})"
        for r in recs
    )

    return f"""You are a skeptical music critic for AmongstFriends. You did NOT make the recommendations below — a different analyst did. Your job is to try to REJECT each one.

The pool map of the user (the analyst's own brief):
{map_block}
User's ratings (numbered — low ratings are evidence; cite the number when you use one):
{review_block}

Candidate recommendations:
{recs_block}

Reject a rec ONLY if one of these holds:
- LANE VOID: it does not sit in any of the map's lanes, and no stated thread connects it to the pool. (This includes a "stretch" with no thread, and a "core" pick that is not actually inside its claimed lane.)
- REPEL COLLISION: it trips anything in the "Repels" constraints, or it echoes one of the user's LOW-rated albums (cite the review number).

Calibration — you are a quality filter, not a gatekeeper:
- If the user's pool is small (fewer than ~8 albums) or a lane is "weak", be LENIENT: weak evidence is not grounds to reject.
- An era or energy mismatch alone is NOT grounds to reject.
- A pick being famous or "obvious" is NOT grounds to reject.
- Keep a rec if you are on the fence. Reject AT MOST 3.

Keep a rec ONLY if neither rejection rule fires.
Also report "unfilled_roles": which of the plan's roles (core / adjacent / stretch) no surviving rec covers. Use [] if every role is covered.

Respond ONLY with a valid JSON object. No markdown, no preamble.
{{
  "verdicts": [
    {{"artist": "Artist Name", "album": "Exact Album Title", "verdict": "keep", "reason": "one line"}}
  ],
  "unfilled_roles": ["stretch"]
}}"""


def build_prompt_pool_topup(reviews, pmap: dict | None, kept_briefs: list[dict], needed: int, max_needed: int = 6) -> str:
    """Topup: fill a short list with more certain picks.

    Runs when critic rejections (and their failed rescues) or artist dedupe
    leave the batch under the target size. Unguided and existence-first — a
    correct short list beats a long list of guesses, so the model may return
    fewer entries than asked.
    """
    map_block = _map_block(pmap) if pmap else ""
    review_block, artists, albums = _format_reviews_numbered(reviews, 30)
    kept_block = "\n".join(f"  {r['artist']} — {r['album']}" for r in kept_briefs) or "  (nothing yet)"
    n = min(needed, max_needed)
    map_section = f"{map_block}\n" if map_block else ""

    return f"""You are a music recommendation engine for the app AmongstFriends.

FILL THE REST: the recommendations below for this user are short of the target. Provide up to {n} MORE real albums.

{map_section}The list so far (do not repeat these artists or albums):
{kept_block}

User's ratings (numbered):
{review_block}

Rules:
- Each entry is a SPECIFIC album (artist + exact album title) that genuinely exists. If you are not certain an album with that exact title exists, choose one you ARE certain about — or return FEWER entries. A correct short list beats a long list of guesses.
- Never suggest artists the user has already reviewed: {artists}
- Never include albums the user has already reviewed: {albums}
- Prefer picks inside the user's strongest taste areas (core); a bold pick only if it has a stated thread to the pool.
- Treat "Repels" as hard constraints.
- Keep the reason to one or two sentences, like a friend's tip.
- Also report per entry: "genre" (a short coarse label) and "era" (the decade, e.g. "2010s").

Respond ONLY with a valid JSON array of at most {n} objects. No markdown, no preamble.
[
  {{"artist": "Artist Name", "album": "Exact Album Title", "confidence": 0.8, "reason": "one or two sentences, like a friend's tip", "genre": "downtempo soul", "era": "2010s", "role": "core", "lane": "downtempo soul"}}
]"""
