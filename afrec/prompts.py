"""Prompt builders — the experimentable core of the engine.

This is the module you edit when experimenting with "changes to the engine".
The evaluator (eval/evaluate.py) calls these through the same code path production
uses, so anything you change here is exactly what gets measured.

Two-call design (avoids hallucinated cross-artist album combinations):
  Call 1  — build_prompt_artists : taste profile → 8–10 new artists
  Call 2  — build_prompt_albums  : per-artist → one real album each
  Retry   — build_retry_prompt   : replace Spotify-unverifiable albums
"""

from __future__ import annotations

import random

MAX_REVIEWS_FOR_LLM = 20


def _format_reviews_for_prompt(reviews: list[dict], max_reviews: int = MAX_REVIEWS_FOR_LLM) -> tuple[str, str, str]:
    """
    Shared helper for all prompt builders.
    Returns (review_block, reviewed_artists_str, reviewed_albums_str).
    Reviews are sorted by rating (best first) then shuffled so no single
    artist gets positional prominence.
    """
    top = sorted(reviews, key=lambda r: r["rating"], reverse=True)[:max_reviews]
    random.shuffle(top)

    lines = []
    for r in top:
        line = f"  - {r['artist']} — {r['album']} ({r['rating']}/5)"
        if r.get("text"):
            line += f'\n    Note: "{r["text"]}"'
        lines.append(line)

    review_block = "\n".join(lines)
    reviewed_artists = ", ".join(sorted({f'"{r["artist"]}"' for r in top}))
    reviewed_albums = ", ".join(f'"{r["artist"]} - {r["album"]}"' for r in top)
    return review_block, reviewed_artists, reviewed_albums


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

Respond ONLY with a valid JSON array with exactly one entry per artist. No markdown, no preamble.
[
  {{
    "artist": "Artist Name",
    "album": "Real Album Title",
    "confidence": 0.88,
    "reason": "Friend-style reason tied to their taste."
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

Respond ONLY with a valid JSON array of exactly {needed} item(s). No markdown, no explanation.
[
  {{
    "album": "Exact Album Title",
    "artist": "Artist Name",
    "confidence": 0.88,
    "reason": "One or two sentences, friend-style recommendation."
  }}
]"""
