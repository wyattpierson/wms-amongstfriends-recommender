#!/usr/bin/env python3
"""
AmongstFriends Music Recommendation Engine
- Calls Firebase callable function get_user_reviews to fetch reviews
- Sends reviews to local Ollama LLM for recommendations
- Calls Firebase callable function submit_recommendation for each result
"""

import json
import os
import sys
import argparse
import urllib.parse
import requests
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv

# ── Load config from .env ─────────────────────────────────────────────────────
# All secrets and environment-specific values live in a .env file next to this
# script. Copy .env.example to .env and fill in your values.
# Never commit .env or serviceAccountKey.json to git.

load_dotenv()

def _require(key: str) -> str:
    """Get a required env var, exit with a clear message if missing."""
    val = os.getenv(key)
    if not val:
        print(f"❌ Missing required config: {key}")
        print(f"   Add it to your .env file. See .env.example for reference.")
        sys.exit(1)
    return val

def _optional(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

# ── Config ────────────────────────────────────────────────────────────────────

SERVICE_ACCOUNT_PATH = _optional("SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")

FIREBASE_PROJECT_ID  = _require("FIREBASE_PROJECT_ID")
FIREBASE_REGION      = _optional("FIREBASE_REGION", "us-central1")
FIREBASE_WEB_API_KEY = _require("FIREBASE_WEB_API_KEY")

OLLAMA_URL   = _optional("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = _optional("OLLAMA_MODEL", "llama3")

SPOTIFY_CLIENT_ID     = _optional("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = _optional("SPOTIFY_CLIENT_SECRET")
SPOTIFY_MATCH_THRESHOLD = float(_optional("SPOTIFY_MATCH_THRESHOLD", "0.5"))

BOT_USER_ID        = _optional("BOT_USER_ID", "oPduqxdtQ62yvTVm4KPO")
MIN_REVIEWS_TO_RUN = int(_optional("MIN_REVIEWS_TO_RUN", "3"))
MAX_REVIEWS_FOR_LLM = int(_optional("MAX_REVIEWS_FOR_LLM", "20"))

# ── Content types ─────────────────────────────────────────────────────────────
# Known catalog types returned by get_user_reviews.
# Only types listed here will be processed — others are silently skipped.
# To add a new type in the future: add an entry here and a matching
# build_prompt_<type>() function below.
SUPPORTED_TYPES = {
    "album":      "music",      # ← active
    "restaurant": "restaurant", # ← not yet implemented, here for documentation
    "book":       "book",       # ← not yet implemented, here for documentation
    "other":      "other",      # ← not yet implemented, here for documentation
}
ACTIVE_TYPE = "album"  # only this type is processed in the current run

# ── Firebase Auth: Get an ID token for LLMBot ─────────────────────────────────
#
# Firebase callable functions require a Firebase ID token in the Authorization
# header — they won't accept raw service account credentials directly.
#
# The flow:
#   1. Use the service account to mint a custom token for "LLMBot"
#   2. Exchange that custom token for a real Firebase ID token via REST
#   3. Use that ID token in all callable function requests
#
# This is the standard server-to-server pattern for callable functions.

def get_firebase_id_token() -> str:
    """Mint a custom token for LLMBot and exchange it for a Firebase ID token."""

    # Step 1: Create a custom token signed by the service account
    custom_token: bytes = auth.create_custom_token(BOT_USER_ID)

    # Step 2: Exchange for an ID token via Firebase Auth REST API
    exchange_url = (
        f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
        f"?key={FIREBASE_WEB_API_KEY}"
    )
    resp = requests.post(exchange_url, json={
        "token": custom_token.decode("utf-8"),
        "returnSecureToken": True,
    }, timeout=15)

    if not resp.ok:
        print(f"❌ Failed to exchange custom token for ID token.")
        print(f"   Status: {resp.status_code}")
        print(f"   Body:   {resp.text}")
        print(f"\n   Double-check that FIREBASE_WEB_API_KEY is set correctly.")
        sys.exit(1)

    return resp.json()["idToken"]


# ── Firebase Callable Function Caller ─────────────────────────────────────────

def call_firebase_function(function_name: str, payload: dict, id_token: str) -> dict:
    """
    Call a Firebase HTTPS callable function.
    Callable functions expect the body wrapped as { "data": <payload> }
    and return { "result": <your return value> }.
    """
    url = (
        f"https://{FIREBASE_REGION}-{FIREBASE_PROJECT_ID}"
        f".cloudfunctions.net/{function_name}"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {id_token}",
    }

    try:
        resp = requests.post(url, json={"data": payload}, headers=headers, timeout=30)
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot reach Firebase function '{function_name}'. Check your internet connection.")
        sys.exit(1)

    if not resp.ok:
        print(f"❌ Firebase function '{function_name}' returned HTTP {resp.status_code}:")
        print(f"   {resp.text}")
        sys.exit(1)

    outer = resp.json()
    # Callable functions wrap return values in "result"
    return outer.get("result", outer)


# ── Parse Reviews ─────────────────────────────────────────────────────────────

def parse_reviews(raw_reviews: list, content_type: str = ACTIVE_TYPE) -> list[dict]:
    """
    Normalize review objects from get_user_reviews into a flat structure,
    filtering to only the requested content_type.

    Known types: album, restaurant, book, other (and any future additions).
    Reviews with a non-matching type are skipped and counted.

    Input shape per review:
    {
      catalog: {
        type: "album",
        metadata: { albumName, artistName },
        name: "...",
      },
      rating: 3,
      reviewId: "...",
      text: "...",
      userId: "...",
      groupId: "..."
    }
    """
    parsed = []
    skipped: dict[str, int] = {}  # type -> count of skipped reviews

    for r in raw_reviews:
        catalog     = r.get("catalog", {})
        review_type = catalog.get("type", "unknown")

        if review_type != content_type:
            skipped[review_type] = skipped.get(review_type, 0) + 1
            continue

        metadata = catalog.get("metadata", {})
        parsed.append({
            "reviewId": r.get("reviewId", ""),
            "type":     review_type,
            "album":    metadata.get("albumName") or catalog.get("name", "Unknown Album"),
            "artist":   metadata.get("artistName", "Unknown Artist"),
            "rating":   int(r.get("rating", 3)),
            "text":     r.get("text", ""),
            "groupId":  r.get("groupId", "default-group"),
        })

    if skipped:
        skipped_summary = ", ".join(f"{count} {t}" for t, count in sorted(skipped.items()))
        print(f"   (Skipped {skipped_summary} review(s) — not '{content_type}' type)")

    return parsed


# ── Build LLM Prompts ─────────────────────────────────────────────────────────
# We use two separate LLM calls to avoid hallucinated artist+album combinations:
#   Call 1 — artist discovery: given the taste profile, suggest artists to explore
#   Call 2 — album recall:     for each artist, name one specific real album
#
# To add a new type: add build_prompt_<type>_artists() and
# build_prompt_<type>_albums() and wire into the two dispatchers below.

import random as _random

def _format_reviews_for_prompt(reviews: list[dict]) -> tuple[str, str, str]:
    """
    Shared helper used by both prompt builders.
    Returns (review_block, reviewed_artists_str, reviewed_albums_str).
    Shuffles the review order so no single artist gets positional prominence.
    """
    top = sorted(reviews, key=lambda r: r["rating"], reverse=True)[:MAX_REVIEWS_FOR_LLM]
    _random.shuffle(top)

    lines = []
    for r in top:
        line = f"  - {r['artist']} — {r['album']} ({r['rating']}/5)"
        if r["text"]:
            line += f"\n    Note: \"{r['text']}\""
        lines.append(line)

    review_block     = "\n".join(lines)
    reviewed_artists = ", ".join(sorted({f'"{r["artist"]}"' for r in top}))
    reviewed_albums  = ", ".join(f'"{r["artist"]} - {r["album"]}"' for r in top)
    return review_block, reviewed_artists, reviewed_albums


def build_prompt_artists(reviews: list[dict], content_type: str = ACTIVE_TYPE) -> str:
    """Dispatcher for Call 1: suggest artists."""
    if content_type == "album":
        return build_prompt_album_artists(reviews)
    raise NotImplementedError(f"No artist prompt builder for type: {content_type!r}")


def build_prompt_albums(artists: list[str], reviews: list[dict], content_type: str = ACTIVE_TYPE) -> str:
    """Dispatcher for Call 2: for confirmed artists, find a specific album."""
    if content_type == "album":
        return build_prompt_album_albums(artists, reviews)
    raise NotImplementedError(f"No album prompt builder for type: {content_type!r}")


def build_prompt_album_artists(reviews: list[dict]) -> str:
    """
    Call 1 — artist discovery.
    Intentionally does NOT ask for album titles yet. The model is much more
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


def build_prompt_album_albums(artists: list[str], reviews: list[dict]) -> str:
    """
    Call 2 — album recall.
    Given confirmed artists, ask for one specific real album each. Factual recall
    of a single artist's discography is far more reliable than cross-artist associations.
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


# ── Call Ollama ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> list[dict]:
    print(f"  Calling Ollama ({OLLAMA_MODEL})...")

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 1500,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot reach Ollama. Is it running? Try: ollama serve")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ Ollama timed out. Try a smaller model or increase the timeout.")
        sys.exit(1)

    raw = resp.json().get("response", "").strip()

    # Strip markdown fences if the model wraps output in ```json ... ```
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw   = "\n".join(lines[1:])
        if raw.strip().endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

    try:
        recommendations = json.loads(raw)
        if not isinstance(recommendations, list):
            raise ValueError("Expected a JSON array at the top level")
        return recommendations
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Could not parse Ollama response as JSON: {e}")
        print(f"\nRaw Ollama output:\n{raw}")
        sys.exit(1)


def call_ollama_artists(prompt: str) -> list[str]:
    """
    Variant of call_ollama for Call 1 — expects a JSON array of strings (artist names)
    rather than a list of recommendation dicts.
    """
    print(f"  Calling Ollama for artist suggestions ({OLLAMA_MODEL})...")

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 512},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot reach Ollama. Is it running? Try: ollama serve")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ Ollama timed out.")
        sys.exit(1)

    raw = resp.json().get("response", "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if raw.strip().endswith("```"):
            raw = raw[:raw.rfind("```")]
        raw = raw.strip()

    try:
        artists = json.loads(raw)
        if not isinstance(artists, list):
            raise ValueError("Expected a JSON array")
        # Tolerate mixed input — filter to strings only
        artists = [a for a in artists if isinstance(a, str) and a.strip()]
        return artists
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Could not parse artist list from Ollama: {e}")
        print(f"\nRaw Ollama output:\n{raw}")
        sys.exit(1)


# ── Spotify Search URL ────────────────────────────────────────────────────────

def spotify_search_url(artist: str, album: str) -> str:
    query = urllib.parse.quote(f"{artist} {album}")
    return f"https://open.spotify.com/search/{query}"


# ── Spotify Verification ─────────────────────────────────────────────────────

def spotify_get_token() -> str | None:
    """Get a Spotify client credentials token. Returns None if credentials not configured."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        timeout=10,
    )
    if not resp.ok:
        print(f"  ⚠️  Could not get Spotify token: {resp.status_code} {resp.text}")
        return None
    return resp.json().get("access_token")


def _similarity(a: str, b: str) -> float:
    """
    Simple character-overlap similarity between two strings (0.0-1.0).
    Case-insensitive. Good enough to catch hallucinated titles without
    requiring an external library like difflib or rapidfuzz.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    matches = sum(1 for ch in shorter if ch in longer)
    return matches / len(longer)


def spotify_verify(artist: str, album: str, token: str) -> tuple[bool, str | None, str | None]:
    """
    Search Spotify for an album and check whether the result actually matches.

    Returns:
        (verified, canonical_album, spotify_url)
        - verified:        True if a confident match was found
        - canonical_album: Spotify's official album title (use this instead of the LLM's version)
        - spotify_url:     Direct link to the album on Spotify
    """
    query = urllib.parse.quote(f"album:{album} artist:{artist}")
    resp = requests.get(
        f"https://api.spotify.com/v1/search?q={query}&type=album&limit=3",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if not resp.ok:
        print(f"  ⚠️  Spotify search failed: {resp.status_code}")
        return False, None, None

    items = resp.json().get("albums", {}).get("items", [])
    if not items:
        return False, None, None

    # Check top results for a match on both artist and album name
    for item in items:
        spotify_album   = item.get("name", "")
        spotify_artists = [a["name"] for a in item.get("artists", [])]
        spotify_url     = item.get("external_urls", {}).get("spotify")

        album_score  = _similarity(album, spotify_album)
        artist_score = max(_similarity(artist, sa) for sa in spotify_artists)

        if album_score >= SPOTIFY_MATCH_THRESHOLD and artist_score >= SPOTIFY_MATCH_THRESHOLD:
            return True, spotify_album, spotify_url

    return False, None, None


# ── Spotify Batch Verification ───────────────────────────────────────────────

def verify_recommendations(
    recommendations: list[dict],
    spotify_token: str | None,
    round_label: str = "Round 1",
) -> tuple[list[dict], list[dict]]:
    """
    Run a list of LLM recommendations through Spotify verification.

    Returns:
        (verified, failed)
        - verified: recs that passed, with canonical album/link attached
        - failed:   recs that were not found on Spotify (hallucinated)
    """
    if not spotify_token:
        # No Spotify configured — pass everything through unverified
        for rec in recommendations:
            rec["link"] = spotify_search_url(rec.get("artist", ""), rec.get("album", ""))
        return recommendations, []

    verified = []
    failed   = []

    for rec in recommendations:
        artist = rec.get("artist", "Unknown Artist")
        album  = rec.get("album", "Unknown Album")

        found, canonical_album, spotify_url = spotify_verify(artist, album, spotify_token)

        if not found:
            print(f"  ❌ [{round_label}] Not found on Spotify: {artist} — {album}")
            failed.append(rec)
            continue

        if canonical_album and canonical_album.lower() != album.lower():
            print(f"  ✏️  [{round_label}] Title corrected: '{album}' → '{canonical_album}'")

        rec["album"] = canonical_album or album
        rec["link"]  = spotify_url or spotify_search_url(artist, rec["album"])
        print(f"  ✅ [{round_label}] Verified: {artist} — {rec['album']}")
        verified.append(rec)

    return verified, failed


# ── Retry Prompt ──────────────────────────────────────────────────────────────

def build_retry_prompt(
    reviews: list[dict],
    verified: list[dict],
    failed: list[dict],
    needed: int,
) -> str:
    """
    Build a correction prompt that tells the LLM what passed and what was
    hallucinated, and asks it to provide replacements only for the failed ones.
    """
    verified_block = "\n".join(
        f"  - {r['artist']} — {r['album']} ✅" for r in verified
    ) or "  (none)"

    failed_block = "\n".join(
        f"  - {r['artist']} — {r['album']} ❌ (not found on Spotify — this album does not exist)"
        for r in failed
    )

    already_reviewed = ", ".join(
        f'"{r["artist"]} - {r["album"]}"' for r in reviews[:20]
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


# ── Submit Recommendations ────────────────────────────────────────────────────

def submit_recommendations(
    recommendations: list[dict],
    to_user_id: str,
    group_id: str,
    id_token: str,
    dry_run: bool,
):
    """Submit a pre-verified list of recommendations. All items should already have a 'link' field."""
    submitted = 0

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Submitting {len(recommendations)} verified recommendations...\n")

    for i, rec in enumerate(recommendations, 1):
        artist     = rec.get("artist", "Unknown Artist")
        album      = rec.get("album", "Unknown Album")
        reason     = rec.get("reason", "")
        confidence = rec.get("confidence", 0)
        link       = rec.get("link", spotify_search_url(artist, album))

        print(f"  [{i}/{len(recommendations)}] {artist} — {album} ({confidence:.0%} confidence)")
        print(f"  Reason: {reason}")
        print(f"  Link:   {link}")

        message = f"{artist} — {album}\n\n{reason}"
        payload = {
            "fromUserId": BOT_USER_ID,
            "toUserId":   to_user_id,
            "groupId":    group_id,
            "message":    message,
            "link":       link,
        }

        if dry_run:
            print("  ↳ Skipped (dry run)\n")
            submitted += 1
            continue

        result = call_firebase_function("submit_recommendation", payload, id_token)

        if result.get("success"):
            print(f"  ✅ Submitted — id: {result.get('recommendationId')}\n")
            submitted += 1
        else:
            print(f"  ⚠️  Unexpected response: {result}\n")

    print(f"  Summary: {submitted} {'would be ' if dry_run else ''}submitted")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AmongstFriends Music Recommender"
    )
    parser.add_argument(
        "user_id",
        help="Firebase userId to fetch reviews for and deliver recommendations to"
    )
    parser.add_argument(
        "--group-id",
        default="default-group",
        help="groupId passed to submit_recommendation (default: default-group)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch reviews and generate recs but skip submit_recommendation calls"
    )
    args = parser.parse_args()

    user_id  = args.user_id
    group_id = args.group_id

    print(f"\n🎵 AmongstFriends Recommender")
    print(f"   User:    {user_id}")
    print(f"   Group:   {group_id}")
    print(f"   Dry run: {args.dry_run}")
    print("─" * 50)

    # 1. Init Firebase Admin SDK
    print("\n1. Initializing Firebase...")
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    print("   ✅ Firebase initialized")

    # 2. Authenticate as LLMBot to get an ID token for callable functions
    print("\n2. Authenticating as LLMBot...")
    id_token = get_firebase_id_token()
    print("   ✅ ID token obtained")

    # 3. Fetch reviews via get_user_reviews callable function
    print("\n3. Fetching reviews...")
    response    = call_firebase_function("get_user_reviews", {"userId": user_id}, id_token)
    raw_reviews = response.get("reviews", [])

    if not raw_reviews:
        print("   No reviews found for this user. Exiting.")
        sys.exit(0)

    reviews = parse_reviews(raw_reviews)
    print(f"   Found {len(reviews)} reviews:")
    for r in sorted(reviews, key=lambda x: x["rating"], reverse=True):
        stars = "★" * r["rating"] + "☆" * (5 - r["rating"])
        print(f"   {stars}  {r['artist']} — {r['album']}")
        if r["text"]:
            print(f"          \"{r['text'][:80]}{'...' if len(r['text']) > 80 else ''}\"")

    if len(reviews) < MIN_REVIEWS_TO_RUN:
        print(f"\n⚠️  Only {len(reviews)} review(s) — need at least {MIN_REVIEWS_TO_RUN} to run.")
        sys.exit(0)

    # 4a. Call 1 — ask Ollama to suggest artists (no albums yet)
    print(f"\n4. Generating artist suggestions with Ollama ({OLLAMA_MODEL})...")
    artist_prompt = build_prompt_artists(reviews)
    suggested_artists = call_ollama_artists(artist_prompt)

    # Filter out any artists the user has already reviewed (model sometimes slips)
    reviewed_artist_names = {r["artist"].lower().strip() for r in reviews}
    suggested_artists = [
        a for a in suggested_artists
        if a.lower().strip() not in reviewed_artist_names
    ]

    print(f"   Got {len(suggested_artists)} new artist suggestions:")
    for a in suggested_artists:
        print(f"   • {a}")

    if not suggested_artists:
        print("⚠️  No new artists suggested. Exiting.")
        sys.exit(0)

    # 4b. Call 2 — ask Ollama for one specific real album per artist
    print(f"\n   Finding albums for each artist...")
    album_prompt    = build_prompt_albums(suggested_artists, reviews)
    recommendations = call_ollama(album_prompt)

    print(f"   Got {len(recommendations)} album recommendations:")
    for rec in recommendations:
        print(f"   • {rec.get('artist')} — {rec.get('album')} ({rec.get('confidence', 0):.0%} confidence)")

    # 5. Verify against Spotify (round 1)
    spotify_token = spotify_get_token()
    if spotify_token:
        print("\n5. Verifying recommendations against Spotify...")
    else:
        print("\n5. Skipping Spotify verification (credentials not configured)...")

    verified, failed = verify_recommendations(recommendations, spotify_token, round_label="Round 1")

    # 5b. Retry with reflection if anything failed and Spotify is configured
    if failed and spotify_token:
        print(f"\n   {len(failed)} failed verification — asking Ollama for replacements (round 2)...")
        retry_prompt = build_retry_prompt(reviews, verified, failed, needed=len(failed))
        retry_recs   = call_ollama(retry_prompt)
        retry_verified, retry_failed = verify_recommendations(retry_recs, spotify_token, round_label="Round 2")

        if retry_failed:
            print(f"   {len(retry_failed)} still failed after retry — dropping them.")
        if retry_verified:
            print(f"   {len(retry_verified)} replacement(s) verified and added.")

        verified = verified + retry_verified
    elif failed and not spotify_token:
        verified = verified + failed

    if not verified:
        print("\n⚠️  No verified recommendations to submit. Exiting.")
        sys.exit(0)

    print(f"\n   Final: {len(verified)} verified recommendation(s) ready to submit.")

    # 6. Submit verified recommendations
    print("\n6. Submitting recommendations...")
    submit_recommendations(
        recommendations=verified,
        to_user_id=user_id,
        group_id=group_id,
        id_token=id_token,
        dry_run=args.dry_run,
    )

    print("✅ Done!")


if __name__ == "__main__":
    main()