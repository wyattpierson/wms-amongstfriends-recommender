"""Musicae.io (via RapidAPI) — the *measurement* layer: how it sounds.

The only free-ish source left for objective audio measurements (Spotify
killed its own audio-features endpoints in 2024 — that's literally why
Musicae exists). Distributed through RapidAPI: one X-RapidAPI-Key +
X-RapidAPI-Host header pair, no OAuth.

**Free tier: 10 requests/day.** Batches of ≤5 track IDs per call. This
module is deliberately quota-dumb: it counts nothing. Budget, latching,
and the disk cache live in afrec/enrich.py — the only place that should
ever decide "do we fire this call".

Auth: MUSICAE_API_KEY from the environment (.env). NOTE: the key must be
SUBSCRIBED to the "DJ Track Audio Analysis API" on RapidAPI (free plan,
no card) — a wrong or unsubscribed key returns 403 "not subscribed"
(verified 2026-10-05: the 403 gives no hint which of the two it is).

Endpoint (verified live 2026-10-05): GET /v2/audio-analysis?ids=a,b,c
(≤5 comma-separated Spotify track IDs) → {"audio_analysis": [entry, ...]}
where entry = {track, rhythm, harmony, score, mood, genres}. There is NO
POST /v2/audio-analysis (404) — an earlier draft of this module assumed
one. Single-track alternative: GET /v2/audio-analysis/{id}.
"""

from __future__ import annotations

import os

import requests

from .sanitize import clean_label

HOST = "dj-track-audio-analysis-api.p.rapidapi.com"
BASE = f"https://{HOST}"
TIMEOUT_S = 20
BATCH_SIZE = 5          # ≤5 track IDs per /v2/audio-analysis call

# The 7 Spotify-identical continuous features (the fit math's v1 set).
SPOTIFY_FEATURES = (
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
)


class MusicaeError(RuntimeError):
    """Musicae/RapidAPI rejected the request (HTTP error, network, quota).

    Mirrors SpotifyAPIError: "the API is unavailable" is NOT "the track
    doesn't exist" (a genuine miss → missing entry in the result list).
    enrich.Enricher's latch turns the first failure into a loud one-time
    degrade instead of per-call noise.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def api_key() -> str | None:
    return os.getenv("MUSICAE_API_KEY") or None


def _headers() -> dict:
    key = api_key()
    if not key:
        raise MusicaeError("MUSICAE_API_KEY not set (add it to .env)")
    return {"X-RapidAPI-Key": key, "X-RapidAPI-Host": HOST,
            "accept": "application/json"}


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_track(raw: dict) -> dict | None:
    """Extract the whitelist of fields we keep from one raw analysis entry.
    Everything else in the response is discarded. Missing sections degrade
    to None — the response schema is verified against the live API by
    `python eval/backfill.py smoke`, and this parser must never raise on
    shape drift, only return less.
    """
    t = (raw or {}).get("track") or {}
    ids = t.get("ids") or {}
    track_id = ids.get("spotify") or raw.get("track_id") or None

    rhythm = (raw or {}).get("rhythm") or {}
    harmony = (raw or {}).get("harmony") or {}
    score = (raw or {}).get("score") or {}
    mood = (raw or {}).get("mood") or {}

    features = {k: _num(score.get(k)) for k in SPOTIFY_FEATURES}
    features = {k: v for k, v in features.items() if v is not None}

    return {
        "track_id": track_id,
        "name": clean_label(t.get("name")),
        "duration_ms": _num(t.get("duration_ms")) or None,
        "loudness": _num(t.get("loudness")),
        "bpm": _num(rhythm.get("bpm")),
        "key": _num(harmony.get("key")),
        "mode": clean_label(harmony.get("mode"), maxlen=12),
        "camelot": clean_label(harmony.get("camelot"), maxlen=6),
        "mood": clean_label(mood.get("label"), maxlen=24),
        "features": features or None,
        "genres": [g for g in (clean_label(x, maxlen=30) for x in (raw or {}).get("genres") or []) if g][:3],
    }


def audio_analysis(track_ids: list[str]) -> list[dict]:
    """Audio analysis for ≤N Spotify track IDs, batched 5 per call.

    Returns one parsed dict per ID found (order of input, misses skipped).
    Raises MusicaeError on API failure (403 not-subscribed, 429 quota,
    network). Each batch is one request → callers must budget in batches,
    not in IDs.
    """
    ids = [i for i in dict.fromkeys(track_ids) if i]
    out: list[dict] = []
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i:i + BATCH_SIZE]
        try:
            # GET with comma-separated ids (≤5) — verified live shape.
            resp = requests.get(
                f"{BASE}/v2/audio-analysis",
                params={"ids": ",".join(batch)},
                headers=_headers(),
                timeout=TIMEOUT_S,
            )
        except requests.RequestException as e:
            raise MusicaeError(f"Musicae unreachable: {e}")
        if not resp.ok:
            try:
                detail = str(resp.json().get("message") or resp.text[:120])
            except ValueError:
                detail = resp.text[:120]
            hint = " — daily quota exhausted?" if resp.status_code == 429 else (
                " — key not subscribed to this API on RapidAPI?" if resp.status_code == 403 else "")
            raise MusicaeError(
                f"Musicae API {resp.status_code} {detail}{hint} (batch of {len(batch)})",
                status_code=resp.status_code,
            )
        try:
            body = resp.json()
        except ValueError:
            raise MusicaeError(f"Musicae returned an unparseable body (batch of {len(batch)})")
        entries = (body.get("audio_analysis") or body.get("tracks")
                   if isinstance(body, dict) else body)
        if not isinstance(entries, list):
            # Shape drift: keep the raw shape visible for the smoke test,
            # degrade to nothing rather than raising mid-run.
            raise MusicaeError(
                f"Musicae unexpected response shape (batch of {len(batch)}): {str(body)[:160]}"
            )
        for e in entries:
            parsed = _parse_track(e if isinstance(e, dict) else {})
            if parsed:
                out.append(parsed)
    return out
