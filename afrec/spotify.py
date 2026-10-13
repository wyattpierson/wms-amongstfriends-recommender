"""Spotify verification of LLM-generated artist/album pairs.

This is the "ground truth" check that catches hallucinated albums. It is
optional: if no Spotify client credentials are configured, the pipeline simply
passes recommendations through unverified (link = a Spotify search URL).

The verifier behind a small `Verifier` callable interface so the pipeline does
not need to know Spotify exists — swap in a fake in tests, or another catalog
in the future.
"""

from __future__ import annotations

import os
import re
import urllib.parse

import requests


class SpotifyConfigError(RuntimeError):
    pass


def search_url(artist: str, album: str) -> str:
    """Fallback link: a Spotify search for the artist + album."""
    query = urllib.parse.quote(f"{artist} {album}")
    return f"https://open.spotify.com/search/{query}"


def get_token(client_id: str | None = None, client_secret: str | None = None) -> str | None:
    """
    Get a Spotify client-credentials access token.
    Returns None (not an error) if credentials aren't configured.
    """
    cid = client_id or os.getenv("SPOTIFY_CLIENT_ID")
    csecret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not csecret:
        return None
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(cid, csecret),
        timeout=10,
    )
    if not resp.ok:
        raise SpotifyConfigError(f"Could not get Spotify token: {resp.status_code} {resp.text}")
    return resp.json().get("access_token")


def _similarity(a: str, b: str) -> float:
    """
    Cheap character-overlap similarity (0.0–1.0), case-insensitive.
    Good enough for fuzzy album/artist matching without an external dep.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    matches = sum(1 for ch in shorter if ch in longer)
    return matches / len(longer)


class SpotifyAPIError(RuntimeError):
    """Spotify rejected the request itself (HTTP error, incl. 429 QUOTA_EXCEEDED).

    Distinct from "album not found" (a genuine miss): an API error says
    nothing about the album. Conflating the two silently drops good recs —
    the 2026-10-04 all-zero gemma run was half quota exhaustion, not
    hallucination. pipeline.VerificationGuard is the circuit breaker that
    turns these into a loud, flagged degrade instead of silent drops.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def verify(
    artist: str,
    album: str,
    token: str,
    match_threshold: float = 0.5,
) -> tuple[bool, str | None, str | None, int | None]:
    """
    Search Spotify for an album and check the result actually matches.

    Returns (found, canonical_album, spotify_url, release_year):
        - found:           True if a confident match was found
        - canonical_album: Spotify's official album title (prefer over LLM's)
        - spotify_url:     direct link to the album on Spotify
        - release_year:    4-digit year parsed from the match's release_date, or None

    Raises SpotifyAPIError when Spotify itself rejects the request (429
    quota, 5xx, network failure, unparseable body) — "the API is
    unavailable" is NOT "the album doesn't exist", and the two must not
    be conflated.
    """
    query = urllib.parse.quote(f"album:{album} artist:{artist}")
    try:
        resp = requests.get(
            f"https://api.spotify.com/v1/search?q={query}&type=album&limit=3",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if not resp.ok:
            reason = ""
            try:
                body = resp.json()
                err = body.get("error") or {}
                reason = str(err.get("reason") or err.get("message") or "")
            except ValueError:
                pass
            hint = " — daily quota exhausted?" if resp.status_code == 429 else ""
            raise SpotifyAPIError(
                f"Spotify API {resp.status_code} {('— ' + reason) if reason else ''}{hint} "
                f"(search '{album}' / {artist})",
                status_code=resp.status_code,
            )
        items = resp.json().get("albums", {}).get("items", [])
    except SpotifyAPIError:
        raise
    except requests.RequestException as e:
        raise SpotifyAPIError(f"Spotify API unreachable (search '{album}' / {artist}): {e}")
    except ValueError:
        raise SpotifyAPIError(f"Spotify API returned an unparseable body (search '{album}' / {artist})")
    if not items:
        return False, None, None, None

    for item in items:
        spotify_album = item.get("name", "")
        spotify_artists = [a.get("name", "") for a in item.get("artists", [])]
        spotify_url = item.get("external_urls", {}).get("spotify")

        album_score = _similarity(album, spotify_album)
        artist_score = max((_similarity(artist, sa) for sa in spotify_artists), default=0.0)

        if album_score >= match_threshold and artist_score >= match_threshold:
            return True, spotify_album, spotify_url, _release_year(item.get("release_date"))

    return False, None, None, None


def _release_year(release_date) -> int | None:
    """Parse a Spotify release_date ("YYYY-MM-DD" or "YYYY") into a 4-digit year."""
    if not release_date:
        return None
    head = str(release_date)[:4]
    return int(head) if head.isdigit() else None


def _base_name(name: str) -> str:
    """Album name without parentheticals: 'Settle (Deluxe)' → 'settle'.
    Spotify lists many releases only as 'X (Deluxe)'/'X (Remixes)'; the LLM
    (and every other catalog) says 'X'."""
    return re.sub(r"\s*\([^)]*\)", "", name).strip().lower()


def resolve_track_id(
    artist: str,
    album: str,
    token: str,
    match_threshold: float = 0.5,
) -> tuple[str | None, int | None]:
    """
    One Spotify track search for an (artist, album) → the top matching
    track's Spotify ID (what Musicae's audio-analysis needs as input) plus
    the album's release year, free from the same response (Last.fm's
    releaseDate is thin on electronic artists; Spotify's is reliable).

    Returns (None, None) when nothing matches confidently (a genuine miss).
    Raises SpotifyAPIError when Spotify itself rejects the request — same
    contract as verify(), so the enrichment latch degrades instead of
    burning quota on retries.
    """
    query = urllib.parse.quote(f"{album} {artist}")
    try:
        resp = requests.get(
            f"https://api.spotify.com/v1/search?q={query}&type=track&limit=10",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if not resp.ok:
            hint = " — daily quota exhausted?" if resp.status_code == 429 else ""
            raise SpotifyAPIError(
                f"Spotify API {resp.status_code} {hint} (track resolve '{album}' / {artist})",
                status_code=resp.status_code,
            )
        items = resp.json().get("tracks", {}).get("items", [])
    except SpotifyAPIError:
        raise
    except requests.RequestException as e:
        raise SpotifyAPIError(f"Spotify API unreachable (track resolve '{album}' / {artist}): {e}")
    except ValueError:
        raise SpotifyAPIError(f"Spotify API returned an unparseable body (track resolve '{album}' / {artist})")

    for item in items:
        album_name = (item.get("album") or {}).get("name", "")
        track_artists = [a.get("name", "") for a in item.get("artists", [])]
        album_score = max(_similarity(album, album_name),
                          _similarity(_base_name(album), _base_name(album_name)))
        artist_score = max((_similarity(artist, a) for a in track_artists), default=0.0)
        if album_score >= match_threshold and artist_score >= match_threshold:
            release = str((item.get("album") or {}).get("release_date") or "")
            year = int(release[:4]) if release[:4].isdigit() else None
            return item.get("id"), year
    return None, None


def verifier_for(
    token: str,
    match_threshold: float = 0.5,
) -> callable:
    """Build a callable `fn(artist, album) -> (found, canonical, url, release_year)` for the pipeline."""
    def _check(artist: str, album: str) -> tuple[bool, str | None, str | None, int | None]:
        return verify(artist, album, token, match_threshold)
    return _check
