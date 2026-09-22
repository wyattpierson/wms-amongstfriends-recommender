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


def verify(
    artist: str,
    album: str,
    token: str,
    match_threshold: float = 0.5,
) -> tuple[bool, str | None, str | None]:
    """
    Search Spotify for an album and check the result actually matches.

    Returns (found, canonical_album, spotify_url):
        - found:           True if a confident match was found
        - canonical_album: Spotify's official album title (prefer over LLM's)
        - spotify_url:     direct link to the album on Spotify
    """
    query = urllib.parse.quote(f"album:{album} artist:{artist}")
    resp = requests.get(
        f"https://api.spotify.com/v1/search?q={query}&type=album&limit=3",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if not resp.ok:
        return False, None, None

    items = resp.json().get("albums", {}).get("items", [])
    if not items:
        return False, None, None

    for item in items:
        spotify_album = item.get("name", "")
        spotify_artists = [a.get("name", "") for a in item.get("artists", [])]
        spotify_url = item.get("external_urls", {}).get("spotify")

        album_score = _similarity(album, spotify_album)
        artist_score = max((_similarity(artist, sa) for sa in spotify_artists), default=0.0)

        if album_score >= match_threshold and artist_score >= match_threshold:
            return True, spotify_album, spotify_url

    return False, None, None


def verifier_for(
    token: str,
    match_threshold: float = 0.5,
) -> callable:
    """Build a callable `fn(artist, album) -> (found, canonical, url)` for the pipeline."""
    def _check(artist: str, album: str) -> tuple[bool, str | None, str | None]:
        return verify(artist, album, token, match_threshold)
    return _check
