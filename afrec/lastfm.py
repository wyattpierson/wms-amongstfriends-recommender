"""Last.fm — the *label* layer: what humans say an artist/album is.

Name-keyed (artist/album strings), free API key, ~50 req/s — cheap enough
that we never batch or cache at this layer's own rate budget; the disk
cache lives in afrec/enrich.py on top of us.

Division of labor: **Last.fm = identity, Musicae = measurement.** Tags are
crowd-sourced folksonomy (noisy, sometimes stale), so every string leaves
this module through clean_label and tags carry a minimum-count threshold.

Auth: LASTFM_API_KEY from the environment (.env). No OAuth for the
methods we use.
"""

from __future__ import annotations

import os

import requests

from .sanitize import clean_label

BASE = "https://ws.audioscrobbler.com/2.0/"
TIMEOUT_S = 10
MIN_TAG_COUNT = 2   # folksonomy noise floor: 1-vote tags are single-user opinions


class LastfmError(RuntimeError):
    """Last.fm rejected the request (HTTP error, network, API error body).

    Distinct from "artist/album not found" (a genuine miss, returned as
    None by the lookups below).
    """


def api_key() -> str | None:
    return os.getenv("LASTFM_API_KEY") or None


def _call(method: str, **params) -> dict:
    key = api_key()
    if not key:
        raise LastfmError("LASTFM_API_KEY not set (add it to .env)")
    params.update(method=method, api_key=key, format="json")
    try:
        resp = requests.get(BASE, params=params, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise LastfmError(f"Last.fm unreachable ({method}): {e}")
    if resp.status_code == 429:
        raise LastfmError(f"Last.fm 429 rate-limited ({method})")
    if not resp.ok:
        raise LastfmError(f"Last.fm HTTP {resp.status_code} ({method})")
    try:
        body = resp.json()
    except ValueError:
        raise LastfmError(f"Last.fm returned an unparseable body ({method})")
    if body.get("error"):
        raise LastfmError(
            f"Last.fm error {body['error']}: {str(body.get('message', ''))[:120]} ({method})"
        )
    return body


def _tags(node: dict | None, *, limit: int, min_count: int = MIN_TAG_COUNT) -> list[tuple[str, int]]:
    """[(clean_tag, count)…], by count desc, noise floor applied, capped.

    Last.fm shape drift: artist.getinfo returns tags WITHOUT counts (and the
    node is 'tags', not 'toptags'); artist.getTopTags / album responses carry
    counts. So the min_count noise floor applies only to COUNTED lists — an
    uncounted list is already server-ranked and is kept as-is."""
    entries = []
    for t in (node or {}).get("tag", []):
        name = clean_label(t.get("name"))
        if not name:
            continue
        raw = t.get("count")
        entries.append((name, int(raw) if raw is not None else None))
    counted = [(n, c) for n, c in entries if c is not None and c >= min_count]
    if counted:
        counted.sort(key=lambda tc: (-tc[1], tc[0]))
        return counted[:limit]
    return [(n, 0) for n, _c in entries][:limit]


def artist_info(artist: str, *, tag_limit: int = 8, similar_limit: int = 10) -> dict | None:
    """Sanitized artist facts: name, top (tag, count) pairs, similar artists.

    Returns None when Last.fm has no such artist (a genuine miss, not an
    error). Raises LastfmError on API failure.
    """
    body = _call("artist.getinfo", artist=artist, inc="tags+similar")
    a = body.get("artist") or {}
    if not a.get("name"):
        return None
    similar = [clean_label(s.get("name")) for s in a.get("similar", {}).get("artist", [])]
    return {
        "name": clean_label(a.get("name")),
        "tags": (_tags(a.get("toptags"), limit=tag_limit)
                 or _tags(a.get("tags"), limit=tag_limit)),
        "similar": [s for s in similar if s][:similar_limit],
    }


def album_info(artist: str, album: str, *, tag_limit: int = 8) -> dict | None:
    """Sanitized album facts: name, release year (→ era), top tags, mbid.

    Returns None when Last.fm has no such album. Raises LastfmError on API failure.
    """
    body = _call("album.getinfo", artist=artist, album=album)
    a = body.get("album") or {}
    if not a.get("name"):
        return None
    release_date = a.get("releaseDate")
    if isinstance(release_date, dict):        # {"#text": "1998-05-22", "precision": ...}
        release_date = release_date.get("#text")
    release_date = str(release_date or "")
    year = int(release_date[:4]) if release_date[:4].isdigit() else None
    mbid = a.get("mbid") or None
    if mbid and len(mbid) != 36:      # defensive: MBIDs are UUIDs
        mbid = None
    artist_field = a.get("artist")     # sometimes a dict, sometimes a bare string
    artist_name = artist_field.get("name") if isinstance(artist_field, dict) else (artist_field or artist)
    return {
        "name": clean_label(a.get("name")),
        "artist": clean_label(artist_name),
        "year": year,
        "tags": (_tags(a.get("toptags"), limit=tag_limit)
                 or _tags(a.get("tags"), limit=tag_limit)),
        "mbid": mbid,
    }
