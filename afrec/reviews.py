"""Normalization of raw review payloads from get_user_reviews.

Kept transport-agnostic: given the JSON list the Firebase function returns,
produce a flat, uniform shape the prompts and pipeline can rely on.
"""

from __future__ import annotations

# Known catalog types returned by get_user_reviews. Only types listed here are
# processed; others are silently skipped (and counted for reporting).
SUPPORTED_TYPES = {
    "album": "music",
    "restaurant": "restaurant",
    "book": "book",
    "other": "other",
}

ACTIVE_TYPE = "album"  # only this type is processed in the current run


def parse_reviews(raw_reviews: list, content_type: str = ACTIVE_TYPE) -> tuple[list[dict], dict[str, int]]:
    """
    Normalize raw review objects into a flat structure, filtering to only the
    requested content_type.

    Returns:
        (parsed, skipped)
        - parsed:  list of {reviewId, type, album, artist, rating, text, groupId}
        - skipped: mapping of skipped review type → count (e.g. {"restaurant": 2})
    """
    parsed: list[dict] = []
    skipped: dict[str, int] = {}

    for r in raw_reviews:
        catalog = r.get("catalog", {}) or {}
        review_type = catalog.get("type", "unknown")

        if review_type != content_type:
            skipped[review_type] = skipped.get(review_type, 0) + 1
            continue

        metadata = catalog.get("metadata", {}) or {}
        parsed.append({
            "reviewId": r.get("reviewId", ""),
            "type": review_type,
            "album": metadata.get("albumName") or catalog.get("name", "Unknown Album"),
            "artist": metadata.get("artistName", "Unknown Artist"),
            "rating": int(r.get("rating", 3)),
            "text": r.get("text", ""),
            "groupId": r.get("groupId", "default-group"),
        })

    return parsed, skipped


def summary_line(skipped: dict[str, int], content_type: str) -> str:
    """Human-readable summary of skipped (non-matching) reviews, or '' if none."""
    if not skipped:
        return ""
    parts = ", ".join(f"{count} {t}" for t, count in sorted(skipped.items()))
    return f"(skipped {parts} review(s) — not '{content_type}' type)"
