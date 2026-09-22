#!/usr/bin/env python3
"""
Make an eval test case out of a REAL user's Firebase profile.

Fetches the user's raw reviews (`get_user_reviews`, the same call as
`recommend.py`, via the bot ID token) and saves them as
`eval/fixtures/<profile>.json` — keeping only the MUSIC/ALBUM entries, since
that is exactly what the engine feeds to the LLM (`parse_reviews` filters on
`ACTIVE_TYPE="album"`; books/restaurants/`other` never reach the prompts). So
the fixture stays music-only and production-faithful.

It also scaffolds a companion `eval/golden/<profile>.json` seed (empty
good/bad lists + a pointer to grow it via --interactive labels), and prints a
summary of what was saved so you can eyeball the profile before judging.

Nothing else happens — no LLM, no Spotify, no writes back to Firebase
(read-only: exactly the `get_user_reviews` callable, the same call
`recommend.py` makes in step 2).

Usage (from the repo root, or anywhere):
  venv/bin/python eval/make_fixture.py --user-id oPd... --profile alex
  venv/bin/python eval/make_fixture.py --user-id oPd... --profile alex --force  # re-fetch

Then, to start judging:
  venv/bin/python eval/evaluate.py --interactive --profile alex --models llama3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths & imports (same pattern as evaluate.py) ─────────────────────────
EVAL_DIR = Path(__file__).resolve().parent        # …/amongstfriends-recommender/eval
REPO_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(REPO_ROOT))                # for `afrec.*`

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from afrec import firebase
from afrec.reviews import parse_reviews, ACTIVE_TYPE


def is_music(r: dict) -> bool:
    """True if this raw review is a music/album entry (the only kind the engine consumes)."""
    return (r.get("catalog") or {}).get("type") == ACTIVE_TYPE


def filter_music(raw_reviews: list) -> list:
    """Keep only album/music reviews — drops books, restaurants, `other`, etc.

    afrec.reviews.parse_reviews (ACTIVE_TYPE="album") already ignores non-music
    entries, so this is exactly the subset the production engine feeds to the LLM.
    Storing only that subset keeps the fixture music-only and unambiguous.
    """
    return [r for r in raw_reviews if is_music(r)]


def _slug(name: str) -> str:
    """Filesystem/profile name: lowercase alnum with underscores."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "profile"


def write_fixture(profile: str, user_id: str, description: str, music_reviews: list,
                  force: bool = False) -> Path:
    fixtures_dir = EVAL_DIR / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / f"{profile}.json"
    if path.exists() and not force:
        sys.exit(f"❌ {path} already exists — refusing to overwrite. "
                 f"Re-run with --force to re-fetch and replace it.")
    doc = {
        "profile": profile,
        "description": description or f"Real user profile fetched from Firebase ({profile}).",
        "source": "firebase",
        "source_user_id": user_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": "Fetched via eval/make_fixture.py from get_user_reviews, kept to MUSIC/ALBUM "
                 "reviews only (books/restaurants/`other` stripped) — exactly the rows the "
                 "production engine feeds to the LLM. Do not hand-edit reviews; re-fetch "
                 "with --force instead.",
        "reviews": music_reviews,
    }
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path


def write_golden_seed(profile: str, user_id: str) -> Path:
    golden_dir = EVAL_DIR / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    path = golden_dir / f"{profile}.json"
    if path.exists():
        print(f"   (golden for '{profile}' already exists — left untouched)")
        return path
    doc = {
        "profile": profile,
        "description": f"Good/bad picks for the {profile} profile (real user {user_id}).",
        "annotated_by": "wyatt (seed — fill from interactive labels)",
        "annotated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "notes": "EMPTY SEED — 0 good / 0 bad means the golden column scores +0.000 "
                 "for everyone (no discrimination yet). Judge the engine with "
                 "`evaluate.py --interactive --profile " + profile + "` and promote the "
                 "artists that keep coming up good/bad here. Entry = string OR "
                 "{\"artist\": ..., \"album\": ... (optional), \"note\": ...}.",
        "good": [],
        "bad": [],
    }
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save a real Firebase user's reviews as an eval fixture (+ golden seed).")
    parser.add_argument("--user-id", required=True, help="Firebase userId (same value you'd pass to recommend.py)")
    parser.add_argument("--profile", help="Name for the fixture (default: the user id). This becomes --profile for evaluate.py.")
    parser.add_argument("--description", default="", help="One line describing the listener, for the leaderboard/README.")
    parser.add_argument("--no-golden", action="store_true", help="Skip scaffolding golden/<profile>.json")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing fixtures/<profile>.json (re-fetch replaces it)")
    args = parser.parse_args()

    profile = _slug(args.profile or args.user_id)

    print("\n📥 Fetching real profile → eval fixture")
    print(f"   User:        {args.user_id}")
    print(f"   Profile:     {profile}")

    try:
        firebase.initialize()
        id_token = firebase.get_firebase_id_token()
    except (firebase.FirebaseConfigError, firebase.FirebaseError) as e:
        print(f"❌ Firebase: {e}")
        return 1

    try:
        raw = firebase.get_user_reviews(args.user_id, id_token)
    except firebase.FirebaseError as e:
        print(f"❌ get_user_reviews: {e}")
        return 1

    if not raw:
        print(f"❌ No reviews found for user {args.user_id} — nothing to fixture.")
        return 1

    # The engine only consumes music/album reviews (ACTIVE_TYPE); books/restaurants/
    # `other` never reach the prompts. Keep only that subset so the fixture is
    # music-only and matches exactly what the production engine sees.
    music = filter_music(raw)
    dropped = len(raw) - len(music)
    print(f"   music/album reviews: {len(music)}"
          + (f"  ({dropped} non-music reviews excluded)" if dropped else ""))

    if not music:
        print(f"⚠️  User has {len(raw)} review(s) but none are music/albums — "
              f"nothing to recommend from. Aborting.")
        return 1

    reviews, _skipped = parse_reviews(music)
    if len(reviews) < 3:
        print(f"⚠️  Only {len(reviews)} album review(s) — below MIN_REVIEWS_TO_RUN (3); "
              f"recommend.py would refuse this user. Saving anyway.")

    print("\n   Albums by rating:")
    for r in sorted(reviews, key=lambda x: x["rating"], reverse=True)[:10]:
        stars = "★" * r["rating"] + "☆" * (5 - r["rating"])
        print(f"   {stars}  {r['artist']} — {r['album']}")

    fixture_path = write_fixture(profile, args.user_id, args.description, music,
                                 force=args.force)
    print(f"\n✅ Fixture   → {fixture_path}   ({len(music)} music reviews)")

    golden_path = None if args.no_golden else write_golden_seed(profile, args.user_id)
    if golden_path:
        print(f"✅ Golden    → {EVAL_DIR / 'golden' / f'{profile}.json'}  (empty seed — fill as you judge)")

    print(f"""
Next steps:
  # Judge it live (labels grow golden/labeled.jsonl):
  venv/bin/python eval/evaluate.py --interactive --profile {profile} --models llama3

  # Or sweep models against it:
  venv/bin/python eval/evaluate.py --models llama3,mistral --profile {profile}

As you label, promote repeat good/bad artists into:
  eval/golden/{profile}.json
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
