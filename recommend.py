#!/usr/bin/env python3
"""
AmongstFriends Music Recommendation Engine (production entry point).

Flow:
    Firebase get_user_reviews  →  LLM (artists → albums)  →  Spotify verify
    →  retry on hallucinations  →  Firebase submit_recommendation

All the interesting logic lives in the `afrec` package:
    afrec.prompts    — prompt builders (what you tweak)
    afrec.pipeline   — the two-call flow + verification + retry
    afrec.spotify    — the hallucination check
    afrec.llm        — the LLM client (llama.cpp llama-server / Ollama, swappable)
    afrec.firebase   — auth + callable functions (isolated)

To evaluate models/prompt changes without touching Firebase, see `eval/README.md`
(run `python eval/evaluate.py --models llama3,qwen2.5`).
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from afrec import firebase, pipeline, spotify
from afrec.llm import make_llm, LLMConnectionError, LLMTimeoutError, LLMHTTPError, LLMError
from afrec.reviews import parse_reviews, summary_line, ACTIVE_TYPE


def _min_reviews() -> int:
    return int(os.getenv("MIN_REVIEWS_TO_RUN", "3"))


def main():
    parser = argparse.ArgumentParser(description="AmongstFriends Music Recommender")
    parser.add_argument("user_id", help="Firebase userId to fetch reviews for and deliver recommendations to")
    parser.add_argument("--group-id", default=os.getenv("GROUP_ID", "default-group"),
                        help="groupId passed to submit_recommendation")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch reviews + generate recs but skip submit_recommendation")
    parser.add_argument("--no-spotify", action="store_true",
                        help="skip Spotify verification (pass recs through unverified)")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL"),
                        help="Model name (cosmetic for llama-server, which serves whatever GGUF you loaded)")
    args = parser.parse_args()

    user_id = args.user_id
    group_id = args.group_id

    print("\n🎵 AmongstFriends Recommender")
    print(f"   User:    {user_id}")
    print(f"   Group:   {group_id}")
    print(f"   Model:   {args.model}")
    print(f"   Dry run: {args.dry_run}")
    print("─" * 50)

    # 1. Firebase
    print("\n1. Initializing Firebase...")
    try:
        firebase.initialize()
    except firebase.FirebaseConfigError as e:
        print(f"❌ {e}")
        sys.exit(1)
    print("   ✅ Firebase initialized")

    print("\n2. Authenticating as LLMBot...")
    try:
        id_token = firebase.get_firebase_id_token()
    except (firebase.FirebaseError, firebase.FirebaseConfigError) as e:
        print(f"❌ {e}")
        sys.exit(1)
    print("   ✅ ID token obtained")

    # 2. Fetch + normalize reviews
    print("\n3. Fetching reviews...")
    try:
        raw_reviews = firebase.get_user_reviews(user_id, id_token)
    except firebase.FirebaseError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not raw_reviews:
        print("   No reviews found for this user. Exiting.")
        sys.exit(0)

    reviews, skipped = parse_reviews(raw_reviews)
    skip_note = summary_line(skipped, ACTIVE_TYPE)
    print(f"   Found {len(reviews)} {ACTIVE_TYPE} review(s)" + (f"  {skip_note}" if skip_note else ""))
    for r in sorted(reviews, key=lambda x: x["rating"], reverse=True):
        stars = "★" * r["rating"] + "☆" * (5 - r["rating"])
        print(f"   {stars}  {r['artist']} — {r['album']}")
        if r["text"]:
            print(f"          \"{r['text'][:80]}{'...' if len(r['text']) > 80 else ''}\"")

    if len(reviews) < _min_reviews():
        print(f"\n⚠️  Only {len(reviews)} review(s) — need at least {_min_reviews()} to run.")
        sys.exit(0)

    # 3. Build the LLM + optional verifier, then run the engine
    # Backend/URL come from LLM_BACKEND / LLM_URL in .env (default: llama.cpp
    # llama-server on localhost:8092). Override with --model if needed.
    llm = make_llm(args.model)

    verifier = None
    if not args.no_spotify:
        try:
            token = spotify.get_token()
        except spotify.SpotifyConfigError as e:
            print(f"⚠️  {e}")
            token = None
        if token:
            threshold = float(os.getenv("SPOTIFY_MATCH_THRESHOLD", "0.5"))
            verifier = spotify.verifier_for(token, threshold)

    print(f"\n4. Generating recommendations with {llm.name} ...")
    try:
        outcome = pipeline.run(reviews, llm, verifier=verifier, log=print)
    except LLMConnectionError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except LLMTimeoutError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except LLMHTTPError as e:
        print(f"❌ {e}\n   {e.body}")
        sys.exit(1)
    except LLMError as e:
        print(f"❌ LLM error: {e}")
        sys.exit(1)

    if not outcome.ok:
        print("⚠️  No recommendations survived. Exiting.")
        for st in outcome.stages:
            if not st.ok:
                print(f"   stage '{st.stage}': {st.error}")
        sys.exit(0)

    print(f"\n   Final: {len(outcome.recs)} recommendation(s)"
          + (f" ({outcome.verified_count} Spotify-verified, {outcome.failed_count} dropped)" if verifier else ", unverified"))
    for rec in outcome.recs:
        print(f"   • {rec.artist} — {rec.album} ({rec.confidence:.0%})"
              + ("" if rec.verified else "  [unverified]"))

    # 4. Submit
    print("\n5. " + ("Would submit (dry run)" if args.dry_run else "Submitting recommendations..."))
    firebase.submit_many(
        outcome.recs,
        to_user_id=user_id,
        group_id=group_id,
        id_token=id_token,
        dry_run=args.dry_run,
    )

    print("✅ Done!")


if __name__ == "__main__":
    main()
