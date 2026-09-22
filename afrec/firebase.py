"""All Firebase interaction, isolated in one module.

Keeping Firebase here (auth, callable functions, reads/writes) means the rest
of the engine — prompts, LLM calls, Spotify checks, orchestration — runs with
zero Firebase dependency. That's what lets eval/evaluate.py try models and prompt
revisions offline against saved review profiles.

Authentication pattern (server-to-server for callable functions):
  1. Service account mints a custom token for the bot user (BOT_USER_ID)
  2. Custom token exchanged for a real Firebase ID token via Identity Toolkit
  3. ID token used as Bearer auth on the callable functions
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import firebase_admin
from firebase_admin import credentials, auth

# afrec/firebase.py → REPO_ROOT is one level up (…/amongstfriends-recommender).
try:  # prefer python-dotenv; fall back to a tiny parser (keeps firebase.py lightweight)
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(path):  # type: ignore[no-redef]
        p = Path(path)
        if not p.exists():
            return
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


class FirebaseConfigError(RuntimeError):
    pass


class FirebaseError(RuntimeError):
    pass


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise FirebaseConfigError(f"Missing required config: {key} (add it to .env)")
    return val


# ── Config ────────────────────────────────────────────────────────────────────

def _default_service_account_path() -> str:
    """Env value if set, else the key at the repo root (resolve relative values against the repo root too)."""
    val = os.getenv("SERVICE_ACCOUNT_PATH", "")
    if val:
        p = Path(val).expanduser()
        return str(p if p.is_absolute() else (REPO_ROOT / p))
    fallback = REPO_ROOT / "serviceAccountKey.json"
    default = str(fallback) if fallback.exists() else "serviceAccountKey.json"
    os.environ["SERVICE_ACCOUNT_PATH"] = default   # so _require()/ensure_configured() see it
    return default


SERVICE_ACCOUNT_PATH = _default_service_account_path()
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
FIREBASE_REGION = os.getenv("FIREBASE_REGION", "us-central1")
FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY", "")
BOT_USER_ID = os.getenv("BOT_USER_ID", "oPduqxdtQ62yvTVm4KPO")


def ensure_configured() -> None:
    """Raise FirebaseConfigError naming the first missing env var, if any."""
    for key in ("FIREBASE_PROJECT_ID", "FIREBASE_WEB_API_KEY", "SERVICE_ACCOUNT_PATH"):
        _require(key)
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise FirebaseConfigError(f"Service account key not found at: {SERVICE_ACCOUNT_PATH}")


def initialize() -> None:
    """Lazily initialize the Firebase Admin SDK (idempotent)."""
    if firebase_admin._apps:
        return
    ensure_configured()
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)


def get_firebase_id_token() -> str:
    """Mint a custom token for the bot user and exchange it for a Firebase ID token."""
    initialize()
    custom_token: bytes = auth.create_custom_token(BOT_USER_ID)

    exchange_url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
        f"?key={FIREBASE_WEB_API_KEY}"
    )
    resp = requests.post(
        exchange_url,
        json={"token": custom_token.decode("utf-8"), "returnSecureToken": True},
        timeout=15,
    )
    if not resp.ok:
        raise FirebaseError(
            f"Failed to exchange custom token for ID token. "
            f"Status {resp.status_code}: {resp.text} "
            f"(double-check FIREBASE_WEB_API_KEY)"
        )
    return resp.json()["idToken"]


def call_function(function_name: str, payload: dict, id_token: str | None = None) -> dict:
    """
    Call a Firebase HTTPS callable function.
    Body is wrapped as {data: payload}; the return is unwrapped from {result: ...}.
    Caller supplies the id_token (see get_firebase_id_token) to avoid re-minting per call.
    """
    url = (
        f"https://{FIREBASE_REGION}-{FIREBASE_PROJECT_ID}"
        f".cloudfunctions.net/{function_name}"
    )
    headers = {"Content-Type": "application/json"}
    if id_token:
        headers["Authorization"] = f"Bearer {id_token}"
    try:
        resp = requests.post(url, json={"data": payload}, headers=headers, timeout=30)
    except requests.exceptions.ConnectionError as e:
        raise FirebaseError(f"Cannot reach Firebase function '{function_name}'. Check connectivity.") from e

    if not resp.ok:
        raise FirebaseError(
            f"Firebase function '{function_name}' returned HTTP {resp.status_code}: {resp.text}"
        )
    outer = resp.json()
    return outer.get("result", outer)


def get_user_reviews(user_id: str, id_token: str) -> list:
    """Fetch the raw review list for a user via the get_user_reviews callable."""
    response = call_function("get_user_reviews", {"userId": user_id}, id_token)
    return response.get("reviews", []) or []


def submit_recommendation(payload: dict, id_token: str) -> dict:
    """Submit one recommendation via the submit_recommendation callable."""
    return call_function("submit_recommendation", payload, id_token)


def submit_many(
    recs: list,
    to_user_id: str,
    group_id: str,
    id_token: str,
    *,
    dry_run: bool = False,
    from_user_id: str | None = None,
    log=print,
) -> int:
    """
    Submit a list of already-verified Rec objects. Returns count submitted.
    Each rec needs artist/album/reason/confidence/link.
    """
    from_user_id = from_user_id or BOT_USER_ID
    submitted = 0
    log(f"\n{'[DRY RUN] ' if dry_run else ''}Submitting {len(recs)} recommendation(s)...\n")

    for i, rec in enumerate(recs, 1):
        artist = rec.artist
        album = rec.album
        reason = rec.reason
        confidence = rec.confidence
        link = rec.link

        log(f"  [{i}/{len(recs)}] {artist} — {album} ({confidence:.0%} confidence)")
        if reason:
            log(f"  Reason: {reason}")
        log(f"  Link:   {link}")

        if dry_run:
            log("  ↳ skipped (dry run)\n")
            submitted += 1
            continue

        message = f"{artist} — {album}\n\n{reason}".strip()
        payload = {
            "fromUserId": from_user_id,
            "toUserId": to_user_id,
            "groupId": group_id,
            "message": message,
            "link": link,
        }
        result = submit_recommendation(payload, id_token)
        if result.get("success"):
            log(f"  ✅ submitted — id: {result.get('recommendationId')}\n")
            submitted += 1
        else:
            log(f"  ⚠️ unexpected response: {result}\n")

    log(f"  Summary: {submitted} {'would be ' if dry_run else ''}submitted")
    return submitted
