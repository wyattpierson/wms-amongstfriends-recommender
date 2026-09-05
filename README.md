# AmongstFriends — LLM Recommendation Engine

A local Python script that reads a user's reviews from the AmongstFriends Firebase backend, runs them through a local Ollama LLM to generate personalized recommendations, and submits those recommendations back via Firebase Functions.

Currently supports: **music (albums)**. Restaurants, books, and other types are stubbed for future expansion.

---


## To run
* source venv/bin/activate
* pip install python-dotenv
  * I don't think this is needed, but just in case
* python recommend.py Jdwvuss3sFch1QiBvT7BanC0CI92 --dry-run



## How it works

```
get_user_reviews()        Ollama (local)        submit_recommendation()
  Firebase Function   →   llama3 / mistral   →   Firebase Function
  (fetch album reviews)   (generate recs)         (one call per rec)
```

1. Authenticates as `LLMBot` using your service account key
2. Calls `get_user_reviews` to fetch all reviews for a given user
3. Filters to album reviews only, ignores restaurants/books/etc.
4. Sends the review list to your local Ollama model
5. Calls `submit_recommendation` once for each recommendation returned

---

## Prerequisites

### 1. Python 3.10+

Check with:
```bash
python3 --version
```

If you need to install or upgrade: https://www.python.org/downloads/

### 2. Ollama running locally

Install from https://ollama.com and pull a model:
```bash
ollama pull llama3       # recommended starting point on 64GB RAM
# or
ollama pull mixtral      # stronger but slower
# or
ollama pull mistral      # good balance of speed and quality
```

Make sure it's running before you use the script:
```bash
ollama serve             # start the server (runs on localhost:11434)
ollama list              # confirm your model is available
```

> You don't need to run `ollama serve` manually if the Ollama desktop app is open — it runs the server automatically.

### 3. Firebase service account key

This is how the script authenticates to Firebase without any user login flow.

1. Go to [Firebase Console](https://console.firebase.google.com) → your project → **Project Settings** → **Service Accounts**
2. Click **Generate new private key** → confirm → download the JSON file
3. Rename it to `serviceAccountKey.json` and place it in the same folder as `recommend.py`

> ⚠️ Never commit this file to git. Add `serviceAccountKey.json` to your `.gitignore`.

### 4. Firebase Web API Key

This is a separate value needed to exchange auth tokens. It's not a secret (it's embedded in iOS apps), but keep it out of public repos anyway.

Find it at: **Firebase Console → Project Settings → General → Web API key**

### 5. Firebase Functions deployed

The script calls two Firebase callable functions that must already be deployed in your project:

| Function | What it does |
|---|---|
| `get_user_reviews` | Returns all reviews for a given userId |
| `submit_recommendation` | Creates a recommendation from LLMBot to a user |

If either function isn't deployed yet, the script will fail at that step with an HTTP error.

---

## Setup

### Install Python dependencies

```bash
pip install firebase-admin requests
```

Or if you prefer a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
pip install firebase-admin requests
```

### Configure the script

Open `recommend.py` and fill in the four values near the top:

```python
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"   # path to your key file — fine to leave as-is if it's in the same folder
FIREBASE_PROJECT_ID  = "your-project-id"           # e.g. "amongstfriends-prod"
FIREBASE_REGION      = "us-central1"               # change only if your functions are deployed elsewhere
FIREBASE_WEB_API_KEY = "your-firebase-web-api-key" # from Firebase Console → Project Settings → General
OLLAMA_MODEL         = "llama3"                    # must match a model you have pulled locally
```

Your folder should look like this when you're done:
```
recommend.py
serviceAccountKey.json    ← downloaded from Firebase
README.md
```

---

## Running

### Always dry-run first

This fetches reviews and generates recommendations but does **not** call `submit_recommendation`. Safe to run as many times as you want.

```bash
python3 recommend.py <userId> --dry-run
```

You'll see the full output — which reviews were found, what the LLM recommended, and what would have been submitted — without actually writing anything.

### Run for real

```bash
python3 recommend.py <userId>
```

### With a specific group

If you want recommendations submitted to a specific group (instead of `default-group`):

```bash
python3 recommend.py <userId> --group-id <groupId>
```

### Finding a userId

User IDs are the Firebase Auth UIDs. You can find them in:
- **Firebase Console → Authentication → Users** — listed in the UID column
- Or from the example in the codebase: `feyXuB6dyGf7LZWUF1mQ5lSY96g2`

### Full example

```bash
# Dry run for a specific user
python3 recommend.py feyXuB6dyGf7LZWUF1mQ5lSY96g2 --dry-run

# Real run
python3 recommend.py feyXuB6dyGf7LZWUF1mQ5lSY96g2

# Real run with a specific group
python3 recommend.py feyXuB6dyGf7LZWUF1mQ5lSY96g2 --group-id some-group-id
```

---

## What the output looks like

```
🎵 AmongstFriends Recommender
   User:    feyXuB6dyGf7LZWUF1mQ5lSY96g2
   Group:   default-group
   Dry run: False
──────────────────────────────────────────────────

1. Initializing Firebase...
   ✅ Firebase initialized

2. Authenticating as LLMBot...
   ✅ ID token obtained

3. Fetching reviews...
   Found 4 album reviews:
   (Skipped 2 restaurant, 1 book review(s) — not 'album' type)
   ★★★★★  Frank Ocean — Blonde
   ★★★★☆  Bon Iver — For Emma, Forever Ago
   ★★★☆☆  Holo — Astro
          "Very nicely lowkey vibey electronic. Almost great but doesn't..."
   ★★☆☆☆  Coldplay — Music of the Spheres

4. Generating recommendations with Ollama (llama3)...
   Got 3 recommendations:
   • Sufjan Stevens — Carrie & Lowell (91% confidence)
   • James Blake — Overgrown (88% confidence)
   • Grouper — Ruins (85% confidence)

5. Submitting recommendations...

  [1/3] Sufjan Stevens — Carrie & Lowell (91% confidence)
  Reason:  Shares the same quiet emotional intensity as Blonde...
  Link:    https://open.spotify.com/search/Sufjan%20Stevens%20Carrie%20%26%20Lowell
  ✅ Submitted — id: rec_abc123

  [2/3] James Blake — Overgrown (88% confidence)
  ...

✅ Done!
```

---

## Troubleshooting

**`❌ Cannot reach Ollama`**
Ollama isn't running. Either open the Ollama desktop app or run `ollama serve` in a terminal.

**`❌ Failed to exchange custom token for ID token`**
Your `FIREBASE_WEB_API_KEY` is wrong or missing. Double-check it in Firebase Console → Project Settings → General.

**`❌ Firebase function 'get_user_reviews' returned HTTP 403`**
The function is rejecting the `LLMBot` identity. Check that your Firebase Function doesn't whitelist specific UIDs — you may need to add a bypass for `LLMBot` in the function code.

**`❌ Firebase function 'get_user_reviews' returned HTTP 404`**
The function name doesn't match what's deployed, or it's deployed in a different region. Check `FIREBASE_REGION` in the script config.

**`❌ Could not parse Ollama response as JSON`**
The model returned something the script couldn't parse. This usually happens with smaller/older models that don't follow JSON instructions reliably. Try `llama3` or `mixtral` if you're on a smaller model. The raw output will be printed so you can see what it returned.

**`⚠️ Only N review(s) — need at least 3 to run`**
The user hasn't reviewed enough albums yet. The minimum is set to 3 (`MIN_REVIEWS_TO_RUN` in the script) to ensure the LLM has enough signal to make meaningful recommendations.

**Recommendations are for albums the user already reviewed**
This can occasionally happen with smaller models that don't reliably follow the "don't repeat these" instruction. The prompt includes an explicit list of already-reviewed albums, but larger models (llama3, mixtral) respect it much more consistently.

---

## Tuning

A few constants at the top of `recommend.py` you might want to adjust:

| Constant | Default | What it does |
|---|---|---|
| `OLLAMA_MODEL` | `llama3` | Which local model to use |
| `MIN_REVIEWS_TO_RUN` | `3` | Skip users with fewer album reviews than this |
| `MAX_REVIEWS_FOR_LLM` | `20` | How many reviews to include in the prompt (higher = more context, slower) |
| `BOT_USER_ID` | `LLMBot` | The `fromUserId` sent in each `submit_recommendation` call |

---

## Adding a new recommendation type in the future

When you're ready to support restaurants, books, etc.:

1. In `recommend.py`, add a `build_prompt_restaurant()` function modelled after `build_prompt_album()`
2. Add a case for it in the `build_prompt()` dispatcher
3. Add a Spotify-equivalent link builder if relevant (or pass `None` for `link`)
4. Pass `--type restaurant` on the CLI (you'd also add that argument to the argparse block)

The `parse_reviews()` filtering and the `SUPPORTED_TYPES` dict are already set up to handle this — only the prompt and link logic need adding.
