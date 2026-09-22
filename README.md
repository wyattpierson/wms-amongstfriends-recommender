# AmongstFriends — LLM Recommendation Engine

Reads a user's album reviews from the AmongstFriends Firebase backend, runs them
through a local LLM to generate personalized album recommendations, verifies each
pick against the real Spotify API (retrying hallucinations), and submits them back
via Firebase Functions.

Currently supports **music (albums)**; other types are stubbed for the future.

> **Quick reference:** `make` (or `make help`) lists every command in this repo.
> This README explains *what things are and when to use them*; the deep-dive on
> evaluation is [`eval/README.md`](eval/README.md).

> **What's generic, what's personal.** Reusable, in my opinion: the
> local-LLM → Spotify-verification pipeline pattern (`afrec/llm.py`,
> `afrec/pipeline.py`, `afrec/spotify.py`) and the whole evaluation harness
> (`eval/` — fixtures, golden sets with contrast pairs, live human judging,
> Elo duels, static dashboard). Specific to me: the Firebase wiring
> (`afrec/firebase.py`, `recommend.py`) is built for the AmongstFriends app,
> and the `wyatt` fixture/golden is my own review data and taste. If you only
> read one thing, read [`eval/README.md`](eval/README.md).

---

## What's here

```
afrec/        THE ENGINE — the thing being measured (never imports from eval/)
  prompts.py     ← the file you edit to tune the engine
  pipeline.py    orchestration: artists → albums → verify → retry
  spotify.py     hallucination check (verify artist+album against Spotify)
  llm.py         LLM client — llama.cpp llama-server (default) or Ollama, via .env
  firebase.py    auth + remote function calls (production only)
  reviews.py     normalizes raw review payloads; filters to album reviews

recommend.py  PRODUCTION entry point: Firebase → engine → Firebase

eval/         MEASUREMENT — everything for evaluating and improving the engine
  evaluate.py    the harness → leaderboard (quality / real r1 / golden)
  taste.py       human judge: 1–5 scoring, model duels, Elo, calibration stats
  goldenset.py   golden-set + label-corpus scoring logic
  humaneval.py   storage/stats for taste.py verdicts
  make_fixture.py  turn a real Firebase user into a test profile
  mock_ollama_server.py  fake LLM — run the harness with no model at all
  site.py        static-site generator (your judging data → site/ for GitHub Pages)
  fixtures/      the INPUTS  — saved user-review profiles (2 ship)
  golden/        the JUDGE   — your good/bad lists + auto-grown label corpora
  tests/         offline tests (no network, no model)
  output/        gitignored — leaderboards, TSVs, per-model JSON, CSVs

.env            all configuration (model, URLs, Firebase, Spotify) — not in git
```

The split is deliberate: tune the engine in `afrec/` without touching the
evaluator, and change how we measure without touching the engine.

---

## Setup (one time)

1. **Python 3.10+**, then:
   ```bash
   make setup            # venv + pip install (firebase-admin, requests, python-dotenv)
   source venv/bin/activate
   ```
2. **`serviceAccountKey.json`** in the repo root — Firebase Console → Project
   Settings → Service Accounts → *Generate new private key*. (gitignored; never commit.)
3. **`.env`** in the repo root (gitignored):

   | Key | What |
   |---|---|
   | `FIREBASE_PROJECT_ID` / `FIREBASE_REGION` | your project + where functions are deployed |
   | `FIREBASE_WEB_API_KEY` | Project Settings → General (not a secret, but keep it out of git) |
   | `LLM_BACKEND` | `llama.cpp` (default) or `ollama` |
   | `LLM_URL` | e.g. `http://localhost:8092` (llama-server) or `http://localhost:11434` |
   | `LLM_MODEL` | model name — cosmetic for llama-server (it serves whatever GGUF you loaded); a real pulled name for Ollama |
   | `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | for existence verification; leave empty to run without it |

4. **A local model server** (or point `LLM_URL` at a remote one):
   ```bash
   ./build/bin/llama-server -m /path/to/model.gguf --port 8092
   curl http://localhost:8092/v1/models   # confirm it's up
   ```
5. **Firebase functions deployed** — `get_user_reviews` and
   `submit_recommendation` must exist in your project (production runs only;
   evaluation doesn't need them).

---

## I want to… (command menu)

Every one of these is a `make` target — `make help` reprints this list.
Model name/URL come from `.env`; override with `MODELS=...` where shown.
`FB_UID` is the Firebase Auth UID (Console → Authentication → Users).

### Run the recommender on a real user (production — touches Firebase)

| I want to… | Command |
|---|---|
| Test safely: fetch reviews + generate recs, **submit nothing** | `make dry FB_UID=<uid>` |
| Really run it and submit recommendations | `make run FB_UID=<uid>` |
| Submit into a specific group instead of `default-group` | `make run FB_UID=<uid> GROUP=<id>` |
| Run without Spotify verification | add `--no-spotify` to the `recommend.py` call |

> **Always `make dry` first.**

### Automated — measure with no human in the loop

The machine runs the engine on your saved profiles and scores it. You don't
watch anything; the golden set is just *data it reads* (see below). Proxies
only tell you the engine *works* — they can't tell you it's *good* for you;
that's the human-judging tier below.

| I want to… | Command |
|---|---|
| Full leaderboard: every profile, real Spotify checks | `make eval` |
| Fast loop while editing `afrec/prompts.py` (one profile, seconds, no Spotify) | `make fast PROFILE=jazzy_hiphop` |
| Dump every individual rec to a spreadsheet for eyeballing | `make csv PROFILE=... CSV=recs.csv` |
| A/B a prompt change | run `eval/evaluate.py ... --output-dir out/before`, edit, re-run to `out/after`, diff the leaderboards (details: `eval/README.md`, playbook item 3) |
| Sweep several models at once | `make eval MODELS=modelA,modelB` |

### Human, offline — write your taste down (once, per profile)

You author `eval/golden/<profile>.json`: artists/albums that are clearly a
good or bad pick for that listener — including **contrast pairs** (two
near-identical picks the listener splits on: the real taste test) and
**novel** entries (original-but-right moves, scored separately). You do
this **ahead of time, without seeing any model output** — that's what makes
the leaderboard's `golden`/`pairs`/`novel` columns model-independent. Then
`make eval` scores every model against it, automated.
→ [how to write a golden set](eval/golden/README.md)

### Human, live — watch a model's output and judge it (the real signal)

Interactive sessions: a model generates real recommendations and **you** make
the call, one judgment at a time. Three tools, three different units of
judgment — pick the one matching your question:

| I want to… | What *you* are evaluating | Command |
|---|---|---|
| Vet individual picks | **one recommendation at a time** — good / bad / unsure | `make label PROFILE=jazzy_hiphop` |
| Grade an answer as a whole | **the entire response**, rated 1–5 — catches a great pick your golden list never mentioned | `make score PROFILE=... ROUNDS=2` |
| Compare models head-to-head | **which of 2+ responses is better** for the same profile (shown as randomized A/B to fight position bias) | `make duel PROFILE=... MODELS=modelA,modelB` |
| Review everything you've judged so far | (automated rollup of the above — nothing to watch) | `make stats` |

Every verdict appends to `eval/golden/*.jsonl` — over time this becomes your
per-model human-preference corpus (and raw material for future fine-tuning/DPO).

### Manage test cases (automated)

| I want to… | Command |
|---|---|
| Turn a real Firebase user into a profile (read-only fetch) | `make fixture FB_UID=<uid> NAME=alex` |
| Exercise the harness with **no model at all** | `make mock` (then `--base-url http://localhost:11500`) |
| Sanity-check everything offline (no network, no model) | `make test` |

### See your results on a website

| I want to… | Command |
|---|---|
| Show all my judging data (scores, duels, labels, golden sets, latest leaderboards) as a static site | `make site` → commit `site/` → GitHub Pages |

---

## What the numbers mean (30-second version)

The leaderboard ranks models by:

- **`quality`** — blended proxy: instruction-following + LLM confidence + **existence**
- **`real r1`** — fraction of the model's *first-pass* albums that really exist on
  Spotify. The honest anti-hallucination number. (`verified %` reads ~100% *by
  construction* — the pipeline retries until something exists — so don't trust it alone.)
- **`golden`** — your taste, applied automatically: how well the picks match the
  good/bad list you wrote *in advance*. No human is needed at run time — the
  human is only in the loop when you author/grow the list
- **`pairs`** — contrast-pair accuracy: for each good/bad near-twin in your
  golden set, did it rec the right side? (+1 / 0 / −1 per pair). The number
  that separates taste-inference from "more of the same genre"
- **`novel`** — how many *original-but-right* moves (adjacent scene,
  cross-genre bridge) it reached for, out of the ones you listed

The proxy metrics tell you the engine *works*; **your taste** (`label` / `score` /
`duel` / golden sets) tells you it's *good*. The full explanation — and the
ranked "how to improve the evaluation" playbook — is in
[`eval/README.md`](eval/README.md).

---

## How a run works

```
get_user_reviews()       local LLM                 submit_recommendation()
  Firebase Function   →   artists → albums →      →   Firebase Function
  (fetch album reviews)   verify → retry              (one call per rec)
```

1. Authenticates as `LLMBot` with the service account key
2. Fetches the user's reviews, keeps albums only
3. Call 1: 8–10 new artists → Call 2: one album per artist
4. Verifies each (artist, album) on Spotify; failures get one retry call where
   the model is told what failed
5. Submits each surviving recommendation

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `❌ Cannot reach llama-server` | Server isn't running or `LLM_URL` is wrong. `curl $LLM_URL/v1/models` |
| `❌ Failed to exchange custom token for ID token` | Wrong/missing `FIREBASE_WEB_API_KEY` |
| Firebase function `403` | Function is rejecting `LLMBot` — add a bypass for that UID in the function code |
| Firebase function `404` | Function name/region mismatch — check `FIREBASE_REGION` |
| `❌ Could not parse LLM response as JSON` | Small model not following JSON instructions; try a bigger GGUF (raw output is printed) |
| `⚠️ Only N review(s) — need at least 3` | User hasn't reviewed enough albums (`MIN_REVIEWS_TO_RUN` env knob) |
| Eval: `no fixtures found` | Missing files in `eval/fixtures/` |
| Eval: spotify "off (no creds)" | Set `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` in `.env` |
| Eval: golden `0g/0b` everywhere | Your golden lists don't overlap what the model suggests — broaden them (`eval/README.md`, playbook item 1) |

---

## Tuning knobs

- **`afrec/prompts.py`** — the prompts (artists / albums / retry). This is *the*
  file to edit when you want to change what the engine asks for.
- Env: `MIN_REVIEWS_TO_RUN` (default 3), `SPOTIFY_MATCH_THRESHOLD` (default 0.5),
  `GROUP_ID` (default `default-group`), plus the `.env` table above.
- CLI: `--temperature` on the eval tools (lower = less creative, fewer hallucinations).
