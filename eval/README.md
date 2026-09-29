# Evaluation — the single root for all eval work

> This is the **deep dive**: what each number means and how to improve the
> evaluation. For the quick "I want to X → command" menu, see the root
> [`README.md`](../README.md) (or just run `make help`).

Everything about **measuring and improving** the recommendation engine lives in
this one directory. The engine itself (the thing being measured) lives in
`afrec/` at the repo root. The two are deliberately separate: you can tune the
engine (`afrec/prompts.py`, `afrec/pipeline.py`) without touching anything
here, and you can change how we *measure* it here without touching the engine.

```
eval/
  EVAL_PLAN.md           the working plan — human-eval workflow, hardware limits, model quirks (read first)
  EVAL_CHECKLIST.md      the to-do list — every remaining command to type, in order (the "what's left")
  evaluate.py            the harness — run it to get a leaderboard (this is "the eval")
  goldenset.py           scoring + labeling logic (golden sets, interactive corpus)
  model_options.py       per-model LLM options loader (model_options.json — e.g. Qwen no-think)
  taste.py               HUMAN JUDGE — rate a whole response 1–5, or duel models (single duel)
  duels.py               HUMAN JUDGE, BATCHED — whole duel matrices (agent showdowns / model showdowns), resumable
  humaneval.py           storage + stats for taste.py + duels.py (scores.jsonl, duels.jsonl, Elo)
  fixtures/<profile>.json   the INPUTS — saved user-review profiles the engine runs against
  golden/<profile>.json     the JUDGE — human good/bad picks per profile (your taste)
  golden/labeled.jsonl      the CORPUS — every live good/bad verdict you give, appended
  golden/scores.jsonl       the CORPUS — every 1–5 rating of a full response, appended
  golden/duels.jsonl        the CORPUS — every model-duel ranking, appended (by taste.py and duels.py)
  mock_ollama_server.py    fake LLM server (mimics llama-server), so you can exercise the harness with no model
  make_fixture.py          BUILD a test case from a REAL user — fetches their Firebase reviews,
                           keeps only the music/album ones (same source recommend.py reads),
                           saves eval/fixtures/<profile>.json (+ empty golden seed); --force to re-fetch
  tests/
    test_agents.py             agent registry + run stamping, fully offline
    test_offline.py            pipeline + harness, no Spotify, no network
    test_spotify_existence.py  proves the "does it exist on Spotify" metric is honest
    test_humaneval.py          scores/duels/Elo records, fully offline
  output/                generated (gitignored): leaderboards, TSVs, per-model JSON, CSVs
  output/duel_sessions/    duel session state (resume points for duels.py — safe to delete; duels.jsonl is the record)
```

Engine (what's being measured): see `afrec/` at the repo root. `recommend.py`
is the production entry point. Neither imports anything from `eval/`.

**Two variables, tracked separately:** the **model** (the brain) and the
**agent** (the playbook — prompts + call flow + retry policy, one module per
agent in `afrec/agent_*.py`, registered in `afrec/agents.py`). The two in
the box: `artist-then-album@v1` (two-call baseline, the default) and
`album-first@v1` (one-shot direct album picks — the A/B). Every automated
result and human verdict records **both**, and every command takes `--agent`
(make: `AGENT=`). Run `make agents` for the reference table.
`make select AGENT=<tag>` changes which agent production runs. When you ship
a new strategy, register a new agent (don't edit a published one) and compare
tags on the same model + profiles. Duel Elo is keyed `model [agent]` so a
playbook change gets its own ladder instead of polluting the model's history.

---

## Who is doing the evaluating? (automated vs. human)

Every tool falls into one of three tiers, by **where the human is in the loop**:

**1. Automated — no human in the loop.** `evaluate.py` runs the engine against
`fixtures/` and scores it: the proxy metrics (instruction-following, confidence,
Spotify existence) plus your pre-written `golden/<profile>.json` lists. The
tests and the mock server are automated too. You just read the leaderboard.

**2. Human, offline — you write your taste down in advance.** The golden sets:
*you* decide "for this listener, X is clearly a good pick, Y is clearly bad" —
without looking at any model output. That's the only human input the automated
leaderboard needs, and because it's model-independent, the `golden` column is
comparable across models. (See [`golden/README.md`](golden/README.md).)

**3. Human, live — you watch the model's real output and judge it.** The model
generates actual recommendations and **you make the call**, one judgment at a
time. The three interactive tools ask different questions:

| Tool | What *you* are deciding | Unit of judgment | Recorded in |
|---|---|---|---|
| `evaluate.py --interactive` | Is **this one recommendation** good or bad? | a single rec, one at a time (good / bad / **unsure** — mark it and go listen later) | `golden/labeled.jsonl` (model **and agent** tagged) |
| `taste.py pending` | The **to-listen queue**: the unsure recs you can't judge without hearing them | list the queue, then `--judge` to append final verdicts after you've listened | `golden/labeled.jsonl` (`unknown` rows don't count in stats until resolved) |
| `taste.py score` | How good was **the whole response**? | the full set of recs, rated 1–5 | `golden/scores.jsonl` |
| `taste.py duel` | **Which of 2+ responses is better?** | relative, one duel at a time (randomized A/B/C) | `golden/duels.jsonl` → win-rates, Elo |
| `duels.py agents` / `duels.py models` | The same judgment, **batched into a resumable session**: e.g. every model's two playbooks vs each other, or each model with its own preferred agent | many duels in one command; progress saved after each duel | `golden/duels.jsonl` + `output/duel_sessions/` (resume state) |

Why three live tools? Different units of judgment answer different questions:
**label** has the finest granularity — it finds the specific rec that missed and
builds the most corpus per session. **score** judges the *answer as a whole*,
which rewards a great pick nobody "planned" for (the anti-golden-set-bias
signal), and `--rounds N` tells you how *consistent* a model is. **duel** is a
*relative* judgment — easier for humans to make reliably than absolute ratings —
and accumulates into a no-metrics "which model do I actually prefer" leaderboard.
`taste.py stats` rolls all three up (automated).

```
 fixtures/*.json  (inputs)
        │
   afrec/ engine  ← model under test
        │
   generated recommendations
        │
   ┌────┴───────────────────────────────────────┐
   │ AUTOMATED: evaluate.py                     │   HUMAN (live): you watch
   │  • proxy metrics                           │     • label:  each rec — good/bad?
   │  • golden/<profile>.json                   │     • score:  whole response — 1–5?
   │    (human-written in tier 2, then static)  │     • duel:   which of A/B is better?
   └────┬───────────────────────────────────────┘
        ▼                                       ▼
   leaderboard (output/)             golden/*.jsonl → taste.py stats
```

---

## Quick start

Note: with llama-server the `--models` name is just a label for the leaderboards
(the server serves whatever GGUF you loaded). Swap GGUFs, re-run, and the
leaderboard tracks each model by the name you gave it.

### Automated (no human in the loop)

```bash
# Run every saved taste profile (Spotify on if creds present):
python eval/evaluate.py --models my-model

# Fast iteration while you edit afrec/prompts.py — one profile, no Spotify
# (model name/URL come from .env):
python eval/evaluate.py --profile jazzy_hiphop --no-spotify

# Dump every individual recommendation to a CSV for human review:
python eval/evaluate.py --profile jazzy_hiphop --export-csv recs.csv

# No model on this box? Point at a remote llama-server, or run the mock:
python eval/mock_ollama_server.py --port 11500 &
python eval/evaluate.py --models mockA --base-url http://localhost:11500 --no-spotify
```

### Human, live (you watch the output and judge it)

```bash
# Judge a model's real recommendations live (grows the human-preference corpus):
python eval/evaluate.py --interactive --profile jazzy_hiphop

# Rate a model's WHOLE response 1–5 (rewards good answers nobody "planned" for):
python eval/taste.py score --profile jazzy_hiphop --models my-model --rounds 2

# Duel two or more models on the same profile — you pick the best
# (e.g. two llama-servers on different ports via separate runs, or Ollama names):
python eval/taste.py duel --profile wyatt --models modelA,modelB

# Duel a WHOLE MATRIX in one resumable session (duels.py):
python eval/duels.py agents                # agent showdown: each model vs its own other playbook, all profiles
python eval/duels.py models --contestants modelA:album-first@v1,modelB --profiles wyatt,jazzy_hiphop
python eval/duels.py status                # progress across all sessions
# Quit anytime (q / Ctrl-C); re-run the same command to pick up where you left off.

# Where do things stand (avg 1–5 scores, duel win-rates, Elo)?
python eval/taste.py stats
```

### Human, offline / test-case management

```bash
# Turn a REAL user's Firebase profile into a test case (read-only fetch, same call
# as recommend.py), then judge it:
python eval/make_fixture.py --user-id <userId> --profile alex
python eval/evaluate.py --interactive --profile alex

# Write your taste down for the automated leaderboard (tier 2):
#   edit golden/<profile>.json  →  golden/README.md
```

Run the offline tests any time (no network, no LLM server):

```bash
python eval/tests/test_agents.py
python eval/tests/test_offline.py
python eval/tests/test_spotify_existence.py
python eval/tests/test_humaneval.py
```

### See it all on a static website

```bash
make site    # → docs/  (self-contained HTML, reads your local judging data)
```

Open `docs/index.html` locally, or commit `docs/` and point GitHub Pages at the
`main` branch, `/docs` folder (Pages can only serve the root or /docs from a branch)
it — scores, duels + Elo, labels, your golden sets, and the latest
leaderboards, regenerated from local files on each push. No services, no
APIs: `eval/site.py` only reads `eval/golden/*.jsonl` and the newest
`eval/output/leaderboard_*.md` files.

---

## What "quality" actually is (read this before trusting any number)

`quality` is a **proxy**, not ground truth. It blends (higher = better):

| Component | What it measures | Honest? |
|---|---|---|
| **instruction-following** | no repeat artists, one album per artist, covers the suggested artists | yes (deterministic) |
| **LLM confidence** | the model's self-reported 0–1 confidence | weak (models are overconfident) |
| **existence rate** | fraction of the model's *proposed* albums that **really exist on Spotify** | yes (live API) |

The existence term is where the interesting truth is — see below.

### The key metric: "how much of the output actually exists on Spotify?"

- **`real r1`** (a.k.a. `raw_existence_rate`) = `round1_found / total_candidates`.
  Of the albums the model **produced on its first pass**, what fraction exist as
  real Spotify albums? *This is the anti-hallucination number.*
- **`rescued`** = how many of the misses the engine's retry step fixed with a real album.
- **`verified` / `verified_share`** = the *final* recs that resolved. ⚠️ This reads
  ~100% **by construction** — the pipeline retries until something exists — so it
  **hides** the hallucination rate. That's why `quality` is driven by `real r1`,
  not by `verified_share`.
- Verified against the **real** `api.spotify.com` (not a name-guess). A made-up album
  returns "not found"; a real one returns a direct link.
- **`⚠️ UNTRUSTED` cutoff** — if a model/agent's avg `real r1` is ≤25% (i.e. ≥75% of its
  round-1 recs don't exist on Spotify), the leaderboard flags it **UNTRUSTED** (on the model
  row, the per-profile status, and an `untrusted` column in the TSV). That pairing is
  untrustworthy and is *not worth human eval time* — this is the automated first-pass
  gate before you spend listening sessions on a model. Threshold: `UNTRUSTED_REAL_R1_MAX`
  at the top of `evaluate.py`.
- **Thinking models (Qwen3-style)** emit a `reasoning_content` field alongside `content`.
  At temp 0.7 they sometimes bury the whole answer in their reasoning and finish with an
  **empty** `content` — the stage parses nothing and the run scores 0 recs. The harness now
  (1) retries a parse failure once and (2) unwraps double-nested arrays. Thinking is handled
  **per model** via `model_options.json` (e.g. Qwen3.8-27B-Q8_0 → `enable_thinking: false`),
  merged into every generate call by both `evaluate.py` and `taste.py` — ~20× faster on Qwen
  with the same quality, and a no-op for the other models in the pool. Override per run with
  `--thinking` (spot-check) or `--no-thinking` (force off). Full story: `EVAL_PLAN.md` →
  "Known model quirks".

**Reading a row:**
```
| profile | real r1  | rescued | avg conf | quality |
|   5/9 (56%) |    2      |   0.94  |   0.68   |
```
→ the model got 5 of 9 right on the first try; the retry rescued 2 of the 4 misses;
final recs are all real; but its raw accuracy is 56%, and `quality` reflects that.

`eval/tests/test_spotify_existence.py` locks this behavior in: a fake model emits 4
real + 4 fake albums and the test asserts `raw_existence_rate == 0.5`,
`final_existence_rate == 1.0`, and (crucially) that `quality` ≈ the 0.5 reality —
not the ~1.0 a masked metric would show.

---

## HOW TO IMPROVE THE EVALUATION

Ranked by leverage (do the top ones first):

### 1. Improve the golden sets — the biggest lever
The `golden` column in the leaderboard is your **taste**, turned into a number.
We intentionally keep **only 2 profiles** (and 2 seed golden sets): music taste
drifts and the scene changes, so a big static set goes stale fast. The seed
files under `golden/` are a handful of good/bad artists each — a floor to
*build from*, not a finished yardstick. Grow them over time, mostly from lived
experience: `--interactive` labels and `taste.py` verdicts tell you who belongs
on the list, and you promote the repeat offenders. (Older, settled canon — like
the jazz profile — changes least; newer scenes drift fastest.)
Replace the seed guesses with artists
you *actually* like and dislike for each profile (any that the engine is likely to
surface). Every good/bad entry you add makes the leaderboard discriminate "good
engine" from "bad engine" for **you**.

- One file per profile: `golden/<profile>.json`. Bare strings or objects both work:
  ```json
  { "good": ["Nas", {"artist":"Gang Starr","album":"Daily Production","note":"deep cut"}],
    "bad":  ["Drake", "Kanye West"] }
  ```
- Scoring = `(good_hits − bad_hits) / n_recs`, clamped to `[-1, 1]` → `+0.250` means
  the model's picks lean toward your good list. On top of the base score the
  leaderboard also shows **contrast pairs** (right side of a good/bad near-twin,
  +1/0/−1 each) and **novel hits** (original-but-right moves, listed separately).
- **Model-independent**: any model is scored against the same yardstick, so the
  number is directly comparable across models.
- Full details: [`golden/README.md`](golden/README.md).
- **Watch for 0g/0b**: if the model never suggests any artist on your lists you'll
  get `0g / 0b`. That's a *data* gap (lists too narrow / wrong genre), not a bug —
  broaden the golden set for that profile.

### 2. Grow the interactive corpus (`--interactive`, `taste.py`)
Two complementary human-judgment tools:

- **`evaluate.py --interactive`** — label each individual rec **good / bad /
  unsure**; every verdict lands in `golden/labeled.jsonl` with model + timestamp.
- **`taste.py score`** — rate the **whole response** 1–5 (1 bad · 2 weak ·
  3 neutral · 4 good · 5 great). This is the anti-golden-set-bias signal:
  a model can surface a great answer nobody planned for, and it still counts as
  a 5. Use `--rounds N` to sample N independent generations and rate each —
  this also tells you how *consistent* a model is. Each verdict records which
  **agent** generated the response, so `taste.py stats` splits scores by
  `model [agent]`.
- **`taste.py duel`** — two or more models answer the same profile; responses
  are shown as **randomized letters** (A/B/C) to fight position bias; you pick
  the best (and optionally rank the rest). Every ranking feeds a running
  win-rate + **Elo** table (`taste.py stats`), so over time you get a
  no-metrics-needed "which model do I actually prefer" leaderboard.
- **`duels.py` — the batched, resumable version of duel.** For matrix
  comparisons ("all 4 mediums × 3 profiles, each agent against the other" is
  12 duels — no one wants to type 12 commands), `duels.py agents` / `duels.py models`
  build the whole queue, run it in one terminal session, and save progress
  after every duel: quit with `q` or Ctrl-C, re-run the same command later,
  and it resumes. Contestants are labeled `Model [agent]` for agent duels, so
  the two playbooks get separate Elo ladders per model; `duels.py status`
  shows where every session stands. One model is loaded at a time (agent
  duels), so it respects the one-big-model hardware limit — it even warns
  before a model duel that would load two big models.
- - **Async judging (the listening queue)** — you don't have to know every answer
  on the spot. In `--interactive`, press **u** on a rec you'd have to *hear* to judge
  (it's recorded as `unknown`, which `taste.py stats` ignores). Later, queue those in
  Spotify (`python eval/taste.py pending` lists them, deduped, with the artist/album
  and your note), listen whenever, then `python eval/taste.py pending --judge` walks
  the queue and appends your final good/bad verdicts. `stats` shows the pending count
  so you never lose track of what's still owed.
- **Confidence calibration** — free by-product of scoring: each rated response
  also recorded the model's self-claimed confidence per rec, so
  `taste.py stats` reports bias (claims vs your 1–5 average), a correlation
  when you've rated ≥3 responses, and a *claim vs taste* red-flag table —
  "you rated it 1, the model claimed 0.9" is the tell that a model can't
  sense when it's off.

Inspect any of it:

```bash
python eval/taste.py stats
# or dig into raw records:
python -c "import sys; sys.path.insert(0,'eval'); import goldenset as G, humaneval as H; \n           print('labeled recs', G.labeled_stats()['total'], '| scores', H.score_stats()['n'], '| duels', len(H.load_duels()))"
```

These corpora are your real taste signal (vs. the generic leaderboard proxy)
— and together with `labeled.jsonl` they're the raw material for any future
fine-tuning / DPO-style preference training.

### 3. A/B the engine while you change it
Two ways to hold everything else fixed:

**Quick loop (still designing the change):** hold the model fixed, edit
`afrec/prompts.py` (or `afrec/pipeline.py`), re-run:
```bash
# before edit
python eval/evaluate.py --profile jazzy_hiphop --no-spotify --output-dir out/before
# edit afrec/prompts.py
# after edit
python eval/evaluate.py --profile jazzy_hiphop --no-spotify --output-dir out/after
diff out/before/leaderboard_*.md out/after/leaderboard_*.md
```
`--no-spotify` makes each run ~seconds so you can iterate fast; drop the flag on the
final comparison to see real existence rates.

**Keep the version (the change is a real strategy shift):** ship it as a new
agent (new module + `_register(...)`, see `afrec/agent_album_first.py` as
the template) and compare by tag — the leaderboards, TSVs and human corpora
all record which agent produced each number, so results from different
playbooks never silently mix. The two built-ins make the first comparison:
```bash
python eval/evaluate.py --models llama3 --agent artist-then-album --output-dir out/ata
python eval/evaluate.py --models llama3 --agent album-first --output-dir out/af
```
For the *taste* side of the playbook question (the automated columns can't
see it), run the agent showdown: `python eval/duels.py agents` — each model
duels its own other playbook on every profile, batched and resumable,
standings in `taste.py stats` as `Model [agent]` ladders.

### 4. Add a model to the sweep
`--models a,b,c` — any names (with llama-server they're just labels; with Ollama
they must be pulled model names). The leaderboard ranks them by `quality`,
`real r1`, and `golden` side by side. See which *kind* of thing each model is bad at
via the per-profile table.

### 5. Add a taste profile (fixture)
Only 2 profiles ship on purpose (see §1). To add one:
- **From a real user — the preferred route now:**
  ```bash
  python eval/make_fixture.py --user-id <userId> --profile alex
  ```
  Fetches the user's real Firebase reviews (the same read-only `get_user_reviews`
  call `recommend.py` uses), keeps the music/album reviews (books/restaurants/etc.
  are stripped), saves `fixtures/alex.json`, and scaffolds an empty `golden/alex.json`
  seed to grow via `--interactive` labels. `--force` re-fetches over an existing one.
- **By hand:** copy a `fixtures/<name>.json`, edit the `reviews` (see `fixtures/` for the shape).
Either way the profile automatically appears in every run. But reconsider how many you
keep — a small set you grow is more trustworthy than a wide set you abandon.

### 6. Add or adjust a metric
All metric math is in `score_outcome()` in `eval/evaluate.py` (one function — search
for `def score_outcome`). To add a signal (e.g. "recs the user already owns",
"genre spread", "reason length"), compute it there, return a key in its dict, then
display it in `render_leaderboard()` + `write_tsv()`. Because `quality` is a blend
right in that function, you can re-weight or swap terms here without touching the
engine. Keep any new key model-independent if you want it comparable across models.

### 7. Tighten the existence ground truth
The verifier is `afrec/spotify.py` (`verify()` against the real API). To be more
strict (e.g. require an *album* result, not a search fallback; enforce a similarity
threshold; reject "artist only" matches), change the verifier — the harness just
calls it. `test_spotify_existence.py` already fakes a verifier, so you can test your
new logic offline.

### 8. Keep it reproducible (know the limits)
`quality` depends on sampling (temperature, non-deterministic). For a
defensible comparison: repeat a run a few times and compare the **mean**, or lower
`--temperature 0` for tighter, less creative (fewer hallucinations) sampling. The
`per_<model>.json` files store the full `RunOutcome` (each stage's parsed output) so
you can diff exactly what changed between two runs.

---

## How to verify it's all wired up
- `python eval/tests/test_agents.py` — agent registry, tag resolution, and that
  every run is stamped with its agent version. Fully offline.
- `python eval/tests/test_offline.py` — no-spotify and (fake-verify) spotify paths +
  leaderboard render. Uses `FakeLLM`, so it's deterministic and offline.
- `python eval/tests/test_spotify_existence.py` — proves the existence metric isn't
  masked by retry.
- `python eval/tests/test_humaneval.py` — taste.py's score/duel/Elo records, offline.
- `python eval/mock_ollama_server.py --port 11500` then a real E2E at `--base-url
  http://localhost:11500` — the whole harness, no model required.

## Troubleshooting
- **`no fixtures found`** → files missing from `eval/fixtures/`.
- **spotify "off (no creds)"** → set `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` in
  the repo-root `.env`.
- **connection error** → the server isn't up or `--base-url`/`LLM_URL` is wrong;
  check with `curl <base-url>/v1/models` (llama-server) or `ollama list` (Ollama).
- **golden `0g/0b` everywhere** → your golden lists don't overlap what the model
  suggests; broaden them (see §1).
- **CSV `verified=no` for real-sounding albums** → Spotify genuinely doesn't have
  that exact artist+album; the `spotify_url` column is a search link you can open to
  confirm.
