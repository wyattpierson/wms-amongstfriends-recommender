# EVAL_PLAN — rating the models × agents (working doc)

Living plan for the human-in-the-loop evaluation of the AmongstFriends
recommender. This is the source of truth for the workflow — chat history
is not. Last updated: 2026-09-23.

## Goal

Produce a trustworthy per-(model × agent) ranking on **taste fit** (not
catalog recall), using:

1. **Automated triage** — existence on Spotify (`real r1`) + the UNTRUSTED
   cutoff (`raw_existence_rate ≤ 0.25` ⇒ ⚠️ UNTRUSTED, auto-dropped).
2. **Human labels** — interactive good/bad/unsure per rec, accumulated in
   `eval/golden/labeled.jsonl` (the durable corpus).
3. **Async listening** — `taste.py pending` queue for "unsure" recs, judged
   later after actually listening.

The final deliverable is `taste.py stats`: per-model and per-agent
`good_rate` (human), with `real r1` (automated) as the trust gate.

## Status

- [x] Fixtures: `jazzy_hiphop`, `alt_rnb`, `warm_groove`, `wyatt`
- [x] Golden **seeds** (5–7 entries each) — see "Golden policy" below
- [x] UNTRUSTED hard cutoff in `evaluate.py` (leaderboard flag + TSV column)
- [x] `taste.py pending` / `pending --judge` async listening workflow
- [x] Qwen thinking-model fixes (see "Known model quirks")
- [x] Spotify creds in `.env`
- [ ] Triage sweep across all models × 2 agents (jazzy_hiphop first)
- [ ] Interactive labeling for models that survive triage
- [ ] Listening-queue passes for "unsure" recs
- [ ] Golden curation from labeled data (end of each fixture)
- [ ] Repeat for `alt_rnb`, `warm_groove`, then `wyatt` (18 reviews — slowest)

## Golden policy (revised)

The hand-written golden sets are **seeds, not ground truth** — they are
narrow (e.g. jazzy_hiphop seeds are all jazz-adjacent hip-hop) and any
engine will "fail" them partly for genre-coverage reasons, not taste
reasons. Consequences:

- **Ignore the `golden` columns during triage.** Triage decision is based
  on `real r1` / UNTRUSTED only.
- The real golden set is what **you label**. Every interactive run and
  every `pending --judge` verdict appends to `labeled.jsonl`.
- **At the end of each fixture**, curate: keep the seeds, add good novel
  recs you loved (with notes), add bad recs with reasons. Only then does
  the golden column become meaningful for that fixture.
- `labeled_stats` already ignores `unknown` labels in `good_rate`, so
  unresolved pending items don't skew stats.

## Hardware constraints (M5 MacBook Pro, 64 GB)

- At most **one big model per command** (27B–32B ≈ 30 GB at Q8).
- Small models (7–9B) can run **in pairs**.
- Qwen is **Q8-only now** (`Qwen3.8-27B-Q8_0`, ~30 GB) — the Q5 file was
  deleted 2026-09-23, so Qwen must run **alone** (never paired with smalls).
  (The pool still has a stale Q5 registration; it 500s on request — ignore
  it, or remove it from the pool config if you keep it around.)
- Pool server: `http://localhost:8092` (models auto-load on demand).

## Per-fixture drill

### Step 1 — Triage sweep (fast, no human)

```bash
# big model, alone, both agents (Qwen runs no-think automatically —
# see eval/model_options.json; add --thinking only for quality spot-checks)
venv/bin/python eval/evaluate.py --models Qwen3.8-27B-Q8_0 \
    --agent artist-then-album@v1 --profile jazzy_hiphop
venv/bin/python eval/evaluate.py --models Qwen3.8-27B-Q8_0 \
    --agent album-first@v1 --profile jazzy_hiphop

# smalls in pairs, both agents
venv/bin/python eval/evaluate.py \
    --models Dolphin3.0-Llama3.1-8B.Q8_0,Meta-Llama-3.1-8B-Instruct-Q8_0 \
    --agent artist-then-album@v1 --profile jazzy_hiphop
venv/bin/python eval/evaluate.py \
    --models Dolphin3.0-Llama3.1-8B.Q8_0,Meta-Llama-3.1-8B-Instruct-Q8_0 \
    --agent album-first@v1 --profile jazzy_hiphop
# (repeat for Mistral/gemma-E4B, Phi-4-mini/gemma-26B-A4B as desired)
```

Read only: `real r1` + the UNTRUSTED flag. UNTRUSTED ⇒ done, don't label it.
Borderline (25–50%) ⇒ label, but expect to throw most recs away.

### Step 2 — Interactive labeling (the actual eval)

One model × one agent × one profile per run:

```bash
venv/bin/python eval/evaluate.py --models <model> --agent <agent-tag> \
    --profile jazzy_hiphop --interactive
# g = good, b = bad, u = unsure (goes to the listening queue)
```

Rules of thumb:
- Label **taste fit**, not existence — Spotify already checks existence.
- "Good" = you'd actually listen to it knowing this person's taste.
- When in doubt → `u`. Resolving 50% later is fine; that's what
  `taste.py pending` is for.

### Step 3 — Listening queue (async)

```bash
venv/bin/python eval/taste.py pending            # what to listen to (deduped)
# …listen via the Spotify links…
venv/bin/python eval/taste.py pending --judge    # g/b/; per item
```

### Step 4 — Scoreboard

```bash
venv/bin/python eval/taste.py stats
```

Per-model and per-agent `good_rate` (human) is the ranking. Cross-check
against `real r1` from triage. A model that is real but boring loses to a
model that is real AND fits.

### Step 5 — List-level judgment (whole-list quality + duels)

Labels judge recs one at a time; they can't see a list where every rec is
fine but the whole thing skews one genre and misses the core of the taste.
That's what this step is for (the warm_groove/phi-4-mini case: six "good"
recs, wrong big picture):

```bash
# 1. Rate the whole response 1–5 — "good recs, wrong big picture" is a 3–4
#    with a note, not a 5. 2 rounds to see consistency.
venv/bin/python eval/taste.py score --profile <fixture> --models <model> \
    --agent <agent-tag>@v1 --rounds 2

# 2. Duels, batched + resumable (one command for the whole matrix):
venv/bin/python eval/duels.py agents    # each model vs its own other playbook, all 3 fixtures
venv/bin/python eval/duels.py models --contestants <a>:<agent>@v1,<b> --profiles <fixture>,...
venv/bin/python eval/duels.py status    # progress; re-run the same command to resume after a quit
```

Agent-duel standings show up in `taste.py stats` as separate `Model [agent]`
Elo ladders — that's the head-to-head playbook verdict the automated columns
can't give. Model duels keep the bare-model ladder; each contestant may name
its own preferred agent. Keep at most one big model per matchup (the driver
warns before it would load two).

### Step 6 — Golden curation (end of fixture)

From the run outputs + `labeled.jsonl`:
- add the best novel recs to `eval/golden/<profile>.json` (good, with notes)
- add clearly-bad recs (bad, with reasons)
- keep the seeds unless they were actively wrong

Then move to the next fixture. `wyatt` (18 reviews) last — slowest and
most personal.

## Known model quirks

### Qwen3.8-27B (thinking model)

Symptom seen 2026-09-23: `artists` stage returned an **empty string**
after ~59 s → `ResponseParseError` → 0 recs, run failed.

Root cause: Qwen3-style models emit a `reasoning_content` field alongside
`content`. At temperature 0.7 the model occasionally buries the whole
answer in its reasoning and finishes with empty `content`. The engine only
reads `content`. Non-deterministic — the same prompt succeeds on retry.

Second quirk: with thinking **off**, Qwen consistently double-wraps arrays
(`[["a","b"],…]`), which silently parsed as "1 artist" (a list) and was
filtered out by the agent.

Fixes (shipped):
1. `afrec/pipeline.py::_run_stage` — **one retry** on `ResponseParseError`
   (flake guard; `LLMError` still fails fast).
2. `afrec/llm.py::_extract_json` — **unwraps** single-element nested lists.
3. **Per-model no-think (automatic)** — `eval/model_options.json` maps
   `Qwen3.8-27B-Q8_0` → `chat_template_kwargs: {"enable_thinking": false}`;
   `evaluate.py` and `taste.py` merge it in per model (`model_options.py`).
   ~20× faster (100–131 s vs 137–160 s *per stage*), same quality
   (0.718–0.800). No flag needed. CLI overrides: `--thinking` (force on,
   spot-checks) / `--no-thinking` (force off, all models).

Template facts (read from the GGUFs, 2026-09-23): only Qwen3 defaults
thinking **on**; gemma-4's template defaults thinking **off** (kwarg is a
no-op); Dolphin/Meta-Llama/Mistral/Phi-4 templates don't reference the
kwarg at all — so the payload is safe pool-wide. Qwen's template also
supports `reasoning_effort` (xhigh default / medium / low) if we ever want
a thinking-on-but-brief middle ground.

Recommendation: no flags for sweeps; `--thinking` only for a deliberate
quality spot-check. If a model still fails after the retry, check
`eval/output/per_<model>.json` → `stages[].raw`.

**Why no-thinking is the default (and the residual risk).** Setting
`enable_thinking: false` is not just hiding output — it changes behavior:
the model skips its internal deliberation and answers directly. For *this*
task (match taste from short review inputs) that's fine: the bottleneck is
catalog knowledge, not reasoning, and our runs show equal-or-better quality
(0.718 / 0.800 / 0.800) at ~20× the speed and none of the empty-content
flakes. Keeping thinking on and *parsing the answer out of* the long output
is strictly worse for us (slower + flaky + no measured gain). **Residual
risk:** we haven't confirmed parity on *human* labels yet. Once the labeled
corpus has data, run one `--thinking` vs no-thinking session on the leading
model/agent and compare `good_rate` before locking the default.

## Decision criteria (provisional)

- **UNTRUSTED**: `raw_existence_rate ≤ 0.25` — automatic, no human time.
- **Worth keeping**: `real r1 ≥ 0.75` AND `good_rate ≥ 0.5` on ≥ 15 human
  labels (more is better).
- **Agent comparison**: same models, both agents, compare `good_rate` and
  `real r1` deltas per fixture.
- Revisit thresholds once we have labeled data from ≥ 2 fixtures.

## Files

| Path | Role |
|---|---|
| `eval/evaluate.py` | batch sweep + `--interactive` labeling + UNTRUSTED flag |
| `eval/taste.py` | `stats` / `models` / `pending [--judge]` / `score` / `duel` (single) |
| `eval/duels.py` | `agents` / `models` / `status` — batched, resumable duel sessions |
| `eval/EVAL_CHECKLIST.md` | the to-do list — every remaining command to type, in order (this plan's "what's left") |
| `eval/output/duel_sessions/` | duel session resume state (safe to delete; `duels.jsonl` is the record) |
| `eval/model_options.py` + `model_options.json` | per-model LLM options (Qwen no-think), merged into every generate call |
| `eval/golden/labeled.jsonl` | **the** durable human corpus (append-only) |
| `eval/golden/<profile>.json` | curated seeds (+ your additions) |
| `eval/output/leaderboard_*.md` | per-sweep leaderboard |
| `eval/output/runs_*.tsv` | machine-readable rows (has `untrusted` col) |
| `eval/output/per_<model>.json` | stage-level debug (raw model output) |
