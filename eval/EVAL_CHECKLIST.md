# EVAL_CHECKLIST — remaining evals (concrete commands)

Snapshot as of 2026-09-24. Companion to `EVAL_PLAN.md` (the "why"); this file
is the "what's left to type". Run everything from the repo root.

## State so far (already done — do not re-run)

- [x] Triage sweep, **jazzy_hiphop**, both agents: `Qwen3.8-27B-Q8_0`,
      `Dolphin3.0-Llama3.1-8B.Q8_0`, `Meta-Llama-3.1-8B-Instruct-Q8_0`,
      `Mistral-7B-Instruct-v0.3-Q8_0`, `microsoft_Phi-4-mini-instruct-Q8_0`
- [x] Interactive labeling: `microsoft_Phi-4-mini-instruct-Q8_0` ×
      `artist-then-album@v1` × **warm_groove** (6 recs, all good — but the
      list skewed jazz-hip-hop; see Phase 8 note on capturing this)
- [ ] Everything else below. `scores.jsonl` / `duels.jsonl` do not exist yet.

Models in the pool (`http://localhost:8092`):

| model | size class |
|---|---|
| `Qwen3.8-27B-Q8_0` | BIG — runs alone |
| `gemma-4-26B-A4B-it-UD-Q5_K_M` | MoE (4B active) — can pair with one small |
| `Dolphin3.0-Llama3.1-8B.Q8_0` | small — pairable |
| `Meta-Llama-3.1-8B-Instruct-Q8_0` | small — pairable |
| `Mistral-7B-Instruct-v0.3-Q8_0` | small — pairable |
| `microsoft_Phi-4-mini-instruct-Q8_0` | small — pairable |
| `gemma-4-E4B-it-Q8_0` | small — pairable |

Agents: `artist-then-album@v1` (selected default) and `album-first@v1`.
Profiles: `jazzy_hiphop`, `warm_groove`, `alt_rnb`, `wyatt`.

---

## Phase 1 — Finish triage, jazzy_hiphop (fast, no human)

Only the two gemma models haven't been swept. Wait for the in-flight
phi-4-mini session to finish before loading anything new.

- [ ] gemma-4-26B-A4B (alone, both agents):
  ```bash
  venv/bin/python eval/evaluate.py --models gemma-4-26B-A4B-it-UD-Q5_K_M \
      --agent artist-then-album@v1 --profile jazzy_hiphop
  venv/bin/python eval/evaluate.py --models gemma-4-26B-A4B-it-UD-Q5_K_M \
      --agent album-first@v1 --profile jazzy_hiphop
  ```
- [ ] gemma-4-E4B (pair with Mistral, both agents):
  ```bash
  venv/bin/python eval/evaluate.py \
      --models Mistral-7B-Instruct-v0.3-Q8_0,gemma-4-E4B-it-Q8_0 \
      --agent artist-then-album@v1 --profile jazzy_hiphop
  venv/bin/python eval/evaluate.py \
      --models Mistral-7B-Instruct-v0.3-Q8_0,gemma-4-E4B-it-Q8_0 \
      --agent album-first@v1 --profile jazzy_hiphop
  ```
  (Mistral re-runs are harmless; read the gemma row.)

Read only `real r1` + UNTRUSTED flag. Any gemma at `raw_existence_rate ≤ 0.25`
is out — delete its rows from Phase 2.

## Phase 2 — Interactive labeling, jazzy_hiphop (the actual eval)

One model × one agent × one profile per run. `g` = good, `b` = bad,
`u` = unsure (→ listening queue). Label **taste fit**, not existence.
Order: start with the triage leaders (Qwen, Mistral, Meta-Llama), then the rest.

- [ ] `Qwen3.8-27B-Q8_0` × `artist-then-album@v1`
- [ ] `Qwen3.8-27B-Q8_0` × `album-first@v1`
- [ ] `Mistral-7B-Instruct-v0.3-Q8_0` × `artist-then-album@v1`
- [ ] `Mistral-7B-Instruct-v0.3-Q8_0` × `album-first@v1`
- [ ] `Meta-Llama-3.1-8B-Instruct-Q8_0` × `artist-then-album@v1`
- [ ] `Meta-Llama-3.1-8B-Instruct-Q8_0` × `album-first@v1`
- [ ] `Dolphin3.0-Llama3.1-8B.Q8_0` × `artist-then-album@v1`
- [ ] `Dolphin3.0-Llama3.1-8B.Q8_0` × `album-first@v1`
- [ ] `microsoft_Phi-4-mini-instruct-Q8_0` × `artist-then-album@v1`
- [ ] `microsoft_Phi-4-mini-instruct-Q8_0` × `album-first@v1`
- [ ] `gemma-4-26B-A4B-it-UD-Q5_K_M` × `artist-then-album@v1` *(only if it survived Phase 1)*
- [ ] `gemma-4-26B-A4B-it-UD-Q5_K_M` × `album-first@v1` *(only if it survived Phase 1)*
- [ ] `gemma-4-E4B-it-Q8_0` × `artist-then-album@v1` *(only if it survived Phase 1)*
- [ ] `gemma-4-E4B-it-Q8_0` × `album-first@v1` *(only if it survived Phase 1)*

Template for every one of them:

```bash
venv/bin/python eval/evaluate.py --models <MODEL> --agent <AGENT>@v1 \
    --profile jazzy_hiphop --interactive
```

## Phase 3 — Listening queue (whenever 'u' items accumulate)

- [ ] `venv/bin/python eval/taste.py pending` → queue the links in Spotify
- [ ] after listening: `venv/bin/python eval/taste.py pending --judge`

## Phase 4 — Golden curation, jazzy_hiphop (end of fixture)

- [ ] Curate `eval/golden/jazzy_hiphop.json`: keep seeds, add loved novel
      recs (with notes), add clearly-bad recs (with reasons)
- [ ] `venv/bin/python eval/taste.py stats` → per-model / per-agent `good_rate`

## Phase 5 — Qwen thinking parity check (from EVAL_PLAN)

Once Phase 2 has labels: one `--thinking` session vs the no-thinking runs on
the leading Qwen agent, compare `good_rate`, then lock the default.

- [ ] `venv/bin/python eval/evaluate.py --models Qwen3.8-27B-Q8_0 \
      --agent <leading-agent>@v1 --profile jazzy_hiphop --interactive --thinking`
- [ ] compare vs the no-thinking `good_rate` in `taste.py stats`

## Phase 6 — warm_groove (2nd fixture)

Triage only the models that survived jazzy triage (drop any UNTRUSTED).
`microsoft_Phi-4-mini-instruct-Q8_0` × `artist-then-album@v1` interactive is
**already done** — do not repeat it.

- [ ] Triage (Qwen alone, smalls in pairs, both agents):
  ```bash
  venv/bin/python eval/evaluate.py --models Qwen3.8-27B-Q8_0 \
      --agent artist-then-album@v1 --profile warm_groove
  venv/bin/python eval/evaluate.py --models Qwen3.8-27B-Q8_0 \
      --agent album-first@v1 --profile warm_groove
  venv/bin/python eval/evaluate.py \
      --models Dolphin3.0-Llama3.1-8B.Q8_0,Meta-Llama-3.1-8B-Instruct-Q8_0 \
      --agent artist-then-album@v1 --profile warm_groove
  venv/bin/python eval/evaluate.py \
      --models Dolphin3.0-Llama3.1-8B.Q8_0,Meta-Llama-3.1-8B-Instruct-Q8_0 \
      --agent album-first@v1 --profile warm_groove
  venv/bin/python eval/evaluate.py \
      --models Mistral-7B-Instruct-v0.3-Q8_0,microsoft_Phi-4-mini-instruct-Q8_0 \
      --agent artist-then-album@v1 --profile warm_groove
  venv/bin/python eval/evaluate.py \
      --models Mistral-7B-Instruct-v0.3-Q8_0,microsoft_Phi-4-mini-instruct-Q8_0 \
      --agent album-first@v1 --profile warm_groove
  # + gemma pairs if they survived Phase 1 (same pattern)
  ```
- [ ] Interactive labeling (one run each; skip phi-4-mini × artist-then-album):
  - Qwen × both agents
  - Mistral × both agents
  - Meta-Llama × both agents
  - Dolphin × both agents
  - `microsoft_Phi-4-mini-instruct-Q8_0` × **`album-first@v1` only**
  - surviving gemmas × both agents
  ```bash
  venv/bin/python eval/evaluate.py --models <MODEL> --agent <AGENT>@v1 \
      --profile warm_groove --interactive
  ```
- [ ] `venv/bin/python eval/taste.py pending --judge` (as needed)
- [ ] Curate `eval/golden/warm_groove.json`
- [ ] `venv/bin/python eval/taste.py stats`

## Phase 7 — alt_rnb (3rd fixture)

Same drill as Phase 6, `--profile alt_rnb`, same survivor list:

- [ ] Triage: Qwen alone ×2 agents; smalls in pairs ×2 agents
- [ ] Interactive labeling: surviving models × both agents (`--interactive`)
- [ ] `taste.py pending --judge` (as needed)
- [ ] Curate `eval/golden/alt_rnb.json`
- [ ] `taste.py stats`

## Phase 8 — List-level judgment: `score` and `duel` (nothing recorded yet)

This is the layer that captures **whole-list quality** — exactly the
warm_groove/phi-4-mini case where every rec was individually good but the six
skewed one genre and missed the core of the taste. `score` rates the *entire*
response 1–5 (not per rec): "good recs, wrong big picture" is a 3–4 with a
note, not a 5. `duel` pits whole lists against each other, so a skewed list
loses to a better-balanced one. Use these once Phase 2/6 have narrowed the field.

- [ ] Score phi-4-mini on warm_groove first — it directly measures the
      "individually good, overall skewed" case you just hit:
  ```bash
  venv/bin/python eval/taste.py score --profile warm_groove \
      --models microsoft_Phi-4-mini-instruct-Q8_0 \
      --agent artist-then-album@v1 --rounds 2
  ```
  (Give it the honest whole-list rating; put the genre-skew in the note.)
- [ ] Score the top 2–3 (model × agent) combos from `taste.py stats`, 2
      rounds each, on `jazzy_hiphop`:
  ```bash
  venv/bin/python eval/taste.py score --profile jazzy_hiphop \
      --models <TOP-MODEL> --agent <AGENT>@v1 --rounds 2
  ```
### 8b. Agent showdown — which playbook wins? (batched, resumable)

`eval/duels.py agents` runs the whole matrix in ONE command: each model is
loaded once and duels itself — `artist-then-album` vs `album-first` — on every
profile. You still judge each duel (randomized letters, like `taste.py duel`);
progress is saved after every duel, so `q`/Ctrl-C and re-run the same command
to resume. One model at a time → respects the one-big-model limit.

- [ ] 4 medium models × 3 profiles = 12 duels:
  ```bash
  venv/bin/python eval/duels.py agents
  ```
  (defaults: Dolphin3.0, Meta-Llama-3.1, Mistral-7B, Phi-4-mini ×
  jazzy_hiphop, warm_groove, alt_rnb. Add `--dry-run` to preview the queue,
  `--session NAME` to pin the resume name, `venv/bin/python eval/duels.py status`
  to check progress.)

### 8c. Model showdown — which model wins, each with its preferred agent?

- [ ] Pick the survivors from 8a/8b, then (keep at most ONE big model per
      matchup — Qwen + a small is fine, two bigs is not; the driver warns):
  ```bash
  venv/bin/python eval/duels.py models \
      --contestants Qwen3.8-27B-Q8_0:album-first@v1,Mistral-7B-Instruct-v0.3-Q8_0 \
      --profiles jazzy_hiphop,warm_groove,alt_rnb
  ```
  (Omit `:agent` to use the selected agent for that model.)

- [ ] `venv/bin/python eval/taste.py stats` — response scores, Elo, win-rates
      (agent-duel standings appear as `Model [agent]` units, so the two
      playbooks get separate ladders per model)

## Phase 9 — wyatt (last; 18 reviews — slowest, most personal)

- [ ] Triage: survivors × both agents, `--profile wyatt`
- [ ] Interactive labeling: survivors × both agents, `--interactive`
- [ ] `taste.py pending --judge`, curate `eval/golden/wyatt.json`,
  score/duel the top combos, `taste.py stats`

---

## Final

- [ ] Revisit decision thresholds (EVAL_PLAN "Decision criteria") once
      ≥ 2 fixtures are labeled
- [ ] Lock production agent: `make select AGENT=<tag>` if album-first wins
