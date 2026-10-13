# AmongstFriends recommender — one command surface for everything.
#
#   make            → shows this help
#   make <target>   → runs it; pass vars like  make dry FB_UID=<firebaseUid>
#
# Model name / URL come from .env (LLM_MODEL / LLM_URL); override with MODELS=.
# See README.md for the "I want to..." menu this file implements.

# Python: project venv if it exists (after `make setup`), else python3 to
# bootstrap. Override anytime:  make site PYTHON=/usr/bin/python3.12
ifeq ($(wildcard venv/bin/python),)
PYTHON ?= python3
else
PYTHON ?= venv/bin/python
endif
# NOTE: no PROFILE default — `eval` with no PROFILE must run ALL profiles.
# Targets that need exactly one profile fall back to jazzy_hiphop per-target
# (see fast/label/score) or require it explicitly (csv/duel).
MODELS  ?=
AGENT   ?=
ROUNDS  ?= 2
PORT    ?= 11500
MIN_COUNT ?= 2
ARTIST  ?=

# Only append to MODELS / AGENT if they weren't given on the command line
ifneq ($(MODELS),)
MODELS_OPT := --models $(MODELS)
else
MODELS_OPT :=
endif
ifneq ($(AGENT),)
AGENT_OPT := --agent $(AGENT)
else
AGENT_OPT :=
endif
# ENRICH=1 offers the enrichment layer's catalog facts to agents that accept
# them (pool-map@v1.2+): tags + sound measurements in the map call.
ENRICH_OPT := $(if $(ENRICH),--enrich)

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "AmongstFriends recommender — commands (see README.md for when to use each)"
	@echo ""
	@echo "  setup                          create venv + install deps"
	@echo ""
	@echo "  RUN THE RECOMMENDER (production, touches Firebase)"
	@echo "  dry  FB_UID=<uid>                fetch reviews + generate, submit NOTHING"
	@echo "  run  FB_UID=<uid> [GROUP=<id>]   real run — submits recommendations"
	@echo ""
	@echo "  AUTOMATED (no human in the loop)"
	@echo "  eval    [PROFILE=... AGENT=..] full leaderboard, all profiles, Spotify on"
	@echo "  fast    [PROFILE=... AGENT=..] one profile, no Spotify — seconds, for prompt editing"
	@echo "  csv     PROFILE=... CSV=out    every individual rec into a spreadsheet"
	@echo "  agents                         reference table of engine agent versions"
	@echo "  select AGENT=album-first@v1    set the agent production runs (writes .env)"
	@echo ""
	@echo "  HUMAN, LIVE (interactive — you watch the model and rate it)"
	@echo "  label   [PROFILE=... AGENT=..] rate each rec good/bad/unsure (grows the corpus)"
	@echo "  score   [PROFILE=... ROUNDS=2] rate whole responses 1-5"
	@echo "  duel    PROFILE=... MODELS=a,b  models face off, you pick; feeds Elo"
	@echo "  duels   DUELS='agents'|'models ...'|'status'  batched, resumable duel sessions"
	@echo "  grade   [LIMIT=3 MODEL=..]     grade individual recs from saved score/duel records"
	@echo "  stats                            current scores, win-rates, Elo, calibration"
	@echo ""
	@echo "  TAG HARVEST — understand + tag the recs the engine keeps making"
	@echo "  harvest [--min-count 2]          offline: which untagged artists recur (runs_history)"
	@echo "  harvest-lookup ARTIST='X' [SOURCE=auto|lastfm|musicbrainz]  sanitized suggest (human review)"
	@echo ""
	@echo "  ENRICHMENT — cache catalog facts (year/tags/measurements) for albums"
	@echo "  enrich-status                    offline: cache coverage + days left"
	@echo "  enrich-backfill [BUDGET=10 DELAY=1 DRY=1]  run whenever; free-tier paced"
	@echo "  enrich-smoke                     1 Last.fm + 1 Spotify + 1 Musicae call (key check)"
	@echo "  HUMAN, OFFLINE — write your taste down once per profile:"
	@echo "    edit eval/golden/<profile>.json  (good/bad artists or albums;"
	@echo "    the automated leaderboard scores every model against it)"
	@echo "  TEST CASES"
	@echo "  fixture FB_UID=<uid> NAME=alex   turn a real Firebase user into a profile"
	@echo "  mock    [PORT=11500]           start fake LLM server (no model needed)"
	@echo ""
	@echo ""
	@echo "  SITE"
	@echo "  site                            build the static docs/ from your judging data"
	@echo "                                  (push it → GitHub Pages shows it;"
	@echo "                                  Pages source: main branch, /docs)"
	@echo ""
	@echo "  CHECKS"
	@echo "  test                            offline tests (no network, no model)"

.PHONY: setup
setup:
	$(PYTHON) -m venv venv
	./venv/bin/pip install -r requirements.txt
	@echo "Done — make targets now use venv/bin/python automatically (no activate needed)"

# ── Production ────────────────────────────────────────────────────────────────

.PHONY: dry
dry:
	@test -n "$(FB_UID)" || { echo "usage: make dry FB_UID=<firebaseUid>"; exit 1; }
	$(PYTHON) recommend.py $(FB_UID) --dry-run

.PHONY: run
run:
	@test -n "$(FB_UID)" || { echo "usage: make run FB_UID=<firebaseUid> [GROUP=<id>]"; exit 1; }
	$(PYTHON) recommend.py $(FB_UID) --group-id $(or $(GROUP),default-group)

# ── Measure ───────────────────────────────────────────────────────────────────

.PHONY: eval
eval:
	$(PYTHON) eval/evaluate.py $(MODELS_OPT) $(AGENT_OPT) $(ENRICH_OPT) $(if $(PROFILE),--profile $(PROFILE))

.PHONY: fast
fast:
	$(PYTHON) eval/evaluate.py $(MODELS_OPT) $(AGENT_OPT) $(ENRICH_OPT) --profile $(or $(PROFILE),jazzy_hiphop) --no-spotify

.PHONY: csv
csv:
	@test -n "$(CSV)" && test -n "$(PROFILE)" || { echo "usage: make csv PROFILE=... CSV=out.csv"; exit 1; }
	$(PYTHON) eval/evaluate.py $(MODELS_OPT) $(AGENT_OPT) $(ENRICH_OPT) --profile $(PROFILE) --export-csv $(CSV)

.PHONY: agents
agents:
	$(PYTHON) -m afrec.agents

.PHONY: select
select:
	@test -n "$(AGENT)" || { echo "usage: make select AGENT=album-first@v1"; exit 1; }
	$(PYTHON) -c 'import sys; from pathlib import Path; tag=sys.argv[1]; p=Path(".env"); lines=[l for l in p.read_text().splitlines() if l.strip() and not l.startswith("AFREC_AGENT=")] if p.exists() else []; lines.append(f"AFREC_AGENT={tag}"); p.write_text("\n".join(lines)+"\n"); print(f"selected agent: {tag}  (written to .env — production & default evals now use it)")' "$(AGENT)"

# ── Human judging ─────────────────────────────────────────────────────────────

.PHONY: label
label:
	$(PYTHON) eval/evaluate.py $(MODELS_OPT) $(AGENT_OPT) $(ENRICH_OPT) --interactive --profile $(or $(PROFILE),jazzy_hiphop)

.PHONY: score
score:
	$(PYTHON) eval/taste.py score $(MODELS_OPT) $(AGENT_OPT) --profile $(or $(PROFILE),jazzy_hiphop) --rounds $(ROUNDS)

.PHONY: duel
duel:
	@test -n "$(MODELS)" && test -n "$(PROFILE)" || { echo "usage: make duel PROFILE=... MODELS=modelA,modelB"; exit 1; }
	$(PYTHON) eval/taste.py duel --models $(MODELS) $(AGENT_OPT) --profile $(PROFILE)

# Batched, resumable duels (eval/duels.py). DUELS passes straight through:
#   make duels DUELS='agents'                                  # 4 mediums x 3 profiles, agent A vs B
#   make duels DUELS='models --contestants a:agent@v1,b --profiles p1,p2'
#   make duels DUELS='agents --dry-run'                        # preview the queue
#   make duels DUELS='status'                                  # session progress
#   make duels DUELS='agents --agent-a pool-map@v1.1 --agent-b pool-map@v1.2 \
#                          --models Mistral-7B-Instruct-v0.3-Q8_0 --enrich'
#     ^ --enrich offers the enrichment layer's catalog facts to agents that
#       accept them (pool-map@v1.2+). Off by default: without it, v1.2 runs
#       byte-identical to v1.1, so an A/B of the two is meaningless.
# Quit mid-session (q / Ctrl-C) and re-run the same command to resume.
.PHONY: duels
duels:
	@test -n "$(DUELS)" || { echo "usage: make duels DUELS='agents' | DUELS='models --contestants ...' | DUELS='status'"; exit 1; }
	$(PYTHON) eval/duels.py $(DUELS)

# Grade individual recs from SAVED score/duel records (offline backfill — no
# model calls):   make grade                       # newest 10 records
#                 make grade LIMIT=3 MODEL=mistral # a few, one model
.PHONY: grade
grade:
	$(PYTHON) eval/taste.py grade $(if $(PROFILE),--profile $(PROFILE)) $(if $(MODEL),--model $(MODEL)) $(if $(LIMIT),--limit $(LIMIT))

.PHONY: stats
stats:
	$(PYTHON) eval/taste.py stats

# ── Tag harvest (eval/harvest.py) ─────────────────────────────────────────────
# Offline consensus from eval/output/runs_history.jsonl (model self-reports,
# display-only data) + MusicBrainz lookup (sanitized; NEVER fed to any model).
.PHONY: harvest
harvest:
	$(PYTHON) eval/harvest.py report --min-count $(MIN_COUNT)

.PHONY: harvest-lookup
harvest-lookup:
	@test -n "$(ARTIST)" || { echo "usage: make harvest-lookup ARTIST='Name Here' [SOURCE=auto|lastfm|musicbrainz]"; exit 1; }
	$(PYTHON) eval/harvest.py lookup "$(ARTIST)" $(if $(SOURCE),--source $(SOURCE)) --suggest

# ── Enrichment cache (eval/backfill.py) ───────────────────────────────────────
# Free tier: Musicae 10 calls/day (5 albums/call) → run these whenever, at the
# pace you choose. The cache makes every run strictly additive.
#   make enrich-status                    offline: what's cached / what's missing
#   make enrich-backfill                  Last.fm (free) + up to 10 Musicae calls
#   make enrich-backfill BUDGET=3 DELAY=30
#   make enrich-backfill DRY=1            print the plan, fire nothing
.PHONY: enrich-status
enrich-status:
	$(PYTHON) eval/backfill.py status

.PHONY: enrich-backfill
enrich-backfill:
	$(PYTHON) eval/backfill.py backfill $(if $(BUDGET),--budget $(BUDGET)) $(if $(DELAY),--delay $(DELAY)) $(if $(SOURCES),--sources $(SOURCES)) $(if $(DRY),--dry-run) $(if $(LASTFM_ONLY),--lastfm-only) $(if $(MUSICAE_ONLY),--musicae-only)

.PHONY: enrich-smoke
enrich-smoke:
	$(PYTHON) eval/backfill.py smoke

# ── Test cases ────────────────────────────────────────────────────────────────

.PHONY: fixture
fixture:
	@test -n "$(FB_UID)" || { echo "usage: make fixture FB_UID=<firebaseUid> NAME=<profile>"; exit 1; }
	$(PYTHON) eval/make_fixture.py --user-id $(FB_UID) $(if $(NAME),--profile $(NAME))

.PHONY: mock
mock:
	$(PYTHON) eval/mock_ollama_server.py --port $(PORT)

.PHONY: site
site:
	$(PYTHON) eval/site.py

# ── Checks ────────────────────────────────────────────────────────────────────

.PHONY: site test
test:
	$(PYTHON) eval/tests/test_agents.py
	$(PYTHON) eval/tests/test_offline.py
	$(PYTHON) eval/tests/test_spotify_existence.py
	$(PYTHON) eval/tests/test_humaneval.py
	$(PYTHON) eval/tests/test_goldenset.py
	$(PYTHON) eval/tests/test_enrich.py
