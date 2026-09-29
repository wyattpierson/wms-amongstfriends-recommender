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
PROFILE ?= jazzy_hiphop
MODELS  ?=
AGENT   ?=
ROUNDS  ?= 2
PORT    ?= 11500

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
	@echo "  stats                            current scores, win-rates, Elo, calibration"
	@echo ""
	@echo "  HUMAN, OFFLINE — write your taste down once per profile:"
	@echo "    edit eval/golden/<profile>.json  (good/bad artists or albums;"
	@echo "    the automated leaderboard scores every model against it)"
	@echo "  TEST CASES"
	@echo "  fixture FB_UID=<uid> NAME=alex   turn a real Firebase user into a profile"
	@echo "  mock    [PORT=11500]           start fake LLM server (no model needed)"
	@echo ""
	@echo ""
	@echo "  SITE"
	@echo "  site                            build the static site/ from your judging data"
	@echo "                                  (push it → GitHub Pages shows it)"
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
	$(PYTHON) eval/evaluate.py $(MODELS_OPT) $(AGENT_OPT) $(if $(PROFILE),--profile $(PROFILE))

.PHONY: fast
fast:
	$(PYTHON) eval/evaluate.py $(MODELS_OPT) $(AGENT_OPT) --profile $(PROFILE) --no-spotify

.PHONY: csv
csv:
	@test -n "$(CSV)" || { echo "usage: make csv PROFILE=... CSV=out.csv"; exit 1; }
	$(PYTHON) eval/evaluate.py $(MODELS_OPT) $(AGENT_OPT) --profile $(PROFILE) --export-csv $(CSV)

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
	$(PYTHON) eval/evaluate.py $(MODELS_OPT) $(AGENT_OPT) --interactive --profile $(PROFILE)

.PHONY: score
score:
	$(PYTHON) eval/taste.py score $(MODELS_OPT) $(AGENT_OPT) --profile $(PROFILE) --rounds $(ROUNDS)

.PHONY: duel
duel:
	@test -n "$(MODELS)" || { echo "usage: make duel PROFILE=... MODELS=modelA,modelB"; exit 1; }
	$(PYTHON) eval/taste.py duel --models $(MODELS) $(AGENT_OPT) --profile $(PROFILE)

# Batched, resumable duels (eval/duels.py). DUELS passes straight through:
#   make duels DUELS='agents'                                  # 4 mediums x 3 profiles, agent A vs B
#   make duels DUELS='models --contestants a:agent@v1,b --profiles p1,p2'
#   make duels DUELS='agents --dry-run'                        # preview the queue
#   make duels DUELS='status'                                  # session progress
# Quit mid-session (q / Ctrl-C) and re-run the same command to resume.
.PHONY: duels
duels:
	@test -n "$(DUELS)" || { echo "usage: make duels DUELS='agents' | DUELS='models --contestants ...' | DUELS='status'"; exit 1; }
	$(PYTHON) eval/duels.py $(DUELS)

.PHONY: stats
stats:
	$(PYTHON) eval/taste.py stats

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
