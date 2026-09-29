"""AmongstFriends recommendation engine.

Modules:
    llm       — pluggable LLM client (llama.cpp llama-server / Ollama; backend-agnostic)
    reviews   — normalization of raw review payloads
    prompts   — prompt builders (the "engine" you want to experiment with)
    spotify   — Spotify verification of LLM-generated artist/album pairs
    pipeline  — orchestration: artists call → albums call → verify → retry
    agent_registry          — the registry mechanism (Agent type, register, lookup, selection)
    agents    — the catalog: imports every agent module + the reference table (make agents)
    agent_artist_then_album — the default two-call artists→albums agent
    agent_album_first       — the one-shot direct-albums agent (the A/B)
    firebase  — all Firebase interaction, isolated here so the rest can run offline

`recommend.py` is the production entry point (Firebase in/out).
`eval/evaluate.py` exercises `pipeline` offline against saved review profiles,
across models and prompt revisions — no Firebase required. `afrec/` (engine) and
`eval/` (measurement) are kept separate on purpose: see `eval/README.md`.

An *agent* is the engine's versioned playbook (prompts + call flow + retry);
each lives in `afrec/agent_<id>.py`, `pipeline.run()` is the orchestrator
that dispatches to the selected agent (AFREC_AGENT in .env, else the
default). Every scored run records which agent produced it.
`python -m afrec.agents` prints the reference table.
"""

__version__ = "0.3.0"
