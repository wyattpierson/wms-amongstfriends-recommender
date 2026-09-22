"""AmongstFriends recommendation engine.

Modules:
    llm       — pluggable LLM client (llama.cpp llama-server / Ollama; backend-agnostic)
    reviews   — normalization of raw review payloads
    prompts   — prompt builders (the "engine" you want to experiment with)
    spotify   — Spotify verification of LLM-generated artist/album pairs
    pipeline  — orchestration: artists call → albums call → verify → retry
    firebase  — all Firebase interaction, isolated here so the rest can run offline

`recommend.py` is the production entry point (Firebase in/out).
`eval/evaluate.py` exercises `pipeline` offline against saved review profiles,
across models and prompt revisions — no Firebase required. `afrec/` (engine) and
`eval/` (measurement) are kept separate on purpose: see `eval/README.md`.
"""

__version__ = "0.2.0"
