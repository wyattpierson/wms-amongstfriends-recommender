"""Per-model LLM options — `afrec/model_options.json`.

This is the CANONICAL home of the table, in the engine (not eval/), because
the fixes it carries are model-side behavior, not agent-side: production
(`recommend.py`) and eval must feed the model identically. afrec/llm.py
merges this model's entry into every LlamaCPPLM.generate() payload
automatically; explicit caller options (e.g. eval's --thinking/--no-thinking
override) are merged on top and win.

Why this file exists: the llama-server API does not expose which models are
"thinking" models (Qwen3-style: they emit reasoning_content and run ~20x
slower per stage). So we keep an explicit map here of model name -> extra
options, merged into every generate() call for that model.

Keys are the model NAME as the client is constructed with (the llama-server
alias, or a GGUF path/filename — lookups also try the bare filename with the
.gguf suffix stripped). OllamaLM does not apply the table (different API
surface); the entries are llama.cpp /chat_completion options.

Template facts (read from the GGUFs in ~/llama_models, 2026-09-23):
  - Qwen3.8-27: thinking ON by default; enable_thinking=false renders an
    empty think block (the official /no_think mechanism); also supports
    reasoning_effort (xhigh default / medium / low).
  - gemma-4-26B / gemma-4-E4B: the template says default(false) for
    enable_thinking, BUT on hard prompts (long-context JSON generation) the
    model THOUGHT SPONTANEOUSLY anyway — ~1,200-1,500 thinking tokens in
    reasoning_content before any content. Those thinking tokens consume the
    agent's max_tokens budget (map 2500 / discovery 1500 / critic 1200), so
    the visible JSON came back truncated mid-string (or empty when thinking
    ate the whole budget) and every profile scored 0 recs (2026-10-04, E4B).
    enable_thinking=false suppresses the spontaneous thinking (verified on
    the exact failing prompts: finish=stop, reasoning=0, full JSON in-budget).
    KEEP the entry for both gemma models.
  - Dolphin / Meta-Llama-3.1 / Mistral / Phi-4: no thinking references in
    the template at all; the kwarg is a no-op.

So the blanket {"chat_template_kwargs": {"enable_thinking": false}} payload
is safe for every model in the pool — but this file makes the intent
per-model explicit and gives us a home for future per-model tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

OPTIONS_PATH = Path(__file__).resolve().parent / "model_options.json"


def options_for(model: str) -> dict | None:
    """Raw extra options for a model, or None if it has no entry.

    Lookup tries the exact string, then the bare filename, then the bare
    filename without .gguf — so table keys can be plain model names while
    callers pass aliases or GGUF paths.
    """
    if not model or not OPTIONS_PATH.exists():
        return None
    try:
        table = json.loads(OPTIONS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    candidates = [model]
    bare = Path(model).name
    if bare != model:
        candidates.append(bare)
    if bare.endswith(".gguf"):
        candidates.append(bare[: -len(".gguf")])
    for key in candidates:
        opts = table.get(key)
        if isinstance(opts, dict):
            return dict(opts)
    return None


def merged(model: str, thinking: bool | None = None) -> dict | None:
    """Options for a model, with an optional thinking override applied.

    thinking=None  -> use the file entry as-is
    thinking=True  -> force enable_thinking on  (spot-check mode)
    thinking=False -> force enable_thinking off
    """
    base = dict(options_for(model) or {})
    if thinking is not None:
        ctk = dict(base.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = thinking
        base["chat_template_kwargs"] = ctk
    return base or None
