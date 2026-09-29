"""Per-model LLM options — `eval/model_options.json`.

Why this file exists: the llama-server API does not expose which models are
"thinking" models (Qwen3-style: they emit reasoning_content and run ~20x
slower per stage). So we keep an explicit map here of model name -> extra
options, merged into every generate() call for that model.

Template facts (read from the GGUFs in ~/llama_models, 2026-09-23):
  - Qwen3.8-27: thinking ON by default; enable_thinking=false renders an
    empty think block (the official /no_think mechanism); also supports
    reasoning_effort (xhigh default / medium / low).
  - gemma-4-26B / gemma-4-E4B: thinking OFF by default (default(false));
    the kwarg is a no-op for them.
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
    """Raw extra options for a model, or None if it has no entry."""
    if not OPTIONS_PATH.exists():
        return None
    try:
        table = json.loads(OPTIONS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    opts = table.get(model)
    return dict(opts) if isinstance(opts, dict) else None


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
