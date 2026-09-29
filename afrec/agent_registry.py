"""
afrec.agent_registry — the mechanism for versioned recommendation *agents*.

The model is the **brain**; the agent is the **playbook** (prompts, call
flow, retry policy). Keeping them separate lets you A/B:

  - same model, two agents  →  "does the new strategy work?"
  - same agent, two models  →  "which brain is best at THIS strategy?"

Every scored run (automated metrics AND human verdicts) records BOTH the
model and the agent tag, so results from different playbooks never silently
mix.

This file is the pure mechanism: it knows NOTHING about specific agents.
It provides the `Agent` type, the registry, and:

    _register(...)     how an agent module publishes a version
    add_alias(...)     how a retired tag (post-rename) keeps resolving
    get_agent(tag)     full tag / bare id (→ latest) / alias
    default_agent()    the registry default
    selected_agent()   $AFREC_AGENT (from .env) → default
    resolve_agent(x)   Agent | str | None → Agent   (what pipeline.run uses)
    list_agents()      everything registered

Concrete agents live in `afrec/agent_*.py`; `afrec/agents.py` imports them
(which triggers their `_register(...)` calls) and owns the human-facing
reference table (`make agents` / `python -m afrec.agents`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from . import pipeline

# (reviews, llm, *, verifier, temperature, num_predict_artists, num_predict_albums,
#  extra_options, log) -> pipeline.RunOutcome
AgentRun = Callable[..., pipeline.RunOutcome]

_REGISTRY: dict[str, "Agent"] = {}
_DEFAULT_TAG: str | None = None
_ALIASES: dict[str, str] = {}   # old tag → current tag (for records written before a rename)


@dataclass(frozen=True)
class Agent:
    id: str
    version: str
    name: str
    summary: str
    run: AgentRun
    details: dict = field(default_factory=dict)   # structured knobs: calls, retry policy, prompts, …

    @property
    def tag(self) -> str:
        return f"{self.id}@{self.version}"


def _register(
    *,
    id: str,
    version: str,
    name: str,
    summary: str,
    run: AgentRun,
    details: dict | None = None,
    default: bool = False,
) -> Agent:
    """Register an agent version. Called by agent modules at import time.

    A registered version is immutable: to change a strategy, register a NEW
    version (v2, v3…) and compare tags — never re-register an existing one.
    The returned agent's `run` stamps every RunOutcome with the tag.
    """
    global _DEFAULT_TAG
    tag = f"{id}@{version}"
    if tag in _REGISTRY:
        raise ValueError(f"agent '{tag}' is already registered — never re-register a published version")

    def _run(reviews, llm, **kw):
        outcome = run(reviews, llm, **kw)
        outcome.agent = tag           # stamp the version on every run
        return outcome

    agent = Agent(id=id, version=version, name=name, summary=summary,
                  run=_run, details=details or {})
    _REGISTRY[tag] = agent
    if default or _DEFAULT_TAG is None:
        _DEFAULT_TAG = tag
    return agent


def add_alias(old_tag: str, new_tag: str) -> None:
    """Map a retired tag (e.g. after a rename) to its current agent."""
    if new_tag not in _REGISTRY:
        raise ValueError(f"cannot alias to unregistered agent '{new_tag}'")
    _ALIASES[old_tag] = new_tag


def get_agent(tag: str) -> Agent:
    """Resolve a full tag ('artist-then-album@v1'), a bare id ('album-first'
    → latest registered version of that id), or an alias."""
    tag = _ALIASES.get(tag, tag)
    if tag in _REGISTRY:
        return _REGISTRY[tag]
    bare = tag.split("@", 1)[0]
    matches = [a for t, a in _REGISTRY.items() if a.id == bare]
    if not matches:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"unknown agent '{tag}'. Available: {known}")
    return matches[-1]   # dict order == registration order → last registered wins


def default_agent() -> Agent:
    assert _DEFAULT_TAG, "no agents registered"
    return _REGISTRY[_DEFAULT_TAG]


def resolve_agent(agent: "Agent | str | None") -> Agent:
    """Accept an Agent instance, a tag string, or None (→ the selected agent)."""
    if agent is None:
        return selected_agent()
    if isinstance(agent, Agent):
        return agent
    return get_agent(agent)


def selected_agent() -> Agent:
    """
    The agent the production/pipeline path runs when no agent is given.

    Resolution: $AFREC_AGENT (set in .env — `make select AGENT=<tag>`) →
    registry default. An unknown $AFREC_AGENT is a configuration error and
    raises: production should fail loudly, not silently switch playbooks.
    """
    env_tag = os.getenv("AFREC_AGENT", "").strip()
    if env_tag:
        try:
            return get_agent(env_tag)
        except ValueError:
            raise ValueError(
                f"AFREC_AGENT='{env_tag}' is not a registered agent. "
                f"Available: {', '.join(sorted(_REGISTRY))}. "
                f"Fix .env or `make select AGENT=<tag>`."
            ) from None
    return default_agent()


def list_agents() -> list[Agent]:
    return list(_REGISTRY.values())
