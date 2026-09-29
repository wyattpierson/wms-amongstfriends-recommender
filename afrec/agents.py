"""
afrec.agents — the agent catalog: WHERE the agents are, and how to browse them.

The registry mechanism (Agent type, registration, lookup, selection) lives in
`afrec/agent_registry.py`. This file:

  1. re-exports the registry API, so `from afrec.agents import get_agent, …`
     keeps working everywhere;
  2. imports every agent module — each one self-registers on import, which is
     the ONLY list of agents that exists (a file that isn't imported here is
     not an agent);
  3. owns the human-facing reference table:

         $ make agents            # or: python -m afrec.agents

To add an agent: create `afrec/agent_<id>.py` ending in `_register(...)`,
then add its import below.
"""

from __future__ import annotations

from .agent_registry import (
    Agent,
    AgentRun,
    _register,
    add_alias,
    get_agent,
    default_agent,
    resolve_agent,
    selected_agent,
    list_agents,
)

# ── THE CATALOG ──────────────────────────────────────────────────────────────
# Each module self-registers on import. Order here = registration order =
# what "latest version of a bare id" means.
from . import agent_artist_then_album, agent_album_first  # noqa: E402,F401


def render_table() -> str:
    """Human-readable reference table: every registered agent + its knobs."""
    from .agent_registry import _REGISTRY

    selected = selected_agent().tag
    default = default_agent().tag
    lines = [
        "Agent reference table — every registered engine strategy",
        "",
        "An agent = the playbook (prompts + call flow + retry). The model is the brain;",
        "every scored run records both.  Production runs the SELECTED agent;",
        "eval commands take --agent <tag> (make: AGENT=<tag>) to override.",
        "",
    ]
    current_id = None
    for a in _REGISTRY.values():
        if a.id != current_id:
            current_id = a.id
            lines.append(f"■ {a.id}")
        marks = []
        if a.tag == selected:
            marks.append("selected")
        if a.tag == default:
            marks.append("default")
        mark_txt = f"   ({', '.join(marks)})" if marks else ""
        lines.append(f"  {a.tag:<28} {a.summary}{mark_txt}")
        knob_txt = "  ·  ".join(f"{k}: {v}" for k, v in a.details.items())
        if knob_txt:
            lines.append(f"{' ' * 30}{knob_txt}")
    lines += [
        "",
        "Compare two versions:  make eval AGENT=artist-then-album@v1   vs   AGENT=album-first@v1",
        "(same model, same profiles, different playbook — diff the leaderboards)",
        "Switch production:     make select AGENT=album-first@v1",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Running as `python -m afrec.agents` executes this file as __main__, so use
    # the *package* copy (the registry itself lives in afrec.agent_registry,
    # one canonical copy either way).
    from afrec import agents as _agents
    print(_agents.render_table())
