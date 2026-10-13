"""Tool registry. winnow turns transcript moments into tool calls; dispatch() runs them.

Register a new integration (CRM, ticketing, Slack...) with the @tool decorator:

    @tool("create_action_item")
    def create_ticket(text, owner="", due="", at=None):
        ...
        return {"ticket": "ABC-123"}

A call to a tool that isn't registered yet is kept with status "pending", so it can be
replayed once the tool exists (see replay_pending in pipeline.py).
"""
import logging
from typing import Callable

log = logging.getLogger(__name__)
REGISTRY: dict[str, Callable[..., dict]] = {}


def tool(name: str | None = None):
    def register(fn):
        REGISTRY[name or fn.__name__] = fn
        return fn
    return register


def available() -> list[str]:
    return sorted(REGISTRY)


def dispatch(call: dict) -> dict:
    """Run {"name", "args"} -> {"status": done|pending|error, "result": ...}."""
    fn = REGISTRY.get(call.get("name", ""))
    if fn is None:
        return {"status": "pending", "result": {"reason": f"tool {call.get('name')!r} is not available yet"}}
    try:
        return {"status": "done", "result": fn(**call.get("args", {})) or {}}
    except Exception as exc:  # a broken integration must not lose the moment
        log.exception("tool %s failed", call.get("name"))
        return {"status": "error", "result": {"error": f"{type(exc).__name__}: {exc}"}}


# Built-in tools: they record the moment in the app's own database (the pipeline stores every
# moment), and return a small receipt. Replace or wrap them to push to real systems.

@tool()
def log_decision(text, at=None):
    return {"logged": "decision"}


@tool()
def create_action_item(text, owner="", due="", at=None):
    return {"logged": "action_item", "owner": owner or "unassigned", "due": due or "unscheduled"}


@tool()
def flag_risk(text, at=None):
    return {"logged": "risk"}


@tool()
def open_question(text, at=None):
    return {"logged": "question"}


@tool()
def mark_key_moment(text, at=None):
    return {"logged": "key_moment"}


from . import usecases  # noqa: E402,F401  registers quote_lookup, dispute_guidance, alert_manager
