import pytest

from copilot import tools
from copilot.winnow import TOOL_FOR_KIND


def test_every_winnow_kind_has_a_builtin_tool():
    assert set(TOOL_FOR_KIND.values()) <= set(tools.available())


def test_dispatch_done():
    out = tools.dispatch({"name": "create_action_item", "args": {"text": "x", "owner": "Sam"}})
    assert out == {"status": "done", "result": {"logged": "action_item", "owner": "Sam", "due": "unscheduled"}}


def test_dispatch_unknown_tool_is_pending():
    assert tools.dispatch({"name": "post_to_crm", "args": {}})["status"] == "pending"


def test_dispatch_error_is_captured():
    @tools.tool("explodes")
    def explodes(**kwargs):
        raise RuntimeError("boom")
    try:
        out = tools.dispatch({"name": "explodes", "args": {}})
        assert out["status"] == "error" and "boom" in out["result"]["error"]
    finally:
        tools.REGISTRY.pop("explodes")
