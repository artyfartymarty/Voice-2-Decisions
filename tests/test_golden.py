"""Golden calls (tests/golden/calls.json): each must fire exactly its expected tool calls, and nothing it forbids.

The scripts run here as text partials, with no speech model. On the GPU box `python manage.py doctor --golden`
streams the same calls' WAVs through the real models and checks them against the same expectations.
"""
import pytest

from copilot import golden
from copilot.models import Call

CASES = golden.load()


@pytest.mark.django_db
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_golden_call(case):
    result = golden.play_script(case)
    assert golden.compare(case, result) == [], "\n".join(result["transcript"])


@pytest.mark.django_db
def test_golden_runs_leave_nothing_behind():
    golden.play_script(CASES[0])
    assert not Call.objects.exists()


def test_golden_compare_catches_regressions():
    case = CASES[0]
    good = {"tools": [{"tool": e["tool"], "args": e["args"], "status": "done"} for e in case["expect"]],
            "roles": ["agent", "caller"], "warnings": [], "transcript": []}
    assert golden.compare(case, good) == []
    wrong_site = {**good, "tools": [{**good["tools"][0], "args": {"customer_number": "204511", "icao": "KDAL"}}] + good["tools"][1:]}
    assert any("quote_lookup" in f for f in golden.compare(case, wrong_site))
    twice = {**good, "tools": good["tools"] + good["tools"][:1]}
    assert any("expected 1x, got 2x" in f for f in golden.compare(case, twice))
    stuck = {**good, "warnings": [{"code": "repeated_text", "message": "same text 3 times"}]}
    assert golden.compare(case, stuck) == ["transcript check: same text 3 times"]
    one_speaker = {**good, "roles": ["agent"]}
    assert golden.compare(case, one_speaker) == ["no caller found among speakers (got ['agent'])"]


def test_golden_wavs_exist():
    from django.conf import settings
    for case in CASES:
        if "wav" in case:
            assert (settings.BASE_DIR / case["wav"]).is_file(), case["wav"]
