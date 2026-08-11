"""Automatic Match Report (new reporting track, Part A) validation.

Deliberately end-to-end against REAL match data (match 3857276, same cached
match used throughout this project's reporting/explainer tests), not
synthetic fixtures -- this project's established preference for real-data
verification over memory/assumption.
"""

import asyncio
import os
import re

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import pytest

from production.src.models.explainer import generate_tactical_explanation_with_source
from production.src.reporting.match_report import (
    build_match_report_narrative_prompt,
    generate_automatic_match_report,
)
from production.src.reporting.pass_network import generate_pass_network_aggregated

# Same pattern test_explainer.py's/test_zone_explainer.py's own honesty check
# uses -- duplicated here, not cross-imported, since it's a small,
# self-contained regex and tests should not depend on each other.
_UNSUPPORTED_CLAIM_PATTERN = re.compile(
    r"\b(second|seconds|minute|minutes|recovering|recovers|sprint|sprinting)\b", re.IGNORECASE
)

MATCH_ID = 3857276


def test_generate_automatic_match_report_real_match_both_sides():
    """Real match, real 360 coverage on both sides (same match used by
    Milestones 13-15's simulator/explainer tests) -- confirms the compiled
    document assembles all four real sub-reports without recomputing
    anything (generate_team_report/generate_team_opposition_analysis/
    generate_pass_network/fetch_alerts are called, not reimplemented)."""
    report = generate_automatic_match_report(MATCH_ID)

    assert report["no_data"] is False
    assert len(report["teams"]) == 2
    assert report["teams"] == sorted(report["teams"])  # deterministic ordering, stated in the docstring

    for team in report["teams"]:
        team_report = report["team_reports"][team]
        assert team_report["team_name"] == team
        assert "threat_by_pitch_zone" in team_report

        opposition = report["opposition_analysis"][team]
        assert opposition["team_name"] == team
        assert "build_up_tendency" in opposition
        assert "set_piece_reliance" in opposition

    assert report["pass_network"]["match_id"] == MATCH_ID
    assert isinstance(report["alerts"], list)
    assert report["alert_count"] == len(report["alerts"])


def test_generate_automatic_match_report_accepts_aggregated_pass_network_fn():
    """ADR-021 condition-2 compliance: the caller (api.py, under
    PUBLIC_DEPLOYMENT) passes generate_pass_network_aggregated in instead
    of the default -- confirms this module's own `pass_network_fn`
    parameter is real, not decorative, and that the aggregated variant's
    output (no `nodes`/`edges`) flows straight through unmodified."""
    report = generate_automatic_match_report(MATCH_ID, pass_network_fn=generate_pass_network_aggregated)
    assert "nodes" not in report["pass_network"]
    assert "edges" not in report["pass_network"]
    assert "num_players" in report["pass_network"]


def test_generate_automatic_match_report_no_data_for_unknown_match():
    """A match_id with no fetchable event data at all -- `_teams_in_match`
    returns an empty set, fewer than 2 teams, so this must return the
    honest `no_data` response rather than crash or fabricate teams."""
    report = generate_automatic_match_report(match_id=1)
    assert report["no_data"] is True
    assert "reason" in report


def test_build_match_report_narrative_prompt_contains_only_real_assembled_numbers():
    """The prompt text must be traceable to `generate_automatic_match_report`'s
    own real output -- spot-checks a handful of real numbers/team names
    appear verbatim in the built prompt, confirming this is a real
    grounding step, not a template with placeholder text."""
    report = generate_automatic_match_report(MATCH_ID)
    prompt = build_match_report_narrative_prompt(report)

    for team in report["teams"]:
        assert team in prompt

    assert str(MATCH_ID) in prompt
    assert f"Tactical alerts raised during this match: {report['alert_count']}" in prompt
    # Honesty framing must be present, not just the system_instruction --
    # defense in depth, matching build_zone_explanation_prompt's own
    # explicit inline reinforcement.
    assert "MUST NOT state any temporal/duration claim" in prompt


def test_automatic_match_report_real_gemini_narrative_passes_honesty_check():
    """THE real, triggered honesty test (Part A.4) -- skipped, not failed,
    if GEMINI_API_KEY isn't set (same convention as
    test_explainer.py::test_explainer_real_gemini_integration). Reuses the
    SAME `generate_tactical_explanation_with_source` dispatcher and the
    SAME `_HONESTY_SYSTEM_INSTRUCTION` every other real-Gemini call in this
    project already goes through -- nothing new is added to the honesty
    enforcement itself; this only confirms it holds for THIS prompt shape
    too (a whole-match compilation, not a single-frame/single-team
    attribution).
    """
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip(
            "GEMINI_API_KEY is not set. This test exercises the real Gemini Flash-Lite "
            "integration and requires a real API key (see test_explainer.py's own skip "
            "message for the full convention)."
        )

    report = generate_automatic_match_report(MATCH_ID)
    prompt = build_match_report_narrative_prompt(report)

    narrative, source = asyncio.run(generate_tactical_explanation_with_source(prompt))

    assert isinstance(narrative, str)
    assert narrative.strip()

    unsupported = _UNSUPPORTED_CLAIM_PATTERN.findall(narrative)
    assert not unsupported, (
        f"real Gemini match-report narrative contains unsupported temporal/movement claim(s): "
        f"{unsupported} -- the honesty-constraint system instruction needs strengthening, not "
        "this test relaxing"
    )

    print(f"\n=== Real Gemini match-report narrative (source={source}) ===\n{narrative}")
