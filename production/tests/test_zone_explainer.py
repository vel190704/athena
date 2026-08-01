"""Milestone 43 validation: the natural-language zone-explanation layer
built on top of Milestone 40's zone-level Integrated Gradients attribution.

Deliberately end-to-end against REAL match data (Argentina, the same
4-match/316-frame sample already validated in `test_reporting.py` and
documented in `REPORTING_FINDINGS.md`), same discipline as
`test_explainer.py`/`test_reporting.py`.
"""

import asyncio
import os
import re

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np

from production.src.models.explainer import generate_explanation
from production.src.reporting.team_report import generate_team_report
from production.src.reporting.zone_explainer import (
    aggregate_zone_attributions,
    build_zone_explanation_prompt,
    identify_notable_zones,
)

ARGENTINA_MATCH_IDS = [3857264, 3857289, 3857300, 3869151]

# Step 3's honesty check, made an executable assertion: none of these
# temporal/movement-specific words should ever appear in generated output,
# since no temporal/transition-state analysis exists anywhere in this
# project to genuinely support them.
_UNSUPPORTED_CLAIM_PATTERN = re.compile(
    r"\b(second|seconds|minute|minutes|recovering|recovers|sprint|sprinting)\b", re.IGNORECASE
)


def _grid_with_nan(grid_with_none: list[list[float | None]]) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in col] for col in grid_with_none])


def test_identify_notable_zones_real_argentina_data():
    """Step 4.1/4.2: real, already-computed Argentina zone attributions
    (Milestone 40's `aggregate_zone_attributions`) -- asserts the
    identified zones are sane and consistent with `REPORTING_FINDINGS.md`'s
    already-reported aggregate pattern (negative in the team's own half,
    positive in the attacking third)."""
    zone_agg = aggregate_zone_attributions("Argentina", ARGENTINA_MATCH_IDS)
    assert zone_agg["frames_used"] > 0

    grid = _grid_with_nan(zone_agg["mean_attacking_control_attribution_grid"])
    zones = identify_notable_zones(grid, top_n=3, source_label="attacking_control")

    assert zones, "expected at least one notable zone on real Argentina data"
    print("\nNotable zones (Argentina, attacking_control):")
    for z in zones:
        print(f"  {z}")

    positive_zones = [z for z in zones if z["sign"] == "positive"]
    negative_zones = [z for z in zones if z["sign"] == "negative"]
    assert positive_zones, "expected at least one positive (threat-driving) zone"
    assert negative_zones, "expected at least one negative (protective) zone"

    # Sanity check against REPORTING_FINDINGS.md's already-reported
    # aggregate pattern: positive zones should sit in the attacking third
    # (x > FINAL_THIRD_X = 66m); negative zones should sit in the team's
    # own half (x < 60m, i.e. not the attacking third).
    for z in positive_zones:
        x_m, _y_m = z["centroid_xy_m"]
        assert x_m > 60.0, f"expected positive zone {z['name']!r} to be in the attacking half, centroid x={x_m}"
    for z in negative_zones:
        x_m, _y_m = z["centroid_xy_m"]
        assert x_m < 60.0, f"expected negative zone {z['name']!r} to be in the team's own half, centroid x={x_m}"

    # Every zone's "source" must be traceable to what was actually passed in.
    for z in zones:
        assert z["source"] == "attacking_control"
        assert z["cell_count"] > 0
        assert isinstance(z["aggregate_attribution"], float)


def test_identify_notable_zones_per_sign_thresholding():
    """The threshold that decides "notable" must be computed SEPARATELY
    per sign (positive vs. negative), not from a single shared max(|grid|)
    -- otherwise a grid whose positive extreme is much larger than its
    negative extreme (the real, measured Argentina case) would silently
    lose every negative zone. Constructed here with a 3x magnitude
    asymmetry, matching the real data's own asymmetry in kind."""
    grid = np.zeros((10, 7))
    grid[8, 5] = 0.9  # strong positive extreme
    grid[9, 5] = 0.8
    grid[0, 0] = -0.3  # much weaker negative extreme (1/3 the positive one)
    grid[0, 1] = -0.25

    zones = identify_notable_zones(grid, top_n=5)
    signs = {z["sign"] for z in zones}
    assert "positive" in signs
    assert "negative" in signs, "a much-weaker negative extreme must still be found via its OWN threshold"


def test_build_zone_explanation_prompt_structure():
    """The prompt must preserve the EXACT literal header strings
    `explainer.generate_explanation`'s regex parsing depends on -- this is
    a structural requirement (Step 2's reuse constraint), not a style
    choice."""
    fake_team_report = {
        "team_name": "Testland",
        "threat_by_pitch_zone": {"defensive_third": 0.05, "middle_third": 0.10, "attacking_third": 0.40},
    }
    fake_zones = [
        {"name": "attacking third, central channel", "sign": "positive", "aggregate_attribution": 0.5, "cell_count": 4, "centroid_xy_m": (85.0, 34.0), "source": "attacking_control"},
        {"name": "defensive third, right side", "sign": "negative", "aggregate_attribution": -0.2, "cell_count": 3, "centroid_xy_m": (10.0, 10.0), "source": "attacking_control"},
    ]
    prompt = build_zone_explanation_prompt(fake_team_report, fake_zones, frames_used=123)

    assert "Predicted threat:" in prompt
    assert "Top factors INCREASING this threat estimate:\n" in prompt
    assert "Top factors DECREASING this threat estimate:\n" in prompt
    assert "\n\nProvide a concise" in prompt
    assert "- attacking_third_central_channel: +0.5000" in prompt
    assert "- defensive_third_right_side: -0.2000" in prompt
    assert "123 real, sampled" in prompt

    # Step 3 honesty constraint must be stated IN the prompt itself, not
    # just followed by convention.
    assert "MUST NOT state any temporal/duration claim" in prompt


def test_full_zone_explanation_pipeline_real_data_and_honesty_check():
    """Step 4.3: generates the full explanation text for real Argentina
    data via the REUSED (unmodified) Milestone 15 mock executor, prints it
    for direct review, and asserts it contains no claim Step 3's honesty
    check would flag as unsupported (temporal/duration/movement language
    this project has no real computation behind)."""
    team_report = generate_team_report("Argentina", ARGENTINA_MATCH_IDS)
    zone_agg = aggregate_zone_attributions("Argentina", ARGENTINA_MATCH_IDS)

    grid = _grid_with_nan(zone_agg["mean_attacking_control_attribution_grid"])
    zones = identify_notable_zones(grid, top_n=3, source_label="attacking_control")

    prompt = build_zone_explanation_prompt(team_report, zones, frames_used=zone_agg["frames_used"])
    explanation = asyncio.run(generate_explanation(prompt))

    print("\n=== PROMPT ===")
    print(prompt)
    print("\n=== GENERATED EXPLANATION ===")
    print(explanation)

    unsupported = _UNSUPPORTED_CLAIM_PATTERN.findall(explanation)
    assert not unsupported, f"explanation contains unsupported temporal/movement claim(s): {unsupported}"

    # Every zone identifier referenced in the explanation must trace back
    # to a zone this run actually computed -- not an invented name.
    known_identifiers = {z["name"].replace(", ", "_").replace(" ", "_") for z in zones}
    for token in re.findall(r"[a-z]+_third[a-z_]*", explanation):
        assert token in known_identifiers, f"explanation references unknown zone {token!r} not in {known_identifiers}"
