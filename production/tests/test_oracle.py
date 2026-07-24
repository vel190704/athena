"""Milestone 20 validation: Oracle Substitution Validation (RQ5 ground-truth
closure). Uses real substitutions from match 3857276, the same cached match
used throughout Milestones 13-19, so results are directly comparable to
this project's other RQ5 evidence.

Reporting philosophy, consistent with Milestones 13/14's counterfactual
tests: classifications/deltas are printed as findings, not hard-asserted
in a particular direction -- this is an OBSERVATIONAL method (see
oracle_validator.py's module docstring), so a specific delta sign or
magnitude is not something correctness requires. What IS hard-asserted is
methodological soundness: valid float ranges, required keys, and -- most
importantly -- that `perspective_verified` is True for every single
result, since a False here would mean the critical fixed-team-perspective
requirement was violated.
"""

import math
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from production.src.ingestion.statsbomb_io import fetch_match_events
from production.src.models.explainer import load_deterministic_mlp
from production.src.pipeline.oracle_validator import find_substitutions, validate_oracle_substitutions

MATCH_ID = 3857276

REQUIRED_KEYS = {
    "sub_id",
    "team",
    "minute",
    "period",
    "player_off",
    "player_on",
    "threat_pre",
    "threat_post",
    "actual_delta",
    "classification",
    "perspective_verified",
    "overlapping_with",
}


def test_find_substitutions_returns_real_substitutions():
    events = fetch_match_events(MATCH_ID)
    substitutions = find_substitutions(events)

    assert len(substitutions) >= 1, "expected at least 1 substitution in match 3857276"
    for sub in substitutions:
        assert isinstance(sub["minute"], int)
        assert sub["period"] in (1, 2)
        assert isinstance(sub["team"], str) and sub["team"]
        assert isinstance(sub["player_off"], str) and sub["player_off"]
        assert isinstance(sub["player_on"], str) and sub["player_on"]

    print(f"\nFound {len(substitutions)} substitutions in match {MATCH_ID}:")
    for sub in substitutions:
        print(f"  #{sub['sub_id']}: period {sub['period']}, minute {sub['minute']}, {sub['team']}: "
              f"{sub['player_off']} -> {sub['player_on']}")


def test_validate_oracle_substitutions_end_to_end():
    model, normalization_mean, normalization_std, run_id = load_deterministic_mlp()
    print(f"\nLoaded deterministically-selected MLP from MLflow run_id={run_id}")

    results = validate_oracle_substitutions(MATCH_ID, model, normalization_mean, normalization_std)

    assert len(results) >= 1, (
        "expected at least 1 non-skipped substitution result -- if every substitution was "
        "skipped, check the printed skip reasons above"
    )

    for result in results:
        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"result missing required keys: {REQUIRED_KEYS - result.keys()}"
        )

        assert isinstance(result["threat_pre"], float) and math.isfinite(result["threat_pre"])
        assert isinstance(result["threat_post"], float) and math.isfinite(result["threat_post"])
        assert 0.0 <= result["threat_pre"] <= 1.0
        assert 0.0 <= result["threat_post"] <= 1.0
        assert isinstance(result["actual_delta"], float) and math.isfinite(result["actual_delta"])
        assert abs(result["actual_delta"] - (result["threat_post"] - result["threat_pre"])) < 1e-9

        # CRITICAL assertion: a False here means the fixed-team-perspective
        # requirement (Step 2 of this milestone) was violated for this
        # substitution -- this must be a hard failure, not a soft warning.
        assert result["perspective_verified"] is True, (
            f"perspective_verified is False for sub #{result['sub_id']} ({result['team']}, "
            f"minute {result['minute']}) -- this is a critical correctness violation"
        )

        assert isinstance(result["overlapping_with"], list)

    print(f"\n=== Oracle Substitution Validation (match {MATCH_ID}) ===")
    header = (
        f"{'sub_id':>6} {'team':<10} {'minute':>6} {'off':<22} {'on':<22} "
        f"{'pre':>8} {'post':>8} {'delta':>8} {'classification':<28} {'verified':>8} {'overlaps':<12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['sub_id']:>6} {result['team']:<10} {result['minute']:>6} "
            f"{result['player_off']:<22.22} {result['player_on']:<22.22} "
            f"{result['threat_pre']:>8.4f} {result['threat_post']:>8.4f} {result['actual_delta']:>+8.4f} "
            f"{result['classification']:<28} {str(result['perspective_verified']):>8} "
            f"{str(result['overlapping_with']):<12}"
        )

    overlapping_results = [r for r in results if r["overlapping_with"]]
    print(f"\n{len(overlapping_results)} / {len(results)} results have an overlapping substitution window "
          "(their deltas should be interpreted with extra caution -- see oracle_validator.py's module docstring).")
