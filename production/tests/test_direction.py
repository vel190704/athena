"""direction.py validation -- retained per ADR-009 for the evidence trail,
though direction.py is no longer called from feature_extractor.py (see
ADR-009: StatsBomb's event/360 data is already recorded relative to the
acting team's own attacking-left-to-right perspective, so no coordinate
flip is needed; direction.py's near-universal period1==period2 result
across teams IS the evidence for that conclusion, not a bug to fix).
"""

from production.src.ingestion.statsbomb_io import fetch_match_events
from production.src.pipeline.direction import build_direction_lookup, infer_attacking_direction

MATCH_ID = 3857276  # cached real StatsBomb open-data match


def test_build_direction_lookup_period2_is_structurally_guaranteed_opposite():
    """`build_direction_lookup`'s period-2 value must ALWAYS be the exact
    opposite of period 1's -- by construction, this is a rules-based
    derivation (teams swap ends at half-time), not an independent
    inference, so it must hold for every team with a resolved period-1
    direction, with no exceptions. This remains a valid correctness check
    of direction.py's own internal logic even though the module is no
    longer wired into feature_extractor.py.
    """
    events = fetch_match_events(MATCH_ID)
    lookup = build_direction_lookup(events, match_id=MATCH_ID)

    assert len(lookup) > 0
    for team, directions in lookup.items():
        if directions[1] is None:
            assert directions[2] is None
        else:
            assert directions[2] == -directions[1], (
                f"{team}: period 2 direction {directions[2]} is not the guaranteed "
                f"opposite of period 1's {directions[1]}"
            )


def test_independent_period2_inference_agrees_with_period1_evidence_for_adr009():
    """This is the evidence FOR ADR-009's per-actor-coordinate-convention
    conclusion, not a test of a "guaranteed swap that must be enforced."

    Calls infer_attacking_direction independently for both periods
    (bypassing build_direction_lookup's guaranteed-opposite derivation
    entirely) and reports how often the two agree. If StatsBomb's raw
    `location` values were recorded in a shared, physically-fixed
    coordinate frame (Milestone 10's original assumption), a team's
    independently-measured period-2 direction should be the OPPOSITE of
    its period-1 direction (teams swap ends at half-time) -- i.e. d2 ==
    -d1. Verified empirically across all 20 cached matches (40
    team-periods): this agreement happened in only 3/40 cases -- the other
    37 both independently resolved to the SAME direction (+1) in both
    periods. That near-universal period1==period2 result is only possible
    if the raw coordinates are already recorded relative to each acting
    team's own attacking-rightward perspective (confirmed independently via
    goal-kick clustering, turnover-coordinate mirroring, own-goal event
    pairs, and 360 freeze-frame goalkeeper clustering -- see ADR-009), NOT
    a shared physical frame. This test reports the rate rather than gating
    on a specific value, since the point is the historical finding, not a
    number that should stay fixed as the heuristic or cached data evolves.
    """
    events = fetch_match_events(MATCH_ID)
    team_names = {e["team"]["name"] for e in events if "team" in e}

    agree = disagree = no_signal = 0
    for team in team_names:
        d1 = infer_attacking_direction(events, team, period=1, match_id=MATCH_ID)
        d2 = infer_attacking_direction(events, team, period=2, match_id=MATCH_ID)
        if d1 is None or d2 is None:
            no_signal += 1
        elif d2 == -d1:
            agree += 1
        else:
            disagree += 1

    print(
        f"\nIndependent period1/period2 agreement for match {MATCH_ID}: "
        f"agree={agree}, disagree={disagree}, no_signal={no_signal} (of {len(team_names)} teams) "
        "-- a low agreement rate is evidence FOR ADR-009's per-actor convention, not against it."
    )
    assert agree + disagree + no_signal == len(team_names)
