"""Attacking-direction inference (Milestone 10, ADR-003) -- RETIRED from the
feature-extraction pipeline as of ADR-009. Kept in the codebase for the
evidence trail and in case a future data source genuinely needs it (see
below), but no longer called from feature_extractor.py.

Historical context: this module was built on the assumption that StatsBomb
records a single, shared, physically-fixed 0-120 x 0-80 coordinate system
for the whole match, with each team's attacking direction within that
shared frame unknown and requiring per-team, per-period inference (a coin
toss decides which end a team starts on, and it isn't guaranteed correct
even in period 1). Under that assumption, a team's period 2 direction was
derived as the guaranteed opposite of its period 1 direction (teams swap
ends at half-time), anchored to a correctly-inferred period 1 value.

That assumption turned out to be wrong. ADR-009 documents the actual
finding: StatsBomb's event `location` AND 360 freeze-frame coordinates are
already recorded relative to the ACTING team's own attacking-left-to-right
perspective, in both halves -- confirmed via goal-kick clustering,
turnover-coordinate mirroring, own-goal event pairs, and (for freeze-frame
data specifically) goalkeeper position clustering by teammate flag. Given
that, this module's independent per-period heuristic isn't measuring a
team's real-world attacking direction at all -- it's measuring a property
of the data's own encoding, which is why it resolved to the same direction
in both periods for 37 of 40 real team-periods checked (exactly the
"near-universal period1==period2" result that IS itself part of ADR-009's
evidence, not a bug here).

This module is still correctly implemented for what it computes, and is
retained because a genuinely shared-coordinate-frame data source (e.g. a
future Module 4 computer-vision/broadcast pipeline extracting raw pixel
coordinates from video, which will NOT arrive pre-normalized to an acting
team's frame) will need exactly this kind of per-team, per-period
direction inference. See ADR-009 for the full decision record.

All thresholds/heuristics here operate on StatsBomb's NATIVE 0-120
coordinate space, before any ADR-002 rescaling into the project's 100x68
grid.
"""

FINAL_FIFTH_X = 96.0  # 4/5 * 120: StatsBomb's raw-space final fifth boundary
SHOT_MEAN_X_THRESHOLD = 60.0  # pitch center in raw 0-120 space


def _team_events(events: list, team_name: str, period: int, type_name: str) -> list:
    return [
        e
        for e in events
        if e["period"] == period
        and e["type"]["name"] == type_name
        and e["team"]["name"] == team_name
    ]


def _final_fifth_touch_x_values(events: list, team_name: str, period: int) -> list[float]:
    """End-x of this team's Pass/Carry events that reach EITHER final fifth
    of the raw pitch (x > 96 or x < 24).

    Both ends are checked, not just x > 96: a team attacking toward
    decreasing x concentrates its deep touches near x < 24, not x > 96.
    Filtering to only x > 96 would make this fallback structurally
    incapable of ever detecting a -1 direction, defeating its purpose as a
    fallback for teams with zero shots in the period.
    """
    touch_x_values = []
    for e in _team_events(events, team_name, period, "Pass"):
        end_location = e.get("pass", {}).get("end_location")
        if end_location is not None:
            touch_x_values.append(end_location[0])
    for e in _team_events(events, team_name, period, "Carry"):
        end_location = e.get("carry", {}).get("end_location")
        if end_location is not None:
            touch_x_values.append(end_location[0])

    return [x for x in touch_x_values if x > FINAL_FIFTH_X or x < (120.0 - FINAL_FIFTH_X)]


def _infer_direction_signal(events: list, team_name: str, period: int) -> int | None:
    """Compute the direction signal with no side effects (no printing) --
    used both by the public `infer_attacking_direction` (which prints a
    warning if this comes back None) and by the period-2 validation-only
    check in `build_direction_lookup` (where a None result is unremarkable
    -- it just means there's nothing to cross-validate against -- and would
    otherwise trigger a misleading "excluding this team's chains" warning
    for a team whose period 2 direction was never in doubt).
    """
    shot_events = _team_events(events, team_name, period, "Shot")
    if shot_events:
        mean_x = sum(e["location"][0] for e in shot_events) / len(shot_events)
        return 1 if mean_x > SHOT_MEAN_X_THRESHOLD else -1

    final_fifth_touch_x_values = _final_fifth_touch_x_values(events, team_name, period)
    if final_fifth_touch_x_values:
        mean_x = sum(final_fifth_touch_x_values) / len(final_fifth_touch_x_values)
        return 1 if mean_x > SHOT_MEAN_X_THRESHOLD else -1

    return None


def infer_attacking_direction(
    events: list, team_name: str, period: int, match_id: int | None = None
) -> int | None:
    """Infer `team_name`'s attacking direction in `period` from raw (0-120)
    StatsBomb coordinates.

    Returns +1 (attacking toward increasing x), -1 (attacking toward
    decreasing x), or None if neither signal is available for this
    team/period (callers should exclude this team's chains for this period
    rather than guessing -- a warning is printed here either way).

    Primary signal: mean x of the team's Shot events in this period (a team
    shoots at the goal it's attacking). Falls back to the mean x of the
    team's final-fifth Pass/Carry end-locations if the team had zero shots
    in this period.
    """
    direction = _infer_direction_signal(events, team_name, period)
    if direction is not None:
        return direction

    match_ref = f"match {match_id}" if match_id is not None else "this match"
    print(
        f"[direction] WARNING: no shots or final-fifth touches for {team_name} in period "
        f"{period} of {match_ref} -- cannot infer attacking direction; excluding this "
        f"team's chains for this period."
    )
    return None


def build_direction_lookup(events: list, match_id: int | None = None) -> dict[str, dict[int, int | None]]:
    """Build a {team_name: {period: direction}} lookup for one match.

    Period 1's direction is the PRIMARY inference (infer_attacking_direction
    with period=1). Period 2's direction is NOT independently inferred as
    the primary method -- it is set directly to the guaranteed opposite of
    period 1 (`-direction[team][1]`), since teams swap ends at half-time.

    As a cheap validation-only check, if period 2 also has enough shots or
    final-fifth touches to independently estimate a direction, that estimate
    is computed and compared against the guaranteed value; a disagreement
    is printed as a warning (a genuine data anomaly worth seeing) but does
    NOT override the guaranteed value.

    A team whose period 1 direction can't be inferred has both periods set
    to None (there is no valid anchor for the guaranteed-opposite shortcut).
    """
    team_names = {e["team"]["name"] for e in events if "team" in e}

    lookup: dict[str, dict[int, int | None]] = {}
    for team_name in team_names:
        period1_direction = infer_attacking_direction(events, team_name, period=1, match_id=match_id)

        if period1_direction is None:
            lookup[team_name] = {1: None, 2: None}
            continue

        period2_direction = -period1_direction
        lookup[team_name] = {1: period1_direction, 2: period2_direction}

        # Validation-only: uses the silent signal helper, not the public
        # warning-emitting function -- a team with no period-2 shots/touches
        # (common; period 2's direction was never in doubt, it's guaranteed)
        # should not print a spurious "excluding this team's chains" warning.
        period2_independent_estimate = _infer_direction_signal(events, team_name, period=2)
        if (
            period2_independent_estimate is not None
            and period2_independent_estimate != period2_direction
        ):
            print(
                f"[direction] WARNING: {team_name} in match "
                f"{match_id if match_id is not None else '?'}, period 2: independently "
                f"inferred direction ({period2_independent_estimate:+d}) disagrees with the "
                f"guaranteed opposite-of-period-1 direction ({period2_direction:+d}). Using "
                f"the guaranteed value; this disagreement itself is a data anomaly worth "
                f"investigating."
            )

    return lookup
