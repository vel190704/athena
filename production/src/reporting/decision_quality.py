"""Decision Quality (Phase 4, final item -- correctly sequenced last, per
the roadmap's own framing, since it requires composing multiple
already-built signals against real event outcomes).

Answers: was a player's pass under pressure the RIGHT choice, given what
was ACTUALLY available to them at that moment? Composes 3 real signals:
(a) Press Resistance's own real per-event pressure/success signal, (b)
Passing Lane Visualizer's own real per-pass lane-openness computation for
the option the player ACTUALLY chose, and (c) a NEW per-frame computation
(Step 0's own finding -- see below) of the BEST AVAILABLE ALTERNATIVE lane
at that same moment, against the REAL recorded outcome (completed or not).

=============================================================================
STEP 0: THE REAL SCOPING FINDING (read before changing anything below) --
tonight's pattern has been that composition tasks reveal a genuine,
non-obvious problem in Step 0. This one did too, but NOT the one this
task's own framing predicted -- worth stating precisely rather than
forcing the finding to match the prediction.

1. Pressure + outcome signal: FULLY AVAILABLE, reused directly, no new
   engineering. `statsbomb_io.event_is_under_pressure` (the real
   `under_pressure` flag) is genuinely per-event. Pass completion reuses
   the SAME one-line convention `player_report._is_successful_pass_under_
   pressure` already established (`pass.outcome.name != "Incomplete"`) --
   reimplemented here as the identical check (per this project's own
   stated convention against importing another module's leading-
   underscore, module-private helper across file boundaries -- see that
   function's own docstring), not a new definition.

2. ACTUAL-CHOICE lane openness: FULLY AVAILABLE, reused directly --
   CONTRARY to this task's own hinted expectation that this would be the
   blocker. Checked directly (not assumed) before writing anything else:
   `team_report._lane_openness_for_pass` already computes a REAL
   per-SINGLE-PASS, single-frame lane-openness score (`1.0 - max
   defending-team control along the sampled segment from the real pass's
   own `location` to its own `end_location``) -- it is NOT a season
   aggregate at its own core; it is simply a private helper that
   `_team_passing_lane_samples`/`generate_team_passing_lanes` happen to
   consume internally and then throw away after aggregating into
   per-TEAMMATE-PAIR means. The per-decision value genuinely already
   existed; it was just never surfaced on its own. Reused here UNMODIFIED,
   via direct cross-file call (the same "reuse a substantial existing
   computation rather than duplicate real physics-adjacent logic"
   precedent `tactical_events.py` already set by importing `pass_network.
   _is_complete_pass` directly).

3. BEST-ALTERNATIVE lane openness: THIS is the real, genuine scoping gap
   -- confirmed directly. No existing function anywhere in this project
   evaluates lane openness from a passer's own location to any target
   OTHER than the actual pass's own recorded `end_location` -- there is
   no "openness to every visible teammate" computation anywhere.
   JUDGED BUILDABLE within this task's own reasonable scope (unlike
   Tactical Timeline's own deferred batch-momentum decision): it reuses
   the EXACT SAME sampling/engine-query pattern `_lane_openness_for_pass`
   already established (`team_report._lane_sample_points`, a generic
   sampling utility, and `BiomechanicalPitchControl`'s own general-purpose
   query-point interface -- both imported UNMODIFIED) generalized to an
   ARBITRARY target point instead of one hardcoded to the pass's own
   `end_location`. This is a new CALLING PATTERN of already-existing,
   unmodified physics, not a new physics model -- see
   `_lane_openness_to_target` below.

CONCLUSION: Decision Quality is built at FULL scope (actual choice vs.
real best alternative, not a coarser fallback).

=============================================================================
STEP 1 (ADR-021), THE OTHER REAL FINDING THIS STEP 0 SURFACED: unlike
every other Phase 4 composition tonight, this ONE genuinely risks the
Pass-Network/Passing-Lane risk pattern (a NAMED individual + a precise
LOCATION together) -- `team_report._team_passing_lane_samples`'s own real
per-pass fields (`passer_id`, `passer_name`, `passer_location`, ...)
already carry exactly that combination, and Decision Quality's own RAW
per-decision output necessarily does too (the question itself --
"was THIS player's choice at THIS moment the right one" -- is inherently
about a named individual's own specific location). Resolved the SAME way
Pass Network/Passing Lanes already resolved this exact risk (ADR-014's
"scope the constraint, don't remove the capability" precedent): a RAW,
per-decision variant (`generate_decision_quality_analysis`, LOCAL/PRIVATE
ONLY) and an AGGREGATED, public-safe counterpart
(`generate_decision_quality_analysis_aggregated`, per-player rates only,
no location, no individual pass) -- see ADR-021's own addendum for the
full reasoning.
"""

import logging

import torch

from production.src.ingestion.statsbomb_io import (
    X_SCALE,
    Y_SCALE,
    event_is_under_pressure,
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.reporting.team_report import (
    LANE_MIN_COVERED_SAMPLES,
    LANE_SAMPLE_POINTS,
    _lane_openness_for_pass,
    _lane_sample_points,
)
from production.src.spatial.control import BiomechanicalPitchControl

logger = logging.getLogger(__name__)

# STEP 0's own disclosed judgment call: how close the CHOSEN lane's
# openness must be to the REAL best available alternative to count as a
# "good decision" -- a real, hand-picked tolerance, not zero (an exact
# tie is an unreasonably strict bar given real floating-point pitch-
# control noise between two geometrically similar lanes), verified
# against this project's own real data before finalizing (see Step 3 in
# this feature's own report/test file for the exact real distribution of
# openness_gap this was checked against).
GOOD_DECISION_OPENNESS_TOLERANCE = 0.05


def _lane_openness_to_target(
    start_xy: torch.Tensor,
    target_xy: torch.Tensor,
    defending_pos: torch.Tensor,
    defending_vel: torch.Tensor,
    defending_fatigue: torch.Tensor,
    ball_pos: torch.Tensor,
    engine: BiomechanicalPitchControl,
) -> float | None:
    """Step 0.3's new piece: `team_report._lane_openness_for_pass`'s own
    EXACT algorithm (`1.0 - max` per-sample `max`-defender-control along
    the sampled segment -- see that function's own docstring for the two
    distinct reductions this mirrors), generalized to an ARBITRARY
    `target_xy` instead of one hardcoded to a real pass event's own
    `end_location`. Reuses `_lane_sample_points`/`BiomechanicalPitchControl`
    UNMODIFIED -- not a new physics model, a new calling pattern of an
    existing one. Returns `None` under the SAME real coverage condition
    that function already established (fewer than `LANE_MIN_COVERED_SAMPLES`
    sample points within the engine's own `mask_radius` of the ball).
    """
    lane_points = _lane_sample_points(start_xy, target_xy, LANE_SAMPLE_POINTS)
    _, control_probabilities, _ = engine(defending_pos, defending_vel, defending_fatigue, lane_points, ball_pos)
    if control_probabilities.shape[1] < LANE_MIN_COVERED_SAMPLES:
        return None
    defending_control_per_sample = control_probabilities.max(dim=0).values
    return 1.0 - defending_control_per_sample.max().item()


def _is_successful_pass_under_pressure(event: dict) -> bool:
    """Reimplements `player_report._is_successful_pass_under_pressure`'s
    EXACT one-line check (outcome-key-value ABSENCE-of-"Incomplete" =
    complete), per this project's own stated convention against importing
    another module's leading-underscore, private helper across file
    boundaries for small, single-purpose checks like this one -- NOT a
    new or different completion definition."""
    return event.get("pass", {}).get("outcome", {}).get("name") != "Incomplete"


def generate_decision_quality_analysis(team_name: str, match_id: int) -> dict:
    """RAW, per-decision Decision Quality analysis for `team_name`'s own
    real passes made UNDER PRESSURE in `match_id` -- LOCAL/PRIVATE USE
    ONLY (ADR-021: see this module's own Step 1 comment and ADR-021's own
    addendum; `generate_decision_quality_analysis_aggregated` below is
    the public-safe counterpart).

    For each real, 360-covered, under-pressure Pass event by `team_name`:
    the REAL chosen lane's openness (`team_report._lane_openness_for_pass`,
    unmodified), the REAL best-available-alternative lane's openness
    (`_lane_openness_to_target`, evaluated against every OTHER real,
    visible teammate in the SAME freeze frame -- excluding the actor
    themselves), the real recorded outcome, and a derived
    `openness_gap`/`good_decision` verdict (Step 0's own disclosed
    `GOOD_DECISION_OPENNESS_TOLERANCE`).

    SINGLE match_id, not a `match_ids` list -- a "best alternative
    available AT THAT MOMENT" is an inherently single-match, single-frame
    concept, the same reasoning `generate_weak_spot_lifetime_analysis`/
    `tactical_events.detect_tactical_events` already established for their
    own within-match temporal/spatial data.
    """
    events = fetch_match_events(match_id)
    frames = fetch_match_360(match_id)
    if events is None or frames is None:
        return {
            "team_name": team_name,
            "match_id": match_id,
            "no_data": True,
            "reason": "No event or 360 data available for this match_id.",
        }
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    engine = BiomechanicalPitchControl()
    decisions: list[dict] = []

    for event in events:
        if event.get("type", {}).get("name") != "Pass":
            continue
        if event.get("team", {}).get("name") != team_name:
            continue
        if not event_is_under_pressure(event):
            continue
        passer = event.get("player")
        if passer is None or passer.get("id") is None:
            continue
        location = event.get("location")
        end_location = event.get("pass", {}).get("end_location")
        if location is None or end_location is None:
            continue
        frame_data = frames_by_event_uuid.get(event["id"])
        if frame_data is None:
            continue

        chosen_openness = _lane_openness_for_pass(event, frame_data, engine)
        if chosen_openness is None:
            continue

        parsed = parse_360_frame(event, frame_data)
        defending_mask = ~parsed["is_teammate"]
        defending_pos = parsed["player_pos"][defending_mask]
        if defending_pos.shape[0] == 0:
            continue
        defending_vel = parsed["player_vel"][defending_mask]
        defending_fatigue = parsed["fatigue_mod"][defending_mask]

        # Every OTHER real, visible teammate (excluding the actor
        # themselves) in this SAME frame is a real candidate alternative
        # target -- their own real freeze-frame position, not a
        # hypothetical one.
        teammate_mask = parsed["is_teammate"] & (~parsed["is_actor"])
        teammate_positions = parsed["player_pos"][teammate_mask]

        start_xy = torch.tensor([location[0] * X_SCALE, location[1] * Y_SCALE], dtype=torch.float32)
        alternative_opennesses = []
        for i in range(teammate_positions.shape[0]):
            alt = _lane_openness_to_target(
                start_xy, teammate_positions[i], defending_pos, defending_vel, defending_fatigue,
                parsed["ball_pos"], engine,
            )
            if alt is not None:
                alternative_opennesses.append(alt)
        if not alternative_opennesses:
            continue
        best_alternative_openness = max(alternative_opennesses)

        successful = _is_successful_pass_under_pressure(event)
        openness_gap = chosen_openness - best_alternative_openness

        decisions.append(
            {
                "player_id": passer["id"],
                "player_name": passer["name"],
                "period": event["period"],
                "minute": event["minute"] + event["second"] / 60.0,
                "location": [location[0] * X_SCALE, location[1] * Y_SCALE],
                "chosen_lane_openness": chosen_openness,
                "best_alternative_lane_openness": best_alternative_openness,
                "n_alternatives_considered": len(alternative_opennesses),
                "openness_gap": openness_gap,
                "successful": successful,
                "good_decision": openness_gap >= -GOOD_DECISION_OPENNESS_TOLERANCE,
            }
        )

    n_good = sum(1 for d in decisions if d["good_decision"])
    n_successful = sum(1 for d in decisions if d["successful"])
    n_good_and_successful = sum(1 for d in decisions if d["good_decision"] and d["successful"])

    return {
        "team_name": team_name,
        "match_id": match_id,
        "no_data": False,
        "good_decision_openness_tolerance": GOOD_DECISION_OPENNESS_TOLERANCE,
        "decisions": decisions,
        "total_decisions": len(decisions),
        "good_decision_count": n_good,
        "good_decision_share": (n_good / len(decisions)) if decisions else None,
        "successful_count": n_successful,
        "successful_share": (n_successful / len(decisions)) if decisions else None,
        "good_and_successful_count": n_good_and_successful,
    }


def generate_decision_quality_analysis_aggregated(team_name: str, match_id: int) -> dict:
    """ADR-021 condition-2-compliant variant of
    `generate_decision_quality_analysis` -- mirrors `pass_network.
    generate_pass_network_aggregated`'s own raw-pop-then-summarize pattern
    exactly (reuses the raw function internally rather than re-deriving
    its own fetch/filter logic; the raw `decisions` list -- and every
    individual player name/location/openness value inside it -- is a
    LOCAL variable of this function only, popped via `dict.pop`, never
    left on the returned dict).

    Returns real per-PLAYER decision counts/rates ONLY -- no location, no
    individual pass, no per-decision detail -- the SAME aggregate-rate
    shape `generate_player_press_resistance_index` already established as
    condition-2-EXEMPT (a rate/count pair, no spatial or temporal context
    finer than a per-player total).
    """
    full = generate_decision_quality_analysis(team_name, match_id)
    if full.get("no_data"):
        return full

    decisions = full.pop("decisions")

    per_player: dict[int, dict] = {}
    for decision in decisions:
        player_id = decision["player_id"]
        entry = per_player.setdefault(
            player_id,
            {
                "player_id": player_id,
                "player_name": decision["player_name"],
                "total_decisions": 0,
                "good_decision_count": 0,
                "successful_count": 0,
            },
        )
        entry["total_decisions"] += 1
        entry["good_decision_count"] += int(decision["good_decision"])
        entry["successful_count"] += int(decision["successful"])

    player_summary = [
        {
            **entry,
            "good_decision_share": entry["good_decision_count"] / entry["total_decisions"],
            "successful_share": entry["successful_count"] / entry["total_decisions"],
        }
        for entry in per_player.values()
    ]
    player_summary.sort(key=lambda p: -p["total_decisions"])

    return {**full, "player_summary": player_summary}
