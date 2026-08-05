"""Milestone 20 (Module 8 / RQ5): Oracle Substitution Validation.

Closes RQ5's loop by comparing the trained MLP's predicted threat just
BEFORE a real historical substitution against just AFTER it, using the
model itself (not the counterfactual perturbation heuristics of Milestone
13) as the "oracle" -- README.txt Module 8's "Oracle Substitutions
Validation: ... If a sub occurred at minute 70, the simulator runs minutes
68-70 to verify it predicts the real-world xT shift post-substitution."

PERMANENT METHODOLOGICAL CAVEAT (read before trusting any result here):
this measures an OBSERVATIONAL before/after snapshot around a real
substitution, NOT an isolated causal effect. Other events inside the same
+/-2 minute window -- goals, cards, other substitutions, general momentum
shifts -- can and do confound the measured delta. A "threat increased
after sub" classification is evidence the model's assessment of the
substituting team's danger changed across that window, not proof the
substitution CAUSED that change. This is a real limitation of the
observational method itself, not something fixable in code, and every
result returned by `validate_oracle_substitutions` should be read with
this in mind.

CRITICAL DESIGN REQUIREMENT -- fixed team perspective: every scalar
feature in this project (since Milestone 3) is computed relative to
whichever team is ACTING/in-possession at the specific event a frame is
drawn from (`is_teammate`). The pre-frame and post-frame for a given
substitution MUST both be drawn from the SUBSTITUTING team's own
possession events, or the "before vs after" delta silently compares two
different teams' tactical situations rather than one continuous
measurement of the same team. See `validate_oracle_substitutions` for how
this is enforced and verified.
"""

import logging

from production.src.constants import TIME_BIN
from production.src.ingestion.statsbomb_io import fetch_match_events, parse_360_frame
from production.src.models.evaluation import predict_cumulative_incidence
from production.src.pipeline.feature_extractor import extract_features
from production.src.serving.api import _find_qualifying_frame_for_minute

logger = logging.getLogger(__name__)

# TIME_BIN (15s horizon, matching every prior milestone) now comes from
# production.src.constants (engineering-review de-duplication -- was
# defined locally here before; value unchanged).

# Hand-picked heuristic threshold for LABELING purposes only -- NOT a
# statistically validated significance threshold. Consistent with how
# every other hand-tuned constant in this project has been treated (Kalman
# Q/R, pitch-control radii, the WebSocket spike-detection threshold): it
# exists to produce a readable classification, not a claim of statistical
# significance.
CLASSIFICATION_THRESHOLD = 0.02

WINDOW_HALF_WIDTH_MINUTES = 2  # the "+/-2 minute" window from README.txt Module 8


def find_substitutions(events: list) -> list[dict]:
    """Extracts every Substitution event's (minute, period, team,
    player_off, player_on), using StatsBomb's REAL field layout -- verified
    directly against this project's cached match data before writing this
    function, per the project's established "verify schema, don't guess"
    rule (see Milestone 5's event-type surprises, Milestone 9's
    competition-JSON structure):

      - `event["team"]`  -- the team MAKING the substitution (a dict with
        "id"/"name"). There is no separate "substituting team" field;
        `team` on a Substitution event already IS that team.
      - `event["player"]` -- the OUTGOING player (`player_off`).
      - `event["substitution"]["replacement"]` -- the INCOMING player
        (`player_on`). This is nested under `substitution`, NOT a
        top-level field.

    Returns a list of dicts, in match order, each with a `sub_id` (a
    simple 0-indexed position within THIS match's substitution list --
    used later to cross-reference overlapping-window flags).
    """
    substitution_events = [event for event in events if event["type"]["name"] == "Substitution"]

    substitutions = []
    for sub_id, event in enumerate(substitution_events):
        substitutions.append(
            {
                "sub_id": sub_id,
                "minute": event["minute"],
                "period": event["period"],
                "team": event["team"]["name"],
                "team_id": event["team"]["id"],
                "player_off": event["player"]["name"],
                "player_on": event["substitution"]["replacement"]["name"],
            }
        )
    return substitutions


def _windows_overlap(period_a, start_a, end_a, period_b, start_b, end_b) -> bool:
    """Two [minute-2, minute+2] windows overlap only if they're in the SAME
    period -- raw minute values can numerically overlap ACROSS periods
    (period 1's stoppage time and period 2's opening minutes both fall in,
    e.g., the 45-50 range), but there is a real halftime break between
    them in actual match time, so cross-period windows never truly
    overlap regardless of their minute numbers.
    """
    if period_a != period_b:
        return False
    return start_a <= end_b and start_b <= end_a


def validate_oracle_substitutions(match_id, model, normalization_mean, normalization_std) -> list[dict]:
    """For every real substitution in `match_id`, compares the trained
    model's predicted cumulative incidence (threat_15s) just before vs.
    just after, from the SUBSTITUTING team's own perspective throughout.

    "Threat" here consistently means "danger the substituting team is
    generating," measured only from events where THAT team is the one
    acting/in-possession (`is_teammate` in the parsed frame) -- never from
    the opposing team's events, even if those happen to be the nearest
    frames in time. Mixing perspectives would silently compare two
    different teams' tactical situations across the "before" and "after"
    snapshots, making any delta meaningless.

    For each substitution:
      1. `minute - 2 < 0` is skipped outright (no valid pre-window exists).
      2. The pre-frame is searched via `_find_qualifying_frame_for_minute`
         (Milestone 18, extended -- see that function's docstring) with
         `period=sub_period`, `team_id=sub_team_id`, starting at
         `minute - 2`, and capped with `max_minute=minute` so the search
         can never cross past the substitution itself and accidentally
         return a post-substitution event as if it were "pre."
      3. The post-frame is searched the same way starting at `minute + 2`,
         with no upper bound.
      4. If either search comes back empty, the substitution is skipped
         and the reason is printed (not silently dropped).
      5. Before computing any delta, `pre_event["team"]["id"]` and
         `post_event["team"]["id"]` are both asserted to equal the
         substituting team's id -- the explicit same-perspective spot
         check this milestone requires (same spirit as Milestone 12's
         same-reference-frame check). The result is a genuinely
         verified boolean, not an assumed one.
      6. Overlapping-window substitutions (same period, overlapping
         [minute-2, minute+2] ranges) are flagged via `overlapping_with`
         on BOTH results -- their deltas should be read with extra
         caution, since more than one change happened in the same window.

    Returns a list of result dicts, one per NON-skipped substitution.
    """
    events = fetch_match_events(match_id)
    substitutions = find_substitutions(events)

    windows = [
        (sub["period"], sub["minute"] - WINDOW_HALF_WIDTH_MINUTES, sub["minute"] + WINDOW_HALF_WIDTH_MINUTES)
        for sub in substitutions
    ]
    overlapping_with = [[] for _ in substitutions]
    for i in range(len(substitutions)):
        period_i, start_i, end_i = windows[i]
        for j in range(len(substitutions)):
            if i == j:
                continue
            period_j, start_j, end_j = windows[j]
            if _windows_overlap(period_i, start_i, end_i, period_j, start_j, end_j):
                overlapping_with[i].append(substitutions[j]["sub_id"])

    results = []
    for sub in substitutions:
        sub_id = sub["sub_id"]
        minute = sub["minute"]
        period = sub["period"]
        team_id = sub["team_id"]
        team_name = sub["team"]

        if minute - WINDOW_HALF_WIDTH_MINUTES < 0:
            logger.info(
                f"Skipping sub #{sub_id} ({team_name}, minute {minute}): "
                f"minute-{WINDOW_HALF_WIDTH_MINUTES} < 0, no valid pre-window."
            )
            continue

        pre_result = _find_qualifying_frame_for_minute(
            match_id,
            minute - WINDOW_HALF_WIDTH_MINUTES,
            period=period,
            team_id=team_id,
            max_minute=minute,
        )
        if pre_result is None:
            logger.info(
                f"Skipping sub #{sub_id} ({team_name}, minute {minute}): "
                f"no pre-frame found for {team_name} within the search window."
            )
            continue

        post_result = _find_qualifying_frame_for_minute(
            match_id,
            minute + WINDOW_HALF_WIDTH_MINUTES,
            period=period,
            team_id=team_id,
        )
        if post_result is None:
            logger.info(
                f"Skipping sub #{sub_id} ({team_name}, minute {minute}): "
                f"no post-frame found for {team_name} within the search window."
            )
            continue

        pre_event, pre_frame_data = pre_result
        post_event, post_frame_data = post_result

        # CRITICAL: explicit same-perspective spot-check before computing
        # any delta -- both frames must genuinely be the substituting
        # team's own possession events, not merely assumed to be because
        # the search was filtered on team_id (defense in depth against a
        # future refactor of the search helper silently dropping the
        # filter).
        perspective_verified = (
            pre_event["team"]["id"] == team_id and post_event["team"]["id"] == team_id
        )
        assert perspective_verified, (
            f"perspective mismatch for sub #{sub_id}: pre team="
            f"{pre_event['team']['id']}, post team={post_event['team']['id']}, expected {team_id}"
        )

        pre_features = extract_features(parse_360_frame(pre_event, pre_frame_data))
        post_features = extract_features(parse_360_frame(post_event, post_frame_data))

        threat_pre = predict_cumulative_incidence(
            model, pre_features, normalization_mean, normalization_std, time_bin=TIME_BIN
        )
        threat_post = predict_cumulative_incidence(
            model, post_features, normalization_mean, normalization_std, time_bin=TIME_BIN
        )
        actual_delta = threat_post - threat_pre

        if actual_delta > CLASSIFICATION_THRESHOLD:
            classification = "threat increased after sub"
        elif actual_delta < -CLASSIFICATION_THRESHOLD:
            classification = "threat decreased after sub"
        else:
            classification = "no significant change"

        results.append(
            {
                "sub_id": sub_id,
                "team": team_name,
                "minute": minute,
                "period": period,
                "player_off": sub["player_off"],
                "player_on": sub["player_on"],
                "threat_pre": threat_pre,
                "threat_post": threat_post,
                "actual_delta": actual_delta,
                "classification": classification,
                "perspective_verified": perspective_verified,
                "overlapping_with": overlapping_with[sub_id],
            }
        )

    return results
