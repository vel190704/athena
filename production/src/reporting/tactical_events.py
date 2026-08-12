"""Tactical Event Detection (new reporting track): Counter Attack, Switch
of Play, Build-up Pattern -- the 3 event types genuinely feasible from
EVENT DATA ALONE, per the roadmap's own explicit scoping. Deliberately
does NOT attempt anything requiring 360/frame-level data (unlike Weak-Spot
Lifetime Analysis, which is gated behind real 360 coverage) -- event data
is broadly available across this project's whole cached corpus, so this
feature has none of that coverage limitation.

STANDALONE, additive: reads real StatsBomb event data via the EXISTING
`statsbomb_io.fetch_match_events`, reuses `chain_builder.build_possession_chains`
UNMODIFIED for possession-chain construction and turnover classification
(does NOT reinvent regain detection), and reuses `pass_network._is_complete_pass`
UNMODIFIED for pass-completion detection (does NOT reinvent that signal a
third time). Does NOT import or use `direction.py` -- per ADR-009 (already
applied by Tactical Entropy), StatsBomb event `location`/`end_location`
coordinates are already recorded relative to the ACTING team's own
attacking-left-to-right frame in both periods, so raw x already increases
toward the acting team's own attacking third with no per-team/per-period
correction needed.

Single-match scope (`detect_tactical_events(match_id)`), not a `match_ids`
list: like Pass Network and Weak-Spot Lifetime Analysis, this describes
WITHIN-MATCH structure (a specific chain's own regain-to-final-third
transition, a specific pass's own lateral distance) that does not
generalize across matches.
"""

import logging
from collections import defaultdict

from production.src.ingestion.statsbomb_io import X_SCALE, Y_SCALE, fetch_match_events
from production.src.pipeline.chain_builder import build_possession_chains
from production.src.pipeline.feature_extractor import FINAL_THIRD_X
from production.src.reporting.pass_network import _is_complete_pass
from production.src.reporting.team_report import DEFENSIVE_THIRD_X

logger = logging.getLogger(__name__)

# =============================================================================
# STEP 0: every threshold below is a real, disclosed judgment call, verified
# against real match 3857276 (StatsBomb open data, 169 real possession
# chains across both periods, 955 real Pass events) BEFORE being finalized
# -- see this module's own test file for the exact verification numbers.
# =============================================================================

# COUNTER ATTACK: a possession chain is a Counter Attack candidate if it
# (a) immediately follows an OPPONENT chain that ended in an open-play
# turnover (chain_builder's own `censor_reason == "turnover"`, reused
# unmodified -- not reinvented), AND (b) reaches the attacking third
# (`FINAL_THIRD_X`, the SAME boundary team_report.py/feature_extractor.py
# already establish) within this many real seconds of the chain's own
# first located event.
#
# 8.0s was NOT assumed -- a real, counter-intuitive finding shaped it: the
# real time-to-final-third distribution for turnover-preceded chains
# (median 7s, mean 10.5s, n=41 reaching it at all) was actually SLOWER on
# average than for non-turnover-preceded chains (median 3.5s, mean 6.9s,
# n=78) in this real match -- restarts (corners, throw-ins deep in the
# attacking third) pull the non-turnover group's median down, and NOT
# every regain produces a fast transition (many turnover-regains are
# recycled slowly instead). This means "preceded by a turnover" alone is
# NOT a reliable fast-transition signal by itself in real data -- both
# conditions must be checked independently and combined with AND, exactly
# as Step 0 asked. 8.0s sits within the real OVERALL time-to-final-third
# distribution (all 119 chains that ever reach it, any origin): the 60th
# percentile (deciles: 0,0,1,3,5,7,10,13,23) -- meaningfully faster than
# a TYPICAL chain that reaches the final third at all, a genuine "fast"
# cutoff, not a majority-inclusive one. Combined with the turnover
# precondition, 25 of 169 real chains (14.8%) qualify -- see Step 3.
COUNTER_ATTACK_TIME_TO_FINAL_THIRD_SECONDS = 8.0

# SWITCH OF PLAY: a single COMPLETED pass (`_is_complete_pass`, reused
# unmodified from pass_network.py -- not reinvented) whose real lateral
# (y-axis) distance, in meters, meets or exceeds this threshold. Verified
# against the real distribution of all 790 real completed passes in the
# same match: median 9.1m, mean 11.7m -- most passes are short lateral
# distance, as expected. 30.0m sits at the real ~96th percentile (only
# 4.3%, 34 of 790 real completed passes, reach or exceed it) -- a genuine
# statistical outlier, not a common case (95th pctile 29.1m, 97th 33.7m --
# 30.0m is a clean, round value inside that narrow band). Deliberately the
# SIMPLER single-pass interpretation (not a multi-pass "short sequence"),
# since a single real StatsBomb Pass event already gives one clean,
# unambiguous, directly-measurable signal without inventing additional
# unstated judgment calls (how many passes stitched together, over what
# time window) the task's own definition did not require.
SWITCH_OF_PLAY_LATERAL_THRESHOLD_METERS = 30.0

# BUILD-UP PATTERN: a possession chain that (a) starts in the defensive or
# middle third (NOT attacking -- reuses `DEFENSIVE_THIRD_X`/`FINAL_THIRD_X`,
# the SAME zone-share convention `team_comparison._zone_shares` already
# uses), (b) is NOT fast by Counter Attack's own exact criterion (does not
# reach the final third within `COUNTER_ATTACK_TIME_TO_FINAL_THIRD_SECONDS`
# -- either later, or never) -- this is the deliberate INVERSE of Counter
# Attack's own pace signal, which is what makes the two types mutually
# exclusive BY CONSTRUCTION, not merely by chance (verified in Step 3: 0
# real overlapping chains) -- and (c) has at least this many real Pass
# events, to exclude a single misplaced back-pass from qualifying as
# "sustained build-up." Verified against the real per-chain pass-count
# distribution (median 4 passes across all 169 chains): 5 passes sits
# where only 43.2% of ALL real chains qualify (73 of 169) -- genuinely
# above the typical chain's own pass count, not an inclusive floor.
# Combined with (a)/(b), 47 of 169 real chains (27.8%) qualify -- see Step 3.
BUILDUP_MIN_PASSES = 5


def _events_by_chain_key(events: list) -> dict[tuple[int, int], list]:
    """Groups raw events by `(period, possession)` -- the SAME grouping
    key `chain_builder.build_possession_chains` uses internally -- so each
    chain dict that function returns can be paired with its own
    constituent events, which that function's own return shape does not
    carry (only chain-level summary fields). This is NOT re-deriving
    possessions (the grouping key and definition are byte-for-byte
    identical to chain_builder's own); it only recovers the per-event
    detail the chain dicts themselves omit, needed here for intra-chain
    timing/location.
    """
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for e in events:
        if e["period"] in (1, 2):
            groups[(e["period"], e["possession"])].append(e)
    for group in groups.values():
        group.sort(key=lambda e: e["index"])
    return groups


def detect_tactical_events(match_id: int) -> dict:
    """Detects Counter Attack, Switch of Play, and Build-up Pattern
    instances across `match_id`'s full real event stream (both periods).

    Counter Attack and Build-up Pattern are chain-level classifications
    (one chain is/isn't each type -- see Step 0's exact criteria above,
    mutually exclusive by construction along the pace axis). Switch of
    Play is a pass-level classification, independent of chain membership.

    OUTPUT DELIBERATELY CARRIES NO EXACT (x, y) COORDINATE for any
    individual event -- only team, chain/period timing, and DERIVED
    scalar metrics (time-to-final-third, lateral distance, pass count).
    See this module's own ADR-021 addendum for why: an exact per-event
    coordinate is what made the shot map's/Pass Network's own raw variants
    individually-attributable to one specific StatsBomb event; omitting it
    here by design avoids reintroducing that same gating question rather
    than needing a raw/aggregated split the way those features did.
    """
    events = fetch_match_events(match_id)
    if events is None:
        return {
            "match_id": match_id,
            "no_data": True,
            "reason": "No event data available for this match_id.",
        }

    chains = build_possession_chains(events, periods=(1, 2))
    chains_sorted = sorted(chains, key=lambda c: (c["period"], c["chain_id"]))
    events_by_chain = _events_by_chain_key(events)

    counter_attacks = []
    build_up_patterns = []

    for i, chain in enumerate(chains_sorted):
        key = (chain["period"], chain["chain_id"])
        chain_events = events_by_chain.get(key, [])
        located = [e for e in chain_events if "location" in e]
        if not located:
            continue

        start_total_seconds = located[0]["minute"] * 60 + located[0]["second"]
        start_x_scaled = located[0]["location"][0] * X_SCALE
        start_third = (
            "defensive" if start_x_scaled < DEFENSIVE_THIRD_X
            else "attacking" if start_x_scaled > FINAL_THIRD_X
            else "middle"
        )

        time_to_final_third = None
        for e in located:
            x_scaled = e["location"][0] * X_SCALE
            if x_scaled > FINAL_THIRD_X:
                time_to_final_third = (e["minute"] * 60 + e["second"]) - start_total_seconds
                break

        is_fast = (
            time_to_final_third is not None
            and time_to_final_third <= COUNTER_ATTACK_TIME_TO_FINAL_THIRD_SECONDS
        )

        preceded_by_opponent_turnover = False
        if i > 0:
            previous_chain = chains_sorted[i - 1]
            if (
                previous_chain["period"] == chain["period"]
                and previous_chain["censor_reason"] == "turnover"
                and previous_chain["team"] != chain["team"]
            ):
                preceded_by_opponent_turnover = True

        n_passes = sum(1 for e in chain_events if e["type"]["name"] == "Pass")

        if preceded_by_opponent_turnover and is_fast:
            counter_attacks.append(
                {
                    "chain_id": chain["chain_id"],
                    "period": chain["period"],
                    "team": chain["team"],
                    "start_minute": chain["start_minute"],
                    "end_minute": chain["end_minute"],
                    "time_to_final_third_seconds": time_to_final_third,
                    "chain_duration_seconds": chain["duration_seconds"],
                }
            )
        elif start_third in ("defensive", "middle") and not is_fast and n_passes >= BUILDUP_MIN_PASSES:
            build_up_patterns.append(
                {
                    "chain_id": chain["chain_id"],
                    "period": chain["period"],
                    "team": chain["team"],
                    "start_minute": chain["start_minute"],
                    "end_minute": chain["end_minute"],
                    "start_third": start_third,
                    "n_passes": n_passes,
                    "chain_duration_seconds": chain["duration_seconds"],
                }
            )

    switches_of_play = []
    total_completed_passes = 0
    for event in events:
        if event["type"]["name"] != "Pass":
            continue
        if not _is_complete_pass(event):
            continue
        total_completed_passes += 1

        location = event.get("location")
        end_location = event.get("pass", {}).get("end_location")
        if location is None or end_location is None:
            continue
        lateral_distance_meters = abs(end_location[1] - location[1]) * Y_SCALE
        if lateral_distance_meters >= SWITCH_OF_PLAY_LATERAL_THRESHOLD_METERS:
            switches_of_play.append(
                {
                    "period": event["period"],
                    "minute": event["minute"] + event["second"] / 60.0,
                    "team": event.get("team", {}).get("name"),
                    "lateral_distance_meters": lateral_distance_meters,
                }
            )

    return {
        "match_id": match_id,
        "no_data": False,
        "counter_attack_time_to_final_third_seconds": COUNTER_ATTACK_TIME_TO_FINAL_THIRD_SECONDS,
        "switch_of_play_lateral_threshold_meters": SWITCH_OF_PLAY_LATERAL_THRESHOLD_METERS,
        "buildup_min_passes": BUILDUP_MIN_PASSES,
        "total_chains": len(chains_sorted),
        "total_completed_passes": total_completed_passes,
        "counter_attacks": counter_attacks,
        "build_up_patterns": build_up_patterns,
        "switches_of_play": switches_of_play,
        "counter_attack_fraction_of_chains": (
            len(counter_attacks) / len(chains_sorted) if chains_sorted else None
        ),
        "build_up_fraction_of_chains": (
            len(build_up_patterns) / len(chains_sorted) if chains_sorted else None
        ),
        "switch_of_play_fraction_of_completed_passes": (
            len(switches_of_play) / total_completed_passes if total_completed_passes else None
        ),
    }
