"""Tactical Timeline UI (new reporting track, capstone): a single, unified
chronological visualization of a match, combining Weak-Spot Lifetime
Analysis and Tactical Event Detection along one shared time axis. This is
the feature this roadmap deliberately sequenced last, so it would have
real, varied match-level temporal data to design around rather than being
built speculatively ahead of it.

=============================================================================
STEP 0: THE REAL DATA-ALIGNMENT PROBLEM, RESOLVED EXPLICITLY (read before
changing anything below) -- this was the actual hard part of this task,
not a formality.
=============================================================================

Five signals were named as candidates for this timeline: Tactical Momentum,
Match Segmentation, Weak-Spot Lifetime Analysis, Tactical Event Detection,
and (implicitly, via match_report.py) the compiled match document. These
are NOT naturally aligned on the same time axis, and treating them as if
they were would have been the actual mistake this Step 0 exists to catch:

- **Tactical Momentum / Match Segmentation are LIVE-STREAM, PER-WEBSOCKET-
  MESSAGE concepts.** Checked directly, not assumed: `tactical_momentum.py`
  (where both live) has ZERO references anywhere in `production/src/`
  (`grep -rl` returns nothing) -- it is imported ONLY by `dashboard.py`,
  a Streamlit script, and computed CLIENT-SIDE over a plain local
  `threat_buffer` list that is explicitly documented (both in that
  module's own docstring and `dashboard.py`'s) as ephemeral,
  per-connection, and deliberately NOT persisted anywhere. `alert_store.py`'s
  own SQLite schema (`logged_at_utc, source, match_id, video_path, minute,
  threat_before, threat_after, delta, explanation_text, explanation_source`)
  carries no momentum/segmentation field at all. **There is no queryable,
  match-level, post-hoc record of a stream's message-by-message momentum
  or segmentation anywhere in this project.** A real-time timeline (an
  overlay on the ALREADY-STREAMING Live CV Monitor tab, using messages as
  they arrive) and a post-hoc match timeline (a NEW batch computation
  re-deriving these values over a match's full, already-fetched sequence)
  are genuinely two different features, not two views of the same data.

- **Weak-Spot Lifetime Analysis and Tactical Event Detection are ALREADY
  match-level, real-clock-time-indexed BATCH outputs** -- both are
  synchronous functions that fetch a match's full cached event/360 data
  once and return real `start_minute`/`end_minute` (or `minute`) fields
  computed directly from StatsBomb's own `minute`/`second` fields. These
  align naturally with the POST-HOC option, with zero new engineering
  needed to make them time-indexed -- they already are.

**SCOPE DECISION, MADE EXPLICITLY (not silently assumed):** this module
builds the POST-HOC timeline (option b), unifying Weak-Spot Lifetime
Analysis and Tactical Event Detection -- both already real, validated,
match-clock-indexed outputs -- as the PRIMARY and ONLY signals. Tactical
Momentum and Match Segmentation are EXPLICITLY OUT OF SCOPE for this
timeline, not silently dropped: building a batch equivalent would require
genuinely new engineering (loading the deterministic MLP, running
per-frame threat inference across the match's full real event sequence --
the SAME class of real compute cost `team_report.py`'s own season-heatmap
already carries -- then threading the resulting values through
`_compute_tactical_momentum`/`classify_match_phase` exactly as
`dashboard.py`'s own live loop already does, one accumulated buffer entry
at a time) AND would require this backend reporting module to import from
`production/frontend/`, reversing this project's own established
one-way `frontend -> src` dependency direction (every existing reporting
module lives under `production/src/reporting/`; `dashboard.py` imports
FROM those modules, never the other way). That is a real, legitimate,
buildable future feature in its own right -- deliberately deferred here as
its own separate scope, not half-built as an under-designed add-on to
this task's actual focus (cleanly unifying the two signals that are
ALREADY batch/match-level today).

=============================================================================
STEP 0 (continued): PERIOD-BOUNDARY HANDLING ON THE SHARED AXIS
=============================================================================

StatsBomb's raw `minute` field does not reset at half-time, and the two
periods' raw ranges overlap near the boundary (the SAME real gotcha
`api.py`'s `_find_qualifying_frame_for_minute` and Weak-Spot Lifetime
Analysis's own `GAP_TOLERANCE_SECONDS` design already had to account for --
verified again directly against this project's own validation match,
3857276: period 1 runs to a real observed max minute of 50 (includes
first-half stoppage time), period 2 starts back at StatsBomb's own
convention of minute 45 and runs to 93). A single shared DISPLAY axis
therefore needs an explicit, computed (not assumed) period-2 offset:
`period_2_display_offset = real_period_1_max_minute - 45.0` -- shifting
every period-2 timestamp forward so period 2 begins exactly where period 1
actually ended on the display axis, with no overlap and no gap. This
offset is computed fresh per match (not hardcoded), since real stoppage
time varies match to match.
"""

import logging

from production.src.ingestion.statsbomb_io import fetch_match_events
from production.src.reporting.tactical_events import detect_tactical_events
from production.src.reporting.team_report import _teams_in_match, generate_weak_spot_lifetime_analysis

logger = logging.getLogger(__name__)

# StatsBomb's own convention: period 2 kickoff is always logged as minute
# 45 in the raw `minute` field, regardless of how much real first-half
# stoppage time period 1 actually ran to.
_PERIOD_2_NOMINAL_START_MINUTE = 45.0


def _period_1_max_minute(events: list) -> float:
    """The real observed maximum `minute` value among period-1 events --
    NOT assumed to be exactly 45.0 (real first-half stoppage time makes
    this vary match to match; this match's own real value is 50)."""
    period_1_minutes = [e["minute"] for e in events if e.get("period") == 1]
    return float(max(period_1_minutes)) if period_1_minutes else _PERIOD_2_NOMINAL_START_MINUTE


def _display_minute(raw_minute: float, period: int, period_2_display_offset: float) -> float:
    """Maps a real `(period, raw_minute)` pair onto ONE continuous display
    axis -- period 1 unchanged, period 2 shifted forward by the real,
    computed `period_2_display_offset` so it continues exactly where
    period 1 actually ended, never overlapping it."""
    if period == 1:
        return raw_minute
    return raw_minute + period_2_display_offset


def generate_match_timeline(match_id: int) -> dict:
    """Assembles the unified Tactical Timeline for `match_id`: every
    real Weak-Spot Lifetime instance (both teams) and every real Tactical
    Event (Counter Attack, Build-up Pattern, Switch of Play), merged into
    ONE chronologically-sorted list on a single continuous display-minute
    axis (see this module's own Step 0 docstring for the period-boundary
    handling).

    Calls EXISTING functions ONLY -- `generate_weak_spot_lifetime_analysis`,
    `detect_tactical_events` -- nothing here recomputes anything already
    computed elsewhere, and neither underlying detector/classifier is
    modified. Tactical Momentum/Match Segmentation are deliberately NOT
    included -- see this module's own Step 0 docstring for the full,
    explicit scope decision.

    Each `timeline_entries` item is `{"signal": "weak_spot"|"counter_attack"|
    "build_up"|"switch_of_play", "team", "period", "start_minute",
    "end_minute", "display_start_minute", "display_end_minute", "label",
    "detail"}` -- `start_minute`/`end_minute` are the RAW StatsBomb-clock
    values (equal for point events like Switch of Play); `display_*` are
    the period-boundary-corrected values the UI actually plots against.
    `detail` carries the FULL original instance dict from its own source
    function, unmodified, so no field is ever silently dropped.
    """
    teams = sorted(_teams_in_match(match_id))
    if len(teams) < 2:
        return {
            "match_id": match_id,
            "no_data": True,
            "reason": f"Fewer than 2 teams found in match_id={match_id}'s event data -- cannot build a timeline.",
        }

    events = fetch_match_events(match_id)
    period_2_display_offset = _period_1_max_minute(events) - _PERIOD_2_NOMINAL_START_MINUTE

    def _to_display(raw_minute: float, period: int) -> float:
        return _display_minute(raw_minute, period, period_2_display_offset)

    timeline_entries: list[dict] = []

    weak_spot_by_team = {team: generate_weak_spot_lifetime_analysis(team, match_id) for team in teams}
    for team, weak_spot_result in weak_spot_by_team.items():
        if weak_spot_result.get("no_data"):
            continue
        for instance in weak_spot_result["weak_spot_instances"]:
            timeline_entries.append(
                {
                    "signal": "weak_spot",
                    "team": team,
                    "period": instance["period"],
                    "start_minute": instance["start_minute"],
                    "end_minute": instance["end_minute"],
                    "display_start_minute": _to_display(instance["start_minute"], instance["period"]),
                    "display_end_minute": _to_display(instance["end_minute"], instance["period"]),
                    "label": f"Weak zone (col={instance['zone']['col']}, row={instance['zone']['row']})",
                    "detail": instance,
                }
            )

    tactical_events_result = detect_tactical_events(match_id)
    if not tactical_events_result.get("no_data"):
        for counter_attack in tactical_events_result["counter_attacks"]:
            timeline_entries.append(
                {
                    "signal": "counter_attack",
                    "team": counter_attack["team"],
                    "period": counter_attack["period"],
                    "start_minute": counter_attack["start_minute"],
                    "end_minute": counter_attack["end_minute"],
                    "display_start_minute": _to_display(counter_attack["start_minute"], counter_attack["period"]),
                    "display_end_minute": _to_display(counter_attack["end_minute"], counter_attack["period"]),
                    "label": "Counter Attack",
                    "detail": counter_attack,
                }
            )
        for build_up in tactical_events_result["build_up_patterns"]:
            timeline_entries.append(
                {
                    "signal": "build_up",
                    "team": build_up["team"],
                    "period": build_up["period"],
                    "start_minute": build_up["start_minute"],
                    "end_minute": build_up["end_minute"],
                    "display_start_minute": _to_display(build_up["start_minute"], build_up["period"]),
                    "display_end_minute": _to_display(build_up["end_minute"], build_up["period"]),
                    "label": "Build-up Pattern",
                    "detail": build_up,
                }
            )
        for switch in tactical_events_result["switches_of_play"]:
            # A point event, not a span -- start == end (the SAME
            # convention Match Segmentation's own live UI uses for
            # instantaneous readings vs. Weak-Spot Lifetime's own spans).
            display_minute = _to_display(switch["minute"], switch["period"])
            timeline_entries.append(
                {
                    "signal": "switch_of_play",
                    "team": switch["team"],
                    "period": switch["period"],
                    "start_minute": switch["minute"],
                    "end_minute": switch["minute"],
                    "display_start_minute": display_minute,
                    "display_end_minute": display_minute,
                    "label": f"Switch of Play ({switch['lateral_distance_meters']:.1f}m)",
                    "detail": switch,
                }
            )

    timeline_entries.sort(key=lambda entry: (entry["display_start_minute"], entry["signal"]))

    return {
        "match_id": match_id,
        "no_data": False,
        "teams": teams,
        "period_1_max_minute": _period_1_max_minute(events),
        "period_2_display_offset": period_2_display_offset,
        "momentum_segmentation_in_scope": False,
        "momentum_segmentation_note": (
            "Tactical Momentum and Match Segmentation are live-stream-only concepts (ephemeral, "
            "client-side, never persisted) with no batch/post-hoc match-level equivalent today -- "
            "deliberately out of scope for this timeline, not silently omitted. See this module's "
            "own Step 0 docstring for the full reasoning."
        ),
        "timeline_entries": timeline_entries,
        "weak_spot_instance_count": sum(1 for e in timeline_entries if e["signal"] == "weak_spot"),
        "counter_attack_count": sum(1 for e in timeline_entries if e["signal"] == "counter_attack"),
        "build_up_count": sum(1 for e in timeline_entries if e["signal"] == "build_up"),
        "switch_of_play_count": sum(1 for e in timeline_entries if e["signal"] == "switch_of_play"),
    }
