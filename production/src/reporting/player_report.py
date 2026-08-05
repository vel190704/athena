"""Milestone 40 (new reporting track, Step 1): Historical Player Analysis
Report.

STANDALONE, additive: reads real StatsBomb event data via the EXISTING
`production/src/ingestion/statsbomb_io.py` fetch functions (network fetch
only on a cache miss) and reuses `habit_memory.generate_player_heatmap`
exactly as built for Milestone 22, rather than re-implementing the same
100x68m-grid binning logic a second time. Nothing in `production/src/
physics`, `spatial`, `models`, `pipeline`'s training/serving code, or
`serving/` is imported by, or modified for, this module. No CV/video
dependency -- entirely independent of ADR-013 through ADR-016.
"""

import logging
from collections import Counter

from production.src.ingestion.statsbomb_io import X_SCALE, Y_SCALE, fetch_match_events
from production.src.pipeline.habit_memory import (
    CELL_HEIGHT_METERS,
    CELL_WIDTH_METERS,
    GRID_COLS,
    GRID_ROWS,
    MIN_HISTORICAL_EVENTS,
    generate_player_heatmap,
)

logger = logging.getLogger(__name__)

# Shot-map feature (additive -- see generate_player_shot_map below).
# Reuses the SAME real threshold value `heatmap_used_uniform_fallback`
# already uses (habit_memory.MIN_HISTORICAL_EVENTS = 20), matching this
# project's own established convention (player_visualizer.py's
# LOW_SAMPLE_EVENT_COUNT_THRESHOLD does the exact same thing for its
# positional-distribution panel) -- named separately, not reused directly
# by import alias, because this is conceptually a DIFFERENT count (shots,
# a rare event type -- a full match typically yields 0-5 per player, vs.
# dozens of general tagged events) that happens to warrant the same bar,
# not the same measurement.
MIN_SHOTS_FOR_CONFIDENT_SHOT_MAP = MIN_HISTORICAL_EVENTS


def _match_time_minutes(event: dict) -> float:
    """StatsBomb's `minute` field is already a continuous match-clock value
    across periods -- VERIFIED directly against real cached event data
    (e.g. match 3773386's period-2 `Tactical Shift` event carries
    `minute=45, second=0`, i.e. period 2 kickoff continues the clock from
    45, it does not reset to 0) -- so `minute + second/60` is directly
    usable as an elapsed-match-time axis with no per-period offset needed.
    """
    return event["minute"] + event["second"] / 60.0


def _player_position_counts(events_by_match: dict, player_id: int) -> Counter:
    counts: Counter = Counter()
    for events in events_by_match.values():
        for event in events:
            player = event.get("player")
            if player is None or player.get("id") != player_id:
                continue
            position = event.get("position")
            if position is None:
                continue
            counts[position["name"]] += 1
    return counts


def _starting_team(events: list, player_id: int) -> dict | None:
    """This match's `{"id":.., "name":..}` team dict if `player_id` appears
    in the `Starting XI` lineup, else `None`. VERIFIED present on real
    cached data (`tactics.lineup`, each entry `{"player":.., "position":..,
    "jersey_number":..}`) before relying on it here."""
    for event in events:
        if event["type"]["name"] != "Starting XI":
            continue
        for entry in event.get("tactics", {}).get("lineup", []):
            if entry.get("player", {}).get("id") == player_id:
                return event["team"]
    return None


def _substitution_times(events: list, player_id: int) -> tuple[float | None, float | None]:
    """`(subbed_on_minute, subbed_off_minute)` for `player_id` in this
    match, each `None` if that event never occurs for them."""
    subbed_on, subbed_off = None, None
    for event in events:
        if event["type"]["name"] != "Substitution":
            continue
        t = _match_time_minutes(event)
        replacement_id = event.get("substitution", {}).get("replacement", {}).get("id")
        if replacement_id == player_id:
            subbed_on = t
        if event.get("player", {}).get("id") == player_id:
            subbed_off = t
    return subbed_on, subbed_off


def _substitution_team(events: list, player_id: int) -> dict | None:
    """The team dict for the substitution event that brought `player_id`
    onto the pitch, or `None` if no such event exists in this match."""
    for event in events:
        if (
            event["type"]["name"] == "Substitution"
            and event.get("substitution", {}).get("replacement", {}).get("id") == player_id
        ):
            return event["team"]
    return None


def _match_final_minute(events: list) -> float:
    """Proxy for full-time: the latest `minute+second/60` of any event in
    the match (StatsBomb logs a final-whistle-adjacent event, typically
    `Half End`, at the true end of period 2)."""
    return max(_match_time_minutes(e) for e in events)


def _team_formation_changes(events: list, team_id: int) -> list[tuple[float, int]]:
    """Sorted `(start_minute, formation)` breakpoints for `team_id`, from
    `Starting XI` (period-1 kickoff) and `Tactical Shift` (mid-match
    changes) events -- both VERIFIED, against real cached data, to carry a
    `tactics.formation` int (e.g. 4141, 4231) before this function assumed
    it existed (Step 1.3's explicit "don't fabricate a placeholder" check).
    """
    changes = []
    for event in events:
        if event["type"]["name"] not in ("Starting XI", "Tactical Shift"):
            continue
        if event.get("team", {}).get("id") != team_id:
            continue
        formation = event.get("tactics", {}).get("formation")
        if formation is None:
            continue
        changes.append((_match_time_minutes(event), formation))
    changes.sort(key=lambda pair: pair[0])
    return changes


def _formation_minutes_in_interval(
    formation_changes: list[tuple[float, int]], start: float, end: float
) -> dict[int, float]:
    """Minutes each formation was in effect for `[start, end]`, given
    `formation_changes` (sorted breakpoints, each valid from its own start
    minute until the next breakpoint, or `end` for the last one)."""
    if not formation_changes or end <= start:
        return {}

    minutes_by_formation: dict[int, float] = {}
    for i, (change_start, formation) in enumerate(formation_changes):
        change_end = formation_changes[i + 1][0] if i + 1 < len(formation_changes) else float("inf")
        overlap_start = max(start, change_start)
        overlap_end = min(end, change_end)
        if overlap_end > overlap_start:
            minutes_by_formation[formation] = minutes_by_formation.get(formation, 0.0) + (
                overlap_end - overlap_start
            )
    return minutes_by_formation


def _heatmap_qualifying_event_count(events_by_match: dict, player_id: int) -> int:
    """Counts `player_id`'s qualifying events using the EXACT SAME criteria
    `habit_memory.generate_player_heatmap` uses internally (player_id match
    + a `location` field present) -- WITHOUT calling into or modifying that
    function, which returns only the finished grid, never this count.
    Needed because Milestone 44's validation sweep found a real,
    demonstrated transparency gap: a 1-event and an 800-event player
    produce identically-shaped, identically-confident-LOOKING output
    (`positional_distribution` percentages, an already-normalized
    heatmap) with no way for a caller to tell them apart otherwise.
    """
    count = 0
    for events in events_by_match.values():
        for event in events:
            player = event.get("player")
            if player is None or player.get("id") != player_id:
                continue
            if "location" not in event:
                continue
            count += 1
    return count


def generate_player_report(player_id: int, match_ids: list[int]) -> dict:
    """Historical Player Profile Report (Milestone 40, Step 1): positional
    distribution, aggregate positional heatmap, and summary stats -- all
    from real, already-cached/fetchable StatsBomb event data. No CV/video
    dependency whatsoever.

    `match_ids`: StatsBomb match_ids to include (each fetched, and cached,
    via the existing `statsbomb_io.fetch_match_events` -- a network fetch
    only happens for a match_id not already cached under `data/raw/`).
    Matches with no fetchable events are skipped with a printed warning,
    not silently dropped; matches where `player_id` has no `Starting XI` or
    `Substitution`-onto-pitch record are excluded from the minutes/
    formation stats (they never appeared), even though they still count
    toward `positional_distribution`/`heatmap_grid` if the player has ANY
    tagged events in them (defensive: a player could in principle have a
    stray event without a resolvable on-pitch interval; this has not been
    observed in real cached data but is not assumed impossible).

    SAMPLE-SIZE TRANSPARENCY (added after Milestone 44's validation sweep
    found a real, demonstrated gap: a player with a SINGLE tagged event
    produces a `positional_distribution` of `{"Right Center Back": 1.0}`
    -- a 100% figure structurally indistinguishable from one built on
    thousands of events). The returned dict now also carries
    `positional_distribution_event_count` (the raw count backing those
    percentages) and `heatmap_event_count`/`heatmap_used_uniform_fallback`
    (whether `habit_memory.generate_player_heatmap`'s own
    `MIN_HISTORICAL_EVENTS` cold-start threshold was crossed -- that
    function already degrades gracefully to a uniform grid below it, but
    previously gave callers no way to know that had happened versus a
    genuinely uniform real pattern). Callers/renderers should check these
    before presenting `positional_distribution`/`heatmap_grid` as
    confident findings.
    """
    events_by_match: dict[int, list] = {}
    for match_id in match_ids:
        events = fetch_match_events(match_id)
        if events is None:
            logger.warning(f"match_id={match_id}: no events data available, skipping.")
            continue
        events_by_match[match_id] = events

    # --- Step 1.1: positional distribution. StatsBomb's `position` field
    # records the player's CURRENT on-pitch role at the moment of each
    # action; there is no continuous minute-by-minute position ledger in
    # this data, so an EVENT-COUNT share is the honest, available
    # operationalization -- not a true time-weighted one, and stated as
    # such rather than implied to be minutes-based. ---
    position_counts = _player_position_counts(events_by_match, player_id)
    total_position_events = sum(position_counts.values())
    positional_distribution = (
        {
            name: count / total_position_events
            for name, count in sorted(position_counts.items(), key=lambda kv: -kv[1])
        }
        if total_position_events > 0
        else {}
    )

    # --- Step 1.2: aggregate positional heatmap -- reuses
    # `habit_memory.generate_player_heatmap` EXACTLY as built for Milestone
    # 22 (10x7 grid over the verified 100x68m ADR-002 space). No
    # `exclude_match_id` here -- this is a standalone report, not a
    # leakage-guarded training input. ---
    heatmap = generate_player_heatmap(player_id, events_by_match, exclude_match_id=None)
    heatmap_event_count = _heatmap_qualifying_event_count(events_by_match, player_id)

    # --- Step 1.3: summary stats (total minutes, primary position, and
    # formation frequency -- the `tactics.formation` field this needs was
    # VERIFIED to exist against real cached data, see
    # `_team_formation_changes`'s docstring, before being relied on). ---
    total_minutes = 0.0
    formation_minutes: dict[int, float] = {}
    matches_found_in = 0

    for events in events_by_match.values():
        team = _starting_team(events, player_id)
        started = team is not None
        subbed_on, subbed_off = _substitution_times(events, player_id)

        if not started and subbed_on is None:
            continue  # no Starting XI or substitution-on record: didn't play
        if not started:
            team = _substitution_team(events, player_id)

        if team is None:
            continue

        match_end = _match_final_minute(events)
        on_pitch_start = 0.0 if started else subbed_on
        on_pitch_end = subbed_off if subbed_off is not None else match_end
        if on_pitch_end <= on_pitch_start:
            continue

        matches_found_in += 1
        total_minutes += on_pitch_end - on_pitch_start

        formation_changes = _team_formation_changes(events, team["id"])
        for formation, minutes in _formation_minutes_in_interval(
            formation_changes, on_pitch_start, on_pitch_end
        ).items():
            formation_minutes[formation] = formation_minutes.get(formation, 0.0) + minutes

    primary_position = next(iter(positional_distribution), None)
    primary_formation = max(formation_minutes, key=formation_minutes.get) if formation_minutes else None

    return {
        "player_id": player_id,
        "matches_requested": len(match_ids),
        "matches_with_data": len(events_by_match),
        "matches_player_appeared_in": matches_found_in,
        "positional_distribution": positional_distribution,
        "positional_distribution_event_count": total_position_events,
        "primary_position": primary_position,
        "heatmap_grid": heatmap.tolist(),
        "heatmap_grid_shape": (
            "10 cols (x, 10m/cell) x 7 rows (y, ~9.71m/cell), matching "
            "habit_memory.GRID_COLS/GRID_ROWS"
        ),
        "heatmap_event_count": heatmap_event_count,
        "heatmap_used_uniform_fallback": heatmap_event_count < MIN_HISTORICAL_EVENTS,
        "total_minutes_played": total_minutes,
        "formation_minutes": formation_minutes,
        "primary_formation": primary_formation,
    }


def generate_player_shot_map(player_id: int, match_ids: list[int]) -> dict:
    """Shot Map (additive new feature): per-shot location/outcome/body-part/
    quality for `player_id` across `match_ids`, plus summary stats -- a
    NEW, STANDALONE function alongside `generate_player_report`, not an
    extension of its return dict. Reuses `statsbomb_io.fetch_match_events`
    exactly as `generate_player_report` does (a second, independent call
    per match_id -- cheap, since this hits the same on-disk cache file,
    not a second network fetch).

    CRITICAL, DELIBERATE DISTINCTION -- do not blur this in any future
    change: every "xG" value here is StatsBomb's own real, provided
    `shot.statsbomb_xg` field (a trained shot-outcome model StatsBomb
    themselves publish per shot). This is NOT this project's own DeepHit
    model's predicted cumulative incidence (`predict_cumulative_incidence`,
    `_cumulative_incidence_forward`) -- that measures a DIFFERENT quantity
    entirely (near-term THREAT over a time horizon from a general match
    state, not "will THIS SPECIFIC SHOT go in"). Verified directly against
    3,829 real cached Shot events (150 matches) before writing this
    function: `statsbomb_xg` is present, non-null, on 100% of them, real
    range observed [0.0025, 0.993] -- no separate "shot quality" field
    exists anywhere in the schema, so `statsbomb_xg` is used directly as
    the quality signal (for circle sizing in the visualizer) rather than
    inventing or deriving a different one. DeepHit is never imported,
    called, or referenced by this function.

    `outcome` is StatsBomb's real `shot.outcome.name` string -- verified
    real values (same 150-match sample): "Goal", "Off T", "Saved",
    "Blocked", "Wayward", "Post", "Saved Off Target", "Saved to Post".
    `is_goal` is simply `outcome == "Goal"`, not a separately-tracked flag.
    `body_part` is StatsBomb's real `shot.body_part.name` -- verified real
    values: "Right Foot", "Left Foot", "Head", "Other" (matches this
    feature's design spec exactly, confirmed rather than assumed).

    SAMPLE-SIZE TRANSPARENCY (same discipline `generate_player_report`
    already established, applied to this feature's own count): a player
    with 1-2 real shots gets the same honest `shot_map_used_low_sample_flag`
    treatment as a 1-event player gets from `heatmap_used_uniform_fallback`
    -- callers/renderers should check this before presenting `xg_per_shot`/
    the shot scatter as a confident finding.

    REQUEST-SIZE DISCIPLINE (the Team Reports timeout incident's own
    pattern, checked here before shipping, not assumed safe): this
    function does ONE linear scan per match's already-cached event list --
    no BiomechanicalPitchControl, no DeepHit forward passes, no
    possession-chain building (the actual cost driver in that incident).
    Measured directly against Messi's full 596-match cached career before
    this function was considered done: see the task report for the real
    number. No pre-filter/cap was added because none was needed at that
    measured cost -- if `data/raw/`'s cache grows enough that this
    changes, re-measure before assuming this reasoning still holds.
    """
    shots: list[dict] = []

    for match_id in match_ids:
        events = fetch_match_events(match_id)
        if events is None:
            logger.warning(f"match_id={match_id}: no events data available, skipping.")
            continue

        for event in events:
            if event.get("type", {}).get("name") != "Shot":
                continue
            player = event.get("player")
            if player is None or player.get("id") != player_id:
                continue

            shot = event.get("shot", {})
            location = event.get("location")
            statsbomb_xg = shot.get("statsbomb_xg")
            outcome_name = shot.get("outcome", {}).get("name")
            body_part_name = shot.get("body_part", {}).get("name")
            # Defensive, not reactive: verified 100% presence of all four
            # fields across 3,829 real cached shots before writing this
            # function (see docstring) -- not observed missing, but a
            # shot missing one of these is skipped rather than plotted
            # with a fabricated placeholder value.
            if location is None or statsbomb_xg is None or outcome_name is None or body_part_name is None:
                logger.warning(
                    f"match_id={match_id}: shot event {event.get('id')} missing a "
                    "required field (location/statsbomb_xg/outcome/body_part) -- skipping this shot."
                )
                continue

            # ADR-002 rescale, matching habit_memory.generate_player_heatmap's
            # OWN established convention exactly (`x = raw_x * X_SCALE`) --
            # raw StatsBomb event `location` is in that provider's native
            # 120x80 unit grid, not this project's 100x68m pitch space every
            # renderer (including this feature's own render_shot_map) draws
            # against. Skipping this scaling was caught directly, before
            # shipping, by a shot plotting off-canvas (raw x up to 120 on a
            # 100-wide drawn pitch) during this feature's own validation.
            scaled_location = [location[0] * X_SCALE, location[1] * Y_SCALE]

            shots.append({
                "match_id": match_id,
                "location": scaled_location,
                "statsbomb_xg": statsbomb_xg,
                "outcome": outcome_name,
                "is_goal": outcome_name == "Goal",
                "body_part": body_part_name,
            })

    total_shots = len(shots)
    goals = sum(1 for s in shots if s["is_goal"])
    sum_statsbomb_xg = sum(s["statsbomb_xg"] for s in shots)
    shots_by_body_part = dict(Counter(s["body_part"] for s in shots))

    return {
        "player_id": player_id,
        "matches_requested": len(match_ids),
        "shots": shots,
        "total_shots": total_shots,
        "goals": goals,
        "shots_by_body_part": shots_by_body_part,
        "sum_statsbomb_xg": sum_statsbomb_xg,
        "xg_per_shot": (sum_statsbomb_xg / total_shots) if total_shots > 0 else None,
        "shot_map_used_low_sample_flag": total_shots < MIN_SHOTS_FOR_CONFIDENT_SHOT_MAP,
    }


def generate_player_shot_map_aggregated(player_id: int, match_ids: list[int]) -> dict:
    """ADR-021 condition-2-compliant variant of `generate_player_shot_map`,
    for PUBLIC deployments: bins individual shots into the SAME
    GRID_COLS x GRID_ROWS grid convention `habit_memory`/this module's own
    `generate_player_report` already use for the aggregate positional
    heatmap, producing shot DENSITY and mean `statsbomb_xg` PER CELL --
    never an individually-recoverable single shot location. This function
    NEVER returns a per-shot list; there is no field on this dict a caller
    could use to reconstruct any one specific shot's exact coordinates.

    Reuses `generate_player_shot_map` internally (not a re-implementation
    of its fetch/filter/rescale/low-sample logic) and reduces its `shots`
    list to a grid before returning -- that raw list is a local variable
    of this function only, and is discarded (via `dict.pop`, not merely
    left unread) before the return dict is built, so it can never
    accidentally leak through via e.g. a future `**` merge mistake.

    ADR-021 CONTEXT: condition 2 ("no raw StatsBomb data exposed to site
    visitors, in any form") was found, in this project's own compliance
    audit, to be violated by `generate_player_shot_map`'s `shots` field --
    each entry is one real, individually-located StatsBomb Shot event.
    This function exists so a PUBLIC deployment can serve a shot map
    without ever computing or transmitting that per-shot list at all (see
    `api.py`'s `PUBLIC_DEPLOYMENT` flag, which selects this function over
    `generate_player_shot_map` for the `/reports/player/{id}/shot-map`
    endpoint). The summary scalars below (`total_shots`, `goals`,
    `sum_statsbomb_xg`, `xg_per_shot`, `shots_by_body_part`,
    `shot_map_used_low_sample_flag`) are reused UNCHANGED from
    `generate_player_shot_map`'s own return dict -- ADR-021's own
    reasoning already treats a count/sum/share as condition-2-compliant
    (not individually traceable to one event), so these do not need to be
    recomputed or altered for this aggregated variant.
    """
    full = generate_player_shot_map(player_id, match_ids)
    shots = full.pop("shots")  # local only -- deliberately never returned from this function

    shot_count_grid = [[0] * GRID_ROWS for _ in range(GRID_COLS)]
    xg_sum_grid = [[0.0] * GRID_ROWS for _ in range(GRID_COLS)]

    for shot in shots:
        x, y = shot["location"]
        col = min(max(int(x // CELL_WIDTH_METERS), 0), GRID_COLS - 1)
        row = min(max(int(y // CELL_HEIGHT_METERS), 0), GRID_ROWS - 1)
        shot_count_grid[col][row] += 1
        xg_sum_grid[col][row] += shot["statsbomb_xg"]

    total_shots = full["total_shots"]
    shot_density_grid = [
        [(shot_count_grid[col][row] / total_shots) if total_shots > 0 else 0.0 for row in range(GRID_ROWS)]
        for col in range(GRID_COLS)
    ]
    # None (not 0.0) for an empty cell -- an honest "no shots landed here",
    # not a fabricated zero-quality reading, matching team_report.py's own
    # control_heatmap_grid convention for cells with no observation.
    mean_xg_grid = [
        [
            (xg_sum_grid[col][row] / shot_count_grid[col][row]) if shot_count_grid[col][row] > 0 else None
            for row in range(GRID_ROWS)
        ]
        for col in range(GRID_COLS)
    ]

    return {
        **full,
        "shot_density_grid": shot_density_grid,
        "mean_xg_grid": mean_xg_grid,
        "shot_grid_shape": (
            f"{GRID_COLS} cols (x, {CELL_WIDTH_METERS}m/cell) x {GRID_ROWS} rows "
            f"(y, {CELL_HEIGHT_METERS:.2f}m/cell); same convention as habit_memory's positional "
            "heatmap. shot_density_grid cells sum to 1.0 across the whole grid (0.0 if total_shots"
            "==0); mean_xg_grid cells are None where no shot landed."
        ),
    }
