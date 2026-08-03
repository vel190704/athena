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

from collections import Counter

from production.src.ingestion.statsbomb_io import fetch_match_events
from production.src.pipeline.habit_memory import (
    MIN_HISTORICAL_EVENTS,
    generate_player_heatmap,
)


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
            print(f"[player_report] match_id={match_id}: no events data available, skipping.")
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
