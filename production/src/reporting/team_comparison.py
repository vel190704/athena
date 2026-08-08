"""General-purpose team-season style comparison tool.

Given any two (team, season) pairs, produces a real, data-grounded
side-by-side comparison of playing style. Reuses EXISTING, unmodified
machinery throughout -- `statsbomb_io.py`'s fetch functions (Milestone 9),
`team_report.py`'s `generate_team_report` (Milestone 40) when 360 data is
available, and `habit_memory`'s established 10x7 grid convention
(Milestone 22) for the location-only fallback grid. Nothing in
`BiomechanicalPitchControl`, `feature_extractor.py`, or `team_report.py`
is modified to build this.

TWO ANALYSIS MODES, AUTO-DETECTED PER COMPARISON, NEVER MIXED:

1. **`pitch_control_360`** -- used only when BOTH team-seasons have at
   least one 360-covered match. Reuses `team_report.generate_team_report`
   exactly (full `BiomechanicalPitchControl` weak/strong-zone analysis)
   for both sides.
2. **`event_location_activity_map`** -- the common case. Aggregates raw
   event `location` fields (no 360, no pitch-control physics) into the
   same 10x7 grid `habit_memory`/`player_report`/`team_report` already
   use. Used whenever EITHER side lacks 360 data, so the two sides are
   always compared on the SAME footing -- a 360-based result for one team
   is never diffed against a location-only result for the other.

Per a full-catalog scan of StatsBomb's live `competitions.json`, only 8
season/club combinations in the ENTIRE open-data catalog have any 360
coverage at all (each one a single-club-centric collection: Barcelona
2020/21, Bayer Leverkusen 2023/24, PSG 2021/22 and 2022/23, plus a
handful of international tournaments). Every other team-season --
including every example this module's own validation runs use -- falls
back to mode 2.

DATA RICHNESS (reuses Milestone 44's validation-sweep discipline
exactly): a team-season built from very few matches must never look as
confident as one built from a full season. `LOW_SAMPLE_MATCH_THRESHOLD`
(named and stated openly, the same way `habit_memory.MIN_HISTORICAL_
EVENTS` is) flags this explicitly in the output, and a reliability
caveat is attached to the comparison itself (not just a buried field)
whenever one side is thin.
"""

from production.src.ingestion.statsbomb_io import (
    X_SCALE,
    Y_SCALE,
    fetch_competition_matches,
    fetch_competitions_index,
    fetch_match_360,
    fetch_match_events,
)
from production.src.pipeline.feature_extractor import FINAL_THIRD_X, PITCH_WIDTH
from production.src.pipeline.habit_memory import (
    CELL_HEIGHT_METERS,
    CELL_WIDTH_METERS,
    GRID_COLS,
    GRID_ROWS,
)
from production.src.reporting.team_report import DEFENSIVE_THIRD_X, generate_team_report

# Named and stated openly, mirroring habit_memory.MIN_HISTORICAL_EVENTS's
# convention exactly (Milestone 44's validation sweep): a team-season
# built from fewer matches than this is flagged, never silently presented
# as equally confident as a well-supported side.
LOW_SAMPLE_MATCH_THRESHOLD = 10


def _season_start_year(season_name: str) -> int | None:
    """"2016/2017" -> 2016; a non-split single-year season_name (e.g. a
    tournament's "2023") -> 2023. Returns None for anything unparseable
    rather than raising -- a genuinely malformed season_name should be
    skipped by the caller, not crash the whole comparison."""
    try:
        return int(season_name.split("/")[0])
    except (ValueError, IndexError):
        return None


def _resolve_team_season_matches(team_name: str, start_year: int) -> list[dict]:
    """Searches EVERY competition/season in the LIVE `competitions.json`
    index (Milestone 9's own established discipline: match validity is
    never assumed from a hardcoded list) whose season starts in
    `start_year`, and returns one dict per match where `team_name`
    actually played (home or away) -- across ALL matching competitions,
    not just one. A team appearing in more than one competition in the
    same year (e.g. league + continental cup) has every one of them
    included; this mirrors `team_report.py`'s own `match_ids`-list
    contract, which never assumes a single competition either.

    Each returned dict: `match_id`, `competition_id`, `season_id`,
    `competition_name`, `season_name`, `has_360` (from that match's own
    `match_status_360` field in the matches list -- NOT the season-level
    `match_available_360` flag on `competitions.json`, which can be set
    even when only a small fraction of that season's individual matches
    actually carry 360 data).
    """
    index = fetch_competitions_index()
    matches_info = []

    for entry in index:
        if _season_start_year(entry["season_name"]) != start_year:
            continue

        comp_id, season_id = entry["competition_id"], entry["season_id"]
        matches = fetch_competition_matches(comp_id, season_id)

        for m in matches:
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            if team_name not in (home, away):
                continue
            matches_info.append({
                "match_id": m["match_id"],
                "competition_id": comp_id,
                "season_id": season_id,
                "competition_name": entry["competition_name"],
                "season_name": entry["season_name"],
                "has_360": m.get("match_status_360") == "available",
            })

    return matches_info


def _team_season_summary(team_name: str, start_year: int) -> dict:
    matches_info = _resolve_team_season_matches(team_name, start_year)
    competitions = sorted({(m["competition_name"], m["season_name"]) for m in matches_info})
    return {
        "team_name": team_name,
        "start_year": start_year,
        "match_ids": [m["match_id"] for m in matches_info],
        "match_ids_360": [m["match_id"] for m in matches_info if m["has_360"]],
        "n_matches": len(matches_info),
        "n_matches_360": sum(1 for m in matches_info if m["has_360"]),
        "competitions": [{"competition_name": c, "season_name": s} for c, s in competitions],
    }


def _richness_flag(n_matches: int) -> str:
    return (
        "LOW SAMPLE -- too thin for season-level claims"
        if n_matches < LOW_SAMPLE_MATCH_THRESHOLD
        else "well-supported"
    )


def _reliability_caveat(data_richness: dict) -> str | None:
    """A plain-language, always-visible warning (not just a code-level
    flag) whenever one side of the comparison is built from far fewer
    matches than the other -- Step 2's explicit requirement that this
    NOT be buried."""
    a, b = data_richness["team_a"], data_richness["team_b"]
    a_low = a["matches"] < LOW_SAMPLE_MATCH_THRESHOLD
    b_low = b["matches"] < LOW_SAMPLE_MATCH_THRESHOLD
    if not a_low and not b_low:
        return None

    low_side, other_side = (a, b) if a_low else (b, a)
    ratio = (other_side["matches"] / low_side["matches"]) if low_side["matches"] > 0 else float("inf")
    return (
        f"CAVEAT: {low_side['team']} {low_side['season']}'s side of this comparison is built "
        f"from only {low_side['matches']} match(es), while {other_side['team']} "
        f"{other_side['season']}'s side has {other_side['matches']} ({ratio:.0f}x more). "
        f"This comparison is NOT equally reliable on both sides -- treat "
        f"{low_side['team']} {low_side['season']}'s numbers as illustrative at best, not a "
        "real season-level characterization."
    )


def _build_location_activity_grid(match_ids: list[int], team_name: str) -> tuple[list[list[float]], int]:
    """Aggregates `team_name`'s own event `location` fields (across
    `match_ids`) into the SAME 10x7 grid convention `habit_memory`/
    `player_report`/`team_report` already use (ADR-002's 100x68m space,
    via the same `X_SCALE`/`Y_SCALE` `statsbomb_io.py` applies at
    ingestion) -- a plain EVENT-COUNT DENSITY grid, no pitch-control
    physics involved. This is the location-only fallback, used whenever
    360 data isn't available for at least one side.

    Per ADR-009, StatsBomb records event locations already relative to
    the ACTING team's own attacking-left-to-right frame -- this is what
    makes aggregating locations across many different matches/opponents
    into one "this team's own activity map" meaningful at all; without
    that guarantee, "attacking third" would mean a different real part of
    the pitch depending on which end the team happened to be defending
    that match.

    Returns `(grid, total_located_events)`, where `grid[col][row]` is
    that cell's SHARE of the team's total located events (summing to 1.0
    across the whole grid) -- a density, not a raw count, so two teams
    with very different total event counts are still directly comparable
    cell-by-cell.
    """
    counts = [[0] * GRID_ROWS for _ in range(GRID_COLS)]
    total = 0

    for match_id in match_ids:
        events = fetch_match_events(match_id)
        if events is None:
            continue
        for event in events:
            if event.get("team", {}).get("name") != team_name:
                continue
            location = event.get("location")
            if location is None:
                continue
            x = location[0] * X_SCALE
            y = location[1] * Y_SCALE
            col = min(int(x // CELL_WIDTH_METERS), GRID_COLS - 1)
            row = min(int(y // CELL_HEIGHT_METERS), GRID_ROWS - 1)
            counts[col][row] += 1
            total += 1

    if total == 0:
        return [[0.0] * GRID_ROWS for _ in range(GRID_COLS)], 0

    density = [[counts[col][row] / total for row in range(GRID_ROWS)] for col in range(GRID_COLS)]
    return density, total


def _zone_shares(grid: list[list[float]]) -> dict[str, float]:
    """Coarse zone aggregation of a 10x7 density/value grid into
    defensive/middle/attacking thirds (reusing `team_report.
    DEFENSIVE_THIRD_X` / `feature_extractor.FINAL_THIRD_X` exactly, not
    re-derived) and left/right half (split at `PITCH_WIDTH / 2`), keyed
    by each cell's CENTER coordinate -- the same convention `team_report.
    py`'s own zone bucketing uses for a single ball position, applied
    here per-cell."""
    defensive = middle = attacking = 0.0
    left = right = 0.0

    for col in range(GRID_COLS):
        cell_center_x = col * CELL_WIDTH_METERS + CELL_WIDTH_METERS / 2
        for row in range(GRID_ROWS):
            value = grid[col][row]
            if value is None:
                continue
            cell_center_y = row * CELL_HEIGHT_METERS + CELL_HEIGHT_METERS / 2

            if cell_center_x < DEFENSIVE_THIRD_X:
                defensive += value
            elif cell_center_x > FINAL_THIRD_X:
                attacking += value
            else:
                middle += value

            if cell_center_y < PITCH_WIDTH / 2:
                left += value
            else:
                right += value

    return {
        "defensive_third": defensive,
        "middle_third": middle,
        "attacking_third": attacking,
        "left_half": left,
        "right_half": right,
    }


def _location_summary(label_a: str, zones_a: dict, label_b: str, zones_b: dict) -> str:
    """`label_a`/`label_b` are SEASON-QUALIFIED labels (e.g. "Barcelona
    2008"), not bare team names -- `team_a == team_b` is a real, expected
    case (comparing one club across two eras), and a bare team name on
    both sides of the sentence would be ambiguous/confusing (a real
    usability gap found and fixed during this module's own Step 2
    validation, not a hypothetical concern)."""
    diffs = {zone: zones_a[zone] - zones_b[zone] for zone in zones_a}
    zone, diff = max(diffs.items(), key=lambda kv: abs(kv[1]))
    leader, follower = (label_a, label_b) if diff > 0 else (label_b, label_a)
    leader_share = zones_a[zone] if diff > 0 else zones_b[zone]
    follower_share = zones_b[zone] if diff > 0 else zones_a[zone]
    ratio = (leader_share / follower_share) if follower_share > 0 else float("inf")
    zone_label = zone.replace("_", " ")
    return (
        f"The largest activity difference is in the {zone_label}: {leader} concentrated "
        f"{leader_share * 100:.1f}% of its located events there vs. {follower}'s "
        f"{follower_share * 100:.1f}% ({ratio:.1f}x)."
    )


def _compare_location(
    label_a: str, match_ids_a: list[int], team_a: str,
    label_b: str, match_ids_b: list[int], team_b: str,
) -> dict:
    grid_a, n_a = _build_location_activity_grid(match_ids_a, team_a)
    grid_b, n_b = _build_location_activity_grid(match_ids_b, team_b)

    cell_diff = [[grid_a[c][r] - grid_b[c][r] for r in range(GRID_ROWS)] for c in range(GRID_COLS)]

    zones_a, zones_b = _zone_shares(grid_a), _zone_shares(grid_b)
    zone_diff = {zone: zones_a[zone] - zones_b[zone] for zone in zones_a}

    return {
        "team_a_activity_grid": grid_a,
        "team_a_total_located_events": n_a,
        "team_b_activity_grid": grid_b,
        "team_b_total_located_events": n_b,
        "activity_grid_units": "share of the team's own located events falling in this cell (each grid sums to 1.0)",
        "grid_shape": (
            f"{GRID_COLS} cols (x, {CELL_WIDTH_METERS}m/cell) x {GRID_ROWS} rows "
            f"(y, {CELL_HEIGHT_METERS:.2f}m/cell) -- same convention as habit_memory/"
            "player_report/team_report"
        ),
        "zone_shares": {"team_a": zones_a, "team_b": zones_b},
        "zone_diff_a_minus_b": zone_diff,
        "cell_diff_a_minus_b": cell_diff,
        "summary": _location_summary(label_a, zones_a, label_b, zones_b),
    }


def _control_summary(label_a: str, zone_diff: dict, label_b: str) -> str:
    """`label_a`/`label_b` are season-qualified labels -- see
    `_location_summary`'s docstring for why (`team_a == team_b` is a
    real, expected case)."""
    if not zone_diff:
        return "No comparable pitch-control threat zones were computed for both sides."
    zone, diff = max(zone_diff.items(), key=lambda kv: abs(kv[1]))
    leader = label_a if diff > 0 else label_b
    zone_label = zone.replace("_", " ")
    return (
        f"The largest pitch-control-based threat difference is in the {zone_label}: "
        f"{leader} shows {abs(diff) * 100:.1f} percentage points higher predicted threat "
        f"there than the other side."
    )


def _compare_360(
    label_a: str, match_ids_a: list[int], team_a: str,
    label_b: str, match_ids_b: list[int], team_b: str,
) -> dict:
    """360-mode: reuses `team_report.generate_team_report` EXACTLY, for
    both sides, using only the 360-COVERED subset of each team-season's
    matches -- nothing in `team_report.py` is reimplemented or modified
    here."""
    report_a = generate_team_report(team_a, match_ids_a)
    report_b = generate_team_report(team_b, match_ids_b)

    grid_a, grid_b = report_a["control_heatmap_grid"], report_b["control_heatmap_grid"]
    cell_diff = [
        [
            (grid_a[c][r] - grid_b[c][r])
            if grid_a[c][r] is not None and grid_b[c][r] is not None
            else None
            for r in range(len(grid_a[0]))
        ]
        for c in range(len(grid_a))
    ]

    zone_diff = {
        zone: report_a["threat_by_pitch_zone"][zone] - report_b["threat_by_pitch_zone"][zone]
        for zone in report_a["threat_by_pitch_zone"]
        if report_a["threat_by_pitch_zone"][zone] is not None and report_b["threat_by_pitch_zone"][zone] is not None
    }

    return {
        "team_a_report": report_a,
        "team_b_report": report_b,
        "cell_diff_a_minus_b": cell_diff,
        "threat_by_pitch_zone_diff_a_minus_b": zone_diff,
        "summary": _control_summary(label_a, zone_diff, label_b),
    }


def compare_team_seasons(team_a: str, season_a: int, team_b: str, season_b: int) -> dict:
    """General-purpose team-season style comparison. `season_a`/`season_b`
    are season START years (e.g. 2016 for the 2016/2017 season, or a
    single-year tournament season like 2023 as-is).

    See this module's docstring for the two analysis modes and how the
    choice between them is auto-detected (never mixed) and for the
    data-richness/reliability-caveat discipline (Milestone 44's pattern,
    reused exactly).
    """
    info_a = _team_season_summary(team_a, season_a)
    info_b = _team_season_summary(team_b, season_b)

    data_richness = {
        "team_a": {"team": team_a, "season": season_a, "matches": info_a["n_matches"], "flag": _richness_flag(info_a["n_matches"])},
        "team_b": {"team": team_b, "season": season_b, "matches": info_b["n_matches"], "flag": _richness_flag(info_b["n_matches"])},
    }

    use_360 = info_a["n_matches_360"] > 0 and info_b["n_matches_360"] > 0
    analysis_mode = "pitch_control_360" if use_360 else "event_location_activity_map"
    mode_reason = (
        "Both team-seasons have at least one 360-covered match -- full BiomechanicalPitchControl "
        "weak/strong-zone analysis (team_report.generate_team_report) used for both sides."
        if use_360 else
        "At least one team-season has NO 360-covered matches (the common case -- only 8 "
        "season/club combinations in the entire open-data catalog have any 360 coverage at "
        "all). Falling back to event-LOCATION activity maps (no pitch-control physics) for "
        "BOTH sides, so the two are never compared on different footings."
    )

    label_a, label_b = f"{team_a} {season_a}", f"{team_b} {season_b}"
    if use_360:
        result = _compare_360(label_a, info_a["match_ids_360"], team_a, label_b, info_b["match_ids_360"], team_b)
    else:
        result = _compare_location(label_a, info_a["match_ids"], team_a, label_b, info_b["match_ids"], team_b)

    return {
        "team_a": team_a,
        "season_a": season_a,
        "team_b": team_b,
        "season_b": season_b,
        "analysis_mode": analysis_mode,
        "mode_reason": mode_reason,
        "data_richness": data_richness,
        "reliability_caveat": _reliability_caveat(data_richness),
        "competitions_used": {"team_a": info_a["competitions"], "team_b": info_b["competitions"]},
        **result,
    }


# ============================================================================
# Session/Match Comparison (additive extension): the SAME tool as
# compare_team_seasons above, at a finer granularity -- one team, two
# SPECIFIC matches, not two seasons. compare_team_seasons itself is
# untouched; this reuses `_compare_360`/`_compare_location` directly
# (both already generic over an arbitrary match_ids list -- neither cares
# whether that list represents a whole season or a single match), just
# parameterized with single-match_id lists and a match-level richness/
# 360-detection basis instead of a season-level one. See ADR-021's
# "Session/Match Comparison" addendum for the full Step 0 gating
# reasoning (aggregating one match's several-hundred-plus located events
# into the same 10x7 grid remains safely AGGREGATE under condition 2 --
# verified against this project's real cache, not assumed).
# ============================================================================

# Step 1: match-level richness cannot reuse LOW_SAMPLE_MATCH_THRESHOLD's
# match-COUNT convention -- there is always exactly 1 match per side at
# this granularity by definition, so a count-based threshold is
# meaningless here. Flagged instead by real located-EVENT count within
# that one match.
#
# VERIFIED against real data before picking a number (not reused
# verbatim from MIN_HISTORICAL_EVENTS=20, which sits so far below this
# specific real distribution's floor it would be a meaningless gate at
# THIS granularity): a full scan of every real (team, match) combination
# in this project's whole data/raw/ cache (1,868 pairs) found
# min=896, p5=1,147, median=1,788, mean=1,851, max=3,472 located events.
# 500 sits comfortably below the real observed FLOOR (896) -- no real
# match in this cache would ever be flagged -- while still being a
# meaningful gate against a genuinely degenerate case (an abandoned
# match, partial event coverage, or a team-name typo yielding
# near-zero matched events), the same "named, explicit,
# real-data-verified constant" convention this project's own
# MIN_HISTORICAL_EVENTS/MIN_UNDER_PRESSURE_EVENTS_FOR_CONFIDENT_PRI/
# MIN_TRANSITIONS_FOR_CONFIDENT_PASS_ENTROPY already established,
# recalibrated to match-level comparison's own real distribution.
MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON = 500


def _match_located_event_count(match_id: int, team_name: str) -> int:
    """Real located-event count for `team_name` in ONE specific match --
    the Step 1 basis for match-level richness/low-sample flagging.
    Computed independently of analysis mode (used for the richness check
    regardless of whether `pitch_control_360` or
    `event_location_activity_map` ends up being selected for the
    comparison itself), since a thin real event count is a genuine
    signal-scarcity concern in EITHER mode, not just the location-grid
    one."""
    events = fetch_match_events(match_id)
    if events is None:
        return 0
    return sum(
        1 for e in events
        if e.get("team", {}).get("name") == team_name and e.get("location") is not None
    )


def _match_360_available(match_id: int) -> bool:
    """Whether a SPECIFIC match_id has real, fetchable 360 freeze-frame
    data -- checked via a real `fetch_match_360` call, treating a `None`
    result as unavailable, the SAME verify-via-a-real-fetch discipline
    `team_report.py`'s own chain-frame builder already uses (Step 0.2).
    Deliberately NOT this module's own `_resolve_team_season_matches`
    `match_status_360`-from-the-matches-list approach: that requires
    first resolving which competition/season a match_id belongs to,
    extra work this function's caller (two already-known match_ids)
    doesn't need -- a direct fetch is simpler here and matches
    `team_report.py`'s own established pattern exactly."""
    return fetch_match_360(match_id) is not None


def _match_richness_flag(located_events: int) -> str:
    return (
        "LOW SAMPLE -- too thin for match-level claims"
        if located_events < MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON
        else "well-supported"
    )


def _match_reliability_caveat(data_richness: dict) -> str | None:
    """Match-level analog of `_reliability_caveat` above -- SAME
    plain-language, always-visible warning discipline, using
    MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON (event-count-based,
    per Step 1) rather than LOW_SAMPLE_MATCH_THRESHOLD (match-count-based,
    which cannot apply at this granularity). A new, separate function --
    not a modification of `_reliability_caveat` -- since the two operate
    on differently-shaped richness dicts (`located_events` vs `matches`).
    """
    a, b = data_richness["team_a"], data_richness["team_b"]
    a_low = a["located_events"] < MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON
    b_low = b["located_events"] < MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON
    if not a_low and not b_low:
        return None

    low_side, other_side = (a, b) if a_low else (b, a)
    ratio = (
        (other_side["located_events"] / low_side["located_events"])
        if low_side["located_events"] > 0 else float("inf")
    )
    return (
        f"CAVEAT: {low_side['team']}'s match {low_side['match_id']} side of this comparison has "
        f"only {low_side['located_events']} real located event(s), while match "
        f"{other_side['match_id']}'s side has {other_side['located_events']} ({ratio:.0f}x more). "
        f"This comparison is NOT equally reliable on both sides -- treat match "
        f"{low_side['match_id']}'s numbers as illustrative at best, not a confident "
        "match-level characterization."
    )


def compare_team_matches(team_name: str, match_id_a: int, match_id_b: int) -> dict:
    """Session/Match Comparison: the SAME general-purpose style-comparison
    tool as `compare_team_seasons`, at a finer granularity -- one team,
    two SPECIFIC matches, not two (team, season) pairs.

    Reuses `_compare_360`/`_compare_location` UNCHANGED
    (parameterized with single-match_id lists instead of a season's full
    list) -- `compare_team_seasons` itself is not modified. Match-level
    labels (`"{team_name} (match {match_id})"`) disambiguate the two
    sides the same way `compare_team_seasons`'s season-qualified labels
    do, since `match_id_a`/`match_id_b` for the SAME team is the expected
    use case (comparing one team's own two different matches), not an
    edge case.

    See ADR-021's "Session/Match Comparison" addendum for the full Step 0
    gating reasoning (this remains condition-2-EXEMPT, unconditional, at
    this finer granularity) and Step 1's real-data-verified
    MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON threshold.
    """
    n_a = _match_located_event_count(match_id_a, team_name)
    n_b = _match_located_event_count(match_id_b, team_name)

    data_richness = {
        "team_a": {
            "team": team_name, "match_id": match_id_a, "located_events": n_a,
            "flag": _match_richness_flag(n_a),
        },
        "team_b": {
            "team": team_name, "match_id": match_id_b, "located_events": n_b,
            "flag": _match_richness_flag(n_b),
        },
    }

    use_360 = _match_360_available(match_id_a) and _match_360_available(match_id_b)
    analysis_mode = "pitch_control_360" if use_360 else "event_location_activity_map"
    mode_reason = (
        "Both specific matches have real 360 freeze-frame coverage -- full BiomechanicalPitchControl "
        "weak/strong-zone analysis (team_report.generate_team_report) used for both sides."
        if use_360 else
        "At least one of the two specific matches has NO 360 coverage. Falling back to event-LOCATION "
        "activity maps (no pitch-control physics) for BOTH matches, so the two are never compared on "
        "different footings -- same discipline compare_team_seasons already applies at season level."
    )

    label_a = f"{team_name} (match {match_id_a})"
    label_b = f"{team_name} (match {match_id_b})"
    if use_360:
        result = _compare_360(label_a, [match_id_a], team_name, label_b, [match_id_b], team_name)
    else:
        result = _compare_location(label_a, [match_id_a], team_name, label_b, [match_id_b], team_name)

    return {
        "team_name": team_name,
        "match_id_a": match_id_a,
        "match_id_b": match_id_b,
        "analysis_mode": analysis_mode,
        "mode_reason": mode_reason,
        "data_richness": data_richness,
        "reliability_caveat": _match_reliability_caveat(data_richness),
        **result,
    }
