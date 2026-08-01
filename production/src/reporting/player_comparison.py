"""General-purpose player-era style comparison tool.

Given any player and two seasons, produces a real, data-grounded
side-by-side comparison of that player's positional role and spatial
activity between the two eras. Mirrors `team_comparison.py`'s design
exactly, for individual players instead of teams, and reuses existing,
unmodified machinery throughout: `player_report.py`'s
`generate_player_report` for each side's positional distribution/heatmap,
`data_fallback.py`'s `find_or_fetch_player_matches` to resolve a
player+season into real match_ids (searching/fetching across the live
StatsBomb catalog, not requiring the caller to already know the right
competition/season), and `team_comparison.py`'s own `_zone_shares` /
`_season_start_year` helpers, imported directly rather than
reimplemented a second time. Nothing in `player_report.py`,
`team_report.py`, `team_comparison.py`, `BiomechanicalPitchControl`, or
`feature_extractor.py` is modified to build this.

A REAL ARCHITECTURAL DIFFERENCE FROM `team_comparison.py`, stated
plainly rather than glossed over: `team_comparison.py` has two genuinely
different code paths depending on 360 availability (full
`BiomechanicalPitchControl` pitch-control analysis vs. an event-location
activity map), because `generate_team_report` itself branches on 360
data. `generate_player_report` does NOT -- read directly, it only ever
calls `fetch_match_events` and `habit_memory.generate_player_heatmap`
(itself built from event `location`, never from a 360 freeze-frame or
`BiomechanicalPitchControl`). So there is only ONE real mode here:
event-location/heatmap-based, regardless of 360 availability. This
module still checks 360 availability per season (never assumed) and
reports it, but strictly as an INFORMATIONAL diagnostic -- "would a
future pitch-control-level player comparison be possible in principle"
-- not a functional mode switch, because the function being reused
(`generate_player_report`) has no pitch-control mode to switch into.
Presenting this as a real second mode when the underlying reused
function has no such path would overstate what this tool actually does.

DATA RICHNESS (Milestone 44's validation-sweep discipline, reused
exactly -- same threshold `player_report.py`/`player_visualizer.py`
already use for this exact signal, not a new number invented here):
`LOW_SAMPLE_EVENT_COUNT_THRESHOLD = habit_memory.MIN_HISTORICAL_EVENTS`.
A season whose `heatmap_event_count` falls below this is flagged
explicitly, and a plain-language reliability caveat is attached to the
comparison itself whenever either side is thin.

POSITIONAL ROLE DIFF, kept as its own explicit section, separate from
the spatial zone diff: the one thing genuinely specific to comparing a
PLAYER across eras (not applicable to a team) is that their tagged
on-pitch ROLE can itself change between seasons -- a winger moving
inside to a central role, for instance. `generate_player_report`'s
`positional_distribution` field already carries exactly this signal;
this module diffs it directly, never conflating it with the spatial
heatmap diff (a real shift in WHERE a player's events are recorded can
happen with or without a change in their tagged ROLE, and the two
questions should not be blurred into one number).
"""

from production.src.pipeline.habit_memory import MIN_HISTORICAL_EVENTS
from production.src.reporting.data_fallback import find_or_fetch_player_matches
from production.src.reporting.player_report import generate_player_report
from production.src.reporting.team_comparison import _season_start_year, _zone_shares
from production.src.ingestion.statsbomb_io import fetch_competitions_index
from production.src.reporting.team_trend_data import _season_label

# Reused directly from habit_memory -- the SAME threshold player_report.py/
# player_visualizer.py already use for `heatmap_used_uniform_fallback`,
# not a separately-invented number (Milestone 44's own discipline).
LOW_SAMPLE_EVENT_COUNT_THRESHOLD = MIN_HISTORICAL_EVENTS


def _resolve_player_season_matches(
    player_id: int,
    start_year: int,
    candidate_team_names: list[str] | None = None,
) -> list[dict]:
    """Resolves `(player_id, start_year)` to real match dicts by scoping
    `data_fallback.find_or_fetch_player_matches` to just the
    competitions whose season starts in `start_year` (via
    `team_comparison._season_start_year`, imported directly rather than
    redefined a second time) -- narrower and faster than searching the
    player's entire career every time a single season is needed.
    """
    index = fetch_competitions_index()
    competitions = [
        (c["competition_id"], c["season_id"])
        for c in index
        if _season_start_year(c["season_name"]) == start_year
    ]
    return find_or_fetch_player_matches(player_id, candidate_team_names=candidate_team_names, competitions=competitions)


def _richness_flag(heatmap_event_count: int, uniform_fallback: bool) -> str:
    return (
        f"LOW SAMPLE -- uniform heatmap fallback ({heatmap_event_count} events "
        f"< {LOW_SAMPLE_EVENT_COUNT_THRESHOLD})"
        if uniform_fallback
        else "well-supported"
    )


def _reliability_caveat(data_richness: dict) -> str | None:
    a, b = data_richness["season_a"], data_richness["season_b"]
    a_low = a["heatmap_used_uniform_fallback"]
    b_low = b["heatmap_used_uniform_fallback"]
    if not a_low and not b_low:
        return None

    low_side, other_side = (a, b) if a_low else (b, a)
    return (
        f"CAVEAT: {low_side['season_label']}'s side of this comparison has only "
        f"{low_side['heatmap_event_count']} qualifying event(s) (below the "
        f"{LOW_SAMPLE_EVENT_COUNT_THRESHOLD}-event cold-start threshold, uniform-fallback "
        f"territory), while {other_side['season_label']} has {other_side['heatmap_event_count']}. "
        "This comparison is NOT equally reliable on both sides -- treat "
        f"{low_side['season_label']}'s numbers as illustrative at best."
    )


def _positional_role_diff(dist_a: dict, dist_b: dict) -> dict:
    positions = set(dist_a) | set(dist_b)
    return {pos: dist_b.get(pos, 0.0) - dist_a.get(pos, 0.0) for pos in positions}


def _positional_role_summary(label_a: str, dist_a: dict, label_b: str, dist_b: dict, diff: dict) -> str:
    if not diff:
        return "No positional data available for this comparison."
    position, delta = max(diff.items(), key=lambda kv: abs(kv[1]))
    grower_label, other_label = (label_b, label_a) if delta > 0 else (label_a, label_b)
    grower_share = dist_b.get(position, 0.0) if delta > 0 else dist_a.get(position, 0.0)
    other_share = dist_a.get(position, 0.0) if delta > 0 else dist_b.get(position, 0.0)
    return (
        f"The largest positional-role shift is at {position}: {grower_label} played there "
        f"{grower_share * 100:.1f}% of tagged events vs. {other_label}'s {other_share * 100:.1f}% "
        f"({abs(delta) * 100:.1f} percentage points)."
    )


def _spatial_summary(label_a: str, zones_a: dict, label_b: str, zones_b: dict) -> str:
    diffs = {zone: zones_b[zone] - zones_a[zone] for zone in zones_a}
    zone, delta = max(diffs.items(), key=lambda kv: abs(kv[1]))
    leader_label, follower_label = (label_b, label_a) if delta > 0 else (label_a, label_b)
    leader_share = zones_b[zone] if delta > 0 else zones_a[zone]
    follower_share = zones_a[zone] if delta > 0 else zones_b[zone]
    ratio = (leader_share / follower_share) if follower_share > 0 else float("inf")
    zone_label = zone.replace("_", " ")
    return (
        f"The largest spatial activity difference is in the {zone_label}: {leader_label} "
        f"concentrated {leader_share * 100:.1f}% of heatmap density there vs. "
        f"{follower_label}'s {follower_share * 100:.1f}% ({ratio:.1f}x)."
    )


def compare_player_seasons(
    player_id: int,
    season_a: int,
    season_b: int,
    *,
    candidate_team_names: list[str] | None = None,
) -> dict:
    """General-purpose player-era comparison. `season_a`/`season_b` are
    season START years (e.g. 2005 for the 2005/06 season).
    `candidate_team_names` narrows the underlying
    `find_or_fetch_player_matches` search (strongly recommended whenever
    the player's clubs/national team are known -- see `data_fallback.py`'s
    cost note) but is not required.

    See this module's docstring for why there is only one real analysis
    mode here (unlike `team_comparison.py`), and for the data-richness/
    reliability-caveat discipline.
    """
    matches_a = _resolve_player_season_matches(player_id, season_a, candidate_team_names)
    matches_b = _resolve_player_season_matches(player_id, season_b, candidate_team_names)
    match_ids_a = [m["match_id"] for m in matches_a]
    match_ids_b = [m["match_id"] for m in matches_b]

    report_a = generate_player_report(player_id, match_ids_a)
    report_b = generate_player_report(player_id, match_ids_b)

    label_a = f"the {_season_label(season_a)} season"
    label_b = f"the {_season_label(season_b)} season"

    data_richness = {
        "season_a": {
            "season": season_a,
            "season_label": label_a,
            "matches_found": len(matches_a),
            "matches_player_appeared_in": report_a["matches_player_appeared_in"],
            "positional_distribution_event_count": report_a["positional_distribution_event_count"],
            "heatmap_event_count": report_a["heatmap_event_count"],
            "heatmap_used_uniform_fallback": report_a["heatmap_used_uniform_fallback"],
            "flag": _richness_flag(report_a["heatmap_event_count"], report_a["heatmap_used_uniform_fallback"]),
        },
        "season_b": {
            "season": season_b,
            "season_label": label_b,
            "matches_found": len(matches_b),
            "matches_player_appeared_in": report_b["matches_player_appeared_in"],
            "positional_distribution_event_count": report_b["positional_distribution_event_count"],
            "heatmap_event_count": report_b["heatmap_event_count"],
            "heatmap_used_uniform_fallback": report_b["heatmap_used_uniform_fallback"],
            "flag": _richness_flag(report_b["heatmap_event_count"], report_b["heatmap_used_uniform_fallback"]),
        },
    }

    # Informational only -- see module docstring: generate_player_report
    # has no pitch-control code path, so this can never change WHICH
    # analysis actually runs, only report whether it theoretically could
    # in a future extension.
    n_360_a = sum(1 for m in matches_a if m.get("has_360"))
    n_360_b = sum(1 for m in matches_b if m.get("has_360"))
    pitch_control_diagnostic = {
        "season_a_360_matches": n_360_a,
        "season_b_360_matches": n_360_b,
        "pitch_control_possible_in_principle": n_360_a > 0 and n_360_b > 0,
        "note": (
            "Informational only. generate_player_report (reused unmodified here) has no "
            "pitch-control code path at all -- it never calls fetch_match_360 or "
            "BiomechanicalPitchControl, unlike generate_team_report. This diagnostic reports "
            "whether 360 data exists for both seasons, not whether it was used, because it "
            "never is by this tool."
        ),
    }

    # --- Positional role diff (own section, per this module's explicit scope) ---
    role_diff = _positional_role_diff(report_a["positional_distribution"], report_b["positional_distribution"])
    positional_role = {
        "season_a_distribution": report_a["positional_distribution"],
        "season_b_distribution": report_b["positional_distribution"],
        "diff_b_minus_a": role_diff,
        "summary": _positional_role_summary(
            label_a, report_a["positional_distribution"], label_b, report_b["positional_distribution"], role_diff
        ),
    }

    # --- Spatial zone diff (own section, reusing team_comparison._zone_shares exactly) ---
    zones_a = _zone_shares(report_a["heatmap_grid"])
    zones_b = _zone_shares(report_b["heatmap_grid"])
    zone_diff = {zone: zones_b[zone] - zones_a[zone] for zone in zones_a}
    spatial_activity = {
        "season_a_zone_shares": zones_a,
        "season_b_zone_shares": zones_b,
        "diff_b_minus_a": zone_diff,
        "summary": _spatial_summary(label_a, zones_a, label_b, zones_b),
    }

    return {
        "player_id": player_id,
        "season_a": season_a,
        "season_b": season_b,
        "data_richness": data_richness,
        "reliability_caveat": _reliability_caveat(data_richness),
        "pitch_control_diagnostic": pitch_control_diagnostic,
        "positional_role": positional_role,
        "spatial_activity": spatial_activity,
    }


def compare_player_across_eras(
    player_id: int,
    season_list: list[int],
    *,
    candidate_team_names: list[str] | None = None,
) -> list[dict]:
    """Pairwise comparisons across CONSECUTIVE entries in `season_list`
    (e.g. `[2005, 2014, 2021]` -> [2005-vs-2014, 2014-vs-2021]) -- reuses
    `compare_player_seasons` directly for each pair, no separate N-way
    comparison structure."""
    return [
        compare_player_seasons(player_id, season_list[i], season_list[i + 1], candidate_team_names=candidate_team_names)
        for i in range(len(season_list) - 1)
    ]
