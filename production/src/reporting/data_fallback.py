"""Auto-fallback coverage expansion for the reporting track:
`find_or_fetch_team_matches` / `find_or_fetch_player_matches`.

STATUS: this was previously discussed/planned in this project's working
history but never actually written into any file (confirmed by an
exhaustive codebase search before this file was created) -- this is a
fresh implementation, not a resurrection of prior code. It reuses
Milestone 9's existing StatsBomb fetch/cache machinery
(`statsbomb_io.fetch_competitions_index` / `fetch_competition_matches` /
`fetch_match_events`) EXACTLY as built -- nothing in `statsbomb_io.py`,
`player_report.py`, or `team_report.py` is modified or reimplemented
here. This module only ORCHESTRATES those existing functions across
MULTIPLE competitions/seasons, then hands the resulting match_ids to the
existing, unmodified `generate_player_report`/`generate_team_report`.

WHAT PROBLEM THIS SOLVES: `generate_player_report`/`generate_team_report`
both require the caller to already know which real StatsBomb match_ids
to pass in. Before this module, discovering those match_ids for a player/
team not already exercised by this project's existing test fixtures
meant manually re-deriving them (as happened repeatedly in this
project's own history -- e.g. the Real Madrid/Barcelona team-comparison
scoping work). This module automates that discovery: given a team name
(or a player_id, optionally narrowed by candidate team names), it
searches EVERY competition/season in the LIVE `competitions.json` index
(never a hardcoded list -- the same discipline
`statsbomb_io.find_360_competitions` already applies) and returns every
real match_id where that team/player actually appears, fetching (and
therefore caching, via `statsbomb_io`'s own existing cache) whatever
match-lists/events aren't already local.

COST NOTE, stated plainly: `find_or_fetch_team_matches` is cheap (one
match-list fetch per competition-season, no event-level fetch needed,
since team names are already present in the match-list JSON itself).
`find_or_fetch_player_matches` is NOT cheap in the fully unscoped case --
match-list JSON carries no player roster, so confirming a player actually
played requires fetching that match's full EVENT file. Passing
`candidate_team_names` (when the caller has a reasonable guess, e.g. a
player's known clubs) narrows the candidate-match set before any event
fetch happens, and should always be preferred over an unscoped
catalog-wide player search when a hint is available.
"""

from production.src.ingestion.statsbomb_io import (
    fetch_competition_matches,
    fetch_competitions_index,
    fetch_match_events,
)


def _all_competition_season_pairs() -> list[tuple[int, int, str, str]]:
    """`(competition_id, season_id, competition_name, season_name)` for
    every entry in the LIVE competitions index -- never a hardcoded list,
    matching `statsbomb_io.find_360_competitions`'s own established
    discipline of trusting the live source over memory."""
    index = fetch_competitions_index()
    return [
        (c["competition_id"], c["season_id"], c["competition_name"], c["season_name"])
        for c in index
    ]


def find_or_fetch_team_matches(
    team_name: str,
    competitions: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Searches (and fetches/caches as needed, via the existing
    `fetch_competition_matches`) every competition/season for `team_name`
    -- across the WHOLE live catalog by default, or restricted to the
    given `(competition_id, season_id)` pairs if provided.

    Returns one dict per match found: `match_id`, `competition_id`,
    `season_id`, `competition_name`, `season_name`, `opponent`, `has_360`
    (from that match's own `match_status_360` field, NOT the season-level
    `match_available_360` flag on `competitions.json` -- the same
    distinction `team_comparison.py`'s `_resolve_team_season_matches`
    already draws, reused here for consistency). Every returned
    `match_id` is immediately usable as-is in
    `generate_team_report(team_name, match_ids)` -- no further
    transformation needed.
    """
    pairs = _all_competition_season_pairs()
    if competitions is not None:
        wanted = set(competitions)
        pairs = [p for p in pairs if (p[0], p[1]) in wanted]

    found = []
    for comp_id, season_id, comp_name, season_name in pairs:
        matches = fetch_competition_matches(comp_id, season_id)
        for m in matches:
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            if team_name not in (home, away):
                continue
            found.append({
                "match_id": m["match_id"],
                "competition_id": comp_id,
                "season_id": season_id,
                "competition_name": comp_name,
                "season_name": season_name,
                "opponent": away if home == team_name else home,
                "has_360": m.get("match_status_360") == "available",
            })
    return found


def find_or_fetch_player_matches(
    player_id: int,
    candidate_team_names: list[str] | None = None,
    competitions: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Searches (and fetches/caches as needed) every competition/season
    for a real tagged event by `player_id`.

    `candidate_team_names`: if given, only matches involving one of these
    teams are even considered for an event-level fetch (see module
    docstring's cost note) -- strongly recommended whenever the caller
    has any reasonable idea which club(s)/national team(s) the player
    represented, since an unscoped catalog-wide search fetches EVERY
    match's full event file to check.

    Returns one dict per match where `player_id` has at least one tagged
    event: `match_id`, `competition_id`, `season_id`, `competition_name`,
    `season_name`, `has_360` (see `find_or_fetch_team_matches`'s
    docstring for why this is the per-match field, not the season-level
    flag). Every returned `match_id` is immediately usable in
    `generate_player_report(player_id, match_ids)`.
    """
    pairs = _all_competition_season_pairs()
    if competitions is not None:
        wanted = set(competitions)
        pairs = [p for p in pairs if (p[0], p[1]) in wanted]

    candidate_matches = []
    for comp_id, season_id, comp_name, season_name in pairs:
        matches = fetch_competition_matches(comp_id, season_id)
        for m in matches:
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            if candidate_team_names is not None and home not in candidate_team_names and away not in candidate_team_names:
                continue
            candidate_matches.append((
                m["match_id"], comp_id, season_id, comp_name, season_name,
                m.get("match_status_360") == "available",
            ))

    found = []
    for match_id, comp_id, season_id, comp_name, season_name, has_360 in candidate_matches:
        events = fetch_match_events(match_id)
        if events is None:
            continue
        if any(e.get("player", {}).get("id") == player_id for e in events):
            found.append({
                "match_id": match_id,
                "competition_id": comp_id,
                "season_id": season_id,
                "competition_name": comp_name,
                "season_name": season_name,
                "has_360": has_360,
            })
    return found
