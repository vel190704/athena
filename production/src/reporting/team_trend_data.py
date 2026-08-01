"""New reporting track: season-by-season team trend data from
football-data.co.uk, covering the "big five" European leagues' top
flight (Premier League, La Liga, Serie A, Bundesliga, Ligue 1).

SCOPE BOUNDARY (do not blur this): football-data.co.uk provides MATCH
RESULTS AND TEAM-LEVEL STATS ONLY -- goals, shots, corners, cards, fouls,
home/away record. It carries NO event-level data and NO pitch
coordinates. This means it can NEVER feed `BiomechanicalPitchControl`,
and this module produces no heatmap and no pitch-control weak-zone
analysis -- that remains `team_report.py`'s job, entirely unchanged and
not imported here. This module answers a different question ("how has
this team's results/output trended year over year") from
`team_report.py`'s ("where is this team spatially strong/weak,
aggregated from historical StatsBomb match footage"). The two are never
combined into one number anywhere in this file, and nothing in
`production/src/reporting/team_report.py` or the physics/spatial/models
stack it depends on is imported or modified here.

DATA SOURCE / LICENSE BASIS (verified directly before writing this file,
not assumed):
- football-data.co.uk's own notes page (footbal-data.co.uk/notes.txt)
  states no explicit license or redistribution terms.
- Its main site states the data is free, but its own stated scope is
  narrower than a generic open license: "All data provided by
  Football-Data are made available for the purposes of league match
  prediction only" (site's own wording), with an explicit disclaimer of
  responsibility for data accuracy. Usage here is analytical/research
  reporting on already-completed matches, consistent with that framing,
  not a redistribution of the raw CSV files themselves.
- The PDDL-licensed GitHub mirror (github.com/footballcsv/england and
  siblings) was checked directly and found STALE: it stops at the
  2020-21 season, more than four seasons behind. It is NOT usable for a
  report claiming coverage through 2025/26.
- Conclusion: this module fetches directly from football-data.co.uk's
  own CSV endpoints (confirmed live and current -- a complete 2025/26
  Premier League season, 380 played matches, was downloaded and read
  directly while building this module), not the stale mirror.

COMPLIANCE SCOPE (state this plainly; do not describe this data source
as simply "free to use"): football-data.co.uk's own stated scope --
"for the purposes of league match prediction only" (notes.txt) -- is
narrower than a general research-use license, and no clean license (MIT,
PDDL, CC-BY, or similar) was found anywhere for these files. This is a
REAL, UNRESOLVED licensing ambiguity, not a solved one, and it is handled
the same conservative way ADR-014 handles the AGPL-derived pitch-keypoint
model: this feature is scoped to strictly PERSONAL, NON-DISTRIBUTED
research use only. Concretely, that means:
- The cached CSVs and any DataFrame/report this module produces are for
  local analysis, not for republishing, redistributing, or bundling the
  underlying data (raw or aggregated) anywhere outside this local
  research use.
- Nothing in this module is wired into `production/src/serving/api.py`'s
  live WebSocket/REST layer or any other network-served endpoint -- same
  restriction ADR-014 places on the AGPL-lineage keypoint model, for the
  same reason (an unresolved license should not be allowed to reach a
  served application by default).
- If this feature is ever extended toward a served/distributed use case,
  the licensing question above must be revisited and resolved first --
  this note is not a permanent green light, it is the current
  conservative stance pending that resolution.

SCHEMA NOTE (verified across two+ leagues, not assumed uniform): England's
CSVs (E0) include a `Referee` column that Spain/Germany/Italy/France's
(SP1/D1/I1/F1) do NOT. This module never reads `Referee`, so that
difference doesn't matter here -- but it is real, and is exactly the kind
of assumption this project's own history (see RESEARCH_FINDINGS.md
§4b) has repeatedly found costly to skip checking. The columns this
module DOES depend on (goals, full-time result, shots, shots-on-target,
fouls, corners, yellow/red cards) were confirmed present, under the same
names, in all five leagues' current-season files.
"""

from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"

# football-data.co.uk's own codes for the "big five" leagues' top flight.
LEAGUE_CODES = {
    "premier_league": "E0",
    "la_liga": "SP1",
    "serie_a": "I1",
    "bundesliga": "D1",
    "ligue_1": "F1",
}

# Same data/raw/-style local caching convention as
# production/src/ingestion/statsbomb_io.py's CACHE_DIR -- a plain,
# gitignored, disk-cached directory keyed by what was fetched.
CACHE_DIR = Path("data/raw/football_data_co_uk")

# Verified present, under these exact names, in every one of the five
# leagues' current-season CSVs (Referee is NOT in this list -- see the
# module docstring's schema note; it's England-only and unused here).
_REQUIRED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]
_STAT_COLUMNS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]


def _season_code(start_year: int) -> str:
    """2025 -> "2526" (the 2025/26 season) -- football-data.co.uk's own
    two-year, four-digit folder-naming convention. Plain modulo handles
    the turn-of-century case (1999 -> "9900") with no special-casing."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _cache_path(league_code: str, start_year: int) -> Path:
    return CACHE_DIR / f"{league_code}_{_season_code(start_year)}.csv"


def _fetch_season_csv(league_code: str, start_year: int) -> pd.DataFrame | None:
    """Downloads (direct HTTP GET, no scraping/browser tooling) or reads
    from the local cache one league-season's match-level CSV.

    Returns None -- never raises -- when the league/season combination
    doesn't exist on football-data.co.uk (a real, expected case: a
    season before this league's archive starts, or one not yet played).
    Callers use this to report an honest gap rather than crash.
    """
    cache_path = _cache_path(league_code, start_year)
    if cache_path.exists():
        matches = pd.read_csv(cache_path, encoding="utf-8-sig")
    else:
        url = BASE_URL.format(season_code=_season_code(start_year), league_code=league_code)
        response = requests.get(url, timeout=15)
        if response.status_code != 200 or not response.content:
            print(
                f"[team_trend_data] {league_code} {_season_label(start_year)}: "
                f"not available at {url} (status {response.status_code}) -- skipping."
            )
            return None

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(response.content)
        matches = pd.read_csv(cache_path, encoding="utf-8-sig")

    missing = [c for c in _REQUIRED_COLUMNS if c not in matches.columns]
    if missing:
        print(
            f"[team_trend_data] {league_code} {_season_label(start_year)}: "
            f"missing expected columns {missing} -- skipping season."
        )
        return None

    # A season fetched mid-way through (or a postponed fixture) can carry
    # rows with no final score yet -- these are not played matches and
    # must not be counted, not silently treated as a 0-0 draw.
    before = len(matches)
    matches = matches.dropna(subset=["FTHG", "FTAG", "FTR"])
    if len(matches) < before:
        print(
            f"[team_trend_data] {league_code} {_season_label(start_year)}: "
            f"dropped {before - len(matches)} row(s) with no final score "
            "(likely an in-progress season / postponed fixture)."
        )

    matches = matches.copy()
    matches[_STAT_COLUMNS] = matches[_STAT_COLUMNS].fillna(0)
    return matches


def _aggregate_season_to_teams(matches: pd.DataFrame) -> dict[str, dict]:
    """One season's match-level rows -> {team_name: aggregated stats
    dict}. Every stat here comes from columns verified (see module
    docstring) to exist under the same name across all five leagues --
    nothing league-specific (e.g. `Referee`) is touched.
    """
    home = pd.DataFrame({
        "team": matches["HomeTeam"],
        "is_home": True,
        "goals_for": matches["FTHG"],
        "goals_against": matches["FTAG"],
        "result": matches["FTR"].map({"H": "W", "D": "D", "A": "L"}),
        "shots_for": matches["HS"],
        "shots_on_target_for": matches["HST"],
        "fouls": matches["HF"],
        "corners": matches["HC"],
        "yellow_cards": matches["HY"],
        "red_cards": matches["HR"],
    })
    away = pd.DataFrame({
        "team": matches["AwayTeam"],
        "is_home": False,
        "goals_for": matches["FTAG"],
        "goals_against": matches["FTHG"],
        "result": matches["FTR"].map({"H": "L", "D": "D", "A": "W"}),
        "shots_for": matches["AS"],
        "shots_on_target_for": matches["AST"],
        "fouls": matches["AF"],
        "corners": matches["AC"],
        "yellow_cards": matches["AY"],
        "red_cards": matches["AR"],
    })
    per_team_match = pd.concat([home, away], ignore_index=True)

    team_stats = {}
    for team, group in per_team_match.groupby("team"):
        home_group = group[group["is_home"]]
        away_group = group[~group["is_home"]]
        matches_played = len(group)
        wins = int((group["result"] == "W").sum())
        draws = int((group["result"] == "D").sum())
        losses = int((group["result"] == "L").sum())
        points = wins * 3 + draws

        def _record(g: pd.DataFrame) -> str:
            return (
                f"{int((g['result'] == 'W').sum())}W-"
                f"{int((g['result'] == 'D').sum())}D-"
                f"{int((g['result'] == 'L').sum())}L"
            )

        team_stats[team] = {
            "matches_played": matches_played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "points": points,
            "points_per_game": points / matches_played if matches_played else None,
            "win_rate": wins / matches_played if matches_played else None,
            "goals_scored": int(group["goals_for"].sum()),
            "goals_conceded": int(group["goals_against"].sum()),
            "goal_difference": int(group["goals_for"].sum() - group["goals_against"].sum()),
            "home_record": _record(home_group),
            "away_record": _record(away_group),
            "shots_for": int(group["shots_for"].sum()),
            "shots_on_target_for": int(group["shots_on_target_for"].sum()),
            "corners_for": int(group["corners"].sum()),
            "fouls_committed": int(group["fouls"].sum()),
            "yellow_cards": int(group["yellow_cards"].sum()),
            "red_cards": int(group["red_cards"].sum()),
        }

    return team_stats


def fetch_team_season_stats(league_code: str, season_range) -> pd.DataFrame:
    """Fetches (downloading + locally caching) every requested season of
    `league_code` (a football-data.co.uk code -- see `LEAGUE_CODES`),
    aggregates each season's match rows to TEAM-season level, and returns
    one combined long-format DataFrame: one row per (team, season).

    `season_range`: an iterable of season START years, e.g.
    `range(2016, 2026)` for the 2016/17 through 2025/26 seasons
    inclusive.

    A league/season combination that doesn't exist on football-data.co.uk
    (too early for this league's archive, or not yet played) is skipped
    with a printed note -- never fabricated, never a crash.
    """
    if league_code not in LEAGUE_CODES.values():
        raise ValueError(f"Unknown league_code {league_code!r}; expected one of {sorted(LEAGUE_CODES.values())}")

    all_rows = []
    for start_year in season_range:
        matches = _fetch_season_csv(league_code, start_year)
        if matches is None:
            continue

        season_label = _season_label(start_year)
        for team_name, stats in _aggregate_season_to_teams(matches).items():
            row = {"league_code": league_code, "season": season_label, "team": team_name}
            row.update(stats)
            all_rows.append(row)

    columns = ["league_code", "season", "team"]
    return pd.DataFrame(all_rows, columns=columns if not all_rows else None)


_TREND_METRICS = ["goals_scored", "goals_conceded", "points", "goal_difference"]


def _compute_year_over_year_deltas(ordered_labels: list[str], seasons_found: dict[str, dict]) -> list[dict]:
    """Deltas between each pair of CONSECUTIVELY-FOUND seasons in the
    output (not necessarily consecutive calendar years -- a team can have
    a gap season in between). `consecutive=False` flags a delta that
    actually spans a gap, so a caller/reader is never misled into reading
    e.g. a 3-year jump as a single season's swing.
    """
    deltas = []
    for prev_label, curr_label in zip(ordered_labels, ordered_labels[1:]):
        prev, curr = seasons_found[prev_label], seasons_found[curr_label]
        prev_start_year = int(prev_label.split("-")[0])
        curr_start_year = int(curr_label.split("-")[0])

        entry = {
            "from_season": prev_label,
            "to_season": curr_label,
            "consecutive": (curr_start_year - prev_start_year) == 1,
        }
        for metric in _TREND_METRICS:
            entry[f"{metric}_delta"] = curr[metric] - prev[metric]
        entry["points_per_game_delta"] = curr["points_per_game"] - prev["points_per_game"]
        entry["win_rate_delta"] = curr["win_rate"] - prev["win_rate"]
        deltas.append(entry)
    return deltas


def generate_team_trend_report(
    team_name: str,
    start_season: int,
    end_season: int,
    *,
    known_aliases: list[str] | None = None,
) -> dict:
    """Season-by-season results/output trend report for `team_name`
    (football-data.co.uk's own spelling, e.g. "Man City", "Nott'm
    Forest") across [`start_season`, `end_season`] inclusive, both given
    as season START years (e.g. 2016 means the 2016/17 season).

    Searches ALL FIVE covered top-flight leagues for each requested
    season, rather than requiring a single pre-specified league -- this
    is what lets a team that moved between the top flight and a lower
    division (relegation/promotion) show up as an honest gap rather than
    require the caller to already know which league covers which years.
    NOTE: this only covers the five leagues' TOP FLIGHT (per this
    project's explicit scope) -- a season spent in a second division
    still shows as a gap here, even though the team was actively playing
    that season; this module does not claim otherwise.

    `known_aliases`: optional list of alternate spellings to also search
    for (e.g. a team that changed its football-data.co.uk-listed name
    partway through the range). Each season is matched against
    `team_name` and every alias; if more than one matches in the same
    season (should not happen under a real rename, but reported rather
    than silently picking one if it does), the FIRST match in
    `[team_name] + known_aliases` order is used and a note is printed.

    Returns a dict with `season_stats` (one entry per season actually
    found), `gap_seasons` (season labels where NEITHER `team_name` nor
    any alias appeared in any of the five leagues), and
    `year_over_year_deltas` (see `_compute_year_over_year_deltas`) --
    gaps are always reported explicitly, never silently dropped from the
    output structure.
    """
    names_to_match = [team_name] + list(known_aliases or [])
    full_range = range(start_season, end_season + 1)

    # Fetch each league's full requested range ONCE (not once per
    # season), so an N-season query issues 5 total fetch-or-cache passes,
    # not 5*N.
    league_frames = {code: fetch_team_season_stats(code, full_range) for code in LEAGUE_CODES.values()}

    seasons_found: dict[str, dict] = {}
    gap_seasons: list[str] = []

    for start_year in full_range:
        season_label = _season_label(start_year)
        matched_row = None

        for name in names_to_match:
            candidates = []
            for df in league_frames.values():
                if df.empty:
                    continue
                hit = df[(df["season"] == season_label) & (df["team"] == name)]
                if not hit.empty:
                    candidates.append(hit.iloc[0].to_dict())
            if candidates:
                if len(candidates) > 1:
                    print(
                        f"[team_trend_data] {name!r} matched in {len(candidates)} leagues for "
                        f"{season_label} -- using the first found; this should not normally happen."
                    )
                matched_row = candidates[0]
                break

        if matched_row is None:
            gap_seasons.append(season_label)
        else:
            seasons_found[season_label] = matched_row

    ordered_labels = sorted(seasons_found.keys(), key=lambda s: int(s.split("-")[0]))

    return {
        "team_name": team_name,
        "known_aliases": list(known_aliases or []),
        "seasons_requested": end_season - start_season + 1,
        "seasons_found": len(seasons_found),
        "gap_seasons": gap_seasons,
        "season_stats": {label: seasons_found[label] for label in ordered_labels},
        "year_over_year_deltas": _compute_year_over_year_deltas(ordered_labels, seasons_found),
    }
