"""Lightweight, report-independent enumeration of what's ACTUALLY cached in
`data/raw/` -- built so the dashboard's Player Reports / Team Reports tabs
can offer a dropdown reflecting real cache contents (including everything
`data_fallback.py`'s runs have pulled in over time), instead of a small,
hand-picked preset list that silently drifts out of date the moment new
data is fetched.

Deliberately does NOT call `generate_player_report`/`generate_team_report`
for every candidate -- doing that per-candidate would mean running the
full, expensive report pipeline once per candidate just to build a
dropdown list (4,000+ players at the current cache size). It DOES,
however, independently re-run the same possession-chain-building step
`team_report.py` itself uses (see `_scan_360_usable_matches`) -- that
step is what actually determines whether a team's report contains real
content, and a verification audit found raw cached-match-count has zero
predictive relationship to it (see "TEAM LOW-SAMPLE DEFINITION" below).

WHAT THIS READS vs. SKIPS, stated explicitly:
- Team/player identity + season grouping: every cached
  `{match_id}_events.json` file's `player`/`team` fields, plus cached
  `matches_{comp}_{season}.json` files (to resolve which
  competition/season a match_id belongs to).
- Team 360-usability (see below): cached `{match_id}_360.json` files
  (existence AND content -- these are read, unlike everything else this
  module treats as existence-only) for matches that have one, plus
  `chain_builder.build_possession_chains` on that match's own already-
  loaded events. Reuses `build_possession_chains` because it is validated
  pipeline infrastructure `team_report.py` itself sits on top of, not
  `team_report.py`'s own report-generation logic -- exactly the same
  "reuse the validated engine, write your own orchestration" principle
  the independent verification audit itself used.
- SKIPS entirely: any positional/heatmap aggregation,
  `BiomechanicalPitchControl` pitch-control physics, MLflow/model access.
  No cumulative-incidence or control value is ever computed here -- only
  whether a representative chain frame WOULD exist for a match, the same
  yes/no `team_report.py`'s own `_match_representative_chain_frames`
  computes, reused independently.

TEAM LOW-SAMPLE DEFINITION (post-audit correction): an independent
verification audit found THREE disagreeing "is this team's data usable"
checks across the codebase (this module's own prior cache-count-based
`total_matches`, `team_report.py`'s internal `matches_used`, and
`dashboard.py`'s separate `matches_used < 2` check) -- and confirmed raw
cached-match count has NO predictive relationship to whether a team
report actually contains data (Real Madrid: 68 cached matches, but only 2
have usable 360 coverage). This module is now the SINGLE authoritative
source: `total_matches_360` (a team's matches that are BOTH cached AND
360-usable, computed via `_scan_360_usable_matches` below) is the only
count `low_sample` is based on. `dashboard.py` no longer has its own
separate threshold -- see that file's own comment at the call site.
`total_matches_cached` (the old metric) is kept as a secondary,
INFORMATIONAL field only, never used for `low_sample`.

TEAM NAME MERGING (post-audit correction): StatsBomb tags the same real
club under a long form in some matches and a short form in others (e.g.
"Stade Malherbe Caen" vs. "Caen" -- confirmed the SAME club, since
`matches_{comp}_{season}.json` lists PSG's opponent as the former while
every event in that exact match tags it the latter). Left unmerged, a
dropdown built from event data alone would silently split one real team's
coverage into two confusing, incomplete-looking entries -- and picking
the "wrong" one for a specific match previously produced a broken,
0-match report (`generate_team_report` matches on the event-derived name
exactly). `TEAM_NAME_MERGES` (below) presents ONE entry per real team,
built from an explicit, individually-verified list -- NOT a fuzzy/
automatic algorithm, because an automatic pass over the current cache
also raised "Guinea" vs. "Equatorial Guinea", which are genuinely
DIFFERENT national teams that happen to have played each other (match
3922239 literally is Guinea vs. Equatorial Guinea) -- automatic merging
would have silently created a real, new bug of exactly this module's own
class. Each season entry still tracks match_ids PER VARIANT
(`match_ids_by_variant`), not just a flattened list, because
`generate_team_report(team_name, match_ids)` takes exactly one name and
a merged entry's own match_ids can span variants even within the SAME
competition-season -- see `dashboard.py`'s Team Reports tab for how a
multi-variant selection is handled (multiple report calls, not a silent
partial one).
"""

import json
from pathlib import Path

from production.src.pipeline.chain_builder import build_possession_chains

CACHE_DIR = Path("data/raw")

# Reused, not re-derived: the one authoritative "is this a low sample"
# boundary each unit already has elsewhere in this project --
# `team_comparison.LOW_SAMPLE_MATCH_THRESHOLD` for teams (matches),
# `habit_memory.MIN_HISTORICAL_EVENTS` for players (events). Importing
# both here would pull in `team_comparison.py`'s full StatsBomb/physics
# import chain and `habit_memory.py`'s, for two integers -- restated
# directly instead, with the source named, matching this project's own
# `constants.py`-style "duplicate a plain value rather than force an
# import" precedent when the alternative is a heavy/unnecessary import
# graph for a dashboard-only enumeration helper.
LOW_SAMPLE_MATCH_THRESHOLD = 10  # team_comparison.LOW_SAMPLE_MATCH_THRESHOLD
LOW_SAMPLE_EVENT_THRESHOLD = 20  # habit_memory.MIN_HISTORICAL_EVENTS

# CONFIRMED long/short-form StatsBomb naming splits for the SAME real
# club, found via a full-inventory sweep (word-subset + fuzzy-similarity
# heuristics over every cached team name) and MANUALLY verified one by
# one before being added here -- see module docstring. Maps the longer
# variant to the canonical (shorter) one, matching the form that appears
# more often across this project's own cache. Two confirmed as of this
# sweep; candidates considered and REJECTED as genuinely different real
# teams: "Guinea"/"Equatorial Guinea" (distinct national teams -- one
# cached match, 3922239, is literally a Guinea vs. Equatorial Guinea
# fixture), "Namibia"/"Gambia"/"Zambia", "Granada"/"Canada" (all
# coincidental short-string similarity between unrelated real teams, not
# a naming-convention split).
TEAM_NAME_MERGES: dict[str, str] = {
    "Stade Malherbe Caen": "Caen",
    "Olympique de Marseille": "Marseille",
}


def _canonical_team_name(name: str) -> str:
    return TEAM_NAME_MERGES.get(name, name)


def _load_competitions_index() -> dict[tuple[int, int], tuple[str, str]]:
    path = CACHE_DIR / "competitions.json"
    if not path.exists():
        return {}
    index = json.loads(path.read_text())
    return {(c["competition_id"], c["season_id"]): (c["competition_name"], c["season_name"]) for c in index}


def _match_id_to_compseason() -> dict[int, tuple[int, int]]:
    """From cached `matches_{comp}_{season}.json` files ONLY -- these are
    small (one per competition-season, not per match), so reading every
    one of them is negligible next to the event-file scan below."""
    out: dict[int, tuple[int, int]] = {}
    for mf in CACHE_DIR.glob("matches_*.json"):
        parts = mf.stem.split("_")
        comp_id, season_id = int(parts[1]), int(parts[2])
        try:
            matches = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for m in matches:
            out[m["match_id"]] = (comp_id, season_id)
    return out


def _scan_cached_events() -> tuple[dict, dict]:
    """The main expensive pass this module makes: opens every cached
    `{match_id}_events.json` file exactly once and extracts BOTH team and
    player appearances from its actual event data. Returns `(teams, players)`:

    - `teams`: `{canonical_team_name: {(comp_id, season_id): {variant_name: set[match_ids]}}}`
      -- see `TEAM_NAME_MERGES`/module docstring for why this is keyed by
      variant within each season rather than a flat match_id set.
    - `players`: `{(player_id, name): {(comp_id, season_id): {"match_ids": set[int], "event_count": int}}}`

    Not cached itself (callers each do a fresh scan; the dashboard layer
    is where real caching across reruns belongs, via `st.cache_data`).
    """
    match_to_compseason = _match_id_to_compseason()

    teams: dict[str, dict[tuple[int, int], dict[str, set[int]]]] = {}
    players: dict[tuple[int, str], dict[tuple[int, int], dict]] = {}

    for ef in CACHE_DIR.glob("*_events.json"):
        match_id = int(ef.stem.split("_")[0])
        cs = match_to_compseason.get(match_id)
        if cs is None:
            # An events file with no corresponding cached match-list entry
            # -- can't attribute a competition/season, so skip rather than
            # guess. Not observed in the current cache, but handled rather
            # than assumed impossible.
            continue
        try:
            events = json.loads(ef.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if events is None:
            continue

        for e in events:
            team = e.get("team")
            if team is not None:
                variant_name = team["name"]
                canonical = _canonical_team_name(variant_name)
                by_variant = teams.setdefault(canonical, {}).setdefault(cs, {})
                by_variant.setdefault(variant_name, set()).add(match_id)

            player = e.get("player")
            if player is not None and player.get("id") is not None:
                key = (player["id"], player["name"])
                rec = players.setdefault(key, {}).setdefault(cs, {"match_ids": set(), "event_count": 0})
                rec["match_ids"].add(match_id)
                rec["event_count"] += 1

    return teams, players


def _scan_360_usable_matches() -> set[int]:
    """Independently reimplements `team_report._match_representative_chain_frames`'s
    yes/no question ("does this match have at least one possession chain
    with a representative event that BOTH has a `location` AND a cached
    360 freeze-frame?") for every match that has a cached `_360.json`
    file -- WITHOUT importing `team_report.py` or calling
    `generate_team_report`. Reuses `build_possession_chains`
    (`chain_builder.py`, validated pipeline infrastructure, not
    `team_report.py`'s own logic).

    Restricted to matches with a cached `_360.json` file (a fast filename
    check, no parsing) BEFORE any chain-building runs -- at the current
    cache size this is ~120 of ~830 cached matches, keeping the real cost
    (possession-chain building, not physics/ML) small and bounded rather
    than proportional to the full event-file cache.
    """
    usable: set[int] = set()
    frame_files = {int(f.stem.split("_")[0]): f for f in CACHE_DIR.glob("*_360.json")}

    for match_id, frame_path in frame_files.items():
        event_path = CACHE_DIR / f"{match_id}_events.json"
        if not event_path.exists():
            continue
        try:
            events = json.loads(event_path.read_text())
            frames = json.loads(frame_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if events is None or frames is None:
            continue

        frame_event_uuids = {f["event_uuid"] for f in frames}
        chains = build_possession_chains(events, periods=(1, 2))

        by_period_possession: dict[tuple[int, int], list] = {}
        for e in events:
            if e["period"] in (1, 2):
                by_period_possession.setdefault((e["period"], e["possession"]), []).append(e)
        for group in by_period_possession.values():
            group.sort(key=lambda e: e["index"])

        for chain in chains:
            chain_events = by_period_possession.get((chain["period"], chain["chain_id"]), [])
            if any(e["id"] in frame_event_uuids and "location" in e for e in chain_events):
                usable.add(match_id)
                break

    return usable


def enumerate_cached_teams() -> list[dict]:
    """One entry per real team (long/short-form naming variants merged --
    see `TEAM_NAME_MERGES`). Each entry: `team_name` (canonical),
    `seasons` (list of `{competition_name, season_name, match_ids,
    match_ids_by_variant, match_ids_360}`, sorted), `total_matches_cached`
    (informational only), `total_matches_360` (the authoritative sample
    size -- see module docstring), `low_sample`
    (`total_matches_360 < LOW_SAMPLE_MATCH_THRESHOLD`).
    """
    comp_lookup = _load_competitions_index()
    teams, _players = _scan_cached_events()
    usable_360 = _scan_360_usable_matches()
    return _reshape_teams(teams, comp_lookup, usable_360)


def enumerate_cached_players() -> list[dict]:
    """One entry per (player_id, name). Each entry: `player_id`, `name`,
    `seasons` (list of
    `{competition_name, season_name, match_ids, event_count}`, sorted --
    this is what lets a caller offer "early Messi" vs. "PSG Messi" as
    distinct, selectable season groups instead of one aggregate),
    `total_matches`, `total_events` (a cheap, RAW tagged-event count --
    NOT `generate_player_report`'s own narrower
    `positional_distribution_event_count`/`heatmap_event_count`, which
    additionally require a `position`/`location` field respectively; this
    total is deliberately a cheaper proxy, good enough for dropdown
    triage/sorting, not claimed identical to the real report's own
    figures), `low_sample` (`total_events < LOW_SAMPLE_EVENT_THRESHOLD`).

    UNCHANGED by the post-audit team fixes above -- Part 1 of the
    verification audit confirmed player_report.py's own event-count-based
    threshold has no discrepancies, so nothing here needed correcting.
    """
    comp_lookup = _load_competitions_index()
    _teams, players = _scan_cached_events()
    return _reshape_players(players, comp_lookup)


def enumerate_cached_candidates() -> tuple[list[dict], list[dict]]:
    """`(teams, players)`, sharing ONE scan of the cached event files
    instead of two -- the entrypoint the dashboard should use when it
    needs both (Player Reports AND Team Reports tabs), so switching
    between them never triggers a second, fully redundant ~equal-cost
    scan. `enumerate_cached_teams`/`enumerate_cached_players` remain
    available standalone for callers that only need one side.
    """
    comp_lookup = _load_competitions_index()
    teams, players = _scan_cached_events()
    usable_360 = _scan_360_usable_matches()
    return _reshape_teams(teams, comp_lookup, usable_360), _reshape_players(players, comp_lookup)


def _reshape_teams(
    teams: dict[str, dict[tuple[int, int], dict[str, set[int]]]],
    comp_lookup: dict[tuple[int, int], tuple[str, str]],
    usable_360: set[int],
) -> list[dict]:
    results = []
    for team_name, seasons in teams.items():
        season_entries = []
        all_match_ids: set[int] = set()
        all_match_ids_360: set[int] = set()
        for (comp_id, season_id), by_variant in seasons.items():
            comp_name, season_name = comp_lookup.get(
                (comp_id, season_id), (f"competition {comp_id}", f"season {season_id}")
            )
            season_match_ids: set[int] = set()
            for variant_matches in by_variant.values():
                season_match_ids.update(variant_matches)
            season_match_ids_360 = season_match_ids & usable_360

            season_entries.append({
                "competition_name": comp_name,
                "season_name": season_name,
                "match_ids": sorted(season_match_ids),
                "match_ids_by_variant": {v: sorted(ids) for v, ids in by_variant.items()},
                "match_ids_360": sorted(season_match_ids_360),
            })
            all_match_ids.update(season_match_ids)
            all_match_ids_360.update(season_match_ids_360)
        season_entries.sort(key=lambda s: (s["competition_name"], s["season_name"]))

        total_matches_360 = len(all_match_ids_360)
        results.append({
            "team_name": team_name,
            "seasons": season_entries,
            "total_matches_cached": len(all_match_ids),
            "total_matches_360": total_matches_360,
            "low_sample": total_matches_360 < LOW_SAMPLE_MATCH_THRESHOLD,
        })
    results.sort(key=lambda t: t["team_name"])
    return results


def _reshape_players(
    players: dict[tuple[int, str], dict[tuple[int, int], dict]],
    comp_lookup: dict[tuple[int, int], tuple[str, str]],
) -> list[dict]:
    results = []
    for (player_id, name), seasons in players.items():
        season_entries = []
        all_match_ids: set[int] = set()
        total_events = 0
        for cs, rec in seasons.items():
            comp_name, season_name = comp_lookup.get(cs, (f"competition {cs[0]}", f"season {cs[1]}"))
            season_entries.append({
                "competition_name": comp_name,
                "season_name": season_name,
                "match_ids": sorted(rec["match_ids"]),
                "event_count": rec["event_count"],
            })
            all_match_ids.update(rec["match_ids"])
            total_events += rec["event_count"]
        season_entries.sort(key=lambda s: (s["competition_name"], s["season_name"]))
        results.append({
            "player_id": player_id,
            "name": name,
            "seasons": season_entries,
            "total_matches": len(all_match_ids),
            "total_events": total_events,
            "low_sample": total_events < LOW_SAMPLE_EVENT_THRESHOLD,
        })
    results.sort(key=lambda p: p["name"])
    return results
