"""Pass Network report (new reporting-track addition).

STANDALONE, additive: reads real StatsBomb event data via the EXISTING
`statsbomb_io.fetch_match_events` (network fetch only on a cache miss),
plus that same module's `X_SCALE`/`Y_SCALE` rescale constants (the same
combined import `player_report.py` already establishes for its shot map).
This module does NOT import or modify `chain_builder.py`, `direction.py`,
or anything under `production/src/physics`, `spatial`, or `models` --
`statsbomb_io`'s existing fetch functions/constants are its only external
data dependency. No CV/video dependency whatsoever.

Single-match scope, deliberately: unlike `player_report.py`/`team_report.py`
(which aggregate across a caller-supplied `match_ids` list), a pass
network is inherently a per-match shape -- a player's average position and
who they combined with only mean something within one match's own
tactical setup, not blended across a career.

ADR-021 condition-2 GATING DECISION (see that ADR's own addendum for the
full reasoning): the RAW network below (real player names at individually-
attributable average locations, connected by real pairwise completed-pass
counts) is treated as RAW, individually-attributable data under condition
2 -- the SAME treatment as the shot map, not an exception. `generate_pass_network`
is the local/private-only variant; `generate_pass_network_aggregated` is
the public-safe counterpart (per-player totals/degree only, no location,
no pairwise edge weights).
"""

import logging
from collections import defaultdict

from production.src.ingestion.statsbomb_io import X_SCALE, Y_SCALE, fetch_match_events

logger = logging.getLogger(__name__)


def _is_complete_pass(event: dict) -> bool:
    """True iff this 'Pass' event has NO `pass.outcome` key at all.

    Mirrors the exact "nested outcome key ABSENCE signals success, not a
    null value" pattern `chain_builder._is_incomplete_reception` already
    established and proved against real data -- reused here as the same
    underlying signal, not reinvented. Deliberately BROADER than that
    function's own check, though: `_is_incomplete_reception` only tests
    `outcome.name == "Incomplete"` (all it needs for ITS purpose, chain-
    termination classification). A pass network instead needs "was this a
    genuinely successful pass to a teammate," which must ALSO exclude
    `outcome.name in {"Out", "Pass Offside", "Unknown", ...}` -- StatsBomb
    populates `pass.outcome` for ANY non-success reason, not just literal
    "Incomplete", so "no outcome key at all" is the correct completion
    signal here, not a narrower name match.

    VERIFIED against match 3857276's real 955 cached Pass events before
    trusting this: every one of the 790 passes with no `outcome` key had a
    `pass.recipient` present (confirms completion implies a resolvable
    recipient); conversely, passes WITH an outcome key (Incomplete/Out/
    Pass Offside/Unknown) sometimes STILL carry a `pass.recipient` --
    StatsBomb records the intended target even when the pass failed to
    reach them, e.g. 59 real 'Incomplete' passes in that match had a
    `recipient` field naming the passer's own intended teammate. This is
    exactly why completion must be decided from `outcome`, never from
    `recipient` presence -- see `_completed_pass_recipient` below.
    """
    return "outcome" not in event.get("pass", {})


def _completed_pass_recipient(event: dict) -> dict | None:
    """The real receiving player's `{"id":.., "name":..}` dict for a
    CONFIRMED-complete pass (see `_is_complete_pass`), or `None`.

    Only ever called after `_is_complete_pass` has already returned True --
    `pass.recipient` is NOT trusted as a completion signal by itself (see
    that function's docstring: it can be present on a real, verified
    INCOMPLETE pass too, naming the intended-but-never-reached teammate).
    """
    return event.get("pass", {}).get("recipient")


def _starting_xi_player_ids(events: list) -> dict[int, dict]:
    """`{player_id: {"name":.., "team":..}}` for every player who appears
    in EITHER team's real `Starting XI` `tactics.lineup` for this match
    (field presence VERIFIED against match 3857276's real cached data
    before relying on it -- same `tactics.lineup` structure
    `player_report._starting_team` already established and verified).

    Filter choice (Step 1.3): STARTING XI membership, not a minutes-played
    threshold. A pass network conventionally shows one team's XI's own
    passing shape; a starting-XI filter is simpler, directly available in
    this data with no derived on-pitch-interval computation needed (unlike
    `player_report._substitution_times`'s own minutes-played logic, built
    for a different purpose -- career totals, not one match's shape), and
    avoids a late substitute's few end-of-match passes distorting a node's
    average position toward wherever they happened to touch the ball in
    a handful of minutes.
    """
    players: dict[int, dict] = {}
    for event in events:
        if event["type"]["name"] != "Starting XI":
            continue
        team = event.get("team", {}).get("name")
        for entry in event.get("tactics", {}).get("lineup", []):
            player = entry.get("player")
            if player is None or player.get("id") is None:
                continue
            players[player["id"]] = {"name": player["name"], "team": team}
    return players


def generate_pass_network(match_id: int) -> dict:
    """RAW pass network for `match_id` -- LOCAL/PRIVATE USE ONLY (ADR-021
    condition 2: see this module's own docstring and `generate_pass_network_aggregated`
    below for the public-safe counterpart). Real per-player average
    location and real pairwise completed-pass edge weights, individually
    attributable to named players -- structurally the same class of data
    the shot map's raw variant exposes, gated the same way.

    Scope: both teams' Starting XI players only (see `_starting_xi_player_ids`).
    A pass is only counted as a network edge when BOTH the passer and the
    completed-pass recipient are Starting XI players -- a starting player's
    passes into a substitute (after that substitute enters) are excluded,
    same as a substitute's own passes are, since neither has a
    whole-match average position to plot meaningfully.

    Returns `{"no_data": True, "reason": ...}` (no `nodes`/`edges` key) if
    `match_id` has no fetchable event data, mirroring
    `api.py`'s existing "no_data" convention for a caller-supplied
    match/season with nothing usable.
    """
    events = fetch_match_events(match_id)
    if events is None:
        logger.warning(f"match_id={match_id}: no events data available.")
        return {"match_id": match_id, "no_data": True, "reason": "No event data available for this match_id."}

    starting_xi = _starting_xi_player_ids(events)

    location_sum: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    location_count: dict[int, int] = defaultdict(int)
    edge_completed_counts: dict[tuple[int, int], int] = defaultdict(int)

    total_pass_attempts_considered = 0
    excluded_non_starting_xi_passes = 0

    for event in events:
        if event.get("type", {}).get("name") != "Pass":
            continue
        passer = event.get("player")
        location = event.get("location")
        if passer is None or location is None:
            continue
        passer_id = passer["id"]
        if passer_id not in starting_xi:
            excluded_non_starting_xi_passes += 1
            continue

        total_pass_attempts_considered += 1
        scaled_location = [location[0] * X_SCALE, location[1] * Y_SCALE]
        location_sum[passer_id][0] += scaled_location[0]
        location_sum[passer_id][1] += scaled_location[1]
        location_count[passer_id] += 1

        if not _is_complete_pass(event):
            continue
        recipient = _completed_pass_recipient(event)
        if recipient is None or recipient.get("id") is None:
            # Verified not observed in real cached data for a confirmed-
            # complete pass (see _is_complete_pass's docstring), but not
            # assumed impossible -- skipped, not fabricated.
            logger.warning(
                f"match_id={match_id}: complete pass event {event.get('id')} has no resolvable "
                "recipient -- skipping this edge."
            )
            continue
        recipient_id = recipient["id"]
        if recipient_id not in starting_xi:
            continue  # completed pass into a substitute -- not part of the starting-XI shape
        edge_completed_counts[(passer_id, recipient_id)] += 1

    nodes = [
        {
            "player_id": player_id,
            "name": starting_xi[player_id]["name"],
            "team": starting_xi[player_id]["team"],
            "avg_location": [
                location_sum[player_id][0] / location_count[player_id],
                location_sum[player_id][1] / location_count[player_id],
            ],
            "pass_attempts": location_count[player_id],
        }
        for player_id in location_count
    ]
    edges = [
        {
            "passer_id": passer_id,
            "recipient_id": recipient_id,
            "passer_name": starting_xi[passer_id]["name"],
            "recipient_name": starting_xi[recipient_id]["name"],
            "completed_passes": count,
        }
        for (passer_id, recipient_id), count in edge_completed_counts.items()
    ]

    return {
        "match_id": match_id,
        "no_data": False,
        "nodes": nodes,
        "edges": edges,
        "starting_xi_player_ids": sorted(starting_xi.keys()),
        "total_pass_attempts_considered": total_pass_attempts_considered,
        "excluded_non_starting_xi_passes": excluded_non_starting_xi_passes,
    }


def generate_pass_network_aggregated(match_id: int) -> dict:
    """ADR-021 condition-2-compliant variant of `generate_pass_network`,
    for PUBLIC deployments -- mirrors `player_report.generate_player_shot_map_aggregated`'s
    own raw-pop-then-summarize pattern exactly (reuses `generate_pass_network`
    internally rather than re-implementing its fetch/filter logic; the raw
    `nodes`/`edges` lists are LOCAL variables of this function only,
    popped via `dict.pop` -- never left on the returned dict, so they
    cannot leak through a future `**` merge mistake).

    Returns real per-player TOTALS (completed passes sent/received,
    distinct passing-partner count) and network-level summary stats
    (player/edge counts, density, total completed passes) -- no
    individual player's average LOCATION and no PAIRWISE edge weight (the
    two things that make the raw variant individually-attributable in the
    same way the shot map's per-shot `location` does) appear anywhere in
    this return dict.
    """
    full = generate_pass_network(match_id)
    if full.get("no_data"):
        return full

    nodes = full.pop("nodes")
    edges = full.pop("edges")

    completed_sent: dict[int, int] = defaultdict(int)
    completed_received: dict[int, int] = defaultdict(int)
    partners: dict[int, set] = defaultdict(set)
    for edge in edges:
        completed_sent[edge["passer_id"]] += edge["completed_passes"]
        completed_received[edge["recipient_id"]] += edge["completed_passes"]
        partners[edge["passer_id"]].add(edge["recipient_id"])
        partners[edge["recipient_id"]].add(edge["passer_id"])

    player_summary = [
        {
            "player_id": node["player_id"],
            "name": node["name"],
            "team": node["team"],
            "completed_passes_sent": completed_sent.get(node["player_id"], 0),
            "completed_passes_received": completed_received.get(node["player_id"], 0),
            "distinct_passing_partners": len(partners.get(node["player_id"], set())),
        }
        for node in nodes
    ]

    num_players = len(nodes)
    num_possible_directed_edges = num_players * (num_players - 1)
    total_completed_passes = sum(edge["completed_passes"] for edge in edges)

    return {
        **full,
        "player_summary": player_summary,
        "num_players": num_players,
        "num_edges": len(edges),
        "network_density": (len(edges) / num_possible_directed_edges) if num_possible_directed_edges > 0 else 0.0,
        "total_completed_passes_in_network": total_completed_passes,
    }
