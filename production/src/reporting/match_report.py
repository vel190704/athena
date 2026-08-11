"""Automatic Match Report (new reporting track, Part A): pure COMPILATION
of already-computed outputs for ONE match into a single document.

Step 0 dedup check (done before writing this module, see the task's own
report): this capability genuinely does not exist anywhere yet. Producing
the same picture today requires a user to separately open the Team Reports
tab twice (once per side), the Opposition Analysis panel twice, the Pass
Network tab once, and the Alerts History tab filtered by hand to one
`match_id` -- six separate lookups, no compiled single document. This
module closes exactly that COMPILATION gap, nothing else.

Calls EXISTING report-generation functions ONLY -- `generate_team_report`,
`generate_team_opposition_analysis`, `generate_pass_network`(`_aggregated`),
`alert_store.fetch_alerts` -- none of them modified, none of their logic
reimplemented or duplicated here. The one genuinely NEW piece
(`build_match_report_narrative_prompt`) is a plain, SYNCHRONOUS
prompt-builder mirroring `zone_explainer.build_zone_explanation_prompt`'s
own role exactly: it only assembles a grounded prompt STRING from this
module's own compiled dict. The actual async Gemini/mock dispatch stays in
`api.py`, via `explainer.generate_tactical_explanation_with_source`,
UNMODIFIED -- the same real/mock split (and the same honesty
`system_instruction`) every other LLM-touching reporting feature in this
project already uses.
"""

import logging

from production.src.reporting.pass_network import generate_pass_network
from production.src.reporting.team_report import (
    _teams_in_match,
    generate_team_opposition_analysis,
    generate_team_report,
)
from production.src.serving.alert_store import fetch_alerts

logger = logging.getLogger(__name__)


def generate_automatic_match_report(match_id: int, pass_network_fn=generate_pass_network) -> dict:
    """Compiles team reports for BOTH sides, opposition analysis for BOTH
    sides, this match's pass network, and this `match_id`'s own alert
    history into one document. No LLM narrative here -- see
    `build_match_report_narrative_prompt` (this module) and `api.py`'s new
    `/reports/match/{match_id}` endpoint for that separate, async step.

    `pass_network_fn`: `generate_pass_network` (default, real per-player
    locations/edges) or `generate_pass_network_aggregated` -- the caller's
    choice. `api.py` passes the aggregated variant under
    `PUBLIC_DEPLOYMENT`, mirroring the existing
    `/reports/pass-network/{match_id}` endpoint's own gating decision
    exactly, made once at the call site -- not duplicated as new gating
    logic in this reporting-layer module (this module, like every other
    `production/src/reporting/*.py` module, does not read `PUBLIC_DEPLOYMENT`
    itself).

    `team_reports`/`opposition_analysis` are keyed by team name and scoped
    to `[match_id]` only (a single-match slice of the same multi-match
    functions the Team Reports tab already calls) -- both real functions
    already handle a team with no 360 coverage for this match gracefully
    (`matches_used=0`, an all-None heatmap, no crash; see
    `generate_team_report`'s own docstring), so no special-casing is added
    here for that.

    Teams are the real teams that played in `match_id` (`_teams_in_match`,
    reused unmodified), sorted alphabetically for a deterministic `teams`
    ordering -- StatsBomb's own event data carries no explicit
    home/away role field this project's ingestion layer currently
    surfaces, so "alphabetical" is a labeling choice, not a real home/away
    claim.
    """
    teams = sorted(_teams_in_match(match_id))
    if len(teams) < 2:
        return {
            "match_id": match_id,
            "teams": teams,
            "no_data": True,
            "reason": (
                f"Fewer than 2 teams found in match_id={match_id}'s event data -- "
                "cannot compile a match report."
            ),
        }

    team_reports = {team: generate_team_report(team, [match_id]) for team in teams}
    opposition_analysis = {team: generate_team_opposition_analysis(team, [match_id]) for team in teams}
    pass_network = pass_network_fn(match_id)
    alerts = fetch_alerts(match_id=match_id, limit=500)

    return {
        "match_id": match_id,
        "teams": teams,
        "no_data": False,
        "team_reports": team_reports,
        "opposition_analysis": opposition_analysis,
        "pass_network": pass_network,
        "alerts": alerts,
        "alert_count": len(alerts),
    }


def _pass_network_summary(pass_network: dict) -> dict:
    """Normalizes `generate_pass_network`'s (raw) and
    `generate_pass_network_aggregated`'s (public-safe) two different
    return shapes into the SAME three summary numbers, for the narrative
    prompt only -- neither shape is modified, this just reads whichever
    fields are actually present. Returns `None` values (not zeros) when
    `pass_network["no_data"]` is True, so the prompt can state "not
    available" honestly rather than a fabricated zero.
    """
    if pass_network.get("no_data"):
        return {"num_players": None, "num_edges": None, "total_completed_passes": None}

    if "nodes" in pass_network:  # raw generate_pass_network shape
        edges = pass_network.get("edges", [])
        return {
            "num_players": len(pass_network.get("nodes", [])),
            "num_edges": len(edges),
            "total_completed_passes": sum(e["completed_passes"] for e in edges),
        }

    # aggregated generate_pass_network_aggregated shape
    return {
        "num_players": pass_network.get("num_players"),
        "num_edges": pass_network.get("num_edges"),
        "total_completed_passes": pass_network.get("total_completed_passes_in_network"),
    }


def build_match_report_narrative_prompt(compiled_report: dict) -> str:
    """Builds a prompt for a short, whole-match narrative summary,
    following the SAME discipline as `explainer.build_tactical_prompt` /
    `zone_explainer.build_zone_explanation_prompt` (system role framing,
    real computed inputs only, an explicit ask for a concise output) --
    but scoped to a WHOLE MATCH's compiled document rather than one
    frame's or one team's attribution.

    HONESTY CONSTRAINT: this prompt only ever states numbers already
    present in `compiled_report` (itself built purely from EXISTING,
    unmodified report functions in `generate_automatic_match_report`
    above) -- team names, per-team mean threat and weakest control zone
    (only if that side had real 360 coverage for this match), build-up
    long-pass share, set-piece shot share, pass-network totals, and the
    alert count/delta range. It explicitly instructs against any
    temporal/duration claim or specific player-movement claim, the same
    reinforcement `build_zone_explanation_prompt` already states inline as
    defense in depth -- the actual enforcement is the SAME
    `_HONESTY_SYSTEM_INSTRUCTION` every real-Gemini call in this project
    already goes through (`explainer.generate_tactical_explanation_with_source`),
    unmodified, not a new constraint invented here.

    Deliberately does NOT reuse `build_tactical_prompt`'s exact literal
    header strings ("Predicted threat:", "Top factors INCREASING...") --
    unlike `build_zone_explanation_prompt`, this prompt describes TWO
    teams' worth of genuinely different-shaped facts (zone/phase threat,
    build-up tendency, set-piece reliance, pass-network totals, alert
    history), which does not fit that single-scalar-plus-factor-list
    shape. This means `explainer.generate_explanation`'s regex-based mock
    executor will not parse this prompt meaningfully (it will report
    "threat is unknown%, no factors identified") when `GEMINI_API_KEY` is
    unset -- a degraded but still SAFE fallback (it never fabricates a
    claim; see `generate_explanation`'s own code), not a crash, so this is
    an accepted, explicitly-stated tradeoff rather than a silent gap.
    """
    match_id = compiled_report["match_id"]
    teams = compiled_report["teams"]

    def _team_facts(team: str) -> str:
        report = compiled_report["team_reports"].get(team, {})
        opposition = compiled_report["opposition_analysis"].get(team, {})

        lines = [f"Team: {team}"]
        if report.get("matches_used", 0) > 0:
            threat_by_zone = report.get("threat_by_pitch_zone") or {}
            known = {k: v for k, v in threat_by_zone.items() if v is not None}
            if known:
                zone_text = ", ".join(f"{zone}: {value * 100:.1f}%" for zone, value in known.items())
                lines.append(f"  Threat by pitch zone (mean predicted shot probability): {zone_text}")
            weak_zones = report.get("weakest_control_zones") or []
            if weak_zones:
                wz = weak_zones[0]
                lines.append(
                    f"  Weakest pitch-control zone: grid cell (col={wz['col']}, row={wz['row']}), "
                    f"mean control={wz['mean_control']:.3f}"
                )
        else:
            lines.append("  No 360 freeze-frame coverage for this match -- no zone/control data available.")

        build_up = opposition.get("build_up_tendency") or {}
        long_pass_share = build_up.get("long_pass_share")
        if long_pass_share is not None:
            lines.append(f"  Build-up long-pass share: {long_pass_share * 100:.1f}%")
        set_piece = opposition.get("set_piece_reliance") or {}
        set_piece_share = set_piece.get("set_piece_shot_share")
        if set_piece_share is not None:
            lines.append(f"  Set-piece shot share: {set_piece_share * 100:.1f}%")

        return "\n".join(lines)

    team_facts_text = "\n".join(_team_facts(team) for team in teams)

    pn_summary = _pass_network_summary(compiled_report.get("pass_network", {}))
    if pn_summary["num_players"] is not None:
        pass_network_text = (
            f"Pass network: {pn_summary['num_players']} players, {pn_summary['num_edges']} distinct "
            f"passing connections, {pn_summary['total_completed_passes']} total completed passes."
        )
    else:
        pass_network_text = "Pass network: no event data available for this match."

    alerts = compiled_report.get("alerts", [])
    if alerts:
        deltas = [a["delta"] for a in alerts]
        alerts_text = (
            f"Tactical alerts raised during this match: {len(alerts)}, with threat-delta magnitude "
            f"ranging from {min(deltas):+.3f} to {max(deltas):+.3f}."
        )
    else:
        alerts_text = "Tactical alerts raised during this match: 0."

    prompt = (
        "You are an expert football tactical analyst producing a short post-match report. Your job "
        "is to summarize the REAL, already-computed data below for this match. You explain "
        "already-computed data; you do not make your own prediction, and you MUST NOT state any "
        "temporal/duration claim (e.g. 'for N seconds', 'in the final minutes') or any specific "
        "player movement/recovery claim -- neither is derivable from this data.\n\n"
        f"Match ID: {match_id}\n"
        f"Teams: {', '.join(teams)}\n\n"
        f"{team_facts_text}\n\n"
        f"{pass_network_text}\n\n"
        f"{alerts_text}\n\n"
        "Provide a concise 3-4 sentence narrative summary of this match's tactical picture, "
        "referencing only the real facts given above."
    )
    return prompt
