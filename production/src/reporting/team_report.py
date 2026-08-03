"""Milestone 40 (new reporting track, Step 2): Historical Team Analysis
Report.

STANDALONE, additive: reads real StatsBomb event/360 data via the EXISTING
`statsbomb_io` fetch functions, and runs the EXISTING, UNMODIFIED
`BiomechanicalPitchControl` engine and the deterministically-selected
trained DeepHit MLP (`explainer.load_deterministic_mlp`,
`evaluation.predict_cumulative_incidence`) exactly as those were built and
validated for Milestones 1-15/24. Nothing in `production/src/physics`,
`spatial`, `models`, `pipeline`'s training/serving code, or `serving/` is
modified for this module -- this file only calls their existing public
functions. No CV/video dependency -- entirely independent of ADR-013
through ADR-016.
"""

import torch

from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.models.evaluation import predict_cumulative_incidence
from production.src.models.explainer import load_deterministic_mlp
from production.src.pipeline.chain_builder import build_possession_chains
from production.src.pipeline.feature_extractor import (
    FINAL_THIRD_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
    extract_features,
)
from production.src.pipeline.habit_memory import (
    CELL_HEIGHT_METERS,
    CELL_WIDTH_METERS,
    GRID_COLS,
    GRID_ROWS,
)
from production.src.spatial.control import BiomechanicalPitchControl

# Mirrors feature_extractor.FINAL_THIRD_X (66.0, an established, hand-tuned
# asymmetric threshold -- NOT exactly PITCH_LENGTH/3) for the defensive
# third boundary: PITCH_LENGTH - FINAL_THIRD_X, keeping the same convention
# reflected rather than inventing an independent 33.33 split.
DEFENSIVE_THIRD_X = PITCH_LENGTH - FINAL_THIRD_X

# 15-minute game-phase buckets -- a reasonable, common match-report
# granularity; not an established project convention to reuse (no prior
# milestone bucketed by game phase), so chosen fresh and stated plainly.
GAME_PHASE_BUCKET_MINUTES = 15.0

DEFAULT_THREAT_TIME_BIN = 3  # matches this project's established Brier@15s convention (BIN_SIZE_SECONDS=5.0 * 3 = 15s)


def _build_pitch_grid() -> torch.Tensor:
    """Full PITCH_LENGTH x PITCH_WIDTH grid (1m spacing), matching
    `feature_extractor._build_pitch_grid` exactly -- reimplemented locally
    (rather than importing that leading-underscore, module-private name
    across files) from the same public `PITCH_LENGTH`/`PITCH_WIDTH`
    constants."""
    xs = torch.arange(0, PITCH_LENGTH, dtype=torch.float32)
    ys = torch.arange(0, PITCH_WIDTH, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)


def _match_representative_chain_frames(match_id: int) -> list[tuple[dict, dict, float]]:
    """`(chain, parsed_frame, event_minute)` triples for every possession
    chain in `match_id` that has at least one 360-covered event -- the SAME
    representative-event-per-chain pattern `train.py`'s
    `_match_chains_with_features` uses (one 360 frame per chain, not every
    single individual event), reimplemented locally here rather than
    importing that leading-underscore, training-pipeline-private helper.
    Chains with no 360-covered event are skipped, same as that pattern.
    `event_minute` (`minute + second/60`, continuous across periods -- see
    `player_report._match_time_minutes`) is carried alongside the parsed
    frame since `parse_360_frame`'s own return dict does not expose it, and
    game-phase bucketing below needs it.
    """
    events = fetch_match_events(match_id)
    frames = fetch_match_360(match_id)
    if events is None or frames is None:
        return []

    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}
    chains = build_possession_chains(events, periods=(1, 2))

    events_by_period_possession: dict[tuple[int, int], list] = {}
    for e in events:
        if e["period"] in (1, 2):
            events_by_period_possession.setdefault((e["period"], e["possession"]), []).append(e)
    for group in events_by_period_possession.values():
        group.sort(key=lambda e: e["index"])

    triples = []
    for chain in chains:
        chain_events = events_by_period_possession.get((chain["period"], chain["chain_id"]), [])
        rep_event, rep_frame = None, None
        for e in chain_events:
            frame = frames_by_event_uuid.get(e["id"])
            if frame is not None and "location" in e:
                rep_event, rep_frame = e, frame
                break
        if rep_event is None:
            continue
        parsed = parse_360_frame(rep_event, rep_frame)
        event_minute = rep_event["minute"] + rep_event["second"] / 60.0
        triples.append((chain, parsed, event_minute))
    return triples


def _teams_in_match(match_id: int) -> set[str]:
    events = fetch_match_events(match_id)
    if events is None:
        return set()
    return {e["team"]["name"] for e in events if "team" in e}


def generate_team_report(team_name: str, match_ids: list[int]) -> dict:
    """Historical Team Analysis Report (Milestone 40, Step 2): an aggregate
    pitch-control weak-spot heatmap and an aggregate DeepHit threat pattern
    by pitch zone / game phase, both computed from real match data via the
    EXISTING, unmodified physics/ML stack.

    `match_ids`: StatsBomb match_ids to include. A match is skipped (with a
    printed note) if `team_name` did not play in it at all, or has no
    fetchable events/360 data.
    """
    engine = BiomechanicalPitchControl()
    pitch_grid = _build_pitch_grid()

    control_sum = [[0.0] * GRID_ROWS for _ in range(GRID_COLS)]
    control_count = [[0] * GRID_ROWS for _ in range(GRID_COLS)]

    threat_by_zone: dict[str, list] = {"defensive_third": [], "middle_third": [], "attacking_third": []}
    threat_by_phase: dict[str, list] = {}

    matches_used = 0
    model = normalization_mean = normalization_std = None  # lazy-loaded, only if a threat frame exists

    for match_id in match_ids:
        teams = _teams_in_match(match_id)
        if team_name not in teams:
            print(f"[team_report] match_id={match_id}: {team_name!r} did not play, skipping.")
            continue

        triples = _match_representative_chain_frames(match_id)
        if not triples:
            print(f"[team_report] match_id={match_id}: no 360-covered chains found, skipping.")
            continue
        matches_used += 1

        for chain, parsed, event_minute in triples:
            is_team_attacking = chain["team"] == team_name
            team_mask = parsed["is_teammate"] if is_team_attacking else ~parsed["is_teammate"]

            team_pos = parsed["player_pos"][team_mask]
            team_vel = parsed["player_vel"][team_mask]
            team_fatigue = parsed["fatigue_mod"][team_mask]

            if team_pos.shape[0] > 0:
                active_coords, control_probabilities, _ = engine(
                    team_pos, team_vel, team_fatigue, pitch_grid, parsed["ball_pos"]
                )
                team_control = control_probabilities.max(dim=0).values  # [N_active], same rule as feature_extractor._team_control

                cols = (active_coords[:, 0] // CELL_WIDTH_METERS).long().clamp(0, GRID_COLS - 1)
                rows = (active_coords[:, 1] // CELL_HEIGHT_METERS).long().clamp(0, GRID_ROWS - 1)
                for col, row, value in zip(cols.tolist(), rows.tolist(), team_control.tolist()):
                    control_sum[col][row] += value
                    control_count[col][row] += 1

            # Step 2.2: threat pattern is only meaningful for FRAMES WHERE
            # team_name IS THE ACTING/ATTACKING SIDE -- predicted
            # cumulative incidence models the acting team's own near-term
            # shot probability (Milestones 8/13); a defending-side frame's
            # prediction would represent the OPPONENT's threat, not
            # team_name's.
            if not is_team_attacking:
                continue

            if model is None:
                model, normalization_mean, normalization_std, run_id = load_deterministic_mlp()
                print(f"[team_report] using deterministic MLP run_id={run_id}")

            features = extract_features(parsed, engine)
            cumulative_incidence = predict_cumulative_incidence(
                model, features, normalization_mean, normalization_std, time_bin=DEFAULT_THREAT_TIME_BIN
            )

            ball_x = parsed["ball_pos"][0].item()
            if ball_x < DEFENSIVE_THIRD_X:
                zone = "defensive_third"
            elif ball_x > FINAL_THIRD_X:
                zone = "attacking_third"
            else:
                zone = "middle_third"
            threat_by_zone[zone].append(cumulative_incidence)

            phase_bucket = _phase_bucket_label(event_minute)
            threat_by_phase.setdefault(phase_bucket, []).append(cumulative_incidence)

    control_heatmap = [
        [
            (control_sum[col][row] / control_count[col][row]) if control_count[col][row] > 0 else None
            for row in range(GRID_ROWS)
        ]
        for col in range(GRID_COLS)
    ]

    weak_zones = sorted(
        (
            {"col": col, "row": row, "mean_control": control_heatmap[col][row]}
            for col in range(GRID_COLS)
            for row in range(GRID_ROWS)
            if control_heatmap[col][row] is not None
        ),
        key=lambda z: z["mean_control"],
    )[:5]

    threat_by_zone_mean = {
        zone: (sum(values) / len(values) if values else None) for zone, values in threat_by_zone.items()
    }
    # Sort by the bucket's actual numeric start minute, not the label
    # string -- alphabetical order would misplace any bucket whose start
    # minute has a different digit count (e.g. "105-120'" would sort
    # before "90-105'"), which matters once extra-time chains appear.
    threat_by_phase_mean = {
        phase: (sum(values) / len(values))
        for phase, values in sorted(threat_by_phase.items(), key=lambda kv: int(kv[0].split("-")[0]))
    }

    return {
        "team_name": team_name,
        "matches_requested": len(match_ids),
        "matches_used": matches_used,
        "control_heatmap_grid": control_heatmap,
        "control_heatmap_grid_shape": (
            f"{GRID_COLS} cols (x, {CELL_WIDTH_METERS}m/cell) x {GRID_ROWS} rows "
            f"(y, {CELL_HEIGHT_METERS:.2f}m/cell); None = no frame in this sample had an "
            "active (near-ball) cell there"
        ),
        "weakest_control_zones": weak_zones,
        "threat_by_pitch_zone": threat_by_zone_mean,
        "threat_by_game_phase": threat_by_phase_mean,
        "threat_time_bin_seconds": DEFAULT_THREAT_TIME_BIN * 5.0,
    }


def _phase_bucket_label(event_minute: float) -> str:
    """`GAME_PHASE_BUCKET_MINUTES`-wide (15-minute) game-phase bucket label
    from the representative event's continuous match-clock minute (see
    `_match_representative_chain_frames`'s docstring for why this is
    already continuous across periods, no half-time offset needed).
    Stoppage time beyond 90' (e.g. minute 93) falls into whichever bucket
    its raw minute lands in (e.g. "90-105'"), not clamped to "75-90'" --
    an open-ended top bucket would hide genuinely late chances, which
    matters for a report meant to surface tactical patterns honestly.
    """
    bucket_start = int(event_minute // GAME_PHASE_BUCKET_MINUTES) * int(GAME_PHASE_BUCKET_MINUTES)
    bucket_end = bucket_start + int(GAME_PHASE_BUCKET_MINUTES)
    return f"{bucket_start}-{bucket_end}'"
