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

import logging
import math
from collections import defaultdict

import torch

from production.src.ingestion.statsbomb_io import (
    X_SCALE,
    Y_SCALE,
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
    MIN_HISTORICAL_EVENTS,
)
from production.src.spatial.control import BiomechanicalPitchControl

logger = logging.getLogger(__name__)

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
            logger.info(f"match_id={match_id}: {team_name!r} did not play, skipping.")
            continue

        triples = _match_representative_chain_frames(match_id)
        if not triples:
            logger.info(f"match_id={match_id}: no 360-covered chains found, skipping.")
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
                logger.info(f"using deterministic MLP run_id={run_id}")

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


# ============================================================================
# Tactical Entropy (additive new feature): Shannon CONDITIONAL (bigram)
# entropy over a team's pass-DIRECTION transition sequence -- a real
# statistical measure of predictability/unpredictability in a team's own
# passing patterns. See `generate_team_pass_entropy`'s own docstring for
# the full Step 0 definitions (this is a genuine judgment call, same
# discipline as the shot map's "on-target" choice / Press Resistance
# Index's shot-success choice -- documented explicitly, not left implicit).
# ============================================================================

# Step 0.1 (pass "type" categorization): direction relative to the team's
# own attacking direction, in this project's 100x68m rescaled space
# (ADR-002) -- NOT raw StatsBomb 0-120 units, matching pass_network.py's
# own established rescale convention. VERIFIED against ADR-009 before
# choosing this approach (re-confirmed here, not assumed to still hold):
# StatsBomb event `location` is already recorded relative to the ACTING
# team's own attacking-left-to-right perspective, in BOTH halves -- there
# is no per-team/per-period direction inference to do here, unlike a raw
# shared-coordinate-frame source. `pipeline/direction.py` exists for THAT
# different case specifically and was explicitly retired from this
# project's own StatsBomb feature pipeline for this exact reason (see
# ADR-009) -- reusing it here would be actively WRONG (solving a problem
# this data doesn't have), not merely redundant, so it is deliberately
# NOT imported or called anywhere in this section.
#
# 5.0m was chosen as the Forward/Backward threshold after checking the
# real distribution of pass end_location-minus-location x-deltas across
# match 3857276's 955 real cached passes (mean +6.0 raw units [+5.0m],
# range -44.1 to +72.4 raw units): a 5.0m band produces a reasonably
# balanced 3-way split on that real match (433 Forward / 231 Backward /
# 291 Sideways at the equivalent raw threshold) rather than a degenerate
# near-all-one-category result. A reasonable, stated choice, not a
# provably optimal one.
PASS_DIRECTION_THRESHOLD_METERS = 5.0
PASS_TYPE_CATEGORIES = ("Forward", "Backward", "Sideways")

# Step 0.4 low-sample flag: reuses the SAME MIN_HISTORICAL_EVENTS (20)
# threshold value this project already established (habit_memory.py,
# player_report.MIN_SHOTS_FOR_CONFIDENT_SHOT_MAP,
# player_report.MIN_UNDER_PRESSURE_EVENTS_FOR_CONFIDENT_PRI), following
# the same naming precedent: a new, descriptively-named constant set
# equal to MIN_HISTORICAL_EVENTS, applied here to TOTAL TRANSITION COUNT
# (not raw pass count -- a transition needs two consecutive same-chain
# passes, so this is the more conservative, directly-relevant sample size
# for an entropy estimate specifically).
MIN_TRANSITIONS_FOR_CONFIDENT_PASS_ENTROPY = MIN_HISTORICAL_EVENTS


def _pass_direction_category(event: dict) -> str | None:
    """Categorizes one real Pass event's direction (Step 0.1 -- see this
    module's Tactical Entropy section header comment for the full
    reasoning and real-data threshold justification).

    Returns None if `location`/`pass.end_location` are absent -- verified
    100% present across 955 real cached passes in match 3857276 (both
    completed and incomplete), but not assumed to hold universally;
    skipped, not fabricated, if genuinely missing.
    """
    location = event.get("location")
    end_location = event.get("pass", {}).get("end_location")
    if location is None or end_location is None:
        return None
    delta_x_meters = (end_location[0] - location[0]) * X_SCALE
    if delta_x_meters > PASS_DIRECTION_THRESHOLD_METERS:
        return "Forward"
    if delta_x_meters < -PASS_DIRECTION_THRESHOLD_METERS:
        return "Backward"
    return "Sideways"


def _is_complete_pass_for_entropy(event: dict) -> bool:
    """Mirrors pass_network.py's `_is_complete_pass` EXACT "no
    `pass.outcome` key at all" completion signal (reused, not reinvented
    -- see that function's own real-data verification against match
    3857276's 955 cached passes). Reimplemented locally here, rather than
    imported across files, per this project's established convention
    against importing another module's own leading-underscore,
    module-private helper directly (see e.g. team_report.py's own
    `_build_pitch_grid` docstring, or player_report.py's Press Resistance
    Index section, for the same convention applied previously).

    Used here ONLY for the transparency fields
    (`completed_pass_attempts_considered` below) -- NOT to gate which
    passes enter the transition sequence (see `generate_team_pass_entropy`'s
    own docstring, Step 0.1, for why completion status deliberately does
    NOT filter inclusion here, unlike Pass Network's raw edges).
    """
    return "outcome" not in event.get("pass", {})


def _team_pass_category_sequences(match_id: int, team_name: str) -> list[list[tuple[str, bool]]]:
    """One `(category, is_complete)` list per real StatsBomb possession
    chain where `team_name` had the ball, in event order -- the unit
    `generate_team_pass_entropy` computes transitions within (Step 0.3).

    Chain boundaries: StatsBomb's own `possession` field, grouped by
    `(period, possession)` -- the SAME grouping key `chain_builder.py`'s
    `build_possession_chains` establishes. `chain_builder.py` exposes only
    each chain's METADATA (team, timing, censor reason), not its actual
    event list, so this function regroups the raw events by
    `(period, possession)` itself, the SAME pattern this module's own
    `_match_representative_chain_frames` already uses for exactly the
    same reason (not a new third regrouping convention).
    `build_possession_chains` IS called here, to get each chain's
    authoritative team attribution (`chain["team"]`) from chain_builder's
    own classification, rather than re-deriving "whose chain is this"
    independently from `possession_team`.

    A transition is NEVER counted across a chain boundary (Step 0.3): a
    turnover-and-regain is not a meaningful tactical sequence for this
    team's own passing shape. Each chain's category list is kept separate
    here for exactly that reason -- `generate_team_pass_entropy` only
    counts a transition between two categories adjacent WITHIN one of
    these per-chain lists, never between the last category of one chain
    and the first category of another (even a different chain awarded to
    the SAME team later, e.g. after a brief opposition touch).
    """
    events = fetch_match_events(match_id)
    if events is None:
        return []

    chains = build_possession_chains(events, periods=(1, 2))
    team_chain_keys = {
        (chain["period"], chain["chain_id"]) for chain in chains if chain["team"] == team_name
    }
    if not team_chain_keys:
        return []

    events_by_chain: dict[tuple[int, int], list] = {}
    for e in events:
        if e["period"] not in (1, 2):
            continue
        key = (e["period"], e["possession"])
        if key in team_chain_keys:
            events_by_chain.setdefault(key, []).append(e)
    for group in events_by_chain.values():
        group.sort(key=lambda e: e["index"])

    sequences = []
    for chain_events in events_by_chain.values():
        sequence = []
        for e in chain_events:
            if e["type"]["name"] != "Pass":
                continue
            if e.get("team", {}).get("name") != team_name:
                continue
            category = _pass_direction_category(e)
            if category is None:
                continue
            sequence.append((category, _is_complete_pass_for_entropy(e)))
        sequences.append(sequence)
    return sequences


def generate_team_pass_entropy(team_name: str, match_ids: list[int]) -> dict:
    """Tactical Entropy (additive new feature): Shannon CONDITIONAL
    (bigram) entropy over `team_name`'s pass-DIRECTION transition
    sequence, aggregated across `match_ids` -- H(type_t+1 | type_t),
    averaged over all observed type_t, weighted by how often each type_t
    itself occurs as a "from" state.

    STEP 0 DEFINITIONS (a genuine judgment call, documented explicitly --
    same discipline the shot map's "on-target" choice and Press
    Resistance Index's shot-success choice already established):

    1. PASS TYPE: direction (Forward/Backward/Sideways) relative to the
       team's own attacking direction, via a `PASS_DIRECTION_THRESHOLD_METERS`
       (5.0m) band on the end_location-minus-location x delta, in this
       project's 100x68m rescaled space. See this module's Tactical
       Entropy section header comment for the full real-data verification
       (including why `pipeline/direction.py` is deliberately NOT used --
       ADR-009 found StatsBomb data needs no per-team/period direction
       inference at all, so that module would be solving the wrong
       problem here, not merely a redundant one).
    2. CONDITIONAL, NOT UNIGRAM, ENTROPY: this function measures the
       unpredictability of what pass type FOLLOWS a given pass type --
       H(type_t+1 | type_t) = -sum_j P(j|i) * log2(P(j|i)), averaged over
       i weighted by P(i) -- NOT the plain diversity of pass types on
       their own (which would ignore order entirely). This is the correct
       reading of "transition sequence" per this feature's own roadmap
       framing: a team could show high UNIGRAM diversity (uses all three
       pass types often) while still being perfectly predictable in
       SEQUENCE (e.g. always Forward immediately after Backward) -- a
       unigram measure would miss that entirely; conditional entropy is
       built specifically to capture it.
    3. SEQUENCE SCOPE: passes are ordered using `chain_builder.py`'s own
       possession-chain grouping (`(period, possession)`, via
       `_team_pass_category_sequences` above), and a transition is NEVER
       counted across a chain boundary -- a turnover-and-regain is not a
       meaningful tactical sequence for this team's own passing shape.
       Completion status (reusing pass_network.py's exact `_is_complete_pass`
       signal, mirrored as `_is_complete_pass_for_entropy` above) does
       NOT gate which passes enter the sequence: an attempted pass's
       TYPE (what the team tried) is a real tactical choice whether or
       not it succeeded, and excluding failed attempts would silently
       bias this toward only a team's successful shape, hiding a genuine
       part of "predictability" (e.g. a team that predictably keeps
       trying the same risky pass type even when it often fails).
       `completed_pass_attempts_considered` below reports the completion
       breakdown for transparency instead of using it as a filter.
    4. ZERO-PROBABILITY HANDLING: standard Shannon convention, 0*log2(0)
       treated as 0 -- implemented by skipping any (from, to) pair with
       zero observed count when summing entropy terms (so log2(0) is
       never literally evaluated), and skipping any `type_t` never
       observed as a "from" state when computing the weighted average
       (zero weight, not a NaN/divide-by-zero term). A team with a
       narrow real repertoire (e.g. only ever passes Forward) correctly
       yields an entropy of exactly 0.0 (perfectly predictable), not an
       error -- see test_team_pass_entropy.py's dedicated real-data test
       for this exact case.

    Returns the transition COUNT matrix (so a viewer can judge sample
    size), the row-normalized transition PROBABILITY matrix, the overall
    conditional entropy in bits, a `normalized_entropy` (entropy /
    log2(3), i.e. rescaled to 0=fully predictable..1=uniformly random
    over 3 categories, for easier at-a-glance reading), and
    `pass_entropy_used_low_sample_flag` (same low-sample-flagging
    discipline as every other aggregate feature in this project -- see
    `MIN_TRANSITIONS_FOR_CONFIDENT_PASS_ENTROPY` above).
    """
    transition_counts: dict[str, dict[str, int]] = {
        a: {b: 0 for b in PASS_TYPE_CATEGORIES} for a in PASS_TYPE_CATEGORIES
    }
    total_pass_attempts_considered = 0
    completed_pass_attempts_considered = 0
    matches_used = 0

    for match_id in match_ids:
        teams = _teams_in_match(match_id)
        if team_name not in teams:
            logger.info(f"match_id={match_id}: {team_name!r} did not play, skipping.")
            continue

        sequences = _team_pass_category_sequences(match_id, team_name)
        if not sequences:
            logger.info(f"match_id={match_id}: no possession chains found for {team_name!r}, skipping.")
            continue
        matches_used += 1

        for sequence in sequences:
            total_pass_attempts_considered += len(sequence)
            completed_pass_attempts_considered += sum(1 for _, is_complete in sequence if is_complete)
            categories_only = [category for category, _ in sequence]
            for i in range(len(categories_only) - 1):
                transition_counts[categories_only[i]][categories_only[i + 1]] += 1

    total_transitions = sum(sum(row.values()) for row in transition_counts.values())

    transition_probabilities: dict[str, dict[str, float | None]] = {}
    for a in PASS_TYPE_CATEGORIES:
        row_total = sum(transition_counts[a].values())
        if row_total == 0:
            transition_probabilities[a] = {b: None for b in PASS_TYPE_CATEGORIES}
        else:
            transition_probabilities[a] = {
                b: transition_counts[a][b] / row_total for b in PASS_TYPE_CATEGORIES
            }

    conditional_entropy_bits = None
    if total_transitions > 0:
        conditional_entropy_bits = 0.0
        for a in PASS_TYPE_CATEGORIES:
            row_total = sum(transition_counts[a].values())
            if row_total == 0:
                continue  # Step 0.4: type_t never observed -- zero weight, not a NaN term
            p_from = row_total / total_transitions
            row_entropy = 0.0
            for b in PASS_TYPE_CATEGORIES:
                count = transition_counts[a][b]
                if count == 0:
                    continue  # Step 0.4: 0 * log2(0) treated as 0, log2(0) never evaluated
                p = count / row_total
                row_entropy -= p * math.log2(p)
            conditional_entropy_bits += p_from * row_entropy

    max_possible_entropy_bits = math.log2(len(PASS_TYPE_CATEGORIES))
    normalized_entropy = (
        conditional_entropy_bits / max_possible_entropy_bits if conditional_entropy_bits is not None else None
    )

    return {
        "team_name": team_name,
        "matches_requested": len(match_ids),
        "matches_used": matches_used,
        "pass_type_categories": list(PASS_TYPE_CATEGORIES),
        "pass_direction_threshold_meters": PASS_DIRECTION_THRESHOLD_METERS,
        "transition_counts": transition_counts,
        "transition_probabilities": transition_probabilities,
        "total_transitions": total_transitions,
        "total_pass_attempts_considered": total_pass_attempts_considered,
        "completed_pass_attempts_considered": completed_pass_attempts_considered,
        "conditional_entropy_bits": conditional_entropy_bits,
        "max_possible_entropy_bits": max_possible_entropy_bits,
        "normalized_entropy": normalized_entropy,
        "pass_entropy_used_low_sample_flag": (
            total_transitions < MIN_TRANSITIONS_FOR_CONFIDENT_PASS_ENTROPY
        ),
    }


# ============================================================================
# Passing Lane Visualizer (additive new feature): the SAME
# BiomechanicalPitchControl pitch-control field this module's own
# control_heatmap_grid already computes, sampled specifically along
# candidate PASSING LANES (the straight-line segment between a real
# passer and a real recipient) rather than binned into a full-pitch grid.
#
# OPPORTUNITY, NOT HISTORY -- explicitly distinct from pass_network.py's
# edges: Pass Network counts whether/how often a pass was actually
# COMPLETED (a real outcome). This feature instead scores how CONTESTED
# the space along a given real pass's trajectory was, independent of
# whether that pass succeeded -- a genuinely different signal (e.g. a
# skilled pass through heavy traffic that still succeeds scores LOW
# openness despite a good outcome; an intercepted pass through a
# genuinely open lane -- rare, but real -- would score HIGH openness
# despite a bad outcome).
#
# A REAL, LOAD-BEARING DATA-STRUCTURE FINDING THAT SHAPES THIS FEATURE'S
# ACTUAL ACHIEVABLE SCOPE (checked directly before building anything, not
# assumed): `parse_360_frame`'s own docstring already establishes that a
# 360 freeze-frame entry carries NO player id/name for anyone except the
# event's own ACTING player -- "there is NO way to know who any of the
# other ~21 visible players are" (verified against 21,273 real
# freeze-frames, Milestone 22). This means a SPECIFIC NAMED teammate
# pair's positions can NEVER both be read directly from one freeze-frame
# (only the acting player -- the passer -- has a known identity in that
# frame). The roadmap's own suggested framing ("mean lane-openness
# between each pair of teammates who actually played together, across
# ALL FRAMES where both were on the pitch") is therefore NOT achievable
# from this data at all -- it would require identifying a specific named
# teammate's own anonymous dot in an arbitrary frame, which nothing in
# this codebase (and no field in this data) supports.
#
# What IS achievable, and what this feature actually builds instead: for
# each real Pass EVENT (the ball's actual attempted trajectory, from a
# NAMED passer via `event.player` to a NAMED recipient via
# `event.pass.recipient` -- both real, known identities straight from the
# event's own metadata, no freeze-frame identity-matching needed), sample
# pitch control along that event's own real straight-line trajectory
# (`event.location` -> `event.pass.end_location`), using the SAME
# freeze-frame's DEFENDING team positions -- which need NO individual
# identity at all, only the `is_teammate` flag `parse_360_frame` already
# exposes. Aggregated per NAMED (passer, recipient) pair across every
# usable real pass in the given match_ids. This is anchored to passes
# that were actually ATTEMPTED (a hybrid of opportunity -- the lane's
# real geometric openness, independent of outcome -- and history -- only
# pairs a pass was actually tried between get sampled at all), not a
# survey of every conceivable teammate pair at every frame; stated
# plainly here as a genuine, real scope limitation, not implied to be
# broader than it is.
# ============================================================================

# Step 0.1: 11 evenly-spaced points (inclusive of both ends) along a
# pass's own real straight-line trajectory -- a reasonable, explicit
# choice (not proven-optimal): real median pass distance in a 360-covered
# sample match (3773386) was 12.0m, so 11 points gives ~1.2m spacing at
# the median, comparable to a player's own body/positioning scale.
LANE_SAMPLE_POINTS = 11

# BiomechanicalPitchControl only evaluates control within `mask_radius`
# (30.0m default) of the BALL -- verified directly (control.py's own
# sparse-masking, ADR-005) before relying on it, not assumed to cover an
# arbitrary lane end to end. VERIFIED against real data before deciding
# how to handle this: 92.9% of match 3773386's 1,060 real passes (with a
# named recipient) were fully within 30m of their own passer end to end,
# so most real lanes are ENTIRELY covered; requiring a STRICT MAJORITY
# (> half) of LANE_SAMPLE_POINTS to remain in-mask handles the remaining
# ~7% (long diagonal switches) by using only the ball-proximal portion of
# the lane, rather than fabricating a value for the unmeasured far
# portion or silently discarding every long pass outright.
LANE_MIN_COVERED_SAMPLES = 6

# Step 1 low-sample threshold: reuses MIN_HISTORICAL_EVENTS (20) directly
# (not recalibrated, unlike Session/Match Comparison's threshold) --
# VERIFIED this is a MEANINGFUL bar here, not a vacuous one: real
# per-(passer,recipient)-pair attempt counts in ONE 360-covered match
# (3773386) ranged min=1, median=4, max=22 across 150 distinct real
# pairs -- 20 real, multi-match-aggregated attempts is a genuine,
# discriminating confidence bar for a SPECIFIC pair, unlike match-level
# comparison's own located-event counts (which never dipped below 896 in
# this project's whole cache, making 20 meaningless there).
MIN_PASS_SAMPLES_FOR_CONFIDENT_LANE_OPENNESS = MIN_HISTORICAL_EVENTS


def _lane_sample_points(start: torch.Tensor, end: torch.Tensor, n: int) -> torch.Tensor:
    """`n` evenly-spaced points (inclusive of both ends) along the
    straight-line segment from `start` to `end`, in the SAME 100x68m
    rescaled space `parse_360_frame`'s own `player_pos`/`ball_pos`
    already use. Passed directly as the `pitch_grid` argument to
    `BiomechanicalPitchControl` -- REUSING that engine's own existing,
    general-purpose query-point interface exactly as designed (it
    already accepts an arbitrary `[N_total, 2]` tensor of query points,
    not hardcoded to the dense 100x68 grid `_build_pitch_grid` happens to
    construct for the full heatmap above) -- not a new sampling
    convention invented for this feature.
    """
    t = torch.linspace(0.0, 1.0, n).unsqueeze(1)  # [n, 1]
    return start.unsqueeze(0) * (1.0 - t) + end.unsqueeze(0) * t  # [n, 2]


def _lane_openness_for_pass(event: dict, frame_data: dict, engine: BiomechanicalPitchControl) -> float | None:
    """Real lane-openness score for ONE real Pass event with a matched
    360 frame: `1.0 - max(defending-team control along the sampled
    segment)`. Two DISTINCT reductions, deliberately not conflated:

    1. Per sample point, `control_probabilities.max(dim=0).values` --
       the SAME per-cell reduction `generate_team_report`'s own
       `control_heatmap_grid` above already uses (the highest INDIVIDUAL
       defending player's control probability at that point) -- reused
       exactly, not reinvented.
    2. Across the lane's own sample points, `.max()` -- Step 0.1's own
       NEW choice for THIS feature specifically: a single point of high
       defensive control anywhere along the lane can kill it even if the
       rest is open, so MAX (not mean) is the conservative, realistic
       reduction for "was this lane genuinely open," matching the
       roadmap's own stated reasoning for preferring max here.

    Returns `None` (skip, don't fabricate) if `location`/`end_location`
    are missing, no defending players are visible in this freeze-frame,
    or fewer than `LANE_MIN_COVERED_SAMPLES` of the lane's own sample
    points fall within the engine's `mask_radius` of the ball (Step 0.1).
    """
    location = event.get("location")
    end_location = event.get("pass", {}).get("end_location")
    if location is None or end_location is None:
        return None

    parsed = parse_360_frame(event, frame_data)
    defending_mask = ~parsed["is_teammate"]
    defending_pos = parsed["player_pos"][defending_mask]
    defending_vel = parsed["player_vel"][defending_mask]
    defending_fatigue = parsed["fatigue_mod"][defending_mask]
    if defending_pos.shape[0] == 0:
        return None

    start = torch.tensor([location[0] * X_SCALE, location[1] * Y_SCALE], dtype=torch.float32)
    end = torch.tensor([end_location[0] * X_SCALE, end_location[1] * Y_SCALE], dtype=torch.float32)
    lane_points = _lane_sample_points(start, end, LANE_SAMPLE_POINTS)

    _, control_probabilities, _ = engine(defending_pos, defending_vel, defending_fatigue, lane_points, parsed["ball_pos"])
    if control_probabilities.shape[1] < LANE_MIN_COVERED_SAMPLES:
        return None

    defending_control_per_sample = control_probabilities.max(dim=0).values
    return 1.0 - defending_control_per_sample.max().item()


def _team_passing_lane_samples(match_id: int, team_name: str, engine: BiomechanicalPitchControl) -> list[dict]:
    """Real per-pass lane-openness samples for `team_name`'s own passes
    in ONE match -- one dict per usable real pass event (a NAMED
    recipient AND a matched 360 frame; VERIFIED against match 3773386:
    985 of 1,118 real Pass events, 88%, cleared both). NOT filtered by
    completion (see this section's own header comment, Step 0.3) --
    reuses the exact same "an attempted pass's own trajectory is real
    information regardless of outcome" reasoning `generate_team_pass_entropy`
    already established for a different purpose.
    """
    events = fetch_match_events(match_id)
    frames = fetch_match_360(match_id)
    if events is None or frames is None:
        return []
    frames_by_uuid = {f["event_uuid"]: f for f in frames}

    samples = []
    for event in events:
        if event.get("type", {}).get("name") != "Pass":
            continue
        if event.get("team", {}).get("name") != team_name:
            continue
        passer = event.get("player")
        recipient = event.get("pass", {}).get("recipient")
        if passer is None or passer.get("id") is None or recipient is None or recipient.get("id") is None:
            continue
        frame_data = frames_by_uuid.get(event["id"])
        if frame_data is None:
            continue

        openness = _lane_openness_for_pass(event, frame_data, engine)
        if openness is None:
            continue

        location, end_location = event["location"], event["pass"]["end_location"]
        samples.append({
            "passer_id": passer["id"],
            "passer_name": passer["name"],
            "recipient_id": recipient["id"],
            "recipient_name": recipient["name"],
            "passer_location": [location[0] * X_SCALE, location[1] * Y_SCALE],
            "recipient_location": [end_location[0] * X_SCALE, end_location[1] * Y_SCALE],
            "lane_openness": openness,
        })
    return samples


def generate_team_passing_lanes(team_name: str, match_ids: list[int]) -> dict:
    """RAW Passing Lane data for `team_name` -- LOCAL/PRIVATE USE ONLY
    (ADR-021 condition 2: see this feature's own addendum;
    `generate_team_passing_lanes_aggregated` below is the public-safe
    counterpart). Real per-player average location (`nodes`) PLUS real
    per-(passer, recipient)-pair mean lane-openness (`lanes`),
    individually attributable to named players -- structurally the SAME
    risk class `pass_network.py`'s raw edges already established (a
    named pair + a precise average location + a real score), gated the
    exact same way, NOT the Session/Match Comparison precedent (which
    has no location anywhere in its own output at all).

    `matches_used` counts a match toward the total only if `team_name`
    played in it AND at least one usable real pass sample was found
    there (360-covered, named recipient) -- matches
    `generate_team_report`'s own `matches_used` semantics.
    """
    engine = BiomechanicalPitchControl()

    all_samples: list[dict] = []
    matches_used = 0
    for match_id in match_ids:
        teams = _teams_in_match(match_id)
        if team_name not in teams:
            logger.info(f"match_id={match_id}: {team_name!r} did not play, skipping.")
            continue
        samples = _team_passing_lane_samples(match_id, team_name, engine)
        if samples:
            matches_used += 1
        all_samples.extend(samples)

    pair_openness: dict[tuple[int, int], list[float]] = defaultdict(list)
    pair_names: dict[tuple[int, int], tuple[str, str]] = {}
    location_sum: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    location_count: dict[int, int] = defaultdict(int)
    player_names: dict[int, str] = {}

    for s in all_samples:
        key = (s["passer_id"], s["recipient_id"])
        pair_openness[key].append(s["lane_openness"])
        pair_names[key] = (s["passer_name"], s["recipient_name"])
        for role_id, role_name, role_loc in (
            (s["passer_id"], s["passer_name"], s["passer_location"]),
            (s["recipient_id"], s["recipient_name"], s["recipient_location"]),
        ):
            location_sum[role_id][0] += role_loc[0]
            location_sum[role_id][1] += role_loc[1]
            location_count[role_id] += 1
            player_names[role_id] = role_name

    nodes = [
        {
            "player_id": player_id,
            "name": player_names[player_id],
            "avg_location": [
                location_sum[player_id][0] / location_count[player_id],
                location_sum[player_id][1] / location_count[player_id],
            ],
        }
        for player_id in location_count
    ]

    lanes = []
    for (passer_id, recipient_id), values in pair_openness.items():
        passer_name, recipient_name = pair_names[(passer_id, recipient_id)]
        n = len(values)
        lanes.append({
            "passer_id": passer_id,
            "passer_name": passer_name,
            "recipient_id": recipient_id,
            "recipient_name": recipient_name,
            "mean_lane_openness": sum(values) / n,
            "n_pass_samples": n,
            "passing_lane_used_low_sample_flag": n < MIN_PASS_SAMPLES_FOR_CONFIDENT_LANE_OPENNESS,
        })
    lanes.sort(key=lambda lane: -lane["mean_lane_openness"])

    return {
        "team_name": team_name,
        "matches_requested": len(match_ids),
        "matches_used": matches_used,
        "total_pass_samples_used": len(all_samples),
        "nodes": nodes,
        "lanes": lanes,
    }


def generate_team_passing_lanes_aggregated(team_name: str, match_ids: list[int]) -> dict:
    """ADR-021 condition-2-compliant variant of `generate_team_passing_lanes`,
    for PUBLIC deployments -- mirrors `pass_network.generate_pass_network_aggregated`'s
    own raw-pop-then-summarize pattern exactly (reuses `generate_team_passing_lanes`
    internally rather than re-implementing its fetch/sampling logic; the
    raw `nodes` list -- the ONLY field carrying a player's precise
    average LOCATION -- is a LOCAL variable of this function only, popped
    via `dict.pop`, never left on the returned dict).

    `lanes` (named pairs + mean openness + real sample counts) is kept
    UNCHANGED from the raw variant: per this feature's own ADR-021
    addendum, a named pair with a scalar score and no location is its own
    genuinely different, lower-risk shape from `nodes` -- the same
    reasoning that already keeps `pass_network_aggregated`'s per-player
    names in its `player_summary` while stripping only location/edges.
    """
    full = generate_team_passing_lanes(team_name, match_ids)
    full.pop("nodes")
    return full


# ============================================================================
# Opposition Analysis (additive new feature): 3 SPECIFIC, computable
# opposition-scouting metrics for `team_name` -- deliberately NOT a vague
# "insights" dump (Step 0's own explicit scoping discipline). Each
# answers a genuinely different real scouting question:
#   1. Weak-zone pitch control (where is this team spatially weak
#      defensively) -- REUSED, NOT RECOMPUTED, from generate_team_report's
#      EXISTING `weakest_control_zones` field. Deliberately NOT
#      re-derived here: dashboard.py's Opposition Analysis panel reads
#      the ALREADY-FETCHED team_report_dict the SAME tab's own pitch-
#      control panel already requested, and re-presents that one field
#      under an "opposition scouting: where to attack this team" label,
#      at the PRESENTATION layer, with ZERO additional backend
#      computation -- no wrapper function for this metric exists in this
#      module at all, since one would either recompute (forbidden) or
#      just re-export a field callers can already read directly.
#   2. Build-up length tendency (`generate_team_opposition_analysis`
#      below) -- does this team build up short/patient or long/direct;
#      shapes whether a high press or a compact mid-block is the better
#      counter.
#   3. Set-piece shot reliance (`generate_team_opposition_analysis`
#      below) -- how much of this team's shot volume comes from
#      manufactured dead-ball situations vs open play; shapes whether
#      restart marking discipline matters more than open-play defending
#      against them.
# A fourth or fifth metric (counter-attack frequency, high-press
# resistance, ...) would be a real, defensible addition but was
# deliberately NOT built here -- 3 specific, justified metrics, not an
# open-ended feature dump, per Step 0's own explicit instruction.
# ============================================================================

# Step 0.2's length threshold, scoped to defensive/middle-third passes
# only (build-up phase, not attacking-third passes which are naturally
# shorter/more incisive and answer a different question). VERIFIED
# against real data before choosing 25.0m: match 3773386's 532 real
# Barcelona build-up passes (defensive + middle third) had median 13.2m,
# p75 18.8m -- 25.0m sits comfortably above the bulk of the real
# distribution, isolating a genuinely distinct minority (9.8% of this
# real sample) as "long," rather than a threshold so loose it captures a
# fifth of all passes (20m -> 21.1%) or so strict it captures almost
# none (30m -> 4.7%). A reasonable, stated choice, not a provably
# optimal one -- same discipline as every prior length/distance judgment
# call this project has made explicit (Press Resistance Index's shot
# definition, Tactical Entropy's direction threshold).
BUILDUP_LONG_PASS_THRESHOLD_METERS = 25.0

# Step 0.3's set-piece definition. VERIFIED real StatsBomb `play_pattern`
# values against match 3773386 before choosing which ones count (9
# distinct real values observed: "Regular Play", "From Throw In", "From
# Free Kick", "From Corner", "From Kick Off", "From Goal Kick", "From
# Keeper", "Other", "From Counter"). Deliberately scoped to
# {"From Corner", "From Free Kick"} only -- the two patterns real
# football scouting conventionally means by "set piece" (a manufactured,
# rehearsed attacking situation) -- NOT "From Throw In"/"From Kick Off"/
# "From Goal Kick"/"From Keeper", which are also technically dead-ball
# restarts but are not what "set-piece reliance" colloquially asks about
# in a scouting context (a throw-in is not a rehearsed goal-threat
# situation the way a corner or free kick routine is).
SET_PIECE_PLAY_PATTERNS = frozenset({"From Corner", "From Free Kick"})

MIN_BUILDUP_PASSES_FOR_CONFIDENT_LENGTH_TENDENCY = MIN_HISTORICAL_EVENTS
MIN_SHOTS_FOR_CONFIDENT_SET_PIECE_RELIANCE = MIN_HISTORICAL_EVENTS


def generate_team_opposition_analysis(team_name: str, match_ids: list[int]) -> dict:
    """Opposition Analysis (additive new feature): `team_name`'s
    build-up-length tendency and set-piece shot reliance, aggregated
    across `match_ids` -- see this section's own header comment for the
    3-metric scoping (this function computes metrics 2 and 3; metric 1,
    the pitch-control weak zones, is reused directly from
    `generate_team_report`'s existing output, not recomputed here).

    Event-data only -- no 360 freeze-frame coverage needed (unlike
    Passing Lanes above), same class as Tactical Entropy/Press Resistance
    Index.

    `build_up_tendency`: `long_pass_share` among this team's own
    defensive-and-middle-third passes only (Step 0.2) -- the attacking
    third is deliberately excluded, since final-third passes answer a
    different question (incisiveness, not build-up shape).

    `set_piece_reliance`: `set_piece_shot_share` -- what fraction of this
    team's real shots (any outcome, not just goals) originated from a
    real `play_pattern` in `SET_PIECE_PLAY_PATTERNS` (Step 0.3).

    Both sub-metrics carry their OWN low-sample flag (reusing
    MIN_HISTORICAL_EVENTS, same convention as every prior feature this
    session), since they are independent real-data counts that can be
    thin even when the other one isn't (e.g. a low-shot match can still
    have plenty of real build-up passes).
    """
    total_buildup_passes = 0
    long_buildup_passes = 0
    total_shots = 0
    set_piece_shots = 0
    matches_used = 0

    for match_id in match_ids:
        teams = _teams_in_match(match_id)
        if team_name not in teams:
            logger.info(f"match_id={match_id}: {team_name!r} did not play, skipping.")
            continue
        events = fetch_match_events(match_id)
        if events is None:
            continue
        matches_used += 1

        for event in events:
            if event.get("team", {}).get("name") != team_name:
                continue
            type_name = event.get("type", {}).get("name")

            if type_name == "Pass":
                location = event.get("location")
                end_location = event.get("pass", {}).get("end_location")
                if location is None or end_location is None:
                    continue
                if location[0] * X_SCALE >= FINAL_THIRD_X:
                    continue  # attacking-third pass -- not build-up (Step 0.2)
                dx = (end_location[0] - location[0]) * X_SCALE
                dy = (end_location[1] - location[1]) * Y_SCALE
                distance = (dx * dx + dy * dy) ** 0.5
                total_buildup_passes += 1
                if distance > BUILDUP_LONG_PASS_THRESHOLD_METERS:
                    long_buildup_passes += 1

            elif type_name == "Shot":
                total_shots += 1
                play_pattern = event.get("play_pattern", {}).get("name")
                if play_pattern in SET_PIECE_PLAY_PATTERNS:
                    set_piece_shots += 1

    return {
        "team_name": team_name,
        "matches_requested": len(match_ids),
        "matches_used": matches_used,
        "build_up_tendency": {
            "total_buildup_passes": total_buildup_passes,
            "long_passes": long_buildup_passes,
            "long_pass_share": (
                long_buildup_passes / total_buildup_passes if total_buildup_passes > 0 else None
            ),
            "long_pass_threshold_meters": BUILDUP_LONG_PASS_THRESHOLD_METERS,
            "build_up_tendency_used_low_sample_flag": (
                total_buildup_passes < MIN_BUILDUP_PASSES_FOR_CONFIDENT_LENGTH_TENDENCY
            ),
        },
        "set_piece_reliance": {
            "total_shots": total_shots,
            "set_piece_shots": set_piece_shots,
            "set_piece_shot_share": (set_piece_shots / total_shots) if total_shots > 0 else None,
            "set_piece_play_patterns": sorted(SET_PIECE_PLAY_PATTERNS),
            "set_piece_reliance_used_low_sample_flag": (
                total_shots < MIN_SHOTS_FOR_CONFIDENT_SET_PIECE_RELIANCE
            ),
        },
    }
