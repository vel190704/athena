"""Milestone 40 (new reporting track, Step 3): zone-level (pitch-control
grid-cell) attribution -- extends Milestone 15's Integrated-Gradients
explainer from attributing across 4 SCALAR features to attributing across
PITCH-CONTROL GRID CELLS, enabling statements like "the right half-space
is systematically open" rather than only "space_behind_defending_line
mattered."

WHY THIS IS TRACTABLE WITHOUT RETRAINING OR MODIFYING THE EXISTING MODEL:
`feature_extractor.extract_features`'s 4 scalar features are themselves
just MASKED SUMS over the per-cell control-probability tensors
`BiomechanicalPitchControl` already produces (`attacking_control_near_ball
= att_control[near_ball_mask].sum()`, etc.) -- a differentiable
computation, in principle, all the way from grid cells to the MLP's
output. The one thing that breaks it in the ORIGINAL code is
`extract_features` calling `.item()` at the end, which detaches the
result from autograd. `_zone_features_from_grid` below reimplements the
EXACT SAME four formulas (verified against `feature_extractor.py`'s
source, not paraphrased) as a pure, differentiable function of two new
LEAF tensors (per-cell attacking/defending control, `requires_grad=True`)
instead of calling the original function -- nothing in `feature_extractor.
py`, `control.py`, or the trained MLP is modified; this module only adds
a differentiable REPLAY of the aggregation step for one already-computed
frame, so Captum can attribute back through it.

SCOPE, STATED PLAINLY: this attributes the MLP's cumulative-incidence
output back to per-cell CONTROL VALUES for one frame at a time (the same
frame's own `active_grid_coords`, which come from `BiomechanicalPitchControl`'s
existing sparse ball-distance mask -- ADR-005). It does NOT (yet) attribute
further back through the physics ODE to player positions/velocities, and
the resulting attribution map's cell set is frame-specific (the ~2,000-
2,800 cells within `mask_radius` of THAT frame's ball position), not a
fixed grid comparable cell-for-cell across different frames without
re-binning (see `zone_attributions_to_grid` for that step).
"""

import logging

import numpy as np
import torch
from captum.attr import IntegratedGradients
from scipy.ndimage import label

from production.src.pipeline.feature_extractor import (
    FINAL_THIRD_X,
    NEAR_BALL_RADIUS,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from production.src.pipeline.habit_memory import (
    CELL_HEIGHT_METERS,
    CELL_WIDTH_METERS,
    GRID_COLS,
    GRID_ROWS,
)
from production.src.spatial.control import BiomechanicalPitchControl

logger = logging.getLogger(__name__)


def _active_grid_and_controls(
    engine: BiomechanicalPitchControl,
    parsed_frame: dict,
    pitch_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the EXISTING, unmodified engine once per side (mirrors
    `feature_extractor._team_control`'s own two-call pattern exactly), and
    returns `(active_coords, attacking_control, defending_control,
    highest_defending_x)` -- all still attached to the ORIGINAL physics
    computation graph at this point (detached in `compute_zone_attributions`,
    which is where a NEW, purpose-built leaf tensor takes over for
    attribution).
    """
    is_teammate = parsed_frame["is_teammate"]
    attacking_mask = is_teammate
    defending_mask = ~is_teammate

    def _run(mask):
        pos = parsed_frame["player_pos"][mask]
        if pos.shape[0] == 0:
            return None, None
        vel = parsed_frame["player_vel"][mask]
        fatigue = parsed_frame["fatigue_mod"][mask]
        active_coords, control_probabilities, _ = engine(
            pos, vel, fatigue, pitch_grid, parsed_frame["ball_pos"]
        )
        return active_coords, control_probabilities.max(dim=0).values

    att_coords, att_control = _run(attacking_mask)
    def_coords, def_control = _run(defending_mask)
    active_coords = att_coords if att_coords is not None else def_coords
    if active_coords is None:
        raise ValueError("neither team has a visible player in this frame -- no control to attribute")

    n_active = active_coords.shape[0]
    if att_control is None:
        att_control = torch.zeros(n_active)
    if def_control is None:
        def_control = torch.zeros(n_active)

    defending_positions = parsed_frame["player_pos"][defending_mask]
    highest_defending_x = (
        defending_positions[:, 0].max() if defending_positions.shape[0] > 0 else torch.tensor(PITCH_LENGTH)
    )

    return active_coords, att_control, def_control, highest_defending_x


def _zone_features_from_grid(
    attacking_control: torch.Tensor,  # [B, N_active], LEAF, requires_grad
    defending_control: torch.Tensor,  # [B, N_active], LEAF, requires_grad
    active_coords: torch.Tensor,      # [N_active, 2], fixed geometry, no grad needed
    ball_pos: torch.Tensor,
    highest_defending_x: torch.Tensor,
) -> torch.Tensor:
    """Reimplements `feature_extractor.extract_features`'s 4-scalar
    aggregation EXACTLY (same masks, same formulas -- verified against that
    function's source), but keeps every step as a tensor operation instead
    of calling `.item()`, so gradients flow from the returned `[B, 4]`
    tensor back to `attacking_control`/`defending_control`. Returns
    features in `survival_dataset.FEATURE_KEYS`'s exact order.

    `B` is a batch dimension Captum's `IntegratedGradients` controls, NOT
    something callers set directly -- it evaluates this function at
    `n_steps` interpolated points along the baseline-to-input path
    simultaneously (`B = n_steps`, default 50), not once. Masks below are
    `[N_active]` (frame geometry never varies across interpolation steps)
    and are applied via elementwise multiply + `sum(dim=-1)`, not boolean
    indexing, specifically so they broadcast correctly against the `[B,
    N_active]` control tensors instead of assuming `B == 1`.
    """
    dist_to_ball = torch.linalg.norm(active_coords - ball_pos.unsqueeze(0), dim=-1)  # [N_active]
    near_ball_mask = (dist_to_ball <= NEAR_BALL_RADIUS).float()  # [N_active]
    final_third_mask = (active_coords[:, 0] > FINAL_THIRD_X).float()
    behind_line_mask = (active_coords[:, 0] < highest_defending_x).float()

    attacking_control_near_ball = (attacking_control * near_ball_mask).sum(dim=-1)  # [B]
    defending_control_near_ball = (defending_control * near_ball_mask).sum(dim=-1)
    attacking_control_final_third = (attacking_control * final_third_mask).sum(dim=-1)
    space_behind_defending_line = ((1.0 - defending_control) * behind_line_mask).sum(dim=-1)

    return torch.stack(
        [
            attacking_control_near_ball,
            defending_control_near_ball,
            attacking_control_final_third,
            space_behind_defending_line,
        ],
        dim=-1,
    )  # [B, 4]


def compute_zone_attributions(
    parsed_frame: dict,
    engine: BiomechanicalPitchControl,
    model: torch.nn.Module,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    pitch_grid: torch.Tensor,
    time_bin: int = 3,
) -> dict:
    """Integrated Gradients attribution of the MLP's cumulative-incidence
    output back to PER-CELL attacking/defending control values, for ONE
    parsed 360 frame -- the zone-level extension of `explainer.
    compute_attributions`'s 4-scalar version.

    BASELINE, STATED EXPLICITLY (a real, different-in-kind choice from
    `explainer.compute_attributions`'s "zero NORMALIZED feature space =
    average training match state" baseline): here the baseline is ZERO
    CONTROL at every cell for both teams -- "no team can reach any cell,"
    an absence-of-control reference point, not an average-match-state one.
    This is the natural zero for a probability-valued quantity, but it is
    NOT the same kind of "average" the existing 4-feature explainer uses,
    and should not be read as directly comparable to those attributions.

    Returns `{"active_coords": [[x,y],...], "attacking_control_attribution":
    [...], "defending_control_attribution": [...], "cumulative_incidence":
    float}`, all in the same order (one entry per active cell in this
    frame).
    """
    active_coords, att_control, def_control, highest_defending_x = _active_grid_and_controls(
        engine, parsed_frame, pitch_grid
    )

    # Detach from the ORIGINAL physics graph and re-attach as fresh LEAF
    # tensors -- this module attributes w.r.t. the CONTROL VALUES
    # themselves, not back through the ODE/Newton-Raphson solve to player
    # positions (see module docstring's stated scope).
    att_leaf = att_control.detach().clone().unsqueeze(0).requires_grad_(True)  # [1, N_active]
    def_leaf = def_control.detach().clone().unsqueeze(0).requires_grad_(True)  # [1, N_active]
    active_coords_fixed = active_coords.detach().clone()
    ball_pos_fixed = parsed_frame["ball_pos"].detach().clone()
    highest_defending_x_fixed = highest_defending_x.detach().clone()

    def forward_fn(att_grid: torch.Tensor, def_grid: torch.Tensor) -> torch.Tensor:
        # att_grid/def_grid: [B, N_active] -- B is Captum's internal
        # interpolation-step batch (n_steps, default 50), NOT a fixed 1;
        # see `_zone_features_from_grid`'s docstring.
        features = _zone_features_from_grid(
            att_grid, def_grid, active_coords_fixed, ball_pos_fixed, highest_defending_x_fixed
        )  # [B, 4]
        normalized = (features - normalization_mean) / normalization_std  # [B, 4]
        pmf = model(normalized)  # [B, num_bins]
        cumulative = torch.cumsum(pmf, dim=1)
        return cumulative[:, time_bin]  # [B]

    baseline_att = torch.zeros_like(att_leaf)
    baseline_def = torch.zeros_like(def_leaf)

    integrated_gradients = IntegratedGradients(forward_fn)
    att_attr, def_attr = integrated_gradients.attribute(
        (att_leaf, def_leaf), baselines=(baseline_att, baseline_def)
    )

    with torch.no_grad():
        cumulative_incidence = forward_fn(att_leaf, def_leaf).item()

    return {
        "active_coords": active_coords_fixed.tolist(),
        "attacking_control_attribution": att_attr.squeeze(0).tolist(),
        "defending_control_attribution": def_attr.squeeze(0).tolist(),
        "cumulative_incidence": cumulative_incidence,
    }


def zone_attributions_to_grid(attribution_result: dict) -> dict:
    """Re-bins one frame's per-cell attributions (frame-specific, ~2,000-
    2,800 active cells) into the SAME `GRID_COLS x GRID_ROWS` coarse grid
    `habit_memory`/`team_report` use, by SUMMING attribution within each
    coarse cell -- this is the step that turns "a scatter of per-cell
    numbers for one frame" into a fixed, human-readable zone map (e.g. "the
    right half-space is systematically open"), and what would need to be
    AVERAGED ACROSS MANY FRAMES (not done here -- see this module's
    Step-3-feasibility caveats) before that kind of aggregate statement
    would be evidence-backed rather than a single-frame anecdote.
    """
    att_sum = [[0.0] * GRID_ROWS for _ in range(GRID_COLS)]
    def_sum = [[0.0] * GRID_ROWS for _ in range(GRID_COLS)]

    for (x, y), att_val, def_val in zip(
        attribution_result["active_coords"],
        attribution_result["attacking_control_attribution"],
        attribution_result["defending_control_attribution"],
    ):
        col = min(max(int(x // CELL_WIDTH_METERS), 0), GRID_COLS - 1)
        row = min(max(int(y // CELL_HEIGHT_METERS), 0), GRID_ROWS - 1)
        att_sum[col][row] += att_val
        def_sum[col][row] += def_val

    return {
        "attacking_control_attribution_grid": att_sum,
        "defending_control_attribution_grid": def_sum,
        "grid_shape": f"{GRID_COLS} cols x {GRID_ROWS} rows, same convention as habit_memory/team_report",
    }


def aggregate_zone_attributions(team_name: str, match_ids: list[int], time_bin: int = 3) -> dict:
    """Averages per-frame zone attributions (`compute_zone_attributions` +
    `zone_attributions_to_grid`) across every 360-covered chain frame where
    `team_name` is the ACTING/ATTACKING side, across `match_ids` -- the
    aggregation step needed before a statement like "the right half-space
    is systematically open" is actually evidence-backed, rather than one
    frame's anecdote (see `zone_attributions_to_grid`'s docstring).

    Reuses `team_report`'s own match-iteration/team-filtering helpers
    (`_match_representative_chain_frames`, `_teams_in_match`) rather than
    re-deriving the same logic a third time. Loads the deterministic MLP
    once (`explainer.load_deterministic_mlp`), lazily, only if at least one
    qualifying frame is found.
    """
    from production.src.models.explainer import load_deterministic_mlp
    from production.src.reporting.team_report import (
        _match_representative_chain_frames,
        _teams_in_match,
    )

    engine = BiomechanicalPitchControl()
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, PITCH_LENGTH, dtype=torch.float32),
        torch.arange(0, PITCH_WIDTH, dtype=torch.float32),
        indexing="ij",
    )
    pitch_grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    att_sum = [[0.0] * GRID_ROWS for _ in range(GRID_COLS)]
    def_sum = [[0.0] * GRID_ROWS for _ in range(GRID_COLS)]
    att_count = [[0] * GRID_ROWS for _ in range(GRID_COLS)]
    def_count = [[0] * GRID_ROWS for _ in range(GRID_COLS)]

    model = normalization_mean = normalization_std = None
    frames_used = 0

    for match_id in match_ids:
        if team_name not in _teams_in_match(match_id):
            continue
        for chain, parsed, _minute in _match_representative_chain_frames(match_id):
            if chain["team"] != team_name:
                continue
            if model is None:
                model, normalization_mean, normalization_std, run_id = load_deterministic_mlp()
                logger.info(f"using deterministic MLP run_id={run_id}")

            result = compute_zone_attributions(
                parsed, engine, model, normalization_mean, normalization_std, pitch_grid, time_bin=time_bin
            )
            grid = zone_attributions_to_grid(result)
            frames_used += 1
            for col in range(GRID_COLS):
                for row in range(GRID_ROWS):
                    a = grid["attacking_control_attribution_grid"][col][row]
                    d = grid["defending_control_attribution_grid"][col][row]
                    if a != 0.0:
                        att_sum[col][row] += a
                        att_count[col][row] += 1
                    if d != 0.0:
                        def_sum[col][row] += d
                        def_count[col][row] += 1

    mean_att = [
        [(att_sum[c][r] / att_count[c][r]) if att_count[c][r] > 0 else None for r in range(GRID_ROWS)]
        for c in range(GRID_COLS)
    ]
    mean_def = [
        [(def_sum[c][r] / def_count[c][r]) if def_count[c][r] > 0 else None for r in range(GRID_ROWS)]
        for c in range(GRID_COLS)
    ]

    return {
        "team_name": team_name,
        "frames_used": frames_used,
        "mean_attacking_control_attribution_grid": mean_att,
        "mean_defending_control_attribution_grid": mean_def,
        "grid_shape": f"{GRID_COLS} cols x {GRID_ROWS} rows, same convention as habit_memory/team_report",
        "note": (
            "Mean is over frames where a cell was active (non-zero attribution), not over all "
            "frames_used -- a cell far from the ball in most frames legitimately has fewer "
            "observations than one near typical play."
        ),
    }


# =============================================================================
# Milestone 43 (new reporting track): natural-language zone-explanation layer.
#
# ADDITIVE ONLY -- everything above this line is Milestone 40's original
# zone-level attribution logic, UNCHANGED. Nothing below modifies it; these
# functions only CONSUME `aggregate_zone_attributions`'s output (a 10x7 grid
# of mean attribution values, `None` for cells with no observations) and
# `team_report.generate_team_report`'s output.
#
# THIS CLOSES THE THIRD LAYER of the original three-layer ask (state
# estimation / threat estimation / tactical explanation) for the zone-level
# signal specifically -- Milestone 15 already closed it for the 4 SCALAR
# features; this does the same job for spatial zones.
# =============================================================================

# HAND-PICKED, stated explicitly (this project's standing practice for every
# tunable constant): a cell counts as part of a "notable" region only if its
# |attribution| is at least this FRACTION of the grid's own max |attribution|
# -- a RELATIVE threshold, not an absolute one, since raw attribution
# magnitude depends on the baseline/sample and an absolute cutoff picked for
# one team/report would not transfer to another. Not tuned against any
# validation criterion.
NOTABLE_ZONE_THRESHOLD_FRACTION = 0.5

# Reuses team_report.py's OWN defensive-third boundary formula
# (`PITCH_LENGTH - FINAL_THIRD_X`) rather than importing that module (a
# small, self-contained constant -- not worth a cross-module dependency for
# one number, but deliberately computed the SAME way, not re-guessed).
_DEFENSIVE_THIRD_X = PITCH_LENGTH - FINAL_THIRD_X


def _zone_name(col: int, row: int) -> str:
    """Human-readable zone description for grid cell `(col, row)`.

    X-AXIS (pitch third): reuses `feature_extractor.FINAL_THIRD_X` (66.0)
    and its mirror `_DEFENSIVE_THIRD_X` (34.0) -- the SAME established
    thirds boundary `team_report.py` already uses, not a re-invented one.

    Y-AXIS (left/right side): HAND-PICKED AND ARBITRARY -- stated
    explicitly, the same caveat `tactical_map_renderer.py`'s
    `_to_canvas_xy` and `player_visualizer.py`'s `POSITION_LOCATIONS`
    already state for their own y-axis choices. This project has no
    established real-world left/right convention for the y-axis (StatsBomb
    coordinates are per-ACTOR-oriented for x/attacking-direction only, per
    ADR-009 -- that ADR says nothing about y/width-axis handedness). Rows
    0-1 (~19.4m nearest y=0) are labeled "right side", rows 5-6 (~19.4m
    nearest y=PITCH_WIDTH) "left side", the middle 3 rows "central
    channel" -- a label, not a verified real-world claim.
    """
    x_m = (col + 0.5) * CELL_WIDTH_METERS
    if x_m < _DEFENSIVE_THIRD_X:
        third = "defensive third"
    elif x_m > FINAL_THIRD_X:
        third = "attacking third"
    else:
        third = "middle third"

    if row <= 1:
        side = "right side"
    elif row >= GRID_ROWS - 2:
        side = "left side"
    else:
        side = "central channel"

    return f"{third}, {side}"


def _zone_identifier(zone_name: str) -> str:
    """`zone_name` ("defensive third, right side") -> a `\\w+`-safe token
    ("defensive_third_right_side") -- required because this module reuses
    `explainer.generate_explanation`'s EXISTING regex-based mock executor
    UNMODIFIED (see `build_zone_explanation_prompt`), and that executor's
    `_FACTOR_NAME_PATTERN` only matches `[\\w_]+` (no spaces, no commas).
    The natural-language name is preserved in each zone dict for the
    PROMPT's own body text; only this identifier form goes into the
    `- name: value` factor lines the mock executor's regex parses.
    """
    return zone_name.replace(", ", "_").replace(" ", "_")


def identify_notable_zones(
    zone_attributions: np.ndarray, top_n: int = 3, source_label: str = "control"
) -> list[dict]:
    """Step 1: identifies the most notable POSITIVE (threat-driving) and
    NEGATIVE (protective/solid) CONTIGUOUS regions in a per-cell attribution
    grid -- not just individual highest-magnitude cells.

    METHOD, STATED EXPLICITLY: connected-component grouping (4-connectivity,
    `scipy.ndimage.label`) of cells whose value clears
    `NOTABLE_ZONE_THRESHOLD_FRACTION` (a hand-picked, relative threshold --
    see that constant's own comment) of that SIGN's OWN extreme value --
    i.e. positive cells are thresholded against `0.5 * max(positive
    values)`, negative cells against `0.5 * |min(negative values)|`,
    SEPARATELY, not against a single shared `max(|grid|)`. This was found
    to matter on real data while building this: Argentina's real aggregate
    grid has a positive extreme (~0.045) over 2x the negative extreme's
    magnitude (~-0.019); a single shared threshold pegged to the larger
    side's extreme suppressed EVERY negative cell entirely, silently
    erasing the "systematically weak in the defensive half" signal
    `REPORTING_FINDINGS.md` already reported from the same data. Per-sign
    thresholds fix this without changing the underlying rule's spirit
    ("notable relative to how extreme this GRID'S values get"), just
    applying it fairly to both signs instead of only the larger one. This
    is a genuine clustering of ADJACENT notable cells into coherent
    regions (matching what "the right half-space is systematically open"
    actually describes -- a contiguous area, not a scattered set of
    individually-extreme cells), not a re-derivation of pre-named pitch
    zones.

    `zone_attributions`: a `[GRID_COLS, GRID_ROWS]` array (e.g.
    `aggregate_zone_attributions`'s `mean_attacking_control_attribution_grid`
    or `mean_defending_control_attribution_grid`, converted from its native
    list-of-lists-with-`None` form via `np.nan` for missing cells -- see
    `test_zone_explainer.py` for that conversion).
    `source_label`: which underlying quantity this grid came from (e.g.
    `"attacking_control"`) -- embedded into each returned zone's `"source"`
    field, since `identify_notable_zones` itself only ever sees ONE grid and
    cannot otherwise know which of the two `aggregate_zone_attributions`
    grids it was given. THIS IS THE LIMIT of what "which pitch-control
    feature was most responsible" can honestly mean here: the zone-level
    attribution (`compute_zone_attributions`) attributes to attacking-control
    and defending-control VALUES DIRECTLY, not to Milestone 15's 4 named
    scalar features (`attacking_control_near_ball`, etc.) -- so the most
    specific derivable statement is "attacking_control" vs "defending_control",
    not a finer breakdown.

    Returns up to `top_n` zone dicts (across both signs combined, ranked by
    `|aggregate_attribution|`): `{"name": str, "sign": "positive"|"negative",
    "aggregate_attribution": float, "cell_count": int, "centroid_xy_m":
    (float, float), "source": str}`.
    """
    grid = np.asarray(zone_attributions, dtype=np.float64)
    valid = ~np.isnan(grid)
    if not valid.any():
        return []

    positive_values = grid[valid & (grid > 0)]
    negative_values = grid[valid & (grid < 0)]
    positive_threshold = NOTABLE_ZONE_THRESHOLD_FRACTION * positive_values.max() if positive_values.size else None
    negative_threshold = (
        NOTABLE_ZONE_THRESHOLD_FRACTION * abs(negative_values.min()) if negative_values.size else None
    )
    if positive_threshold is None and negative_threshold is None:
        return []

    mask_sign_pairs = []
    if positive_threshold is not None:
        mask_sign_pairs.append((valid & (grid >= positive_threshold), "positive"))
    if negative_threshold is not None:
        mask_sign_pairs.append((valid & (grid <= -negative_threshold), "negative"))

    zones = []
    for mask, sign in mask_sign_pairs:
        if not mask.any():
            continue
        labeled, num_components = label(mask)
        for component_id in range(1, num_components + 1):
            component_mask = labeled == component_id
            cells = np.argwhere(component_mask)  # [(col, row), ...]
            values = grid[component_mask]
            centroid_col = float(cells[:, 0].mean())
            centroid_row = float(cells[:, 1].mean())
            zone_name = _zone_name(round(centroid_col), round(centroid_row))
            zones.append(
                {
                    "name": zone_name,
                    "sign": sign,
                    "aggregate_attribution": float(values.sum()),
                    "cell_count": len(cells),
                    "centroid_xy_m": (
                        (centroid_col + 0.5) * CELL_WIDTH_METERS,
                        (centroid_row + 0.5) * CELL_HEIGHT_METERS,
                    ),
                    "source": source_label,
                }
            )

    zones.sort(key=lambda z: -abs(z["aggregate_attribution"]))
    return zones[:top_n]


def build_zone_explanation_prompt(
    team_report: dict,
    notable_zones: list[dict],
    frames_used: int | None = None,
) -> str:
    """Step 2: builds a prompt for a zone-level tactical explanation,
    following the SAME structure/discipline as `explainer.
    build_tactical_prompt` (system role framing explicitly stating
    "explain, don't predict" per ADR-006, real computed inputs, an explicit
    ask for a concise output) but scoped to ZONE-level findings.

    STRUCTURAL NOTE: this prompt deliberately reuses several EXACT literal
    header strings from `build_tactical_prompt` ("Predicted threat:",
    "Top factors INCREASING this threat estimate:", "Top factors DECREASING
    this threat estimate:", "\\n\\nProvide a concise") -- not a style
    preference, a REQUIREMENT, because Step 2.3 reuses `explainer.
    generate_explanation`'s EXISTING regex-based mock executor UNMODIFIED,
    and that executor's regexes match these exact strings. Zone names
    (which contain spaces/commas) are converted to `\\w+`-safe identifiers
    (`_zone_identifier`) for the `- name: value` factor lines specifically,
    since the mock executor's `_FACTOR_NAME_PATTERN` cannot match a name
    containing spaces or commas.

    `frames_used`: the real sample size behind `notable_zones` (typically
    `aggregate_zone_attributions`'s own `frames_used`), stated in the
    prompt's framing paragraph if given -- if omitted, the prompt says so
    honestly ("exact count not provided") rather than fabricating a number.

    HONESTY CONSTRAINT, STATED DIRECTLY IN THE PROMPT TEXT ITSELF (Step 3):
    this data is a STATIC, per-frame-AVERAGED spatial pattern -- there is no
    temporal/transition-state layer built anywhere in this project (no
    frame-to-frame trajectory analysis, no player-specific movement-
    direction computation). The prompt explicitly instructs against any
    duration claim (e.g. "for N seconds") or specific player-movement claim,
    because neither is derivable from what `notable_zones` actually
    contains -- a spatial magnitude and sign, nothing time-indexed.
    """
    threat_by_zone = team_report.get("threat_by_pitch_zone", {}) or {}
    known_threats = [v for v in threat_by_zone.values() if v is not None]
    mean_threat_pct = (sum(known_threats) / len(known_threats) * 100.0) if known_threats else 0.0

    sample_note = (
        f"aggregated across {frames_used} real, sampled 360-covered attacking frames"
        if frames_used is not None
        else "aggregated across this report's sampled attacking frames (exact frame count not provided to this prompt)"
    )

    positive_zones = [z for z in notable_zones if z["sign"] == "positive"]
    negative_zones = [z for z in notable_zones if z["sign"] == "negative"]

    def _format_bucket(zones: list[dict], direction_label: str) -> str:
        if not zones:
            return f"No zones {direction_label} were identified."
        lines = "\n".join(
            f"- {_zone_identifier(z['name'])}: {z['aggregate_attribution']:+.4f}" for z in zones
        )
        if len(zones) < 2:
            lines += f"\n(Only {len(zones)} zone {direction_label} found, not more.)"
        return lines

    positive_text = _format_bucket(positive_zones, "increasing this threat")
    negative_text = _format_bucket(negative_zones, "decreasing this threat")

    team_name = team_report.get("team_name", "this team")

    prompt = (
        f"You are an expert football tactical analyst reviewing {team_name}'s aggregate "
        "spatial pitch-control pattern. Your job is to explain WHY these zone-level "
        "attributions look the way they do, based ONLY on the spatial attribution data "
        "provided below. You explain this already-computed attribution; you do not make "
        "your own prediction. This data is a STATIC, per-frame-AVERAGED spatial pattern "
        f"({sample_note}), NOT a time-series or player-tracking analysis -- you MUST NOT "
        "state any temporal/duration claim (e.g. 'open for N seconds') or any specific "
        "player movement/recovery claim, because neither is derivable from this data.\n\n"
        f"Predicted threat: {mean_threat_pct:.1f}%\n\n"
        "Top factors INCREASING this threat estimate:\n"
        f"{positive_text}\n\n"
        "Top factors DECREASING this threat estimate:\n"
        f"{negative_text}\n\n"
        "Provide a concise 2-3 sentence tactical explanation of why the model predicted "
        "this threat level, referencing the spatial zones above by name and their real, "
        "computed attribution magnitude."
    )
    return prompt
