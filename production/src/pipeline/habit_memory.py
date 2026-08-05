"""Milestone 22 (Module 6): Bayesian Habit Blending.

STEP 0 FINDING (verified against real cached 360 data -- 21,273
freeze-frames across 6 cached matches -- not assumed): StatsBomb's public
360 freeze-frame entries expose ONLY `location`, `teammate`, `actor`,
`keeper` per visible player -- NEVER a player id or name, not even for the
entry with `actor: True`. The only reliably known player identity anywhere
in this pipeline is the ACTING player of the parent event, via
`event["player"]["id"]`/`["name"]` (present on every 360-covered event
checked). This means player-specific habit priors can only be built and
applied for the ONE player whose identity is actually known at a given
frame -- the actor -- never for the other ~21 visible players. This
module does NOT invent or guess identity for non-actor players (e.g. via
nearest-position matching across frames); that would silently build on a
false premise. See `statsbomb_io.parse_360_frame`'s docstring and the
`is_actor`/`actor_player_id` fields it now exposes.

Coordinate space: all pitch coordinates here are in the SAME rescaled
100 x 68 meter space established by ADR-002 and used throughout
`feature_extractor.py`/`control.py` (verified: `PITCH_LENGTH = 100.0`,
`PITCH_WIDTH = 68.0`), NOT StatsBomb's native 120 x 80 unit space.
`generate_player_heatmap` explicitly rescales raw event locations via
`statsbomb_io.X_SCALE`/`Y_SCALE` before binning, so the resulting heatmap
lives in the same space as `bayesian_blend_habit`'s live player position
and can be combined with it directly.
"""

import logging

import numpy as np

from production.src.ingestion.statsbomb_io import X_SCALE, Y_SCALE
from production.src.pipeline.feature_extractor import PITCH_LENGTH, PITCH_WIDTH

logger = logging.getLogger(__name__)

# Coarse grid over the verified 100x68m pitch space -- 10 columns (10m/cell
# in x) x 7 rows (~9.71m/cell in y).
GRID_COLS = 10
GRID_ROWS = 7
CELL_WIDTH_METERS = PITCH_LENGTH / GRID_COLS
CELL_HEIGHT_METERS = PITCH_WIDTH / GRID_ROWS

# Cold-start threshold (tunable): fewer qualifying historical events than
# this and generate_player_heatmap falls back to a uniform prior rather
# than a sparse/noisy one.
MIN_HISTORICAL_EVENTS = 20

# Cell centers in GRID-INDEX units (e.g. column 0's center is at grid-index
# 0.5), shaped for broadcasting -- reused by both the mock heatmap
# generator and the live-position likelihood so both are built the exact
# same vectorized way (no native Python loop over grid cells).
_COL_CENTERS_GRID = (np.arange(GRID_COLS) + 0.5).reshape(-1, 1)  # [GRID_COLS, 1]
_ROW_CENTERS_GRID = (np.arange(GRID_ROWS) + 0.5).reshape(1, -1)  # [1, GRID_ROWS]


def _uniform_heatmap() -> np.ndarray:
    return np.ones((GRID_COLS, GRID_ROWS), dtype=np.float64) / (GRID_COLS * GRID_ROWS)


def _gaussian_grid(x0_grid: float, y0_grid: float, sigma_grid_x: float, sigma_grid_y: float) -> np.ndarray:
    """Vectorized Gaussian evaluated at every grid cell's center, entirely
    in GRID-INDEX units (both the center `(x0_grid, y0_grid)` and the
    sigmas must already be in grid-index units -- callers convert from
    real meters before calling this).
    """
    dx = (_COL_CENTERS_GRID - x0_grid) / sigma_grid_x  # [GRID_COLS, 1]
    dy = (_ROW_CENTERS_GRID - y0_grid) / sigma_grid_y  # [1, GRID_ROWS]
    return np.exp(-0.5 * (dx ** 2 + dy ** 2))  # broadcasts to [GRID_COLS, GRID_ROWS]


def generate_player_heatmap(
    player_id: int,
    events_by_match: dict,
    exclude_match_id: int | None = None,
) -> np.ndarray:
    """Historical positional Prior for `player_id`, binned into the
    GRID_COLS x GRID_ROWS grid over the verified 100x68m pitch space.

    `events_by_match` is a dict `{match_id: [raw StatsBomb event dicts]}`,
    NOT a flat list. StatsBomb's raw event dicts carry no `match_id` field
    of their own (verified: an event's keys are id/index/period/timestamp/
    minute/second/type/possession/possession_team/play_pattern/team/
    duration/tactics -- there is no per-event match reference), so
    match-level exclusion below requires this match-keyed grouping; a flat
    list with no match association could not support `exclude_match_id` at
    all.

    DATA LEAKAGE GUARD: if `exclude_match_id` is given, every event from
    that match is excluded before building the Prior -- the same
    training-split-only discipline established for feature normalization
    since Milestone 7 (never derive a "historical"/prior statistic from the
    very match/moment being predicted).

    COLD-START FALLBACK: if fewer than MIN_HISTORICAL_EVENTS qualifying
    events remain (this player's own events, with a `location`, outside
    the excluded match), returns a UNIFORM grid instead of a sparse/noisy
    Prior -- this makes the downstream Bayesian update degrade gracefully
    to "trust the live evidence fully" when there's no reliable historical
    signal, rather than overfitting to a handful of points.
    """
    counts = np.zeros((GRID_COLS, GRID_ROWS), dtype=np.float64)
    num_qualifying = 0

    for match_id, events in events_by_match.items():
        if exclude_match_id is not None and match_id == exclude_match_id:
            continue
        for event in events:
            player = event.get("player")
            if player is None or player.get("id") != player_id:
                continue
            if "location" not in event:
                continue

            raw_x, raw_y = event["location"][0], event["location"][1]
            x = raw_x * X_SCALE  # ADR-002 rescale, matching parse_360_frame
            y = raw_y * Y_SCALE

            col = min(max(int(x // CELL_WIDTH_METERS), 0), GRID_COLS - 1)
            row = min(max(int(y // CELL_HEIGHT_METERS), 0), GRID_ROWS - 1)

            counts[col, row] += 1.0
            num_qualifying += 1

    if num_qualifying < MIN_HISTORICAL_EVENTS:
        logger.warning(
            f"player_id={player_id}: only {num_qualifying} qualifying historical "
            f"event(s) (< {MIN_HISTORICAL_EVENTS}) -- falling back to a uniform prior."
        )
        return _uniform_heatmap()

    return counts / counts.sum()


def build_player_match_buckets(events_by_match: dict) -> dict:
    """Milestone 23: precomputes, ONCE, every player's historical (x, y)
    positions (already rescaled into the 100x68m space via ADR-002's
    X_SCALE/Y_SCALE, matching `generate_player_heatmap`'s convention) per
    match, from raw StatsBomb events -- `{player_id: {match_id: [(x, y),
    ...]}}`.

    This exists so that training a model on many samples doesn't re-scan
    the full event corpus once per sample: `heatmap_from_buckets` below
    aggregates these precomputed per-match buckets cheaply (summing
    coordinate lists already grouped by player/match), which is what makes
    a distinct, correctly-scoped heatmap per (player, sample) pair
    tractable across thousands of samples.
    """
    buckets: dict = {}
    for match_id, events in events_by_match.items():
        for event in events:
            player = event.get("player")
            if player is None or "location" not in event:
                continue
            player_id = player["id"]
            x = event["location"][0] * X_SCALE
            y = event["location"][1] * Y_SCALE
            buckets.setdefault(player_id, {}).setdefault(match_id, []).append((x, y))
    return buckets


def heatmap_from_buckets(
    player_id: int,
    buckets: dict,
    included_match_ids,
    exclude_match_id: int | None = None,
) -> tuple:
    """Aggregates a player's PRECOMPUTED per-match coordinate buckets (see
    `build_player_match_buckets`) over `included_match_ids`, excluding
    `exclude_match_id` if present among them, into the same
    GRID_COLS x GRID_ROWS Prior `generate_player_heatmap` produces --
    including the identical cold-start fallback rule -- without
    re-scanning any raw events.

    `included_match_ids` should already be the caller's TRAINING-split
    match set (Milestone 23 Step 1's leakage discipline); this function
    itself only handles the per-sample `exclude_match_id` exclusion, not
    the train/val partitioning, which is the caller's responsibility.

    Returns (heatmap, num_qualifying_events, is_cold_start) -- the extra
    two values let a caller building many heatmaps in bulk tally diversity/
    cold-start diagnostics without recomputing them separately.
    """
    player_buckets = buckets.get(player_id, {})
    counts = np.zeros((GRID_COLS, GRID_ROWS), dtype=np.float64)
    num_qualifying = 0

    for match_id in included_match_ids:
        if exclude_match_id is not None and match_id == exclude_match_id:
            continue
        for x, y in player_buckets.get(match_id, []):
            col = min(max(int(x // CELL_WIDTH_METERS), 0), GRID_COLS - 1)
            row = min(max(int(y // CELL_HEIGHT_METERS), 0), GRID_ROWS - 1)
            counts[col, row] += 1.0
            num_qualifying += 1

    if num_qualifying < MIN_HISTORICAL_EVENTS:
        return _uniform_heatmap(), num_qualifying, True

    return counts / counts.sum(), num_qualifying, False


def generate_mock_heatmap(position: tuple[float, float], sigma_meters: float = 15.0) -> np.ndarray:
    """Synthetic Gaussian-blob heatmap centered at `position` (x, y) in
    real pitch-meter coordinates -- for testing, not derived from any real
    player's history. `sigma_meters` is deliberately wider than
    `bayesian_blend_habit`'s default live-position sigma (5.0m): a
    historical PRIOR is expected to represent a smoothed tendency over many
    events, not a single tightly-localized observation.
    """
    x0_grid = position[0] / CELL_WIDTH_METERS
    y0_grid = position[1] / CELL_HEIGHT_METERS
    sigma_grid_x = sigma_meters / CELL_WIDTH_METERS
    sigma_grid_y = sigma_meters / CELL_HEIGHT_METERS

    grid = _gaussian_grid(x0_grid, y0_grid, sigma_grid_x, sigma_grid_y)
    return grid / grid.sum()


def bayesian_blend_habit(
    player_pos: tuple[float, float],
    player_heatmap: np.ndarray,
    sigma_meters: float = 5.0,
) -> tuple[float, float]:
    """Multiplicative Bayesian blend of a historical Prior
    (`player_heatmap`, GRID_COLS x GRID_ROWS, sums to 1) with a live
    observed position (`player_pos`, real pitch-meter (x, y)).

    `sigma_meters` is explicitly converted into GRID-CELL units before
    building the Gaussian likelihood --
    `sigma_grid_x = sigma_meters / CELL_WIDTH_METERS`,
    `sigma_grid_y = sigma_meters / CELL_HEIGHT_METERS` -- rather than used
    directly as a standard deviation in grid-INDEX space. Skipping this
    conversion is an easy, silent unit-mismatch bug: a "5 meter" live-
    position blob would otherwise end up either far too tight (if treated
    as 5 grid-index units, i.e. 50m in x) or far too wide (if the grid
    were finer) relative to what "5 meters" actually means on this
    specific ~10m x ~9.7m-per-cell grid.

    Returns (expected_x_meters, expected_y_meters): the Posterior's
    centroid, converted back from grid-index space to real pitch meters.
    """
    x0_grid = player_pos[0] / CELL_WIDTH_METERS
    y0_grid = player_pos[1] / CELL_HEIGHT_METERS
    sigma_grid_x = sigma_meters / CELL_WIDTH_METERS
    sigma_grid_y = sigma_meters / CELL_HEIGHT_METERS

    likelihood = _gaussian_grid(x0_grid, y0_grid, sigma_grid_x, sigma_grid_y)

    posterior_unnormalized = player_heatmap * likelihood
    # Epsilon floor BEFORE normalizing: if the Prior is sparse and the
    # live-position Gaussian's mass falls mostly on near-zero Prior cells,
    # the elementwise product can be all/near-zero everywhere, and
    # dividing by a ~0 sum produces NaN. Adding a tiny floor first
    # guarantees a strictly positive sum, so this degrades to "posterior
    # ~= uniform" instead of NaN in that adversarial case.
    posterior_unnormalized = posterior_unnormalized + 1e-9
    posterior = posterior_unnormalized / posterior_unnormalized.sum()

    expected_x_grid = (posterior * _COL_CENTERS_GRID).sum()
    expected_y_grid = (posterior * _ROW_CENTERS_GRID).sum()

    expected_x_meters = float(expected_x_grid * CELL_WIDTH_METERS)
    expected_y_meters = float(expected_y_grid * CELL_HEIGHT_METERS)

    return expected_x_meters, expected_y_meters
