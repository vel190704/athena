"""Milestone 42 (new reporting track, Step 1): Player Report Visualization.

Pure RENDERING layer over Milestone 40's `generate_player_report` output
-- does not recompute, adjust, or otherwise touch any report value.
Nothing in `player_report.py` is modified or reimplemented here.

LIBRARY CHOICE: matplotlib, not PIL. This dashboard needs several things
matplotlib gives directly (patches for the pitch outline, `imshow` +
colormaps + a colorbar for the smoothed heatmap, proportionally-sized
scatter markers, multi-line text layout, a `GridSpec` panel layout, all
composed into one importable Figure) -- PIL would require hand-rolling
each of those (manual anti-aliased circle/line drawing, manual text
wrapping, no built-in colormap/colorbar support), for no benefit here
since this is an offline, one-shot static PNG export, not a
performance-sensitive pixel-buffer pipeline (that's what `production/src/
cv/tactical_map_renderer.py` is, and why THAT module uses OpenCV instead).
"""

import logging

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from production.src.pipeline.feature_extractor import (
    FINAL_THIRD_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from production.src.pipeline.habit_memory import (
    GRID_COLS,
    GRID_ROWS,
    MIN_HISTORICAL_EVENTS,
)
from production.src.reporting.pitch_diagram import draw_pitch_outline

logger = logging.getLogger(__name__)

# Reuses habit_memory's own cold-start threshold (not a separately-invented
# number) as the "too few events to be a confident distribution" cutoff for
# the positional-distribution panel's low-sample warning below -- the same
# real threshold `player_report.py`'s `heatmap_used_uniform_fallback`
# already checks against, applied here to the OTHER field
# (`positional_distribution`) that has no such fallback mechanism of its
# own but suffers the identical small-sample-looks-confident problem.
LOW_SAMPLE_EVENT_COUNT_THRESHOLD = MIN_HISTORICAL_EVENTS

# HAND-BUILT layout for visualization purposes only -- StatsBomb's
# `position` field is a role LABEL, not a coordinate; it carries no (x, y)
# location anywhere in the schema. These placements are a reasonable,
# common analytics-dashboard convention (formation-slot positions on a
# 100x68m pitch, attacking direction = increasing x per ADR-002/009),
# invented for this milestone, not sourced from or verified against any
# StatsBomb data. Covers every position name observed across a 40-match
# sample of this project's real cached data (`data/raw/*_events.json`).
POSITION_LOCATIONS: dict[str, tuple[float, float]] = {
    "Goalkeeper": (5, 34),
    "Right Back": (18, 10),
    "Right Center Back": (18, 24),
    "Center Back": (18, 34),
    "Left Center Back": (18, 44),
    "Left Back": (18, 58),
    "Right Wing Back": (35, 6),
    "Left Wing Back": (35, 62),
    "Right Defensive Midfield": (35, 24),
    "Center Defensive Midfield": (35, 34),
    "Left Defensive Midfield": (35, 44),
    "Right Midfield": (50, 8),
    "Right Center Midfield": (50, 24),
    "Center Midfield": (50, 34),
    "Left Center Midfield": (50, 44),
    "Left Midfield": (50, 60),
    "Right Attacking Midfield": (65, 20),
    "Center Attacking Midfield": (65, 34),
    "Left Attacking Midfield": (65, 48),
    "Right Wing": (80, 6),
    "Left Wing": (80, 62),
    "Right Center Forward": (85, 26),
    "Center Forward": (88, 34),
    "Left Center Forward": (85, 42),
}

MIN_DOT_RADIUS_M = 3.0
MAX_DOT_RADIUS_M = 9.0
HEATMAP_GAUSSIAN_SIGMA = 0.9  # in GRID_COLS x GRID_ROWS cell units

# Matches pitch_diagram.draw_pitch_outline's own ax.set_ylim(-2, PITCH_WIDTH+2)
# -- the label-clamp below needs the SAME bound the pitch itself is drawn
# against, not a re-guessed one, or the two could silently drift apart.
_PITCH_Y_AXIS_MIN = -2.0
# Rough estimate (pitch-meters) of a 2-line label's rendered height at this
# module's fontsize -- not exact font-metrics, just enough headroom that a
# label placed va='top' at (label_y - this) still lands inside the axis
# limits rather than being clipped by them.
_LABEL_HEIGHT_ESTIMATE_M = 6.0

# --- Shot map (additive new feature) -- constants -------------------------
# Reuses the SAME low-sample threshold value player_report.py's
# generate_player_shot_map computes shot_map_used_low_sample_flag from
# (MIN_HISTORICAL_EVENTS=20) -- imported from that function's own return
# value at render time, not re-derived here, so this file never needs its
# own copy of the threshold number.
#
# Sizing constants deliberately separate from MIN_DOT_RADIUS_M/
# MAX_DOT_RADIUS_M above (the positional-distribution panel's own dots) --
# a shot map typically plots far more markers at once (Messi: 2,646 real
# shots for a full career) than the positional panel ever does (at most
# ~15-20 distinct position labels), so shot markers are sized smaller to
# stay legible when many overlap.
MIN_SHOT_DOT_RADIUS_M = 1.2
MAX_SHOT_DOT_RADIUS_M = 4.5

# Reference-image convention (goal = filled/red, no-goal = hollow) --
# StatsBomb's own real `shot.outcome.name == "Goal"` value drives which of
# these two styles a given shot gets; nothing else about the outcome
# (Saved/Blocked/Off T/etc.) is separately styled, matching the
# reference's own two-state (scored/not-scored) legend, not the shot's
# full outcome taxonomy.
GOAL_SHOT_COLOR = "#e63946"
NO_GOAL_SHOT_EDGE_COLOR = "#f1faee"

# Crop the shot-map pitch to the attacking third only (reference-image
# framing), rather than reusing the full 100m pitch every other panel in
# this file uses -- shots overwhelmingly cluster near the box, so a full-
# pitch view would compress virtually all real markers into a small
# corner of the canvas. Reuses FINAL_THIRD_X (=66.0) directly from
# feature_extractor.py -- the SAME attacking-third boundary
# `team_report.py`'s own zone bucketing already established -- rather
# than inventing a new, separate cropping threshold.
SHOT_MAP_X_MIN = FINAL_THIRD_X


def _draw_positional_distribution(ax, positional_distribution: dict[str, float], event_count: int) -> None:
    draw_pitch_outline(ax)
    ax.set_title(f"Positional distribution (% of {event_count} tagged events)", color="white", fontsize=11)

    if not positional_distribution:
        ax.text(
            PITCH_LENGTH / 2, PITCH_WIDTH / 2, "No positional data", color="white", ha="center", va="center"
        )
        return

    # Milestone 44's validation sweep found a real gap here: a 1-event
    # player's "100%" figure looked identical to a well-supported one.
    # The title above now always states the real event count backing
    # these percentages; below `LOW_SAMPLE_EVENT_COUNT_THRESHOLD`, an
    # explicit low-confidence banner is ALSO drawn, since a title alone is
    # easy to skim past on a dashboard whose visual focus is the dots.
    if event_count < LOW_SAMPLE_EVENT_COUNT_THRESHOLD:
        ax.text(
            PITCH_LENGTH / 2, PITCH_WIDTH + 5,
            f"LOW SAMPLE ({event_count} event{'s' if event_count != 1 else ''}) -- not a confident distribution",
            color="#ff4444", ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=5,
        )

    max_share = max(positional_distribution.values())
    for position_name, share in positional_distribution.items():
        location = POSITION_LOCATIONS.get(position_name)
        if location is None:
            logger.warning(f"no plot location for position {position_name!r} -- skipping dot.")
            continue
        x, y = location
        radius = MIN_DOT_RADIUS_M + (MAX_DOT_RADIUS_M - MIN_DOT_RADIUS_M) * (share / max_share)
        ax.scatter(
            [x], [y], s=radius**2 * 12, color="#ffcc00", edgecolors="black", linewidths=0.8, alpha=0.85, zorder=3
        )

        # Label-position clamp: the default placement is BELOW the dot
        # (va='top', text grows downward from label_y). For a dot near the
        # bottom edge (e.g. "Right Wing" at y=6), that default would push
        # the label below `_PITCH_Y_AXIS_MIN` and clip it off-canvas. In
        # that case, flip the label ABOVE the dot instead (va='bottom')
        # rather than just nudging it a few meters -- a dot this close to
        # one edge is, by construction, far from the other, so flipping
        # always has room. The symmetric case (a dot near the TOP edge)
        # doesn't need this: the default already places its label toward
        # the pitch center, away from that edge.
        label_below_y = y - radius - 3
        if label_below_y - _LABEL_HEIGHT_ESTIMATE_M < _PITCH_Y_AXIS_MIN:
            label_y, va = y + radius + 3, "bottom"
        else:
            label_y, va = label_below_y, "top"

        ax.text(
            x, label_y, f"{position_name}\n{share*100:.1f}%",
            color="white", ha="center", va=va, fontsize=6.5, zorder=4,
        )


def _draw_heatmap(ax, heatmap_grid: list[list[float]], event_count: int, used_uniform_fallback: bool) -> None:
    draw_pitch_outline(ax)
    ax.set_title(f"Aggregate positional heatmap ({event_count} events)", color="white", fontsize=11)

    grid = np.array(heatmap_grid)  # [GRID_COLS, GRID_ROWS]
    smoothed = gaussian_filter(grid, sigma=HEATMAP_GAUSSIAN_SIGMA)
    # imshow expects [rows, cols] with origin at the array's [0,0] -- grid
    # is [x_cols, y_rows], so transpose to [y_rows, x_cols] and let
    # `extent`/`origin='lower'` map array indices onto pitch meters
    # directly (col index -> x, row index -> y), matching
    # `habit_memory.py`'s own `col = x // CELL_WIDTH_METERS` binning
    # convention exactly.
    ax.imshow(
        smoothed.T,
        extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
        origin="lower",
        cmap="hot",
        alpha=0.75,
        aspect="auto",
        zorder=2,
    )

    # Milestone 44's validation sweep found this heatmap is otherwise
    # visually IDENTICAL in kind (a smoothed, colored grid) whether it's
    # `habit_memory.generate_player_heatmap`'s real computation or its own
    # uniform cold-start fallback (< MIN_HISTORICAL_EVENTS qualifying
    # events) -- the fallback previously only printed a console warning,
    # invisible to anyone looking at just the rendered image.
    if used_uniform_fallback:
        ax.text(
            PITCH_LENGTH / 2, PITCH_WIDTH + 5,
            f"UNIFORM FALLBACK ({event_count} events, below the 20-event cold-start threshold)",
            color="#ff4444", ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=5,
        )


def _draw_summary_text(ax, player_report: dict) -> None:
    ax.axis("off")
    lines = [
        f"Player ID: {player_report['player_id']}",
        "",
        f"Primary position: {player_report['primary_position'] or 'n/a'}",
        f"Total minutes played: {player_report['total_minutes_played']:.1f}",
        f"Primary formation: {player_report['primary_formation'] or 'n/a'}",
        "",
        f"Matches with data: {player_report['matches_with_data']}/{player_report['matches_requested']}",
        f"Matches player appeared in: {player_report['matches_player_appeared_in']}",
        "",
        f"Positional distribution sample: {player_report.get('positional_distribution_event_count', 'n/a')} events",
        f"Heatmap sample: {player_report.get('heatmap_event_count', 'n/a')} events"
        + (" (UNIFORM FALLBACK)" if player_report.get("heatmap_used_uniform_fallback") else ""),
    ]

    formation_minutes = player_report.get("formation_minutes") or {}
    if formation_minutes:
        lines.append("")
        lines.append("Formation minutes:")
        for formation, minutes in sorted(formation_minutes.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {formation}: {minutes:.1f} min")

    ax.text(
        0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
        fontsize=10, color="black", family="monospace",
    )


def render_player_dashboard(player_report: dict, output_path: str) -> None:
    """Renders Milestone 40's `generate_player_report` output as a single
    static PNG dashboard: a positional-distribution pitch diagram (left),
    and summary-stats text + a smoothed aggregate heatmap (right).
    """
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"Player Report -- player_id={player_report['player_id']}", fontsize=15, y=0.98)

    grid = GridSpec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.4], figure=fig)

    ax_positions = fig.add_subplot(grid[:, 0])
    _draw_positional_distribution(
        ax_positions,
        player_report.get("positional_distribution", {}),
        player_report.get("positional_distribution_event_count", 0),
    )

    ax_text = fig.add_subplot(grid[0, 1])
    _draw_summary_text(ax_text, player_report)

    ax_heatmap = fig.add_subplot(grid[1, 1])
    _draw_heatmap(
        ax_heatmap,
        player_report.get("heatmap_grid", [[0.0] * GRID_ROWS] * GRID_COLS),
        player_report.get("heatmap_event_count", 0),
        player_report.get("heatmap_used_uniform_fallback", False),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)


# ============================================================================
# Shot map (additive new feature): pure RENDERING layer over
# `player_report.generate_player_shot_map`'s output, exactly like
# `render_player_dashboard` above is over `generate_player_report`'s --
# does not recompute, adjust, or otherwise touch any shot-map value.
# Nothing above this line (render_player_dashboard and everything it
# calls) is modified.
# ============================================================================


def _draw_shot_scatter(ax, shots: list[dict], low_sample: bool) -> None:
    draw_pitch_outline(ax)
    ax.set_xlim(SHOT_MAP_X_MIN, PITCH_LENGTH + 2)  # attacking-third crop -- see module-level comment
    ax.set_title(f"Shot map ({len(shots)} shots)", color="white", fontsize=11)

    if not shots:
        ax.text(
            (SHOT_MAP_X_MIN + PITCH_LENGTH) / 2, PITCH_WIDTH / 2, "No shot data",
            color="white", ha="center", va="center",
        )
        return

    # Milestone 44's validation-sweep convention, reused verbatim (same
    # banner style `_draw_positional_distribution`'s LOW SAMPLE text and
    # `_draw_heatmap`'s UNIFORM FALLBACK text already use) -- not a new,
    # separately-invented low-sample visual.
    if low_sample:
        ax.text(
            (SHOT_MAP_X_MIN + PITCH_LENGTH) / 2, PITCH_WIDTH + 5,
            f"LOW SAMPLE ({len(shots)} shot{'s' if len(shots) != 1 else ''}) -- not a confident shot pattern",
            color="#ff4444", ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=5,
        )

    xg_values = [s["statsbomb_xg"] for s in shots]
    max_xg = max(xg_values)
    for s in shots:
        x, y = s["location"][0], s["location"][1]
        xg = s["statsbomb_xg"]
        # Same min/max-radius interpolation pattern
        # `_draw_positional_distribution` already uses for share -> radius,
        # applied here to xg -> radius (0 xg -> MIN, max observed xg ->
        # MAX -- relative to THIS shot set, not a fixed absolute scale,
        # since a single low-xG-heavy player's shots would otherwise all
        # render at the same tiny size).
        radius = MIN_SHOT_DOT_RADIUS_M + (MAX_SHOT_DOT_RADIUS_M - MIN_SHOT_DOT_RADIUS_M) * (
            xg / max_xg if max_xg > 0 else 0.0
        )
        if s["is_goal"]:
            ax.scatter(
                [x], [y], s=radius**2 * 12, color=GOAL_SHOT_COLOR, edgecolors="black",
                linewidths=0.8, alpha=0.9, zorder=4,
            )
        else:
            ax.scatter(
                [x], [y], s=radius**2 * 12, facecolors="none", edgecolors=NO_GOAL_SHOT_EDGE_COLOR,
                linewidths=1.1, alpha=0.85, zorder=3,
            )

    # Reference-image-style legend: two representative markers (goal /
    # no-goal), not one per real shot -- explains the fill convention
    # without cluttering the plot with a full xG size legend (the sidebar
    # stats panel already states xg_per_shot numerically).
    ax.scatter([], [], color=GOAL_SHOT_COLOR, edgecolors="black", linewidths=0.8, label="Goal")
    ax.scatter([], [], facecolors="none", edgecolors=NO_GOAL_SHOT_EDGE_COLOR, linewidths=1.1, label="No goal")
    ax.legend(loc="lower left", fontsize=7, facecolor="#2e8b3d", labelcolor="white", framealpha=0.6)


def _draw_shot_map_stats(ax, shot_map: dict) -> None:
    ax.axis("off")
    total_shots = shot_map["total_shots"]
    xg_per_shot = shot_map["xg_per_shot"]
    lines = [
        f"Player ID: {shot_map['player_id']}",
        "",
        f"Total shots: {total_shots}",
        f"Goals: {shot_map['goals']}",
        f"Sum statsbomb_xg: {shot_map['sum_statsbomb_xg']:.2f}",
        f"xG per shot: {xg_per_shot:.3f}" if xg_per_shot is not None else "xG per shot: n/a",
        "",
        "Shots by body part:",
    ]
    body_part_counts = shot_map.get("shots_by_body_part") or {}
    if body_part_counts:
        for body_part, count in sorted(body_part_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {body_part}: {count}")
    else:
        lines.append("  n/a")

    lines.append("")
    lines.append(
        "xG = StatsBomb's own real statsbomb_xg per shot -- NOT this"
    )
    lines.append(
        "project's DeepHit threat model (a different quantity)."
    )

    ax.text(
        0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
        fontsize=10, color="black", family="monospace",
    )


def render_shot_map(shot_map_data: dict, output_path: str) -> None:
    """Renders `player_report.generate_player_shot_map`'s output as a
    single static PNG: an attacking-third pitch diagram with shot markers
    sized by `statsbomb_xg` and colored/filled by outcome (left), and a
    stats sidebar -- total shots, body-part breakdown, goals, total xG,
    xG per shot (right). A NEW, STANDALONE function alongside
    `render_player_dashboard`, not a modification of it.

    ADR-021 condition 2 (no raw StatsBomb data exposed to PUBLIC
    deployments): this function plots each individual shot's exact
    location, so it must only ever be called for LOCAL/private use --
    `render_shot_map_aggregated` below is the PUBLIC-deployment
    counterpart. Not gated inside this function itself (that decision
    belongs to the caller -- see `api.py`'s `PUBLIC_DEPLOYMENT` flag and
    `dashboard.py`'s shot-map panel).
    """
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(f"Shot Map -- player_id={shot_map_data['player_id']}", fontsize=15, y=0.98)

    grid = GridSpec(1, 2, width_ratios=[1.5, 1.0], figure=fig)

    ax_shots = fig.add_subplot(grid[0, 0])
    _draw_shot_scatter(ax_shots, shot_map_data.get("shots", []), shot_map_data.get("shot_map_used_low_sample_flag", False))

    ax_stats = fig.add_subplot(grid[0, 1])
    _draw_shot_map_stats(ax_stats, shot_map_data)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)


# ============================================================================
# Shot map, AGGREGATED variant (ADR-021 condition-2 compliance fix): pure
# RENDERING layer over `player_report.generate_player_shot_map_aggregated`'s
# output -- binned density/xG grids only, no individual shot marker
# anywhere. Consumes ONLY `shot_density_grid`/`mean_xg_grid` and the same
# summary scalars `_draw_shot_map_stats` above already reads; never reads
# a `"shots"` key (that field does not exist on this function's input
# dict in the first place). Nothing above this line is modified.
# ============================================================================


def _draw_shot_density_heatmap(ax, shot_density_grid: list[list[float]], total_shots: int, low_sample: bool) -> None:
    draw_pitch_outline(ax)
    ax.set_xlim(SHOT_MAP_X_MIN, PITCH_LENGTH + 2)  # same attacking-third crop as the raw scatter version
    ax.set_title(f"Shot density (aggregated, {total_shots} shots)", color="white", fontsize=11)

    grid = np.array(shot_density_grid)
    if grid.sum() == 0:
        ax.text(
            (SHOT_MAP_X_MIN + PITCH_LENGTH) / 2, PITCH_WIDTH / 2, "No shot data",
            color="white", ha="center", va="center",
        )
        return

    # Same gaussian-smoothed imshow convention `_draw_heatmap` already uses
    # for the aggregate positional heatmap panel -- styled consistently,
    # not a new visual language invented for this one panel.
    smoothed = gaussian_filter(grid, sigma=HEATMAP_GAUSSIAN_SIGMA)
    ax.imshow(
        smoothed.T, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH], origin="lower",
        cmap="hot", alpha=0.75, aspect="auto", zorder=2,
    )

    # Same LOW SAMPLE banner convention as _draw_shot_scatter/_draw_heatmap
    # -- reused verbatim, not reinvented for this aggregated panel.
    if low_sample:
        ax.text(
            (SHOT_MAP_X_MIN + PITCH_LENGTH) / 2, PITCH_WIDTH + 5,
            f"LOW SAMPLE ({total_shots} shot{'s' if total_shots != 1 else ''}) -- not a confident shot pattern",
            color="#ff4444", ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=5,
        )


def _draw_mean_xg_heatmap(ax, mean_xg_grid: list[list[float | None]]) -> None:
    draw_pitch_outline(ax)
    ax.set_xlim(SHOT_MAP_X_MIN, PITCH_LENGTH + 2)
    ax.set_title("Mean statsbomb_xg per cell (aggregated)", color="white", fontsize=11)

    # None-for-unobserved-cell handling matches _draw_control_heatmap's own
    # masked-array convention exactly (team_visualizer.py) -- a cell with no
    # shots is drawn as genuinely blank, not a fabricated 0 xG reading.
    grid = np.array([[np.nan if v is None else v for v in col] for col in mean_xg_grid], dtype=np.float64)
    populated = grid[~np.isnan(grid)]
    if populated.size == 0:
        ax.text(
            (SHOT_MAP_X_MIN + PITCH_LENGTH) / 2, PITCH_WIDTH / 2, "No shot data",
            color="white", ha="center", va="center",
        )
        return

    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(
        masked.T, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH], origin="lower",
        cmap="YlOrRd", vmin=0.0, vmax=max(float(populated.max()), 0.01), alpha=0.8, aspect="auto", zorder=2,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean statsbomb_xg", fontsize=8)


def render_shot_map_aggregated(shot_map_aggregated_data: dict, output_path: str) -> None:
    """Renders `player_report.generate_player_shot_map_aggregated`'s
    output as a single static PNG: a shot-DENSITY heatmap (left) and a
    mean-`statsbomb_xg`-per-cell heatmap (middle), both binned into the
    same grid `_draw_heatmap`'s aggregate positional panel already uses,
    plus the same stats sidebar `render_shot_map` uses (right) -- no
    individual shot marker anywhere in this figure. The PUBLIC-deployment
    counterpart to `render_shot_map` above (ADR-021 condition 2).
    """
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(
        f"Shot Map (aggregated) -- player_id={shot_map_aggregated_data['player_id']}", fontsize=15, y=0.98
    )

    grid = GridSpec(1, 3, width_ratios=[1.3, 1.3, 1.0], figure=fig)

    ax_density = fig.add_subplot(grid[0, 0])
    _draw_shot_density_heatmap(
        ax_density,
        shot_map_aggregated_data.get("shot_density_grid", [[0.0] * GRID_ROWS] * GRID_COLS),
        shot_map_aggregated_data.get("total_shots", 0),
        shot_map_aggregated_data.get("shot_map_used_low_sample_flag", False),
    )

    ax_xg = fig.add_subplot(grid[0, 1])
    _draw_mean_xg_heatmap(ax_xg, shot_map_aggregated_data.get("mean_xg_grid", [[None] * GRID_ROWS] * GRID_COLS))

    ax_stats = fig.add_subplot(grid[0, 2])
    _draw_shot_map_stats(ax_stats, shot_map_aggregated_data)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)
