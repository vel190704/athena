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

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from production.src.reporting.pitch_diagram import draw_pitch_outline
from production.src.pipeline.feature_extractor import PITCH_LENGTH, PITCH_WIDTH
from production.src.pipeline.habit_memory import GRID_COLS, GRID_ROWS, MIN_HISTORICAL_EVENTS

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
            print(f"[player_visualizer] no plot location for position {position_name!r} -- skipping dot.")
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
