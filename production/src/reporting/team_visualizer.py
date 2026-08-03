"""Milestone 42 (new reporting track, Step 2): Team Report Visualization.

Pure RENDERING layer over Milestone 40's `generate_team_report` output --
does not recompute, adjust, or otherwise touch any report value. Nothing
in `team_report.py` is modified or reimplemented here (see
`player_visualizer.py`'s docstring for the matplotlib-vs-PIL rationale,
identical here).

SAMPLE-SIZE CAVEAT, STATED HERE BECAUSE IT CANNOT BE FIXED IN THIS FILE:
`generate_team_report`'s return dict exposes `matches_used`/
`matches_requested` but NOT a per-FRAME count (the raw per-frame value
lists are reduced to means before the dict is returned). This module
therefore captions with the match-level counts only, and says so
explicitly, rather than fabricating or omitting a frame count -- adding
that field would mean modifying `team_report.py`'s return contract, which
this milestone's scope explicitly excludes.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from production.src.pipeline.feature_extractor import PITCH_LENGTH, PITCH_WIDTH
from production.src.reporting.pitch_diagram import draw_pitch_outline


def _draw_control_heatmap(ax, control_heatmap_grid: list[list[float | None]]) -> None:
    draw_pitch_outline(ax)
    ax.set_title("Pitch-control heatmap (weak vs. strong zones)", color="white", fontsize=11)

    grid = np.array(
        [[np.nan if v is None else v for v in col] for col in control_heatmap_grid], dtype=np.float64
    )  # [GRID_COLS, GRID_ROWS]

    populated = grid[~np.isnan(grid)]
    if populated.size == 0:
        ax.text(PITCH_LENGTH / 2, PITCH_WIDTH / 2, "No control data", color="white", ha="center", va="center")
        return

    # Diverging around this SAMPLE's own mean control (not a fixed 0.5) --
    # aggregate control values on real data skew well below 0.5 (a team is
    # rarely the dominant side across the whole sparse-masked grid), so
    # centering at 0.5 would paint almost everything "weak" and hide the
    # actual relative pattern this heatmap exists to surface.
    center = float(populated.mean())
    vmin, vmax = float(populated.min()), float(populated.max())
    if vmin == vmax:
        vmin, vmax = center - 0.01, center + 0.01
    norm = TwoSlopeNorm(vmin=min(vmin, center - 1e-6), vcenter=center, vmax=max(vmax, center + 1e-6))

    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(
        masked.T,
        extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
        origin="lower",
        cmap="RdYlGn",
        norm=norm,
        alpha=0.8,
        aspect="auto",
        zorder=2,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"mean control (sample mean={center:.3f})", fontsize=8)


def _draw_threat_by_zone(ax, threat_by_pitch_zone: dict) -> None:
    zones = ["defensive_third", "middle_third", "attacking_third"]
    values = [threat_by_pitch_zone.get(z) for z in zones]
    labels = [z.replace("_", " ") for z in zones]
    plot_values = [v if v is not None else 0.0 for v in values]

    bars = ax.bar(labels, plot_values, color=["#c0392b", "#f39c12", "#27ae60"])
    for bar, v in zip(bars, values):
        label = "n/a" if v is None else f"{v*100:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
    ax.set_title("Predicted threat by pitch zone", fontsize=10)
    ax.set_ylabel("mean cumulative incidence")
    ax.set_ylim(0, max(plot_values + [0.01]) * 1.3)


def _draw_threat_by_phase(ax, threat_by_game_phase: dict) -> None:
    phases = list(threat_by_game_phase.keys())  # generate_team_report already returns these sorted numerically
    values = [threat_by_game_phase[p] for p in phases]

    ax.plot(phases, values, marker="o", color="#2980b9")
    ax.set_title("Predicted threat by game phase", fontsize=10)
    ax.set_ylabel("mean cumulative incidence")
    ax.tick_params(axis="x", rotation=45, labelsize=7)


def render_team_dashboard(team_report: dict, output_path: str) -> None:
    """Renders Milestone 40's `generate_team_report` output as a single
    static PNG dashboard: the pitch-control weak/strong-zone heatmap
    (left), and the threat-by-zone / threat-by-phase pattern (right),
    with a sample-size caption.
    """
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"Team Report -- {team_report['team_name']}", fontsize=15, y=0.98)

    grid = GridSpec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0], figure=fig)

    ax_heatmap = fig.add_subplot(grid[:, 0])
    _draw_control_heatmap(ax_heatmap, team_report.get("control_heatmap_grid", []))

    ax_zone = fig.add_subplot(grid[0, 1])
    _draw_threat_by_zone(ax_zone, team_report.get("threat_by_pitch_zone", {}))

    ax_phase = fig.add_subplot(grid[1, 1])
    _draw_threat_by_phase(ax_phase, team_report.get("threat_by_game_phase", {}))

    caption = (
        f"Built from {team_report['matches_used']} matches (of {team_report['matches_requested']} requested). "
        "Per-frame count is not exposed by generate_team_report's current return contract -- "
        "match-level count shown for transparency about sample size, not a frame-level one."
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color="dimgray", wrap=True)

    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)
