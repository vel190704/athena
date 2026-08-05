"""Team Trend two-season comparison visualization -- the football-data.co.uk
-track counterpart to `team_visualizer.py` (StatsBomb-track). Pure
RENDERING layer over `team_trend_data.compare_team_trend_seasons`'s output
-- does not recompute, adjust, or otherwise touch any comparison value.
Nothing in `team_trend_data.py` is modified or reimplemented here (see
`player_visualizer.py`'s docstring for the matplotlib-vs-PIL rationale,
identical here).

SCOPE BOUNDARY, same as `team_trend_data.py`'s own: this file draws only
plain match-results/output bar charts (goals, points, shots, cards) --
no pitch diagram, no heatmap, no event-location plotting of any kind,
because the underlying data has no coordinates at all. Never imports
`pitch_diagram.py` or anything from `team_visualizer.py`.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Core counting/result stats shown as grouped bars (season_a vs season_b).
# Deliberately a SUBSET of team_trend_data.COMPARISON_METRICS -- shots-on-
# target/corners/fouls are still in the raw comparison dict (and the "Raw
# comparison data" JSON expander in dashboard.py), just not each given
# their own bar here, to keep the chart legible rather than 14 bars wide.
_BAR_METRICS = ["points", "wins", "draws", "losses", "goals_scored", "goals_conceded"]
_BAR_LABELS = {
    "points": "Points", "wins": "Wins", "draws": "Draws", "losses": "Losses",
    "goals_scored": "Goals For", "goals_conceded": "Goals Against",
}

_SEASON_A_COLOR = "#4a7fb5"
_SEASON_B_COLOR = "#e07b39"
_POSITIVE_DELTA_COLOR = "#2e8b3d"
_NEGATIVE_DELTA_COLOR = "#c0392b"


def _draw_no_data(ax, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=11, wrap=True)


def _draw_season_bars(ax, comparison: dict) -> None:
    stats_a, stats_b = comparison["season_a_stats"], comparison["season_b_stats"]
    label_a, label_b = comparison["season_a"], comparison["season_b"]

    x = range(len(_BAR_METRICS))
    width = 0.35
    values_a = [stats_a[m] for m in _BAR_METRICS]
    values_b = [stats_b[m] for m in _BAR_METRICS]

    ax.bar([i - width / 2 for i in x], values_a, width, label=label_a, color=_SEASON_A_COLOR)
    ax.bar([i + width / 2 for i in x], values_b, width, label=label_b, color=_SEASON_B_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_BAR_LABELS[m] for m in _BAR_METRICS], rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{comparison['team_name']}: {label_a} vs {label_b}", fontsize=11)
    ax.legend(fontsize=8)


def _draw_delta_bars(ax, comparison: dict) -> None:
    diff = comparison["diff_b_minus_a"]
    values = [diff[f"{m}_delta"] for m in _BAR_METRICS]
    colors = [_POSITIVE_DELTA_COLOR if v >= 0 else _NEGATIVE_DELTA_COLOR for v in values]

    bars = ax.bar(range(len(_BAR_METRICS)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(_BAR_METRICS)))
    ax.set_xticklabels([_BAR_LABELS[m] for m in _BAR_METRICS], rotation=30, ha="right", fontsize=8)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:+d}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=7,
        )
    ax.set_title(
        f"Delta ({comparison['season_b']} minus {comparison['season_a']})", fontsize=11,
    )
    # Same point-of-confusion clarification the dashboard's own caption
    # states (Feature 3.5) -- also baked directly into the rendered image
    # itself, not just the surrounding Streamlit page, since this PNG can
    # be saved/shared independently of the page it was generated on.
    ax.text(
        0.5, -0.32, "Negative = a real decrease vs. season_a, not an error.",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="dimgray",
    )


def render_team_trend_comparison(comparison: dict, output_path: str) -> None:
    """Renders `team_trend_data.compare_team_trend_seasons`'s output as a
    single static PNG: grouped bars for season_a vs season_b (left), and
    a delta bar chart (right) -- both restricted to plain match-results/
    output counting stats (`_BAR_METRICS`), matching this data source's
    own SCOPE BOUNDARY (no coordinates, no pitch-control physics
    anywhere in this file).

    Gracefully renders a "no data" message instead of crashing when
    either season wasn't found in any of the five covered leagues
    (`season_a_found`/`season_b_found` False) -- the same discipline
    every other renderer in this project's reporting track already
    applies to a missing/insufficient-sample case.
    """
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Team Trend Comparison -- {comparison['team_name']}", fontsize=15, y=0.98)

    if not comparison["season_a_found"] or not comparison["season_b_found"]:
        ax = fig.add_subplot(1, 1, 1)
        _draw_no_data(ax, comparison["summary"])
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_path, dpi=130, facecolor="white")
        plt.close(fig)
        return

    grid = GridSpec(1, 2, width_ratios=[1.0, 1.0], figure=fig)

    ax_bars = fig.add_subplot(grid[0, 0])
    _draw_season_bars(ax_bars, comparison)

    ax_delta = fig.add_subplot(grid[0, 1])
    _draw_delta_bars(ax_delta, comparison)

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)
