"""Tactical Timeline UI visualization.

Pure RENDERING layer over `match_timeline.generate_match_timeline`'s
output -- does not recompute, adjust, or otherwise touch any of its
values. Same matplotlib/Agg convention as `pass_network_visualizer.py`/
`team_visualizer.py`/`player_visualizer.py` (see those modules' own
docstrings for the library-choice reasoning) -- reused here, not a new
visual language invented for this one chart.

WEAK-SPOT VOLUME, STATED EXPLICITLY: a real match's weak-spot instance
list can run into the thousands (9,171 across both teams for this
project's own validation match) -- the overwhelming majority single-frame
noise, per Weak-Spot Lifetime Analysis's own established finding. Plotting
every one as an individual Gantt span would be unreadable, not more
honest. This renderer shows only the TOP `MAX_WEAK_SPOTS_PLOTTED`
longest-duration instances (already duration-sorted by the source data),
the SAME "duration-sorted ranking surfaces genuine persistence above the
noise floor, don't plot the raw noise" convention the dashboard's own
Weak-Spot Lifetime panel table already established (its own "Top 20 of N
total" caption) -- stated on the chart itself, not silently truncated.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MAX_WEAK_SPOTS_PLOTTED = 20

_LANES = ("counter_attack", "build_up", "switch_of_play", "weak_spot")
_LANE_LABELS = {
    "counter_attack": "Counter Attack",
    "build_up": "Build-up Pattern",
    "switch_of_play": "Switch of Play",
    "weak_spot": f"Weak Spot (top {MAX_WEAK_SPOTS_PLOTTED} by duration)",
}
_LANE_Y = {signal: i for i, signal in enumerate(_LANES)}

_TEAM_COLORS = ("#ffcc00", "#3399ff")  # matches pass_network_visualizer's own NODE_COLOR palette family
_BACKGROUND_COLOR = "#1a1a1a"
_TEXT_COLOR = "white"


def render_match_timeline(timeline_data: dict, output_path: str) -> None:
    """Renders `match_timeline.generate_match_timeline`'s output as a
    single static PNG: one horizontal lane per signal type (Counter
    Attack / Build-up Pattern / Switch of Play as spans or point markers;
    Weak Spot as the top-N-by-duration spans, see module docstring), all
    plotted against the SAME continuous `display_start_minute`/
    `display_end_minute` axis the data assembly already computed, with a
    vertical dashed line marking the real, computed period-2 boundary
    (`period_1_max_minute`) -- never a fixed/assumed 45.0.

    Two colors distinguish the two teams (a fixed legend, not per-team
    computed) -- consistent regardless of which team happens to be
    `teams[0]`/`teams[1]` in this specific match's own alphabetical
    ordering.
    """
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=_BACKGROUND_COLOR)
    ax.set_facecolor(_BACKGROUND_COLOR)

    if timeline_data.get("no_data"):
        ax.text(0.5, 0.5, "No timeline data available", color=_TEXT_COLOR, ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, facecolor=_BACKGROUND_COLOR)
        plt.close(fig)
        return

    teams = timeline_data["teams"]
    team_color = {team: _TEAM_COLORS[i % len(_TEAM_COLORS)] for i, team in enumerate(teams)}

    entries = timeline_data["timeline_entries"]
    weak_spot_entries = sorted(
        (e for e in entries if e["signal"] == "weak_spot"),
        key=lambda e: -(e["display_end_minute"] - e["display_start_minute"]),
    )[:MAX_WEAK_SPOTS_PLOTTED]
    span_signals = {"counter_attack", "build_up"}

    for entry in [e for e in entries if e["signal"] != "weak_spot"] + weak_spot_entries:
        lane_y = _LANE_Y[entry["signal"]]
        color = team_color.get(entry["team"], "gray")
        start, end = entry["display_start_minute"], entry["display_end_minute"]

        if entry["signal"] in span_signals or entry["signal"] == "weak_spot":
            width = max(end - start, 0.15)  # a visible minimum width so genuinely short spans still render
            ax.barh(lane_y, width, left=start, height=0.6, color=color, edgecolor="black", linewidth=0.3, alpha=0.85)
        else:  # switch_of_play -- a point event, not a span
            ax.scatter([start], [lane_y], color=color, marker="^", s=60, edgecolors="black", linewidths=0.5, zorder=3)

    ax.axvline(
        timeline_data["period_1_max_minute"], color="white", linestyle="--", linewidth=1.0, alpha=0.6
    )
    ax.text(
        timeline_data["period_1_max_minute"], len(_LANES) - 0.4, "Half-time",
        color=_TEXT_COLOR, fontsize=8, ha="center", rotation=90, va="top",
    )

    ax.set_yticks([_LANE_Y[s] for s in _LANES])
    ax.set_yticklabels([_LANE_LABELS[s] for s in _LANES], color=_TEXT_COLOR)
    ax.set_xlabel("Match time (display minute, continuous across both periods)", color=_TEXT_COLOR)
    ax.tick_params(colors=_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(_TEXT_COLOR)
    ax.set_ylim(-0.6, len(_LANES) - 0.2)

    legend_handles = [Patch(facecolor=team_color[team], edgecolor="black", label=team) for team in teams]
    legend = ax.legend(handles=legend_handles, loc="upper right", facecolor=_BACKGROUND_COLOR, labelcolor=_TEXT_COLOR)
    legend.get_frame().set_edgecolor(_TEXT_COLOR)

    ax.set_title(
        f"Tactical Timeline -- match_id={timeline_data['match_id']} "
        f"({' vs '.join(teams)})",
        color=_TEXT_COLOR,
    )

    fig.tight_layout()
    fig.savefig(output_path, facecolor=_BACKGROUND_COLOR)
    plt.close(fig)
