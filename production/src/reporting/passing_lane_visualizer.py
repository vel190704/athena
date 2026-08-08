"""Passing Lane Visualizer Report Visualization.

Pure RENDERING layer over `team_report.generate_team_passing_lanes`/
`generate_team_passing_lanes_aggregated`'s output -- does not recompute,
adjust, or otherwise touch any report value. Nothing in `team_report.py`
is modified or reimplemented here. Same matplotlib/Agg convention as
`pass_network_visualizer.py`/`team_visualizer.py` (see those modules' own
docstrings for the library-choice reasoning); reuses `pitch_diagram.
draw_pitch_outline` exactly as `pass_network_visualizer.py` already does.

VISUALLY DISTINCT FROM Pass Network's edges, deliberately (they measure
different things -- see team_report.py's own Passing Lane Visualizer
section header comment): Pass Network draws uniform WHITE lines sized by
real completed-pass VOLUME. This module instead draws lines on a
red(closed)-to-green(open) diverging colormap keyed to each lane's own
`mean_lane_openness` SCORE -- a line's color here answers "how contested
was this space," not "how often was this pass made."
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from production.src.reporting.pitch_diagram import draw_pitch_outline

MIN_NODE_RADIUS_M = 1.5
MAX_NODE_RADIUS_M = 3.5
MIN_LANE_LINEWIDTH = 0.8
MAX_LANE_LINEWIDTH = 4.0
NODE_COLOR = "#ffcc00"
NODE_LABEL_COLOR = "black"
# Diverging red(0.0, closed)->yellow(0.5)->green(1.0, open) colormap,
# keyed to mean_lane_openness directly (already a natural [0, 1] scale --
# no further normalization needed against this match/season's own min/max,
# unlike Pass Network's volume-based line-width scaling, which DOES
# normalize against the request's own max since raw completed-pass counts
# have no natural fixed ceiling the way a control PROBABILITY does).
LANE_COLORMAP = matplotlib.colormaps["RdYlGn"]


def _short_label(name: str) -> str:
    """Last whitespace-separated token of a player's full name -- same
    convention `pass_network_visualizer._short_label` already
    established, reimplemented locally rather than imported across
    modules (this project's convention for a small, module-private
    helper -- see e.g. team_report.py's own `_build_pitch_grid`
    docstring for the same reasoning applied elsewhere)."""
    return name.split()[-1]


def _draw_team_passing_lanes(ax, team_name: str, nodes: list[dict], lanes: list[dict]) -> None:
    draw_pitch_outline(ax)
    ax.set_title(f"{team_name} -- passing lane openness ({len(lanes)} pairs)", color="white", fontsize=11)

    if not nodes:
        ax.text(50, 34, "No passing lane data", color="white", ha="center", va="center")
        return

    node_by_id = {n["player_id"]: n for n in nodes}
    max_samples = max((lane["n_pass_samples"] for lane in lanes), default=1)

    for lane in lanes:
        passer = node_by_id.get(lane["passer_id"])
        recipient = node_by_id.get(lane["recipient_id"])
        if passer is None or recipient is None:
            continue
        x1, y1 = passer["avg_location"]
        x2, y2 = recipient["avg_location"]
        # Line WIDTH still scales with real sample count (so a
        # well-supported lane visually stands out from a single-sample
        # one) -- but COLOR (the actual openness signal this feature is
        # built to show) is keyed to mean_lane_openness, not volume.
        linewidth = MIN_LANE_LINEWIDTH + (MAX_LANE_LINEWIDTH - MIN_LANE_LINEWIDTH) * (
            lane["n_pass_samples"] / max_samples if max_samples > 0 else 0.0
        )
        color = LANE_COLORMAP(lane["mean_lane_openness"])
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.75, zorder=2)

    for node in nodes:
        x, y = node["avg_location"]
        ax.scatter(
            [x], [y], s=MIN_NODE_RADIUS_M**2 * 12, color=NODE_COLOR,
            edgecolors="black", linewidths=0.8, zorder=3,
        )
        ax.text(
            x, y, _short_label(node["name"]), color=NODE_LABEL_COLOR, ha="center", va="center",
            fontsize=6.5, fontweight="bold", zorder=4,
        )


def render_passing_lanes(passing_lanes_data: dict, output_path: str) -> None:
    """Renders `team_report.generate_team_passing_lanes`'s output as a
    single static PNG: each player's own real average location, connected
    by lines colored red(closed)->green(open) by each pair's real mean
    lane-openness score.

    ADR-021 condition 2 (no raw StatsBomb data exposed to PUBLIC
    deployments): this function plots each player's real individual
    average location, so it must only ever be called for LOCAL/private
    use -- `render_passing_lanes_aggregated` below is the PUBLIC-
    deployment counterpart. Not gated inside this function itself (same
    convention `render_pass_network` already established -- that
    decision belongs to the caller, see `api.py`'s `PUBLIC_DEPLOYMENT`
    flag and `dashboard.py`'s passing-lanes panel).
    """
    team_name = passing_lanes_data["team_name"]
    nodes = passing_lanes_data.get("nodes", [])
    lanes = passing_lanes_data.get("lanes", [])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    _draw_team_passing_lanes(ax, team_name, nodes, lanes)

    sm = cm.ScalarMappable(cmap=LANE_COLORMAP)
    sm.set_array([0.0, 1.0])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean lane openness (0=closed, 1=open)", fontsize=8)

    fig.suptitle(
        f"Passing Lane Openness -- {passing_lanes_data['matches_used']} of "
        f"{passing_lanes_data['matches_requested']} matches used, "
        f"{passing_lanes_data['total_pass_samples_used']} real pass samples",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)


# ============================================================================
# Passing lanes, AGGREGATED variant (ADR-021 condition-2 compliance): pure
# RENDERING layer over `team_report.generate_team_passing_lanes_aggregated`'s
# output -- named pairs ranked by openness score, no location, no pitch
# diagram (there is nothing to place at a real coordinate once `nodes` is
# gone). Consumes ONLY `lanes`; never reads a `"nodes"` key (which does
# not exist on this function's input dict in the first place).
# ============================================================================


def render_passing_lanes_aggregated(passing_lanes_aggregated_data: dict, output_path: str) -> None:
    """Renders `team_report.generate_team_passing_lanes_aggregated`'s
    output: a horizontal bar chart of named (passer, recipient) pairs
    ranked by real mean lane-openness, bars colored the SAME red->green
    scale `render_passing_lanes` uses -- no pitch diagram, no location,
    mirroring `render_pass_network_aggregated`'s own bar-chart fallback.
    """
    team_name = passing_lanes_aggregated_data["team_name"]
    lanes = sorted(
        passing_lanes_aggregated_data.get("lanes", []), key=lambda lane: -lane["mean_lane_openness"]
    )[:20]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"{team_name} -- top passing lanes by openness (aggregated, no location)", fontsize=11)

    if not lanes:
        ax.axis("off")
        ax.text(0.5, 0.5, "No passing lane data", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = [f"{_short_label(lane['passer_name'])} -> {_short_label(lane['recipient_name'])}" for lane in lanes]
        openness = [lane["mean_lane_openness"] for lane in lanes]
        colors = [LANE_COLORMAP(v) for v in openness]
        y_pos = range(len(lanes))
        ax.barh(list(y_pos), openness, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean lane openness (0=closed, 1=open)", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)
