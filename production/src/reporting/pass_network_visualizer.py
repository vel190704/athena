"""Pass Network Report Visualization.

Pure RENDERING layer over `pass_network.generate_pass_network`/
`generate_pass_network_aggregated`'s output -- does not recompute, adjust,
or otherwise touch any report value. Nothing in `pass_network.py` is
modified or reimplemented here. Same matplotlib/Agg convention as
`player_visualizer.py`/`team_visualizer.py` (see that module's own
docstring for the library-choice reasoning).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from production.src.reporting.pitch_diagram import draw_pitch_outline

MIN_NODE_RADIUS_M = 1.5
MAX_NODE_RADIUS_M = 4.5
MIN_EDGE_LINEWIDTH = 0.6
MAX_EDGE_LINEWIDTH = 6.0
NODE_COLOR = "#ffcc00"
NODE_LABEL_COLOR = "black"
EDGE_COLOR = "white"


def _short_label(name: str) -> str:
    """Last whitespace-separated token of a player's full name -- a
    plotted 11-a-side pitch has no room for full names; matches the
    common real-world pass-network convention of labeling by surname."""
    return name.split()[-1]


def _draw_team_pass_network(ax, team_name: str, nodes: list[dict], edges: list[dict]) -> None:
    draw_pitch_outline(ax)
    ax.set_title(f"{team_name} ({len(nodes)} players)", color="white", fontsize=11)

    if not nodes:
        ax.text(50, 34, "No pass data", color="white", ha="center", va="center")
        return

    max_attempts = max(n["pass_attempts"] for n in nodes)
    max_completed = max((e["completed_passes"] for e in edges), default=1)
    node_by_id = {n["player_id"]: n for n in nodes}

    for edge in edges:
        passer = node_by_id.get(edge["passer_id"])
        recipient = node_by_id.get(edge["recipient_id"])
        if passer is None or recipient is None:
            continue
        x1, y1 = passer["avg_location"]
        x2, y2 = recipient["avg_location"]
        linewidth = MIN_EDGE_LINEWIDTH + (MAX_EDGE_LINEWIDTH - MIN_EDGE_LINEWIDTH) * (
            edge["completed_passes"] / max_completed if max_completed > 0 else 0.0
        )
        ax.plot([x1, x2], [y1, y2], color=EDGE_COLOR, linewidth=linewidth, alpha=0.55, zorder=2)

    for node in nodes:
        x, y = node["avg_location"]
        radius = MIN_NODE_RADIUS_M + (MAX_NODE_RADIUS_M - MIN_NODE_RADIUS_M) * (
            node["pass_attempts"] / max_attempts if max_attempts > 0 else 0.0
        )
        ax.scatter([x], [y], s=radius**2 * 12, color=NODE_COLOR, edgecolors="black", linewidths=0.8, zorder=3)
        ax.text(
            x, y, _short_label(node["name"]), color=NODE_LABEL_COLOR, ha="center", va="center",
            fontsize=6.5, fontweight="bold", zorder=4,
        )


def render_pass_network(pass_network_data: dict, output_path: str) -> None:
    """Renders `pass_network.generate_pass_network`'s output as a single
    static PNG: one pitch panel per team, nodes at each Starting XI
    player's own average location (sized by pass attempts), edges between
    teammates sized by real completed-pass count.

    ADR-021 condition 2 (no raw StatsBomb data exposed to PUBLIC
    deployments): this function plots each player's real individual
    average location and real pairwise edge weights, so it must only ever
    be called for LOCAL/private use -- `render_pass_network_aggregated`
    below is the PUBLIC-deployment counterpart. Not gated inside this
    function itself (that decision belongs to the caller -- see `api.py`'s
    `PUBLIC_DEPLOYMENT` flag and `dashboard.py`'s pass-network panel).
    """
    match_id = pass_network_data["match_id"]
    nodes = pass_network_data.get("nodes", [])
    edges = pass_network_data.get("edges", [])
    teams = sorted({n["team"] for n in nodes}) if nodes else []

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"Pass Network -- match_id={match_id}", fontsize=15, y=0.98)
    grid = GridSpec(1, max(len(teams), 1), figure=fig)

    for i, team in enumerate(teams):
        ax = fig.add_subplot(grid[0, i])
        team_nodes = [n for n in nodes if n["team"] == team]
        team_ids = {n["player_id"] for n in team_nodes}
        team_edges = [e for e in edges if e["passer_id"] in team_ids and e["recipient_id"] in team_ids]
        _draw_team_pass_network(ax, team, team_nodes, team_edges)

    if not teams:
        ax = fig.add_subplot(grid[0, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, "No pass network data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)


# ============================================================================
# Pass network, AGGREGATED variant (ADR-021 condition-2 compliance):  pure
# RENDERING layer over `pass_network.generate_pass_network_aggregated`'s
# output -- per-player TOTALS only, no location, no pairwise edge weight.
# Consumes ONLY `player_summary`/the network-level scalar stats; never
# reads a `"nodes"`/`"edges"` key (neither field exists on this function's
# input dict in the first place).
# ============================================================================


def _draw_team_pass_totals(ax, team_name: str, player_summary: list[dict]) -> None:
    ax.set_title(f"{team_name} -- completed passes sent", color="black", fontsize=11)
    team_players = sorted(
        (p for p in player_summary if p["team"] == team_name),
        key=lambda p: -p["completed_passes_sent"],
    )
    if not team_players:
        ax.axis("off")
        ax.text(0.5, 0.5, "No pass data", ha="center", va="center", transform=ax.transAxes)
        return

    names = [_short_label(p["name"]) for p in team_players]
    sent = [p["completed_passes_sent"] for p in team_players]
    y_pos = range(len(team_players))
    ax.barh(list(y_pos), sent, color="#2e8b3d")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Completed passes sent", fontsize=9)


def render_pass_network_aggregated(pass_network_aggregated_data: dict, output_path: str) -> None:
    """Renders `pass_network.generate_pass_network_aggregated`'s output:
    one bar-chart panel per team (completed passes sent, per player,
    total only -- no location, no pairwise breakdown), plus a network-
    level summary line (player/edge counts, density, total completed
    passes) in the figure title.
    """
    match_id = pass_network_aggregated_data["match_id"]
    player_summary = pass_network_aggregated_data.get("player_summary", [])
    teams = sorted({p["team"] for p in player_summary}) if player_summary else []

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"Pass Network (aggregated) -- match_id={match_id} -- "
        f"{pass_network_aggregated_data.get('num_players', 0)} players, "
        f"{pass_network_aggregated_data.get('num_edges', 0)} edges, "
        f"density={pass_network_aggregated_data.get('network_density', 0.0):.2f}",
        fontsize=12, y=0.98,
    )
    grid = GridSpec(1, max(len(teams), 1), figure=fig)

    for i, team in enumerate(teams):
        ax = fig.add_subplot(grid[0, i])
        _draw_team_pass_totals(ax, team, player_summary)

    if not teams:
        ax = fig.add_subplot(grid[0, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, "No pass network data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=130, facecolor="white")
    plt.close(fig)
