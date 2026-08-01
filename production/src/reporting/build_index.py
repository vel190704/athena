"""Milestone 45 (new reporting track): a minimal static HTML browsing
index for rendered player/team dashboards.

Pure INDEXING/ORCHESTRATION layer -- does not modify, recompute, or
reimplement anything in `player_report.py`, `player_visualizer.py`,
`team_report.py`, or `team_visualizer.py`. This file only IMPORTS and
CALLS those existing functions exactly as built, and writes their
already-rendered output (plus a small metadata sidecar this module needs
for the index) to a persistent directory.

WHY A `manifest.json` SIDECAR, NOT PURE FILENAME SCANNING: a PNG file
alone carries no information about whether it's a player or a team
dashboard, or whether its underlying report was low-sample
(`heatmap_used_uniform_fallback`, added by Milestone 44's validation
sweep). `generate_report_index` needs that metadata to group entries and
draw the "Low Sample" tag Step 1.2 requires -- a manifest written
alongside the PNGs at render time (by `render_validation_dashboards`
below) is the simplest way to carry it, without parsing meaning out of
filenames (fragile) or re-deriving it by re-running the report generators
every time the index is rebuilt (wasteful, and would silently go stale
the moment a PNG is regenerated without re-running the indexer).

WHY THESE DASHBOARDS NEEDED RE-RENDERING AT ALL: Milestone 42's and
Milestone 44's dashboard PNGs were written to ephemeral per-session
scratch paths, never a persistent project location -- there was nothing
for this module to "scan" until `render_validation_dashboards` (below)
re-ran the existing, unmodified render pipeline into a real, persistent
directory (`reports_output/`, gitignored -- a generated-artifacts
directory, same convention as `mlruns/`/`data/raw/`).
"""

import html
import json
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"

# HAND-PICKED, stated explicitly (this project's standing practice): teams
# have no direct equivalent of `heatmap_used_uniform_fallback` --
# `generate_team_report` exposes `matches_used`/`matches_requested` but no
# per-frame count (see `team_visualizer.py`'s own caption caveat). A team
# built from this few matches is flagged "Low Sample" here as a reasonable,
# analogous heuristic, not a re-derivation of a real statistical threshold.
TEAM_LOW_SAMPLE_MATCH_THRESHOLD = 2


def render_validation_dashboards(output_dir: str) -> None:
    """POPULATION HELPER -- NOT part of the core indexing deliverable
    (`generate_report_index` below). Re-renders the same 8 profiles
    covered by `REPORTING_FINDINGS.md` (Messi + the 6 Milestone 44
    validation-sweep cases + Argentina) into `output_dir`, calling
    `generate_player_report`/`render_player_dashboard`/
    `generate_team_report`/`render_team_dashboard` EXACTLY as built, and
    writes `manifest.json` alongside them with the metadata
    `generate_report_index` needs.
    """
    from production.src.reporting.player_report import generate_player_report
    from production.src.reporting.player_visualizer import render_player_dashboard
    from production.src.reporting.team_report import generate_team_report
    from production.src.reporting.team_visualizer import render_team_dashboard

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    leverkusen_matches = [3895052, 3895060, 3895067, 3895074, 3895086, 3895095, 3895107, 3895113]

    player_cases = [
        (5503, "Lionel Messi", [3773386, 3857264, 3857289, 3857300, 3869151, 3869321, 3869519, 3869685]),
        (32602, "Kristijan Jakic (0 events)", [3869684]),
        (99479, "Yu-Min Cho (1 event)", [3857262]),
        (33401, "Amine Adli (multi-position)", leverkusen_matches),
        (8667, "Lukas Hradecky (GK)", leverkusen_matches),
    ]

    team_cases = [
        ("Argentina", [3857264, 3857289, 3857300, 3869151]),
        ("Barcelona", [3773386]),
        ("Bayer Leverkusen", leverkusen_matches),
    ]

    manifest = []

    for player_id, display_name, match_ids in player_cases:
        report = generate_player_report(player_id, match_ids)
        slug = display_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        png_filename = f"player_{player_id}_{slug}.png"
        render_player_dashboard(report, str(output_path / png_filename))

        low_sample = bool(report.get("heatmap_used_uniform_fallback"))
        reason = None
        if low_sample:
            reason = (
                f"heatmap_used_uniform_fallback=True "
                f"(heatmap_event_count={report.get('heatmap_event_count')}, "
                f"positional_distribution_event_count={report.get('positional_distribution_event_count')})"
            )
        manifest.append(
            {
                "type": "player",
                "display_name": display_name,
                "png_filename": png_filename,
                "low_sample": low_sample,
                "low_sample_reason": reason,
            }
        )
        print(f"[build_index] rendered player {display_name!r} -> {png_filename} (low_sample={low_sample})")

    for team_name, match_ids in team_cases:
        report = generate_team_report(team_name, match_ids)
        slug = team_name.lower().replace(" ", "_")
        png_filename = f"team_{slug}.png"
        render_team_dashboard(report, str(output_path / png_filename))

        low_sample = report["matches_used"] <= TEAM_LOW_SAMPLE_MATCH_THRESHOLD
        reason = (
            f"matches_used={report['matches_used']} (<= {TEAM_LOW_SAMPLE_MATCH_THRESHOLD})" if low_sample else None
        )
        manifest.append(
            {
                "type": "team",
                "display_name": team_name,
                "png_filename": png_filename,
                "low_sample": low_sample,
                "low_sample_reason": reason,
            }
        )
        print(f"[build_index] rendered team {team_name!r} -> {png_filename} (low_sample={low_sample})")

    with open(output_path / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f, indent=2)


def generate_report_index(output_dir: str) -> str:
    """Step 1: scans `output_dir` for rendered dashboard PNGs (via
    `manifest.json`, written alongside them by `render_validation_dashboards`
    -- see this module's docstring for why a manifest, not filename
    parsing, is the metadata source) and writes a single static
    `index.html` into the same directory, grouped into "Players"/"Teams"
    sections, each entry linking to and thumbnailing its dashboard image.

    Per Step 1.2: an entry whose manifest record has `low_sample=True`
    (mirroring `heatmap_used_uniform_fallback` for players, or a
    similarly-low match count for teams -- see `TEAM_LOW_SAMPLE_MATCH_
    THRESHOLD`) gets a visible "⚠ Low Sample" tag directly on its card,
    not just in the underlying dashboard image -- the index is the first
    thing anyone sees, so it must not hide the same caveat the dashboard
    itself already surfaces.

    Genuinely minimal, per Step 1.3: plain HTML + inlined CSS, no
    framework, no build step, no server -- opens directly from the
    filesystem in any browser (`file://.../index.html`).

    PNGs found in `output_dir` with NO corresponding manifest entry are
    still listed (never silently dropped), under an "Unlabeled" section,
    rather than assumed to be low-sample or high-sample without evidence
    either way.

    Returns the path to the written `index.html`.
    """
    output_path = Path(output_dir)
    manifest_path = output_path / MANIFEST_FILENAME

    entries: list[dict] = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            entries = json.load(f)
    else:
        print(f"[build_index] no {MANIFEST_FILENAME} found in {output_dir} -- every PNG will be Unlabeled.")

    known_filenames = {e["png_filename"] for e in entries}
    for png_file in sorted(output_path.glob("*.png")):
        if png_file.name not in known_filenames:
            entries.append(
                {
                    "type": "unknown",
                    "display_name": png_file.stem,
                    "png_filename": png_file.name,
                    "low_sample": False,
                    "low_sample_reason": None,
                }
            )

    players = [e for e in entries if e["type"] == "player"]
    teams = [e for e in entries if e["type"] == "team"]
    unknown = [e for e in entries if e["type"] == "unknown"]

    def _card(entry: dict) -> str:
        name = html.escape(entry["display_name"])
        filename = html.escape(entry["png_filename"])
        tag = ""
        if entry.get("low_sample"):
            reason = html.escape(entry.get("low_sample_reason") or "low sample")
            tag = f'<span class="low-sample-tag" title="{reason}">&#9888; Low Sample</span>'
        return f"""
        <div class="card">
          <a href="{filename}"><img src="{filename}" loading="lazy" alt="{name}"></a>
          <div class="card-label">{name}{tag}</div>
        </div>"""

    def _section(title: str, items: list[dict]) -> str:
        if not items:
            return ""
        cards = "\n".join(_card(e) for e in sorted(items, key=lambda e: e["display_name"]))
        return f'<h2>{html.escape(title)}</h2>\n<div class="grid">{cards}\n</div>\n'

    body = _section("Players", players) + _section("Teams", teams) + _section("Unlabeled", unknown)
    if not (players or teams or unknown):
        body = "<p><em>No rendered dashboards found in this directory.</em></p>"

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Project Athena -- Reporting Dashboards</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem; background: #f5f5f5; color: #222; }}
  h1 {{ margin-bottom: 0.2rem; }}
  p.subtitle {{ color: #555; margin-top: 0; }}
  h2 {{ margin-top: 2.2rem; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 1.2rem; margin-top: 1rem; }}
  .card {{ width: 320px; background: white; border: 1px solid #ddd; border-radius: 6px; padding: 0.6rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card img {{ width: 100%; border-radius: 4px; display: block; }}
  .card-label {{ margin-top: 0.5rem; font-size: 0.95rem; }}
  .low-sample-tag {{ display: inline-block; background: #ffe1e1; color: #a80000; border: 1px solid #ffb3b3;
                      border-radius: 4px; padding: 1px 6px; font-size: 0.8rem; margin-left: 0.5rem; font-weight: bold; }}
</style>
</head>
<body>
<h1>Project Athena -- Reporting Dashboards</h1>
<p class="subtitle">Milestone 40/42/43/44 player &amp; team analysis reports. Static, local, offline HTML --
open directly from the filesystem, no server required.</p>
{body}
</body>
</html>
"""

    index_path = output_path / "index.html"
    index_path.write_text(doc)
    return str(index_path)
