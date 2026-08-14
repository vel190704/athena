"""Milestones 17 & 19 (Module 3/9 UI), extended with the reporting track's
Streamlit integration: Project Athena's dashboard, now ten tabs wide
(Pass Network and Alerts History were added after this docstring's "five
tabs" count was first written but before this correction; Match Report and
Tactical Chat are this session's own additions; Home is the newest addition,
a pure read-only orientation layer positioned leftmost/first -- see their
own tab sections below for what each does).

"Live CV Monitor" holds the two ORIGINAL panels, unchanged in behavior:

  1. "Tactical What-If Simulator" (Milestone 19) -- a plain synchronous
     REST call to `/simulate` (Milestone 18), triggered by "Run Simulation".
     Fast, request/response, no blocking loop.
  2. "Live Tactical Threat Monitor" (Milestone 17) -- a WebSocket stream,
     triggered by "Start Stream". See the ARCHITECTURAL DECISION below.
     Extended (additive) with Tactical Momentum: a rolling-window
     smoothing + trend indicator computed CLIENT-SIDE over the same
     `threat_buffer` this panel already accumulates -- see
     `_compute_tactical_momentum`'s own docstring for the exact
     definitions (window sizes, thresholds, classification labels).
     Deliberately NOT persisted anywhere (no new `alert_store.py` table,
     no new endpoint) -- a live, ephemeral, per-connection computation,
     the same scope the existing threat chart itself already has; there
     is no reason to persist a client-side smoothing of an already-
     ephemeral live number. ADR-021 condition 2: inherits the compliance
     audit's existing "StatsBomb-sourced live tactical stream... already
     condition-2-compliant by construction" finding (see ADR-021's Update
     section) for BOTH `source=statsbomb` and `source=cv` -- this feature
     reads ONLY the already-compliant `threat_15s` scalar out of each
     message (never the `players`/`ball` fields `source=cv` messages also
     carry) and reduces it further (an average, then a difference of two
     averages), so it can only carry LESS information than the signal
     already found compliant, never more. No new gating needed.
     Further extended (additive, this session) with Match Segmentation: a
     discrete game-phase label ("Building Attack" / "Transition" /
     "Defensive Consolidation" / "Stable") derived ENTIRELY from Tactical
     Momentum's own already-computed `smoothed_now`/`classification`
     output -- see `classify_match_phase`'s own docstring (same
     `tactical_momentum.py` module) for the exact decision table and the
     majority-vote hysteresis window that prevents it from flipping on a
     single noisy message. Same ephemeral, unpersisted, ADR-021-exempt
     scope as Tactical Momentum, for the same reason (a further reduction
     of an already-exempt derived signal).

The four new tabs ("Player Reports", "Team Reports", "Team Trends",
"Team Comparison") are a pure UI WIRING layer over the existing reporting
modules (`player_report.py`, `team_report.py`, `team_trend_data.py`,
`team_comparison.py`, and their visualizers) -- none of that
report-generation logic is modified, reimplemented, or duplicated here.

ADR-018 (read before modifying the reporting tabs): Player Reports, Team
Reports, and Team Comparison no longer import their report-generation
functions directly -- they call the new `/reports/player/{player_id}`,
`/reports/team/{team_name}`, and `/reports/team-comparison` endpoints on
`api.py` over HTTP instead, reusing the same `rest_base_url` sidebar
config the What-If Simulator already uses. This closes a real
dual-entrypoint gap: previously this Streamlit process talked to MLflow
and `data/raw/` independently of `api.py`, which only worked because both
processes happened to run on the same machine. See ADR-018 for the full
reasoning, including why `team_trend_data.py` is a deliberate, NAMED
EXCEPTION to this: that module's own docstring already states it must
never be served over a network endpoint (an unresolved football-data.co.uk
licensing scope, the same conservative stance ADR-014 applies to the CV
track's AGPL-derived model) -- so the Team Trends tab below still imports
and calls `generate_team_trend_report` directly, unchanged, and still
needs `data/raw/` write access and network access to football-data.co.uk
from wherever this dashboard process itself runs. Full multi-machine
separation therefore holds for three of the four reporting tabs, not all
four -- stated plainly, not implied to be complete.

All four tabs' results are still wrapped in `st.cache_data` so Streamlit's
rerun-the-whole-script-on-any-widget-interaction model doesn't silently
re-trigger expensive report generation (now an HTTP round-trip for three
of the four tabs) on every tab switch or unrelated click.

ARCHITECTURAL DECISION -- read before modifying this file:

This dashboard deliberately uses the SYNCHRONOUS `websocket-client` library
(the `websocket` module) instead of the async `websockets` library, and
runs exactly ONE long-lived, blocking receive loop inside a single
Streamlit script execution (triggered by the "Start Stream" button),
updating `st.empty()` placeholders in-place as messages arrive.

Streamlit's execution model reruns the ENTIRE script top-to-bottom on every
widget interaction / rerun. That model does not compose with holding a
live async websocket connection open in `st.session_state` across reruns:
the connection object would either be silently dropped in a later rerun,
or -- if it survives -- there is no supported way to keep pumping messages
into the UI in the background between reruns. The common failure mode this
produces is a dashboard that updates once (on the run that opened the
connection) and then silently stops, because nothing is left running to
call `recv()` again after that script execution ends.

The single-blocking-loop pattern avoids this class of bug entirely: the
ENTIRE stream lifetime -- connect, every message, every UI update, and
disconnect -- happens within one script execution, so there is never a
connection object that needs to survive a rerun. The trade-off (accepted
deliberately, per Milestone 17's spec) is that the loop is NOT
interactively stoppable mid-run: Streamlit cannot process a new "Stop"
button click while the script is blocked inside this loop's `recv()`
calls. Instead, the loop is bounded up front by an explicit max-duration
and max-message-count control, and simply ends on its own after that -- an
honest alternative to a Stop button that would not actually work while the
script is blocked.

PERMANENT CONSEQUENCE, NOW APPLYING TO THE WHOLE APP, NOT JUST ONE PANEL
(do not paper over this): `st.tabs()` does NOT lazily execute only the
selected tab's code -- Streamlit runs this ENTIRE script top-to-bottom on
every rerun, every tab's body included, regardless of which tab is
visually selected. Because the "Live CV Monitor" tab's blocking loop is
written FIRST in this file's top-to-bottom order, a running stream blocks
that same single script execution before Python ever reaches the Player
Reports / Team Reports / Team Trends / Team Comparison tabs' code below
it -- so while a stream is running, NONE of those tabs will update or
respond to input either, for the exact same underlying reason, not a
separate limitation. This is surfaced directly in the top-level caption
below (visible regardless of which tab is open) and must stay that way if
this file is restructured further.

Do NOT "fix" this by moving the loop into session_state, a background
thread, or an st.fragment/rerun-driven poll unless you have specifically
re-verified that approach does not reintroduce the update-then-silently-
stop failure mode described above.
"""

import base64
import html
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

# `streamlit run production/frontend/dashboard.py` puts THIS SCRIPT'S OWN
# directory (production/frontend/) on sys.path -- not the repo root --
# regardless of the shell's current working directory when the command
# is issued. Every `from production...` import below is an absolute
# import rooted at the repo root, so without this, they fail with
# `ModuleNotFoundError: No module named 'production'` the moment
# streamlit (not `python -m`, not pytest, not a shell with an editable
# install already on its venv's sys.path) is what actually launches this
# file -- a real failure mode found via an actual browser launch, not
# hypothetical. Inserting the repo root explicitly makes this script
# self-sufficient regardless of how or from where it's started.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st
import websocket

import production.src.reporting.candidate_index as candidate_index_module
from production.src.reporting.candidate_index import enumerate_cached_candidates
from production.src.reporting.match_timeline_visualizer import render_match_timeline
from production.src.reporting.pass_network_visualizer import (
    render_pass_network,
    render_pass_network_aggregated,
)
from production.src.reporting.passing_lane_visualizer import (
    render_passing_lanes,
    render_passing_lanes_aggregated,
)
from production.src.reporting.player_visualizer import (
    render_player_dashboard,
    render_player_match_timeline,
    render_player_match_timeline_aggregated,
    render_player_match_touch_map,
    render_player_match_touch_map_aggregated,
    render_shot_map,
    render_shot_map_aggregated,
)
from production.src.reporting.team_trend_data import (
    compare_team_trend_seasons,
    generate_team_trend_report,
)
from production.src.reporting.team_trend_visualizer import render_team_trend_comparison
from production.src.reporting.team_visualizer import render_team_dashboard

from production.frontend.tactical_momentum import (
    ELEVATED_THREAT_LEVEL,
    MOMENTUM_MIN_MESSAGES_FOR_TREND,
    MOMENTUM_SMOOTHING_WINDOW_MESSAGES,
    MOMENTUM_TREND_LOOKBACK_MESSAGES,
    MOMENTUM_TREND_THRESHOLD,
    SEGMENT_DWELL_MESSAGES,
    _compute_tactical_momentum,
    classify_match_phase,
)

# ADR-021 condition-2 / Team Trends serving-contradiction compliance fix:
# this dashboard process's OWN copy of the same flag api.py checks (see
# that file's own comment for the full reasoning). Read ONCE at import
# time, same convention. This flag does two independent things in this
# file: (1) decides whether the shot-map panel may ever render/display
# individual per-shot data (belt-and-suspenders on top of api.py's own
# server-side gate -- see the shot-map panel below for why this is
# checked in BOTH places rather than trusted from only one), and (2)
# fully hides the Team Trends tab, since that tab's own data source
# (football-data.co.uk, via team_trend_data.py) has its own separate,
# pre-existing "never served/distributed" restriction that a public
# deployment of THIS dashboard process would otherwise silently violate
# regardless of api.py's flag (see team_trend_data.py's own docstring and
# ADR-018's Update section).
PUBLIC_DEPLOYMENT = os.environ.get("PUBLIC_DEPLOYMENT", "false").strip().lower() == "true"

DEFAULT_REST_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws/tactical-stream"
DEFAULT_MATCH_ID = "3857276"
# Milestone 33: server-side default for the CV data source -- MUST resolve
# inside the backend's data/raw/ directory (ALLOWED_CV_VIDEO_DIRECTORY in
# api.py). `data/raw/` is entirely gitignored (see .gitignore), so this is
# NOT guaranteed to exist in a fresh clone or a different machine -- a
# plausible path, not a committed asset.
#
# STALE-COMMENT CORRECTION (verified directly, not assumed, during a UI
# walkthrough follow-up after this comment was originally written): this
# used to say "every CV milestone since 25 has noted no persistent real
# footage lives under this path in THIS environment." That is no longer
# true HERE -- a real file genuinely exists at this exact path in this
# environment (11,356,053 bytes, a valid MP4 container, confirmed via
# cv2.VideoCapture: 1284x728, 28fps, 970 frames, and a decoded frame shows
# real Premier League broadcast footage, not a blank/placeholder image).
# This is the SAME private, unannotated local clip Milestone 34B's "First
# Real-Footage Validation" already used and documented (see
# docs/CV_PIPELINE_FINDINGS.md and docs/FULL_PROJECT_REPORT.md's
# Milestone 34B entry -- identical specs), not a new or unexplained file.
# "StatsBomb Replay" (below) remains the default `st.radio` selection
# (index 0) regardless -- it needs no local file at all and is therefore
# the more broadly reproducible out-of-the-box path across environments,
# even though this specific environment does have real CV footage
# available. Left here as a plausible path for anyone who DOES have (or,
# as in this environment, already has) a real local clip to point this at.
DEFAULT_CV_VIDEO_PATH = "data/raw/test_match.mp4"
MAX_THREAT_BUFFER_LEN = 60
MAX_ALERT_BUFFER_LEN = 20
RECV_TIMEOUT_SECONDS = 60.0  # how long to wait for a single message before treating the stream as stalled
SIMULATE_REQUEST_TIMEOUT_SECONDS = 5.0  # mandatory -- see What-If section below

# Tactical Momentum (additive new feature): rolling-window smoothing +
# trend indicator computed CLIENT-SIDE over the same `threat_buffer` the
# existing line chart already accumulates (see MAX_THREAT_BUFFER_LEN
# above) -- no new WebSocket field, no api.py change. The pure computation
# itself lives in the separate `tactical_momentum.py` module (plain
# Python, no `streamlit` import) specifically so it can be unit-tested via
# a normal `import`, since this script cannot safely be imported directly
# -- see that module's own docstring for the full Step 0 definition of
# what these constants mean and why these exact values were chosen.
TACTICAL_ACTIONS = ["high_press", "drop_deep", "force_wide", "no_change"]
# ADR-018: report endpoints do real network fetches (StatsBomb), MLflow
# artifact loads, and pitch-control physics across potentially many chains --
# a single /simulate-style 5s budget is too tight for these. 60s matches this
# file's own RECV_TIMEOUT_SECONDS as a "generous but bounded" convention.
REPORT_REQUEST_TIMEOUT_SECONDS = 60.0

# AI Tactical Chat (new reporting track, Part B): each turn rebuilds a
# match's full context package (a generate_automatic_match_report-scale
# call, measured at ~2s for the cached match this dashboard defaults to)
# PLUS a real Gemini round-trip -- generously bounded above both costs
# combined, well under REPORT_REQUEST_TIMEOUT_SECONDS's own 60s heavy-report
# budget since a chat reply is expected to feel conversational, not
# report-generation-slow.
CHAT_REQUEST_TIMEOUT_SECONDS = 30.0

# Player Similarity Search's own precompute rebuild -- a genuinely slow,
# manually-triggered operation (MEASURED against this project's real full
# searchable population, ~5,000 players: see player_similarity.py's own
# docstring for the exact real timing), so it needs a much longer client
# timeout than every other report request in this file rather than
# sharing REPORT_REQUEST_TIMEOUT_SECONDS's 60s "generous but bounded"
# budget, which this operation would blow through by design, not by bug.
SIMILARITY_INDEX_REBUILD_TIMEOUT_SECONDS = 1800.0

# Timeout-incident fix, MEASURED (not guessed) directly against this
# project's own real cached data before being chosen -- see the task that
# added this constant for the full breakdown. Real Madrid's 68-match
# request (2 of them 360-covered) timed out at 61.90s; pre-filtering to
# just the 2 360-covered matches dropped that to 4.82s -- confirming most
# of the original cost was wasted network round-trips checking coverage
# generate_team_report has no cheap way to know in advance, which
# candidate_index.py's own 360-scan already does. But pre-filtering ALONE
# is not sufficient: PSG's full, genuinely well-supported, ALREADY
# pre-filtered 51-match case (every one of them real, no waste) still took
# 100.27s -- real BiomechanicalPitchControl/DeepHit compute. Two real
# calibration points: Bayer Leverkusen's 31 matches took 36.77s and
# 47.55s across two separate runs (~1.2-1.5s/match); PSG's 51 took 100.27s
# (~2.0s/match) -- genuine run-to-run variance, not just scale. 25 is
# chosen to stay under REPORT_REQUEST_TIMEOUT_SECONDS even at the WORSE
# observed rate (25 * 2.0s = 50s, ~10s margin for model-loading/fixed
# overhead) while trimming as little as possible off the single largest
# genuinely well-supported real case in this cache (Bayer Leverkusen's 31
# -- 6 matches short of the cap, a disclosed, minor, deliberate
# trade-off, not an oversight).
TEAM_REPORT_MAX_360_MATCHES_PER_REQUEST = 25

# --- Reporting-tab candidate lists (dynamic, from what's actually cached) -
# Previously a small, hand-picked preset dict (Milestone 44's original
# validation-sweep cases only) -- replaced with a REAL scan of data/raw/
# via candidate_index.py, so the dropdown reflects everything actually
# cached, including whatever data_fallback.py's own runs have pulled in
# since (e.g. Ronaldo, Real Madrid, PSG, Bayern Munich, Messi's full
# tracked career) -- not just the original 5 players / 3 teams. See
# candidate_index.py's own module docstring for exactly what this scan
# reads (cached event/match-list JSON only) versus skips (no
# positional-distribution/heatmap aggregation, no pitch-control physics,
# no MLflow/model access -- a pure enumeration, not a report).
#
# ARCHITECTURAL NOTE (disclosed, not silently glossed over): this scan
# reads `data/raw/` DIRECTLY from the Streamlit process, same as this
# dashboard did for reporting DATA before ADR-018 -- meaning populating
# these dropdowns still assumes `dashboard.py` runs with its own
# `data/raw/` access, even though the actual report GENERATION for a
# selected candidate correctly goes through api.py's HTTP boundary
# (unchanged). This is a real, narrower re-introduction of a co-location
# assumption, scoped to dropdown population only -- not something ADR-018
# claimed to solve for this not-yet-existing feature, and not something
# this task asked to extend api.py to cover (a `/candidates/...` endpoint
# would remove it; out of this task's stated scope, which is a
# dashboard/enumeration-layer change only).
REFRESH_CACHE_LIST_TTL_SECONDS = 3600  # 1 hour -- generous; paired with a manual refresh button below for on-demand invalidation

# Alerts History tab (ADR-019 persistence, surfaced here for the first time):
# a SHORT TTL, unlike REFRESH_CACHE_LIST_TTL_SECONDS above -- that cache backs
# a slow, mostly-static scan of data/raw/ (what candidates EXIST), while this
# one backs a fast SQLite query over genuinely time-sensitive data (new
# alerts can be logged continuously, e.g. by a live WebSocket stream running
# in another session against the same backend). 30s keeps the tab from
# re-querying on every unrelated widget interaction/tab switch (Streamlit
# reruns the whole script on both) while still staying well under a minute
# stale for a "check what's happened recently" panel.
ALERTS_HISTORY_CACHE_TTL_SECONDS = 30
ALERTS_HISTORY_REQUEST_TIMEOUT_SECONDS = 10.0  # a single indexed SQLite SELECT -- fast, not a report-generation call
ALERTS_HISTORY_DEFAULT_LIMIT = 500  # matches fetch_alerts's own default (alert_store.py)


@st.cache_data(show_spinner="Scanning data/raw/ for cached players/teams (one-time, ~15-20s at current cache size)...", ttl=REFRESH_CACHE_LIST_TTL_SECONDS)
def _cached_candidate_index(_cache_bust: int) -> tuple[list[dict], list[dict]]:
    """`_cache_bust` is never read -- its only job is to participate in
    `st.cache_data`'s cache key, so the "Refresh cache list" button
    (which increments it in `st.session_state`) can force a fresh scan on
    demand, on top of the TTL above."""
    return enumerate_cached_candidates()


# --- Cached wrappers ------------------------------------------------------
# Streamlit reruns this entire script on almost any widget interaction;
# without these, switching tabs or touching an unrelated widget would
# silently re-trigger a real report generation (now an HTTP round-trip to
# api.py for three of the four tabs -- see ADR-018) every single time.
# Keyed on tuples (not lists) for a cleanly hashable cache key.
#
# `rest_base_url` is deliberately an explicit PARAMETER of each of these
# functions (not read from an outer/global variable inside the function
# body) so it correctly participates in `st.cache_data`'s cache key: if the
# user edits the "REST API Base URL" sidebar field, that must invalidate any
# previously-cached report fetched from a DIFFERENT backend, not silently
# keep serving a stale response from whichever URL happened to be active the
# first time a given (player_id, match_ids) combination was requested.
# Without this, a base-URL change would go unnoticed by the cache -- a real,
# subtle bug this file's own original (pre-ADR-018) cache signature
# (player_id, match_ids only) would have reintroduced.
@st.cache_data(show_spinner=False)
def _cached_player_report(rest_base_url: str, player_id: int, match_ids: tuple[int, ...]) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/player/{player_id}",
        params={"match_ids": list(match_ids)},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_player_png(report: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_player_dashboard(report, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Shot map (additive new feature): a dedicated pair of cached wrappers,
# mirroring _cached_player_report/_cached_player_png's own pattern exactly
# -- calls the NEW, separate /reports/player/{player_id}/shot-map endpoint
# (see api.py's own comment for why this is a dedicated endpoint, not a
# field added to the existing player-report response).
@st.cache_data(show_spinner=False)
def _cached_player_shot_map(rest_base_url: str, player_id: int, match_ids: tuple[int, ...]) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/player/{player_id}/shot-map",
        params={"match_ids": list(match_ids)},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_player_shot_map_png(shot_map: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_shot_map(shot_map, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _cached_player_shot_map_aggregated_png(shot_map_aggregated: dict) -> bytes:
    """ADR-021 condition-2 compliance fix -- the PUBLIC-deployment
    counterpart to `_cached_player_shot_map_png` above. Only ever called
    on a dict that has already been confirmed (see the shot-map panel
    below) to carry no `"shots"` key."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_shot_map_aggregated(shot_map_aggregated, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Player Dashboard (additive new feature): match-level views extending the
# existing Player Reports tab. Match Summary is unconditionally aggregate
# (ADR-021: not gated -- see player_report.py's own Player Dashboard
# section for the full Step 0 reasoning); Touch Map and Timeline each get
# a raw + aggregated cached-PNG pair, same pattern as the shot map above.
@st.cache_data(show_spinner=False)
def _cached_player_match_summary(rest_base_url: str, player_id: int, match_ids: tuple[int, ...]) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/player/{player_id}/match-summary",
        params={"match_ids": list(match_ids)},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# Press Resistance Index (additive new feature): season/multi-match
# aggregate rate, same ADR-021-unconditional class as match summary above
# -- not gated behind PUBLIC_DEPLOYMENT (see the endpoint's own docstring
# in api.py and generate_player_press_resistance_index's docstring in
# player_report.py for the full exemption reasoning).
@st.cache_data(show_spinner=False)
def _cached_player_press_resistance_index(rest_base_url: str, player_id: int, match_ids: tuple[int, ...]) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/player/{player_id}/press-resistance",
        params={"match_ids": list(match_ids)},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# Player Similarity Search (new ML work, additive, its own feature --
# see docs/adr/ADR-021's own addendum for the exemption reasoning). The
# LIVE query is a fast lookup against an ALREADY-PRECOMPUTED index --
# cached here too (st.cache_data) purely to avoid a redundant HTTP round
# trip on an unrelated widget rerun within the same session, NOT because
# the query itself is slow. Rebuilding the index itself (below) is
# deliberately NOT cached -- every click must trigger a real rebuild.
@st.cache_data(show_spinner=False)
def _cached_similar_players(rest_base_url: str, player_id: int, top_k: int) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/player/{player_id}/similar",
        params={"top_k": top_k},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _rebuild_similarity_index(rest_base_url: str) -> dict:
    """NOT `@st.cache_data` -- Step 3.3's own explicit "manual trigger,
    no automatic staleness/TTL" discipline means every real button click
    must genuinely re-POST, never silently return a cached prior result.
    Uses SIMILARITY_INDEX_REBUILD_TIMEOUT_SECONDS, not the shared
    REPORT_REQUEST_TIMEOUT_SECONDS every other report request uses (see
    that constant's own comment for why)."""
    response = requests.post(
        f"{rest_base_url}/reports/player-similarity/rebuild",
        timeout=SIMILARITY_INDEX_REBUILD_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_player_match_touch_map(rest_base_url: str, player_id: int, match_id: int) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/player/{player_id}/match/{match_id}/touch-map",
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_player_match_touch_map_png(touch_map: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_player_match_touch_map(touch_map, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _cached_player_match_touch_map_aggregated_png(touch_map_aggregated: dict) -> bytes:
    """ADR-021 condition-2 compliance -- the PUBLIC-deployment counterpart
    to `_cached_player_match_touch_map_png` above. Only ever called on a
    dict already confirmed (see the touch-map panel below) to carry no
    `"touches"` key."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_player_match_touch_map_aggregated(touch_map_aggregated, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _cached_player_match_timeline(rest_base_url: str, player_id: int, match_id: int) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/player/{player_id}/match/{match_id}/timeline",
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_player_match_timeline_png(timeline: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_player_match_timeline(timeline, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _cached_player_match_timeline_aggregated_png(timeline_aggregated: dict) -> bytes:
    """ADR-021 condition-2 compliance -- the PUBLIC-deployment counterpart
    to `_cached_player_match_timeline_png` above. Only ever called on a
    dict already confirmed (see the timeline panel below) to carry no
    `"timeline"` key."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_player_match_timeline_aggregated(timeline_aggregated, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _cached_team_report(rest_base_url: str, team_name: str, match_ids: tuple[int, ...]) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/team/{requests.utils.quote(team_name, safe='')}",
        params={"match_ids": list(match_ids)},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# Tactical Entropy (additive new feature): unlike generate_team_report
# above, this needs no 360 freeze-frame coverage at all (event data only)
# -- so it is fetched against `_variant_to_match_ids` (every raw cached
# match in the selection), NOT `_variant_to_match_ids_360` (the smaller
# 360-covered subset the pitch-control report needs). ADR-021: NOT gated
# behind PUBLIC_DEPLOYMENT (see the endpoint's own docstring in api.py and
# generate_team_pass_entropy's docstring in team_report.py). Measured
# directly before skipping a request-size cap here (same discipline as
# the Team Report timeout incident): 300 requested match_ids (174 real
# Barcelona matches used) completed in ~8.5s, a pure event-scan/dict-count
# cost with no physics/ML inference -- far cheaper than team_report's own
# per-match pitch-control cost, so no separate cap constant was added.
@st.cache_data(show_spinner=False)
def _cached_team_pass_entropy(rest_base_url: str, team_name: str, match_ids: tuple[int, ...]) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/team/{requests.utils.quote(team_name, safe='')}/pass-entropy",
        params={"match_ids": list(match_ids)},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_team_png(report: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_team_dashboard(report, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Passing Lane Visualizer (additive new feature): unlike Tactical
# Entropy, this DOES need 360 freeze-frame coverage (it reuses
# BiomechanicalPitchControl, same requirement as the pitch-control report
# above) -- fetched against `_variant_to_match_ids_360` (the SAME
# 360-covered, already-capped match list the pitch-control report itself
# uses), not the raw uncapped list.
@st.cache_data(show_spinner=False)
def _cached_team_passing_lanes(rest_base_url: str, team_name: str, match_ids: tuple[int, ...]) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/team/{requests.utils.quote(team_name, safe='')}/passing-lanes",
        params={"match_ids": list(match_ids)},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_passing_lanes_png(passing_lanes: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_passing_lanes(passing_lanes, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _cached_passing_lanes_aggregated_png(passing_lanes_aggregated: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_passing_lanes_aggregated(passing_lanes_aggregated, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Opposition Analysis (additive new feature): event-data only, like
# Tactical Entropy -- fetched against `_variant_to_match_ids` (the RAW,
# un-360-filtered match selection), not `_variant_to_match_ids_360`. The
# THIRD metric (pitch-control weak zones) is deliberately NOT fetched
# here at all -- dashboard.py's own Opposition Analysis panel re-presents
# the ALREADY-FETCHED team_report_dict's own `weakest_control_zones`
# field (the SAME dict the pitch-control panel above already requested
# in this same tab), with zero additional backend computation. See
# generate_team_opposition_analysis's own docstring in team_report.py.
@st.cache_data(show_spinner=False)
def _cached_team_opposition_analysis(rest_base_url: str, team_name: str, match_ids: tuple[int, ...]) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/team/{requests.utils.quote(team_name, safe='')}/opposition-analysis",
        params={"match_ids": list(match_ids)},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# Weak-Spot Lifetime Analysis (new reporting track): GET
# /reports/team/{team_name}/weak-spot-lifetime/{match_id}. SINGLE match_id,
# not a match_ids tuple like the panels above -- a weak-spot "lifetime" is
# an inherently within-match temporal concept (see
# generate_weak_spot_lifetime_analysis's own docstring), so this panel gets
# its own single-match_id input rather than reusing the multi-variant
# season/match-list selection machinery the rest of this tab uses.
@st.cache_data(show_spinner=False)
def _cached_weak_spot_lifetime(
    rest_base_url: str, team_name: str, match_id: int, include_recommendations: bool = False
) -> dict:
    """`include_recommendations` (Weak-Spot Exploitation Recommendation,
    additive new feature): forwarded straight through to the endpoint's
    own opt-in query param -- see get_weak_spot_lifetime's own docstring
    in api.py for why this defaults to False (byte-for-byte unchanged
    response shape/cost unless a caller explicitly asks for more)."""
    response = requests.get(
        f"{rest_base_url}/reports/team/{requests.utils.quote(team_name, safe='')}"
        f"/weak-spot-lifetime/{match_id}",
        params={"include_recommendations": include_recommendations},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# Decision Quality (Phase 4, final item): GET
# /reports/team/{team_name}/decision-quality/{match_id}. SINGLE match_id,
# same "best alternative available AT THAT MOMENT" reasoning Weak-Spot
# Lifetime Analysis's own panel already established.
@st.cache_data(show_spinner=False)
def _cached_decision_quality(rest_base_url: str, team_name: str, match_id: int) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/team/{requests.utils.quote(team_name, safe='')}"
        f"/decision-quality/{match_id}",
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# team_trend_data.py is a deliberate, NAMED EXCEPTION to ADR-018's
# consolidation -- see this file's module docstring and ADR-018 itself.
# That module's own docstring already states it must never be wired into
# api.py's served layer (unresolved football-data.co.uk licensing scope),
# so this tab keeps calling generate_team_trend_report directly, unlike the
# other three reporting tabs.
@st.cache_data(show_spinner=False)
def _cached_team_trend_report(team_name: str, start_season: int, end_season: int) -> dict:
    return generate_team_trend_report(team_name, start_season, end_season)


# Feature 3 (Compare Two Seasons): same deliberate, named ADR-018 exception
# as _cached_team_trend_report above -- compare_team_trend_seasons is
# called in-process for the exact same licensing reason, never through
# api.py. This section is gated behind the SAME PUBLIC_DEPLOYMENT check as
# the rest of this tab (see the tab body below) -- no separate/weaker gate.
@st.cache_data(show_spinner=False)
def _cached_team_trend_comparison(team_name: str, season_a: int, season_b: int) -> dict:
    return compare_team_trend_seasons(team_name, season_a, season_b)


@st.cache_data(show_spinner=False)
def _cached_team_trend_comparison_png(comparison: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_team_trend_comparison(comparison, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _cached_team_comparison(
    rest_base_url: str, team_a: str, season_a: int, team_b: str, season_b: int
) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/team-comparison",
        params={"team_a": team_a, "season_a": season_a, "team_b": team_b, "season_b": season_b},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# Session/Match Comparison (additive extension of the Team-Season Style
# Comparison tool above, at a finer granularity -- one team, two SPECIFIC
# matches, not two seasons). ADR-021: unconditional, same as the
# season-level comparison above -- see ADR-021's "Session/Match
# Comparison" addendum for the full reasoning.
@st.cache_data(show_spinner=False)
def _cached_team_match_comparison(
    rest_base_url: str, team_name: str, match_id_a: int, match_id_b: int
) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/team-comparison/match",
        params={"team_name": team_name, "match_id_a": match_id_a, "match_id_b": match_id_b},
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_pass_network(rest_base_url: str, match_id: int) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/pass-network/{match_id}",
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_pass_network_png(pass_network: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_pass_network(pass_network, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _cached_pass_network_aggregated_png(pass_network_aggregated: dict) -> bytes:
    """ADR-021 condition-2 compliance -- the PUBLIC-deployment counterpart
    to `_cached_pass_network_png` above. Only ever called on a dict that
    has already been confirmed (see the Pass Network panel below) to
    carry no `"nodes"`/`"edges"` key."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_pass_network_aggregated(pass_network_aggregated, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Automatic Match Report (new reporting track, Part A): GET
# /reports/match/{match_id}. A slower call than the other reporting
# endpoints (it aggregates two full Team Report calls plus more, "heavy"
# tier on the API side) -- REPORT_REQUEST_TIMEOUT_SECONDS (60s) is reused
# unchanged rather than a new, longer timeout constant, matching how the
# existing Team Report panel already budgets for a single heavy call at
# that same 60s figure.
@st.cache_data(show_spinner=False)
def _cached_match_report(rest_base_url: str, match_id: int) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/match/{match_id}",
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# Tactical Event Detection (new reporting track): GET
# /reports/match/{match_id}/tactical-events -- OWN dedicated fetch, not
# folded into _cached_match_report above, matching that endpoint's own
# separation from generate_automatic_match_report on the API side.
@st.cache_data(show_spinner=False)
def _cached_tactical_events(rest_base_url: str, match_id: int) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/match/{match_id}/tactical-events",
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# Tactical Timeline UI (new reporting track, capstone): GET
# /reports/match/{match_id}/timeline -- OWN dedicated fetch, same
# separation-from-Match-Report convention Tactical Event Detection's own
# fetch above already established.
@st.cache_data(show_spinner=False)
def _cached_match_timeline(rest_base_url: str, match_id: int) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/match/{match_id}/timeline",
        timeout=REPORT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _cached_match_timeline_png(timeline_data: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        render_match_timeline(timeline_data, tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Alerts History tab: GET /alerts/history (ADR-019's persistence store, via
# alert_store.fetch_alerts -- see api.py). Every filter param is Optional on
# the endpoint's own side (None = no filter, AND-combined) -- `requests`
# itself drops any dict entry whose value is None from the querystring
# (confirmed directly, not assumed), so passing match_id/source/start_utc/
# end_utc through unconditionally, even when a filter is unset, is correct
# and needs no manual "only include if set" branching here.
#
# No `X-API-Key` header is sent -- confirmed directly that NO existing call
# in this file sends one either (grepped this module for "X-API-Key"/
# "headers=" before adding this). Per ADR-022, `API_KEY` is unset by
# default (every environment this project has actually run in, including
# its own test suite), so this is zero-friction, unchanged behavior; if an
# operator ever sets `API_KEY` server-side without also updating this
# dashboard to send the header, EVERY existing REST call in this file
# (`/simulate`, `/reports/*`) would already 401 identically -- a pre-existing
# gap this one new call does not introduce or worsen.
@st.cache_data(show_spinner=False, ttl=ALERTS_HISTORY_CACHE_TTL_SECONDS)
def _cached_alerts_history(
    rest_base_url: str,
    match_id: int | None,
    source: str | None,
    start_utc: str | None,
    end_utc: str | None,
    limit: int,
) -> list[dict]:
    response = requests.get(
        f"{rest_base_url}/alerts/history",
        params={
            "match_id": match_id,
            "source": source,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "limit": limit,
        },
        timeout=ALERTS_HISTORY_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# --- Home tab helpers ------------------------------------------------------
# Step 0 scope (all five candidate sections included, none deferred):
#   1. Data coverage summary -- reuses `_cached_candidate_index` (the exact
#      cache-bust'd wrapper around `enumerate_cached_candidates` the Player
#      Reports/Team Reports tabs already use for their dropdowns), so Home
#      does NOT trigger a second ~15-20s cache scan -- it shares the same
#      `st.cache_data` entry.
#   2. Recent activity -- reuses `_cached_alerts_history` (same helper the
#      Alerts History tab calls), no filters, capped to
#      `HOME_RECENT_ALERTS_LIMIT` for a compact preview.
#   3. Quick actions -- derived from the SAME alerts fetch in (2), not a
#      second query: the most recent alert's `match_id` (alerts are already
#      returned most-recent-first by `fetch_alerts`'s own `ORDER BY
#      logged_at_utc DESC`). Tactical Event Detection and Tactical Timeline
#      both already read match_id from the SAME `match_report_id_input`
#      session_state key inside `tab_match_report` (confirmed directly --
#      see both call sites), so one shortcut covers both "deepest views"
#      named in this task. If there is no alert history yet, this section
#      states that plainly rather than fabricating a "recent" match.
#   4. System status -- new `_cached_health`/`_cached_metrics` wrappers
#      calling the EXISTING `/health`/`/metrics` endpoints (api.py, Fix 3)
#      exactly like every other `_cached_*` wrapper in this file calls an
#      existing endpoint -- neither endpoint's own logic is touched.
#   No section is deferred -- all five of the task's candidate sections are
#   in scope and genuinely buildable from already-existing calls alone.
HOME_RECENT_ALERTS_LIMIT = 5
HEALTH_METRICS_CACHE_TTL_SECONDS = 5  # short: this panel wants to look "live", not stale for a full 30s like Alerts History
HEALTH_METRICS_REQUEST_TIMEOUT_SECONDS = 5.0  # /health and /metrics are documented as cheap/fast/no-side-effects (api.py) -- a short timeout is appropriate, not conservative-for-a-heavy-call


@st.cache_data(show_spinner=False, ttl=HEALTH_METRICS_CACHE_TTL_SECONDS)
def _cached_health(rest_base_url: str) -> dict:
    response = requests.get(f"{rest_base_url}/health", timeout=HEALTH_METRICS_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False, ttl=HEALTH_METRICS_CACHE_TTL_SECONDS)
def _cached_metrics(rest_base_url: str) -> dict:
    response = requests.get(f"{rest_base_url}/metrics", timeout=HEALTH_METRICS_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _home_data_coverage_summary(teams: list[dict], players: list[dict]) -> dict:
    """Pure aggregation over `_cached_candidate_index`'s already-scanned
    output -- no file I/O, no re-scan. `total_matches_cached`/
    `total_matches_360` are real DISTINCT match counts (a union across every
    team's own season match_ids), not a sum of each team's own count, which
    would double-count every match (two teams per match)."""
    all_match_ids: set[int] = set()
    all_match_ids_360: set[int] = set()
    for team in teams:
        for season in team["seasons"]:
            all_match_ids.update(season["match_ids"])
            all_match_ids_360.update(season["match_ids_360"])
    return {
        "total_players_cached": len(players),
        "total_teams_cached": len(teams),
        "total_matches_cached": len(all_match_ids),
        "total_matches_360": len(all_match_ids_360),
    }


def _home_most_recent_match_id(recent_alerts: list[dict]) -> int | None:
    """`recent_alerts` is already most-recent-first (`fetch_alerts`'s own
    `ORDER BY logged_at_utc DESC`) -- the first entry with a non-null
    `match_id` is the most recently referenced match. Returns `None` if
    there is no alert history yet, or no alert carries a `match_id`
    (`video_path`-sourced CV alerts can be logged with `match_id=None`)."""
    for alert in recent_alerts:
        if alert.get("match_id") is not None:
            return alert["match_id"]
    return None


def _fetch_report_safely(
    fetch_fn,
    rest_base_url: str | None = None,
    *,
    context_label: str | None = None,
    not_found_message: str | None = None,
) -> dict | None:
    """Calls `fetch_fn()` and returns its parsed JSON (or whatever it
    returns), or `None` if the call failed -- rendering ONE consistent
    `st.error` template, rather than letting an exception crash this tab's
    script execution. This is this file's single canonical error-state
    pattern (UX polish pass, Part C) -- every tab that can fail (a real
    network call to `rest_base_url`, OR a real network call to an external
    source like football-data.co.uk) routes its failure through here, so
    "connection error" / "timeout" / "server error" all look and read the
    same everywhere, not just in the ~30 call sites that already used this
    helper before this pass.

    `rest_base_url`: set for calls to THIS project's own FastAPI backend
    (the common case -- every `_cached_*` HTTP wrapper above) so the
    Timeout/ConnectionError messages can name it and suggest starting
    `uvicorn`. `context_label`: set instead for calls that do NOT talk to
    this project's own backend (e.g. `team_trend_data.py`'s own direct
    `requests.get()` against football-data.co.uk from the Team Trends tab)
    -- produces the same template with backend-appropriate wording, not a
    misleading "start uvicorn" suggestion for a failure that has nothing to
    do with this project's own API server. Exactly one of the two should be
    given; `context_label` takes precedence if both are.

    `not_found_message`: if given, a 404 `HTTPError` renders `st.info(...)`
    with this message instead of the generic HTTPError `st.error` -- for
    call sites where "not found" is an expected, non-alarming outcome (e.g.
    Player Similarity Search's "no similarity index has been built yet,
    click Rebuild first" -- a real, previously-duplicated special case this
    parameter absorbs into the shared helper instead of a one-off inline
    try/except).

    The catch-all `Exception` branch is a real, deliberate addition (UX
    polish pass): previously, a non-`requests` failure (e.g. a pandas/CSV
    parsing error from `team_trend_data.py`'s own direct, in-process calls,
    which were never routed through this helper at all before this pass)
    would propagate uncaught and crash the whole script with Streamlit's
    own generic traceback box, not a clean, consistent message.
    """
    label = context_label or (f"backend at {rest_base_url}" if rest_base_url else "the backend")
    try:
        return fetch_fn()
    except requests.exceptions.Timeout:
        st.error(
            f"Request to {label} timed out after {REPORT_REQUEST_TIMEOUT_SECONDS:.0f}s -- it did "
            "not respond in time."
        )
    except requests.exceptions.ConnectionError:
        if rest_base_url and context_label is None:
            st.error(
                f"Backend unreachable at {rest_base_url} -- confirm the FastAPI server is running "
                "(uvicorn production.src.serving.api:app)."
            )
        else:
            st.error(f"Could not reach {label} -- confirm you have a working internet connection.")
    except requests.exceptions.HTTPError as exc:
        if not_found_message and exc.response is not None and exc.response.status_code == 404:
            st.info(not_found_message)
        else:
            st.error(f"Request to {label} failed: {exc}")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        st.error(f"Unexpected error from {label}: {exc}")
    return None


# ============================================================================
# UX polish pass, Part A: cross-linking between tabs.
#
# STEP 0 FINDING, verified directly rather than assumed (this Streamlit
# version, 1.60.0, exactly what `requirements-lock.txt` pins): `st.tabs()`
# DOES support a real, genuine "open this specific tab" mechanism via its
# `default=<tab label>` parameter -- confirmed by reading `help(st.tabs)`
# against the actually-installed version, then confirmed END-TO-END with a
# real, triggered `AppTest` run (a button click sets `st.session_state` and
# calls `st.rerun()`; the next run's `st.tabs(..., default=...)` picks it up
# with no exception). This is a genuinely stronger result than what this
# task's own Step 0 anticipated ("even if it can't force-switch the active
# tab") -- `default=` achieves real auto-navigation, not just a pre-filled
# field the user still has to manually click into.
#
# ONE deliberately NOT taken: `st.tabs(..., key=..., on_change="rerun")` --
# Streamlit's OTHER, bidirectional mechanism for tab control (see the "Set
# the default tab" vs. "Programmatically control the tab state" examples in
# `help(st.tabs)`). That mode fundamentally changes what tab-switching DOES
# for the WHOLE app: switching tabs stops being a free, client-side-only
# interaction and starts triggering a full server rerun on every click, and
# tab bodies would need to start using `.open` to skip inactive tabs to make
# that worthwhile -- this file's own module docstring "PERMANENT CONSEQUENCE"
# section documents that EVERY tab's code currently runs on EVERY rerun
# regardless of which tab is visible, an architectural property this whole
# file depends on (most concretely, `tab_cv`'s blocking loop). Opting into
# `on_change="rerun"` here would be a real, invasive change to that model as
# a side effect of a "UI polish" feature -- explicitly out of scope
# ("no new backend logic" and, in spirit, no unrequested architecture
# changes). `default=` alone needs neither `key` nor `on_change` and changes
# nothing about how tab-switching or reruns already work -- confirmed by the
# same AppTest run above completing with no exception and every tab's
# content still present.
#
# The chosen cross-link TARGET is always "Custom" mode on the destination
# tab (Player Reports / Team Reports), never a preset -- see the preset
# selectboxes' own `key=` comments above for why: a preset's OPTION STRING
# is a long, dynamically-scanned label a cross-link button would have to
# reconstruct exactly, whereas "Custom" is a stable literal and its own
# fields take plain player_id/team_name/match_ids values directly.
#
# A SECOND real constraint, found and fixed by actually running this (a
# real, triggered AppTest failure -- clicking a cross-link button silently
# did nothing, no exception, nothing changed -- not reasoned about in
# advance): a `_render_cross_link_button` call CANNOT live inside a block
# gated by `if <some other button>_clicked:` -- e.g. `if
# pass_network_generate_clicked: ... _render_cross_link_button(...)`.
# `st.button()`'s own return value is `True` for exactly ONE script run
# (the run immediately following its own click) and `False` on every
# subsequent rerun, including the very rerun a NESTED cross-link button's
# own `st.rerun()` triggers -- so on that next run, `pass_network_generate_
# clicked` is `False` again, the whole `if` block (cross-link button
# included) never re-executes, and the pending click is silently orphaned:
# the code that would have processed it never runs. This is a real bug in
# the actual browser too, not an AppTest artifact -- confirmed by writing
# temporary debug prints directly into this function and observing they
# never fired on the second run. The fix, applied at each of the three
# real cross-link call sites (Pass Network, Player Similarity Search,
# Match Report): the FETCH stays gated behind its own "Generate" button
# (unchanged), but the fetched RESULT is stored in `st.session_state` and
# the RENDER code (including any cross-link buttons) reads from
# `st.session_state` and runs unconditionally on every rerun as long as a
# result is present -- not gated by the transient click boolean at all.
#
# A THIRD real constraint, ALSO found by actually running this (not the
# first guess): fixing the second constraint above made the FIRST
# constraint's own exception (`st.session_state.<key> cannot be modified
# after the widget with key <key> is instantiated`) start firing for
# real -- proving directly that `st.rerun()` does NOT, in fact, bypass
# that ordering rule the way this docstring originally (incorrectly)
# claimed. Verified with a minimal, isolated repro outside this file
# before touching this fix: the exception fires at the exact
# `st.session_state[key] = value` line itself, synchronously, before
# `st.rerun()` is ever reached -- so if the target widget (e.g. Player
# Reports' own preset selectbox) already ran earlier in THIS SAME script
# execution (true here: Pass Network/Match Report/Player Similarity
# Search all run physically AFTER Player Reports/Team Reports), setting
# its session_state directly, from inside the click handler, always
# raises. The REAL fix: split into two phases. PHASE 1 (here, at click
# time): store the requested prefills under a key that is NOT any
# widget's own key (`_pending_cross_link`), then `st.rerun()`. PHASE 2 (at
# the very top of this script, before `st.tabs()` -- see that call's own
# comment): pop `_pending_cross_link` and apply each prefill directly,
# which is legal there because it runs before ANY tab's widgets have been
# instantiated on the new run at all -- the same reasoning the Home tab's
# own quick-action button already relies on (session_state must be set
# before the target widget runs in the SAME execution), just applied at
# the one point in this file where "before every widget" is actually
# guaranteed: the top of the script itself.
def _render_cross_link_button(label: str, *, target_tab: str, prefills: dict[str, str]) -> None:
    """Renders a "View full report" button; on click, stashes `prefills`
    and `target_tab` under `_pending_cross_link` (never a widget's own
    key, so this write is always legal regardless of what else has
    rendered this run) and calls `st.rerun()`. The actual prefill
    application happens separately, at the top of this script -- see this
    function's own module-level comment for why."""
    if st.button(label):
        st.session_state["_pending_cross_link"] = {"target_tab": target_tab, "prefills": prefills}
        st.rerun()


# ============================================================================
# UX polish pass, Part B: export/share for compiled reports.
#
# STEP 0 SCOPE, stated explicitly:
#
# Format -- static HTML, not PDF. Reuses `build_index.py`'s own established
# minimal-HTML pattern (plain HTML + inlined CSS, no framework, no build
# step -- see that module's own docstring) rather than introducing a new
# dependency: PDF generation would need a real new library this project has
# never used anywhere (weasyprint/reportlab/pdfkit, each with its own real
# install/system-dependency footprint); HTML needs nothing beyond the
# stdlib `html`/`base64` modules this file now imports. The ONE real
# difference from `build_index.py`'s own pattern: that module links to
# SEPARATE PNG files on disk (`<img src="filename.png">`, fine for a
# persistent local directory of files that all stay together); a
# `st.download_button`-delivered file is a SINGLE file a user can move or
# open anywhere, possibly with no server or sibling files present at all --
# so every image here is embedded as a base64 `data:` URI instead
# (`_image_to_data_uri` below), making the exported file genuinely
# self-contained, per this task's own explicit requirement.
#
# Scope -- Match Report, Player Report, Team Report ONLY, not all 10 tabs.
# These are this dashboard's three "compiled document" views (a real
# takeaway someone would want to save/share as one file); the rest are
# either live/ephemeral (Live CV Monitor, Alerts History, Tactical Chat),
# comparison/trend views whose own natural output is already a chart+table
# on-screen rather than a single narrative document (Team Trends, Team
# Comparison), or a single visualization already easy to screenshot (Pass
# Network). Within Player/Team Reports specifically, the export mirrors
# what the tab actually RENDERS on screen for its own MAIN report (the
# dashboard image + its most central metrics/tables) -- it does not
# separately capture every optional, separately-triggered sub-panel a tab
# can also produce (e.g. Team Reports' own opt-in Weak-Spot Lifetime
# Analysis / Decision Quality panels, each already heavy, separately
# button-triggered features in their own right) -- stated here explicitly,
# not silently incomplete.
def _image_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _html_metric_row(metrics: list[tuple[str, str]]) -> str:
    """`metrics`: `[(label, value), ...]` -- mirrors `st.metric`'s own
    label/value shape, rendered as simple bordered cards (no framework)."""
    cards = "".join(
        f'<div class="metric"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div></div>'
        for label, value in metrics
    )
    return f'<div class="metric-row">{cards}</div>'


def _html_table_from_records(records: list[dict]) -> str:
    """`records`: a list of flat dicts, all sharing the same keys (the
    shape every `pd.DataFrame`-backed table in this file already uses) --
    rendered as a plain HTML table, column order taken from the first
    record."""
    if not records:
        return "<p><em>No rows.</em></p>"
    columns = list(records[0].keys())
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns) + "</tr>"
        for r in records
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"


def _build_standalone_html_export(*, title: str, generated_note: str, sections: list[str]) -> str:
    """Wraps `sections` (pre-built HTML fragment strings) in one
    self-contained document -- same minimal plain-HTML-plus-inlined-CSS
    convention as `build_index.py`'s own `generate_report_index` (see that
    function's own docstring for the precedent this mirrors), styled
    consistently with it (the same card/table look) rather than inventing
    a new visual language for exported files."""
    body = "\n".join(sections)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem; background: #f5f5f5; color: #222; }}
  .doc {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ margin-bottom: 0.2rem; }}
  p.subtitle {{ color: #555; margin-top: 0; }}
  h2 {{ margin-top: 2.2rem; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }}
  img {{ max-width: 100%; border-radius: 4px; display: block; margin: 1rem 0; border: 1px solid #ddd; background: white; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
  th {{ background: #eee; }}
  .metric-row {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.8rem 0; }}
  .metric {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 0.6rem 1rem; }}
  .metric .label {{ font-size: 0.8rem; color: #666; }}
  .metric .value {{ font-size: 1.3rem; font-weight: bold; }}
</style>
</head>
<body>
<div class="doc">
<h1>{html.escape(title)}</h1>
<p class="subtitle">{html.escape(generated_note)}</p>
{body}
</div>
</body>
</html>
"""


def _fetch_chat_reply_safely(rest_base_url: str, session_id: str, match_id: int, message: str) -> dict | None:
    """POST /chat/tactical, reusing `_fetch_report_safely`'s exact same
    Timeout/ConnectionError/HTTPError handling -- NOT wrapped in
    `st.cache_data` (unlike the GET report fetchers above): a chat reply is
    not idempotent given identical arguments the way a report fetch is --
    the server-side session history that grounds this exact prompt has
    changed by the time an identical question might be asked again, so
    caching would risk silently replaying a stale reply."""

    def _do_post():
        response = requests.post(
            f"{rest_base_url}/chat/tactical",
            json={"session_id": session_id, "match_id": match_id, "message": message},
            timeout=CHAT_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    return _fetch_report_safely(_do_post, rest_base_url)


st.set_page_config(page_title="Project Athena Dashboard", layout="wide")
st.title("Project Athena Dashboard")
st.caption(
    "Note: the What-If Simulator and the Live Stream (both in the 'Live CV Monitor' tab) "
    "cannot run at the same time -- and while a live stream is running, EVERY OTHER TAB in "
    "this app (Player Reports, Team Reports, Team Trends, Team Comparison) is also blocked "
    "and unresponsive, not just the What-If panel. This is one single-threaded Streamlit "
    "script: a running stream blocks the entire script execution, tabs included, until it "
    "finishes or hits its max-duration/max-message cap. See this file's module docstring "
    "for why."
)

# --- StatsBomb attribution (ADR-020 clause 1.4 compliance fix) ------------
# "The User is required to accredit any publication of analysis formed
# from StatsBomb Data with the StatsBomb brand logo." Placed here,
# BEFORE the tabs are created and inside its own `st.sidebar` block, so
# it renders at the TOP of the persistent sidebar -- visible on every
# page load regardless of which tab is selected or clicked into, exactly
# the "visible without requiring a click" bar ADR-020 sets. Deliberately
# UNCONDITIONAL -- not gated behind PUBLIC_DEPLOYMENT the way the shot-map/
# Team-Trends fixes are: those exist to avoid EXPOSING data publicly;
# this exists to satisfy a real, currently-active licensing obligation
# that applies to ANY use of StatsBomb data, local or public alike, so
# gating it behind a deployment-mode flag would be wrong, not just
# unnecessary.
with st.sidebar:
    st.markdown(
        "**Data provided by [StatsBomb](https://statsbomb.com)** -- the Player Reports, "
        "Team Reports, Team Comparison, and Shot Map features are built on StatsBomb's "
        "open event/360 data, used under their Public Data User Agreement. See "
        "[ADR-020](docs/adr/ADR-020-statsbomb-open-data-licensing-scope.md) for the full "
        "licensing review."
    )
    st.divider()

# UX polish pass, Part A (cross-linking): PHASE 2 of the two-phase design
# `_render_cross_link_button`'s own module comment explains in full --
# consuming `_pending_cross_link` HERE, before `st.tabs()` and therefore
# before every tab's own widgets instantiate this run, is what makes
# setting a target widget's session_state (e.g.
# `player_report_preset_selectbox`) legal: Streamlit only forbids setting
# a widget's session_state AFTER that widget has already rendered in the
# SAME run, and nothing has rendered yet at this point in the script.
# `.pop`, not `.get` -- clears the pending request in the SAME step it's
# applied, so a cross-link jump fires exactly once, not on every
# subsequent rerun.
_pending_cross_link = st.session_state.pop("_pending_cross_link", None)
_cross_link_target_tab = None
if _pending_cross_link is not None:
    for _cl_key, _cl_value in _pending_cross_link["prefills"].items():
        st.session_state[_cl_key] = _cl_value
    _cross_link_target_tab = _pending_cross_link["target_tab"]

(
    tab_home, tab_cv, tab_player, tab_team, tab_trends, tab_compare, tab_pass_network, tab_alerts,
    tab_match_report, tab_chat,
) = st.tabs(
    [
        "Home", "Live CV Monitor", "Player Reports", "Team Reports", "Team Trends", "Team Comparison",
        "Pass Network", "Alerts History", "Match Report", "Tactical Chat",
    ],
    default=_cross_link_target_tab,
)

# `tab_home`'s LABEL is first above -- deliberately, so it renders as the
# leftmost/default-visible tab -- but its own `with tab_home:` body is
# written right after `with tab_cv:` below, NOT here and NOT at the end of
# the file. Two independent constraints, both real, both checked directly
# rather than assumed:
#
# 1. `st.tabs()` decouples VISUAL tab order (the labels list above) from
#    PHYSICAL source order (where each `with tab_x:` block sits in this
#    script) -- Streamlit renders each returned tab object into the visual
#    slot matching its position in the labels list, regardless of where its
#    `with` block appears in the file. The module docstring's "PERMANENT
#    CONSEQUENCE" section establishes a real invariant this file depends
#    on: `with tab_cv:`'s blocking WebSocket loop must stay FIRST in
#    physical source order (this script reruns top-to-bottom on every
#    interaction, and every tab body physically AFTER that loop is blocked
#    from updating until it ends) -- so `tab_home`'s body cannot be placed
#    ahead of `tab_cv`'s without breaking that invariant for every other
#    tab, even though doing so would not change Home's own VISUAL position
#    at all.
#
# 2. Home's "Quick actions" section (Step 0 §3) sets
#    `st.session_state["match_report_id_input"]` to jump the Match
#    Report/Tactical Timeline tab's shared match_id field to whatever match
#    was most recently referenced. Streamlit forbids setting a widget's
#    session_state key AFTER that widget has already been instantiated
#    within the SAME script run -- so `with tab_home:` must execute BEFORE
#    `with tab_match_report:` does, in physical source order, or that
#    assignment would raise `StreamlitAPIException` on every button click.
#
# Both constraints are satisfied by placing `with tab_home:` immediately
# after `with tab_cv:` -- second in physical order, well before
# `tab_match_report`. This also happens to be a strictly BETTER position
# for a lightweight landing tab than appending it last (tonight's usual
# convention for new tabs): Home is blocked only by an active CV stream in
# `tab_cv` (rare, user-triggered), not by every other tab's code as well.

# ============================================================================
# TAB: Live CV Monitor -- the two ORIGINAL panels (Milestones 17-19, 33),
# unchanged in behavior. Written FIRST in this file's top-to-bottom order
# on purpose: see the module docstring's "PERMANENT CONSEQUENCE" section
# for why that ordering is what makes the blocking-loop caveat above
# actually true, not just documented.
# ============================================================================
with tab_cv:
    # --- Sidebar: connection settings, shared by both panels ---------------
    with st.sidebar:
        st.header("Connection Settings")
        rest_base_url = st.text_input("REST API Base URL", value=DEFAULT_REST_BASE_URL)
        ws_url = st.text_input("WebSocket URL", value=DEFAULT_WS_URL)
        match_id = st.text_input("Match ID", value=DEFAULT_MATCH_ID)

        st.divider()
        st.header("Player/Team Report Candidates")
        st.caption(
            "The Player Reports / Team Reports dropdowns are built from a scan of "
            "data/raw/ (cached ~1 hour). Click to force a fresh scan if you've just "
            "fetched new data (e.g. via data_fallback.py)."
        )
        if "candidate_cache_bust" not in st.session_state:
            st.session_state.candidate_cache_bust = 0
        if st.button("Refresh cache list"):
            st.session_state.candidate_cache_bust += 1

        st.divider()
        st.header("Live Stream Settings")
        data_source_label = st.radio("Data Source", ["StatsBomb Replay", "CV Video Feed"], index=0)
        video_path = None
        if data_source_label == "CV Video Feed":
            video_path = st.text_input(
                "Video Path",
                value=DEFAULT_CV_VIDEO_PATH,
                help=(
                    "Server-side file path -- must resolve INSIDE the backend's data/raw/ directory. "
                    "The backend rejects (with a clean error, not a crash) any path that resolves "
                    "outside it, so don't point this at an arbitrary location on disk. The Match ID "
                    "field above is ignored for this data source. NOTE: no video ships with this "
                    "project's git history -- data/raw/ is gitignored, so a fresh clone/different "
                    "machine has none by default and you'd need to point this at your own local clip. "
                    "This specific environment happens to already have one at the default path above "
                    "(verified: real Premier League broadcast footage, not a placeholder)."
                ),
            )
        max_duration_seconds = st.number_input(
            "Max stream duration (seconds)", min_value=1, max_value=3600, value=300
        )
        max_messages = st.number_input(
            "Max message count", min_value=1, max_value=5000, value=200
        )
        start_clicked = st.button("Start Stream", type="primary")

    # ========================================================================
    # Panel 1: Tactical What-If Simulator (Milestone 19) -- rendered/checked
    # FIRST, before the live-stream section below. This ordering matters, not
    # just visually: on any script execution where "Run Simulation" was clicked,
    # this panel's single fast REST call runs and completes here, and execution
    # then falls through the (unclicked) "Start Stream" button below and the
    # script ends normally. On a script execution where "Start Stream" was
    # clicked instead, this panel's button check below is simply False and is
    # skipped in a single line, before execution reaches the live section's
    # blocking loop. Neither panel's code has to "wait" on the other in either
    # case -- but see the module docstring: this does NOT mean both can be
    # triggered in the same run. Only one button click is being processed per
    # script execution, ever.
    # ========================================================================
    st.header("Tactical What-If Simulator")

    action = st.selectbox("Tactical Action", TACTICAL_ACTIONS)
    minute = st.number_input("Match Minute", min_value=0, value=10, step=1)
    run_simulation_clicked = st.button("Run Simulation")

    def _fetch_simulation() -> dict:
        response = requests.get(
            f"{rest_base_url}/simulate",
            params={"match_id": match_id, "minute": int(minute), "action": action},
            timeout=SIMULATE_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    if run_simulation_clicked:
        # UX polish pass (Part C): this was this file's own ORIGINAL
        # loading/error pattern (a `st.empty()` placeholder + 4 duplicated
        # inline `except` clauses) -- `_fetch_report_safely`'s own
        # docstring even claimed to mirror it, but this panel was never
        # actually migrated to use the shared helper. `st.empty()` isn't
        # load-bearing here: this whole block only runs on the exact rerun
        # `run_simulation_clicked` is True, and Streamlit resets button
        # state every rerun, so there's no stale-content-from-a-previous-run
        # case for it to guard against -- confirmed by checking, not
        # assumed. Now the SAME `st.spinner` + `_fetch_report_safely`
        # pattern every other tab uses.
        with st.spinner("Running simulation..."):
            result = _fetch_report_safely(_fetch_simulation, rest_base_url)

        if result is not None:
            baseline = result["baseline_threat_15s"]
            simulated = result["simulated_threat_15s"]
            delta = result["delta"]

            metric_cols = st.columns(3)
            metric_cols[0].metric("Baseline Threat (15s)", f"{baseline * 100:.2f}%")
            metric_cols[1].metric("Simulated Threat (15s)", f"{simulated * 100:.2f}%")
            # delta_color="inverse" is deliberate: st.metric's DEFAULT
            # coloring shows a positive delta as green ("good news"), which
            # is backwards here -- a positive delta means predicted THREAT
            # went UP. "inverse" makes an increase render red/warning and a
            # decrease render green/reassuring, matching what the number
            # actually means tactically.
            metric_cols[2].metric(
                "Delta (simulated - baseline)",
                f"{delta * 100:+.2f} pp",
                delta=f"{delta * 100:+.2f} pp",
                delta_color="inverse",
            )

    # ========================================================================
    # Panel 1b: Coach Mode (new reporting track, Part C) -- the narrow,
    # genuinely new gap Step 0's dedup check identified beyond the What-If
    # Simulator above: ranking ALL tactical actions at once for the SAME
    # match/minute this panel's own inputs already select, instead of
    # requiring one manual "Run Simulation" click per action. Reuses this
    # panel's own `match_id`/`minute` inputs directly (no separate Coach
    # Mode inputs) -- GET /coach-mode, not a new simulation pipeline.
    # ========================================================================
    st.subheader("Coach Mode -- rank all tactical actions")
    st.caption(
        "Runs and ranks every tactical action at once for the match/minute above, instead of "
        "one What-If Simulator run per action. Recommends the action with the lowest simulated "
        "threat (GET /coach-mode)."
    )
    coach_mode_clicked = st.button("Run Coach Mode")

    def _fetch_coach_mode() -> dict:
        response = requests.get(
            f"{rest_base_url}/coach-mode",
            params={"match_id": match_id, "minute": int(minute)},
            timeout=SIMULATE_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    if coach_mode_clicked:
        # UX polish pass (Part C): was its own `st.empty()` placeholder --
        # same simplification as the What-If Simulator panel above, same
        # reasoning (not load-bearing; button-click gating already
        # prevents stale content across unrelated reruns).
        with st.spinner("Ranking all tactical actions..."):
            coach_mode_result = _fetch_report_safely(_fetch_coach_mode, rest_base_url)
        if coach_mode_result is not None:
            st.metric("Baseline Threat (15s)", f"{coach_mode_result['baseline_threat_15s'] * 100:.2f}%")
            st.success(f"Recommended action: **{coach_mode_result['recommended_action']}**")
            rankings_df = pd.DataFrame(coach_mode_result["rankings"])
            rankings_df["simulated_threat_15s"] = rankings_df["simulated_threat_15s"] * 100
            rankings_df["delta"] = rankings_df["delta"] * 100
            rankings_df = rankings_df.rename(columns={
                "action": "Action",
                "simulated_threat_15s": "Simulated Threat (15s) %",
                "delta": "Delta (pp)",
            })
            st.dataframe(rankings_df, width="stretch")

    st.divider()

    # ========================================================================
    # Panel 2: Live Tactical Threat Monitor (Milestone 17) -- unchanged.
    # ========================================================================
    st.header("Live Tactical Threat Monitor")

    status_placeholder = st.empty()
    chart_col, alerts_col = st.columns(2)

    with chart_col:
        st.subheader("Live Threat Probability (rolling window)")
        chart_placeholder = st.empty()
        st.caption("Tactical Momentum (rolling smoothing + trend over the same buffer above)")
        momentum_placeholder = st.empty()
        momentum_placeholder.info(
            f"Momentum: warming up (0/{MOMENTUM_MIN_MESSAGES_FOR_TREND} messages)"
        )
        st.caption("Match Segmentation (discrete game-phase, derived from Tactical Momentum above)")
        segment_placeholder = st.empty()
        segment_placeholder.info(
            f"Phase: warming up (0/{MOMENTUM_MIN_MESSAGES_FOR_TREND} messages)"
        )

    with alerts_col:
        st.subheader("Tactical Alerts")
        alerts_placeholder = st.empty()

    status_placeholder.info("Idle -- configure settings in the sidebar and click Start Stream.")

    def _render_alerts(alerts_buffer: list[str]) -> None:
        """Renders the alerts feed, most-recent-first, capped at
        MAX_ALERT_BUFFER_LEN entries (see module docstring: this buffer is a
        plain local list for the duration of the single blocking loop, not
        session_state -- it never needs to survive a rerun)."""
        if not alerts_buffer:
            alerts_placeholder.write("No alerts yet.")
            return
        alerts_placeholder.markdown("\n\n".join(f"- {text}" for text in alerts_buffer))

    def _render_momentum(momentum: dict) -> None:
        """Renders `_compute_tactical_momentum`'s return value. Uses the
        SAME `delta_color="inverse"` convention the What-If Simulator's
        own delta metric above already established (a rising number here
        means rising THREAT, which is bad news, not good news -- default
        st.metric coloring has that backwards)."""
        if momentum["status"] == "warming_up":
            momentum_placeholder.info(
                f"Momentum: warming up ({momentum['messages_so_far']}/"
                f"{momentum['messages_needed']} messages)"
            )
            return
        with momentum_placeholder.container():
            st.metric(
                "Tactical Momentum",
                momentum["classification"],
                delta=(
                    f"{momentum['trend'] * 100:+.2f} pp smoothed "
                    f"(last {MOMENTUM_TREND_LOOKBACK_MESSAGES} messages)"
                ),
                delta_color="inverse",
                help=(
                    f"Smoothed threat_15s (last {MOMENTUM_SMOOTHING_WINDOW_MESSAGES} messages): "
                    f"{momentum['smoothed_now'] * 100:.2f}%. Classification threshold: "
                    f"+/-{MOMENTUM_TREND_THRESHOLD * 100:.1f} pp trend."
                ),
            )

    # Match Segmentation (additive new feature): a discrete game-phase
    # label, alongside (not replacing) the momentum indicator above --
    # each phase rendered via a different Streamlit status container so
    # the 4 real categories stay visually distinguishable at a glance.
    _SEGMENT_RENDER_STYLE = {
        "Building Attack": "error",  # elevated, live threat -- the "pay attention" state
        "Transition": "warning",
        "Defensive Consolidation": "success",
        "Stable": "info",
    }

    def _render_segment(phase_result: dict) -> None:
        """Renders `classify_match_phase`'s return value. Deliberately
        shows BOTH the hysteresis-smoothed `phase` (the headline label)
        and the RAW pre-hysteresis `raw_phase` (in the caption) -- the
        same "show the smoothed AND the underlying number" transparency
        `_render_momentum` above already applies to `classification` vs.
        `trend`."""
        if phase_result["status"] == "warming_up":
            segment_placeholder.info(
                f"Phase: warming up ({phase_result['messages_so_far']}/"
                f"{phase_result['messages_needed']} messages)"
            )
            return
        phase = phase_result["phase"]
        style = _SEGMENT_RENDER_STYLE.get(phase, "info")
        render_fn = getattr(segment_placeholder, style)
        render_fn(
            f"Match Phase: **{phase}**\n\n"
            f"Smoothed threat_15s: {phase_result['smoothed_now'] * 100:.2f}% "
            f"(elevated threshold: {ELEVATED_THREAT_LEVEL * 100:.0f}%), "
            f"momentum: {phase_result['momentum_classification']}, "
            f"raw (pre-hysteresis) phase: {phase_result['raw_phase']} "
            f"(dwell window: {SEGMENT_DWELL_MESSAGES} messages)."
        )

    if start_clicked:
        # Milestone 33: which data source's query params to build depends on
        # the sidebar selection -- source=cv requires video_path (server-side,
        # must resolve inside data/raw/); source=statsbomb (the default,
        # unchanged from Milestone 17) uses match_id.
        if data_source_label == "CV Video Feed":
            connection_url = f"{ws_url}?source=cv&video_path={video_path}"
        else:
            connection_url = f"{ws_url}?source=statsbomb&match_id={match_id}"

        # Rolling, CAPPED buffers -- plain local variables, intentionally NOT
        # session_state, since the entire stream lifetime happens within this
        # single, uninterrupted script execution (see module docstring).
        threat_buffer: list[float] = []
        alerts_buffer: list[str] = []
        message_count = 0
        latest_real_time_lag_sec: float | None = None  # only ever set by the CV source

        status_placeholder.info("Connecting...")

        try:
            ws_connection = websocket.create_connection(connection_url, timeout=RECV_TIMEOUT_SECONDS)
        except Exception as exc:
            status_placeholder.error(
                f"Connection failed ({exc}). Confirm the FastAPI backend is running, then click "
                "Start Stream to retry."
            )
            ws_connection = None

        if ws_connection is not None:
            start_time = time.monotonic()
            stream_error = None

            try:
                while True:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= max_duration_seconds:
                        break
                    if message_count >= max_messages:
                        break

                    try:
                        raw_message = ws_connection.recv()
                    except Exception as exc:
                        stream_error = str(exc)
                        break

                    if not raw_message:
                        # Empty frame signals the server closed the connection.
                        stream_error = "Server closed the connection."
                        break

                    try:
                        message = json.loads(raw_message)
                    except json.JSONDecodeError:
                        # A malformed frame shouldn't take down the whole
                        # session -- skip it and keep streaming.
                        continue

                    message_count += 1
                    message_type = message.get("type")

                    if message_type == "threat":
                        threat_buffer.append(message.get("threat_15s", 0.0))
                        if len(threat_buffer) > MAX_THREAT_BUFFER_LEN:
                            threat_buffer.pop(0)
                        chart_placeholder.line_chart(pd.DataFrame({"threat_15s": threat_buffer}))
                        _render_momentum(_compute_tactical_momentum(threat_buffer))
                        _render_segment(classify_match_phase(threat_buffer))
                        # real_time_lag_sec (Milestone 33, CV source only): how
                        # far behind real video time the stream currently is.
                        # Surfaced honestly rather than silently either
                        # sprinting through the match or claiming a pace it
                        # isn't keeping -- see api.py's _stream_cv_source.
                        if "real_time_lag_sec" in message:
                            latest_real_time_lag_sec = message["real_time_lag_sec"]

                    elif message_type == "alert":
                        alert_text = message.get("explanation", "(empty alert)")
                        alerts_buffer.insert(0, alert_text)
                        if len(alerts_buffer) > MAX_ALERT_BUFFER_LEN:
                            alerts_buffer.pop()
                        _render_alerts(alerts_buffer)

                    lag_suffix = ""
                    if latest_real_time_lag_sec is not None:
                        if latest_real_time_lag_sec > 0.5:
                            lag_suffix = f" -- running {latest_real_time_lag_sec:.1f}s behind real-time"
                        else:
                            lag_suffix = " -- keeping real-time pace"
                    status_placeholder.info(f"Streaming... ({message_count} messages received){lag_suffix}")
            finally:
                try:
                    ws_connection.close()
                except Exception:
                    pass

            if stream_error is not None:
                status_placeholder.error(
                    f"Connection lost -- {stream_error} Click Start Stream to reconnect."
                )
            else:
                status_placeholder.success(
                    f"Stream ended after {message_count} messages "
                    f"({time.monotonic() - start_time:.1f}s) -- click Start Stream to continue."
                )

# ============================================================================
# TAB: Home -- landing/orientation view. VISUALLY leftmost (see the
# `st.tabs()` call's own comment above for why this `with` block is
# nonetheless physically SECOND, right after `tab_cv`, not first or last).
# Pure read-only summary/orchestration layer: every figure below is reused
# from an existing scan (`candidate_index.py`), an existing query
# (`alert_store.py`, via the same `/alerts/history` endpoint the Alerts
# History tab already calls), or an existing endpoint (`/health`,
# `/metrics`) -- nothing here is computed fresh, and no report-generation/
# model/physics logic is touched. See the "Home tab helpers" section above
# for the full Step 0 scope writeup.
# ============================================================================
with tab_home:
    st.header("Home")
    st.caption(
        "Orientation before picking a tab -- what's cached locally, what's happened recently, "
        "quick shortcuts into the deepest views, and whether the backend is up. Every figure "
        "on this tab is reused from an existing scan, query, or endpoint; nothing here is "
        "computed fresh."
    )

    # --- System status (Step 0 §4: existing GET /health, GET /metrics) -----
    st.subheader("System Status")
    home_health = _fetch_report_safely(lambda: _cached_health(rest_base_url), rest_base_url)
    home_metrics = _fetch_report_safely(lambda: _cached_metrics(rest_base_url), rest_base_url)

    status_cols = st.columns(4)
    if home_health is not None:
        status_cols[0].metric("Backend Status", home_health["status"])
        status_cols[1].metric("Model Loaded", "Yes" if home_health["model_loaded"] else "No")
        home_uptime = home_health.get("uptime_seconds")
        status_cols[2].metric("Uptime", f"{home_uptime / 60:.1f} min" if home_uptime is not None else "n/a")
    else:
        status_cols[0].metric("Backend Status", "unreachable")
        status_cols[1].metric("Model Loaded", "n/a")
        status_cols[2].metric("Uptime", "n/a")
    status_cols[3].metric(
        "Alerts Logged (Total)",
        home_metrics.get("total_alerts_logged", "n/a") if home_metrics is not None else "n/a",
    )

    st.divider()

    # --- Data coverage summary (Step 0 §1) ----------------------------------
    st.subheader("Data Coverage")
    home_cache_bust = st.session_state.get("candidate_cache_bust", 0)
    home_teams, home_players = _cached_candidate_index(home_cache_bust)
    home_coverage = _home_data_coverage_summary(home_teams, home_players)

    coverage_cols = st.columns(4)
    coverage_cols[0].metric("Players Cached", home_coverage["total_players_cached"])
    coverage_cols[1].metric("Teams Cached", home_coverage["total_teams_cached"])
    coverage_cols[2].metric("Matches Cached", home_coverage["total_matches_cached"])
    coverage_cols[3].metric("Matches with 360 Coverage", home_coverage["total_matches_360"])
    st.caption("See the Player Reports / Team Reports tabs to explore this data.")

    st.divider()

    # --- Recent activity (Step 0 §2) + Quick actions (Step 0 §3) -----------
    st.subheader("Recent Activity")
    home_recent_alerts = _fetch_report_safely(
        lambda: _cached_alerts_history(rest_base_url, None, None, None, None, HOME_RECENT_ALERTS_LIMIT),
        rest_base_url,
    )

    if home_recent_alerts is None:
        st.info("Could not fetch alert history -- see the error above.")
    elif len(home_recent_alerts) == 0:
        st.info("No alerts logged yet. Once activity starts, it will show here and in the Alerts History tab.")
    else:
        home_display_df = pd.DataFrame(home_recent_alerts)[[
            "logged_at_utc", "source", "match_id", "minute", "delta",
        ]].rename(columns={
            "logged_at_utc": "Timestamp (UTC)",
            "source": "Source",
            "match_id": "Match ID",
            "minute": "Minute",
            "delta": "Threat Delta",
        })
        st.dataframe(home_display_df, width="stretch")

        home_total_logged = home_metrics.get("total_alerts_logged") if home_metrics is not None else None
        if home_total_logged is not None:
            st.caption(
                f"Showing {len(home_recent_alerts)} most recent of {home_total_logged} total -- "
                "see the Alerts History tab for full history and filters."
            )
        else:
            st.caption("See the Alerts History tab for full history and filters.")

        st.markdown("**Quick actions**")
        home_recent_match_id = _home_most_recent_match_id(home_recent_alerts)
        if home_recent_match_id is not None:
            if st.button(f"Jump to match_id={home_recent_match_id} in Match Report / Tactical Timeline"):
                st.session_state["match_report_id_input"] = str(home_recent_match_id)
                st.success(
                    f"Match ID field in the Match Report tab is now set to {home_recent_match_id} -- "
                    "click that tab to view. Covers both Automatic Match Report/Tactical Event "
                    "Detection and Tactical Timeline, which share the same Match ID field."
                )
        else:
            st.caption("No recent alert carries a match_id yet -- no quick-action shortcut to offer.")

# ============================================================================
# TAB: Player Reports -- report DATA now fetched over HTTP from api.py's
# /reports/player/{player_id} (ADR-018); PNG rendering still calls
# player_visualizer.py directly (pure client-side rendering of an
# already-fetched dict, no MLflow/data/raw access of its own). Neither
# player_report.py nor player_visualizer.py is modified. See module
# docstring for the caching rationale.
# ============================================================================
with tab_player:
    st.header("Player Report")

    _, _cached_players = _cached_candidate_index(st.session_state.candidate_cache_bust)

    # Label format matches this tab's own pre-existing preset convention
    # ("Name (id) -- LOW SAMPLE, N events" / "-- well-supported") --
    # LOW_SAMPLE_EVENT_THRESHOLD (candidate_index.py, = habit_memory.
    # MIN_HISTORICAL_EVENTS = 20) is the SAME cutoff `generate_player_report`
    # itself effectively uses for `heatmap_used_uniform_fallback`, so this
    # label is a real signal, not an arbitrary one. `total_events` here is
    # a cheap, raw tagged-event count -- see candidate_index.py's own
    # docstring for exactly how it differs from the real report's own
    # (narrower) `positional_distribution_event_count`.
    _player_labels: dict[str, dict] = {}
    for _p in _cached_players:
        _tag = f"LOW SAMPLE, {_p['total_events']} event(s)" if _p["low_sample"] else "well-supported"
        _label = f"{_p['name']} ({_p['player_id']}) -- {_tag}, {len(_p['seasons'])} season(s) cached"
        _player_labels[_label] = _p

    # `key=` on these three widgets (UX polish pass, Part A): added so a
    # "View full report" cross-link button elsewhere in this file can
    # pre-fill them via `st.session_state` before an `st.rerun()` -- see
    # `_render_cross_link_button`'s own docstring for why "Custom" mode
    # specifically (not the preset dropdown) is the cross-link target: a
    # preset's own OPTION STRING is a long, dynamically-scanned label
    # ("Name (id) -- well-supported, N season(s) cached") a cross-link
    # button would have to reconstruct byte-for-byte to select it via
    # session_state, which is fragile and tightly coupled to
    # candidate_index.py's own label formatting; "Custom" is always a
    # stable, literal string in both this list and Team Reports' own preset
    # list, and its own fields take plain player_id/match_ids values
    # directly -- a far more robust target. Zero behavior change for a user
    # who never uses a cross-link button: an explicit `key` only enables
    # EXTERNAL control via session_state, it doesn't change a widget's
    # default behavior otherwise.
    preset_label = st.selectbox(
        "Player", list(_player_labels.keys()) + ["Custom"], key="player_report_preset_selectbox"
    )
    if preset_label == "Custom":
        player_id_input = st.text_input(
            "Player ID (StatsBomb player_id)", value="", key="player_report_custom_player_id"
        )
        match_ids_input = st.text_input(
            "Match IDs (comma-separated StatsBomb match_id list)", value="",
            key="player_report_custom_match_ids",
        )
        player_id = int(player_id_input) if player_id_input.strip() else None
        match_ids = (
            tuple(int(m.strip()) for m in match_ids_input.split(",") if m.strip())
            if match_ids_input.strip() else ()
        )
    else:
        _candidate = _player_labels[preset_label]
        player_id = _candidate["player_id"]

        # Season sub-selector (Step 2.2): a player with cached data across
        # multiple competition-seasons (Messi: 25, not the single flat
        # entry a hardcoded preset would have offered) gets ONE dropdown
        # entry here, plus this multiselect -- letting a user combine any
        # subset of their cached seasons (a single season for "early
        # Messi", the two Ligue 1 seasons for "PSG Messi", or everything
        # for a full-career aggregate) rather than only ever an all-time
        # aggregate. Defaults to ALL cached seasons selected, matching
        # what a flat preset would have shown by default.
        _season_labels: dict[str, dict] = {}
        for _s in _candidate["seasons"]:
            _slabel = f"{_s['competition_name']} {_s['season_name']} ({_s['event_count']} events, {len(_s['match_ids'])} match(es))"
            _season_labels[_slabel] = _s
        if len(_season_labels) > 1:
            _selected_season_labels = st.multiselect(
                "Season(s)", options=list(_season_labels.keys()), default=list(_season_labels.keys())
            )
        else:
            _selected_season_labels = list(_season_labels.keys())

        match_ids = tuple(
            sorted({mid for lbl in _selected_season_labels for mid in _season_labels[lbl]["match_ids"]})
        )

    generate_clicked = st.button("Generate Player Report")

    if generate_clicked:
        if not player_id or not match_ids:
            st.error("Provide a player_id and at least one match_id.")
        else:
            with st.spinner("Generating report..."):
                report = _fetch_report_safely(
                    lambda: _cached_player_report(rest_base_url, player_id, match_ids), rest_base_url
                )

            if report is not None:
                png_bytes = _cached_player_png(report)

                # Milestone 44's validation sweep found a real gap: a
                # 1-event player's positional_distribution/heatmap look just
                # as "confident" as a well-supported player's. That fix must
                # survive into THIS UI as a real, visible Streamlit element
                # -- not just as banners baked into the PNG image, which a
                # user could plausibly skim past. These fields are passed
                # through the /reports/player/{player_id} endpoint's JSON
                # response completely unchanged (ADR-018), so this check
                # works identically whether report came from a direct call
                # or, as now, over HTTP.
                if report.get("heatmap_used_uniform_fallback"):
                    st.warning(
                        f"LOW SAMPLE: only {report.get('heatmap_event_count', 0)} qualifying event(s) "
                        "for this player -- the heatmap below is a UNIFORM FALLBACK (habit_memory's "
                        "own cold-start threshold, MIN_HISTORICAL_EVENTS), not a real learned pattern. "
                        "Treat this report as illustrative, not a confident finding."
                    )
                elif report.get("positional_distribution_event_count", 0) < 20:
                    st.warning(
                        f"LOW SAMPLE: positional distribution is based on only "
                        f"{report.get('positional_distribution_event_count', 0)} tagged event(s) -- "
                        "not a confident distribution."
                    )

                st.image(png_bytes, caption=f"Player Report -- player_id={player_id}", width="stretch")

                with st.expander("Raw report data"):
                    st.json(report)

                # --- Shot Map (additive new feature) ------------------------
                # A SEPARATE section, alongside (not replacing or
                # reorganizing) the positional-distribution/heatmap panels
                # above -- fetched from the NEW, dedicated
                # /reports/player/{player_id}/shot-map endpoint (see
                # api.py's own comment for why this is a separate endpoint
                # rather than a field added to the existing player-report
                # response). xG values shown here are StatsBomb's own real
                # statsbomb_xg per shot -- NOT this project's DeepHit
                # threat model, a different quantity (see
                # generate_player_shot_map's docstring).
                st.divider()
                st.subheader("Shot Map")
                # UX polish pass (Part B): tracks whether an exportable shot
                # map image actually got rendered this run -- stays None on
                # the "no data"/"configuration error" paths below, so the
                # Export button (further down this tab) can honestly omit
                # it rather than reference an undefined variable.
                _export_shot_map_png = None
                with st.spinner("Generating shot map..."):
                    shot_map = _fetch_report_safely(
                        lambda: _cached_player_shot_map(rest_base_url, player_id, match_ids), rest_base_url
                    )
                if shot_map is not None:
                    # Same low-sample visual convention as the positional/
                    # heatmap warnings above -- reused, not reinvented.
                    if shot_map.get("shot_map_used_low_sample_flag"):
                        st.warning(
                            f"LOW SAMPLE: only {shot_map.get('total_shots', 0)} shot(s) for this player -- "
                            "treat the shot map below as illustrative, not a confident pattern."
                        )

                    # ADR-021 condition-2 compliance fix. DEFENSE IN DEPTH,
                    # deliberately checking BOTH signals rather than trusting
                    # either alone:
                    #   1. response_has_raw_shots -- the ACTUAL shape of what
                    #      api.py returned (ground truth: api.py's own
                    #      PUBLIC_DEPLOYMENT flag already decided server-side
                    #      whether generate_player_shot_map's raw `shots` list
                    #      was ever computed at all).
                    #   2. this process's OWN PUBLIC_DEPLOYMENT flag.
                    # If this dashboard's flag says public but the response
                    # still carries raw shots (the two processes' flags are
                    # misconfigured out of sync), FAIL CLOSED: refuse to
                    # render or display anything from this panel rather than
                    # silently trust either signal alone -- a compliance
                    # boundary should never rely on one unverified check.
                    response_has_raw_shots = "shots" in shot_map

                    if PUBLIC_DEPLOYMENT and response_has_raw_shots:
                        st.error(
                            "Configuration error: this dashboard process has PUBLIC_DEPLOYMENT=true, "
                            "but the API server returned raw per-shot data -- refusing to render or "
                            "display it. Set PUBLIC_DEPLOYMENT=true on the API server process too "
                            "(see README.md's 'Public deployment mode' section)."
                        )
                    elif PUBLIC_DEPLOYMENT or not response_has_raw_shots:
                        # Aggregated path: either this deployment is public
                        # (and the server correctly agreed, per the check
                        # above), or the server is simply already running in
                        # aggregated mode regardless of this process's own
                        # flag -- either way, no raw per-shot data exists on
                        # this dict, so it's safe to render/display.
                        shot_map_png = _cached_player_shot_map_aggregated_png(shot_map)
                        st.image(
                            shot_map_png,
                            caption=f"Shot Map (aggregated) -- player_id={player_id}",
                            width="stretch",
                        )
                        with st.expander("Raw shot map data (aggregated grids only -- no per-shot data)"):
                            st.json(shot_map)
                        _export_shot_map_png = shot_map_png
                    else:
                        # Local/private mode, response confirmed to carry
                        # real per-shot data -- unchanged from before this
                        # flag existed.
                        shot_map_png = _cached_player_shot_map_png(shot_map)
                        st.image(shot_map_png, caption=f"Shot Map -- player_id={player_id}", width="stretch")
                        with st.expander("Raw shot map data"):
                            st.json(shot_map)
                        _export_shot_map_png = shot_map_png

                # --- Player Dashboard: match-level views (additive new feature) ---
                # A THIRD section, alongside (not replacing) the positional-
                # distribution/heatmap panels and the Shot Map above.
                # Match Summary is unconditionally aggregate (ADR-021: not
                # gated). Touch Map and Timeline are per-MATCH views (a
                # touch map/timeline for the whole requested match_ids set
                # combined would not mean anything -- same single-match-
                # scope reasoning the Pass Network tab already established),
                # so they get their own match_id selector, scoped to the
                # matches already selected for this player above.
                st.divider()
                st.subheader("Player Dashboard: Match-by-Match Views")

                with st.spinner("Generating match summary..."):
                    match_summary = _fetch_report_safely(
                        lambda: _cached_player_match_summary(rest_base_url, player_id, match_ids), rest_base_url
                    )
                if match_summary is not None:
                    st.write(f"Appeared in {match_summary['matches_player_appeared_in']} of {match_summary['matches_requested']} requested match(es).")
                    if match_summary["matches"]:
                        summary_df = pd.DataFrame([
                            {
                                "Match ID": m["match_id"],
                                "Team": m["team"],
                                "Opponent": m["opponent"],
                                "Minutes Played": round(m["minutes_played"], 1),
                                "Total Tagged Events": m["total_tagged_events"],
                                "Shots": m["event_type_counts"].get("Shot", 0),
                                "Passes": m["event_type_counts"].get("Pass", 0),
                            }
                            for m in match_summary["matches"]
                        ])
                        st.dataframe(summary_df, width="stretch")
                        with st.expander("Raw match summary data (full event-type counts per match)"):
                            st.json(match_summary)
                        _export_match_summary_records = summary_df.to_dict("records")
                    else:
                        st.info("No match-level data -- player did not appear in any requested match.")
                        _export_match_summary_records = []

                    # --- Export (UX polish pass, Part B) --------------------
                    # Scope, stated explicitly: the main dashboard image,
                    # the Shot Map, and the match summary table -- the
                    # report's three "always attempted on Generate" panels.
                    # Press Resistance Index / touch map / timeline /
                    # Player Similarity Search below are each their OWN,
                    # separately-triggered sub-panel (own button, own
                    # spinner) -- consistent with Team Report's own export
                    # scoping (see this file's Part B module-level comment).
                    _player_report_sections = [
                        f"<h2>Report</h2><img src=\"{_image_to_data_uri(png_bytes)}\" alt=\"Player report\">"
                    ]
                    if _export_shot_map_png is not None:
                        _player_report_sections.append(
                            f"<h2>Shot Map</h2><img src=\"{_image_to_data_uri(_export_shot_map_png)}\" alt=\"Shot map\">"
                        )
                    _player_report_sections.append(
                        "<h2>Match-by-Match Summary</h2>" + _html_table_from_records(_export_match_summary_records)
                    )
                    _player_report_html = _build_standalone_html_export(
                        title=f"Player Report -- player_id={player_id}",
                        generated_note=(
                            "Project Athena -- Player Report. Static, self-contained HTML -- opens "
                            "directly from the filesystem, no server required."
                        ),
                        sections=_player_report_sections,
                    )
                    st.download_button(
                        "Export Player Report (HTML)",
                        data=_player_report_html,
                        file_name=f"player_report_{player_id}.html",
                        mime="text/html",
                    )

                    st.markdown("**Press Resistance Index** (successful action while under pressure)")
                    with st.spinner("Generating Press Resistance Index..."):
                        pri = _fetch_report_safely(
                            lambda: _cached_player_press_resistance_index(rest_base_url, player_id, match_ids),
                            rest_base_url,
                        )
                    if pri is not None:
                        if pri["press_resistance_index_used_low_sample_flag"]:
                            st.warning(
                                f"LOW SAMPLE: only {pri['overall']['under_pressure_attempts']} under-pressure "
                                "action(s) across the requested match(es) -- treat the rate(s) below as "
                                "illustrative, not a confident finding."
                            )
                        overall = pri["overall"]
                        overall_rate = overall["success_rate"]
                        st.metric(
                            "Overall Press Resistance Rate",
                            f"{overall_rate:.1%}" if overall_rate is not None else "N/A",
                            help=(
                                f"{overall['successful_under_pressure']} successful / "
                                f"{overall['under_pressure_attempts']} total under-pressure actions "
                                "(Pass, Dribble, Shot combined)."
                            ),
                        )
                        pri_df = pd.DataFrame([
                            {
                                "Event Type": event_type.capitalize(),
                                "Under-Pressure Attempts": stats["under_pressure_attempts"],
                                "Successful": stats["successful_under_pressure"],
                                "Success Rate": f"{stats['success_rate']:.1%}" if stats["success_rate"] is not None else "N/A",
                            }
                            for event_type, stats in pri["event_types"].items()
                        ])
                        st.dataframe(pri_df, width="stretch")
                        with st.expander("Raw Press Resistance Index data"):
                            st.json(pri)

                    if match_ids:
                        st.markdown("**Touch Map / Key-Event Timeline** (pick one match)")
                        selected_match_id = st.selectbox(
                            "Match ID for match-level views", options=list(match_ids), key="player_dashboard_match_id"
                        )

                        with st.spinner("Generating touch map..."):
                            touch_map = _fetch_report_safely(
                                lambda: _cached_player_match_touch_map(rest_base_url, player_id, selected_match_id),
                                rest_base_url,
                            )
                        if touch_map is not None and not touch_map.get("no_data"):
                            if touch_map.get("touch_map_used_low_sample_flag"):
                                st.warning(
                                    f"LOW SAMPLE: only {touch_map.get('total_touches', 0)} touch(es) in this "
                                    "match -- treat the touch map below as illustrative, not a confident pattern."
                                )
                            # Same defense-in-depth PUBLIC_DEPLOYMENT check as the
                            # Shot Map/Pass Network panels above -- reused, not
                            # reinvented. See those panels' own comments for the
                            # full reasoning.
                            response_has_raw_touches = "touches" in touch_map
                            if PUBLIC_DEPLOYMENT and response_has_raw_touches:
                                st.error(
                                    "Configuration error: this dashboard process has PUBLIC_DEPLOYMENT=true, "
                                    "but the API server returned raw per-touch location data -- refusing to "
                                    "render or display it. Set PUBLIC_DEPLOYMENT=true on the API server "
                                    "process too (see README.md's 'Public deployment mode' section)."
                                )
                            elif PUBLIC_DEPLOYMENT or not response_has_raw_touches:
                                touch_png = _cached_player_match_touch_map_aggregated_png(touch_map)
                                st.image(
                                    touch_png,
                                    caption=f"Touch Map (aggregated) -- player_id={player_id}, match_id={selected_match_id}",
                                    width="stretch",
                                )
                                with st.expander("Raw touch map data (aggregated grid only -- no per-touch data)"):
                                    st.json(touch_map)
                            else:
                                touch_png = _cached_player_match_touch_map_png(touch_map)
                                st.image(
                                    touch_png,
                                    caption=f"Touch Map -- player_id={player_id}, match_id={selected_match_id}",
                                    width="stretch",
                                )
                                with st.expander("Raw touch map data"):
                                    st.json(touch_map)

                        with st.spinner("Generating key-event timeline..."):
                            timeline = _fetch_report_safely(
                                lambda: _cached_player_match_timeline(rest_base_url, player_id, selected_match_id),
                                rest_base_url,
                            )
                        if timeline is not None and not timeline.get("no_data"):
                            response_has_raw_timeline = "timeline" in timeline
                            if PUBLIC_DEPLOYMENT and response_has_raw_timeline:
                                st.error(
                                    "Configuration error: this dashboard process has PUBLIC_DEPLOYMENT=true, "
                                    "but the API server returned a raw per-event timeline -- refusing to "
                                    "render or display it. Set PUBLIC_DEPLOYMENT=true on the API server "
                                    "process too (see README.md's 'Public deployment mode' section)."
                                )
                            elif PUBLIC_DEPLOYMENT or not response_has_raw_timeline:
                                timeline_png = _cached_player_match_timeline_aggregated_png(timeline)
                                st.image(
                                    timeline_png,
                                    caption=f"Key-Event Timeline (aggregated) -- player_id={player_id}, match_id={selected_match_id}",
                                    width="stretch",
                                )
                                with st.expander("Raw timeline data (event-type counts per time bucket only)"):
                                    st.json(timeline)
                            else:
                                timeline_png = _cached_player_match_timeline_png(timeline)
                                st.image(
                                    timeline_png,
                                    caption=f"Key-Event Timeline -- player_id={player_id}, match_id={selected_match_id}",
                                    width="stretch",
                                )
                                with st.expander("Raw timeline data"):
                                    st.json(timeline)

                st.divider()
                st.subheader("Player Similarity Search")
                st.caption(
                    "Cosine similarity over a 15-feature style profile (positional role, Press "
                    "Resistance Index, shot volume/quality/technique) -- searches for similar TYPE "
                    "of player, not similar overall activity level. Reads an offline-precomputed "
                    "index; does not recompute the population live. See ADR-021's own addendum for "
                    "why this is unconditional (not gated behind PUBLIC_DEPLOYMENT)."
                )

                rebuild_col, _ = st.columns([1, 3])
                with rebuild_col:
                    if st.button("Rebuild similarity index", help=(
                        "A real, slow (measured ~16 minutes across this project's full real "
                        "population) operation, manually triggered ONLY -- there is no automatic "
                        "rebuild. Run this once after fetching new player data, not on every visit."
                    )):
                        # UX polish pass (Part C): was its own inline
                        # try/except catching only the broad
                        # `RequestException` parent class (Timeout/
                        # ConnectionError/HTTPError all collapsed into one
                        # generic message, unlike every other tab's own
                        # 3-way-differentiated error). Now the same shared
                        # helper, same message templates, everywhere.
                        with st.spinner("Rebuilding similarity index (this genuinely takes a while)..."):
                            rebuild_result = _fetch_report_safely(
                                lambda: _rebuild_similarity_index(rest_base_url), rest_base_url
                            )
                        if rebuild_result is not None:
                            st.success(
                                f"Indexed {rebuild_result['searchable_population_size']} of "
                                f"{rebuild_result['total_cached_population_size']} cached players "
                                f"in {rebuild_result['build_duration_seconds']:.1f}s."
                            )
                            st.cache_data.clear()

                top_k = st.number_input("Top-K similar players", min_value=1, max_value=20, value=5, step=1)
                if st.button("Find Similar Players"):
                    # UX polish pass (Part C): was its own inline try/except
                    # duplicating `_fetch_report_safely`'s logic (plus a
                    # special 404 case) -- now absorbed into the shared
                    # helper via `not_found_message`, so this special case
                    # no longer needs its own copy of the other 3 exception
                    # branches.
                    with st.spinner("Querying similarity index..."):
                        similar = _fetch_report_safely(
                            lambda: _cached_similar_players(rest_base_url, player_id, int(top_k)),
                            rest_base_url,
                            not_found_message=(
                                "No similarity index found yet -- click 'Rebuild similarity index' above first."
                            ),
                        )
                    # UX polish pass (Part A): stored in session_state and
                    # rendered below, outside this transient `if
                    # ..._clicked:` gate -- see `_render_cross_link_button`'s
                    # own module comment for the "orphaned click" bug this
                    # fixes (found by actually testing it, not assumed).
                    st.session_state["_similar_players_result"] = {"player_id": player_id, "data": similar}

                _sp_result = st.session_state.get("_similar_players_result")
                if _sp_result is not None:
                    _sp_player_id = _sp_result["player_id"]
                    similar = _sp_result["data"]
                    if similar is not None:
                        if similar.get("no_data"):
                            st.info(similar.get("reason", "This player is not in the searchable population."))
                        else:
                            st.write(
                                f"Players most similar to **{similar['name']}** (player_id={_sp_player_id}), "
                                f"out of {similar['searchable_population_size']} searchable players:"
                            )
                            similar_df = pd.DataFrame([
                                {
                                    "Player": s["name"],
                                    "Player ID": s["player_id"],
                                    "Similarity": f"{s['similarity']:.3f}",
                                    "Driven by": ", ".join(s["matched_features"]),
                                }
                                for s in similar["similar_players"]
                            ])
                            st.dataframe(similar_df, width="stretch")
                            with st.expander("Raw player similarity data"):
                                st.json(similar)

                            # --- Cross-linking (UX polish pass, Part A) -----
                            # Real navigation problem this closes (Step 0):
                            # a similar player's name/player_id appears here
                            # with no path to their own Player Report.
                            # HONEST LIMITATION, stated explicitly rather
                            # than silently guessed: this endpoint's own
                            # response (`find_similar_players`) carries no
                            # match_ids for the MATCHED players -- only
                            # name/player_id/similarity/matched_features
                            # (confirmed directly in player_similarity.py).
                            # Cross-referencing `_cached_players` (already
                            # loaded in this same tab, no new call) recovers
                            # match_ids for any matched player who is ALSO
                            # in the local cache -- the common case, since
                            # the similarity index itself is built from that
                            # same cached population. When a match isn't
                            # found (an edge case, not the norm), the button
                            # still pre-fills player_id and gets the user to
                            # the right tab in Custom mode -- they supply
                            # match_ids themselves -- rather than being
                            # silently omitted.
                            st.markdown("**Jump to a similar player's full report:**")
                            _similar_match_ids_by_player: dict[int, list[int]] = {
                                _p["player_id"]: sorted(
                                    {mid for _s in _p["seasons"] for mid in _s["match_ids"]}
                                )
                                for _p in _cached_players
                            }
                            _similar_cols = st.columns(3)
                            for _sim_i, _sim_player in enumerate(similar["similar_players"]):
                                _sim_match_ids = _similar_match_ids_by_player.get(_sim_player["player_id"], [])
                                with _similar_cols[_sim_i % 3]:
                                    _render_cross_link_button(
                                        f"{_sim_player['name']} →",
                                        target_tab="Player Reports",
                                        prefills={
                                            "player_report_preset_selectbox": "Custom",
                                            "player_report_custom_player_id": str(_sim_player["player_id"]),
                                            "player_report_custom_match_ids": ",".join(
                                                str(m) for m in _sim_match_ids
                                            ),
                                        },
                                    )

# ============================================================================
# TAB: Team Reports -- report DATA now fetched over HTTP from api.py's
# /reports/team/{team_name} (ADR-018); PNG rendering still calls
# team_visualizer.py directly (pure client-side rendering, no MLflow/
# data/raw access of its own). Neither team_report.py nor
# team_visualizer.py is modified.
# ============================================================================
with tab_team:
    st.header("Team Report")

    # Cross-panel data-tiering note (static, always visible -- not
    # conditional on any one request's outcome): different report types
    # on this tab have genuinely different real-data requirements, so it
    # is expected and correct -- not a bug -- for one panel to succeed
    # while another fails on the IDENTICAL team/season selection. Added
    # after a real reproduced case (Arsenal, Premier League 2003/04: Team
    # Report correctly shows "no 360-covered matches, widen your
    # selection" while Tactical Entropy correctly succeeds with real
    # numbers on that same selection) left no on-page explanation for why.
    st.info(
        "This tab mixes two different kinds of report, with different real-data requirements: "
        "**Team Report** (pitch-control heatmap, threat-by-zone) is the one report in this entire "
        "dashboard built on 360 freeze-frame coverage (via `BiomechanicalPitchControl`), which is far "
        "rarer than plain event data -- a team/season with 0 360-covered matches will correctly show "
        "a 'no 360-covered matches' message. **Tactical Entropy** (pass-direction predictability) only "
        "needs event data, no 360 coverage required, so it can succeed on the same selection even when "
        "Team Report can't. Seeing one panel work and the other not for the identical selection is "
        "expected behavior, not an error. Verified directly, not assumed: every OTHER report in this "
        "dashboard -- Player Reports (season report, shot map, touch map, timeline, Press Resistance "
        "Index, match summary), Pass Network (both raw and aggregated), and Tactical Entropy here -- "
        "is event-data-only too, same tier as Tactical Entropy; Team Report's pitch-control panel is "
        "the sole exception, not one of several."
    )

    _cached_teams, _ = _cached_candidate_index(st.session_state.candidate_cache_bust)

    # Post-audit correction: label now reflects `total_matches_360` (the
    # SAME 360-covered-chain count team_report.py's own `matches_used`
    # measures -- candidate_index.py independently reimplements that
    # chain-building step, cheaply, without running physics/ML), not raw
    # cached match count. An earlier verification audit found the two have
    # essentially no relationship (Real Madrid: 68 cached matches, only 2
    # with usable 360 coverage) -- this label would have silently called
    # Real Madrid "well-supported" under the old metric. See
    # candidate_index.py's own module docstring for the full reasoning.
    _team_labels: dict[str, dict] = {}
    for _t in _cached_teams:
        _tag = f"LOW SAMPLE, {_t['total_matches_360']} 360-covered match(es)" if _t["low_sample"] else "well-supported"
        _label = f"{_t['team_name']} -- {_tag} (of {_t['total_matches_cached']} cached), {len(_t['seasons'])} season(s)"
        _team_labels[_label] = _t

    # `key=` added for the same cross-link reason as the Player Reports
    # tab's own preset selectbox -- see that widget's own comment. Team
    # Reports' own "Custom" fields (`team_report_name`/`team_report_match_ids`
    # below) already had explicit keys before this pass.
    team_preset_label = st.selectbox(
        "Team", list(_team_labels.keys()) + ["Custom"], key="team_report_preset_selectbox"
    )
    if team_preset_label == "Custom":
        team_name_input = st.text_input("Team name (StatsBomb team name)", value="", key="team_report_name")
        team_match_ids_input = st.text_input(
            "Match IDs (comma-separated StatsBomb match_id list)", value="", key="team_report_match_ids"
        )
        team_name = team_name_input.strip()
        # Custom mode: exactly one caller-provided name/match_ids pair,
        # same as before -- the multi-variant handling below only applies
        # to candidates resolved through candidate_index.py's own
        # TEAM_NAME_MERGES, since a manually-typed name is unambiguous.
        # No 360-based pre-filtering/cap here either -- candidate_index.py
        # has no coverage data for arbitrary caller-provided match_ids, and
        # a user typing exact match_ids in has already opted out of the
        # dropdown's guardrails deliberately.
        _variant_to_match_ids: dict[str, tuple[int, ...]] = (
            {team_name: tuple(int(m.strip()) for m in team_match_ids_input.split(",") if m.strip())}
            if team_name_input.strip() and team_match_ids_input.strip() else {}
        )
        _variant_to_match_ids_360 = _variant_to_match_ids
    else:
        _team_candidate = _team_labels[team_preset_label]
        team_name = _team_candidate["team_name"]

        # Season sub-selector, same pattern as the Player Reports tab --
        # Barcelona (24 cached seasons) gets one dropdown entry plus this
        # multiselect, not 24 flat entries.
        #
        # DEFAULT (post-timeout-incident fix): the MOST RECENT season only,
        # not "all seasons" -- a real request (Real Madrid, 19 seasons +
        # cups, 68 raw matches, only 2 with usable 360 coverage) timed out
        # at 60s because "select all" silently handed team_report.py a
        # scope no case in this project's history had been tested against.
        # Chose "most recent season only" over "no default" (the other
        # option Step 1 allowed): an empty default means clicking Generate
        # with no changes always just shows the existing "provide a
        # team/match_ids" error, for every team, even ones with only 1-2
        # cached seasons -- worse first-run UX than a small, safe, WORKING
        # default the user can deliberately widen via the multiselect
        # below. A single season is bounded by construction (this cache's
        # largest single season is ~38 raw matches, not 68+), so it can't
        # reproduce the incident's request shape even by accident.
        _team_season_labels: dict[str, dict] = {}
        for _s in _team_candidate["seasons"]:
            _n_360 = len(_s["match_ids_360"])
            _slabel = f"{_s['competition_name']} {_s['season_name']} ({_n_360} of {len(_s['match_ids'])} 360-covered)"
            _team_season_labels[_slabel] = _s

        def _season_recency_key(label: str) -> tuple[int, int]:
            season_name = _team_season_labels[label]["season_name"]
            start_year_str = season_name.split("/")[0]
            start_year = int(start_year_str) if start_year_str.isdigit() else -1
            return (start_year, len(_team_season_labels[label]["match_ids"]))

        _most_recent_season_label = max(_team_season_labels, key=_season_recency_key) if _team_season_labels else None
        _default_season_labels = [_most_recent_season_label] if _most_recent_season_label else []

        if len(_team_season_labels) > 1:
            _selected_team_season_labels = st.multiselect(
                "Season(s)", options=list(_team_season_labels.keys()), default=_default_season_labels,
                help=(
                    "Defaults to the most recent cached season only, not all of them -- selecting every "
                    "season for a team with many of them can request far more matches than have usable "
                    "360 coverage, which is slow for no benefit. Widen this deliberately if you want more."
                ),
            )
        else:
            _selected_team_season_labels = list(_team_season_labels.keys())

        # Post-audit correction (Caen/Marseille class of bug):
        # `_team_candidate` may merge MULTIPLE StatsBomb name variants of
        # the same real club (e.g. "Marseille"/"Olympique de Marseille"),
        # and -- confirmed directly during the audit -- a single season
        # can contain matches tagged under BOTH variants. Since
        # `generate_team_report(team_name, match_ids)` matches on exactly
        # one name (unchanged, unmodified logic), a selection spanning
        # multiple variants is grouped here into one call PER variant,
        # so every real match is actually captured by SOME call -- never
        # silently dropped the way picking a single name string would.
        #
        # Timeout-incident fix: ALSO computed here, per variant, using
        # ONLY information candidate_index.py already has cheaply (no new
        # expensive check) -- `_variant_to_match_ids` (every raw cached
        # match in the selection) and `_variant_to_match_ids_360`
        # (the subset ALSO known to be 360-covered). The latter, capped at
        # TEAM_REPORT_MAX_360_MATCHES_PER_REQUEST, is what's actually sent
        # to generate_team_report below -- never the raw list.
        _variant_to_match_ids_raw: dict[str, set[int]] = {}
        _variant_to_match_ids_360_raw: dict[str, set[int]] = {}
        for _lbl in _selected_team_season_labels:
            _season = _team_season_labels[_lbl]
            _season_360_set = set(_season["match_ids_360"])
            for _variant, _ids in _season["match_ids_by_variant"].items():
                _variant_to_match_ids_raw.setdefault(_variant, set()).update(_ids)
                _variant_to_match_ids_360_raw.setdefault(_variant, set()).update(set(_ids) & _season_360_set)
        _variant_to_match_ids = {v: tuple(sorted(ids)) for v, ids in _variant_to_match_ids_raw.items()}
        _variant_to_match_ids_360 = {v: tuple(sorted(ids)) for v, ids in _variant_to_match_ids_360_raw.items()}

        _total_raw_selected = sum(len(ids) for ids in _variant_to_match_ids.values())
        _total_360_selected = sum(len(ids) for ids in _variant_to_match_ids_360.values())

        # Step 2: warn BEFORE the button is clicked, not after a timeout --
        # both conditions use only the cheap data above, already fetched
        # for the labels/multiselect.
        if _total_raw_selected > 0 and _total_360_selected < _total_raw_selected:
            st.warning(
                f"This selection includes {_total_raw_selected} cached match(es) for {team_name}, but "
                f"only {_total_360_selected} have the 360 freeze-frame coverage a real team report needs "
                f"-- the other {_total_raw_selected - _total_360_selected} would contribute nothing. Only "
                "the 360-covered matches will actually be sent when you click Generate."
            )
        if _total_360_selected > TEAM_REPORT_MAX_360_MATCHES_PER_REQUEST:
            st.warning(
                f"{_total_360_selected} 360-covered matches in this selection -- capped to "
                f"{TEAM_REPORT_MAX_360_MATCHES_PER_REQUEST} per request (measured: real pitch-control/"
                "threat computation runs at roughly 1.2-2.0s per match, so a request this size risked "
                "taking 60s+ on genuine computation alone, not wasted work -- confirmed directly: a "
                "51-match well-supported request took just over 100s). Narrow the season selection above "
                "for a specific sub-range instead of relying on this cap, if you need a different subset."
            )

        def _capped(match_ids: tuple[int, ...]) -> tuple[int, ...]:
            """Most-recent-N by match_id, a simple, deterministic (if
            imperfect -- StatsBomb match_ids are not strictly globally
            chronological across competitions) recency proxy; exact
            precision doesn't matter for a safety cap the way it would for
            a real feature."""
            return tuple(sorted(sorted(match_ids, reverse=True)[:TEAM_REPORT_MAX_360_MATCHES_PER_REQUEST]))

        _variant_to_match_ids_360 = {v: _capped(ids) for v, ids in _variant_to_match_ids_360.items()}

        # Fix 1 (zero-usable-match repro: Atlético Madrid, La Liga 2020/21
        # -- 2 raw cached matches, 0 with 360 coverage): a variant with
        # ZERO 360-covered matches must not remain as a dict key mapping
        # to an empty tuple. `_variant_to_match_ids_360_raw.setdefault(
        # _variant, set())` above always creates the key even when the
        # season-360 intersection is empty, so `not _variant_to_match_ids_360`
        # below (a plain dict-truthiness check) would NOT catch this --
        # a dict with one key and an empty-tuple value is still truthy.
        # Without this filter, that empty tuple reached
        # `_cached_team_report(rest_base_url, variant, ())`, which
        # `requests` sends as NO `match_ids` param at all -- reproduced
        # directly as api.py's raw 422 before this fix (see api.py's own
        # comment on this same case).
        _variant_to_match_ids_360 = {v: ids for v, ids in _variant_to_match_ids_360.items() if ids}

    generate_team_clicked = st.button("Generate Team Report")

    if generate_team_clicked:
        if not team_name or not _variant_to_match_ids_360:
            st.error(
                "Provide a team name and at least one match_id."
                if team_preset_label == "Custom"
                else "No 360-covered matches in this selection -- widen the season selection above."
            )
        elif len(_variant_to_match_ids_360) > 1:
            st.info(
                f"This selection spans {len(_variant_to_match_ids_360)} different StatsBomb name variants "
                f"for {team_name} ({', '.join(f'{v!r} ({len(ids)} match(es))' for v, ids in _variant_to_match_ids_360.items())}). "
                "generate_team_report's own pitch-control aggregation can't be safely combined after the "
                "fact (its return contract doesn't expose the per-cell counts a correct re-average would "
                "need) without modifying that function -- so each variant is reported separately below, "
                "rather than silently reporting only one and dropping the other's real coverage."
            )
            for _variant, _ids in _variant_to_match_ids_360.items():
                st.subheader(f"Variant: {_variant!r} ({len(_ids)} match(es))")
                with st.spinner(f"Generating report for {_variant!r}..."):
                    _variant_report = _fetch_report_safely(
                        lambda _v=_variant, _i=_ids: _cached_team_report(rest_base_url, _v, _i), rest_base_url
                    )
                if _variant_report is not None:
                    _variant_png = _cached_team_png(_variant_report)
                    st.info(
                        f"Built from {_variant_report['matches_used']} matches (of {_variant_report['matches_requested']} requested)."
                    )
                    if _variant_report["matches_used"] < candidate_index_module.LOW_SAMPLE_MATCH_THRESHOLD:
                        st.warning(
                            f"LOW SAMPLE: only {_variant_report['matches_used']} 360-covered match(es) used -- "
                            "treat this variant's pitch-control/threat pattern as illustrative, not a "
                            "confident finding."
                        )
                    st.image(_variant_png, caption=f"Team Report -- {_variant}", width="stretch")
                    with st.expander(f"Raw report data ({_variant!r})"):
                        st.json(_variant_report)
        else:
            ((_single_variant, _single_match_ids),) = _variant_to_match_ids_360.items()
            with st.spinner("Generating report..."):
                team_report_dict = _fetch_report_safely(
                    lambda: _cached_team_report(rest_base_url, _single_variant, _single_match_ids), rest_base_url
                )

            if team_report_dict is not None and team_report_dict.get("no_data"):
                # Fix 1 defense-in-depth: the dict-filter fix above already
                # prevents this dashboard from ever sending an empty
                # match_ids selection in the normal preset flow, so this
                # should not fire in practice -- kept anyway as a second,
                # independent layer (e.g. a stale candidate-index cache,
                # or any future caller of api.py's endpoint that isn't
                # this dashboard) rather than trusting the client-side
                # filter alone to be the only thing standing between a
                # user and api.py's raw 422.
                st.info(
                    team_report_dict.get("reason")
                    or "No 360-covered matches available for this selection -- try a different season."
                )
            elif team_report_dict is not None:
                team_png_bytes = _cached_team_png(team_report_dict)

                # team_report.py/team_visualizer.py's existing sample-size
                # caption (matches_used/matches_requested) is baked into the
                # rendered PNG already -- surfaced HERE too as a real
                # Streamlit element, per this tab's explicit requirement,
                # not just left embedded in the image. Passed through the
                # /reports/team/{team_name} endpoint's JSON response
                # unchanged (ADR-018).
                st.info(
                    f"Built from {team_report_dict['matches_used']} matches "
                    f"(of {team_report_dict['matches_requested']} requested). "
                    "Per-frame count is not exposed by generate_team_report's current return "
                    "contract -- match-level count shown for transparency about sample size, "
                    "not a frame-level one (see team_visualizer.py's own caption)."
                )
                # Post-audit correction: this used to be dashboard.py's OWN
                # separate `matches_used < 2` check -- a third, disagreeing
                # "is this usable" threshold alongside candidate_index.py's
                # (10, cache-count-based at the time) and team_report.py's
                # own internal 360-based matches_used. Now there is exactly
                # ONE authoritative threshold constant
                # (candidate_index.LOW_SAMPLE_MATCH_THRESHOLD, still 10),
                # applied here to the REAL, live matches_used this exact
                # selection actually produced -- not a separately
                # pre-computed estimate.
                if team_report_dict["matches_used"] < candidate_index_module.LOW_SAMPLE_MATCH_THRESHOLD:
                    st.warning(
                        f"LOW SAMPLE: only {team_report_dict['matches_used']} 360-covered match(es) used "
                        f"(threshold: {candidate_index_module.LOW_SAMPLE_MATCH_THRESHOLD}) -- treat this "
                        "team's pitch-control/threat pattern as illustrative, not a confident finding."
                    )

                st.image(team_png_bytes, caption=f"Team Report -- {team_name}", width="stretch")

                with st.expander("Raw report data"):
                    st.json(team_report_dict)

                # --- Export (UX polish pass, Part B) ------------------------
                # Scope, stated explicitly: the main dashboard image plus
                # Tactical Entropy/Opposition Analysis for this SAME single
                # variant (`_single_variant`) -- both re-fetched here via
                # their own already-`st.cache_data`-wrapped functions, so
                # this is a real cache HIT (no new network cost), not a
                # second computation; simpler than threading an accumulator
                # through their own per-variant render loops further below,
                # which also cover the rarer multi-variant case this export
                # button does not (a team whose name changed across the
                # requested seasons -- disclosed, not silently dropped).
                # Weak-Spot Lifetime Analysis / Decision Quality below are
                # each their OWN separately-triggered sub-panel (own
                # button, own spinner, real extra compute cost) -- excluded
                # from this export for the same reason Player Report's own
                # Press Resistance Index/touch map/timeline are.
                _team_export_sections = [
                    f"<h2>Report</h2><img src=\"{_image_to_data_uri(team_png_bytes)}\" alt=\"Team report\">"
                ]
                # Raw (un-360-filtered) match_ids for this SAME variant --
                # Tactical Entropy/Opposition Analysis both need no 360
                # coverage, so they use the raw selection, not
                # `_single_match_ids` (which is 360-filtered/capped, for
                # the pitch-control report above only). Falls back to
                # `_single_match_ids` only if this variant is somehow
                # absent from the raw dict (not expected in practice --
                # `_variant_to_match_ids_360` is derived FROM
                # `_variant_to_match_ids`).
                _export_raw_match_ids = _variant_to_match_ids.get(_single_variant, _single_match_ids)
                _export_entropy = _cached_team_pass_entropy(rest_base_url, _single_variant, _export_raw_match_ids)
                if _export_entropy is not None and _export_entropy["conditional_entropy_bits"] is not None:
                    _team_export_sections.append(
                        "<h2>Tactical Entropy</h2>"
                        + _html_metric_row([
                            ("Conditional Entropy", f"{_export_entropy['conditional_entropy_bits']:.3f} bits"),
                            ("Normalized (0=predictable, 1=random)", f"{_export_entropy['normalized_entropy']:.3f}"),
                            ("Real Transitions Observed", str(_export_entropy["total_transitions"])),
                        ])
                    )
                _export_opposition = _cached_team_opposition_analysis(rest_base_url, _single_variant, _export_raw_match_ids)
                if _export_opposition is not None:
                    _opp_metrics: list[tuple[str, str]] = []
                    _long_pass_share = (_export_opposition.get("build_up_tendency") or {}).get("long_pass_share")
                    if _long_pass_share is not None:
                        _opp_metrics.append(("Build-up long-pass share", f"{_long_pass_share * 100:.1f}%"))
                    _set_piece_share = (_export_opposition.get("set_piece_reliance") or {}).get("set_piece_shot_share")
                    if _set_piece_share is not None:
                        _opp_metrics.append(("Set-piece shot share", f"{_set_piece_share * 100:.1f}%"))
                    if _opp_metrics:
                        _team_export_sections.append("<h2>Opposition Analysis</h2>" + _html_metric_row(_opp_metrics))

                _team_export_html = _build_standalone_html_export(
                    title=f"Team Report -- {team_name}",
                    generated_note=(
                        "Project Athena -- Team Report. Static, self-contained HTML -- opens directly "
                        "from the filesystem, no server required."
                    ),
                    sections=_team_export_sections,
                )
                st.download_button(
                    "Export Team Report (HTML)",
                    data=_team_export_html,
                    file_name=f"team_report_{team_name.replace(' ', '_')}.html",
                    mime="text/html",
                )

        # Tactical Entropy (additive new feature): rendered for EVERY
        # variant in `_variant_to_match_ids` (the RAW, un-360-filtered
        # match selection -- see `_cached_team_pass_entropy`'s own
        # comment for why), independent of the single/multi-variant
        # 360-report branching above -- this feature needs no 360
        # coverage at all, so it always runs once per variant regardless
        # of which branch the pitch-control report itself took.
        if team_name and _variant_to_match_ids:
            st.divider()
            st.subheader("Tactical Entropy (pass-direction predictability)")
            for _entropy_variant, _entropy_match_ids in _variant_to_match_ids.items():
                with st.spinner(f"Computing Tactical Entropy for {_entropy_variant!r}..."):
                    entropy_report = _fetch_report_safely(
                        lambda _v=_entropy_variant, _i=_entropy_match_ids: _cached_team_pass_entropy(
                            rest_base_url, _v, _i
                        ),
                        rest_base_url,
                    )
                if entropy_report is None:
                    continue

                st.markdown(f"**{_entropy_variant}** -- {entropy_report['matches_used']} of "
                            f"{entropy_report['matches_requested']} requested match(es) used")

                if entropy_report["pass_entropy_used_low_sample_flag"]:
                    st.warning(
                        f"LOW SAMPLE: only {entropy_report['total_transitions']} real pass-to-pass "
                        "transition(s) observed -- treat the entropy value below as illustrative, "
                        "not a confident finding."
                    )

                if entropy_report["conditional_entropy_bits"] is None:
                    st.info("No pass-to-pass transitions available for this selection.")
                else:
                    entropy_cols = st.columns(3)
                    entropy_cols[0].metric(
                        "Conditional Entropy",
                        f"{entropy_report['conditional_entropy_bits']:.3f} bits",
                        help=(
                            f"Max possible (uniform over {len(entropy_report['pass_type_categories'])} "
                            f"categories): {entropy_report['max_possible_entropy_bits']:.3f} bits."
                        ),
                    )
                    entropy_cols[1].metric(
                        "Normalized (0=predictable, 1=random)",
                        f"{entropy_report['normalized_entropy']:.3f}",
                    )
                    entropy_cols[2].metric(
                        "Real Transitions Observed", entropy_report["total_transitions"],
                    )

                    _entropy_cats = entropy_report["pass_type_categories"]
                    transition_df = pd.DataFrame(entropy_report["transition_probabilities"]).T
                    transition_df = transition_df.reindex(index=_entropy_cats, columns=_entropy_cats)
                    st.caption("Transition probability matrix: P(next pass type = column | current pass type = row)")
                    st.dataframe(transition_df.style.format("{:.2f}", na_rep="n/a"), width="stretch")

                with st.expander(f"Raw Tactical Entropy data ({_entropy_variant!r})"):
                    st.json(entropy_report)

        # Passing Lane Visualizer (additive new feature): unlike Tactical
        # Entropy, this DOES need 360 coverage -- fetched against
        # `_variant_to_match_ids_360` (the SAME 360-covered, already-
        # capped match list the pitch-control report above uses), not
        # the raw uncapped `_variant_to_match_ids`.
        if team_name and _variant_to_match_ids_360:
            st.divider()
            st.subheader("Passing Lane Visualizer (pitch-control-based lane openness)")
            for _lane_variant, _lane_match_ids in _variant_to_match_ids_360.items():
                with st.spinner(f"Computing passing lanes for {_lane_variant!r}..."):
                    lanes_report = _fetch_report_safely(
                        lambda _v=_lane_variant, _i=_lane_match_ids: _cached_team_passing_lanes(
                            rest_base_url, _v, _i
                        ),
                        rest_base_url,
                    )
                if lanes_report is None:
                    continue

                st.markdown(f"**{_lane_variant}** -- {lanes_report['matches_used']} of "
                            f"{lanes_report['matches_requested']} 360-covered match(es) used, "
                            f"{lanes_report['total_pass_samples_used']} real pass samples")

                # ADR-021 condition-2 compliance fix. DEFENSE IN DEPTH,
                # same pattern as the Shot Map / Pass Network panels
                # above -- see those panels' own comments for the full
                # reasoning; mirrored here unchanged. Only `nodes`
                # (per-player average location) is the gated field --
                # `lanes` (named pairs + scores) is present either way,
                # per this feature's own ADR-021 addendum.
                response_has_raw_lanes = "nodes" in lanes_report

                if PUBLIC_DEPLOYMENT and response_has_raw_lanes:
                    st.error(
                        "Configuration error: this dashboard process has PUBLIC_DEPLOYMENT=true, "
                        "but the API server returned raw per-player location data -- refusing to "
                        "render or display it. Set PUBLIC_DEPLOYMENT=true on the API server process "
                        "too (see README.md's 'Public deployment mode' section)."
                    )
                elif PUBLIC_DEPLOYMENT or not response_has_raw_lanes:
                    lanes_png = _cached_passing_lanes_aggregated_png(lanes_report)
                    st.image(
                        lanes_png,
                        caption=f"Passing Lane Openness (aggregated, no location) -- {_lane_variant}",
                        width="stretch",
                    )
                    with st.expander(f"Raw passing-lane data ({_lane_variant!r}, no location)"):
                        st.json(lanes_report)
                else:
                    lanes_png = _cached_passing_lanes_png(lanes_report)
                    st.image(lanes_png, caption=f"Passing Lane Openness -- {_lane_variant}", width="stretch")
                    with st.expander(f"Raw passing-lane data ({_lane_variant!r})"):
                        st.json(lanes_report)

                low_sample_lanes = [
                    lane for lane in lanes_report.get("lanes", []) if lane["passing_lane_used_low_sample_flag"]
                ]
                if low_sample_lanes:
                    st.caption(
                        f"{len(low_sample_lanes)} of {len(lanes_report.get('lanes', []))} pairs are "
                        "LOW SAMPLE (fewer than the confidence threshold's real pass samples) -- "
                        "treat those specific pairs' openness scores as illustrative, not confident."
                    )

        # Opposition Analysis (additive new feature): 3 specific
        # opposition-scouting metrics. Metrics 2/3 (build-up tendency,
        # set-piece reliance) are event-data-only, fetched against
        # `_variant_to_match_ids` like Tactical Entropy. Metric 1
        # (pitch-control weak zones) re-presents `_cached_team_report`'s
        # own `weakest_control_zones` field -- called again here
        # deliberately (not passed down from the panel above, which may
        # not have run in this exact shape for every variant in
        # multi-variant mode), but this hits `st.cache_data`'s cache for
        # any (variant, match_ids) pair already fetched above, so no
        # REAL additional backend computation happens for that part.
        if team_name and _variant_to_match_ids:
            st.divider()
            st.subheader("Opposition Analysis (scouting: how to play against this team)")
            for _opp_variant, _opp_match_ids in _variant_to_match_ids.items():
                with st.spinner(f"Computing opposition analysis for {_opp_variant!r}..."):
                    opp_report = _fetch_report_safely(
                        lambda _v=_opp_variant, _i=_opp_match_ids: _cached_team_opposition_analysis(
                            rest_base_url, _v, _i
                        ),
                        rest_base_url,
                    )
                if opp_report is None:
                    continue

                st.markdown(f"**{_opp_variant}** -- {opp_report['matches_used']} of "
                            f"{opp_report['matches_requested']} requested match(es) used")

                opp_cols = st.columns(2)
                with opp_cols[0]:
                    st.markdown("**Build-up tendency**")
                    bt = opp_report["build_up_tendency"]
                    if bt["long_pass_share"] is None:
                        st.info("No real build-up (defensive/middle-third) passes available.")
                    else:
                        st.metric(
                            f"Long pass share (>{bt['long_pass_threshold_meters']:.0f}m)",
                            f"{bt['long_pass_share'] * 100:.1f}%",
                            help=f"{bt['long_passes']} long of {bt['total_buildup_passes']} real build-up passes.",
                        )
                        if bt["build_up_tendency_used_low_sample_flag"]:
                            st.caption(
                                f"LOW SAMPLE: only {bt['total_buildup_passes']} real build-up passes."
                            )
                with opp_cols[1]:
                    st.markdown("**Set-piece reliance**")
                    sp = opp_report["set_piece_reliance"]
                    if sp["set_piece_shot_share"] is None:
                        st.info("No real shots available.")
                    else:
                        st.metric(
                            "Shots from set pieces",
                            f"{sp['set_piece_shot_share'] * 100:.1f}%",
                            help=f"{sp['set_piece_shots']} set-piece of {sp['total_shots']} real shots "
                                 f"({', '.join(sp['set_piece_play_patterns'])}).",
                        )
                        if sp["set_piece_reliance_used_low_sample_flag"]:
                            st.caption(f"LOW SAMPLE: only {sp['total_shots']} real shots.")

                st.markdown("**Pitch-control weak zones (where to attack this team)**")
                _opp_360_ids = _variant_to_match_ids_360.get(_opp_variant, ())
                if not _opp_360_ids:
                    st.info("No 360-covered matches for this selection -- weak-zone data unavailable.")
                else:
                    _opp_team_report = _fetch_report_safely(
                        lambda _v=_opp_variant, _i=_opp_360_ids: _cached_team_report(rest_base_url, _v, _i),
                        rest_base_url,
                    )
                    if _opp_team_report is not None and _opp_team_report.get("weakest_control_zones"):
                        st.caption(
                            "Lowest mean pitch-control cells for this team (col/row grid indices, "
                            "10x7 -- reused, unmodified, from the Team Report panel above)."
                        )
                        st.dataframe(pd.DataFrame(_opp_team_report["weakest_control_zones"]), width="stretch")
                    else:
                        st.info("No weak-zone data available for this selection.")

                with st.expander(f"Raw opposition analysis data ({_opp_variant!r})"):
                    st.json(opp_report)

        # ====================================================================
        # Weak-Spot Lifetime Analysis (new reporting track): extends the
        # Opposition Analysis panel's own "pitch-control weak zones" concept
        # above (a STATIC, season-aggregate view collapsing every frame into
        # one grid) with a TEMPORAL one -- how long a specific zone actually
        # stayed weak, in real match-clock time, within ONE match. See
        # generate_weak_spot_lifetime_analysis's own docstring in
        # team_report.py for the full Step 0 definitions (WEAK_CONTROL_
        # THRESHOLD, GAP_TOLERANCE_SECONDS). Co-located in this same tab
        # (not a new tab) since it's a direct temporal extension of what's
        # already here, matching this panel's own conceptual home.
        # ====================================================================
        if team_name:
            st.divider()
            st.subheader("Weak-Spot Lifetime Analysis (temporal, single match)")
            st.caption(
                "How long a specific pitch zone stayed WEAK (low defending-team pitch control) across "
                "one match's real 360-covered frame sequence, in time order -- distinct from the "
                "season-aggregate weak-zone heatmap above, which discards temporal order entirely. "
                "Needs real 360 freeze-frame coverage for this specific match_id."
            )
            weak_spot_match_id_input = st.text_input(
                "Match ID (StatsBomb match_id)", value=DEFAULT_MATCH_ID, key="weak_spot_match_id_input"
            )
            weak_spot_include_recommendations = st.checkbox(
                "Include exploitation recommendations (top 20 instances -- adds ~10s, loads 2 more models)",
                value=False,
                key="weak_spot_include_recommendations",
                help=(
                    "For each of the top 20 longest-lived weak-spot instances: which defensive action "
                    "(/coach-mode's own high_press/drop_deep/force_wide/no_change ranking) most reduces "
                    "predicted threat at that instance's own real match state, plus a Deep Ensemble "
                    "confidence signal (5 independently-trained members' real disagreement -- higher "
                    "spread means lower confidence). This is the match STATE's own best fix, not a "
                    "causal decomposition isolating that one grid cell alone -- see team_report.py's own "
                    "Step 0 comment for the full disclosed scope."
                ),
            )
            weak_spot_run_clicked = st.button("Analyze Weak-Spot Lifetimes")

            if weak_spot_run_clicked:
                try:
                    weak_spot_match_id = int(weak_spot_match_id_input.strip())
                except ValueError:
                    st.error(f"Match ID must be a whole number -- got {weak_spot_match_id_input!r}.")
                else:
                    spinner_text = (
                        "Analyzing weak-spot lifetimes and computing exploitation recommendations..."
                        if weak_spot_include_recommendations else "Analyzing weak-spot lifetimes..."
                    )
                    with st.spinner(spinner_text):
                        weak_spot_result = _fetch_report_safely(
                            lambda: _cached_weak_spot_lifetime(
                                rest_base_url, team_name, weak_spot_match_id, weak_spot_include_recommendations
                            ),
                            rest_base_url,
                        )

                    if weak_spot_result is not None:
                        if weak_spot_result.get("no_data"):
                            st.info(weak_spot_result.get("reason", "No data available for this match_id."))
                        else:
                            coverage = weak_spot_result["event_360_coverage_fraction"]
                            st.metric(
                                "360 coverage of located events",
                                f"{coverage * 100:.1f}%" if coverage is not None else "N/A",
                                help=(
                                    f"{weak_spot_result['total_360_covered_located_events']} of "
                                    f"{weak_spot_result['total_located_events']} real located events had "
                                    "a matching 360 frame. This is the START/END coverage of the event "
                                    "stream, not a guarantee any SPECIFIC zone was continuously observed "
                                    "-- see the caption below."
                                ),
                            )
                            st.caption(
                                f"Weak threshold: mean defending control <= {weak_spot_result['weak_control_threshold']}. "
                                f"Gap tolerance: {weak_spot_result['gap_tolerance_seconds']:.0f}s between consecutive "
                                f"real observations of the SAME zone (a specific zone is only observed in frames "
                                f"where the ball comes within the physics engine's own mask radius of it -- a real "
                                f"subset of every 360-covered frame, denser near typical play areas, sparser "
                                f"elsewhere). {weak_spot_result['defending_frames_used']} real defending frames used."
                            )

                            longest = weak_spot_result.get("longest_lived_weak_spot")
                            if longest is not None:
                                st.markdown(
                                    f"**Longest-lived weak spot**: zone (col={longest['zone']['col']}, "
                                    f"row={longest['zone']['row']}), period {longest['period']}, "
                                    f"{longest['start_minute']:.2f}' -> {longest['end_minute']:.2f}' "
                                    f"({longest['duration_minutes']:.2f} real minutes, "
                                    f"{longest['frame_count']} consecutive real frames)."
                                )

                            instances_df = pd.DataFrame(weak_spot_result["weak_spot_instances"])
                            if not instances_df.empty:
                                instances_df["col"] = instances_df["zone"].apply(lambda z: z["col"])
                                instances_df["row"] = instances_df["zone"].apply(lambda z: z["row"])
                                display_df = instances_df[[
                                    "col", "row", "period", "start_minute", "end_minute",
                                    "duration_minutes", "frame_count",
                                ]].rename(columns={
                                    "col": "Col", "row": "Row", "period": "Period",
                                    "start_minute": "Start (min)", "end_minute": "End (min)",
                                    "duration_minutes": "Duration (min)", "frame_count": "Frames",
                                })
                                st.dataframe(display_df.head(20), width="stretch")
                                st.caption(
                                    f"Top 20 of {len(weak_spot_result['weak_spot_instances'])} total real weak-spot "
                                    "instances found (already sorted by duration, longest first)."
                                )

                                if weak_spot_include_recommendations:
                                    recommended_rows = []
                                    for inst in weak_spot_result["weak_spot_instances"][:20]:
                                        rec = inst.get("recommendation")
                                        if rec is None:
                                            continue
                                        recommended_rows.append({
                                            "Col": inst["zone"]["col"],
                                            "Row": inst["zone"]["row"],
                                            "Duration (min)": round(inst["duration_minutes"], 2),
                                            "Recommended Action": rec["recommended_action"],
                                            "Baseline Threat": f"{rec['baseline_threat_15s'] * 100:.1f}%",
                                            "Best Delta (pp)": f"{rec['rankings'][0]['delta'] * 100:+.1f}",
                                            "Confidence (ensemble std)": round(
                                                rec["confidence"]["ensemble_std_cumulative_incidence"], 4
                                            ),
                                        })
                                    st.markdown("**Exploitation Recommendations**")
                                    if recommended_rows:
                                        st.dataframe(pd.DataFrame(recommended_rows), width="stretch")
                                        st.caption(
                                            "Recommended Action = the defensive posture that most reduces predicted "
                                            "threat at that instance's own real match state (lowest simulated threat "
                                            "among high_press/drop_deep/force_wide/no_change, /coach-mode's own "
                                            "ranking). Confidence = the Deep Ensemble's real 5-member disagreement "
                                            "(standard deviation of cumulative incidence) on the recommended "
                                            "action's own resulting state -- LOWER means the 5 independently-trained "
                                            "members agree more, i.e. HIGHER confidence in the recommendation."
                                        )
                                    else:
                                        st.info("No recommendations could be computed (no real frame found near any instance).")

                            with st.expander("Raw weak-spot lifetime data"):
                                st.json(weak_spot_result)

        # ====================================================================
        # Decision Quality (Phase 4, final item): was a player's pass under
        # pressure the RIGHT choice, given the real best available
        # alternative at that moment? ADR-021: the FIRST Phase 4 composition
        # that genuinely needed gating (named player + precise location),
        # not exemption -- mirrors the Pass Network/Shot Map panels' own
        # defense-in-depth check exactly (inspects the ACTUAL response, not
        # just this process's own PUBLIC_DEPLOYMENT flag).
        # ====================================================================
        if team_name:
            st.divider()
            st.subheader("Decision Quality (real pass choices under pressure)")
            st.caption(
                "For each real pass made under pressure: the real lane-openness of the option chosen "
                "vs. the real BEST available alternative at that same moment (any other visible "
                "teammate), against the real recorded outcome. Composes Press Resistance's own "
                "pressure/success signal with a new per-frame generalization of Passing Lane "
                "Visualizer's own lane-openness computation."
            )
            decision_quality_match_id_input = st.text_input(
                "Match ID (StatsBomb match_id)", value=DEFAULT_MATCH_ID, key="decision_quality_match_id_input"
            )
            decision_quality_run_clicked = st.button("Analyze Decision Quality")

            if decision_quality_run_clicked:
                try:
                    decision_quality_match_id = int(decision_quality_match_id_input.strip())
                except ValueError:
                    st.error(f"Match ID must be a whole number -- got {decision_quality_match_id_input!r}.")
                else:
                    with st.spinner("Analyzing decision quality..."):
                        decision_quality_result = _fetch_report_safely(
                            lambda: _cached_decision_quality(rest_base_url, team_name, decision_quality_match_id),
                            rest_base_url,
                        )

                    if decision_quality_result is not None:
                        if decision_quality_result.get("no_data"):
                            st.info(decision_quality_result.get("reason", "No data available for this match_id."))
                        else:
                            # Defense in depth, same as the Pass Network/Shot Map panels:
                            # check the ACTUAL response shape, not just this process's own flag.
                            response_has_raw_decisions = "decisions" in decision_quality_result
                            if PUBLIC_DEPLOYMENT and response_has_raw_decisions:
                                st.error(
                                    "Configuration error: this dashboard process has PUBLIC_DEPLOYMENT=true, "
                                    "but the API server returned raw per-decision data (named players, "
                                    "precise locations) -- refusing to render or display it. Set "
                                    "PUBLIC_DEPLOYMENT=true on the API server process too."
                                )
                            elif PUBLIC_DEPLOYMENT or not response_has_raw_decisions:
                                st.markdown("**Per-player decision quality (aggregated -- no location)**")
                                st.dataframe(pd.DataFrame(decision_quality_result["player_summary"]), width="stretch")
                                with st.expander("Raw decision quality data (aggregated)"):
                                    st.json(decision_quality_result)
                            else:
                                metric_cols = st.columns(3)
                                metric_cols[0].metric("Total decisions", decision_quality_result["total_decisions"])
                                good_share = decision_quality_result["good_decision_share"]
                                metric_cols[1].metric(
                                    "Good decision share",
                                    f"{good_share * 100:.1f}%" if good_share is not None else "N/A",
                                    help=(
                                        f"Chosen lane openness within "
                                        f"{decision_quality_result['good_decision_openness_tolerance']} of the "
                                        "real best available alternative."
                                    ),
                                )
                                successful_share = decision_quality_result["successful_share"]
                                metric_cols[2].metric(
                                    "Successful share",
                                    f"{successful_share * 100:.1f}%" if successful_share is not None else "N/A",
                                )

                                decisions_df = pd.DataFrame(decision_quality_result["decisions"])
                                if not decisions_df.empty:
                                    display_df = decisions_df[[
                                        "player_name", "period", "minute", "chosen_lane_openness",
                                        "best_alternative_lane_openness", "openness_gap", "successful", "good_decision",
                                    ]].rename(columns={
                                        "player_name": "Player", "period": "Period", "minute": "Minute",
                                        "chosen_lane_openness": "Chosen Openness",
                                        "best_alternative_lane_openness": "Best Alternative Openness",
                                        "openness_gap": "Gap", "successful": "Successful", "good_decision": "Good Decision",
                                    })
                                    st.dataframe(display_df, width="stretch")

                                with st.expander("Raw decision quality data"):
                                    st.json(decision_quality_result)

# ============================================================================
# TAB: Team Trends -- UI wiring over team_trend_data.py, unmodified. A
# DELIBERATE, NAMED EXCEPTION to ADR-018 (see module docstring): this tab
# still calls generate_team_trend_report directly, in-process, because
# that module's own docstring forbids wiring it into api.py's served layer
# pending resolution of its data source's licensing scope. This is the one
# reporting tab that still requires dashboard.py to run with its own
# data/raw/ and football-data.co.uk network access -- not fully separated
# from the backend the way the other three tabs now are.
#
# Team Trends serving-contradiction fix (this ADR-018 exception was always
# CONDITIONED on dashboard.py itself never being publicly deployed -- see
# team_trend_data.py's own updated docstring and ADR-018's own Update
# section): when PUBLIC_DEPLOYMENT is set, this ENTIRE TAB is disabled --
# not degraded, not shown with a caveat -- because in-process is no longer
# a meaningful distinction from "served" once the process itself is public.
# generate_team_trend_report is never called in this mode: no
# football-data.co.uk network request is made, no data/raw/ write happens,
# nothing from this data source reaches a public visitor at all.
# ============================================================================
with tab_trends:
    if PUBLIC_DEPLOYMENT:
        st.header("Team Trend Report (football-data.co.uk)")
        st.info(
            "This tab is disabled in this deployment (PUBLIC_DEPLOYMENT=true). "
            "football-data.co.uk's own stated terms ('for the purposes of league match "
            "prediction only', notes.txt) are scoped to personal, non-distributed research "
            "use only -- a real, unresolved licensing ambiguity handled conservatively, the "
            "same way ADR-014 handles the AGPL-derived pitch-keypoint CV model, and the same "
            "way team_trend_data.py's own docstring already states this feature must never be "
            "served/distributed. Run this dashboard locally with PUBLIC_DEPLOYMENT unset (or "
            "'false') to use this tab. See REPORTING_FINDINGS.md §8 and ADR-018 for the full "
            "compliance note."
        )
    else:
        st.header("Team Trend Report (football-data.co.uk)")

        st.caption(
            "Data source: football-data.co.uk. Per its stated terms ('for the purposes of "
            "league match prediction only', notes.txt), this feature is scoped to personal, "
            "non-distributed research use only -- a real, unresolved licensing ambiguity "
            "handled conservatively, the same way ADR-014 handles the AGPL-derived "
            "pitch-keypoint CV model. See REPORTING_FINDINGS.md §8 for the full compliance note."
        )

        trend_team_name = st.text_input("Team name (football-data.co.uk spelling, e.g. 'Man City')", value="Man City")
        trend_col1, trend_col2 = st.columns(2)
        with trend_col1:
            trend_start_season = st.number_input("Start season (start year, e.g. 2019 for 2019/20)", min_value=1990, max_value=2100, value=2019, step=1)
        with trend_col2:
            trend_end_season = st.number_input("End season (start year, e.g. 2025 for 2025/26)", min_value=1990, max_value=2100, value=2025, step=1)

        generate_trend_clicked = st.button("Generate Trend Report")

        if generate_trend_clicked:
            if trend_start_season > trend_end_season:
                st.error("Start season must be <= end season.")
            else:
                # UX polish pass (Part C): this call is direct/in-process,
                # not an HTTP call to this project's own backend
                # (team_trend_data.py's own module docstring: it must never
                # be served over api.py) -- but it DOES make its own real
                # network call to football-data.co.uk internally, and that
                # call previously had ZERO exception handling anywhere on
                # this path (a real gap the Step 0 audit found: this was
                # the one tab in the whole dashboard where a genuine
                # network failure -- no internet, football-data.co.uk down
                # -- would crash the script with an unhandled traceback
                # instead of a clean message). Routed through the SAME
                # shared helper every other tab already uses, via
                # `context_label` since there is no `rest_base_url` here.
                with st.spinner("Fetching and aggregating season data..."):
                    trend_report = _fetch_report_safely(
                        lambda: _cached_team_trend_report(
                            trend_team_name.strip(), int(trend_start_season), int(trend_end_season)
                        ),
                        context_label="football-data.co.uk",
                    )
                if trend_report is None:
                    pass  # _fetch_report_safely already rendered the error
                else:
                    st.write(
                        f"Seasons found: {trend_report['seasons_found']} / {trend_report['seasons_requested']} requested."
                    )

                    # gap_seasons: reused directly, shown honestly -- never silently
                    # omitted just because this is now a UI instead of a printed dict.
                    if trend_report["gap_seasons"]:
                        st.warning(
                            "Gap seasons (team not found in any of the five covered top-flight leagues -- "
                            "relegated, not yet promoted, or otherwise absent that year): "
                            + ", ".join(trend_report["gap_seasons"])
                        )

                    season_stats = trend_report["season_stats"]
                    if season_stats:
                        trend_df = pd.DataFrame.from_dict(season_stats, orient="index")
                        trend_df.index.name = "season"

                        st.subheader("Year-by-year trend")
                        chart_metrics = [m for m in ["points", "goals_scored", "goals_conceded", "win_rate"] if m in trend_df.columns]
                        if chart_metrics:
                            st.line_chart(trend_df[chart_metrics])

                        st.subheader("Raw per-season data")
                        st.dataframe(trend_df)

                        if trend_report["year_over_year_deltas"]:
                            st.subheader("Year-over-year deltas")
                            deltas_df = pd.DataFrame(trend_report["year_over_year_deltas"])
                            st.dataframe(deltas_df)
                            non_consecutive = deltas_df[~deltas_df["consecutive"]]
                            if not non_consecutive.empty:
                                st.info(
                                    "Rows marked consecutive=False span a gap season -- not an "
                                    "adjacent-year comparison, shown as such rather than implied to be one."
                                )
                    else:
                        # UX polish pass (Part C): was `st.error` -- an
                        # empty RESULT (this team genuinely has no seasons
                        # in the requested range) is not a SYSTEM failure;
                        # every other tab's analogous "found nothing" case
                        # (Pass Network, Match Report, Alerts History) uses
                        # `st.info`, not the alarming red error box. Same
                        # visual language now, not just the same wording.
                        st.info(f"No seasons found for {trend_team_name!r} in the requested range across any covered league.")

                    with st.expander("Raw report data"):
                        st.json(trend_report)

        # --- Compare Two Seasons (Feature 3, additive) -----------------
        # A SEPARATE section, alongside (not replacing) the year-by-year
        # trend view above -- same team-vs-itself pattern as
        # team_comparison.py's two-DIFFERENT-teams comparison, but for
        # this data source's own plain results/output stats (no
        # pitch-control/event-location data exists here to compare).
        # Governed by the EXACT SAME PUBLIC_DEPLOYMENT gate as the rest of
        # this tab (this whole block is inside the `else:` branch above --
        # zero separate/weaker condition for this section specifically).
        st.divider()
        st.subheader("Compare Two Seasons")
        st.caption(
            "Compares one team against its own past/future self across two specific seasons -- "
            "goals, points, shots, cards, etc. Every delta below is season_b MINUS season_a: a "
            "NEGATIVE value is a real decrease (e.g. fewer goals scored, a lower points total), "
            "not an error or missing data. A positive value is a real increase."
        )

        compare_team_name = st.text_input(
            "Team name (football-data.co.uk spelling)", value="Man City", key="trend_compare_team_name"
        )
        compare_col1, compare_col2 = st.columns(2)
        with compare_col1:
            compare_season_a = st.number_input(
                "Season A (start year)", min_value=1990, max_value=2100, value=2019, step=1, key="trend_compare_season_a"
            )
        with compare_col2:
            compare_season_b = st.number_input(
                "Season B (start year)", min_value=1990, max_value=2100, value=2025, step=1, key="trend_compare_season_b"
            )

        compare_clicked = st.button("Compare Seasons")

        if compare_clicked:
            # UX polish pass (Part C): same fix as "Generate Trend Report"
            # above -- this call previously had no exception handling at
            # all around its own real football-data.co.uk network call.
            with st.spinner("Fetching and comparing both seasons..."):
                comparison = _fetch_report_safely(
                    lambda: _cached_team_trend_comparison(
                        compare_team_name.strip(), int(compare_season_a), int(compare_season_b)
                    ),
                    context_label="football-data.co.uk",
                )

            if comparison is None:
                pass  # _fetch_report_safely already rendered the error
            elif not comparison["season_a_found"] or not comparison["season_b_found"]:
                st.warning(comparison["summary"])
            else:
                st.write(comparison["summary"])
                comparison_png = _cached_team_trend_comparison_png(comparison)
                st.image(
                    comparison_png,
                    caption=f"{compare_team_name} -- {comparison['season_a']} vs {comparison['season_b']}",
                    width="stretch",
                )

            if comparison is not None:
                with st.expander("Raw comparison data"):
                    st.json(comparison)

# ============================================================================
# TAB: Team Comparison -- report DATA now fetched over HTTP from api.py's
# /reports/team-comparison (ADR-018). team_comparison.py itself is not
# modified.
# ============================================================================
with tab_compare:
    st.header("Team-Season Style Comparison")

    # Session/Match Comparison (additive extension, same tool at a finer
    # granularity -- one team, two SPECIFIC matches, not two seasons; see
    # ADR-021's own addendum). A mode toggle, not a new tab, per the
    # roadmap's own "extends" framing: reuses this tab's existing
    # rendering structure below (analysis mode header, richness columns,
    # reliability caveat, summary, zone table/threat diff, raw JSON
    # expander) almost verbatim -- the two granularities' `data_richness`
    # sub-dicts genuinely differ in UNITS (match count vs. real located-
    # event count within one match, per Step 1), so the richness columns
    # branch on `comparison_granularity` below rather than assuming
    # identical field names would be honest to reuse unchanged.
    comparison_granularity = st.radio("Comparison Granularity", ["Season-level", "Match-level"], index=0)

    if comparison_granularity == "Season-level":
        compare_col_a, compare_col_b = st.columns(2)
        with compare_col_a:
            st.subheader("Team A")
            compare_team_a = st.text_input("Team A name (StatsBomb team name)", value="Barcelona")
            compare_season_a = st.number_input("Team A season (start year)", min_value=1990, max_value=2100, value=2008, step=1)
        with compare_col_b:
            st.subheader("Team B")
            compare_team_b = st.text_input("Team B name (StatsBomb team name)", value="Barcelona")
            compare_season_b = st.number_input("Team B season (start year)", min_value=1990, max_value=2100, value=2015, step=1)
    else:
        compare_match_team = st.text_input("Team name (StatsBomb team name)", value="Barcelona", key="compare_match_team")
        compare_match_col_a, compare_match_col_b = st.columns(2)
        with compare_match_col_a:
            compare_match_id_a = st.text_input("Match A (StatsBomb match_id)", value=DEFAULT_MATCH_ID, key="compare_match_id_a")
        with compare_match_col_b:
            compare_match_id_b = st.text_input("Match B (StatsBomb match_id)", value=DEFAULT_MATCH_ID, key="compare_match_id_b")

    generate_comparison_clicked = st.button("Compare")

    if generate_comparison_clicked and comparison_granularity == "Match-level":
        try:
            _match_id_a_int = int(compare_match_id_a.strip())
            _match_id_b_int = int(compare_match_id_b.strip())
        except ValueError:
            st.error("Match A/B must both be whole numbers.")
            comparison = None
        else:
            with st.spinner("Fetching match data and computing comparison..."):
                comparison = _fetch_report_safely(
                    lambda: _cached_team_match_comparison(
                        rest_base_url, compare_match_team.strip(), _match_id_a_int, _match_id_b_int
                    ),
                    rest_base_url,
                )
    elif generate_comparison_clicked:
        with st.spinner("Fetching match data and computing comparison..."):
            comparison = _fetch_report_safely(
                lambda: _cached_team_comparison(
                    rest_base_url,
                    compare_team_a.strip(), int(compare_season_a),
                    compare_team_b.strip(), int(compare_season_b),
                ),
                rest_base_url,
            )
    else:
        comparison = None

    if generate_comparison_clicked:
        if comparison is not None:
            st.subheader(f"Analysis mode: `{comparison['analysis_mode']}`")
            st.caption(comparison["mode_reason"])

            richness_col_a, richness_col_b = st.columns(2)
            if comparison_granularity == "Season-level":
                with richness_col_a:
                    ra = comparison["data_richness"]["team_a"]
                    st.metric(f"{ra['team']} {ra['season']} matches", ra["matches"])
                    st.caption(ra["flag"])
                with richness_col_b:
                    rb = comparison["data_richness"]["team_b"]
                    st.metric(f"{rb['team']} {rb['season']} matches", rb["matches"])
                    st.caption(rb["flag"])
            else:
                with richness_col_a:
                    ra = comparison["data_richness"]["team_a"]
                    st.metric(f"{ra['team']} (match {ra['match_id']}) located events", ra["located_events"])
                    st.caption(ra["flag"])
                with richness_col_b:
                    rb = comparison["data_richness"]["team_b"]
                    st.metric(f"{rb['team']} (match {rb['match_id']}) located events", rb["located_events"])
                    st.caption(rb["flag"])

            # THE critical requirement for this tab: a low-sample side's
            # reliability caveat must be a prominent, impossible-to-miss
            # element -- never a quiet footnote a user could scroll past.
            # Passed through the /reports/team-comparison endpoint's JSON
            # response unchanged (ADR-018).
            if comparison["reliability_caveat"]:
                st.error(comparison["reliability_caveat"])

            st.subheader("Summary")
            st.write(comparison["summary"])

            if comparison["analysis_mode"] == "event_location_activity_map":
                st.subheader("Zone shares (share of located events)")
                zone_df = pd.DataFrame(comparison["zone_shares"])
                st.dataframe(zone_df)
                st.subheader("Zone diff (A - B)")
                st.json(comparison["zone_diff_a_minus_b"])
            else:
                st.subheader("Threat-by-pitch-zone diff (A - B)")
                st.json(comparison["threat_by_pitch_zone_diff_a_minus_b"])

            with st.expander("Raw comparison data"):
                st.json(comparison)


# ============================================================================
# TAB: Pass Network -- report DATA fetched over HTTP from api.py's new
# /reports/pass-network/{match_id} (same ADR-018 "call api.py, don't
# reimplement" principle as the other reporting tabs); PNG rendering calls
# pass_network_visualizer.py directly (pure client-side rendering, no
# MLflow/StatsBomb access of its own -- same split as every other
# reporting tab's PNG step).
#
# ADR-021 condition-2 compliance: SAME gating pattern as the Shot Map panel
# in the Player Reports tab above, applied here rather than reinvented --
# see that panel's own comment for the full defense-in-depth reasoning
# (this dashboard process's OWN PUBLIC_DEPLOYMENT flag, PLUS an inspection
# of whether the actual API response still carries a raw `nodes`/`edges`
# field, failing closed on any mismatch between the two signals).
# ============================================================================
with tab_pass_network:
    st.header("Pass Network")
    st.caption(
        "Single-match pass network: each Starting XI player's own average location, connected "
        "by real completed-pass counts to their teammates. Built from real StatsBomb event data "
        "via the new GET /reports/pass-network/{match_id} endpoint."
    )

    pass_network_match_id_input = st.text_input(
        "Match ID (StatsBomb match_id)", value=DEFAULT_MATCH_ID, key="pass_network_match_id_input"
    )
    pass_network_generate_clicked = st.button("Generate Pass Network")

    if pass_network_generate_clicked:
        try:
            pass_network_match_id = int(pass_network_match_id_input.strip())
        except ValueError:
            st.error(f"Match ID must be a whole number -- got {pass_network_match_id_input!r}.")
            st.session_state.pop("_pass_network_result", None)
        else:
            with st.spinner("Generating pass network..."):
                pass_network = _fetch_report_safely(
                    lambda: _cached_pass_network(rest_base_url, pass_network_match_id), rest_base_url
                )
            # UX polish pass (Part A): stored in session_state and RENDERED
            # BELOW, outside this transient `if ..._clicked:` gate -- see
            # `_render_cross_link_button`'s own module comment for why a
            # cross-link button cannot safely live inside this block
            # directly (its own `st.rerun()` would silently orphan itself,
            # a real bug found by actually testing this, not assumed).
            st.session_state["_pass_network_result"] = {"match_id": pass_network_match_id, "data": pass_network}

    _pn_result = st.session_state.get("_pass_network_result")
    if _pn_result is not None:
        pass_network_match_id = _pn_result["match_id"]
        pass_network = _pn_result["data"]
        if pass_network is not None:
            if pass_network.get("no_data"):
                st.info(pass_network.get("reason", "No pass network data available for this match_id."))
            else:
                # ADR-021 condition-2 compliance fix. DEFENSE IN DEPTH,
                # deliberately checking BOTH signals rather than trusting
                # either alone -- see the Shot Map panel's own comment
                # (Player Reports tab) for the full reasoning; mirrored
                # here unchanged:
                #   1. response_has_raw_network -- the ACTUAL shape of
                #      what api.py returned (ground truth: api.py's own
                #      PUBLIC_DEPLOYMENT flag already decided server-side
                #      whether generate_pass_network's raw nodes/edges
                #      were ever computed at all).
                #   2. this process's OWN PUBLIC_DEPLOYMENT flag.
                response_has_raw_network = "nodes" in pass_network

                if PUBLIC_DEPLOYMENT and response_has_raw_network:
                    st.error(
                        "Configuration error: this dashboard process has PUBLIC_DEPLOYMENT=true, "
                        "but the API server returned raw per-player location/edge data -- refusing "
                        "to render or display it. Set PUBLIC_DEPLOYMENT=true on the API server "
                        "process too (see README.md's 'Public deployment mode' section)."
                    )
                elif PUBLIC_DEPLOYMENT or not response_has_raw_network:
                    pass_network_png = _cached_pass_network_aggregated_png(pass_network)
                    st.image(
                        pass_network_png,
                        caption=f"Pass Network (aggregated) -- match_id={pass_network_match_id}",
                        width="stretch",
                    )
                    with st.expander("Raw pass network data (per-player totals only -- no location/edges)"):
                        st.json(pass_network)
                else:
                    pass_network_png = _cached_pass_network_png(pass_network)
                    st.image(
                        pass_network_png,
                        caption=f"Pass Network -- match_id={pass_network_match_id}",
                        width="stretch",
                    )
                    with st.expander("Raw pass network data"):
                        st.json(pass_network)

                    # --- Cross-linking (UX polish pass, Part A) -------------
                    # RAW variant only, deliberately -- the aggregated
                    # variant's own `player_summary` carries no player
                    # NAME (see generate_pass_network_aggregated's own
                    # docstring), so there is no entity here to link to
                    # in that mode. Real navigation problem this closes
                    # (Step 0): a player's name appears in this tab only
                    # baked into the PNG image itself -- there was
                    # previously no way to jump from a name seen here to
                    # that same player's own Player Report.
                    st.markdown("**Jump to a player's full report:**")
                    _pn_nodes = sorted(pass_network["nodes"], key=lambda n: (n["team"], n["name"]))
                    _pn_cols = st.columns(3)
                    for _pn_i, _pn_node in enumerate(_pn_nodes):
                        with _pn_cols[_pn_i % 3]:
                            _render_cross_link_button(
                                f"{_pn_node['name']} ({_pn_node['team']}) →",
                                target_tab="Player Reports",
                                prefills={
                                    "player_report_preset_selectbox": "Custom",
                                    "player_report_custom_player_id": str(_pn_node["player_id"]),
                                    "player_report_custom_match_ids": str(pass_network_match_id),
                                },
                            )


# ============================================================================
# TAB: Alerts History (ADR-019's persistence store, surfaced here for the
# first time) -- a pure UI wiring layer over GET /alerts/history, the same
# "call api.py over HTTP, don't reimplement" principle ADR-018 established
# for the four reporting tabs above. Read-only: this tab never logs an
# alert itself, only fetches and displays what the Live CV Monitor's two
# panels (or any other client) already logged via `log_alert`.
#
# Placed LAST, after Team Comparison, not folded into the "Live CV Monitor"
# tab: that tab already carries the module docstring's documented
# "PERMANENT CONSEQUENCE" (a running stream/simulation blocks the entire
# script, every other tab included) -- adding a third, unrelated panel
# there would only add to that same blocking surface for no benefit, since
# this feature has no interaction with the live stream/simulator beyond
# reading what they already persisted. A new tab keeps this read-only
# browsing feature independent of that blocking behavior entirely, the
# same way the four reporting tabs already are.
# ============================================================================
with tab_alerts:
    st.header("Alerts History")
    st.caption(
        "Browse previously-logged tactical alerts (ADR-019). Both the What-If Simulator and "
        "the Live Tactical Threat Monitor (Live CV Monitor tab) log a companion history entry "
        "for every alert they raise -- this tab only reads that history; it never logs "
        "anything itself."
    )

    alerts_match_id_input = st.text_input(
        "Match ID (optional)",
        value="",
        help=(
            "Free-text StatsBomb match_id, exact match only. Deliberately free text, not a "
            "dropdown built from candidate_index.py's enumeration used elsewhere in this "
            "dashboard: that scan reads data/raw/'s cached event files, a DIFFERENT population "
            "from what's actually in the alerts history table below (a match_id could have "
            "cached event data but zero alerts, or vice versa), and is a slow, ~15-20s one-time "
            "scan besides -- a mismatched, non-cheap fit for this specific filter."
        ),
        key="alerts_match_id_input",
    )
    alerts_source_input = st.selectbox("Source", ["All", "statsbomb", "cv"], key="alerts_source_input")

    alerts_filter_by_date = st.checkbox("Filter by date range", value=False, key="alerts_filter_by_date")
    alerts_date_range = None
    if alerts_filter_by_date:
        alerts_date_range = st.date_input(
            "Date range (UTC, both ends inclusive)",
            value=(date.today() - timedelta(days=30), date.today()),
            key="alerts_date_range",
        )

    alerts_fetch_clicked = st.button("Fetch Alert History")

    if alerts_fetch_clicked:
        alerts_match_id: int | None = None
        alerts_match_id_error = False
        if alerts_match_id_input.strip():
            try:
                alerts_match_id = int(alerts_match_id_input.strip())
            except ValueError:
                st.error(f"Match ID must be a whole number -- got {alerts_match_id_input!r}.")
                alerts_match_id_error = True

        if not alerts_match_id_error:
            alerts_source = None if alerts_source_input == "All" else alerts_source_input

            start_utc = end_utc = None
            if alerts_filter_by_date and alerts_date_range and len(alerts_date_range) == 2:
                start_date, end_date = alerts_date_range
                # logged_at_utc is stored as datetime.now(UTC).isoformat() (alert_store.py) --
                # always this exact "+00:00"-suffixed ISO-8601 form, which sorts correctly as
                # plain text (fetch_alerts compares these as strings, not parsed dates). Matching
                # that same form here, at each day's real start/end instant, is what makes the
                # ">= start_utc" / "<= end_utc" string comparisons on the API side correct.
                start_utc = f"{start_date.isoformat()}T00:00:00+00:00"
                end_utc = f"{end_date.isoformat()}T23:59:59.999999+00:00"

            with st.spinner("Fetching alert history..."):
                alerts = _fetch_report_safely(
                    lambda: _cached_alerts_history(
                        rest_base_url, alerts_match_id, alerts_source, start_utc, end_utc,
                        ALERTS_HISTORY_DEFAULT_LIMIT,
                    ),
                    rest_base_url,
                )

            if alerts is not None:
                if len(alerts) == 0:
                    st.info("No alerts found for the given filters.")
                else:
                    st.write(f"{len(alerts)} alert(s) found (most recent first).")
                    display_df = pd.DataFrame(alerts)[[
                        "logged_at_utc", "source", "match_id", "video_path", "minute",
                        "threat_before", "threat_after", "delta",
                        "explanation_text", "explanation_source",
                    ]].rename(columns={
                        "logged_at_utc": "Timestamp (UTC)",
                        "source": "Source",
                        "match_id": "Match ID",
                        "video_path": "Video Path",
                        "minute": "Minute",
                        "threat_before": "Threat Before",
                        "threat_after": "Threat After",
                        "delta": "Delta",
                        "explanation_text": "Explanation",
                        "explanation_source": "Explanation Source",
                    })
                    st.dataframe(display_df, width="stretch")

                    with st.expander("Raw alert data"):
                        st.json(alerts)


# ============================================================================
# TAB: Match Report (new reporting track, Part A -- Automatic Match Report).
# A pure UI wiring layer over the new GET /reports/match/{match_id}
# endpoint, the SAME "call api.py over HTTP, don't reimplement" principle
# ADR-018 established for the other reporting tabs. See match_report.py's
# module docstring for the Step 0 dedup check: this compiles data already
# available across the Team Reports / Pass Network / Alerts History tabs
# into ONE document plus a grounded narrative, rather than duplicating any
# of those tabs' own computation.
# ============================================================================
with tab_match_report:
    st.header("Automatic Match Report")
    st.caption(
        "Compiles both teams' Team Reports, both teams' Opposition Analysis, this match's Pass "
        "Network, and this match's own Alerts History into one document, plus a short narrative "
        "grounded strictly in that real, already-computed data (GET /reports/match/{match_id})."
    )

    match_report_id_input = st.text_input(
        "Match ID (StatsBomb match_id)", value=DEFAULT_MATCH_ID, key="match_report_id_input"
    )
    match_report_generate_clicked = st.button("Generate Match Report")

    if match_report_generate_clicked:
        try:
            match_report_match_id = int(match_report_id_input.strip())
        except ValueError:
            st.error(f"Match ID must be a whole number -- got {match_report_id_input!r}.")
            st.session_state.pop("_match_report_result", None)
        else:
            with st.spinner("Compiling match report (this aggregates two full Team Reports -- can take a while)..."):
                match_report = _fetch_report_safely(
                    lambda: _cached_match_report(rest_base_url, match_report_match_id), rest_base_url
                )
            # UX polish pass (Part A/B): stored in session_state and
            # rendered below, outside this transient `if ..._clicked:` gate
            # -- both the cross-link buttons AND the Export download button
            # need to keep working across reruns they themselves trigger
            # (or any other rerun elsewhere in the app); see
            # `_render_cross_link_button`'s own module comment for the full
            # "orphaned click" bug this fixes, found by actually testing it.
            st.session_state["_match_report_result"] = {
                "match_id": match_report_match_id, "data": match_report,
            }

    _mr_result = st.session_state.get("_match_report_result")
    if _mr_result is not None:
        match_report_match_id = _mr_result["match_id"]
        match_report = _mr_result["data"]
        if match_report is not None:
            if match_report.get("no_data"):
                st.info(match_report.get("reason", "No match report data available for this match_id."))
            else:
                st.subheader(f"{' vs '.join(match_report['teams'])} (match_id={match_report_match_id})")

                narrative_source = match_report.get("narrative_source", "unknown")
                st.markdown(f"**Narrative** _(source: {narrative_source})_")
                st.write(match_report.get("narrative", ""))

                team_cols = st.columns(len(match_report["teams"]))
                for col, team in zip(team_cols, match_report["teams"]):
                    with col:
                        st.markdown(f"**{team}**")
                        team_report = match_report["team_reports"].get(team, {})
                        if team_report.get("matches_used", 0) > 0:
                            threat_by_zone = team_report.get("threat_by_pitch_zone") or {}
                            for zone, value in threat_by_zone.items():
                                if value is not None:
                                    st.metric(f"Threat ({zone})", f"{value * 100:.1f}%")
                        else:
                            st.caption("No 360 coverage for this match -- no zone/control data.")

                        opposition = match_report["opposition_analysis"].get(team, {})
                        long_pass_share = (opposition.get("build_up_tendency") or {}).get("long_pass_share")
                        if long_pass_share is not None:
                            st.metric("Build-up long-pass share", f"{long_pass_share * 100:.1f}%")
                        set_piece_share = (opposition.get("set_piece_reliance") or {}).get("set_piece_shot_share")
                        if set_piece_share is not None:
                            st.metric("Set-piece shot share", f"{set_piece_share * 100:.1f}%")

                        # Cross-linking (UX polish pass, Part A). Real
                        # navigation problem this closes (Step 0): a
                        # team name appears in this compiled report with
                        # no path to that team's own full Team Report.
                        _render_cross_link_button(
                            f"View full {team} Team Report →",
                            target_tab="Team Reports",
                            prefills={
                                "team_report_preset_selectbox": "Custom",
                                "team_report_name": team,
                                "team_report_match_ids": str(match_report_match_id),
                            },
                        )

                st.metric("Tactical alerts this match", match_report.get("alert_count", 0))

                with st.expander("Raw compiled match report data"):
                    st.json(match_report)

                # --- Export (UX polish pass, Part B) --------------------
                # Mirrors exactly what's rendered above -- narrative,
                # per-team metrics, alert count -- not the raw
                # `pass_network`/`opposition_analysis` sub-dicts this
                # tab itself never visualizes beyond the raw-JSON
                # expander (see this section's own module-level comment
                # for the full Step 0 scope).
                _match_report_sections = [
                    f"<h2>Narrative <em>(source: {html.escape(narrative_source)})</em></h2>"
                    f"<p>{html.escape(match_report.get('narrative', ''))}</p>",
                ]
                for _team in match_report["teams"]:
                    _team_report = match_report["team_reports"].get(_team, {})
                    _opposition = match_report["opposition_analysis"].get(_team, {})
                    _team_metrics: list[tuple[str, str]] = []
                    if _team_report.get("matches_used", 0) > 0:
                        for _zone, _value in (_team_report.get("threat_by_pitch_zone") or {}).items():
                            if _value is not None:
                                _team_metrics.append((f"Threat ({_zone})", f"{_value * 100:.1f}%"))
                    _long_pass_share = (_opposition.get("build_up_tendency") or {}).get("long_pass_share")
                    if _long_pass_share is not None:
                        _team_metrics.append(("Build-up long-pass share", f"{_long_pass_share * 100:.1f}%"))
                    _set_piece_share = (_opposition.get("set_piece_reliance") or {}).get("set_piece_shot_share")
                    if _set_piece_share is not None:
                        _team_metrics.append(("Set-piece shot share", f"{_set_piece_share * 100:.1f}%"))
                    _match_report_sections.append(
                        f"<h2>{html.escape(_team)}</h2>" + _html_metric_row(_team_metrics)
                    )
                _match_report_sections.append(
                    _html_metric_row([("Tactical alerts this match", str(match_report.get("alert_count", 0)))])
                )

                _match_report_html = _build_standalone_html_export(
                    title=f"Match Report -- {' vs '.join(match_report['teams'])} (match_id={match_report_match_id})",
                    generated_note=(
                        "Project Athena -- Automatic Match Report. Static, self-contained HTML -- "
                        "opens directly from the filesystem, no server required."
                    ),
                    sections=_match_report_sections,
                )
                st.download_button(
                    "Export Match Report (HTML)",
                    data=_match_report_html,
                    file_name=f"match_report_{match_report_match_id}.html",
                    mime="text/html",
                )

    # ========================================================================
    # Tactical Event Detection (new reporting track): OWN dedicated button/
    # fetch, not folded into "Generate Match Report" above -- mirrors the
    # API side's own separate endpoint (own real compute cost, not bundled
    # into every Automatic Match Report call by default). Reuses this tab's
    # own match_report_id_input field so the match_id is only entered once.
    # ========================================================================
    st.divider()
    st.subheader("Tactical Event Detection")
    st.caption(
        "Detects Counter Attack, Switch of Play, and Build-up Pattern instances from this match's "
        "real possession-chain/event structure (event data only -- no 360 dependency, unlike "
        "Weak-Spot Lifetime Analysis in the Team Reports tab). Uses the Match ID above "
        "(GET /reports/match/{match_id}/tactical-events)."
    )
    tactical_events_clicked = st.button("Detect Tactical Events")

    if tactical_events_clicked:
        try:
            tactical_events_match_id = int(match_report_id_input.strip())
        except ValueError:
            st.error(f"Match ID must be a whole number -- got {match_report_id_input!r}.")
        else:
            with st.spinner("Detecting tactical events..."):
                tactical_events = _fetch_report_safely(
                    lambda: _cached_tactical_events(rest_base_url, tactical_events_match_id), rest_base_url
                )

            if tactical_events is not None:
                if tactical_events.get("no_data"):
                    st.info(tactical_events.get("reason", "No data available for this match_id."))
                else:
                    event_cols = st.columns(3)
                    event_cols[0].metric(
                        "Counter Attacks",
                        len(tactical_events["counter_attacks"]),
                        help=(
                            f"{tactical_events['counter_attack_fraction_of_chains'] * 100:.1f}% of "
                            f"{tactical_events['total_chains']} real possession chains. Threshold: reaches "
                            f"the attacking third within "
                            f"{tactical_events['counter_attack_time_to_final_third_seconds']:.0f}s of an "
                            "opponent turnover."
                        ),
                    )
                    event_cols[1].metric(
                        "Build-up Patterns",
                        len(tactical_events["build_up_patterns"]),
                        help=(
                            f"{tactical_events['build_up_fraction_of_chains'] * 100:.1f}% of "
                            f"{tactical_events['total_chains']} real possession chains. Threshold: "
                            f"starts in defensive/middle third, NOT fast, >= "
                            f"{tactical_events['buildup_min_passes']} real passes."
                        ),
                    )
                    event_cols[2].metric(
                        "Switches of Play",
                        len(tactical_events["switches_of_play"]),
                        help=(
                            f"{tactical_events['switch_of_play_fraction_of_completed_passes'] * 100:.1f}% of "
                            f"{tactical_events['total_completed_passes']} real completed passes. Threshold: "
                            f">= {tactical_events['switch_of_play_lateral_threshold_meters']:.0f}m real "
                            "lateral distance."
                        ),
                    )

                    detail_tabs = st.tabs(["Counter Attacks", "Build-up Patterns", "Switches of Play"])
                    with detail_tabs[0]:
                        if tactical_events["counter_attacks"]:
                            st.dataframe(pd.DataFrame(tactical_events["counter_attacks"]), width="stretch")
                        else:
                            st.info("No Counter Attacks detected.")
                    with detail_tabs[1]:
                        if tactical_events["build_up_patterns"]:
                            st.dataframe(pd.DataFrame(tactical_events["build_up_patterns"]), width="stretch")
                        else:
                            st.info("No Build-up Patterns detected.")
                    with detail_tabs[2]:
                        if tactical_events["switches_of_play"]:
                            st.dataframe(pd.DataFrame(tactical_events["switches_of_play"]), width="stretch")
                        else:
                            st.info("No Switches of Play detected.")

                    with st.expander("Raw tactical events data"):
                        st.json(tactical_events)

    # ========================================================================
    # Tactical Timeline UI (new reporting track, capstone): unifies Weak-Spot
    # Lifetime Analysis and Tactical Event Detection onto one shared time
    # axis. Placed here, alongside the compiled Match Report and Tactical
    # Event Detection panels (not a new tab) -- the roadmap's own explicit
    # framing for this feature, since it's a visualization OVER those two
    # match-level signals, not a separate concern. Tactical Momentum/Match
    # Segmentation are DELIBERATELY NOT part of this timeline -- see
    # match_timeline.py's own Step 0 docstring for the full, explicit scope
    # reasoning (live-stream-only concepts, no batch/post-hoc equivalent
    # exists today) -- stated again here, visibly, so a dashboard user sees
    # the same honest scope note the API response itself carries.
    # ========================================================================
    st.divider()
    st.subheader("Tactical Timeline")
    st.caption(
        "A single, unified chronological view of this match: Weak-Spot Lifetime instances "
        "(Team Reports tab) and Tactical Events (Counter Attack / Build-up Pattern / Switch of "
        "Play, above) plotted on one continuous time axis, correctly handling the real "
        "half-time minute-numbering boundary. Uses the Match ID above "
        "(GET /reports/match/{match_id}/timeline)."
    )
    st.info(
        "Tactical Momentum and Match Segmentation are NOT included here -- both are live-stream-"
        "only concepts (ephemeral, client-side, computed in the Live CV Monitor tab) with no "
        "persisted or batch match-level equivalent today. This timeline unifies the two signals "
        "that already exist as real, match-clock-indexed batch data."
    )
    timeline_clicked = st.button("Build Tactical Timeline")

    if timeline_clicked:
        try:
            timeline_match_id = int(match_report_id_input.strip())
        except ValueError:
            st.error(f"Match ID must be a whole number -- got {match_report_id_input!r}.")
        else:
            with st.spinner("Building tactical timeline (Weak-Spot Lifetime for both teams + Tactical Events)..."):
                timeline_data = _fetch_report_safely(
                    lambda: _cached_match_timeline(rest_base_url, timeline_match_id), rest_base_url
                )

            if timeline_data is not None:
                if timeline_data.get("no_data"):
                    st.info(timeline_data.get("reason", "No timeline data available for this match_id."))
                else:
                    timeline_png = _cached_match_timeline_png(timeline_data)
                    st.image(
                        timeline_png,
                        caption=f"Tactical Timeline -- match_id={timeline_match_id}",
                        width="stretch",
                    )
                    timeline_cols = st.columns(4)
                    timeline_cols[0].metric("Counter Attacks", timeline_data["counter_attack_count"])
                    timeline_cols[1].metric("Build-up Patterns", timeline_data["build_up_count"])
                    timeline_cols[2].metric("Switches of Play", timeline_data["switch_of_play_count"])
                    timeline_cols[3].metric("Weak-Spot Instances", timeline_data["weak_spot_instance_count"])
                    st.caption(
                        f"Period 2 display offset: +{timeline_data['period_2_display_offset']:.1f} min "
                        f"(real period-1 max minute: {timeline_data['period_1_max_minute']:.1f}, not assumed 45.0)."
                    )

                    with st.expander("Raw timeline data"):
                        st.json(timeline_data)


# ============================================================================
# TAB: Tactical Chat (new reporting track, Part B -- AI Tactical Chat). A
# pure UI wiring layer over the new POST /chat/tactical endpoint.
#
# PLACED IN ITS OWN TAB, not folded into "Live CV Monitor" (the other option
# this task's own instructions named): the module docstring's documented
# "PERMANENT CONSEQUENCE" already means EVERY tab in this file is equally
# blocked whenever the Live CV Monitor tab's stream/simulator loop is
# running -- placing Chat there specifically would not avoid that shared
# blocking surface (it applies file-wide, not per-tab), and would only
# conflate an unrelated, purely request/response feature with that tab's
# already-documented blocking-loop caveat for no benefit. A new tab keeps
# Chat's own state (conversation history, session id) independent, the
# same reasoning the Match Report and Alerts History tabs above already
# follow.
# ============================================================================
with tab_chat:
    st.header("AI Tactical Chat")
    st.caption(
        "Ask follow-up questions about a specific match. Every reply is grounded ONLY in that "
        "match's real, already-computed data (team reports, opposition analysis, pass network, "
        "recent alerts) -- rebuilt fresh before every single reply, never cached across turns. "
        "If you ask something this system hasn't computed (a future prediction, a substitution "
        "recommendation, an injury status), it will say so plainly rather than guess. "
        "Powered by POST /chat/tactical."
    )

    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = str(uuid.uuid4())
    if "chat_display_history" not in st.session_state:
        st.session_state["chat_display_history"] = []

    chat_match_id_input = st.text_input(
        "Match ID (StatsBomb match_id) -- context is rebuilt from this match on every message",
        value=DEFAULT_MATCH_ID,
        key="chat_match_id_input",
    )

    if st.button("Reset conversation", key="chat_reset_button"):
        st.session_state["chat_session_id"] = str(uuid.uuid4())
        st.session_state["chat_display_history"] = []
        st.rerun()

    for chat_turn in st.session_state["chat_display_history"]:
        with st.chat_message(chat_turn["role"]):
            st.write(chat_turn["text"])
            if chat_turn.get("source"):
                st.caption(f"(source: {chat_turn['source']})")

    chat_user_message = st.chat_input("Ask about this match's tactical picture...")
    if chat_user_message:
        try:
            chat_match_id = int(chat_match_id_input.strip())
        except ValueError:
            st.error(f"Match ID must be a whole number -- got {chat_match_id_input!r}.")
        else:
            st.session_state["chat_display_history"].append({"role": "user", "text": chat_user_message})
            with st.chat_message("user"):
                st.write(chat_user_message)

            with st.chat_message("assistant"):
                with st.spinner("Rebuilding this match's context and generating a reply..."):
                    chat_result = _fetch_chat_reply_safely(
                        rest_base_url, st.session_state["chat_session_id"], chat_match_id, chat_user_message
                    )
                if chat_result is not None:
                    reply_text = chat_result["reply"]
                    reply_source = chat_result.get("reply_source", "unknown")
                    st.write(reply_text)
                    st.caption(f"(source: {reply_source})")
                    st.session_state["chat_display_history"].append(
                        {"role": "assistant", "text": reply_text, "source": reply_source}
                    )
                # On a real request failure, `_fetch_chat_reply_safely` has
                # already rendered `st.error(...)` inside this same
                # `st.chat_message("assistant")` container -- nothing is
                # appended to `chat_display_history` in that case, so a
                # failed turn does not leave a fabricated assistant message
                # in the conversation the model would otherwise be shown as
                # its own prior turn on a retry.
