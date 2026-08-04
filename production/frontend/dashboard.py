"""Milestones 17 & 19 (Module 3/9 UI), extended with the reporting track's
Streamlit integration: Project Athena's dashboard, now five tabs wide.

"Live CV Monitor" holds the two ORIGINAL panels, unchanged in behavior:

  1. "Tactical What-If Simulator" (Milestone 19) -- a plain synchronous
     REST call to `/simulate` (Milestone 18), triggered by "Run Simulation".
     Fast, request/response, no blocking loop.
  2. "Live Tactical Threat Monitor" (Milestone 17) -- a WebSocket stream,
     triggered by "Start Stream". See the ARCHITECTURAL DECISION below.

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

import json
import sys
import tempfile
import time
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
from production.src.reporting.player_visualizer import (
    render_player_dashboard,
    render_shot_map,
)
from production.src.reporting.team_trend_data import generate_team_trend_report
from production.src.reporting.team_visualizer import render_team_dashboard

DEFAULT_REST_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws/tactical-stream"
DEFAULT_MATCH_ID = "3857276"
# Milestone 33: server-side default for the CV data source -- MUST resolve
# inside the backend's data/raw/ directory (ALLOWED_CV_VIDEO_DIRECTORY in
# api.py); this is just a plausible-looking placeholder, not a file
# guaranteed to exist. Every CV milestone since 25 has noted no persistent
# real footage lives under this path in this environment -- "StatsBomb
# Replay" (below) is therefore the data source actually usable out of the
# box, and stays the default `st.radio` selection (index 0) for exactly
# that reason. Left here, unchanged, as a plausible path for anyone who
# DOES have a real local clip to point this at deliberately.
DEFAULT_CV_VIDEO_PATH = "data/raw/test_match.mp4"
MAX_THREAT_BUFFER_LEN = 60
MAX_ALERT_BUFFER_LEN = 20
RECV_TIMEOUT_SECONDS = 60.0  # how long to wait for a single message before treating the stream as stalled
SIMULATE_REQUEST_TIMEOUT_SECONDS = 5.0  # mandatory -- see What-If section below
TACTICAL_ACTIONS = ["high_press", "drop_deep", "force_wide", "no_change"]
# ADR-018: report endpoints do real network fetches (StatsBomb), MLflow
# artifact loads, and pitch-control physics across potentially many chains --
# a single /simulate-style 5s budget is too tight for these. 60s matches this
# file's own RECV_TIMEOUT_SECONDS as a "generous but bounded" convention.
REPORT_REQUEST_TIMEOUT_SECONDS = 60.0

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
def _cached_team_report(rest_base_url: str, team_name: str, match_ids: tuple[int, ...]) -> dict:
    response = requests.get(
        f"{rest_base_url}/reports/team/{requests.utils.quote(team_name, safe='')}",
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


# team_trend_data.py is a deliberate, NAMED EXCEPTION to ADR-018's
# consolidation -- see this file's module docstring and ADR-018 itself.
# That module's own docstring already states it must never be wired into
# api.py's served layer (unresolved football-data.co.uk licensing scope),
# so this tab keeps calling generate_team_trend_report directly, unlike the
# other three reporting tabs.
@st.cache_data(show_spinner=False)
def _cached_team_trend_report(team_name: str, start_season: int, end_season: int) -> dict:
    return generate_team_trend_report(team_name, start_season, end_season)


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


def _fetch_report_safely(fetch_fn, rest_base_url: str) -> dict | None:
    """Calls `fetch_fn()` (a zero-arg closure around one of the `_cached_*`
    HTTP wrappers above) and returns its parsed JSON, or `None` if the
    request failed -- rendering a clean `st.error` in the same style this
    file's original What-If Simulator panel already uses, rather than
    letting an unhandled `requests` exception crash this tab's script
    execution (ADR-018: these three tabs now make a real network call,
    where before they were in-process function calls that could not fail
    this way)."""
    try:
        return fetch_fn()
    except requests.exceptions.Timeout:
        st.error(
            f"Report request timed out after {REPORT_REQUEST_TIMEOUT_SECONDS:.0f}s -- the backend "
            f"at {rest_base_url} did not respond in time."
        )
    except requests.exceptions.ConnectionError:
        st.error(
            f"Backend unreachable at {rest_base_url} -- confirm the FastAPI server is running "
            "(uvicorn production.src.serving.api:app)."
        )
    except requests.exceptions.HTTPError as exc:
        st.error(f"Report request failed: {exc}")
    return None


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

tab_cv, tab_player, tab_team, tab_trends, tab_compare = st.tabs(
    ["Live CV Monitor", "Player Reports", "Team Reports", "Team Trends", "Team Comparison"]
)

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
                    "field above is ignored for this data source. NOTE: no real footage ships with "
                    "this project by default -- you must point this at your own local clip."
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

    simulation_result_placeholder = st.empty()

    if run_simulation_clicked:
        simulation_result_placeholder.info("Running simulation...")
        try:
            response = requests.get(
                f"{rest_base_url}/simulate",
                params={"match_id": match_id, "minute": int(minute), "action": action},
                timeout=SIMULATE_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            simulation_result_placeholder.error(
                f"Request timed out after {SIMULATE_REQUEST_TIMEOUT_SECONDS:.0f}s -- the backend at "
                f"{rest_base_url} did not respond in time."
            )
        except requests.exceptions.ConnectionError:
            simulation_result_placeholder.error(
                f"Backend unreachable at {rest_base_url} -- confirm the FastAPI server is running "
                "(uvicorn production.src.serving.api:app)."
            )
        except requests.exceptions.HTTPError as exc:
            simulation_result_placeholder.error(f"Simulation request failed: {exc}")
        except Exception as exc:
            simulation_result_placeholder.error(f"Simulation request failed: {exc}")
        else:
            baseline = result["baseline_threat_15s"]
            simulated = result["simulated_threat_15s"]
            delta = result["delta"]

            with simulation_result_placeholder.container():
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

    preset_label = st.selectbox("Player", list(_player_labels.keys()) + ["Custom"])
    if preset_label == "Custom":
        player_id_input = st.text_input("Player ID (StatsBomb player_id)", value="")
        match_ids_input = st.text_input("Match IDs (comma-separated StatsBomb match_id list)", value="")
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
                with st.spinner("Generating shot map..."):
                    shot_map = _fetch_report_safely(
                        lambda: _cached_player_shot_map(rest_base_url, player_id, match_ids), rest_base_url
                    )
                if shot_map is not None:
                    shot_map_png = _cached_player_shot_map_png(shot_map)

                    # Same low-sample visual convention as the positional/
                    # heatmap warnings above -- reused, not reinvented.
                    if shot_map.get("shot_map_used_low_sample_flag"):
                        st.warning(
                            f"LOW SAMPLE: only {shot_map.get('total_shots', 0)} shot(s) for this player -- "
                            "treat the shot map below as illustrative, not a confident pattern."
                        )

                    st.image(shot_map_png, caption=f"Shot Map -- player_id={player_id}", width="stretch")

                    with st.expander("Raw shot map data"):
                        st.json(shot_map)

# ============================================================================
# TAB: Team Reports -- report DATA now fetched over HTTP from api.py's
# /reports/team/{team_name} (ADR-018); PNG rendering still calls
# team_visualizer.py directly (pure client-side rendering, no MLflow/
# data/raw access of its own). Neither team_report.py nor
# team_visualizer.py is modified.
# ============================================================================
with tab_team:
    st.header("Team Report")

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

    team_preset_label = st.selectbox("Team", list(_team_labels.keys()) + ["Custom"])
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

            if team_report_dict is not None:
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

# ============================================================================
# TAB: Team Trends -- UI wiring over team_trend_data.py, unmodified. A
# DELIBERATE, NAMED EXCEPTION to ADR-018 (see module docstring): this tab
# still calls generate_team_trend_report directly, in-process, because
# that module's own docstring forbids wiring it into api.py's served layer
# pending resolution of its data source's licensing scope. This is the one
# reporting tab that still requires dashboard.py to run with its own
# data/raw/ and football-data.co.uk network access -- not fully separated
# from the backend the way the other three tabs now are.
# ============================================================================
with tab_trends:
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
            with st.spinner("Fetching and aggregating season data..."):
                trend_report = _cached_team_trend_report(trend_team_name.strip(), int(trend_start_season), int(trend_end_season))

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
                st.error(f"No seasons found for {trend_team_name!r} in the requested range across any covered league.")

            with st.expander("Raw report data"):
                st.json(trend_report)

# ============================================================================
# TAB: Team Comparison -- report DATA now fetched over HTTP from api.py's
# /reports/team-comparison (ADR-018). team_comparison.py itself is not
# modified.
# ============================================================================
with tab_compare:
    st.header("Team-Season Style Comparison")

    compare_col_a, compare_col_b = st.columns(2)
    with compare_col_a:
        st.subheader("Team A")
        compare_team_a = st.text_input("Team A name (StatsBomb team name)", value="Barcelona")
        compare_season_a = st.number_input("Team A season (start year)", min_value=1990, max_value=2100, value=2008, step=1)
    with compare_col_b:
        st.subheader("Team B")
        compare_team_b = st.text_input("Team B name (StatsBomb team name)", value="Barcelona")
        compare_season_b = st.number_input("Team B season (start year)", min_value=1990, max_value=2100, value=2015, step=1)

    generate_comparison_clicked = st.button("Compare")

    if generate_comparison_clicked:
        with st.spinner("Fetching match data and computing comparison..."):
            comparison = _fetch_report_safely(
                lambda: _cached_team_comparison(
                    rest_base_url,
                    compare_team_a.strip(), int(compare_season_a),
                    compare_team_b.strip(), int(compare_season_b),
                ),
                rest_base_url,
            )

        if comparison is not None:
            st.subheader(f"Analysis mode: `{comparison['analysis_mode']}`")
            st.caption(comparison["mode_reason"])

            richness_col_a, richness_col_b = st.columns(2)
            with richness_col_a:
                ra = comparison["data_richness"]["team_a"]
                st.metric(f"{ra['team']} {ra['season']} matches", ra["matches"])
                st.caption(ra["flag"])
            with richness_col_b:
                rb = comparison["data_richness"]["team_b"]
                st.metric(f"{rb['team']} {rb['season']} matches", rb["matches"])
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
