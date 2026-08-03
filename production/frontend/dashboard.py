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

from production.src.reporting.player_visualizer import render_player_dashboard
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

# --- Reporting-tab UI presets -------------------------------------------
# Cheaply enumerable, already-validated (player/team, match_ids) pairs
# from this project's own history (Milestone 44's validation sweep /
# build_index.py's `render_validation_dashboards`) -- reused here purely
# as convenient UI dropdown entries, NOT a re-implementation of anything
# in player_report.py/team_report.py. "Custom" is always available
# alongside these for any other cached (or fetchable) player/team.
_LEVERKUSEN_MATCH_IDS = (3895052, 3895060, 3895067, 3895074, 3895086, 3895095, 3895107, 3895113)
KNOWN_PLAYER_PRESETS = {
    "Lionel Messi (5503) -- well-supported": (5503, (3773386, 3857264, 3857289, 3857300, 3869151, 3869321, 3869519, 3869685)),
    "Kristijan Jakić (32602) -- LOW SAMPLE, 0 events": (32602, (3869684,)),
    "Yu-Min Cho (99479) -- LOW SAMPLE, 1 event": (99479, (3857262,)),
    "Amine Adli (33401) -- multi-position": (33401, _LEVERKUSEN_MATCH_IDS),
    "Lukáš Hrádecký (8667) -- goalkeeper": (8667, _LEVERKUSEN_MATCH_IDS),
}
KNOWN_TEAM_PRESETS = {
    "Argentina": (3857264, 3857289, 3857300, 3869151),
    "Barcelona -- LOW SAMPLE, 1 match": (3773386,),
    "Bayer Leverkusen": _LEVERKUSEN_MATCH_IDS,
}


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

    preset_label = st.selectbox("Player", list(KNOWN_PLAYER_PRESETS.keys()) + ["Custom"])
    if preset_label == "Custom":
        player_id_input = st.text_input("Player ID (StatsBomb player_id)", value="")
        match_ids_input = st.text_input("Match IDs (comma-separated StatsBomb match_id list)", value="")
        player_id = int(player_id_input) if player_id_input.strip() else None
        match_ids = (
            tuple(int(m.strip()) for m in match_ids_input.split(",") if m.strip())
            if match_ids_input.strip() else ()
        )
    else:
        player_id, match_ids = KNOWN_PLAYER_PRESETS[preset_label]

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

# ============================================================================
# TAB: Team Reports -- report DATA now fetched over HTTP from api.py's
# /reports/team/{team_name} (ADR-018); PNG rendering still calls
# team_visualizer.py directly (pure client-side rendering, no MLflow/
# data/raw access of its own). Neither team_report.py nor
# team_visualizer.py is modified.
# ============================================================================
with tab_team:
    st.header("Team Report")

    team_preset_label = st.selectbox("Team", list(KNOWN_TEAM_PRESETS.keys()) + ["Custom"])
    if team_preset_label == "Custom":
        team_name_input = st.text_input("Team name (StatsBomb team name)", value="", key="team_report_name")
        team_match_ids_input = st.text_input(
            "Match IDs (comma-separated StatsBomb match_id list)", value="", key="team_report_match_ids"
        )
        team_name = team_name_input.strip()
        team_match_ids = (
            tuple(int(m.strip()) for m in team_match_ids_input.split(",") if m.strip())
            if team_match_ids_input.strip() else ()
        )
    else:
        team_name = team_preset_label.split(" -- ")[0]
        team_match_ids = KNOWN_TEAM_PRESETS[team_preset_label]

    generate_team_clicked = st.button("Generate Team Report")

    if generate_team_clicked:
        if not team_name or not team_match_ids:
            st.error("Provide a team name and at least one match_id.")
        else:
            with st.spinner("Generating report..."):
                team_report_dict = _fetch_report_safely(
                    lambda: _cached_team_report(rest_base_url, team_name, team_match_ids), rest_base_url
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
                if team_report_dict["matches_used"] < 2:
                    st.warning(
                        f"LOW SAMPLE: only {team_report_dict['matches_used']} match used -- treat "
                        "this team's pitch-control/threat pattern as illustrative, not a confident "
                        "season-level finding."
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
