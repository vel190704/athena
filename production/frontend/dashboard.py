"""Milestones 17 & 19 (Module 3/9 UI): Project Athena's tactical dashboard.

Two independent panels share this one script:

  1. "Tactical What-If Simulator" (Milestone 19) -- a plain synchronous
     REST call to `/simulate` (Milestone 18), triggered by "Run Simulation".
     Fast, request/response, no blocking loop.
  2. "Live Tactical Threat Monitor" (Milestone 17) -- a WebSocket stream,
     triggered by "Start Stream". See the ARCHITECTURAL DECISION below.

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

PERMANENT CONSEQUENCE FOR MILESTONE 19 (do not paper over this): because
that blocking loop occupies the ENTIRE script execution while it runs,
Streamlit cannot process ANY other widget interaction -- including the
What-If panel's "Run Simulation" button -- until the loop ends. The two
panels are laid out on the same page for convenience, but they are NOT
usable at the same time: whichever action was clicked (Start Stream, or
Run Simulation) is the only one that runs in that script execution. This
is a real, permanent limitation of the single-blocking-loop architecture,
not a bug to fix or a scenario to engineer around -- it is documented here
and surfaced directly in the UI caption below.

Do NOT "fix" this by moving the loop into session_state, a background
thread, or an st.fragment/rerun-driven poll unless you have specifically
re-verified that approach does not reintroduce the update-then-silently-
stop failure mode described above.
"""

import json
import time

import pandas as pd
import requests
import streamlit as st
import websocket

DEFAULT_REST_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws/tactical-stream"
DEFAULT_MATCH_ID = "3857276"
# Milestone 33: server-side default for the CV data source -- MUST resolve
# inside the backend's data/raw/ directory (ALLOWED_CV_VIDEO_DIRECTORY in
# api.py); this is just a plausible-looking placeholder, not a file
# guaranteed to exist.
DEFAULT_CV_VIDEO_PATH = "data/raw/test_match.mp4"
MAX_THREAT_BUFFER_LEN = 60
MAX_ALERT_BUFFER_LEN = 20
RECV_TIMEOUT_SECONDS = 10.0  # how long to wait for a single message before treating the stream as stalled
SIMULATE_REQUEST_TIMEOUT_SECONDS = 5.0  # mandatory -- see What-If section below
TACTICAL_ACTIONS = ["high_press", "drop_deep", "force_wide", "no_change"]

st.set_page_config(page_title="Project Athena: Live Tactical Threat Monitor", layout="wide")
st.title("Project Athena: Live Tactical Threat Monitor")
st.caption(
    "Note: the What-If Simulator and the Live Stream cannot run at the same time -- "
    "the live stream (if started) will block interaction with this panel until it finishes. "
    "See this file's module docstring for why."
)

# --- Sidebar: connection settings, shared by both panels -------------------
with st.sidebar:
    st.header("Connection Settings")
    rest_base_url = st.text_input("REST API Base URL", value=DEFAULT_REST_BASE_URL)
    ws_url = st.text_input("WebSocket URL", value=DEFAULT_WS_URL)
    match_id = st.text_input("Match ID", value=DEFAULT_MATCH_ID)

    st.divider()
    st.header("Live Stream Settings")
    data_source_label = st.radio("Data Source", ["StatsBomb Replay", "CV Video Feed"])
    video_path = None
    if data_source_label == "CV Video Feed":
        video_path = st.text_input(
            "Video Path",
            value=DEFAULT_CV_VIDEO_PATH,
            help=(
                "Server-side file path -- must resolve INSIDE the backend's data/raw/ directory. "
                "The backend rejects (with a clean error, not a crash) any path that resolves "
                "outside it, so don't point this at an arbitrary location on disk. The Match ID "
                "field above is ignored for this data source."
            ),
        )
    max_duration_seconds = st.number_input(
        "Max stream duration (seconds)", min_value=1, max_value=3600, value=300
    )
    max_messages = st.number_input(
        "Max message count", min_value=1, max_value=5000, value=200
    )
    start_clicked = st.button("Start Stream", type="primary")

# ============================================================================
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
# ============================================================================
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

# ============================================================================
# Panel 2: Live Tactical Threat Monitor (Milestone 17) -- unchanged.
# ============================================================================
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
