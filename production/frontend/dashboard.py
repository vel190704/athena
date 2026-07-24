"""Milestone 17 (Module 3/9 UI): Project Athena's live tactical dashboard.

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
and max-message-count control (Step 2.1), and simply ends on its own after
that -- an honest alternative to a Stop button that would not actually
work while the script is blocked.

Do NOT "fix" this by moving the loop into session_state, a background
thread, or an st.fragment/rerun-driven poll unless you have specifically
re-verified that approach does not reintroduce the update-then-silently-
stop failure mode described above.
"""

import json
import time

import pandas as pd
import streamlit as st
import websocket

DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws/tactical-stream"
DEFAULT_MATCH_ID = "3857276"
MAX_THREAT_BUFFER_LEN = 60
MAX_ALERT_BUFFER_LEN = 20
RECV_TIMEOUT_SECONDS = 10.0  # how long to wait for a single message before treating the stream as stalled

st.set_page_config(page_title="Project Athena: Live Tactical Threat Monitor", layout="wide")
st.title("Project Athena: Live Tactical Threat Monitor")

# --- Sidebar: connection + run-bound controls (Step 2.1) -------------------
with st.sidebar:
    st.header("Stream Settings")
    ws_url = st.text_input("WebSocket URL", value=DEFAULT_WS_URL)
    match_id = st.text_input("Match ID", value=DEFAULT_MATCH_ID)
    max_duration_seconds = st.number_input(
        "Max stream duration (seconds)", min_value=1, max_value=3600, value=300
    )
    max_messages = st.number_input(
        "Max message count", min_value=1, max_value=5000, value=200
    )
    start_clicked = st.button("Start Stream", type="primary")

# --- Main layout: status line + two columns (chart / alerts) --------------
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
    connection_url = f"{ws_url}?match_id={match_id}"

    # Rolling, CAPPED buffers -- plain local variables, intentionally NOT
    # session_state, since the entire stream lifetime happens within this
    # single, uninterrupted script execution (see module docstring).
    threat_buffer: list[float] = []
    alerts_buffer: list[str] = []
    message_count = 0

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

                elif message_type == "alert":
                    alert_text = message.get("explanation", "(empty alert)")
                    alerts_buffer.insert(0, alert_text)
                    if len(alerts_buffer) > MAX_ALERT_BUFFER_LEN:
                        alerts_buffer.pop()
                    _render_alerts(alerts_buffer)

                status_placeholder.info(f"Streaming... ({message_count} messages received)")
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
