"""Tactical Momentum (additive new feature): a rolling-window smoothing +
trend indicator over the already-streamed `threat_15s` signal that
dashboard.py's Live CV Monitor tab receives over `/ws/tactical-stream`.

Deliberately a PLAIN Python module -- no `streamlit` import, no side
effects at import time -- kept separate from `dashboard.py` (a Streamlit
SCRIPT, not a safely importable module: it executes top-level Streamlit
calls that assume a live ScriptRunContext) specifically so this pure
computation can be unit-tested directly via a normal `import`, without
going through `streamlit.testing.v1.AppTest`. `dashboard.py` imports
`_compute_tactical_momentum` and the constants below and calls them over
the SAME `threat_buffer` list it already accumulates from incoming
WebSocket messages -- this module holds no buffer of its own and performs
no I/O; all buffering/streaming/rendering stays in dashboard.py.
"""

# Chosen INDEPENDENTLY of dashboard.py's MAX_THREAT_BUFFER_LEN (60) -- that
# buffer controls how much history the LINE CHART retains for visual
# inspection, a separate concern from how many points denoise one momentum
# reading. 10 was chosen so several independent windows (6, at the
# buffer's full 60-message length) fit inside the retained buffer, and so
# momentum updates on roughly a 10-message lag (~10s at this stream's
# default delay=1.0s pacing, per simulator.py's `live_match_stream`) --
# responsive enough to be useful live, without letting single-message
# noise dominate it.
MOMENTUM_SMOOTHING_WINDOW_MESSAGES = 10

# The discrete-derivative lookback: trend = smoothed_now - smoothed_then,
# where smoothed_then is the same MOMENTUM_SMOOTHING_WINDOW_MESSAGES-wide
# average as of this many messages ago. Kept as a SEPARATE named constant
# from the smoothing window above (even though both currently equal 10)
# since they are conceptually distinct: one is "how many points denoise a
# single reading," the other is "how far back to compare against."
MOMENTUM_TREND_LOOKBACK_MESSAGES = 10

MOMENTUM_MIN_MESSAGES_FOR_TREND = MOMENTUM_SMOOTHING_WINDOW_MESSAGES + MOMENTUM_TREND_LOOKBACK_MESSAGES

# Half of api.py's own SPIKE_THRESHOLD (0.05) -- see _compute_tactical_
# momentum's docstring for the full reasoning.
MOMENTUM_TREND_THRESHOLD = 0.025


def _compute_tactical_momentum(threat_buffer: list[float]) -> dict:
    """Tactical Momentum: a rolling-window smoothing + trend indicator
    over the already-streamed `threat_15s` signal. Pure, side-effect-free
    computation over a plain list of floats -- reads the SAME
    `threat_buffer` dashboard.py's existing line chart already
    accumulates, does not duplicate any buffering logic, and never
    touches the WebSocket message itself (works identically regardless of
    which field(s) a given message carries beyond `threat_15s` -- see
    dashboard.py's module docstring for why that matters for source=cv).

    STEP 0 DEFINITIONS -- a genuine judgment call, documented explicitly
    here rather than left implicit (same discipline player_report.py's
    shot-map "on-target" choice already established for this project):

    1. SMOOTHING: a simple moving average over the most recent
       MOMENTUM_SMOOTHING_WINDOW_MESSAGES (10) raw `threat_15s` readings.
       See the module-level constant's own comment for why 10, and why
       independent of dashboard.py's MAX_THREAT_BUFFER_LEN (60).
    2. TREND: a discrete derivative, NOT a linear regression -- the
       smoothed value now MINUS the smoothed value from
       MOMENTUM_TREND_LOOKBACK_MESSAGES (10) messages ago. A full
       regression would fit a line to samples that are event-paced, not
       evenly time-spaced (see `live_match_stream`'s own docstring: one
       message per 360-tagged event, not one per fixed real-world
       second), for marginal benefit over a simple two-point difference
       of two already-smoothed values -- not used here for that reason.
    3. CLASSIFICATION: `trend > +MOMENTUM_TREND_THRESHOLD` (0.025, i.e.
       2.5 percentage points of threat probability) -> "Building";
       `trend < -MOMENTUM_TREND_THRESHOLD` -> "Fading"; otherwise ->
       "Stable". 0.025 is deliberately HALF of api.py's own
       SPIKE_THRESHOLD (0.05): SPIKE_THRESHOLD is calibrated to flag a
       single, noisy message-to-message jump, whereas this trend value is
       already a difference of two SMOOTHED (10-point-averaged) readings
       -- a real, sustained shift half that size already reflects genuine
       directional movement rather than raw single-message noise, and
       reasonably deserves a lower bar than the raw spike-alert
       threshold. A reasonable, stated choice, not a provably optimal
       one -- validated against a real match stream (see
       test_dashboard.py) rather than picked and left unchecked.
    4. INSUFFICIENT DATA: fewer than MOMENTUM_MIN_MESSAGES_FOR_TREND (20
       = 10 + 10) raw `threat_15s` readings in the buffer -> returns a
       `{"status": "warming_up", ...}` dict instead of a fabricated or
       crashing trend/classification. Once enough data exists, returns
       `{"status": "ready", "smoothed_now": ..., "trend": ...,
       "classification": ...}`.
    """
    n = len(threat_buffer)
    if n < MOMENTUM_MIN_MESSAGES_FOR_TREND:
        return {
            "status": "warming_up",
            "messages_so_far": n,
            "messages_needed": MOMENTUM_MIN_MESSAGES_FOR_TREND,
        }

    smoothed_now = (
        sum(threat_buffer[-MOMENTUM_SMOOTHING_WINDOW_MESSAGES:]) / MOMENTUM_SMOOTHING_WINDOW_MESSAGES
    )
    lookback_window = threat_buffer[
        -(MOMENTUM_TREND_LOOKBACK_MESSAGES + MOMENTUM_SMOOTHING_WINDOW_MESSAGES):
        -MOMENTUM_TREND_LOOKBACK_MESSAGES
    ]
    smoothed_then = sum(lookback_window) / MOMENTUM_SMOOTHING_WINDOW_MESSAGES

    trend = smoothed_now - smoothed_then
    if trend > MOMENTUM_TREND_THRESHOLD:
        classification = "Building"
    elif trend < -MOMENTUM_TREND_THRESHOLD:
        classification = "Fading"
    else:
        classification = "Stable"

    return {
        "status": "ready",
        "smoothed_now": smoothed_now,
        "trend": trend,
        "classification": classification,
    }
