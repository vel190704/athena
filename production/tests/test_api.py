"""Milestone 16 validation: FastAPI WebSocket live tactical stream.

TestClient drives the ASGI app in-process (no real network/uvicorn
process) with delay=0.0 for fast execution, against the same cached real
match (3857276) used throughout Milestones 13-15 -- consistent with this
project's preference for testing against real data over synthetic
fixtures.
"""

import os
import queue
import threading
import time

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from fastapi.testclient import TestClient

from production.src.serving.api import app

MATCH_ID = 3857276


def _try_receive_json(websocket, timeout: float = 2.0):
    """Attempts one more `receive_json()`, but gives up after `timeout`
    seconds instead of blocking forever. Starlette's WebSocketTestSession
    has no built-in receive timeout, so the blocking call is run on a
    background thread and joined with a deadline; if nothing arrives in
    time, the thread is abandoned (it's a daemon) and this returns None
    rather than hanging the suite when no further message is ever sent.
    """
    result: queue.Queue = queue.Queue(maxsize=1)

    def _worker():
        try:
            result.put(("ok", websocket.receive_json()))
        except Exception as exc:
            result.put(("error", exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        status, value = result.get(timeout=timeout)
    except queue.Empty:
        return None

    if status == "error":
        raise value
    return value


def _assert_valid_message(message: dict) -> None:
    assert "type" in message, f"message missing 'type' field: {message}"

    if message["type"] == "threat":
        assert "minute" in message, f"threat message missing 'minute': {message}"
        assert "threat_15s" in message, f"threat message missing 'threat_15s': {message}"
        threat = message["threat_15s"]
        assert isinstance(threat, float), f"threat_15s should be a float, got {type(threat)}"
        assert 0.0 <= threat <= 1.0, f"threat_15s out of [0, 1]: {threat}"
    elif message["type"] == "alert":
        assert "explanation" in message, f"alert message missing 'explanation': {message}"
        explanation = message["explanation"]
        assert isinstance(explanation, str) and len(explanation) > 0, (
            f"alert explanation should be a non-empty string, got {explanation!r}"
        )
    else:
        raise AssertionError(f"unexpected message type: {message['type']!r}")


def test_tactical_stream_yields_valid_threat_and_optional_alert_messages():
    with TestClient(app) as client:
        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0"
        ) as websocket:
            messages = [websocket.receive_json() for _ in range(3)]
            for message in messages:
                _assert_valid_message(message)

            threat_count = sum(1 for m in messages if m["type"] == "threat")
            alert_count = sum(1 for m in messages if m["type"] == "alert")
            print(f"\nReceived {len(messages)} messages: {threat_count} threat, {alert_count} alert")

            # Buffer for any in-flight background alert task to finish and
            # be received -- purely opportunistic: whether a spike occurs
            # within this window depends on the match's real data and is
            # NOT asserted either way (per Step 4.4).
            time.sleep(0.5)
            trailing = _try_receive_json(websocket, timeout=1.5)
            if trailing is not None:
                _assert_valid_message(trailing)
                print(f"Trailing message after buffer: {trailing}")
                if trailing["type"] == "alert":
                    print(f"Alert fired during normal-threshold streaming: {trailing['explanation']}")


def test_per_connection_spike_state_is_isolated():
    """Validates Step 3.3/3.4's per-connection (not global) state
    requirement.

    Connection A opens with an artificially low spike_threshold (0.0), so
    across a handful of real frames its previous_threat_15s is very likely
    driven upward by at least one genuine spike -- this is the Step 5.2
    "extra check" that reliably forces the alert path (and, with it, the
    connection_lock-guarded interleaved send) to actually execute.

    Connection B then opens FRESH with the normal threshold, and its very
    FIRST message is hard-asserted to be type="threat", never "alert": a
    brand-new connection's previous_threat_15s always starts at None, so
    no spike delta can be computed on message 1 -- REGARDLESS of whatever
    connection A's state looked like. If previous_threat_15s were
    accidentally module/global state instead of connection-local, A's
    activity could leak into B's very first comparison and incorrectly
    fire an alert here.
    """
    with TestClient(app) as client:
        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0&spike_threshold=0.0"
        ) as connection_a:
            a_messages = [connection_a.receive_json() for _ in range(5)]
            for message in a_messages:
                _assert_valid_message(message)
            a_alert_count = sum(1 for m in a_messages if m["type"] == "alert")
            print(f"\nConnection A (spike_threshold=0.0) alerts observed in first 5 messages: {a_alert_count}")

            # Let any in-flight alert task on connection A finish before
            # tearing the connection down, to avoid a noisy (harmless)
            # "send on closed connection" background-task warning.
            time.sleep(0.5)

        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0"
        ) as connection_b:
            first_message = connection_b.receive_json()
            _assert_valid_message(first_message)
            assert first_message["type"] == "threat", (
                "connection B's first message must be type='threat' -- its own "
                "previous_threat_15s must start at None regardless of connection A's prior "
                f"activity, so no spike alert can fire on message 1. Got: {first_message}"
            )

    print(
        f"\nPer-connection isolation confirmed: connection B's first message was "
        f"type={first_message['type']!r} (never an alert), independent of connection A's "
        "spike_threshold=0.0 activity on a separate connection."
    )
