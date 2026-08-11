"""ADR-022 Update (Phase 2): rate limiting.

Validates BOTH halves of the requirement: (1) PUBLIC_DEPLOYMENT unset
(local dev / this project's own test suite default) sees ZERO behavior
change -- no request is ever throttled; (2) PUBLIC_DEPLOYMENT=true
genuinely enforces the configured tiered limits, with real, TRIGGERED
429s (REST) / close-code-1013s (WebSocket), not just code review. Same
`TestClient`-in-process convention `test_api.py` already establishes.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import production.src.serving.api as api_module
from production.src.serving.api import RATE_LIMIT_TIERS, app

MATCH_ID = 3857276
ARGENTINA_MATCH_IDS = [3857264]


@pytest.fixture(autouse=True)
def _reset_rate_limiter_buckets():
    """Every `_RateLimiter`'s own `_buckets` dict is real, in-process
    module state that persists across tests within one pytest session
    (the same reason `test_api.py`'s own `_model`/`_yolo_checkpoint_warmed`
    module globals need no per-test reset -- but a rate limiter's bucket
    state, unlike those, is exactly what each test below needs to
    control precisely). Cleared before AND after every test in this file
    so no test's real token consumption leaks into another's."""
    for limiter in api_module._rate_limiters.values():
        limiter._buckets.clear()
    yield
    for limiter in api_module._rate_limiters.values():
        limiter._buckets.clear()


# --- Step 1.1: OFF by default (PUBLIC_DEPLOYMENT unset) -------------------


def test_rate_limiting_off_by_default_many_rapid_requests_all_succeed():
    """Sends MORE requests than even the tightest real tier's capacity
    (heavy=6/min) to a heavy-tier endpoint, with PUBLIC_DEPLOYMENT left
    at its real default (unset) -- every single one must succeed (no
    429), proving this is genuinely OFF, not just "set very high"."""
    assert api_module.PUBLIC_DEPLOYMENT is False
    with TestClient(app) as client:
        statuses = [
            client.get("/reports/team/Argentina", params={"match_ids": ARGENTINA_MATCH_IDS}).status_code
            for _ in range(10)
        ]
    assert all(status == 200 for status in statuses), statuses
    assert 429 not in statuses


def test_rate_limit_key_function_scoping(monkeypatch):
    """Step 0.2: genuinely different keying logic per mode, not one
    hardcoded scheme -- verified directly on the function itself."""
    monkeypatch.setattr(api_module, "API_KEY", None)
    assert api_module._rate_limit_key("1.2.3.4") == "ip:1.2.3.4"
    assert api_module._rate_limit_key(None) == "ip:unknown"

    monkeypatch.setattr(api_module, "API_KEY", "shared-secret")
    assert api_module._rate_limit_key("1.2.3.4") == "key:shared-secret"
    assert api_module._rate_limit_key("5.6.7.8") == "key:shared-secret"  # same key regardless of caller IP


# --- Step 3.2: a REAL, triggered 429 under PUBLIC_DEPLOYMENT=true ---------


def test_rate_limit_triggers_real_429_at_configured_heavy_tier_capacity(monkeypatch):
    """THE real, triggered test -- not code review. Uses the REAL
    configured "heavy" tier capacity (6/min, RATE_LIMIT_TIERS["heavy"]),
    completely unmodified, against a real heavy-tier endpoint
    (/reports/team/{team_name}). The 7th rapid request within the same
    minute must be rejected with a real 429 and a real Retry-After
    header; the first 6 must NOT be rate-limited.

    `generate_team_report` itself IS stubbed (not the rate limiter) --
    a REAL, measured finding from this test's own first draft: the
    "heavy" tier's own 6/min refill rate (1 token per 10s) is close
    enough to team_report's own real multi-second pitch-control cost
    that 6 genuinely slow sequential real calls could let the bucket
    partially refill mid-test, making the 7th call's rejection
    genuinely timing-dependent/flaky -- not a bug in the rate limiter,
    but a real interaction between two independently-reasonable real
    costs. Stubbing the endpoint's OWN compute (the exact same technique
    already used for the similarity_rebuild test below, for the exact
    same reason) removes that race while leaving the rate-limiting
    dependency itself, which is what this test verifies, completely
    real and untouched.
    """
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setattr(api_module, "generate_team_report", lambda team_name, match_ids: {"team_name": team_name})
    assert RATE_LIMIT_TIERS["heavy"] == 6.0

    with TestClient(app) as client:
        responses = [
            client.get("/reports/team/Argentina", params={"match_ids": ARGENTINA_MATCH_IDS})
            for _ in range(7)
        ]

    statuses = [r.status_code for r in responses]
    assert statuses[:6].count(429) == 0, f"first 6 requests (== configured capacity) should not be rate-limited: {statuses}"
    assert statuses[6] == 429, f"7th request (over configured capacity) should be rate-limited: {statuses}"
    assert "Retry-After" in responses[6].headers
    assert int(responses[6].headers["Retry-After"]) >= 1


def test_rate_limit_triggers_real_429_at_configured_similarity_rebuild_capacity(monkeypatch):
    """Same real, triggered discipline, for the similarity_rebuild tier's
    OWN much tighter real capacity (1 per 30 minutes) -- the 2nd rapid
    request must 429. `build_player_similarity_index` itself is stubbed
    (NOT the rate limiter) purely so this test doesn't trigger a real
    ~27-minute rebuild -- the rate-limiting dependency itself, which is
    what this test verifies, is completely real and unmodified.
    """
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setattr(
        api_module, "build_player_similarity_index",
        lambda: {"searchable_population_size": 0, "total_cached_population_size": 0, "build_duration_seconds": 0.0},
    )

    with TestClient(app) as client:
        first = client.post("/reports/player-similarity/rebuild")
        second = client.post("/reports/player-similarity/rebuild")

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers
    # A real ~30-minute window -- the returned Retry-After should be a
    # real, large number (not e.g. a leftover 1-second default), roughly
    # up to 1800s, confirming this tier's own distinct window is genuinely
    # wired through, not accidentally sharing another tier's short one.
    assert 1000 < int(second.headers["Retry-After"]) <= 1800


def test_rate_limit_response_body_names_the_tier(monkeypatch):
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setattr(api_module, "generate_team_report", lambda team_name, match_ids: {"team_name": team_name})
    with TestClient(app) as client:
        for _ in range(6):
            client.get("/reports/team/Argentina", params={"match_ids": ARGENTINA_MATCH_IDS})
        limited = client.get("/reports/team/Argentina", params={"match_ids": ARGENTINA_MATCH_IDS})
    assert limited.status_code == 429
    assert "heavy" in limited.json()["detail"]


# --- Step 3.3: /health and /metrics stay effectively unthrottled ----------


def test_health_endpoint_unthrottled_under_public_deployment(monkeypatch):
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    with TestClient(app) as client:
        statuses = [client.get("/health").status_code for _ in range(20)]
    assert all(status == 200 for status in statuses)


def test_metrics_endpoint_effectively_unthrottled_under_public_deployment(monkeypatch):
    """20 rapid requests, well under the metrics tier's real 300/min
    capacity -- none should be rate-limited."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    assert RATE_LIMIT_TIERS["metrics"] == 300.0
    with TestClient(app) as client:
        statuses = [client.get("/metrics").status_code for _ in range(20)]
    assert all(status == 200 for status in statuses)


# --- Step 2.2: WebSocket connection-rate limiting (real, triggered) -------


def test_websocket_connection_rate_limit_real_triggered(monkeypatch):
    """Real, triggered test against the REAL configured
    RATE_LIMIT_TIERS["websocket_connect"] capacity (10/min): the 11th
    rapid connection attempt within the same minute must be rejected at
    the handshake with close code 1013 ("Try Again Later"), before
    `accept()` -- the first 10 must connect successfully."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    assert RATE_LIMIT_TIERS["websocket_connect"] == 10.0

    with TestClient(app) as client:
        for i in range(10):
            with client.websocket_connect(f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0") as ws:
                ws.receive_json()  # confirm the connection genuinely accepted and is streaming

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1013


def test_websocket_connection_rate_limiting_off_by_default():
    """PUBLIC_DEPLOYMENT unset (real default): more connection attempts
    than the configured websocket_connect capacity must ALL succeed."""
    assert api_module.PUBLIC_DEPLOYMENT is False
    with TestClient(app) as client:
        for _ in range(12):
            with client.websocket_connect(f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0") as ws:
                ws.receive_json()
