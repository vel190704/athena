"""Engineering-review action item: regression coverage for
`production/frontend/dashboard.py`'s five Streamlit tabs -- had no test
file at all, per the review's own finding. Uses Streamlit's own official
`streamlit.testing.v1.AppTest` framework (simulates real widget
interactions and inspects rendered output programmatically) rather than
just importing the module -- the same tool and the same real cases used
during this dashboard's original interactive validation, now persisted
as real regression tests instead of a one-off manual session.

Real, already-cached StatsBomb/football-data.co.uk data throughout (no
mocked report output) -- these tests exercise the ACTUAL reporting
functions via the dashboard's real cached wrappers, the same discipline
`test_reporting.py`/`test_team_comparison.py`/etc. already use.

ADR-018 UPDATE: Player Reports, Team Reports, and Team Comparison now call
`api.py` over REAL HTTP (`requests.get`), not an in-process function call
-- `AppTest` actually executes `dashboard.py`'s real code, including these
real `requests.get()` calls over an actual socket, so unlike every other
test in this suite (which use FastAPI's in-process `TestClient`), these
three tabs' tests need a REAL, listening backend to hit. The
`live_api_server` fixture below starts one (a real `uvicorn.Server`,
full lifespan -- genuine MLflow model load, genuine YOLO checkpoint warm,
exactly what every `with TestClient(app) as client:` block elsewhere in
this project's test suite already triggers, just over an actual bound
socket instead of an in-process ASGI transport) on an OS-assigned free
port, and each affected test points the dashboard's "REST API Base URL"
sidebar field at it before clicking its tab's Generate/Compare button.
The Team Trends tab needs no such fixture: `team_trend_data.py` is a
deliberate, named exception to ADR-018 (see that ADR) and is still called
in-process, unchanged.
"""

import json
import threading
import time

import pytest
import requests
import streamlit as st
import uvicorn
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.testing.v1 import AppTest

from fastapi.testclient import TestClient

import production.src.serving.api as api_module
from production.frontend.tactical_momentum import (
    ELEVATED_THREAT_LEVEL,
    MOMENTUM_MIN_MESSAGES_FOR_TREND,
    MOMENTUM_TREND_THRESHOLD,
    SEGMENT_DWELL_MESSAGES,
    _compute_tactical_momentum,
    classify_match_phase,
)
from production.src.serving import alert_store as alert_store_module
from production.src.serving.api import app as _fastapi_app

DASHBOARD_PATH = "production/frontend/dashboard.py"
APP_TIMEOUT_SECONDS = 180
# Same real cached match test_api.py's own tactical-stream tests use (and
# dashboard.py's own DEFAULT_MATCH_ID) -- delay=0.0 so these tests don't
# pay the streaming pacing cost for real messages they don't need to wait
# real-time for.
TACTICAL_STREAM_MATCH_ID = 3857276


@pytest.fixture(scope="module")
def live_api_server():
    config = uvicorn.Config(_fastapi_app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 120
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("live_api_server did not start in time")

    port = server.servers[0].sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    # Belt-and-suspenders: `server.started` flips only after the ASGI
    # lifespan's own startup (real MLflow load + YOLO warm) has already
    # completed, since Starlette blocks request handling until lifespan
    # startup finishes -- but confirm the app actually answers before
    # handing the URL to a test, rather than trusting a flag alone.
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/openapi.json", timeout=2).status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        raise RuntimeError("live_api_server did not become ready in time")

    yield base_url

    server.should_exit = True
    thread.join(timeout=10)


def test_dashboard_loads_all_ten_tabs_no_exception():
    """Was "...nine tabs..." until this session's Home tab was added.
    Unlike every prior tab addition (Match Report/Tactical Chat, appended
    AFTER every pre-existing tab so no existing index shifted), Home is
    inserted FIRST/leftmost by design -- so every existing `at.tabs[N]`
    index used elsewhere in this file shifts by +1 (see e.g. `at.tabs[3]`
    now resolving to Team Reports, previously `at.tabs[2]`)."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    assert not at.exception
    assert len(at.tabs) == 10
    headers = [h.value for tab in at.tabs for h in tab.header]
    assert "Home" in headers
    assert "Player Report" in headers
    assert "Team Report" in headers
    assert "Team Trend Report (football-data.co.uk)" in headers
    assert "Team-Season Style Comparison" in headers
    assert "Pass Network" in headers
    assert "Alerts History" in headers
    assert "Automatic Match Report" in headers
    assert "AI Tactical Chat" in headers


# ============================================================================
# Home tab (new, this session): a pure read-only orientation layer. Real
# HTTP through `live_api_server`, same discipline as the Alerts History/
# Player/Team Reports tabs above (ADR-018) -- `/health`, `/metrics`,
# `/alerts/history` are all real endpoints on the real FastAPI app, not
# mocked. `_isolated_alerts_db` keeps this test's logged alerts out of the
# real `data/app_state/alerts.db`, same as the Alerts History tests.
# ============================================================================


def test_home_tab_renders_real_status_coverage_activity_and_quick_action(live_api_server, _isolated_alerts_db):
    alert_store_module.log_alert(
        source="statsbomb", match_id=3857276, video_path=None, minute=10.0,
        threat_before=0.05, threat_after=0.20, explanation_text="home tab test alert", explanation_source="mock",
    )
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    assert not at.exception
    home_tab = at.tabs[0]
    assert home_tab.header[0].value == "Home"

    # --- System status: real /health + /metrics data, not mocked --------
    subheaders = [s.value for s in home_tab.subheader]
    assert "System Status" in subheaders
    metric_labels = [m.label for m in home_tab.metric]
    assert "Backend Status" in metric_labels
    backend_status_metric = next(m for m in home_tab.metric if m.label == "Backend Status")
    assert backend_status_metric.value == "ok"
    model_loaded_metric = next(m for m in home_tab.metric if m.label == "Model Loaded")
    assert model_loaded_metric.value == "Yes"

    # --- Data coverage: real candidate_index scan, not fabricated --------
    assert "Data Coverage" in subheaders
    players_metric = next(m for m in home_tab.metric if m.label == "Players Cached")
    teams_metric = next(m for m in home_tab.metric if m.label == "Teams Cached")
    matches_metric = next(m for m in home_tab.metric if m.label == "Matches Cached")
    matches_360_metric = next(m for m in home_tab.metric if m.label == "Matches with 360 Coverage")
    assert int(players_metric.value) > 0
    assert int(teams_metric.value) > 0
    assert int(matches_metric.value) > 0
    # Real invariant, not an assumed fixed number: 360 coverage is a subset of all cached matches.
    assert int(matches_360_metric.value) <= int(matches_metric.value)

    # --- Recent activity: the real alert just logged above ---------------
    assert "Recent Activity" in subheaders
    assert len(home_tab.dataframe) == 1
    df = home_tab.dataframe[0].value
    assert df.shape[0] == 1
    assert df["Match ID"].tolist() == [3857276]

    # --- Quick action: sets the shared Match Report match_id field -------
    assert len(home_tab.button) == 1
    assert "3857276" in home_tab.button[0].label
    home_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)

    assert not at.exception
    match_report_tab = at.tabs[8]
    assert match_report_tab.text_input[0].value == "3857276"


def test_home_tab_no_alerts_shows_info_not_broken_table(live_api_server, _isolated_alerts_db):
    """An empty, isolated alerts db -- Home's Recent Activity section must
    render a clean info message and offer no quick-action button, the same
    'no alerts found' honesty the Alerts History tab's own empty-state test
    establishes, rather than a broken/empty dataframe or a fabricated
    shortcut."""
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    assert not at.exception
    home_tab = at.tabs[0]
    assert len(home_tab.dataframe) == 0
    assert any("No alerts logged yet" in i.value for i in home_tab.info)
    assert len(home_tab.button) == 0


def test_team_reports_tab_data_tiering_note_renders_statically_no_backend_needed():
    """Cross-panel context note (UI walkthrough follow-up): a static,
    always-visible clarification that Team Report needs 360 freeze-frame
    coverage while Tactical Entropy (and every other report on this
    dashboard except Team Report) only needs event data -- so one panel
    succeeding while another fails on the identical team/season selection
    is expected, not a bug. Renders unconditionally at the top of the
    Team Reports tab, before any button click or API call -- confirmed
    here by NOT using `live_api_server` at all (no backend needed for
    this specific assertion), which itself proves the note is static
    (reacting to no request outcome) rather than conditional logic.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    assert not at.exception
    team_tab = at.tabs[3]
    info_texts = [i.value for i in team_tab.info]
    assert any("360 freeze-frame coverage" in t and "Tactical Entropy" in t for t in info_texts)
    assert any("event-data-only" in t for t in info_texts)


# ============================================================================
# TAB: Live CV Monitor -- Tactical Momentum (additive new feature): a
# rolling-window smoothing + trend indicator computed CLIENT-SIDE over the
# `threat_15s` values `/ws/tactical-stream` already sends. `_compute_
# tactical_momentum` lives in the separate, plain-Python `tactical_
# momentum.py` module specifically so it's importable and unit-testable
# without going through `AppTest` (dashboard.py itself cannot safely be
# imported as a plain module -- see its own docstring's import-path
# comment). Real data throughout, same discipline as the rest of this
# file: real `threat_15s` values pulled from a REAL cached match via
# `/ws/tactical-stream` (the exact same endpoint dashboard.py's own
# streaming loop calls), not synthetic numbers -- reuses `TestClient`
# in-process (test_api.py's own pattern for this endpoint) rather than
# the heavier `live_api_server` + real-socket fixture other tests in this
# file need, since this feature never touches Streamlit rendering itself.
# ============================================================================


def _collect_real_threat_sequence(count: int) -> list[float]:
    with TestClient(_fastapi_app) as client:
        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={TACTICAL_STREAM_MATCH_ID}&delay=0.0"
        ) as ws:
            threats: list[float] = []
            while len(threats) < count:
                message = ws.receive_json()
                if message["type"] == "threat":
                    threats.append(message["threat_15s"])
    return threats


def test_compute_tactical_momentum_empty_buffer_warming_up_no_crash():
    """The insufficient-data edge case at the very start of a stream
    (buffer length 0) -- must not crash or fabricate a trend."""
    momentum = _compute_tactical_momentum([])
    assert momentum == {
        "status": "warming_up",
        "messages_so_far": 0,
        "messages_needed": MOMENTUM_MIN_MESSAGES_FOR_TREND,
    }


def test_compute_tactical_momentum_warming_up_boundary_real_data():
    """Real threat_15s values (not synthetic) around the exact
    MOMENTUM_MIN_MESSAGES_FOR_TREND boundary: one message short must still
    report "warming_up" with the real, honest count; exactly at the
    minimum must switch to "ready" with a real classification."""
    threats = _collect_real_threat_sequence(MOMENTUM_MIN_MESSAGES_FOR_TREND)
    assert len(threats) == MOMENTUM_MIN_MESSAGES_FOR_TREND

    one_short = _compute_tactical_momentum(threats[:-1])
    assert one_short["status"] == "warming_up"
    assert one_short["messages_so_far"] == MOMENTUM_MIN_MESSAGES_FOR_TREND - 1
    assert one_short["messages_needed"] == MOMENTUM_MIN_MESSAGES_FOR_TREND

    at_minimum = _compute_tactical_momentum(threats)
    assert at_minimum["status"] == "ready"
    assert at_minimum["classification"] in {"Building", "Fading", "Stable"}


def test_compute_tactical_momentum_real_data_shows_genuine_rise_and_fall():
    """Step 3.1's real sanity check: replayed against 150 real messages
    from a real cached match (source=statsbomb), the classification must
    show genuine variation -- not stuck on one label -- and, specifically,
    both a real "Building" (threat genuinely rising) and a real "Fading"
    (threat genuinely falling) period must actually occur. VERIFIED
    manually against this exact match/stream before writing this
    assertion: real threat_15s rises from ~0.03 to ~0.33 and back down
    across these messages, producing sustained real trend values well
    past MOMENTUM_TREND_THRESHOLD in both directions (e.g. a real
    +0.21 smoothed trend around message 40, a real -0.22 smoothed trend
    around message 100) -- not borderline, noise-driven flips."""
    threats = _collect_real_threat_sequence(150)

    classifications = []
    for i in range(MOMENTUM_MIN_MESSAGES_FOR_TREND, len(threats) + 1):
        momentum = _compute_tactical_momentum(threats[:i])
        assert momentum["status"] == "ready"
        classifications.append(momentum["classification"])

    assert "Building" in classifications
    assert "Fading" in classifications
    assert "Stable" in classifications


def test_compute_tactical_momentum_trend_sign_matches_threshold_real_data():
    """Cross-check independent of the function's own internal threshold
    comparison: for every real "ready" reading, the classification must
    agree with a fresh, independently-computed sign check against
    MOMENTUM_TREND_THRESHOLD -- catches a flipped comparison operator or
    an inverted trend sign that a same-formula assertion could not."""
    threats = _collect_real_threat_sequence(80)

    for i in range(MOMENTUM_MIN_MESSAGES_FOR_TREND, len(threats) + 1):
        momentum = _compute_tactical_momentum(threats[:i])
        trend = momentum["trend"]
        if trend > MOMENTUM_TREND_THRESHOLD:
            assert momentum["classification"] == "Building"
        elif trend < -MOMENTUM_TREND_THRESHOLD:
            assert momentum["classification"] == "Fading"
        else:
            assert momentum["classification"] == "Stable"


# ============================================================================
# TAB: Live CV Monitor -- Match Segmentation (additive new feature): a
# discrete game-phase classifier derived entirely from Tactical Momentum's
# own already-computed output (`smoothed_now`, `classification`) -- no new
# data extraction, Tactical Momentum's own logic above unmodified. Lives in
# the SAME `tactical_momentum.py` module for the same testability reason.
# ============================================================================


def test_classify_match_phase_empty_buffer_warming_up_no_crash():
    """Mirrors Tactical Momentum's own equivalent test -- segmentation
    cannot exist before momentum itself does."""
    phase = classify_match_phase([])
    assert phase == {
        "status": "warming_up",
        "messages_so_far": 0,
        "messages_needed": MOMENTUM_MIN_MESSAGES_FOR_TREND,
    }


def test_classify_match_phase_real_data_produces_all_four_categories():
    """Step 3.1's real sanity check: replayed against 350 real messages
    from the real cached validation match (source=statsbomb), the phase
    classification must show genuine variation across all 4 real category
    labels -- not stuck on one -- and must transition sensibly (not on
    every single message). VERIFIED manually before writing this
    assertion: this exact real stream produces a real 22.2%/33.4%/26.5%/
    16.9% split across (Stable/Defensive Consolidation/Building Attack/
    Transition) with 332 real classified messages and 32 real phase
    transitions -- a healthy, non-degenerate distribution, not a
    near-constant or flickering-every-message one."""
    threats = _collect_real_threat_sequence(350)

    phases = []
    for i in range(MOMENTUM_MIN_MESSAGES_FOR_TREND, len(threats) + 1):
        result = classify_match_phase(threats[:i])
        assert result["status"] == "ready"
        assert result["phase"] in {"Building Attack", "Defensive Consolidation", "Transition", "Stable"}
        phases.append(result["phase"])

    assert set(phases) == {"Building Attack", "Defensive Consolidation", "Transition", "Stable"}

    # "Not flickering on every message": the number of real transitions
    # must be meaningfully smaller than the number of messages classified
    # (the hysteresis window's whole purpose) -- a same-message-to-message
    # flip rate would defeat the point of the dwell/majority-vote filter.
    transitions = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i - 1])
    assert transitions < len(phases) * 0.5


def test_classify_match_phase_raw_classification_matches_decision_table_real_data():
    """Cross-check independent of the function's own internal branching:
    for every real 'ready' reading, the RAW (pre-hysteresis) phase must
    agree with a fresh, independently-computed 2x3 decision-table lookup
    over (elevated: smoothed_now >= ELEVATED_THREAT_LEVEL) x (momentum
    classification) -- catches a flipped comparison or a swapped label
    that a same-formula assertion could not, the same discipline Tactical
    Momentum's own trend-sign cross-check applies."""
    threats = _collect_real_threat_sequence(200)

    for i in range(MOMENTUM_MIN_MESSAGES_FOR_TREND, len(threats) + 1):
        momentum = _compute_tactical_momentum(threats[:i])
        result = classify_match_phase(threats[:i])
        elevated = momentum["smoothed_now"] >= ELEVATED_THREAT_LEVEL

        if elevated and momentum["classification"] in ("Building", "Stable"):
            expected_raw = "Building Attack"
        elif not elevated and momentum["classification"] == "Fading":
            expected_raw = "Defensive Consolidation"
        elif not elevated and momentum["classification"] == "Stable":
            expected_raw = "Stable"
        else:
            expected_raw = "Transition"

        assert result["raw_phase"] == expected_raw


def test_classify_match_phase_hysteresis_suppresses_single_message_spike():
    """Step 3.2: THE real hysteresis/dwell-window test, a constructed
    synthetic noisy sequence (not real match data -- this needs a
    precisely controlled disturbance, the same reason Tactical Momentum's
    own boundary test above uses a real-data boundary slice rather than a
    fabricated one only where control, not realism, is what's actually
    under test).

    A long flat baseline (0.04, well below ELEVATED_THREAT_LEVEL) warms up
    past MOMENTUM_MIN_MESSAGES_FOR_TREND, then ONE single isolated spike
    (0.95) is injected, then baseline resumes. VERIFIED by direct trace
    before writing these assertions: the RAW (pre-hysteresis) classification
    flips the INSTANT the spike enters `smoothed_now`'s own 10-message
    window (message index 25), but the hysteresis-smoothed `phase` does
    NOT flip until index 27 -- exactly 2 messages later, matching
    SEGMENT_DWELL_MESSAGES=5's own majority-vote arithmetic (a flip
    requires 3 of the last 5 raw votes to agree; the spike only occupies
    1 of 5 lookback buffer-states at index 25, 2 of 5 at index 26, and
    reaches 3 of 5 -- a majority -- only at index 27). This is the ACTUAL
    noise-suppression guarantee: an isolated single-message disturbance
    cannot flip the displayed phase on its own.
    """
    baseline = 0.04
    spike = 0.95
    buffer = [baseline] * 25
    buffer.append(spike)
    buffer.extend([baseline] * 10)

    spike_index = 25  # 0-indexed position of the single spike in `buffer`

    # Steady baseline before the spike: phase must be "Stable" throughout.
    for i in range(MOMENTUM_MIN_MESSAGES_FOR_TREND, spike_index + 1):
        result = classify_match_phase(buffer[:i])
        assert result["status"] == "ready"
        assert result["phase"] == "Stable", f"expected Stable before the spike at index {i}"

    # The spike's own message and the immediately following one: the RAW
    # classification has already flipped, but the hysteresis-smoothed
    # `phase` must NOT flip yet -- this is the actual thing under test.
    at_spike = classify_match_phase(buffer[: spike_index + 1])
    assert at_spike["raw_phase"] != "Stable", "sanity check: the spike should visibly perturb the RAW classification"
    assert at_spike["phase"] == "Stable", "a single isolated spike must not flip the hysteresis-smoothed phase"

    one_after = classify_match_phase(buffer[: spike_index + 2])
    assert one_after["phase"] == "Stable", "hysteresis must still hold 1 message after the isolated spike"

    # Once the spike has persisted in enough of the lookback window to win
    # a genuine majority, the phase IS allowed to reflect it -- hysteresis
    # suppresses NOISE, it does not permanently ignore a real (even if
    # transient) signal once it has been consistently observed.
    majority_reached = classify_match_phase(buffer[: spike_index + 3])
    assert majority_reached["phase"] != "Stable"


def test_classify_match_phase_recovers_to_stable_after_spike_clears_window():
    """The other half of the same real trace: once the single spike has
    fully scrolled out of BOTH of Tactical Momentum's own smoothing
    windows (smoothed_now's and smoothed_then's, together spanning
    MOMENTUM_MIN_MESSAGES_FOR_TREND=20 messages), the phase must recover
    to "Stable" -- confirming the disturbance's effect is bounded and
    temporary, not a permanent corruption of the classifier's state (this
    function is PURE and stateless by design; there is no persistent state
    that COULD get stuck, but this confirms that holds in practice too)."""
    baseline = 0.04
    spike = 0.95
    buffer = [baseline] * 25 + [spike] + [baseline] * 30

    final = classify_match_phase(buffer)
    assert final["phase"] == "Stable"
    assert final["raw_phase"] == "Stable"
    assert final["smoothed_now"] == baseline


def test_player_reports_tab_low_sample_warning_renders_real_data(live_api_server):
    """Yu-Min Cho (1 real tagged event): the LOW SAMPLE warning must
    render as a real, visible st.warning element -- the exact regression
    this tab exists to guard against (Milestone 44's original finding).
    ADR-018: this tab's report data now comes over real HTTP, so it needs
    `live_api_server`; the sidebar's REST API Base URL must be pointed at
    it before the Generate button is clicked.

    Candidate-index update: the dropdown is now a real scan of data/raw/
    (candidate_index.py), so its exact label text is dynamic (event
    counts, season counts) -- matched here by the fixed "(99479)" player_id
    substring, not the full formatted string, so this test doesn't need
    updating every time the label format itself changes.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    cho_label = next(o for o in player_tab.selectbox[0].options if "(99479)" in o)
    player_tab.selectbox[0].set_value(cho_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    assert len(player_tab.warning) >= 1
    assert any("LOW SAMPLE" in w.value for w in player_tab.warning)
    # 4 images now, not 2: player report + Shot Map + the Player Dashboard's
    # own additive Touch Map + Key-Event Timeline (rendered in their own
    # section below the existing report -- see player_report.py's Player
    # Dashboard section / dashboard.py's own comment for why).
    assert len(player_tab.image) == 4


def test_player_reports_tab_well_supported_no_false_positive_warning_real_data(live_api_server):
    """Messi (well-supported): no warning should fire -- the mirror case,
    confirming the check above isn't just always-on. Selected explicitly
    by player_id substring -- the dropdown's default (first alphabetical)
    entry is no longer guaranteed to be Messi now that it lists every
    real cached player, so this can't rely on the widget's default value
    the way the old hardcoded-preset version could."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    assert len(player_tab.warning) == 0
    # 4 images now: player report + Shot Map + Touch Map + Key-Event
    # Timeline (see comment above).
    assert len(player_tab.image) == 4


# ============================================================================
# ADR-021 condition-2 compliance fix: PUBLIC_DEPLOYMENT gates the shot-map
# panel's rendered variant AND the Team Trends tab entirely. `live_api_server`
# runs api_module.app in-process (a background thread, same Python process
# as this test), so monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", ...)
# genuinely changes what that live server returns on its next request --
# combined with monkeypatch.setenv for dashboard.py's OWN copy of the flag
# (re-read fresh on every AppTest .run(), since Streamlit re-executes the
# whole script top-to-bottom on every rerun -- there is no stale-import
# concern here the way there might be for a one-shot script).
# ============================================================================


def test_player_reports_shot_map_panel_renders_aggregated_when_public_deployment_set(live_api_server, monkeypatch):
    """Both flags set consistently (the correctly-configured case): the
    shot-map panel must render the aggregated variant, and its raw-data
    JSON expander -- the ACTUAL rendered page content, not just the
    underlying function's return value -- must carry no `"shots"` field
    anywhere."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "true")
    # st.cache_data's cache is process-global, not per-AppTest-instance --
    # in REAL usage PUBLIC_DEPLOYMENT never changes mid-process (it's an
    # env var read once at startup), so this staleness never occurs
    # outside a test suite that flips the flag between runs of the same
    # (rest_base_url, player_id, match_ids) cache key, as these tests do.
    # Cleared explicitly so each test's actual HTTP round-trip reflects
    # ITS OWN flag state, not a previous test's cached response.
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    # No "Configuration error" fail-closed message -- both flags agree, so
    # the dashboard's defense-in-depth check must not trip.
    assert not any("Configuration error" in e.value for e in player_tab.error)
    # 4 images now: player report, shot map (aggregated-grid render, not
    # the raw scatter), Touch Map (aggregated), Key-Event Timeline
    # (aggregated) -- the Player Dashboard section's own two panels are
    # ALSO gated and ALSO agree with PUBLIC_DEPLOYMENT here.
    assert len(player_tab.image) == 4

    # The shot map's OWN raw-data JSON expander is the SECOND st.json call
    # on this tab (index 1: player report's own "Raw report data" comes
    # first, then shot map's, then the Player Dashboard section's match
    # summary/touch map/timeline expanders) -- parse the ACTUAL rendered
    # JSON text, not the underlying dict the endpoint returned, to confirm
    # no per-shot field reached the page.
    shot_map_json = json.loads(player_tab.json[1].value)
    assert "shots" not in shot_map_json
    assert "shot_density_grid" in shot_map_json
    assert "mean_xg_grid" in shot_map_json
    assert '"shots"' not in player_tab.json[1].value  # raw-text belt-and-suspenders, same as the API-level test


def test_player_reports_shot_map_panel_unaffected_when_public_deployment_unset(live_api_server, monkeypatch):
    """Explicit control for the default state (mirrors
    test_shot_map_endpoint_public_deployment_off_by_default_confirmed in
    test_api.py, at the dashboard-UI level): confirms the flag genuinely
    defaults to off and the raw per-shot scatter still renders, with the
    real per-shot data still present in the raw-data expander -- byte-for-
    byte the same behavior as before this fix existed."""
    monkeypatch.delenv("PUBLIC_DEPLOYMENT", raising=False)
    assert api_module.PUBLIC_DEPLOYMENT is False
    st.cache_data.clear()  # see the previous test's comment on why this is needed here

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    # 4 images now: player report, shot map (raw scatter), Touch Map
    # (raw), Key-Event Timeline (raw) -- the Player Dashboard section's
    # own two panels are unaffected by this flag too, same as shot map.
    assert len(player_tab.image) == 4
    # Shot map's own raw-data expander is the SECOND st.json call on this
    # tab now (index 1) -- see the flag-agree test above for the full
    # ordering explanation.
    shot_map_json = json.loads(player_tab.json[1].value)
    assert "shots" in shot_map_json
    assert len(shot_map_json["shots"]) == shot_map_json["total_shots"] > 0


def test_player_reports_shot_map_panel_fails_closed_on_mismatched_flags(live_api_server, monkeypatch):
    """Defense-in-depth check itself, under real test: dashboard.py's flag
    says public, but the live api.py server's flag was left off (a real,
    plausible misconfiguration -- e.g. only one of two deployed processes
    got the env var). The panel must refuse to render/display anything
    from the mismatched response rather than silently show raw per-shot
    data just because ITS OWN flag was the only one checked."""
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "true")
    # api_module.PUBLIC_DEPLOYMENT deliberately left at its real default
    # (False) here -- this IS the misconfiguration under test.
    assert api_module.PUBLIC_DEPLOYMENT is False
    st.cache_data.clear()  # see the first flag test's comment on why this is needed here

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    # 3 configuration errors now: shot map, Touch Map, and Key-Event
    # Timeline all independently fail closed under this same mismatch
    # (each runs its own defense-in-depth check -- see dashboard.py's
    # Player Dashboard section).
    config_errors = [e.value for e in player_tab.error if "Configuration error" in e.value]
    assert len(config_errors) == 3
    # Only the ORIGINAL player-report image may render -- no shot-map,
    # Touch Map, or Timeline image (raw or aggregated) anywhere.
    assert len(player_tab.image) == 1
    # 3 raw-data expanders, not 1: the player report's own, PLUS the
    # Player Dashboard's match-summary table, PLUS the Press Resistance
    # Index panel -- both of those are UNCONDITIONALLY aggregate (ADR-021:
    # never gated, see player_report.py's own Player Dashboard / Press
    # Resistance Index sections), so they render regardless of this
    # mismatch; shot map/Touch Map/Timeline each contribute zero
    # expanders here since all three failed closed.
    assert len(player_tab.json) == 3


# ============================================================================
# Player Dashboard (additive new feature, on top of the existing Player
# Reports tab): match-by-match summary table (not gated), touch map + key-
# event timeline (both ADR-021-gated, same pattern as the Shot Map panel
# above). All four tests here select Messi's EARLIEST cached season only
# (2004/2005), not his full career -- a deliberate, lighter real-data
# selection (matches this repo's own memory constraints on the night this
# feature was built; still real, non-trivial data, just smaller) rather
# than the shot-map tests' default full-career selection.
# ============================================================================


def test_player_dashboard_renders_real_data_default_local_private(live_api_server):
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    season_multiselect = player_tab.multiselect[0]
    early_season_label = next(o for o in season_multiselect.options if "2004/2005" in o)
    season_multiselect.set_value([early_season_label])
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    assert len(player_tab.error) == 0
    # 4 images: player report, shot map, touch map, timeline.
    assert len(player_tab.image) == 4
    # 2 dataframes: the match summary table, PLUS the Press Resistance
    # Index's per-event-type table (both unconditionally aggregate, see
    # ADR-021 -- neither is gated behind PUBLIC_DEPLOYMENT).
    assert len(player_tab.dataframe) == 2
    # A second selectbox (match_id for the match-level views), alongside
    # the player-selector selectbox.
    assert len(player_tab.selectbox) == 2


def test_player_dashboard_touch_map_and_timeline_render_aggregated_when_public_deployment_set(live_api_server, monkeypatch):
    """Both flags set consistently: the touch map and timeline panels
    must both render their AGGREGATED variant, with no raw per-touch
    location or raw per-event timeline anywhere in their raw-data
    expanders -- same defense-in-depth pattern as the Shot Map panel's own
    equivalent test above."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "true")
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    season_multiselect = player_tab.multiselect[0]
    early_season_label = next(o for o in season_multiselect.options if "2004/2005" in o)
    season_multiselect.set_value([early_season_label])
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    assert not any("Configuration error" in e.value for e in player_tab.error)
    assert len(player_tab.image) == 4  # still 4: aggregated variants replace raw ones 1:1, none dropped

    json_texts = [j.value for j in player_tab.json]
    assert not any('"touches"' in t for t in json_texts)
    assert not any('"timeline"' in t for t in json_texts)
    assert any('"touch_density_grid"' in t for t in json_texts)
    assert any('"buckets"' in t for t in json_texts)


def test_player_dashboard_touch_map_and_timeline_unaffected_when_public_deployment_unset(live_api_server, monkeypatch):
    """Explicit control for the default state: confirms the flag genuinely
    defaults to off and the raw touch map/timeline still render, with real
    per-touch/per-event data still present in their raw-data expanders."""
    monkeypatch.delenv("PUBLIC_DEPLOYMENT", raising=False)
    assert api_module.PUBLIC_DEPLOYMENT is False
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    season_multiselect = player_tab.multiselect[0]
    early_season_label = next(o for o in season_multiselect.options if "2004/2005" in o)
    season_multiselect.set_value([early_season_label])
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    assert len(player_tab.image) == 4
    json_texts = [j.value for j in player_tab.json]
    assert any('"touches"' in t for t in json_texts)
    assert any('"timeline"' in t for t in json_texts)


def test_player_dashboard_touch_map_and_timeline_fail_closed_on_mismatched_flags(live_api_server, monkeypatch):
    """Dashboard says public, server says private -- both the touch map
    and timeline panels must refuse to render/display anything, same
    fail-closed discipline as the Shot Map panel's own equivalent test."""
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "true")
    assert api_module.PUBLIC_DEPLOYMENT is False
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    season_multiselect = player_tab.multiselect[0]
    early_season_label = next(o for o in season_multiselect.options if "2004/2005" in o)
    season_multiselect.set_value([early_season_label])
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    # 3 configuration errors: shot map, touch map, timeline all fail closed.
    config_errors = [e.value for e in player_tab.error if "Configuration error" in e.value]
    assert len(config_errors) == 3
    # Only the original player-report image renders.
    assert len(player_tab.image) == 1


def test_team_trends_tab_disabled_when_public_deployment_set(monkeypatch):
    """The Team Trends serving-contradiction fix: with PUBLIC_DEPLOYMENT
    set, the entire tab must be replaced with an explanatory message --
    no button, no text input, nothing that could trigger
    generate_team_trend_report (which would mean a real
    football-data.co.uk network request / data/raw/ write happening from
    a public deployment)."""
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "true")

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    trends_tab = next(tab for tab in at.tabs if any("Team Trend" in h.value for h in tab.header))
    assert not at.exception
    assert len(trends_tab.button) == 0
    assert len(trends_tab.text_input) == 0
    assert any("disabled in this deployment" in i.value for i in trends_tab.info)


def test_team_trends_tab_unaffected_when_public_deployment_unset(monkeypatch):
    """Explicit control: with the flag unset (default), the tab's real
    interactive elements are still present -- unchanged from before this
    fix."""
    monkeypatch.delenv("PUBLIC_DEPLOYMENT", raising=False)

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    trends_tab = next(tab for tab in at.tabs if any("Team Trend" in h.value for h in tab.header))
    assert not at.exception
    # 2 buttons/text_inputs now, not 1: "Generate Trend Report" (year-by-
    # year view) plus the new, additive "Compare Seasons" section (Feature
    # 3) -- both live in the same non-public branch of this tab.
    assert len(trends_tab.button) == 2
    assert len(trends_tab.text_input) == 2


def test_player_reports_tab_season_subselector_produces_distinct_reports_real_data(live_api_server):
    """Candidate-index update (new coverage): Messi has 25 real cached
    competition-seasons, not one flat aggregate entry -- selecting just
    "La Liga 2004/2005" (his earliest cached season) must produce a
    genuinely different, much smaller report than selecting every cached
    season, proving the season multiselect actually scopes the request
    rather than silently always fetching the full career.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[2]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    season_multiselect = player_tab.multiselect[0]
    early_season_label = next(o for o in season_multiselect.options if "2004/2005" in o)
    season_multiselect.set_value([early_season_label])
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    early_report = json.loads(player_tab.json[0].value)
    # The full career is 596 matches/132,319 events (verified directly
    # against candidate_index.py's own scan) -- a single early season
    # must be a tiny fraction of that, not the full aggregate.
    assert early_report["matches_requested"] < 10
    assert early_report["positional_distribution_event_count"] < 500


def test_team_reports_tab_renders_real_data(live_api_server):
    """Bayer Leverkusen (well-supported under the POST-AUDIT 360-based
    metric -- 31 of its default season's 34 cached matches are actually
    360-covered, the strongest real single-season case in this cache):
    confirms the Team Reports tab -- newly HTTP-based per ADR-018 -- still
    renders its sample-size info banner and image with no LOW SAMPLE
    (quality) warning, end to end through the new endpoint. Matched by
    team-name prefix, not the full dynamic label -- see the player-tab
    tests above for why.

    NOT asserting zero warnings overall (timeout-incident fix, Step 2):
    this exact case also legitimately shows a "3 of 34 not 360-covered"
    transparency warning AND the request-cap warning (31 exceeds
    TEAM_REPORT_MAX_360_MATCHES_PER_REQUEST=25 -- see that constant's own
    comment for why the cap deliberately trims this specific real case by
    6 matches rather than being raised to avoid it) -- those are a
    DIFFERENT, correct, and expected category of informational warning,
    not a data-quality problem with this well-supported team.

    NOT Argentina (this test's team before the low-sample-definition
    fix): the verification audit found Argentina has only 7 real
    360-covered matches despite 22 cached -- genuinely low-sample under
    the corrected metric, so it now correctly triggers the warning this
    test asserts ABSENT. Using it here would make this test assert the
    wrong thing. See test_team_reports_tab_low_sample_warning_fires_for_real_low_sample_team_real_data
    for that positive case instead.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    team_tab = at.tabs[3]
    leverkusen_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Bayer Leverkusen "))
    team_tab.selectbox[0].set_value(leverkusen_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]
    team_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]

    assert not at.exception
    # 2 images, not 1: the pitch-control Team Report's own image, PLUS
    # the Passing Lane Visualizer's raw image (an additive new panel in
    # this same tab, PUBLIC_DEPLOYMENT unset by default in this test so
    # the raw, not aggregated, variant renders).
    assert len(team_tab.image) == 2
    assert any("Built from" in i.value for i in team_tab.info)
    assert not any("LOW SAMPLE" in w.value for w in team_tab.warning)


def test_team_reports_tab_low_sample_warning_fires_for_real_low_sample_team_real_data(live_api_server):
    """Mirror case, and the exact regression the post-audit fix exists to
    prevent: Real Madrid has 68 cached matches but only 2 are 360-covered
    -- confirmed directly during the verification audit. Before the fix,
    this team's dropdown label called it "well-supported" (cache-count
    based) and dashboard.py's own separate `matches_used < 2` check meant
    even the real report's low sample wouldn't have warned (2 is not
    < 2). Both are gone now: the label and the warning both come from the
    same LOW_SAMPLE_MATCH_THRESHOLD (10) applied to the real,
    360-covered match count.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    team_tab = at.tabs[3]
    real_madrid_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Real Madrid "))
    assert "LOW SAMPLE" in real_madrid_label, f"dropdown label should already flag this: {real_madrid_label!r}"
    team_tab.selectbox[0].set_value(real_madrid_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]
    team_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]

    assert not at.exception
    # 2 images, not 1: the pitch-control Team Report's own image, PLUS
    # the Passing Lane Visualizer's raw image (an additive new panel in
    # this same tab, PUBLIC_DEPLOYMENT unset by default in this test so
    # the raw, not aggregated, variant renders).
    assert len(team_tab.image) == 2
    assert len(team_tab.warning) >= 1
    assert any("LOW SAMPLE" in w.value for w in team_tab.warning)


def test_team_reports_tab_default_selection_is_single_most_recent_season_real_data():
    """Timeout-incident fix, Step 1: the season multiselect's default must
    be a single season, not every cached one -- Real Madrid has 22 cached
    seasons (was 19 pre-existing; a separate, real, exhaustive
    data_fallback.find_or_fetch_player_matches search for Cristiano
    Ronaldo, run with candidate_team_names including "Real Madrid" across
    the FULL unscoped catalog, checked every historical "Real Madrid"
    match for Ronaldo's presence -- including matches from eras before he
    was born, e.g. Copa del Rey 1982/1983 and La Liga 1973/1974 -- and
    fetch_match_events's own established caching convention caches any
    checked match's events regardless of whether the player was actually
    found in it, so those 3 old, unrelated, non-360-covered matches are
    now genuinely part of Real Madrid's own cached footprint too). Real,
    confirmed directly against the live cache before updating this
    assumption -- not guessed. Selecting all 22 (the literal reported
    failure shape) must never be what a user gets just by clicking
    through without touching the multiselect. No live_api_server needed
    -- this only checks the widget's default value, before any Generate
    click."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    team_tab = at.tabs[3]
    real_madrid_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Real Madrid "))
    team_tab.selectbox[0].set_value(real_madrid_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]

    season_multiselect = team_tab.multiselect[0]
    assert len(season_multiselect.options) == 22, "test assumption: Real Madrid has 22 cached seasons"
    assert len(season_multiselect.value) == 1, (
        f"default should be exactly one season, got {len(season_multiselect.value)}: {season_multiselect.value}"
    )


def test_team_reports_tab_warns_before_generate_click_for_wasteful_selection_real_data():
    """Timeout-incident fix, Step 2: selecting ALL of Real Madrid's 22
    seasons (71 raw matches, still only 2 360-covered -- the 3 additional
    raw matches found by the exhaustive Ronaldo search above are all
    non-360-covered old-era matches, so this split's own "2 have the 360"
    half is unchanged from before that search) must show a clear
    st.warning IMMEDIATELY, before the Generate button is ever clicked --
    not only after a slow/empty result. No live_api_server needed -- this
    checks UI state before any report request is made.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    team_tab = at.tabs[3]
    real_madrid_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Real Madrid "))
    team_tab.selectbox[0].set_value(real_madrid_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]

    season_multiselect = team_tab.multiselect[0]
    season_multiselect.set_value(season_multiselect.options)  # reproduce the exact reported failure shape
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]

    # No button click yet -- this warning must already be visible.
    assert any("71 cached match(es)" in w.value and "2 have the 360" in w.value for w in team_tab.warning), (
        f"expected a pre-generation warning naming the real 71-cached/2-360-covered split, got: "
        f"{[w.value for w in team_tab.warning]}"
    )


def test_team_reports_tab_reproduces_and_fixes_timeout_incident_real_data(live_api_server):
    """THE regression test for the reported incident itself: Real Madrid,
    all 19 cached seasons selected (68 raw matches) -- measured directly
    (outside this test suite) to take 61.90s against the real,
    UNFILTERED pipeline, timing out against this file's own
    REPORT_REQUEST_TIMEOUT_SECONDS=60.0. After the fix (candidate_index.py
    pre-filtering the request down to the 2 already-known 360-covered
    matches before it ever reaches generate_team_report), the SAME
    selection must complete correctly in well under a minute -- generous
    slack (30s) over the ~2-5s actually measured, so this test isn't
    flaky under normal system load, while still catching a real
    regression back to the old behavior (which would fail this bound by
    2x).
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    team_tab = at.tabs[3]
    real_madrid_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Real Madrid "))
    team_tab.selectbox[0].set_value(real_madrid_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]
    season_multiselect = team_tab.multiselect[0]
    season_multiselect.set_value(season_multiselect.options)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]

    start = time.monotonic()
    team_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - start
    team_tab = at.tabs[3]

    assert not at.exception
    assert elapsed < 30.0, f"took {elapsed:.1f}s -- the whole point of this fix was to make this fast, not just not-timeout"
    # 2 images, not 1: the pitch-control Team Report's own image, PLUS
    # the Passing Lane Visualizer's raw image (an additive new panel in
    # this same tab, PUBLIC_DEPLOYMENT unset by default in this test so
    # the raw, not aggregated, variant renders).
    assert len(team_tab.image) == 2
    assert any("Built from 2 matches" in i.value for i in team_tab.info), (
        "the pre-filter should have sent exactly the 2 known 360-covered matches, not all 68"
    )


def test_team_reports_tab_caps_large_valid_selection_real_data(live_api_server):
    """Timeout-incident fix, Step 3: pre-filtering alone is NOT enough --
    measured directly that PSG's full, genuinely well-supported, ALREADY
    pre-filtered 51-match request took 100.27s on real computation alone.
    Selecting all of PSG's 3 cached seasons (95 raw, 51 360-covered) must
    warn about the cap AND actually enforce it (request truncated to
    TEAM_REPORT_MAX_360_MATCHES_PER_REQUEST=25), not just warn and then
    send the full 51 anyway.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    team_tab = at.tabs[3]
    psg_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Paris Saint-Germain "))
    team_tab.selectbox[0].set_value(psg_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]
    season_multiselect = team_tab.multiselect[0]
    season_multiselect.set_value(season_multiselect.options)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]

    assert any("capped to 25" in w.value for w in team_tab.warning), (
        f"expected a pre-generation cap warning, got: {[w.value for w in team_tab.warning]}"
    )

    team_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[3]

    assert not at.exception
    # 2 images, not 1: the pitch-control Team Report's own image, PLUS
    # the Passing Lane Visualizer's raw image (an additive new panel in
    # this same tab, PUBLIC_DEPLOYMENT unset by default in this test so
    # the raw, not aggregated, variant renders).
    assert len(team_tab.image) == 2
    assert any("Built from 25 matches (of 25 requested)" in i.value for i in team_tab.info), (
        "the request actually sent must be capped to 25, not the full 51"
    )


def test_team_comparison_tab_reliability_caveat_renders_as_error_real_data(live_api_server):
    """Real Madrid 2016 vs. Barcelona 2008: the reliability caveat must
    render as a prominent st.error, not a quiet footnote -- Step 5's
    explicit requirement, now persisted as a regression test."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    compare_tab = at.tabs[5]
    compare_tab.text_input[0].set_value("Real Madrid")
    compare_tab.number_input[0].set_value(2016)
    compare_tab.text_input[1].set_value("Barcelona")
    compare_tab.number_input[1].set_value(2008)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    compare_tab = at.tabs[5]
    compare_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    compare_tab = at.tabs[5]

    assert not at.exception
    assert len(compare_tab.error) == 1
    assert "NOT equally reliable" in compare_tab.error[0].value


def test_team_trends_tab_gap_seasons_render_and_compliance_caption_visible_real_data():
    """Norwich, 2018-2025: gap seasons must render as a visible warning,
    and the football-data.co.uk compliance-scope caption must always be
    visible regardless of which team is queried."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    trends_tab = at.tabs[4]
    assert any("personal, non-distributed research" in c.value for c in trends_tab.caption)

    trends_tab.text_input[0].set_value("Norwich")
    trends_tab.number_input[0].set_value(2018)
    trends_tab.number_input[1].set_value(2025)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    trends_tab = at.tabs[4]
    trends_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    trends_tab = at.tabs[4]

    assert not at.exception
    assert len(trends_tab.warning) == 1
    assert "Gap seasons" in trends_tab.warning[0].value


def test_team_trends_tab_compare_two_seasons_real_data():
    """Feature 3, end-to-end through the real dashboard UI: Man City's
    default 2019/2025 comparison must render a real image plus the
    negative-delta clarification caption, additive alongside (not
    replacing) the existing year-by-year 'Generate Trend Report' section
    -- both sections' own widgets/results must coexist on the same tab."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    trends_tab = at.tabs[4]
    # The negative-delta clarification caption (Feature 3.5) must be
    # visible before any button is even clicked -- not buried behind a
    # result the user might never trigger.
    assert any("NEGATIVE value is a real decrease" in c.value for c in trends_tab.caption)

    # Compare Seasons uses its own, second text_input/number_inputs --
    # defaults (Man City, 2019, 2025) are already real, known football
    # history (see test_team_trend_data.py's own docstring), so clicking
    # straight through with no changes is itself a real test case.
    trends_tab.button[1].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    trends_tab = at.tabs[4]

    assert not at.exception
    assert len(trends_tab.image) == 1
    comparison_json = json.loads(trends_tab.json[-1].value)
    assert comparison_json["season_a_found"] is True
    assert comparison_json["season_b_found"] is True
    assert comparison_json["diff_b_minus_a"]["points_delta"] == -3
    assert "NEGATIVE" in comparison_json["diff_convention"]

    # The original year-by-year section's own widgets must still be
    # present and untouched -- additive, not a replacement.
    assert len(trends_tab.button) == 2
    assert len(trends_tab.text_input) == 2


# ============================================================================
# Pass Network tab: GET /reports/pass-network/{match_id} via api.py, same
# real-HTTP-through-live_api_server discipline as the other reporting tabs
# (ADR-018). ADR-021 condition-2 compliance tests mirror the Shot Map
# panel's own PUBLIC_DEPLOYMENT flag tests exactly (see those above) --
# same monkeypatch pattern, applied to this new tab instead of reinvented.
# ============================================================================

PASS_NETWORK_MATCH_ID = "3857276"  # Canada vs. Morocco -- 22 real Starting XI players, 171 real edges


def test_pass_network_tab_renders_real_data_default_local_private(live_api_server):
    """Flag unset (the default): the real per-player pitch-network image
    renders, and the raw-data expander carries real node/edge data."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    pn_tab = at.tabs[6]
    pn_tab.text_input[0].set_value(PASS_NETWORK_MATCH_ID)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]
    pn_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]

    assert not at.exception
    assert len(pn_tab.error) == 0
    assert len(pn_tab.image) == 1
    raw = json.loads(pn_tab.json[0].value)
    assert len(raw["nodes"]) == 22
    assert len(raw["edges"]) > 0


def test_pass_network_tab_renders_aggregated_when_public_deployment_set(live_api_server, monkeypatch):
    """Both flags set consistently (the correctly-configured public
    case): the aggregated variant must render, and its raw-data expander
    must carry no `nodes`/`edges`/`avg_location` anywhere."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "true")
    st.cache_data.clear()  # see the shot-map flag tests above for why this matters across real HTTP reruns

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    pn_tab = at.tabs[6]
    pn_tab.text_input[0].set_value(PASS_NETWORK_MATCH_ID)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]
    pn_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]

    assert not at.exception
    assert not any("Configuration error" in e.value for e in pn_tab.error)
    assert len(pn_tab.image) == 1
    raw_text = pn_tab.json[0].value
    assert '"nodes"' not in raw_text
    assert '"edges"' not in raw_text
    assert "avg_location" not in raw_text
    assert '"player_summary"' in raw_text


def test_pass_network_tab_unaffected_when_public_deployment_unset(live_api_server, monkeypatch):
    """Explicit control for the default state: confirms the flag genuinely
    defaults to off and the raw network still renders with real per-player
    location data still present in the raw expander."""
    monkeypatch.delenv("PUBLIC_DEPLOYMENT", raising=False)
    assert api_module.PUBLIC_DEPLOYMENT is False
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    pn_tab = at.tabs[6]
    pn_tab.text_input[0].set_value(PASS_NETWORK_MATCH_ID)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]
    pn_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]

    assert not at.exception
    assert len(pn_tab.image) == 1
    raw = json.loads(pn_tab.json[0].value)
    assert "nodes" in raw
    assert "avg_location" in raw["nodes"][0]


def test_pass_network_tab_fails_closed_on_mismatched_flags(live_api_server, monkeypatch):
    """Defense-in-depth check itself, under real test: dashboard.py's flag
    says public, but the live api.py server's flag was left off. The
    panel must refuse to render/display anything from the mismatched
    response rather than silently show raw network data."""
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "true")
    # api_module.PUBLIC_DEPLOYMENT deliberately left at its real default
    # (False) here -- this IS the misconfiguration under test.
    assert api_module.PUBLIC_DEPLOYMENT is False
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    pn_tab = at.tabs[6]
    pn_tab.text_input[0].set_value(PASS_NETWORK_MATCH_ID)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]
    pn_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]

    assert not at.exception
    assert any("Configuration error" in e.value for e in pn_tab.error)
    assert len(pn_tab.image) == 0
    assert len(pn_tab.json) == 0


# ============================================================================
# Alerts History tab (ADR-019's persistence store, surfaced for the first
# time): GET /alerts/history via api.py, same real-HTTP-through-live_api_server
# discipline as the Player/Team Reports tabs above (ADR-018).
#
# `_isolated_alerts_db` mirrors test_alert_store.py's own `_isolated_db`
# fixture exactly (monkeypatching alert_store's module-level DB_DIR/DB_PATH
# into pytest's tmp_path) so these tests never read or write the real
# data/app_state/alerts.db -- `live_api_server` runs api.py's real FastAPI
# app in a background thread of THIS SAME test process, so the monkeypatch
# is visible to it too: fetch_alerts/log_alert look up DB_PATH via their
# own defining module's globals when they run, not a copy frozen at import
# time, regardless of how api.py imported them.
# ============================================================================


@pytest.fixture
def _isolated_alerts_db(tmp_path, monkeypatch):
    monkeypatch.setattr(alert_store_module, "DB_DIR", tmp_path / "app_state")
    monkeypatch.setattr(alert_store_module, "DB_PATH", tmp_path / "app_state" / "alerts.db")


def test_alerts_history_tab_renders_and_filters_by_match_id_real_data(live_api_server, _isolated_alerts_db):
    """Two real alerts logged against match_id=111, one against a
    different match (222) -- filtering by 111 must return exactly the two
    matching rows, in a real, readable table (not a raw JSON dump), and
    the raw-data expander below it must reflect that SAME filtered set."""
    alert_store_module.log_alert(
        source="statsbomb", match_id=111, video_path=None, minute=5.0,
        threat_before=0.10, threat_after=0.30, explanation_text="alert A", explanation_source="mock",
    )
    alert_store_module.log_alert(
        source="statsbomb", match_id=111, video_path=None, minute=10.0,
        threat_before=0.20, threat_after=0.60, explanation_text="alert B", explanation_source="mock",
    )
    alert_store_module.log_alert(
        source="cv", match_id=222, video_path="data/raw/other.mp4", minute=3.0,
        threat_before=0.05, threat_after=0.40, explanation_text="alert C (different match)", explanation_source="gemini",
    )
    st.cache_data.clear()  # see the PUBLIC_DEPLOYMENT tests above for why this matters across real HTTP reruns

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    alerts_tab = at.tabs[7]
    alerts_tab.text_input[0].set_value("111")
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[7]
    alerts_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[7]

    assert not at.exception
    assert len(alerts_tab.dataframe) == 1
    df = alerts_tab.dataframe[0].value
    assert df.shape[0] == 2
    assert set(df["Match ID"]) == {111}
    assert set(df["Explanation"]) == {"alert A", "alert B"}

    raw = json.loads(alerts_tab.json[0].value)
    assert len(raw) == 2
    assert all(r["match_id"] == 111 for r in raw)


def test_alerts_history_tab_empty_result_shows_info_message_not_broken_table(live_api_server, _isolated_alerts_db):
    """An empty, isolated alerts db -- no rows logged at all -- must render
    a clean 'no alerts found' message, not an empty/broken dataframe."""
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    alerts_tab = at.tabs[7]
    alerts_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[7]

    assert not at.exception
    assert len(alerts_tab.dataframe) == 0
    assert any("No alerts found" in i.value for i in alerts_tab.info)


def test_alerts_history_tab_source_filter_genuinely_filters_real_data(live_api_server, _isolated_alerts_db):
    """One statsbomb alert, one cv alert -- filtering by source=cv must
    return only the cv one, confirming the filter actually filters rather
    than silently ignoring the selectbox and returning everything."""
    alert_store_module.log_alert(
        source="statsbomb", match_id=1, video_path=None, minute=1.0,
        threat_before=0.10, threat_after=0.20, explanation_text="sb alert", explanation_source="mock",
    )
    alert_store_module.log_alert(
        source="cv", match_id=None, video_path="data/raw/clip.mp4", minute=2.0,
        threat_before=0.10, threat_after=0.50, explanation_text="cv alert", explanation_source="gemini",
    )
    st.cache_data.clear()

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    alerts_tab = at.tabs[7]
    alerts_tab.selectbox[0].set_value("cv")
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[7]
    alerts_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[7]

    assert not at.exception
    assert len(alerts_tab.dataframe) == 1
    df = alerts_tab.dataframe[0].value
    assert df.shape[0] == 1
    assert df["Source"].tolist() == ["cv"]
    assert df["Explanation"].tolist() == ["cv alert"]


def test_alerts_history_tab_invalid_match_id_shows_clean_error_no_crash():
    """A non-numeric Match ID must be validated BEFORE any request is
    made -- a clean st.error, no unhandled exception, and no dataframe.
    No live_api_server needed: this validation happens client-side, before
    any HTTP call, the same 'no live server needed for pre-request
    validation' convention the Team Reports tab's own pre-generation
    warning tests above already establish."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    alerts_tab = at.tabs[7]
    alerts_tab.text_input[0].set_value("not-a-number")
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[7]
    alerts_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[7]

    assert not at.exception
    assert any("must be a whole number" in e.value for e in alerts_tab.error)
    assert len(alerts_tab.dataframe) == 0


# ============================================================================
# UX polish pass: Part A (cross-linking), Part B (export/share), Part C
# (consistent loading/error states).
#
# `_capture_media_managers`: `st.download_button`'s actual file content
# isn't reachable from AppTest's own public API (its `DownloadButton`
# wrapper exposes only a `value: bool`, and `Runtime.instance()` is torn
# down -- set back to `None` -- at the end of every single `AppTest.run()`
# call, per that method's own source). Monkeypatching `MediaFileManager.
# __init__` to record every instance as AppTest constructs a fresh one on
# each `run()` is the only way found (by directly reading AppTest's own
# source, not guessed) to reach the REAL bytes a real download would
# serve, so these export tests can assert on genuine file content rather
# than "the button rendered."
# ============================================================================


@pytest.fixture
def _capture_media_managers(monkeypatch):
    managers: list[MediaFileManager] = []
    real_init = MediaFileManager.__init__

    def _capturing_init(self, *a, **kw):
        real_init(self, *a, **kw)
        managers.append(self)

    monkeypatch.setattr(MediaFileManager, "__init__", _capturing_init)
    return managers


def _get_download_button_content(managers: list[MediaFileManager], download_button) -> bytes:
    file_id = download_button.proto.url.split("/")[-1].rsplit(".", 1)[0]
    for manager in reversed(managers):
        try:
            return manager._storage._files_by_id[file_id].content
        except KeyError:
            continue
    raise AssertionError(f"download button file_id {file_id!r} not found in any captured MediaFileManager")


def test_pass_network_to_player_report_cross_link_prefills_correctly(live_api_server):
    """Part A, real case 1. A real navigation problem (Step 0): a player's
    name in the Pass Network tab has no path to their own Player Report --
    confirms the "View full report" button genuinely pre-fills the target
    tab's Custom fields, not just that it renders. This is also the
    regression test for two real bugs found and fixed while building this
    (see `dashboard.py`'s own module comments on `_render_cross_link_button`):
    (1) a cross-link button nested inside a transient `if ..._clicked:`
    block silently orphans its own click (fixed by persisting the fetched
    result in `st.session_state` and rendering unconditionally); (2)
    `st.rerun()` does NOT bypass Streamlit's "cannot set a widget's
    session_state after it's instantiated this run" rule (fixed by
    applying prefills at the top of the script, before `st.tabs()`, not
    inside the button's own click handler)."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    pn_tab = at.tabs[6]
    pn_tab.text_input[0].set_value(str(TACTICAL_STREAM_MATCH_ID))
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]
    pn_tab.button[0].click()  # "Generate Pass Network"
    at.run(timeout=APP_TIMEOUT_SECONDS)
    pn_tab = at.tabs[6]

    assert not at.exception
    assert len(pn_tab.button) > 1, "expected real per-player cross-link buttons to have rendered"
    clicked_label = pn_tab.button[1].label
    pn_tab.button[1].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)

    assert not at.exception
    assert at.session_state["player_report_preset_selectbox"] == "Custom"
    clicked_player_id = at.session_state["player_report_custom_player_id"]
    assert clicked_player_id.isdigit()
    assert clicked_label.endswith(" →")  # the button's own label names a player, not their numeric id
    assert at.session_state["player_report_custom_match_ids"] == str(TACTICAL_STREAM_MATCH_ID)

    player_tab = at.tabs[2]
    assert player_tab.selectbox[0].value == "Custom"
    text_input_values = {ti.label: ti.value for ti in player_tab.text_input}
    assert text_input_values["Player ID (StatsBomb player_id)"] == clicked_player_id
    assert text_input_values["Match IDs (comma-separated StatsBomb match_id list)"] == str(TACTICAL_STREAM_MATCH_ID)


def test_match_report_to_team_report_cross_link_prefills_correctly(live_api_server):
    """Part A, real case 2. A real navigation problem (Step 0): a team
    name in a compiled Match Report has no path to that team's own full
    Team Report."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    mr_tab = at.tabs[8]
    mr_tab.text_input[0].set_value(str(TACTICAL_STREAM_MATCH_ID))
    at.run(timeout=APP_TIMEOUT_SECONDS)
    mr_tab = at.tabs[8]
    mr_tab.button[0].click()  # "Generate Match Report"
    at.run(timeout=APP_TIMEOUT_SECONDS)
    mr_tab = at.tabs[8]

    assert not at.exception
    cross_link_buttons = [b for b in mr_tab.button if b.label.startswith("View full")]
    assert len(cross_link_buttons) == 2, "expected one cross-link button per team in this real 2-team match"
    target_button = cross_link_buttons[0]
    target_team = target_button.label.removeprefix("View full ").removesuffix(" Team Report →")
    target_button.click()
    at.run(timeout=APP_TIMEOUT_SECONDS)

    assert not at.exception
    assert at.session_state["team_report_preset_selectbox"] == "Custom"
    assert at.session_state["team_report_name"] == target_team
    assert at.session_state["team_report_match_ids"] == str(TACTICAL_STREAM_MATCH_ID)

    team_tab = at.tabs[3]
    assert team_tab.selectbox[0].value == "Custom"
    text_input_values = {ti.label: ti.value for ti in team_tab.text_input}
    assert text_input_values["Team name (StatsBomb team name)"] == target_team
    assert text_input_values["Match IDs (comma-separated StatsBomb match_id list)"] == str(TACTICAL_STREAM_MATCH_ID)


def test_match_report_export_produces_self_contained_html(live_api_server, _capture_media_managers):
    """Part B, real case 1: the exported Match Report HTML is a genuine,
    complete, standalone document -- opens with no live server, no
    external references."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    mr_tab = at.tabs[8]
    mr_tab.text_input[0].set_value(str(TACTICAL_STREAM_MATCH_ID))
    at.run(timeout=APP_TIMEOUT_SECONDS)
    mr_tab = at.tabs[8]
    mr_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    mr_tab = at.tabs[8]

    assert not at.exception
    assert len(mr_tab.download_button) == 1
    content = _get_download_button_content(_capture_media_managers, mr_tab.download_button[0])
    html_text = content.decode("utf-8")

    assert html_text.strip().lower().startswith("<!doctype html>")
    assert html_text.strip().lower().endswith("</html>")
    assert f"match_id={TACTICAL_STREAM_MATCH_ID}" in html_text
    # Genuinely self-contained: no reference to a live server or the
    # public internet anywhere in the document.
    assert "http://" not in html_text
    assert "https://" not in html_text


def test_player_report_export_embeds_real_images_no_external_refs(live_api_server, _capture_media_managers):
    """Part B, real case 2: the exported Player Report HTML embeds its
    real rendered images as base64 `data:` URIs, not links to external
    files -- confirmed by parsing the actual `<img>` tags, not just
    substring-checking the raw text."""
    from html.parser import HTMLParser

    class _ImgCollector(HTMLParser):
        def __init__(self):
            super().__init__()
            self.img_srcs: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag == "img":
                self.img_srcs.append(dict(attrs).get("src", ""))

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    player_tab = at.tabs[2]
    player_tab.button[0].click()  # "Generate Player Report" (default preset)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[2]

    assert not at.exception
    assert len(player_tab.download_button) == 1
    content = _get_download_button_content(_capture_media_managers, player_tab.download_button[0])
    html_text = content.decode("utf-8")

    parser = _ImgCollector()
    parser.feed(html_text)
    assert len(parser.img_srcs) >= 1, "expected at least the main report image embedded"
    for src in parser.img_srcs:
        assert src.startswith("data:image/png;base64,"), f"image src is not a self-contained data URI: {src[:60]!r}"
        assert len(src) > 1000, "embedded image data URI looks suspiciously small/empty"
    assert "http://" not in html_text
    assert "https://" not in html_text


def test_team_trends_network_failure_shows_clean_error_not_crash(monkeypatch, tmp_path):
    """Part C, real regression test: the Step 0 audit's own finding --
    Team Trends was the ONE tab in this dashboard with ZERO exception
    handling around a call that makes a real network request
    (`team_trend_data.py`'s own internal `requests.get()` against
    football-data.co.uk) -- confirmed here with a genuinely triggered
    failure (not reasoned about), by making that underlying call raise a
    real `requests.exceptions.ConnectionError` and confirming the tab
    shows a clean, consistent `st.error` instead of an unhandled
    traceback."""
    import production.src.reporting.team_trend_data as team_trend_data_module

    def _raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("simulated: football-data.co.uk unreachable")

    monkeypatch.setattr(team_trend_data_module.requests, "get", _raise_connection_error)
    # This project's own real local disk cache (data/raw/football_data_co_uk/)
    # very plausibly already has every league/season CSV this default query
    # range would touch, from ordinary prior use of this feature -- which
    # would make `_fetch_season_csv` take its `cache_path.exists()` branch
    # and never call `requests.get` at all, silently not exercising the
    # mock above. Redirected to an empty `tmp_path` so every single
    # league/season fetch is a genuine cache MISS, forcing the real
    # network path (and therefore the mock) to actually run.
    monkeypatch.setattr(team_trend_data_module, "CACHE_DIR", tmp_path)
    # PUBLIC_DEPLOYMENT is unset in this test process (never set anywhere
    # in production/tests/, per ADR-021) -- Team Trends is not hidden.

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    trends_tab = at.tabs[4]
    trends_tab.text_input[0].set_value("A Team With No Local Cache Whatsoever")
    at.run(timeout=APP_TIMEOUT_SECONDS)
    trends_tab = at.tabs[4]
    trends_tab.button[0].click()  # "Generate Trend Report"
    at.run(timeout=APP_TIMEOUT_SECONDS)
    trends_tab = at.tabs[4]

    assert not at.exception, "a real network failure must render a clean st.error, not crash the script"
    assert any("football-data.co.uk" in e.value for e in trends_tab.error)


def test_error_message_template_consistent_across_report_pattern_categories(monkeypatch):
    """Part C, real case: the SAME error-message template now renders for
    a connection failure across all three report-generation pattern
    categories this task named -- live-stream (Live CV Monitor's What-If
    Simulator), single-entity (Player Reports), and comparison (Team
    Comparison) -- confirming the Step 0 audit's fix (extracting one
    shared `_fetch_report_safely` error path) actually reached all three,
    not just the ~30 call sites that already used it before this pass."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value("http://127.0.0.1:1")  # unreachable, fast refusal
    at.run(timeout=APP_TIMEOUT_SECONDS)

    expected_fragment = "Backend unreachable at http://127.0.0.1:1"

    cv_tab = at.tabs[1]
    cv_tab.button[0].click()  # "Run Simulation"
    at.run(timeout=30)
    cv_tab = at.tabs[1]
    assert not at.exception
    assert any(expected_fragment in e.value for e in cv_tab.error)

    player_tab = at.tabs[2]
    player_tab.button[0].click()  # "Generate Player Report"
    at.run(timeout=30)
    player_tab = at.tabs[2]
    assert not at.exception
    assert any(expected_fragment in e.value for e in player_tab.error)

    compare_tab = at.tabs[5]
    compare_tab.button[0].click()  # "Compare"
    at.run(timeout=30)
    compare_tab = at.tabs[5]
    assert not at.exception
    assert any(expected_fragment in e.value for e in compare_tab.error)
