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

import threading
import time

import pytest
import requests
import uvicorn
from streamlit.testing.v1 import AppTest

from production.src.serving.api import app as _fastapi_app

DASHBOARD_PATH = "production/frontend/dashboard.py"
APP_TIMEOUT_SECONDS = 180


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


def test_dashboard_loads_all_five_tabs_no_exception():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    assert not at.exception
    assert len(at.tabs) == 5
    headers = [h.value for tab in at.tabs for h in tab.header]
    assert "Player Report" in headers
    assert "Team Report" in headers
    assert "Team Trend Report (football-data.co.uk)" in headers
    assert "Team-Season Style Comparison" in headers


def test_player_reports_tab_low_sample_warning_renders_real_data(live_api_server):
    """Yu-Min Cho (1 real tagged event): the LOW SAMPLE warning must
    render as a real, visible st.warning element -- the exact regression
    this tab exists to guard against (Milestone 44's original finding).
    ADR-018: this tab's report data now comes over real HTTP, so it needs
    `live_api_server`; the sidebar's REST API Base URL must be pointed at
    it before the Generate button is clicked."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[1]
    player_tab.selectbox[0].set_value("Yu-Min Cho (99479) -- LOW SAMPLE, 1 event")
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

    assert not at.exception
    assert len(player_tab.warning) >= 1
    assert any("LOW SAMPLE" in w.value for w in player_tab.warning)
    assert len(player_tab.image) == 1


def test_player_reports_tab_well_supported_no_false_positive_warning_real_data(live_api_server):
    """Messi (well-supported): no warning should fire -- the mirror case,
    confirming the check above isn't just always-on."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    player_tab = at.tabs[1]
    player_tab.button[0].click()  # default selectbox value is the Messi preset
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

    assert not at.exception
    assert len(player_tab.warning) == 0
    assert len(player_tab.image) == 1


def test_team_reports_tab_renders_real_data(live_api_server):
    """Argentina (well-supported, >=2 matches): confirms the Team Reports
    tab -- newly HTTP-based per ADR-018 -- still renders its sample-size
    info banner and image with no low-sample warning, end to end through
    the new endpoint."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    team_tab = at.tabs[2]
    team_tab.selectbox[0].set_value("Argentina")
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]
    team_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

    assert not at.exception
    assert len(team_tab.image) == 1
    assert any("Built from" in i.value for i in team_tab.info)
    assert len(team_tab.warning) == 0


def test_team_comparison_tab_reliability_caveat_renders_as_error_real_data(live_api_server):
    """Real Madrid 2016 vs. Barcelona 2008: the reliability caveat must
    render as a prominent st.error, not a quiet footnote -- Step 5's
    explicit requirement, now persisted as a regression test."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    at.sidebar.text_input[0].set_value(live_api_server)

    compare_tab = at.tabs[4]
    compare_tab.text_input[0].set_value("Real Madrid")
    compare_tab.number_input[0].set_value(2016)
    compare_tab.text_input[1].set_value("Barcelona")
    compare_tab.number_input[1].set_value(2008)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    compare_tab = at.tabs[4]
    compare_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    compare_tab = at.tabs[4]

    assert not at.exception
    assert len(compare_tab.error) == 1
    assert "NOT equally reliable" in compare_tab.error[0].value


def test_team_trends_tab_gap_seasons_render_and_compliance_caption_visible_real_data():
    """Norwich, 2018-2025: gap seasons must render as a visible warning,
    and the football-data.co.uk compliance-scope caption must always be
    visible regardless of which team is queried."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    trends_tab = at.tabs[3]
    assert any("personal, non-distributed research" in c.value for c in trends_tab.caption)

    trends_tab.text_input[0].set_value("Norwich")
    trends_tab.number_input[0].set_value(2018)
    trends_tab.number_input[1].set_value(2025)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    trends_tab = at.tabs[3]
    trends_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    trends_tab = at.tabs[3]

    assert not at.exception
    assert len(trends_tab.warning) == 1
    assert "Gap seasons" in trends_tab.warning[0].value
