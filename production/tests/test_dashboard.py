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

    player_tab = at.tabs[1]
    cho_label = next(o for o in player_tab.selectbox[0].options if "(99479)" in o)
    player_tab.selectbox[0].set_value(cho_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

    assert not at.exception
    assert len(player_tab.warning) >= 1
    assert any("LOW SAMPLE" in w.value for w in player_tab.warning)
    # 2 images now, not 1: the original player-report dashboard image plus
    # the new, additive Shot Map image (rendered in its own section below
    # the existing report -- see generate_player_shot_map/render_shot_map).
    assert len(player_tab.image) == 2


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

    player_tab = at.tabs[1]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

    assert not at.exception
    assert len(player_tab.warning) == 0
    # 2 images now, not 1: the original player-report dashboard image plus
    # the new, additive Shot Map image (see comment above).
    assert len(player_tab.image) == 2


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

    player_tab = at.tabs[1]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

    season_multiselect = player_tab.multiselect[0]
    early_season_label = next(o for o in season_multiselect.options if "2004/2005" in o)
    season_multiselect.set_value([early_season_label])
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

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

    team_tab = at.tabs[2]
    leverkusen_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Bayer Leverkusen "))
    team_tab.selectbox[0].set_value(leverkusen_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]
    team_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

    assert not at.exception
    assert len(team_tab.image) == 1
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

    team_tab = at.tabs[2]
    real_madrid_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Real Madrid "))
    assert "LOW SAMPLE" in real_madrid_label, f"dropdown label should already flag this: {real_madrid_label!r}"
    team_tab.selectbox[0].set_value(real_madrid_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]
    team_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

    assert not at.exception
    assert len(team_tab.image) == 1
    assert len(team_tab.warning) >= 1
    assert any("LOW SAMPLE" in w.value for w in team_tab.warning)


def test_team_reports_tab_default_selection_is_single_most_recent_season_real_data():
    """Timeout-incident fix, Step 1: the season multiselect's default must
    be a single season, not every cached one -- Real Madrid has 19 cached
    seasons; selecting all of them (the literal reported failure shape)
    must never be what a user gets just by clicking through without
    touching the multiselect. No live_api_server needed -- this only
    checks the widget's default value, before any Generate click."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    team_tab = at.tabs[2]
    real_madrid_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Real Madrid "))
    team_tab.selectbox[0].set_value(real_madrid_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

    season_multiselect = team_tab.multiselect[0]
    assert len(season_multiselect.options) == 19, "test assumption: Real Madrid has 19 cached seasons"
    assert len(season_multiselect.value) == 1, (
        f"default should be exactly one season, got {len(season_multiselect.value)}: {season_multiselect.value}"
    )


def test_team_reports_tab_warns_before_generate_click_for_wasteful_selection_real_data():
    """Timeout-incident fix, Step 2: selecting ALL of Real Madrid's 19
    seasons (68 raw matches, only 2 360-covered -- confirmed during the
    audit) must show a clear st.warning IMMEDIATELY, before the Generate
    button is ever clicked -- not only after a slow/empty result. No
    live_api_server needed -- this checks UI state before any report
    request is made.
    """
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    team_tab = at.tabs[2]
    real_madrid_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Real Madrid "))
    team_tab.selectbox[0].set_value(real_madrid_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

    season_multiselect = team_tab.multiselect[0]
    season_multiselect.set_value(season_multiselect.options)  # reproduce the exact reported failure shape
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

    # No button click yet -- this warning must already be visible.
    assert any("68 cached match(es)" in w.value and "2 have the 360" in w.value for w in team_tab.warning), (
        f"expected a pre-generation warning naming the real 68-cached/2-360-covered split, got: "
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

    team_tab = at.tabs[2]
    real_madrid_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Real Madrid "))
    team_tab.selectbox[0].set_value(real_madrid_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]
    season_multiselect = team_tab.multiselect[0]
    season_multiselect.set_value(season_multiselect.options)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

    start = time.monotonic()
    team_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - start
    team_tab = at.tabs[2]

    assert not at.exception
    assert elapsed < 30.0, f"took {elapsed:.1f}s -- the whole point of this fix was to make this fast, not just not-timeout"
    assert len(team_tab.image) == 1
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

    team_tab = at.tabs[2]
    psg_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Paris Saint-Germain "))
    team_tab.selectbox[0].set_value(psg_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]
    season_multiselect = team_tab.multiselect[0]
    season_multiselect.set_value(season_multiselect.options)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

    assert any("capped to 25" in w.value for w in team_tab.warning), (
        f"expected a pre-generation cap warning, got: {[w.value for w in team_tab.warning]}"
    )

    team_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

    assert not at.exception
    assert len(team_tab.image) == 1
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
