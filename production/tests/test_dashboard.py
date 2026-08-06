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
from streamlit.testing.v1 import AppTest

import production.src.serving.api as api_module
from production.src.serving import alert_store as alert_store_module
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


def test_dashboard_loads_all_six_tabs_no_exception():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    assert not at.exception
    assert len(at.tabs) == 6
    headers = [h.value for tab in at.tabs for h in tab.header]
    assert "Player Report" in headers
    assert "Team Report" in headers
    assert "Team Trend Report (football-data.co.uk)" in headers
    assert "Team-Season Style Comparison" in headers
    assert "Alerts History" in headers


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

    player_tab = at.tabs[1]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

    assert not at.exception
    # No "Configuration error" fail-closed message -- both flags agree, so
    # the dashboard's defense-in-depth check must not trip.
    assert not any("Configuration error" in e.value for e in player_tab.error)
    # Still exactly 2 images: the player-report image, plus ONE shot-map
    # image (the aggregated-grid render, not the raw scatter).
    assert len(player_tab.image) == 2

    # The shot map's OWN raw-data JSON expander (rendered after the player
    # report's own "Raw report data" expander) is the last st.json call on
    # this tab -- parse the ACTUAL rendered JSON text, not the underlying
    # dict the endpoint returned, to confirm no per-shot field reached the
    # page.
    shot_map_json = json.loads(player_tab.json[-1].value)
    assert "shots" not in shot_map_json
    assert "shot_density_grid" in shot_map_json
    assert "mean_xg_grid" in shot_map_json
    assert '"shots"' not in player_tab.json[-1].value  # raw-text belt-and-suspenders, same as the API-level test


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

    player_tab = at.tabs[1]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

    assert not at.exception
    assert len(player_tab.image) == 2
    shot_map_json = json.loads(player_tab.json[-1].value)
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

    player_tab = at.tabs[1]
    messi_label = next(o for o in player_tab.selectbox[0].options if "(5503)" in o)
    player_tab.selectbox[0].set_value(messi_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]
    player_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

    assert not at.exception
    assert any("Configuration error" in e.value for e in player_tab.error)
    # Only the ORIGINAL player-report image may render -- no shot-map
    # image (raw or aggregated) and no raw-data expander for it at all.
    assert len(player_tab.image) == 1
    assert len(player_tab.json) == 1  # only the player report's own expander, not a second one for the shot map


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

    team_tab = at.tabs[2]
    real_madrid_label = next(o for o in team_tab.selectbox[0].options if o.startswith("Real Madrid "))
    team_tab.selectbox[0].set_value(real_madrid_label)
    at.run(timeout=APP_TIMEOUT_SECONDS)
    team_tab = at.tabs[2]

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


def test_team_trends_tab_compare_two_seasons_real_data():
    """Feature 3, end-to-end through the real dashboard UI: Man City's
    default 2019/2025 comparison must render a real image plus the
    negative-delta clarification caption, additive alongside (not
    replacing) the existing year-by-year 'Generate Trend Report' section
    -- both sections' own widgets/results must coexist on the same tab."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    trends_tab = at.tabs[3]
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
    trends_tab = at.tabs[3]

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

    alerts_tab = at.tabs[5]
    alerts_tab.text_input[0].set_value("111")
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[5]
    alerts_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[5]

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

    alerts_tab = at.tabs[5]
    alerts_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[5]

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

    alerts_tab = at.tabs[5]
    alerts_tab.selectbox[0].set_value("cv")
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[5]
    alerts_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[5]

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

    alerts_tab = at.tabs[5]
    alerts_tab.text_input[0].set_value("not-a-number")
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[5]
    alerts_tab.button[0].click()
    at.run(timeout=APP_TIMEOUT_SECONDS)
    alerts_tab = at.tabs[5]

    assert not at.exception
    assert any("must be a whole number" in e.value for e in alerts_tab.error)
    assert len(alerts_tab.dataframe) == 0
