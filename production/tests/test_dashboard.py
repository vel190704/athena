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
"""

from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = "production/frontend/dashboard.py"
APP_TIMEOUT_SECONDS = 180


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


def test_player_reports_tab_low_sample_warning_renders_real_data():
    """Yu-Min Cho (1 real tagged event): the LOW SAMPLE warning must
    render as a real, visible st.warning element -- the exact regression
    this tab exists to guard against (Milestone 44's original finding)."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

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


def test_player_reports_tab_well_supported_no_false_positive_warning_real_data():
    """Messi (well-supported): no warning should fire -- the mirror case,
    confirming the check above isn't just always-on."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

    player_tab = at.tabs[1]
    player_tab.button[0].click()  # default selectbox value is the Messi preset
    at.run(timeout=APP_TIMEOUT_SECONDS)
    player_tab = at.tabs[1]

    assert not at.exception
    assert len(player_tab.warning) == 0
    assert len(player_tab.image) == 1


def test_team_comparison_tab_reliability_caveat_renders_as_error_real_data():
    """Real Madrid 2016 vs. Barcelona 2008: the reliability caveat must
    render as a prominent st.error, not a quiet footnote -- Step 5's
    explicit requirement, now persisted as a regression test."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=APP_TIMEOUT_SECONDS)

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
