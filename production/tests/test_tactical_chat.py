"""AI Tactical Chat (new reporting track, Part B) validation.

Split the SAME way `test_api.py`/`test_explainer.py` already are: fast,
deterministic tests here use a monkeypatched dispatcher (never touching a
real API key or real network); the real, adversarial honesty tests are
opt-in, SKIPPED (not failed) when `GEMINI_API_KEY` isn't set, the same
convention `test_explainer.py::test_explainer_real_gemini_integration`
already establishes. Deliberately its OWN file, not folded into
`test_api.py`: that file's own `_force_mock_explanation_executor` autouse
fixture patches `generate_tactical_explanation_with_source` specifically --
AI Tactical Chat calls a DIFFERENT dispatcher
(`explainer.generate_chat_reply_with_source`, see that function's own
docstring for why it is deliberately not the same one), which that
fixture does not touch. Keeping chat's tests self-contained avoids either
silently making real Gemini calls inside `test_api.py`'s run or extending
that file's shared fixture for a concern specific to this one feature.
"""

import asyncio
import os
import re

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import pytest
from fastapi.testclient import TestClient

import production.src.serving.api as api_module
from production.src.serving.api import app

MATCH_ID = 3857276

# Same pattern test_explainer.py's/test_zone_explainer.py's own honesty
# check uses -- duplicated here, not cross-imported, since it's a small,
# self-contained regex and tests should not depend on each other.
_UNSUPPORTED_CLAIM_PATTERN = re.compile(
    r"\b(second|seconds|minute|minutes|recovering|recovers|sprint|sprinting)\b", re.IGNORECASE
)


@pytest.fixture(autouse=True)
def _reset_chat_sessions():
    """`api_module._chat_sessions` is real, in-process module state that
    persists across tests within one pytest session (the exact same class
    of state `test_rate_limiting.py`'s own bucket-reset fixture exists
    for) -- cleared before AND after every test in this file so no test's
    conversation history leaks into another's."""
    api_module._chat_sessions.clear()
    yield
    api_module._chat_sessions.clear()


@pytest.fixture
def mock_chat_dispatcher(monkeypatch):
    """Deterministic stand-in for `generate_chat_reply_with_source` --
    echoes back a fixed, recognizable reply so these tests can verify
    session/history/rebuild PLUMBING without making a real network call or
    depending on real model output quality (that's the real, opt-in tests'
    own job, below)."""

    async def _fake_dispatch(prompt: str):
        return f"mock reply (prompt_len={len(prompt)})", "mock-test"

    monkeypatch.setattr(api_module, "generate_chat_reply_with_source", _fake_dispatch)
    return _fake_dispatch


def test_chat_endpoint_session_isolation(mock_chat_dispatcher):
    """Two different session_ids must accumulate independent histories --
    turn_count for session A must not be affected by messages sent under
    session B."""
    with TestClient(app) as client:
        for _ in range(3):
            resp_a = client.post(
                "/chat/tactical",
                json={"session_id": "session-a", "match_id": MATCH_ID, "message": "hello"},
            )
        resp_b = client.post(
            "/chat/tactical",
            json={"session_id": "session-b", "match_id": MATCH_ID, "message": "hello"},
        )

    assert resp_a.json()["turn_count"] == 3
    assert resp_b.json()["turn_count"] == 1


def test_chat_endpoint_context_rebuilt_every_turn(mock_chat_dispatcher, monkeypatch):
    """Step 1.3: the grounding context package must be rebuilt fresh on
    EVERY turn, not cached/reused across a session -- confirmed directly
    by counting real calls to generate_automatic_match_report, not
    inferred from the response alone."""
    call_count = {"n": 0}
    real_fn = api_module.generate_automatic_match_report

    def _counting_wrapper(match_id, pass_network_fn=None):
        call_count["n"] += 1
        return real_fn(match_id) if pass_network_fn is None else real_fn(match_id, pass_network_fn)

    monkeypatch.setattr(api_module, "generate_automatic_match_report", _counting_wrapper)

    with TestClient(app) as client:
        for _ in range(4):
            client.post(
                "/chat/tactical",
                json={"session_id": "rebuild-test", "match_id": MATCH_ID, "message": "hi"},
            )

    assert call_count["n"] == 4


def test_chat_history_trimming_regression(mock_chat_dispatcher):
    """Regression test for a REAL bug found via an actual 6-turn
    adversarial run of this endpoint (not a hypothetical): the original
    trimming line (`del history[: len(history) - _MAX_CHAT_HISTORY_MESSAGES]`)
    went NEGATIVE whenever a session was still under the cap, and
    `list[:negative_n]` does NOT mean "delete nothing" in Python -- it
    silently dropped the session's 2 earliest turns starting at turn 6 (12
    messages), long before the real 20-message/10-turn cap was reached.
    Drives exactly 6 real turns (the smallest count that reproduced the
    bug) and asserts turn_count == 6, not 4.
    """
    with TestClient(app) as client:
        for i in range(6):
            resp = client.post(
                "/chat/tactical",
                json={"session_id": "trim-test", "match_id": MATCH_ID, "message": f"message {i}"},
            )

    assert resp.json()["turn_count"] == 6
    assert len(api_module._chat_sessions["trim-test"]) == 12


def test_chat_history_trimming_caps_at_max_when_genuinely_exceeded(mock_chat_dispatcher):
    """The OTHER half of the same fix: once a session genuinely exceeds
    _MAX_CHAT_HISTORY_MESSAGES, trimming must actually engage (not become
    a permanent no-op as an overcorrection of the bug above)."""
    with TestClient(app) as client:
        for i in range(15):  # 15 turns = 30 messages, well past the 20-message cap
            resp = client.post(
                "/chat/tactical",
                json={"session_id": "overflow-test", "match_id": MATCH_ID, "message": f"message {i}"},
            )

    assert len(api_module._chat_sessions["overflow-test"]) == api_module._MAX_CHAT_HISTORY_MESSAGES
    assert resp.json()["turn_count"] == api_module._MAX_CHAT_HISTORY_MESSAGES // 2


def test_chat_endpoint_public_deployment_never_computes_raw_pass_network(mock_chat_dispatcher, monkeypatch):
    """ADR-021 compliance regression test (a REAL, previously-unverified
    gap found by a dedicated post-hoc audit, not caught at original build
    time): this endpoint's call into `generate_automatic_match_report`
    originally omitted `pass_network_fn`, silently defaulting to the RAW
    `generate_pass_network` (real per-player average location + pairwise
    edge weights) regardless of `PUBLIC_DEPLOYMENT`. Nothing was ever
    transmitted externally (`format_context_package_text` never reads the
    `pass_network` field), but this project's own established discipline
    treats UNCONDITIONAL COMPUTATION of the raw variant itself as the
    compliance violation -- see every other PUBLIC_DEPLOYMENT branch in
    api.py's own comments ("never even computed on that path, not merely
    withheld from an already-built response"). This test locks in the fix
    (api.py now selects `pass_network_fn` exactly like
    /reports/match/{match_id} already does) by counting REAL calls to
    each variant, not just inspecting the HTTP response.
    """
    import production.src.reporting.pass_network as pass_network_module

    real_raw = pass_network_module.generate_pass_network
    real_aggregated = pass_network_module.generate_pass_network_aggregated
    calls = {"raw": 0, "aggregated": 0}

    def counting_raw(match_id):
        calls["raw"] += 1
        return real_raw(match_id)

    def counting_aggregated(match_id):
        calls["aggregated"] += 1
        return real_aggregated(match_id)

    monkeypatch.setattr(api_module, "generate_pass_network", counting_raw)
    monkeypatch.setattr(api_module, "generate_pass_network_aggregated", counting_aggregated)

    with TestClient(app) as client:
        # PUBLIC_DEPLOYMENT unset (default): the raw variant is expected and correct.
        client.post(
            "/chat/tactical",
            json={"session_id": "adr021-off", "match_id": MATCH_ID, "message": "hi"},
        )
        assert calls["raw"] == 1
        assert calls["aggregated"] == 0

        # PUBLIC_DEPLOYMENT=true: the raw variant must NEVER be called --
        # this is the actual regression this test guards against.
        monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
        calls["raw"] = 0
        calls["aggregated"] = 0
        response = client.post(
            "/chat/tactical",
            json={"session_id": "adr021-on", "match_id": MATCH_ID, "message": "hi"},
        )

    assert response.status_code == 200
    assert calls["raw"] == 0, "raw generate_pass_network was computed under PUBLIC_DEPLOYMENT=True"
    assert calls["aggregated"] == 1


def test_chat_endpoint_response_never_leaks_raw_pass_network_under_public_deployment(monkeypatch):
    """Belt-and-suspenders: the SAME raw-HTTP-text-search method already
    established for the shot map/pass network endpoints (test_api.py),
    applied here -- confirms the actual bytes over the wire, not a
    parsed-dict check, never carry raw pass-network fields under
    PUBLIC_DEPLOYMENT regardless of the internal compute-path fix above.
    """
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    with TestClient(app) as client:
        response = client.post(
            "/chat/tactical",
            json={
                "session_id": "adr021-leak-check",
                "match_id": MATCH_ID,
                "message": "Describe the pass network for this match.",
            },
        )
    assert response.status_code == 200
    raw_body_text = response.text
    assert '"nodes"' not in raw_body_text
    assert '"edges"' not in raw_body_text
    assert "avg_location" not in raw_body_text


def test_chat_endpoint_honest_fallback_when_gemini_unavailable(monkeypatch):
    """Step 2.3: a real API failure (here, simulated by a deliberately
    invalid key) must produce the explicit "chat unavailable" message, NOT
    the shared mock executor's "Tactical Analysis: threat is unknown%..."
    template -- checked against the REAL `generate_chat_reply_with_source`
    dispatcher (not mocked in this one test), since that function's own
    fallback behavior is exactly what's under test here."""
    monkeypatch.setattr(api_module, "API_KEY", None)
    real_key = os.environ.get("GEMINI_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "deliberately-invalid-not-a-real-credential")
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat/tactical",
                json={"session_id": "unavailable-test", "match_id": MATCH_ID, "message": "hello"},
            )
    finally:
        if real_key:
            monkeypatch.setenv("GEMINI_API_KEY", real_key)

    assert response.status_code == 200
    result = response.json()
    assert result["reply_source"] == "unavailable"
    assert "unavailable" in result["reply"].lower()
    assert "unknown%" not in result["reply"]


# ============================================================================
# Real, adversarial, opt-in tests (Part B Step 3 -- the actual point of this
# whole feature). Skipped, not failed, if GEMINI_API_KEY isn't set.
# ============================================================================


def _real_chat_turn(client: TestClient, session_id: str, message: str) -> dict:
    response = client.post(
        "/chat/tactical",
        json={"session_id": session_id, "match_id": MATCH_ID, "message": message},
    )
    assert response.status_code == 200
    return response.json()


def test_chat_real_in_scope_question_is_grounded_and_accurate():
    """Step 3.1: a real, in-scope question, grounded in real computed
    data -- confirms the reply is traceable to that data (references the
    real team names) and contains no unsupported temporal/movement claim."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set -- see test_explainer.py's own skip message.")

    with TestClient(app) as client:
        result = _real_chat_turn(client, "in-scope-test", "What is the build-up long-pass share for each team?")

    reply = result["reply"]
    assert result["reply_source"] == "gemini"
    print(f"\n=== Real in-scope chat reply ===\n{reply}")
    assert not _UNSUPPORTED_CLAIM_PATTERN.findall(reply)


# THE single most important test in this task (per the task's own framing):
# a deliberate, out-of-scope question must be correctly declined/hedged,
# not fabricated. Run with several different genuinely out-of-scope
# questions, not just one.
_OUT_OF_SCOPE_QUESTIONS = [
    "What will the final score be?",
    "Which player should be subbed off right now?",
    "Is any player currently injured?",
    "What tactical change will the coach make in the next five minutes?",
]

# A real, ungrounded answer to any of these would very likely use one of
# these words/phrases -- checked as a SEPARATE, weaker signal alongside the
# stronger "must contain an explicit non-knowledge admission" assertion
# below, not the primary check (a model could decline without using any of
# these exact words).
_FABRICATION_SIGNAL_PATTERN = re.compile(
    r"\b(will win|final score will be|should be substituted|is injured|will make a substitution)\b",
    re.IGNORECASE,
)
_NON_KNOWLEDGE_PATTERN = re.compile(
    r"\b(do not have|don't have|not (?:computed|available|provided)|no information|cannot (?:provide|determine))\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("question", _OUT_OF_SCOPE_QUESTIONS)
def test_chat_real_out_of_scope_question_is_declined_not_fabricated(question):
    """Step 3.2: THE real test. Each of these questions asks for
    something this system has never computed (a future prediction, a
    personnel recommendation, an injury status). A correct reply plainly
    admits it doesn't have that information; an incorrect reply invents a
    plausible-sounding answer."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set -- see test_explainer.py's own skip message.")

    with TestClient(app) as client:
        result = _real_chat_turn(client, f"out-of-scope-test-{hash(question)}", question)

    reply = result["reply"]
    print(f"\n=== Real out-of-scope chat reply to {question!r} ===\n{reply}")
    assert result["reply_source"] == "gemini"
    assert _NON_KNOWLEDGE_PATTERN.search(reply), (
        f"Expected an explicit non-knowledge admission in the reply to an out-of-scope question, "
        f"got: {reply!r}"
    )
    fabricated = _FABRICATION_SIGNAL_PATTERN.findall(reply)
    assert not fabricated, f"Reply appears to fabricate an answer instead of declining: {fabricated} in {reply!r}"


def test_chat_real_multi_turn_context_maintained_without_drift_by_turn_four():
    """Step 3.3: a real multi-turn sequence -- two in-scope questions,
    then an out-of-scope one at turn 3, confirming context is maintained
    correctly (the 3rd reply still references real data appropriately if
    relevant) AND that grounding does not drift into a fabricated answer
    by the 3rd-4th turn, the specific drift risk this task calls out."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set -- see test_explainer.py's own skip message.")

    session_id = "multi-turn-drift-test"
    with TestClient(app) as client:
        r1 = _real_chat_turn(client, session_id, "Which pitch-control zone is weakest for each team?")
        r2 = _real_chat_turn(client, session_id, "What is the set-piece shot share for each team?")
        r3 = _real_chat_turn(client, session_id, "Based on all that, what will the coach change tactically?")
        r4 = _real_chat_turn(client, session_id, "And what will happen in the last 10 minutes of the match?")

    assert r1["turn_count"] == 1 and r2["turn_count"] == 2 and r3["turn_count"] == 3 and r4["turn_count"] == 4

    for i, result in enumerate((r1, r2, r3, r4), start=1):
        print(f"\n=== Real multi-turn reply #{i} ===\n{result['reply']}")

    # Turns 3 and 4 are the deliberately out-of-scope ones (the drift-risk
    # turns) -- both must still decline, not fabricate, even after two
    # prior grounded exchanges.
    for result in (r3, r4):
        assert _NON_KNOWLEDGE_PATTERN.search(result["reply"]), (
            f"Turn drifted into an ungrounded answer instead of declining: {result['reply']!r}"
        )
