"""Milestone 18 validation: the /simulate REST counterfactual endpoint.

Cross-validated against an INDEPENDENTLY-implemented batch prediction path
(`production.src.models.evaluation.predict_cumulative_incidence`,
established since Milestone 7/13) rather than the endpoint's own private
helper -- comparing the endpoint to its own internal function would just be
comparing a function to itself and would never catch a silently-diverged
extraction path.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from fastapi.testclient import TestClient

from production.src.ingestion.statsbomb_io import parse_360_frame
from production.src.models.evaluation import predict_cumulative_incidence
from production.src.models.explainer import load_deterministic_mlp
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.simulator import perturb_features
from production.src.serving.api import _find_qualifying_frame_for_minute, app

MATCH_ID = 3857276
TIME_BIN = 3  # 15s horizon, matching every prior milestone
TOLERANCE = 1e-4


def test_simulate_endpoint_returns_valid_response():
    with TestClient(app) as client:
        response = client.get(f"/simulate?match_id={MATCH_ID}&minute=10&action=high_press")

    assert response.status_code == 200
    body = response.json()

    for key in ("match_id", "minute", "action", "baseline_threat_15s", "simulated_threat_15s", "delta"):
        assert key in body, f"missing key {key!r} in response: {body}"

    assert body["match_id"] == MATCH_ID
    assert body["minute"] == 10
    assert body["action"] == "high_press"
    assert isinstance(body["baseline_threat_15s"], float)
    assert isinstance(body["simulated_threat_15s"], float)
    assert isinstance(body["delta"], float)
    assert 0.0 <= body["baseline_threat_15s"] <= 1.0
    assert 0.0 <= body["simulated_threat_15s"] <= 1.0

    expected_delta = body["simulated_threat_15s"] - body["baseline_threat_15s"]
    assert abs(body["delta"] - expected_delta) < 1e-9, (
        f"delta {body['delta']} does not equal simulated - baseline ({expected_delta})"
    )


def test_simulate_endpoint_cross_validates_against_batch_pipeline():
    """Independently re-derives baseline/simulated cumulative incidence
    using the ORIGINAL Milestone 13 batch functions (`extract_features`,
    `perturb_features`, `evaluation.predict_cumulative_incidence`) against
    the SAME deterministically-selected model (Milestone 15/16's selection
    rule -- not "latest run", which could resolve to a different one of the
    two nearly-identical seed-42/seed-43 MLP runs) and the SAME resolved
    match frame (via the endpoint's own frame-resolution helper, since
    frame *selection* isn't what's under test here -- prediction
    consistency with the established batch pipeline is). If the REST
    endpoint were secretly using a different/simplified extraction path,
    this would drift outside tolerance.
    """
    minute = 10
    action = "high_press"

    model, normalization_mean, normalization_std, run_id = load_deterministic_mlp()

    result = _find_qualifying_frame_for_minute(MATCH_ID, minute)
    assert result is not None
    event, frame_data = result
    parsed_frame = parse_360_frame(event, frame_data)
    baseline_features = extract_features(parsed_frame)
    simulated_features = perturb_features(baseline_features, action)

    expected_baseline = predict_cumulative_incidence(
        model, baseline_features, normalization_mean, normalization_std, time_bin=TIME_BIN
    )
    expected_simulated = predict_cumulative_incidence(
        model, simulated_features, normalization_mean, normalization_std, time_bin=TIME_BIN
    )

    with TestClient(app) as client:
        response = client.get(f"/simulate?match_id={MATCH_ID}&minute={minute}&action={action}")

    assert response.status_code == 200
    body = response.json()

    print(f"\n=== Cross-validation (match {MATCH_ID}, minute={minute}, action={action}, run_id={run_id}) ===")
    print(f"REST endpoint : baseline={body['baseline_threat_15s']:.6f}, simulated={body['simulated_threat_15s']:.6f}")
    print(f"Batch pipeline: baseline={expected_baseline:.6f}, simulated={expected_simulated:.6f}")

    assert abs(body["baseline_threat_15s"] - expected_baseline) < TOLERANCE, (
        f"REST baseline_threat_15s={body['baseline_threat_15s']} vs batch expected={expected_baseline}"
    )
    assert abs(body["simulated_threat_15s"] - expected_simulated) < TOLERANCE, (
        f"REST simulated_threat_15s={body['simulated_threat_15s']} vs batch expected={expected_simulated}"
    )


def test_simulate_endpoint_rejects_invalid_action():
    with TestClient(app) as client:
        response = client.get(f"/simulate?match_id={MATCH_ID}&minute=10&action=nonsense_action")

    assert response.status_code in (400, 422), (
        f"expected a clean 400/422 for an invalid action, got {response.status_code}: {response.text}"
    )


def test_simulate_endpoint_404_past_end_of_match():
    """Case (a): a minute genuinely beyond the match's total duration.
    This cached match's last event (in either period) is minute=93, so 200
    is unambiguously past the actual end of the match -- not merely past
    some 360-coverage boundary that stops earlier than the match itself
    (see the module-level note below on why case (b) isn't exercised).
    """
    with TestClient(app) as client:
        response = client.get(f"/simulate?match_id={MATCH_ID}&minute=200&action=no_change")

    assert response.status_code == 404
    body = response.json()
    assert "200" in body["detail"]
    assert str(MATCH_ID) in body["detail"]


# Case (b) -- a minute value known to fall in a real 360-coverage GAP
# *within* the match's actual duration -- was checked for directly against
# the cached data and does NOT exist for this match: events with both a
# `location` and an associated 360 frame reach all the way to minute=93 in
# period 2, identical to the match's actual final event minute (also 93).
# Coverage is effectively continuous through the whole match, so there is
# no minute strictly inside [0, 93] for which
# `_find_qualifying_frame_for_minute` would fail to find a later
# qualifying frame -- only querying past the match's actual end (case a,
# above) can trigger a genuine 404 here. Asserting a second, distinct
# in-match "gap" case would be fabricating a scenario this data doesn't
# actually contain, which the milestone instructions explicitly warn
# against.
