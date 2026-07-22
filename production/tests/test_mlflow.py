"""Milestone 8 validation: MLflow logging works end-to-end.

Uses an ISOLATED tracking URI (pytest's tmp_path) for the entire test.
mlflow.set_tracking_uri is process-global state -- without pointing it at a
throwaway location first, this test would write into the same local
mlruns/ store used by real training runs, polluting the experiment history
every time the test suite runs. The original tracking URI is restored
afterward so this test can't leak into any other test in the same session.
"""

import os

# See production/src/pipeline/train.py for why this is needed: recent
# mlflow versions put the local file-store backend into "maintenance mode"
# behind this opt-in env var, and this project's isolated test tracking URI
# below is still a file:// store.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow


def test_mlflow_log_and_retrieve_roundtrip(tmp_path):
    original_tracking_uri = mlflow.get_tracking_uri()
    isolated_tracking_uri = f"file://{tmp_path}/mlruns"

    try:
        mlflow.set_tracking_uri(isolated_tracking_uri)

        with mlflow.start_run(run_name="test_run") as run:
            run_id = run.info.run_id
            assert run_id is not None

            mlflow.log_param("dummy_param", "test_value")
            mlflow.log_metric("dummy_metric", 0.42)

            artifact_path = tmp_path / "dummy_artifact.txt"
            artifact_path.write_text("dummy artifact content")
            mlflow.log_artifact(str(artifact_path))

        # Retrieve against the SAME isolated tracking URI -- a real check
        # that logging actually persisted and is queryable, not just that
        # the calls didn't throw.
        retrieved_run = mlflow.get_run(run_id)
        assert retrieved_run.data.params["dummy_param"] == "test_value"
        assert retrieved_run.data.metrics["dummy_metric"] == 0.42
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)
