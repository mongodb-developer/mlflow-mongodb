"""Run operations through the MLflow plugin interface."""

import pytest
from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.tracking import MlflowClient


def test_run_lifecycle_logging_and_info_updates_through_mlflow_client(store):
    client = MlflowClient(tracking_uri=store.store_uri)
    experiment_id = client.create_experiment(
        "client-runs", artifact_location="s3://artifacts/client"
    )
    run = client.create_run(
        experiment_id, start_time=1000, tags={"team": "risk"}, run_name="client-run"
    )
    run_id = run.info.run_id

    client.log_batch(
        run_id,
        metrics=[Metric("loss", 0.4, 1000, 0)],
        params=[Param("depth", "3")],
        tags=[RunTag("purpose", "evaluation")],
        synchronous=True,
    )
    client.log_metric(run_id, "loss", 0.3, timestamp=1001, step=1, synchronous=True)
    client.set_tag(run_id, "team", "platform", synchronous=True)
    client.update_run(run_id, status="FINISHED", name="$client-renamed")

    loaded = client.get_run(run_id)
    assert loaded.info.artifact_uri == f"s3://artifacts/client/{run_id}/artifacts"
    assert loaded.info.run_name == "$client-renamed"
    assert loaded.info.status == "FINISHED"
    assert loaded.data.metrics == {"loss": 0.3}
    assert loaded.data.tags["team"] == "platform"
    assert loaded.data.tags["purpose"] == "evaluation"
    assert store.get_run(run_id).info.run_name == "$client-renamed"
    assert [metric.value for metric in client.get_metric_history(run_id, "loss")] == [0.4, 0.3]

    with pytest.raises(MlflowException, match="Changing param values") as caught:
        client.log_param(run_id, "depth", "4", synchronous=True)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    client.delete_run(run_id)
    assert client.get_run(run_id).info.lifecycle_stage == "deleted"
    client.restore_run(run_id)
    assert client.get_run(run_id).info.lifecycle_stage == "active"
