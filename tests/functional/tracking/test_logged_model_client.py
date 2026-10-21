"""Logged-model operations through the MLflow plugin interface."""

import pytest
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking import MlflowClient


def test_logged_model_lifecycle_tags_metrics_and_search_through_client(store):
    client = MlflowClient(tracking_uri=store.store_uri)
    experiment_id = client.create_experiment("client-logged-models")
    run = client.create_run(experiment_id, run_name="source")
    model = client.create_logged_model(
        experiment_id,
        name="classifier",
        source_run_id=run.info.run_id,
        model_type="classifier",
        tags={"team.name": "risk"},
        params={"depth": "3"},
    )
    assert model.status == LoggedModelStatus.PENDING
    client.log_metric(
        run.info.run_id,
        "loss",
        5.0,
        timestamp=1000,
        step=1,
        model_id=model.model_id,
        synchronous=True,
    )
    client.set_logged_model_tags(model.model_id, {"team.name": "$platform", "purpose": "eval"})
    client.delete_logged_model_tag(model.model_id, "purpose")

    loaded = client.get_logged_model(model.model_id)
    assert loaded.source_run_id == run.info.run_id
    assert loaded.model_type == "classifier"
    assert loaded.tags == {"team.name": "$platform"}
    assert loaded.params == {"depth": "3"}
    assert loaded.metrics[0].value == 5
    assert loaded.metrics[0].run_id == run.info.run_id
    page = client.search_logged_models(
        [experiment_id],
        filter_string="metrics.loss < 6 AND params.depth = '3'",
        order_by=[{"field_name": "metrics.loss", "ascending": False}],
        max_results=1,
    )
    assert [result.model_id for result in page] == [model.model_id]
    assert page.token is None
    assert page[0].to_proto() == loaded.to_proto()
    client.delete_logged_model(model.model_id)
    with pytest.raises(MlflowException) as caught:
        client.get_logged_model(model.model_id)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert client.search_logged_models([experiment_id]) == []
