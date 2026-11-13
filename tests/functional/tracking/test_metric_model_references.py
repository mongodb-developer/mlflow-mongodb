"""Model metric references validated against real MongoDB records."""

import pytest
from mlflow.entities import LifecycleStage, Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, ErrorCode

MISSING_MODEL_ID = "m-" + "f" * 32
OTHER_MISSING_MODEL_ID = "m-" + "e" * 32


@pytest.fixture
def metric_models(store, run):
    active = store.create_logged_model(run.info.experiment_id, name="active-model")
    deleted = store.create_logged_model(run.info.experiment_id, name="deleted-model")
    store.delete_logged_model(deleted.model_id)
    return {
        "active": active.model_id,
        "deleted": deleted.model_id,
        "missing": MISSING_MODEL_ID,
        "other_missing": OTHER_MISSING_MODEL_ID,
    }


def _stored_run(store, run_id):
    return store._database[store._settings.runs_collection_name].find_one({"_id": run_id})


@pytest.mark.parametrize(
    "model_names,missing_ids",
    [
        (("missing", "active"), [MISSING_MODEL_ID]),
        (("active", "missing"), [MISSING_MODEL_ID]),
        (("deleted", "missing"), [MISSING_MODEL_ID]),
        (
            ("missing", "active", "other_missing", "missing"),
            [OTHER_MISSING_MODEL_ID, MISSING_MODEL_ID],
        ),
    ],
)
def test_missing_metric_model_rejects_mixed_batch_without_any_writes(
    store, run, metric_models, model_names, missing_ids
):
    before = _stored_run(store, run.info.run_id)
    metrics = [Metric("ordinary", 1, 1000, 0)]
    metrics.extend(
        Metric("loss", step + 1, 1000, step, model_id=metric_models[name])
        for step, name in enumerate(model_names)
    )

    with pytest.raises(MlflowException) as caught:
        store.log_batch(run.info.run_id, metrics, [Param("depth", "3")], [RunTag("team", "risk")])

    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.message == (
        f"Unable to log metrics: logged models with IDs {missing_ids} not found."
    )
    assert _stored_run(store, run.info.run_id) == before
    assert (
        store._database[store._settings.run_metrics_collection_name].count_documents(
            {"run_id": run.info.run_id}
        )
        == 0
    )


def test_async_batch_reports_missing_model_without_any_writes(store, run, metric_models):
    before = _stored_run(store, run.info.run_id)
    operations = store.log_batch_async(
        run.info.run_id,
        [
            Metric("loss", 1, 1000, 0, model_id=metric_models["active"]),
            Metric("loss", 2, 1000, 1, model_id=MISSING_MODEL_ID),
        ],
        [Param("depth", "3")],
        [RunTag("team", "risk")],
    )

    with pytest.raises(MlflowException, match="logged models with IDs"):
        operations.wait()

    assert _stored_run(store, run.info.run_id) == before
    assert (
        store._database[store._settings.run_metrics_collection_name].count_documents(
            {"run_id": run.info.run_id}
        )
        == 0
    )


def test_async_batch_accepts_repeated_references_to_deleted_model(store, run, metric_models):
    model_id = metric_models["deleted"]
    collection = store._database[store._settings.logged_models_collection_name]
    before = collection.find_one({"_id": model_id})

    store.log_batch_async(
        run.info.run_id,
        [
            Metric("loss", 1, 1000, 0, model_id=model_id),
            Metric("accuracy", 2, 1000, 0, model_id=model_id),
        ],
        [],
        [],
    ).wait()

    loaded = store.get_logged_model(model_id, allow_deleted=True)
    assert before["lifecycle_stage"] == LifecycleStage.DELETED
    assert {metric.key for metric in loaded.metrics} == {"loss", "accuracy"}
    assert all(metric.model_id == model_id for metric in loaded.metrics)
    assert collection.find_one({"_id": model_id}) == before
    assert len(store.get_metric_history(run.info.run_id, "loss")) == 1
    assert len(store.get_metric_history(run.info.run_id, "accuracy")) == 1
    assert (
        store._database[store._settings.run_metrics_collection_name].count_documents(
            {"run_id": run.info.run_id}
        )
        == 4
    )
