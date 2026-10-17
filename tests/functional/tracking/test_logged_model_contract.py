"""Logged-model persistence contracts against MongoDB."""

from unittest.mock import patch
from uuid import UUID

import pytest
from mlflow.entities import LifecycleStage, Metric
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from mlflow_mongodb.infrastructure.errors import RepositoryNotFoundError, RepositoryTagNotFoundError


def _stored_model(store, model_id):
    return store._database[store._settings.logged_models_collection_name].find_one(
        {"_id": model_id}
    )


def test_create_get_preserve_metadata_and_k_v_arrays(store, run, clock):
    model = store.create_logged_model(
        run.info.experiment_id,
        name="classifier",
        source_run_id=run.info.run_id,
        model_type="classifier",
        tags=[LoggedModelTag("team.name", "$risk")],
        params=[LoggedModelParameter("depth", "3")],
    )
    other = store.create_logged_model(run.info.experiment_id, name="classifier")
    loaded = store.get_logged_model(model.model_id)
    document = _stored_model(store, model.model_id)

    assert model.model_id != other.model_id
    assert model.model_id == "m-" + UUID(hex=model.model_id[2:]).hex
    assert loaded.name == "classifier"
    assert loaded.experiment_id == run.info.experiment_id
    assert loaded.source_run_id == run.info.run_id
    assert loaded.model_type == "classifier"
    assert loaded.status == LoggedModelStatus.PENDING
    assert loaded.status_message is None
    assert loaded.creation_timestamp == loaded.last_updated_timestamp == clock["now"]
    assert loaded.artifact_location == (
        f"{store.get_experiment(run.info.experiment_id).artifact_location}/models/{model.model_id}/artifacts"
    )
    assert loaded.tags == {"team.name": "$risk"}
    assert loaded.params == {"depth": "3"}
    assert document["tags"] == [{"k": "team.name", "v": "$risk"}]
    assert document["params"] == [{"k": "depth", "v": "3"}]
    assert document["lifecycle_stage"] == LifecycleStage.ACTIVE
    assert "model_id" not in document and "metrics" not in document


def test_omitted_name_is_generated_and_persisted(store, run):
    with patch("mlflow_mongodb.tracking.store._generate_random_name", return_value="generated"):
        model = store.create_logged_model(run.info.experiment_id)
    assert model.name == store.get_logged_model(model.model_id).name == "generated"
    assert model.tags == {} and model.params == {}


@pytest.mark.parametrize("missing", [True, False])
def test_creation_requires_existing_active_experiment(store, missing):
    experiment_id = "missing" if missing else store.create_experiment("deleted-model-parent")
    if not missing:
        store.delete_experiment(experiment_id)
    with pytest.raises(MlflowException) as caught:
        store.create_logged_model(experiment_id, name="model")
    expected = RESOURCE_DOES_NOT_EXIST if missing else INVALID_PARAMETER_VALUE
    assert caught.value.error_code == ErrorCode.Name(expected)
    assert store._database[store._settings.logged_models_collection_name].count_documents({}) == 0


@pytest.mark.parametrize("field", ["tags", "params"])
def test_duplicate_creation_entries_do_not_insert_partial_model(store, run, field):
    entity = LoggedModelTag if field == "tags" else LoggedModelParameter
    with pytest.raises(MlflowException) as caught:
        store.create_logged_model(
            run.info.experiment_id,
            name="model",
            **{field: [entity("key", "first"), entity("key", "second")]},
        )
    assert caught.value.error_code == ErrorCode.Name(BAD_REQUEST)
    assert store._database[store._settings.logged_models_collection_name].count_documents({}) == 0


def test_get_collects_history_across_runs_and_preserves_metric_identity(store, run):
    model = store.create_logged_model(run.info.experiment_id, name="model")
    other = store.create_logged_model(run.info.experiment_id, name="other")
    other_run = store.create_run(run.info.experiment_id, "user", 1000, [], "other-run")
    store.log_batch(
        run.info.run_id,
        [
            Metric(
                "loss",
                0.4,
                1000,
                1,
                model_id=model.model_id,
                dataset_name="eval",
                dataset_digest="abc",
            ),
            Metric(
                "loss",
                0.2,
                2000,
                2,
                model_id=model.model_id,
                dataset_name="train",
                dataset_digest="def",
            ),
            Metric("loss", 9.0, 3000, 3, model_id=other.model_id),
            Metric("loss", 8.0, 4000, 4),
        ],
        [],
        [],
    )
    store.log_metric(other_run.info.run_id, Metric("loss", 0.3, 1500, 1, model_id=model.model_id))

    metrics = store.get_logged_model(model.model_id).metrics

    assert [(m.value, m.timestamp, m.step) for m in metrics] == [
        (0.2, 2000, 2),
        (0.3, 1500, 1),
        (0.4, 1000, 1),
    ]
    assert all(m.model_id == model.model_id for m in metrics)
    assert metrics[1].run_id == other_run.info.run_id
    assert metrics[2].dataset_name == "eval" and metrics[2].dataset_digest == "abc"
    document = store._database[store._settings.run_metrics_collection_name].find_one(
        {"model_id": model.model_id}
    )
    assert "k" in document and "v" in document
    assert "key" not in document and "value" not in document


def test_repeated_delete_preserves_data_and_refreshes_timestamp(store, run, clock):
    model = store.create_logged_model(
        run.info.experiment_id,
        name="model",
        tags=[LoggedModelTag("team", "risk")],
        params=[LoggedModelParameter("depth", "3")],
    )
    store.log_metric(run.info.run_id, Metric("loss", 5.0, 1000, 0, model_id=model.model_id))
    original = _stored_model(store, model.model_id)
    for advance in (0, 0, 10):
        clock["now"] += advance
        store.delete_logged_model(model.model_id)
        stored = _stored_model(store, model.model_id)
        assert stored["lifecycle_stage"] == LifecycleStage.DELETED
        assert stored["last_updated_timestamp"] == clock["now"]
        assert stored["tags"] == original["tags"] and stored["params"] == original["params"]
    with pytest.raises(MlflowException) as caught:
        store.get_logged_model(model.model_id)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert store.get_logged_model(model.model_id, allow_deleted=True).metrics[0].value == 5
    assert store.search_logged_models([run.info.experiment_id]) == []


@pytest.mark.parametrize("deleted", [False, True])
def test_tag_mutation_handles_literals_duplicates_empty_batches_and_deleted_models(
    store, run, deleted
):
    model = store.create_logged_model(
        run.info.experiment_id, name="model", tags=[LoggedModelTag("preserve", "yes")]
    )
    if deleted:
        store.delete_logged_model(model.model_id)
    before = _stored_model(store, model.model_id)
    store.set_logged_model_tags(model.model_id, [])
    store.set_logged_model_tags(
        model.model_id,
        [
            LoggedModelTag("team.name", "old"),
            LoggedModelTag("team.name", "$new"),
            LoggedModelTag("$literal", "$value"),
        ],
    )
    store.set_logged_model_tags(model.model_id, [LoggedModelTag("team.name", "$new")])
    loaded = store.get_logged_model(model.model_id, allow_deleted=True)
    assert loaded.tags == {"preserve": "yes", "team.name": "$new", "$literal": "$value"}
    store.delete_logged_model_tag(model.model_id, "team.name")
    assert store.get_logged_model(model.model_id, allow_deleted=True).tags == {
        "preserve": "yes",
        "$literal": "$value",
    }
    document = _stored_model(store, model.model_id)
    assert all(set(tag) == {"k", "v"} for tag in document["tags"])
    assert document["last_updated_timestamp"] == before["last_updated_timestamp"]
    with pytest.raises(MlflowException) as caught:
        store.delete_logged_model_tag(model.model_id, "team.name")
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.parametrize(
    "operation,args",
    [
        ("get_logged_model", ()),
        ("delete_logged_model", ()),
        ("set_logged_model_tags", ([],)),
        ("delete_logged_model_tag", ("team",)),
    ],
)
def test_missing_model_operations_report_public_error(store, operation, args):
    with pytest.raises(MlflowException) as caught:
        getattr(store, operation)("m-" + "f" * 32, *args)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_repository_distinguishes_missing_model_and_tag_without_unrelated_changes(store, run):
    model = store.create_logged_model(
        run.info.experiment_id, name="model", tags=[LoggedModelTag("team", "risk")]
    )
    repository = store._logged_model_repository
    before = _stored_model(store, model.model_id)
    with pytest.raises(RepositoryTagNotFoundError):
        repository.delete_tag(model_id=model.model_id, key="missing")
    assert _stored_model(store, model.model_id) == before
    with pytest.raises(RepositoryNotFoundError):
        repository.set_tags(model_id="missing", tags={})
    with pytest.raises(RepositoryNotFoundError):
        repository.mark_deleted(model_id="missing", last_updated_timestamp=1000)
    with pytest.raises(RepositoryNotFoundError):
        repository.delete_tag(model_id="missing", key="team")
    assert _stored_model(store, model.model_id) == before


def test_async_run_metric_logging_is_visible_on_logged_model(store, run):
    model = store.create_logged_model(run.info.experiment_id, name="async-model")
    store.log_metric_async(
        run.info.run_id, Metric("loss", 5.0, 1000, 1, model_id=model.model_id)
    ).wait()
    metrics = store.get_logged_model(model.model_id).metrics
    assert len(metrics) == 1
    assert metrics[0].value == 5
    assert metrics[0].model_id == model.model_id
    assert metrics[0].run_id == run.info.run_id
