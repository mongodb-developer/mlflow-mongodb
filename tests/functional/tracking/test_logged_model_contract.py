"""Logged-model persistence contracts against MongoDB."""

import json
import sys
from unittest.mock import patch
from uuid import UUID

import pytest
from mlflow.entities import LifecycleStage, Metric, RunTag
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_MODELS

from mlflow_mongodb.infrastructure.errors import RepositoryNotFoundError, RepositoryTagNotFoundError


def _stored_model(store, model_id):
    return store._database[store._settings.logged_models_collection_name].find_one(
        {"_id": model_id}
    )


def test_get_logged_model_dedupes_and_sanitizes_metrics(store, run):
    model = store.create_logged_model(run.info.experiment_id, name="retried-metric")
    metric = Metric("loss", float("inf"), 1000, 1, model_id=model.model_id)
    store.log_batch(run.info.run_id, metrics=[metric, metric], params=[], tags=[])
    store.log_batch(run.info.run_id, metrics=[metric], params=[], tags=[])

    metrics = store.get_logged_model(model.model_id).metrics

    assert len(metrics) == 1
    assert metrics[0].key == "loss"
    assert metrics[0].value == pytest.approx(sys.float_info.max)
    assert metrics[0].timestamp == 1000
    assert metrics[0].step == 1
    assert metrics[0].model_id == model.model_id
    assert metrics[0].run_id == run.info.run_id


@pytest.mark.parametrize(
    "first_dataset,second_dataset",
    [
        ((None, None), (None, None)),
        (("eval", "a"), ("train", "b")),
    ],
)
def test_shared_points_keep_both_models_and_one_paginated_run_history(
    store, run, first_dataset, second_dataset
):
    first = store.create_logged_model(run.info.experiment_id, name="first")
    second = store.create_logged_model(run.info.experiment_id, name="second")
    metrics = [
        Metric(
            "loss",
            timestamp / 10000,
            timestamp,
            1,
            model_id=model.model_id,
            dataset_name=dataset[0],
            dataset_digest=dataset[1],
        )
        for model, dataset in ((first, first_dataset), (second, second_dataset))
        for timestamp in (1000, 2000)
    ]
    metrics.append(Metric("loss", 0.3, 3000, 1))
    store.log_batch(run.info.run_id, metrics=[*metrics, *metrics], params=[], tags=[])
    store.log_batch(run.info.run_id, metrics=metrics, params=[], tags=[])

    for model, dataset in ((first, first_dataset), (second, second_dataset)):
        history = store.get_logged_model(model.model_id).metrics
        assert len(history) == 2
        assert sorted(metric.timestamp for metric in history) == [1000, 2000]
        assert all(metric.model_id == model.model_id for metric in history)
        assert all(metric.run_id == run.info.run_id for metric in history)
        assert {(metric.dataset_name, metric.dataset_digest) for metric in history} == {dataset}

    first_page = store.get_metric_history(run.info.run_id, "loss", max_results=1)
    second_page = store.get_metric_history(
        run.info.run_id, "loss", max_results=1, page_token=first_page.token
    )
    third_page = store.get_metric_history(
        run.info.run_id, "loss", max_results=1, page_token=second_page.token
    )
    assert first_page.token and second_page.token
    assert third_page.token is None
    history = [*first_page, *second_page, *third_page]
    assert [metric.timestamp for metric in history] == [1000, 2000, 3000]
    assert [metric.value for metric in history] == pytest.approx([0.1, 0.2, 0.3])
    assert all(metric.model_id is None for metric in history)
    assert all(metric.dataset_name is None and metric.dataset_digest is None for metric in history)

    collection = store._database[store._settings.run_metrics_collection_name]
    assert collection.count_documents({"scope": "run", "run_id": run.info.run_id}) == 3
    assert collection.count_documents({"scope": "model", "run_id": run.info.run_id}) == 4
    assert collection.count_documents({"run_id": run.info.run_id}) == 7
    document = collection.find_one({"scope": "run", "run_id": run.info.run_id})
    assert set(document) == {"_id", "scope", "run_id", "k", "v", "timestamp", "step"}


@pytest.mark.parametrize(
    "replacement",
    [
        {"value": 0.8},
        {"dataset_name": "train", "dataset_digest": "b"},
    ],
)
def test_conflicting_model_point_is_rejected_without_replacing_existing_point(
    store, run, replacement
):
    model = store.create_logged_model(run.info.experiment_id, name="conflict")
    point = {
        "key": "loss",
        "value": 0.4,
        "timestamp": 1000,
        "step": 1,
        "model_id": model.model_id,
        "dataset_name": "eval",
        "dataset_digest": "a",
    }
    store.log_metric(run.info.run_id, Metric(**point))

    with pytest.raises(MlflowException) as caught:
        store.log_metric(run.info.run_id, Metric(**{**point, **replacement}))

    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.message == "Unable to log run batch."
    assert caught.value.__cause__ is None
    history = store.get_logged_model(model.model_id).metrics
    assert len(history) == 1
    assert history[0].value == pytest.approx(0.4)
    assert history[0].dataset_name == "eval"
    assert history[0].dataset_digest == "a"


@pytest.mark.parametrize("status", [LoggedModelStatus.READY, LoggedModelStatus.FAILED])
@pytest.mark.parametrize("deleted", [False, True])
def test_finalize_preserves_model_data_and_returns_metric_history(
    store, run, clock, status, deleted
):
    model = store.create_logged_model(
        run.info.experiment_id,
        name="finalized",
        tags=[LoggedModelTag("team", "risk")],
        params=[LoggedModelParameter("depth", "3")],
    )
    store.log_metric(run.info.run_id, Metric("loss", 0.4, 1000, 1, model_id=model.model_id))
    if deleted:
        store.delete_logged_model(model.model_id)
    before = _stored_model(store, model.model_id)
    clock["now"] += 1000

    finalized = store.finalize_logged_model(model.model_id, status)
    repeated = store.finalize_logged_model(model.model_id, status)

    assert finalized.status == repeated.status == status
    assert finalized.last_updated_timestamp == clock["now"]
    assert finalized.tags == {"team": "risk"}
    assert finalized.params == {"depth": "3"}
    assert len(finalized.metrics) == 1
    assert finalized.metrics[0].value == pytest.approx(0.4)
    assert finalized.metrics[0].model_id == model.model_id
    assert _stored_model(store, model.model_id) == {
        **before,
        "status": status.value,
        "last_updated_timestamp": clock["now"],
    }


def test_finalize_missing_model_does_not_insert(store):
    model_id = "m-" + "f" * 32
    with pytest.raises(MlflowException) as caught:
        store.finalize_logged_model(model_id, LoggedModelStatus.READY)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert _stored_model(store, model_id) is None


def test_record_logged_model_appends_history_and_preserves_other_tags(store, run):
    models = [Model(artifact_path="first"), Model(artifact_path="second")]
    store.set_tag(run.info.run_id, RunTag("keep", "$value"))
    for model in models:
        store.record_logged_model(run.info.run_id, model)

    loaded = store.get_run(run.info.run_id)
    expected = [model.get_tags_dict() for model in models]
    assert json.loads(loaded.data.tags[MLFLOW_LOGGED_MODELS]) == expected
    assert loaded.data.tags["keep"] == "$value"
    document = store._database[store._settings.runs_collection_name].find_one(
        {"_id": run.info.run_id}
    )
    stored_tags = {tag["k"]: tag["v"] for tag in document["tags"]}
    assert json.loads(stored_tags[MLFLOW_LOGGED_MODELS]) == expected


def test_record_logged_model_rejects_deleted_run_without_changing_it(store, run):
    store.delete_run(run.info.run_id)
    collection = store._database[store._settings.runs_collection_name]
    before = collection.find_one({"_id": run.info.run_id})
    with pytest.raises(MlflowException) as caught:
        store.record_logged_model(run.info.run_id, Model(artifact_path="model"))
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert collection.find_one({"_id": run.info.run_id}) == before


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
