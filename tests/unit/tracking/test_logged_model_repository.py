"""Logged-model repository contracts and persistence boundaries."""

from unittest.mock import MagicMock, Mock, call

import pytest
from bson.errors import InvalidBSON, InvalidDocument
from mlflow.entities import LifecycleStage
from pymongo import ReturnDocument
from pymongo.errors import OperationFailure

import mlflow_mongodb.tracking.repositories.logged_models as model_module
from mlflow_mongodb.infrastructure.errors import (
    RepositoryNotFoundError,
    RepositoryPersistenceError,
    RepositoryTagNotFoundError,
)
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.repositories import LoggedModelRepository

MODEL_ID = "m-" + "a" * 32
CREATE_ARGS = {
    "model_id": MODEL_ID,
    "experiment_id": "42",
    "name": "example-model",
    "artifact_location": "s3://artifacts/model",
    "creation_timestamp": 1000,
    "status": "PENDING",
    "lifecycle_stage": LifecycleStage.ACTIVE,
    "source_run_id": None,
    "model_type": "classifier",
    "tags": {"team.name": "$risk"},
    "params": {"depth": "3"},
}


@pytest.fixture
def collections():
    return {"logged_models": MagicMock(), "run_metrics": MagicMock()}


@pytest.fixture
def repository(collections, logged_model_document_factory):
    database = MagicMock()
    database.__getitem__.side_effect = collections.__getitem__
    collections["logged_models"].find_one.return_value = logged_model_document_factory(_id=MODEL_ID)
    return LoggedModelRepository(database)


def _cursor(documents):
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__iter__.side_effect = lambda: iter(documents)
    cursor.sort.return_value = cursor
    return cursor


def test_initialization_uses_configured_collections():
    database = MagicMock()
    models, metrics = MagicMock(), MagicMock()
    database.__getitem__.side_effect = {"models": models, "history": metrics}.__getitem__
    settings = MongoDBSettings(
        logged_models_collection_name="models", run_metrics_collection_name="history"
    )

    repository = LoggedModelRepository(database, settings=settings)

    assert repository._settings is settings
    assert repository._collection is models
    assert repository._metrics_collection is metrics
    assert database.__getitem__.call_args_list == [call("models"), call("history")]
    assert models.create_index.call_count == 3
    assert metrics.create_index.call_count == 2


@pytest.mark.parametrize("stage", ["models", "model_index", "metrics", "metric_index"])
def test_initialization_translates_backend_failures(stage):
    failure = OperationFailure("Private initialization details")
    models, metrics = MagicMock(), MagicMock()
    database = MagicMock()

    def select(name):
        if (stage == "models" and name == "logged_models") or (
            stage == "metrics" and name == "run_metrics"
        ):
            raise failure
        return models if name == "logged_models" else metrics

    database.__getitem__.side_effect = select
    if stage == "model_index":
        models.create_index.side_effect = failure
    if stage == "metric_index":
        metrics.create_index.side_effect = failure

    with pytest.raises(RepositoryPersistenceError) as caught:
        LoggedModelRepository(database)
    assert str(caught.value) == "Unable to initialize logged model repository."
    assert caught.value.__cause__ is failure


def test_create_persists_k_v_arrays_and_initial_metadata(repository, collections):
    record = repository.create(**CREATE_ARGS)
    document = collections["logged_models"].insert_one.call_args.args[0]

    assert document["_id"] == record.model_id == MODEL_ID
    assert document["tags"] == [{"k": "team.name", "v": "$risk"}]
    assert document["params"] == [{"k": "depth", "v": "3"}]
    assert record.tags[0].key == "team.name"
    assert record.params[0].value == "3"
    assert record.creation_timestamp == record.last_updated_timestamp == 1000
    assert record.status_message is None
    assert "metrics" not in document


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_create_translates_database_failures(repository, collections, error_type):
    failure = error_type("Private database details")
    collections["logged_models"].insert_one.side_effect = failure

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.create(**CREATE_ARGS)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_find_translates_database_failures(repository, collections, error_type):
    failure = error_type("Private database details")
    collections["logged_models"].find_one.side_effect = failure

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.find_by_id(MODEL_ID)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_delete_translates_database_failures(repository, collections, error_type):
    failure = error_type("Private database details")
    collections["logged_models"].update_one.side_effect = failure

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.mark_deleted(model_id=MODEL_ID, last_updated_timestamp=2000)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_set_tags_translates_database_failures(repository, collections, error_type):
    failure = error_type("Private database details")
    collections["logged_models"].update_one.side_effect = failure

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.set_tags(model_id=MODEL_ID, tags={})
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_delete_tag_translates_database_failures(repository, collections, error_type):
    failure = error_type("Private database details")
    collections["logged_models"].find_one_and_update.side_effect = failure

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.delete_tag(model_id=MODEL_ID, key="team")
    assert caught.value.__cause__ is failure


def test_find_missing_and_decode_record(repository, collections):
    record = repository.find_by_id(MODEL_ID)
    assert record.model_id == MODEL_ID
    assert record.tags[0].key == "team"
    collections["logged_models"].find_one.return_value = None
    assert repository.find_by_id(MODEL_ID) is None


@pytest.mark.parametrize("failure_type", [KeyError, InvalidBSON])
def test_record_conversion_errors_are_not_translated(repository, monkeypatch, failure_type):
    failure = failure_type("Conversion failure")
    monkeypatch.setattr(model_module.LoggedModelRecord, "from_document", Mock(side_effect=failure))
    with pytest.raises(failure_type) as caught:
        repository.find_by_id(MODEL_ID)
    assert caught.value is failure


def test_delete_succeeds_when_existing_model_is_unchanged(repository, collections):
    collections["logged_models"].update_one.return_value = Mock(matched_count=1, modified_count=0)
    repository.mark_deleted(model_id=MODEL_ID, last_updated_timestamp=1000)


def test_delete_rejects_missing_model(repository, collections):
    collections["logged_models"].update_one.return_value = Mock(matched_count=0, modified_count=0)
    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.mark_deleted(model_id=MODEL_ID, last_updated_timestamp=1000)
    assert caught.value.args == (MODEL_ID,)


def test_set_tags_succeeds_when_existing_model_is_unchanged(repository, collections):
    collections["logged_models"].update_one.return_value = Mock(matched_count=1, modified_count=0)
    repository.set_tags(model_id=MODEL_ID, tags={})


def test_set_tags_rejects_missing_model(repository, collections):
    collections["logged_models"].update_one.return_value = Mock(matched_count=0, modified_count=0)
    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.set_tags(model_id=MODEL_ID, tags={})
    assert caught.value.args == (MODEL_ID,)


@pytest.mark.parametrize("document", [None, {"_id": MODEL_ID, "tags": [{"k": "other"}]}])
def test_delete_tag_distinguishes_missing_model_and_tag(repository, collections, document):
    collections["logged_models"].find_one_and_update.return_value = document
    error_type = RepositoryNotFoundError if document is None else RepositoryTagNotFoundError
    with pytest.raises(error_type):
        repository.delete_tag(model_id=MODEL_ID, key="team")
    kwargs = collections["logged_models"].find_one_and_update.call_args.kwargs
    assert kwargs["return_document"] == ReturnDocument.BEFORE
    collections["logged_models"].find_one.assert_not_called()


def test_delete_existing_tag_succeeds(repository, collections):
    collections["logged_models"].find_one_and_update.return_value = {
        "_id": MODEL_ID,
        "tags": [{"k": "team.name"}],
    }
    repository.delete_tag(model_id=MODEL_ID, key="team.name")


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidBSON])
def test_metric_history_partial_cursor_failures_are_translated_and_cursor_closed(
    repository, collections, error_type
):
    failure = error_type("Private deferred cursor details")

    def documents():
        yield {}
        raise failure

    cursor = _cursor([])
    cursor.__iter__.side_effect = documents
    collections["run_metrics"].find.return_value = cursor
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.get_metric_history(MODEL_ID)
    assert caught.value.__cause__ is failure
    cursor.__exit__.assert_called_once()


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_metric_history_cursor_creation_failures_are_translated(
    repository, collections, error_type
):
    failure = error_type("Private query details")
    collections["run_metrics"].find.side_effect = failure
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.get_metric_history(MODEL_ID)
    assert caught.value.__cause__ is failure


def test_metric_history_conversion_is_outside_boundary_and_cursor_is_closed(
    repository, collections, monkeypatch
):
    cursor = _cursor([{"model_id": MODEL_ID}])
    collections["run_metrics"].find.return_value = cursor
    error = InvalidBSON("Conversion failure")
    monkeypatch.setattr(model_module.RunMetricRecord, "from_document", Mock(side_effect=error))
    with pytest.raises(InvalidBSON) as caught:
        repository.get_metric_history(MODEL_ID)
    assert caught.value is error
    cursor.__exit__.assert_called_once()


def test_empty_metric_history_returns_empty_tuple(repository, collections):
    collections["run_metrics"].find.return_value = _cursor([])
    assert repository.get_metric_history(MODEL_ID) == ()


@pytest.mark.parametrize(
    "operation,builder",
    [
        ("set_tags", "build_merge_array_expression"),
        ("delete_tag", "build_remove_array_element_update"),
    ],
)
def test_update_builder_errors_are_not_reported_as_persistence_failures(
    repository, collections, monkeypatch, operation, builder
):
    error = ValueError("Invalid builder input")
    monkeypatch.setattr(model_module, builder, Mock(side_effect=error))
    with pytest.raises(ValueError) as caught:
        if operation == "set_tags":
            repository.set_tags(model_id=MODEL_ID, tags={"team": "risk"})
        else:
            repository.delete_tag(model_id=MODEL_ID, key="team")
    assert caught.value is error
    collections["logged_models"].update_one.assert_not_called()
    collections["logged_models"].find_one_and_update.assert_not_called()
