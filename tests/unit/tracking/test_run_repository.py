"""Run repository domain behavior and visible persistence boundaries."""

from unittest.mock import MagicMock, Mock, call

import pytest
from bson.errors import InvalidBSON, InvalidDocument
from mlflow.entities import LifecycleStage
from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError, OperationFailure

import mlflow_mongodb.tracking.repositories.runs as run_module
from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotActiveError,
    RepositoryNotFoundError,
    RepositoryPersistenceError,
)
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.repositories import RunRepository

RUN_ID = "a" * 32
CREATE_ARGS = {
    "run_id": RUN_ID,
    "experiment_id": "42",
    "name": "example-run",
    "artifact_uri": "s3://artifacts/run",
    "user_id": "test-user",
    "status": "RUNNING",
    "start_time": 1000,
    "lifecycle_stage": LifecycleStage.ACTIVE,
    "tags": {"team": "risk"},
}


@pytest.fixture
def collections():
    return {"runs": MagicMock(), "run_metrics": MagicMock()}


@pytest.fixture
def repository(collections, run_document_factory):
    database = MagicMock()
    database.__getitem__.side_effect = collections.__getitem__
    collections["runs"].find_one.return_value = run_document_factory(_id=RUN_ID)
    collections["runs"].find_one_and_update.return_value = run_document_factory(_id=RUN_ID)
    return RunRepository(database)


def test_initialization_selects_configured_collections_and_creates_indexes():
    runs = MagicMock()
    metrics = MagicMock()
    database = MagicMock()
    database.__getitem__.side_effect = {"custom_runs": runs, "custom_metrics": metrics}.__getitem__
    settings = MongoDBSettings(
        runs_collection_name="custom_runs", run_metrics_collection_name="custom_metrics"
    )

    repository = RunRepository(database, settings=settings)

    assert repository._settings is settings
    assert repository._collection is runs
    assert repository._metrics_collection is metrics
    assert database.__getitem__.call_args_list == [call("custom_runs"), call("custom_metrics")]
    runs.create_index.assert_called_once_with(
        [("experiment_id", ASCENDING)], name=RunRepository.EXPERIMENT_INDEX
    )
    assert metrics.create_index.call_args_list == [
        call(
            [("run_id", ASCENDING), ("k", ASCENDING), ("timestamp", ASCENDING)],
            name="run_metrics_run_k_timestamp",
            partialFilterExpression={"scope": "run"},
        ),
        call(
            [
                (field, ASCENDING)
                for field in (
                    "run_id",
                    "k",
                    "timestamp",
                    "step",
                    "v",
                )
            ],
            name="run_metrics_identity",
            unique=True,
            partialFilterExpression={"scope": "run"},
        ),
        call(
            [
                ("model_id", ASCENDING),
                ("run_id", ASCENDING),
                ("k", ASCENDING),
                ("timestamp", ASCENDING),
                ("step", ASCENDING),
            ],
            name="model_metrics_identity",
            unique=True,
            partialFilterExpression={"scope": "model"},
        ),
    ]


@pytest.mark.parametrize("step", ["runs", "runs_index", "run_metrics", "metrics_index"])
def test_initialization_translates_failure_at_either_collection_or_index(step):
    backend_error = OperationFailure("Private collection/index details")
    collections = {"runs": MagicMock(), "run_metrics": MagicMock()}
    database = MagicMock()

    def select(name):
        if name == step:
            raise backend_error
        return collections[name]

    database.__getitem__.side_effect = select
    if step == "runs_index":
        collections["runs"].create_index.side_effect = backend_error
    elif step == "metrics_index":
        collections["run_metrics"].create_index.side_effect = backend_error

    with pytest.raises(RepositoryPersistenceError) as caught:
        RunRepository(database)

    assert str(caught.value) == "Unable to initialize run repository."
    assert caught.value.__cause__ is backend_error


def test_settings_validation_is_not_reported_as_persistence_failure(monkeypatch):
    failure = ValueError("Invalid MongoDB collection name")
    monkeypatch.setattr(run_module, "MongoDBSettings", Mock(side_effect=failure))
    database = MagicMock()

    with pytest.raises(ValueError) as caught:
        RunRepository(database)

    assert caught.value is failure
    database.__getitem__.assert_not_called()


@pytest.mark.parametrize("error_type", [DuplicateKeyError, OperationFailure, InvalidDocument])
def test_create_preserves_duplicate_identity_and_translates_other_backend_failures(
    repository, collections, error_type
):
    backend_error = error_type("Private write details")
    collections["runs"].insert_one.side_effect = backend_error
    expected_type = (
        RepositoryAlreadyExistsError
        if error_type is DuplicateKeyError
        else RepositoryPersistenceError
    )

    with pytest.raises(expected_type) as caught:
        repository.create(**CREATE_ARGS)

    assert caught.value.__cause__ is backend_error
    if error_type is DuplicateKeyError:
        assert caught.value.args == (RUN_ID,)


def test_create_returns_record_with_initialized_run_fields(repository, collections):
    record = repository.create(**CREATE_ARGS)

    document = collections["runs"].insert_one.call_args.args[0]
    assert record.run_id == RUN_ID
    assert record.name == CREATE_ARGS["name"]
    assert record.params == ()
    assert document["params"] == []
    assert document["metrics"] == []
    assert document["inputs"] == {"datasets": [], "models": []}
    assert document["outputs"] == []
    assert document["tags"] == [{"k": "team", "v": "risk"}]


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidBSON])
def test_find_translates_driver_and_response_decoding_failures(repository, collections, error_type):
    backend_error = error_type("Private read details")
    collections["runs"].find_one.side_effect = backend_error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.find_by_id(RUN_ID)

    assert str(caught.value) == "Unable to get run."
    assert caught.value.__cause__ is backend_error


def test_find_decodes_stored_parameters(repository, collections, run_document_factory):
    collections["runs"].find_one.return_value = run_document_factory(
        params=[{"k": "depth", "v": "3"}, {"k": "label", "v": "$literal"}]
    )

    record = repository.find_by_id(RUN_ID)

    assert [(param.key, param.value) for param in record.params] == [
        ("depth", "3"),
        ("label", "$literal"),
    ]


def test_find_returns_none_for_missing_run(repository, collections):
    collections["runs"].find_one.return_value = None

    assert repository.find_by_id(RUN_ID) is None


@pytest.mark.parametrize("error_type", [KeyError, InvalidBSON])
def test_record_conversion_is_outside_persistence_translation(
    repository, collections, monkeypatch, error_type
):
    failure = error_type("DTO conversion failure")
    monkeypatch.setattr(run_module.RunRecord, "from_document", Mock(side_effect=failure))

    with pytest.raises(error_type) as caught:
        repository.find_by_id(RUN_ID)

    assert caught.value is failure
    collections["runs"].find_one.assert_called_once_with({"_id": RUN_ID})


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidBSON])
def test_metric_history_translates_failure_after_partial_cursor_iteration(
    repository, collections, monkeypatch, error_type
):
    backend_error = error_type("Private cursor details")
    convert = Mock()
    monkeypatch.setattr(run_module.RunMetricRecord, "from_document", convert)

    def iterate():
        yield {"k": "loss", "v": 0.4, "timestamp": 1000, "step": 1}
        raise backend_error

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.__iter__.side_effect = iterate
    collections["run_metrics"].find.return_value = cursor

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.get_metric_history(run_id=RUN_ID, metric_key="loss", offset=2, limit=3)

    assert caught.value.__cause__ is backend_error
    convert.assert_not_called()
    cursor.__exit__.assert_called_once()
    cursor.sort.assert_called_once_with(
        [("timestamp", ASCENDING), ("step", ASCENDING), ("v", ASCENDING), ("_id", ASCENDING)]
    )
    cursor.skip.assert_called_once_with(2)
    cursor.limit.assert_called_once_with(3)


def test_metric_history_converts_materialized_documents(repository, collections):
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.__iter__.return_value = iter(
        [{"k": "loss", "v": 0.4, "timestamp": 1000, "step": 1, "run_id": RUN_ID}]
    )
    collections["run_metrics"].find.return_value = cursor

    records = repository.get_metric_history(run_id=RUN_ID, metric_key="loss")

    assert [(record.key, record.value, record.run_id) for record in records] == [
        ("loss", 0.4, RUN_ID)
    ]
    collections["run_metrics"].find.assert_called_once_with(
        {"scope": "run", "run_id": RUN_ID, "k": "loss"}
    )
    cursor.limit.assert_not_called()


@pytest.mark.parametrize("method", ["mark_deleted", "restore"])
def test_lifecycle_updates_match_run_id_without_a_stage_precondition(
    repository, collections, run_document_factory, method
):
    stage = LifecycleStage.DELETED if method == "mark_deleted" else LifecycleStage.ACTIVE
    deleted_time = 2000 if method == "mark_deleted" else None
    collections["runs"].find_one_and_update.return_value = run_document_factory(
        _id=RUN_ID, lifecycle_stage=stage, deleted_time=deleted_time
    )
    kwargs = {"run_id": RUN_ID}
    if method == "mark_deleted":
        kwargs["deleted_time"] = deleted_time

    record = getattr(repository, method)(**kwargs)

    assert record.lifecycle_stage == stage
    collections["runs"].find_one_and_update.assert_called_once_with(
        {"_id": RUN_ID},
        {"$set": {"lifecycle_stage": stage, "deleted_time": deleted_time}},
        return_document=ReturnDocument.AFTER,
    )


@pytest.mark.parametrize("method", ["mark_deleted", "restore"])
def test_lifecycle_database_failures_are_translated_by_the_shared_transition(
    repository, collections, method
):
    backend_error = OperationFailure("Private lifecycle details")
    collections["runs"].find_one_and_update.side_effect = backend_error
    kwargs = {"run_id": RUN_ID}
    if method == "mark_deleted":
        kwargs["deleted_time"] = 2000

    with pytest.raises(RepositoryPersistenceError) as caught:
        getattr(repository, method)(**kwargs)

    assert str(caught.value) == "Unable to update run lifecycle."
    assert caught.value.__cause__ is backend_error


@pytest.mark.parametrize("method", ["mark_deleted", "restore"])
def test_lifecycle_transitions_report_missing_match_as_domain_error(
    repository, collections, method
):
    collections["runs"].find_one_and_update.return_value = None
    kwargs = {"run_id": RUN_ID}
    if method == "mark_deleted":
        kwargs["deleted_time"] = 2000

    with pytest.raises(RepositoryNotFoundError) as caught:
        getattr(repository, method)(**kwargs)

    assert caught.value.args == (RUN_ID,)


def test_update_builder_failure_passes_through_before_database_write(
    repository, collections, monkeypatch
):
    failure = InvalidDocument("Update builder failure")
    monkeypatch.setattr(run_module, "build_merge_array_expression", Mock(side_effect=failure))

    with pytest.raises(InvalidDocument) as caught:
        repository.update_info(run_id=RUN_ID, status=None, end_time=None, run_name="$name")

    assert caught.value is failure
    collections["runs"].find_one_and_update.assert_not_called()


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_update_info_translates_database_failure(repository, collections, error_type):
    backend_error = error_type("Private update details")
    collections["runs"].find_one_and_update.side_effect = backend_error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.update_info(run_id=RUN_ID, status="FINISHED", end_time=2000, run_name=None)

    assert str(caught.value) == "Unable to update run info."
    assert caught.value.__cause__ is backend_error


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_batch_update_failure_prevents_metric_insertion(repository, collections, error_type):
    backend_error = error_type("Private batch update details")
    collections["runs"].find_one_and_update.side_effect = backend_error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.log_batch(
            run_id=RUN_ID,
            metrics=[{"k": "loss", "v": 0.4, "timestamp": 1000, "step": 1}],
            params=[],
            tags=[{"k": "team", "v": "risk"}],
        )

    assert str(caught.value) == "Unable to update run batch data."
    assert caught.value.__cause__ is backend_error
    collections["runs"].find_one.assert_not_called()
    collections["run_metrics"].bulk_write.assert_not_called()


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_batch_metric_insertion_failure_follows_run_update(repository, collections, error_type):
    backend_error = error_type("Private metric insertion details")
    collections["run_metrics"].bulk_write.side_effect = backend_error
    operations = Mock()
    operations.attach_mock(collections["runs"].find_one_and_update, "update")
    operations.attach_mock(collections["run_metrics"].bulk_write, "insert")

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.log_batch(
            run_id=RUN_ID,
            metrics=[{"k": "loss", "v": 0.4, "timestamp": 1000, "step": 1}],
            params=[],
            tags=[],
        )

    assert str(caught.value) == "Unable to insert run batch metrics."
    assert caught.value.__cause__ is backend_error
    assert [operation[0] for operation in operations.mock_calls] == ["update", "insert"]
    collections["runs"].find_one.assert_not_called()


@pytest.mark.parametrize("method", ["log_inputs", "log_outputs"])
@pytest.mark.parametrize("previous", [None, {"lifecycle_stage": LifecycleStage.DELETED}])
def test_input_output_operations_distinguish_missing_and_inactive_runs(
    repository, collections, method, previous
):
    collections["runs"].find_one_and_update.return_value = previous
    kwargs = {"run_id": RUN_ID, "models": []}
    if method == "log_inputs":
        kwargs["datasets"] = []
    expected_type = RepositoryNotFoundError if previous is None else RepositoryNotActiveError

    with pytest.raises(expected_type) as caught:
        getattr(repository, method)(**kwargs)

    expected_args = (RUN_ID,) if previous is None else (RUN_ID, LifecycleStage.DELETED)
    assert caught.value.args == expected_args
    args, kwargs = collections["runs"].find_one_and_update.call_args
    assert args[0] == {"_id": RUN_ID}
    assert kwargs["return_document"] == ReturnDocument.BEFORE


@pytest.mark.parametrize("method", ["log_inputs", "log_outputs"])
def test_input_output_backend_failures_are_translated(repository, collections, method):
    backend_error = InvalidDocument("Private pipeline encoding details")
    collections["runs"].find_one_and_update.side_effect = backend_error
    kwargs = {"run_id": RUN_ID, "models": []}
    if method == "log_inputs":
        kwargs["datasets"] = []

    with pytest.raises(RepositoryPersistenceError) as caught:
        getattr(repository, method)(**kwargs)

    assert caught.value.__cause__ is backend_error
