from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, ErrorCode

import mlflow_mongodb.tracking.store as tracking_store
from mlflow_mongodb.infrastructure.errors import RepositoryPersistenceError
from mlflow_mongodb.tracking.store import MongoDBTrackingStore


def test_create_experiment_sanitizes_and_logs_persistence_failure(
    store, experiment_repository, caplog
):
    persistence_error = RepositoryPersistenceError("Private connection and query details")
    experiment_repository.create.side_effect = persistence_error

    with pytest.raises(MlflowException) as caught:
        store.create_experiment("example-experiment")

    experiment_repository.create.assert_called_once()
    assert caught.value.message == "Unable to create experiment."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    records = [record for record in caplog.records if record.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].args == (persistence_error,)


def test_repository_initialization_failure_is_logged_once(monkeypatch, caplog):
    persistence_error = RepositoryPersistenceError("Private initialization details")
    initialize = Mock(side_effect=persistence_error)
    monkeypatch.setattr(tracking_store, "ExperimentRepository", initialize)
    store = MongoDBTrackingStore("mongodb://localhost:27017/mlflow")
    database = Mock()
    store.__dict__["_database"] = database

    with pytest.raises(MlflowException) as caught:
        store.search_experiments()

    initialize.assert_called_once_with(database, settings=store._settings)
    assert caught.value.message == "Unable to initialize experiment repository."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    records = [record for record in caplog.records if record.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].args == (persistence_error,)
