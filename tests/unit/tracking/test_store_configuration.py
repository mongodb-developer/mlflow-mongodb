from unittest.mock import MagicMock, Mock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from pymongo.errors import ConfigurationError

from mlflow_mongodb.tracking.store import MongoDBTrackingStore


def test_mongo_client_is_lazy_and_cached(monkeypatch):
    client = Mock()
    client_factory = Mock(return_value=client)
    monkeypatch.setattr("mlflow_mongodb.tracking.store.MongoClient", client_factory)
    store = MongoDBTrackingStore("mongodb://localhost:27017/mlflow")

    client_factory.assert_not_called()

    assert store._mongo_client is client
    assert store._mongo_client is client
    client_factory.assert_called_once_with("mongodb://localhost:27017/mlflow")


def test_mongo_client_requires_tracking_uri():
    store = MongoDBTrackingStore()

    with pytest.raises(MlflowException, match="MongoDB tracking URI is required") as caught:
        store._mongo_client

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_mongo_client_translates_invalid_uri(monkeypatch):
    client_factory = Mock(side_effect=ConfigurationError("Private URI details"))
    monkeypatch.setattr("mlflow_mongodb.tracking.store.MongoClient", client_factory)
    store = MongoDBTrackingStore("not-mongodb://localhost/mlflow")

    with pytest.raises(MlflowException) as caught:
        store._mongo_client

    assert caught.value.message == "Invalid MongoDB tracking URI."
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    client_factory.assert_called_once_with("not-mongodb://localhost/mlflow")


def test_database_requires_name_in_uri():
    client = Mock()
    client.get_default_database.side_effect = ConfigurationError("Private database details")
    store = MongoDBTrackingStore("mongodb://localhost:27017")
    store.__dict__["_mongo_client"] = client

    with pytest.raises(MlflowException, match="must include a database name") as caught:
        store._database

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    client.get_default_database.assert_called_once_with()


def test_database_is_lazy_and_cached():
    database = Mock()
    client = Mock()
    client.get_default_database.return_value = database
    store = MongoDBTrackingStore("mongodb://localhost:27017/mlflow")
    store.__dict__["_mongo_client"] = client

    client.get_default_database.assert_not_called()

    assert store._database is database
    assert store._database is database
    client.get_default_database.assert_called_once_with()


def test_store_uses_configured_experiment_collection(monkeypatch):
    monkeypatch.setenv("MLFLOW_MONGODB_EXPERIMENTS_COLLECTION", "custom_experiments")
    database = MagicMock()
    collection = MagicMock()
    database.__getitem__.return_value = collection
    store = MongoDBTrackingStore("mongodb://localhost:27017/mlflow")
    store.__dict__["_database"] = database

    repository = store._experiment_repository

    assert repository._collection is collection
    assert store._experiment_repository is repository
    database.__getitem__.assert_called_once_with("custom_experiments")
    assert collection.create_index.call_count == 2


def test_store_rejects_invalid_configured_experiment_collection(monkeypatch):
    monkeypatch.setenv("MLFLOW_MONGODB_EXPERIMENTS_COLLECTION", "invalid$name")

    with pytest.raises(ValueError, match="Invalid MongoDB collection name"):
        MongoDBTrackingStore("mongodb://localhost:27017/mlflow")
