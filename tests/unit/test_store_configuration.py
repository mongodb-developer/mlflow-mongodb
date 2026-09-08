"""Unit tests for MongoDB store configuration."""

import pytest
from mlflow.exceptions import MlflowException

from mlflow_mongodb import MongoDBModelRegistryStore


def test_mongo_client_requires_registry_uri():
    store = MongoDBModelRegistryStore()

    with pytest.raises(MlflowException, match="MongoDB registry URI is required"):
        store._mongo_client


def test_mongo_client_rejects_non_mongodb_uri():
    store = MongoDBModelRegistryStore("not-mongodb://localhost/mlflow")

    with pytest.raises(MlflowException, match="Invalid MongoDB registry URI"):
        store._mongo_client


def test_database_requires_name_in_uri():
    store = MongoDBModelRegistryStore("mongodb://localhost:27017")

    with pytest.raises(MlflowException, match="must include a database name"):
        store._database


def test_database_uses_name_from_uri_without_connecting():
    store = MongoDBModelRegistryStore("mongodb://localhost:27017/mlflow")

    assert store._database.name == "mlflow"
