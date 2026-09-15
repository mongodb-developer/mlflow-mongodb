"""Unit tests for MongoDB store configuration."""

from unittest.mock import MagicMock, call

import pytest
from mlflow.exceptions import MlflowException

from mlflow_mongodb import MongoDBModelRegistryStore
from mlflow_mongodb.settings import MongoDBSettings


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


def test_store_uses_configured_collection_names(monkeypatch):
    registered_models_collection = "custom_registered_models"
    model_versions_collection = "custom_model_versions"
    monkeypatch.setenv(
        "MLFLOW_MONGODB_REGISTERED_MODELS_COLLECTION",
        registered_models_collection,
    )
    monkeypatch.setenv(
        "MLFLOW_MONGODB_MODEL_VERSIONS_COLLECTION",
        model_versions_collection,
    )
    database = MagicMock()
    collections = {}
    database.__getitem__.side_effect = lambda name: collections.setdefault(name, MagicMock())
    store = MongoDBModelRegistryStore("mongodb://localhost:27017/mlflow")
    store.__dict__["_database"] = database

    registered_model_repository = store._registered_model_repository
    model_version_repository = store._model_version_repository

    assert registered_model_repository._collection is collections[registered_models_collection]
    assert model_version_repository._collection is collections[model_versions_collection]
    assert (
        model_version_repository._registered_models_collection
        is collections[registered_models_collection]
    )
    assert database.__getitem__.call_args_list == [
        call(registered_models_collection),
        call(model_versions_collection),
        call(registered_models_collection),
    ]


@pytest.mark.parametrize(
    "invalid_name",
    ["", "invalid\x00name", "invalid$name", "system.models"],
)
@pytest.mark.parametrize(
    "setting_field",
    [
        "registered_models_collection_name",
        "model_versions_collection_name",
    ],
)
def test_mongodb_settings_reject_invalid_collection_names(setting_field, invalid_name):
    with pytest.raises(ValueError, match="Invalid MongoDB collection name"):
        MongoDBSettings(**{setting_field: invalid_name})


@pytest.mark.parametrize(
    "environment_variable",
    [
        "MLFLOW_MONGODB_REGISTERED_MODELS_COLLECTION",
        "MLFLOW_MONGODB_MODEL_VERSIONS_COLLECTION",
    ],
)
def test_store_rejects_invalid_configured_collection_name(monkeypatch, environment_variable):
    monkeypatch.setenv(environment_variable, "invalid$name")

    with pytest.raises(ValueError, match="Invalid MongoDB collection name"):
        MongoDBModelRegistryStore("mongodb://localhost:27017/mlflow")
