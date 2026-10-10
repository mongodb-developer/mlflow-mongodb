"""Unit tests for persistence failures at the public registry store boundary."""

from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, ErrorCode

import mlflow_mongodb.model_registry.store as registry_store
from mlflow_mongodb import MongoDBModelRegistryStore
from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryPersistenceError,
)


def test_create_model_version_handles_failure_after_version_allocation(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
):
    registered_model_repository.find_by_name.return_value = registered_model_record_factory()
    registered_model_repository.allocate_next_version.return_value = 1
    model_version_repository.create.side_effect = RepositoryPersistenceError(
        "Private insert failure"
    )

    with pytest.raises(MlflowException) as caught:
        store.create_model_version("example-model", "s3://artifacts/model")

    registered_model_repository.allocate_next_version.assert_called_once()
    model_version_repository.create.assert_called_once()
    assert caught.value.message == "Unable to create model version."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


def test_duplicate_model_recovery_translates_lookup_failure(store, registered_model_repository):
    registered_model_repository.create.side_effect = RepositoryAlreadyExistsError("example-model")
    registered_model_repository.find_by_name.side_effect = RepositoryPersistenceError(
        "Private recovery lookup failure"
    )

    with pytest.raises(MlflowException) as caught:
        store.create_registered_model("example-model")

    registered_model_repository.find_by_name.assert_called_once_with("example-model")
    assert caught.value.message == "Unable to create registered model."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


@pytest.mark.parametrize(
    "constructor, operation, message",
    [
        pytest.param(
            "RegisteredModelRepository",
            lambda store: store.create_registered_model("example-model"),
            "Unable to initialize registered-model repository.",
            id="registered-models",
        ),
        pytest.param(
            "ModelVersionRepository",
            lambda store: store.search_model_versions(),
            "Unable to initialize model-version repository.",
            id="model-versions",
        ),
    ],
)
def test_repository_initialization_is_logged_once_through_public_operation(
    monkeypatch,
    caplog,
    constructor,
    operation,
    message,
):
    persistence_error = RepositoryPersistenceError("Private initialization failure")
    initialize = Mock(side_effect=persistence_error)
    monkeypatch.setattr(registry_store, constructor, initialize)
    store = MongoDBModelRegistryStore("mongodb://localhost:27017/mlflow")
    store.__dict__["_database"] = Mock()

    with pytest.raises(MlflowException) as caught:
        operation(store)

    initialize.assert_called_once()
    assert caught.value.message == message
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    records = [record for record in caplog.records if record.name == registry_store.logger.name]
    assert len(records) == 1
    assert records[0].args == (persistence_error,)
