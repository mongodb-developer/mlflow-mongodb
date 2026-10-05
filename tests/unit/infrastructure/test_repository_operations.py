"""Unit tests for the shared repository persistence boundary."""

import pytest
from bson.errors import InvalidBSON, InvalidDocument
from pymongo.errors import (
    AutoReconnect,
    DocumentTooLarge,
    DuplicateKeyError,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from mlflow_mongodb.infrastructure.errors import RepositoryPersistenceError
from mlflow_mongodb.infrastructure.repository_operations import repository_operation


def test_successful_operation_completes():
    documents = []

    with repository_operation():
        documents.append({"name": "example-model"})

    assert documents == [{"name": "example-model"}]


@pytest.mark.parametrize(
    "error_type",
    [
        OperationFailure,
        AutoReconnect,
        ServerSelectionTimeoutError,
        InvalidDocument,
        InvalidBSON,
        DocumentTooLarge,
        DuplicateKeyError,
    ],
)
def test_backend_failures_are_translated_without_logging(error_type, caplog):
    backend_error = error_type("private database details")

    with pytest.raises(RepositoryPersistenceError) as caught, repository_operation():
        raise backend_error

    assert str(caught.value) == "A database operation failed."
    assert caught.value.__cause__ is backend_error
    assert not caplog.records


def test_operation_message_is_preserved():
    backend_error = OperationFailure("private database details")

    with (
        pytest.raises(RepositoryPersistenceError) as caught,
        repository_operation("Unable to read registered model."),
    ):
        raise backend_error

    assert str(caught.value) == "Unable to read registered model."
    assert caught.value.__cause__ is backend_error


@pytest.mark.parametrize(
    "error_type", [ValueError, KeyError, TypeError, RepositoryPersistenceError]
)
def test_other_errors_pass_through(error_type):
    original_error = error_type("operation failed")

    with pytest.raises(error_type) as caught, repository_operation():
        raise original_error

    assert caught.value is original_error


def test_driver_only_boundary_does_not_translate_bson_errors():
    original_error = InvalidBSON("BSON failure outside a database command")

    with pytest.raises(InvalidBSON) as caught, repository_operation(errors=(PyMongoError,)):
        raise original_error

    assert caught.value is original_error


def test_driver_only_boundary_translates_driver_errors():
    backend_error = OperationFailure("private database details")

    with (
        pytest.raises(RepositoryPersistenceError) as caught,
        repository_operation(errors=(PyMongoError,)),
    ):
        raise backend_error

    assert caught.value.__cause__ is backend_error
