"""Unit tests for shared error translation at public store boundaries."""

import inspect
import traceback
from logging import Logger
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE, ErrorCode
from pymongo.errors import OperationFailure

from mlflow_mongodb.infrastructure.errors import RepositoryPersistenceError
from mlflow_mongodb.infrastructure.store_errors import handle_persistence_error


def test_success_preserves_result_arguments_and_function_metadata():
    logger = Mock(spec=Logger)
    result = object()

    def operation(name, *, description):
        """A witness store operation."""
        assert name == "example-model"
        assert description == "description"
        return result

    decorated = handle_persistence_error("Unable to create model.", logger=logger)(operation)

    assert decorated("example-model", description="description") is result
    assert decorated.__name__ == operation.__name__
    assert decorated.__doc__ == operation.__doc__
    assert inspect.signature(decorated) == inspect.signature(operation)
    logger.error.assert_not_called()


def test_persistence_failure_is_logged_and_translated_without_leaking_private_details():
    logger = Mock(spec=Logger)
    backend_error = OperationFailure("Private backend failure: host=mongodb.internal")
    persistence_error = RepositoryPersistenceError("Private repository failure: database=registry")
    persistence_error.__cause__ = backend_error

    @handle_persistence_error("Unable to create model.", logger=logger)
    def operation():
        raise persistence_error

    with pytest.raises(MlflowException) as caught:
        operation()

    assert caught.value.message == "Unable to create model."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    public_traceback = "".join(traceback.format_exception(caught.value))
    assert str(persistence_error) not in public_traceback
    assert str(backend_error) not in public_traceback
    logger.error.assert_called_once_with("Unable to create model: %s", persistence_error)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ValueError("Invalid input"), id="validation"),
        pytest.param(MlflowException("Invalid input", INVALID_PARAMETER_VALUE), id="mlflow-error"),
    ],
)
def test_other_errors_pass_through_without_logging(error):
    logger = Mock(spec=Logger)

    @handle_persistence_error("Unable to create model.", logger=logger)
    def operation():
        raise error

    with pytest.raises(type(error)) as caught:
        operation()

    assert caught.value is error
    logger.error.assert_not_called()
