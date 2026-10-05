"""Shared error translation at public store boundaries."""

from functools import wraps
from logging import Logger

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

from mlflow_mongodb.infrastructure.errors import RepositoryPersistenceError


def handle_persistence_error(message: str, *, logger: Logger):
    """Log repository persistence failures and translate them into safe MLflow errors."""
    log_message = f"{message.removesuffix('.')}: %s"

    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RepositoryPersistenceError as exc:
                logger.error(log_message, exc)
                raise MlflowException(message, INTERNAL_ERROR) from None

        return wrapped

    return decorate
