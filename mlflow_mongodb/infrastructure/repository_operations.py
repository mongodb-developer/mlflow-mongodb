"""Visible error boundaries for repository database operations."""

from collections.abc import Iterator
from contextlib import contextmanager

from bson.errors import BSONError
from pymongo.errors import PyMongoError

from mlflow_mongodb.infrastructure.errors import RepositoryPersistenceError


@contextmanager
def repository_operation(
    message: str = "A database operation failed.",
    *,
    errors: tuple[type[Exception], ...] = (PyMongoError, BSONError),
) -> Iterator[None]:
    """Translate configured backend failures within an explicit database boundary."""
    try:
        yield
    except errors as exc:
        raise RepositoryPersistenceError(message) from exc
