"""Fixtures for functional tests backed by a real MongoDB server."""

import os
import re
import threading
from collections.abc import Iterator

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from mlflow_mongodb import MongoDBModelRegistryStore
from mlflow_mongodb.repositories import (
    ModelVersionRepository,
    RegisteredModelRepository,
)

MONGODB_URI_ENV_VAR = "MONGODB_URI"
SERVER_SELECTION_TIMEOUT_MS = 3_000
TEST_DATABASE_PATTERN = re.compile(r"(^|[-_])test($|[-_])", re.IGNORECASE)
APPLICATION_COLLECTIONS = (
    ModelVersionRepository.COLLECTION_NAME,
    RegisteredModelRepository.COLLECTION_NAME,
)


def _clear_application_collections(store: MongoDBModelRegistryStore) -> None:
    """Delete test documents without dropping collections or their indexes."""
    for collection_name in APPLICATION_COLLECTIONS:
        store._database[collection_name].delete_many({})


def _wait_for_prompt_linking_threads() -> None:
    """Wait for MLflow's asynchronous prompt-linking work before database cleanup."""
    for thread in threading.enumerate():
        if thread.name.startswith("link_prompt_to_experiment_thread"):
            thread.join(timeout=5)
            if thread.is_alive():
                raise TimeoutError(f"Thread {thread.name} did not complete within 5 seconds")


@pytest.fixture(scope="session")
def mongodb_uri() -> str:
    """Return the URI of the dedicated functional-test database."""
    uri = os.environ.get(MONGODB_URI_ENV_VAR)
    if not uri:
        pytest.skip(f"Set {MONGODB_URI_ENV_VAR} to run MongoDB functional tests")
    return uri


@pytest.fixture(scope="session")
def mongodb_store(mongodb_uri: str) -> Iterator[MongoDBModelRegistryStore]:
    """Create one real store and initialize its collections once per test session."""
    client = MongoClient(
        mongodb_uri,
        serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
    )
    store = MongoDBModelRegistryStore(store_uri=mongodb_uri)
    store.__dict__["_mongo_client"] = client

    try:
        database = store._database
    except Exception:
        client.close()
        raise

    if not TEST_DATABASE_PATTERN.search(database.name):
        client.close()
        pytest.fail(
            f"Refusing to clean MongoDB database {database.name!r}. "
            "The functional-test database name must contain a distinct 'test' segment, "
            "for example 'mlflow_functional_test'."
        )

    try:
        client.admin.command("ping")
        server_version = tuple(client.server_info()["versionArray"][:2])
    except PyMongoError as exc:
        client.close()
        pytest.fail(f"Cannot connect to MongoDB using {MONGODB_URI_ENV_VAR}: {exc}")

    if server_version < (8, 0):
        client.close()
        pytest.fail(
            "MongoDB 8.0 or newer is required; "
            f"the functional-test server reports {server_version[0]}.{server_version[1]}"
        )

    _clear_application_collections(store)

    # Repository construction creates the production indexes. Do this once, outside any
    # future test transaction, then preserve the indexes between tests.
    store._registered_model_repository
    store._model_version_repository

    try:
        yield store
    finally:
        try:
            _wait_for_prompt_linking_threads()
        finally:
            try:
                client.drop_database(database.name)
            finally:
                client.close()


@pytest.fixture
def store(mongodb_store: MongoDBModelRegistryStore) -> Iterator[MongoDBModelRegistryStore]:
    """Return the shared real store and clean its documents after the test."""
    try:
        yield mongodb_store
    finally:
        try:
            _wait_for_prompt_linking_threads()
        finally:
            _clear_application_collections(mongodb_store)
