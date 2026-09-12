"""Fixtures for functional tests backed by a real MongoDB server."""

import os
from collections.abc import Iterator

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from mlflow_mongodb import MongoDBModelRegistryStore

MONGODB_URI_ENV_VAR = "MONGODB_URI"
SERVER_SELECTION_TIMEOUT_MS = 3_000


def _clear_application_collections(store: MongoDBModelRegistryStore) -> None:
    """Delete test documents without dropping collections or their indexes."""
    for collection_name in (
        store._settings.model_versions_collection_name,
        store._settings.registered_models_collection_name,
    ):
        store._database[collection_name].delete_many({})


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
    # Inject the client so the fixture controls its lifetime and can close it after the
    # functional test session.
    store.__dict__["_mongo_client"] = client

    try:
        database = store._database
    except Exception:
        client.close()
        raise

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
            client.drop_database(database.name)
        finally:
            client.close()


@pytest.fixture
def store(mongodb_store: MongoDBModelRegistryStore) -> Iterator[MongoDBModelRegistryStore]:
    """Return the shared real store and clean its documents after the test."""
    try:
        yield mongodb_store
    finally:
        _clear_application_collections(mongodb_store)
