import os
from collections.abc import Iterator

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from mlflow_mongodb.tracking.store import MongoDBTrackingStore

MONGODB_URI_ENV_VAR = "MONGODB_URI"
SERVER_SELECTION_TIMEOUT_MS = 3_000


def _clear_application_collections(store: MongoDBTrackingStore) -> None:
    store._database[store._settings.experiments_collection_name].delete_many({})


@pytest.fixture(scope="session")
def mongodb_uri() -> str:
    uri = os.environ.get(MONGODB_URI_ENV_VAR)
    if not uri:
        pytest.skip(f"Set {MONGODB_URI_ENV_VAR} to run MongoDB functional tests")
    return uri


@pytest.fixture(scope="session")
def mongodb_store(mongodb_uri: str) -> Iterator[MongoDBTrackingStore]:
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS)
    store = MongoDBTrackingStore(store_uri=mongodb_uri, artifact_uri="s3://experiment-artifacts")
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
    store._experiment_repository

    try:
        yield store
    finally:
        try:
            client.drop_database(database.name)
        finally:
            client.close()


@pytest.fixture
def store(mongodb_store: MongoDBTrackingStore) -> Iterator[MongoDBTrackingStore]:
    try:
        yield mongodb_store
    finally:
        _clear_application_collections(mongodb_store)


@pytest.fixture
def clock(monkeypatch):
    current = {"now": 1_700_000_000_000}
    monkeypatch.setattr(
        "mlflow_mongodb.tracking.store.get_current_time_millis", lambda: current["now"]
    )
    return current
