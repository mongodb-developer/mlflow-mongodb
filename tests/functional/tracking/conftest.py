from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from mlflow.entities import TraceInfo, TraceLocation, TraceState
from mlflow.tracking._tracking_service.utils import _get_store  # ruff: ignore[import-private-name]
from pymongo import MongoClient
from pymongo.errors import PyMongoError

if TYPE_CHECKING:
    from mlflow_mongodb.tracking.store import MongoDBTrackingStore

MONGODB_URI_ENV_VAR = "MONGODB_URI"
SERVER_SELECTION_TIMEOUT_MS = 3_000


def _clear_application_collections(store: MongoDBTrackingStore) -> None:
    store._database[store._settings.experiments_collection_name].delete_many({})
    store._database[store._settings.runs_collection_name].delete_many({})
    store._database[store._settings.run_metrics_collection_name].delete_many({})
    store._database[store._settings.logged_models_collection_name].delete_many({})
    store._database[store._settings.traces_collection_name].delete_many({})
    store._database[store._settings.spans_collection_name].delete_many({})
    store._database[store._settings.assessments_collection_name].delete_many({})


@pytest.fixture(scope="session")
def mongodb_uri() -> str:
    uri = os.environ.get(MONGODB_URI_ENV_VAR)
    if not uri:
        pytest.skip(f"Set {MONGODB_URI_ENV_VAR} to run MongoDB functional tests")
    return uri


@pytest.fixture(scope="session")
def mongodb_store(mongodb_uri: str) -> Iterator[MongoDBTrackingStore]:
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS)
    store = _get_store(store_uri=mongodb_uri, artifact_uri="s3://experiment-artifacts")
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
    store._run_repository
    store._logged_model_repository
    store._trace_repository

    try:
        yield store
    finally:
        try:
            client.drop_database(database.name)
        finally:
            client.close()


@pytest.fixture
def store(mongodb_store: MongoDBTrackingStore, clock) -> Iterator[MongoDBTrackingStore]:
    # Reopen the repository after each cleanup, as a client opening a fresh database would.
    mongodb_store.__dict__.pop("_experiment_repository", None)
    timestamp = clock["now"]
    clock["now"] = 0
    mongodb_store._experiment_repository
    clock["now"] = timestamp
    try:
        yield mongodb_store
    finally:
        try:
            mongodb_store.shut_down_async_logging()
        finally:
            _clear_application_collections(mongodb_store)


@pytest.fixture
def run(store: MongoDBTrackingStore):
    experiment_id = store.create_experiment("run-experiment")
    return store.create_run(
        experiment_id=experiment_id,
        user_id="test-user",
        start_time=1_700_000_000_000,
        tags=[],
        run_name="functional-run",
    )


@pytest.fixture
def clock(monkeypatch):
    current = {"now": 1_700_000_000_000}
    monkeypatch.setattr(
        "mlflow_mongodb.tracking.store.get_current_time_millis", lambda: current["now"]
    )
    return current


@pytest.fixture
def trace_info_factory(store):
    experiment_id = store.create_experiment("trace-experiment")

    def factory(**overrides):
        arguments = {
            "trace_id": f"tr-{uuid4().hex}",
            "trace_location": TraceLocation.from_experiment_id(experiment_id),
            "request_time": 1_700_000_000_000,
            "state": TraceState.OK,
            "execution_duration": 10,
        }
        arguments.update(overrides)
        return TraceInfo(**arguments)

    return factory


@pytest.fixture
def trace(store, trace_info_factory):
    return store.start_trace(trace_info_factory())
