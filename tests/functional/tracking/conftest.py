from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from mlflow.entities import TraceInfo, TraceLocation, TraceState
from mlflow.entities.assessment import Expectation
from mlflow.entities.span import Span
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey, TraceSizeStatsKey
from mlflow.tracking._tracking_service.utils import _get_store  # ruff: ignore[import-private-name]
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, Status, StatusCode, TraceFlags
from pymongo import MongoClient
from pymongo.errors import PyMongoError

if TYPE_CHECKING:
    from mlflow_mongodb.tracking.store import MongoDBTrackingStore

MONGODB_URI_ENV_VAR = "MONGODB_URI"
SERVER_SELECTION_TIMEOUT_MS = 3_000


def _clear_application_collections(store: MongoDBTrackingStore) -> None:
    store._database[store._settings.experiments_collection_name].delete_many({})
    store._database[store._settings.experiment_id_counters_collection_name].delete_many({})
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


@pytest.fixture
def persisted_trace_factory(store, trace_info_factory):
    """Seed span documents for read/delete coverage before span ingestion is implemented."""

    def factory(*, span_count=3, **overrides):
        overrides.setdefault(
            "trace_metadata",
            {TraceMetadataKey.SIZE_STATS: json.dumps({TraceSizeStatsKey.NUM_SPANS: span_count})},
        )
        info = trace_info_factory(**overrides)
        assessment = Expectation(
            name="answer", value={"expected": [1, True]}, trace_id=info.trace_id
        )
        assessment.assessment_id = f"a-{info.trace_id}"
        info.assessments = [assessment]
        info = store.start_trace(info)
        otel_trace_id = int(info.trace_id.removeprefix("tr-"), 16)
        root_context = SpanContext(otel_trace_id, 1, False, TraceFlags(1))
        start_time = info.request_time * 1_000_000
        spans = [
            Span(
                ReadableSpan(
                    name=f"span-{index}",
                    context=SpanContext(otel_trace_id, index + 1, False, TraceFlags(1)),
                    parent=root_context if index else None,
                    # Put the root after its children by time to exercise root-first reads.
                    start_time=start_time + (2 if index == 0 else 0),
                    end_time=start_time + 10,
                    attributes={
                        SpanAttributeKey.REQUEST_ID: json.dumps(info.trace_id),
                        SpanAttributeKey.INPUTS: json.dumps({"prompt": "$literal"}),
                        SpanAttributeKey.OUTPUTS: json.dumps({"answer": [1, True]}),
                    },
                    status=Status(StatusCode.OK),
                    resource=Resource.get_empty(),
                )
            )
            for index in range(span_count)
        ]
        documents = [
            {
                "trace_id": info.trace_id,
                "span_id": span.span_id,
                "parent_span_id": span.parent_id,
                "start_time_ns": span.start_time_ns,
                "end_time_ns": span.end_time_ns,
                "content": span.to_dict(),
            }
            for span in reversed(spans)
        ]
        if documents:
            store._trace_repository._spans_collection.insert_many(documents)
        return info, spans

    return factory


@pytest.fixture
def span_factory():
    def factory(
        *,
        trace_id,
        span_id=1,
        parent_id=None,
        start_time=1_700_000_000_000_000_000,
        end_time=1_700_000_000_010_000_000,
        status=StatusCode.OK,
        attributes=None,
        resource_attributes=None,
    ):
        values = {SpanAttributeKey.REQUEST_ID: trace_id, SpanAttributeKey.SPAN_TYPE: "CHAIN"}
        values.update(attributes or {})
        return Span(
            ReadableSpan(
                name=f"span-{span_id}",
                context=SpanContext(trace_id=42, span_id=span_id, is_remote=False),
                parent=(
                    SpanContext(trace_id=42, span_id=parent_id, is_remote=False)
                    if parent_id is not None
                    else None
                ),
                start_time=start_time,
                end_time=end_time,
                status=Status(status),
                attributes={key: json.dumps(value) for key, value in values.items()},
                resource=Resource(resource_attributes or {}),
            )
        )

    return factory
