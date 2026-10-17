import json
from collections.abc import Mapping
from unittest.mock import create_autospec
from uuid import UUID

import pytest
from mlflow.entities import LifecycleStage
from mlflow.entities.assessment import Feedback
from mlflow.entities.span import Span
from mlflow.tracing.constant import SpanAttributeKey
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, Status, StatusCode

from mlflow_mongodb.tracking.repositories import (
    ExperimentRepository,
    LoggedModelPage,
    LoggedModelRepository,
    RunRepository,
    TraceRepository,
)
from mlflow_mongodb.tracking.store import MongoDBTrackingStore
from mlflow_mongodb.tracking.types import (
    ExperimentRecord,
    ExperimentTagRecord,
    LoggedModelRecord,
    RunRecord,
    SpanRecord,
    TraceRecord,
)

FIXED_TIMESTAMP = 1_700_000_000_000
EXPERIMENT_ID = "42"
ARTIFACT_URI = "s3://experiment-artifacts"


@pytest.fixture
def experiment_repository():
    return create_autospec(ExperimentRepository, instance=True)


@pytest.fixture
def store(monkeypatch, experiment_repository):
    monkeypatch.setattr(
        "mlflow_mongodb.tracking.store.get_current_time_millis",
        lambda: FIXED_TIMESTAMP,
    )
    monkeypatch.setattr("mlflow_mongodb.tracking.store.uuid4", lambda: UUID(int=42))
    mongodb_store = MongoDBTrackingStore(
        store_uri="mongodb://localhost:27017/mlflow",
        artifact_uri=ARTIFACT_URI,
    )
    mongodb_store.__dict__["_experiment_repository"] = experiment_repository
    return mongodb_store


@pytest.fixture
def experiment_record_factory():
    def factory(
        *,
        experiment_id=EXPERIMENT_ID,
        name="example-experiment",
        artifact_location=f"{ARTIFACT_URI}/{EXPERIMENT_ID}",
        lifecycle_stage=LifecycleStage.ACTIVE,
        creation_time=FIXED_TIMESTAMP,
        last_update_time=FIXED_TIMESTAMP,
        tags: Mapping[str, str] | None = None,
    ):
        return ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_location,
            lifecycle_stage=lifecycle_stage,
            creation_time=creation_time,
            last_update_time=last_update_time,
            tags=tuple(
                ExperimentTagRecord(key=key, value=value) for key, value in (tags or {}).items()
            ),
        )

    return factory


@pytest.fixture
def run_repository():
    return create_autospec(RunRepository, instance=True)


@pytest.fixture
def run_document_factory():
    def factory(**overrides):
        document = {
            "_id": UUID(int=42).hex,
            "experiment_id": EXPERIMENT_ID,
            "name": "example-run",
            "artifact_uri": f"{ARTIFACT_URI}/{EXPERIMENT_ID}/{UUID(int=42).hex}/artifacts",
            "user_id": "test-user",
            "status": "RUNNING",
            "start_time": FIXED_TIMESTAMP,
            "end_time": None,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "tags": [{"k": "team", "v": "risk"}],
            "params": [],
            "metrics": [],
            "inputs": {"datasets": [], "models": []},
            "outputs": [],
        }
        document.update(overrides)
        return document

    return factory


@pytest.fixture
def run_record_factory(run_document_factory):
    def factory(**overrides):
        return RunRecord.from_document(run_document_factory(**overrides))

    return factory


@pytest.fixture
def run_store(
    store,
    run_repository,
    experiment_repository,
    experiment_record_factory,
    run_record_factory,
):
    store.__dict__["_run_repository"] = run_repository
    experiment_repository.find_by_id.return_value = experiment_record_factory()
    for method in ("create", "find_by_id", "update_info", "mark_deleted", "restore"):
        getattr(run_repository, method).return_value = run_record_factory()
    run_repository.get_metric_history.return_value = []
    return store


@pytest.fixture
def logged_model_repository():
    return create_autospec(LoggedModelRepository, instance=True)


@pytest.fixture
def logged_model_document_factory():
    def factory(**overrides):
        document = {
            "_id": f"m-{UUID(int=42).hex}",
            "experiment_id": EXPERIMENT_ID,
            "name": "example-model",
            "artifact_location": (
                f"{ARTIFACT_URI}/{EXPERIMENT_ID}/models/m-{UUID(int=42).hex}/artifacts"
            ),
            "creation_timestamp": FIXED_TIMESTAMP,
            "last_updated_timestamp": FIXED_TIMESTAMP,
            "status": "PENDING",
            "status_message": None,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "source_run_id": None,
            "model_type": None,
            "tags": [{"k": "team", "v": "risk"}],
            "params": [{"k": "depth", "v": "3"}],
        }
        document.update(overrides)
        return document

    return factory


@pytest.fixture
def logged_model_record_factory(logged_model_document_factory):
    def factory(**overrides):
        return LoggedModelRecord.from_document(logged_model_document_factory(**overrides))

    return factory


@pytest.fixture
def logged_model_store(
    store,
    logged_model_repository,
    experiment_repository,
    experiment_record_factory,
    logged_model_record_factory,
):
    store.__dict__["_logged_model_repository"] = logged_model_repository
    experiment_repository.find_by_id.return_value = experiment_record_factory()
    logged_model_repository.create.return_value = logged_model_record_factory()
    logged_model_repository.find_by_id.return_value = logged_model_record_factory()
    logged_model_repository.get_metric_history.return_value = ()
    logged_model_repository.search.return_value = LoggedModelPage((), False)
    return store


@pytest.fixture
def trace_repository():
    return create_autospec(TraceRepository, instance=True)


@pytest.fixture
def trace_record_factory():
    def factory(**overrides):
        document = {
            "_id": f"tr-{UUID(int=42).hex}",
            "experiment_id": EXPERIMENT_ID,
            "request_time": FIXED_TIMESTAMP,
            "state": "OK",
            "execution_duration": 10,
            "tags": [{"k": "team", "v": "risk"}],
            "trace_metadata": [{"k": "source", "v": "test"}],
        }
        assessments = overrides.pop("assessments", ())
        document.update(overrides)
        return TraceRecord.from_document(document, assessments=assessments)

    return factory


@pytest.fixture
def trace_store(store, trace_repository, trace_record_factory, experiment_record_factory):
    store.__dict__["_trace_repository"] = trace_repository
    store._experiment_repository.find_by_id.return_value = experiment_record_factory()
    trace_repository.start_trace.return_value = trace_record_factory()
    trace_repository.get_trace_info.return_value = trace_record_factory()
    trace_repository.batch_get_trace_infos.return_value = []
    trace_repository.batch_get_traces.return_value = []
    trace_repository.search_trace_infos.return_value = []
    trace_repository.get_spans.return_value = []
    trace_repository.query_trace_metrics.return_value = []
    return store


@pytest.fixture
def feedback_factory(trace_record_factory):
    def factory(**overrides):
        assessment_id = overrides.pop("assessment_id", "a-original")
        arguments = {
            "name": "quality",
            "value": True,
            "trace_id": trace_record_factory().trace_id,
            "span_id": "0000000000000001",
            "metadata": {"reviewer": "alice"},
            "create_time_ms": FIXED_TIMESTAMP - 1000,
            "last_update_time_ms": FIXED_TIMESTAMP - 500,
        }
        arguments.update(overrides)
        assessment = Feedback(**arguments)
        assessment.assessment_id = assessment_id
        return assessment

    return factory


@pytest.fixture
def span_factory(trace_record_factory):
    def factory(
        *,
        span_id=1,
        parent_id=None,
        trace_id=None,
        start_time=1_000_000,
        end_time=11_000_000,
        status=StatusCode.OK,
        attributes=None,
        resource_attributes=None,
    ):
        trace_id = trace_id or trace_record_factory().trace_id
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


@pytest.fixture
def span_record_factory(span_factory):
    def factory(**kwargs):
        return SpanRecord.from_document(
            MongoDBTrackingStore._span_to_document(span_factory(**kwargs))
        )

    return factory
