"""Persistence DTOs owned by the tracking store."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final


class SpanDimensionKey:
    """Internal field names for queryable span model attributes."""

    MODEL: Final = "model"
    MODEL_PROVIDER: Final = "provider"


RUN_METRIC_SCOPE = "run"
MODEL_METRIC_SCOPE = "model"


@dataclass(frozen=True)
class ExperimentTagRecord:
    """Stored experiment tag data."""

    key: str
    value: str


@dataclass(frozen=True)
class ExperimentRecord:
    """Typed representation of an experiment document."""

    experiment_id: int
    name: str
    artifact_location: str
    lifecycle_stage: str
    creation_time: int
    last_update_time: int
    tags: tuple[ExperimentTagRecord, ...]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=document["_id"],
            name=document["name"],
            artifact_location=document["artifact_location"],
            lifecycle_stage=document["lifecycle_stage"],
            creation_time=document["creation_time"],
            last_update_time=document["last_update_time"],
            tags=tuple(ExperimentTagRecord(tag["k"], tag["v"]) for tag in document.get("tags", [])),
        )


@dataclass(frozen=True)
class RunParamRecord:
    """Stored run parameter data."""

    key: str
    value: str


@dataclass(frozen=True)
class TraceRecord:
    """Trace metadata stored independently from span payloads."""

    trace_id: str
    experiment_id: str
    request_time: int
    state: str
    execution_duration: int | None
    client_request_id: str | None
    request_preview: str | None
    response_preview: str | None
    tags: tuple[ExperimentTagRecord, ...]
    trace_metadata: tuple[ExperimentTagRecord, ...]
    assessments: tuple[dict[str, Any], ...]
    span_stats: dict[str, Any] | None = None
    run_ids: tuple[str, ...] = ()

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        assessments: tuple[dict[str, Any], ...] = (),
    ) -> "TraceRecord":
        return cls(
            trace_id=document["_id"],
            experiment_id=document["experiment_id"],
            request_time=document["request_time"],
            state=document["state"],
            execution_duration=document.get("execution_duration"),
            client_request_id=document.get("client_request_id"),
            request_preview=document.get("request_preview"),
            response_preview=document.get("response_preview"),
            tags=tuple(ExperimentTagRecord(t["k"], t["v"]) for t in document.get("tags", [])),
            trace_metadata=tuple(
                ExperimentTagRecord(m["k"], m["v"]) for m in document.get("trace_metadata", [])
            ),
            assessments=assessments,
            span_stats=document.get("span_stats"),
            run_ids=tuple(document.get("run_ids", [])),
        )


@dataclass(frozen=True)
class SpanRecord:
    """Persisted span payload returned when loading a trace."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time_ns: int
    end_time_ns: int | None
    content: dict[str, Any]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "SpanRecord":
        return cls(
            trace_id=document["trace_id"],
            span_id=document["span_id"],
            parent_span_id=document.get("parent_span_id"),
            start_time_ns=document["start_time_ns"],
            end_time_ns=document.get("end_time_ns"),
            content=document["content"],
        )


@dataclass(frozen=True)
class SpanSummaryRecord:
    """Span fields needed to recompute a trace summary."""

    span_id: str
    parent_span_id: str | None
    status: str
    start_time_ns: int
    end_time_ns: int | None
    token_usage: dict[str, Any] | None
    cost: dict[str, Any] | None
    trace_fields: dict[str, Any]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "SpanSummaryRecord":
        return cls(
            span_id=document["span_id"],
            parent_span_id=document.get("parent_span_id"),
            status=document["status"],
            start_time_ns=document["start_time_ns"],
            end_time_ns=document.get("end_time_ns"),
            token_usage=document.get("token_usage"),
            cost=document.get("cost"),
            trace_fields=document.get("trace_fields", {}),
        )


@dataclass(frozen=True)
class RunMetricRecord:
    """Stored metric from run history or the latest-per-key summary."""

    key: str
    value: float
    timestamp: int
    step: int
    model_id: str | None = None
    dataset_name: str | None = None
    dataset_digest: str | None = None
    run_id: str | None = None

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "RunMetricRecord":
        return cls(
            key=document["k"],
            value=document["v"],
            timestamp=document["timestamp"],
            step=document["step"],
            model_id=document.get("model_id"),
            dataset_name=document.get("dataset_name"),
            dataset_digest=document.get("dataset_digest"),
            run_id=document.get("run_id"),
        )


@dataclass(frozen=True)
class DatasetInputRecord:
    """Dataset metadata and input tags embedded in a run."""

    name: str
    digest: str
    source_type: str
    source: str
    schema: str | None
    profile: str | None
    tags: tuple[ExperimentTagRecord, ...]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "DatasetInputRecord":
        dataset = document["dataset"]
        return cls(
            name=dataset["name"],
            digest=dataset["digest"],
            source_type=dataset["source_type"],
            source=dataset["source"],
            schema=dataset.get("schema"),
            profile=dataset.get("profile"),
            tags=tuple(ExperimentTagRecord(tag["k"], tag["v"]) for tag in document.get("tags", [])),
        )


@dataclass(frozen=True)
class ModelOutputRecord:
    """A logged model and its output step embedded in a run."""

    model_id: str
    step: int

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ModelOutputRecord":
        return cls(model_id=document["model_id"], step=document["step"])


@dataclass(frozen=True)
class RunRecord:
    """Typed representation of a run document."""

    run_id: str
    experiment_id: str
    name: str
    artifact_uri: str
    user_id: str
    status: str
    start_time: int
    end_time: int | None
    lifecycle_stage: str
    tags: tuple[ExperimentTagRecord, ...]
    params: tuple[RunParamRecord, ...] = ()
    metrics: tuple[RunMetricRecord, ...] = ()
    dataset_inputs: tuple[DatasetInputRecord, ...] = ()
    model_inputs: tuple[str, ...] = ()
    model_outputs: tuple[ModelOutputRecord, ...] = ()

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "RunRecord":
        # Runs created before input logging used an empty array placeholder.
        inputs = document.get("inputs") or {}
        return cls(
            run_id=document["_id"],
            experiment_id=document["experiment_id"],
            name=document["name"],
            artifact_uri=document["artifact_uri"],
            user_id=document["user_id"],
            status=document["status"],
            start_time=document["start_time"],
            end_time=document.get("end_time"),
            lifecycle_stage=document["lifecycle_stage"],
            params=tuple(
                RunParamRecord(param["k"], param["v"]) for param in document.get("params", [])
            ),
            metrics=tuple(
                RunMetricRecord.from_document(metric) for metric in document.get("metrics", [])
            ),
            dataset_inputs=tuple(
                DatasetInputRecord.from_document(dataset_input)
                for dataset_input in inputs.get("datasets", [])
            ),
            model_inputs=tuple(model["model_id"] for model in inputs.get("models", [])),
            model_outputs=tuple(
                ModelOutputRecord.from_document(model) for model in document.get("outputs", [])
            ),
            tags=tuple(ExperimentTagRecord(tag["k"], tag["v"]) for tag in document.get("tags", [])),
        )


@dataclass(frozen=True)
class LoggedModelTagRecord:
    """Stored logged-model tag data."""

    key: str
    value: str


@dataclass(frozen=True)
class LoggedModelParameterRecord:
    """Stored logged-model parameter data."""

    key: str
    value: str


@dataclass(frozen=True)
class LoggedModelRecord:
    """Typed representation of a logged-model document."""

    model_id: str
    experiment_id: str
    name: str
    artifact_location: str
    creation_timestamp: int
    last_updated_timestamp: int
    status: str
    status_message: str | None
    lifecycle_stage: str
    source_run_id: str | None
    model_type: str | None
    tags: tuple[LoggedModelTagRecord, ...]
    params: tuple[LoggedModelParameterRecord, ...]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "LoggedModelRecord":
        return cls(
            model_id=document["_id"],
            experiment_id=document["experiment_id"],
            name=document["name"],
            artifact_location=document["artifact_location"],
            creation_timestamp=document["creation_timestamp"],
            last_updated_timestamp=document["last_updated_timestamp"],
            status=document["status"],
            status_message=document["status_message"],
            lifecycle_stage=document["lifecycle_stage"],
            source_run_id=document["source_run_id"],
            model_type=document["model_type"],
            tags=tuple(
                LoggedModelTagRecord(tag["k"], tag["v"]) for tag in document.get("tags", [])
            ),
            params=tuple(
                LoggedModelParameterRecord(param["k"], param["v"])
                for param in document.get("params", [])
            ),
        )
