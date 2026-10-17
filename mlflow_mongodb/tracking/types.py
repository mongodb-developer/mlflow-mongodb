"""Persistence DTOs owned by the tracking store."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentTagRecord:
    """Stored experiment tag data."""

    key: str
    value: str


@dataclass(frozen=True)
class ExperimentRecord:
    """Typed representation of an experiment document."""

    experiment_id: str
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
