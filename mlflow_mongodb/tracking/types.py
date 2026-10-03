"""Experiment persistence DTOs owned by the tracking store."""

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
