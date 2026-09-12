"""Shared persistence DTOs for the MongoDB repositories."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bson import ObjectId


@dataclass(frozen=True)
class RegisteredModelTagRecord:
    """Stored registered-model tag data."""

    key: str
    value: str


@dataclass(frozen=True)
class RegisteredModelAliasRecord:
    """Stored registered-model alias data."""

    alias: str
    version: int


@dataclass(frozen=True)
class RegisteredModelRecord:
    """Typed representation of a registered-model document."""

    model_id: ObjectId
    name: str
    creation_timestamp: int
    last_updated_timestamp: int
    description: str | None
    tags: tuple[RegisteredModelTagRecord, ...]
    aliases: tuple[RegisteredModelAliasRecord, ...]
    deployment_job_id: str | None
    version_counter: int

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "RegisteredModelRecord":
        return cls(
            model_id=document["_id"],
            name=document["name"],
            creation_timestamp=document["creation_timestamp"],
            last_updated_timestamp=document["last_updated_timestamp"],
            description=document.get("description"),
            tags=tuple(
                RegisteredModelTagRecord(tag["key"], tag["value"])
                for tag in document.get("tags", [])
            ),
            aliases=tuple(
                RegisteredModelAliasRecord(alias["alias"], alias["version"])
                for alias in document.get("aliases", [])
            ),
            deployment_job_id=document.get("deployment_job_id"),
            version_counter=document.get("version_counter", 0),
        )


@dataclass(frozen=True)
class ModelVersionTagRecord:
    """Stored model-version tag data."""

    key: str
    value: str


@dataclass(frozen=True)
class ModelVersionRecord:
    """Typed representation of a model-version document."""

    model_version_id: ObjectId
    registered_model_id: ObjectId
    version: int
    creation_timestamp: int
    last_updated_timestamp: int | None
    description: str | None
    user_id: str | None
    current_stage: str
    source: str | None
    storage_location: str | None
    run_id: str | None
    run_link: str | None
    status: str
    status_message: str | None
    tags: tuple[ModelVersionTagRecord, ...]
    model_id: str | None

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ModelVersionRecord":
        return cls(
            model_version_id=document["_id"],
            registered_model_id=document["registered_model_id"],
            version=document["version"],
            creation_timestamp=document["creation_timestamp"],
            last_updated_timestamp=document.get("last_updated_timestamp"),
            description=document.get("description"),
            user_id=document.get("user_id"),
            current_stage=document["current_stage"],
            source=document.get("source"),
            storage_location=document.get("storage_location"),
            run_id=document.get("run_id"),
            run_link=document.get("run_link"),
            status=document["status"],
            status_message=document.get("status_message"),
            tags=tuple(
                ModelVersionTagRecord(tag["key"], tag["value"]) for tag in document.get("tags", [])
            ),
            model_id=document.get("model_id"),
        )


@dataclass(frozen=True)
class RegisteredModelDetails:
    """A registered model and its latest model versions."""

    registered_model: RegisteredModelRecord
    latest_versions: tuple[ModelVersionRecord, ...]


@dataclass(frozen=True)
class ModelVersionSearchResult:
    """A model version and its joined registered-model data."""

    model_version: ModelVersionRecord
    registered_model: RegisteredModelRecord
