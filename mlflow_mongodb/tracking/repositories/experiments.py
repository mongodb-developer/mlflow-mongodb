"""Persistence operations for experiments."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotActiveError,
    RepositoryNotFoundError,
    RepositoryPersistenceError,
)
from mlflow_mongodb.infrastructure.repository_operations import repository_operation
from mlflow_mongodb.infrastructure.search_filters import build_value_condition
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.types import ExperimentRecord


@dataclass(frozen=True)
class ExperimentFilter:
    """A validated comparison using public experiment attributes or literal tag keys."""

    field_type: Literal["attribute", "tag"]
    key: str
    comparator: str
    value: str | float | tuple[str, ...] | None


@dataclass(frozen=True)
class ExperimentOrder:
    """An ordering by a public experiment attribute."""

    key: str
    ascending: bool


class ExperimentRepository:
    """Store experiment documents with embedded tags in MongoDB."""

    UNIQUE_NAME_INDEX = "experiments_name_unique"

    def __init__(self, database: Database, settings: MongoDBSettings | None = None):
        self._settings = settings or MongoDBSettings()
        with repository_operation("Unable to select collection.", errors=(PyMongoError,)):
            self._collection = database[self._settings.experiments_collection_name]
        with repository_operation("Unable to initialize repository."):
            self._collection.create_index(
                [("name", ASCENDING)], unique=True, name=self.UNIQUE_NAME_INDEX
            )
            self._collection.create_index(
                [("lifecycle_stage", ASCENDING), ("creation_time", DESCENDING), ("_id", ASCENDING)],
                name="experiments_lifecycle_creation_id",
            )

    def search(
        self,
        *,
        lifecycle_stages: Sequence[str],
        filters: Sequence[ExperimentFilter],
        order_by: Sequence[ExperimentOrder],
        offset: int,
        limit: int,
    ) -> list[ExperimentRecord]:
        """Translate validated search criteria and read the requested slice in MongoDB."""
        clauses = [{"lifecycle_stage": {"$in": list(lifecycle_stages)}}]
        for search_filter in filters:
            if search_filter.field_type == "attribute":
                field = "_id" if search_filter.key == "experiment_id" else search_filter.key
                clauses.append(
                    {field: self._value_condition(search_filter.comparator, search_filter.value)}
                )
            elif search_filter.comparator == "IS NULL":
                clauses.append({"tags": {"$not": {"$elemMatch": {"k": search_filter.key}}}})
            elif search_filter.comparator == "IS NOT NULL":
                clauses.append({"tags": {"$elemMatch": {"k": search_filter.key}}})
            else:
                tag_match = {
                    "k": search_filter.key,
                    "v": self._value_condition(search_filter.comparator, search_filter.value),
                }
                if search_filter.comparator == "!=":
                    tag_match = {
                        "$and": [tag_match, {"v": {"$ne": None}}],
                    }
                clauses.append({"tags": {"$elemMatch": tag_match}})
        sort_fields = [
            (
                "_id" if order.key == "experiment_id" else order.key,
                ASCENDING if order.ascending else DESCENDING,
            )
            for order in order_by
        ]
        with repository_operation():
            cursor = (
                self._collection.find({"$and": clauses}).sort(sort_fields).skip(offset).limit(limit)
            )
            documents = list(cursor)
        return [ExperimentRecord.from_document(document) for document in documents]

    @staticmethod
    def _value_condition(comparator: str, value: str | float | tuple[str, ...]):
        if comparator not in ("=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "NOT IN"):
            raise ValueError(f"Unsupported experiment comparator: {comparator}")

        return build_value_condition(comparator, value)

    def create(
        self,
        *,
        experiment_id: str,
        name: str,
        artifact_location: str,
        lifecycle_stage: str,
        creation_timestamp: int,
        tags: Mapping[str, str],
    ) -> str:
        with repository_operation("Unable to create experiment."):
            try:
                self._collection.insert_one(
                    {
                        "_id": experiment_id,
                        "name": name,
                        "artifact_location": artifact_location,
                        "lifecycle_stage": lifecycle_stage,
                        "creation_time": creation_timestamp,
                        "last_update_time": creation_timestamp,
                        "tags": [{"k": key, "v": value} for key, value in tags.items()],
                    }
                )
            except DuplicateKeyError as exc:
                # Do not report an ID collision as a duplicate experiment name.
                if exc.details and exc.details.get("keyPattern") == {"_id": 1}:
                    raise RepositoryPersistenceError("Unable to create experiment.") from exc
                raise RepositoryAlreadyExistsError(name) from exc
        return experiment_id

    def find_by_id(self, experiment_id: str) -> ExperimentRecord | None:
        with repository_operation("Unable to get experiment."):
            document = self._collection.find_one({"_id": experiment_id})
        return ExperimentRecord.from_document(document) if document is not None else None

    def find_by_name(self, name: str) -> ExperimentRecord | None:
        with repository_operation("Unable to get experiment by name."):
            document = self._collection.find_one({"name": name})
        return ExperimentRecord.from_document(document) if document is not None else None

    def rename(
        self,
        *,
        experiment_id: str | None,
        new_name: str,
        last_update_time: int,
    ) -> ExperimentRecord:
        """Atomically rename an active experiment, preserving name uniqueness."""
        with repository_operation("Unable to rename experiment."):
            try:
                document = self._collection.find_one_and_update(
                    {"_id": experiment_id, "lifecycle_stage": "active"},
                    {"$set": {"name": new_name, "last_update_time": last_update_time}},
                    return_document=ReturnDocument.AFTER,
                )
            except DuplicateKeyError as exc:
                raise RepositoryAlreadyExistsError(new_name) from exc
        if document is None:
            if self.find_by_id(experiment_id) is None:
                raise RepositoryNotFoundError(experiment_id)
            raise RepositoryNotActiveError(experiment_id)
        return ExperimentRecord.from_document(document)

    def mark_deleted(
        self,
        *,
        experiment_id: str | None,
        last_update_time: int,
    ) -> ExperimentRecord:
        return self._transition_lifecycle(
            experiment_id=experiment_id,
            current_stage="active",
            next_stage="deleted",
            updates={"last_update_time": last_update_time},
        )

    def restore(
        self,
        *,
        experiment_id: str | None,
        last_update_time: int,
    ) -> ExperimentRecord:
        return self._transition_lifecycle(
            experiment_id=experiment_id,
            current_stage="deleted",
            next_stage="active",
            updates={"last_update_time": last_update_time},
        )

    def _transition_lifecycle(
        self,
        *,
        experiment_id: str | None,
        current_stage: str,
        next_stage: str,
        updates: dict[str, object],
    ) -> ExperimentRecord:
        with repository_operation("Unable to update experiment lifecycle."):
            document = self._collection.find_one_and_update(
                {"_id": experiment_id, "lifecycle_stage": current_stage},
                {"$set": {"lifecycle_stage": next_stage, **updates}},
                return_document=ReturnDocument.AFTER,
            )
        if document is None:
            raise RepositoryNotFoundError(experiment_id)
        return ExperimentRecord.from_document(document)
