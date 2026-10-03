"""Persistence operations for experiments."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from bson.errors import BSONError
from mlflow.utils.search_utils import SearchUtils
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError

from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotActiveError,
    RepositoryNotFoundError,
    RepositoryPersistenceError,
)
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
    COMPARISON_OPERATORS: ClassVar[dict[str, str]] = {
        "=": "$eq",
        "!=": "$ne",
        "<": "$lt",
        "<=": "$lte",
        ">": "$gt",
        ">=": "$gte",
    }

    def __init__(self, database: Database, settings: MongoDBSettings | None = None):
        self._settings = settings or MongoDBSettings()
        try:
            self._collection = database[self._settings.experiments_collection_name]
            self._collection.create_index(
                [("name", ASCENDING)], unique=True, name=self.UNIQUE_NAME_INDEX
            )
            self._collection.create_index(
                [("lifecycle_stage", ASCENDING), ("creation_time", DESCENDING), ("_id", ASCENDING)],
                name="experiments_lifecycle_creation_id",
            )
        except (PyMongoError, BSONError) as exc:
            raise RepositoryPersistenceError("A database operation failed.") from exc

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
                clauses.append(
                    {
                        "tags": {
                            "$elemMatch": {
                                "k": search_filter.key,
                                "v": self._value_condition(
                                    search_filter.comparator, search_filter.value
                                ),
                            }
                        }
                    }
                )
        sort_fields = [
            (
                "_id" if order.key == "experiment_id" else order.key,
                ASCENDING if order.ascending else DESCENDING,
            )
            for order in order_by
        ]
        try:
            cursor = (
                self._collection.find({"$and": clauses}).sort(sort_fields).skip(offset).limit(limit)
            )
            return [ExperimentRecord.from_document(document) for document in cursor]
        except (PyMongoError, BSONError) as exc:
            raise RepositoryPersistenceError("A database operation failed.") from exc

    @classmethod
    def _value_condition(
        cls, comparator: str, value: str | float | tuple[str, ...] | None
    ) -> dict[str, Any]:
        if comparator in (SearchUtils.LIKE_OPERATOR, SearchUtils.ILIKE_OPERATOR):
            pattern = re.escape(value).replace("%", ".*").replace("_", ".")
            # MLflow's Python regex helper lacks DOTALL and strict end-of-string anchoring.
            return {
                "$regex": f"\\A{pattern}\\z",
                "$options": "is" if comparator == SearchUtils.ILIKE_OPERATOR else "s",
            }
        if comparator in ("IN", "NOT IN"):
            return {"$in" if comparator == "IN" else "$nin": list(value)}
        return {cls.COMPARISON_OPERATORS[comparator]: value}

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
        document = {
            "_id": experiment_id,
            "name": name,
            "artifact_location": artifact_location,
            "lifecycle_stage": lifecycle_stage,
            "creation_time": creation_timestamp,
            "last_update_time": creation_timestamp,
            "tags": [{"k": key, "v": value} for key, value in tags.items()],
        }
        try:
            self._collection.insert_one(document)
        except DuplicateKeyError as exc:
            # Do not report an ID collision as a duplicate experiment name.
            if exc.details and exc.details.get("keyPattern") == {"_id": 1}:
                raise RepositoryPersistenceError("Unable to create experiment.") from exc
            raise RepositoryAlreadyExistsError(name) from exc
        except (PyMongoError, BSONError) as exc:
            raise RepositoryPersistenceError("Unable to create experiment.") from exc
        return experiment_id

    def find_by_id(self, experiment_id: str) -> ExperimentRecord | None:
        document = self._collection.find_one({"_id": experiment_id})
        return ExperimentRecord.from_document(document) if document is not None else None

    def find_by_name(self, name: str) -> ExperimentRecord | None:
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
        try:
            document = self._collection.find_one_and_update(
                {"_id": experiment_id, "lifecycle_stage": "active"},
                {"$set": {"name": new_name, "last_update_time": last_update_time}},
                return_document=ReturnDocument.AFTER,
            )
            if document is None:
                if self.find_by_id(experiment_id) is None:
                    raise RepositoryNotFoundError(experiment_id)
                raise RepositoryNotActiveError(experiment_id)
            return ExperimentRecord.from_document(document)
        except DuplicateKeyError as exc:
            raise RepositoryAlreadyExistsError(new_name) from exc
        except (PyMongoError, BSONError) as exc:
            raise RepositoryPersistenceError("Unable to rename experiment.") from exc

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
        updates = {"lifecycle_stage": next_stage, **updates}
        document = self._collection.find_one_and_update(
            {"_id": experiment_id, "lifecycle_stage": current_stage},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise RepositoryNotFoundError(experiment_id)
        return ExperimentRecord.from_document(document)
