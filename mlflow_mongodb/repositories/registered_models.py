"""Persistence operations for registered models."""

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from bson import ObjectId
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.client_session import ClientSession
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from mlflow_mongodb.repositories.types import (
    ModelVersionRecord,
    RegisteredModelDetails,
    RegisteredModelRecord,
)


@dataclass(frozen=True)
class RegisteredModelFilter:
    """A validated filter to apply to registered-model documents."""

    field_type: Literal["attribute", "tag"]
    key: str
    comparator: Literal["=", "!=", "LIKE", "ILIKE"]
    value: str
    include_missing: bool = False


@dataclass(frozen=True)
class RegisteredModelOrder:
    """A validated registered-model sort field."""

    key: str
    ascending: bool


@dataclass(frozen=True)
class RegisteredModelPage:
    """One materialized page of registered-model records."""

    records: tuple[RegisteredModelDetails, ...]
    has_more: bool


class RegisteredModelAlreadyExistsError(Exception):
    """Raised when a registered model name is already stored."""


class RegisteredModelNotFoundError(Exception):
    """Raised when a registered model name is not stored."""


class RegisteredModelRepository:
    """Store registered-model documents in MongoDB."""

    COLLECTION_NAME = "registered_models"
    MODEL_VERSIONS_COLLECTION_NAME = "model_versions"
    UNIQUE_NAME_INDEX = "registered_models_name_unique"
    TAGS_INDEX = "registered_models_tags_key_value"

    def __init__(self, database: Database):
        self._collection = database[self.COLLECTION_NAME]
        self._collection.create_index(
            [("name", ASCENDING)],
            unique=True,
            name=self.UNIQUE_NAME_INDEX,
        )
        self._collection.create_index(
            [("tags.key", ASCENDING), ("tags.value", ASCENDING)],
            name=self.TAGS_INDEX,
        )

    def create(
        self,
        *,
        name: str,
        creation_timestamp: int,
        description: str | None,
        tags: Mapping[str, str],
        deployment_job_id: str | None,
    ) -> RegisteredModelRecord:
        document: dict[str, Any] = {
            "name": name,
            "creation_timestamp": creation_timestamp,
            "last_updated_timestamp": creation_timestamp,
            "description": description,
            "tags": [{"key": key, "value": value} for key, value in tags.items()],
            "aliases": [],
            "deployment_job_id": deployment_job_id,
            "version_counter": 0,
        }

        try:
            result = self._collection.insert_one(document)
        except DuplicateKeyError as exc:
            raise RegisteredModelAlreadyExistsError(name) from exc

        document["_id"] = result.inserted_id
        return RegisteredModelRecord.from_document(document)

    def allocate_next_version(
        self,
        *,
        model_id: ObjectId,
        last_updated_timestamp: int,
    ) -> int:
        document = self._collection.find_one_and_update(
            {"_id": model_id},
            {
                "$inc": {"version_counter": 1},
                "$set": {"last_updated_timestamp": last_updated_timestamp},
            },
            projection={"version_counter": True},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise RegisteredModelNotFoundError(str(model_id))

        return document["version_counter"]

    def update(
        self,
        *,
        name: str,
        last_updated_timestamp: int,
        description: str | None,
        deployment_job_id: str | None,
    ) -> RegisteredModelRecord:
        fields_to_update = {
            "last_updated_timestamp": last_updated_timestamp,
            "description": description,
        }
        if deployment_job_id is not None:
            fields_to_update["deployment_job_id"] = deployment_job_id

        document = self._collection.find_one_and_update(
            {"name": name},
            {"$set": fields_to_update},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def touch(
        self,
        *,
        model_id: ObjectId,
        last_updated_timestamp: int,
        session: ClientSession | None = None,
    ) -> RegisteredModelRecord:
        document = self._collection.find_one_and_update(
            {"_id": model_id},
            {"$set": {"last_updated_timestamp": last_updated_timestamp}},
            return_document=ReturnDocument.AFTER,
            session=session,
        )
        if document is None:
            raise RegisteredModelNotFoundError(str(model_id))

        return RegisteredModelRecord.from_document(document)

    def rename(
        self,
        *,
        name: str,
        new_name: str,
        last_updated_timestamp: int,
    ) -> RegisteredModelRecord:
        try:
            document = self._collection.find_one_and_update(
                {"name": name},
                {
                    "$set": {
                        "name": new_name,
                        "last_updated_timestamp": last_updated_timestamp,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError as exc:
            raise RegisteredModelAlreadyExistsError(new_name) from exc

        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def find_by_name(
        self,
        name: str,
        *,
        session: ClientSession | None = None,
    ) -> RegisteredModelRecord | None:
        document = self._collection.find_one({"name": name}, session=session)
        return RegisteredModelRecord.from_document(document) if document is not None else None

    def find_by_name_with_latest_versions(
        self,
        name: str,
        *,
        stages: Sequence[str] | None = None,
        session: ClientSession | None = None,
    ) -> RegisteredModelDetails | None:
        documents = self._collection.aggregate(
            [
                {"$match": {"name": name}},
                {"$limit": 1},
                self._latest_versions_lookup_stage(stages=stages),
            ],
            session=session,
        )
        document = next(documents, None)
        return self._to_details(document) if document is not None else None

    def find_latest_version_by_name(
        self,
        name: str,
        *,
        session: ClientSession | None = None,
    ) -> RegisteredModelDetails | None:
        documents = self._collection.aggregate(
            [
                {"$match": {"name": name}},
                {"$limit": 1},
                self._latest_version_lookup_stage(),
            ],
            session=session,
        )
        document = next(documents, None)
        return self._to_details(document) if document is not None else None

    def delete(
        self,
        name: str,
        *,
        session: ClientSession | None = None,
    ) -> RegisteredModelRecord:
        document = self._collection.find_one_and_delete({"name": name}, session=session)
        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def set_tag(self, *, name: str, key: str, value: str) -> RegisteredModelRecord:
        document = self._collection.find_one_and_update(
            {"name": name},
            [
                {
                    "$set": {
                        "tags": {
                            "$concatArrays": [
                                {
                                    "$filter": {
                                        "input": {"$ifNull": ["$tags", []]},
                                        "as": "stored_tag",
                                        "cond": {
                                            "$ne": [
                                                "$$stored_tag.key",
                                                {"$literal": key},
                                            ]
                                        },
                                    }
                                },
                                {"$literal": [{"key": key, "value": value}]},
                            ]
                        }
                    }
                }
            ],
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def delete_tag(self, *, name: str, key: str) -> RegisteredModelRecord:
        document = self._collection.find_one_and_update(
            {"name": name},
            {"$pull": {"tags": {"key": key}}},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def set_alias_by_name(
        self,
        *,
        name: str,
        alias: str,
        version: int,
    ) -> RegisteredModelRecord:
        document = self._collection.find_one_and_update(
            {"name": name},
            [
                {
                    "$set": {
                        "aliases": {
                            "$concatArrays": [
                                {
                                    "$filter": {
                                        "input": {"$ifNull": ["$aliases", []]},
                                        "as": "stored_alias",
                                        "cond": {
                                            "$ne": [
                                                "$$stored_alias.alias",
                                                {"$literal": alias},
                                            ]
                                        },
                                    }
                                },
                                {"$literal": [{"alias": alias, "version": version}]},
                            ]
                        }
                    }
                }
            ],
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def delete_alias_by_name(self, *, name: str, alias: str) -> RegisteredModelRecord:
        document = self._collection.find_one_and_update(
            {"name": name},
            {"$pull": {"aliases": {"alias": alias}}},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def delete_aliases_for_version_and_touch(
        self,
        *,
        model_id: ObjectId,
        version: int,
        last_updated_timestamp: int,
        session: ClientSession | None = None,
    ) -> RegisteredModelRecord:
        document = self._collection.find_one_and_update(
            {"_id": model_id},
            {
                "$pull": {"aliases": {"version": version}},
                "$set": {"last_updated_timestamp": last_updated_timestamp},
            },
            return_document=ReturnDocument.AFTER,
            session=session,
        )
        if document is None:
            raise RegisteredModelNotFoundError(str(model_id))

        return RegisteredModelRecord.from_document(document)

    def search(
        self,
        *,
        filters: Sequence[RegisteredModelFilter],
        order_by: Sequence[RegisteredModelOrder],
        offset: int,
        max_results: int,
    ) -> RegisteredModelPage:
        query = self._build_search_query(filters)
        sort = {order.key: ASCENDING if order.ascending else DESCENDING for order in order_by}
        pipeline: list[dict[str, Any]] = [{"$match": query}]
        if sort:
            pipeline.append({"$sort": sort})
        pipeline.extend([
            {"$skip": offset},
            {"$limit": max_results + 1},
            self._latest_versions_lookup_stage(),
        ])
        documents = list(self._collection.aggregate(pipeline))

        has_more = len(documents) > max_results
        records = tuple(self._to_details(document) for document in documents[:max_results])
        return RegisteredModelPage(records=records, has_more=has_more)

    @classmethod
    def _latest_versions_lookup_stage(
        cls,
        *,
        stages: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        version_match = (
            {"current_stage": {"$in": list(stages)}}
            if stages is not None
            else {"current_stage": {"$ne": STAGE_DELETED_INTERNAL}}
        )
        return {
            "$lookup": {
                "from": cls.MODEL_VERSIONS_COLLECTION_NAME,
                "localField": "_id",
                "foreignField": "registered_model_id",
                "pipeline": [
                    {
                        "$match": version_match,
                    },
                    {"$sort": {"version": DESCENDING}},
                    {
                        "$group": {
                            "_id": "$current_stage",
                            "model_version": {"$first": "$$ROOT"},
                        }
                    },
                    {"$replaceRoot": {"newRoot": "$model_version"}},
                    {"$sort": {"version": DESCENDING}},
                ],
                "as": "latest_versions",
            }
        }

    @classmethod
    def _latest_version_lookup_stage(cls) -> dict[str, Any]:
        return {
            "$lookup": {
                "from": cls.MODEL_VERSIONS_COLLECTION_NAME,
                "localField": "_id",
                "foreignField": "registered_model_id",
                "pipeline": [
                    {
                        "$match": {
                            "current_stage": {"$ne": STAGE_DELETED_INTERNAL},
                        }
                    },
                    {"$sort": {"version": DESCENDING}},
                    {"$limit": 1},
                ],
                "as": "latest_versions",
            }
        }

    @staticmethod
    def _to_details(document: Mapping[str, Any]) -> RegisteredModelDetails:
        return RegisteredModelDetails(
            registered_model=RegisteredModelRecord.from_document(document),
            latest_versions=tuple(
                ModelVersionRecord.from_document(model_version)
                for model_version in document.get("latest_versions", [])
            ),
        )

    @classmethod
    def _build_search_query(cls, filters: Sequence[RegisteredModelFilter]) -> dict[str, Any]:
        clauses = [cls._build_filter_clause(search_filter) for search_filter in filters]
        return {"$and": clauses} if clauses else {}

    @classmethod
    def _build_filter_clause(cls, search_filter: RegisteredModelFilter) -> dict[str, Any]:
        value_condition = cls._build_value_condition(
            search_filter.comparator,
            search_filter.value,
        )
        if search_filter.field_type == "attribute":
            return {search_filter.key: value_condition}

        tag_match = {
            "tags": {
                "$elemMatch": {
                    "key": search_filter.key,
                    "value": value_condition,
                }
            }
        }
        if not search_filter.include_missing:
            return tag_match

        return {
            "$or": [
                tag_match,
                {
                    "tags": {
                        "$not": {
                            "$elemMatch": {"key": search_filter.key},
                        }
                    }
                },
            ]
        }

    @staticmethod
    def _build_value_condition(comparator: str, value: str):
        if comparator == "=":
            return value
        if comparator == "!=":
            return {"$ne": value}
        if comparator not in ("LIKE", "ILIKE"):
            raise ValueError(f"Unsupported registered-model comparator: {comparator}")

        regex = re.escape(value).replace("%", ".*").replace("_", ".")
        if not value.startswith("%"):
            regex = f"^{regex}"
        if not value.endswith("%"):
            regex = f"{regex}$"
        flags = re.DOTALL | (re.IGNORECASE if comparator == "ILIKE" else 0)
        return re.compile(regex, flags)
