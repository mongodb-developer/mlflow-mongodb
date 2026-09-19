"""Persistence operations for registered models."""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from bson import ObjectId
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from mlflow_mongodb.repositories._helpers import (
    build_remove_array_element_update,
    build_replace_array_element_pipeline,
)
from mlflow_mongodb.repositories.types import (
    ModelVersionRecord,
    RegisteredModelDetails,
    RegisteredModelRecord,
)
from mlflow_mongodb.settings import MongoDBSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredModelFilter:
    """A validated filter to apply to registered-model documents."""

    field_type: Literal["attribute", "tag"]
    key: str
    comparator: Literal["=", "!=", "LIKE", "ILIKE"]
    value: str


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

    UNIQUE_NAME_INDEX = "registered_models_name_unique"
    TAGS_INDEX = "registered_models_tags_key_value"

    def __init__(self, database: Database, settings: MongoDBSettings | None = None):
        self._settings = settings or MongoDBSettings()
        self._collection = database[self._settings.registered_models_collection_name]
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
        """Create a registered model.

        Args:
            name: Registered model name.
            creation_timestamp: Creation timestamp.
            description: Model description.
            tags: Initial registered-model tags keyed by tag name.
            deployment_job_id: Optional deployment job ID.

        Returns:
            The created :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelAlreadyExistsError: If a model with the same name
                already exists.
        """
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
            logger.error("Unable to create registered model: %s", exc)
            raise RegisteredModelAlreadyExistsError(name) from exc

        document["_id"] = result.inserted_id
        return RegisteredModelRecord.from_document(document)

    def allocate_next_version(
        self,
        *,
        model_id: ObjectId,
        last_updated_timestamp: int,
    ) -> int:
        """Allocate the next model-version number.

        Args:
            model_id: MongoDB identifier of the registered model.
            last_updated_timestamp: Timestamp to store for the update.

        Returns:
            The newly allocated model-version number.

        Raises:
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
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
        """Update metadata of a registered model.

        Args:
            name: Registered model name.
            last_updated_timestamp: Timestamp for the update.
            description: New description.
            deployment_job_id: Optional deployment job ID. If ``None``, the
                existing value is left unchanged.

        Returns:
            The updated :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
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
    ) -> RegisteredModelRecord:
        """Update only a registered model's last-updated timestamp.

        Args:
            model_id: MongoDB identifier of the registered model.
            last_updated_timestamp: Timestamp to store for the update.

        Returns:
            The updated :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
        document = self._collection.find_one_and_update(
            {"_id": model_id},
            {"$set": {"last_updated_timestamp": last_updated_timestamp}},
            return_document=ReturnDocument.AFTER,
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
        """Rename a registered model.

        Args:
            name: Current registered model name.
            new_name: New registered model name.
            last_updated_timestamp: Timestamp to store for the update.

        Returns:
            The renamed :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelAlreadyExistsError: If another model already has
                the new name.
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
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
            logger.error("Unable to rename registered model: %s", exc)
            raise RegisteredModelAlreadyExistsError(new_name) from exc

        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def find_by_name(
        self,
        name: str,
    ) -> RegisteredModelRecord | None:
        """Find a registered model by name.

        Args:
            name: Registered model name.

        Returns:
            The matching :class:`RegisteredModelRecord`, or ``None`` if no
            registered model has the specified name.
        """
        document = self._collection.find_one({"name": name})
        return RegisteredModelRecord.from_document(document) if document is not None else None

    def find_by_name_with_latest_versions(
        self,
        name: str,
        *,
        stages: Sequence[str] | None = None,
    ) -> RegisteredModelDetails | None:
        """Find a registered model and its latest versions by name.

        Args:
            name: Registered model name.
            stages: Optional model-version stages to include when selecting
                latest versions.

        Returns:
            The registered model and its latest matching
            :class:`ModelVersionRecord` objects, or ``None`` if no registered
            model has the specified name.
        """
        documents = self._collection.aggregate(
            [
                {"$match": {"name": name}},
                {"$limit": 1},
                self._latest_versions_lookup_stage(stages=stages),
            ],
        )
        document = next(documents, None)
        return self._to_details(document) if document is not None else None

    def find_latest_version_by_name(
        self,
        name: str,
    ) -> RegisteredModelDetails | None:
        """Find a registered model and its latest version by name.

        Args:
            name: Registered model name.

        Returns:
            The registered model and its latest :class:`ModelVersionRecord`,
            or ``None`` if no registered model has the specified name.
        """
        documents = self._collection.aggregate(
            [
                {"$match": {"name": name}},
                {"$limit": 1},
                self._latest_version_lookup_stage(),
            ],
        )
        document = next(documents, None)
        return self._to_details(document) if document is not None else None

    def delete(
        self,
        name: str,
    ) -> RegisteredModelRecord:
        """Delete a registered model by name.

        Args:
            name: Registered model name.

        Returns:
            The deleted :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
        document = self._collection.find_one_and_delete({"name": name})
        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def set_tag(self, *, name: str, key: str, value: str) -> RegisteredModelRecord:
        """Set or replace a tag on a registered model.

        Args:
            name: Registered model name.
            key: Tag key to set.
            value: Tag value.

        Returns:
            The updated :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
        document = self._collection.find_one_and_update(
            {"name": name},
            build_replace_array_element_pipeline(
                array_field="tags",
                key_field="key",
                key=key,
                element={"key": key, "value": value},
            ),
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def delete_tag(self, *, name: str, key: str) -> RegisteredModelRecord:
        """Delete a tag from a registered model.

        Args:
            name: Registered model name.
            key: Tag key to delete.

        Returns:
            The updated :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
        document = self._collection.find_one_and_update(
            {"name": name},
            build_remove_array_element_update(
                array_field="tags",
                key_field="key",
                key=key,
            ),
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
        """Set or replace an alias on a registered model.

        Args:
            name: Registered model name.
            alias: Alias name to set.
            version: Model-version number assigned to the alias.

        Returns:
            The updated :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
        document = self._collection.find_one_and_update(
            {"name": name},
            build_replace_array_element_pipeline(
                array_field="aliases",
                key_field="alias",
                key=alias,
                element={"alias": alias, "version": version},
            ),
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise RegisteredModelNotFoundError(name)

        return RegisteredModelRecord.from_document(document)

    def delete_alias_by_name(self, *, name: str, alias: str) -> RegisteredModelRecord:
        """Delete an alias from a registered model.

        Args:
            name: Registered model name.
            alias: Alias name to delete.

        Returns:
            The updated :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
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
    ) -> RegisteredModelRecord:
        """Delete aliases for a model version and update the model timestamp.

        Args:
            model_id: MongoDB identifier of the registered model.
            version: Model-version number whose aliases should be deleted.
            last_updated_timestamp: Timestamp to store for the update.

        Returns:
            The updated :class:`RegisteredModelRecord`.

        Raises:
            RegisteredModelNotFoundError: If the registered model does not
                exist.
        """
        document = self._collection.find_one_and_update(
            {"_id": model_id},
            {
                "$pull": {"aliases": {"version": version}},
                "$set": {"last_updated_timestamp": last_updated_timestamp},
            },
            return_document=ReturnDocument.AFTER,
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
        """Search registered models and include their latest versions.

        Args:
            filters: Validated registered-model filters.
            order_by: Validated fields and directions used for sorting.
            offset: Number of matching records to skip.
            max_results: Maximum number of records to return.

        Returns:
            A page containing the matching registered models and whether more
            results are available.
        """
        query = self._build_search_query(filters)
        sort = {order.key: ASCENDING if order.ascending else DESCENDING for order in order_by}
        pipeline: list[dict[str, Any]] = [{"$match": query}]
        if sort:
            pipeline.append({"$sort": sort})
        pipeline.extend(
            [
                {"$skip": offset},
                # Fetch one extra record to determine whether another page exists;
                # it is not included in the returned records below.
                {"$limit": max_results + 1},
                self._latest_versions_lookup_stage(),
            ]
        )
        documents = list(self._collection.aggregate(pipeline))
        records = tuple(self._to_details(document) for document in documents[:max_results])
        return RegisteredModelPage(records=records, has_more=len(documents) > max_results)

    def _latest_versions_lookup_stage(
        self,
        *,
        stages: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Build a pipeline stage that adds the latest model versions to each registered model.

        Returns:
            A MongoDB aggregation stage that populates ``latest_versions``.
        """

        version_match = (
            {"current_stage": {"$in": list(stages)}}
            if stages is not None
            else {"current_stage": {"$ne": STAGE_DELETED_INTERNAL}}
        )
        return {
            "$lookup": {
                "from": self._settings.model_versions_collection_name,
                # Join model versions and reduce them to the latest version
                # for each requested stage.
                "localField": "_id",
                "foreignField": "registered_model_id",
                "pipeline": [
                    {
                        "$match": version_match,
                    },
                    # Required by the per-stage $first accumulator below.
                    {"$sort": {"version": DESCENDING}},
                    {
                        "$group": {
                            "_id": "$current_stage",
                            "model_version": {"$first": "$$ROOT"},
                        }
                    },
                    # Return version documents rather than group wrappers.
                    {"$replaceRoot": {"newRoot": "$model_version"}},
                    {"$sort": {"version": DESCENDING}},
                ],
                "as": "latest_versions",
            }
        }

    def _latest_version_lookup_stage(self) -> dict[str, Any]:
        return {
            "$lookup": {
                "from": self._settings.model_versions_collection_name,
                # Join versions for the registered model and select the
                # highest non-deleted version overall.
                "localField": "_id",
                "foreignField": "registered_model_id",
                "pipeline": [
                    {
                        # Soft-deleted versions are not eligible for latest
                        # version results.
                        "$match": {
                            "current_stage": {"$ne": STAGE_DELETED_INTERNAL},
                        }
                    },
                    # Select the highest-numbered remaining version.
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
        # A missing ``is_prompt`` tag is treated as ``false`` by MLflow.
        include_missing = search_filter.key == IS_PROMPT_TAG_KEY and (
            (search_filter.comparator == "=" and search_filter.value.lower() == "false")
            or (search_filter.comparator == "!=" and search_filter.value.lower() == "true")
        )
        if not include_missing:
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
