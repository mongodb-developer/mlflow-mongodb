"""Persistence operations for model versions."""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from bson import ObjectId
from mlflow.entities.model_registry.model_version_stages import (
    STAGE_ARCHIVED,
    STAGE_DELETED_INTERNAL,
)
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
    ModelVersionSearchResult,
    RegisteredModelRecord,
)
from mlflow_mongodb.settings import MongoDBSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelVersionFilter:
    """A validated filter to apply to model-version search results."""

    field_type: Literal["attribute", "tag"]
    key: str
    comparator: str
    value: str | int | float | tuple[str, ...]


@dataclass(frozen=True)
class ModelVersionOrder:
    """A validated model-version sort field."""

    key: Literal[
        "name",
        "version_number",
        "creation_timestamp",
        "last_updated_timestamp",
    ]
    ascending: bool


@dataclass(frozen=True)
class ModelVersionPage:
    """One materialized page of model-version search results."""

    records: tuple[ModelVersionSearchResult, ...]
    has_more: bool


class ModelVersionAlreadyExistsError(Exception):
    """Raised when a model version number is already stored for a registered model."""


class ModelVersionNotFoundError(Exception):
    """Raised when a model version is not stored for a registered model."""


class ModelVersionRepository:
    """Read and write model-version documents in MongoDB."""

    UNIQUE_VERSION_INDEX = "model_versions_registered_model_id_version_unique"
    LATEST_VERSION_INDEX = "model_versions_registered_model_stage_version"
    TAGS_INDEX = "model_versions_tags_key_value"

    def __init__(self, database: Database, settings: MongoDBSettings | None = None):
        self._settings = settings or MongoDBSettings()
        self._collection = database[self._settings.model_versions_collection_name]
        self._registered_models_collection = database[
            self._settings.registered_models_collection_name
        ]
        self._collection.create_index(
            [("registered_model_id", ASCENDING), ("version", ASCENDING)],
            unique=True,
            name=self.UNIQUE_VERSION_INDEX,
        )
        self._collection.create_index(
            [
                ("registered_model_id", ASCENDING),
                ("current_stage", ASCENDING),
                ("version", DESCENDING),
            ],
            name=self.LATEST_VERSION_INDEX,
        )
        self._collection.create_index(
            [("tags.key", ASCENDING), ("tags.value", ASCENDING)],
            name=self.TAGS_INDEX,
        )

    def create(
        self,
        *,
        registered_model_id: ObjectId,
        version: int,
        creation_timestamp: int,
        description: str | None,
        current_stage: str,
        source: str | None,
        storage_location: str | None,
        run_id: str | None,
        run_link: str | None,
        status: str,
        tags: Mapping[str, str],
        model_id: str | None,
    ) -> ModelVersionRecord:
        """Create a model version for a registered model.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            version: Model-version number.
            creation_timestamp: Creation timestamp.
            description: Model-version description.
            current_stage: Initial model-version stage.
            source: Source location of the model.
            storage_location: Resolved model storage location.
            run_id: Optional source run ID.
            run_link: Optional source run link.
            status: Model-version status.
            tags: Initial model-version tags keyed by tag name.
            model_id: Optional logged-model identifier.

        Returns:
            The created :class:`ModelVersionRecord`.

        Raises:
            ModelVersionAlreadyExistsError: If the version already exists for
                the registered model.
        """
        document: dict[str, Any] = {
            "registered_model_id": registered_model_id,
            "version": version,
            "creation_timestamp": creation_timestamp,
            "last_updated_timestamp": creation_timestamp,
            "description": description,
            "user_id": None,
            "current_stage": current_stage,
            "source": source,
            "storage_location": storage_location,
            "run_id": run_id,
            "run_link": run_link,
            "status": status,
            "status_message": None,
            "tags": [{"key": key, "value": value} for key, value in tags.items()],
            "model_id": model_id,
        }

        try:
            result = self._collection.insert_one(document)
        except DuplicateKeyError as exc:
            logger.error("Unable to create model version: %s", exc)
            raise ModelVersionAlreadyExistsError(f"{registered_model_id}:{version}") from exc

        document["_id"] = result.inserted_id
        return ModelVersionRecord.from_document(document)

    def update_description(
        self,
        *,
        registered_model_id: ObjectId,
        version: int,
        description: str | None,
        last_updated_timestamp: int,
    ) -> ModelVersionRecord:
        """Update a non-deleted model version's description.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            version: Model-version number.
            description: New model-version description.
            last_updated_timestamp: Timestamp to store for the update.

        Returns:
            The updated :class:`ModelVersionRecord`.

        Raises:
            ModelVersionNotFoundError: If the model version does not exist or
                has been soft-deleted.
        """
        document = self._collection.find_one_and_update(
            {
                "registered_model_id": registered_model_id,
                "version": version,
                "current_stage": {"$ne": STAGE_DELETED_INTERNAL},
            },
            {
                "$set": {
                    "description": description,
                    "last_updated_timestamp": last_updated_timestamp,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise ModelVersionNotFoundError(f"{registered_model_id}:{version}")

        return ModelVersionRecord.from_document(document)

    def transition_stage(
        self,
        *,
        registered_model_id: ObjectId,
        version: int,
        stage: str,
        last_updated_timestamp: int,
    ) -> ModelVersionRecord:
        """Transition a non-deleted model version to a new stage.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            version: Model-version number.
            stage: New model-version stage.
            last_updated_timestamp: Timestamp to store for the update.

        Returns:
            The updated :class:`ModelVersionRecord`.

        Raises:
            ModelVersionNotFoundError: If the model version does not exist or
                has been soft-deleted.
        """
        document = self._collection.find_one_and_update(
            {
                "registered_model_id": registered_model_id,
                "version": version,
                "current_stage": {"$ne": STAGE_DELETED_INTERNAL},
            },
            {
                "$set": {
                    "current_stage": stage,
                    "last_updated_timestamp": last_updated_timestamp,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise ModelVersionNotFoundError(f"{registered_model_id}:{version}")

        return ModelVersionRecord.from_document(document)

    def archive_other_versions_in_stage(
        self,
        *,
        registered_model_id: ObjectId,
        version: int,
        stage: str,
        last_updated_timestamp: int,
    ) -> None:
        """Archive other model versions currently in a stage.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            version: Model-version number to keep in the stage.
            stage: Stage whose other versions should be archived.
            last_updated_timestamp: Timestamp to store for the updates.
        """
        self._collection.update_many(
            {
                "registered_model_id": registered_model_id,
                "version": {"$ne": version},
                "current_stage": stage,
            },
            {
                "$set": {
                    "current_stage": STAGE_ARCHIVED,
                    "last_updated_timestamp": last_updated_timestamp,
                }
            },
        )

    def touch_all_for_registered_model(
        self,
        *,
        registered_model_id: ObjectId,
        last_updated_timestamp: int,
    ) -> None:
        """Update the timestamp of every version for a registered model.

        This is used after renaming a registered model to keep all associated
        model-version timestamps aligned with the parent-model update. Model
        versions reference the registered model by ID, so the rename does not
        require changing a duplicated model name on each version.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            last_updated_timestamp: Timestamp to store for the updates.
        """
        self._collection.update_many(
            {"registered_model_id": registered_model_id},
            {"$set": {"last_updated_timestamp": last_updated_timestamp}},
        )

    def soft_delete(
        self,
        *,
        registered_model_id: ObjectId,
        version: int,
        last_updated_timestamp: int,
    ) -> ModelVersionRecord:
        """Soft-delete and redact a model version.

        The version document remains in MongoDB, but its stage is changed to
        MLflow's internal deleted marker so normal model-version queries no
        longer return it. Sensitive or externally resolved metadata is
        redacted, while the version identity and lifecycle history remain
        available for internal cleanup and uniqueness checks.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            version: Model-version number.
            last_updated_timestamp: Timestamp to store for the deletion.

        Returns:
            The redacted, soft-deleted :class:`ModelVersionRecord`.

        Raises:
            ModelVersionNotFoundError: If the model version does not exist or
                has already been soft-deleted.
        """
        document = self._collection.find_one_and_update(
            {
                "registered_model_id": registered_model_id,
                "version": version,
                "current_stage": {"$ne": STAGE_DELETED_INTERNAL},
            },
            {
                "$set": {
                    "current_stage": STAGE_DELETED_INTERNAL,
                    "last_updated_timestamp": last_updated_timestamp,
                    "description": None,
                    "user_id": None,
                    "source": "REDACTED-SOURCE-PATH",
                    "run_id": "REDACTED-RUN-ID",
                    "run_link": "REDACTED-RUN-LINK",
                    "status_message": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise ModelVersionNotFoundError(f"{registered_model_id}:{version}")

        return ModelVersionRecord.from_document(document)

    def delete_all_for_registered_model(
        self,
        *,
        registered_model_id: ObjectId,
    ) -> int:
        """Permanently delete every model version owned by a registered model.

        Args:
            registered_model_id: MongoDB identifier of the registered model.

        Returns:
            The number of deleted model-version documents.
        """
        result = self._collection.delete_many({"registered_model_id": registered_model_id})
        return result.deleted_count

    def set_tag(
        self,
        *,
        registered_model_id: ObjectId,
        version: int,
        key: str,
        value: str,
    ) -> ModelVersionRecord:
        """Set or replace a tag on a non-deleted model version.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            version: Model-version number.
            key: Tag key to set.
            value: Tag value.

        Returns:
            The updated :class:`ModelVersionRecord`.

        Raises:
            ModelVersionNotFoundError: If the model version does not exist or
                has been soft-deleted.
        """
        document = self._collection.find_one_and_update(
            {
                "registered_model_id": registered_model_id,
                "version": version,
                "current_stage": {"$ne": STAGE_DELETED_INTERNAL},
            },
            build_replace_array_element_pipeline(
                array_field="tags",
                key_field="key",
                key=key,
                element={"key": key, "value": value},
            ),
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise ModelVersionNotFoundError(f"{registered_model_id}:{version}")

        return ModelVersionRecord.from_document(document)

    def delete_tag(
        self,
        *,
        registered_model_id: ObjectId,
        version: int,
        key: str,
    ) -> ModelVersionRecord:
        """Delete a tag from a non-deleted model version.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            version: Model-version number.
            key: Tag key to delete.

        Returns:
            The updated :class:`ModelVersionRecord`.

        Raises:
            ModelVersionNotFoundError: If the model version does not exist or
                has been soft-deleted.
        """
        document = self._collection.find_one_and_update(
            {
                "registered_model_id": registered_model_id,
                "version": version,
                "current_stage": {"$ne": STAGE_DELETED_INTERNAL},
            },
            build_remove_array_element_update(
                array_field="tags",
                key_field="key",
                key=key,
            ),
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise ModelVersionNotFoundError(f"{registered_model_id}:{version}")

        return ModelVersionRecord.from_document(document)

    def find_by_version(
        self,
        *,
        registered_model_id: ObjectId,
        version: int,
    ) -> ModelVersionRecord | None:
        """Find a non-deleted model version by registered model and number.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            version: Model-version number.

        Returns:
            The matching :class:`ModelVersionRecord`, or ``None`` if the
            version does not exist or has been soft-deleted.
        """
        document = self._collection.find_one(
            {
                "registered_model_id": registered_model_id,
                "version": version,
                "current_stage": {"$ne": STAGE_DELETED_INTERNAL},
            }
        )
        return ModelVersionRecord.from_document(document) if document is not None else None

    def exists_for_registered_model(
        self,
        *,
        registered_model_name: str,
        version: int,
    ) -> bool:
        """Check whether a non-deleted version belongs to a registered model.

        Args:
            registered_model_name: Registered model name.
            version: Model-version number.

        Returns:
            ``True`` if the registered model exists and owns the specified
            non-deleted version; otherwise, ``False``.
        """
        documents = self._registered_models_collection.aggregate(
            [
                {
                    "$match": {"name": registered_model_name},
                },
                {"$limit": 1},
                {
                    "$lookup": {
                        "from": self._settings.model_versions_collection_name,
                        "localField": "_id",
                        "foreignField": "registered_model_id",
                        "pipeline": [
                            {
                                "$match": {
                                    "version": version,
                                    "current_stage": {"$ne": STAGE_DELETED_INTERNAL},
                                }
                            },
                            {"$limit": 1},
                            {"$project": {"_id": True}},
                        ],
                        "as": "matching_versions",
                    }
                },
                {"$match": {"matching_versions.0": {"$exists": True}}},
                {"$project": {"_id": True}},
                {"$limit": 1},
            ]
        )
        return next(documents, None) is not None

    def find_latest_by_stages(
        self,
        *,
        registered_model_id: ObjectId,
        stages: Sequence[str],
    ) -> tuple[ModelVersionRecord, ...]:
        """Find the latest model version for each requested stage.

        Args:
            registered_model_id: MongoDB identifier of the registered model.
            stages: Model-version stages to include.

        Returns:
            A tuple containing at most one highest-numbered
            :class:`ModelVersionRecord` for each requested stage, ordered by
            version number descending.
        """
        documents = self._collection.aggregate(
            [
                {
                    "$match": {
                        "registered_model_id": registered_model_id,
                        "current_stage": {"$in": list(stages)},
                    }
                },
                {"$sort": {"current_stage": ASCENDING, "version": DESCENDING}},
                {"$group": {"_id": "$current_stage", "record": {"$first": "$$ROOT"}}},
                {"$replaceRoot": {"newRoot": "$record"}},
                {"$sort": {"version": DESCENDING}},
            ]
        )
        return tuple(ModelVersionRecord.from_document(document) for document in documents)

    def search(
        self,
        *,
        filters: Sequence[ModelVersionFilter],
        order_by: Sequence[ModelVersionOrder],
        exclude_prompts: bool,
        offset: int,
        max_results: int,
    ) -> ModelVersionPage:
        """Search model versions and include their registered models.

        Args:
            filters: Validated model-version and registered-model filters.
            order_by: Validated fields and directions used for sorting.
            exclude_prompts: Whether prompt model versions should be excluded.
            offset: Number of matching records to skip.
            max_results: Maximum number of records to return.

        Returns:
            A page containing matching model versions with their registered
            models and whether more results are available.
        """
        # Version filters can be applied before joining registered models; name
        # filters must be applied after the join because name belongs to the parent.
        version_filters = [
            search_filter
            for search_filter in filters
            if not (search_filter.field_type == "attribute" and search_filter.key == "name")
            and not (
                exclude_prompts
                and search_filter.field_type == "tag"
                and search_filter.key == IS_PROMPT_TAG_KEY
            )
        ]
        name_filters = [
            search_filter
            for search_filter in filters
            if search_filter.field_type == "attribute" and search_filter.key == "name"
        ]

        # All normal searches exclude soft-deleted versions.
        version_clauses: list[dict[str, Any]] = [{"current_stage": {"$ne": STAGE_DELETED_INTERNAL}}]
        version_clauses.extend(self._build_filter_clauses(version_filters))
        if exclude_prompts:
            # Prompt exclusion is represented by the registered-model prompt tag.
            version_clauses.append(
                {
                    "tags": {
                        "$not": {
                            "$elemMatch": {
                                "key": IS_PROMPT_TAG_KEY,
                                "value": "true",
                            }
                        }
                    }
                }
            )

        # Filter versions first, then join the owning registered model for the
        # returned entity and any parent-model filters.
        pipeline: list[dict[str, Any]] = [
            {"$match": self._combine_clauses(version_clauses)},
            {
                "$lookup": {
                    "from": self._settings.registered_models_collection_name,
                    "localField": "registered_model_id",
                    "foreignField": "_id",
                    "as": "registered_model",
                }
            },
            {"$unwind": "$registered_model"},
        ]

        name_clauses = self._build_filter_clauses(
            name_filters,
            attribute_prefix="registered_model.",
        )
        if name_clauses:
            # Apply registered-model name filters after the lookup and unwind.
            pipeline.append({"$match": self._combine_clauses(name_clauses)})

        # Apply the requested order and pagination after all filters.
        sort_fields = {
            self._order_field(order.key): ASCENDING if order.ascending else DESCENDING
            for order in order_by
        }
        pipeline.append({"$sort": sort_fields})
        if offset:
            pipeline.append({"$skip": offset})
        # Fetch one extra record to determine whether another page exists.
        pipeline.append({"$limit": max_results + 1})

        documents = list(self._collection.aggregate(pipeline))
        has_more = len(documents) > max_results
        records = tuple(
            ModelVersionSearchResult(
                model_version=ModelVersionRecord.from_document(document),
                registered_model=RegisteredModelRecord.from_document(document["registered_model"]),
            )
            for document in documents[:max_results]
        )
        return ModelVersionPage(records=records, has_more=has_more)

    @classmethod
    def _build_filter_clauses(
        cls,
        filters: Sequence[ModelVersionFilter],
        attribute_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Build MongoDB clauses for model-version attributes and tags.

        Attribute filters are mapped to MongoDB fields, optionally using a
        prefix for fields on the joined registered model. Tag filters with the
        same key are combined into one ``$elemMatch`` clause so all conditions
        apply to a single tag entry.

        Args:
            filters: Validated model-version or registered-model filters.
            attribute_prefix: Prefix for attribute fields on a joined document.

        Returns:
            MongoDB query clauses representing the supplied filters.
        """
        clauses = []
        tag_filters: dict[str, list[ModelVersionFilter]] = {}
        for search_filter in filters:
            if search_filter.field_type == "tag":
                # Group constraints by key so they apply to the same tag entry.
                tag_filters.setdefault(search_filter.key, []).append(search_filter)
                continue

            # Attribute filters target either the version or the joined model,
            # depending on the requested field.
            field = f"{attribute_prefix}{cls._attribute_field(search_filter.key)}"
            clauses.append(
                cls._build_attribute_clause(
                    field,
                    search_filter.comparator,
                    search_filter.value,
                )
            )

        for key, filters_for_key in tag_filters.items():
            # Require one array element to satisfy the key and all value
            # conditions for that tag.
            element_clauses = [{"key": key}]
            element_clauses.extend(
                {
                    "value": cls._build_value_condition(
                        search_filter.comparator,
                        search_filter.value,
                    )
                }
                for search_filter in filters_for_key
            )
            clauses.append(
                {
                    "tags": {
                        "$elemMatch": {
                            "$and": element_clauses,
                        }
                    }
                }
            )
        return clauses

    @classmethod
    def _build_attribute_clause(
        cls,
        field: str,
        comparator: str,
        value: str | int | float | tuple[str, ...],
    ) -> dict[str, Any]:
        condition = cls._build_value_condition(comparator, value)
        if comparator != "!=":
            return {field: condition}

        # SQL comparisons against NULL are unknown rather than true. MongoDB's `$ne` also matches
        # a missing field, so explicitly require a stored, non-null value for equivalent behavior.
        return {
            "$and": [
                {field: {"$exists": True}},
                {field: {"$ne": None}},
                {field: condition},
            ]
        }

    @staticmethod
    def _build_value_condition(
        comparator: str,
        value: str | int | float | tuple[str, ...],
    ):
        if comparator == "=":
            return value
        if comparator == "!=":
            return {"$ne": value}
        if comparator == "IN":
            return {"$in": list(value)}
        numeric_operators = {
            ">": "$gt",
            ">=": "$gte",
            "<": "$lt",
            "<=": "$lte",
        }
        if comparator in numeric_operators:
            return {numeric_operators[comparator]: value}
        if comparator not in ("LIKE", "ILIKE"):
            raise ValueError(f"Unsupported model-version comparator: {comparator}")

        regex = re.escape(str(value)).replace("%", ".*").replace("_", ".")
        if not str(value).startswith("%"):
            regex = f"^{regex}"
        if not str(value).endswith("%"):
            regex = f"{regex}$"
        flags = re.DOTALL | (re.IGNORECASE if comparator == "ILIKE" else 0)
        return re.compile(regex, flags)

    @staticmethod
    def _combine_clauses(clauses: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": list(clauses)}

    @staticmethod
    def _attribute_field(key: str) -> str:
        if key == "version_number":
            return "version"
        if key == "source_path":
            return "source"
        return key

    @staticmethod
    def _order_field(key: str) -> str:
        if key == "name":
            return "registered_model.name"
        if key == "version_number":
            return "version"
        return key
