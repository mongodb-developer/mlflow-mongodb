"""Persistence operations for logged models."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from mlflow.entities import LifecycleStage
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.database import Database

from mlflow_mongodb.infrastructure.array_operations import (
    build_merge_array_expression,
    build_remove_array_element_update,
)
from mlflow_mongodb.infrastructure.errors import (
    RepositoryNotFoundError,
    RepositoryTagNotFoundError,
)
from mlflow_mongodb.infrastructure.repository_operations import repository_operation
from mlflow_mongodb.infrastructure.search_filters import COMPARISON_OPERATORS, like_regex
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.types import MODEL_METRIC_SCOPE, LoggedModelRecord, RunMetricRecord


@dataclass(frozen=True)
class LoggedModelFilter:
    """A parsed comparison with a validated attribute or literal entity key."""

    field_type: Literal["attribute", "metric", "param", "tag"]
    key: str
    comparator: str
    value: str | float | tuple[str, ...]


@dataclass(frozen=True)
class LoggedModelOrder:
    """One sort expression, optionally scoped to a metric dataset."""

    field_type: Literal["attribute", "metric"]
    key: str
    ascending: bool
    dataset_name: str | None = None
    dataset_digest: str | None = None


@dataclass(frozen=True)
class LoggedModelSearchResult:
    """A model and its associated metric history."""

    model: LoggedModelRecord
    metrics: tuple[RunMetricRecord, ...]


@dataclass(frozen=True)
class LoggedModelPage:
    """One page of models and whether another model exists after it."""

    records: tuple[LoggedModelSearchResult, ...]
    has_more: bool


class LoggedModelRepository:
    """Store logged-model metadata, tags, and parameters in one document."""

    def __init__(self, database: Database, settings: MongoDBSettings | None = None):
        self._settings = settings or MongoDBSettings()
        with repository_operation("Unable to initialize logged model repository."):
            self._collection = database[self._settings.logged_models_collection_name]
            # The model ID is the document's _id, which already has a unique index.
            self._collection.create_index(
                [
                    ("experiment_id", ASCENDING),
                    ("creation_timestamp", DESCENDING),
                    ("_id", ASCENDING),
                ],
                name="logged_models_experiment_creation_id",
            )
            for field in ("tags", "params"):
                self._collection.create_index(
                    [
                        ("experiment_id", ASCENDING),
                        (f"{field}.k", ASCENDING),
                        (f"{field}.v", ASCENDING),
                    ],
                    name=f"logged_models_experiment_{field}",
                )
            self._metrics_collection = database[self._settings.run_metrics_collection_name]
            self._metrics_collection.create_index(
                [
                    ("model_id", ASCENDING),
                    ("k", ASCENDING),
                    ("timestamp", DESCENDING),
                    ("step", DESCENDING),
                    ("run_id", ASCENDING),
                    ("_id", ASCENDING),
                ],
                name="run_metrics_model_k_latest",
                partialFilterExpression={"scope": MODEL_METRIC_SCOPE},
            )
            self._metrics_collection.create_index(
                [
                    ("model_id", ASCENDING),
                    ("dataset_name", ASCENDING),
                    ("dataset_digest", ASCENDING),
                ],
                name="run_metrics_model_dataset",
                partialFilterExpression={"scope": MODEL_METRIC_SCOPE},
            )

    def create(
        self,
        *,
        model_id: str,
        experiment_id: str,
        name: str,
        artifact_location: str,
        creation_timestamp: int,
        status: str,
        lifecycle_stage: str,
        source_run_id: str | None,
        model_type: str | None,
        tags: Mapping[str, str],
        params: Mapping[str, str],
    ) -> LoggedModelRecord:
        document = {
            "_id": model_id,
            "experiment_id": experiment_id,
            "name": name,
            "artifact_location": artifact_location,
            "creation_timestamp": creation_timestamp,
            "last_updated_timestamp": creation_timestamp,
            "status": status,
            "status_message": None,
            "lifecycle_stage": lifecycle_stage,
            "source_run_id": source_run_id,
            "model_type": model_type,
            "tags": [{"k": key, "v": value} for key, value in tags.items()],
            "params": [{"k": key, "v": value} for key, value in params.items()],
        }
        with repository_operation("Unable to create logged model."):
            self._collection.insert_one(document)
        return LoggedModelRecord.from_document(document)

    def find_by_id(self, model_id: str) -> LoggedModelRecord | None:
        with repository_operation("Unable to get logged model."):
            document = self._collection.find_one({"_id": model_id})
        return LoggedModelRecord.from_document(document) if document is not None else None

    def get_metric_history(self, model_id: str) -> tuple[RunMetricRecord, ...]:
        """Read the complete metric history associated with a model."""
        return self._get_metrics_by_model([model_id]).get(model_id, ())

    def finalize(
        self, *, model_id: str, status: str, last_updated_timestamp: int
    ) -> LoggedModelRecord:
        with repository_operation("Unable to finalize logged model."):
            document = self._collection.find_one_and_update(
                {"_id": model_id},
                {"$set": {"status": status, "last_updated_timestamp": last_updated_timestamp}},
                return_document=ReturnDocument.AFTER,
            )
        if document is None:
            raise RepositoryNotFoundError(model_id)
        return LoggedModelRecord.from_document(document)

    def mark_deleted(self, *, model_id: str, last_updated_timestamp: int) -> None:
        """Mark a model deleted, refreshing its timestamp even when already deleted."""
        with repository_operation("Unable to delete logged model."):
            result = self._collection.update_one(
                {"_id": model_id},
                {
                    "$set": {
                        "lifecycle_stage": LifecycleStage.DELETED,
                        "last_updated_timestamp": last_updated_timestamp,
                    }
                },
            )
        # A repeated deletion in the same millisecond can match without modifying.
        if result.matched_count == 0:
            raise RepositoryNotFoundError(model_id)

    def set_tags(self, *, model_id: str, tags: Mapping[str, str]) -> None:
        """Merge tags and check model existence in one update, including empty batches."""
        tags_update = build_merge_array_expression(
            "$tags", [{"k": k, "v": v} for k, v in tags.items()], "k"
        )
        with repository_operation("Unable to set logged model tags."):
            result = self._collection.update_one(
                {"_id": model_id},
                [{"$set": {"tags": tags_update}}],
            )
        # Empty batches and unchanged tags still match an existing model.
        if result.matched_count == 0:
            raise RepositoryNotFoundError(model_id)

    def delete_tag(self, *, model_id: str, key: str) -> None:
        """Remove a tag and distinguish a missing model from a missing tag atomically."""
        update = build_remove_array_element_update(array_field="tags", key_field="k", key=key)
        with repository_operation("Unable to delete logged model tag."):
            document = self._collection.find_one_and_update(
                {"_id": model_id},
                update,
                projection={"_id": 1, "tags.k": 1},
                return_document=ReturnDocument.BEFORE,
            )
        if document is None:
            raise RepositoryNotFoundError(model_id)
        if not any(tag["k"] == key for tag in document.get("tags", [])):
            raise RepositoryTagNotFoundError(key)

    def search(
        self,
        *,
        experiment_ids: Sequence[str],
        filters: Sequence[LoggedModelFilter],
        datasets: Sequence[Mapping[str, str | None]],
        order_by: Sequence[LoggedModelOrder],
        offset: int,
        max_results: int,
    ) -> LoggedModelPage:
        """Filter and page models in MongoDB, then batch-read their metric history."""
        if not experiment_ids:
            return LoggedModelPage((), False)
        clauses = [
            {"experiment_id": {"$in": list(experiment_ids)}},
            {"lifecycle_stage": {"$ne": LifecycleStage.DELETED}},
        ]
        metric_filters = []
        for search_filter in filters:
            if search_filter.field_type == "metric":
                metric_filters.append(search_filter)
                continue
            condition = self._value_condition(search_filter.comparator, search_filter.value)
            if search_filter.field_type == "attribute":
                field = "_id" if search_filter.key == "model_id" else search_filter.key
                clauses.append({field: condition})
            else:
                field = "tags" if search_filter.field_type == "tag" else "params"
                clauses.append(
                    {
                        field: {
                            "$elemMatch": {
                                "k": search_filter.key,
                                "v": condition,
                            }
                        }
                    }
                )

        pipeline: list[dict[str, Any]] = [{"$match": {"$and": clauses}}]
        pipeline.extend(self._metric_stages(metric_filters, datasets, order_by))
        pipeline.extend(self._order_stages(order_by))
        pipeline.extend([{"$skip": offset}, {"$limit": max_results + 1}])
        with (
            repository_operation("Unable to search logged models."),
            self._collection.aggregate(pipeline) as cursor,
        ):
            documents = list(cursor)
        page_documents = documents[:max_results]

        metrics_by_model = self._get_metrics_by_model(
            [document["_id"] for document in page_documents]
        )

        return LoggedModelPage(
            records=tuple(
                LoggedModelSearchResult(
                    LoggedModelRecord.from_document(document),
                    metrics_by_model.get(document["_id"], ()),
                )
                for document in page_documents
            ),
            has_more=len(documents) > max_results,
        )

    def _get_metrics_by_model(
        self, model_ids: Sequence[str]
    ) -> dict[str, tuple[RunMetricRecord, ...]]:
        """Load metric histories for model retrieval."""
        # A separate batched read avoids joining an unbounded metric array into
        # each aggregation result, which would be subject to the BSON size limit.
        metrics_by_model = defaultdict(list)
        if model_ids:
            with repository_operation("Unable to read logged model metric history."):
                metrics = self._metrics_collection.find(
                    {"scope": MODEL_METRIC_SCOPE, "model_id": {"$in": list(model_ids)}},
                    {
                        "_id": 0,
                        "model_id": 1,
                        "run_id": 1,
                        "k": 1,
                        "v": 1,
                        "timestamp": 1,
                        "step": 1,
                        "dataset_name": 1,
                        "dataset_digest": 1,
                    },
                ).sort(
                    [
                        ("model_id", ASCENDING),
                        ("k", ASCENDING),
                        ("timestamp", DESCENDING),
                        ("step", DESCENDING),
                        ("run_id", ASCENDING),
                        ("_id", ASCENDING),
                    ]
                )
                with metrics:
                    metric_documents = list(metrics)
            for metric in metric_documents:
                metrics_by_model[metric["model_id"]].append(RunMetricRecord.from_document(metric))

        return {model_id: tuple(metrics) for model_id, metrics in metrics_by_model.items()}

    def _metric_stages(
        self,
        metric_filters: Sequence[LoggedModelFilter],
        datasets: Sequence[Mapping[str, str | None]],
        order_by: Sequence[LoggedModelOrder],
    ) -> list[dict[str, Any]]:
        """Group each model's metrics into filter flags and latest sort values."""
        dataset_conditions = self._dataset_conditions(datasets)
        dataset_expression = {
            "$or": [self._metric_scope_expression(scope) for scope in dataset_conditions],
        }
        candidates = []
        group: dict[str, Any] = {"_id": "$model_id"}
        required_matches = {}
        for index, search_filter in enumerate(metric_filters):
            match = {
                "k": search_filter.key,
                "v": self._value_condition(search_filter.comparator, search_filter.value),
            }
            conditions = [
                self._metric_scope_expression({"k": search_filter.key}),
                {"$isNumber": "$v"},
                {COMPARISON_OPERATORS[search_filter.comparator]: ["$v", search_filter.value]},
            ]
            if dataset_conditions:
                match["$or"] = dataset_conditions
                conditions.append(dataset_expression)
            candidates.append(match)
            field = f"filter_{index}"
            # Separate flags allow different history entries to satisfy each filter.
            group[field] = {"$max": {"$and": conditions}}
            required_matches[f"_metric_summary.{field}"] = True

        if dataset_conditions and not metric_filters:
            candidates.append({"$or": dataset_conditions})
            group["dataset_match"] = {"$max": dataset_expression}
            required_matches["_metric_summary.dataset_match"] = True

        order_matches = {}
        for index, order in enumerate(order_by):
            if order.field_type != "metric":
                continue
            scope = {"k": order.key}
            if order.dataset_name:
                scope["dataset_name"] = order.dataset_name
            if order.dataset_digest:
                scope["dataset_digest"] = order.dataset_digest
            candidates.append(scope)
            match_field = f"_order_match_{index}"
            order_matches[match_field] = self._metric_scope_expression(scope)
            # Prioritize this order's scope, then its latest row. Other scopes
            # must neither displace that row nor supply a value when it is absent.
            group[f"order_{index}"] = {
                "$top": {
                    "sortBy": {
                        match_field: DESCENDING,
                        "timestamp": DESCENDING,
                        "step": DESCENDING,
                        "run_id": ASCENDING,
                        "_id": ASCENDING,
                    },
                    "output": {"$cond": [f"${match_field}", "$v", None]},
                }
            }

        if not candidates:
            return []

        # Fetch the union of rows needed by filtering and ordering. Intersecting
        # their scopes here would lose valid matches or the latest sort value.
        metric_pipeline: list[dict[str, Any]] = [
            {"$match": {"scope": MODEL_METRIC_SCOPE, "$or": candidates}}
        ]
        if order_matches:
            metric_pipeline.append({"$set": order_matches})
        metric_pipeline.append({"$group": group})
        pipeline = [
            {
                "$lookup": {
                    "from": self._metrics_collection.name,
                    "localField": "_id",
                    "foreignField": "model_id",
                    "pipeline": metric_pipeline,
                    "as": "_metric_summary",
                }
            },
            {
                "$set": {
                    "_metric_summary": {"$arrayElemAt": ["$_metric_summary", 0]},
                }
            },
        ]
        if required_matches:
            pipeline.append({"$match": required_matches})
        return pipeline

    @staticmethod
    def _metric_scope_expression(scope: Mapping[str, Any]) -> dict[str, Any]:
        """Compare metric identity fields with literal values."""
        comparisons = [
            {"$eq": [f"${field}", {"$literal": value}]} for field, value in scope.items()
        ]
        return comparisons[0] if len(comparisons) == 1 else {"$and": comparisons}

    @staticmethod
    def _dataset_conditions(datasets: Sequence[Mapping[str, str | None]]) -> list[dict[str, Any]]:
        clauses = []
        for dataset in datasets:
            clause = {"dataset_name": dataset["dataset_name"]}
            if "dataset_digest" in dataset:
                clause["dataset_digest"] = dataset["dataset_digest"]
            clauses.append(clause)
        return clauses

    def _order_stages(self, order_by: Sequence[LoggedModelOrder]) -> list[dict[str, Any]]:
        pipeline = []
        computed = {}
        sort = {}
        nullable_attributes = {"model_type", "source_run_id", "status_message"}
        for index, order in enumerate(order_by):
            direction = ASCENDING if order.ascending else DESCENDING
            if order.field_type == "attribute":
                field = "_id" if order.key == "model_id" else order.key
                nullable = order.key in nullable_attributes
            else:
                field = f"_metric_summary.order_{index}"
                nullable = True

            # Descending MongoDB sort already places null/missing values last.
            # Ascending nullable sorts need a rank field to put them last too.
            if nullable and order.ascending:
                null_field = f"_sort_null_{index}"
                computed[null_field] = {
                    "$eq": [{"$ifNull": [f"${field}", None]}, None],
                }
                sort[null_field] = ASCENDING
            sort[field] = direction

        if computed:
            pipeline.append({"$set": computed})
        pipeline.append({"$sort": sort})
        return pipeline

    @staticmethod
    def _value_condition(comparator: str, value: str | float | tuple[str, ...]) -> dict[str, Any]:
        if comparator in ("LIKE", "ILIKE"):
            return {"$regex": like_regex(value, comparator)}
        if comparator == "!=":
            return {"$exists": True, "$nin": [None, value]}
        if comparator == "NOT IN":
            if not value:
                return {"$exists": True}
            return {"$exists": True, "$nin": [None, *value]}
        if comparator == "IN":
            return {"$in": list(value)}
        return {COMPARISON_OPERATORS[comparator]: value}
