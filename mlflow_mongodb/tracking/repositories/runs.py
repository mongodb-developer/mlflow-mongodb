"""Persistence operations for runs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mlflow.entities import LifecycleStage
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT, MLFLOW_RUN_NAME
from pymongo import ASCENDING, DESCENDING, ReturnDocument, UpdateOne
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from mlflow_mongodb.infrastructure.array_operations import build_merge_array_expression
from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotActiveError,
    RepositoryNotFoundError,
    RepositoryParamConflictError,
)
from mlflow_mongodb.infrastructure.repository_operations import repository_operation
from mlflow_mongodb.infrastructure.search_filters import build_value_condition
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.types import (
    MODEL_METRIC_SCOPE,
    RUN_METRIC_SCOPE,
    RunMetricRecord,
    RunRecord,
)


@dataclass(frozen=True)
class RunFilter:
    """A validated run search comparison."""

    field_type: str
    key: str
    comparator: str
    value: str | float | tuple[str, ...] | None


@dataclass(frozen=True)
class RunOrder:
    """Ordering by a run attribute or an embedded key/value entry."""

    field_type: str
    key: str
    ascending: bool


class RunRepository:
    """Store run documents with small run fields embedded in the document."""

    EXPERIMENT_INDEX = "runs_experiment_id"
    _METRIC_IDENTITY_FIELDS = (
        "run_id",
        "k",
        "timestamp",
        "step",
        "v",
    )

    def __init__(self, database: Database, settings: MongoDBSettings | None = None):
        self._settings = settings or MongoDBSettings()
        with repository_operation("Unable to initialize run repository."):
            self._collection = database[self._settings.runs_collection_name]
            self._collection.create_index(
                [("experiment_id", ASCENDING)], name=self.EXPERIMENT_INDEX
            )
            self._metrics_collection = database[self._settings.run_metrics_collection_name]
            self._metrics_collection.create_index(
                [("run_id", ASCENDING), ("k", ASCENDING), ("timestamp", ASCENDING)],
                name="run_metrics_run_k_timestamp",
                partialFilterExpression={"scope": RUN_METRIC_SCOPE},
            )
            self._metrics_collection.create_index(
                [(field, ASCENDING) for field in self._METRIC_IDENTITY_FIELDS],
                name="run_metrics_identity",
                unique=True,
                partialFilterExpression={"scope": RUN_METRIC_SCOPE},
            )
            self._metrics_collection.create_index(
                [
                    ("model_id", ASCENDING),
                    ("run_id", ASCENDING),
                    ("k", ASCENDING),
                    ("timestamp", ASCENDING),
                    ("step", ASCENDING),
                ],
                name="model_metrics_identity",
                unique=True,
                partialFilterExpression={"scope": MODEL_METRIC_SCOPE},
            )

    def search(
        self,
        *,
        experiment_ids: Sequence[str],
        lifecycle_stages: Sequence[str],
        filters: Sequence[RunFilter],
        order_by: Sequence[RunOrder],
        offset: int,
        limit: int | None,
    ) -> list[RunRecord]:
        """Filter, sort, and paginate run documents in MongoDB."""
        clauses = [
            {"experiment_id": {"$in": list(experiment_ids)}},
            {"lifecycle_stage": {"$in": list(lifecycle_stages)}},
        ]
        clauses.extend(self._search_condition(clause) for clause in filters)
        pipeline = [{"$match": {"$and": clauses}}]
        values = {}
        ranks = {}
        sort = {}
        for index, order in enumerate(order_by):
            field = f"_search_value_{index}"
            rank = f"_search_rank_{index}"
            values[field] = self._search_order_value(order)
            reference = f"${field}"
            missing = {"$eq": [reference, None]}
            if order.field_type == "metric":
                ranks[rank] = {
                    "$switch": {
                        "branches": [
                            {"case": missing, "then": 2},
                            {"case": {"$eq": [reference, {"$literal": float("nan")}]}, "then": 1},
                        ],
                        "default": 0,
                    }
                }
            else:
                ranks[rank] = {"$cond": [missing, 1, 0]}
            sort[rank] = ASCENDING
            sort[field] = ASCENDING if order.ascending else DESCENDING
        pipeline.extend([{"$set": values}, {"$set": ranks}, {"$sort": sort}])
        if offset:
            pipeline.append({"$skip": offset})
        if limit is not None:
            pipeline.append({"$limit": limit})
        pipeline.append({"$unset": [*values, *ranks]})
        with repository_operation("Unable to search runs."):
            documents = list(self._collection.aggregate(pipeline))
        return [RunRecord.from_document(document) for document in documents]

    @staticmethod
    def _search_condition(clause: RunFilter) -> dict:
        """Filter a run attribute, dataset input, or embedded metric/parameter/tag."""
        field_type, key, comparator, value = (
            clause.field_type,
            clause.key,
            clause.comparator,
            clause.value,
        )
        if field_type == "attribute":
            if key == "run_name":
                return RunRepository._embedded_condition("tag", MLFLOW_RUN_NAME, comparator, value)
            return RunRepository._attribute_condition(key, comparator, value)
        if field_type == "dataset":
            return RunRepository._dataset_condition(key, comparator, value)
        return RunRepository._embedded_condition(field_type, key, comparator, value)

    @staticmethod
    def _attribute_condition(key, comparator, value) -> dict:
        """Filter a non-null top-level run attribute."""
        field = RunRepository._attribute_field(key)
        condition = {field: build_value_condition(comparator, value)}
        return {"$and": [condition, {field: {"$ne": None}}]}

    @staticmethod
    def _dataset_condition(key, comparator, value) -> dict:
        """Filter dataset inputs by name, digest, or their context tag."""
        if key == "context":
            match = {
                "tags": {
                    "$elemMatch": {
                        "k": MLFLOW_DATASET_CONTEXT,
                        "v": build_value_condition(comparator, value),
                    }
                }
            }
            if comparator in ("!=", "NOT IN"):
                match["tags"]["$elemMatch"] = {
                    "$and": [match["tags"]["$elemMatch"], {"v": {"$ne": None}}]
                }
        else:
            field = f"dataset.{key}"
            match = {
                "$and": [{field: build_value_condition(comparator, value)}, {field: {"$ne": None}}]
            }
        return {"inputs.datasets": {"$elemMatch": match}}

    @staticmethod
    def _embedded_condition(field_type, key, comparator, value) -> dict:
        """Filter embedded metrics, parameters, or tags by key and value or key existence."""
        field = RunRepository._embedded_field(field_type)
        if comparator == "IS NULL":
            return {field: {"$not": {"$elemMatch": {"k": key}}}}
        if comparator == "IS NOT NULL":
            return {field: {"$elemMatch": {"k": key}}}
        match = {
            "$and": [
                {"k": key, "v": build_value_condition(comparator, value)},
                {"v": {"$ne": None}},
            ]
        }
        if field_type == "metric" and comparator != "!=":
            # BSON orders NaN numerically; MLflow numeric comparisons must exclude it.
            match["$and"].append({"v": {"$ne": float("nan")}})
        return {field: {"$elemMatch": match}}

    @staticmethod
    def _embedded_field(field_type: str) -> str:
        if field_type == "metric":
            return "metrics"
        if field_type == "parameter":
            return "params"
        return "tags"

    @staticmethod
    def _attribute_field(key: str) -> str:
        if key == "run_id":
            return "_id"
        if key == "run_name":
            return "name"
        return key

    @staticmethod
    def _search_order_value(order: RunOrder) -> dict:
        if order.field_type == "attribute":
            field = RunRepository._attribute_field(order.key)
            return {"$ifNull": [f"${field}", None]}
        field = RunRepository._embedded_field(order.field_type)
        return {
            "$ifNull": [
                {
                    "$arrayElemAt": [
                        {
                            "$map": {
                                "input": {
                                    "$filter": {
                                        "input": {"$ifNull": [f"${field}", []]},
                                        "as": "entry",
                                        "cond": {"$eq": ["$$entry.k", {"$literal": order.key}]},
                                    }
                                },
                                "as": "entry",
                                "in": "$$entry.v",
                            }
                        },
                        0,
                    ]
                },
                None,
            ]
        }

    def create(
        self,
        *,
        run_id: str,
        experiment_id: str,
        name: str,
        artifact_uri: str,
        user_id: str,
        status: str,
        start_time: int,
        lifecycle_stage: str,
        tags: Mapping[str, str],
    ) -> RunRecord:
        document = {
            "_id": run_id,
            "experiment_id": experiment_id,
            "name": name,
            "artifact_uri": artifact_uri,
            "user_id": user_id,
            "status": status,
            "start_time": start_time,
            "end_time": None,
            "lifecycle_stage": lifecycle_stage,
            "tags": [{"k": key, "v": value} for key, value in tags.items()],
            "params": [],
            "metrics": [],
            "inputs": {"datasets": [], "models": []},
            "outputs": [],
        }
        with repository_operation("Unable to create run."):
            try:
                self._collection.insert_one(document)
            except DuplicateKeyError as exc:
                raise RepositoryAlreadyExistsError(run_id) from exc
        return RunRecord.from_document(document)

    def find_by_id(self, run_id: str | None) -> RunRecord | None:
        with repository_operation("Unable to get run."):
            document = self._collection.find_one({"_id": run_id})
        return RunRecord.from_document(document) if document is not None else None

    def get_metric_history(
        self,
        *,
        run_id: str,
        metric_key: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[RunMetricRecord]:
        with repository_operation("Unable to read metric history."):
            cursor = (
                self._metrics_collection.find(
                    {"scope": RUN_METRIC_SCOPE, "run_id": run_id, "k": metric_key}
                )
                .sort(
                    [
                        ("timestamp", ASCENDING),
                        ("step", ASCENDING),
                        ("v", ASCENDING),
                        # Break ties consistently when otherwise identical events are stored.
                        ("_id", ASCENDING),
                    ]
                )
                .skip(offset)
            )
            if limit is not None:
                cursor = cursor.limit(limit)
            with cursor:
                documents = list(cursor)
        return [RunMetricRecord.from_document(document) for document in documents]

    def mark_deleted(
        self,
        *,
        run_id: str | None,
        deleted_time: int,
    ) -> RunRecord:
        return self._transition_lifecycle(
            run_id=run_id,
            next_stage=LifecycleStage.DELETED,
            updates={"deleted_time": deleted_time},
        )

    def mark_all_deleted(self, *, experiment_id: str, deleted_time: int) -> None:
        with repository_operation("Unable to delete experiment runs."):
            self._collection.update_many(
                {"experiment_id": experiment_id},
                {
                    "$set": {
                        "lifecycle_stage": LifecycleStage.DELETED,
                        "deleted_time": deleted_time,
                    }
                },
            )

    def restore_all(self, *, experiment_id: str) -> None:
        with repository_operation("Unable to restore experiment runs."):
            self._collection.update_many(
                {"experiment_id": experiment_id},
                {"$set": {"lifecycle_stage": LifecycleStage.ACTIVE, "deleted_time": None}},
            )

    def restore(self, *, run_id: str | None) -> RunRecord:
        return self._transition_lifecycle(
            run_id=run_id,
            next_stage=LifecycleStage.ACTIVE,
            updates={"deleted_time": None},
        )

    def update_info(
        self,
        *,
        run_id: str | None,
        status: str | None,
        end_time: int | None,
        run_name: str | None,
    ) -> RunRecord:
        # Keep the pipeline valid even when no optional fields are supplied.
        # The conditional update preserves this lifecycle value.
        fields_to_update = {"lifecycle_stage": {"$literal": LifecycleStage.ACTIVE}}
        if status is not None:
            fields_to_update["status"] = {"$literal": status}
        if end_time is not None:
            fields_to_update["end_time"] = {"$literal": end_time}
        if run_name:
            # Literal values prevent names starting with '$' from being treated
            # as aggregation expressions. Replace the name tag in the same write.
            fields_to_update["name"] = {"$literal": run_name}
            fields_to_update["tags"] = build_merge_array_expression(
                "$tags", [{"k": MLFLOW_RUN_NAME, "v": run_name}], "k"
            )

        with repository_operation("Unable to update run info."):
            updated = self._collection.find_one_and_update(
                {"_id": run_id},
                [
                    {
                        "$replaceWith": {
                            "$cond": [
                                {"$eq": ["$lifecycle_stage", LifecycleStage.ACTIVE]},
                                {"$mergeObjects": ["$$ROOT", fields_to_update]},
                                "$$ROOT",
                            ]
                        }
                    }
                ],
                return_document=ReturnDocument.AFTER,
            )
        if updated is None:
            raise RepositoryNotFoundError(run_id)
        if updated.get("lifecycle_stage") != LifecycleStage.ACTIVE:
            raise RepositoryNotActiveError(run_id, updated.get("lifecycle_stage"))
        return RunRecord.from_document(updated)

    def log_batch(self, *, run_id, metrics, params, tags) -> None:
        run_metrics = [
            {field: metric[field] for field in ("k", "v", "timestamp", "step")}
            for metric in metrics
        ]
        conditions = [{"$eq": ["$lifecycle_stage", LifecycleStage.ACTIVE]}]
        for parameter in params:
            conflicting_params = {
                "$filter": {
                    "input": {"$ifNull": ["$params", []]},
                    "as": "stored",
                    "cond": {
                        "$and": [
                            {"$eq": ["$$stored.k", {"$literal": parameter["k"]}]},
                            {"$ne": ["$$stored.v", {"$literal": parameter["v"]}]},
                        ]
                    },
                }
            }
            conditions.append({"$eq": [{"$size": conflicting_params}, 0]})

        fields_to_update = {}
        if metrics:
            fields_to_update["metrics"] = self._latest_metrics_expression(run_metrics)
        if params:
            params_by_key = {param["k"]: param for param in params}
            fields_to_update["params"] = build_merge_array_expression(
                "$params", list(params_by_key.values()), "k"
            )
        if tags:
            # Resolve duplicate batch keys once, with the last value winning.
            tags_by_key = {tag["k"]: tag for tag in tags}
            fields_to_update["tags"] = build_merge_array_expression(
                "$tags", list(tags_by_key.values()), "k"
            )
            if MLFLOW_RUN_NAME in tags_by_key:
                fields_to_update["name"] = {"$literal": tags_by_key[MLFLOW_RUN_NAME]["v"]}

        with repository_operation("Unable to update run batch data."):
            previous = self._collection.find_one_and_update(
                {"_id": run_id},
                [
                    {
                        "$replaceWith": {
                            "$cond": [
                                {"$and": conditions},
                                {"$mergeObjects": ["$$ROOT", fields_to_update]},
                                "$$ROOT",
                            ]
                        }
                    }
                ],
                projection={"_id": 0, "lifecycle_stage": 1, "params": 1},
                return_document=ReturnDocument.BEFORE,
            )
        if previous is None:
            raise RepositoryNotFoundError(run_id)
        if previous["lifecycle_stage"] != LifecycleStage.ACTIVE:
            raise RepositoryNotActiveError(run_id, previous["lifecycle_stage"])
        existing_params = {
            parameter["k"]: parameter["v"] for parameter in previous.get("params", [])
        }
        for parameter in params:
            if parameter["k"] in existing_params:
                old_value = existing_params[parameter["k"]]
                if old_value != parameter["v"]:
                    raise RepositoryParamConflictError(
                        parameter["k"], old_value, parameter["v"], run_id
                    )

        if metrics:
            metric_documents = [
                {**metric, "run_id": run_id, "scope": RUN_METRIC_SCOPE} for metric in run_metrics
            ]
            metric_documents.extend(
                {**metric, "run_id": run_id, "scope": MODEL_METRIC_SCOPE}
                for metric in metrics
                if metric.get("model_id") is not None
            )
            operations = [
                UpdateOne(
                    # Match the complete point: a conflicting model value or dataset
                    # must violate its unique index rather than become a silent no-op.
                    document,
                    {"$setOnInsert": document},
                    upsert=True,
                )
                for document in metric_documents
            ]
            with repository_operation("Unable to insert run batch metrics."):
                self._metrics_collection.bulk_write(operations, ordered=True)

    def log_inputs(self, *, run_id: str, datasets: list[dict], models: list[dict]) -> None:
        """Attach dataset and model inputs with one atomic database operation."""
        inputs = {
            "datasets": self._append_inputs_expression(
                "$inputs.datasets", datasets, ("dataset.name", "dataset.digest")
            ),
            "models": self._append_inputs_expression("$inputs.models", models, ("model_id",)),
        }
        # Match by ID and guard the mutation inside the pipeline so the same
        # operation distinguishes a missing run from an inactive one.
        fields_to_update = {
            "inputs": {
                "$cond": [{"$eq": ["$lifecycle_stage", LifecycleStage.ACTIVE]}, inputs, "$inputs"]
            }
        }
        with repository_operation("Unable to log run inputs."):
            previous = self._collection.find_one_and_update(
                {"_id": run_id},
                [{"$set": fields_to_update}],
                projection={"_id": 0, "lifecycle_stage": 1},
                return_document=ReturnDocument.BEFORE,
            )
        if previous is None:
            raise RepositoryNotFoundError(run_id)
        if previous["lifecycle_stage"] != LifecycleStage.ACTIVE:
            raise RepositoryNotActiveError(run_id, previous["lifecycle_stage"])

    def log_outputs(self, *, run_id: str, models: list[dict]) -> None:
        """Append model outputs, preserving submission order and duplicates."""
        outputs = {"$concatArrays": [{"$ifNull": ["$outputs", []]}, {"$literal": models}]}
        fields_to_update = {
            "outputs": {
                "$cond": [{"$eq": ["$lifecycle_stage", LifecycleStage.ACTIVE]}, outputs, "$outputs"]
            }
        }
        with repository_operation("Unable to log run outputs."):
            previous = self._collection.find_one_and_update(
                {"_id": run_id},
                [{"$set": fields_to_update}],
                projection={"_id": 0, "lifecycle_stage": 1},
                return_document=ReturnDocument.BEFORE,
            )
        if previous is None:
            raise RepositoryNotFoundError(run_id)
        if previous["lifecycle_stage"] != LifecycleStage.ACTIVE:
            raise RepositoryNotActiveError(run_id, previous["lifecycle_stage"])

    @staticmethod
    def _append_inputs_expression(field: str, inputs: list[dict], identity: tuple[str, ...]):
        """Append unseen inputs, preserving the first entry and its tags."""
        existing_keys = {
            "$map": {
                "input": "$$value",
                "as": "stored",
                "in": [f"$$stored.{key}" for key in identity],
            }
        }
        return {
            "$reduce": {
                "input": {"$literal": inputs},
                "initialValue": {"$ifNull": [field, []]},
                "in": {
                    "$cond": [
                        {"$in": [[f"$$this.{key}" for key in identity], existing_keys]},
                        "$$value",
                        {"$concatArrays": ["$$value", ["$$this"]]},
                    ]
                },
            }
        }

    @staticmethod
    def _latest_metrics_expression(metrics):
        """Merge a batch against the summaries present when the update executes."""
        matching_metrics = {
            "$filter": {
                "input": "$$value",
                "as": "stored",
                "cond": {"$eq": ["$$stored.k", "$$this.k"]},
            }
        }
        other_metrics = {
            "$filter": {
                "input": "$$value",
                "as": "stored",
                "cond": {"$ne": ["$$stored.k", "$$this.k"]},
            }
        }

        def ordering(metric):
            # MLflow ranks NaN as zero for value ties. Preserve the original
            # BSON value (including NaN and infinities) in the stored metric.
            value = f"{metric}.v"
            return [
                f"{metric}.step",
                f"{metric}.timestamp",
                {"$cond": [{"$eq": [value, {"$literal": float("nan")}]}, 0, value]},
            ]

        return {
            "$reduce": {
                "input": {"$literal": metrics},
                "initialValue": {"$ifNull": ["$metrics", []]},
                "in": {
                    "$let": {
                        "vars": {
                            "current": {"$ifNull": [{"$arrayElemAt": [matching_metrics, 0]}, None]}
                        },
                        "in": {
                            "$cond": [
                                {
                                    "$or": [
                                        {"$eq": ["$$current", None]},
                                        {"$gt": [ordering("$$this"), ordering("$$current")]},
                                    ]
                                },
                                {"$concatArrays": [other_metrics, ["$$this"]]},
                                "$$value",
                            ]
                        },
                    }
                },
            }
        }

    def _transition_lifecycle(
        self,
        *,
        run_id: str | None,
        next_stage: str,
        updates: dict[str, object],
    ) -> RunRecord:
        updates = {"lifecycle_stage": next_stage, **updates}
        with repository_operation("Unable to update run lifecycle."):
            document = self._collection.find_one_and_update(
                {"_id": run_id},
                {"$set": updates},
                return_document=ReturnDocument.AFTER,
            )
        if document is None:
            raise RepositoryNotFoundError(run_id)
        return RunRecord.from_document(document)
