"""Persistence operations for runs."""

from collections.abc import Mapping

from mlflow.entities import LifecycleStage
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from pymongo import ASCENDING, ReturnDocument
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
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.types import RunMetricRecord, RunRecord


class RunRepository:
    """Store run documents with small run fields embedded in the document."""

    EXPERIMENT_INDEX = "runs_experiment_id"

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
            )

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
                self._metrics_collection.find({"run_id": run_id, "k": metric_key})
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
        # The filter already guarantees that this lifecycle value is unchanged.
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
                {"_id": run_id, "lifecycle_stage": LifecycleStage.ACTIVE},
                [{"$set": fields_to_update}],
                return_document=ReturnDocument.AFTER,
            )
        if updated is None:
            raise RepositoryNotFoundError(run_id)
        return RunRecord.from_document(updated)

    def log_batch(self, *, run_id, metrics, params, tags) -> None:
        with repository_operation("Unable to read run for batch logging."):
            document = self._collection.find_one(
                {"_id": run_id},
                {"_id": 0, "lifecycle_stage": 1, "params": 1},
            )
        if document is None:
            raise RepositoryNotFoundError(run_id)
        if document["lifecycle_stage"] != LifecycleStage.ACTIVE:
            raise RepositoryNotActiveError(run_id, document["lifecycle_stage"])

        existing_params = {
            parameter["k"]: parameter["v"] for parameter in document.get("params", [])
        }
        for parameter in params:
            old_value = existing_params.get(parameter["k"])
            if old_value is not None and old_value != parameter["v"]:
                raise RepositoryParamConflictError(
                    parameter["k"], old_value, parameter["v"], run_id
                )

        if metrics:
            metric_documents = [{**metric, "run_id": run_id} for metric in metrics]
            with repository_operation("Unable to insert run batch metrics."):
                self._metrics_collection.insert_many(metric_documents, ordered=True)

        fields_to_update = {}
        if metrics:
            fields_to_update["metrics"] = self._latest_metrics_expression(metrics)
        if params:
            new_params = [param for param in params if param["k"] not in existing_params]
            fields_to_update["params"] = build_merge_array_expression("$params", new_params, "k")
        if tags:
            # Resolve duplicate batch keys once, with the last value winning.
            tags_by_key = {tag["k"]: tag for tag in tags}
            fields_to_update["tags"] = build_merge_array_expression(
                "$tags", list(tags_by_key.values()), "k"
            )
            if MLFLOW_RUN_NAME in tags_by_key:
                fields_to_update["name"] = {"$literal": tags_by_key[MLFLOW_RUN_NAME]["v"]}

        if fields_to_update:
            with repository_operation("Unable to update run batch data."):
                result = self._collection.update_one(
                    {"_id": run_id, "lifecycle_stage": LifecycleStage.ACTIVE},
                    [{"$set": fields_to_update}],
                )
            if result.matched_count == 0:
                raise RepositoryNotFoundError(run_id)

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
