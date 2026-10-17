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
    RepositoryNotFoundError,
)
from mlflow_mongodb.infrastructure.repository_operations import repository_operation
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.types import RunRecord


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
