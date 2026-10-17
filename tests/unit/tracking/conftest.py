from collections.abc import Mapping
from unittest.mock import create_autospec
from uuid import UUID

import pytest
from mlflow.entities import LifecycleStage

from mlflow_mongodb.tracking.repositories import (
    ExperimentRepository,
    LoggedModelPage,
    LoggedModelRepository,
    RunRepository,
)
from mlflow_mongodb.tracking.store import MongoDBTrackingStore
from mlflow_mongodb.tracking.types import (
    ExperimentRecord,
    ExperimentTagRecord,
    LoggedModelRecord,
    RunRecord,
)

FIXED_TIMESTAMP = 1_700_000_000_000
EXPERIMENT_ID = "42"
ARTIFACT_URI = "s3://experiment-artifacts"


@pytest.fixture
def experiment_repository():
    return create_autospec(ExperimentRepository, instance=True)


@pytest.fixture
def store(monkeypatch, experiment_repository):
    monkeypatch.setattr(
        "mlflow_mongodb.tracking.store.get_current_time_millis",
        lambda: FIXED_TIMESTAMP,
    )
    monkeypatch.setattr("mlflow_mongodb.tracking.store.uuid4", lambda: UUID(int=42))
    mongodb_store = MongoDBTrackingStore(
        store_uri="mongodb://localhost:27017/mlflow",
        artifact_uri=ARTIFACT_URI,
    )
    mongodb_store.__dict__["_experiment_repository"] = experiment_repository
    return mongodb_store


@pytest.fixture
def experiment_record_factory():
    def factory(
        *,
        experiment_id=EXPERIMENT_ID,
        name="example-experiment",
        artifact_location=f"{ARTIFACT_URI}/{EXPERIMENT_ID}",
        lifecycle_stage=LifecycleStage.ACTIVE,
        creation_time=FIXED_TIMESTAMP,
        last_update_time=FIXED_TIMESTAMP,
        tags: Mapping[str, str] | None = None,
    ):
        return ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_location,
            lifecycle_stage=lifecycle_stage,
            creation_time=creation_time,
            last_update_time=last_update_time,
            tags=tuple(
                ExperimentTagRecord(key=key, value=value) for key, value in (tags or {}).items()
            ),
        )

    return factory


@pytest.fixture
def run_repository():
    return create_autospec(RunRepository, instance=True)


@pytest.fixture
def run_document_factory():
    def factory(**overrides):
        document = {
            "_id": UUID(int=42).hex,
            "experiment_id": EXPERIMENT_ID,
            "name": "example-run",
            "artifact_uri": f"{ARTIFACT_URI}/{EXPERIMENT_ID}/{UUID(int=42).hex}/artifacts",
            "user_id": "test-user",
            "status": "RUNNING",
            "start_time": FIXED_TIMESTAMP,
            "end_time": None,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "tags": [{"k": "team", "v": "risk"}],
            "params": [],
            "metrics": [],
            "inputs": {"datasets": [], "models": []},
            "outputs": [],
        }
        document.update(overrides)
        return document

    return factory


@pytest.fixture
def run_record_factory(run_document_factory):
    def factory(**overrides):
        return RunRecord.from_document(run_document_factory(**overrides))

    return factory


@pytest.fixture
def run_store(
    store,
    run_repository,
    experiment_repository,
    experiment_record_factory,
    run_record_factory,
):
    store.__dict__["_run_repository"] = run_repository
    experiment_repository.find_by_id.return_value = experiment_record_factory()
    for method in ("create", "find_by_id", "update_info", "mark_deleted", "restore"):
        getattr(run_repository, method).return_value = run_record_factory()
    run_repository.get_metric_history.return_value = []
    return store


@pytest.fixture
def logged_model_repository():
    return create_autospec(LoggedModelRepository, instance=True)


@pytest.fixture
def logged_model_document_factory():
    def factory(**overrides):
        document = {
            "_id": f"m-{UUID(int=42).hex}",
            "experiment_id": EXPERIMENT_ID,
            "name": "example-model",
            "artifact_location": (
                f"{ARTIFACT_URI}/{EXPERIMENT_ID}/models/m-{UUID(int=42).hex}/artifacts"
            ),
            "creation_timestamp": FIXED_TIMESTAMP,
            "last_updated_timestamp": FIXED_TIMESTAMP,
            "status": "PENDING",
            "status_message": None,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "source_run_id": None,
            "model_type": None,
            "tags": [{"k": "team", "v": "risk"}],
            "params": [{"k": "depth", "v": "3"}],
        }
        document.update(overrides)
        return document

    return factory


@pytest.fixture
def logged_model_record_factory(logged_model_document_factory):
    def factory(**overrides):
        return LoggedModelRecord.from_document(logged_model_document_factory(**overrides))

    return factory


@pytest.fixture
def logged_model_store(
    store,
    logged_model_repository,
    experiment_repository,
    experiment_record_factory,
    logged_model_record_factory,
):
    store.__dict__["_logged_model_repository"] = logged_model_repository
    experiment_repository.find_by_id.return_value = experiment_record_factory()
    logged_model_repository.create.return_value = logged_model_record_factory()
    logged_model_repository.find_by_id.return_value = logged_model_record_factory()
    logged_model_repository.get_metric_history.return_value = ()
    logged_model_repository.search.return_value = LoggedModelPage((), False)
    return store
