from collections.abc import Mapping
from unittest.mock import create_autospec
from uuid import UUID

import pytest
from mlflow.entities import LifecycleStage

from mlflow_mongodb.tracking.repositories import ExperimentRepository
from mlflow_mongodb.tracking.store import MongoDBTrackingStore
from mlflow_mongodb.tracking.types import ExperimentRecord, ExperimentTagRecord

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
