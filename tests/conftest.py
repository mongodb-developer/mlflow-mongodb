"""Shared fixtures for unit tests."""

from collections.abc import Mapping
from unittest.mock import create_autospec

import pytest
from bson import ObjectId
from mlflow.entities.model_registry.model_version_stages import STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient

from mlflow_mongodb import MongoDBModelRegistryStore
from mlflow_mongodb.repositories import (
    ModelVersionRecord,
    ModelVersionRepository,
    ModelVersionTagRecord,
    RegisteredModelAliasRecord,
    RegisteredModelDetails,
    RegisteredModelRecord,
    RegisteredModelRepository,
    RegisteredModelTagRecord,
)

FIXED_TIMESTAMP = 1_700_000_000_000
REGISTERED_MODEL_ID = ObjectId("64b000000000000000000001")
MODEL_VERSION_ID = ObjectId("64b000000000000000000002")


@pytest.fixture
def registered_model_repository():
    """Return a strict mock of the registered-model repository."""
    return create_autospec(RegisteredModelRepository, instance=True)


@pytest.fixture
def model_version_repository():
    """Return a strict mock of the model-version repository."""
    return create_autospec(ModelVersionRepository, instance=True)


@pytest.fixture
def tracking_client():
    """Return a strict mock of the MLflow tracking client."""
    return create_autospec(MlflowClient, instance=True)


@pytest.fixture
def store(monkeypatch, registered_model_repository, model_version_repository, tracking_client):
    """Return a store wired to mocks so unit tests never connect to MongoDB."""
    monkeypatch.setattr(
        "mlflow_mongodb.model_registry_store.get_current_time_millis",
        lambda: FIXED_TIMESTAMP,
    )

    mongodb_store = MongoDBModelRegistryStore(
        store_uri="mongodb://localhost:27017/mlflow",
        tracking_uri="http://localhost:5000",
    )
    mongodb_store.__dict__["_registered_model_repository"] = registered_model_repository
    mongodb_store.__dict__["_model_version_repository"] = model_version_repository
    mongodb_store.__dict__["_tracking_client"] = tracking_client
    return mongodb_store


@pytest.fixture
def registered_model_record_factory():
    """Build registered-model records with deterministic defaults."""

    def factory(
        *,
        model_id=REGISTERED_MODEL_ID,
        name="example-model",
        creation_timestamp=FIXED_TIMESTAMP,
        last_updated_timestamp=FIXED_TIMESTAMP,
        description=None,
        tags: Mapping[str, str] | None = None,
        aliases: Mapping[str, int] | None = None,
        deployment_job_id=None,
        version_counter=0,
    ):
        return RegisteredModelRecord(
            model_id=model_id,
            name=name,
            creation_timestamp=creation_timestamp,
            last_updated_timestamp=last_updated_timestamp,
            description=description,
            tags=tuple(
                RegisteredModelTagRecord(key=key, value=value)
                for key, value in (tags or {}).items()
            ),
            aliases=tuple(
                RegisteredModelAliasRecord(alias=alias, version=version)
                for alias, version in (aliases or {}).items()
            ),
            deployment_job_id=deployment_job_id,
            version_counter=version_counter,
        )

    return factory


@pytest.fixture
def model_version_record_factory():
    """Build model-version records with deterministic defaults."""

    def factory(
        *,
        model_version_id=MODEL_VERSION_ID,
        registered_model_id=REGISTERED_MODEL_ID,
        version=1,
        creation_timestamp=FIXED_TIMESTAMP,
        last_updated_timestamp=FIXED_TIMESTAMP,
        description=None,
        user_id=None,
        current_stage=STAGE_NONE,
        source="prompt-template",
        storage_location="prompt-template",
        run_id=None,
        run_link=None,
        status=ModelVersionStatus.to_string(ModelVersionStatus.READY),
        status_message=None,
        tags: Mapping[str, str] | None = None,
        model_id=None,
    ):
        return ModelVersionRecord(
            model_version_id=model_version_id,
            registered_model_id=registered_model_id,
            version=version,
            creation_timestamp=creation_timestamp,
            last_updated_timestamp=last_updated_timestamp,
            description=description,
            user_id=user_id,
            current_stage=current_stage,
            source=source,
            storage_location=storage_location,
            run_id=run_id,
            run_link=run_link,
            status=status,
            status_message=status_message,
            tags=tuple(
                ModelVersionTagRecord(key=key, value=value) for key, value in (tags or {}).items()
            ),
            model_id=model_id,
        )

    return factory


@pytest.fixture
def registered_model_details_factory(registered_model_record_factory):
    """Build joined registered-model details returned by repository lookups."""

    def factory(*, registered_model=None, latest_versions=()):
        return RegisteredModelDetails(
            registered_model=registered_model or registered_model_record_factory(),
            latest_versions=tuple(latest_versions),
        )

    return factory
