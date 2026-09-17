"""Unit tests for MongoDB-owned model-registry store orchestration."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.model_registry.model_version_stages import ALL_STAGES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from mlflow_mongodb import MongoDBModelRegistryStore
from mlflow_mongodb.repositories import (
    ModelVersionAlreadyExistsError,
    ModelVersionNotFoundError,
    RegisteredModelAlreadyExistsError,
    RegisteredModelNotFoundError,
)

MODEL_NAME = "fraud-detector"
RENAMED_MODEL_NAME = "fraud-detector-v2"
SOURCE = "s3://model-artifacts/fraud-detector"
STORAGE_LOCATION = "s3://resolved-model-artifacts/fraud-detector"
FIXED_TIMESTAMP = 1_700_000_000_000


def test_tracking_client_is_lazy_and_cached(monkeypatch):
    tracking_client = Mock()
    client_factory = Mock(return_value=tracking_client)
    monkeypatch.setattr(
        "mlflow_mongodb.model_registry_store.MlflowClient",
        client_factory,
    )

    store = MongoDBModelRegistryStore(
        store_uri="mongodb://localhost:27017/mlflow",
        tracking_uri="http://tracking.example",
    )

    client_factory.assert_not_called()

    assert store._tracking_client is tracking_client
    assert store._tracking_client is tracking_client

    client_factory.assert_called_once_with(
        tracking_uri="http://tracking.example",
    )


def test_create_registered_model_converts_entity(
    store,
    registered_model_repository,
    registered_model_record_factory,
):
    record = registered_model_record_factory(
        name=MODEL_NAME,
        description="Detect fraudulent transactions",
        tags={"team": "risk"},
        deployment_job_id="42",
    )
    registered_model_repository.create.return_value = record

    model = store.create_registered_model(
        MODEL_NAME,
        tags=[RegisteredModelTag("team", "risk")],
        description="Detect fraudulent transactions",
        deployment_job_id=42,
    )

    assert model.name == MODEL_NAME
    assert model.description == "Detect fraudulent transactions"
    assert model.tags == {"team": "risk"}
    assert model.aliases == {}
    assert model.deployment_job_id == "42"
    registered_model_repository.create.assert_called_once_with(
        name=MODEL_NAME,
        creation_timestamp=FIXED_TIMESTAMP,
        description="Detect fraudulent transactions",
        tags={"team": "risk"},
        deployment_job_id="42",
    )


def test_create_registered_model_translates_duplicate_error(
    store,
    registered_model_repository,
    registered_model_record_factory,
):
    registered_model_repository.create.side_effect = RegisteredModelAlreadyExistsError(MODEL_NAME)
    registered_model_repository.find_by_name.return_value = registered_model_record_factory(
        name=MODEL_NAME
    )

    with pytest.raises(MlflowException, match="already exists") as exc_info:
        store.create_registered_model(MODEL_NAME)

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_update_registered_model_converts_entity_with_latest_versions(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
):
    record = registered_model_record_factory(
        name=MODEL_NAME,
        description="Updated description",
        aliases={"champion": 3},
        deployment_job_id="84",
    )
    latest_version = model_version_record_factory(
        registered_model_id=record.model_id,
        version=3,
        current_stage="Production",
        source=SOURCE,
        tags={"quality": "approved"},
    )
    registered_model_repository.update.return_value = record
    model_version_repository.find_latest_by_stages.return_value = (latest_version,)

    model = store.update_registered_model(
        MODEL_NAME,
        "Updated description",
        deployment_job_id=84,
    )

    assert model.description == "Updated description"
    assert model.deployment_job_id == "84"
    assert len(model.latest_versions) == 1
    assert model.latest_versions[0].version == 3
    assert model.latest_versions[0].aliases == ["champion"]
    registered_model_repository.update.assert_called_once_with(
        name=MODEL_NAME,
        last_updated_timestamp=FIXED_TIMESTAMP,
        description="Updated description",
        deployment_job_id="84",
    )
    model_version_repository.find_latest_by_stages.assert_called_once_with(
        registered_model_id=record.model_id,
        stages=ALL_STAGES,
    )


def test_update_registered_model_translates_not_found_error(
    store,
    registered_model_repository,
):
    registered_model_repository.update.side_effect = RegisteredModelNotFoundError(MODEL_NAME)

    with pytest.raises(MlflowException, match="Registered Model.*not found") as exc_info:
        store.update_registered_model(MODEL_NAME, "new description")

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_rename_registered_model_converts_latest_versions(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
):
    renamed_record = registered_model_record_factory(name=RENAMED_MODEL_NAME)
    latest_version = model_version_record_factory(
        registered_model_id=renamed_record.model_id,
        version=2,
    )
    registered_model_repository.rename.return_value = renamed_record
    model_version_repository.find_latest_by_stages.return_value = (latest_version,)

    model = store.rename_registered_model(MODEL_NAME, RENAMED_MODEL_NAME)

    assert model.name == RENAMED_MODEL_NAME
    assert [version.version for version in model.latest_versions] == [2]
    assert model.latest_versions[0].name == RENAMED_MODEL_NAME
    registered_model_repository.rename.assert_called_once_with(
        name=MODEL_NAME,
        new_name=RENAMED_MODEL_NAME,
        last_updated_timestamp=FIXED_TIMESTAMP,
    )


@pytest.mark.parametrize(
    ("repository_error", "expected_code", "message"),
    [
        (
            RegisteredModelNotFoundError(MODEL_NAME),
            ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
            "not found",
        ),
        (
            RegisteredModelAlreadyExistsError(RENAMED_MODEL_NAME),
            ErrorCode.Name(RESOURCE_ALREADY_EXISTS),
            "already exists",
        ),
    ],
)
def test_rename_registered_model_translates_repository_errors(
    store,
    registered_model_repository,
    repository_error,
    expected_code,
    message,
):
    registered_model_repository.rename.side_effect = repository_error

    with pytest.raises(MlflowException, match=message) as exc_info:
        store.rename_registered_model(MODEL_NAME, RENAMED_MODEL_NAME)

    assert exc_info.value.error_code == expected_code


def test_get_registered_model_converts_joined_latest_versions(
    store,
    registered_model_repository,
    registered_model_record_factory,
    model_version_record_factory,
    registered_model_details_factory,
):
    record = registered_model_record_factory(
        name=MODEL_NAME,
        tags={"team": "risk"},
        aliases={"champion": 4},
    )
    latest_version = model_version_record_factory(
        registered_model_id=record.model_id,
        version=4,
        source=SOURCE,
    )
    registered_model_repository.find_by_name_with_latest_versions.return_value = (
        registered_model_details_factory(
            registered_model=record,
            latest_versions=(latest_version,),
        )
    )

    model = store.get_registered_model(MODEL_NAME)

    assert model.tags == {"team": "risk"}
    assert model.aliases == {"champion": 4}
    assert model.latest_versions[0].version == 4
    assert model.latest_versions[0].aliases == ["champion"]


def test_get_registered_model_translates_missing_details(
    store,
    registered_model_repository,
):
    registered_model_repository.find_by_name_with_latest_versions.return_value = None

    with pytest.raises(MlflowException, match="Registered Model.*not found") as exc_info:
        store.get_registered_model(MODEL_NAME)

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_get_latest_versions_canonicalizes_and_deduplicates_stages(
    store,
    registered_model_repository,
    registered_model_record_factory,
    model_version_record_factory,
    registered_model_details_factory,
):
    record = registered_model_record_factory(name=MODEL_NAME, aliases={"candidate": 2})
    version = model_version_record_factory(
        registered_model_id=record.model_id,
        version=2,
        current_stage="Staging",
        source=SOURCE,
    )
    registered_model_repository.find_by_name_with_latest_versions.return_value = (
        registered_model_details_factory(
            registered_model=record,
            latest_versions=(version,),
        )
    )

    latest_versions = store.get_latest_versions(
        MODEL_NAME,
        stages=["staging", "Staging", "production"],
    )

    assert [model_version.version for model_version in latest_versions] == [2]
    assert latest_versions[0].aliases == ["candidate"]
    registered_model_repository.find_by_name_with_latest_versions.assert_called_once_with(
        MODEL_NAME,
        stages=("Staging", "Production"),
    )


def test_get_latest_versions_translates_missing_model(
    store,
    registered_model_repository,
):
    registered_model_repository.find_by_name_with_latest_versions.return_value = None

    with pytest.raises(MlflowException, match="Registered Model.*not found") as exc_info:
        store.get_latest_versions(MODEL_NAME)

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_create_model_version_converts_entity(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
):
    model = registered_model_record_factory(name=MODEL_NAME)
    version_record = model_version_record_factory(
        registered_model_id=model.model_id,
        version=5,
        description="Fifth candidate",
        source=SOURCE,
        storage_location=SOURCE,
        run_id="run-5",
        run_link="https://mlflow.example/runs/run-5",
        tags={"quality": "candidate"},
        model_id="m-5",
    )
    registered_model_repository.find_by_name.return_value = model
    registered_model_repository.allocate_next_version.return_value = 5
    model_version_repository.create.return_value = version_record

    version = store.create_model_version(
        MODEL_NAME,
        SOURCE,
        run_id="run-5",
        tags=[ModelVersionTag("quality", "candidate")],
        run_link="https://mlflow.example/runs/run-5",
        description="Fifth candidate",
        model_id="m-5",
    )

    assert version.name == MODEL_NAME
    assert version.version == 5
    assert version.tags == {"quality": "candidate"}
    assert version.source == SOURCE
    registered_model_repository.allocate_next_version.assert_called_once_with(
        model_id=model.model_id,
        last_updated_timestamp=FIXED_TIMESTAMP,
    )
    create_args = model_version_repository.create.call_args.kwargs
    assert create_args == {
        "registered_model_id": model.model_id,
        "version": 5,
        "creation_timestamp": FIXED_TIMESTAMP,
        "description": "Fifth candidate",
        "current_stage": "None",
        "source": SOURCE,
        "storage_location": SOURCE,
        "run_id": "run-5",
        "run_link": "https://mlflow.example/runs/run-5",
        "status": "READY",
        "tags": {"quality": "candidate"},
        "model_id": "m-5",
    }


@pytest.mark.parametrize("failure_point", ["lookup", "allocation"])
def test_create_model_version_translates_missing_model(
    store,
    registered_model_repository,
    registered_model_record_factory,
    failure_point,
):
    if failure_point == "lookup":
        registered_model_repository.find_by_name.return_value = None
    else:
        registered_model_repository.find_by_name.return_value = registered_model_record_factory(
            name=MODEL_NAME
        )
        registered_model_repository.allocate_next_version.side_effect = (
            RegisteredModelNotFoundError(MODEL_NAME)
        )

    with pytest.raises(MlflowException, match="Registered Model.*not found") as exc_info:
        store.create_model_version(MODEL_NAME, SOURCE)

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_create_model_version_translates_allocated_version_collision(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
):
    registered_model_repository.find_by_name.return_value = registered_model_record_factory(
        name=MODEL_NAME
    )
    registered_model_repository.allocate_next_version.return_value = 3
    model_version_repository.create.side_effect = ModelVersionAlreadyExistsError("collision")

    with pytest.raises(MlflowException, match="allocated version already exists") as exc_info:
        store.create_model_version(MODEL_NAME, SOURCE)

    assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


def test_resolve_registered_model_uri_uses_version_download_location(store, monkeypatch):
    get_download_uri = Mock(return_value=STORAGE_LOCATION)
    monkeypatch.setattr(store, "get_model_version_download_uri", get_download_uri)

    storage_location, run_id = store._resolve_model_version_source(
        "models:/fraud-detector/7",
        "run-7",
        None,
    )

    assert storage_location == STORAGE_LOCATION
    assert run_id == "run-7"
    get_download_uri.assert_called_once_with(MODEL_NAME, "7")


def test_resolve_logged_model_uri_uses_tracking_metadata(store, tracking_client):
    tracking_client.get_logged_model.return_value = SimpleNamespace(
        artifact_location=STORAGE_LOCATION,
        source_run_id="source-run",
    )

    storage_location, run_id = store._resolve_model_version_source(
        "models:/m-123",
        None,
        None,
    )

    assert storage_location == STORAGE_LOCATION
    assert run_id == "source-run"
    tracking_client.get_logged_model.assert_called_once_with("m-123")


def test_resolve_source_fills_run_id_from_explicit_model_id(store, tracking_client):
    tracking_client.get_logged_model.return_value = SimpleNamespace(source_run_id="source-run")

    storage_location, run_id = store._resolve_model_version_source(
        SOURCE,
        None,
        "m-123",
    )

    assert storage_location == SOURCE
    assert run_id == "source-run"
    tracking_client.get_logged_model.assert_called_once_with("m-123")


def test_create_model_version_fills_run_id_from_explicit_model_id(
    store,
    registered_model_repository,
    model_version_repository,
    tracking_client,
    registered_model_record_factory,
    model_version_record_factory,
):
    registered_model = registered_model_record_factory(name=MODEL_NAME)
    version_record = model_version_record_factory(
        registered_model_id=registered_model.model_id,
        version=1,
        source=SOURCE,
        storage_location=SOURCE,
        run_id="source-run",
        model_id="m-123",
    )
    tracking_client.get_logged_model.return_value = SimpleNamespace(source_run_id="source-run")
    registered_model_repository.find_by_name.return_value = registered_model
    registered_model_repository.allocate_next_version.return_value = 1
    model_version_repository.create.return_value = version_record

    created = store.create_model_version(
        MODEL_NAME,
        SOURCE,
        run_id=None,
        model_id="m-123",
    )

    assert created.run_id == "source-run"
    assert model_version_repository.create.call_args.kwargs["run_id"] == "source-run"
    tracking_client.get_logged_model.assert_called_once_with("m-123")


def test_resolve_model_version_source_wraps_resolution_error(store, monkeypatch):
    resolution_error = RuntimeError("tracking service unavailable")
    monkeypatch.setattr(
        store,
        "_resolve_models_uri",
        Mock(side_effect=resolution_error),
    )

    with pytest.raises(
        MlflowException,
        match="Unable to resolve the model source",
    ) as exc_info:
        store._resolve_model_version_source("models:/fraud-detector/7", None, None)

    assert exc_info.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


def test_transition_model_version_stage_converts_entity(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
):
    model = registered_model_record_factory(name=MODEL_NAME)
    touched_model = registered_model_record_factory(
        name=MODEL_NAME,
        aliases={"champion": 2},
    )
    transitioned = model_version_record_factory(
        registered_model_id=model.model_id,
        version=2,
        current_stage="Production",
        source=SOURCE,
    )
    registered_model_repository.find_by_name.return_value = model
    registered_model_repository.touch.return_value = touched_model
    model_version_repository.transition_stage.return_value = transitioned

    version = store.transition_model_version_stage(
        MODEL_NAME,
        "2",
        "production",
        archive_existing_versions=True,
    )

    assert version.version == 2
    assert version.current_stage == "Production"
    assert version.aliases == ["champion"]
    model_version_repository.transition_stage.assert_called_once_with(
        registered_model_id=model.model_id,
        version=2,
        stage="Production",
        last_updated_timestamp=FIXED_TIMESTAMP,
    )
    model_version_repository.archive_other_versions_in_stage.assert_called_once_with(
        registered_model_id=model.model_id,
        version=2,
        stage="Production",
        last_updated_timestamp=FIXED_TIMESTAMP,
    )
    registered_model_repository.touch.assert_called_once_with(
        model_id=model.model_id,
        last_updated_timestamp=FIXED_TIMESTAMP,
    )


@pytest.mark.parametrize("failure_point", ["version", "model_touch"])
def test_transition_model_version_stage_translates_repository_not_found(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
    failure_point,
):
    model = registered_model_record_factory(name=MODEL_NAME)
    registered_model_repository.find_by_name.return_value = model
    if failure_point == "version":
        model_version_repository.transition_stage.side_effect = ModelVersionNotFoundError("2")
    else:
        model_version_repository.transition_stage.return_value = model_version_record_factory(
            registered_model_id=model.model_id,
            version=2,
            current_stage="Staging",
        )
        registered_model_repository.touch.side_effect = RegisteredModelNotFoundError(MODEL_NAME)

    with pytest.raises(MlflowException, match="Model Version.*not found") as exc_info:
        store.transition_model_version_stage(
            MODEL_NAME,
            2,
            "Staging",
            archive_existing_versions=False,
        )

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.parametrize("operation", ["update", "transition", "delete", "delete_tag"])
def test_model_version_operations_translate_missing_registered_model(
    store,
    registered_model_repository,
    operation,
):
    registered_model_repository.find_by_name.return_value = None
    operations = {
        "update": lambda: store.update_model_version(MODEL_NAME, 2, "description"),
        "transition": lambda: store.transition_model_version_stage(
            MODEL_NAME,
            2,
            "Staging",
            archive_existing_versions=False,
        ),
        "delete": lambda: store.delete_model_version(MODEL_NAME, 2),
        "delete_tag": lambda: store.delete_model_version_tag(MODEL_NAME, 2, "team"),
    }

    with pytest.raises(MlflowException, match="Model Version.*not found") as exc_info:
        operations[operation]()

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_set_registered_model_alias_translates_repository_not_found(
    store,
    registered_model_repository,
    model_version_repository,
):
    model_version_repository.exists_for_registered_model.return_value = True
    registered_model_repository.set_alias_by_name.side_effect = RegisteredModelNotFoundError(
        MODEL_NAME
    )

    with pytest.raises(MlflowException, match="Model Version.*not found") as exc_info:
        store.set_registered_model_alias(MODEL_NAME, "candidate", 2)

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_delete_model_version_translates_alias_cleanup_not_found(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
):
    registered_model_repository.find_by_name.return_value = registered_model_record_factory(
        name=MODEL_NAME
    )
    registered_model_repository.delete_aliases_for_version_and_touch.side_effect = (
        RegisteredModelNotFoundError(MODEL_NAME)
    )

    with pytest.raises(MlflowException, match="Model Version.*not found") as exc_info:
        store.delete_model_version(MODEL_NAME, 2)

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    model_version_repository.soft_delete.assert_called_once()


def test_get_model_version_by_alias_translates_missing_alias_target(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
):
    registered_model = registered_model_record_factory(
        name=MODEL_NAME,
        aliases={"candidate": 2},
    )
    registered_model_repository.find_by_name.return_value = registered_model
    model_version_repository.find_by_version.return_value = None

    with pytest.raises(MlflowException, match="Model Version.*not found") as exc_info:
        store.get_model_version_by_alias(MODEL_NAME, "candidate")

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.parametrize(
    ("storage_location", "expected_uri"),
    [(STORAGE_LOCATION, STORAGE_LOCATION), (None, SOURCE)],
)
def test_get_model_version_download_uri_prefers_resolved_storage_location(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
    storage_location,
    expected_uri,
):
    model = registered_model_record_factory(name=MODEL_NAME)
    registered_model_repository.find_by_name.return_value = model
    model_version_repository.find_by_version.return_value = model_version_record_factory(
        registered_model_id=model.model_id,
        source=SOURCE,
        storage_location=storage_location,
    )

    assert store.get_model_version_download_uri(MODEL_NAME, "1") == expected_uri
    model_version_repository.find_by_version.assert_called_once_with(
        registered_model_id=model.model_id,
        version=1,
    )


@pytest.mark.parametrize("model_exists", [False, True])
def test_get_model_version_download_uri_translates_not_found(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_exists,
):
    registered_model_repository.find_by_name.return_value = (
        registered_model_record_factory(name=MODEL_NAME) if model_exists else None
    )
    model_version_repository.find_by_version.return_value = None

    with pytest.raises(MlflowException, match="Model Version.*not found") as exc_info:
        store.get_model_version_download_uri(MODEL_NAME, 99)

    assert exc_info.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
