"""Functional tests for the registered-model lifecycle against MongoDB."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from mlflow_mongodb import MongoDBModelRegistryStore

MODEL_NAME = "mongodb-functional-model"
RENAMED_MODEL_NAME = "mongodb-functional-model-renamed"


def test_registered_model_and_version_lifecycle(store: MongoDBModelRegistryStore):
    created = store.create_registered_model(
        MODEL_NAME,
        tags=[RegisteredModelTag("team", "risk")],
        description="Initial model",
    )
    assert created.name == MODEL_NAME
    assert created.description == "Initial model"
    assert created.tags == {"team": "risk"}

    updated = store.update_registered_model(MODEL_NAME, "Updated model")
    assert updated.description == "Updated model"

    renamed = store.rename_registered_model(MODEL_NAME, RENAMED_MODEL_NAME)
    assert renamed.name == RENAMED_MODEL_NAME
    with pytest.raises(MlflowException, match="Registered Model.*not found") as old_name_error:
        store.get_registered_model(MODEL_NAME)
    assert old_name_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    store.set_registered_model_tag(
        RENAMED_MODEL_NAME,
        RegisteredModelTag("environment", "functional"),
    )
    assert store.get_registered_model(RENAMED_MODEL_NAME).tags == {
        "team": "risk",
        "environment": "functional",
    }
    store.delete_registered_model_tag(RENAMED_MODEL_NAME, "team")
    assert store.get_registered_model(RENAMED_MODEL_NAME).tags == {"environment": "functional"}

    version_one = store.create_model_version(
        RENAMED_MODEL_NAME,
        source="s3://models/mongodb-functional-model/1",
        run_id="run-1",
        tags=[ModelVersionTag("candidate", "one")],
        description="First version",
    )
    version_two = store.create_model_version(
        RENAMED_MODEL_NAME,
        source="s3://models/mongodb-functional-model/2",
        run_id="run-2",
        tags=[ModelVersionTag("candidate", "two")],
        description="Second version",
    )
    assert (version_one.version, version_two.version) == (1, 2)

    updated_version = store.update_model_version(
        RENAMED_MODEL_NAME,
        version_one.version,
        "First version, reviewed",
    )
    assert updated_version.description == "First version, reviewed"

    store.transition_model_version_stage(
        RENAMED_MODEL_NAME,
        version_one.version,
        "Production",
        archive_existing_versions=False,
    )
    transitioned = store.transition_model_version_stage(
        RENAMED_MODEL_NAME,
        version_two.version,
        "Production",
        archive_existing_versions=True,
    )
    assert transitioned.current_stage == "Production"
    assert (
        store.get_model_version(RENAMED_MODEL_NAME, version_one.version).current_stage == "Archived"
    )
    assert (
        store.get_model_version(RENAMED_MODEL_NAME, version_two.version).current_stage
        == "Production"
    )
    assert {
        (version.version, version.current_stage)
        for version in store.get_latest_versions(
            RENAMED_MODEL_NAME,
            stages=["Production", "Archived"],
        )
    } == {(1, "Archived"), (2, "Production")}

    store.set_registered_model_alias(
        RENAMED_MODEL_NAME,
        "champion",
        version_two.version,
    )
    assert store.get_model_version_by_alias(RENAMED_MODEL_NAME, "champion").version == 2

    store.delete_model_version(RENAMED_MODEL_NAME, version_two.version)
    with pytest.raises(MlflowException, match="Model Version.*not found") as deleted_version_error:
        store.get_model_version(RENAMED_MODEL_NAME, version_two.version)
    assert deleted_version_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert store.get_registered_model(RENAMED_MODEL_NAME).aliases == {}

    registered_model = store._registered_model_repository.find_by_name(RENAMED_MODEL_NAME)
    assert registered_model is not None
    store.delete_registered_model(RENAMED_MODEL_NAME)

    with pytest.raises(MlflowException, match="Registered Model.*not found") as deleted_model_error:
        store.get_registered_model(RENAMED_MODEL_NAME)
    assert deleted_model_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert (
        store._database[store._settings.model_versions_collection_name].count_documents(
            {"registered_model_id": registered_model.model_id}
        )
        == 0
    )


def test_concurrent_model_version_allocation_is_unique_and_sequential(
    store: MongoDBModelRegistryStore,
):
    version_count = 8
    store.create_registered_model(MODEL_NAME)
    start_barrier = threading.Barrier(version_count)

    def create_version(worker_number):
        start_barrier.wait(timeout=10)
        return store.create_model_version(
            MODEL_NAME,
            source=f"s3://models/concurrent/{worker_number}",
        )

    with ThreadPoolExecutor(max_workers=version_count) as executor:
        versions = list(executor.map(create_version, range(version_count)))

    assert sorted(version.version for version in versions) == list(range(1, version_count + 1))
    stored_versions = store.search_model_versions(
        f"name = '{MODEL_NAME}'",
        max_results=version_count,
        order_by=["version_number ASC"],
    )
    assert [version.version for version in stored_versions] == list(range(1, version_count + 1))
