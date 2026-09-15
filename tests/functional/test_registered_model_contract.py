"""Functional registered-model contracts shared with MLflow's reference stores."""

import pytest
from mlflow.entities.model_registry import RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from mlflow_mongodb import MongoDBModelRegistryStore


def _latest_versions_by_stage(versions):
    return {version.current_stage: version.version for version in versions}


def test_registered_model_create_get_and_update_contract(
    store: MongoDBModelRegistryStore,
    monkeypatch,
):
    clock = {"now": 1_700_000_000_000}
    monkeypatch.setattr(
        "mlflow_mongodb.model_registry_store.get_current_time_millis",
        lambda: clock["now"],
    )

    model = store.create_registered_model(
        "registered-model-contract",
        tags=[
            RegisteredModelTag("team", "risk"),
            RegisteredModelTag("environment", "test"),
        ],
    )

    assert model.name == "registered-model-contract"
    assert model.description is None
    assert model.creation_timestamp == clock["now"]
    assert model.last_updated_timestamp == clock["now"]
    assert model.latest_versions == []
    assert model.tags == {"team": "risk", "environment": "test"}

    stored = store.get_registered_model(model.name)
    assert stored.creation_timestamp == model.creation_timestamp
    assert stored.last_updated_timestamp == model.last_updated_timestamp
    assert stored.latest_versions == []
    assert stored.tags == model.tags

    clock["now"] += 1_000
    updated = store.update_registered_model(
        model.name,
        "Reviewed model",
        deployment_job_id=42,
    )
    assert updated.description == "Reviewed model"
    assert updated.deployment_job_id == "42"
    assert updated.last_updated_timestamp == clock["now"]
    assert store.get_registered_model(model.name).description == "Reviewed model"
    assert store.get_registered_model(model.name).deployment_job_id == "42"

    with pytest.raises(MlflowException, match="already exists") as duplicate_error:
        store.create_registered_model(model.name)
    assert duplicate_error.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

    other_model = store.create_registered_model(f"{model.name}-other")
    assert other_model.name == f"{model.name}-other"

    for invalid_name in (None, ""):
        with pytest.raises(
            MlflowException,
            match="Missing value for required parameter 'name'",
        ) as invalid_name_error:
            store.create_registered_model(invalid_name)
        assert invalid_name_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_rename_registered_model_preserves_versions_and_rejects_conflicts(
    store: MongoDBModelRegistryStore,
    monkeypatch,
):
    original_name = "rename-contract-original"
    new_name = "rename-contract-new"

    with pytest.raises(MlflowException, match="Registered Model.*not found") as missing_error:
        store.rename_registered_model("rename-contract-missing", "rename-contract-missing-new")
    assert missing_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    store.create_registered_model(original_name)
    store.create_model_version(original_name, "s3://models/rename/1")
    store.create_model_version(original_name, "s3://models/rename/2")

    rename_timestamp = 1_700_000_100_000
    monkeypatch.setattr(
        "mlflow_mongodb.model_registry_store.get_current_time_millis",
        lambda: rename_timestamp,
    )

    renamed = store.rename_registered_model(original_name, new_name)

    assert renamed.name == new_name
    assert [version.version for version in renamed.latest_versions] == [2]
    for version_number in (1, 2):
        version = store.get_model_version(new_name, version_number)
        assert version.name == new_name
        assert version.last_updated_timestamp == rename_timestamp

    with pytest.raises(MlflowException, match="Registered Model.*not found") as old_name_error:
        store.get_registered_model(original_name)
    assert old_name_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    store.create_registered_model(original_name)
    with pytest.raises(MlflowException, match="already exists") as conflict_error:
        store.rename_registered_model(new_name, original_name)
        assert conflict_error.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

    for invalid_name in (None, ""):
        with pytest.raises(
            MlflowException,
            match="Missing value for required parameter 'new_name'",
        ) as invalid_name_error:
            store.rename_registered_model(original_name, invalid_name)
        assert invalid_name_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_delete_registered_model_removes_versions_and_is_not_idempotent(
    store: MongoDBModelRegistryStore,
):
    name = "delete-registered-model-contract"
    store.create_registered_model(name)
    version = store.create_model_version(name, "s3://models/delete/1")

    store.delete_registered_model(name)

    operations = (
        lambda: store.get_registered_model(name),
        lambda: store.update_registered_model(name, "deleted"),
        lambda: store.delete_registered_model(name),
        lambda: store.get_model_version(name, version.version),
    )
    for operation in operations:
        with pytest.raises(MlflowException, match="not found") as missing_error:
            operation()
        assert missing_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_get_latest_versions_selects_each_stage_and_falls_back_after_deletion(
    store: MongoDBModelRegistryStore,
):
    name = "latest-versions-contract"
    store.create_registered_model(name)
    assert store.get_registered_model(name).latest_versions == []

    version_one = store.create_model_version(name, "s3://models/latest/1")
    version_two = store.create_model_version(name, "s3://models/latest/2")
    store.transition_model_version_stage(name, version_two.version, "Production", False)
    version_three = store.create_model_version(name, "s3://models/latest/3")
    store.transition_model_version_stage(name, version_three.version, "Production", False)
    version_four = store.create_model_version(name, "s3://models/latest/4")
    store.transition_model_version_stage(name, version_four.version, "Staging", False)

    expected = {"None": version_one.version, "Production": 3, "Staging": 4}
    assert _latest_versions_by_stage(store.get_registered_model(name).latest_versions) == expected
    assert _latest_versions_by_stage(store.get_latest_versions(name, stages=None)) == expected
    assert _latest_versions_by_stage(store.get_latest_versions(name, stages=[])) == expected
    assert _latest_versions_by_stage(store.get_latest_versions(name, ["pROduction"])) == {
        "Production": 3
    }
    assert _latest_versions_by_stage(store.get_latest_versions(name, ["None", "Production"])) == {
        "None": 1,
        "Production": 3,
    }

    store.delete_model_version(name, version_three.version)

    expected_after_deletion = {"None": 1, "Production": 2, "Staging": 4}
    assert (
        _latest_versions_by_stage(store.get_registered_model(name).latest_versions)
        == expected_after_deletion
    )
    assert _latest_versions_by_stage(store.get_latest_versions(name, ["Production"])) == {
        "Production": 2
    }


def test_registered_model_tags_replace_delete_and_remain_isolated(
    store: MongoDBModelRegistryStore,
):
    first_name = "registered-model-tags-first"
    second_name = "registered-model-tags-second"
    initial_tags = [
        RegisteredModelTag("key", "value"),
        RegisteredModelTag("anotherKey", "another value"),
    ]
    store.create_registered_model(first_name, tags=initial_tags)
    store.create_registered_model(second_name, tags=initial_tags)

    store.set_registered_model_tag(first_name, RegisteredModelTag("new", "new value"))
    store.set_registered_model_tag(first_name, RegisteredModelTag("key", "overridden"))

    assert store.get_registered_model(first_name).tags == {
        "key": "overridden",
        "anotherKey": "another value",
        "new": "new value",
    }
    assert store.get_registered_model(second_name).tags == {
        "key": "value",
        "anotherKey": "another value",
    }

    store.delete_registered_model_tag(first_name, "new")
    store.delete_registered_model_tag(first_name, "key")
    store.delete_registered_model_tag(first_name, "key")
    assert store.get_registered_model(first_name).tags == {"anotherKey": "another value"}
    assert store.get_registered_model(second_name).tags == {
        "key": "value",
        "anotherKey": "another value",
    }

    store.set_registered_model_tag(
        second_name,
        RegisteredModelTag("longTagKey", "a" * 4_999),
    )
    with pytest.raises(
        MlflowException,
        match="'value' exceeds the maximum length",
    ) as long_tag_error:
        store.set_registered_model_tag(
            second_name,
            RegisteredModelTag("longTagKey", "a" * 100_001),
        )
    assert long_tag_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(
        MlflowException,
        match="Missing value for required parameter 'key'",
    ) as invalid_tag_error:
        store.set_registered_model_tag(
            second_name,
            RegisteredModelTag(key=None, value=""),
        )
    assert invalid_tag_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(
        MlflowException,
        match="Missing value for required parameter 'key'",
    ) as invalid_key_error:
        store.delete_registered_model_tag(second_name, None)
    assert invalid_key_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    for operation in (
        lambda: store.set_registered_model_tag(None, RegisteredModelTag("key", "value")),
        lambda: store.delete_registered_model_tag(None, "key"),
    ):
        with pytest.raises(
            MlflowException,
            match="Missing value for required parameter 'name'",
        ) as invalid_name_error:
            operation()
        assert invalid_name_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    store.delete_registered_model(first_name)
    for operation in (
        lambda: store.set_registered_model_tag(
            first_name,
            RegisteredModelTag("key", "value"),
        ),
        lambda: store.delete_registered_model_tag(first_name, "anotherKey"),
    ):
        with pytest.raises(MlflowException, match="Registered Model.*not found") as missing_error:
            operation()
        assert missing_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
