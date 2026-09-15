"""Functional model-version contracts shared with MLflow's reference stores."""

import pytest
from mlflow.entities.model_registry import ModelVersionTag
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from mlflow_mongodb import MongoDBModelRegistryStore


def test_model_version_create_update_and_stage_validation_contract(
    store: MongoDBModelRegistryStore,
    monkeypatch,
):
    name = "model-version-metadata-contract"
    clock = {"now": 1_700_001_000_000}
    monkeypatch.setattr(
        "mlflow_mongodb.model_registry_store.get_current_time_millis",
        lambda: clock["now"],
    )
    store.create_registered_model(name)

    version = store.create_model_version(
        name,
        source="s3://models/metadata/1",
        run_id="run-metadata",
        tags=[
            ModelVersionTag("quality", "candidate"),
            ModelVersionTag("team", "risk"),
        ],
        run_link="https://mlflow.example/runs/run-metadata",
        description="Initial version",
    )

    assert version.name == name
    assert version.version == 1
    assert version.current_stage == "None"
    assert version.creation_timestamp == clock["now"]
    assert version.last_updated_timestamp == clock["now"]
    assert version.description == "Initial version"
    assert version.source == "s3://models/metadata/1"
    assert version.run_id == "run-metadata"
    assert version.run_link == "https://mlflow.example/runs/run-metadata"
    assert version.status == "READY"
    assert version.status_message is None
    assert version.tags == {"quality": "candidate", "team": "risk"}

    stored = store.get_model_version(name, version.version)
    assert stored.source == version.source
    assert stored.tags == version.tags
    coerced_version = store.get_model_version(name, str(version.version))
    assert coerced_version.version == version.version

    version_without_run = store.create_model_version(name, "s3://models/metadata/2")
    assert version_without_run.version == 2
    assert version_without_run.run_id is None

    clock["now"] += 1_000
    updated = store.update_model_version(name, version.version, "Reviewed version")
    assert updated.description == "Reviewed version"
    assert updated.current_stage == "None"
    assert updated.last_updated_timestamp == clock["now"]

    transitioned = store.transition_model_version_stage(
        name,
        version.version,
        "sTaGiNg",
        archive_existing_versions=False,
    )
    assert transitioned.current_stage == "Staging"
    assert store.get_model_version(name, version.version).description == "Reviewed version"

    with pytest.raises(
        MlflowException,
        match="Invalid Model Version stage",
    ) as invalid_stage_error:
        store.transition_model_version_stage(name, version.version, "unknown", False)
    assert invalid_stage_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_transition_without_archiving_preserves_other_active_versions(
    store: MongoDBModelRegistryStore,
):
    name = "transition-without-archive-contract"
    store.create_registered_model(name)
    version_one = store.create_model_version(name, "s3://models/stages/1")
    version_two = store.create_model_version(name, "s3://models/stages/2")
    version_three = store.create_model_version(name, "s3://models/stages/3")

    for inactive_stage in ("Archived", "None"):
        store.transition_model_version_stage(name, version_one.version, inactive_stage, False)

    store.transition_model_version_stage(name, version_one.version, "Staging", False)
    store.transition_model_version_stage(name, version_two.version, "Production", False)
    store.transition_model_version_stage(name, version_three.version, "Staging", False)
    store.transition_model_version_stage(name, version_three.version, "Production", False)

    assert store.get_model_version(name, version_one.version).current_stage == "Staging"
    assert store.get_model_version(name, version_two.version).current_stage == "Production"
    assert store.get_model_version(name, version_three.version).current_stage == "Production"


def test_transition_with_archiving_rejects_inactive_stages_and_archives_peers(
    store: MongoDBModelRegistryStore,
    monkeypatch,
):
    name = "transition-with-archive-contract"
    store.create_registered_model(name)
    version_one = store.create_model_version(name, "s3://models/archive/1")
    version_two = store.create_model_version(name, "s3://models/archive/2")
    version_three = store.create_model_version(name, "s3://models/archive/3")

    for inactive_stage in ("Archived", "None"):
        with pytest.raises(MlflowException, match="not an Active stage"):
            store.transition_model_version_stage(
                name,
                version_one.version,
                inactive_stage,
                True,
            )

    store.transition_model_version_stage(name, version_one.version, "Staging", False)
    store.transition_model_version_stage(name, version_two.version, "Production", False)

    transition_timestamp = 1_700_002_000_000
    monkeypatch.setattr(
        "mlflow_mongodb.model_registry_store.get_current_time_millis",
        lambda: transition_timestamp,
    )
    store.transition_model_version_stage(name, version_three.version, "STAGING", True)

    archived_one = store.get_model_version(name, version_one.version)
    staged_three = store.get_model_version(name, version_three.version)
    assert archived_one.current_stage == "Archived"
    assert staged_three.current_stage == "Staging"
    assert archived_one.last_updated_timestamp == transition_timestamp
    assert staged_three.last_updated_timestamp == transition_timestamp

    store.transition_model_version_stage(name, version_three.version, "Production", True)
    archived_two = store.get_model_version(name, version_two.version)
    production_three = store.get_model_version(name, version_three.version)
    assert archived_two.current_stage == "Archived"
    assert production_three.current_stage == "Production"
    assert archived_two.last_updated_timestamp == production_three.last_updated_timestamp


def test_deleted_model_version_is_redacted_and_unavailable(
    store: MongoDBModelRegistryStore,
):
    name = "deleted-model-version-contract"
    source = "s3://models/deleted/1"
    run_id = "run-to-redact"
    run_link = "https://mlflow.example/runs/run-to-redact"
    store.create_registered_model(name)
    version = store.create_model_version(
        name,
        source=source,
        run_id=run_id,
        run_link=run_link,
        description="Description to redact",
    )
    store.set_registered_model_alias(name, "candidate", version.version)

    assert store.get_model_version_download_uri(name, version.version) == source
    store.transition_model_version_stage(name, version.version, "Production", False)
    store.update_model_version(name, version.version, "Updated description")
    assert store.get_model_version_download_uri(name, version.version) == source

    registered_model = store._registered_model_repository.find_by_name(name)
    assert registered_model is not None
    store.delete_model_version(name, version.version)

    raw_version = store._database[store._settings.model_versions_collection_name].find_one(
        {
            "registered_model_id": registered_model.model_id,
            "version": version.version,
        }
    )
    assert raw_version is not None
    assert raw_version["current_stage"] == STAGE_DELETED_INTERNAL
    assert raw_version["description"] is None
    assert raw_version["source"] == "REDACTED-SOURCE-PATH"
    assert raw_version["run_id"] == "REDACTED-RUN-ID"
    assert raw_version["run_link"] == "REDACTED-RUN-LINK"
    assert store.get_registered_model(name).aliases == {}

    operations = (
        lambda: store.get_model_version(name, version.version),
        lambda: store.update_model_version(name, version.version, "deleted"),
        lambda: store.delete_model_version(name, version.version),
        lambda: store.get_model_version_download_uri(name, version.version),
    )
    for operation in operations:
        with pytest.raises(MlflowException, match="Model Version.*not found") as missing_error:
            operation()
        assert missing_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_model_version_tags_replace_delete_and_remain_isolated(
    store: MongoDBModelRegistryStore,
):
    first_name = "model-version-tags-first"
    second_name = "model-version-tags-second"
    initial_tags = [
        ModelVersionTag("key", "value"),
        ModelVersionTag("anotherKey", "another value"),
    ]
    store.create_registered_model(first_name)
    store.create_registered_model(second_name)
    first_version = store.create_model_version(
        first_name,
        "s3://models/tags/first/1",
        tags=initial_tags,
    )
    other_first_version = store.create_model_version(
        first_name,
        "s3://models/tags/first/2",
        tags=initial_tags,
    )
    second_version = store.create_model_version(
        second_name,
        "s3://models/tags/second/1",
        tags=initial_tags,
    )

    store.set_model_version_tag(
        first_name,
        first_version.version,
        ModelVersionTag("new", "new value"),
    )
    store.set_model_version_tag(
        first_name,
        first_version.version,
        ModelVersionTag("key", "overridden"),
    )
    assert store.get_model_version(first_name, first_version.version).tags == {
        "key": "overridden",
        "anotherKey": "another value",
        "new": "new value",
    }
    assert store.get_model_version(first_name, other_first_version.version).tags == {
        "key": "value",
        "anotherKey": "another value",
    }
    assert store.get_model_version(second_name, second_version.version).tags == {
        "key": "value",
        "anotherKey": "another value",
    }

    store.delete_model_version_tag(first_name, first_version.version, "new")
    store.delete_model_version_tag(first_name, first_version.version, "key")
    store.delete_model_version_tag(first_name, first_version.version, "key")
    assert store.get_model_version(first_name, first_version.version).tags == {
        "anotherKey": "another value"
    }

    store.set_model_version_tag(
        first_name,
        first_version.version,
        ModelVersionTag("longTagKey", "a" * 4_999),
    )
    with pytest.raises(
        MlflowException,
        match="'value' exceeds the maximum length",
    ) as long_tag_error:
        store.set_model_version_tag(
            first_name,
            first_version.version,
            ModelVersionTag("longTagKey", "a" * 100_001),
        )
    assert long_tag_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    store.delete_model_version(second_name, second_version.version)
    for operation in (
        lambda: store.set_model_version_tag(
            second_name,
            second_version.version,
            ModelVersionTag("key", "value"),
        ),
        lambda: store.delete_model_version_tag(second_name, second_version.version, "key"),
    ):
        with pytest.raises(MlflowException, match="Model Version.*not found") as missing_error:
            operation()
        assert missing_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    with pytest.raises(
        MlflowException,
        match="Missing value for required parameter 'key'",
    ) as invalid_tag_error:
        store.set_model_version_tag(
            first_name,
            first_version.version,
            ModelVersionTag(key=None, value=""),
        )
    assert invalid_tag_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    invalid_operations = (
        (
            lambda: store.set_model_version_tag(
                None,
                first_version.version,
                ModelVersionTag("key", "value"),
            ),
            "Missing value for required parameter 'name'",
        ),
        (
            lambda: store.set_model_version_tag(
                first_name,
                "not-a-version",
                ModelVersionTag("key", "value"),
            ),
            "Parameter 'version' must be an integer",
        ),
        (
            lambda: store.delete_model_version_tag(
                first_name,
                first_version.version,
                None,
            ),
            "Missing value for required parameter 'key'",
        ),
        (
            lambda: store.delete_model_version_tag(None, first_version.version, "key"),
            "Missing value for required parameter 'name'",
        ),
        (
            lambda: store.delete_model_version_tag(first_name, "not-a-version", "key"),
            "Parameter 'version' must be an integer",
        ),
    )
    for operation, expected_message in invalid_operations:
        with pytest.raises(
            MlflowException,
            match=expected_message,
        ) as invalid_parameter_error:
            operation()
        assert invalid_parameter_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("copy_to_same_model", [False, True])
def test_copy_model_version_contract(
    store: MongoDBModelRegistryStore,
    copy_to_same_model,
):
    source_name = "copy-model-version-source"
    store.create_registered_model(source_name)
    source_version = store.create_model_version(
        source_name,
        source="s3://models/copy/source",
        run_id="copy-run",
        tags=[
            ModelVersionTag("key", "value"),
            ModelVersionTag("anotherKey", "another value"),
        ],
        run_link="https://mlflow.example/runs/copy-run",
        description="Copied description",
    )
    store.transition_model_version_stage(source_name, source_version.version, "Production", False)

    destination_name = source_name if copy_to_same_model else "copy-model-version-destination"
    copied = store.copy_model_version(source_version, destination_name)

    assert copied.name == destination_name
    assert copied.version == (2 if copy_to_same_model else 1)
    assert copied.current_stage == "None"
    assert copied.description == source_version.description
    assert copied.source == f"models:/{source_name}/{source_version.version}"
    assert copied.run_link == source_version.run_link
    assert copied.run_id == source_version.run_id
    assert copied.status == "READY"
    assert copied.status_message is None
    assert copied.tags == source_version.tags
    assert (
        store.get_model_version_download_uri(destination_name, copied.version)
        == source_version.source
    )

    double_copy = store.copy_model_version(copied, "copy-model-version-second-destination")
    assert double_copy.source == f"models:/{copied.name}/{copied.version}"
    assert (
        store.get_model_version_download_uri(double_copy.name, double_copy.version)
        == source_version.source
    )
