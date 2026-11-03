"""Functional model-version contracts shared with MLflow's reference stores."""

from dataclasses import replace

import pytest
from bson import ObjectId
from mlflow.entities.model_registry import ModelVersionTag
from mlflow.entities.model_registry.model_version_stages import (
    STAGE_ARCHIVED,
    STAGE_DELETED_INTERNAL,
    STAGE_NONE,
    STAGE_PRODUCTION,
    STAGE_STAGING,
)
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotFoundError,
)
from mlflow_mongodb.model_registry.repositories import ModelVersionOrder
from mlflow_mongodb.model_registry.store import MongoDBModelRegistryStore
from mlflow_mongodb.model_registry.types import (
    ModelVersionRecord,
    ModelVersionSearchResult,
    ModelVersionTagRecord,
)


@pytest.fixture
def version_create_args(store):
    store.create_registered_model("repository-create")
    parent = store._registered_model_repository.find_by_name("repository-create")
    return {
        "registered_model_id": parent.model_id,
        "version": 1,
        "creation_timestamp": 1_700_001_000_000,
        "description": "Model description",
        "current_stage": STAGE_NONE,
        "source": "s3://models/source",
        "storage_location": "s3://models/resolved",
        "run_id": "source-run",
        "run_link": "https://mlflow.example/runs/source-run",
        "status": ModelVersionStatus.to_string(ModelVersionStatus.READY),
        "tags": {"team.name": "$risk", "$literal": "value"},
        "model_id": "m-" + "a" * 32,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "description": None,
            "source": None,
            "storage_location": None,
            "run_id": None,
            "run_link": None,
            "tags": {},
            "model_id": None,
        },
    ],
    ids=["complete-metadata", "optional-metadata-absent"],
)
def test_repository_create_persists_metadata_and_returns_record(
    store, version_create_args, overrides
):
    arguments = {**version_create_args, **overrides}

    record = store._model_version_repository.create(**arguments)

    assert isinstance(record.model_version_id, ObjectId)
    assert record == ModelVersionRecord(
        **{
            **arguments,
            "tags": tuple(
                ModelVersionTagRecord(key, value) for key, value in arguments["tags"].items()
            ),
        },
        model_version_id=record.model_version_id,
        last_updated_timestamp=arguments["creation_timestamp"],
        user_id=None,
        status_message=None,
    )
    collection = store._database[store._settings.model_versions_collection_name]
    assert collection.find_one({"_id": record.model_version_id}) == {
        **arguments,
        "_id": record.model_version_id,
        "last_updated_timestamp": arguments["creation_timestamp"],
        "user_id": None,
        "status_message": None,
        "tags": [{"key": key, "value": value} for key, value in arguments["tags"].items()],
    }
    assert (
        store._model_version_repository.find_by_version(
            registered_model_id=record.registered_model_id, version=record.version
        )
        == record
    )


def test_repository_create_duplicate_version_preserves_existing_document(
    store, version_create_args
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})

    with pytest.raises(RepositoryAlreadyExistsError) as caught:
        repository.create(**{**version_create_args, "description": "Replacement"})

    assert caught.value.args == (
        f"{version_create_args['registered_model_id']}:{version_create_args['version']}",
    )
    assert collection.count_documents({}) == 1
    assert collection.find_one({"_id": original.model_version_id}) == before


@pytest.mark.parametrize("description", ["", None])
def test_repository_update_description_changes_only_description_and_timestamp(
    store, version_create_args, description
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    peer = repository.create(**{**version_create_args, "version": 2})
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})
    peer_before = collection.find_one({"_id": peer.model_version_id})
    timestamp = original.creation_timestamp + 1000

    updated = repository.update_description(
        registered_model_id=original.registered_model_id,
        version=original.version,
        description=description,
        last_updated_timestamp=timestamp,
    )

    assert updated == replace(original, description=description, last_updated_timestamp=timestamp)
    assert collection.find_one({"_id": original.model_version_id}) == {
        **before,
        "description": description,
        "last_updated_timestamp": timestamp,
    }
    assert collection.find_one({"_id": peer.model_version_id}) == peer_before


@pytest.mark.parametrize("target", [{"version": 2}, {"registered_model_id": ObjectId()}])
def test_repository_update_description_missing_target_leaves_existing_version_unchanged(
    store, version_create_args, target
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})
    identity = {
        "registered_model_id": original.registered_model_id,
        "version": original.version,
        **target,
    }

    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.update_description(
            **identity,
            description="Updated description",
            last_updated_timestamp=original.creation_timestamp + 1000,
        )

    assert caught.value.resource_id == f"{identity['registered_model_id']}:{identity['version']}"
    assert collection.count_documents({}) == 1
    assert collection.find_one({"_id": original.model_version_id}) == before


@pytest.mark.parametrize("target", [{"version": 2}, {"registered_model_id": ObjectId()}])
def test_repository_transition_stage_missing_target_leaves_existing_version_unchanged(
    store, version_create_args, target
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})
    identity = {
        "registered_model_id": original.registered_model_id,
        "version": original.version,
        **target,
    }

    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.transition_stage(
            **identity,
            stage=STAGE_STAGING,
            last_updated_timestamp=original.creation_timestamp + 1000,
        )

    assert caught.value.resource_id == f"{identity['registered_model_id']}:{identity['version']}"
    assert collection.count_documents({}) == 1
    assert collection.find_one({"_id": original.model_version_id}) == before


@pytest.mark.parametrize("stage", [STAGE_STAGING, STAGE_PRODUCTION])
def test_repository_archive_other_versions_changes_only_matching_peers(
    store, version_create_args, stage
):
    repository = store._model_version_repository
    collection = store._database[store._settings.model_versions_collection_name]
    selected = repository.create(**{**version_create_args, "current_stage": stage})
    peers = [
        repository.create(**{**version_create_args, "version": version, "current_stage": stage})
        for version in (2, 3)
    ]
    untouched = [
        repository.create(
            **{**version_create_args, "version": version, "current_stage": other_stage}
        )
        for version, other_stage in (
            (4, STAGE_NONE),
            (5, STAGE_ARCHIVED),
            (6, STAGE_DELETED_INTERNAL),
        )
    ]
    store.create_registered_model("other-archive-parent")
    other_parent = store._registered_model_repository.find_by_name("other-archive-parent")
    untouched.append(
        repository.create(
            **{
                **version_create_args,
                "registered_model_id": other_parent.model_id,
                "current_stage": stage,
            }
        )
    )
    unchanged = {
        record.model_version_id: collection.find_one({"_id": record.model_version_id})
        for record in [selected, *untouched]
    }
    peer_documents = [collection.find_one({"_id": peer.model_version_id}) for peer in peers]
    timestamp = selected.creation_timestamp + 1000

    repository.archive_other_versions_in_stage(
        registered_model_id=selected.registered_model_id,
        version=selected.version,
        stage=stage,
        last_updated_timestamp=timestamp,
    )

    for before in peer_documents:
        assert collection.find_one({"_id": before["_id"]}) == {
            **before,
            "current_stage": STAGE_ARCHIVED,
            "last_updated_timestamp": timestamp,
        }
    for model_version_id, before in unchanged.items():
        assert collection.find_one({"_id": model_version_id}) == before


def test_repository_touch_all_updates_every_stage_without_changing_other_models(
    store, version_create_args
):
    repository = store._model_version_repository
    collection = store._database[store._settings.model_versions_collection_name]
    versions = [
        repository.create(**{**version_create_args, "version": version, "current_stage": stage})
        for version, stage in enumerate(
            (STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED, STAGE_DELETED_INTERNAL),
            start=1,
        )
    ]
    store.create_registered_model("other-touch-parent")
    other_parent = store._registered_model_repository.find_by_name("other-touch-parent")
    other = repository.create(
        **{**version_create_args, "registered_model_id": other_parent.model_id}
    )
    other_before = collection.find_one({"_id": other.model_version_id})
    before = [collection.find_one({"_id": record.model_version_id}) for record in versions]
    timestamp = version_create_args["creation_timestamp"] + 1000

    repository.touch_all_for_registered_model(
        registered_model_id=version_create_args["registered_model_id"],
        last_updated_timestamp=timestamp,
    )

    for document in before:
        assert collection.find_one({"_id": document["_id"]}) == {
            **document,
            "last_updated_timestamp": timestamp,
        }
    assert collection.find_one({"_id": other.model_version_id}) == other_before


def test_repository_touch_all_missing_parent_leaves_existing_versions_unchanged(
    store, version_create_args
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})

    repository.touch_all_for_registered_model(
        registered_model_id=ObjectId(),
        last_updated_timestamp=original.creation_timestamp + 1000,
    )

    assert collection.count_documents({}) == 1
    assert collection.find_one({"_id": original.model_version_id}) == before


@pytest.mark.parametrize("target", [{"version": 2}, {"registered_model_id": ObjectId()}])
def test_repository_soft_delete_missing_target_preserves_existing_version(
    store, version_create_args, target
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})
    identity = {
        "registered_model_id": original.registered_model_id,
        "version": original.version,
        **target,
    }

    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.soft_delete(
            **identity, last_updated_timestamp=original.creation_timestamp + 1000
        )

    assert caught.value.resource_id == f"{identity['registered_model_id']}:{identity['version']}"
    assert collection.find_one({"_id": original.model_version_id}) == before


def test_repository_soft_delete_repeated_call_preserves_first_deletion_timestamp(
    store, version_create_args
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    repository.soft_delete(
        registered_model_id=original.registered_model_id,
        version=original.version,
        last_updated_timestamp=original.creation_timestamp + 1000,
    )
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})

    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.soft_delete(
            registered_model_id=original.registered_model_id,
            version=original.version,
            last_updated_timestamp=original.creation_timestamp + 2000,
        )

    assert caught.value.resource_id == f"{original.registered_model_id}:{original.version}"
    assert collection.find_one({"_id": original.model_version_id}) == before


def test_repository_delete_all_removes_every_stage_and_preserves_other_models(
    store, version_create_args
):
    repository = store._model_version_repository
    collection = store._database[store._settings.model_versions_collection_name]
    stages = (STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED, STAGE_DELETED_INTERNAL)
    for version, stage in enumerate(stages, start=1):
        repository.create(**{**version_create_args, "version": version, "current_stage": stage})
    store.create_registered_model("other-delete-parent")
    other_parent = store._registered_model_repository.find_by_name("other-delete-parent")
    other = repository.create(
        **{**version_create_args, "registered_model_id": other_parent.model_id}
    )
    other_before = collection.find_one({"_id": other.model_version_id})
    parents = store._database[store._settings.registered_models_collection_name]
    parent_before = parents.find_one({"_id": version_create_args["registered_model_id"]})

    deleted_count = repository.delete_all_for_registered_model(
        registered_model_id=version_create_args["registered_model_id"]
    )

    assert deleted_count == len(stages)
    assert (
        collection.count_documents(
            {"registered_model_id": version_create_args["registered_model_id"]}
        )
        == 0
    )
    assert collection.count_documents({}) == 1
    assert collection.find_one({"_id": other.model_version_id}) == other_before
    assert parents.find_one({"_id": version_create_args["registered_model_id"]}) == parent_before
    assert (
        repository.delete_all_for_registered_model(
            registered_model_id=version_create_args["registered_model_id"]
        )
        == 0
    )


def test_repository_delete_all_missing_parent_returns_zero_without_changing_versions(
    store, version_create_args
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})

    deleted_count = repository.delete_all_for_registered_model(registered_model_id=ObjectId())

    assert deleted_count == 0
    assert collection.count_documents({}) == 1
    assert collection.find_one({"_id": original.model_version_id}) == before


@pytest.mark.parametrize(
    "initial_tags,key,value",
    [
        ({}, "$literal", "$value"),
        ({"team.name": "$old", "keep": "yes"}, "team.name", "$new"),
        ({"$literal": "old", "keep": "yes"}, "$literal", ""),
    ],
)
def test_repository_set_tag_adds_or_replaces_literal_entry_without_other_changes(
    store, version_create_args, initial_tags, key, value
):
    repository = store._model_version_repository
    original = repository.create(**{**version_create_args, "tags": initial_tags})
    peer = repository.create(**{**version_create_args, "version": 2})
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})
    peer_before = collection.find_one({"_id": peer.model_version_id})
    expected_tags = [tag for tag in before["tags"] if tag["key"] != key]
    expected_tags.append({"key": key, "value": value})

    updated = repository.set_tag(
        registered_model_id=original.registered_model_id,
        version=original.version,
        key=key,
        value=value,
    )

    assert updated == replace(
        original,
        tags=tuple(ModelVersionTagRecord(tag["key"], tag["value"]) for tag in expected_tags),
    )
    assert collection.find_one({"_id": original.model_version_id}) == {
        **before,
        "tags": expected_tags,
    }
    assert collection.find_one({"_id": peer.model_version_id}) == peer_before


@pytest.mark.parametrize("target", [{"version": 2}, {"registered_model_id": ObjectId()}])
def test_repository_set_tag_missing_target_preserves_existing_version(
    store, version_create_args, target
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})
    identity = {
        "registered_model_id": original.registered_model_id,
        "version": original.version,
        **target,
    }

    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.set_tag(**identity, key="new", value="value")

    assert caught.value.resource_id == f"{identity['registered_model_id']}:{identity['version']}"
    assert collection.count_documents({}) == 1
    assert collection.find_one({"_id": original.model_version_id}) == before


@pytest.mark.parametrize(
    "initial_tags,key",
    [
        ({}, "absent"),
        ({"team.name": "$risk", "keep": "yes"}, "team.name"),
        ({"$literal": "value", "keep": "yes"}, "$literal"),
    ],
)
def test_repository_delete_tag_preserves_other_entries_and_metadata(
    store, version_create_args, initial_tags, key
):
    repository = store._model_version_repository
    original = repository.create(**{**version_create_args, "tags": initial_tags})
    peer = repository.create(**{**version_create_args, "version": 2})
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})
    peer_before = collection.find_one({"_id": peer.model_version_id})
    expected_tags = [tag for tag in before["tags"] if tag["key"] != key]

    updated = repository.delete_tag(
        registered_model_id=original.registered_model_id,
        version=original.version,
        key=key,
    )

    assert updated == replace(
        original,
        tags=tuple(ModelVersionTagRecord(tag["key"], tag["value"]) for tag in expected_tags),
    )
    assert collection.find_one({"_id": original.model_version_id}) == {
        **before,
        "tags": expected_tags,
    }
    assert collection.find_one({"_id": peer.model_version_id}) == peer_before


@pytest.mark.parametrize("target", [{"version": 2}, {"registered_model_id": ObjectId()}])
def test_repository_delete_tag_missing_target_preserves_existing_version(
    store, version_create_args, target
):
    repository = store._model_version_repository
    original = repository.create(**version_create_args)
    collection = store._database[store._settings.model_versions_collection_name]
    before = collection.find_one({"_id": original.model_version_id})
    identity = {
        "registered_model_id": original.registered_model_id,
        "version": original.version,
        **target,
    }

    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.delete_tag(**identity, key="team.name")

    assert caught.value.resource_id == f"{identity['registered_model_id']}:{identity['version']}"
    assert collection.count_documents({}) == 1
    assert collection.find_one({"_id": original.model_version_id}) == before


@pytest.mark.parametrize(
    "name,version,stage,expected",
    [
        ("repository-create", 1, STAGE_NONE, True),
        ("repository-create", 1, STAGE_DELETED_INTERNAL, False),
        ("repository-create", 2, STAGE_NONE, False),
        ("other-exists-parent", 1, STAGE_NONE, False),
        ("other-exists-parent", 2, STAGE_NONE, True),
        ("empty-exists-parent", 1, STAGE_NONE, False),
        ("missing-parent", 1, STAGE_NONE, False),
        ("Repository-create", 1, STAGE_NONE, False),
    ],
)
def test_repository_exists_for_registered_model_scopes_name_and_version(
    store, version_create_args, name, version, stage, expected
):
    repository = store._model_version_repository
    repository.create(**{**version_create_args, "current_stage": stage})
    store.create_registered_model("other-exists-parent")
    other_parent = store._registered_model_repository.find_by_name("other-exists-parent")
    repository.create(
        **{**version_create_args, "registered_model_id": other_parent.model_id, "version": 2}
    )
    store.create_registered_model("empty-exists-parent")

    exists = repository.exists_for_registered_model(registered_model_name=name, version=version)

    assert exists is expected


@pytest.mark.parametrize(
    "stages,expected_versions",
    [
        ([STAGE_PRODUCTION, STAGE_STAGING, STAGE_ARCHIVED, STAGE_NONE], (9, 8, 7, 3)),
        ([STAGE_STAGING, STAGE_STAGING, STAGE_PRODUCTION], (9, 7)),
        ([], ()),
    ],
)
def test_repository_find_latest_by_stages_selects_highest_version_and_sorts_descending(
    store, version_create_args, stages, expected_versions
):
    repository = store._model_version_repository
    versions = {
        version: repository.create(
            **{
                **version_create_args,
                "version": version,
                "current_stage": stage,
                "creation_timestamp": version_create_args["creation_timestamp"] + 1000 - version,
            }
        )
        for version, stage in (
            (9, STAGE_STAGING),
            (1, STAGE_STAGING),
            (7, STAGE_PRODUCTION),
            (2, STAGE_PRODUCTION),
            (3, STAGE_NONE),
            (8, STAGE_ARCHIVED),
            (4, STAGE_ARCHIVED),
            (10, STAGE_DELETED_INTERNAL),
        )
    }
    store.create_registered_model("other-latest-parent")
    other_parent = store._registered_model_repository.find_by_name("other-latest-parent")
    repository.create(
        **{
            **version_create_args,
            "registered_model_id": other_parent.model_id,
            "version": 99,
            "current_stage": STAGE_STAGING,
        }
    )

    latest = repository.find_latest_by_stages(
        registered_model_id=version_create_args["registered_model_id"], stages=stages
    )

    assert latest == tuple(versions[version] for version in expected_versions)


def test_repository_find_latest_by_stages_missing_parent_returns_empty_tuple(
    store, version_create_args
):
    repository = store._model_version_repository
    repository.create(**version_create_args)

    latest = repository.find_latest_by_stages(registered_model_id=ObjectId(), stages=[STAGE_NONE])

    assert latest == ()


@pytest.mark.parametrize(
    "offset,expected_versions,has_more",
    [(0, (2,), True), (1, (1,), False), (2, (), False)],
)
def test_repository_search_excludes_orphans_and_deleted_versions_before_pagination(
    store, version_create_args, offset, expected_versions, has_more
):
    repository = store._model_version_repository
    parent = store._registered_model_repository.find_by_name("repository-create")
    versions = {
        version: repository.create(**{**version_create_args, "version": version})
        for version in (1, 2)
    }
    repository.create(
        **{**version_create_args, "version": 3, "current_stage": STAGE_DELETED_INTERNAL}
    )
    repository.create(**{**version_create_args, "registered_model_id": ObjectId(), "version": 99})

    page = repository.search(
        filters=(),
        order_by=(ModelVersionOrder("version_number", False),),
        exclude_prompts=False,
        offset=offset,
        max_results=1,
    )

    assert page.records == tuple(
        ModelVersionSearchResult(versions[version], parent) for version in expected_versions
    )
    assert page.has_more is has_more


def test_model_version_create_update_and_stage_validation_contract(
    store: MongoDBModelRegistryStore,
    monkeypatch,
):
    name = "model-version-metadata-contract"
    clock = {"now": 1_700_001_000_000}
    monkeypatch.setattr(
        "mlflow_mongodb.model_registry.store.get_current_time_millis",
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

    parent = store._registered_model_repository.find_by_name(name)
    collection = store._database[store._settings.model_versions_collection_name]
    identity = {"registered_model_id": parent.model_id, "version": version.version}
    before_transition = collection.find_one(identity)
    transitioned = store.transition_model_version_stage(
        name,
        version.version,
        "sTaGiNg",
        archive_existing_versions=False,
    )
    assert transitioned.current_stage == "Staging"
    assert store.get_model_version(name, version.version).description == "Reviewed version"
    assert collection.find_one(identity) == {
        **before_transition,
        "current_stage": STAGE_STAGING,
        "last_updated_timestamp": clock["now"],
    }

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
        "mlflow_mongodb.model_registry.store.get_current_time_millis",
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
    collection = store._database[store._settings.model_versions_collection_name]
    identity = {"registered_model_id": registered_model.model_id, "version": version.version}
    collection.update_one(
        identity, {"$set": {"user_id": "private-user", "status_message": "private-status"}}
    )
    before_deletion = collection.find_one(identity)
    store.delete_model_version(name, version.version)

    raw_version = collection.find_one(identity)
    assert raw_version is not None
    assert raw_version == {
        **before_deletion,
        "current_stage": STAGE_DELETED_INTERNAL,
        "last_updated_timestamp": raw_version["last_updated_timestamp"],
        "description": None,
        "user_id": None,
        "source": "REDACTED-SOURCE-PATH",
        "run_id": "REDACTED-RUN-ID",
        "run_link": "REDACTED-RUN-LINK",
        "status_message": None,
    }
    assert raw_version["last_updated_timestamp"] >= before_deletion["last_updated_timestamp"]
    assert store.get_registered_model(name).aliases == {}

    operations = (
        lambda: store.get_model_version(name, version.version),
        lambda: store.update_model_version(name, version.version, "deleted"),
        lambda: store.transition_model_version_stage(
            name, version.version, "Staging", archive_existing_versions=False
        ),
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
