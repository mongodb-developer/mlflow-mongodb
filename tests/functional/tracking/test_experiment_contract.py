import pytest
from mlflow.entities import Experiment, ExperimentTag, LifecycleStage, ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.utils.validation import MAX_EXPERIMENT_NAME_LENGTH

from mlflow_mongodb.tracking.store import MongoDBTrackingStore


@pytest.mark.parametrize("artifact_location", [None, "s3://custom-artifacts/experiment"])
def test_create_and_get_experiment_preserves_metadata(
    store: MongoDBTrackingStore, clock, artifact_location
):
    experiment_id = store.create_experiment(
        "experiment-metadata",
        artifact_location=artifact_location,
        tags=[
            ExperimentTag("team", "platform"),
            ExperimentTag("purpose", "evaluation"),
            ExperimentTag("team", "risk"),
        ],
    )

    assert isinstance(experiment_id, str)
    assert experiment_id.isdecimal()
    assert experiment_id != "0"

    by_id = store.get_experiment(experiment_id)
    by_name = store.get_experiment_by_name("experiment-metadata")
    assert isinstance(by_id, Experiment)
    assert isinstance(by_name, Experiment)
    for experiment in (by_id, by_name):
        assert experiment.experiment_id == experiment_id
        assert experiment.name == "experiment-metadata"
        assert experiment.artifact_location == (
            artifact_location or f"s3://experiment-artifacts/{experiment_id}"
        )
        assert experiment.lifecycle_stage == LifecycleStage.ACTIVE
        assert experiment.creation_time == clock["now"]
        assert experiment.last_update_time == clock["now"]
        assert experiment.tags == {"team": "risk", "purpose": "evaluation"}

    document = store._database[store._settings.experiments_collection_name].find_one(
        {"_id": experiment_id}
    )
    assert document["tags"] == [
        {"k": "team", "v": "risk"},
        {"k": "purpose", "v": "evaluation"},
    ]


def test_generated_experiment_ids_are_unique(store: MongoDBTrackingStore):
    first_id = store.create_experiment("experiment-first")
    second_id = store.create_experiment("experiment-second")

    assert first_id != second_id
    assert store.get_experiment(first_id).name == "experiment-first"
    assert store.get_experiment(second_id).name == "experiment-second"


def test_get_missing_experiment_contract(store: MongoDBTrackingStore):
    with pytest.raises(MlflowException, match="No Experiment with id") as caught:
        store.get_experiment("missing")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert store.get_experiment_by_name("missing") is None


@pytest.mark.parametrize("deleted", [False, True])
def test_experiment_name_remains_unique_across_lifecycle_stages(
    store: MongoDBTrackingStore, deleted
):
    experiment_id = store.create_experiment("experiment-unique")
    if deleted:
        store.delete_experiment(experiment_id)

    with pytest.raises(MlflowException, match="already exists") as caught:
        store.create_experiment("experiment-unique")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)
    assert store.get_experiment_by_name("experiment-unique").experiment_id == experiment_id
    assert len(store.search_experiments(view_type=ViewType.ALL)) == 1


def test_delete_and_restore_preserve_experiment_metadata(store: MongoDBTrackingStore, clock):
    experiment_id = store.create_experiment(
        "experiment-lifecycle", tags=[ExperimentTag("team", "risk")]
    )
    created = store.get_experiment(experiment_id)
    clock["now"] += 1000

    assert store.delete_experiment(experiment_id) is None

    deleted = store.get_experiment(experiment_id)
    assert deleted.lifecycle_stage == LifecycleStage.DELETED
    assert deleted.creation_time == created.creation_time
    assert deleted.last_update_time == clock["now"]
    assert deleted.tags == created.tags
    assert deleted.artifact_location == created.artifact_location
    assert store.get_experiment_by_name(created.name).lifecycle_stage == LifecycleStage.DELETED
    assert store.search_experiments() == []
    deleted_experiments = store.search_experiments(ViewType.DELETED_ONLY)
    assert [experiment.experiment_id for experiment in deleted_experiments] == [experiment_id]

    with pytest.raises(MlflowException) as repeated_delete:
        store.delete_experiment(experiment_id)
    assert repeated_delete.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    clock["now"] += 1000
    assert store.restore_experiment(experiment_id) is None

    restored = store.get_experiment(experiment_id)
    assert restored.lifecycle_stage == LifecycleStage.ACTIVE
    assert restored.creation_time == created.creation_time
    assert restored.last_update_time == clock["now"]
    assert restored.tags == created.tags
    assert restored.artifact_location == created.artifact_location
    assert store.search_experiments(ViewType.DELETED_ONLY) == []
    assert [experiment.experiment_id for experiment in store.search_experiments()] == [
        experiment_id
    ]

    with pytest.raises(MlflowException) as repeated_restore:
        store.restore_experiment(experiment_id)
    assert repeated_restore.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_delete_missing_experiment(store: MongoDBTrackingStore):
    with pytest.raises(MlflowException) as caught:
        store.delete_experiment("missing")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_restore_missing_experiment(store: MongoDBTrackingStore):
    with pytest.raises(MlflowException) as caught:
        store.restore_experiment("missing")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_rename_experiment_preserves_identity_and_metadata(store: MongoDBTrackingStore, clock):
    experiment_id = store.create_experiment(
        "experiment-original", tags=[ExperimentTag("team", "risk")]
    )
    created = store.get_experiment(experiment_id)
    clock["now"] += 1000

    assert store.rename_experiment(experiment_id, "experiment-renamed") is None

    renamed = store.get_experiment(experiment_id)
    assert renamed.name == "experiment-renamed"
    assert renamed.creation_time == created.creation_time
    assert renamed.last_update_time == clock["now"]
    assert renamed.tags == created.tags
    assert renamed.artifact_location == created.artifact_location
    assert store.get_experiment_by_name("experiment-original") is None
    assert store.get_experiment_by_name("experiment-renamed").experiment_id == experiment_id

    store.rename_experiment(experiment_id, "experiment-renamed")
    assert store.get_experiment(experiment_id).name == "experiment-renamed"


@pytest.mark.parametrize("target_deleted", [False, True])
def test_rename_experiment_rejects_duplicate_name(store: MongoDBTrackingStore, target_deleted):
    original_id = store.create_experiment("experiment-original")
    target_id = store.create_experiment("experiment-target")
    if target_deleted:
        store.delete_experiment(target_id)

    with pytest.raises(MlflowException, match="already exists") as caught:
        store.rename_experiment(original_id, "experiment-target")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)
    assert store.get_experiment(original_id).name == "experiment-original"
    assert store.get_experiment(target_id).name == "experiment-target"


def test_rename_deleted_experiment_is_rejected(store: MongoDBTrackingStore):
    experiment_id = store.create_experiment("experiment-deleted")
    store.delete_experiment(experiment_id)

    with pytest.raises(MlflowException, match="non-active experiment") as caught:
        store.rename_experiment(experiment_id, "experiment-renamed")

    assert caught.value.error_code == ErrorCode.Name(INVALID_STATE)
    assert store.get_experiment(experiment_id).name == "experiment-deleted"


def test_rename_missing_experiment(store: MongoDBTrackingStore):
    with pytest.raises(MlflowException) as caught:
        store.rename_experiment("missing", "experiment-renamed")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.parametrize("name", [None, "", "x" * (MAX_EXPERIMENT_NAME_LENGTH + 1)])
def test_invalid_experiment_name_is_rejected(store: MongoDBTrackingStore, name):
    with pytest.raises(MlflowException) as caught:
        store.create_experiment(name)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert store.search_experiments(view_type=ViewType.ALL) == []
