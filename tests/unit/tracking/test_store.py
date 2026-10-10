import pytest
from mlflow.entities import Experiment, ExperimentTag, LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.utils.validation import (
    MAX_EXPERIMENT_NAME_LENGTH,
    MAX_EXPERIMENT_TAG_KEY_LENGTH,
    MAX_EXPERIMENT_TAG_VAL_LENGTH,
)

from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotActiveError,
    RepositoryNotFoundError,
)

FIXED_TIMESTAMP = 1_700_000_000_000
EXPERIMENT_ID = "42"


@pytest.mark.parametrize("artifact_location", [None, "s3://custom-artifacts/experiment"])
def test_create_experiment_passes_metadata_and_deduplicated_tags(
    store, experiment_repository, artifact_location
):
    experiment_repository.create.return_value = EXPERIMENT_ID

    experiment_id = store.create_experiment(
        "example-experiment",
        artifact_location=artifact_location,
        tags=[
            ExperimentTag("team", "platform"),
            ExperimentTag("purpose", "evaluation"),
            ExperimentTag("team", "risk"),
        ],
    )

    assert experiment_id == EXPERIMENT_ID
    experiment_repository.create.assert_called_once_with(
        experiment_id=EXPERIMENT_ID,
        name="example-experiment",
        artifact_location=artifact_location or f"s3://experiment-artifacts/{EXPERIMENT_ID}",
        lifecycle_stage=LifecycleStage.ACTIVE,
        creation_timestamp=FIXED_TIMESTAMP,
        tags={"team": "risk", "purpose": "evaluation"},
    )


def test_create_experiment_without_tags(store, experiment_repository):
    experiment_repository.create.return_value = EXPERIMENT_ID

    assert store.create_experiment("example-experiment") == EXPERIMENT_ID

    experiment_repository.create.assert_called_once_with(
        experiment_id=EXPERIMENT_ID,
        name="example-experiment",
        artifact_location=f"s3://experiment-artifacts/{EXPERIMENT_ID}",
        lifecycle_stage=LifecycleStage.ACTIVE,
        creation_timestamp=FIXED_TIMESTAMP,
        tags={},
    )


@pytest.mark.parametrize("name", [None, "", 123, "x" * (MAX_EXPERIMENT_NAME_LENGTH + 1)])
def test_create_experiment_validates_name_before_persistence(store, experiment_repository, name):
    with pytest.raises(MlflowException) as caught:
        store.create_experiment(name)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.create.assert_not_called()


@pytest.mark.parametrize(
    "tag",
    [
        ExperimentTag("x" * (MAX_EXPERIMENT_TAG_KEY_LENGTH + 1), "value"),
        ExperimentTag("key", "x" * (MAX_EXPERIMENT_TAG_VAL_LENGTH + 1)),
    ],
)
def test_create_experiment_validates_tags_before_persistence(store, experiment_repository, tag):
    with pytest.raises(MlflowException) as caught:
        store.create_experiment("example-experiment", tags=[tag])

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.create.assert_not_called()


def test_create_experiment_rejects_runs_artifact_uri_before_persistence(
    store, experiment_repository
):
    with pytest.raises(MlflowException, match="Artifact location cannot be a runs:/ URI") as caught:
        store.create_experiment("example-experiment", artifact_location="runs:/run-id/artifacts")

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.create.assert_not_called()


def test_create_experiment_validates_artifact_length_before_persistence(
    store, experiment_repository, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ARTIFACT_LOCATION_MAX_LENGTH", "30")

    with pytest.raises(MlflowException, match="Invalid artifact path length") as caught:
        store.create_experiment("example-experiment", artifact_location="s3://bucket/" + "x" * 30)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.create.assert_not_called()


def test_create_experiment_translates_duplicate_name(store, experiment_repository):
    experiment_repository.create.side_effect = RepositoryAlreadyExistsError("Private duplicate")

    with pytest.raises(MlflowException, match="already exists") as caught:
        store.create_experiment("example-experiment")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)
    assert "Private duplicate" not in caught.value.message
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    experiment_repository.create.assert_called_once()


@pytest.mark.parametrize("experiment_id", [EXPERIMENT_ID, 42])
@pytest.mark.parametrize("lifecycle_stage", [LifecycleStage.ACTIVE, LifecycleStage.DELETED])
def test_get_experiment_converts_record(
    store, experiment_repository, experiment_record_factory, experiment_id, lifecycle_stage
):
    record = experiment_record_factory(
        lifecycle_stage=lifecycle_stage, last_update_time=FIXED_TIMESTAMP + 1, tags={"team": "risk"}
    )
    experiment_repository.find_by_id.return_value = record

    experiment = store.get_experiment(experiment_id)

    assert isinstance(experiment, Experiment)
    assert experiment.experiment_id == record.experiment_id
    assert experiment.name == record.name
    assert experiment.artifact_location == record.artifact_location
    assert experiment.lifecycle_stage == lifecycle_stage
    assert experiment.creation_time == record.creation_time
    assert experiment.last_update_time == record.last_update_time
    assert experiment.tags == {"team": "risk"}
    experiment_repository.find_by_id.assert_called_once_with(EXPERIMENT_ID)


@pytest.mark.parametrize("experiment_id", ["missing", None])
def test_get_experiment_rejects_missing_id(store, experiment_repository, experiment_id):
    experiment_repository.find_by_id.return_value = None

    with pytest.raises(MlflowException, match="No Experiment with id") as caught:
        store.get_experiment(experiment_id)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    experiment_repository.find_by_id.assert_called_once_with(experiment_id)


@pytest.mark.parametrize("lifecycle_stage", [LifecycleStage.ACTIVE, LifecycleStage.DELETED])
def test_get_experiment_by_name_converts_record(
    store, experiment_repository, experiment_record_factory, lifecycle_stage
):
    record = experiment_record_factory(
        lifecycle_stage=lifecycle_stage, last_update_time=FIXED_TIMESTAMP + 1, tags={"team": "risk"}
    )
    experiment_repository.find_by_name.return_value = record

    experiment = store.get_experiment_by_name(record.name)

    assert isinstance(experiment, Experiment)
    assert experiment.experiment_id == record.experiment_id
    assert experiment.name == record.name
    assert experiment.artifact_location == record.artifact_location
    assert experiment.lifecycle_stage == lifecycle_stage
    assert experiment.creation_time == record.creation_time
    assert experiment.last_update_time == record.last_update_time
    assert experiment.tags == {"team": "risk"}
    experiment_repository.find_by_name.assert_called_once_with(record.name)


def test_get_experiment_by_name_returns_none_when_missing(store, experiment_repository):
    experiment_repository.find_by_name.return_value = None

    assert store.get_experiment_by_name("missing") is None

    experiment_repository.find_by_name.assert_called_once_with("missing")


def test_delete_experiment_passes_id_and_timestamp(store, experiment_repository):
    assert store.delete_experiment(EXPERIMENT_ID) is None

    experiment_repository.mark_deleted.assert_called_once_with(
        experiment_id=EXPERIMENT_ID, last_update_time=FIXED_TIMESTAMP
    )


def test_delete_experiment_translates_missing_or_inactive_experiment(store, experiment_repository):
    experiment_repository.mark_deleted.side_effect = RepositoryNotFoundError("Private lifecycle")

    with pytest.raises(MlflowException, match="No Experiment with id=42 exists") as caught:
        store.delete_experiment(EXPERIMENT_ID)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    experiment_repository.mark_deleted.assert_called_once_with(
        experiment_id=EXPERIMENT_ID, last_update_time=FIXED_TIMESTAMP
    )


def test_restore_experiment_passes_id_and_timestamp(store, experiment_repository):
    assert store.restore_experiment(EXPERIMENT_ID) is None

    experiment_repository.restore.assert_called_once_with(
        experiment_id=EXPERIMENT_ID, last_update_time=FIXED_TIMESTAMP
    )


def test_restore_experiment_translates_missing_or_active_experiment(store, experiment_repository):
    experiment_repository.restore.side_effect = RepositoryNotFoundError("Private lifecycle")

    with pytest.raises(MlflowException, match="No Experiment with id=42 exists") as caught:
        store.restore_experiment(EXPERIMENT_ID)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    experiment_repository.restore.assert_called_once_with(
        experiment_id=EXPERIMENT_ID, last_update_time=FIXED_TIMESTAMP
    )


@pytest.mark.parametrize("experiment_id", [EXPERIMENT_ID, 42])
def test_rename_experiment_passes_normalized_id_and_timestamp(
    store, experiment_repository, experiment_id
):
    assert store.rename_experiment(experiment_id, "renamed-experiment") is None

    experiment_repository.rename.assert_called_once_with(
        experiment_id=EXPERIMENT_ID,
        new_name="renamed-experiment",
        last_update_time=FIXED_TIMESTAMP,
    )


@pytest.mark.parametrize("name", [None, "", 123, "x" * (MAX_EXPERIMENT_NAME_LENGTH + 1)])
def test_rename_experiment_validates_name_before_persistence(store, experiment_repository, name):
    with pytest.raises(MlflowException) as caught:
        store.rename_experiment(EXPERIMENT_ID, name)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.rename.assert_not_called()


@pytest.mark.parametrize(
    ("repository_error", "expected_code", "message"),
    [
        (RepositoryNotFoundError("Private lookup"), RESOURCE_DOES_NOT_EXIST, "No Experiment"),
        (RepositoryNotActiveError("Private lifecycle"), INVALID_STATE, "non-active experiment"),
        (
            RepositoryAlreadyExistsError("Private duplicate"),
            RESOURCE_ALREADY_EXISTS,
            "already exists",
        ),
    ],
)
def test_rename_experiment_translates_domain_errors(
    store, experiment_repository, repository_error, expected_code, message
):
    experiment_repository.rename.side_effect = repository_error

    with pytest.raises(MlflowException, match=message) as caught:
        store.rename_experiment(EXPERIMENT_ID, "renamed-experiment")

    assert caught.value.error_code == ErrorCode.Name(expected_code)
    assert "Private" not in caught.value.message
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    experiment_repository.rename.assert_called_once_with(
        experiment_id=EXPERIMENT_ID,
        new_name="renamed-experiment",
        last_update_time=FIXED_TIMESTAMP,
    )
