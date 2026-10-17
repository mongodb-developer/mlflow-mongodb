"""Run persistence contracts against MongoDB."""

from unittest.mock import patch
from uuid import UUID

import pytest
from mlflow.entities import (
    LifecycleStage,
    RunStatus,
    RunTag,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

from mlflow_mongodb.infrastructure.errors import RepositoryNotFoundError


def _stored_run(store, run_id):
    return store._database[store._settings.runs_collection_name].find_one({"_id": run_id})


def test_created_runs_have_unique_ids_and_correct_metadata(store):
    experiment_id = store.create_experiment("run-create", artifact_location="s3://artifacts/exp")
    runs = [
        store.create_run(int(experiment_id), "test-user", 1000, [RunTag("team", "risk")], "named")
        for _ in range(2)
    ]

    assert runs[0].info.run_id != runs[1].info.run_id
    for run in runs:
        assert UUID(hex=run.info.run_id).hex == run.info.run_id
        assert run.info.experiment_id == experiment_id
        assert run.info.artifact_uri == f"s3://artifacts/exp/{run.info.run_id}/artifacts"
        assert run.info.user_id == "test-user"
        assert run.info.start_time == 1000
        assert run.info.status == "RUNNING"
        assert run.info.lifecycle_stage == LifecycleStage.ACTIVE
        assert run.data.tags == {"team": "risk", MLFLOW_RUN_NAME: "named"}
        assert store.get_run(run.info.run_id).info.run_name == "named"


@pytest.mark.parametrize(
    ("argument_name", "tag_name"),
    [("argument", None), (None, "tag"), ("same", "same"), (None, None)],
)
def test_run_name_resolution_is_persisted_as_name_and_tag(store, argument_name, tag_name):
    experiment_id = store.create_experiment("run-naming")
    tags = [] if tag_name is None else [RunTag(MLFLOW_RUN_NAME, tag_name)]

    with patch(
        "mlflow_mongodb.tracking.store._generate_random_name", return_value="generated-run"
    ) as generate_name:
        run = store.create_run(experiment_id, "test-user", 1000, tags, argument_name)
    stored = store.get_run(run.info.run_id)

    assert stored.info.run_name == (argument_name or tag_name or "generated-run")
    assert stored.info.run_name == stored.data.tags[MLFLOW_RUN_NAME]
    if argument_name or tag_name:
        generate_name.assert_not_called()
    else:
        generate_name.assert_called_once_with()


def test_conflicting_run_names_do_not_insert_a_run(store):
    experiment_id = store.create_experiment("run-name-conflict")

    with pytest.raises(MlflowException, match="different values") as caught:
        store.create_run(experiment_id, "test-user", 1000, [RunTag(MLFLOW_RUN_NAME, "tag")], "arg")

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert store._database[store._settings.runs_collection_name].count_documents({}) == 0


@pytest.mark.parametrize("missing", [False, True])
def test_creation_requires_an_existing_active_experiment(store, missing):
    if missing:
        experiment_id = "404"
    else:
        experiment_id = store.create_experiment("run-deleted-parent")
        store.delete_experiment(experiment_id)

    with pytest.raises(MlflowException) as caught:
        store.create_run(experiment_id, "test-user", 1000, [], "named")

    code = RESOURCE_DOES_NOT_EXIST if missing else INVALID_PARAMETER_VALUE
    assert caught.value.error_code == ErrorCode.Name(code)
    assert store._database[store._settings.runs_collection_name].count_documents({}) == 0


def test_update_run_info_persists_status_end_time_and_literal_name(store, run):
    info = store.update_run_info(run.info.run_id, RunStatus.FINISHED, 2000, "$renamed")
    loaded = store.get_run(run.info.run_id)

    assert (info.status, info.end_time, info.run_name) == ("FINISHED", 2000, "$renamed")
    assert loaded.info.status == "FINISHED"
    assert loaded.info.end_time == 2000
    assert loaded.data.tags[MLFLOW_RUN_NAME] == "$renamed"


@pytest.mark.parametrize("missing", [False, True])
def test_repository_update_info_rejects_unmatched_runs_without_changes(store, run, missing):
    run_id = "f" * 32 if missing else run.info.run_id
    if not missing:
        store.delete_run(run_id)
    before = _stored_run(store, run_id)
    original_run = _stored_run(store, run.info.run_id)

    with pytest.raises(RepositoryNotFoundError) as caught:
        store._run_repository.update_info(
            run_id=run_id, status="FINISHED", end_time=2000, run_name="changed"
        )

    assert caught.value.args == (run_id,)
    assert _stored_run(store, run_id) == before
    assert _stored_run(store, run.info.run_id) == original_run
