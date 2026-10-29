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


def test_restore_run_restores_individually_deleted_run(store, run, clock):
    run_id = run.info.run_id
    original = _stored_run(store, run_id)
    experiment = store.get_experiment(run.info.experiment_id)
    store.delete_run(run_id)
    deleted = _stored_run(store, run_id)
    assert deleted["lifecycle_stage"] == LifecycleStage.DELETED
    assert deleted["deleted_time"] == clock["now"]
    clock["now"] += 1000

    assert store.restore_run(run_id) is None

    assert store.get_run(run_id).info.lifecycle_stage == LifecycleStage.ACTIVE
    assert _stored_run(store, run_id) == {
        **original,
        "deleted_time": None,
    }
    parent = store.get_experiment(run.info.experiment_id)
    assert parent.lifecycle_stage == LifecycleStage.ACTIVE
    assert parent.last_update_time == experiment.last_update_time


def test_restore_run_keeps_parent_experiment_deleted(store, run, clock):
    run_id = run.info.run_id
    original = _stored_run(store, run_id)
    store.delete_experiment(run.info.experiment_id)
    experiment = store.get_experiment(run.info.experiment_id)
    assert experiment.lifecycle_stage == LifecycleStage.DELETED
    deleted = _stored_run(store, run_id)
    assert deleted["lifecycle_stage"] == LifecycleStage.DELETED
    assert deleted["deleted_time"] == clock["now"]
    clock["now"] += 1000

    assert store.restore_run(run_id) is None

    assert store.get_run(run_id).info.lifecycle_stage == LifecycleStage.ACTIVE
    assert _stored_run(store, run_id) == {
        **original,
        "deleted_time": None,
    }
    parent = store.get_experiment(run.info.experiment_id)
    assert parent.lifecycle_stage == LifecycleStage.DELETED
    assert parent.last_update_time == experiment.last_update_time


def test_experiment_delete_restore_cascades_to_all_its_runs(store, clock):
    experiment_id = store.create_experiment("cascade-parent")
    other_experiment_id = store.create_experiment("unrelated-parent")
    runs = [
        store.create_run(experiment_id, "test-user", 1000, [RunTag("team", "risk")], name)
        for name in ("active-run", "previously-deleted-run")
    ]
    other_runs = [
        store.create_run(other_experiment_id, "test-user", 1000, [], name)
        for name in ("unrelated-active-run", "unrelated-deleted-run")
    ]
    store.delete_run(runs[1].info.run_id)
    store.delete_run(other_runs[1].info.run_id)
    original_documents = [_stored_run(store, run.info.run_id) for run in runs]
    other_documents = [_stored_run(store, run.info.run_id) for run in other_runs]
    clock["now"] += 1000

    assert store.delete_experiment(experiment_id) is None

    deleted_experiment = store.get_experiment(experiment_id)
    assert deleted_experiment.lifecycle_stage == LifecycleStage.DELETED
    assert deleted_experiment.last_update_time == clock["now"]
    for run, original in zip(runs, original_documents, strict=True):
        assert store.get_run(run.info.run_id).info.lifecycle_stage == LifecycleStage.DELETED
        assert _stored_run(store, run.info.run_id) == {
            **original,
            "lifecycle_stage": LifecycleStage.DELETED,
            "deleted_time": clock["now"],
        }
    assert [_stored_run(store, run.info.run_id) for run in other_runs] == other_documents
    assert store.get_experiment(other_experiment_id).lifecycle_stage == LifecycleStage.ACTIVE
    clock["now"] += 1000

    assert store.restore_experiment(experiment_id) is None

    restored_experiment = store.get_experiment(experiment_id)
    assert restored_experiment.lifecycle_stage == LifecycleStage.ACTIVE
    assert restored_experiment.last_update_time == clock["now"]
    for run, original in zip(runs, original_documents, strict=True):
        assert store.get_run(run.info.run_id).info.lifecycle_stage == LifecycleStage.ACTIVE
        assert _stored_run(store, run.info.run_id) == {
            **original,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "deleted_time": None,
        }
    assert [_stored_run(store, run.info.run_id) for run in other_runs] == other_documents
    assert store.get_experiment(other_experiment_id).lifecycle_stage == LifecycleStage.ACTIVE


def test_update_run_info_rejects_deleted_run_without_changes(store, run):
    run_id = run.info.run_id
    store.delete_run(run_id)
    before = _stored_run(store, run_id)

    with pytest.raises(MlflowException, match="active.*Current state is deleted") as caught:
        store.update_run_info(run_id, RunStatus.FINISHED, 2000, "$changed")

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert _stored_run(store, run_id) == before


def test_update_run_info_rejects_missing_run_without_inserting(store, run):
    run_id = "f" * 32
    original_run = _stored_run(store, run.info.run_id)

    with pytest.raises(MlflowException, match="not found") as caught:
        store.update_run_info(run_id, RunStatus.FINISHED, 2000, "$changed")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert _stored_run(store, run_id) is None
    assert _stored_run(store, run.info.run_id) == original_run
