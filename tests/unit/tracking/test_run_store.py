"""Run validation, entity conversion, and public error boundaries."""

from unittest.mock import MagicMock, Mock
from uuid import UUID

import pytest
from mlflow.entities import (
    LifecycleStage,
    RunStatus,
    RunTag,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

import mlflow_mongodb.tracking.store as tracking_store
from mlflow_mongodb.infrastructure.errors import (
    RepositoryPersistenceError,
)
from mlflow_mongodb.tracking.store import MongoDBTrackingStore

RUN_ID = UUID(int=42).hex
TIMESTAMP = 1_700_000_000_000


def _invoke_store(store, operation):
    calls = {
        "create_run": lambda: store.create_run("42", "test-user", TIMESTAMP, [], "example-run"),
        "get_run": lambda: store.get_run(RUN_ID),
        "delete_run": lambda: store.delete_run(RUN_ID),
        "restore_run": lambda: store.restore_run(RUN_ID),
        "update_run_info": lambda: store.update_run_info(RUN_ID, None, None, None),
        "log_batch": lambda: store.log_batch(RUN_ID, [], [], []),
        "log_inputs": lambda: store.log_inputs(RUN_ID),
        "log_outputs": lambda: store.log_outputs(RUN_ID, []),
        "get_metric_history": lambda: store.get_metric_history(RUN_ID, "loss"),
    }
    return calls[operation]()


@pytest.mark.parametrize(
    ("argument_name", "tag_name", "expected_name"),
    [
        ("explicit", None, "explicit"),
        (None, "tag-name", "tag-name"),
        ("same", "same", "same"),
        (None, None, "generated"),
    ],
)
def test_create_run_resolves_name_and_normalizes_experiment_id(
    run_store,
    run_repository,
    experiment_repository,
    run_record_factory,
    monkeypatch,
    argument_name,
    tag_name,
    expected_name,
):
    monkeypatch.setattr(tracking_store, "_generate_random_name", lambda: "generated")
    tags = [RunTag("team", "old"), RunTag("team", "risk")]
    if tag_name is not None:
        tags.append(RunTag(MLFLOW_RUN_NAME, tag_name))
    run_repository.create.return_value = run_record_factory(
        name=expected_name,
        tags=[{"k": "team", "v": "risk"}, {"k": MLFLOW_RUN_NAME, "v": expected_name}],
    )

    run = run_store.create_run(42, "test-user", TIMESTAMP, tags, argument_name)

    experiment_repository.find_by_id.assert_called_once_with("42")
    run_repository.create.assert_called_once_with(
        run_id=RUN_ID,
        experiment_id="42",
        name=expected_name,
        artifact_uri=f"s3://experiment-artifacts/42/{RUN_ID}/artifacts",
        user_id="test-user",
        status="RUNNING",
        start_time=TIMESTAMP,
        lifecycle_stage=LifecycleStage.ACTIVE,
        tags={"team": "risk", MLFLOW_RUN_NAME: expected_name},
    )
    assert run.info.run_id == RUN_ID
    assert run.info.run_name == expected_name
    assert run.data.tags == {"team": "risk", MLFLOW_RUN_NAME: expected_name}
    assert len(tags) == (3 if tag_name is not None else 2)


@pytest.mark.parametrize("missing", [False, True])
def test_create_run_checks_parent_before_creating_run(
    run_store, run_repository, experiment_repository, experiment_record_factory, missing
):
    experiment_repository.find_by_id.return_value = (
        None if missing else experiment_record_factory(lifecycle_stage=LifecycleStage.DELETED)
    )

    with pytest.raises(MlflowException) as caught:
        _invoke_store(run_store, "create_run")

    expected_code = RESOURCE_DOES_NOT_EXIST if missing else INVALID_PARAMETER_VALUE
    assert caught.value.error_code == ErrorCode.Name(expected_code)
    run_repository.create.assert_not_called()


def test_conflicting_run_names_are_rejected_before_write(run_store, run_repository):
    with pytest.raises(MlflowException, match="different values") as caught:
        run_store.create_run("42", "test-user", TIMESTAMP, [RunTag(MLFLOW_RUN_NAME, "tag")], "arg")

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    run_repository.create.assert_not_called()


def test_get_run_converts_params_metrics_inputs_outputs_and_tags(
    run_store, run_repository, run_record_factory
):
    run_repository.find_by_id.return_value = run_record_factory(
        params=[{"k": "depth", "v": "3"}, {"k": "label", "v": "$literal"}],
        metrics=[{"k": "loss", "v": 0.4, "timestamp": TIMESTAMP, "step": 2}],
        inputs={
            "datasets": [
                {
                    "dataset": {
                        "name": "training",
                        "digest": "abc",
                        "source_type": "test",
                        "source": "{}",
                        "schema": "{}",
                        "profile": "{}",
                    },
                    "tags": [{"k": "context", "v": "training"}],
                }
            ],
            "models": [{"model_id": "m-model"}],
        },
        outputs=[{"model_id": "m-model", "step": 2}],
    )

    run = run_store.get_run(RUN_ID)

    assert run.data.params == {"depth": "3", "label": "$literal"}
    assert run.data.metrics == {"loss": 0.4}
    assert run.data.tags == {"team": "risk"}
    assert run.inputs.dataset_inputs[0].dataset.name == "training"
    assert run.inputs.dataset_inputs[0].tags[0].value == "training"
    assert [model.model_id for model in run.inputs.model_inputs] == ["m-model"]
    assert [(model.model_id, model.step) for model in run.outputs.model_outputs] == [("m-model", 2)]


def test_update_run_info_checks_lifecycle_before_write(
    run_store, run_repository, run_record_factory
):
    run_repository.find_by_id.return_value = run_record_factory(
        lifecycle_stage=LifecycleStage.DELETED
    )

    with pytest.raises(MlflowException, match="active.*Current state is deleted") as caught:
        run_store.update_run_info(RUN_ID, RunStatus.FINISHED, TIMESTAMP + 100, "renamed")

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    run_repository.update_info.assert_not_called()


def test_update_run_info_converts_status_and_returned_record(
    run_store, run_repository, run_record_factory
):
    run_repository.update_info.return_value = run_record_factory(
        name="$renamed", status="FINISHED", end_time=TIMESTAMP + 100
    )

    info = run_store.update_run_info(RUN_ID, RunStatus.FINISHED, TIMESTAMP + 100, "$renamed")

    run_repository.update_info.assert_called_once_with(
        run_id=RUN_ID, status="FINISHED", end_time=TIMESTAMP + 100, run_name="$renamed"
    )
    assert (info.status, info.end_time, info.run_name) == ("FINISHED", TIMESTAMP + 100, "$renamed")


def test_create_run_translates_parent_read_failure(run_store, experiment_repository):
    experiment_repository.find_by_id.side_effect = RepositoryPersistenceError("Private parent read")

    with pytest.raises(MlflowException) as caught:
        _invoke_store(run_store, "create_run")

    assert caught.value.message == "Unable to create run."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


def test_run_repository_initialization_failure_is_logged_once(monkeypatch, caplog):
    error = RepositoryPersistenceError("Private initialization details")
    initialize = Mock(side_effect=error)
    monkeypatch.setattr(tracking_store, "RunRepository", initialize)
    store = MongoDBTrackingStore("mongodb://localhost:27017/mlflow")
    database = Mock()
    store.__dict__["_database"] = database

    with pytest.raises(MlflowException) as caught:
        store.get_run(RUN_ID)

    initialize.assert_called_once_with(database, settings=store._settings)
    assert caught.value.message == "Unable to initialize run repository."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    records = [record for record in caplog.records if record.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].args == (error,)
    assert records[0].exc_info is None


def test_run_repository_is_lazy_cached_and_uses_collection_overrides(monkeypatch):
    monkeypatch.setenv("MLFLOW_MONGODB_RUNS_COLLECTION", "custom_runs")
    monkeypatch.setenv("MLFLOW_MONGODB_RUN_METRICS_COLLECTION", "custom_metrics")
    database = MagicMock()
    store = MongoDBTrackingStore("mongodb://localhost:27017/mlflow")
    store.__dict__["_database"] = database
    database.__getitem__.assert_not_called()

    repository = store._run_repository

    assert store._run_repository is repository
    assert [args.args[0] for args in database.__getitem__.call_args_list] == [
        "custom_runs",
        "custom_metrics",
    ]


@pytest.mark.parametrize("operation", ["get_run", "update_run_info"])
def test_missing_run_read_has_resource_not_found_code(run_store, run_repository, operation):
    run_repository.find_by_id.return_value = None

    with pytest.raises(MlflowException, match="not found") as caught:
        _invoke_store(run_store, operation)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    run_repository.update_info.assert_not_called()


def test_programming_errors_pass_through_the_store_decorator(run_store, run_repository):
    failure = ValueError("DTO conversion failure")
    run_repository.find_by_id.side_effect = failure

    with pytest.raises(ValueError) as caught:
        run_store.get_run(RUN_ID)

    assert caught.value is failure
