"""Run validation, entity conversion, and public error boundaries."""

from unittest.mock import MagicMock, Mock
from uuid import UUID

import pytest
from mlflow.entities import (
    Dataset,
    DatasetInput,
    InputTag,
    LifecycleStage,
    LoggedModelInput,
    LoggedModelOutput,
    Metric,
    Param,
    RunStatus,
    RunTag,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from mlflow.utils.search_utils import SearchUtils

import mlflow_mongodb.tracking.store as tracking_store
from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotActiveError,
    RepositoryNotFoundError,
    RepositoryParamConflictError,
    RepositoryPersistenceError,
)
from mlflow_mongodb.tracking.store import MongoDBTrackingStore
from mlflow_mongodb.tracking.types import RunMetricRecord

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

    experiment_repository.find_by_id.assert_called_once_with(42)
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


def test_update_run_info_translates_inactive_run_without_initial_read(run_store, run_repository):
    run_repository.update_info.side_effect = RepositoryNotActiveError(
        RUN_ID, LifecycleStage.DELETED
    )

    with pytest.raises(MlflowException, match="active.*Current state is deleted") as caught:
        run_store.update_run_info(RUN_ID, RunStatus.FINISHED, TIMESTAMP + 100, "renamed")

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    run_repository.find_by_id.assert_not_called()
    run_repository.update_info.assert_called_once_with(
        run_id=RUN_ID, status="FINISHED", end_time=TIMESTAMP + 100, run_name="renamed"
    )


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
    run_repository.find_by_id.assert_not_called()


def test_batch_encodes_metric_identity_and_key_value_entries(run_store, run_repository):
    metric = Metric(
        "loss", 0.4, TIMESTAMP, 2, model_id="m-model", dataset_name="training", dataset_digest="abc"
    )

    run_store.log_batch(RUN_ID, [metric], [Param("depth", "3")], [RunTag("team", "$risk")])

    run_repository.log_batch.assert_called_once_with(
        run_id=RUN_ID,
        metrics=[
            {
                "k": "loss",
                "v": 0.4,
                "timestamp": TIMESTAMP,
                "step": 2,
                "model_id": "m-model",
                "dataset_name": "training",
                "dataset_digest": "abc",
            }
        ],
        params=[{"k": "depth", "v": "3"}],
        tags=[{"k": "team", "v": "$risk"}],
    )


def test_duplicate_param_keys_are_rejected_before_batch_persistence(run_store, run_repository):
    with pytest.raises(MlflowException) as caught:
        run_store.log_batch(RUN_ID, [], [Param("depth", "3"), Param("depth", "4")], [])

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    run_repository.log_batch.assert_not_called()


def test_inputs_outputs_are_encoded_without_losing_dataset_metadata(run_store, run_repository):
    dataset = Dataset("training", "abc", "test", "{}", "{}", "{}")

    run_store.log_inputs(
        RUN_ID,
        [DatasetInput(dataset, [InputTag("context", "training")])],
        [LoggedModelInput("m-input")],
    )
    run_store.log_outputs(RUN_ID, [LoggedModelOutput("m-output", 2)])

    run_repository.log_inputs.assert_called_once_with(
        run_id=RUN_ID,
        datasets=[
            {"dataset": dataset.to_dictionary(), "tags": [{"k": "context", "v": "training"}]}
        ],
        models=[{"model_id": "m-input"}],
    )
    run_repository.log_outputs.assert_called_once_with(
        run_id=RUN_ID, models=[{"model_id": "m-output", "step": 2}]
    )


def test_invalid_input_container_remains_a_type_error(run_store, run_repository):
    with pytest.raises(TypeError, match="should be a list"):
        run_store.log_inputs(RUN_ID, datasets="invalid")

    run_repository.log_inputs.assert_not_called()


@pytest.mark.parametrize(
    "operation", ["log_batch", "log_inputs", "log_outputs", "get_metric_history"]
)
def test_invalid_run_id_is_rejected_before_repository_access(run_store, run_repository, operation):
    args = {
        "log_batch": ([], [], []),
        "log_inputs": (),
        "log_outputs": ([],),
        "get_metric_history": ("loss",),
    }[operation]

    with pytest.raises(MlflowException) as caught:
        getattr(run_store, operation)("invalid id!", *args)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    getattr(run_repository, operation).assert_not_called()


@pytest.mark.parametrize("max_results", [0, -1, True, 1.5, "2"])
def test_invalid_metric_page_size_is_rejected(run_store, run_repository, max_results):
    with pytest.raises(MlflowException, match="positive integer") as caught:
        run_store.get_metric_history(RUN_ID, "loss", max_results=max_results)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    run_repository.get_metric_history.assert_not_called()


def test_negative_metric_offset_is_rejected(run_store, run_repository):
    with pytest.raises(MlflowException, match="offset must not be negative") as caught:
        run_store.get_metric_history(RUN_ID, "loss", page_token=SearchUtils.create_page_token(-1))

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    run_repository.get_metric_history.assert_not_called()


def test_metric_history_pagination_reads_extra_record_and_preserves_identity(
    run_store, run_repository
):
    run_repository.get_metric_history.return_value = [
        RunMetricRecord("loss", value, TIMESTAMP + index, index, "m-model", "training", "abc")
        for index, value in enumerate((0.5, 0.4, 0.3))
    ]

    page = run_store.get_metric_history(
        RUN_ID, "loss", max_results=2, page_token=SearchUtils.create_page_token(3)
    )

    run_repository.get_metric_history.assert_called_once_with(
        run_id=RUN_ID, metric_key="loss", offset=3, limit=3
    )
    assert [metric.value for metric in page] == [0.5, 0.4]
    assert page.token == SearchUtils.create_page_token(5)
    assert all(metric.run_id == RUN_ID and metric.dataset_digest == "abc" for metric in page)


@pytest.mark.parametrize(
    ("operation", "repository_method", "expected_message"),
    [
        ("create_run", "create", "Unable to create run."),
        ("get_run", "find_by_id", "Unable to get run."),
        ("delete_run", "mark_deleted", "Unable to delete run."),
        ("restore_run", "restore", "Unable to restore run."),
        ("update_run_info", "update_info", "Unable to update run info."),
        ("log_batch", "log_batch", "Unable to log run batch."),
        ("log_inputs", "log_inputs", "Unable to log run inputs."),
        ("log_outputs", "log_outputs", "Unable to log run outputs."),
        ("get_metric_history", "get_metric_history", "Unable to read metric history."),
    ],
)
def test_run_operations_sanitize_and_log_persistence_failure_once(
    run_store, run_repository, caplog, operation, repository_method, expected_message
):
    error = RepositoryPersistenceError("Private database URI and query details")
    getattr(run_repository, repository_method).side_effect = error

    with pytest.raises(MlflowException) as caught:
        _invoke_store(run_store, operation)

    assert caught.value.message == expected_message
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    records = [record for record in caplog.records if record.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].args == (error,)
    assert records[0].exc_info is None


def test_create_run_translates_parent_read_failure(run_store, experiment_repository):
    experiment_repository.find_by_id.side_effect = RepositoryPersistenceError("Private parent read")

    with pytest.raises(MlflowException) as caught:
        _invoke_store(run_store, "create_run")

    assert caught.value.message == "Unable to create run."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


@pytest.mark.parametrize(
    ("operation", "repository_method", "error", "expected_code"),
    [
        ("create_run", "create", RepositoryAlreadyExistsError(RUN_ID), RESOURCE_ALREADY_EXISTS),
        ("delete_run", "mark_deleted", RepositoryNotFoundError(RUN_ID), RESOURCE_DOES_NOT_EXIST),
        ("restore_run", "restore", RepositoryNotFoundError(RUN_ID), RESOURCE_DOES_NOT_EXIST),
        (
            "update_run_info",
            "update_info",
            RepositoryNotFoundError(RUN_ID),
            RESOURCE_DOES_NOT_EXIST,
        ),
        (
            "log_batch",
            "log_batch",
            RepositoryParamConflictError("depth", "3", "4", RUN_ID),
            INVALID_PARAMETER_VALUE,
        ),
        ("log_batch", "log_batch", RepositoryNotFoundError(RUN_ID), RESOURCE_DOES_NOT_EXIST),
        (
            "log_batch",
            "log_batch",
            RepositoryNotActiveError(RUN_ID, LifecycleStage.DELETED),
            INVALID_PARAMETER_VALUE,
        ),
        ("log_inputs", "log_inputs", RepositoryNotFoundError(RUN_ID), RESOURCE_DOES_NOT_EXIST),
        ("log_outputs", "log_outputs", RepositoryNotFoundError(RUN_ID), RESOURCE_DOES_NOT_EXIST),
        (
            "log_inputs",
            "log_inputs",
            RepositoryNotActiveError(RUN_ID, LifecycleStage.DELETED),
            INVALID_PARAMETER_VALUE,
        ),
        (
            "log_outputs",
            "log_outputs",
            RepositoryNotActiveError(RUN_ID, LifecycleStage.DELETED),
            INVALID_PARAMETER_VALUE,
        ),
    ],
)
def test_run_domain_errors_keep_codes_and_suppressed_chaining(
    run_store, run_repository, operation, repository_method, error, expected_code
):
    getattr(run_repository, repository_method).side_effect = error

    with pytest.raises(MlflowException) as caught:
        _invoke_store(run_store, operation)

    assert caught.value.error_code == ErrorCode.Name(expected_code)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


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


def test_missing_run_read_has_resource_not_found_code(run_store, run_repository):
    run_repository.find_by_id.return_value = None

    with pytest.raises(MlflowException, match="not found") as caught:
        run_store.get_run(RUN_ID)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    run_repository.update_info.assert_not_called()


def test_update_run_info_translates_missing_run_without_initial_read(run_store, run_repository):
    run_repository.update_info.side_effect = RepositoryNotFoundError(RUN_ID)

    with pytest.raises(MlflowException, match="not found") as caught:
        run_store.update_run_info(RUN_ID, None, None, None)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    run_repository.find_by_id.assert_not_called()
    run_repository.update_info.assert_called_once_with(
        run_id=RUN_ID, status=None, end_time=None, run_name=None
    )


def test_programming_errors_pass_through_the_store_decorator(run_store, run_repository):
    failure = ValueError("DTO conversion failure")
    run_repository.find_by_id.side_effect = failure

    with pytest.raises(ValueError) as caught:
        run_store.get_run(RUN_ID)

    assert caught.value is failure


@pytest.mark.parametrize("operation", ["log_metric", "log_param", "set_tag"])
def test_inherited_single_value_logging_delegates_to_batch(run_store, run_repository, operation):
    entity = {
        "log_metric": Metric("loss", 0.4, TIMESTAMP, 1),
        "log_param": Param("depth", "3"),
        "set_tag": RunTag("team", "risk"),
    }[operation]

    getattr(run_store, operation)(RUN_ID, entity)

    run_repository.log_batch.assert_called_once()
    args = run_repository.log_batch.call_args.kwargs
    assert args["run_id"] == RUN_ID
    assert len(args["metrics"]) == (1 if operation == "log_metric" else 0)
    assert len(args["params"]) == (1 if operation == "log_param" else 0)
    assert len(args["tags"]) == (1 if operation == "set_tag" else 0)
