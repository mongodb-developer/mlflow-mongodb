"""Run persistence contracts against MongoDB."""

import math
import sys
from unittest.mock import patch
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


def test_latest_metric_uses_step_then_timestamp_then_value_and_keeps_history(store, run):
    run_id = run.info.run_id
    events = [
        Metric("loss", 0.4, 1000, 1),
        Metric("loss", 0.9, 900, 2),
        Metric("loss", 0.8, 1001, 1),
        Metric("loss", 0.6, 950, 2),
        Metric("loss", 0.7, 950, 2),
        Metric("loss", 0.5, 950, 2),
    ]
    store.log_batch(run_id, events[:2], [], [])
    store.log_batch(run_id, events[2:5], [], [])
    store.log_batch(run_id, events[5:], [], [])

    assert store.get_run(run_id).data.metrics == {"loss": 0.7}
    history = store.get_metric_history(run_id, "loss")
    assert [(metric.timestamp, metric.step, metric.value) for metric in history] == sorted(
        (metric.timestamp, metric.step, metric.value) for metric in events
    )
    assert len(history) == len(events)


def test_metric_history_pagination_is_scoped_to_run_points(store, run):
    run_id = run.info.run_id
    model_id = "m-" + "a" * 32
    events = [
        Metric(
            "loss",
            index / 10,
            1000 + index,
            index,
            model_id=model_id,
            dataset_name="training",
            dataset_digest="abc",
        )
        for index in range(5)
    ]
    other = store.create_run(run.info.experiment_id, "test-user", 1000, [], "other")
    store.log_batch(run_id, [*events, Metric("accuracy", 0.9, 1000, 0)], [], [])
    store.log_metric(other.info.run_id, Metric("loss", 99, 1000, 0))

    first = store.get_metric_history(run_id, "loss", max_results=2)
    second = store.get_metric_history(run_id, "loss", max_results=2, page_token=first.token)
    third = store.get_metric_history(run_id, "loss", max_results=2, page_token=second.token)
    history = list(first) + list(second) + list(third)

    assert [metric.value for metric in history] == [metric.value for metric in events]
    assert first.token and second.token
    assert third.token is None
    assert all(metric.run_id == run_id and metric.model_id is None for metric in history)
    assert all(metric.dataset_name is None and metric.dataset_digest is None for metric in history)
    assert store.get_metric_history(run_id, "absent") == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("inf"), sys.float_info.max),
        (float("-inf"), -sys.float_info.max),
        (0.4, 0.4),
        (-2.0, -2.0),
        (0.0, 0.0),
        (sys.float_info.max, sys.float_info.max),
        (-sys.float_info.max, -sys.float_info.max),
    ],
)
def test_metric_value_sanitization_applies_to_history_and_latest(store, run, value, expected):
    store.log_batch(
        run.info.run_id,
        [Metric("value", value, 1000, 0)],
        [],
        [],
    )

    history = store.get_metric_history(run.info.run_id, "value")

    assert len(history) == 1
    assert history[0].value == pytest.approx(expected)
    assert store.get_run(run.info.run_id).data.metrics["value"] == pytest.approx(expected)


def test_nan_is_preserved_in_metric_history_and_latest(store, run):
    store.log_batch(run.info.run_id, [Metric("value", float("nan"), 1000, 0)], [], [])

    history = store.get_metric_history(run.info.run_id, "value")

    assert len(history) == 1
    assert math.isnan(history[0].value)
    assert math.isnan(store.get_run(run.info.run_id).data.metrics["value"])


def test_nan_ranks_as_zero_for_latest_metric_value_ties(store, run):
    run_id = run.info.run_id
    store.log_batch(
        run_id, [Metric("loss", -1.0, 1000, 1), Metric("loss", float("nan"), 1000, 1)], [], []
    )

    assert math.isnan(store.get_run(run_id).data.metrics["loss"])

    store.log_metric(run_id, Metric("loss", 0.1, 1000, 1))
    assert store.get_run(run_id).data.metrics["loss"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("key", "old_value", "new_value"),
    [("depth", "3", "4"), ("label", "$3", "$4"), ("a.b", "yes", "no")],
)
def test_parameters_are_immutable_and_conflict_precedes_metric_write(
    store, run, key, old_value, new_value
):
    run_id = run.info.run_id
    store.log_param(run_id, Param(key, old_value))
    store.log_param(run_id, Param(key, old_value))
    before = _stored_run(store, run_id)

    with pytest.raises(MlflowException, match="Changing param values is not allowed") as caught:
        store.log_batch(
            run_id,
            [Metric("loss", 0.4, 1000, 0)],
            [Param("new_param", "value"), Param(key, new_value)],
            [RunTag(MLFLOW_RUN_NAME, "$changed")],
        )

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    assert _stored_run(store, run_id) == before
    assert store.get_metric_history(run_id, "loss") == []


def test_empty_batch_preserves_active_run(store, run):
    before = _stored_run(store, run.info.run_id)

    store.log_batch(run.info.run_id, [], [], [])

    assert _stored_run(store, run.info.run_id) == before
    assert store.get_metric_history(run.info.run_id, "loss") == []


@pytest.mark.parametrize(
    ("metrics", "params", "tags"),
    [
        ([], [], []),
        ([Metric("loss", 0.4, 1000, 0)], [Param("depth", "3")], [RunTag("team", "risk")]),
    ],
)
def test_batch_with_missing_run_does_not_insert_data(store, run, metrics, params, tags):
    missing_id = "f" * 32
    before = _stored_run(store, run.info.run_id)

    with pytest.raises(MlflowException, match="not found") as caught:
        store.log_batch(missing_id, metrics, params, tags)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert _stored_run(store, missing_id) is None
    assert _stored_run(store, run.info.run_id) == before
    assert store.get_metric_history(run.info.run_id, "loss") == []
    assert (
        store._database[store._settings.run_metrics_collection_name].count_documents(
            {"run_id": missing_id}
        )
        == 0
    )


def test_batch_tags_keep_last_value_and_synchronize_run_name(store, run):
    store.log_batch(
        run.info.run_id,
        [],
        [],
        [RunTag("team", "old"), RunTag("team", "$risk"), RunTag(MLFLOW_RUN_NAME, "$renamed")],
    )
    store.set_tag(run.info.run_id, RunTag("purpose", "evaluation"))

    loaded = store.get_run(run.info.run_id)

    assert loaded.data.tags == {
        "team": "$risk",
        "purpose": "evaluation",
        MLFLOW_RUN_NAME: "$renamed",
    }
    assert loaded.info.run_name == "$renamed"
    assert len(_stored_run(store, run.info.run_id)["tags"]) == 3


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


def test_delete_restore_preserve_data_and_metric_history(store, run, clock):
    run_id = run.info.run_id
    store.log_batch(run_id, [Metric("loss", 0.4, 1000, 0)], [Param("depth", "3")], [])

    store.delete_run(run_id)

    assert store.get_run(run_id).info.lifecycle_stage == LifecycleStage.DELETED
    assert _stored_run(store, run_id)["deleted_time"] == clock["now"]
    assert store.get_metric_history(run_id, "loss")[0].value == pytest.approx(0.4)
    store.restore_run(run_id)
    restored = store.get_run(run_id)
    assert restored.info.lifecycle_stage == LifecycleStage.ACTIVE
    assert restored.data.metrics == {"loss": 0.4}
    assert _stored_run(store, run_id)["deleted_time"] is None
    assert _stored_run(store, run_id)["params"] == [{"k": "depth", "v": "3"}]


@pytest.mark.parametrize(
    "operation",
    [
        "get_run",
        "delete_run",
        "restore_run",
        "update_run_info",
        "log_batch",
        "log_inputs",
        "log_outputs",
    ],
)
def test_missing_run_errors_have_public_code(store, operation):
    args = {
        "get_run": (),
        "delete_run": (),
        "restore_run": (),
        "update_run_info": (None, None, None),
        "log_batch": ([], [], []),
        "log_inputs": (),
        "log_outputs": ([],),
    }[operation]

    with pytest.raises(MlflowException, match="not found") as caught:
        getattr(store, operation)("f" * 32, *args)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


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


@pytest.mark.parametrize("operation", ["update_run_info", "log_inputs", "log_outputs"])
def test_inactive_run_mutations_are_rejected_without_changes(store, run, operation):
    run_id = run.info.run_id
    store.delete_run(run_id)
    before = _stored_run(store, run_id)
    args = {
        "update_run_info": (RunStatus.FINISHED, 2000, "changed"),
        "log_inputs": (),
        "log_outputs": ([LoggedModelOutput("m-model", 1)],),
    }[operation]

    with pytest.raises(MlflowException, match="active.*Current state is deleted") as caught:
        getattr(store, operation)(run_id, *args)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert _stored_run(store, run_id) == before


def test_dataset_and_model_inputs_preserve_first_identity_and_tags(store, run):
    first = DatasetInput(
        Dataset("training", "abc", "test", "{}", "{}", "{}"), [InputTag("context", "first")]
    )
    duplicate = DatasetInput(
        Dataset("training", "abc", "test", "{}"), [InputTag("context", "second")]
    )
    other_digest = DatasetInput(Dataset("training", "def", "test", "{}"), [])
    store.log_inputs(run.info.run_id, [first, duplicate], [LoggedModelInput("m-first")])
    store.log_inputs(
        run.info.run_id,
        [duplicate, other_digest],
        [LoggedModelInput("m-first"), LoggedModelInput("m-second")],
    )

    inputs = store.get_run(run.info.run_id).inputs

    assert [(item.dataset.name, item.dataset.digest) for item in inputs.dataset_inputs] == [
        ("training", "abc"),
        ("training", "def"),
    ]
    assert inputs.dataset_inputs[0].tags[0].value == "first"
    assert inputs.dataset_inputs[0].dataset.schema == "{}"
    assert inputs.dataset_inputs[0].dataset.profile == "{}"
    assert [item.model_id for item in inputs.model_inputs] == ["m-first", "m-second"]


def test_model_outputs_preserve_submission_order_and_duplicates(store, run):
    store.log_outputs(run.info.run_id, [LoggedModelOutput("m-model", 1)])
    store.log_outputs(
        run.info.run_id, [LoggedModelOutput("m-model", 1), LoggedModelOutput("m-model", 2)]
    )

    outputs = store.get_run(run.info.run_id).outputs.model_outputs

    assert [(model.model_id, model.step) for model in outputs] == [
        ("m-model", 1),
        ("m-model", 1),
        ("m-model", 2),
    ]


def test_inherited_async_logging_persists_metrics_params_and_tags(store, run):
    run_id = run.info.run_id
    store.log_batch_async(run_id, [Metric("loss", 0.4, 1000, 0)], [Param("depth", "3")], []).wait()
    store.log_metric_async(run_id, Metric("loss", 0.3, 1001, 1)).wait()
    store.log_param_async(run_id, Param("width", "5")).wait()
    store.set_tag_async(run_id, RunTag("team", "async")).wait()

    loaded = store.get_run(run_id)

    assert loaded.data.metrics == {"loss": 0.3}
    assert loaded.data.tags["team"] == "async"
    assert {item["k"]: item["v"] for item in _stored_run(store, run_id)["params"]} == {
        "depth": "3",
        "width": "5",
    }
    assert [metric.value for metric in store.get_metric_history(run_id, "loss")] == [0.4, 0.3]


def test_async_parameter_conflict_is_reported_when_waiting(store, run):
    run_id = run.info.run_id
    store.log_param(run_id, Param("depth", "3"))

    operation = store.log_batch_async(
        run_id, [Metric("loss", 0.4, 1000, 0)], [Param("depth", "4")], []
    )
    with pytest.raises(MlflowException, match="Changing param values"):
        operation.wait()

    assert store.get_metric_history(run_id, "loss") == []
    assert _stored_run(store, run_id)["params"] == [{"k": "depth", "v": "3"}]


def test_get_run_returns_logged_parameters(store, run):
    store.log_param(run.info.run_id, Param("depth", "3"))

    assert store.get_run(run.info.run_id).data.params == {"depth": "3"}


@pytest.mark.parametrize("nonempty", [False, True])
def test_inactive_batch_logging_matches_mlflow_error_code(store, run, nonempty):
    store.delete_run(run.info.run_id)
    before = _stored_run(store, run.info.run_id)

    with pytest.raises(MlflowException) as caught:
        store.log_batch(
            run.info.run_id,
            [Metric("loss", 0.4, 1000, 1)] if nonempty else [],
            [Param("depth", "3")] if nonempty else [],
            [RunTag("team", "risk")] if nonempty else [],
        )

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert "Current state is deleted" in caught.value.message
    assert _stored_run(store, run.info.run_id) == before
    assert store.get_metric_history(run.info.run_id, "loss") == []


@pytest.mark.parametrize("operation", ["delete_run", "restore_run"])
def test_repeated_lifecycle_operations_succeed_like_sqlalchemy_store(store, run, clock, operation):
    run_id = run.info.run_id
    store.log_batch(run_id, [Metric("loss", 0.4, 1000, 0)], [Param("depth", "3")], [])
    original = _stored_run(store, run_id)
    if operation == "restore_run":
        store.delete_run(run_id)
    getattr(store, operation)(run_id)
    first = _stored_run(store, run_id)
    clock["now"] += 1000

    getattr(store, operation)(run_id)

    repeated = _stored_run(store, run_id)
    expected_stage = LifecycleStage.DELETED if operation == "delete_run" else LifecycleStage.ACTIVE
    assert first["lifecycle_stage"] == repeated["lifecycle_stage"] == expected_stage
    if operation == "delete_run":
        assert first["deleted_time"] == clock["now"] - 1000
        assert repeated["deleted_time"] == clock["now"]
    else:
        assert first["deleted_time"] is None
        assert repeated["deleted_time"] is None
    for field, value in original.items():
        if field not in ("lifecycle_stage", "deleted_time"):
            assert repeated[field] == value
    assert store.get_metric_history(run_id, "loss")[0].value == pytest.approx(0.4)
