"""Logged-model validation, entity conversion, and public errors."""

from unittest.mock import MagicMock, Mock
from uuid import UUID

import pytest
from mlflow.entities import LifecycleStage
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

import mlflow_mongodb.tracking.store as tracking_store
from mlflow_mongodb.infrastructure.errors import (
    RepositoryNotFoundError,
    RepositoryPersistenceError,
    RepositoryTagNotFoundError,
)
from mlflow_mongodb.tracking.types import RunMetricRecord

MODEL_ID = f"m-{UUID(int=42).hex}"
TIMESTAMP = 1_700_000_000_000


def test_finalize_translates_missing_model(logged_model_store, logged_model_repository):
    logged_model_repository.finalize.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException) as caught:
        logged_model_store.finalize_logged_model(MODEL_ID, LoggedModelStatus.READY)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert "private" not in caught.value.message
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    logged_model_repository.get_metric_history.assert_not_called()


def test_finalize_hides_persistence_details(logged_model_store, logged_model_repository, caplog):
    failure = RepositoryPersistenceError("private")
    logged_model_repository.finalize.side_effect = failure
    with pytest.raises(MlflowException) as caught:
        logged_model_store.finalize_logged_model(MODEL_ID, LoggedModelStatus.READY)
    assert caught.value.error_code == "INTERNAL_ERROR"
    assert caught.value.message == "Unable to finalize logged model."
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert caplog.records[-1].args == (failure,)
    logged_model_repository.get_metric_history.assert_not_called()


@pytest.mark.parametrize("value", [None, {}, "model"])
def test_record_rejects_wrong_model_type_before_reading_run(run_store, run_repository, value):
    with pytest.raises(TypeError, match="should be mlflow.models.Model"):
        run_store.record_logged_model("missing", value)
    run_repository.find_by_id.assert_not_called()
    run_repository.log_batch.assert_not_called()


def test_record_validates_history_tag_before_writing(run_store, run_repository, monkeypatch):
    validate = Mock(side_effect=MlflowException.invalid_parameter_value("Invalid tag value"))
    monkeypatch.setattr(tracking_store, "_validate_tag", validate)
    with pytest.raises(MlflowException) as caught:
        run_store.record_logged_model(UUID(int=42).hex, Model(artifact_path="model"))
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    validate.assert_called_once()
    run_repository.log_batch.assert_not_called()


def test_record_missing_run_prevents_write(run_store, run_repository):
    run_repository.find_by_id.return_value = None
    with pytest.raises(MlflowException) as caught:
        run_store.record_logged_model("missing", Model(artifact_path="model"))
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    run_repository.log_batch.assert_not_called()


def test_record_hides_write_failure(run_store, run_repository, caplog):
    failure = RepositoryPersistenceError("private")
    run_repository.log_batch.side_effect = failure
    with pytest.raises(MlflowException) as caught:
        run_store.record_logged_model(UUID(int=42).hex, Model(artifact_path="model"))
    assert caught.value.error_code == "INTERNAL_ERROR"
    assert "private" not in caught.value.message
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert caplog.records[-1].args == (failure,)


@pytest.mark.parametrize("name", ["named", None])
def test_create_builds_pending_model_and_artifact_path(
    logged_model_store, logged_model_repository, logged_model_record_factory, monkeypatch, name
):
    generate = Mock(return_value="generated")
    monkeypatch.setattr(tracking_store, "_generate_random_name", generate)
    expected_name = name or "generated"
    logged_model_repository.create.return_value = logged_model_record_factory(name=expected_name)

    model = logged_model_store.create_logged_model(
        "42",
        name=name,
        source_run_id="source-run",
        model_type="classifier",
        tags=[LoggedModelTag("team", "risk")],
        params=[LoggedModelParameter("depth", "3")],
    )

    logged_model_repository.create.assert_called_once_with(
        model_id=MODEL_ID,
        experiment_id="42",
        name=expected_name,
        artifact_location=f"s3://experiment-artifacts/42/models/{MODEL_ID}/artifacts",
        creation_timestamp=TIMESTAMP,
        status=LoggedModelStatus.PENDING.value,
        lifecycle_stage=LifecycleStage.ACTIVE,
        source_run_id="source-run",
        model_type="classifier",
        tags={"team": "risk"},
        params={"depth": "3"},
    )
    assert model.model_id == MODEL_ID
    assert model.name == expected_name
    assert model.status == LoggedModelStatus.PENDING
    if name:
        generate.assert_not_called()
    else:
        generate.assert_called_once_with()


@pytest.mark.parametrize("missing", [True, False])
def test_create_requires_existing_active_experiment(
    logged_model_store,
    logged_model_repository,
    experiment_repository,
    experiment_record_factory,
    missing,
):
    experiment_repository.find_by_id.return_value = (
        None if missing else experiment_record_factory(lifecycle_stage=LifecycleStage.DELETED)
    )
    with pytest.raises(MlflowException) as caught:
        logged_model_store.create_logged_model("42", name="model")
    code = RESOURCE_DOES_NOT_EXIST if missing else INVALID_PARAMETER_VALUE
    assert caught.value.error_code == ErrorCode.Name(code)
    logged_model_repository.create.assert_not_called()


def test_invalid_name_precedes_experiment_lookup(
    logged_model_store, logged_model_repository, experiment_repository
):
    with pytest.raises(MlflowException):
        logged_model_store.create_logged_model("42", name="invalid/name")
    experiment_repository.find_by_id.assert_not_called()
    logged_model_repository.create.assert_not_called()


@pytest.mark.parametrize("field", ["tags", "params"])
@pytest.mark.parametrize("case", ["duplicate", "null_key", "null_value"])
def test_create_rejects_invalid_entries_before_parent_lookup(
    logged_model_store, logged_model_repository, experiment_repository, field, case
):
    entity = LoggedModelTag if field == "tags" else LoggedModelParameter
    entries = {
        "duplicate": [entity("key", "one"), entity("key", "two")],
        "null_key": [entity(None, "value")],
        "null_value": [entity("key", None)],
    }[case]
    with pytest.raises(MlflowException) as caught:
        logged_model_store.create_logged_model("42", **{field: entries})
    assert caught.value.error_code == ErrorCode.Name(BAD_REQUEST)
    experiment_repository.find_by_id.assert_not_called()
    logged_model_repository.create.assert_not_called()


def test_get_converts_complete_metadata_and_metric_history(
    logged_model_store, logged_model_repository, logged_model_record_factory
):
    logged_model_repository.find_by_id.return_value = logged_model_record_factory(
        source_run_id="run",
        model_type="classifier",
        status="READY",
        status_message="done",
        last_updated_timestamp=TIMESTAMP + 1,
    )
    logged_model_repository.get_metric_history.return_value = (
        RunMetricRecord("loss", 0.4, 1000, 1, MODEL_ID, "eval", "abc", "run"),
        RunMetricRecord("loss", 0.3, 2000, 2, MODEL_ID, "train", "def", "other"),
    )

    model = logged_model_store.get_logged_model(MODEL_ID)

    assert model.experiment_id == "42"
    assert model.creation_timestamp == TIMESTAMP
    assert model.last_updated_timestamp == TIMESTAMP + 1
    assert model.source_run_id == "run"
    assert model.model_type == "classifier"
    assert model.status == LoggedModelStatus.READY
    assert model.status_message == "done"
    assert model.tags == {"team": "risk"}
    assert model.params == {"depth": "3"}
    assert [(m.value, m.run_id, m.dataset_name, m.dataset_digest) for m in model.metrics] == [
        (0.4, "run", "eval", "abc"),
        (0.3, "other", "train", "def"),
    ]
    assert all(m.model_id == MODEL_ID for m in model.metrics)


@pytest.mark.parametrize("deleted", [False, True])
def test_get_missing_or_deleted_model_skips_history(
    logged_model_store, logged_model_repository, logged_model_record_factory, deleted
):
    logged_model_repository.find_by_id.return_value = (
        logged_model_record_factory(lifecycle_stage=LifecycleStage.DELETED) if deleted else None
    )
    with pytest.raises(MlflowException) as caught:
        logged_model_store.get_logged_model(MODEL_ID)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    logged_model_repository.get_metric_history.assert_not_called()


def test_get_can_include_deleted_model(
    logged_model_store, logged_model_repository, logged_model_record_factory
):
    logged_model_repository.find_by_id.return_value = logged_model_record_factory(
        lifecycle_stage=LifecycleStage.DELETED
    )
    assert logged_model_store.get_logged_model(MODEL_ID, allow_deleted=True).model_id == MODEL_ID
    logged_model_repository.get_metric_history.assert_called_once_with(MODEL_ID)


def test_delete_passes_current_timestamp(logged_model_store, logged_model_repository):
    logged_model_store.delete_logged_model(MODEL_ID)
    logged_model_repository.mark_deleted.assert_called_once_with(
        model_id=MODEL_ID, last_updated_timestamp=TIMESTAMP
    )


def test_set_tags_keeps_last_value_without_mutating_input(
    logged_model_store, logged_model_repository
):
    tags = [LoggedModelTag("team", "old"), LoggedModelTag("team", "$new")]
    logged_model_store.set_logged_model_tags(MODEL_ID, tags)
    logged_model_repository.set_tags.assert_called_once_with(
        model_id=MODEL_ID, tags={"team": "$new"}
    )
    assert [tag.value for tag in tags] == ["old", "$new"]


@pytest.mark.parametrize("tag", [LoggedModelTag(None, "value"), LoggedModelTag("key", None)])
def test_set_tags_rejects_null_entries(logged_model_store, logged_model_repository, tag):
    with pytest.raises(MlflowException) as caught:
        logged_model_store.set_logged_model_tags(MODEL_ID, [tag])
    assert caught.value.error_code == ErrorCode.Name(BAD_REQUEST)
    logged_model_repository.set_tags.assert_not_called()


@pytest.mark.parametrize(
    "operation,args,method,error",
    [
        ("delete_logged_model", (MODEL_ID,), "mark_deleted", RepositoryNotFoundError("private")),
        ("set_logged_model_tags", (MODEL_ID, []), "set_tags", RepositoryNotFoundError("private")),
        (
            "delete_logged_model_tag",
            (MODEL_ID, "team"),
            "delete_tag",
            RepositoryNotFoundError("private"),
        ),
        (
            "delete_logged_model_tag",
            (MODEL_ID, "team"),
            "delete_tag",
            RepositoryTagNotFoundError("private"),
        ),
    ],
)
def test_domain_errors_keep_public_code_and_suppress_backend_context(
    logged_model_store, logged_model_repository, caplog, operation, args, method, error
):
    getattr(logged_model_repository, method).side_effect = error
    with pytest.raises(MlflowException) as caught:
        getattr(logged_model_store, operation)(*args)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert "private" not in caught.value.message
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


@pytest.mark.parametrize(
    "operation,args,method,message",
    [
        (
            "create_logged_model",
            ("42", "example-model"),
            "create",
            "Unable to create logged model.",
        ),
        ("get_logged_model", (MODEL_ID,), "find_by_id", "Unable to get logged model."),
        ("get_logged_model", (MODEL_ID,), "get_metric_history", "Unable to get logged model."),
        ("delete_logged_model", (MODEL_ID,), "mark_deleted", "Unable to delete logged model."),
        ("set_logged_model_tags", (MODEL_ID, []), "set_tags", "Unable to set logged model tags."),
        (
            "delete_logged_model_tag",
            (MODEL_ID, "team"),
            "delete_tag",
            "Unable to delete logged model tag.",
        ),
        ("search_logged_models", (["42"],), "search", "Unable to search logged models."),
    ],
)
def test_persistence_errors_are_sanitized_and_logged_once(
    logged_model_store, logged_model_repository, caplog, operation, args, method, message
):
    error = RepositoryPersistenceError("Private URI and query")
    getattr(logged_model_repository, method).side_effect = error
    with pytest.raises(MlflowException) as caught:
        getattr(logged_model_store, operation)(*args)
    assert caught.value.message == message
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    records = [r for r in caplog.records if r.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].args == (error,)
    assert records[0].exc_info is None


@pytest.mark.parametrize(
    "operation,args",
    [
        ("create_logged_model", ("42", "example-model")),
        ("get_logged_model", (MODEL_ID,)),
        ("delete_logged_model", (MODEL_ID,)),
        ("set_logged_model_tags", (MODEL_ID, [])),
        ("delete_logged_model_tag", (MODEL_ID, "team")),
        ("search_logged_models", (["42"],)),
    ],
)
def test_initialization_failure_is_logged_once(
    logged_model_store, monkeypatch, caplog, operation, args
):
    logged_model_store.__dict__.pop("_logged_model_repository")
    logged_model_store.__dict__["_database"] = MagicMock()
    error = RepositoryPersistenceError("Private initialization details")
    monkeypatch.setattr(tracking_store, "LoggedModelRepository", Mock(side_effect=error))
    with pytest.raises(MlflowException) as caught:
        getattr(logged_model_store, operation)(*args)
    assert caught.value.message == "Unable to initialize logged model repository."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    records = [r for r in caplog.records if r.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].args == (error,)


def test_repository_initialization_is_cached(
    logged_model_store, monkeypatch, logged_model_repository
):
    logged_model_store.__dict__.pop("_logged_model_repository")
    database = MagicMock()
    logged_model_store.__dict__["_database"] = database
    initialize = Mock(return_value=logged_model_repository)
    monkeypatch.setattr(tracking_store, "LoggedModelRepository", initialize)
    logged_model_store.get_logged_model(MODEL_ID)
    logged_model_store.get_logged_model(MODEL_ID)
    initialize.assert_called_once_with(database, settings=logged_model_store._settings)
