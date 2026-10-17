"""Trace lifecycle, span ingestion, and public error contracts."""

import json
from unittest.mock import MagicMock, Mock

import pytest
from mlflow.entities import LifecycleStage
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.tracing.constant import (
    TraceMetadataKey,
    TraceSizeStatsKey,
)

import mlflow_mongodb.tracking.store as store_module
from mlflow_mongodb.infrastructure.errors import (
    RepositoryNotFoundError,
    RepositoryPersistenceError,
)
from mlflow_mongodb.tracking._compat import (  # ruff: ignore[import-private-name]
    SPANS_LOCATION,
    SPANS_LOCATION_TAG,
    TRACE_COST_METADATA,
)
from mlflow_mongodb.tracking.store import MongoDBTrackingStore

TRACE_ID = "tr-0000000000000000000000000000002a"
TIMESTAMP = 1_700_000_000_000


def test_start_trace_normalizes_storage_fields_without_mutating_input(
    trace_store, trace_repository, feedback_factory
):
    assessment = feedback_factory(assessment_id=None)
    info = TraceInfo(
        trace_id=TRACE_ID,
        trace_location=TraceLocation.from_experiment_id("42"),
        request_time=TIMESTAMP,
        state=TraceState.OK,
        tags={"team": "risk", SPANS_LOCATION_TAG: "artifact"},
        trace_metadata={
            TraceMetadataKey.SOURCE_RUN: "run",
            TraceMetadataKey.SIZE_STATS: json.dumps({TraceSizeStatsKey.NUM_SPANS: 2}),
            TraceMetadataKey.TOKEN_USAGE: '{"input_tokens": 4}',
            TRACE_COST_METADATA: '{"total_cost": 0.2}',
        },
        assessments=[assessment],
    )

    result = trace_store.start_trace(info)

    arguments = trace_repository.start_trace.call_args.kwargs
    assert arguments["trace_id"] == TRACE_ID
    assert arguments["experiment_id"] == "42"
    assert arguments["request_time"] == TIMESTAMP
    assert arguments["state"] == "OK"
    assert arguments["tags"][SPANS_LOCATION_TAG] == SPANS_LOCATION
    assert arguments["run_ids"] == ["run"]
    assert arguments["span_stats"] == {TraceSizeStatsKey.NUM_SPANS: 2}
    assert arguments["metrics"] == {"token_usage": {"input_tokens": 4}, "cost": {"total_cost": 0.2}}
    assert arguments["assessments"][0]["assessment_id"]
    assert assessment.assessment_id is None
    assert info.tags[SPANS_LOCATION_TAG] == "artifact"
    assert result.trace_id == TRACE_ID


def test_start_trace_fills_missing_assessment_trace_id_without_mutating_input(
    trace_store, trace_repository, feedback_factory
):
    assessment = feedback_factory(trace_id=None)
    info = TraceInfo(
        TRACE_ID,
        TraceLocation.from_experiment_id("42"),
        TIMESTAMP,
        TraceState.OK,
        assessments=[assessment],
    )

    trace_store.start_trace(info)

    documents = trace_repository.start_trace.call_args.kwargs["assessments"]
    assert len(documents) == 1
    assert documents[0]["trace_id"] == TRACE_ID
    assert documents[0]["assessment_id"] == assessment.assessment_id
    assert assessment.trace_id is None


def test_start_trace_rejects_deleted_experiment_before_write(
    trace_store, trace_repository, experiment_repository, experiment_record_factory
):
    experiment_repository.find_by_id.return_value = experiment_record_factory(
        lifecycle_stage=LifecycleStage.DELETED
    )
    info = TraceInfo(TRACE_ID, TraceLocation.from_experiment_id("42"), TIMESTAMP, TraceState.OK)

    with pytest.raises(MlflowException) as caught:
        trace_store.start_trace(info)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    trace_repository.start_trace.assert_not_called()


@pytest.mark.parametrize(
    "value", ["broken", "[]", '{"num_spans": true}', '{"num_spans": -1}', '{"num_spans": 1.5}']
)
def test_invalid_span_stats_are_public_validation_errors(value):
    with pytest.raises(MlflowException) as caught:
        MongoDBTrackingStore._parse_span_stats(value)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", {}),
        ("{}", {}),
        ('{"num_spans": 0}', {"num_spans": 0}),
        ('{"num_spans": 3}', {"num_spans": 3}),
    ],
)
def test_span_stats_decode_at_interface_boundary(value, expected):
    assert MongoDBTrackingStore._parse_span_stats(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"tokens": 1}, {"tokens": 1}),
        ('{"tokens": 2}', {"tokens": 2}),
        ({}, {}),
        (None, None),
        ("bad", None),
        ([1], None),
        ({"tokens": True}, None),
        ({"tokens": "1"}, None),
        ({"tokens": float("nan")}, None),
        ({"tokens": float("inf")}, None),
    ],
)
def test_numeric_stats_accept_only_finite_numeric_objects(value, expected):
    assert MongoDBTrackingStore._numeric_trace_stats(value) == expected


@pytest.mark.parametrize(
    "criteria",
    [
        {},
        {"trace_ids": []},
        {"trace_ids": [TRACE_ID], "max_timestamp_millis": 1},
        {"trace_ids": [TRACE_ID], "max_traces": 1},
        {"max_timestamp_millis": 1, "max_traces": 0},
        {"max_timestamp_millis": 1, "max_traces": -1},
    ],
)
def test_delete_validates_selection_before_repository_call(trace_store, trace_repository, criteria):
    with pytest.raises(MlflowException) as caught:
        trace_store.delete_traces("42", **criteria)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    trace_repository.delete_traces.assert_not_called()


@pytest.mark.parametrize(
    "criteria", [{"trace_ids": [TRACE_ID]}, {"max_timestamp_millis": 0, "max_traces": 2}]
)
def test_delete_returns_repository_count(trace_store, trace_repository, criteria):
    trace_repository.delete_traces.return_value = 2
    assert trace_store.delete_traces("42", **criteria) == 2
    trace_repository.delete_traces.assert_called_once_with(
        experiment_id="42",
        max_timestamp_millis=criteria.get("max_timestamp_millis"),
        max_traces=criteria.get("max_traces"),
        trace_ids=criteria.get("trace_ids"),
    )


def test_get_trace_info_converts_metadata_and_assessments(
    trace_store, trace_repository, trace_record_factory, feedback_factory
):
    assessment = feedback_factory(valid=False)
    trace_repository.get_trace_info.return_value = trace_record_factory(
        assessments=(trace_store._assessment_to_document(assessment),),
        client_request_id="client",
        request_preview="request",
        response_preview="response",
    )

    info = trace_store.get_trace_info(TRACE_ID)

    assert info.trace_id == TRACE_ID
    assert info.experiment_id == "42"
    assert info.state == TraceState.OK
    assert info.tags == {"team": "risk"}
    assert info.trace_metadata == {"source": "test"}
    assert info.client_request_id == "client"
    assert info.request_preview == "request"
    assert info.response_preview == "response"
    assert info.assessments[0].valid is False


def test_get_trace_reports_incomplete_export_after_bounded_retries(
    trace_store, trace_repository, monkeypatch
):
    monkeypatch.setattr("mlflow_mongodb.infrastructure.retry.time.sleep", Mock())
    with pytest.raises(MlflowException, match="not fully exported") as caught:
        trace_store.get_trace(TRACE_ID)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert caught.value.__cause__ is None
    assert trace_repository.get_spans.call_count == 3


def test_partial_trace_can_be_empty_without_retry(trace_store, trace_repository):
    trace = trace_store.get_trace(TRACE_ID, allow_partial=True)
    assert trace.data.spans == []
    trace_repository.get_spans.assert_called_once_with(TRACE_ID)


def test_get_missing_trace_does_not_retry_or_load_spans(trace_store, trace_repository):
    trace_repository.get_trace_info.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException) as caught:
        trace_store.get_trace(TRACE_ID)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert "private" not in caught.value.message
    trace_repository.get_trace_info.assert_called_once_with(TRACE_ID)
    trace_repository.get_spans.assert_not_called()


def test_get_trace_database_failure_is_sanitized_without_retry(
    trace_store, trace_repository, caplog
):
    error = RepositoryPersistenceError("Private query")
    trace_repository.get_spans.side_effect = error
    with pytest.raises(MlflowException) as caught:
        trace_store.get_trace(TRACE_ID)
    assert caught.value.message == "Unable to read trace."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    trace_repository.get_spans.assert_called_once_with(TRACE_ID)
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


@pytest.mark.parametrize("trace_ids,experiment_ids", [([], None), ([TRACE_ID], [])])
def test_batch_metadata_empty_selection_avoids_repository(
    trace_store, trace_repository, trace_ids, experiment_ids
):
    assert trace_store.batch_get_trace_infos(trace_ids, experiment_ids=experiment_ids) == []
    trace_repository.batch_get_trace_infos.assert_not_called()


@pytest.mark.parametrize("trace_ids,experiment_ids", [([], None), ([TRACE_ID], [])])
def test_batch_traces_empty_selection_avoids_repository(
    trace_store, trace_repository, trace_ids, experiment_ids
):
    assert trace_store.batch_get_traces(trace_ids, experiment_ids=experiment_ids) == []
    trace_repository.batch_get_traces.assert_not_called()


def test_trace_repository_initialization_is_cached(trace_store, trace_repository, monkeypatch):
    trace_store.__dict__.pop("_trace_repository")
    database = MagicMock()
    trace_store.__dict__["_database"] = database
    initialize = Mock(return_value=trace_repository)
    monkeypatch.setattr(store_module, "TraceRepository", initialize)
    trace_store.get_trace_info(TRACE_ID)
    trace_store.get_trace_info(TRACE_ID)
    initialize.assert_called_once_with(database, settings=trace_store._settings)


def test_trace_repository_initialization_failure_is_logged_once(trace_store, monkeypatch, caplog):
    trace_store.__dict__.pop("_trace_repository")
    trace_store.__dict__["_database"] = MagicMock()
    error = RepositoryPersistenceError("Private initialization details")
    monkeypatch.setattr(store_module, "TraceRepository", Mock(side_effect=error))
    with pytest.raises(MlflowException) as caught:
        trace_store.get_trace_info(TRACE_ID)
    assert caught.value.message == "Unable to initialize trace repository."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    records = [record for record in caplog.records if record.name == store_module.logger.name]
    assert len(records) == 1
    assert records[0].args == (error,)
    assert records[0].exc_info is None


def test_set_trace_tag_passes_literal_value(trace_store, trace_repository):
    trace_store.set_trace_tag(TRACE_ID, "team", "$value")
    trace_repository.set_trace_tag.assert_called_once_with(
        trace_id=TRACE_ID, key="team", value="$value"
    )


def test_set_trace_tag_missing_trace_keeps_safe_domain_error(trace_store, trace_repository):
    trace_repository.set_trace_tag.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException) as caught:
        trace_store.set_trace_tag(TRACE_ID, "team", "risk")
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert "private" not in caught.value.message
    assert caught.value.__cause__ is None


def test_delete_trace_tag_missing_resource_keeps_safe_domain_error(trace_store, trace_repository):
    trace_repository.delete_trace_tag.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException) as caught:
        trace_store.delete_trace_tag(TRACE_ID, "team")
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert "private" not in caught.value.message
    assert caught.value.__cause__ is None


def test_prompt_links_use_public_name_version_identity(trace_store, trace_repository):
    prompt = Mock(spec=PromptVersion, name="prompt")
    prompt.name = "greeting"
    prompt.version = 2
    trace_store.link_prompts_to_trace(TRACE_ID, [prompt])
    trace_repository.link_prompts.assert_called_once_with(
        trace_id=TRACE_ID, prompt_versions=[{"name": "greeting", "version": "2"}]
    )


def test_empty_prompt_links_skip_repository(trace_store, trace_repository):
    trace_store.link_prompts_to_trace(TRACE_ID, [])
    trace_repository.link_prompts.assert_not_called()


def test_prompt_links_missing_trace_is_safely_translated(trace_store, trace_repository):
    trace_repository.link_prompts.side_effect = RepositoryNotFoundError("private")
    prompt = Mock(spec=PromptVersion)
    prompt.name, prompt.version = "greeting", 1
    with pytest.raises(MlflowException) as caught:
        trace_store.link_prompts_to_trace(TRACE_ID, [prompt])
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert "private" not in caught.value.message
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("trace_ids,run_id", [([TRACE_ID], ""), ([TRACE_ID] * 101, "run")])
def test_run_links_validate_request_before_repository(
    trace_store, trace_repository, trace_ids, run_id
):
    with pytest.raises(MlflowException) as caught:
        trace_store.link_traces_to_run(trace_ids, run_id)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    trace_repository.link_traces_to_run.assert_not_called()


def test_empty_run_links_skip_validation_and_repository(trace_store, trace_repository):
    trace_store.link_traces_to_run([], "")
    trace_repository.link_traces_to_run.assert_not_called()


def test_run_links_accept_maximum_request_size(trace_store, trace_repository):
    trace_ids = [f"tr-{index}" for index in range(100)]
    trace_store.link_traces_to_run(trace_ids, "run")
    trace_repository.link_traces_to_run.assert_called_once_with(trace_ids=trace_ids, run_id="run")


def test_run_link_persistence_failure_is_sanitized(trace_store, trace_repository, caplog):
    error = RepositoryPersistenceError("Private URI and query")
    trace_repository.link_traces_to_run.side_effect = error
    with pytest.raises(MlflowException) as caught:
        trace_store.link_traces_to_run([TRACE_ID], "run")
    assert caught.value.message == "Unable to link traces to run."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None
