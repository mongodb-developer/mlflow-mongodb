"""Trace lifecycle, span ingestion, and public error contracts."""

import asyncio
import json
import logging
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, call

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
    TEMPORARILY_UNAVAILABLE,
    ErrorCode,
)
from mlflow.tracing.constant import (
    TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceSizeStatsKey,
)
from opentelemetry.trace import StatusCode

import mlflow_mongodb.tracking.store as store_module
from mlflow_mongodb.infrastructure.errors import (
    RepositoryNotFoundError,
    RepositoryPersistenceError,
    RepositoryWriteConflictError,
)
from mlflow_mongodb.tracking._compat import (  # ruff: ignore[import-private-name]
    SPAN_COST_ATTRIBUTE,
    SPAN_SESSION_ATTRIBUTE,
    SPAN_TRACE_TAG_PREFIX,
    SPANS_LOCATION,
    SPANS_LOCATION_TAG,
    TRACE_COST_METADATA,
)
from mlflow_mongodb.tracking.store import MongoDBTrackingStore
from mlflow_mongodb.tracking.types import SpanSummaryRecord

TRACE_ID = "tr-0000000000000000000000000000002a"
TIMESTAMP = 1_700_000_000_000


@dataclass
class _SpanAttributeValue:
    message: str


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


@pytest.mark.parametrize("allow_partial", [False, True])
def test_get_trace_sorts_root_first_then_time_and_identity(
    trace_store, trace_repository, span_record_factory, allow_partial
):
    root = span_record_factory(span_id=1, start_time=2_000_000)
    child = span_record_factory(span_id=3, parent_id=1, start_time=1_000_000)
    sibling = span_record_factory(span_id=2, parent_id=1, start_time=1_000_000)
    trace_repository.get_spans.return_value = [child, sibling, root]

    trace = trace_store.get_trace(TRACE_ID, allow_partial=allow_partial)

    assert [span.span_id for span in trace.data.spans] == [
        root.span_id,
        sibling.span_id,
        child.span_id,
    ]
    trace_repository.get_trace_info.assert_called_once_with(TRACE_ID)
    trace_repository.get_spans.assert_called_once_with(TRACE_ID)


def test_get_trace_reloads_metadata_and_spans_after_incomplete_export(
    trace_store, trace_repository, trace_record_factory, span_record_factory, monkeypatch
):
    sleep = Mock()
    monkeypatch.setattr("mlflow_mongodb.infrastructure.retry.time.sleep", sleep)
    root, child = span_record_factory(), span_record_factory(span_id=2, parent_id=1)
    trace_repository.get_trace_info.return_value = trace_record_factory(span_stats={"num_spans": 2})
    trace_repository.get_spans.side_effect = [[], [root], [root, child]]

    trace = trace_store.get_trace(TRACE_ID)

    assert len(trace.data.spans) == 2
    assert trace_repository.get_trace_info.call_count == 3
    assert trace_repository.get_spans.call_count == 3
    assert sleep.call_args_list == [call(1), call(2)]


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


def test_batch_traces_omits_empty_and_incomplete_exports(
    trace_store, trace_repository, trace_record_factory, span_record_factory
):
    span = span_record_factory()
    trace_repository.batch_get_traces.return_value = [
        (trace_record_factory(_id="empty"), []),
        (trace_record_factory(_id="partial", span_stats={"num_spans": 2}), [span]),
        (trace_record_factory(), [span]),
    ]
    result = trace_store.batch_get_traces(["empty", "partial", TRACE_ID], experiment_ids=["42"])
    assert [trace.info.trace_id for trace in result] == [TRACE_ID]
    trace_repository.batch_get_traces.assert_called_once_with(
        ["empty", "partial", TRACE_ID], experiment_ids=["42"]
    )


def test_log_spans_empty_batch_avoids_experiment_lookup(trace_store, experiment_repository):
    assert trace_store.log_spans("", []) == []
    experiment_repository.find_by_id.assert_not_called()


@pytest.mark.parametrize("location", [None, "", 42])
def test_log_spans_rejects_invalid_location_before_lookup(
    trace_store, experiment_repository, span_factory, location
):
    with pytest.raises(MlflowException) as caught:
        trace_store.log_spans(location, [span_factory()])
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.find_by_id.assert_not_called()


def test_log_spans_rejects_deleted_experiment_before_writes(
    trace_store, trace_repository, experiment_repository, experiment_record_factory, span_factory
):
    experiment_repository.find_by_id.return_value = experiment_record_factory(
        lifecycle_stage=LifecycleStage.DELETED
    )
    with pytest.raises(MlflowException) as caught:
        trace_store.log_spans("42", [span_factory()])
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    trace_repository.ensure_traces.assert_not_called()
    trace_repository.log_spans.assert_not_called()


def test_log_spans_deduplicates_before_writing_and_refreshes_each_trace(
    trace_store, trace_repository, span_factory, monkeypatch
):
    first = span_factory(span_id=1, start_time=4_000_000)
    duplicate = span_factory(span_id=1, start_time=9_000_000)
    child = span_factory(span_id=2, parent_id=1, start_time=2_000_000)
    other = span_factory(span_id=1, trace_id="other", start_time=6_000_000)
    refresh = Mock()
    monkeypatch.setattr(trace_store, "_refresh_trace_span_summary", refresh)
    spans = [first, duplicate, child, other]

    assert trace_store.log_spans("42", spans) is spans

    trace_repository.ensure_traces.assert_called_once_with(
        experiment_id="42", request_times={TRACE_ID: 2, "other": 6}
    )
    documents = trace_repository.log_spans.call_args.args[0]
    assert len(documents) == 3
    assert documents[0]["start_time_ns"] == first.start_time_ns
    assert refresh.call_args_list == [call(TRACE_ID, "42", {}), call("other", "42", {})]
    assert [entry[0] for entry in trace_repository.method_calls] == ["ensure_traces", "log_spans"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("../team", "risk"),
        ("invalid?key", "risk"),
    ],
)
def test_invalid_span_trace_tag_is_skipped_without_changing_existing_tags(caplog, key, value):
    tags = {"team": "original", "preserve": "yes"}

    with caplog.at_level(logging.DEBUG, logger=store_module.logger.name):
        MongoDBTrackingStore._add_span_trace_tag(tags, key, value)

    assert tags == {"team": "original", "preserve": "yes"}
    assert caplog.records[-1].message == f"Skipping invalid span-derived trace tag {key!r}"


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_span_trace_tag_serialization_failure_preserves_existing_tags(
    monkeypatch, caplog, error_type
):
    monkeypatch.setattr(
        store_module, "dump_span_attribute_value", Mock(side_effect=error_type("Invalid value"))
    )
    tags = {"team": "original", "preserve": "yes"}

    with caplog.at_level(logging.DEBUG, logger=store_module.logger.name):
        MongoDBTrackingStore._add_span_trace_tag(tags, "team", object())

    assert tags == {"team": "original", "preserve": "yes"}
    assert caplog.records[-1].message == "Skipping invalid span-derived trace tag 'team'"


@pytest.mark.parametrize("message", ["answer", "café"])
def test_span_trace_tag_serializes_dataclass_with_mlflow_encoder(message):
    tags = {}

    MongoDBTrackingStore._add_span_trace_tag(tags, "answer", _SpanAttributeValue(message))

    assert tags == {"answer": json.dumps({"message": message}, ensure_ascii=False)}


@pytest.mark.parametrize("message", ["answer", "café"])
def test_span_preview_serializes_dataclass_with_mlflow_encoder(message):
    preview = MongoDBTrackingStore._span_preview(_SpanAttributeValue(message), "user")

    assert preview == json.dumps({"message": message}, ensure_ascii=False)


@pytest.mark.parametrize(
    "value,expected",
    [
        (["hello", 1], '["hello", 1]'),
        ((1, 2), "[1, 2]"),
        ([], "[]"),
        (42, "42"),
        (0, "0"),
        (1.5, "1.5"),
        (True, "true"),
        (False, "false"),
        ({}, "{}"),
        ({"answer": "café"}, '{"answer": "café"}'),
    ],
)
def test_span_preview_json_encodes_non_string_values(value, expected):
    assert MongoDBTrackingStore._span_preview(value, "user") == expected


def test_span_preview_caps_oversized_helper_result_at_oss_limit(monkeypatch):
    limit = TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS
    oversized = "x" * (limit + 10)
    extract_preview = Mock(return_value=oversized)
    monkeypatch.setattr(store_module, "_get_truncated_preview", extract_preview)

    preview = MongoDBTrackingStore._span_preview("request", "user")

    assert preview == "x" * (limit - 3) + "..."
    assert len(preview) == limit
    extract_preview.assert_called_once_with("request", role="user")


@pytest.mark.parametrize("error_type", [TypeError, ValueError, AttributeError])
def test_span_preview_extraction_failure_returns_none_and_logs_debug(
    monkeypatch, caplog, error_type
):
    error = error_type("Invalid preview content")
    extract_preview = Mock(side_effect=error)
    monkeypatch.setattr(store_module, "_get_truncated_preview", extract_preview)

    with caplog.at_level(logging.DEBUG, logger=store_module.logger.name):
        result = MongoDBTrackingStore._span_preview("request", "user")

    assert result is None
    extract_preview.assert_called_once_with("request", role="user")
    assert caplog.records[-1].levelno == logging.DEBUG
    assert caplog.records[-1].msg == "Could not extract the %s trace preview: %s"
    assert caplog.records[-1].args == ("user", error)


def test_span_document_retains_encoded_payload_and_extracts_native_query_values(span_factory):
    span = span_factory(
        attributes={
            SpanAttributeKey.CHAT_USAGE: {"input_tokens": 4},
            SPAN_COST_ATTRIBUTE: {"total_cost": 0.2},
            SPAN_SESSION_ATTRIBUTE: "session",
            SPAN_TRACE_TAG_PREFIX + "team": "root",
        },
        resource_attributes={"team": "resource", "telemetry.sdk.name": "ignored"},
    )
    document = MongoDBTrackingStore._span_to_document(span)
    assert document["token_usage"] == {"input_tokens": 4}
    assert document["cost"] == {"total_cost": 0.2}
    assert isinstance(document["content"]["attributes"][SpanAttributeKey.CHAT_USAGE], str)
    assert document["trace_fields"]["metadata"][TraceMetadataKey.TRACE_SESSION] == "session"
    assert document["trace_fields"]["resource_tags"] == {"team": "resource"}
    assert document["trace_fields"]["root_tags"] == {"team": "root"}


@pytest.mark.parametrize(
    ("parent_id", "status", "state"),
    [
        (None, StatusCode.OK, "OK"),
        (None, StatusCode.ERROR, "ERROR"),
        (9, StatusCode.OK, "IN_PROGRESS"),
    ],
)
def test_summary_uses_root_state_and_persisted_time_bounds(span_factory, parent_id, status, state):
    span = span_factory(parent_id=parent_id, status=status)
    document = MongoDBTrackingStore._span_to_document(span)
    summary = MongoDBTrackingStore._summarize_spans([SpanSummaryRecord.from_document(document)])
    assert summary["state"] == state
    assert summary["request_time"] == 1
    assert summary["execution_duration"] == 10


def test_batch_root_tags_override_first_resource_tags(span_factory):
    root = span_factory(
        attributes={SPAN_TRACE_TAG_PREFIX + "team": "root"},
        resource_attributes={"team": "resource"},
    )
    child = span_factory(span_id=2, parent_id=1, resource_attributes={"other": "value"})
    _, _, batch_tags = MongoDBTrackingStore._prepare_span_documents([root, child])
    assert batch_tags[root.trace_id] == {"team": "root"}


def test_duplicate_span_uses_selected_resource_tags_and_preserves_first_document(span_factory):
    first = span_factory(parent_id=9, start_time=1_000_000)
    duplicate = span_factory(
        parent_id=9,
        start_time=2_000_000,
        resource_attributes={"team.name": "$risk"},
    )

    documents, documents_by_trace, batch_tags = MongoDBTrackingStore._prepare_span_documents(
        [first, duplicate]
    )

    assert len(documents) == 1
    assert documents[0]["start_time_ns"] == first.start_time_ns
    assert documents[0]["trace_fields"]["resource_tags"] == {}
    assert documents_by_trace[first.trace_id] == documents
    assert batch_tags[first.trace_id] == {"team.name": "$risk"}


def test_summary_does_not_count_usage_twice_when_parent_arrives_later(span_factory):
    child = span_factory(
        span_id=2,
        parent_id=1,
        attributes={
            SpanAttributeKey.CHAT_USAGE: {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        },
    )
    parent = span_factory(
        attributes={
            SpanAttributeKey.CHAT_USAGE: {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        }
    )
    records = [
        SpanSummaryRecord.from_document(MongoDBTrackingStore._span_to_document(span))
        for span in (child, parent)
    ]
    assert MongoDBTrackingStore._summarize_spans(records)["token_usage"] == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }


def test_refresh_retries_with_new_snapshot_after_revision_conflict(
    trace_store, trace_repository, span_factory, monkeypatch
):
    monkeypatch.setattr("mlflow_mongodb.infrastructure.retry.time.sleep", Mock())
    summary_record = SpanSummaryRecord.from_document(
        trace_store._span_to_document(
            span_factory(attributes={SpanAttributeKey.CHAT_USAGE: {"input_tokens": 4}})
        )
    )
    trace_repository.span_summary_snapshot.side_effect = [
        (1, [summary_record]),
        (2, [summary_record]),
    ]
    trace_repository.update_span_summary.side_effect = [
        RepositoryWriteConflictError(TRACE_ID),
        None,
    ]

    trace_store._refresh_trace_span_summary(TRACE_ID, "42", {})

    assert trace_repository.span_summary_snapshot.call_count == 2
    updates = trace_repository.update_span_summary.call_args_list
    assert [entry.kwargs["revision"] for entry in updates] == [1, 2]
    summary = updates[-1].kwargs["summary"]
    assert summary["token_usage"]["input_tokens"] == 4
    assert (
        json.loads(summary["aggregate_metadata"][TraceMetadataKey.TOKEN_USAGE])
        == summary["token_usage"]
    )


def test_refresh_empty_snapshot_does_not_finalize_trace(trace_store, trace_repository):
    trace_repository.span_summary_snapshot.return_value = (1, [])
    with pytest.raises(RepositoryNotFoundError):
        trace_store._refresh_trace_span_summary(TRACE_ID, "42", {})
    trace_repository.update_span_summary.assert_not_called()


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (RepositoryNotFoundError(TRACE_ID), RESOURCE_DOES_NOT_EXIST),
        (RepositoryWriteConflictError(TRACE_ID), TEMPORARILY_UNAVAILABLE),
    ],
)
def test_ingestion_summary_domain_errors_have_distinct_public_codes(
    trace_store, span_factory, monkeypatch, error, code
):
    monkeypatch.setattr(trace_store, "_refresh_trace_span_summary", Mock(side_effect=error))
    with pytest.raises(MlflowException) as caught:
        trace_store.log_spans("42", [span_factory()])
    assert caught.value.error_code == ErrorCode.Name(code)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_async_span_ingestion_delegates_to_sync_store(trace_store, span_factory, monkeypatch):
    spans = [span_factory()]
    log_spans = Mock(return_value=spans)
    monkeypatch.setattr(trace_store, "log_spans", log_spans)
    assert asyncio.run(trace_store.log_spans_async("42", spans)) is spans
    log_spans.assert_called_once_with("42", spans)


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


@pytest.mark.parametrize(
    "stage", ["ensure_traces", "log_spans", "span_summary_snapshot", "update_span_summary"]
)
def test_span_ingestion_translates_failure_at_each_persistence_stage(
    trace_store, trace_repository, span_factory, caplog, stage
):
    span = span_factory()
    summary_record = SpanSummaryRecord.from_document(trace_store._span_to_document(span))
    trace_repository.span_summary_snapshot.return_value = (1, [summary_record])
    error = RepositoryPersistenceError("Private write details")
    getattr(trace_repository, stage).side_effect = error

    with pytest.raises(MlflowException) as caught:
        trace_store.log_spans("42", [span])

    assert caught.value.message == "Unable to log spans."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


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
