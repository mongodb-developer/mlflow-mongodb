"""Trace metadata and payload reads against MongoDB."""

import json

import pytest
from mlflow.entities import TraceLocation
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracing.constant import TraceMetadataKey, TraceSizeStatsKey


@pytest.fixture
def scoped_traces(store, persisted_trace_factory):
    first = persisted_trace_factory()
    other_experiment_id = store.create_experiment("other-trace-experiment")
    second = persisted_trace_factory(
        trace_location=TraceLocation.from_experiment_id(other_experiment_id)
    )
    return first, second


@pytest.mark.parametrize(
    "scope,expected_indexes",
    [(None, [1, 0]), ((0,), [0]), ((1,), [1]), ((), [])],
)
def test_batch_metadata_preserves_request_order_deduplicates_and_scopes_experiments(
    store, scoped_traces, scope, expected_indexes
):
    infos = [info for info, _ in scoped_traces]
    requested_ids = [infos[1].trace_id, "tr-missing", infos[0].trace_id, infos[1].trace_id]
    experiment_ids = None if scope is None else [infos[index].experiment_id for index in scope]

    loaded = store.batch_get_trace_infos(requested_ids, experiment_ids=experiment_ids)

    assert loaded == [infos[index] for index in expected_indexes]
    assert requested_ids == [infos[1].trace_id, "tr-missing", infos[0].trace_id, infos[1].trace_id]


@pytest.mark.parametrize(
    "scope,expected_indexes",
    [(None, [1, 0]), ((0,), [0]), ((1,), [1]), ((), [])],
)
def test_batch_traces_preserves_request_order_deduplicates_and_scopes_experiments(
    store, scoped_traces, scope, expected_indexes
):
    infos = [info for info, _ in scoped_traces]
    requested_ids = [infos[1].trace_id, "tr-missing", infos[0].trace_id, infos[1].trace_id]
    experiment_ids = None if scope is None else [infos[index].experiment_id for index in scope]

    loaded = store.batch_get_traces(requested_ids, experiment_ids=experiment_ids)

    assert [trace.info for trace in loaded] == [infos[index] for index in expected_indexes]
    assert [[span.to_dict() for span in trace.data.spans] for trace in loaded] == [
        [span.to_dict() for span in scoped_traces[index][1]] for index in expected_indexes
    ]
    assert requested_ids == [infos[1].trace_id, "tr-missing", infos[0].trace_id, infos[1].trace_id]


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {TraceMetadataKey.SIZE_STATS: json.dumps({TraceSizeStatsKey.NUM_SPANS: 3})},
    ],
)
def test_get_trace_returns_metadata_assessments_and_root_first_spans(
    store, persisted_trace_factory, metadata
):
    info, spans = persisted_trace_factory(trace_metadata=metadata)

    loaded = store.get_trace(info.trace_id)

    assert loaded.info == info
    assert [span.to_dict() for span in loaded.data.spans] == [span.to_dict() for span in spans]
    assert loaded.data.spans[0].parent_id is None
    assert loaded.data.spans[1].inputs == {"prompt": "$literal"}
    assert loaded.data.spans[1].outputs == {"answer": [1, True]}


@pytest.mark.parametrize("span_count", [0, 1])
def test_partial_trace_returns_available_spans_despite_expected_count(
    store, persisted_trace_factory, span_count
):
    info, spans = persisted_trace_factory(
        span_count=span_count,
        trace_metadata={TraceMetadataKey.SIZE_STATS: json.dumps({TraceSizeStatsKey.NUM_SPANS: 3})},
    )

    loaded = store.get_trace(info.trace_id, allow_partial=True)

    assert loaded.info == info
    assert [span.to_dict() for span in loaded.data.spans] == [span.to_dict() for span in spans]


@pytest.mark.parametrize("span_count,expected_count", [(0, 0), (0, 1), (1, 3)])
def test_batch_traces_omits_empty_incomplete_and_missing_traces(
    store, persisted_trace_factory, span_count, expected_count
):
    complete, spans = persisted_trace_factory()
    incomplete, _ = persisted_trace_factory(
        span_count=span_count,
        trace_metadata={
            TraceMetadataKey.SIZE_STATS: json.dumps({TraceSizeStatsKey.NUM_SPANS: expected_count})
        },
    )

    loaded = store.batch_get_traces([incomplete.trace_id, "tr-missing", complete.trace_id])

    assert [trace.info for trace in loaded] == [complete]
    assert [span.to_dict() for span in loaded[0].data.spans] == [span.to_dict() for span in spans]
    assert store.batch_get_trace_infos([incomplete.trace_id]) == [incomplete]


def test_get_trace_rejects_incomplete_export_after_reading_persisted_spans(
    store, persisted_trace_factory
):
    info, _ = persisted_trace_factory(
        span_count=1,
        trace_metadata={TraceMetadataKey.SIZE_STATS: json.dumps({TraceSizeStatsKey.NUM_SPANS: 3})},
    )

    with pytest.raises(MlflowException) as caught:
        store.get_trace(info.trace_id)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert "not fully exported" in caught.value.message
    assert caught.value.__cause__ is None
    assert store.get_trace_info(info.trace_id) == info
