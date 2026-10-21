"""Trace creation, span ingestion, retrieval, and deletion against MongoDB."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest
from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.protos.service_pb2 import DeleteTraces, LinkTracesToRun
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TraceMetadataKey,
    TraceSizeStatsKey,
)
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from opentelemetry.trace import StatusCode

from mlflow_mongodb.tracking._compat import (  # ruff: ignore[import-private-name]
    SPAN_COST_ATTRIBUTE,
    SPAN_SESSION_ATTRIBUTE,
    SPAN_TRACE_TAG_PREFIX,
    SPANS_LOCATION,
    SPANS_LOCATION_TAG,
    TRACE_COST_METADATA,
)


def test_start_trace_round_trips_metadata_and_native_assessment_values(store, trace_info_factory):
    info = trace_info_factory(
        tags={"team.name": "$risk"},
        trace_metadata={
            TraceMetadataKey.SOURCE_RUN: "run",
            TraceMetadataKey.SIZE_STATS: json.dumps({TraceSizeStatsKey.NUM_SPANS: 2}),
            TraceMetadataKey.TOKEN_USAGE: '{"input_tokens": 4}',
            TRACE_COST_METADATA: '{"total_cost": 0.2}',
        },
        request_preview="$request",
        response_preview="$response",
        client_request_id="$client",
    )
    expectation = Expectation(name="expected", value={"answer": [1, True]}, trace_id=info.trace_id)
    info.assessments = [expectation]
    original = deepcopy(info)

    created = store.start_trace(info)
    loaded = store.get_trace_info(info.trace_id)

    assert created == loaded
    assert loaded.trace_id == info.trace_id
    assert loaded.experiment_id == info.experiment_id
    assert loaded.request_time == info.request_time
    assert loaded.execution_duration == 10
    assert loaded.state == TraceState.OK
    assert loaded.request_preview == "$request"
    assert loaded.response_preview == "$response"
    assert loaded.client_request_id == "$client"
    assert loaded.tags == {
        "team.name": "$risk",
        SPANS_LOCATION_TAG: SPANS_LOCATION,
        MLFLOW_ARTIFACT_LOCATION: (
            f"{store.get_experiment(info.experiment_id).artifact_location}"
            f"/traces/{info.trace_id}/artifacts"
        ),
    }
    assert loaded.trace_metadata == info.trace_metadata
    assert loaded.assessments[0].value == {"answer": [1, True]}
    assert loaded.assessments[0].assessment_id
    assert info == original
    document = store._trace_repository._collection.find_one({"_id": info.trace_id})
    assert document["run_ids"] == ["run"]
    assert document["span_stats"] == {TraceSizeStatsKey.NUM_SPANS: 2}
    assert document["token_usage"] == {"input_tokens": 4}
    assert document["cost"] == {"total_cost": 0.2}
    assert all(set(item) == {"k", "v"} for item in document["tags"] + document["trace_metadata"])
    assessment = store._trace_repository._assessments_collection.find_one(
        {"trace_id": info.trace_id}
    )
    assert assessment["content"]["expectation"] == {"value": {"answer": [1, True]}}


@pytest.mark.parametrize(
    "status,expected_state", [(StatusCode.OK, TraceState.OK), (StatusCode.ERROR, TraceState.ERROR)]
)
def test_child_first_delivery_creates_placeholder_and_late_root_refreshes_summary(
    store, trace_info_factory, span_factory, status, expected_state
):
    info = trace_info_factory()
    child = span_factory(
        trace_id=info.trace_id,
        span_id=2,
        parent_id=1,
        attributes={
            SpanAttributeKey.CHAT_USAGE: {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    )
    root = span_factory(
        trace_id=info.trace_id,
        status=status,
        attributes={
            SPAN_TRACE_TAG_PREFIX + "team": "$root",
            SPAN_SESSION_ATTRIBUTE: "session",
            SpanAttributeKey.CHAT_USAGE: {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            SpanAttributeKey.INPUTS: "question",
            SpanAttributeKey.OUTPUTS: "answer",
        },
        resource_attributes={"team": "resource", "telemetry.sdk.name": "ignored"},
    )

    store.log_spans(info.experiment_id, [child])
    assert store.get_trace_info(info.trace_id).state == TraceState.IN_PROGRESS
    store.log_spans(info.experiment_id, [root, child])
    result = store.get_trace(info.trace_id)

    assert result.info.state == expected_state
    assert result.info.request_time == root.start_time_ns // 1_000_000
    assert result.info.execution_duration == 10
    assert result.info.tags["team"] == "$root"
    assert "telemetry.sdk.name" not in result.info.tags
    assert result.info.trace_metadata[TraceMetadataKey.TRACE_SESSION] == "session"
    assert json.loads(result.info.trace_metadata[TraceMetadataKey.TOKEN_USAGE]) == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert [span.span_id for span in result.data.spans] == [root.span_id, child.span_id]
    assert (
        store._trace_repository._spans_collection.count_documents({"trace_id": info.trace_id}) == 2
    )


def test_concurrent_metadata_and_span_ingestion_preserve_one_complete_trace(
    store, trace_info_factory, span_factory
):
    repository = store._trace_repository

    def write_metadata(barrier, info):
        barrier.wait()
        return store.start_trace(info)

    def write_spans(barrier, info, root):
        barrier.wait()
        return store.log_spans(info.experiment_id, [root])

    with ThreadPoolExecutor(max_workers=2) as executor:
        for _ in range(8):
            info = trace_info_factory(
                tags={"team": "$metadata"},
                trace_metadata={"source": "$authoritative"},
            )
            root = span_factory(trace_id=info.trace_id)
            barrier = Barrier(2, timeout=10)

            metadata = executor.submit(write_metadata, barrier, info)
            spans = executor.submit(write_spans, barrier, info, root)
            assert metadata.result(timeout=30).trace_id == info.trace_id
            assert spans.result(timeout=30) == [root]

            loaded = store.get_trace(info.trace_id)
            assert loaded.info.experiment_id == info.experiment_id
            assert loaded.info.tags["team"] == "$metadata"
            assert loaded.info.trace_metadata["source"] == "$authoritative"
            assert loaded.info.state == TraceState.OK
            assert [item.span_id for item in loaded.data.spans] == [root.span_id]
            assert repository._collection.count_documents({"_id": info.trace_id}) == 1
            assert repository._spans_collection.count_documents({"trace_id": info.trace_id}) == 1
            document = repository._collection.find_one({"_id": info.trace_id})
            assert document["trace_info_finalized"] is True


def test_span_summary_preserves_authoritative_trace_fields(store, trace_info_factory, span_factory):
    info = trace_info_factory(
        request_time=1000,
        execution_duration=77,
        request_preview="explicit",
        tags={"team": "explicit"},
        trace_metadata={
            TraceMetadataKey.TOKEN_USAGE: '{"input_tokens": 20}',
            TRACE_COST_METADATA: '{"total_cost": 0.5}',
        },
    )
    store.start_trace(info)
    root = span_factory(
        trace_id=info.trace_id,
        status=StatusCode.ERROR,
        attributes={
            SPAN_TRACE_TAG_PREFIX + "team": "derived",
            SPAN_TRACE_TAG_PREFIX + "new": "$value",
            SpanAttributeKey.CHAT_USAGE: {"input_tokens": 3},
            SPAN_COST_ATTRIBUTE: {"total_cost": 0.1},
        },
    )

    store.log_spans(info.experiment_id, [root])

    loaded = store.get_trace_info(info.trace_id)
    assert loaded.state == TraceState.OK
    assert loaded.request_time == 1000
    assert loaded.execution_duration == 77
    assert loaded.request_preview == "explicit"
    assert loaded.tags["team"] == "explicit"
    assert loaded.tags["new"] == "$value"
    assert loaded.trace_metadata[TraceMetadataKey.TOKEN_USAGE] == '{"input_tokens": 20}'
    assert loaded.trace_metadata[TRACE_COST_METADATA] == '{"total_cost": 0.5}'
    document = store._trace_repository._collection.find_one({"_id": info.trace_id})
    assert document["token_usage"] == {"input_tokens": 20}
    assert document["cost"] == {"total_cost": 0.5}


def test_batch_reads_restore_request_order_skip_missing_and_scope_experiments(
    store, trace_info_factory, span_factory
):
    first, second = [store.start_trace(trace_info_factory()) for _ in range(2)]
    other_experiment = store.create_experiment("other-traces")
    unrelated = store.start_trace(
        trace_info_factory(trace_location=TraceLocation.from_experiment_id(other_experiment))
    )
    for info in (first, second, unrelated):
        store.log_spans(info.experiment_id, [span_factory(trace_id=info.trace_id)])
    feedback = store.create_assessment(
        Feedback(name="quality", value=True, trace_id=first.trace_id)
    )
    requested = [second.trace_id, "tr-missing", first.trace_id, second.trace_id, unrelated.trace_id]

    infos = store.batch_get_trace_infos(requested, experiment_ids=[first.experiment_id])
    traces = store.batch_get_traces(requested, experiment_ids=[first.experiment_id])

    assert [info.trace_id for info in infos] == [second.trace_id, first.trace_id]
    assert [trace.info.trace_id for trace in traces] == [second.trace_id, first.trace_id]
    assert infos[1].assessments[0].assessment_id == feedback.assessment_id
    assert traces[1].info.assessments == infos[1].assessments
    assert len(traces[0].data.spans) == len(traces[1].data.spans) == 1


def test_batch_complete_reads_omit_empty_and_partially_exported_traces(
    store, trace_info_factory, span_factory
):
    empty = store.start_trace(trace_info_factory())
    partial = store.start_trace(
        trace_info_factory(trace_metadata={TraceMetadataKey.SIZE_STATS: '{"num_spans": 2}'})
    )
    complete = store.start_trace(trace_info_factory())
    for info in (partial, complete):
        store.log_spans(info.experiment_id, [span_factory(trace_id=info.trace_id)])

    result = store.batch_get_traces([empty.trace_id, partial.trace_id, complete.trace_id])

    assert [trace.info.trace_id for trace in result] == [complete.trace_id]
    assert len(store.get_trace(partial.trace_id, allow_partial=True).data.spans) == 1
    assert store.get_trace(empty.trace_id, allow_partial=True).data.spans == []


def test_trace_tags_and_links_are_idempotent_and_preserve_literal_values(store, trace):
    store.set_trace_tag(trace.trace_id, "team.name", "$risk")
    store.set_trace_tag(trace.trace_id, "team.name", "$risk")
    store.set_trace_tag(trace.trace_id, "remove", "value")
    store.delete_trace_tag(trace.trace_id, "remove")
    request = LinkTracesToRun(trace_ids=[trace.trace_id, "tr-missing"], run_id="run")
    store.link_traces_to_run(request.trace_ids, request.run_id)
    store.link_traces_to_run(request.trace_ids, request.run_id)
    refs = [{"name": "greeting", "version": "1"}]
    store._trace_repository.link_prompts(trace_id=trace.trace_id, prompt_versions=refs)
    store._trace_repository.link_prompts(trace_id=trace.trace_id, prompt_versions=refs)

    loaded = store.get_trace_info(trace.trace_id)
    document = store._trace_repository._collection.find_one({"_id": trace.trace_id})

    assert loaded.tags["team.name"] == "$risk"
    assert "remove" not in loaded.tags
    assert document["run_ids"] == ["run"]
    assert document["linked_prompts"] == refs
    assert len([tag for tag in document["tags"] if tag["k"] == "team.name"]) == 1
    assert list(request.trace_ids) == [trace.trace_id, "tr-missing"]
    assert store._trace_repository._collection.count_documents({"_id": "tr-missing"}) == 0


def test_setting_tag_on_missing_trace_returns_not_found_without_creating_trace(store):
    trace_id = "tr-missing"

    with pytest.raises(MlflowException) as caught:
        store.set_trace_tag(trace_id, "team", "$risk")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert caught.value.message == f"Trace with ID '{trace_id}' not found."
    assert caught.value.__cause__ is None
    assert store._trace_repository._collection.count_documents({"_id": trace_id}) == 0


def test_linking_prompts_to_missing_trace_returns_not_found_without_creating_trace(store):
    trace_id = "tr-missing"
    prompt = PromptVersion(name="greeting", version=1, template="Hello")

    with pytest.raises(MlflowException) as caught:
        store.link_prompts_to_trace(trace_id, [prompt])

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert caught.value.message == f"Trace with ID '{trace_id}' not found."
    assert caught.value.__cause__ is None
    assert store._trace_repository._collection.count_documents({"_id": trace_id}) == 0


@pytest.mark.parametrize("target", ["missing_trace", "existing_trace"])
def test_deleting_missing_trace_tag_returns_not_found_and_preserves_existing_trace(
    store, trace, target
):
    trace_id = {"missing_trace": "tr-missing", "existing_trace": trace.trace_id}[target]

    with pytest.raises(MlflowException) as caught:
        store.delete_trace_tag(trace_id, "missing-tag")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert caught.value.message == f"Trace '{trace_id}' or tag 'missing-tag' not found."
    assert caught.value.__cause__ is None
    assert store.get_trace_info(trace.trace_id) == trace
    assert store._trace_repository._collection.count_documents({"_id": "tr-missing"}) == 0


def test_async_span_ingestion_persists_readable_trace(store, trace, span_factory):
    span = span_factory(trace_id=trace.trace_id)

    result = asyncio.run(store.log_spans_async(trace.experiment_id, [span]))

    assert result == [span]
    assert store.get_trace(trace.trace_id).data.spans[0].span_id == span.span_id


@pytest.mark.parametrize("max_traces,remaining", [(None, [3000]), (1, [2000, 3000])])
def test_timestamp_deletion_cascades_to_children_and_respects_count_limit(
    store, trace_info_factory, span_factory, max_traces, remaining
):
    infos = [
        store.start_trace(trace_info_factory(request_time=value)) for value in (1000, 2000, 3000)
    ]
    for info in infos:
        store.log_spans(info.experiment_id, [span_factory(trace_id=info.trace_id)])
        store.create_assessment(Feedback(name="quality", value=True, trace_id=info.trace_id))

    count = store.delete_traces(
        infos[0].experiment_id, max_timestamp_millis=2000, max_traces=max_traces
    )

    assert count == 3 - len(remaining)
    kept = store.batch_get_trace_infos([info.trace_id for info in infos])
    assert [info.request_time for info in kept] == remaining
    repository = store._trace_repository
    assert repository._spans_collection.count_documents({}) == len(remaining)
    assert repository._assessments_collection.count_documents({}) == len(remaining)
    assert store.delete_traces(infos[0].experiment_id, trace_ids=["tr-missing"]) == 0


def test_id_deletion_scopes_ownership_and_cascades_only_selected_trace(
    store, trace_info_factory, span_factory
):
    selected = store.start_trace(trace_info_factory())
    other_experiment = store.create_experiment("other-delete")
    other = store.start_trace(
        trace_info_factory(trace_location=TraceLocation.from_experiment_id(other_experiment))
    )
    for info in (selected, other):
        store.log_spans(info.experiment_id, [span_factory(trace_id=info.trace_id)])
        store.create_assessment(Feedback(name="quality", value=True, trace_id=info.trace_id))

    request = DeleteTraces(
        experiment_id=selected.experiment_id,
        request_ids=[selected.trace_id, other.trace_id],
    )

    assert store.delete_traces(request.experiment_id, trace_ids=request.request_ids) == 1

    assert store.get_trace(other.trace_id).info.assessments
    with pytest.raises(MlflowException) as caught:
        store.get_trace_info(selected.trace_id)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    repository = store._trace_repository
    assert repository._spans_collection.count_documents({"trace_id": selected.trace_id}) == 0
    assert repository._assessments_collection.count_documents({"trace_id": selected.trace_id}) == 0
    assert repository._collection.count_documents({"_id": other.trace_id}) == 1
    assert repository._spans_collection.count_documents({"trace_id": other.trace_id}) == 1
    assert repository._assessments_collection.count_documents({"trace_id": other.trace_id}) == 1
    assert list(request.request_ids) == [selected.trace_id, other.trace_id]
