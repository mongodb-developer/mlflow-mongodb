"""Trace selection and child cleanup against MongoDB."""

import pytest
from mlflow.entities import TraceLocation
from mlflow.protos.service_pb2 import DeleteTraces


@pytest.fixture
def deletion_traces(store, persisted_trace_factory):
    # Reverse insertion order so a limited deletion must use time and ID ordering.
    traces = [
        persisted_trace_factory(trace_id=f"tr-{index:032x}", request_time=timestamp)[0]
        for index, timestamp in ((3, 200), (2, 100), (1, 100))
    ]
    other_experiment_id = store.create_experiment("other-deletion-experiment")
    other, _ = persisted_trace_factory(
        request_time=50,
        trace_location=TraceLocation.from_experiment_id(other_experiment_id),
    )
    return traces, other


@pytest.mark.parametrize(
    "cutoff,max_traces,deleted_indexes",
    [(99, None, []), (100, None, [1, 2]), (100, 1, [2]), (200, 2, [1, 2]), (200, None, [0, 1, 2])],
)
def test_delete_by_timestamp_scopes_experiment_orders_limits_and_removes_children(
    store, deletion_traces, cutoff, max_traces, deleted_indexes
):
    traces, other = deletion_traces
    experiment_id = traces[0].experiment_id

    count = store.delete_traces(experiment_id, max_timestamp_millis=cutoff, max_traces=max_traces)

    assert count == len(deleted_indexes)
    remaining_ids = {
        trace.trace_id for index, trace in enumerate(traces) if index not in deleted_indexes
    }
    remaining_ids.add(other.trace_id)
    repository = store._trace_repository
    assert {document["_id"] for document in repository._collection.find({})} == remaining_ids
    assert {
        document["trace_id"] for document in repository._spans_collection.find({})
    } == remaining_ids
    assert repository._spans_collection.count_documents({}) == 3 * len(remaining_ids)
    assert {
        document["trace_id"] for document in repository._assessments_collection.find({})
    } == remaining_ids
    assert repository._assessments_collection.count_documents({}) == len(remaining_ids)


def test_delete_by_protobuf_ids_deduplicates_scopes_experiment_and_removes_children(
    store, deletion_traces
):
    traces, other = deletion_traces
    request = DeleteTraces(
        experiment_id=traces[0].experiment_id,
        request_ids=[traces[1].trace_id, "tr-missing", other.trace_id, traces[1].trace_id],
    )
    original_ids = list(request.request_ids)

    count = store.delete_traces(request.experiment_id, trace_ids=request.request_ids)

    assert count == 1
    assert list(request.request_ids) == original_ids
    remaining_ids = {traces[0].trace_id, traces[2].trace_id, other.trace_id}
    repository = store._trace_repository
    assert {document["_id"] for document in repository._collection.find({})} == remaining_ids
    assert {
        document["trace_id"] for document in repository._spans_collection.find({})
    } == remaining_ids
    assert repository._spans_collection.count_documents({}) == 9
    assert {
        document["trace_id"] for document in repository._assessments_collection.find({})
    } == remaining_ids
    assert repository._assessments_collection.count_documents({}) == 3
    assert store.delete_traces(request.experiment_id, trace_ids=request.request_ids) == 0
