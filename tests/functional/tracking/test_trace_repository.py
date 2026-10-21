"""Trace document updates and successful cursor execution against MongoDB."""

from copy import deepcopy

import pytest
from mlflow.tracing.constant import TraceMetadataKey
from pymongo.errors import BulkWriteError

from mlflow_mongodb.infrastructure.errors import (
    RepositoryPersistenceError,
    RepositoryWriteConflictError,
)
from mlflow_mongodb.tracking.types import SpanSummaryRecord


@pytest.fixture
def repository(store):
    return store._trace_repository


@pytest.fixture
def start_trace_arguments(trace_info_factory):
    info = trace_info_factory()
    return {
        "trace_id": info.trace_id,
        "experiment_id": info.experiment_id,
        "request_time": info.request_time,
        "state": "OK",
        "execution_duration": 10,
        "client_request_id": "$client",
        "request_preview": None,
        "response_preview": None,
        "tags": {"team.name": "$risk"},
        "trace_metadata": {"source": "$run"},
        "assessments": [],
    }


def test_start_trace_persists_literal_values_and_returns_joined_assessments(
    repository, start_trace_arguments
):
    assessment = {"assessment_id": "a-one", "expectation": {"value": {"answer": [1, True]}}}
    start_trace_arguments["assessments"] = [assessment]
    original = deepcopy(start_trace_arguments)

    record = repository.start_trace(**start_trace_arguments)

    document = repository._collection.find_one({"_id": record.trace_id})
    assert record.trace_id == start_trace_arguments["trace_id"]
    assert record.experiment_id == start_trace_arguments["experiment_id"]
    assert record.request_time == start_trace_arguments["request_time"]
    assert record.state == "OK"
    assert record.execution_duration == 10
    assert record.client_request_id == "$client"
    assert document["tags"] == [{"k": "team.name", "v": "$risk"}]
    assert document["trace_metadata"] == [{"k": "source", "v": "$run"}]
    assert document["trace_info_finalized"] is True
    assert document["authoritative_metadata_keys"] == ["source"]
    assert record.assessments == ({**assessment, "trace_id": record.trace_id},)
    assert start_trace_arguments == original


def test_start_trace_merges_kv_records_without_dropping_unrelated_keys(
    repository, start_trace_arguments
):
    repository.start_trace(
        **{
            **start_trace_arguments,
            "tags": {"preserve": "yes", "team.name": "old"},
            "trace_metadata": {"preserve": "yes", "source": "old"},
        }
    )

    record = repository.start_trace(**start_trace_arguments)

    assert {item.key: item.value for item in record.tags} == {
        "preserve": "yes",
        "team.name": "$risk",
    }
    assert {item.key: item.value for item in record.trace_metadata} == {
        "preserve": "yes",
        "source": "$run",
    }
    document = repository._collection.find_one({"_id": record.trace_id})
    assert set(document["authoritative_metadata_keys"]) == {"preserve", "source"}
    assert len(document["tags"]) == len(document["trace_metadata"]) == 2


@pytest.mark.parametrize(
    "run_ids,expected",
    [
        (None, {"existing"}),
        ([], {"existing"}),
        (["run-b", "existing", "run-b"], {"existing", "run-b"}),
    ],
)
def test_start_trace_unions_run_ids_with_existing_links(
    repository, start_trace_arguments, run_ids, expected
):
    repository.start_trace(**start_trace_arguments, run_ids=["existing"])
    original = deepcopy(run_ids)

    record = repository.start_trace(**start_trace_arguments, run_ids=run_ids)

    assert set(record.run_ids) == expected
    assert len(record.run_ids) == len(expected)
    assert run_ids == original


def test_start_trace_omitted_stats_preserve_existing_native_values(
    repository, start_trace_arguments
):
    repository.start_trace(
        **start_trace_arguments,
        span_stats={"num_spans": 3},
        metrics={"token_usage": {"input_tokens": 4}, "cost": {"total_cost": 0.2}},
    )

    record = repository.start_trace(**start_trace_arguments)

    document = repository._collection.find_one({"_id": record.trace_id})
    assert record.span_stats == {"num_spans": 3}
    assert document["token_usage"] == {"input_tokens": 4}
    assert document["cost"] == {"total_cost": 0.2}


@pytest.mark.parametrize("stats", [{}, {"num_spans": 0}, {"num_spans": 3}])
def test_start_trace_explicit_stats_replace_existing_value(
    repository, start_trace_arguments, stats
):
    repository.start_trace(**start_trace_arguments, span_stats={"num_spans": 9})

    record = repository.start_trace(**start_trace_arguments, span_stats=stats)

    assert record.span_stats == stats


@pytest.mark.parametrize("field", ["token_usage", "cost"])
@pytest.mark.parametrize("value", [None, {}, {"total": 1.5}])
def test_start_trace_explicit_native_metric_replaces_existing_value(
    repository, start_trace_arguments, field, value
):
    repository.start_trace(**start_trace_arguments, metrics={field: {"old": 5}})

    record = repository.start_trace(**start_trace_arguments, metrics={field: value})

    document = repository._collection.find_one({"_id": record.trace_id})
    assert document[field] == value
    assert document["trace_metadata"] == [{"k": "source", "v": "$run"}]


@pytest.mark.parametrize("field", ["request_preview", "response_preview"])
@pytest.mark.parametrize("value,expected", [(None, "old"), ("", ""), ("$preview", "$preview")])
def test_start_trace_preview_update_preserves_or_replaces_stored_value(
    repository, start_trace_arguments, field, value, expected
):
    repository.start_trace(**{**start_trace_arguments, field: "old"})

    record = repository.start_trace(**{**start_trace_arguments, field: value})

    assert getattr(record, field) == expected


def test_ensure_traces_creates_placeholders_without_overwriting_existing_metadata(
    repository, start_trace_arguments
):
    record = repository.start_trace(**start_trace_arguments)
    before = repository._collection.find_one({"_id": record.trace_id})

    repository.ensure_traces(
        experiment_id=record.experiment_id, request_times={record.trace_id: 1, "tr-placeholder": 2}
    )

    assert repository._collection.find_one({"_id": record.trace_id}) == before
    placeholder = repository.get_trace_info("tr-placeholder")
    assert placeholder.experiment_id == record.experiment_id
    assert placeholder.request_time == 2
    assert placeholder.state == "IN_PROGRESS"
    assert placeholder.execution_duration is None
    assert placeholder.trace_metadata == ()
    document = repository._collection.find_one({"_id": placeholder.trace_id})
    assert document["trace_info_finalized"] is False
    assert all(set(tag) == {"k", "v"} for tag in document["tags"])


def test_ensure_traces_rejects_an_existing_trace_owned_by_another_experiment(
    repository, start_trace_arguments
):
    record = repository.start_trace(**start_trace_arguments)
    before = repository._collection.find_one({"_id": record.trace_id})

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.ensure_traces(
            experiment_id="other-experiment", request_times={record.trace_id: 1}
        )

    assert isinstance(caught.value.__cause__, BulkWriteError)
    assert caught.value.__cause__.details["writeErrors"][0]["code"] == 11000
    assert repository._collection.find_one({"_id": record.trace_id}) == before
    assert repository._collection.count_documents({}) == 1


def test_exported_assessments_replace_by_id_preserve_others_and_keep_native_values(
    repository, start_trace_arguments
):
    original = {"assessment_id": "a-one", "trace_id": "wrong", "feedback": {"value": False}}
    other = {"assessment_id": "a-other", "feedback": {"value": True}}
    replacement = {"assessment_id": "a-one", "feedback": {"value": {"answer": [1, True]}}}
    repository.start_trace(**{**start_trace_arguments, "assessments": [original, other]})
    before = deepcopy(replacement)

    record = repository.start_trace(
        **{**start_trace_arguments, "assessments": [original, replacement]}
    )

    assert record.assessments == (
        {**replacement, "trace_id": record.trace_id},
        {**other, "trace_id": record.trace_id},
    )
    assert repository._assessments_collection.count_documents({"trace_id": record.trace_id}) == 2
    assert replacement == before
    assert original["trace_id"] == "wrong"


def test_span_snapshot_returns_revision_and_sorted_projected_records(
    repository, trace, store, span_factory
):
    later = store._span_to_document(span_factory(trace_id=trace.trace_id, span_id=3))
    earlier = store._span_to_document(
        span_factory(
            trace_id=trace.trace_id,
            span_id=2,
            start_time=trace.request_time * 1_000_000 - 1_000_000,
        )
    )
    tied = store._span_to_document(span_factory(trace_id=trace.trace_id, span_id=1))
    unrelated = store._span_to_document(span_factory(trace_id="tr-unrelated", span_id=4))
    repository.log_spans([later, earlier, tied, unrelated])

    revision, records = repository.span_summary_snapshot(
        trace_id=trace.trace_id, experiment_id=trace.experiment_id
    )

    assert revision == 1
    assert records == [
        SpanSummaryRecord.from_document(document) for document in (earlier, tied, later)
    ]
    next_revision, repeated = repository.span_summary_snapshot(
        trace_id=trace.trace_id, experiment_id=trace.experiment_id
    )
    assert next_revision == revision + 1
    assert repeated == records


def test_span_redelivery_does_not_replace_first_stored_payload(
    repository, trace, store, span_factory
):
    first = store._span_to_document(span_factory(trace_id=trace.trace_id))
    changed = store._span_to_document(
        span_factory(trace_id=trace.trace_id, end_time=first["end_time_ns"] + 1_000_000)
    )
    repository.log_spans([first])

    repository.log_spans([changed])

    records = repository.get_spans(trace.trace_id)
    assert len(records) == 1
    assert records[0].content == first["content"]
    assert records[0].end_time_ns == first["end_time_ns"]


def test_summary_update_merges_missing_keys_and_respects_authoritative_metadata(
    repository, start_trace_arguments
):
    record = repository.start_trace(
        **{**start_trace_arguments, "request_preview": "explicit"},
        metrics={"token_usage": {"input_tokens": 10}},
    )
    revision, _ = repository.span_summary_snapshot(
        trace_id=record.trace_id, experiment_id=record.experiment_id
    )
    summary = {
        "state": "ERROR",
        "request_time": 1,
        "execution_duration": 99,
        "request_preview": "derived",
        "response_preview": "$response",
        "tags": {"team.name": "derived", "new": "$tag"},
        "metadata": {"source": "derived", "session": "$session"},
        "token_usage": {"input_tokens": 4},
        "cost": None,
        "aggregate_metadata": {TraceMetadataKey.TOKEN_USAGE: '{"input_tokens": 4}'},
    }

    repository.update_span_summary(
        trace_id=record.trace_id,
        experiment_id=record.experiment_id,
        revision=revision,
        summary=summary,
    )

    updated = repository.get_trace_info(record.trace_id)
    assert updated.state == "OK"
    assert updated.request_time == record.request_time
    assert updated.execution_duration == 10
    assert updated.request_preview == "explicit"
    assert updated.response_preview == "$response"
    assert {item.key: item.value for item in updated.tags} == {"team.name": "$risk", "new": "$tag"}
    assert {item.key: item.value for item in updated.trace_metadata} == {
        "source": "$run",
        "session": "$session",
        TraceMetadataKey.TOKEN_USAGE: '{"input_tokens": 4}',
    }


def test_stale_summary_revision_cannot_overwrite_newer_snapshot(repository, trace):
    revision, _ = repository.span_summary_snapshot(
        trace_id=trace.trace_id, experiment_id=trace.experiment_id
    )
    repository.span_summary_snapshot(trace_id=trace.trace_id, experiment_id=trace.experiment_id)
    summary = {
        "state": "ERROR",
        "tags": {},
        "metadata": {},
        "request_time": 1,
        "execution_duration": 2,
        "request_preview": None,
        "response_preview": None,
        "token_usage": None,
        "cost": None,
    }

    with pytest.raises(RepositoryWriteConflictError):
        repository.update_span_summary(
            trace_id=trace.trace_id,
            experiment_id=trace.experiment_id,
            revision=revision,
            summary=summary,
        )

    assert repository.get_trace_info(trace.trace_id).state == "OK"
