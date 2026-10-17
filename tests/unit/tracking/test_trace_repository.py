"""Trace write contracts, conflicts, and partial cursor failures."""

from unittest.mock import MagicMock, Mock, call

import pytest
from bson.errors import InvalidBSON, InvalidDocument
from pymongo import ReturnDocument
from pymongo.errors import BulkWriteError, DuplicateKeyError, OperationFailure

from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotFoundError,
    RepositoryPersistenceError,
    RepositoryWriteConflictError,
)
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.repositories import TraceRepository

TRACE_ID = "tr-example"


@pytest.fixture
def collections():
    return {name: MagicMock() for name in ("traces", "spans", "assessments")}


@pytest.fixture
def repository(collections):
    database = MagicMock()
    database.__getitem__.side_effect = collections.__getitem__
    return TraceRepository(database)


@pytest.fixture
def failing_cursor():
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.sort.return_value = cursor
    cursor.limit.return_value = cursor
    error = OperationFailure("Private cursor details")

    def iterate():
        yield {"_id": TRACE_ID}
        raise error

    cursor.__iter__.side_effect = iterate
    return cursor, error


@pytest.fixture
def start_trace_arguments():
    return {
        "trace_id": TRACE_ID,
        "experiment_id": "42",
        "request_time": 123,
        "state": "OK",
        "execution_duration": 10,
        "client_request_id": "$client",
        "request_preview": None,
        "response_preview": None,
        "tags": {"team.name": "$risk"},
        "trace_metadata": {"source": "$run"},
        "assessments": [{"assessment_id": "a-one", "feedback": {"value": True}}],
    }


@pytest.fixture
def start_trace_dependencies(repository, monkeypatch, trace_record_factory):
    dependencies = Mock()
    dependencies.get_trace_info.return_value = trace_record_factory(_id=TRACE_ID)
    monkeypatch.setattr(repository, "_upsert_trace", dependencies.upsert_trace)
    monkeypatch.setattr(repository, "_upsert_assessments", dependencies.upsert_assessments)
    monkeypatch.setattr(repository, "get_trace_info", dependencies.get_trace_info)
    return dependencies


@pytest.mark.parametrize(
    "stage,expected_calls",
    [
        ("upsert_trace", ["upsert_trace"]),
        ("upsert_assessments", ["upsert_trace", "upsert_assessments"]),
        ("get_trace_info", ["upsert_trace", "upsert_assessments", "get_trace_info"]),
    ],
)
def test_start_trace_propagates_failure_and_stops_before_subsequent_operations(
    repository, start_trace_arguments, start_trace_dependencies, stage, expected_calls
):
    error = RepositoryPersistenceError("Private database details")
    getattr(start_trace_dependencies, stage).side_effect = error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.start_trace(**start_trace_arguments)

    assert caught.value is error
    assert [entry[0] for entry in start_trace_dependencies.mock_calls] == expected_calls


def test_initialization_uses_configured_collections_and_unique_child_identities():
    collections = {
        name: MagicMock() for name in ("custom_traces", "custom_spans", "custom_assessments")
    }
    database = MagicMock()
    database.__getitem__.side_effect = collections.__getitem__
    settings = MongoDBSettings(
        traces_collection_name="custom_traces",
        spans_collection_name="custom_spans",
        assessments_collection_name="custom_assessments",
    )

    repository = TraceRepository(database, settings=settings)

    assert repository._settings is settings
    assert database.__getitem__.call_args_list == [call(name) for name in collections]
    collections["custom_spans"].create_index.assert_any_call(
        [("trace_id", 1), ("span_id", 1)], unique=True, name="spans_trace_span_unique"
    )
    collections["custom_assessments"].create_index.assert_any_call(
        [("trace_id", 1), ("assessment_id", 1)],
        unique=True,
        name="assessments_trace_assessment_unique",
    )


@pytest.mark.parametrize("collection_name", ["traces", "spans", "assessments"])
def test_index_initialization_failures_preserve_backend_cause(collections, collection_name):
    error = OperationFailure("Private index details")
    collections[collection_name].create_index.side_effect = error
    database = MagicMock()
    database.__getitem__.side_effect = collections.__getitem__
    with pytest.raises(RepositoryPersistenceError) as caught:
        TraceRepository(database)
    assert str(caught.value) == "Unable to initialize trace repository."
    assert caught.value.__cause__ is error


def test_trace_upsert_retries_duplicate_key_with_ownership_filter(repository, collections):
    collection = collections["traces"]
    collection.find_one_and_update.side_effect = [DuplicateKeyError("race"), {"_id": TRACE_ID}]
    update = [{"$set": {"state": {"$literal": "OK"}}}]

    assert repository._upsert_trace(TRACE_ID, "42", update) == {"_id": TRACE_ID}
    assert (
        collection.find_one_and_update.call_args_list
        == [
            call(
                {"_id": TRACE_ID, "experiment_id": "42"},
                update,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        ]
        * 2
    )
    collection.find_one.assert_not_called()


def test_trace_upsert_stops_after_three_identity_conflicts(repository, collections):
    error = DuplicateKeyError("Private identity details")
    collections["traces"].find_one_and_update.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository._upsert_trace(TRACE_ID, "42", {"$set": {"state": "OK"}})
    assert collections["traces"].find_one_and_update.call_count == 3
    assert isinstance(caught.value.__cause__, RepositoryWriteConflictError)
    assert caught.value.__cause__.__cause__ is error


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_trace_upsert_does_not_retry_other_database_errors(repository, collections, error_type):
    error = error_type("Private write details")
    collections["traces"].find_one_and_update.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository._upsert_trace(TRACE_ID, "42", {})
    assert caught.value.__cause__ is error
    assert collections["traces"].find_one_and_update.call_count == 1


def test_ensure_traces_empty_request_times_skips_bulk_write(repository, collections):
    assert repository.ensure_traces(experiment_id="42", request_times={}) is None
    collections["traces"].bulk_write.assert_not_called()


def test_ensure_traces_retries_only_failed_operations_using_each_batch_indexes(
    repository, collections
):
    collection = collections["traces"]
    first_error = BulkWriteError(
        {
            "writeErrors": [
                {"index": 1, "code": 11000, "keyPattern": {"_id": 1}},
                {"index": 3, "code": 11000, "keyPattern": {"_id": 1}},
            ]
        }
    )
    second_error = BulkWriteError(
        {
            "writeErrors": [
                {"index": 1, "code": 11000, "keyPattern": {"_id": 1}},
            ]
        }
    )
    collection.bulk_write.side_effect = [first_error, second_error, Mock()]
    request_times = {f"tr-{index}": index for index in range(4)}

    repository.ensure_traces(experiment_id="42", request_times=request_times)

    calls = collection.bulk_write.call_args_list
    assert len(calls) == 3
    initial_operations = calls[0].args[0]
    assert len(initial_operations) == 4
    assert calls[1].args[0] == [initial_operations[1], initial_operations[3]]
    assert calls[2].args[0] == [initial_operations[3]]
    assert all(entry.kwargs == {"ordered": False} for entry in calls)
    assert request_times == {f"tr-{index}": index for index in range(4)}


def test_ensure_traces_stops_after_three_duplicate_id_failures(repository, collections):
    error = BulkWriteError(
        {
            "writeErrors": [
                {"index": 0, "code": 11000, "keyPattern": {"_id": 1}},
            ]
        }
    )
    collections["traces"].bulk_write.side_effect = error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.ensure_traces(experiment_id="42", request_times={TRACE_ID: 1})

    assert collections["traces"].bulk_write.call_count == 3
    assert str(caught.value) == "Unable to create trace placeholders."
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "details",
    [
        pytest.param({"writeErrors": []}, id="no-write-errors"),
        pytest.param({"writeErrors": [{"index": 0, "code": 121}]}, id="validation-error"),
        pytest.param(
            {
                "writeErrors": [
                    {"index": 0, "code": 11000, "keyPattern": {"name": 1}},
                ]
            },
            id="different-unique-index",
        ),
        pytest.param({"writeErrors": [{"index": 0, "code": 11000}]}, id="unknown-index"),
        pytest.param(
            {
                "writeErrors": [
                    {"index": 0, "code": 11000, "keyPattern": {"_id": 1}},
                    {"index": 1, "code": 121},
                ]
            },
            id="mixed-write-errors",
        ),
        pytest.param(
            {
                "writeErrors": [
                    {"index": 0, "code": 11000, "keyPattern": {"_id": 1}},
                ],
                "writeConcernErrors": [{"code": 64, "errmsg": "Private write concern details"}],
            },
            id="write-concern-error",
        ),
    ],
)
def test_ensure_traces_does_not_retry_non_identity_bulk_failures(repository, collections, details):
    error = BulkWriteError(details)
    collections["traces"].bulk_write.side_effect = error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.ensure_traces(experiment_id="42", request_times={TRACE_ID: 1, "tr-other": 2})

    assert collections["traces"].bulk_write.call_count == 1
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_ensure_traces_does_not_retry_driver_or_encoding_failures(
    repository, collections, error_type
):
    error = error_type("Private placeholder details")
    collections["traces"].bulk_write.side_effect = error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.ensure_traces(experiment_id="42", request_times={TRACE_ID: 1})

    assert collections["traces"].bulk_write.call_count == 1
    assert caught.value.__cause__ is error


def test_ensure_traces_stops_if_a_retry_has_a_different_failure(repository, collections):
    duplicate = BulkWriteError(
        {
            "writeErrors": [
                {"index": 1, "code": 11000, "keyPattern": {"_id": 1}},
            ]
        }
    )
    failure = OperationFailure("Private retry details")
    collection = collections["traces"]
    collection.bulk_write.side_effect = [duplicate, failure]

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.ensure_traces(experiment_id="42", request_times={TRACE_ID: 1, "tr-other": 2})

    assert collection.bulk_write.call_count == 2
    assert caught.value.__cause__ is failure
    calls = collection.bulk_write.call_args_list
    assert calls[1].args[0] == [calls[0].args[0][1]]


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument, OverflowError])
def test_span_write_translates_encoding_and_driver_errors(repository, collections, error_type):
    error = error_type("Private span details")
    collections["spans"].bulk_write.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.log_spans([{"trace_id": TRACE_ID, "span_id": "span"}])
    assert caught.value.__cause__ is error


def test_empty_trace_run_links_skip_database_update(repository, collections):
    assert repository.link_traces_to_run(trace_ids=[], run_id="run") is None
    collections["traces"].update_many.assert_not_called()


def test_empty_prompt_links_skip_database_update(repository, collections):
    assert repository.link_prompts(trace_id=TRACE_ID, prompt_versions=[]) is None
    collections["traces"].find_one_and_update.assert_not_called()


def test_empty_span_write_does_not_call_database(repository, collections):
    repository.log_spans([])
    collections["spans"].bulk_write.assert_not_called()


def test_snapshot_missing_trace_does_not_read_spans(repository, collections):
    collections["traces"].find_one_and_update.return_value = None
    with pytest.raises(RepositoryNotFoundError):
        repository.span_summary_snapshot(trace_id=TRACE_ID, experiment_id="42")
    collections["spans"].find.assert_not_called()


def test_revision_conflict_is_distinct_from_persistence_failure(repository, collections):
    collections["traces"].update_one.return_value.matched_count = 0
    summary = {
        "state": "OK",
        "tags": {},
        "metadata": {},
        "request_time": 1,
        "execution_duration": 2,
        "request_preview": None,
        "response_preview": None,
        "token_usage": None,
        "cost": None,
    }
    with pytest.raises(RepositoryWriteConflictError) as caught:
        repository.update_span_summary(
            trace_id=TRACE_ID, experiment_id="42", revision=3, summary=summary
        )
    assert caught.value.args == (TRACE_ID,)
    assert collections["traces"].update_one.call_args.args[0] == {
        "_id": TRACE_ID,
        "experiment_id": "42",
        "span_revision": 3,
    }


def test_partial_batch_read_failure_closes_cursor_and_preserves_cause(
    repository, collections, failing_cursor
):
    cursor, error = failing_cursor
    collections["traces"].aggregate.return_value = cursor
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.batch_get_traces([TRACE_ID])
    assert caught.value.__cause__ is error
    assert cursor.__exit__.call_count == 1


def test_partial_metadata_read_failure_closes_cursor_and_preserves_cause(
    repository, collections, failing_cursor
):
    cursor, error = failing_cursor
    collections["traces"].aggregate.return_value = cursor
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.batch_get_trace_infos([TRACE_ID])
    assert caught.value.__cause__ is error
    assert cursor.__exit__.call_count == 1


def test_partial_span_read_failure_closes_cursor_and_preserves_cause(
    repository, collections, failing_cursor
):
    cursor, error = failing_cursor
    collections["spans"].find.return_value = cursor
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.get_spans(TRACE_ID)
    assert caught.value.__cause__ is error
    assert cursor.__exit__.call_count == 1


def test_partial_snapshot_failure_closes_cursor_and_preserves_cause(
    repository, collections, failing_cursor
):
    cursor, error = failing_cursor
    collections["traces"].find_one_and_update.return_value = {"span_revision": 1}
    collections["spans"].find.return_value = cursor
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.span_summary_snapshot(trace_id=TRACE_ID, experiment_id="42")
    assert caught.value.__cause__ is error
    assert cursor.__exit__.call_count == 1


def test_partial_deletion_selection_failure_closes_cursor_before_any_deletes(
    repository, collections, failing_cursor
):
    cursor, error = failing_cursor
    collections["traces"].find.return_value = cursor
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.delete_traces(experiment_id="42", max_timestamp_millis=123)
    assert caught.value.__cause__ is error
    assert cursor.__exit__.call_count == 1
    for collection in collections.values():
        collection.delete_many.assert_not_called()


@pytest.mark.parametrize(
    "failing_collection,expected_calls",
    [("spans", (1, 0, 0)), ("assessments", (1, 1, 0)), ("traces", (1, 1, 1))],
)
def test_deletion_failure_stops_cascade_without_deleting_metadata_early(
    repository, collections, failing_collection, expected_calls
):
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.sort.return_value = cursor
    cursor.__iter__.return_value = iter([{"_id": TRACE_ID}])
    collections["traces"].find.return_value = cursor
    error = OperationFailure("Private delete details")
    collections[failing_collection].delete_many.side_effect = error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.delete_traces(experiment_id="42", trace_ids=[TRACE_ID])

    assert caught.value.__cause__ is error
    assert (
        tuple(
            collections[name].delete_many.call_count for name in ("spans", "assessments", "traces")
        )
        == expected_calls
    )


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidBSON])
def test_assessment_read_translates_driver_and_decode_errors(repository, collections, error_type):
    error = error_type("Private read details")
    collections["assessments"].find_one.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.get_assessment(trace_id=TRACE_ID, assessment_id="a-one")
    assert caught.value.__cause__ is error


def test_assessment_missing_is_a_domain_error(repository, collections):
    collections["assessments"].find_one.return_value = None
    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.get_assessment(trace_id=TRACE_ID, assessment_id="a-one")
    assert caught.value.args == ("a-one",)


def test_malformed_assessment_response_is_not_a_database_error(repository, collections):
    collections["assessments"].find_one.return_value = {"malformed": True}
    with pytest.raises(KeyError, match="content"):
        repository.get_assessment(trace_id=TRACE_ID, assessment_id="a-one")


def test_empty_assessments_skip_bulk_write(repository, collections):
    assert (
        repository._upsert_assessments(trace_id=TRACE_ID, experiment_id="42", assessments=[])
        is None
    )
    collections["assessments"].bulk_write.assert_not_called()


def test_duplicate_assessment_does_not_invalidate_original(repository, collections):
    error = DuplicateKeyError("Private duplicate details")
    collections["assessments"].insert_one.side_effect = error
    with pytest.raises(RepositoryAlreadyExistsError) as caught:
        repository.create_assessment(
            trace_id=TRACE_ID,
            experiment_id="42",
            content={"assessment_id": "a-new", "overrides": "a-original"},
        )
    assert caught.value.__cause__ is error
    collections["assessments"].update_one.assert_not_called()


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument, OverflowError])
def test_assessment_create_translates_write_errors(repository, collections, error_type):
    error = error_type("Private assessment details")
    collections["assessments"].insert_one.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.create_assessment(
            trace_id=TRACE_ID, experiment_id="42", content={"assessment_id": "a-new"}
        )
    assert caught.value.__cause__ is error


def test_override_target_disappearing_after_insert_is_not_found(repository, collections):
    collections["assessments"].update_one.return_value.matched_count = 0
    with pytest.raises(RepositoryNotFoundError) as caught:
        repository.create_assessment(
            trace_id=TRACE_ID,
            experiment_id="42",
            content={"assessment_id": "a-new", "overrides": "a-original"},
        )
    assert caught.value.args == ("a-original",)
    assert collections["assessments"].insert_one.call_count == 1


def test_override_invalidation_failure_is_translated_without_claiming_rollback(
    repository, collections
):
    error = OperationFailure("Private update details")
    collection = collections["assessments"]
    collection.update_one.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.create_assessment(
            trace_id=TRACE_ID,
            experiment_id="42",
            content={"assessment_id": "a-new", "overrides": "a-original"},
        )
    assert caught.value.__cause__ is error
    assert collection.insert_one.call_count == 1
    collection.delete_one.assert_not_called()


def test_update_missing_assessment_is_not_found(repository, collections):
    collections["assessments"].find_one_and_update.return_value = None
    with pytest.raises(RepositoryNotFoundError):
        repository.update_assessment(
            trace_id=TRACE_ID, assessment_id="a-one", updates={"rationale": "new"}
        )


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument, OverflowError])
def test_assessment_update_translates_write_errors(repository, collections, error_type):
    error = error_type("Private update details")
    collections["assessments"].find_one_and_update.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.update_assessment(
            trace_id=TRACE_ID, assessment_id="a-one", updates={"rationale": "new"}
        )
    assert caught.value.__cause__ is error


def test_delete_missing_assessment_is_idempotent(repository, collections):
    collection = collections["assessments"]
    collection.find_one.return_value = None
    assert repository.delete_assessment(trace_id=TRACE_ID, assessment_id="a-one") is None
    collection.update_one.assert_not_called()
    collection.delete_one.assert_not_called()


def test_failed_override_restoration_does_not_delete_assessment(repository, collections):
    collection = collections["assessments"]
    collection.find_one.return_value = {"content": {"overrides": "a-original"}}
    error = OperationFailure("Private restoration details")
    collection.update_one.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.delete_assessment(trace_id=TRACE_ID, assessment_id="a-new")
    assert caught.value.__cause__ is error
    collection.delete_one.assert_not_called()


def test_delete_failure_after_restoration_is_translated(repository, collections):
    collection = collections["assessments"]
    collection.find_one.return_value = {"content": {"overrides": "a-original"}}
    error = OperationFailure("Private delete details")
    collection.delete_one.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.delete_assessment(trace_id=TRACE_ID, assessment_id="a-new")
    assert caught.value.__cause__ is error
    assert collection.update_one.call_count == 1
