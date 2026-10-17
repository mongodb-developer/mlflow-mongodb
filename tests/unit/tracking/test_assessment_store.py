"""Assessment conversion, validation order, mutations, and public errors."""

from copy import deepcopy

import pytest
from mlflow.entities.assessment import (
    AssessmentError,
    Expectation,
    ExpectationValue,
    FeedbackValue,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotFoundError,
    RepositoryPersistenceError,
)
from mlflow_mongodb.tracking.store import MongoDBTrackingStore

try:
    from mlflow.entities.assessment import IssueReference
except ImportError:
    IssueReference = None


TRACE_ID = "tr-0000000000000000000000000000002a"
TIMESTAMP = 1_700_000_000_000


@pytest.fixture
def assessment_document(trace_store, feedback_factory):
    return trace_store._assessment_to_document(feedback_factory())


@pytest.fixture
def assessment_store(trace_store, trace_repository, assessment_document):
    trace_repository.get_assessment.return_value = assessment_document
    trace_repository.create_assessment.side_effect = lambda **kwargs: kwargs["content"]
    trace_repository.update_assessment.return_value = assessment_document
    return trace_store


@pytest.mark.parametrize("value", [True, 3, 0.5, "yes", [1, "a"], {"answer": [1, True]}])
def test_feedback_values_remain_native(feedback_factory, value):
    assessment = feedback_factory(value=value)
    document = MongoDBTrackingStore._assessment_to_document(assessment)
    result = MongoDBTrackingStore._assessment_from_document(document)

    assert document["feedback"]["value"] == value
    assert type(document["feedback"]["value"]) is type(value)
    assert result.value == value


@pytest.mark.parametrize("value", [[1, "a"], {"answer": [1, True]}])
def test_feedback_document_copies_mutable_values(feedback_factory, value):
    assessment = feedback_factory(value=value)
    document = MongoDBTrackingStore._assessment_to_document(assessment)
    assert document["feedback"]["value"] == value
    assert document["feedback"]["value"] is not assessment.value


@pytest.mark.parametrize("value", [[1, {"nested": True}], {"answer": "expected"}])
def test_structured_expectation_is_native_and_copied(value):
    assessment = Expectation(name="expected", value=value, trace_id=TRACE_ID)
    document = MongoDBTrackingStore._assessment_to_document(assessment)
    assert document["expectation"] == {"value": value}
    assert document["expectation"]["value"] is not assessment.value
    assert MongoDBTrackingStore._assessment_from_document(document).value == value


def test_expectation_conversion_preserves_validity_override_and_run_link():
    assessment = Expectation(name="expected", value={"answer": 1}, trace_id=TRACE_ID)
    assessment.valid = False
    assessment.overrides = "a-previous"
    assessment.run_id = "run"
    result = MongoDBTrackingStore._assessment_from_document(
        MongoDBTrackingStore._assessment_to_document(assessment)
    )
    assert result.valid is False
    assert result.overrides == "a-previous"
    assert result.run_id == "run"


@pytest.mark.skipif(IssueReference is None, reason="MLflow has no IssueReference entity")
def test_issue_reference_conversion_preserves_native_content_and_run_link():
    assessment = IssueReference(
        issue_id="issue-1",
        issue_name="Incorrect answer",
        trace_id=TRACE_ID,
        run_id="run",
        rationale="Needs review",
    )
    document = MongoDBTrackingStore._assessment_to_document(assessment)
    result = MongoDBTrackingStore._assessment_from_document(document)
    assert document["issue"] == {"issue_name": "Incorrect answer"}
    assert result.name == "issue-1"
    assert result.run_id == "run"
    assert result.rationale == "Needs review"


def test_create_generates_identity_and_timestamps_without_mutating_payload(
    assessment_store, trace_repository, feedback_factory, monkeypatch
):
    monkeypatch.setattr(
        "mlflow_mongodb.tracking.store.generate_assessment_id", lambda: "a-generated"
    )
    assessment = feedback_factory(assessment_id=None, valid=False)
    assessment.create_time_ms = 0
    assessment.last_update_time_ms = 0
    original_metadata = deepcopy(assessment.metadata)

    result = assessment_store.create_assessment(assessment)

    assert result.assessment_id == "a-generated"
    assert result.valid is True
    assert result.create_time_ms == TIMESTAMP
    assert result.last_update_time_ms == TIMESTAMP
    assert assessment.assessment_id == "a-generated"
    assert assessment.valid is False
    assert assessment.create_time_ms == 0
    assert assessment.metadata == original_metadata
    arguments = trace_repository.create_assessment.call_args.kwargs
    assert arguments["trace_id"] == TRACE_ID
    assert arguments["experiment_id"] == "42"
    assert arguments["content"]["feedback"]["value"] is True


def test_create_preserves_supplied_identity_timestamps_and_run_link(
    assessment_store, feedback_factory
):
    assessment = feedback_factory()
    assessment.run_id = "run"
    result = assessment_store.create_assessment(assessment)
    assert result.assessment_id == assessment.assessment_id
    assert result.create_time_ms == assessment.create_time_ms
    assert result.last_update_time_ms == assessment.last_update_time_ms
    assert result.run_id == "run"


def test_create_checks_override_in_same_trace_before_write(
    assessment_store, trace_repository, feedback_factory
):
    assessment = feedback_factory(assessment_id="a-override", overrides="a-original")
    result = assessment_store.create_assessment(assessment)
    assert result.overrides == "a-original"
    trace_repository.get_assessment.assert_called_once_with(
        trace_id=TRACE_ID, assessment_id="a-original"
    )
    assert [entry[0] for entry in trace_repository.method_calls] == [
        "get_trace_info",
        "get_assessment",
        "create_assessment",
    ]


def test_create_missing_trace_fails_before_generating_id_or_checking_override(
    assessment_store, trace_repository, feedback_factory
):
    trace_repository.get_trace_info.side_effect = RepositoryNotFoundError("private")
    assessment = feedback_factory(assessment_id=None, overrides="a-other")
    with pytest.raises(MlflowException) as caught:
        assessment_store.create_assessment(assessment)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert assessment.assessment_id is None
    trace_repository.get_assessment.assert_not_called()
    trace_repository.create_assessment.assert_not_called()


def test_create_missing_override_fails_before_write(
    assessment_store, trace_repository, feedback_factory
):
    trace_repository.get_assessment.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException, match="Assessment with ID 'a-other'") as caught:
        assessment_store.create_assessment(feedback_factory(overrides="a-other"))
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    trace_repository.create_assessment.assert_not_called()


def test_create_override_disappearing_during_write_rechecks_trace_and_reports_missing_assessment(
    assessment_store, trace_repository, feedback_factory, caplog
):
    error = RepositoryNotFoundError("Private repository details")
    trace_repository.create_assessment.side_effect = error
    assessment = feedback_factory(assessment_id="a-override", overrides="a-original")

    with pytest.raises(MlflowException) as caught:
        assessment_store.create_assessment(assessment)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert caught.value.message == (
        f"Assessment with ID 'a-original' not found for trace '{TRACE_ID}'"
    )
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    trace_repository.get_assessment.assert_called_once_with(
        trace_id=TRACE_ID, assessment_id="a-original"
    )
    assert [entry[0] for entry in trace_repository.method_calls] == [
        "get_trace_info",
        "get_assessment",
        "create_assessment",
        "get_trace_info",
    ]
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


def test_duplicate_identity_is_sanitized_and_logged(
    assessment_store, trace_repository, feedback_factory, caplog
):
    error = RepositoryAlreadyExistsError("private")
    trace_repository.create_assessment.side_effect = error
    with pytest.raises(MlflowException) as caught:
        assessment_store.create_assessment(feedback_factory())
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.message == "Unable to create assessment due to an identity conflict."
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


def test_get_existing_assessment_does_not_reload_trace(assessment_store, trace_repository):
    result = assessment_store.get_assessment(TRACE_ID, "a-original")
    assert result.assessment_id == "a-original"
    trace_repository.get_trace_info.assert_not_called()


def test_get_missing_assessment_distinguishes_existing_trace(assessment_store, trace_repository):
    trace_repository.get_assessment.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException, match="Assessment with ID 'a-missing'") as caught:
        assessment_store.get_assessment(TRACE_ID, "a-missing")
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert caught.value.__cause__ is None
    trace_repository.get_trace_info.assert_called_once_with(TRACE_ID)


def test_get_missing_assessment_reports_missing_trace_first(assessment_store, trace_repository):
    trace_repository.get_assessment.side_effect = RepositoryNotFoundError("private")
    trace_repository.get_trace_info.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException, match=f"Trace with ID '{TRACE_ID}'") as caught:
        assessment_store.get_assessment(TRACE_ID, "a-missing")
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.parametrize(
    "updates",
    [
        {"expectation": ExpectationValue(1), "feedback": FeedbackValue(True)},
        {"expectation": ExpectationValue(1)},
        {"feedback": "invalid"},
    ],
)
def test_update_rejects_invalid_feedback_mutation_before_write(
    assessment_store, trace_repository, updates
):
    with pytest.raises(MlflowException) as caught:
        assessment_store.update_assessment(TRACE_ID, "a-original", **updates)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    trace_repository.update_assessment.assert_not_called()


@pytest.mark.parametrize("updates", [{"feedback": FeedbackValue(True)}, {"expectation": "invalid"}])
def test_update_rejects_invalid_expectation_mutation_before_write(
    assessment_store, trace_repository, updates
):
    expectation = Expectation(name="expected", value="answer", trace_id=TRACE_ID)
    trace_repository.get_assessment.return_value = assessment_store._assessment_to_document(
        expectation
    )
    with pytest.raises(MlflowException) as caught:
        assessment_store.update_assessment(TRACE_ID, "a-original", **updates)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    trace_repository.update_assessment.assert_not_called()


def test_update_checks_existence_before_value_validation(assessment_store, trace_repository):
    trace_repository.get_assessment.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException) as caught:
        assessment_store.update_assessment(
            TRACE_ID, "a-missing", expectation=ExpectationValue(1), feedback=FeedbackValue(True)
        )
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    trace_repository.update_assessment.assert_not_called()


def test_update_passes_only_mutable_fields_and_incoming_metadata(
    assessment_store, trace_repository, assessment_document
):
    original = deepcopy(assessment_document)
    assessment_store.update_assessment(
        TRACE_ID,
        "a-original",
        name="",
        rationale="",
        metadata={"new": "value"},
        feedback=FeedbackValue(False),
    )
    updates = trace_repository.update_assessment.call_args.kwargs["updates"]
    assert set(updates) == {
        "assessment_name",
        "rationale",
        "metadata",
        "feedback",
        "error",
        "last_update_time",
    }
    assert isinstance(updates["assessment_name"], str) and not updates["assessment_name"]
    assert isinstance(updates["rationale"], str) and not updates["rationale"]
    assert updates["metadata"] == {"new": "value"}
    assert updates["feedback"]["value"] is False
    assert updates["error"] is None
    assert updates["last_update_time"] != original["last_update_time"]
    assert assessment_document == original


def test_update_without_optional_fields_only_updates_timestamp(assessment_store, trace_repository):
    assessment_store.update_assessment(TRACE_ID, "a-original")
    assert set(trace_repository.update_assessment.call_args.kwargs["updates"]) == {
        "last_update_time"
    }


def test_update_feedback_error_is_preserved(assessment_store, trace_repository):
    error = AssessmentError(error_code="JUDGE_ERROR", error_message="judge failed")
    assessment_store.update_assessment(TRACE_ID, "a-original", feedback=FeedbackValue(None, error))
    feedback = trace_repository.update_assessment.call_args.kwargs["updates"]["feedback"]
    assert feedback["value"] is None
    assert feedback["error"]["error_code"] == "JUDGE_ERROR"
    assert feedback["error"]["error_message"] == "judge failed"


def test_update_feedback_can_clear_existing_error(
    assessment_store, trace_repository, feedback_factory
):
    existing = feedback_factory(
        value=None,
        error=AssessmentError(error_code="JUDGE_ERROR", error_message="judge failed"),
    )
    trace_repository.get_assessment.return_value = assessment_store._assessment_to_document(
        existing
    )
    assessment_store.update_assessment(TRACE_ID, "a-original", feedback=FeedbackValue(True))
    updated = trace_repository.update_assessment.call_args.kwargs["updates"]["feedback"]
    assert updated["value"] is True
    assert "error" not in updated


def test_update_returns_repository_state(assessment_store, trace_repository, feedback_factory):
    stored = feedback_factory(value=False, metadata={"reviewer": "alice", "new": "value"})
    trace_repository.update_assessment.return_value = assessment_store._assessment_to_document(
        stored
    )
    result = assessment_store.update_assessment(TRACE_ID, "a-original", metadata={"new": "value"})
    assert result.value is False
    assert result.metadata == {"reviewer": "alice", "new": "value"}
    assert result.span_id == stored.span_id
    assert result.source == stored.source
    assert result.create_time_ms == stored.create_time_ms


def test_update_structured_expectation_passes_native_value(assessment_store, trace_repository):
    expectation = Expectation(name="expected", value="old", trace_id=TRACE_ID)
    trace_repository.get_assessment.return_value = assessment_store._assessment_to_document(
        expectation
    )
    assessment_store.update_assessment(
        TRACE_ID, "a-original", expectation=ExpectationValue({"answer": [1, True]})
    )
    assert trace_repository.update_assessment.call_args.kwargs["updates"]["expectation"] == {
        "value": {"answer": [1, True]}
    }


def test_update_disappearing_assessment_reports_safe_not_found(assessment_store, trace_repository):
    trace_repository.update_assessment.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException) as caught:
        assessment_store.update_assessment(TRACE_ID, "a-original", name="new")
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert "private" not in caught.value.message
    assert caught.value.__cause__ is None


def test_delete_checks_trace_then_delegates_without_requiring_assessment(
    assessment_store, trace_repository
):
    assert assessment_store.delete_assessment(TRACE_ID, "a-missing") is None
    trace_repository.delete_assessment.assert_called_once_with(
        trace_id=TRACE_ID, assessment_id="a-missing"
    )
    trace_repository.get_assessment.assert_not_called()


def test_delete_missing_trace_does_not_touch_assessments(assessment_store, trace_repository):
    trace_repository.get_trace_info.side_effect = RepositoryNotFoundError("private")
    with pytest.raises(MlflowException) as caught:
        assessment_store.delete_assessment(TRACE_ID, "a-original")
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    trace_repository.delete_assessment.assert_not_called()


def test_get_persistence_error_is_sanitized(assessment_store, trace_repository, caplog):
    error = RepositoryPersistenceError("Private URI and query")
    trace_repository.get_assessment.side_effect = error
    with pytest.raises(MlflowException) as caught:
        assessment_store.get_assessment(TRACE_ID, "a-original")
    assert caught.value.message == "Unable to read assessment."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


def test_create_persistence_error_is_sanitized(
    assessment_store, trace_repository, feedback_factory, caplog
):
    error = RepositoryPersistenceError("Private URI and query")
    trace_repository.create_assessment.side_effect = error
    with pytest.raises(MlflowException) as caught:
        assessment_store.create_assessment(feedback_factory())
    assert caught.value.message == "Unable to create assessment."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert len(caplog.records) == 1
    assert caplog.records[0].args == (error,)
    assert caplog.records[0].exc_info is None


def test_update_persistence_error_is_sanitized(assessment_store, trace_repository, caplog):
    error = RepositoryPersistenceError("Private URI and query")
    trace_repository.update_assessment.side_effect = error
    with pytest.raises(MlflowException) as caught:
        assessment_store.update_assessment(TRACE_ID, "a-original", name="new")
    assert caught.value.message == "Unable to update assessment."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


def test_delete_persistence_error_is_sanitized(assessment_store, trace_repository, caplog):
    error = RepositoryPersistenceError("Private URI and query")
    trace_repository.delete_assessment.side_effect = error
    with pytest.raises(MlflowException) as caught:
        assessment_store.delete_assessment(TRACE_ID, "a-original")
    assert caught.value.message == "Unable to delete assessment."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None
