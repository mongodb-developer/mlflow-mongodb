"""Assessment CRUD and trace joins against MongoDB."""

from copy import deepcopy

import pytest
from mlflow.entities.assessment import (
    AssessmentError,
    Expectation,
    ExpectationValue,
    Feedback,
    FeedbackValue,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode


@pytest.mark.parametrize("value", [True, 7, 0.5, "$yes", [1, "a"], {"answer": [1, True]}])
def test_feedback_create_get_and_trace_join_preserve_native_values(store, trace, value):
    assessment = Feedback(name="quality", value=value, trace_id=trace.trace_id)

    created = store.create_assessment(assessment)
    loaded = store.get_assessment(trace.trace_id, created.assessment_id)

    assert created.assessment_id.startswith("a-")
    assert loaded.value == value
    assert type(loaded.value) is type(value)
    assert loaded.valid is True
    joined = store.get_trace_info(trace.trace_id).assessments
    assert joined == [loaded]
    document = store._trace_repository._assessments_collection.find_one(
        {"assessment_id": created.assessment_id}
    )
    assert document["experiment_id"] == trace.experiment_id
    assert document["content"]["feedback"]["value"] == value
    assert type(document["content"]["feedback"]["value"]) is type(value)


@pytest.mark.parametrize("value", [[1, {"answer": True}], {"answer": [1, "yes"]}])
def test_expectation_create_and_update_store_native_structured_values(store, trace, value):
    created = store.create_assessment(
        Expectation(name="expected", value="old", trace_id=trace.trace_id)
    )

    updated = store.update_assessment(
        trace.trace_id, created.assessment_id, expectation=ExpectationValue(value)
    )

    assert updated.value == value
    assert store.get_assessment(trace.trace_id, created.assessment_id).value == value
    document = store._trace_repository._assessments_collection.find_one(
        {"assessment_id": created.assessment_id}
    )
    assert document["content"]["expectation"] == {"value": value}


def test_update_merges_metadata_preserves_immutable_fields_and_accepts_empty_strings(
    store, trace, clock
):
    assessment = Feedback(
        name="quality",
        value=True,
        trace_id=trace.trace_id,
        span_id="0000000000000001",
        metadata={"preserve": "yes", "replace": "old"},
        create_time_ms=clock["now"] - 10,
        last_update_time_ms=clock["now"] - 10,
    )
    assessment.run_id = "run"
    created = store.create_assessment(assessment)
    incoming = {"replace": "$new", "extra": "value"}
    before = deepcopy(incoming)
    clock["now"] += 5

    updated = store.update_assessment(
        trace.trace_id,
        created.assessment_id,
        name="",
        rationale="",
        metadata=incoming,
        feedback=FeedbackValue(False),
    )

    assert isinstance(updated.name, str) and not updated.name
    assert isinstance(updated.rationale, str) and not updated.rationale
    assert updated.value is False
    assert updated.metadata == {"preserve": "yes", "replace": "$new", "extra": "value"}
    assert updated.assessment_id == created.assessment_id
    assert updated.trace_id == created.trace_id
    assert updated.span_id == created.span_id
    assert updated.source == created.source
    assert updated.run_id == "run"
    assert updated.create_time_ms == created.create_time_ms
    assert updated.last_update_time_ms == clock["now"]
    assert incoming == before
    assert store.get_trace_info(trace.trace_id).assessments == [updated]


def test_feedback_error_can_be_set_and_cleared(store, trace):
    created = store.create_assessment(Feedback(name="quality", value=True, trace_id=trace.trace_id))
    error = AssessmentError(error_code="JUDGE_ERROR", error_message="judge failed")

    failed = store.update_assessment(
        trace.trace_id, created.assessment_id, feedback=FeedbackValue(None, error)
    )

    assert failed.value is None
    assert failed.feedback.error == error
    assert store.get_assessment(trace.trace_id, created.assessment_id).feedback.error == error

    recovered = store.update_assessment(
        trace.trace_id, created.assessment_id, feedback=FeedbackValue(True)
    )

    assert recovered.value is True
    assert recovered.feedback.error is None
    assert store.get_assessment(trace.trace_id, created.assessment_id).feedback.error is None


def test_override_creation_invalidates_original_and_deletion_restores_it(store, trace):
    original = store.create_assessment(
        Feedback(name="quality", value=False, trace_id=trace.trace_id)
    )
    override = store.create_assessment(
        Feedback(
            name="quality",
            value=True,
            trace_id=trace.trace_id,
            overrides=original.assessment_id,
        )
    )

    assert store.get_assessment(trace.trace_id, original.assessment_id).valid is False
    assert override.valid is True
    assert override.overrides == original.assessment_id
    assert {
        item.assessment_id: item.valid for item in store.get_trace_info(trace.trace_id).assessments
    } == {
        original.assessment_id: False,
        override.assessment_id: True,
    }

    store.delete_assessment(trace.trace_id, override.assessment_id)

    assert store.get_assessment(trace.trace_id, original.assessment_id).valid is True
    with pytest.raises(MlflowException) as caught:
        store.get_assessment(trace.trace_id, override.assessment_id)
    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert [item.assessment_id for item in store.get_trace_info(trace.trace_id).assessments] == [
        original.assessment_id
    ]


def test_deleting_assessment_is_idempotent_and_preserves_other_assessments(store, trace):
    first = store.create_assessment(Feedback(name="first", value=True, trace_id=trace.trace_id))
    other = store.create_assessment(Feedback(name="other", value=False, trace_id=trace.trace_id))

    store.delete_assessment(trace.trace_id, first.assessment_id)
    store.delete_assessment(trace.trace_id, first.assessment_id)

    assert store.get_assessment(trace.trace_id, other.assessment_id) == other
    assert store.get_trace_info(trace.trace_id).assessments == [other]


def test_assessment_lookup_is_scoped_to_trace(store, trace, trace_info_factory):
    other = store.start_trace(trace_info_factory())
    assessment = store.create_assessment(
        Feedback(name="quality", value=True, trace_id=trace.trace_id)
    )

    with pytest.raises(MlflowException) as caught:
        store.get_assessment(other.trace_id, assessment.assessment_id)

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert store.get_assessment(trace.trace_id, assessment.assessment_id) == assessment
