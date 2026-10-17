"""Trace creation, span ingestion, retrieval, and deletion against MongoDB."""

import json
from copy import deepcopy

import pytest
from mlflow.entities.assessment import Expectation
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.protos.service_pb2 import LinkTracesToRun
from mlflow.tracing.constant import (
    TraceMetadataKey,
    TraceSizeStatsKey,
)
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

from mlflow_mongodb.tracking._compat import (  # ruff: ignore[import-private-name]
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
