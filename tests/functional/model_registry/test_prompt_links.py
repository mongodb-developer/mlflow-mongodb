"""Inherited prompt-link operations against real tracking backends."""

import json
from uuid import uuid4

import mlflow
import pytest
from mlflow.entities import TraceInfo, TraceLocation, TraceState
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.run_tag import RunTag
from mlflow.store.tracking.abstract_store import AbstractStore as AbstractTrackingStore
from mlflow.tracking._tracking_service.utils import _get_store  # ruff: ignore[import-private-name]

from mlflow_mongodb.model_registry.store import MongoDBModelRegistryStore

LINKED_PROMPTS_TAG_KEY = "mlflow.linkedPrompts"


@pytest.fixture
def prompts(store):
    store.create_prompt("linked-prompt")
    return [
        store.create_prompt_version("linked-prompt", "Hello {{name}}"),
        store.create_prompt_version("linked-prompt", "Welcome {{name}}"),
    ]


@pytest.fixture
def tracking(store, mongodb_uri, tmp_path):
    tracking_store = _get_store(store_uri=mongodb_uri)
    tracking_store.__dict__["_mongo_client"] = store._mongo_client
    experiment_id = tracking_store.create_experiment(
        f"inherited-prompt-links-{uuid4().hex}", artifact_location=tmp_path.as_uri()
    )
    previous_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(mongodb_uri)
        yield tracking_store, experiment_id
    finally:
        mlflow.set_tracking_uri(previous_uri)
        for collection_name in (
            tracking_store._settings.traces_collection_name,
            tracking_store._settings.runs_collection_name,
            tracking_store._settings.logged_models_collection_name,
        ):
            tracking_store._database[collection_name].delete_many({"experiment_id": experiment_id})
        tracking_store._database[tracking_store._settings.experiments_collection_name].delete_one(
            {"_id": int(experiment_id)}
        )


@pytest.fixture
def linking_registry(store, mongodb_uri, tmp_path):
    """Exercise inherited links with a conflicting instance tracking URI."""
    tracking_uri = f"sqlite:///{tmp_path / 'other-tracking.db'}"
    registry = MongoDBModelRegistryStore(store_uri=mongodb_uri, tracking_uri=tracking_uri)
    registry.__dict__["_mongo_client"] = store._mongo_client
    return registry


def test_inherited_prompt_run_links_merge_versions_without_duplicates(
    linking_registry, prompts, tracking
):
    tracking_store, experiment_id = tracking
    run = tracking_store.create_run(
        experiment_id,
        user_id="functional-test",
        start_time=1_700_000_000_000,
        tags=[RunTag("team", "risk")],
        run_name="prompt-linked-run",
    )

    for prompt in [*prompts, prompts[0]]:
        linking_registry.link_prompt_version_to_run(
            prompt.name, str(prompt.version), run.info.run_id
        )

    tags = tracking_store.get_run(run.info.run_id).data.tags
    assert json.loads(tags[LINKED_PROMPTS_TAG_KEY]) == [
        {"name": prompt.name, "version": str(prompt.version)} for prompt in prompts
    ]
    assert tags["team"] == "risk"


def test_inherited_prompt_model_links_merge_versions_without_duplicates(
    linking_registry, prompts, tracking
):
    tracking_store, experiment_id = tracking
    model = tracking_store.create_logged_model(
        experiment_id, name="prompt-linked-model", tags=[LoggedModelTag("team", "risk")]
    )

    for prompt in [*prompts, prompts[0]]:
        linking_registry.link_prompt_version_to_model(
            prompt.name, str(prompt.version), model.model_id
        )

    tags = tracking_store.get_logged_model(model.model_id).tags
    assert json.loads(tags[LINKED_PROMPTS_TAG_KEY]) == [
        {"name": prompt.name, "version": str(prompt.version)} for prompt in prompts
    ]
    assert tags["team"] == "risk"


@pytest.fixture
def trace(tracking):
    tracking_store, experiment_id = tracking
    trace_id = f"tr-{uuid4().hex}"
    return tracking_store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id(experiment_id),
            request_time=1_700_000_000_000,
            state=TraceState.OK,
            execution_duration=10,
            tags={"team": "risk"},
        )
    )


def test_inherited_prompt_trace_links_merge_versions_without_duplicates(
    linking_registry, prompts, tracking, trace
):
    tracking_store, _ = tracking
    linking_registry.link_prompts_to_trace([prompts[0]], trace.trace_id)
    linking_registry.link_prompts_to_trace(prompts, trace.trace_id)
    linking_registry.link_prompts_to_trace([prompts[0]], trace.trace_id)

    expected = [{"name": prompt.name, "version": str(prompt.version)} for prompt in prompts]
    tags = tracking_store.get_trace_info(trace.trace_id).tags
    assert json.loads(tags[LINKED_PROMPTS_TAG_KEY]) == expected
    assert tags["team"] == "risk"


@pytest.mark.skipif(
    not hasattr(AbstractTrackingStore, "link_prompts_to_trace"),
    reason="Installed MLflow does not expose native prompt-to-trace linking",
)
def test_inherited_prompt_trace_links_persist_searchable_native_associations(
    linking_registry, prompts, tracking, trace
):
    tracking_store, experiment_id = tracking

    linking_registry.link_prompts_to_trace(prompts, trace.trace_id)
    linking_registry.link_prompts_to_trace(prompts, trace.trace_id)

    expected = [{"name": prompt.name, "version": str(prompt.version)} for prompt in prompts]
    document = tracking_store._trace_repository._collection.find_one({"_id": trace.trace_id})
    assert document["linked_prompts"] == expected
    for prompt in prompts:
        traces, _ = tracking_store.search_traces(
            experiment_ids=[experiment_id],
            filter_string=f"tags.`{LINKED_PROMPTS_TAG_KEY}` = '{prompt.name}/{prompt.version}'",
        )
        assert [result.trace_id for result in traces] == [trace.trace_id]
