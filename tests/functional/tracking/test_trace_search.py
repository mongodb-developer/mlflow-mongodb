"""Trace filtering, missing values, and pagination through real MongoDB pipelines."""

import pytest
from mlflow.entities import TraceState
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey
from mlflow.utils.search_utils import SearchTraceUtils


@pytest.fixture
def searchable_traces(store, trace_info_factory):
    traces = {
        "a": store.start_trace(
            trace_info_factory(
                request_time=1000,
                tags={TraceTagKey.TRACE_NAME: "alpha", "team.name": "$risk"},
                trace_metadata={"session.key": "one"},
                client_request_id="client",
            )
        ),
        "b": store.start_trace(
            trace_info_factory(
                request_time=2000,
                state=TraceState.ERROR,
                tags={TraceTagKey.TRACE_NAME: "beta", "team.name": "other"},
                trace_metadata={TraceMetadataKey.SOURCE_RUN: "run"},
                client_request_id="other",
            )
        ),
        "c": store.start_trace(
            trace_info_factory(
                request_time=3000,
                state=TraceState.IN_PROGRESS,
                execution_duration=None,
            )
        ),
    }
    store.link_traces_to_run([traces["a"].trace_id], "run")
    store._trace_repository.link_prompts(
        trace_id=traces["a"].trace_id, prompt_versions=[{"name": "greeting", "version": "2"}]
    )
    return traces


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("status = 'ERROR'", ["b"]),
        ("status != 'ERROR'", ["a", "c"]),
        pytest.param(
            "client_request_id != 'other'",
            ["a"],
            marks=pytest.mark.skipif(
                "client_request_id" not in SearchTraceUtils.VALID_SEARCH_ATTRIBUTE_KEYS,
                reason="Installed MLflow has no client-request-ID filters",
            ),
        ),
        ("timestamp_ms >= 2000", ["b", "c"]),
        pytest.param(
            "end_time_ms >= 2010",
            ["b", "c"],
            marks=pytest.mark.skipif(
                "end_time_ms" not in SearchTraceUtils.VALID_SEARCH_ATTRIBUTE_KEYS,
                reason="Installed MLflow has no end-time filters",
            ),
        ),
        ("tags.`team.name` = '$risk'", ["a"]),
        pytest.param(
            "tags.`team.name` IS NULL",
            ["c"],
            marks=pytest.mark.skipif(
                "IS NULL" not in SearchTraceUtils.VALID_TAG_COMPARATORS,
                reason="Installed MLflow has no tag existence filters",
            ),
        ),
        pytest.param(
            "tags.`team.name` IS NOT NULL",
            ["a", "b"],
            marks=pytest.mark.skipif(
                "IS NOT NULL" not in SearchTraceUtils.VALID_TAG_COMPARATORS,
                reason="Installed MLflow has no tag existence filters",
            ),
        ),
        ("request_metadata.`session.key` = 'one'", ["a"]),
        pytest.param(
            "name ILIKE 'AL%'",
            ["a"],
            marks=pytest.mark.skipif(
                "ILIKE" not in SearchTraceUtils.VALID_TAG_COMPARATORS,
                reason="Installed MLflow has no ILIKE trace-name filters",
            ),
        ),
        pytest.param(
            "name RLIKE '^b.*a$'",
            ["b"],
            marks=pytest.mark.skipif(
                "RLIKE" not in SearchTraceUtils.VALID_TAG_COMPARATORS,
                reason="Installed MLflow has no RLIKE trace-name filters",
            ),
        ),
        ("run_id = 'run'", ["a", "b"]),
        pytest.param(
            "prompt = 'greeting/2'",
            ["a"],
            marks=pytest.mark.skipif(
                "prompt" not in SearchTraceUtils.SEARCH_KEY_TO_TAG,
                reason="Installed MLflow has no prompt search aliases",
            ),
        ),
        ("status = 'OK' AND tags.`team.name` = '$risk'", ["a"]),
    ],
)
def test_filters_return_matching_traces(store, searchable_traces, expression, expected):
    experiment_id = searchable_traces["a"].experiment_id

    records, _ = store.search_traces(
        experiment_ids=[experiment_id],
        filter_string=expression,
        order_by=["timestamp_ms ASC"],
    )

    assert [info.trace_id for info in records] == [
        searchable_traces[key].trace_id for key in expected
    ]


def test_custom_order_places_missing_values_last_and_keeps_pages_stable(store, searchable_traces):
    experiment_id = searchable_traces["a"].experiment_id

    first, token = store.search_traces(
        experiment_ids=[experiment_id],
        order_by=["name DESC"],
        max_results=1,
    )
    second, next_token = store.search_traces(
        experiment_ids=[experiment_id],
        order_by=["name DESC"],
        max_results=1,
        page_token=token,
    )
    last, final_token = store.search_traces(
        experiment_ids=[experiment_id],
        order_by=["name DESC"],
        max_results=2,
        page_token=next_token,
    )

    assert [info.trace_id for info in first + second + last] == [
        searchable_traces[key].trace_id for key in ("b", "a", "c")
    ]
    assert token is not None and next_token is not None
    assert final_token is None


@pytest.mark.parametrize(
    "order,expected_indices",
    [
        pytest.param(
            "end_time_ms ASC",
            [3, 1, 2, 0],
            marks=pytest.mark.skipif(
                "end_time_ms" not in SearchTraceUtils.VALID_ORDER_BY_ATTRIBUTE_KEYS,
                reason="Installed MLflow has no end-time ordering",
            ),
        ),
        pytest.param(
            "end_time_ms DESC",
            [0, 2, 3, 1],
            marks=pytest.mark.skipif(
                "end_time_ms" not in SearchTraceUtils.VALID_ORDER_BY_ATTRIBUTE_KEYS,
                reason="Installed MLflow has no end-time ordering",
            ),
        ),
        ("execution_time_ms ASC", [3, 1, 0, 2]),
        ("execution_time_ms DESC", [0, 1, 3, 2]),
    ],
)
def test_attribute_ordering_uses_computed_end_time_and_stored_duration(
    store, trace_info_factory, order, expected_indices
):
    traces = [
        store.start_trace(trace_info_factory(request_time=start, execution_duration=duration))
        for start, duration in ((1000, 3000), (2000, 500), (3000, None), (2500, 0))
    ]

    records, _ = store.search_traces(
        experiment_ids=[traces[0].experiment_id],
        order_by=[order],
    )

    assert [record.trace_id for record in records] == [
        traces[index].trace_id for index in expected_indices
    ]


def test_search_scope_excludes_other_experiments_and_empty_selection(store, searchable_traces):
    experiment_id = store.create_experiment("unrelated-trace-search")

    assert store.search_traces(experiment_ids=[experiment_id]) == ([], None)
    assert store.search_traces(locations=[]) == ([], None)
    records, _ = store.search_traces(experiment_ids=[searchable_traces["a"].experiment_id])
    assert [record.trace_id for record in records] == [
        searchable_traces[key].trace_id for key in ("c", "b", "a")
    ]
