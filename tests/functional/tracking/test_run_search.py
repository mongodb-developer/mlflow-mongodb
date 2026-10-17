"""Run search contracts against real MongoDB documents."""

import pytest
from mlflow.entities import (
    Dataset,
    DatasetInput,
    InputTag,
    LoggedModelInput,
    LoggedModelOutput,
    Metric,
    Param,
    RunStatus,
    RunTag,
    ViewType,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.search_utils import SearchUtils


def _search(store, experiment_id, expression="", **kwargs):
    return store.search_runs([experiment_id], expression, ViewType.ACTIVE_ONLY, **kwargs)


def _run_names(runs):
    return [run.info.run_name for run in runs]


@pytest.fixture
def searchable_runs(store):
    experiment_id = store.create_experiment("searchable-runs")
    runs = {
        name: store.create_run(experiment_id, "test-user", 1000, [], name)
        for name in ("low", "high", "nan", "missing")
    }
    for name, value, depth, priority in [("low", 1.0, "10", "b"), ("high", 2.0, "2", "a")]:
        store.log_batch(
            runs[name].info.run_id,
            [Metric("loss", value, 1000, 1), Metric("score.test", value, 1000, 1)],
            [Param("depth", depth), Param("param.test", depth), Param("empty", "")],
            [RunTag("priority", priority), RunTag("team.name", "Risk"), RunTag("empty", "")],
        )
    store.log_metric(runs["nan"].info.run_id, Metric("loss", float("nan"), 1000, 1))
    return experiment_id, runs


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("metrics.loss > 1", {"high"}),
        ("metrics.loss >= 1", {"low", "high"}),
        ("metrics.loss < 2", {"low"}),
        ("metrics.loss <= 2", {"low", "high"}),
        ("metrics.loss = 1", {"low"}),
        ("metrics.loss != 1", {"high", "nan"}),
        ("params.depth = '10'", {"low"}),
        ("params.depth != '10'", {"high"}),
        ("params.depth LIKE '1%'", {"low"}),
        ("tags.priority != 'a'", {"low"}),
        ("tags.`team.name` ILIKE 'risk'", {"low", "high"}),
        ("tags.`team.name` LIKE 'risk'", set()),
        ("metrics.`score.test` = 2", {"high"}),
        ("params.`param.test` = '2'", {"high"}),
        ("attributes.run_name = 'high'", {"high"}),
        ("attributes.`run name` LIKE 'h%'", {"high"}),
        ("attributes.start_time >= 1000", {"low", "high", "nan", "missing"}),
        ("attributes.end_time != 2000", set()),
        ("params.empty = '' AND tags.empty = ''", {"low", "high"}),
        ("metrics.loss >= 1 AND params.depth = '2' AND tags.priority = 'a'", {"high"}),
        ("tags.priority = 'a' AND tags.priority = 'b'", set()),
    ],
)
def test_search_run_filters(store, searchable_runs, expression, expected):
    experiment_id, _ = searchable_runs
    assert set(_run_names(_search(store, experiment_id, expression))) == expected


@pytest.mark.parametrize(
    ("category", "comparator", "expected"),
    [
        (category, comparator, expected)
        for category, supported in (
            ("params", SearchUtils.VALID_PARAM_COMPARATORS),
            ("tags", SearchUtils.VALID_TAG_COMPARATORS),
        )
        for comparator, expected in (
            ("IS NULL", {"nan", "missing"}),
            ("IS NOT NULL", {"low", "high"}),
        )
        if comparator in supported
    ],
)
def test_search_key_existence_includes_empty_values(
    store, searchable_runs, category, comparator, expected
):
    experiment_id, _ = searchable_runs
    expression = f"{category}.empty {comparator}"
    assert set(_run_names(_search(store, experiment_id, expression))) == expected


@pytest.mark.parametrize(
    ("category", "comparator"),
    [
        (category, comparator)
        for category, supported in (
            ("params", SearchUtils.VALID_PARAM_COMPARATORS),
            ("tags", SearchUtils.VALID_TAG_COMPARATORS),
        )
        for comparator in ("IS NULL", "IS NOT NULL")
        if comparator not in supported
    ],
)
def test_search_rejects_unsupported_key_existence_comparators(
    store, searchable_runs, category, comparator
):
    experiment_id, _ = searchable_runs
    expression = f"{category}.empty {comparator}"
    with pytest.raises(MlflowException) as caught:
        _search(store, experiment_id, expression)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("metrics.loss ASC", ["low", "high", "nan", "missing"]),
        ("metrics.loss DESC", ["high", "low", "nan", "missing"]),
    ],
)
def test_metric_sorting_places_nan_and_missing_after_numbers(
    store, searchable_runs, order, expected
):
    experiment_id, _ = searchable_runs
    page = _search(store, experiment_id, order_by=[order])
    assert _run_names(page) == expected


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("params.depth ASC", ["low", "high"]),
        ("params.depth DESC", ["high", "low"]),
        ("tags.priority ASC", ["high", "low"]),
        ("tags.priority DESC", ["low", "high"]),
        ("metrics.`score.test` DESC", ["high", "low"]),
        ("params.`param.test` DESC", ["high", "low"]),
    ],
)
def test_embedded_field_sorting_places_missing_values_last(store, searchable_runs, order, expected):
    experiment_id, _ = searchable_runs
    page = _search(store, experiment_id, order_by=[order])
    names = _run_names(page)
    assert names[:2] == expected
    assert set(names[2:]) == {"nan", "missing"}
    trailing_run_ids = [run.info.run_id for run in page[2:]]
    assert trailing_run_ids == sorted(trailing_run_ids)


def test_equal_sort_values_are_ordered_by_run_id(store, searchable_runs):
    experiment_id, _ = searchable_runs
    page = _search(store, experiment_id, order_by=["tags.`team.name` ASC"])
    names = _run_names(page)
    assert set(names[:2]) == {"low", "high"}
    assert set(names[2:]) == {"nan", "missing"}
    matching_run_ids = [run.info.run_id for run in page[:2]]
    assert matching_run_ids == sorted(matching_run_ids)
    missing_run_ids = [run.info.run_id for run in page[2:]]
    assert missing_run_ids == sorted(missing_run_ids)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("ASC", ["high", "low", "missing", "nan"]), ("DESC", ["nan", "missing", "low", "high"])],
)
def test_sort_by_run_name_uses_stored_name_field(store, searchable_runs, direction, expected):
    experiment_id, _ = searchable_runs

    page = _search(store, experiment_id, order_by=[f"attributes.run_name {direction}"])

    assert _run_names(page) == expected


def test_default_order_is_start_time_descending_then_run_id(store):
    experiment_id = store.create_experiment("run-search-order")
    old = store.create_run(experiment_id, "test-user", 1000, [], "old")
    newer = [store.create_run(experiment_id, "test-user", 2000, [], name) for name in ("a", "b")]
    page = _search(store, experiment_id)
    assert [run.info.run_id for run in page] == [
        *sorted(run.info.run_id for run in newer),
        old.info.run_id,
    ]
    ascending = _search(store, experiment_id, order_by=["attributes.created ASC"])
    assert ascending[0].info.run_id == old.info.run_id


def test_multi_field_sort_preserves_requested_precedence(store, searchable_runs):
    experiment_id, _ = searchable_runs
    assert _run_names(
        _search(store, experiment_id, order_by=["tags.`team.name` ASC", "metrics.loss DESC"])
    )[:2] == ["high", "low"]


def test_search_scopes_experiments_and_all_lifecycle_views(store):
    experiment_id = store.create_experiment("run-search-scope")
    other_id = store.create_experiment("run-search-other")
    active = store.create_run(experiment_id, "test-user", 1000, [], "active")
    deleted = store.create_run(experiment_id, "test-user", 1000, [], "deleted")
    store.create_run(other_id, "test-user", 1000, [], "other")
    store.delete_run(deleted.info.run_id)
    store.update_run_info(active.info.run_id, RunStatus.FINISHED, 2000, None)
    for view, expected in [
        (ViewType.ACTIVE_ONLY, {"active"}),
        (ViewType.DELETED_ONLY, {"deleted"}),
        (ViewType.ALL, {"active", "deleted"}),
    ]:
        assert set(_run_names(store.search_runs([experiment_id], "", view))) == expected
    assert set(_run_names(store.search_runs([experiment_id, other_id], "", ViewType.ALL))) == {
        "active",
        "deleted",
        "other",
    }
    assert store.search_runs([], "", ViewType.ALL) == []
    assert store.search_runs(["unknown"], "", ViewType.ALL) == []
    assert _run_names(_search(store, experiment_id, "attributes.status = 'FINISHED'")) == ["active"]


def test_search_uses_latest_metric_summary(store, searchable_runs):
    experiment_id, runs = searchable_runs
    run_id = runs["high"].info.run_id
    store.log_metric(run_id, Metric("loss", 0.1, 2000, 2))
    store.log_metric(run_id, Metric("loss", 20.0, 3000, 1))
    assert _search(store, experiment_id, "metrics.loss > 1") == []
    assert _run_names(_search(store, experiment_id, "metrics.loss = 0.1")) == ["high"]
    assert len(store.get_metric_history(run_id, "loss")) == 3


def test_run_id_membership_filters(store, searchable_runs):
    experiment_id, runs = searchable_runs
    run_id = runs["high"].info.run_id
    assert _run_names(_search(store, experiment_id, f"attributes.run_id IN ('{run_id}')")) == [
        "high"
    ]
    assert set(
        _run_names(_search(store, experiment_id, f"attributes.run_id NOT IN ('{run_id}')"))
    ) == {
        "low",
        "nan",
        "missing",
    }


@pytest.mark.parametrize("direction", ["ASC", "DESC"])
def test_nullable_attribute_sorting_keeps_missing_values_last(store, searchable_runs, direction):
    experiment_id, runs = searchable_runs
    store.update_run_info(runs["low"].info.run_id, RunStatus.FINISHED, 2000, None)
    store.update_run_info(runs["high"].info.run_id, RunStatus.FINISHED, 3000, None)
    names = _run_names(_search(store, experiment_id, order_by=[f"attributes.end_time {direction}"]))
    assert names[:2] == (["low", "high"] if direction == "ASC" else ["high", "low"])
    assert set(names[2:]) == {"nan", "missing"}


def test_pagination_has_no_duplicates_or_omissions(store, searchable_runs):
    experiment_id, _ = searchable_runs
    expected = _search(store, experiment_id, order_by=["metrics.loss DESC"])
    first = _search(store, experiment_id, max_results=2, order_by=["metrics.loss DESC"])
    assert first.token is not None
    second = _search(
        store, experiment_id, max_results=2, order_by=["metrics.loss DESC"], page_token=first.token
    )
    assert second.token is None
    assert [run.info.run_id for run in [*first, *second]] == [run.info.run_id for run in expected]
    unbounded = _search(store, experiment_id, max_results=None, order_by=["metrics.loss DESC"])
    assert unbounded.token is None
    assert [run.info.run_id for run in unbounded] == [run.info.run_id for run in expected]


def test_pagination_is_applied_after_filters(store, searchable_runs):
    experiment_id, _ = searchable_runs
    first = _search(
        store, experiment_id, "metrics.loss >= 1", max_results=1, order_by=["metrics.loss DESC"]
    )
    assert _run_names(first) == ["high"]
    assert first.token is not None
    second = _search(
        store,
        experiment_id,
        "metrics.loss >= 1",
        max_results=1,
        order_by=["metrics.loss DESC"],
        page_token=first.token,
    )
    assert _run_names(second) == ["low"]
    assert second.token is None


def test_search_returns_complete_entities_and_preserves_stored_documents(store, run):
    run_id = run.info.run_id
    store.log_batch(
        run_id, [Metric("loss", 0.4, 1000, 1)], [Param("depth", "3")], [RunTag("team.name", "risk")]
    )
    store.log_inputs(
        run_id,
        [
            DatasetInput(
                Dataset("training", "abc", "test", "{}"),
                [InputTag(MLFLOW_DATASET_CONTEXT, "training")],
            )
        ],
        [LoggedModelInput("m-input")],
    )
    store.log_outputs(run_id, [LoggedModelOutput("m-output", 2)])
    collection = store._database[store._settings.runs_collection_name]
    before = collection.find_one({"_id": run_id})
    page = _search(store, run.info.experiment_id, order_by=["tags.`team.name` DESC"])
    assert len(page) == 1
    assert page[0].to_proto() == store.get_run(run_id).to_proto()
    assert page[0].data.params == {"depth": "3"}
    assert page[0].inputs.dataset_inputs[0].tags[0].value == "training"
    assert page[0].inputs.model_inputs[0].model_id == "m-input"
    assert page[0].outputs.model_outputs[0].model_id == "m-output"
    assert collection.find_one({"_id": run_id}) == before
    assert before["params"] == [{"k": "depth", "v": "3"}]
    assert all(set(tag) == {"k", "v"} for tag in before["tags"])


def test_dataset_filters_match_independent_inputs_and_context_tags(store, run):
    store.log_inputs(
        run.info.run_id,
        [
            DatasetInput(
                Dataset("training", "aaa", "test", "{}"),
                [InputTag(MLFLOW_DATASET_CONTEXT, "train")],
            ),
            DatasetInput(
                Dataset("evaluation", "bbb", "test", "{}"),
                [InputTag(MLFLOW_DATASET_CONTEXT, "eval")],
            ),
        ],
    )
    expected_run = store.get_run(run.info.run_id)
    for expression in [
        "datasets.name = 'training'",
        "datasets.digest = 'bbb'",
        "datasets.context = 'eval'",
        "datasets.context != 'eval'",
        "datasets.context NOT IN ('train')",
        "datasets.name = 'training' AND datasets.digest = 'bbb'",
    ]:
        results = _search(store, run.info.experiment_id, expression)
        assert len(results) == 1
        assert results[0].info.run_id == run.info.run_id
        assert results[0].to_proto() == expected_run.to_proto()
    assert _search(store, run.info.experiment_id, "datasets.context = 'missing'") == []
    assert (
        _search(store, run.info.experiment_id, "datasets.name NOT IN ('training', 'evaluation')")
        == []
    )


def test_search_through_mlflow_client(store, searchable_runs):
    experiment_id, _ = searchable_runs
    client = MlflowClient(tracking_uri=store.store_uri)
    page = client.search_runs(
        [experiment_id], "params.depth != '10'", max_results=1, order_by=["metrics.loss DESC"]
    )
    assert _run_names(page) == ["high"]
    assert page.token is None


@pytest.mark.parametrize(
    "expression", ["metrics.loss LIKE 'x'", "attributes.unknown = 'x'", "params.depth > '2'"]
)
def test_invalid_filters_report_mlflow_validation_errors(store, run, expression):
    with pytest.raises(MlflowException) as caught:
        _search(store, run.info.experiment_id, expression)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
