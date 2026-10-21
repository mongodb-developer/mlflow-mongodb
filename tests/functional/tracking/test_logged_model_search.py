"""Logged-model metadata and metric searches against MongoDB."""

import pytest
from mlflow.entities import Metric
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode


def _names(page):
    return [model.name for model in page]


@pytest.fixture
def searchable_models(store, run, request):
    request.getfixturevalue("clock")
    models = {}
    for name, depth, team in [
        ("low", "10", "Risk"),
        ("high", "2", "Platform"),
        ("missing", None, None),
    ]:
        models[name] = store.create_logged_model(
            run.info.experiment_id,
            name=name,
            params=[] if depth is None else [LoggedModelParameter("depth", depth)],
            tags=[]
            if team is None
            else [LoggedModelTag("team.name", team), LoggedModelTag("empty", "")],
            model_type=None if name == "missing" else name,
        )
    store.log_batch(
        run.info.run_id,
        [
            Metric(
                "loss",
                0.05,
                1000,
                5,
                model_id=models["low"].model_id,
                dataset_name="eval",
                dataset_digest="a",
            ),
            Metric(
                "loss",
                0.8,
                2000,
                1,
                model_id=models["low"].model_id,
                dataset_name="train",
                dataset_digest="b",
            ),
            Metric(
                "score.test",
                0.9,
                3000,
                1,
                model_id=models["low"].model_id,
                dataset_name="eval",
                dataset_digest="a",
            ),
            Metric(
                "loss",
                0.2,
                1000,
                1,
                model_id=models["high"].model_id,
                dataset_name="eval",
                dataset_digest="c",
            ),
            Metric(
                "loss",
                0.4,
                2000,
                1,
                model_id=models["high"].model_id,
                dataset_name="train",
                dataset_digest="b",
            ),
        ],
        [],
        [],
    )
    return run.info.experiment_id, models


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("name = 'low'", {"low"}),
        ("params.depth = '10'", {"low"}),
        ("params.depth != '10'", {"high"}),
        ("params.depth IN ('10', '2')", {"low", "high"}),
        ("params.depth NOT IN ('10')", {"high"}),
        ("params.depth NOT IN ()", {"low", "high"}),
        ("params.depth LIKE '1%'", {"low"}),
        ("tags.`team.name` ILIKE 'risk'", {"low"}),
        ("tags.`team.name` LIKE 'risk'", set()),
        ("tags.`team.name` != 'Risk'", {"high"}),
        ("tags.empty = ''", {"low", "high"}),
        ("tags.empty NOT IN ()", {"low", "high"}),
        ("metrics.loss < 0.1", {"low"}),
        ("metrics.loss <= 0.2", {"low", "high"}),
        ("metrics.loss > 0.6", {"low"}),
        ("metrics.loss >= 0.8", {"low"}),
        ("metrics.loss = 0.4", {"high"}),
        ("metrics.loss != 0.4", {"low", "high"}),
        ("metrics.`score.test` > 0.8", {"low"}),
        ("metrics.loss < 0.1 AND metrics.loss > 0.6", {"low"}),
        ("metrics.loss > 0.1 AND params.depth = '2' AND tags.`team.name` = 'Platform'", {"high"}),
        ("tags.`team.name` = 'Risk' AND tags.`team.name` = 'Platform'", set()),
    ],
)
def test_filters_match_metadata_and_any_history_entry(
    store, searchable_models, expression, expected
):
    experiment_id, _ = searchable_models
    assert (
        set(_names(store.search_logged_models([experiment_id], filter_string=expression)))
        == expected
    )


@pytest.mark.parametrize(
    "datasets,expression,expected",
    [
        ([{"dataset_name": "eval"}], None, {"low", "high"}),
        ([{"dataset_name": "eval", "dataset_digest": "a"}], None, {"low"}),
        (
            [
                {"dataset_name": "eval", "dataset_digest": "a"},
                {"dataset_name": "eval", "dataset_digest": "c"},
            ],
            None,
            {"low", "high"},
        ),
        ([{"dataset_name": "absent"}], None, set()),
        ([{"dataset_name": "eval"}], "metrics.loss > 0.1", {"high"}),
        ([{"dataset_name": "train"}], "metrics.loss > 0.6", {"low"}),
        ([{"dataset_name": "eval", "dataset_digest": "c"}], "params.depth = '10'", set()),
    ],
)
def test_dataset_scopes_apply_to_history_filters(
    store, searchable_models, datasets, expression, expected
):
    experiment_id, _ = searchable_models
    results = store.search_logged_models(
        [experiment_id], filter_string=expression, datasets=datasets
    )
    assert set(_names(results)) == expected


@pytest.mark.parametrize(
    "ascending,expected", [(True, ["high", "low", "missing"]), (False, ["low", "high", "missing"])]
)
def test_metric_sort_uses_latest_timestamp_before_step_and_nulls_last(
    store, searchable_models, ascending, expected
):
    experiment_id, _ = searchable_models
    page = store.search_logged_models(
        [experiment_id], order_by=[{"field_name": "metrics.loss", "ascending": ascending}]
    )
    assert _names(page) == expected


def test_filter_and_sort_scopes_are_independent(store, searchable_models):
    experiment_id, _ = searchable_models
    page = store.search_logged_models(
        [experiment_id],
        filter_string="metrics.loss < 0.3",
        datasets=[{"dataset_name": "eval"}],
        order_by=[{"field_name": "metrics.loss", "dataset_name": "train", "ascending": False}],
    )
    assert _names(page) == ["low", "high"]
    # The result entities retain history from both datasets, not just matching rows.
    assert len(page[0].metrics) == 3
    assert {metric.dataset_name for metric in page[0].metrics} == {"eval", "train"}


def test_metric_order_scope_cannot_be_displaced_by_other_metric_rows(store, searchable_models):
    experiment_id, _ = searchable_models
    page = store.search_logged_models(
        [experiment_id],
        order_by=[
            {"field_name": "metrics.loss", "dataset_name": "eval", "dataset_digest": "a"},
            {"field_name": "metrics.score.test", "ascending": False},
        ],
    )
    assert _names(page)[0] == "low"
    assert set(_names(page)[1:]) == {"high", "missing"}


@pytest.mark.parametrize(
    "ascending,expected", [(True, ["high", "low", "missing"]), (False, ["low", "high", "missing"])]
)
def test_nullable_attribute_order_keeps_nulls_last(store, searchable_models, ascending, expected):
    experiment_id, _ = searchable_models
    assert (
        _names(
            store.search_logged_models(
                [experiment_id], order_by=[{"field_name": "model_type", "ascending": ascending}]
            )
        )
        == expected
    )


def test_default_order_and_pagination_are_deterministic(store, run, clock):
    old = store.create_logged_model(run.info.experiment_id, name="old")
    clock["now"] += 100
    newer = [store.create_logged_model(run.info.experiment_id, name=name) for name in ("a", "b")]
    expected = [*sorted(model.model_id for model in newer), old.model_id]
    first = store.search_logged_models([run.info.experiment_id], max_results=2)
    assert first.token is not None
    second = store.search_logged_models(
        [run.info.experiment_id], max_results=2, page_token=first.token
    )
    assert second.token is None
    assert [model.model_id for model in [*first, *second]] == expected
    assert len(set(expected)) == 3
    ascending = store.search_logged_models(
        [run.info.experiment_id], order_by=[{"field_name": "creation_time"}]
    )
    assert ascending[0].model_id == old.model_id


def test_pagination_is_applied_after_filtering_and_sorting(store, searchable_models):
    experiment_id, _ = searchable_models
    arguments = {
        "filter_string": "metrics.loss > 0.1",
        "max_results": 1,
        "order_by": [{"field_name": "metrics.loss", "ascending": False}],
    }
    first = store.search_logged_models([experiment_id], **arguments)
    second = store.search_logged_models([experiment_id], page_token=first.token, **arguments)
    assert _names(first) == ["low"] and _names(second) == ["high"]
    assert first.token is not None and second.token is None


def test_search_scopes_experiments_and_excludes_deleted_models(store, searchable_models):
    experiment_id, models = searchable_models
    other_id = store.create_experiment("other-model-experiment")
    store.create_logged_model(other_id, name="other")
    store.delete_logged_model(models["high"].model_id)
    assert set(_names(store.search_logged_models([experiment_id]))) == {"low", "missing"}
    assert set(_names(store.search_logged_models([experiment_id, other_id]))) == {
        "low",
        "missing",
        "other",
    }
    assert store.search_logged_models([]) == []
    assert store.search_logged_models(["unknown"]) == []
    assert (
        _names(store.search_logged_models([experiment_id], filter_string="status = 'PENDING'"))
        != []
    )


def test_model_id_membership(store, searchable_models):
    experiment_id, models = searchable_models
    assert _names(
        store.search_logged_models(
            [experiment_id], filter_string=f"model_id IN ('{models['low'].model_id}')"
        )
    ) == ["low"]


@pytest.mark.parametrize(
    "alias",
    [
        "creation_time",
        "creation_timestamp",
        "last_updated_timestamp",
    ],
)
def test_timestamp_filter_aliases(store, searchable_models, clock, alias):
    experiment_id, _ = searchable_models
    assert (
        len(store.search_logged_models([experiment_id], filter_string=f"{alias} = {clock['now']}"))
        == 3
    )


def test_search_returns_complete_entities_without_mutating_documents(store, searchable_models):
    experiment_id, models = searchable_models
    collection = store._database[store._settings.logged_models_collection_name]
    before = collection.find_one({"_id": models["low"].model_id})
    result = store.search_logged_models([experiment_id], filter_string="name = 'low'")[0]
    loaded = store.get_logged_model(result.model_id)
    assert result.to_proto() == loaded.to_proto()
    assert result.params == {"depth": "10"} and result.tags["team.name"] == "Risk"
    assert len(result.metrics) == 3
    assert collection.find_one({"_id": result.model_id}) == before


def test_metric_sort_ties_use_step_then_run_id(store, run):
    models = [
        store.create_logged_model(run.info.experiment_id, name=name) for name in ("first", "second")
    ]
    other_run = store.create_run(run.info.experiment_id, "user", 1000, [], "other")
    smaller, larger = sorted([run.info.run_id, other_run.info.run_id])
    store.log_batch(
        smaller,
        [
            Metric("loss", 0.1, 2000, 2, model_id=models[0].model_id),
            Metric("loss", 0.9, 2000, 1, model_id=models[0].model_id),
            Metric("loss", 0.5, 2000, 1, model_id=models[1].model_id),
        ],
        [],
        [],
    )
    store.log_metric(larger, Metric("loss", 0.8, 2000, 2, model_id=models[0].model_id))
    assert _names(
        store.search_logged_models(
            [run.info.experiment_id], order_by=[{"field_name": "metrics.loss"}]
        )
    ) == ["first", "second"]


@pytest.mark.parametrize(
    "expression", ["metrics.loss LIKE 'x'", "unknown = 'x'", "params.depth > '2'"]
)
def test_invalid_filters_report_validation_error(store, run, expression):
    with pytest.raises(MlflowException) as caught:
        store.search_logged_models([run.info.experiment_id], filter_string=expression)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_dataset_metric_order_ignores_newer_rows_outside_scope(store, searchable_models):
    experiment_id, _ = searchable_models
    page = store.search_logged_models(
        [experiment_id],
        order_by=[{"field_name": "metrics.loss", "dataset_name": "eval", "ascending": True}],
    )
    # Evaluation values are 0.05 and 0.2; newer training rows are 0.8 and 0.4.
    assert _names(page) == ["low", "high", "missing"]


def test_multiple_metric_filters_can_match_different_rows_and_keys(store, searchable_models):
    experiment_id, _ = searchable_models
    page = store.search_logged_models(
        [experiment_id],
        filter_string="metrics.loss < 0.1 AND metrics.`score.test` > 0.8",
        datasets=[{"dataset_name": "eval", "dataset_digest": "a"}],
    )
    assert _names(page) == ["low"]


def test_filtering_and_sorting_by_different_metrics_preserves_latest_sort_value(store, run):
    first = store.create_logged_model(run.info.experiment_id, name="first")
    second = store.create_logged_model(run.info.experiment_id, name="second")
    store.log_batch(
        run.info.run_id,
        [
            Metric("loss", 0.1, 5000, 1, model_id=first.model_id),
            Metric("accuracy", 0.3, 1000, 1, model_id=first.model_id),
            Metric("accuracy", 0.9, 2000, 1, model_id=first.model_id),
            Metric("loss", 0.1, 5000, 1, model_id=second.model_id),
            Metric("accuracy", 0.5, 2000, 1, model_id=second.model_id),
        ],
        [],
        [],
    )
    page = store.search_logged_models(
        [run.info.experiment_id],
        filter_string="metrics.loss < 0.2",
        order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
    )
    assert _names(page) == ["first", "second"]
