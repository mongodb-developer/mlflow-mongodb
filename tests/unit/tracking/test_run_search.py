"""Run search validation, pagination, and persistence boundaries."""

from unittest.mock import MagicMock

import pytest
from bson.errors import InvalidBSON, InvalidDocument
from mlflow.entities import LifecycleStage, ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.utils.search_utils import SearchUtils
from pymongo.errors import OperationFailure

import mlflow_mongodb.tracking.store as tracking_store
from mlflow_mongodb.infrastructure.errors import RepositoryPersistenceError
from mlflow_mongodb.tracking.repositories.runs import RunFilter, RunOrder, RunRepository


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("metrics.loss >= 0.5", RunFilter("metric", "loss", ">=", 0.5)),
        ("params.depth = '3'", RunFilter("parameter", "depth", "=", "3")),
        ("tags.`team.name` ILIKE 'Risk%'", RunFilter("tag", "team.name", "ILIKE", "Risk%")),
        ("attributes.created >= 1000", RunFilter("attribute", "start_time", ">=", 1000.0)),
        ("attributes.`run name` = 'sample'", RunFilter("attribute", "run_name", "=", "sample")),
        ("attributes.run_id IN ('a', 'b')", RunFilter("attribute", "run_id", "IN", ("a", "b"))),
        ("datasets.name = 'training'", RunFilter("dataset", "name", "=", "training")),
        ("datasets.context != 'evaluation'", RunFilter("dataset", "context", "!=", "evaluation")),
    ],
)
def test_search_parses_and_normalizes_filters(run_store, expression, expected):
    assert run_store._parse_run_filters(expression) == (expected,)


def test_search_preserves_literal_embedded_keys(run_store):
    assert run_store._parse_run_filters("params.created = 'literal'") == (
        RunFilter("parameter", "created", "=", "literal"),
    )


@pytest.mark.parametrize("expression", [None, ""])
def test_empty_filter_has_no_clauses(run_store, expression):
    assert run_store._parse_run_filters(expression) == ()


@pytest.mark.parametrize("field_type", ["parameter", "tag"])
@pytest.mark.parametrize("comparator", ["IS NULL", "IS NOT NULL"])
def test_existence_filter_follows_installed_mlflow_capabilities(run_store, field_type, comparator):
    allowed = (
        SearchUtils.VALID_PARAM_COMPARATORS
        if field_type == "parameter"
        else SearchUtils.VALID_TAG_COMPARATORS
    )
    expression = f"{field_type}.optional {comparator}"
    if comparator in allowed:
        assert run_store._parse_run_filters(expression) == (
            RunFilter(field_type, "optional", comparator, None),
        )
    else:
        with pytest.raises(MlflowException):
            run_store._parse_run_filters(expression)


@pytest.mark.parametrize(
    ("parsed", "message"),
    [
        ({"type": "unknown", "key": "x", "comparator": "=", "value": "x"}, "expression type"),
        (
            {"type": "attribute", "key": "unknown", "comparator": "=", "value": "x"},
            "attribute name",
        ),
        ({"type": "metric", "key": "loss", "comparator": "LIKE", "value": "x"}, "comparator"),
        ({"type": "dataset", "key": "unknown", "comparator": "=", "value": "x"}, "dataset key"),
    ],
)
def test_search_rejects_invalid_clauses_before_persistence(
    run_store, run_repository, monkeypatch, parsed, message
):
    monkeypatch.setattr(SearchUtils, "parse_search_filter", lambda _expression: [parsed])
    with pytest.raises(MlflowException, match=message) as caught:
        run_store.search_runs(["42"], "ignored", ViewType.ALL)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    run_repository.search.assert_not_called()


@pytest.mark.parametrize("max_results", [0, -1, True, 1.5, "2", SEARCH_MAX_RESULTS_THRESHOLD + 1])
def test_search_rejects_invalid_page_sizes(run_store, run_repository, max_results):
    with pytest.raises(MlflowException, match="max_results") as caught:
        run_store.search_runs(["42"], "", ViewType.ALL, max_results=max_results)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    run_repository.search.assert_not_called()


@pytest.mark.parametrize(
    ("view", "stages"),
    [
        (ViewType.ACTIVE_ONLY, [LifecycleStage.ACTIVE]),
        (ViewType.DELETED_ONLY, [LifecycleStage.DELETED]),
        (ViewType.ALL, [LifecycleStage.ACTIVE, LifecycleStage.DELETED]),
    ],
)
def test_search_scopes_normalized_experiment_ids_and_lifecycle(
    run_store, run_repository, view, stages
):
    run_repository.search.return_value = []
    page = run_store.search_runs([42, "43"], "", view, max_results=2)
    run_repository.search.assert_called_once_with(
        experiment_ids=["42", "43"],
        lifecycle_stages=stages,
        filters=(),
        order_by=(
            RunOrder("attribute", "start_time", False),
            RunOrder("attribute", "run_id", True),
        ),
        offset=0,
        limit=3,
    )
    assert page == []
    assert page.token is None


def test_search_paginates_and_returns_complete_entities(
    run_store, run_repository, run_record_factory
):
    record = run_record_factory(
        params=[{"k": "depth", "v": "3"}],
        metrics=[{"k": "loss", "v": 0.4, "timestamp": 1000, "step": 1}],
        inputs={
            "datasets": [
                {
                    "dataset": {
                        "name": "training",
                        "digest": "abc",
                        "source_type": "test",
                        "source": "{}",
                    },
                    "tags": [],
                }
            ],
            "models": [{"model_id": "m-input"}],
        },
        outputs=[{"model_id": "m-output", "step": 1}],
    )
    run_repository.search.return_value = [record, run_record_factory(), run_record_factory()]
    page = run_store.search_runs(
        ["42"],
        "",
        ViewType.ALL,
        max_results=2,
        page_token=SearchUtils.create_page_token(4),
        order_by=["metrics.loss ASC"],
    )
    assert len(page) == 2
    assert page.token == SearchUtils.create_page_token(6)
    assert page[0].data.params == {"depth": "3"}
    assert page[0].data.metrics == {"loss": 0.4}
    assert page[0].data.tags == {"team": "risk"}
    assert page[0].inputs.dataset_inputs[0].dataset.digest == "abc"
    assert page[0].inputs.model_inputs[0].model_id == "m-input"
    assert page[0].outputs.model_outputs[0].model_id == "m-output"
    assert run_repository.search.call_args.kwargs["offset"] == 4
    assert run_repository.search.call_args.kwargs["order_by"][0] == RunOrder("metric", "loss", True)


@pytest.mark.parametrize("max_results", [2, None])
def test_search_without_extra_result_has_no_next_token(
    run_store, run_repository, run_record_factory, max_results
):
    run_repository.search.return_value = [run_record_factory(), run_record_factory()]
    page = run_store.search_runs(["42"], "", ViewType.ALL, max_results=max_results)
    assert len(page) == 2
    assert page.token is None
    assert run_repository.search.call_args.kwargs["limit"] == (3 if max_results == 2 else None)


@pytest.mark.parametrize("token", ["invalid-base64", SearchUtils.create_page_token(-1)])
def test_search_rejects_invalid_page_token(run_store, run_repository, token):
    with pytest.raises(MlflowException) as caught:
        run_store.search_runs(["42"], "", ViewType.ALL, page_token=token)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    run_repository.search.assert_not_called()


@pytest.mark.parametrize(
    "orders",
    [
        ["metrics.loss ASC", "metric.loss DESC"],
        ["start_time DESC", "created ASC"],
        ["datasets.name ASC"],
    ],
)
def test_search_rejects_duplicate_or_unsupported_ordering(run_store, run_repository, orders):
    with pytest.raises(MlflowException):
        run_store.search_runs(["42"], "", ViewType.ALL, order_by=orders)
    run_repository.search.assert_not_called()


def test_search_sanitizes_persistence_errors(run_store, run_repository, caplog):
    error = RepositoryPersistenceError("Private URI and query")
    run_repository.search.side_effect = error
    with pytest.raises(MlflowException) as caught:
        run_store.search_runs(["42"], "", ViewType.ALL)
    assert caught.value.message == "Unable to search runs."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    records = [record for record in caplog.records if record.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].args == (error,)
    assert records[0].exc_info is None


def test_search_initialization_failure_is_logged_once(run_store, monkeypatch, caplog):
    error = RepositoryPersistenceError("Private initialization details")
    run_store.__dict__.pop("_run_repository")
    run_store.__dict__["_database"] = MagicMock()
    initialize = MagicMock(side_effect=error)
    monkeypatch.setattr(tracking_store, "RunRepository", initialize)
    with pytest.raises(MlflowException) as caught:
        run_store.search_runs(["42"], "", ViewType.ALL)
    assert caught.value.message == "Unable to initialize run repository."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    records = [record for record in caplog.records if record.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].args == (error,)


@pytest.fixture
def search_collection():
    return MagicMock()


@pytest.fixture
def search_repository(search_collection):
    database = MagicMock()
    database.__getitem__.return_value = search_collection
    return RunRepository(database)


def _repository_search(repository, **overrides):
    arguments = {
        "experiment_ids": ["42"],
        "lifecycle_stages": [LifecycleStage.ACTIVE],
        "filters": (),
        "order_by": (
            RunOrder("attribute", "start_time", False),
            RunOrder("attribute", "run_id", True),
        ),
        "offset": 0,
        "limit": 3,
    }
    arguments.update(overrides)
    return repository.search(**arguments)


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidDocument])
def test_repository_search_translates_aggregate_failures(
    search_repository, search_collection, error_type
):
    error = error_type("Private aggregate details")
    search_collection.aggregate.side_effect = error
    with pytest.raises(RepositoryPersistenceError) as caught:
        _repository_search(search_repository)
    assert str(caught.value) == "Unable to search runs."
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("error_type", [OperationFailure, InvalidBSON])
def test_repository_search_consumes_cursor_inside_error_boundary(
    search_repository, search_collection, run_document_factory, error_type
):
    error = error_type("Private cursor details")

    def documents():
        yield run_document_factory()
        raise error

    search_collection.aggregate.return_value = documents()
    with pytest.raises(RepositoryPersistenceError) as caught:
        _repository_search(search_repository)
    assert caught.value.__cause__ is error


def test_repository_search_keeps_record_conversion_outside_error_boundary(
    search_repository, search_collection
):
    search_collection.aggregate.return_value = [{}]
    with pytest.raises(KeyError):
        _repository_search(search_repository)


def test_repository_search_does_not_modify_database_documents(
    search_repository, search_collection, run_document_factory
):
    document = run_document_factory(params=[{"k": "depth", "v": "3"}])
    search_collection.aggregate.return_value = [document]
    record = _repository_search(search_repository)[0]
    assert record.params[0].value == "3"
    assert document["params"] == [{"k": "depth", "v": "3"}]
    search_collection.update_one.assert_not_called()
