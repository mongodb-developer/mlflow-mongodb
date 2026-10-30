"""Evaluation-dataset search compatibility until dataset storage is implemented."""

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.tracking import MlflowClient
from mlflow.utils.search_utils import SearchUtils

import mlflow_mongodb.tracking.store as store_module

pytestmark = pytest.mark.skipif(
    store_module.SearchEvaluationDatasetsUtils is None,
    reason="This MLflow version has no evaluation-dataset search parser.",
)


@pytest.mark.parametrize(
    "filter_string",
    [
        None,
        "",
        "name = 'evaluation'",
        "name != 'other'",
        "name LIKE 'eval%'",
        "name > 'evaluation'",
        "name >= 'evaluation'",
        "name < 'evaluation'",
        "name <= 'evaluation'",
        "created_by = 'reviewer'",
        "created_by > 'reviewer'",
        "last_updated_by = 'reviewer'",
        "last_updated_by < 'reviewer'",
        "created_time >= 1000",
        "created_time LIKE 1000",
        "created_time ILIKE 1000",
        "last_update_time < 2000",
        "last_update_time LIKE 2000",
        "last_update_time ILIKE 2000",
        "tags.team = '$risk'",
        "tags.team ILIKE 'RISK%'",
        "tags.created_time = 'literal'",
        "name = 'evaluation' AND tags.team = 'risk' AND created_time >= 1000",
    ],
)
def test_valid_search_returns_empty_dataset_page(store, filter_string):
    page = store.search_datasets(filter_string=filter_string)

    assert isinstance(page, PagedList)
    assert page == []
    assert page.token is None


@pytest.mark.parametrize("experiment_ids", [None, [], ["missing-experiment"]])
def test_empty_dataset_page_does_not_require_experiments(store, experiment_ids):
    page = store.search_datasets(experiment_ids=experiment_ids)

    assert page == []
    assert page.token is None


@pytest.mark.parametrize(
    "order_by",
    [["name ASC"], ["created_time DESC"], ["last_update_time ASC", "name DESC"]],
)
def test_valid_ordering_and_positive_page_offset_return_empty_page(store, order_by):
    page = store.search_datasets(
        order_by=order_by,
        page_token=SearchUtils.create_page_token(5),
        max_results=1,
    )

    assert page == []
    assert page.token is None


@pytest.mark.parametrize("max_results", [0, -1, True, 1.5, "2", SEARCH_MAX_RESULTS_THRESHOLD + 1])
def test_invalid_page_size_is_rejected(store, max_results):
    with pytest.raises(MlflowException) as caught:
        store.search_datasets(max_results=max_results)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert "max_results" in caught.value.message


@pytest.mark.parametrize(
    "expression",
    ["unknown = 'value'", "attributes.name = 'value'", "name =", "created_time >= 'yesterday'"],
)
def test_invalid_filter_syntax_and_fields_are_rejected(store, expression):
    with pytest.raises(MlflowException) as caught:
        store.search_datasets(filter_string=expression)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize(
    "comparator",
    [
        ">",
        ">=",
        "<",
        "<=",
    ],
)
def test_validator_rejects_unsupported_tag_comparators(store, comparator):
    with pytest.raises(MlflowException) as caught:
        store.search_datasets(filter_string=f"tags.team {comparator} 'risk'")

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.message == (f"Invalid comparator for evaluation dataset tag: {comparator}")
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_tag_ordering_is_rejected(store):
    with pytest.raises(MlflowException) as caught:
        store.search_datasets(order_by=["tags.team ASC"])

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.message == "Invalid order_by entity: tag"


def test_negative_page_offset_is_rejected(store):
    with pytest.raises(MlflowException) as caught:
        store.search_datasets(page_token=SearchUtils.create_page_token(-1))

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.message == "Page offset must not be negative."


@pytest.mark.parametrize("page_token", ["not-a-page-token", "e30="])
def test_malformed_page_token_is_rejected(store, page_token):
    with pytest.raises(MlflowException) as caught:
        store.search_datasets(page_token=page_token)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.skipif(
    not hasattr(MlflowClient, "search_datasets"),
    reason="This MLflow client has no evaluation-dataset search API.",
)
def test_client_can_search_datasets_and_receive_empty_page(store):
    client = MlflowClient(tracking_uri=store.store_uri)

    page = client.search_datasets(filter_string="name LIKE 'eval%'", max_results=10)

    assert isinstance(page, PagedList)
    assert page == []
    assert page.token is None
