"""Logged-model search parsing, validation, and pagination at the store boundary."""

import base64
import json

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.utils.search_utils import SearchLoggedModelsPaginationToken

from mlflow_mongodb.infrastructure.errors import RepositoryInvalidAttributeError
from mlflow_mongodb.tracking.repositories import (
    LoggedModelFilter,
    LoggedModelOrder,
    LoggedModelPage,
    LoggedModelSearchResult,
)
from mlflow_mongodb.tracking.types import RunMetricRecord


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("name = 'model'", LoggedModelFilter("attribute", "name", "=", "model")),
        (
            "creation_time >= 1000",
            LoggedModelFilter("attribute", "creation_timestamp", ">=", 1000.0),
        ),
        (
            "creation_timestamp >= 1000",
            LoggedModelFilter("attribute", "creation_timestamp", ">=", 1000.0),
        ),
        (
            "last_updated_timestamp < 2000",
            LoggedModelFilter("attribute", "last_updated_timestamp", "<", 2000.0),
        ),
        ("metrics.`loss.test` > 0.1", LoggedModelFilter("metric", "loss.test", ">", 0.1)),
        ("tags.`team.name` ILIKE 'risk%'", LoggedModelFilter("tag", "team.name", "ILIKE", "risk%")),
        ("params.depth IN ('3', '4')", LoggedModelFilter("param", "depth", "IN", ("3", "4"))),
        ("tags.team NOT IN ('risk')", LoggedModelFilter("tag", "team", "NOT IN", ("risk",))),
    ],
)
def test_filters_are_parsed_into_repository_clauses(
    logged_model_store, logged_model_repository, expression, expected
):
    logged_model_store.search_logged_models(["42"], filter_string=expression)
    assert logged_model_repository.search.call_args.kwargs["filters"] == (expected,)


@pytest.mark.parametrize(
    "expression",
    [
        "metrics.loss > nan",
        "metrics.loss > inf",
        "metrics.loss LIKE 'x'",
        "params.depth > '3'",
        "name = 'a' OR name = 'b'",
    ],
)
def test_invalid_filters_are_rejected_before_repository_call(
    logged_model_store, logged_model_repository, expression
):
    with pytest.raises(MlflowException) as caught:
        logged_model_store.search_logged_models(["42"], filter_string=expression)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    logged_model_repository.search.assert_not_called()


@pytest.mark.parametrize(
    "expression,error_type",
    [
        ("metrics.loss = 'abc'", ValueError),
        ("params.depth IN (unknown)", ValueError),
        ("params.depth IN (1,,2)", SyntaxError),
    ],
)
def test_search_rejects_invalid_filter_literals(
    logged_model_store, logged_model_repository, caplog, expression, error_type
):
    with pytest.raises(MlflowException) as caught:
        logged_model_store.search_logged_models(["42"], filter_string=expression)

    assert caught.value.message == "Invalid logged model filter string."
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    logged_model_repository.search.assert_not_called()
    records = [
        record for record in caplog.records if record.name == "mlflow_mongodb.tracking.store"
    ]
    assert len(records) == 1
    assert records[0].msg == "Unable to parse logged model filter: %s"
    assert isinstance(records[0].args[0], error_type)
    assert records[0].exc_info is None


def test_search_rejects_invalid_attributes(logged_model_store, logged_model_repository, caplog):
    with pytest.raises(MlflowException) as caught:
        logged_model_store.search_logged_models(
            ["42"], filter_string="unsupported_attribute = 'value'"
        )

    assert caught.value.message == "Invalid logged model attribute: 'unsupported_attribute'."
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    logged_model_repository.search.assert_not_called()
    records = [
        record for record in caplog.records if record.name == "mlflow_mongodb.tracking.store"
    ]
    assert len(records) == 1
    assert records[0].msg == "Unable to validate logged model filter: %s"
    assert isinstance(records[0].args[0], RepositoryInvalidAttributeError)
    assert records[0].args[0].key == "unsupported_attribute"
    assert records[0].exc_info is None


def test_empty_tag_key_is_rejected(logged_model_store, logged_model_repository):
    with pytest.raises(MlflowException, match="Search keys must not be empty") as caught:
        logged_model_store.search_logged_models(["42"], filter_string="tags.`` = 'value'")
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    logged_model_repository.search.assert_not_called()


@pytest.mark.parametrize(
    "expression,message",
    [
        ("last_updated_time = 2000", "Numeric filters require finite numbers"),
        ("params.depth IN (1, 2)", "IN and NOT IN require a list of string values"),
        ("params.depth IN (1)", "IN and NOT IN require a list of string values"),
        ("tags.team = (1)", "String filters require string values"),
    ],
)
def test_invalid_filter_values_are_rejected(
    logged_model_store, logged_model_repository, expression, message
):
    with pytest.raises(MlflowException, match=message) as caught:
        logged_model_store.search_logged_models(["42"], filter_string=expression)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    logged_model_repository.search.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"experiment_ids": "42"},
        {"experiment_ids": [42]},
        {"datasets": "eval"},
        {"datasets": [{}]},
        {"datasets": ["eval"]},
        {"datasets": [{"dataset_name": 1}]},
        {"datasets": [{"dataset_name": "eval", "dataset_digest": 1}]},
        {"max_results": True},
        {"max_results": -1},
        {"max_results": 1.5},
        {"filter_string": 1},
        {"order_by": "name"},
        {"order_by": [{}]},
        {"order_by": [{"field_name": "params.depth"}]},
        {"order_by": [{"field_name": "unknown"}]},
        {"order_by": [{"field_name": "metrics."}]},
        {"order_by": [{"field_name": "name", "ascending": "yes"}]},
        {"order_by": [{"field_name": "name", "dataset_name": "eval"}]},
        {"order_by": [{"field_name": "metrics.loss", "dataset_digest": "abc"}]},
        {"order_by": [{"field_name": "metrics.loss", "dataset_name": 1}]},
        {"page_token": 1},
        {"page_token": "not-base64"},
    ],
)
def test_invalid_search_arguments_do_not_reach_repository(
    logged_model_store, logged_model_repository, kwargs
):
    arguments = {"experiment_ids": ["42"], **kwargs}
    with pytest.raises(MlflowException) as caught:
        logged_model_store.search_logged_models(**arguments)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    logged_model_repository.search.assert_not_called()


def test_order_preserves_precedence_scope_and_default_tie_breakers(
    logged_model_store, logged_model_repository
):
    logged_model_store.search_logged_models(
        ["42"],
        order_by=[
            {
                "field_name": "metrics.loss",
                "ascending": False,
                "dataset_name": "eval",
                "dataset_digest": "abc",
            },
            {"field_name": "name"},
            {"field_name": "name", "ascending": False},
        ],
    )
    assert logged_model_repository.search.call_args.kwargs["order_by"] == (
        LoggedModelOrder("metric", "loss", False, "eval", "abc"),
        LoggedModelOrder("attribute", "name", True),
        LoggedModelOrder("attribute", "creation_timestamp", False),
        LoggedModelOrder("attribute", "model_id", True),
    )


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("creation_time", "creation_timestamp"),
        ("last_updated_time", "last_updated_timestamp"),
    ],
)
def test_order_normalizes_timestamp_aliases(
    logged_model_store, logged_model_repository, alias, canonical
):
    logged_model_store.search_logged_models(["42"], order_by=[{"field_name": alias}])
    assert logged_model_repository.search.call_args.kwargs["order_by"][0] == LoggedModelOrder(
        "attribute", canonical, True
    )


def test_excessive_sort_keys_are_rejected(logged_model_store, logged_model_repository):
    with pytest.raises(MlflowException, match="Too many"):
        logged_model_store.search_logged_models(
            ["42"], order_by=[{"field_name": f"metrics.metric_{index}"} for index in range(16)]
        )
    logged_model_repository.search.assert_not_called()


def test_empty_experiment_scope_avoids_repository(logged_model_store, logged_model_repository):
    page = logged_model_store.search_logged_models([])
    assert page == [] and page.token is None
    logged_model_repository.search.assert_not_called()


def test_search_returns_complete_entities_and_next_offset(
    logged_model_store, logged_model_repository, logged_model_record_factory
):
    record = logged_model_record_factory()
    metrics = (RunMetricRecord("loss", 0.4, 1000, 1, record.model_id, "eval", "abc", "run"),)
    logged_model_repository.search.return_value = LoggedModelPage(
        (LoggedModelSearchResult(record, metrics),), True
    )
    order = [{"field_name": "metrics.loss", "ascending": False}]
    page = logged_model_store.search_logged_models(
        ["42"], filter_string="metrics.loss > 0", max_results=1, order_by=order
    )
    assert page[0].tags == {"team": "risk"}
    assert page[0].params == {"depth": "3"}
    assert page[0].metrics[0].dataset_digest == "abc"
    token = SearchLoggedModelsPaginationToken.decode(page.token)
    assert token.offset == 1
    assert token.experiment_ids == ["42"]
    assert token.filter_string == "metrics.loss > 0"
    assert token.order_by == order
    logged_model_store.search_logged_models(
        ["42"],
        filter_string=token.filter_string,
        max_results=1,
        order_by=order,
        page_token=page.token,
    )
    assert logged_model_repository.search.call_args.kwargs["offset"] == 1


@pytest.mark.parametrize("offset", [-1, True, "1", 2**63])
def test_invalid_token_offsets_are_rejected(logged_model_store, logged_model_repository, offset):
    token = base64.b64encode(
        json.dumps({"experiment_ids": ["42"], "offset": offset}).encode()
    ).decode()
    with pytest.raises(MlflowException):
        logged_model_store.search_logged_models(["42"], page_token=token)
    logged_model_repository.search.assert_not_called()


@pytest.mark.parametrize(
    "token_arguments",
    [
        {"experiment_ids": ["other"]},
        {"experiment_ids": ["42"], "filter_string": "name = 'other'"},
        {"experiment_ids": ["42"], "order_by": [{"field_name": "name"}]},
    ],
)
def test_token_must_match_request(logged_model_store, logged_model_repository, token_arguments):
    token = SearchLoggedModelsPaginationToken(**token_arguments).encode()
    with pytest.raises(MlflowException) as caught:
        logged_model_store.search_logged_models(["42"], page_token=token)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    logged_model_repository.search.assert_not_called()
