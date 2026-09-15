"""Unit tests for model-registry search parsing and store pagination."""

import pytest
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils

from mlflow_mongodb import MongoDBModelRegistryStore
from mlflow_mongodb.repositories import (
    ModelVersionFilter,
    ModelVersionOrder,
    RegisteredModelFilter,
    RegisteredModelOrder,
)


@pytest.mark.parametrize(
    ("filter_string", "expected_filter"),
    [
        (
            "name LIKE 'fraud%'",
            RegisteredModelFilter("attribute", "name", "LIKE", "fraud%"),
        ),
        (
            "tags.team ILIKE 'Plat%'",
            RegisteredModelFilter("tag", "team", "ILIKE", "Plat%"),
        ),
    ],
)
def test_parse_registered_model_filters_adds_prompt_exclusion(
    filter_string,
    expected_filter,
):
    filters = MongoDBModelRegistryStore._parse_registered_model_filters(filter_string)

    assert filters == (
        expected_filter,
        RegisteredModelFilter(
            "tag",
            IS_PROMPT_TAG_KEY,
            "!=",
            "true",
        ),
    )


@pytest.mark.parametrize(
    ("filter_string", "expected_filter"),
    [
        (
            "name ILIKE 'fraud%'",
            ModelVersionFilter("attribute", "name", "ILIKE", "fraud%"),
        ),
        (
            "run_id IN ('run-a', 'run-b')",
            ModelVersionFilter("attribute", "run_id", "IN", ("run-a", "run-b")),
        ),
        (
            "version_number >= 2",
            ModelVersionFilter("attribute", "version_number", ">=", 2),
        ),
        (
            "tags.team LIKE 'risk%'",
            ModelVersionFilter("tag", "team", "LIKE", "risk%"),
        ),
    ],
)
def test_parse_model_version_filters(filter_string, expected_filter):
    filters, exclude_prompts = MongoDBModelRegistryStore._parse_model_version_filters(filter_string)

    assert filters == (expected_filter,)
    assert exclude_prompts is True


@pytest.mark.parametrize(
    ("comparator", "value", "exclude_prompts"),
    [("=", "true", False), ("!=", "false", False), ("=", "false", True), ("!=", "true", True)],
)
def test_parse_model_version_prompt_filter_controls_exclusion(
    comparator,
    value,
    exclude_prompts,
):
    filter_string = f"tags.`{IS_PROMPT_TAG_KEY}` {comparator} '{value}'"

    filters, actual_exclude_prompts = MongoDBModelRegistryStore._parse_model_version_filters(
        filter_string
    )

    assert filters == (ModelVersionFilter("tag", IS_PROMPT_TAG_KEY, comparator, value),)
    assert actual_exclude_prompts is exclude_prompts


@pytest.mark.parametrize(
    ("parser", "invalid_value", "expected_message"),
    [
        (
            MongoDBModelRegistryStore._parse_registered_model_filters,
            "name > 'fraud'",
            "Invalid comparator",
        ),
        (
            MongoDBModelRegistryStore._parse_registered_model_filters,
            "tags.team IN ('risk', 'platform')",
            "Expected a quoted string value",
        ),
        (
            MongoDBModelRegistryStore._parse_model_version_filters,
            "name IN ('fraud', 'credit')",
            "Only the 'run_id' attribute",
        ),
        (
            MongoDBModelRegistryStore._parse_model_version_filters,
            "version_number LIKE '2'",
            "Invalid comparator",
        ),
        (
            MongoDBModelRegistryStore._parse_model_version_filters,
            "unknown_attribute = 'fraud'",
            "Invalid attribute key",
        ),
        (
            MongoDBModelRegistryStore._parse_model_version_filters,
            "name > 'fraud'",
            "Invalid comparator for attribute",
        ),
        (
            MongoDBModelRegistryStore._parse_model_version_filters,
            "tags.team > 'risk'",
            "Invalid comparator for tag",
        ),
    ],
)
def test_filter_parsers_reject_invalid_comparator_contracts(
    parser,
    invalid_value,
    expected_message,
):
    with pytest.raises(MlflowException, match=expected_message) as exc_info:
        parser(invalid_value)

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_registered_model_filter_parser_rejects_unexpected_expression_type(monkeypatch):
    monkeypatch.setattr(
        SearchModelUtils,
        "parse_search_filter",
        lambda _filter_string: [
            {"type": "unexpected", "key": "name", "comparator": "=", "value": "model"}
        ],
    )

    with pytest.raises(
        MlflowException,
        match="Invalid search expression type: unexpected",
    ) as exc_info:
        MongoDBModelRegistryStore._parse_registered_model_filters("name = 'model'")

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_registered_model_order_parser_rejects_unexpected_order_entity(monkeypatch):
    monkeypatch.setattr(
        SearchModelUtils,
        "parse_order_by_for_search_registered_models",
        lambda _order_by: ("tag", "team", True),
    )

    with pytest.raises(
        MlflowException,
        match="Invalid order_by entity: tag",
    ) as exc_info:
        MongoDBModelRegistryStore._parse_registered_model_order(["name ASC"])

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_model_version_filter_parser_rejects_unexpected_token_type(monkeypatch):
    monkeypatch.setattr(
        SearchModelVersionUtils,
        "parse_search_filter",
        lambda _filter_string: [
            {"type": "unexpected", "key": "name", "comparator": "=", "value": "model"}
        ],
    )

    with pytest.raises(
        MlflowException,
        match="Invalid token type: unexpected",
    ) as exc_info:
        MongoDBModelRegistryStore._parse_model_version_filters("name = 'model'")

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_model_version_filter_parser_rejects_unsupported_attribute(monkeypatch):
    monkeypatch.setattr(
        SearchModelVersionUtils,
        "parse_search_filter",
        lambda _filter_string: [
            {
                "type": "attribute",
                "key": "unknown_attribute",
                "comparator": "=",
                "value": "model",
            }
        ],
    )

    with pytest.raises(
        MlflowException,
        match="Invalid attribute name: unknown_attribute",
    ) as exc_info:
        MongoDBModelRegistryStore._parse_model_version_filters("name = 'model'")

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_model_version_order_parser_rejects_unexpected_order_entity(monkeypatch):
    monkeypatch.setattr(
        SearchModelVersionUtils,
        "parse_order_by_for_search_model_versions",
        lambda _order_by: ("tag", "team", True),
    )

    with pytest.raises(
        MlflowException,
        match="Invalid order_by entity: tag",
    ) as exc_info:
        MongoDBModelRegistryStore._parse_model_version_order(["name ASC"])

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_model_version_order_parser_rejects_unsupported_order_key(monkeypatch):
    monkeypatch.setattr(
        SearchModelVersionUtils,
        "parse_order_by_for_search_model_versions",
        lambda _order_by: ("attribute", "source_path", True),
    )

    with pytest.raises(
        MlflowException,
        match="Invalid order by key 'source_path'",
    ) as exc_info:
        MongoDBModelRegistryStore._parse_model_version_order(["source_path ASC"])

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_parse_registered_model_order_adds_deterministic_name_tiebreaker():
    assert MongoDBModelRegistryStore._parse_registered_model_order(None) == (
        RegisteredModelOrder("name", True),
    )
    assert MongoDBModelRegistryStore._parse_registered_model_order(["name DESC"]) == (
        RegisteredModelOrder("name", False),
    )


def test_parse_registered_model_order_normalizes_timestamp_alias():
    assert MongoDBModelRegistryStore._parse_registered_model_order(["timestamp DESC"]) == (
        RegisteredModelOrder("last_updated_timestamp", False),
        RegisteredModelOrder("name", True),
    )

    with pytest.raises(MlflowException, match="duplicate fields") as exc_info:
        MongoDBModelRegistryStore._parse_registered_model_order(
            [
                "timestamp ASC",
                "last_updated_timestamp DESC",
            ]
        )

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_parse_model_version_order_adds_deterministic_tiebreakers():
    assert MongoDBModelRegistryStore._parse_model_version_order(None) == (
        ModelVersionOrder("last_updated_timestamp", False),
        ModelVersionOrder("name", True),
        ModelVersionOrder("version_number", False),
    )
    assert MongoDBModelRegistryStore._parse_model_version_order(["creation_timestamp ASC"]) == (
        ModelVersionOrder("creation_timestamp", True),
        ModelVersionOrder("name", True),
        ModelVersionOrder("version_number", False),
    )


@pytest.mark.parametrize(
    ("parser", "invalid_order", "expected_message"),
    [
        (
            MongoDBModelRegistryStore._parse_registered_model_order,
            ["name ASC", "name DESC"],
            "duplicate fields",
        ),
        (
            MongoDBModelRegistryStore._parse_model_version_order,
            ["name ASC", "name DESC"],
            "duplicate fields",
        ),
        (
            MongoDBModelRegistryStore._parse_model_version_order,
            ["source_path ASC"],
            "Invalid attribute key",
        ),
        (
            MongoDBModelRegistryStore._parse_registered_model_order,
            ["name ASC", "creation_timestamp DESC"],
            "Invalid order by key",
        ),
        (
            MongoDBModelRegistryStore._parse_registered_model_order,
            ["name ASC", "last_updated_timestamp DESC extra"],
            "Invalid order_by clause",
        ),
        (
            MongoDBModelRegistryStore._parse_model_version_order,
            ["name ASC", "run_id DESC"],
            "Invalid attribute key",
        ),
        (
            MongoDBModelRegistryStore._parse_model_version_order,
            ["name ASC", "last_updated_timestamp DESC extra"],
            "Invalid order_by clause",
        ),
    ],
)
def test_order_parsers_reject_invalid_contracts(parser, invalid_order, expected_message):
    with pytest.raises(MlflowException, match=expected_message) as exc_info:
        parser(invalid_order)

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("max_results", [0, -1, 1.5, "1"])
@pytest.mark.parametrize("method_name", ["search_registered_models", "search_model_versions"])
def test_search_rejects_non_positive_or_non_integer_page_sizes(
    store,
    method_name,
    max_results,
):
    with pytest.raises(MlflowException, match="positive integer") as exc_info:
        getattr(store, method_name)(max_results=max_results)

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("method_name", ["search_registered_models", "search_model_versions"])
def test_search_rejects_negative_page_offset(store, method_name):
    with pytest.raises(MlflowException, match="offset must be non-negative") as exc_info:
        getattr(store, method_name)(page_token=SearchUtils.create_page_token(-1))

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_search_model_versions_rejects_malformed_filters(store):
    for malformed_filter in (
        "run_id IN (1, 2, 3)",
        "run_id IN ()",
        "run_id IN (",
        "run_id IN",
        "run_id IN (,)",
        "run_id IN ('run-1',, 'run-2')",
        "name LIKE",
    ):
        with pytest.raises(
            MlflowException,
            match="While parsing a list|Invalid clause",
        ) as invalid_filter_error:
            store.search_model_versions(malformed_filter)

        assert invalid_filter_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_search_registered_models_rejects_malformed_filters(store):
    for malformed_filter in (
        "name != unquoted",
        "run_id = 'run-id'",
        "source_path = 'A/D'",
        "unknown = true",
    ):
        with pytest.raises(
            MlflowException,
            match="not quoted|Invalid attribute key|Invalid clause",
        ) as invalid_filter_error:
            store.search_registered_models(malformed_filter)

        assert invalid_filter_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("method_name", ["search_registered_models", "search_model_versions"])
def test_search_rejects_invalid_tokens_and_excessive_page_sizes(store, method_name):
    method = getattr(store, method_name)

    with pytest.raises(MlflowException, match="Invalid page token") as token_error:
        method(page_token="not-a-page-token")  # ruff: ignore[hardcoded-password-func-arg]
    assert token_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(
        MlflowException,
        match="Invalid value.*max_results",
    ) as page_size_error:
        method(max_results=10**15)
    assert page_size_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
