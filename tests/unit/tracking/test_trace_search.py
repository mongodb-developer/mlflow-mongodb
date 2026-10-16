"""Trace search parsing, query construction, and pagination contracts."""

import re

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey
from mlflow.utils.search_utils import SearchTraceUtils

from mlflow_mongodb.infrastructure.errors import RepositoryPersistenceError
from mlflow_mongodb.tracking._compat import (  # ruff: ignore[import-private-name]
    LINKED_PROMPTS_TAG_KEY,
    TRACE_COST_METADATA,
)
from mlflow_mongodb.tracking.repositories.traces import (
    TraceRepository,
    TraceSearchFilter,
    TraceSearchOrder,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("status = 'OK'", TraceSearchFilter("attribute", "status", "=", "OK")),
        ("timestamp_ms >= 1000", TraceSearchFilter("attribute", "timestamp_ms", ">=", 1000.0)),
        pytest.param(
            "tags.team LIKE 'Risk%'",
            TraceSearchFilter("tag", "team", "LIKE", "Risk%"),
            marks=pytest.mark.skipif(
                "LIKE" not in SearchTraceUtils.VALID_TAG_COMPARATORS,
                reason="Installed MLflow does not support LIKE in trace tags",
            ),
        ),
        ("name = 'agent'", TraceSearchFilter("tag", TraceTagKey.TRACE_NAME, "=", "agent")),
        (
            "run_id = 'run'",
            TraceSearchFilter("request_metadata", TraceMetadataKey.SOURCE_RUN, "=", "run"),
        ),
        pytest.param(
            "prompt = 'greeting/1'",
            TraceSearchFilter("tag", LINKED_PROMPTS_TAG_KEY, "=", "greeting/1"),
            marks=pytest.mark.skipif(
                "prompt" not in SearchTraceUtils.SEARCH_KEY_TO_TAG,
                reason="Installed MLflow has no prompt search alias",
            ),
        ),
    ],
)
def test_parser_normalizes_trace_aliases(trace_store, expression, expected):
    assert trace_store._parse_trace_filters(expression) == [expected]


@pytest.mark.parametrize("expression", [None, ""])
def test_empty_filter_has_no_clauses(trace_store, expression):
    assert trace_store._parse_trace_filters(expression) == []


@pytest.mark.parametrize(
    "clause",
    [
        {"type": "unknown", "key": "x", "comparator": "=", "value": "x"},
        {"type": "attribute", "key": "unknown", "comparator": "=", "value": "x"},
        {"type": "tag", "key": "team", "comparator": ">", "value": "x"},
        {"type": "tag", "key": "team", "comparator": "RLIKE", "value": "["},
        {"type": "tag", "key": LINKED_PROMPTS_TAG_KEY, "comparator": "!=", "value": "name/1"},
        {"type": "tag", "key": LINKED_PROMPTS_TAG_KEY, "comparator": "=", "value": "name"},
        {
            "type": "request_metadata",
            "key": TraceMetadataKey.TOKEN_USAGE,
            "comparator": "LIKE",
            "value": "%",
        },
    ],
)
def test_invalid_clauses_fail_before_repository_call(
    trace_store, trace_repository, monkeypatch, clause
):
    monkeypatch.setattr(
        SearchTraceUtils, "parse_search_filter_for_search_traces", lambda _: [clause]
    )
    with pytest.raises(MlflowException) as caught:
        trace_store.search_traces(experiment_ids=["42"], filter_string="ignored")
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    trace_repository.search_trace_infos.assert_not_called()


@pytest.mark.parametrize("max_results", [0, -1, True, 1.5, "2", SEARCH_MAX_RESULTS_THRESHOLD + 1])
def test_invalid_page_size_fails_before_repository_call(trace_store, trace_repository, max_results):
    with pytest.raises(MlflowException) as caught:
        trace_store.search_traces(experiment_ids=["42"], max_results=max_results)
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    trace_repository.search_trace_infos.assert_not_called()


def test_duplicate_order_fields_fail_before_repository_call(trace_store, trace_repository):
    with pytest.raises(MlflowException, match="duplicate"):
        trace_store.search_traces(
            experiment_ids=["42"], order_by=["timestamp_ms ASC", "timestamp_ms DESC"]
        )
    trace_repository.search_trace_infos.assert_not_called()


@pytest.mark.parametrize(
    "order,field_type",
    [
        pytest.param(
            order,
            field_type,
            marks=pytest.mark.skipif(
                field_type not in SearchTraceUtils._VALID_IDENTIFIERS,
                reason="Installed MLflow rejects this identifier before store validation",
            ),
        )
        for order, field_type in (
            ("span.name ASC", "span"),
            ("feedback.quality ASC", "feedback"),
            ("expectation.answer ASC", "expectation"),
            ("issue.id ASC", "issue"),
        )
    ],
)
def test_unsupported_ordering_field_is_rejected_before_repository_call(
    trace_store, trace_repository, order, field_type
):
    with pytest.raises(MlflowException) as caught:
        trace_store.search_traces(experiment_ids=["42"], order_by=[order])

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.message == f"Invalid trace ordering field: {field_type}"
    trace_repository.search_trace_infos.assert_not_called()


@pytest.mark.parametrize("key", [TraceMetadataKey.TOKEN_USAGE, TRACE_COST_METADATA])
def test_reserved_metadata_ordering_is_rejected(trace_store, trace_repository, key):
    with pytest.raises(MlflowException, match="reserved metadata"):
        trace_store.search_traces(experiment_ids=["42"], order_by=[f"request_metadata.`{key}` ASC"])
    trace_repository.search_trace_infos.assert_not_called()


def test_search_passes_scoped_filters_orders_and_offset(
    trace_store, trace_repository, trace_record_factory
):
    trace_repository.search_trace_infos.return_value = [trace_record_factory()]
    token = SearchTraceUtils.create_page_token(4)

    infos, next_token = trace_store.search_traces(
        experiment_ids=["42"],
        filter_string="status = 'OK'",
        order_by=["timestamp_ms ASC"],
        max_results=1,
        page_token=token,
    )

    trace_repository.search_trace_infos.assert_called_once_with(
        experiment_ids=["42"],
        filters=[TraceSearchFilter("attribute", "status", "=", "OK")],
        order_by=[TraceSearchOrder("attribute", "timestamp_ms", True)],
        offset=4,
        limit=1,
    )
    assert len(infos) == 1
    assert SearchTraceUtils.parse_start_offset_from_page_token(next_token) == 5


def test_negative_page_offset_is_rejected_before_repository_call(trace_store, trace_repository):
    token = SearchTraceUtils.create_page_token(-1)

    with pytest.raises(MlflowException) as caught:
        trace_store.search_traces(experiment_ids=["42"], page_token=token)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.message == "Page offset must not be negative."
    trace_repository.search_trace_infos.assert_not_called()


def test_short_search_page_has_no_next_token(trace_store, trace_repository):
    assert trace_store.search_traces(experiment_ids=["42"], max_results=2) == ([], None)
    assert trace_repository.search_trace_infos.call_count == 1


def test_search_persistence_error_is_sanitized_and_logged(trace_store, trace_repository, caplog):
    error = RepositoryPersistenceError("Private query details")
    trace_repository.search_trace_infos.side_effect = error
    with pytest.raises(MlflowException) as caught:
        trace_store.search_traces(experiment_ids=["42"])
    assert caught.value.message == "Unable to search traces."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


@pytest.mark.parametrize(
    ("operator", "mongo_operator"), [("=", "$eq"), ("!=", "$ne"), (">=", "$gte"), ("<", "$lt")]
)
def test_computed_end_time_comparisons_use_aggregation_expressions(operator, mongo_operator):
    assert TraceRepository._search_filter_condition(
        TraceSearchFilter("attribute", "end_time_ms", operator, 1000)
    ) == {
        "$expr": {
            mongo_operator: [
                {"$add": ["$request_time", {"$ifNull": ["$execution_duration", 0]}]},
                {"$literal": 1000},
            ]
        }
    }


@pytest.mark.parametrize(
    ("operator", "value", "condition"),
    [("!=", "OK", {"$ne": "OK"}), ("NOT IN", ("OK",), {"$nin": ["OK"]})],
)
def test_negative_attribute_comparisons_exclude_missing_and_null(operator, value, condition):
    assert TraceRepository._search_filter_condition(
        TraceSearchFilter("attribute", "status", operator, value)
    ) == {"$and": [{"state": {**condition, "$exists": True}}, {"state": {"$ne": None}}]}


@pytest.mark.parametrize(
    "field_type,field", [("tag", "tags"), ("request_metadata", "trace_metadata")]
)
@pytest.mark.parametrize(
    "operator,expected",
    [
        ("IS NULL", {"$not": {"$elemMatch": {"k": "optional"}}}),
        ("IS NOT NULL", {"$elemMatch": {"k": "optional"}}),
    ],
)
def test_kv_existence_filters_match_keys_without_values(field_type, field, operator, expected):
    assert TraceRepository._search_filter_condition(
        TraceSearchFilter(field_type, "optional", operator, None)
    ) == {field: expected}


@pytest.mark.parametrize(
    "field_type,field", [("tag", "tags"), ("request_metadata", "trace_metadata")]
)
def test_kv_comparisons_keep_key_and_value_in_same_element(field_type, field):
    assert TraceRepository._search_filter_condition(
        TraceSearchFilter(field_type, "team.name", "!=", "risk")
    ) == {field: {"$elemMatch": {"k": "team.name", "v": {"$ne": "risk"}}}}


def test_source_run_filter_matches_explicit_links_and_metadata():
    assert TraceRepository._search_filter_condition(
        TraceSearchFilter("request_metadata", TraceMetadataKey.SOURCE_RUN, "=", "run")
    ) == {
        "$or": [
            {"run_ids": "run"},
            {"trace_metadata": {"$elemMatch": {"k": TraceMetadataKey.SOURCE_RUN, "v": "run"}}},
        ]
    }


def test_prompt_filter_requires_name_and_version_from_same_reference():
    assert TraceRepository._search_filter_condition(
        TraceSearchFilter("tag", LINKED_PROMPTS_TAG_KEY, "=", "greeting/2")
    ) == {"linked_prompts": {"$elemMatch": {"name": "greeting", "version": "2"}}}


@pytest.mark.parametrize("operator,flags", [("LIKE", 0), ("ILIKE", re.IGNORECASE)])
def test_like_comparisons_use_shared_regex_behavior(operator, flags):
    condition = TraceRepository._search_value_condition(operator, "Risk%")
    assert condition.pattern == "^Risk.*"
    assert bool(condition.flags & re.IGNORECASE) == bool(flags)
    assert condition.fullmatch("Risk\nteam")


def test_rlike_retains_regular_expression():
    assert TraceRepository._search_value_condition("RLIKE", "^a[0-9]+$") == {"$regex": "^a[0-9]+$"}
