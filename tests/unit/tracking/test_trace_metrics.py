"""Trace metric validation and conversion at the MLflow interface."""

import json
from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE, ErrorCode
from pymongo.errors import OperationFailure

from mlflow_mongodb.infrastructure.errors import (
    RepositoryPersistenceError,
    RepositoryUnsupportedFieldTypeError,
)
from mlflow_mongodb.tracking._compat import (  # ruff: ignore[import-private-name]
    TRACE_METRICS_AVAILABLE,
)
from mlflow_mongodb.tracking.repositories import TraceRepository

if TRACE_METRICS_AVAILABLE:
    from mlflow.entities.trace_metrics import (
        AggregationType,
        MetricAggregation,
        MetricDataPoint,
        MetricViewType,
    )
    from mlflow.tracing.constant import (
        AssessmentMetricDimensionKey,
        AssessmentMetricKey,
        TraceMetricKey,
    )
else:
    pytest.skip("Trace metric queries require MLflow 3.9 or newer", allow_module_level=True)


@pytest.mark.parametrize(
    "value", [True, False, 1, 1.5, "yes", None, [1, "a"], {"answer": [1, True]}]
)
def test_native_assessment_dimensions_are_serialized_only_for_public_result(value):
    aggregation = MetricAggregation(AggregationType.COUNT)
    rows = [{"_id": {AssessmentMetricDimensionKey.ASSESSMENT_VALUE: value}, "agg_0": 2}]
    points = TraceRepository._to_metric_data_points(
        rows, AssessmentMetricKey.ASSESSMENT_COUNT, [aggregation]
    )
    assert points == [
        MetricDataPoint(
            metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
            dimensions={AssessmentMetricDimensionKey.ASSESSMENT_VALUE: json.dumps(value)},
            values={"COUNT": 2},
        )
    ]
    assert rows[0]["_id"][AssessmentMetricDimensionKey.ASSESSMENT_VALUE] == value


def test_metric_results_omit_null_dimensions_and_empty_aggregates():
    rows = [
        {"_id": {"trace_name": None}, "agg_0": 1},
        {"_id": {"trace_name": "agent"}, "agg_0": None},
        {"_id": {"trace_name": "agent"}, "agg_0": 0},
    ]
    points = TraceRepository._to_metric_data_points(
        rows, TraceMetricKey.LATENCY, [MetricAggregation(AggregationType.SUM)]
    )
    assert points == [MetricDataPoint(TraceMetricKey.LATENCY, {"trace_name": "agent"}, {"SUM": 0})]


def test_time_bucket_is_converted_to_utc_without_mutating_row():
    row = {"_id": {"time_bucket": 0}, "agg_0": 4}
    points = TraceRepository._to_metric_data_points(
        [row], TraceMetricKey.TRACE_COUNT, [MetricAggregation(AggregationType.COUNT)]
    )
    assert points[0].dimensions == {"time_bucket": "1970-01-01T00:00:00+00:00"}
    assert row["_id"]["time_bucket"] == 0


def test_result_aggregation_names_include_requested_percentile():
    points = TraceRepository._to_metric_data_points(
        [{"agg_0": 10, "agg_1": None, "agg_2": 20}],
        TraceMetricKey.LATENCY,
        [
            MetricAggregation(AggregationType.AVG),
            MetricAggregation(AggregationType.SUM),
            MetricAggregation(AggregationType.PERCENTILE, 95),
        ],
    )
    assert points[0].values == {"AVG": 10, "P95": 20}


def test_assessment_metric_filters_exclude_invalid_overrides():
    assert TraceRepository._filter_stages(None, MetricViewType.ASSESSMENTS) == (
        [],
        [{"content.valid": {"$ne": False}}],
    )


@pytest.mark.parametrize(
    "view_type,expression,field_type",
    [
        (MetricViewType.TRACES, "span.name = 'root'", "span"),
        (MetricViewType.ASSESSMENTS, "span.name = 'root'", "span"),
        (MetricViewType.TRACES, "assessment.name = 'quality'", "assessment"),
        (MetricViewType.SPANS, "assessment.name = 'quality'", "assessment"),
    ],
)
def test_metric_filter_view_mismatch_raises_shared_repository_error(
    view_type, expression, field_type
):
    with pytest.raises(RepositoryUnsupportedFieldTypeError) as caught:
        TraceRepository._filter_stages([expression], view_type)

    assert caught.value.field_type == field_type


@pytest.mark.parametrize(
    "field_type,required_view",
    [("span", MetricViewType.SPANS), ("assessment", MetricViewType.ASSESSMENTS)],
)
def test_store_translates_metric_filter_view_mismatch_without_exposing_repository_text(
    trace_store, trace_repository, caplog, field_type, required_view
):
    error = RepositoryUnsupportedFieldTypeError(field_type)
    error.args = ("Private repository details",)
    trace_repository.query_trace_metrics.side_effect = error

    with pytest.raises(MlflowException) as caught:
        trace_store.query_trace_metrics(
            ["42"],
            MetricViewType.TRACES,
            TraceMetricKey.TRACE_COUNT,
            [MetricAggregation(AggregationType.COUNT)],
        )

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.message == (
        f"Filtering by {field_type} is only supported for {required_view} view type, "
        f"got {MetricViewType.TRACES}"
    )
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


def test_query_passes_validated_parameters_and_returns_unpaginated_points(
    trace_store, trace_repository
):
    aggregations = [MetricAggregation(AggregationType.COUNT)]
    point = MetricDataPoint(TraceMetricKey.TRACE_COUNT, {}, {"COUNT": 2})
    trace_repository.query_trace_metrics.return_value = [point]
    result = trace_store.query_trace_metrics(
        ["42"],
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        aggregations,
        start_time_ms=0,
        end_time_ms=60_000,
        time_interval_seconds=60,
        max_results=10,
    )
    assert list(result) == [point]
    assert result.token is None
    trace_repository.query_trace_metrics.assert_called_once_with(
        experiment_ids=["42"],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=aggregations,
        dimensions=None,
        filters=None,
        time_interval_seconds=60,
        start_time_ms=0,
        end_time_ms=60_000,
        max_results=10,
    )


@pytest.mark.parametrize("bounds", [{}, {"start_time_ms": 0}, {"end_time_ms": 1000}])
def test_time_buckets_require_both_bounds(trace_store, trace_repository, bounds):
    with pytest.raises(MlflowException) as caught:
        trace_store.query_trace_metrics(
            ["42"],
            MetricViewType.TRACES,
            TraceMetricKey.TRACE_COUNT,
            [MetricAggregation(AggregationType.COUNT)],
            time_interval_seconds=60,
            **bounds,
        )
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    trace_repository.query_trace_metrics.assert_not_called()


def test_invalid_metric_is_rejected_before_repository_call(trace_store, trace_repository):
    with pytest.raises(MlflowException) as caught:
        trace_store.query_trace_metrics(
            ["42"], MetricViewType.TRACES, "unknown", [MetricAggregation(AggregationType.COUNT)]
        )
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    trace_repository.query_trace_metrics.assert_not_called()


def test_query_repository_failure_is_sanitized(trace_store, trace_repository, caplog):
    error = RepositoryPersistenceError("Private pipeline")
    trace_repository.query_trace_metrics.side_effect = error
    with pytest.raises(MlflowException) as caught:
        trace_store.query_trace_metrics(
            ["42"],
            MetricViewType.TRACES,
            TraceMetricKey.TRACE_COUNT,
            [MetricAggregation(AggregationType.COUNT)],
        )
    assert caught.value.message == "Unable to query trace metrics."
    assert caught.value.error_code == ErrorCode.Name(INTERNAL_ERROR)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert caplog.records[-1].args == (error,)
    assert caplog.records[-1].exc_info is None


def test_zero_metric_limit_returns_empty_without_aggregation():
    collection = MagicMock()
    database = MagicMock()
    database.__getitem__.return_value = collection
    repository = TraceRepository(database)

    result = repository.query_trace_metrics(
        experiment_ids=["42"],
        view_type=MetricViewType.TRACES,
        metric_name=TraceMetricKey.TRACE_COUNT,
        aggregations=[MetricAggregation(AggregationType.COUNT)],
        dimensions=None,
        filters=None,
        time_interval_seconds=None,
        start_time_ms=None,
        end_time_ms=None,
        max_results=0,
    )

    assert result == []
    collection.aggregate.assert_not_called()


def test_partial_metric_cursor_failure_is_translated_and_cursor_closed():
    collection = MagicMock()
    database = MagicMock()
    database.__getitem__.return_value = collection
    repository = TraceRepository(database)
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    error = OperationFailure("Private cursor details")

    def iterate():
        yield {"agg_0": 1}
        raise error

    cursor.__iter__.side_effect = iterate
    collection.aggregate.return_value = cursor

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.query_trace_metrics(
            experiment_ids=["42"],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            dimensions=None,
            filters=None,
            time_interval_seconds=None,
            start_time_ms=None,
            end_time_ms=None,
            max_results=10,
        )
    assert caught.value.__cause__ is error
    assert cursor.__exit__.call_count == 1
