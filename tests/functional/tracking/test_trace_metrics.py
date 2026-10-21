"""Trace, span, and native assessment metric aggregation against MongoDB."""

import json

import pytest
from mlflow.entities import TraceState
from mlflow.entities.assessment import Expectation, Feedback
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from opentelemetry.trace import StatusCode

from mlflow_mongodb.tracking._compat import (  # ruff: ignore[import-private-name]
    SPAN_COST_ATTRIBUTE,
    SPAN_MODEL_ATTRIBUTE,
    SPAN_PROVIDER_ATTRIBUTE,
    TRACE_METRICS_AVAILABLE,
)

if TRACE_METRICS_AVAILABLE:
    from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
    from mlflow.tracing.constant import (
        AssessmentMetricDimensionKey,
        AssessmentMetricKey,
        SpanAttributeKey,
        SpanMetricDimensionKey,
        SpanMetricKey,
        TraceMetadataKey,
        TraceMetricDimensionKey,
        TraceMetricKey,
        TraceTagKey,
    )
else:
    pytest.skip("Trace metric queries require MLflow 3.9 or newer", allow_module_level=True)


def test_trace_latency_groups_by_name_and_utc_time_bucket(store, trace_info_factory):
    for request_time, duration in ((0, 10), (1000, 30), (60_000, 50)):
        store.start_trace(
            trace_info_factory(
                request_time=request_time,
                execution_duration=duration,
                tags={TraceTagKey.TRACE_NAME: "agent"},
            )
        )

    result = store.query_trace_metrics(
        [trace_info_factory().experiment_id],
        MetricViewType.TRACES,
        TraceMetricKey.LATENCY,
        [MetricAggregation(AggregationType.AVG)],
        dimensions=[TraceMetricDimensionKey.TRACE_NAME],
        start_time_ms=0,
        end_time_ms=60_000,
        time_interval_seconds=60,
    )

    assert [(point.dimensions, point.values) for point in result] == [
        ({"time_bucket": "1970-01-01T00:00:00+00:00", "trace_name": "agent"}, {"AVG": 20}),
        ({"time_bucket": "1970-01-01T00:01:00+00:00", "trace_name": "agent"}, {"AVG": 50}),
    ]
    assert result.token is None


def test_span_time_buckets_use_span_start_time_in_nanoseconds(
    store, trace_info_factory, span_factory
):
    trace = store.start_trace(trace_info_factory(request_time=0))
    spans = [
        span_factory(
            trace_id=trace.trace_id,
            span_id=index,
            start_time=start_time,
            end_time=start_time + 1_000_000,
        )
        for index, start_time in enumerate(
            (1_000_000, 59_999_999_999, 60_000_000_000, 61_000_000_000), start=1
        )
    ]
    store.log_spans(trace.experiment_id, spans)

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.SPANS,
        SpanMetricKey.SPAN_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
        time_interval_seconds=60,
        start_time_ms=0,
        end_time_ms=120_000,
    )

    assert [(point.dimensions, point.values) for point in result] == [
        ({"time_bucket": "1970-01-01T00:00:00+00:00"}, {"COUNT": 2}),
        ({"time_bucket": "1970-01-01T00:01:00+00:00"}, {"COUNT": 2}),
    ]


def test_assessment_time_buckets_use_serialized_creation_time(store, trace_info_factory):
    trace = store.start_trace(trace_info_factory(request_time=0))
    for create_time in (1, 59_999, 60_000, 61_000):
        store.create_assessment(
            Feedback(
                name="quality",
                value=True,
                trace_id=trace.trace_id,
                create_time_ms=create_time,
                last_update_time_ms=120_000,
            )
        )

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.ASSESSMENTS,
        AssessmentMetricKey.ASSESSMENT_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
        time_interval_seconds=60,
        start_time_ms=0,
        end_time_ms=120_000,
    )

    assert [(point.dimensions, point.values) for point in result] == [
        ({"time_bucket": "1970-01-01T00:00:00+00:00"}, {"COUNT": 2}),
        ({"time_bucket": "1970-01-01T00:01:00+00:00"}, {"COUNT": 2}),
    ]


def test_trace_counts_group_by_status(store, trace_info_factory):
    for state in (TraceState.OK, TraceState.OK, TraceState.ERROR):
        store.start_trace(trace_info_factory(state=state))

    result = store.query_trace_metrics(
        [trace_info_factory().experiment_id],
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
        dimensions=[TraceMetricDimensionKey.TRACE_STATUS],
    )

    assert len(result) == 2
    assert {point.dimensions["trace_status"]: point.values for point in result} == {
        "OK": {"COUNT": 2},
        "ERROR": {"COUNT": 1},
    }


@pytest.mark.parametrize(
    "filters,expected_count",
    [
        (["trace.status = 'OK'"], 2),
        (["trace.tag.`team.name` = '$risk'"], 2),
        (["trace.metadata.`session.key` = '$session'"], 1),
        ([f"trace.metadata.`{TraceMetadataKey.SOURCE_RUN}` = 'run-one'"], 2),
        (["trace.status = 'OK'", "trace.tag.`team.name` = '$risk'"], 1),
        (["trace.status = 'IN_PROGRESS'"], 0),
    ],
)
def test_trace_metric_filters_match_status_tags_metadata_and_run_links(
    store, trace_info_factory, filters, expected_count
):
    metadata_linked = store.start_trace(
        trace_info_factory(
            tags={"team.name": "$risk"},
            trace_metadata={"session.key": "$session", TraceMetadataKey.SOURCE_RUN: "run-one"},
        )
    )
    # Isolate the metadata association from the explicit run-link representation.
    store._trace_repository._collection.update_one(
        {"_id": metadata_linked.trace_id},
        {"$set": {"run_ids": []}},
    )
    explicitly_linked = store.start_trace(
        trace_info_factory(
            state=TraceState.ERROR,
            tags={"team.name": "$risk"},
            trace_metadata={"session.key": "other"},
        )
    )
    store.link_traces_to_run([explicitly_linked.trace_id], "run-one")
    store.start_trace(
        trace_info_factory(
            tags={"team.name": "other", "decoy": "$risk"},
            trace_metadata={"session.key": "other", "decoy": "$session"},
        )
    )

    result = store.query_trace_metrics(
        [metadata_linked.experiment_id],
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
        filters=filters,
    )

    assert len(result) == 1
    assert result[0].values == {"COUNT": expected_count}


@pytest.mark.parametrize(
    "view_type,metric_name,expression,required_view",
    [
        (
            MetricViewType.TRACES,
            TraceMetricKey.TRACE_COUNT,
            "span.name = 'root'",
            MetricViewType.SPANS,
        ),
        (
            MetricViewType.ASSESSMENTS,
            AssessmentMetricKey.ASSESSMENT_COUNT,
            "span.name = 'root'",
            MetricViewType.SPANS,
        ),
        (
            MetricViewType.TRACES,
            TraceMetricKey.TRACE_COUNT,
            "assessment.name = 'quality'",
            MetricViewType.ASSESSMENTS,
        ),
        (
            MetricViewType.SPANS,
            SpanMetricKey.SPAN_COUNT,
            "assessment.name = 'quality'",
            MetricViewType.ASSESSMENTS,
        ),
    ],
)
def test_metric_filter_view_mismatch_returns_public_validation_error(
    store, trace, view_type, metric_name, expression, required_view
):
    with pytest.raises(MlflowException) as caught:
        store.query_trace_metrics(
            [trace.experiment_id],
            view_type,
            metric_name,
            [MetricAggregation(AggregationType.COUNT)],
            filters=[expression],
        )

    filter_type = expression.split(".", 1)[0]
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.message == (
        f"Filtering by {filter_type} is only supported for {required_view} view type, "
        f"got {view_type}"
    )
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_latency_percentile_uses_requested_percentile_aggregation(store, trace_info_factory):
    for _ in range(3):
        store.start_trace(trace_info_factory(execution_duration=10))

    result = store.query_trace_metrics(
        [trace_info_factory().experiment_id],
        MetricViewType.TRACES,
        TraceMetricKey.LATENCY,
        [MetricAggregation(AggregationType.PERCENTILE, 95)],
    )

    assert len(result) == 1
    assert result[0].values == {"P95": 10}


@pytest.mark.skipif(
    not hasattr(TraceMetricKey, "SESSION_COUNT"),
    reason="Installed MLflow has no session-count metric",
)
def test_session_count_counts_unique_session_ids(store, trace_info_factory):
    for session in ("one", "one", "two"):
        store.start_trace(
            trace_info_factory(trace_metadata={TraceMetadataKey.TRACE_SESSION: session})
        )

    result = store.query_trace_metrics(
        [trace_info_factory().experiment_id],
        MetricViewType.TRACES,
        TraceMetricKey.SESSION_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
    )

    assert len(result) == 1
    assert result[0].values == {"COUNT": 2}


def test_token_sum_uses_native_stats_and_excludes_traces_without_usage(store, trace_info_factory):
    for count in (10, 20):
        store.start_trace(
            trace_info_factory(
                trace_metadata={
                    TraceMetadataKey.TOKEN_USAGE: json.dumps({"input_tokens": count}),
                }
            )
        )
    store.start_trace(trace_info_factory())

    result = store.query_trace_metrics(
        [trace_info_factory().experiment_id],
        MetricViewType.TRACES,
        TraceMetricKey.INPUT_TOKENS,
        [MetricAggregation(AggregationType.SUM)],
    )

    assert len(result) == 1
    assert result[0].values == {"SUM": 30}


@pytest.mark.parametrize(
    "dimension,expected_counts",
    [
        (SpanMetricDimensionKey.SPAN_NAME, {"span-1": 1, "span-2": 1, "span-3": 1}),
        (SpanMetricDimensionKey.SPAN_TYPE, {"CHAIN": 1, "LLM": 2}),
        (SpanMetricDimensionKey.SPAN_STATUS, {"OK": 1, "ERROR": 2}),
    ],
)
def test_span_counts_group_by_stored_span_fields(
    store, trace, span_factory, dimension, expected_counts
):
    spans = [
        span_factory(trace_id=trace.trace_id, span_id=1),
        span_factory(
            trace_id=trace.trace_id,
            span_id=2,
            parent_id=1,
            status=StatusCode.ERROR,
            attributes={SpanAttributeKey.SPAN_TYPE: "LLM"},
        ),
        span_factory(
            trace_id=trace.trace_id,
            span_id=3,
            parent_id=1,
            status=StatusCode.ERROR,
            attributes={SpanAttributeKey.SPAN_TYPE: "LLM"},
        ),
    ]
    store.log_spans(trace.experiment_id, spans)

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.SPANS,
        SpanMetricKey.SPAN_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
        dimensions=[dimension],
    )

    assert len(result) == len(expected_counts)
    assert {
        point.dimensions[dimension]: point.values["COUNT"] for point in result
    } == expected_counts


@pytest.mark.parametrize(
    "duration_ns,expected_ms",
    [(999_999, 0), (1_500_000, 1), (3_250_000, 3)],
)
def test_span_latency_converts_nanoseconds_to_whole_milliseconds(
    store, trace, span_factory, duration_ns, expected_ms
):
    start_time = trace.request_time * 1_000_000
    span = span_factory(
        trace_id=trace.trace_id,
        start_time=start_time,
        end_time=start_time + duration_ns,
    )
    store.log_spans(trace.experiment_id, [span])

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.SPANS,
        SpanMetricKey.LATENCY,
        [MetricAggregation(AggregationType.AVG)],
    )

    assert len(result) == 1
    assert result[0].values == {"AVG": expected_ms}


@pytest.mark.skipif(
    not hasattr(SpanMetricKey, "TOTAL_COST"),
    reason="Installed MLflow has no span-cost metrics",
)
def test_span_metrics_filter_child_rows_and_use_native_cost(store, trace, span_factory):
    spans = [
        span_factory(
            trace_id=trace.trace_id,
            span_id=1,
            attributes={
                SpanAttributeKey.SPAN_TYPE: "LLM",
                SPAN_COST_ATTRIBUTE: {"total_cost": 0.2},
                SPAN_MODEL_ATTRIBUTE: "model-a",
            },
        ),
        span_factory(
            trace_id=trace.trace_id,
            span_id=2,
            parent_id=1,
            attributes={
                SpanAttributeKey.SPAN_TYPE: "CHAIN",
                SPAN_COST_ATTRIBUTE: {"total_cost": 9.0},
                SPAN_MODEL_ATTRIBUTE: "model-b",
            },
        ),
    ]
    store.log_spans(trace.experiment_id, spans)

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.SPANS,
        SpanMetricKey.TOTAL_COST,
        [MetricAggregation(AggregationType.SUM)],
        filters=["span.type = 'LLM'"],
    )

    assert len(result) == 1
    assert result[0].dimensions == {}
    assert result[0].values["SUM"] == pytest.approx(0.2)


@pytest.mark.skipif(
    not all(
        hasattr(SpanMetricDimensionKey, name) for name in ("SPAN_MODEL_NAME", "SPAN_MODEL_PROVIDER")
    ),
    reason="Installed MLflow has no span model metric dimensions",
)
@pytest.mark.parametrize(
    "attribute,stored_field,dimension,value",
    [
        (
            SPAN_MODEL_ATTRIBUTE,
            "model",
            getattr(SpanMetricDimensionKey, "SPAN_MODEL_NAME", None),
            "model-a",
        ),
        (
            SPAN_PROVIDER_ATTRIBUTE,
            "provider",
            getattr(SpanMetricDimensionKey, "SPAN_MODEL_PROVIDER", None),
            "provider-a",
        ),
    ],
)
def test_span_metrics_group_by_normalized_model_attributes(
    store, trace, span_factory, attribute, stored_field, dimension, value
):
    store.log_spans(
        trace.experiment_id,
        [
            span_factory(
                trace_id=trace.trace_id,
                attributes={attribute: value},
            )
        ],
    )

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.SPANS,
        SpanMetricKey.SPAN_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
        dimensions=[dimension],
    )

    assert len(result) == 1
    assert result[0].dimensions == {dimension: value}
    assert result[0].values == {"COUNT": 1}
    document = store._trace_repository._spans_collection.find_one({"trace_id": trace.trace_id})
    assert document["dimension_attributes"] == {stored_field: value}
    assert json.loads(document["content"]["attributes"][attribute]) == value


def test_assessment_counts_group_by_name(store, trace):
    for name in ("quality", "quality", "relevance"):
        store.create_assessment(Feedback(name=name, value=True, trace_id=trace.trace_id))

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.ASSESSMENTS,
        AssessmentMetricKey.ASSESSMENT_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_NAME],
    )

    assert len(result) == 2
    assert {point.dimensions["assessment_name"]: point.values for point in result} == {
        "quality": {"COUNT": 2},
        "relevance": {"COUNT": 1},
    }


@pytest.mark.parametrize("value", [True, False, 3, "yes", {"answer": [1, True]}, [1, "answer"]])
def test_assessment_count_groups_native_values_and_serializes_public_dimension(store, trace, value):
    for _ in range(2):
        store.create_assessment(Feedback(name="quality", value=value, trace_id=trace.trace_id))

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.ASSESSMENTS,
        AssessmentMetricKey.ASSESSMENT_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_VALUE],
        filters=["assessment.name = 'quality'"],
    )

    assert len(result) == 1
    assert result[0].dimensions == {"assessment_value": json.dumps(value)}
    assert result[0].values == {"COUNT": 2}


def test_assessment_numeric_aggregation_excludes_overridden_feedback(store, trace):
    original = store.create_assessment(
        Feedback(name="quality", value=False, trace_id=trace.trace_id)
    )
    override = store.create_assessment(
        Feedback(
            name="quality",
            value=True,
            trace_id=trace.trace_id,
            overrides=original.assessment_id,
        )
    )
    store.create_assessment(Feedback(name="quality", value="no", trace_id=trace.trace_id))
    store.create_assessment(Feedback(name="quality", value=3, trace_id=trace.trace_id))
    store.create_assessment(Feedback(name="quality", value="unscored", trace_id=trace.trace_id))

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.ASSESSMENTS,
        AssessmentMetricKey.ASSESSMENT_VALUE,
        [MetricAggregation(AggregationType.AVG)],
        filters=["assessment.name = 'quality'"],
    )

    assert len(result) == 1
    assert result[0].values == {"AVG": pytest.approx(4 / 3)}
    store.delete_assessment(trace.trace_id, override.assessment_id)
    restored = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.ASSESSMENTS,
        AssessmentMetricKey.ASSESSMENT_VALUE,
        [MetricAggregation(AggregationType.AVG)],
        filters=["assessment.name = 'quality'"],
    )
    assert restored[0].values == {"AVG": 1}


def test_structured_expectation_values_are_grouped_as_native_objects(store, trace):
    value = {"answer": [1, True]}
    store.create_assessment(Expectation(name="expected", value=value, trace_id=trace.trace_id))

    result = store.query_trace_metrics(
        [trace.experiment_id],
        MetricViewType.ASSESSMENTS,
        AssessmentMetricKey.ASSESSMENT_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
        dimensions=[AssessmentMetricDimensionKey.ASSESSMENT_VALUE],
        filters=["assessment.type = 'expectation'"],
    )

    assert len(result) == 1
    assert result[0].dimensions == {"assessment_value": json.dumps(value)}
    assert result[0].values == {"COUNT": 1}


def test_empty_trace_count_returns_zero(store, trace_info_factory):
    result = store.query_trace_metrics(
        [trace_info_factory().experiment_id],
        MetricViewType.TRACES,
        TraceMetricKey.TRACE_COUNT,
        [MetricAggregation(AggregationType.COUNT)],
    )

    assert len(result) == 1
    assert result[0].values == {"COUNT": 0}
