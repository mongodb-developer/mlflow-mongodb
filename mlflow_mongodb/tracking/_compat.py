"""Tracking capabilities and span adapters for supported MLflow 3.x versions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from mlflow.store import tracking as tracking_constants
from mlflow.tracing import constant as tracing_constants
from mlflow.tracing import utils as tracing_utils
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey, TraceTagKey

if TYPE_CHECKING:
    from mlflow.entities.span import Span

    from mlflow_mongodb.tracking.types import SpanSummaryRecord


TRACE_METRICS_AVAILABLE = find_spec("mlflow.entities.trace_metrics") is not None
TRACKING_STORE_SPANS_AVAILABLE = hasattr(tracing_constants, "SpansLocation")
LINKED_PROMPTS_TAG_KEY = getattr(TraceTagKey, "LINKED_PROMPTS", "mlflow.linkedPrompts")
MAX_RESULTS_QUERY_TRACE_METRICS = getattr(
    tracking_constants, "MAX_RESULTS_QUERY_TRACE_METRICS", 1000
)
SPANS_LOCATION_TAG = getattr(TraceTagKey, "SPANS_LOCATION", "mlflow.trace.spansLocation")
SPANS_LOCATION = "TRACKING_STORE" if TRACKING_STORE_SPANS_AVAILABLE else "ARTIFACT_REPO"
TRACE_COST_METADATA = getattr(TraceMetadataKey, "COST", "mlflow.trace.cost")
SPAN_COST_ATTRIBUTE = getattr(SpanAttributeKey, "LLM_COST", "mlflow.llm.cost")
SPAN_MODEL_ATTRIBUTE = getattr(SpanAttributeKey, "MODEL", "mlflow.llm.model")
SPAN_PROVIDER_ATTRIBUTE = getattr(SpanAttributeKey, "MODEL_PROVIDER", "mlflow.llm.provider")
SPAN_SESSION_ATTRIBUTE = getattr(SpanAttributeKey, "SESSION_ID", "session.id")
SPAN_USER_ATTRIBUTE = getattr(SpanAttributeKey, "USER_ID", "user.id")
SPAN_TRACE_TAG_PREFIX = getattr(SpanAttributeKey, "TRACE_TAG_PREFIX", "mlflow.traceTag.")
CONVERSATION_ID_ATTRIBUTE = "gen_ai.conversation.id"

try:
    from mlflow.tracing.otel.translation import translate_span_when_storing
except ModuleNotFoundError as exc:
    if exc.name not in ("mlflow.tracing.otel", "mlflow.tracing.otel.translation"):
        raise

    def translate_span_when_storing(span: Span) -> dict[str, Any]:
        """Preserve native MLflow span content when OTel translation is unavailable."""
        return span.to_dict()


def try_json_loads(value: str) -> Any:
    """Decode a serialized attribute while retaining plain strings."""
    if decoder := getattr(tracing_utils, "try_json_loads", None):
        return decoder(value)
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


@dataclass(frozen=True)
class _SummarySpan:
    span_id: str
    parent_id: str | None
    token_usage: dict[str, Any] | None
    cost: dict[str, Any] | None

    def get_attribute(self, key: str) -> dict[str, Any] | None:
        if key == SpanAttributeKey.CHAT_USAGE:
            return self.token_usage
        if key == SPAN_COST_ATTRIBUTE:
            return self.cost
        return None


def aggregate_span_summaries(
    documents: list[SpanSummaryRecord],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Reuse MLflow's tree-aware aggregation with persisted span summaries."""
    spans = [_SummarySpan(d.span_id, d.parent_span_id, d.token_usage, d.cost) for d in documents]
    usage = tracing_utils.aggregate_usage_from_spans(spans)
    aggregate_cost = getattr(tracing_utils, "aggregate_cost_from_spans", None)
    cost = aggregate_cost(spans) if aggregate_cost is not None else None
    return usage, cost
