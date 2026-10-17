"""Tracking capabilities and span adapters for supported MLflow 3.x versions."""

from __future__ import annotations

import json
from typing import Any

from mlflow.tracing import constant as tracing_constants
from mlflow.tracing import utils as tracing_utils
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey

TRACKING_STORE_SPANS_AVAILABLE = hasattr(tracing_constants, "SpansLocation")
SPANS_LOCATION_TAG = getattr(TraceTagKey, "SPANS_LOCATION", "mlflow.trace.spansLocation")
SPANS_LOCATION = "TRACKING_STORE" if TRACKING_STORE_SPANS_AVAILABLE else "ARTIFACT_REPO"
TRACE_COST_METADATA = getattr(TraceMetadataKey, "COST", "mlflow.trace.cost")


def try_json_loads(value: str) -> Any:
    """Decode a serialized attribute while retaining plain strings."""
    if decoder := getattr(tracing_utils, "try_json_loads", None):
        return decoder(value)
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value
