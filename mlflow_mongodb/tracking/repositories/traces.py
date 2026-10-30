"""Persistence operations for traces, spans, and assessments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, ClassVar, Final

from bson.errors import BSONError
from mlflow.entities.trace_state import TraceState
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey
from pymongo import ASCENDING, DESCENDING, ReplaceOne, ReturnDocument, UpdateOne
from pymongo.database import Database
from pymongo.errors import (
    BulkWriteError,
    DuplicateKeyError,
    PyMongoError,
)

from mlflow_mongodb.infrastructure.array_operations import (
    build_array_value_expression,
    build_merge_array_expression,
    build_remove_array_element_update,
)
from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotFoundError,
    RepositoryPersistenceError,
    RepositoryUnsupportedFieldTypeError,
    RepositoryWriteConflictError,
)
from mlflow_mongodb.infrastructure.repository_operations import repository_operation
from mlflow_mongodb.infrastructure.retry import retry_on_exception
from mlflow_mongodb.infrastructure.search_filters import COMPARISON_OPERATORS, build_value_condition
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking._compat import (
    LINKED_PROMPTS_TAG_KEY,
    SPANS_LOCATION,
    SPANS_LOCATION_TAG,
    TRACE_COST_METADATA,
    TRACE_METRICS_AVAILABLE,
)
from mlflow_mongodb.tracking.types import (
    SpanDimensionKey,
    SpanRecord,
    SpanSummaryRecord,
    TraceRecord,
)

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
        AssessmentMetricSearchKey,
        SpanMetricDimensionKey,
        SpanMetricKey,
        SpanMetricSearchKey,
        TraceMetricDimensionKey,
        TraceMetricKey,
        TraceMetricSearchKey,
    )
    from mlflow.utils.search_utils import SearchTraceMetricsUtils

    _SPAN_COST_METRIC_KEYS = getattr(SpanMetricKey, "cost_keys", tuple)()
    _SESSION_COUNT_METRIC = getattr(TraceMetricKey, "SESSION_COUNT", None)


@dataclass(frozen=True)
class TraceSearchFilter:
    field_type: str
    key: str
    operator: str
    value: Any


@dataclass(frozen=True)
class TraceSearchOrder:
    field_type: str
    key: str
    ascending: bool


class TraceRepository:
    """Own trace metadata and the separate span and assessment collections."""

    _TIME_BUCKET_LABEL = "time_bucket"
    _ASSESSMENT_TYPE_FIELDS: Final = MappingProxyType(
        {
            "feedback": "content.feedback",
            "expectation": "content.expectation",
            "issue": "content.issue",
        }
    )

    SEARCH_ATTRIBUTE_FIELDS: ClassVar[dict[str, str]] = {
        "request_id": "_id",
        "experiment_id": "experiment_id",
        "timestamp_ms": "request_time",
        "execution_time_ms": "execution_duration",
        "status": "state",
        "client_request_id": "client_request_id",
    }

    def __init__(self, database: Database, settings: MongoDBSettings | None = None):
        self._settings = settings or MongoDBSettings()
        with repository_operation("Unable to initialize trace repository."):
            self._collection = database[self._settings.traces_collection_name]
            # _id is the supplied trace ID and already has a unique index.
            self._collection.create_index(
                [("experiment_id", ASCENDING), ("request_time", ASCENDING), ("_id", ASCENDING)],
                name="traces_experiment_request_time_id",
            )
            self._collection.create_index(
                [("experiment_id", ASCENDING), ("request_time", DESCENDING), ("_id", ASCENDING)],
                name="traces_experiment_request_time_desc_id",
            )
            self._collection.create_index(
                [
                    ("experiment_id", ASCENDING),
                    ("run_ids", ASCENDING),
                    ("request_time", ASCENDING),
                ],
                name="traces_experiment_run_ids_request_time",
            )
            self._collection.create_index(
                [("linked_prompts.name", ASCENDING), ("linked_prompts.version", ASCENDING)],
                name="traces_linked_prompts_name_version",
            )
            self._spans_collection = database[self._settings.spans_collection_name]
            self._spans_collection.create_index(
                [("trace_id", ASCENDING), ("span_id", ASCENDING)],
                unique=True,
                name="spans_trace_span_unique",
            )
            self._spans_collection.create_index(
                [("trace_id", ASCENDING), ("start_time_ns", ASCENDING)],
                name="spans_trace_start_time",
            )

            self._assessments_collection = database[self._settings.assessments_collection_name]
            self._assessments_collection.create_index(
                [("trace_id", ASCENDING), ("_id", ASCENDING)],
                name="assessments_trace_id",
            )
            self._assessments_collection.create_index(
                [("experiment_id", ASCENDING), ("_id", ASCENDING)],
                name="assessments_experiment_id",
            )

    def start_trace(
        self,
        *,
        trace_id: str,
        experiment_id: str,
        request_time: int,
        state: str,
        execution_duration: int | None,
        client_request_id: str | None,
        request_preview: str | None,
        response_preview: str | None,
        tags: Mapping[str, str],
        trace_metadata: Mapping[str, str],
        assessments: list[dict[str, Any]],
        run_ids: list[str] | None = None,
        span_stats: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> TraceRecord:
        unique_run_ids = list(dict.fromkeys(run_ids or []))
        fields = {
            "experiment_id": {"$literal": experiment_id},
            "request_time": {"$literal": request_time},
            "state": {"$literal": state},
            "execution_duration": {"$literal": execution_duration},
            "client_request_id": {"$literal": client_request_id},
            "run_ids": {
                "$setUnion": [
                    {"$ifNull": ["$run_ids", []]},
                    {"$literal": unique_run_ids},
                ]
            },
            # Authoritative metadata may arrive after spans. This does not close
            # the trace to further span delivery, even when its state is final.
            "trace_info_finalized": True,
            "authoritative_metadata_keys": {
                "$setUnion": [
                    {"$ifNull": ["$authoritative_metadata_keys", []]},
                    {"$literal": list(trace_metadata)},
                ]
            },
            "tags": build_merge_array_expression(
                "$tags", [{"k": k, "v": v} for k, v in tags.items()], "k"
            ),
            "trace_metadata": build_merge_array_expression(
                "$trace_metadata",
                [{"k": k, "v": v} for k, v in trace_metadata.items()],
                "k",
            ),
        }
        # Write the native stats and serialized metadata together. Omitted stats
        # preserve the existing expected count during partial metadata updates.
        if span_stats is not None:
            fields["span_stats"] = {"$literal": dict(span_stats)}
        for key, value in (metrics or {}).items():
            fields[key] = {"$literal": value}
        for field, value in (
            ("request_preview", request_preview),
            ("response_preview", response_preview),
        ):
            fields[field] = (
                {"$literal": value} if value is not None else {"$ifNull": [f"${field}", None]}
            )

        self._upsert_trace(trace_id, experiment_id, [{"$set": fields}])
        self._upsert_assessments(
            trace_id=trace_id,
            experiment_id=experiment_id,
            assessments=assessments,
        )
        return self.get_trace_info(trace_id)

    @retry_on_exception(
        RepositoryWriteConflictError,
        attempts=3,
        on_exhausted=lambda _: RepositoryPersistenceError(
            "Trace upsert conflicts persisted after 3 attempts."
        ),
    )
    def _upsert_trace(self, trace_id: str, experiment_id: str, update) -> dict[str, Any]:
        # Ownership remains in the filter; failures do not trigger diagnostic reads.
        with repository_operation("Unable to upsert trace."):
            try:
                return self._collection.find_one_and_update(
                    {"_id": trace_id, "experiment_id": experiment_id},
                    update,
                    upsert=True,
                    return_document=ReturnDocument.AFTER,
                )
            except DuplicateKeyError as exc:
                raise RepositoryWriteConflictError(trace_id) from exc

    def ensure_traces(self, *, experiment_id: str, request_times: Mapping[str, int]) -> None:
        """Create missing trace placeholders, retrying concurrent identity conflicts.

        Existing traces retain their metadata. Unordered writes can partially succeed.
        Retry only failed _id upserts, with at most three bulk-write attempts.
        """
        if not request_times:
            return
        operations = [
            UpdateOne(
                {"_id": trace_id, "experiment_id": experiment_id},
                {"$setOnInsert": self._trace_placeholder(request_times[trace_id])},
                upsert=True,
            )
            for trace_id in request_times
        ]
        for attempt in range(3):
            with repository_operation("Unable to create trace placeholders."):
                try:
                    self._collection.bulk_write(operations, ordered=False)
                except BulkWriteError as exc:
                    write_errors = exc.details.get("writeErrors", [])
                    if (
                        attempt == 2
                        or exc.details.get("writeConcernErrors")
                        or not write_errors
                        or any(
                            error.get("code") != 11000 or error.get("keyPattern") != {"_id": 1}
                            for error in write_errors
                        )
                    ):
                        raise
                    failed_indexes = [error["index"] for error in write_errors]
                else:
                    return
            # Successful unordered writes must not be replayed; error indexes refer
            # to the operations submitted in this attempt, not the original batch.
            operations = [operations[index] for index in failed_indexes]

    @staticmethod
    def _trace_placeholder(request_time: int) -> dict[str, Any]:
        return {
            "request_time": request_time,
            "state": TraceState.IN_PROGRESS.value,
            "execution_duration": None,
            "client_request_id": None,
            "request_preview": None,
            "response_preview": None,
            "trace_info_finalized": False,
            "authoritative_metadata_keys": [],
            "run_ids": [],
            "tags": [{"k": SPANS_LOCATION_TAG, "v": SPANS_LOCATION}],
            "trace_metadata": [],
        }

    @staticmethod
    def _assessment_aggregate_value(content: Mapping[str, Any]) -> float | None:
        assessment = content.get("feedback") or content.get("expectation") or {}
        value = assessment.get("value")
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            try:
                numeric_value = float(value)
            except OverflowError:
                return None
            return numeric_value if math.isfinite(numeric_value) else None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"yes", "no"}:
                return 1.0 if normalized == "yes" else 0.0
        return None

    def _upsert_assessments(
        self,
        *,
        trace_id: str,
        experiment_id: str,
        assessments: list[dict[str, Any]],
    ) -> None:
        """Replace supplied assessments in the separate assessment collection."""
        if not assessments:
            return

        assessments_by_id = {assessment["assessment_id"]: assessment for assessment in assessments}
        operations = []
        for assessment_id, assessment in assessments_by_id.items():
            content = dict(assessment)
            content["trace_id"] = trace_id
            operations.append(
                ReplaceOne(
                    {"trace_id": trace_id, "_id": assessment_id},
                    {
                        "trace_id": trace_id,
                        "experiment_id": experiment_id,
                        "_id": assessment_id,
                        "content": content,
                        "aggregate_value": self._assessment_aggregate_value(content),
                    },
                    upsert=True,
                )
            )
        with repository_operation(
            "Unable to persist trace assessments.", errors=(PyMongoError, BSONError, OverflowError)
        ):
            self._assessments_collection.bulk_write(operations, ordered=True)

    def get_assessment(self, *, trace_id: str, assessment_id: str) -> dict[str, Any]:
        with repository_operation("Unable to read assessment."):
            document = self._assessments_collection.find_one(
                {"trace_id": trace_id, "_id": assessment_id},
                {"_id": 0, "content": 1},
            )
        if document is None:
            raise RepositoryNotFoundError(assessment_id)
        return document["content"]

    def create_assessment(
        self, *, trace_id: str, experiment_id: str, content: dict[str, Any]
    ) -> dict[str, Any]:
        document = {
            "trace_id": trace_id,
            "experiment_id": experiment_id,
            "_id": content["assessment_id"],
            "content": content,
            "aggregate_value": self._assessment_aggregate_value(content),
        }
        overridden_id = content.get("overrides")
        with repository_operation(
            "Unable to create assessment.", errors=(PyMongoError, BSONError, OverflowError)
        ):
            if overridden_id:
                result = self._assessments_collection.update_one(
                    {"trace_id": trace_id, "_id": overridden_id},
                    {"$set": {"content.valid": False}},
                )
                if not result.matched_count:
                    raise RepositoryNotFoundError(overridden_id)
            try:
                self._assessments_collection.insert_one(document)
            except DuplicateKeyError as exc:
                raise RepositoryAlreadyExistsError(content["assessment_id"]) from exc
        return content

    def update_assessment(
        self, *, trace_id: str, assessment_id: str, updates: Mapping[str, Any]
    ) -> dict[str, Any]:
        fields = {f"content.{key}": {"$literal": value} for key, value in updates.items()}
        if "feedback" in updates or "expectation" in updates:
            fields["aggregate_value"] = {"$literal": self._assessment_aggregate_value(updates)}
        if "metadata" in updates:
            fields["content.metadata"] = {
                "$mergeObjects": [
                    {"$ifNull": ["$content.metadata", {}]},
                    {"$literal": updates["metadata"]},
                ]
            }
        with repository_operation(
            "Unable to update assessment.", errors=(PyMongoError, BSONError, OverflowError)
        ):
            document = self._assessments_collection.find_one_and_update(
                {"trace_id": trace_id, "_id": assessment_id},
                [{"$set": fields}],
                projection={"_id": 0, "content": 1},
                return_document=ReturnDocument.AFTER,
            )
        if document is None:
            raise RepositoryNotFoundError(assessment_id)
        return document["content"]

    def delete_assessment(self, *, trace_id: str, assessment_id: str) -> None:
        with repository_operation("Unable to delete assessment."):
            document = self._assessments_collection.find_one(
                {"trace_id": trace_id, "_id": assessment_id},
                {"_id": 0, "content": 1},
            )
        if document is None:
            return
        overridden_id = document["content"].get("overrides")
        with repository_operation("Unable to delete assessment."):
            if overridden_id:
                self._assessments_collection.update_one(
                    {"trace_id": trace_id, "_id": overridden_id},
                    {"$set": {"content.valid": True}},
                )
            self._assessments_collection.delete_one({"trace_id": trace_id, "_id": assessment_id})

    def log_spans(self, documents: list[dict[str, Any]]) -> None:
        """Insert immutable spans; delivery of an existing identity is a no-op."""
        if not documents:
            return
        operations = [
            UpdateOne(
                {"trace_id": document["trace_id"], "span_id": document["span_id"]},
                {"$setOnInsert": document},
                upsert=True,
            )
            for document in documents
        ]
        with repository_operation(
            "Unable to persist spans.", errors=(PyMongoError, BSONError, OverflowError)
        ):
            self._spans_collection.bulk_write(operations, ordered=False)

    def span_summary_snapshot(
        self, *, trace_id: str, experiment_id: str
    ) -> tuple[int, list[SpanSummaryRecord]]:
        # Increment AFTER span writes and BEFORE reading them. A later writer
        # invalidates earlier snapshots and then recomputes from persisted spans.
        with repository_operation("Unable to read trace span summary snapshot."):
            trace = self._collection.find_one_and_update(
                {"_id": trace_id, "experiment_id": experiment_id},
                {"$inc": {"span_revision": 1}},
                projection={"span_revision": 1},
                return_document=ReturnDocument.AFTER,
            )
        if trace is None:
            raise RepositoryNotFoundError(trace_id)
        with (
            repository_operation("Unable to read trace span summary snapshot."),
            self._spans_collection.find(
                {"trace_id": trace_id},
                {
                    "_id": 0,
                    "span_id": 1,
                    "parent_span_id": 1,
                    "status": 1,
                    "start_time_ns": 1,
                    "end_time_ns": 1,
                    "token_usage": 1,
                    "cost": 1,
                    "trace_fields": 1,
                },
            ).sort([("start_time_ns", ASCENDING), ("span_id", ASCENDING)]) as cursor,
        ):
            documents = list(cursor)
        return trace["span_revision"], [SpanSummaryRecord.from_document(d) for d in documents]

    def update_span_summary(
        self, *, trace_id: str, experiment_id: str, revision: int, summary: dict[str, Any]
    ) -> None:
        fields = {
            "state": {
                "$cond": [
                    {
                        "$in": [
                            "$state",
                            [TraceState.IN_PROGRESS.value, TraceState.STATE_UNSPECIFIED.value],
                        ]
                    },
                    {"$literal": summary["state"]},
                    "$state",
                ]
            },
            # Span-derived tags replace existing values; root tags already take
            # precedence over resource tags in the summary.
            "tags": build_merge_array_expression(
                "$tags", [{"k": key, "v": value} for key, value in summary["tags"].items()], "k"
            ),
            "trace_metadata": self._merge_missing_records("$trace_metadata", summary["metadata"]),
        }
        for key in ("request_time", "execution_duration"):
            fields[key] = {
                "$cond": [
                    {"$ifNull": ["$trace_info_finalized", False]},
                    f"${key}",
                    {"$literal": summary[key]},
                ]
            }
        for key in ("request_preview", "response_preview"):
            fields[key] = {"$ifNull": [f"${key}", {"$literal": summary[key]}]}

        metadata_updates = []
        for field, metadata_key in (
            ("token_usage", TraceMetadataKey.TOKEN_USAGE),
            ("cost", TRACE_COST_METADATA),
        ):
            if summary[field] is not None:
                is_authoritative = {
                    "$in": [metadata_key, {"$ifNull": ["$authoritative_metadata_keys", []]}]
                }
                fields[field] = {
                    "$cond": [
                        is_authoritative,
                        f"${field}",
                        {"$literal": summary[field]},
                    ]
                }
                metadata_updates.append(
                    {
                        "$set": {
                            "trace_metadata": {
                                "$cond": [
                                    is_authoritative,
                                    "$trace_metadata",
                                    build_merge_array_expression(
                                        "$trace_metadata",
                                        [
                                            {
                                                "k": metadata_key,
                                                "v": summary["aggregate_metadata"][metadata_key],
                                            }
                                        ],
                                        "k",
                                    ),
                                ]
                            }
                        }
                    }
                )

        # Match identity independently of the revision so the returned document
        # distinguishes deletion from a stale snapshot without another read.
        pipeline = [
            {
                "$replaceWith": {
                    "$cond": [
                        {"$eq": ["$span_revision", revision]},
                        {"$mergeObjects": ["$$ROOT", stage["$set"]]},
                        "$$ROOT",
                    ]
                }
            }
            for stage in [{"$set": fields}, *metadata_updates]
        ]
        with repository_operation("Unable to update trace span summary."):
            previous = self._collection.find_one_and_update(
                {"_id": trace_id, "experiment_id": experiment_id},
                pipeline,
                projection={"span_revision": 1},
                return_document=ReturnDocument.BEFORE,
            )
        if previous is None:
            raise RepositoryNotFoundError(trace_id)
        if previous.get("span_revision") != revision:
            raise RepositoryWriteConflictError(trace_id)

    @staticmethod
    def _merge_missing_records(field: str, values: Mapping[str, str]) -> dict:
        return {
            "$concatArrays": [
                {"$ifNull": [field, []]},
                {
                    "$filter": {
                        "input": {"$literal": [{"k": k, "v": v} for k, v in values.items()]},
                        "as": "record",
                        "cond": {
                            "$not": [{"$in": ["$$record.k", {"$ifNull": [f"{field}.k", []]}]}]
                        },
                    }
                },
            ]
        }

    def get_trace_info(self, trace_id: str) -> TraceRecord:
        records = self.batch_get_trace_infos([trace_id])
        if not records:
            raise RepositoryNotFoundError(trace_id)
        return records[0]

    def batch_get_trace_infos(
        self, trace_ids: list[str], *, experiment_ids: list[str] | None = None
    ) -> list[TraceRecord]:
        """Read metadata and assessments without loading span payloads."""
        return [
            record
            for record, _ in self._batch_get_trace_records(
                trace_ids, experiment_ids=experiment_ids, include_spans=False
            )
        ]

    def search_trace_infos(
        self,
        *,
        experiment_ids: list[str] | None,
        filters: list[TraceSearchFilter],
        order_by: list[TraceSearchOrder],
        offset: int,
        limit: int,
    ) -> list[TraceRecord]:
        """Select trace metadata after scoped filtering, sorting, and pagination."""
        # TODO: Support span and assessment filters in search_trace_infos.
        clauses: list[dict[str, Any]] = []
        if experiment_ids is not None:
            if not experiment_ids:
                return []
            clauses.append({"experiment_id": {"$in": experiment_ids}})
        direct_filters = [
            item for item in filters if item.field_type in ("attribute", "tag", "request_metadata")
        ]
        clauses.extend(self._search_filter_condition(item) for item in direct_filters)

        pipeline: list[dict[str, Any]] = []
        if clauses:
            pipeline.append({"$match": {"$and": clauses}})
        pipeline.extend(self._search_order_stages(order_by))
        pipeline.extend([{"$skip": offset}, {"$limit": limit}])
        pipeline.append({"$project": {"_id": 1}})

        with (
            repository_operation("Unable to search trace metadata."),
            self._collection.aggregate(pipeline, allowDiskUse=True) as cursor,
        ):
            documents = list(cursor)
        trace_ids = [document["_id"] for document in documents]
        return self.batch_get_trace_infos(trace_ids, experiment_ids=experiment_ids)

    @staticmethod
    def _search_value_condition(operator: str, value: Any) -> Any:
        if operator == "RLIKE":
            return {"$regex": value}
        return build_value_condition(operator, value)

    @classmethod
    def _search_filter_condition(cls, item: TraceSearchFilter) -> dict[str, Any]:
        if item.field_type == "attribute":
            if item.key == "end_time_ms":
                expression = {"$add": ["$request_time", {"$ifNull": ["$execution_duration", 0]}]}
                return {
                    "$expr": {
                        COMPARISON_OPERATORS[item.operator]: [
                            expression,
                            {"$literal": item.value},
                        ]
                    }
                }
            field = cls.SEARCH_ATTRIBUTE_FIELDS[item.key]
            condition = cls._search_value_condition(item.operator, item.value)
            if item.operator in ("!=", "NOT IN"):
                condition["$exists"] = True
                return {"$and": [{field: condition}, {field: {"$ne": None}}]}
            return {field: condition}

        if item.field_type == "tag" and item.key == LINKED_PROMPTS_TAG_KEY:
            name, version = item.value.split("/", 1)
            return {"linked_prompts": {"$elemMatch": {"name": name, "version": version}}}
        field = "tags" if item.field_type == "tag" else "trace_metadata"
        key_match = {"k": item.key}
        if item.operator == "IS NULL":
            return {field: {"$not": {"$elemMatch": key_match}}}
        if item.operator == "IS NOT NULL":
            return {field: {"$elemMatch": key_match}}
        if (
            item.field_type == "request_metadata"
            and item.key == TraceMetadataKey.SOURCE_RUN
            and item.operator == "="
        ):
            return {
                "$or": [
                    {"run_ids": item.value},
                    {field: {"$elemMatch": {"k": item.key, "v": item.value}}},
                ]
            }
        return {
            field: {
                "$elemMatch": {
                    "k": item.key,
                    "v": cls._search_value_condition(item.operator, item.value),
                }
            }
        }

    def _search_order_stages(self, order_by: list[TraceSearchOrder]) -> list[dict[str, Any]]:
        """Build ordering stages with nulls last and deterministic tie breakers."""
        stages: list[dict[str, Any]] = []
        sort_fields: dict[str, int] = {}
        sort_values: dict[str, Any] = {}
        observed = set()
        for index, item in enumerate(order_by):
            observed.add((item.field_type, item.key))
            if item.field_type == "attribute" and item.key in (
                "timestamp_ms",
                "request_id",
                "experiment_id",
            ):
                sort_fields[self.SEARCH_ATTRIBUTE_FIELDS[item.key]] = (
                    ASCENDING if item.ascending else DESCENDING
                )
                continue
            expression = self._search_order_expression(item)
            value_field = f"_search_sort_{index}"
            null_field = f"_search_null_{index}"
            sort_values[value_field] = expression
            sort_values[null_field] = {"$cond": [{"$eq": [expression, None]}, 1, 0]}
            sort_fields[null_field] = ASCENDING
            sort_fields[value_field] = ASCENDING if item.ascending else DESCENDING
        if sort_values:
            stages.append({"$addFields": sort_values})
        if ("attribute", "timestamp_ms") not in observed:
            sort_fields["request_time"] = DESCENDING
        if ("attribute", "request_id") not in observed:
            sort_fields["_id"] = ASCENDING
        stages.append({"$sort": sort_fields})
        return stages

    @classmethod
    def _search_order_expression(cls, item: TraceSearchOrder) -> Any:
        if item.field_type == "attribute":
            if item.key == "end_time_ms":
                return {"$add": ["$request_time", {"$ifNull": ["$execution_duration", 0]}]}
            return f"${cls.SEARCH_ATTRIBUTE_FIELDS[item.key]}"
        field = "$tags" if item.field_type == "tag" else "$trace_metadata"
        return build_array_value_expression(field, item.key)

    def batch_get_traces(
        self, trace_ids: list[str], *, experiment_ids: list[str] | None = None
    ) -> list[tuple[TraceRecord, list[SpanRecord]]]:
        """Join scoped trace metadata, assessments, and spans in one query."""
        return self._batch_get_trace_records(
            trace_ids, experiment_ids=experiment_ids, include_spans=True
        )

    def _batch_get_trace_records(
        self,
        trace_ids: list[str],
        *,
        experiment_ids: list[str] | None,
        include_spans: bool,
    ) -> list[tuple[TraceRecord, list[SpanRecord]]]:
        if not trace_ids or experiment_ids == []:
            return []

        unique_trace_ids = list(dict.fromkeys(trace_ids))
        query = {"_id": {"$in": unique_trace_ids}}
        if experiment_ids is not None:
            query["experiment_id"] = {"$in": experiment_ids}
        pipeline = [
            {"$match": query},
            {
                "$lookup": {
                    "from": self._assessments_collection.name,
                    "localField": "_id",
                    "foreignField": "trace_id",
                    "pipeline": [
                        {"$sort": {"_id": ASCENDING}},
                        {"$project": {"_id": 0, "content": 1}},
                    ],
                    "as": "_assessments",
                }
            },
        ]
        if include_spans:
            pipeline.append(
                {
                    "$lookup": {
                        "from": self._spans_collection.name,
                        "localField": "_id",
                        "foreignField": "trace_id",
                        "as": "_spans",
                    }
                }
            )

        with (
            repository_operation("Unable to read traces."),
            self._collection.aggregate(pipeline) as cursor,
        ):
            documents = list(cursor)
        documents_by_id = {document["_id"]: document for document in documents}
        records = []
        for trace_id in unique_trace_ids:
            if trace_id not in documents_by_id:
                continue
            document = documents_by_id[trace_id]
            record = TraceRecord.from_document(
                document,
                assessments=tuple(assessment["content"] for assessment in document["_assessments"]),
            )
            spans = (
                [SpanRecord.from_document(span) for span in document["_spans"]]
                if include_spans
                else []
            )
            records.append((record, spans))

        return records

    def query_trace_metrics(
        self,
        *,
        experiment_ids: list[str],
        view_type: MetricViewType,
        metric_name: str,
        aggregations: list[MetricAggregation],
        dimensions: list[str] | None,
        filters: list[str] | None,
        time_interval_seconds: int | None,
        start_time_ms: int | None,
        end_time_ms: int | None,
        max_results: int,
    ) -> list[MetricDataPoint]:
        """Aggregate metrics over trace, span, and assessment documents."""
        if not TRACE_METRICS_AVAILABLE:
            raise NotImplementedError("Trace metric queries require MLflow 3.9 or newer.")
        pipeline = self._build_trace_metrics_pipeline(
            experiment_ids=experiment_ids,
            view_type=view_type,
            metric_name=metric_name,
            aggregations=aggregations,
            dimensions=dimensions,
            filters=filters,
            time_interval_seconds=time_interval_seconds,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            max_results=max_results,
        )
        if max_results == 0:
            return []

        with (
            repository_operation("Unable to query trace metrics."),
            self._collection.aggregate(pipeline, allowDiskUse=True) as cursor,
        ):
            rows = list(cursor)
        if (
            not rows
            and not dimensions
            and not time_interval_seconds
            and aggregations
            and all(agg.aggregation_type == AggregationType.COUNT for agg in aggregations)
        ):
            rows = [{f"agg_{index}": 0 for index in range(len(aggregations))}]
        return self._to_metric_data_points(rows, metric_name, aggregations)

    def set_trace_tag(self, *, trace_id: str, key: str, value: str) -> None:
        tags_update = build_merge_array_expression("$tags", [{"k": key, "v": value}], "k")
        with repository_operation("Unable to set trace tag."):
            document = self._collection.find_one_and_update(
                {"_id": trace_id},
                [{"$set": {"tags": tags_update}}],
                projection={"_id": 1},
                return_document=ReturnDocument.AFTER,
            )
        if document is None:
            raise RepositoryNotFoundError(trace_id)

    def delete_trace_tag(self, *, trace_id: str, key: str) -> None:
        update = build_remove_array_element_update(array_field="tags", key_field="k", key=key)
        with repository_operation("Unable to delete trace tag."):
            document = self._collection.find_one_and_update(
                {"_id": trace_id, "tags.k": key},
                update,
                projection={"_id": 1},
                return_document=ReturnDocument.AFTER,
            )
        if document is None:
            raise RepositoryNotFoundError(trace_id)

    def link_traces_to_run(self, *, trace_ids: list[str], run_id: str) -> None:
        """Add a run reference to existing traces without duplicate entries."""
        if not trace_ids:
            return

        with repository_operation("Unable to link traces to run."):
            self._collection.update_many(
                {"_id": {"$in": trace_ids}}, {"$addToSet": {"run_ids": run_id}}
            )

    def link_prompts(self, *, trace_id: str, prompt_versions: list[Mapping[str, str]]) -> None:
        """Add prompt-version references to a trace without duplicate entries."""
        if not prompt_versions:
            return

        prompt_refs = [dict(pv) for pv in prompt_versions]
        with repository_operation("Unable to link prompts to trace."):
            document = self._collection.find_one_and_update(
                {"_id": trace_id},
                {"$addToSet": {"linked_prompts": {"$each": prompt_refs}}},
                projection={"_id": 1},
                return_document=ReturnDocument.AFTER,
            )
        if document is None:
            raise RepositoryNotFoundError(trace_id)

    def get_spans(self, trace_id: str) -> list[SpanRecord]:
        with (
            repository_operation("Unable to read trace spans."),
            self._spans_collection.find(
                {"trace_id": trace_id},
                {
                    "_id": 0,
                    "trace_id": 1,
                    "span_id": 1,
                    "parent_span_id": 1,
                    "start_time_ns": 1,
                    "end_time_ns": 1,
                    "content": 1,
                },
            ).sort([("start_time_ns", ASCENDING), ("span_id", ASCENDING)]) as cursor,
        ):
            documents = list(cursor)
        return [SpanRecord.from_document(document) for document in documents]

    def delete_traces(
        self,
        *,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        """Best-effort hard deletion of traces selected by validated store criteria.

        Span, assessment, and trace deletion are separate writes. Failures can leave
        partial deletion, and concurrent writers can leave child documents or recreate
        a trace.
        """
        query = {"experiment_id": experiment_id}
        if max_timestamp_millis is not None:
            query["request_time"] = {"$lte": max_timestamp_millis}
        if trace_ids:
            query["_id"] = {"$in": trace_ids}

        with repository_operation("Unable to delete traces."):
            cursor = self._collection.find(query, {"_id": 1}).sort(
                [
                    ("request_time", ASCENDING),
                    ("_id", ASCENDING),
                ]
            )
            if max_traces is not None:
                cursor = cursor.limit(max_traces)
            with cursor:
                documents = list(cursor)

        selected_ids = [document["_id"] for document in documents]
        if not selected_ids:
            return 0
        # Resolve ownership through traces before touching spans. Orphan
        # spans cannot safely be scoped to an experiment with this schema.
        with repository_operation("Unable to delete traces."):
            self._spans_collection.delete_many({"trace_id": {"$in": selected_ids}})
            self._assessments_collection.delete_many({"trace_id": {"$in": selected_ids}})
            # Keeping metadata until span deletion succeeds allows retries
            # to find traces whose span cleanup failed or was interrupted.
            result = self._collection.delete_many(
                {
                    "experiment_id": experiment_id,
                    "_id": {"$in": selected_ids},
                }
            )
        return result.deleted_count

    @staticmethod
    def _metric_value(view_type: MetricViewType, metric_name: str) -> Any:
        if view_type == MetricViewType.TRACES:
            if metric_name == TraceMetricKey.LATENCY:
                return "$execution_duration"
            if metric_name in TraceMetricKey.token_usage_keys():
                return f"$token_usage.{metric_name}"
        elif view_type == MetricViewType.SPANS:
            if metric_name == SpanMetricKey.LATENCY:
                return {
                    "$floor": {
                        "$divide": [
                            {
                                "$subtract": [
                                    "$_metric_row.end_time_ns",
                                    "$_metric_row.start_time_ns",
                                ]
                            },
                            1_000_000,
                        ]
                    }
                }
            if metric_name in _SPAN_COST_METRIC_KEYS:
                return f"$_metric_row.cost.{metric_name}"
        elif view_type == MetricViewType.ASSESSMENTS:
            if metric_name == AssessmentMetricKey.ASSESSMENT_VALUE:
                return "$_metric_row.aggregate_value"
        return None

    @staticmethod
    def _dimension_value(view_type: MetricViewType, dimension: str) -> Any:
        expressions = {
            MetricViewType.TRACES: {
                TraceMetricDimensionKey.TRACE_NAME: build_array_value_expression(
                    "$tags", TraceTagKey.TRACE_NAME
                ),
                TraceMetricDimensionKey.TRACE_STATUS: "$state",
            },
            MetricViewType.SPANS: {
                SpanMetricDimensionKey.SPAN_NAME: "$_metric_row.name",
                SpanMetricDimensionKey.SPAN_TYPE: "$_metric_row.type",
                SpanMetricDimensionKey.SPAN_STATUS: "$_metric_row.status",
            },
            MetricViewType.ASSESSMENTS: {
                AssessmentMetricDimensionKey.ASSESSMENT_NAME: (
                    "$_metric_row.content.assessment_name"
                ),
                AssessmentMetricDimensionKey.ASSESSMENT_VALUE: "$_assessment_value",
            },
        }
        for name, field in (
            ("SPAN_MODEL_NAME", SpanDimensionKey.MODEL),
            ("SPAN_MODEL_PROVIDER", SpanDimensionKey.MODEL_PROVIDER),
        ):
            if key := getattr(SpanMetricDimensionKey, name, None):
                expressions[MetricViewType.SPANS][key] = (
                    f"$_metric_row.dimension_attributes.{field}"
                )
        return expressions[view_type][dimension]

    @classmethod
    def _filter_stages(
        cls, filters: list[str] | None, view_type: MetricViewType
    ) -> tuple[list, list]:
        trace_clauses = []
        row_clauses = []
        if view_type == MetricViewType.ASSESSMENTS:
            row_clauses.append({"content.valid": {"$ne": False}})
        for filter_string in filters or []:
            parsed = SearchTraceMetricsUtils.parse_search_filter(filter_string)
            value = parsed.value
            if parsed.view_type == TraceMetricSearchKey.VIEW_TYPE:
                if parsed.entity == TraceMetricSearchKey.STATUS:
                    trace_clauses.append({"state": value})
                elif parsed.entity == TraceMetricSearchKey.TAG:
                    trace_clauses.append({"tags": {"$elemMatch": {"k": parsed.key, "v": value}}})
                elif parsed.entity == TraceMetricSearchKey.METADATA:
                    clause: dict[str, Any] = {
                        "trace_metadata": {"$elemMatch": {"k": parsed.key, "v": value}}
                    }
                    if parsed.key == TraceMetadataKey.SOURCE_RUN:
                        clause = {"$or": [clause, {"run_ids": value}]}
                    trace_clauses.append(clause)
            elif parsed.view_type == SpanMetricSearchKey.VIEW_TYPE:
                if view_type != MetricViewType.SPANS:
                    raise RepositoryUnsupportedFieldTypeError(parsed.view_type)
                row_clauses.append({parsed.entity: value})
            elif parsed.view_type == AssessmentMetricSearchKey.VIEW_TYPE:
                if view_type != MetricViewType.ASSESSMENTS:
                    raise RepositoryUnsupportedFieldTypeError(parsed.view_type)
                field = (
                    "assessment_name" if parsed.entity == AssessmentMetricSearchKey.NAME else None
                )
                if field:
                    row_clauses.append({f"content.{field}": value})
                else:
                    assessment_field = cls._ASSESSMENT_TYPE_FIELDS.get(value)
                    row_clauses.append(
                        {assessment_field: {"$exists": True}}
                        if assessment_field
                        else {"$expr": False}
                    )
        return trace_clauses, row_clauses

    def _build_trace_metrics_pipeline(
        self,
        *,
        experiment_ids: list[str],
        view_type: MetricViewType,
        metric_name: str,
        aggregations: list[MetricAggregation],
        dimensions: list[str] | None,
        filters: list[str] | None,
        time_interval_seconds: int | None,
        start_time_ms: int | None,
        end_time_ms: int | None,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Build a trace-scoped pipeline with row-level filters for each view."""
        trace_clauses, row_clauses = self._filter_stages(filters, view_type)
        trace_clauses.append({"experiment_id": {"$in": list(map(str, experiment_ids))}})
        if start_time_ms is not None or end_time_ms is not None:
            bounds = {}
            if view_type == MetricViewType.SPANS and metric_name in _SPAN_COST_METRIC_KEYS:
                if start_time_ms is not None:
                    bounds["$gte"] = start_time_ms * 1_000_000
                if end_time_ms is not None:
                    # Include the entire final millisecond, matching SQLAlchemy.
                    bounds["$lt"] = (end_time_ms + 1) * 1_000_000
                row_clauses.append({"start_time_ns": bounds})
            else:
                if start_time_ms is not None:
                    bounds["$gte"] = start_time_ms
                if end_time_ms is not None:
                    bounds["$lte"] = end_time_ms
                trace_clauses.append({"request_time": bounds})

        pipeline = []
        if trace_clauses:
            pipeline.append({"$match": {"$and": trace_clauses}})

        if view_type in (MetricViewType.SPANS, MetricViewType.ASSESSMENTS):
            collection = (
                self._settings.spans_collection_name
                if view_type == MetricViewType.SPANS
                else self._settings.assessments_collection_name
            )

            lookup: dict[str, Any] = {
                "from": collection,
                "localField": "_id",
                "foreignField": "trace_id",
                "as": "_metric_row",
            }
            if row_clauses:
                match = row_clauses[0] if len(row_clauses) == 1 else {"$and": row_clauses}
                lookup["pipeline"] = [{"$match": match}]
            pipeline.extend(
                [
                    {"$lookup": lookup},
                    {"$unwind": "$_metric_row"},
                ]
            )

        metric_match = {}
        if view_type == MetricViewType.TRACES:
            if TraceMetricDimensionKey.TRACE_NAME in (dimensions or []):
                metric_match["tags"] = {"$elemMatch": {"k": TraceTagKey.TRACE_NAME}}
            if metric_name == _SESSION_COUNT_METRIC:
                metric_match["trace_metadata"] = {
                    "$elemMatch": {"k": TraceMetadataKey.TRACE_SESSION}
                }
            elif metric_name in TraceMetricKey.token_usage_keys():
                metric_match[f"token_usage.{metric_name}"] = {"$exists": True}
        elif view_type == MetricViewType.SPANS:
            if metric_name in _SPAN_COST_METRIC_KEYS:
                metric_match[f"_metric_row.cost.{metric_name}"] = {"$exists": True}
        if metric_match:
            pipeline.append({"$match": metric_match})

        pipeline.extend(self._metric_preparation_stages(view_type, metric_name))

        group_keys = {}
        if time_interval_seconds:
            if view_type == MetricViewType.SPANS:
                timestamp = {"$divide": ["$_metric_row.start_time_ns", 1_000_000]}
            elif view_type == MetricViewType.ASSESSMENTS:
                timestamp = {
                    "$toLong": {
                        "$dateFromString": {"dateString": "$_metric_row.content.create_time"}
                    }
                }
            else:
                timestamp = "$request_time"
            size_ms = time_interval_seconds * 1000
            group_keys[self._TIME_BUCKET_LABEL] = {
                "$multiply": [{"$floor": {"$divide": [timestamp, size_ms]}}, size_ms]
            }
        for dimension in dimensions or []:
            group_keys[dimension] = {"$ifNull": [self._dimension_value(view_type, dimension), None]}

        group = {"_id": group_keys or None}
        for index, aggregation in enumerate(aggregations):
            if aggregation.aggregation_type == AggregationType.COUNT:
                if metric_name == _SESSION_COUNT_METRIC:
                    group["_sessions"] = {"$addToSet": {"$ifNull": ["$_session", "$$REMOVE"]}}
                else:
                    group[f"agg_{index}"] = {"$sum": 1}
            elif aggregation.aggregation_type == AggregationType.SUM:
                group[f"agg_{index}"] = {"$sum": "$_metric_value"}
                group["_has_numeric_value"] = {"$max": {"$isNumber": "$_metric_value"}}
            elif aggregation.aggregation_type == AggregationType.AVG:
                group[f"agg_{index}"] = {"$avg": "$_metric_value"}
            elif aggregation.aggregation_type == AggregationType.PERCENTILE:
                group[f"agg_{index}"] = {
                    "$percentile": {
                        "input": "$_metric_value",
                        "p": [aggregation.percentile_value / 100.0],
                        "method": "approximate",
                    }
                }
        pipeline.append({"$group": group})

        calculated = {}
        if metric_name == _SESSION_COUNT_METRIC and view_type == MetricViewType.TRACES:
            for index in range(len(aggregations)):
                calculated[f"agg_{index}"] = {"$size": "$_sessions"}
        for index, aggregation in enumerate(aggregations):
            if aggregation.aggregation_type == AggregationType.SUM:
                calculated[f"agg_{index}"] = {
                    "$cond": ["$_has_numeric_value", f"$agg_{index}", None]
                }
            elif aggregation.aggregation_type == AggregationType.PERCENTILE:
                calculated[f"agg_{index}"] = {"$arrayElemAt": [f"$agg_{index}", 0]}
        if calculated:
            pipeline.append({"$set": calculated})

        if group_keys:
            nonnull_dimensions = {
                f"_id.{key}": {"$ne": None}
                for key in group_keys
                if key != AssessmentMetricDimensionKey.ASSESSMENT_VALUE
            }
            if nonnull_dimensions:
                pipeline.append({"$match": nonnull_dimensions})
            pipeline.append({"$sort": {f"_id.{key}": ASCENDING for key in group_keys}})
        if max_results is not None:
            pipeline.append({"$limit": max_results})
        return pipeline

    @classmethod
    def _metric_preparation_stages(
        cls, view_type: MetricViewType, metric_name: str
    ) -> list[dict[str, Any]]:
        """Build the temporary metric fields needed by each view before grouping."""
        if view_type == MetricViewType.TRACES:
            if metric_name == _SESSION_COUNT_METRIC:
                return [
                    {
                        "$set": {
                            "_session": build_array_value_expression(
                                "$trace_metadata", TraceMetadataKey.TRACE_SESSION
                            )
                        }
                    }
                ]
            value = cls._metric_value(view_type, metric_name)
            if value is not None:
                return [{"$set": {"_metric_value": value}}]
        elif view_type == MetricViewType.SPANS:
            value = cls._metric_value(view_type, metric_name)
            if value is not None:
                return [{"$set": {"_metric_value": value}}]
        elif view_type == MetricViewType.ASSESSMENTS:
            stages = [
                {
                    "$set": {
                        "_assessment_value": {
                            "$ifNull": [
                                "$_metric_row.content.feedback.value",
                                {
                                    "$ifNull": [
                                        "$_metric_row.content.expectation.value",
                                        "$_metric_row.content.issue",
                                    ]
                                },
                            ]
                        }
                    }
                }
            ]
            if metric_name == AssessmentMetricKey.ASSESSMENT_VALUE:
                stages.append(
                    {"$set": {"_metric_value": cls._metric_value(view_type, metric_name)}}
                )
            return stages
        return []

    @classmethod
    def _to_metric_data_points(
        cls,
        rows: list[dict[str, Any]],
        metric_name: str,
        aggregations: list[MetricAggregation],
    ) -> list[MetricDataPoint]:
        """Apply MLflow's result conversion, including null groups and UTC buckets."""
        points = []
        for row in rows:
            dimensions = dict(row.get("_id") or {})
            assessment_value = AssessmentMetricDimensionKey.ASSESSMENT_VALUE
            if assessment_value in dimensions:
                dimensions[assessment_value] = json.dumps(dimensions[assessment_value])
            if any(value is None for value in dimensions.values()):
                continue
            if cls._TIME_BUCKET_LABEL in dimensions:
                dimensions[cls._TIME_BUCKET_LABEL] = datetime.fromtimestamp(
                    float(dimensions[cls._TIME_BUCKET_LABEL]) / 1000, tz=timezone.utc
                ).isoformat()
            values = {
                str(aggregation): row[f"agg_{index}"]
                for index, aggregation in enumerate(aggregations)
                if row.get(f"agg_{index}") is not None
            }
            if values:
                points.append(
                    MetricDataPoint(metric_name=metric_name, dimensions=dimensions, values=values)
                )
        return points
