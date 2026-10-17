"""Persistence operations for traces, spans, and assessments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bson.errors import BSONError
from pymongo import ASCENDING, DESCENDING, ReplaceOne, ReturnDocument
from pymongo.database import Database
from pymongo.errors import (
    DuplicateKeyError,
    PyMongoError,
)

from mlflow_mongodb.infrastructure.array_operations import (
    build_merge_array_expression,
    build_remove_array_element_update,
)
from mlflow_mongodb.infrastructure.errors import (
    RepositoryNotFoundError,
    RepositoryPersistenceError,
    RepositoryWriteConflictError,
)
from mlflow_mongodb.infrastructure.repository_operations import repository_operation
from mlflow_mongodb.infrastructure.retry import retry_on_exception
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.types import (
    SpanRecord,
    TraceRecord,
)


class TraceRepository:
    """Own trace metadata and the separate span and assessment collections."""

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
                [("trace_id", ASCENDING), ("assessment_id", ASCENDING)],
                unique=True,
                name="assessments_trace_assessment_unique",
            )
            self._assessments_collection.create_index(
                [("experiment_id", ASCENDING), ("assessment_id", ASCENDING)],
                name="assessments_experiment_assessment",
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
                    {"trace_id": trace_id, "assessment_id": assessment_id},
                    {
                        "trace_id": trace_id,
                        "experiment_id": experiment_id,
                        "assessment_id": assessment_id,
                        "content": content,
                    },
                    upsert=True,
                )
            )
        with repository_operation(
            "Unable to persist trace assessments.", errors=(PyMongoError, BSONError, OverflowError)
        ):
            self._assessments_collection.bulk_write(operations, ordered=True)

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
                        {"$sort": {"assessment_id": ASCENDING}},
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
