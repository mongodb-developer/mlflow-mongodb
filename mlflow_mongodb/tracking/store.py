"""Skeleton of the MongoDB tracking store for the agreed V1 scope."""

from __future__ import annotations

import asyncio
import binascii
import json
import logging
import math
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from typing import Any
from uuid import uuid4

from mlflow.entities import (
    Assessment,
    Dataset,
    DatasetInput,
    Experiment,
    ExperimentTag,
    InputTag,
    LifecycleStage,
    LoggedModel,
    LoggedModelInput,
    LoggedModelOutput,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunInputs,
    RunOutputs,
    RunStatus,
    RunTag,
    Trace,
    TraceData,
    TraceInfo,
    TraceLocation,
    TraceState,
    ViewType,
)
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import Span
from mlflow.entities.span_status import SpanStatusCode
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    TEMPORARILY_UNAVAILABLE,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import (
    SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import (
    TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceSizeStatsKey,
)
from mlflow.tracing.utils.truncation import _get_truncated_preview
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.utils.mlflow_tags import (
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_RUN_NAME,
    _get_run_name_from_tags,
)
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import (
    SearchExperimentsUtils,
    SearchLoggedModelsPaginationToken,
    SearchLoggedModelsUtils,
    SearchUtils,
)
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import append_to_uri_path, resolve_uri_if_local
from mlflow.utils.validation import (
    _validate_batch_log_data,
    _validate_batch_log_limits,
    _validate_dataset_inputs,
    _validate_experiment_artifact_location,
    _validate_experiment_artifact_location_length,
    _validate_experiment_name,
    _validate_experiment_tag,
    _validate_logged_model_name,
    _validate_metric_name,
    _validate_param_keys_unique,
    _validate_run_id,
    _validate_trace_tag,
)
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConfigurationError

from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryEmptySearchKeyError,
    RepositoryInvalidAttributeError,
    RepositoryInvalidFilterValueError,
    RepositoryNotActiveError,
    RepositoryNotFoundError,
    RepositoryParamConflictError,
    RepositoryTagNotFoundError,
    RepositoryUnsupportedComparatorError,
    RepositoryUnsupportedFieldTypeError,
    RepositoryWriteConflictError,
)
from mlflow_mongodb.infrastructure.retry import retry_on_exception
from mlflow_mongodb.infrastructure.search_filters import (
    ConfiguredSearchFilterValidator,
    SearchFilterClause,
    SearchFilterValidator,
)
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.infrastructure.store_errors import handle_persistence_error
from mlflow_mongodb.tracking._compat import (
    CONVERSATION_ID_ATTRIBUTE,
    SPAN_COST_ATTRIBUTE,
    SPAN_MODEL_ATTRIBUTE,
    SPAN_PROVIDER_ATTRIBUTE,
    SPAN_SESSION_ATTRIBUTE,
    SPAN_TRACE_TAG_PREFIX,
    SPAN_USER_ATTRIBUTE,
    SPANS_LOCATION,
    SPANS_LOCATION_TAG,
    TRACE_COST_METADATA,
    aggregate_span_summaries,
    translate_span_when_storing,
    try_json_loads,
)
from mlflow_mongodb.tracking.repositories import (
    ExperimentRepository,
    LoggedModelRepository,
    RunRepository,
)
from mlflow_mongodb.tracking.repositories.experiments import ExperimentFilter, ExperimentOrder
from mlflow_mongodb.tracking.repositories.logged_models import LoggedModelFilter, LoggedModelOrder
from mlflow_mongodb.tracking.repositories.runs import RunFilter, RunOrder
from mlflow_mongodb.tracking.repositories.traces import (
    TraceRepository,
)
from mlflow_mongodb.tracking.types import (
    LoggedModelRecord,
    RunMetricRecord,
    RunRecord,
    SpanDimensionKey,
    SpanRecord,
    SpanSummaryRecord,
    TraceRecord,
)

logger = logging.getLogger(__name__)


class _TraceNotFullyExportedError(Exception):
    """Raised while a trace's expected spans are still being persisted."""


class MongoDBTrackingStore(AbstractStore):
    """MongoDB tracking store with persistence methods awaiting implementation.

    Batch and single-value async logging use the implementations inherited from
    AbstractStore, which delegate persistence to log_batch. Methods outside V1
    remain inherited and are not a claim of support.
    """

    _RUN_ORDER_FIELD_TYPES = frozenset({"attribute", "metric", "parameter", "tag"})

    def __init__(self, store_uri: str | None = None, artifact_uri: str | None = None) -> None:
        super().__init__()
        self.store_uri = store_uri
        self.artifact_uri = artifact_uri
        self._settings = MongoDBSettings.from_environment()

    @cached_property
    def _mongo_client(self) -> MongoClient:
        if not self.store_uri:
            raise MlflowException(
                "A MongoDB tracking URI is required.", error_code=INVALID_PARAMETER_VALUE
            )
        try:
            return MongoClient(self.store_uri)
        except ConfigurationError as exc:
            logger.error("Unable to create MongoDB tracking client: %s", exc)
            raise MlflowException(
                "Invalid MongoDB tracking URI.", error_code=INVALID_PARAMETER_VALUE
            ) from None

    @cached_property
    def _database(self) -> Database:
        try:
            return self._mongo_client.get_default_database()
        except ConfigurationError as exc:
            logger.error("Unable to select the MongoDB tracking database: %s", exc)
            raise MlflowException(
                "The MongoDB tracking URI must include a database name.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None

    @cached_property
    @handle_persistence_error("Unable to initialize experiment repository.", logger=logger)
    def _experiment_repository(self) -> ExperimentRepository:
        repository = ExperimentRepository(self._database, settings=self._settings)
        artifact_location = resolve_uri_if_local(
            append_to_uri_path(self.artifact_uri or "./mlruns", DEFAULT_EXPERIMENT_ID)
        )
        _validate_experiment_artifact_location_length(artifact_location)
        repository.ensure_default_experiment(
            artifact_location=artifact_location,
            creation_timestamp=get_current_time_millis(),
        )
        return repository

    @cached_property
    @handle_persistence_error("Unable to initialize run repository.", logger=logger)
    def _run_repository(self) -> RunRepository:
        return RunRepository(self._database, settings=self._settings)

    @cached_property
    @handle_persistence_error("Unable to initialize trace repository.", logger=logger)
    def _trace_repository(self) -> TraceRepository:
        return TraceRepository(self._database, settings=self._settings)

    @cached_property
    @handle_persistence_error("Unable to initialize logged model repository.", logger=logger)
    def _logged_model_repository(self) -> LoggedModelRepository:
        return LoggedModelRepository(self._database, settings=self._settings)

    @handle_persistence_error("A database operation failed.", logger=logger)
    def search_experiments(
        self,
        view_type: ViewType = ViewType.ACTIVE_ONLY,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[Experiment]:
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                "a positive integer",
                INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                f"at most {SEARCH_MAX_RESULTS_THRESHOLD}",
                INVALID_PARAMETER_VALUE,
            )

        filters = self._parse_experiment_filters(filter_string)
        orders = self._parse_experiment_order(order_by)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        if offset < 0:
            raise MlflowException("Page offset must not be negative.", INVALID_PARAMETER_VALUE)
        records = self._experiment_repository.search(
            lifecycle_stages=LifecycleStage.view_type_to_stages(view_type),
            filters=filters,
            order_by=orders,
            offset=offset,
            limit=max_results + 1,
        )

        next_page_token = None
        if len(records) > max_results:
            records = records[:max_results]
            next_page_token = SearchUtils.create_page_token(offset + max_results)
        experiments = [
            Experiment(
                experiment_id=record.experiment_id,
                name=record.name,
                artifact_location=record.artifact_location,
                lifecycle_stage=record.lifecycle_stage,
                tags=[ExperimentTag(tag.key, tag.value) for tag in record.tags],
                creation_time=record.creation_time,
                last_update_time=record.last_update_time,
            )
            for record in records
        ]
        return PagedList(experiments, next_page_token)

    @cached_property
    def _experiment_filter_validator(self) -> SearchFilterValidator:
        return ConfiguredSearchFilterValidator(
            field_types=("attribute", "tag"),
            attribute_keys=SearchExperimentsUtils.VALID_SEARCH_ATTRIBUTE_KEYS,
            comparators={
                "attribute": SearchExperimentsUtils.VALID_STRING_ATTRIBUTE_COMPARATORS,
                "tag": SearchExperimentsUtils.VALID_TAG_COMPARATORS,
            },
            field_comparators={
                ("attribute", key): SearchExperimentsUtils.VALID_NUMERIC_ATTRIBUTE_COMPARATORS
                for key in SearchExperimentsUtils.NUMERIC_ATTRIBUTES
            },
        )

    def _parse_experiment_filters(self, filter_string: str | None) -> tuple[ExperimentFilter, ...]:
        filters = []
        for parsed in SearchExperimentsUtils.parse_search_filter(filter_string):
            try:
                clause = self._experiment_filter_validator.validate(parsed)
            except RepositoryUnsupportedFieldTypeError as exc:
                logger.error("Unable to validate experiment filter: %s", exc)
                raise MlflowException.invalid_parameter_value(
                    f"Invalid token type: {exc.field_type}"
                ) from None
            except RepositoryInvalidAttributeError as exc:
                logger.error("Unable to validate experiment filter: %s", exc)
                raise MlflowException.invalid_parameter_value(
                    f"Invalid attribute name: {exc.key}"
                ) from None
            except RepositoryUnsupportedComparatorError as exc:
                logger.error("Unable to validate experiment filter: %s", exc)
                message = (
                    f"Invalid comparator '{exc.comparator}'. "
                    f"Supported comparators: {', '.join(exc.allowed)}."
                )
                raise MlflowException.invalid_parameter_value(message) from None

            value = clause.value
            if (
                clause.field_type == "attribute"
                and clause.key in SearchExperimentsUtils.NUMERIC_ATTRIBUTES
            ):
                # MLflow returns numeric tokens as text; BSON comparisons need numbers.
                value = float(value)
            filters.append(
                ExperimentFilter(clause.field_type, clause.key, clause.comparator, value)
            )
        return tuple(filters)

    @staticmethod
    def _parse_experiment_order(order_by: list[str] | None) -> tuple[ExperimentOrder, ...]:
        orders = []
        for field_type, key, ascending in map(
            SearchExperimentsUtils.parse_order_by_for_search_experiments,
            order_by or ["creation_time DESC", "experiment_id ASC"],
        ):
            if field_type != "attribute":
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order_by entity: {field_type}"
                )
            orders.append(ExperimentOrder(key, ascending))
        if not any(order.key == "experiment_id" for order in orders):
            orders.append(ExperimentOrder("experiment_id", False))
        return tuple(orders)

    @handle_persistence_error("Unable to create experiment.", logger=logger)
    def create_experiment(
        self,
        name: str,
        artifact_location: str | None = None,
        tags: list[ExperimentTag] | None = None,
    ) -> str:
        _validate_experiment_name(name)
        _validate_experiment_artifact_location(artifact_location)
        tags_by_key = {}
        for tag in tags or []:
            _validate_experiment_tag(tag.key, tag.value)
            tags_by_key[tag.key] = tag.value

        # Decimal UUIDs preserve numeric experiment IDs used by prompt filters,
        # without requiring a shared counter; "0" belongs to the default experiment.
        experiment_id = str(uuid4().int)
        artifact_location = resolve_uri_if_local(
            artifact_location or append_to_uri_path(self.artifact_uri or "./mlruns", experiment_id)
        )
        _validate_experiment_artifact_location_length(artifact_location)
        try:
            return self._experiment_repository.create(
                experiment_id=experiment_id,
                name=name,
                artifact_location=artifact_location,
                lifecycle_stage=LifecycleStage.ACTIVE,
                creation_timestamp=get_current_time_millis(),
                tags=tags_by_key,
            )
        except RepositoryAlreadyExistsError as exc:
            logger.error("Unable to create experiment: %s", exc)
            raise MlflowException(
                f"Experiment(name={name}) already exists.", RESOURCE_ALREADY_EXISTS
            ) from None

    @handle_persistence_error("Unable to get experiment.", logger=logger)
    def get_experiment(self, experiment_id: str | None) -> Experiment:
        experiment_id = None if experiment_id is None else str(experiment_id)

        record = self._experiment_repository.find_by_id(experiment_id)
        if record is None:
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            )

        return Experiment(
            experiment_id=record.experiment_id,
            name=record.name,
            artifact_location=record.artifact_location,
            lifecycle_stage=record.lifecycle_stage,
            tags=[ExperimentTag(tag.key, tag.value) for tag in record.tags],
            creation_time=record.creation_time,
            last_update_time=record.last_update_time,
        )

    @handle_persistence_error("Unable to get experiment by name.", logger=logger)
    def get_experiment_by_name(self, experiment_name: str) -> Experiment | None:
        record = self._experiment_repository.find_by_name(experiment_name)
        if record is None:
            return None

        return Experiment(
            experiment_id=record.experiment_id,
            name=record.name,
            artifact_location=record.artifact_location,
            lifecycle_stage=record.lifecycle_stage,
            tags=[ExperimentTag(tag.key, tag.value) for tag in record.tags],
            creation_time=record.creation_time,
            last_update_time=record.last_update_time,
        )

    @handle_persistence_error("Unable to delete experiment.", logger=logger)
    def delete_experiment(self, experiment_id: str) -> None:
        experiment_id = None if experiment_id is None else str(experiment_id)
        try:
            self._experiment_repository.mark_deleted(
                experiment_id=experiment_id,
                last_update_time=get_current_time_millis(),
            )
        except RepositoryNotFoundError as exc:
            logger.error("Unable to delete experiment: %s", exc)
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            ) from None

    @handle_persistence_error("Unable to restore experiment.", logger=logger)
    def restore_experiment(self, experiment_id: str) -> None:
        experiment_id = None if experiment_id is None else str(experiment_id)
        try:
            self._experiment_repository.restore(
                experiment_id=experiment_id,
                last_update_time=get_current_time_millis(),
            )
        except RepositoryNotFoundError as exc:
            logger.error("Unable to restore experiment: %s", exc)
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            ) from None

    @handle_persistence_error("Unable to rename experiment.", logger=logger)
    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        _validate_experiment_name(new_name)
        experiment_id = None if experiment_id is None else str(experiment_id)
        try:
            self._experiment_repository.rename(
                experiment_id=experiment_id,
                new_name=new_name,
                last_update_time=get_current_time_millis(),
            )
        except RepositoryNotFoundError as exc:
            logger.error("Unable to rename experiment: %s", exc)
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            ) from None
        except RepositoryNotActiveError as exc:
            logger.error("Unable to rename experiment: %s", exc)
            raise MlflowException("Cannot rename a non-active experiment.", INVALID_STATE) from None
        except RepositoryAlreadyExistsError as exc:
            logger.error("Unable to rename experiment: %s", exc)
            raise MlflowException(
                f"Experiment(name={new_name}) already exists.", RESOURCE_ALREADY_EXISTS
            ) from None

    @handle_persistence_error("Unable to set experiment tag.", logger=logger)
    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        _validate_experiment_tag(tag.key, tag.value)
        experiment_id = None if experiment_id is None else str(experiment_id)
        try:
            self._experiment_repository.set_tag(
                experiment_id=experiment_id, key=tag.key, value=tag.value
            )
        except RepositoryNotFoundError as exc:
            logger.error("Unable to set experiment tag: %s", exc)
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            ) from None
        except RepositoryNotActiveError as exc:
            logger.error("Unable to set experiment tag: %s", exc)
            raise MlflowException(
                f"The experiment {experiment_id} must be in the 'active' state. "
                f"Current state is {exc.args[1]}.",
                INVALID_PARAMETER_VALUE,
            ) from None

    @handle_persistence_error("Unable to delete experiment tag.", logger=logger)
    def delete_experiment_tag(self, experiment_id: str, key: str) -> None:
        experiment_id = None if experiment_id is None else str(experiment_id)
        try:
            self._experiment_repository.delete_tag(experiment_id=experiment_id, key=key)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to delete experiment tag: %s", exc)
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            ) from None
        except RepositoryNotActiveError as exc:
            logger.error("Unable to delete experiment tag: %s", exc)
            raise MlflowException(
                f"The experiment {experiment_id} must be in the 'active' state. "
                f"Current state is {exc.args[1]}.",
                INVALID_PARAMETER_VALUE,
            ) from None
        except RepositoryTagNotFoundError as exc:
            logger.error("Unable to delete experiment tag: %s", exc)
            raise MlflowException(
                f"No tag with name: {key} in experiment with id {experiment_id}",
                RESOURCE_DOES_NOT_EXIST,
            ) from None

    @handle_persistence_error("Unable to search runs.", logger=logger)
    def _search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str | None,
        run_view_type,
        max_results: int | None,
        order_by: list[str] | None,
        page_token: str | None,
    ) -> tuple[list[Run], str | None]:
        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
                raise MlflowException(
                    f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                    "a positive integer",
                    INVALID_PARAMETER_VALUE,
                )
            if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
                raise MlflowException(
                    f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                    f"at most {SEARCH_MAX_RESULTS_THRESHOLD}",
                    INVALID_PARAMETER_VALUE,
                )
        stages = LifecycleStage.view_type_to_stages(run_view_type)
        filters = self._parse_run_filters(filter_string)
        orders = self._parse_run_order(order_by)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        if offset < 0:
            raise MlflowException("Page offset must not be negative.", INVALID_PARAMETER_VALUE)
        records = self._run_repository.search(
            experiment_ids=[str(experiment_id) for experiment_id in experiment_ids],
            lifecycle_stages=stages,
            filters=filters,
            order_by=orders,
            offset=offset,
            limit=max_results + 1 if max_results is not None else None,
        )
        next_page_token = None
        if max_results is not None and len(records) > max_results:
            records = records[:max_results]
            next_page_token = SearchUtils.create_page_token(offset + max_results)
        return [self._run_from_record(record) for record in records], next_page_token

    @cached_property
    def _run_filter_validator(self) -> SearchFilterValidator:
        return ConfiguredSearchFilterValidator(
            field_types=("attribute", "metric", "parameter", "tag", "dataset"),
            attribute_keys=SearchUtils.VALID_SEARCH_ATTRIBUTE_KEYS,
            comparators={
                "attribute": SearchUtils.VALID_STRING_ATTRIBUTE_COMPARATORS,
                "metric": SearchUtils.VALID_METRIC_COMPARATORS,
                "parameter": SearchUtils.VALID_PARAM_COMPARATORS,
                "tag": SearchUtils.VALID_TAG_COMPARATORS,
                "dataset": SearchUtils.VALID_DATASET_COMPARATORS,
            },
            field_comparators={
                ("attribute", key): SearchUtils.VALID_NUMERIC_ATTRIBUTE_COMPARATORS
                for key in SearchUtils.NUMERIC_ATTRIBUTES
            },
        )

    def _parse_run_filters(self, filter_string):
        filters = []
        for parsed in SearchUtils.parse_search_filter(filter_string):
            try:
                clause = self._run_filter_validator.validate(parsed)
            except RepositoryUnsupportedFieldTypeError as exc:
                logger.error("Unable to validate run filter: %s", exc)
                raise MlflowException.invalid_parameter_value(
                    f"Invalid search expression type '{exc.field_type}'"
                ) from None
            except RepositoryInvalidAttributeError as exc:
                logger.error("Unable to validate run filter: %s", exc)
                raise MlflowException.invalid_parameter_value(
                    f"Invalid attribute name: {exc.key}"
                ) from None
            except RepositoryUnsupportedComparatorError as exc:
                logger.error("Unable to validate run filter: %s", exc)
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator '{exc.comparator}' for {exc.field_type}.{exc.key}"
                ) from None
            filters.append(self._run_filter_from_clause(clause))
        return tuple(filters)

    @staticmethod
    def _run_filter_from_clause(clause) -> RunFilter:
        key = clause.key
        value = clause.value
        if clause.comparator in ("IN", "NOT IN"):
            value = tuple(value)
        if clause.field_type == "attribute":
            key = SearchUtils.translate_key_alias(key)
            if key in SearchUtils.NUMERIC_ATTRIBUTES:
                value = float(value)
        elif clause.field_type == "metric":
            value = float(value)
        elif clause.field_type == "dataset" and key not in SearchUtils.DATASET_ATTRIBUTES:
            raise MlflowException.invalid_parameter_value(f"Invalid dataset key: {key}")
        return RunFilter(clause.field_type, key, clause.comparator, value)

    @classmethod
    def _parse_run_order(cls, order_by: list[str] | None) -> tuple[RunOrder, ...]:
        orders = []
        observed = set()
        for expression in order_by or []:
            field_type, key, ascending = SearchUtils.parse_order_by_for_search_runs(expression)
            if field_type not in cls._RUN_ORDER_FIELD_TYPES:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid identifier type '{field_type}'"
                )
            if field_type == "attribute":
                key = SearchUtils.translate_key_alias(key)
            if (field_type, key) in observed:
                raise MlflowException(f"`order_by` contains duplicate fields: {order_by}")
            observed.add((field_type, key))
            orders.append(RunOrder(field_type, key, ascending))
        if ("attribute", "start_time") not in observed:
            orders.append(RunOrder("attribute", "start_time", False))
        if ("attribute", "run_id") not in observed:
            orders.append(RunOrder("attribute", "run_id", True))
        return tuple(orders)

    @handle_persistence_error("Unable to create run.", logger=logger)
    def create_run(
        self,
        experiment_id: str | None,
        user_id: str,
        start_time: int,
        tags: list[RunTag] | None,
        run_name: str | None,
    ) -> Run:
        experiment_id = None if experiment_id is None else str(experiment_id)
        experiment = self._experiment_repository.find_by_id(experiment_id)
        if experiment is None:
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            )
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                (
                    f"The experiment {experiment.experiment_id} must be in the 'active' state. "
                    f"Current state is {experiment.lifecycle_stage}."
                ),
                INVALID_PARAMETER_VALUE,
            )

        run_id = uuid4().hex
        artifact_uri = append_to_uri_path(experiment.artifact_location, run_id, "artifacts")
        run_tags = list(tags or [])
        run_name_tag = _get_run_name_from_tags(run_tags)
        if run_name and run_name_tag and run_name != run_name_tag:
            raise MlflowException(
                "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                f"different values (run_name='{run_name}', run_name_tag='{run_name_tag}').",
                INVALID_PARAMETER_VALUE,
            )
        resolved_run_name = run_name or run_name_tag or _generate_random_name()
        if not run_name_tag:
            run_tags.append(RunTag(key=MLFLOW_RUN_NAME, value=resolved_run_name))

        try:
            record = self._run_repository.create(
                run_id=run_id,
                experiment_id=experiment_id,
                name=resolved_run_name,
                artifact_uri=artifact_uri,
                user_id=user_id,
                status=RunStatus.to_string(RunStatus.RUNNING),
                start_time=start_time,
                lifecycle_stage=LifecycleStage.ACTIVE,
                tags={tag.key: tag.value for tag in run_tags},
            )
        except RepositoryAlreadyExistsError as exc:
            logger.error("Unable to create run: %s", exc)
            raise MlflowException(
                f"Run with id={run_id} already exists", RESOURCE_ALREADY_EXISTS
            ) from None

        return Run(
            RunInfo(
                run_id=record.run_id,
                experiment_id=record.experiment_id,
                user_id=record.user_id,
                status=record.status,
                start_time=record.start_time,
                end_time=record.end_time,
                lifecycle_stage=record.lifecycle_stage,
                artifact_uri=record.artifact_uri,
                run_name=record.name,
            ),
            RunData(tags=[RunTag(tag.key, tag.value) for tag in record.tags]),
            RunInputs(dataset_inputs=[]),
        )

    @handle_persistence_error("Unable to get run.", logger=logger)
    def get_run(self, run_id: str) -> Run:
        record = self._run_repository.find_by_id(run_id)
        if record is None:
            raise MlflowException(f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST)

        return self._run_from_record(record)

    @staticmethod
    def _run_from_record(record: RunRecord) -> Run:
        return Run(
            RunInfo(
                run_id=record.run_id,
                experiment_id=record.experiment_id,
                user_id=record.user_id,
                status=record.status,
                start_time=record.start_time,
                end_time=record.end_time,
                lifecycle_stage=record.lifecycle_stage,
                artifact_uri=record.artifact_uri,
                run_name=record.name,
            ),
            RunData(
                params=[Param(param.key, param.value) for param in record.params],
                metrics=[
                    Metric(
                        key=metric.key,
                        value=metric.value,
                        timestamp=metric.timestamp,
                        step=metric.step,
                        model_id=metric.model_id,
                        dataset_name=metric.dataset_name,
                        dataset_digest=metric.dataset_digest,
                    )
                    for metric in record.metrics
                ],
                tags=[RunTag(tag.key, tag.value) for tag in record.tags],
            ),
            RunInputs(
                dataset_inputs=[
                    DatasetInput(
                        dataset=Dataset(
                            name=dataset.name,
                            digest=dataset.digest,
                            source_type=dataset.source_type,
                            source=dataset.source,
                            schema=dataset.schema,
                            profile=dataset.profile,
                        ),
                        tags=[InputTag(tag.key, tag.value) for tag in dataset.tags],
                    )
                    for dataset in record.dataset_inputs
                ],
                model_inputs=[LoggedModelInput(model_id) for model_id in record.model_inputs],
            ),
            RunOutputs(
                model_outputs=[
                    LoggedModelOutput(model_id=model.model_id, step=model.step)
                    for model in record.model_outputs
                ]
            ),
        )

    @handle_persistence_error("Unable to delete run.", logger=logger)
    def delete_run(self, run_id: str) -> None:
        try:
            self._run_repository.mark_deleted(
                run_id=run_id,
                deleted_time=get_current_time_millis(),
            )
        except RepositoryNotFoundError as exc:
            logger.error("Unable to delete run: %s", exc)
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from None

    @handle_persistence_error("Unable to restore run.", logger=logger)
    def restore_run(self, run_id: str) -> None:
        try:
            self._run_repository.restore(run_id=run_id)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to restore run: %s", exc)
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from None

    @handle_persistence_error("Unable to update run info.", logger=logger)
    def update_run_info(
        self,
        run_id: str,
        run_status: RunStatus | None,
        end_time: int | None,
        run_name: str | None,
    ) -> RunInfo:
        run = self._run_repository.find_by_id(run_id)
        if run is None:
            raise MlflowException(f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST)
        if run.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                (
                    f"The run {run.run_id} must be in the 'active' state. "
                    f"Current state is {run.lifecycle_stage}."
                ),
                INVALID_PARAMETER_VALUE,
            )

        status = RunStatus.to_string(run_status) if run_status is not None else None
        try:
            updated = self._run_repository.update_info(
                run_id=run_id,
                status=status,
                end_time=end_time,
                run_name=run_name,
            )
        except RepositoryNotFoundError as exc:
            logger.error("Unable to update run info: %s", exc)
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from None

        return RunInfo(
            run_id=updated.run_id,
            experiment_id=updated.experiment_id,
            user_id=updated.user_id,
            status=updated.status,
            start_time=updated.start_time,
            end_time=updated.end_time,
            lifecycle_stage=updated.lifecycle_stage,
            artifact_uri=updated.artifact_uri,
            run_name=updated.name,
        )

    @handle_persistence_error("Unable to log run batch.", logger=logger)
    def log_batch(
        self, run_id: str, metrics: list[Metric], params: list[Param], tags: list[RunTag]
    ) -> None:
        _validate_run_id(run_id)
        metrics, params, tags = _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        _validate_param_keys_unique(params)

        try:
            self._run_repository.log_batch(
                run_id=run_id,
                metrics=[
                    {
                        "k": metric.key,
                        "v": metric.value,
                        "timestamp": metric.timestamp,
                        "step": metric.step,
                        "model_id": metric.model_id,
                        "dataset_name": metric.dataset_name,
                        "dataset_digest": metric.dataset_digest,
                    }
                    for metric in metrics
                ],
                params=[{"k": param.key, "v": param.value} for param in params],
                tags=[{"k": tag.key, "v": tag.value} for tag in tags],
            )
        except RepositoryParamConflictError as exc:
            logger.error("Unable to log run batch: %s", exc)
            key, old_value, new_value, conflicting_run_id = exc.args
            raise MlflowException(
                f"Changing param values is not allowed. Param with key='{key}' was already logged "
                f"with value='{old_value}' for run ID='{conflicting_run_id}'. Attempted logging "
                f"new value '{new_value}'.",
                INVALID_PARAMETER_VALUE,
            ) from None
        except RepositoryNotActiveError as exc:
            logger.error("Unable to log run batch: %s", exc)
            _, lifecycle_stage = exc.args
            raise MlflowException(
                f"The run {run_id} must be in the 'active' state. "
                f"Current state is {lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            ) from None
        except RepositoryNotFoundError as exc:
            logger.error("Unable to log run batch: %s", exc)
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from None

    @handle_persistence_error("Unable to log run inputs.", logger=logger)
    def log_inputs(
        self,
        run_id: str,
        datasets: list[DatasetInput] | None = None,
        models: list[LoggedModelInput] | None = None,
    ) -> None:
        _validate_run_id(run_id)
        if datasets is not None:
            if not isinstance(datasets, list):
                raise TypeError(f"Argument 'datasets' should be a list, got '{type(datasets)}'")
            _validate_dataset_inputs(datasets)

        dataset_inputs = [
            {
                "dataset": dataset_input.dataset.to_dictionary(),
                "tags": [{"k": tag.key, "v": tag.value} for tag in dataset_input.tags],
            }
            for dataset_input in datasets or []
        ]
        model_inputs = [{"model_id": model.model_id} for model in models or []]
        try:
            self._run_repository.log_inputs(
                run_id=run_id, datasets=dataset_inputs, models=model_inputs
            )
        except RepositoryNotFoundError as exc:
            logger.error("Unable to log run inputs: %s", exc)
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from None
        except RepositoryNotActiveError as exc:
            logger.error("Unable to log run inputs: %s", exc)
            _, lifecycle_stage = exc.args
            raise MlflowException(
                f"The run {run_id} must be in the 'active' state. "
                f"Current state is {lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            ) from None

    @handle_persistence_error("Unable to log run outputs.", logger=logger)
    def log_outputs(self, run_id: str, models: list[LoggedModelOutput]) -> None:
        _validate_run_id(run_id)
        model_outputs = [{"model_id": model.model_id, "step": model.step} for model in models]
        try:
            self._run_repository.log_outputs(run_id=run_id, models=model_outputs)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to log run outputs: %s", exc)
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from None
        except RepositoryNotActiveError as exc:
            logger.error("Unable to log run outputs: %s", exc)
            _, lifecycle_stage = exc.args
            raise MlflowException(
                f"The run {run_id} must be in the 'active' state. "
                f"Current state is {lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            ) from None

    @handle_persistence_error("Unable to read metric history.", logger=logger)
    def get_metric_history(
        self,
        run_id: str,
        metric_key: str,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[Metric]:
        _validate_run_id(run_id)
        _validate_metric_name(metric_key)
        if max_results is not None and (
            isinstance(max_results, bool) or not isinstance(max_results, int) or max_results <= 0
        ):
            raise MlflowException(
                "max_results must be a positive integer.", INVALID_PARAMETER_VALUE
            )
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        if offset < 0:
            raise MlflowException("Page offset must not be negative.", INVALID_PARAMETER_VALUE)

        metrics = self._run_repository.get_metric_history(
            run_id=run_id,
            metric_key=metric_key,
            offset=offset,
            limit=max_results + 1 if max_results is not None else None,
        )

        next_token = None
        if max_results is not None and len(metrics) > max_results:
            metrics = metrics[:max_results]
            next_token = SearchUtils.create_page_token(offset + max_results)
        return PagedList(
            [
                Metric(
                    key=metric.key,
                    value=metric.value,
                    timestamp=metric.timestamp,
                    step=metric.step,
                    model_id=metric.model_id,
                    dataset_name=metric.dataset_name,
                    dataset_digest=metric.dataset_digest,
                    run_id=run_id,
                )
                for metric in metrics
            ],
            next_token,
        )

    @handle_persistence_error("Unable to start trace.", logger=logger)
    def start_trace(self, trace_info: TraceInfo) -> TraceInfo:
        experiment = self.get_experiment(trace_info.experiment_id)

        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                f"The experiment {experiment.experiment_id} must be in the 'active' state. "
                f"Current state is {experiment.lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            )

        tags = dict(trace_info.tags)
        # Older clients require artifacts even when spans are also persisted in MongoDB.
        tags[MLFLOW_ARTIFACT_LOCATION] = append_to_uri_path(
            experiment.artifact_location, "traces", trace_info.trace_id, "artifacts"
        )
        tags[SPANS_LOCATION_TAG] = SPANS_LOCATION
        source_run_id = trace_info.trace_metadata.get(TraceMetadataKey.SOURCE_RUN)
        run_ids = [source_run_id] if source_run_id else []
        assessments = []
        for assessment in trace_info.assessments:
            document = self._assessment_to_document(assessment)
            if not document.get("assessment_id"):
                document["assessment_id"] = uuid4().hex
            if not document.get("trace_id"):
                document["trace_id"] = trace_info.trace_id
            assessments.append(document)

        record = self._trace_repository.start_trace(
            trace_id=trace_info.trace_id,
            experiment_id=experiment.experiment_id,
            request_time=trace_info.request_time,
            state=trace_info.state.value,
            execution_duration=trace_info.execution_duration,
            client_request_id=trace_info.client_request_id,
            request_preview=trace_info.request_preview,
            response_preview=trace_info.response_preview,
            tags=tags,
            trace_metadata=trace_info.trace_metadata,
            assessments=assessments,
            run_ids=run_ids,
            span_stats=(
                self._parse_span_stats(trace_info.trace_metadata[TraceMetadataKey.SIZE_STATS])
                if TraceMetadataKey.SIZE_STATS in trace_info.trace_metadata
                else None
            ),
            metrics={
                field: self._numeric_trace_stats(trace_info.trace_metadata[key])
                for field, key in (
                    ("token_usage", TraceMetadataKey.TOKEN_USAGE),
                    ("cost", TRACE_COST_METADATA),
                )
                if key in trace_info.trace_metadata
            },
        )

        return self._to_trace_info(record)

    @staticmethod
    def _parse_span_stats(value: str) -> dict[str, Any]:
        """Decode MLflow's serialized stats at the store boundary."""
        if not value:
            return {}
        try:
            stats = json.loads(value)
        except (TypeError, ValueError) as exc:
            logger.error("Unable to parse trace size stats: %s", exc)
            raise MlflowException.invalid_parameter_value(
                "Invalid trace size-stats JSON."
            ) from None
        if not isinstance(stats, dict):
            raise MlflowException.invalid_parameter_value("Trace size stats must be an object.")
        count = stats.get(TraceSizeStatsKey.NUM_SPANS, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise MlflowException.invalid_parameter_value(
                "Trace size-stats num_spans must be a non-negative integer."
            )
        return stats

    @staticmethod
    def _to_trace_info(record: TraceRecord) -> TraceInfo:
        return TraceInfo(
            trace_id=record.trace_id,
            trace_location=TraceLocation.from_experiment_id(record.experiment_id),
            request_time=record.request_time,
            state=TraceState(record.state),
            execution_duration=record.execution_duration,
            client_request_id=record.client_request_id,
            request_preview=record.request_preview,
            response_preview=record.response_preview,
            tags={tag.key: tag.value for tag in record.tags},
            trace_metadata={item.key: item.value for item in record.trace_metadata},
            assessments=[
                MongoDBTrackingStore._assessment_from_document(item) for item in record.assessments
            ],
        )

    @handle_persistence_error("Unable to delete traces.", logger=logger)
    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        # Keep validation here for compatibility with MLflow versions that do not
        # implement the public delete_traces wrapper in AbstractStore.
        if max_timestamp_millis is None and not trace_ids:
            raise MlflowException.invalid_parameter_value(
                "Either `max_timestamp_millis` or `trace_ids` must be specified."
            )
        if max_timestamp_millis is not None and trace_ids:
            raise MlflowException.invalid_parameter_value(
                "Only one of `max_timestamp_millis` and `trace_ids` can be specified."
            )
        if trace_ids and max_traces is not None:
            raise MlflowException.invalid_parameter_value(
                "`max_traces` can't be specified if `trace_ids` is specified."
            )
        if max_traces is not None and max_traces <= 0:
            raise MlflowException.invalid_parameter_value(
                f"`max_traces` must be a positive integer, received {max_traces}."
            )

        return self._trace_repository.delete_traces(
            experiment_id=experiment_id,
            max_timestamp_millis=max_timestamp_millis,
            max_traces=max_traces,
            trace_ids=list(trace_ids) if trace_ids is not None else None,
        )

    @handle_persistence_error("Unable to read trace metadata.", logger=logger)
    def get_trace_info(self, trace_id: str) -> TraceInfo:
        return self._to_trace_info(self._get_trace_record(trace_id))

    @handle_persistence_error("Unable to read traces.", logger=logger)
    def batch_get_traces(
        self,
        trace_ids: list[str],
        location: str | None = None,  # ruff: ignore[unused-method-argument]
        experiment_ids: list[str] | None = None,
    ) -> list[Trace]:
        """Return complete traces in request order, omitting missing or incomplete traces.

        The connection selects MongoDB; location is unused by this backend.
        """
        if not trace_ids or experiment_ids == []:
            return []

        records = self._trace_repository.batch_get_traces(trace_ids, experiment_ids=experiment_ids)

        traces = []
        for record, span_records in records:
            if not span_records:
                continue
            if record.span_stats and len(span_records) < record.span_stats.get(
                TraceSizeStatsKey.NUM_SPANS, 0
            ):
                continue
            traces.append(self._to_trace(record, span_records))
        return traces

    @handle_persistence_error("Unable to read trace metadata.", logger=logger)
    def batch_get_trace_infos(
        self,
        trace_ids: list[str],
        location: str | None = None,  # ruff: ignore[unused-method-argument]
        experiment_ids: list[str] | None = None,
    ) -> list[TraceInfo]:
        """Return scoped trace metadata in request order without loading spans.

        The connection selects MongoDB; location is unused by this backend.
        """
        if not trace_ids or experiment_ids == []:
            return []

        records = self._trace_repository.batch_get_trace_infos(
            trace_ids, experiment_ids=experiment_ids
        )

        return [self._to_trace_info(record) for record in records]

    def _get_trace_record(self, trace_id: str) -> TraceRecord:
        try:
            record = self._trace_repository.get_trace_info(trace_id)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to read trace metadata: %s", exc)
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None

        return record

    @handle_persistence_error("Unable to read trace.", logger=logger)
    def get_trace(self, trace_id: str, *, allow_partial: bool = False) -> Trace:
        """Load persisted spans, retrying incomplete exports unless partial reads are allowed."""
        try:
            return self._get_trace(trace_id, allow_partial=allow_partial)
        except _TraceNotFullyExportedError as exc:
            logger.error("Unable to read complete trace: %s", exc)
            raise MlflowException(
                f"Trace with ID {trace_id} is not fully exported yet, please try again later.",
                RESOURCE_DOES_NOT_EXIST,
            ) from None

    @retry_on_exception(_TraceNotFullyExportedError, attempts=3, backoff_seconds=(1, 2))
    def _get_trace(self, trace_id: str, *, allow_partial: bool) -> Trace:
        # Refresh metadata and spans on each attempt because they arrive separately.
        # Missing metadata and database errors propagate immediately without retrying.
        record = self._get_trace_record(trace_id)

        span_documents = self._trace_repository.get_spans(trace_id)

        if not allow_partial:
            if trace_stats := record.span_stats:
                expected_spans = trace_stats.get(TraceSizeStatsKey.NUM_SPANS, 0)
                if len(span_documents) < expected_spans:
                    raise _TraceNotFullyExportedError
            # Without stats, a nonempty result has no further completeness check.
            if not span_documents:
                raise _TraceNotFullyExportedError

        return self._to_trace(record, span_documents)

    def _to_trace(self, record: TraceRecord, span_records: list[SpanRecord]) -> Trace:
        trace_info = self._to_trace_info(record)
        spans = [Span.from_dict(span_record.content) for span_record in span_records]
        spans.sort(key=lambda span: (span.parent_id is not None, span.start_time_ns, span.span_id))
        return Trace(info=trace_info, data=TraceData(spans=spans))

    @handle_persistence_error("Unable to log spans.", logger=logger)
    def log_spans(
        self,
        location: str,
        spans: list[Span],
        tracking_uri: str | None = None,  # ruff: ignore[unused-method-argument]
    ) -> list[Span]:
        """Persist spans and refresh their traces, allowing repeated and late delivery.

        The store's connection selects MongoDB; tracking_uri is unused by this backend.
        Writes across collections are separate and may partially succeed on failure.
        """
        if not spans:
            return []
        if not isinstance(location, str) or not location:
            raise MlflowException.invalid_parameter_value("location must be an experiment ID.")

        self._validate_span_ingestion_experiment(location)
        documents, documents_by_trace = self._prepare_span_documents(spans)
        self._ensure_span_traces(location, documents_by_trace)
        self._persist_span_documents(documents)
        self._refresh_span_summaries(location, documents_by_trace)
        return spans

    def _validate_span_ingestion_experiment(self, experiment_id: str) -> None:
        experiment = self.get_experiment(experiment_id)

        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException.invalid_parameter_value(
                f"The experiment {experiment_id} must be in the 'active' state. "
                f"Current state is {experiment.lifecycle_stage}."
            )

    def _ensure_span_traces(
        self,
        experiment_id: str,
        documents_by_trace: defaultdict[str, list[dict[str, Any]]],
    ) -> None:
        request_times = {
            trace_id: min(document["start_time_ns"] for document in trace_documents) // 1_000_000
            for trace_id, trace_documents in documents_by_trace.items()
        }
        self._trace_repository.ensure_traces(
            experiment_id=experiment_id, request_times=request_times
        )

    def _persist_span_documents(self, documents: list[dict[str, Any]]) -> None:
        self._trace_repository.log_spans(documents)

    def _refresh_span_summaries(
        self,
        experiment_id: str,
        documents_by_trace: defaultdict[str, list[dict[str, Any]]],
    ) -> None:
        try:
            for trace_id in documents_by_trace:
                self._refresh_trace_span_summary(trace_id, experiment_id)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to refresh trace span summaries: %s", exc)
            raise MlflowException(
                f"Trace '{exc.args[0]}' was deleted during span ingestion.",
                RESOURCE_DOES_NOT_EXIST,
            ) from None
        except RepositoryWriteConflictError as exc:
            logger.error("Unable to refresh trace span summaries: %s", exc)
            raise MlflowException(
                f"Concurrent span writes prevented refreshing trace '{exc.args[0]}'. "
                "Retry log_spans to refresh its summary; stored spans are deduplicated.",
                TEMPORARILY_UNAVAILABLE,
            ) from None

    @staticmethod
    def _span_attribute_value(value: Any) -> Any:
        # Span.to_dict() retains MLflow's JSON-encoded OTel attribute values.
        # Decode only the fields used for queries; preserve content for Span.from_dict().
        return try_json_loads(value) if isinstance(value, str) else value

    @classmethod
    def _numeric_trace_stats(cls, value: Any) -> dict | None:
        """Read optional numeric usage/cost attributes without failing span ingestion."""
        value = cls._span_attribute_value(value)
        if not isinstance(value, dict):
            return None
        if any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or (isinstance(number, float) and not math.isfinite(number))
            for number in value.values()
        ):
            return None
        return value

    @staticmethod
    def _add_span_trace_tag(tags: dict[str, str], key: str, value: Any) -> None:
        try:
            value = value if isinstance(value, str) else json.dumps(value)
            key, value = _validate_trace_tag(key, value)
        except (MlflowException, TypeError, ValueError):
            logger.debug("Skipping invalid span-derived trace tag %r", key)
            return
        tags[key] = value

    @staticmethod
    def _span_preview(value: Any, role: str) -> str | None:
        try:
            if value is not None and not isinstance(value, (str, dict)):
                value = json.dumps(value)
            preview = _get_truncated_preview(value, role=role)
            # MLflow's helper consults the process-global URI for its size limit.
            # A direct MongoDB store always uses the OSS limit, even in mixed clients.
            if preview is not None and len(preview) > TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS:
                return preview[: TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS - 3] + "..."
            return preview
        except (TypeError, ValueError, AttributeError) as exc:
            logger.debug("Could not extract the %s trace preview: %s", role, exc)
            return None

    @classmethod
    def _span_to_document(cls, span: Span) -> dict[str, Any]:
        """Keep full span content plus compact native fields for summary queries."""

        content = translate_span_when_storing(span)
        attributes = content.get("attributes", {})
        root = span.parent_id is None
        resource_tags = {}
        resource = getattr(span._span, "resource", None)
        if resource is not None:
            for key, value in resource.attributes.items():
                if not key.startswith(("telemetry.sdk.", "mlflow.")):
                    cls._add_span_trace_tag(resource_tags, key, value)
        root_tags = {}
        if root:
            for key, value in attributes.items():
                if key.startswith(SPAN_TRACE_TAG_PREFIX):
                    tag_key = key[len(SPAN_TRACE_TAG_PREFIX) :]
                    if tag_key not in (SPANS_LOCATION_TAG, MLFLOW_ARTIFACT_LOCATION):
                        cls._add_span_trace_tag(
                            root_tags, tag_key, cls._span_attribute_value(value)
                        )

        metadata = {}
        session_id = attributes.get(SPAN_SESSION_ATTRIBUTE) or attributes.get(
            CONVERSATION_ID_ATTRIBUTE
        )
        for key, value in (
            (TraceMetadataKey.TRACE_SESSION, session_id),
            (TraceMetadataKey.TRACE_USER, attributes.get(SPAN_USER_ATTRIBUTE)),
        ):
            if value is not None:
                metadata[key] = str(cls._span_attribute_value(value))

        return {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_id,
            "name": span.name,
            "type": cls._span_attribute_value(attributes.get(SpanAttributeKey.SPAN_TYPE))
            or span.span_type,
            "status": span.status.status_code.value,
            "start_time_ns": span.start_time_ns,
            "end_time_ns": span.end_time_ns,
            "content": content,
            "dimension_attributes": {
                field: cls._span_attribute_value(attributes[attribute])
                for field, attribute in (
                    (SpanDimensionKey.MODEL, SPAN_MODEL_ATTRIBUTE),
                    (SpanDimensionKey.MODEL_PROVIDER, SPAN_PROVIDER_ATTRIBUTE),
                )
                if attribute in attributes
            },
            "token_usage": cls._numeric_trace_stats(attributes.get(SpanAttributeKey.CHAT_USAGE)),
            "cost": cls._numeric_trace_stats(attributes.get(SPAN_COST_ATTRIBUTE)),
            "trace_fields": {
                "metadata": metadata,
                "resource_tags": resource_tags,
                "root_tags": root_tags,
                "request_preview": cls._span_preview(
                    attributes.get(SpanAttributeKey.INPUTS), "user"
                )
                if root
                else None,
                "response_preview": cls._span_preview(
                    attributes.get(SpanAttributeKey.OUTPUTS), "assistant"
                )
                if root
                else None,
            },
        }

    @staticmethod
    def _summarize_spans(documents: list[SpanSummaryRecord]) -> dict[str, Any]:
        """Recompute from unique persisted spans, including parents arriving in later batches."""

        start_ms = min(document.start_time_ns for document in documents) // 1_000_000
        end_times = [d.end_time_ns for d in documents if d.end_time_ns is not None]
        root = next((d for d in documents if d.parent_span_id is None), None)
        metadata = {}
        tags = {}
        for document in documents:
            for key, value in document.trace_fields["metadata"].items():
                metadata.setdefault(key, value)
            for key, value in document.trace_fields["resource_tags"].items():
                tags.setdefault(key, value)
        if root:
            tags.update(root.trace_fields["root_tags"])

        if not root:
            state = TraceState.IN_PROGRESS.value
        elif root.status == SpanStatusCode.ERROR.value:
            state = TraceState.ERROR.value
        else:
            state = TraceState.OK.value

        token_usage, cost = aggregate_span_summaries(documents)
        return {
            "request_time": start_ms,
            "execution_duration": max(end_times) // 1_000_000 - start_ms if end_times else None,
            "state": state,
            "request_preview": root.trace_fields["request_preview"] if root else None,
            "response_preview": root.trace_fields["response_preview"] if root else None,
            "tags": tags,
            "metadata": metadata,
            "token_usage": token_usage,
            "cost": cost,
        }

    @classmethod
    def _prepare_span_documents(
        cls,
        spans: list[Span],
    ) -> tuple[list[dict[str, Any]], defaultdict[str, list[dict[str, Any]]]]:
        # First delivery wins, including repeated identities in the same batch.
        # Prepare documents before creating placeholders; the driver validates BSON on write.
        documents_by_identity = {}
        for span in spans:
            identity = (span.trace_id, span.span_id)
            if identity not in documents_by_identity:
                documents_by_identity[identity] = cls._span_to_document(span)

        documents_by_trace = defaultdict(list)
        for document in documents_by_identity.values():
            documents_by_trace[document["trace_id"]].append(document)
        return list(documents_by_identity.values()), documents_by_trace

    @retry_on_exception(RepositoryWriteConflictError, attempts=3, backoff_seconds=(0.01, 0.05))
    def _refresh_trace_span_summary(self, trace_id: str, experiment_id: str) -> None:
        revision, documents = self._trace_repository.span_summary_snapshot(
            trace_id=trace_id, experiment_id=experiment_id
        )
        if not documents:
            # Span deletion precedes trace deletion; do not finalize an empty
            # snapshot or invent a successful write while deletion is in progress.
            raise RepositoryNotFoundError(trace_id)
        summary = self._summarize_spans(documents)
        # MLflow's trace_metadata contract requires strings. The corresponding
        # native fields are stored in the same atomic update for MongoDB queries.
        summary["aggregate_metadata"] = {
            key: json.dumps(summary[field])
            for field, key in (
                ("token_usage", TraceMetadataKey.TOKEN_USAGE),
                ("cost", TRACE_COST_METADATA),
            )
            if summary[field] is not None
        }
        self._trace_repository.update_span_summary(
            trace_id=trace_id, experiment_id=experiment_id, revision=revision, summary=summary
        )

    async def log_spans_async(self, location: str, spans: list[Span]) -> list[Span]:
        return await asyncio.to_thread(self.log_spans, location, spans)

    @handle_persistence_error("Unable to set trace tag.", logger=logger)
    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        key, value = _validate_trace_tag(key, value)
        try:
            self._trace_repository.set_trace_tag(trace_id=trace_id, key=key, value=value)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to set trace tag: %s", exc)
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None

    @handle_persistence_error("Unable to delete trace tag.", logger=logger)
    def delete_trace_tag(self, trace_id: str, key: str) -> None:
        try:
            self._trace_repository.delete_trace_tag(trace_id=trace_id, key=key)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to delete trace tag: %s", exc)
            raise MlflowException(
                f"Trace '{trace_id}' or tag '{key}' not found.",
                RESOURCE_DOES_NOT_EXIST,
            ) from None

    @staticmethod
    def _assessment_to_document(assessment: Assessment) -> dict[str, Any]:
        document = assessment.to_dictionary()
        if assessment.expectation is not None:
            document["expectation"] = {"value": deepcopy(assessment.expectation.value)}
        elif assessment.feedback is not None:
            document["feedback"]["value"] = deepcopy(assessment.feedback.value)
        if assessment.run_id is not None:
            document["run_id"] = assessment.run_id
        return document

    @staticmethod
    def _assessment_from_document(document: dict[str, Any]) -> Assessment:
        assessment = Assessment.from_dictionary(document)
        assessment.valid = document.get("valid", True)
        assessment.overrides = document.get("overrides")
        assessment.run_id = document.get("run_id", assessment.run_id)
        return assessment

    # Logged models

    @handle_persistence_error("Unable to create logged model.", logger=logger)
    def create_logged_model(
        self,
        experiment_id: str,
        name: str | None = None,
        source_run_id: str | None = None,
        tags: list[LoggedModelTag] | None = None,
        params: list[LoggedModelParameter] | None = None,
        model_type: str | None = None,
    ) -> LoggedModel:
        """Create a pending logged model and persist its metadata atomically."""
        _validate_logged_model_name(name)

        # Preserve SQLAlchemyStore's rejection of duplicate and null entries
        # before embedding them in arrays in a single MongoDB document.
        for field, entries in (("params", params), ("tags", tags)):
            seen_keys = set()
            for entry in entries or []:
                if entry.key is None or entry.value is None or entry.key in seen_keys:
                    raise MlflowException(
                        f"Logged model {field} must have unique, non-null keys "
                        "and non-null values.",
                        BAD_REQUEST,
                    )
                seen_keys.add(entry.key)

        experiment = self.get_experiment(experiment_id)

        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                (
                    f"The experiment {experiment.experiment_id} must be in the 'active' state. "
                    f"Current state is {experiment.lifecycle_stage}."
                ),
                INVALID_PARAMETER_VALUE,
            )

        model_id = f"m-{uuid4().hex}"
        artifact_location = append_to_uri_path(
            experiment.artifact_location, "models", model_id, "artifacts"
        )
        record = self._logged_model_repository.create(
            model_id=model_id,
            experiment_id=experiment.experiment_id,
            name=name or _generate_random_name(),
            artifact_location=artifact_location,
            creation_timestamp=get_current_time_millis(),
            status=LoggedModelStatus.PENDING.value,
            lifecycle_stage=LifecycleStage.ACTIVE,
            source_run_id=source_run_id,
            model_type=model_type,
            tags={tag.key: tag.value for tag in tags or []},
            params={param.key: param.value for param in params or []},
        )

        return self._to_logged_model(record)

    @staticmethod
    def _to_logged_model(
        record: LoggedModelRecord, metrics: tuple[RunMetricRecord, ...] = ()
    ) -> LoggedModel:
        return LoggedModel(
            model_id=record.model_id,
            experiment_id=record.experiment_id,
            name=record.name,
            artifact_location=record.artifact_location,
            creation_timestamp=record.creation_timestamp,
            last_updated_timestamp=record.last_updated_timestamp,
            status=LoggedModelStatus(record.status),
            status_message=record.status_message,
            source_run_id=record.source_run_id,
            model_type=record.model_type,
            tags=[LoggedModelTag(tag.key, tag.value) for tag in record.tags],
            params=[LoggedModelParameter(param.key, param.value) for param in record.params],
            metrics=[
                Metric(
                    key=metric.key,
                    value=metric.value,
                    timestamp=metric.timestamp,
                    step=metric.step,
                    model_id=record.model_id,
                    run_id=metric.run_id,
                    dataset_name=metric.dataset_name,
                    dataset_digest=metric.dataset_digest,
                )
                for metric in metrics
            ]
            or None,
        )

    @staticmethod
    def _normalize_logged_model_search_key(key: str) -> str:
        if key in ("creation_timestamp_ms", "last_updated_timestamp_ms"):
            return key.removesuffix("_ms")
        return key

    @staticmethod
    def _stored_logged_model_attribute_key(key: str) -> str:
        if key == "creation_time":
            return "creation_timestamp"
        if key == "last_updated_time":
            return "last_updated_timestamp"
        return key

    @staticmethod
    def _is_numeric_logged_model_filter(clause: SearchFilterClause) -> bool:
        # Preserve classification after the existing attribute-to-storage key translation.
        return clause.field_type == "metric" or (
            clause.field_type == "attribute"
            and MongoDBTrackingStore._stored_logged_model_attribute_key(clause.key)
            in SearchLoggedModelsUtils.NUMERIC_ATTRIBUTES
        )

    @staticmethod
    def _validate_logged_model_filter_value(clause: SearchFilterClause) -> None:
        if not clause.key:
            raise RepositoryEmptySearchKeyError("Search key must not be empty.")
        if MongoDBTrackingStore._is_numeric_logged_model_filter(clause):
            if clause.field_type == "attribute" and not isinstance(clause.value, (int, float)):
                raise RepositoryInvalidFilterValueError("finite numbers")
            if not math.isfinite(clause.value):
                raise RepositoryInvalidFilterValueError("finite numbers")
        elif clause.comparator in ("IN", "NOT IN"):
            if not isinstance(clause.value, (list, tuple)) or not all(
                isinstance(value, str) for value in clause.value
            ):
                raise RepositoryInvalidFilterValueError("a list of string values")
        elif not isinstance(clause.value, str):
            raise RepositoryInvalidFilterValueError("string values")

    @staticmethod
    def _logged_model_attribute_key(key: str, *, order_by: bool = False) -> str:
        # MLflow's filter parser emits SQL timestamp names; MongoDB stores entity names.
        key = MongoDBTrackingStore._normalize_logged_model_search_key(key)
        valid_keys = (
            SearchLoggedModelsUtils.VALID_ORDER_BY_ATTRIBUTE_KEYS
            if order_by
            else SearchLoggedModelsUtils.VALID_SEARCH_ATTRIBUTE_KEYS
        )
        if key not in valid_keys:
            raise MlflowException.invalid_parameter_value(
                f"Invalid logged model attribute: {key!r}."
            )
        return MongoDBTrackingStore._stored_logged_model_attribute_key(key)

    @staticmethod
    def _validate_logged_model_datasets(datasets: list[dict[str, Any]] | None) -> None:
        if datasets is None:
            return
        if not isinstance(datasets, list):
            raise MlflowException.invalid_parameter_value(
                "`datasets` must be a list of dictionaries."
            )
        for dataset in datasets:
            if not isinstance(dataset, dict) or not dataset.get("dataset_name"):
                raise MlflowException.invalid_parameter_value(
                    "`dataset_name` in the `datasets` clause must be specified."
                )
            if not isinstance(dataset["dataset_name"], str) or (
                dataset.get("dataset_digest") is not None
                and not isinstance(dataset["dataset_digest"], str)
            ):
                raise MlflowException.invalid_parameter_value(
                    "Dataset names and digests must be strings."
                )

    @cached_property
    def _logged_model_filter_validator(self) -> SearchFilterValidator:
        # parse_filter_string already validates operators with Entity.validate_op; its rules
        # differ from SearchLoggedModelsUtils for tag/param membership and timestamp aliases.
        return ConfiguredSearchFilterValidator(
            field_types=("attribute", "metric", "param", "tag"),
            attribute_keys=SearchLoggedModelsUtils.VALID_SEARCH_ATTRIBUTE_KEYS,
            rules=(self._validate_logged_model_filter_value,),
            uppercase_comparators=False,
        )

    def _parse_logged_model_filters(
        self, filter_string: str | None
    ) -> tuple[LoggedModelFilter, ...]:
        # The parser imports SQL models; defer it until MLflow finishes store discovery.
        from mlflow.utils.search_logged_model_utils import (  # ruff: ignore[import-outside-top-level]
            parse_filter_string,
        )

        if filter_string is not None and not isinstance(filter_string, str):
            raise MlflowException.invalid_parameter_value("`filter_string` must be a string.")
        try:
            comparisons = parse_filter_string(filter_string)
        except (ValueError, TypeError, SyntaxError) as exc:
            logger.error("Unable to parse logged model filter: %s", exc)
            raise MlflowException.invalid_parameter_value(
                "Invalid logged model filter string."
            ) from None

        filters = []
        for comparison in comparisons:
            field_type = comparison.entity.type.name.lower()
            key = comparison.entity.key
            if field_type == "attribute":
                key = self._normalize_logged_model_search_key(key)
            parsed = {
                "type": field_type,
                "key": key,
                "comparator": comparison.op,
                "value": comparison.value,
            }
            try:
                clause = self._logged_model_filter_validator.validate(parsed)
            except RepositoryInvalidAttributeError as exc:
                logger.error("Unable to validate logged model filter: %s", exc)
                raise MlflowException.invalid_parameter_value(
                    f"Invalid logged model attribute: {exc.key!r}."
                ) from None
            except RepositoryEmptySearchKeyError as exc:
                logger.error("Unable to validate logged model filter: %s", exc)
                raise MlflowException.invalid_parameter_value(
                    "Search keys must not be empty."
                ) from None
            except RepositoryInvalidFilterValueError as exc:
                logger.error("Unable to validate logged model filter value: %s", exc)
                messages = {
                    "finite numbers": "Numeric filters require finite numbers.",
                    "a list of string values": "IN and NOT IN require a list of string values.",
                    "string values": "String filters require string values.",
                }
                raise MlflowException.invalid_parameter_value(messages[exc.expected]) from None

            key = (
                self._stored_logged_model_attribute_key(clause.key)
                if clause.field_type == "attribute"
                else clause.key
            )
            value = clause.value
            if clause.comparator in ("IN", "NOT IN") and not self._is_numeric_logged_model_filter(
                clause
            ):
                value = tuple(value)
            filters.append(LoggedModelFilter(clause.field_type, key, clause.comparator, value))
        return tuple(filters)

    @classmethod
    def _parse_logged_model_order(
        cls, order_by: list[dict[str, Any]] | None
    ) -> tuple[LoggedModelOrder, ...]:
        if order_by is not None and not isinstance(order_by, list):
            raise MlflowException.invalid_parameter_value(
                "`order_by` must be a list of dictionaries."
            )
        orders = []
        seen = set()
        for order in order_by or []:
            if not isinstance(order, dict) or not isinstance(order.get("field_name"), str):
                raise MlflowException.invalid_parameter_value(
                    "`field_name` in the `order_by` clause must be specified as a string."
                )
            field = order["field_name"]
            if "." in field:
                entity, key = field.split(".", 1)
                if entity != "metrics" or not key:
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid order by field name: {field!r}. Only metrics support a prefix."
                    )
                field_type = "metric"
            else:
                key = cls._logged_model_attribute_key(field, order_by=True)
                field_type = "attribute"
            ascending = order.get("ascending", True)
            if not isinstance(ascending, bool):
                raise MlflowException.invalid_parameter_value("`ascending` must be a boolean.")
            dataset_name = order.get("dataset_name")
            dataset_digest = order.get("dataset_digest")
            if any(
                value is not None and not isinstance(value, str)
                for value in (dataset_name, dataset_digest)
            ):
                raise MlflowException.invalid_parameter_value(
                    "Dataset names and digests must be strings."
                )
            if dataset_digest and not dataset_name:
                raise MlflowException.invalid_parameter_value(
                    "`dataset_digest` can only be specified if `dataset_name` is also specified."
                )
            if field_type != "metric" and (dataset_name or dataset_digest):
                raise MlflowException.invalid_parameter_value(
                    "Dataset ordering applies only to metrics."
                )
            identity = (field_type, key, dataset_name or None, dataset_digest or None)
            # Later repetitions of the same sort expression cannot change its ordering.
            if identity not in seen:
                seen.add(identity)
                orders.append(
                    LoggedModelOrder(field_type, key, ascending, dataset_name, dataset_digest)
                )
        for key, ascending in (("creation_timestamp", False), ("model_id", True)):
            if not any(order.field_type == "attribute" and order.key == key for order in orders):
                orders.append(LoggedModelOrder("attribute", key, ascending))
        sort_keys = sum(
            2
            if order.field_type == "metric"
            or order.key
            in {
                "model_type",
                "source_run_id",
                "status_message",
            }
            else 1
            for order in orders
        )
        if sort_keys > 32:
            raise MlflowException.invalid_parameter_value("Too many order_by fields.")
        return tuple(orders)

    @staticmethod
    def _parse_logged_model_page_token(
        page_token: str | None,
        experiment_ids: list[str],
        filter_string: str | None,
        order_by: list[dict[str, Any]] | None,
    ) -> int:
        if page_token is not None and not isinstance(page_token, str):
            raise MlflowException.invalid_parameter_value("Invalid logged model page token.")
        if not page_token:
            return 0
        try:
            token = SearchLoggedModelsPaginationToken.decode(page_token)
        except (MlflowException, ValueError, TypeError, AttributeError, binascii.Error) as exc:
            logger.error("Unable to parse logged model page token: %s", exc)
            raise MlflowException.invalid_parameter_value(
                "Invalid logged model page token."
            ) from None
        if (
            isinstance(token.offset, bool)
            or not isinstance(token.offset, int)
            or not 0 <= token.offset < 2**63
        ):
            raise MlflowException.invalid_parameter_value("Invalid logged model page token offset.")
        token.validate(experiment_ids, filter_string or None, order_by or None)
        return token.offset

    @handle_persistence_error("Unable to search logged models.", logger=logger)
    def search_logged_models(
        self,
        experiment_ids: list[str],
        filter_string: str | None = None,
        datasets: list[dict[str, Any]] | None = None,
        max_results: int | None = None,
        order_by: list[dict[str, Any]] | None = None,
        page_token: str | None = None,
    ) -> PagedList[LoggedModel]:
        """Search model metadata and associated metrics within the requested experiments."""
        self._validate_logged_model_datasets(datasets)
        if not isinstance(experiment_ids, list) or not all(
            isinstance(experiment_id, str) for experiment_id in experiment_ids
        ):
            raise MlflowException.invalid_parameter_value(
                "`experiment_ids` must be a list of strings."
            )
        offset = self._parse_logged_model_page_token(
            page_token, experiment_ids, filter_string, order_by
        )
        if isinstance(max_results, bool) or (
            max_results is not None and not isinstance(max_results, int)
        ):
            raise MlflowException.invalid_parameter_value("`max_results` must be an integer.")
        max_results = max_results or SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT
        if max_results < 1:
            raise MlflowException.invalid_parameter_value(
                "`max_results` must be a positive integer."
            )
        filters = self._parse_logged_model_filters(filter_string)
        orders = self._parse_logged_model_order(order_by)
        if not experiment_ids:
            return PagedList([], None)
        page = self._logged_model_repository.search(
            experiment_ids=experiment_ids,
            filters=filters,
            datasets=datasets or [],
            order_by=orders,
            offset=offset,
            max_results=max_results,
        )

        next_token = (
            SearchLoggedModelsPaginationToken(
                experiment_ids=experiment_ids,
                filter_string=filter_string or None,
                order_by=order_by or None,
                offset=offset + max_results,
            ).encode()
            if page.has_more
            else None
        )
        return PagedList(
            [self._to_logged_model(result.model, result.metrics) for result in page.records],
            next_token,
        )

    @handle_persistence_error("Unable to get logged model.", logger=logger)
    def get_logged_model(self, model_id: str, allow_deleted: bool = False) -> LoggedModel:
        """Fetch model metadata and its complete associated metric history."""
        record = self._logged_model_repository.find_by_id(model_id)
        if record is None or (
            not allow_deleted and record.lifecycle_stage == LifecycleStage.DELETED
        ):
            raise MlflowException(
                f"Logged model with ID '{model_id}' not found.", RESOURCE_DOES_NOT_EXIST
            )
        metrics = self._logged_model_repository.get_metric_history(record.model_id)

        return self._to_logged_model(record, metrics)

    @handle_persistence_error("Unable to delete logged model.", logger=logger)
    def delete_logged_model(self, model_id: str) -> None:
        """Soft-delete a logged model and refresh its last-updated timestamp."""
        try:
            self._logged_model_repository.mark_deleted(
                model_id=model_id,
                last_updated_timestamp=get_current_time_millis(),
            )
        except RepositoryNotFoundError as exc:
            logger.error("Unable to delete logged model: %s", exc)
            raise MlflowException(
                f"Logged model with ID '{model_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None

    @handle_persistence_error("Unable to set logged model tags.", logger=logger)
    def set_logged_model_tags(self, model_id: str, tags: list[LoggedModelTag]) -> None:
        """Set model tags, keeping the last value for each key in the batch."""
        tags_by_key = {tag.key: tag.value for tag in tags}
        if any(k is None or v is None for k, v in tags_by_key.items()):
            raise MlflowException(
                "Logged model tags must have non-null keys and values.", BAD_REQUEST
            )
        try:
            self._logged_model_repository.set_tags(model_id=model_id, tags=tags_by_key)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to set logged model tags: %s", exc)
            raise MlflowException(
                f"Logged model with ID '{model_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None

    @handle_persistence_error("Unable to delete logged model tag.", logger=logger)
    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        """Delete a model tag, failing if the model or tag does not exist."""
        try:
            self._logged_model_repository.delete_tag(model_id=model_id, key=key)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to delete logged model tag: %s", exc)
            raise MlflowException(
                f"Logged model with ID '{model_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None
        except RepositoryTagNotFoundError as exc:
            logger.error("Unable to delete logged model tag: %s", exc)
            raise MlflowException(
                f"No tag with key {key!r} found for model with ID {model_id!r}.",
                RESOURCE_DOES_NOT_EXIST,
            ) from None

    @handle_persistence_error("Unable to link prompts to trace.", logger=logger)
    def link_prompts_to_trace(self, trace_id: str, prompt_versions: list[PromptVersion]) -> None:
        """Associate prompt versions with a trace by their public name/version identity."""
        if not prompt_versions:
            return

        refs = [
            {"name": prompt_version.name, "version": str(prompt_version.version)}
            for prompt_version in prompt_versions
        ]
        try:
            self._trace_repository.link_prompts(trace_id=trace_id, prompt_versions=refs)
        except RepositoryNotFoundError as exc:
            logger.error("Unable to link prompts to trace: %s", exc)
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None

    @handle_persistence_error("Unable to link traces to run.", logger=logger)
    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        """Add an idempotent run association to at most 100 traces per request."""
        if not trace_ids:
            return
        if not run_id:
            raise MlflowException.invalid_parameter_value("run_id cannot be empty")
        if len(trace_ids) > 100:
            raise MlflowException.invalid_parameter_value(
                "Cannot link more than 100 traces to a run in a single request. "
                f"Provided {len(trace_ids)} traces."
            )

        self._trace_repository.link_traces_to_run(trace_ids=list(trace_ids), run_id=run_id)

    def unlink_traces_from_run(self, trace_ids: list[str], run_id: str) -> None:
        raise NotImplementedError
