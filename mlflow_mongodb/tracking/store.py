"""Skeleton of the MongoDB tracking store for the agreed V1 scope."""

import binascii
import logging
import math
from functools import cached_property
from typing import Any
from uuid import uuid4

from mlflow.entities import (
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
    ViewType,
)
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import (
    SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, _get_run_name_from_tags
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
)
from mlflow_mongodb.infrastructure.search_filters import (
    ConfiguredSearchFilterValidator,
    SearchFilterClause,
    SearchFilterValidator,
)
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.infrastructure.store_errors import handle_persistence_error
from mlflow_mongodb.tracking.repositories import (
    ExperimentRepository,
    LoggedModelRepository,
    RunRepository,
)
from mlflow_mongodb.tracking.repositories.experiments import ExperimentFilter, ExperimentOrder
from mlflow_mongodb.tracking.repositories.logged_models import LoggedModelFilter, LoggedModelOrder
from mlflow_mongodb.tracking.repositories.runs import RunFilter, RunOrder
from mlflow_mongodb.tracking.types import LoggedModelRecord, RunMetricRecord, RunRecord

logger = logging.getLogger(__name__)


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

    @cached_property
    @handle_persistence_error("Unable to initialize logged model repository.", logger=logger)
    def _logged_model_repository(self) -> LoggedModelRepository:
        return LoggedModelRepository(self._database, settings=self._settings)

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
