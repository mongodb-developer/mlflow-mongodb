"""Skeleton of the MongoDB tracking store for the agreed V1 scope."""

import logging
from functools import cached_property
from uuid import uuid4

from mlflow.entities import (
    Dataset,
    DatasetInput,
    Experiment,
    ExperimentTag,
    InputTag,
    LifecycleStage,
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
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import (
    SEARCH_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, _get_run_name_from_tags
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import (
    SearchExperimentsUtils,
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
    _validate_metric_name,
    _validate_param_keys_unique,
    _validate_run_id,
)
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConfigurationError

from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryInvalidAttributeError,
    RepositoryNotActiveError,
    RepositoryNotFoundError,
    RepositoryParamConflictError,
    RepositoryTagNotFoundError,
    RepositoryUnsupportedComparatorError,
    RepositoryUnsupportedFieldTypeError,
)
from mlflow_mongodb.infrastructure.search_filters import (
    ConfiguredSearchFilterValidator,
    SearchFilterValidator,
)
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.infrastructure.store_errors import handle_persistence_error
from mlflow_mongodb.tracking.repositories import (
    ExperimentRepository,
    RunRepository,
)
from mlflow_mongodb.tracking.repositories.experiments import ExperimentFilter, ExperimentOrder
from mlflow_mongodb.tracking.repositories.runs import RunFilter, RunOrder
from mlflow_mongodb.tracking.types import RunRecord

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
