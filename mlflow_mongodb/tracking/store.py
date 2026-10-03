"""MongoDB tracking store with experiment persistence."""

import logging
from functools import cached_property
from uuid import uuid4

from mlflow.entities import Experiment, ExperimentTag, LifecycleStage, ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import append_to_uri_path, resolve_uri_if_local
from mlflow.utils.validation import (
    _validate_experiment_artifact_location,
    _validate_experiment_artifact_location_length,
    _validate_experiment_name,
    _validate_experiment_tag,
)
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConfigurationError

from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotActiveError,
    RepositoryNotFoundError,
    RepositoryPersistenceError,
)
from mlflow_mongodb.tracking.repositories import ExperimentRepository
from mlflow_mongodb.tracking.repositories.experiments import ExperimentFilter, ExperimentOrder

logger = logging.getLogger(__name__)


class MongoDBTrackingStore(AbstractStore):
    """Persist experiments in MongoDB; other tracking operations remain inherited."""

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
    def _experiment_repository(self) -> ExperimentRepository:
        return ExperimentRepository(self._database, settings=self._settings)

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
        try:
            records = self._experiment_repository.search(
                lifecycle_stages=LifecycleStage.view_type_to_stages(view_type),
                filters=filters,
                order_by=orders,
                offset=offset,
                limit=max_results + 1,
            )
        except RepositoryPersistenceError as exc:
            logger.error("Unable to search experiments: %s", exc)
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

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

    @staticmethod
    def _parse_experiment_filters(filter_string: str | None) -> tuple[ExperimentFilter, ...]:
        filters = []
        for parsed in SearchExperimentsUtils.parse_search_filter(filter_string):
            field_type = parsed["type"]
            key = parsed["key"]
            comparator = parsed["comparator"].upper()
            value = parsed["value"]
            if SearchExperimentsUtils.is_numeric_attribute(field_type, key, comparator):
                value = float(value)
            elif not (
                SearchExperimentsUtils.is_string_attribute(field_type, key, comparator)
                or SearchExperimentsUtils.is_tag(field_type, comparator)
            ):
                raise MlflowException.invalid_parameter_value(f"Invalid token type: {field_type}")
            filters.append(ExperimentFilter(field_type, key, comparator, value))
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
        # without requiring a shared counter or reserving the default ID "0".
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
        except RepositoryPersistenceError as exc:
            logger.error("Unable to create experiment: %s", exc)
            raise MlflowException("Unable to create experiment.", INTERNAL_ERROR) from None

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

    def delete_experiment(self, experiment_id: str) -> None:
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

    def restore_experiment(self, experiment_id: str) -> None:
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
        except RepositoryPersistenceError as exc:
            logger.error("Unable to rename experiment: %s", exc)
            raise MlflowException("Unable to rename experiment.", INTERNAL_ERROR) from None
