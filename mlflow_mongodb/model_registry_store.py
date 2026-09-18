"""MLflow Model Registry store backed by MongoDB."""

import logging
import urllib.parse
from functools import cached_property

from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelAlias,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.model_version_stages import (
    ALL_STAGES,
    DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS,
    STAGE_NONE,
    get_canonical_stage,
)
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from mlflow.prompt.registry_utils import (
    add_prompt_filter_string,
    handle_resource_already_exist_error,
    has_prompt_tag,
)
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.tracking.client import MlflowClient
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
    _validate_model_alias_name,
    _validate_model_name,
    _validate_model_renaming,
    _validate_model_version,
    _validate_model_version_tag,
    _validate_registered_model_tag,
    _validate_tag_name,
)
from pymongo import MongoClient
from pymongo.errors import ConfigurationError

from mlflow_mongodb.repositories import (
    ModelVersionAlreadyExistsError,
    ModelVersionFilter,
    ModelVersionNotFoundError,
    ModelVersionOrder,
    ModelVersionRecord,
    ModelVersionRepository,
    RegisteredModelAlreadyExistsError,
    RegisteredModelDetails,
    RegisteredModelFilter,
    RegisteredModelNotFoundError,
    RegisteredModelOrder,
    RegisteredModelRecord,
    RegisteredModelRepository,
)
from mlflow_mongodb.settings import MongoDBSettings

logger = logging.getLogger(__name__)

try:
    from mlflow.utils.validation import (
        _REGISTERED_MODEL_ALIAS_LATEST,
        _validate_model_alias_name_reserved,
    )
except ImportError:
    # MLflow versions before 3.3 perform this check inside _validate_model_alias_name.
    _REGISTERED_MODEL_ALIAS_LATEST = "latest"

    def _validate_model_alias_name_reserved(_alias):
        return None


class MongoDBModelRegistryStore(AbstractStore):
    """MongoDB store for registry URIs using the ``mongodb`` scheme.

    The persistence operations are intentionally left as stubs. They define the
    implementation surface that will be filled in as the backend is developed.
    """

    _REGISTERED_MODEL_ORDER_KEYS = frozenset({"name", "last_updated_timestamp"})

    def __init__(self, store_uri=None, tracking_uri=None):
        super().__init__(store_uri=store_uri, tracking_uri=tracking_uri)
        self.store_uri = store_uri
        self.tracking_uri = tracking_uri
        self._settings = MongoDBSettings.from_environment()

    @cached_property
    def _mongo_client(self):
        if not self.store_uri:
            raise MlflowException(
                "A MongoDB registry URI is required.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        try:
            return MongoClient(self.store_uri)
        except ConfigurationError as error:
            logger.error("Unable to create MongoDB client: %s", error)
            raise MlflowException(
                "Invalid MongoDB registry URI.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None

    @cached_property
    def _database(self):
        try:
            return self._mongo_client.get_default_database()
        except ConfigurationError as error:
            logger.error("Unable to select the MongoDB registry database: %s", error)
            raise MlflowException(
                "The MongoDB registry URI must include a database name.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None

    @cached_property
    def _registered_model_repository(self):
        return RegisteredModelRepository(self._database, settings=self._settings)

    @cached_property
    def _model_version_repository(self):
        return ModelVersionRepository(self._database, settings=self._settings)

    @cached_property
    def _tracking_client(self):
        return MlflowClient(tracking_uri=self.tracking_uri)

    def _to_mlflow_registered_model_details(
        self,
        details: RegisteredModelDetails,
    ) -> RegisteredModel:
        """Convert repository details into an MLflow registered model."""
        return self._to_mlflow_registered_model(
            details.registered_model,
            details.latest_versions,
        )

    def _to_mlflow_registered_model(
        self,
        record: RegisteredModelRecord,
        latest_version_records: tuple[ModelVersionRecord, ...] = (),
    ) -> RegisteredModel:
        """Convert a registered-model record and versions into an MLflow model."""
        return RegisteredModel(
            name=record.name,
            creation_timestamp=record.creation_timestamp,
            last_updated_timestamp=record.last_updated_timestamp,
            description=record.description,
            latest_versions=[
                self._to_mlflow_model_version(version_record, record)
                for version_record in latest_version_records
            ],
            tags=[RegisteredModelTag(tag.key, tag.value) for tag in record.tags],
            aliases=[RegisteredModelAlias(alias.alias, alias.version) for alias in record.aliases],
            deployment_job_id=record.deployment_job_id,
        )

    @staticmethod
    def _to_mlflow_model_version(
        record: ModelVersionRecord,
        registered_model: RegisteredModelRecord,
    ) -> ModelVersion:
        """Convert a model-version record into an MLflow model version."""
        aliases = [
            alias.alias
            for alias in registered_model.aliases
            if str(alias.version) == str(record.version)
        ]
        return ModelVersion(
            name=registered_model.name,
            version=record.version,
            creation_timestamp=record.creation_timestamp,
            last_updated_timestamp=record.last_updated_timestamp,
            description=record.description,
            user_id=record.user_id,
            current_stage=record.current_stage,
            source=record.source,
            run_id=record.run_id,
            status=record.status,
            status_message=record.status_message,
            tags=[ModelVersionTag(tag.key, tag.value) for tag in record.tags],
            run_link=record.run_link,
            aliases=aliases,
            model_id=record.model_id,
        )

    def _resolve_models_uri(self, parsed_model_uri, run_id):
        """Resolve an internal ``models:/`` URI to an artifact location.

        A ``models:/`` URI is a logical MLflow registry reference, not the
        physical location of model files. It can refer either to a logged
        model by ID or to a registered model version by name and version.

        Returns the resolved artifact location and the supplied or inferred
        source run ID.
        """
        if parsed_model_uri.model_id is not None:
            model = self._tracking_client.get_logged_model(parsed_model_uri.model_id)
            return model.artifact_location, run_id or model.source_run_id

        return (
            self.get_model_version_download_uri(
                parsed_model_uri.name,
                parsed_model_uri.version,
            ),
            run_id,
        )

    def _resolve_model_version_source(self, source, run_id, model_id):
        """Resolve a model-version source and infer its source run when possible.

        ``source`` is preserved as the caller-provided value, while the
        returned storage location is the physical artifact location used by
        the registry. Direct artifact URIs are already resolved; internal
        ``models:/`` URIs are looked up through the registry or tracking
        store. A logged model ID can also provide a source run ID when one was
        not explicitly supplied.
        """
        storage_location = source
        if urllib.parse.urlparse(source).scheme == "models":
            parsed_model_uri = _parse_model_uri(source)
            try:
                storage_location, run_id = self._resolve_models_uri(parsed_model_uri, run_id)
            except Exception as error:
                logger.error("Unable to resolve model source: %s", error)
                raise MlflowException(
                    "Unable to resolve the model source.",
                ) from None

        if not run_id and model_id:
            model = self._tracking_client.get_logged_model(model_id)
            run_id = model.source_run_id

        return storage_location, run_id

    @staticmethod
    def _parse_registered_model_filters(filter_string):
        """Parse an MLflow filter string into normalized repository filters.

        For example, ``name = 'fraud-model'`` or ``tag.is_prompt = 'false'``.
        """
        filter_string = add_prompt_filter_string(filter_string, is_prompt=False)
        parsed_filters = SearchModelUtils.parse_search_filter(filter_string)

        filters = []
        for parsed_filter in parsed_filters:
            field_type = parsed_filter["type"]
            key = parsed_filter["key"]
            comparator = parsed_filter["comparator"].upper()
            value = parsed_filter["value"]
            # These (is_string_attribute and is_tag) helpers validate the comparator by raising an
            # MlflowException.
            if field_type == "attribute":
                SearchModelUtils.is_string_attribute(field_type, key, comparator)
            elif field_type == "tag":
                SearchModelUtils.is_tag(field_type, comparator)
            else:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid search expression type: {field_type}"
                )
            filters.append(
                RegisteredModelFilter(
                    field_type=field_type,
                    key=key,
                    comparator=comparator,
                    value=value,
                )
            )
        return tuple(filters)

    @classmethod
    def _parse_registered_model_order(cls, order_by):
        """Parse registered-model order clauses and add a deterministic name tie-breaker."""
        parsed_order = []
        observed_fields = set()
        for order_by_clause in order_by or []:
            if order_by_clause == "timestamp" or order_by_clause.startswith("timestamp "):
                clause = order_by_clause.replace("timestamp", "last_updated_timestamp", 1)
            else:
                clause = order_by_clause
            field_type, key, ascending = (
                SearchModelUtils.parse_order_by_for_search_registered_models(clause)
            )
            if field_type != "attribute":
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order_by entity: {field_type}"
                )
            if key not in cls._REGISTERED_MODEL_ORDER_KEYS:
                raise MlflowException(
                    f"Invalid order by key '{key}' specified. Valid keys are "
                    f"{cls._REGISTERED_MODEL_ORDER_KEYS}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if key in observed_fields:
                raise MlflowException.invalid_parameter_value(
                    f"`order_by` contains duplicate fields: {order_by}"
                )
            observed_fields.add(key)
            parsed_order.append(RegisteredModelOrder(key=key, ascending=ascending))

        # Use the model name as a deterministic tie-breaker and default ordering.
        if "name" not in observed_fields:
            parsed_order.append(RegisteredModelOrder(key="name", ascending=True))
        return tuple(parsed_order)

    @staticmethod
    def _parse_model_version_filters(filter_string):
        """Parse model-version filter expressions into normalized repository filters."""
        parsed_filters = SearchModelVersionUtils.parse_search_filter(filter_string)

        filters = []
        querying_prompts = None
        for parsed_filter in parsed_filters:
            field_type = parsed_filter["type"]
            key = parsed_filter["key"]
            comparator = parsed_filter["comparator"].upper()
            value = parsed_filter["value"]

            if field_type == "attribute":
                if key not in SearchModelVersionUtils.VALID_SEARCH_ATTRIBUTE_KEYS:
                    raise MlflowException(
                        f"Invalid attribute name: {key}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if key in SearchModelVersionUtils.NUMERIC_ATTRIBUTES:
                    if (
                        comparator
                        not in SearchModelVersionUtils.VALID_NUMERIC_ATTRIBUTE_COMPARATORS
                    ):
                        raise MlflowException(
                            f"Invalid comparator for attribute {key}: {comparator}",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                    value = int(value)
                elif (
                    comparator not in SearchModelVersionUtils.VALID_STRING_ATTRIBUTE_COMPARATORS
                    or (comparator == "IN" and key != "run_id")
                ):
                    raise MlflowException(
                        f"Invalid comparator for attribute: {comparator}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
            elif field_type == "tag":
                if comparator not in SearchModelVersionUtils.VALID_TAG_COMPARATORS:
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid comparator for tag: {comparator}"
                    )
                if key == IS_PROMPT_TAG_KEY and querying_prompts is None:
                    querying_prompts = (comparator == "=" and value.lower() == "true") or (
                        comparator == "!=" and value.lower() == "false"
                    )
            else:
                raise MlflowException(
                    f"Invalid token type: {field_type}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            if comparator == "IN":
                value = tuple(value)
            filters.append(
                ModelVersionFilter(
                    field_type=field_type,
                    key=key,
                    comparator=comparator,
                    value=value,
                )
            )

        return tuple(filters), not bool(querying_prompts)

    @staticmethod
    def _parse_model_version_order(order_by):
        order_by = order_by or [
            "last_updated_timestamp DESC",
            "name ASC",
            "version_number DESC",
        ]
        parsed_order = []
        observed_fields = set()
        for order_by_clause in order_by:
            field_type, key, ascending = (
                SearchModelVersionUtils.parse_order_by_for_search_model_versions(order_by_clause)
            )
            if field_type != "attribute":
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order_by entity: {field_type}"
                )
            if key not in SearchModelVersionUtils.VALID_ORDER_BY_ATTRIBUTE_KEYS:
                raise MlflowException(
                    f"Invalid order by key '{key}' specified. Valid keys are "
                    f"{SearchModelVersionUtils.VALID_ORDER_BY_ATTRIBUTE_KEYS}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if key in observed_fields:
                raise MlflowException.invalid_parameter_value(
                    f"`order_by` contains duplicate fields: {order_by}"
                )
            observed_fields.add(key)
            parsed_order.append(ModelVersionOrder(key=key, ascending=ascending))

        if "name" not in observed_fields:
            parsed_order.append(ModelVersionOrder(key="name", ascending=True))
        if "version_number" not in observed_fields:
            parsed_order.append(ModelVersionOrder(key="version_number", ascending=False))
        return tuple(parsed_order)

    def create_registered_model(self, name, tags=None, description=None, deployment_job_id=None):
        """
        Create a registered model.

        Args:
            name: Registered model name.
            tags: Optional initial registered-model tags.
            description: Optional model description.
            deployment_job_id: Optional deployment job identifier.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        _validate_model_name(name)
        tags_by_key = {}
        for tag in tags or []:
            _validate_registered_model_tag(tag.key, tag.value)
            tags_by_key[tag.key] = tag.value

        creation_timestamp = get_current_time_millis()
        deployment_job_id = str(deployment_job_id) if deployment_job_id is not None else None
        try:
            record = self._registered_model_repository.create(
                name=name,
                creation_timestamp=creation_timestamp,
                description=description,
                tags=tags_by_key,
                deployment_job_id=deployment_job_id,
            )
        except RegisteredModelAlreadyExistsError:
            existing_record = self._registered_model_repository.find_by_name(name)
            existing_tags = (
                {tag.key: tag.value for tag in existing_record.tags} if existing_record else {}
            )
            handle_resource_already_exist_error(
                name,
                has_prompt_tag(existing_tags),
                has_prompt_tag(tags_by_key),
            )

        return self._to_mlflow_registered_model(record)

    def update_registered_model(self, name, description, deployment_job_id=None):
        """
        Update a registered model's description and deployment job.

        Args:
            name: Registered model name.
            description: New model description.
            deployment_job_id: Optional deployment job identifier.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        _validate_model_name(name)
        deployment_job_id = str(deployment_job_id) if deployment_job_id is not None else None

        try:
            record = self._registered_model_repository.update(
                name=name,
                last_updated_timestamp=get_current_time_millis(),
                description=description,
                deployment_job_id=deployment_job_id,
            )
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to update registered model: %s", exc)
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

        latest_version_records = self._model_version_repository.find_latest_by_stages(
            registered_model_id=record.model_id,
            stages=ALL_STAGES,
        )
        return self._to_mlflow_registered_model(record, latest_version_records)

    def rename_registered_model(self, name, new_name):
        """
        Rename a registered model.

        Args:
            name: Current registered model name.
            new_name: New registered model name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        _validate_model_name(name)
        _validate_model_renaming(new_name)

        last_updated_timestamp = get_current_time_millis()
        try:
            record = self._registered_model_repository.rename(
                name=name,
                new_name=new_name,
                last_updated_timestamp=last_updated_timestamp,
            )
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to rename registered model: %s", exc)
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None
        except RegisteredModelAlreadyExistsError as exc:
            logger.error("Unable to rename registered model: %s", exc)
            raise MlflowException(
                f"Registered Model (name={new_name}) already exists.",
                error_code=RESOURCE_ALREADY_EXISTS,
            ) from None

        self._model_version_repository.touch_all_for_registered_model(
            registered_model_id=record.model_id,
            last_updated_timestamp=last_updated_timestamp,
        )
        latest_version_records = self._model_version_repository.find_latest_by_stages(
            registered_model_id=record.model_id,
            stages=ALL_STAGES,
        )
        return self._to_mlflow_registered_model(record, latest_version_records)

    def delete_registered_model(self, name):
        """
        Delete a registered model and its model versions.

        Args:
            name: Registered model name.
        """
        _validate_model_name(name)

        try:
            registered_model = self._registered_model_repository.delete(name)
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to delete registered model: %s", exc)
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

        # Remove every version so a registered model or a prompt
        # cannot leave orphaned documents.
        self._model_version_repository.delete_all_for_registered_model(
            registered_model_id=registered_model.model_id,
        )

    def search_registered_models(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search registered models using MLflow filter and ordering expressions.

        Args:
            filter_string: Optional MLflow registered-model filter expression.
            max_results: Maximum number of models to return.
            order_by: Optional list of MLflow ordering expressions.
            page_token: Optional token for retrieving the next page.

        Returns:
            A paginated list of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects.
        """
        if max_results is None:
            max_results = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT
        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        if offset < 0:
            raise MlflowException(
                "Invalid page token, offset must be non-negative",
                error_code=INVALID_PARAMETER_VALUE,
            )

        page = self._registered_model_repository.search(
            filters=self._parse_registered_model_filters(filter_string),
            order_by=self._parse_registered_model_order(order_by),
            offset=offset,
            max_results=max_results,
        )
        next_page_token = (
            SearchUtils.create_page_token(offset + max_results) if page.has_more else None
        )
        return PagedList(
            [self._to_mlflow_registered_model_details(details) for details in page.records],
            next_page_token,
        )

    def get_registered_model(self, name):
        """
        Retrieve a registered model by name.

        Args:
            name: Registered model name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        _validate_model_name(name)

        details = self._registered_model_repository.find_by_name_with_latest_versions(name)
        if details is None:
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return self._to_mlflow_registered_model_details(details)

    def get_latest_versions(self, name, stages=None):
        """
        Retrieve the latest model versions for the requested stages.

        Args:
            name: Registered model name.
            stages: Optional list of stages to filter by.

        Returns:
            A list of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        _validate_model_name(name)

        requested_stages = ALL_STAGES if stages is None or len(stages) == 0 else stages
        canonical_stages = tuple(
            dict.fromkeys(get_canonical_stage(stage) for stage in requested_stages)
        )
        details = self._registered_model_repository.find_by_name_with_latest_versions(
            name,
            stages=canonical_stages,
        )
        if details is None:
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return [
            self._to_mlflow_model_version(record, details.registered_model)
            for record in details.latest_versions
        ]

    def set_registered_model_tag(self, name, tag):
        """
        Set a tag on a registered model.

        Args:
            name: Registered model name.
            tag: Registered-model tag to set.
        """
        _validate_model_name(name)
        _validate_registered_model_tag(tag.key, tag.value)

        try:
            self._registered_model_repository.set_tag(
                name=name,
                key=tag.key,
                value=tag.value,
            )
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to set registered-model tag: %s", exc)
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

    def delete_registered_model_tag(self, name, key):
        """
        Delete a tag from a registered model.

        Args:
            name: Registered model name.
            key: Tag key to delete.
        """
        _validate_model_name(name)
        _validate_tag_name(key)

        try:
            self._registered_model_repository.delete_tag(name=name, key=key)
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to delete registered-model tag: %s", exc)
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

    def set_registered_model_alias(self, name, alias, version):
        """
        Assign an alias to a registered model version.

        Args:
            name: Registered model name.
            alias: Alias name to assign.
            version: Model-version number targeted by the alias.
        """
        _validate_model_name(name)
        _validate_model_alias_name(alias)
        _validate_model_alias_name_reserved(alias)
        _validate_model_version(version)
        version = int(version)

        if not self._model_version_repository.exists_for_registered_model(
            registered_model_name=name,
            version=version,
        ):
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # The version existence check and alias update affect separate documents and are not
        # transactional. A concurrent version deletion can leave a dangling alias, while a
        # concurrent registered-model rename or replacement can make the name lookup stale.
        try:
            self._registered_model_repository.set_alias_by_name(
                name=name,
                alias=alias,
                version=version,
            )
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to set registered-model alias: %s", exc)
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

    def delete_registered_model_alias(self, name, alias):
        """
        Delete an alias from a registered model.

        Args:
            name: Registered model name.
            alias: Alias name to delete.
        """
        _validate_model_name(name)
        _validate_model_alias_name(alias)

        try:
            self._registered_model_repository.delete_alias_by_name(
                name=name,
                alias=alias,
            )
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to delete registered-model alias: %s", exc)
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        local_model_path=None,  # ruff: ignore[unused-method-argument]
        model_id=None,
    ):
        """
        Create a new model version from given source and run ID.

        Args:
            name: Registered model name.
            source: URI indicating the location of the model artifacts.
            run_id: Run ID from MLflow tracking server that generated the model.
            tags: A list of :py:class:`mlflow.entities.model_registry.ModelVersionTag`
                instances associated with this model version.
            run_link: Link to the run from an MLflow tracking server that generated this model.
            description: Description of the version.
            local_model_path: Unused.
            model_id: The ID of the model (from an Experiment) that is being promoted to a
                registered model version, if applicable.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
            created in the backend.
        """
        _validate_model_name(name)
        tags_by_key = {}
        for tag in tags or []:
            _validate_model_version_tag(tag.key, tag.value)
            tags_by_key[tag.key] = tag.value

        storage_location, run_id = self._resolve_model_version_source(
            source,
            run_id,
            model_id,
        )
        registered_model = self._registered_model_repository.find_by_name(name)
        if registered_model is None:
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Initialize the repository, including its unique version index, before allocating a
        # version number so an index-creation failure cannot consume a number.
        creation_timestamp = get_current_time_millis()
        try:
            version = self._registered_model_repository.allocate_next_version(
                model_id=registered_model.model_id,
                last_updated_timestamp=creation_timestamp,
            )
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to allocate model-version number: %s", exc)
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

        try:
            record = self._model_version_repository.create(
                registered_model_id=registered_model.model_id,
                version=version,
                creation_timestamp=creation_timestamp,
                description=description,
                current_stage=STAGE_NONE,
                source=source,
                storage_location=storage_location,
                run_id=run_id,
                run_link=run_link,
                status=ModelVersionStatus.to_string(ModelVersionStatus.READY),
                tags=tags_by_key,
                model_id=model_id,
            )
        except ModelVersionAlreadyExistsError as exc:
            logger.error("Unable to create model version: %s", exc)
            raise MlflowException(
                f"Model Version creation error (name={name}, version={version}): "
                "the allocated version already exists."
            ) from None

        return self._to_mlflow_model_version(record, registered_model)

    def update_model_version(self, name, version, description):
        """
        Update a model version's description.

        Args:
            name: Registered model name.
            version: Registered model version.
            description: New model-version description.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        _validate_model_name(name)
        _validate_model_version(version)
        version = int(version)

        registered_model = self._registered_model_repository.find_by_name(name)
        if registered_model is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        try:
            record = self._model_version_repository.update_description(
                registered_model_id=registered_model.model_id,
                version=version,
                description=description,
                last_updated_timestamp=get_current_time_millis(),
            )
        except ModelVersionNotFoundError as exc:
            logger.error("Unable to update model version: %s", exc)
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

        return self._to_mlflow_model_version(record, registered_model)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        """
        Update a model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If ``True``, archive all existing model versions in the
                target stage. This is valid only for active stages; otherwise, an error is raised.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        canonical_stage = get_canonical_stage(stage)
        if (
            archive_existing_versions
            and canonical_stage not in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS
        ):
            raise MlflowException(
                "Model version transition cannot archive existing model versions "
                f"because '{stage}' is not an Active stage. Valid stages are "
                f"{DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS}"
            )

        _validate_model_name(name)
        _validate_model_version(version)
        version = int(version)

        registered_model = self._registered_model_repository.find_by_name(name)
        if registered_model is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        last_updated_timestamp = get_current_time_millis()
        try:
            record = self._model_version_repository.transition_stage(
                registered_model_id=registered_model.model_id,
                version=version,
                stage=canonical_stage,
                last_updated_timestamp=last_updated_timestamp,
            )
        except ModelVersionNotFoundError as exc:
            logger.error("Unable to transition model-version stage: %s", exc)
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

        if archive_existing_versions:
            self._model_version_repository.archive_other_versions_in_stage(
                registered_model_id=registered_model.model_id,
                version=version,
                stage=canonical_stage,
                last_updated_timestamp=last_updated_timestamp,
            )

        try:
            registered_model = self._registered_model_repository.touch(
                model_id=registered_model.model_id,
                last_updated_timestamp=last_updated_timestamp,
            )
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to update registered model after stage transition: %s", exc)
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

        return self._to_mlflow_model_version(record, registered_model)

    def delete_model_version(self, name, version):
        """
        Soft delete a model version.

        Args:
            name: Registered model name.
            version: Registered model version.
        """
        _validate_model_name(name)
        _validate_model_version(version)
        version = int(version)

        registered_model = self._registered_model_repository.find_by_name(name)
        if registered_model is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        last_updated_timestamp = get_current_time_millis()
        try:
            self._model_version_repository.soft_delete(
                registered_model_id=registered_model.model_id,
                version=version,
                last_updated_timestamp=last_updated_timestamp,
            )
        except ModelVersionNotFoundError as exc:
            logger.error("Unable to delete model version: %s", exc)
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

        try:
            self._registered_model_repository.delete_aliases_for_version_and_touch(
                model_id=registered_model.model_id,
                version=version,
                last_updated_timestamp=last_updated_timestamp,
            )
        except RegisteredModelNotFoundError as exc:
            logger.error("Unable to update registered model after model-version deletion: %s", exc)
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

    def get_model_version(self, name, version):
        """
        Retrieve a model version by registered-model name and version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        _validate_model_name(name)
        _validate_model_version(version)
        version = int(version)

        registered_model = self._registered_model_repository.find_by_name(name)
        model_version = (
            self._model_version_repository.find_by_version(
                registered_model_id=registered_model.model_id,
                version=version,
            )
            if registered_model is not None
            else None
        )
        if model_version is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return self._to_mlflow_model_version(model_version, registered_model)

    def get_model_version_download_uri(self, name, version):
        """
        Retrieve the artifact URI for a model version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            The resolved model artifact URI.
        """
        _validate_model_name(name)
        _validate_model_version(version)
        version = int(version)

        registered_model = self._registered_model_repository.find_by_name(name)
        model_version = (
            self._model_version_repository.find_by_version(
                registered_model_id=registered_model.model_id,
                version=version,
            )
            if registered_model is not None
            else None
        )
        if model_version is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return model_version.storage_location or model_version.source

    def search_model_versions(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search model versions using MLflow filter and ordering expressions.

        Args:
            filter_string: Optional MLflow model-version filter expression.
            max_results: Maximum number of model versions to return.
            order_by: Optional list of MLflow ordering expressions.
            page_token: Optional token for retrieving the next page.

        Returns:
            A paginated list of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        if max_results is None:
            max_results = SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT
        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        if offset < 0:
            raise MlflowException(
                "Invalid page token, offset must be non-negative",
                error_code=INVALID_PARAMETER_VALUE,
            )

        filters, exclude_prompts = self._parse_model_version_filters(filter_string)
        page = self._model_version_repository.search(
            filters=filters,
            order_by=self._parse_model_version_order(order_by),
            exclude_prompts=exclude_prompts,
            offset=offset,
            max_results=max_results,
        )
        next_page_token = (
            SearchUtils.create_page_token(offset + max_results) if page.has_more else None
        )
        return PagedList(
            [
                self._to_mlflow_model_version(
                    result.model_version,
                    result.registered_model,
                )
                for result in page.records
            ],
            next_page_token,
        )

    def set_model_version_tag(self, name, version, tag):
        """
        Set a tag on a model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            tag: Model-version tag to set.
        """
        _validate_model_name(name)
        _validate_model_version(version)
        version = int(version)
        _validate_model_version_tag(tag.key, tag.value)

        registered_model = self._registered_model_repository.find_by_name(name)
        if registered_model is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        try:
            self._model_version_repository.set_tag(
                registered_model_id=registered_model.model_id,
                version=version,
                key=tag.key,
                value=tag.value,
            )
        except ModelVersionNotFoundError as exc:
            logger.error("Unable to set model-version tag: %s", exc)
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

    def delete_model_version_tag(self, name, version, key):
        """
        Delete a tag from a model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key to delete.
        """
        _validate_model_name(name)
        _validate_model_version(version)
        version = int(version)
        _validate_tag_name(key)

        registered_model = self._registered_model_repository.find_by_name(name)
        if registered_model is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        try:
            self._model_version_repository.delete_tag(
                registered_model_id=registered_model.model_id,
                version=version,
                key=key,
            )
        except ModelVersionNotFoundError as exc:
            logger.error("Unable to delete model-version tag: %s", exc)
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from None

    def get_model_version_by_alias(self, name, alias):
        """
        Retrieve the model version targeted by an alias.

        Args:
            name: Registered model name.
            alias: Alias name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        _validate_model_name(name)
        _validate_model_alias_name(alias)

        if alias.lower() == _REGISTERED_MODEL_ALIAS_LATEST:
            details = self._registered_model_repository.find_latest_version_by_name(name)
            if details is None:
                raise MlflowException(
                    f"Registered Model with name={name} not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            if not details.latest_versions:
                raise MlflowException(
                    f"Latest version not found for model {name}.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return self._to_mlflow_model_version(
                details.latest_versions[0],
                details.registered_model,
            )

        registered_model = self._registered_model_repository.find_by_name(name)
        if registered_model is None:
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        stored_alias = next(
            (
                stored_alias
                for stored_alias in registered_model.aliases
                if stored_alias.alias == alias
            ),
            None,
        )
        if stored_alias is None:
            raise MlflowException(
                f"Registered model alias {alias} not found.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        model_version = self._model_version_repository.find_by_version(
            registered_model_id=registered_model.model_id,
            version=stored_alias.version,
        )
        if model_version is None:
            raise MlflowException(
                f"Model Version (name={name}, version={stored_alias.version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return self._to_mlflow_model_version(model_version, registered_model)
