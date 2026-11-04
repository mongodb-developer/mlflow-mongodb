"""Unit tests for the MLflow prompt API backed by the MongoDB registry store."""

from unittest.mock import Mock

import pytest
from mlflow.entities.model_registry import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY, PROMPT_TEXT_TAG_KEY
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException

from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryNotFoundError,
)

PROMPT_NAME = "support-assistant"
PROMPT_TEMPLATE = "Answer {{question}} using the supplied context."
PROMPT_DESCRIPTION = "Prompt used by the support assistant"
PROMPT_TAGS = {"team": "support", "environment": "test"}
VERSION_DESCRIPTION = "Initial prompt wording"
VERSION_TAGS = {"author": "Ada"}


@pytest.mark.parametrize(
    "message,error_code",
    [
        ("Invalid prompt reference.", INVALID_PARAMETER_VALUE),
        ("Unable to persist prompt associations.", INTERNAL_ERROR),
        ("Linking prompts to traces is not supported in FileStore.", INTERNAL_ERROR),
    ],
)
def test_prompt_trace_link_does_not_hide_validation_or_persistence_errors(
    store, monkeypatch, message, error_code
):
    prompt = PromptVersion(name=PROMPT_NAME, version=1, template=PROMPT_TEMPLATE)
    monkeypatch.setattr(store, "get_prompt_version", Mock(return_value=prompt))
    tracking = Mock(spec=["get_trace_info", "link_prompts_to_trace", "set_trace_tag"])
    tracking.get_trace_info.return_value.tags = {}
    failure = MlflowException(message, error_code)
    tracking.link_prompts_to_trace.side_effect = failure
    store.__dict__["_tracking_store"] = tracking

    with pytest.raises(MlflowException) as caught:
        store.link_prompts_to_trace([prompt], "tr-existing")

    assert caught.value is failure
    tracking.set_trace_tag.assert_not_called()


def test_prompt_link_tracking_resolution_does_not_expose_uri_credentials(
    store, monkeypatch, caplog
):
    store.tracking_uri = "unsupported-mongodb://fake-user:fake-password@localhost/test"
    prompt = PromptVersion(name=PROMPT_NAME, version=1, template=PROMPT_TEMPLATE)
    monkeypatch.setattr(store, "get_prompt_version", lambda _name, _version: prompt)

    with pytest.raises(UnsupportedModelRegistryStoreURIException) as caught:
        store.link_prompts_to_trace([prompt], "tr-existing")

    assert "unsupported-mongodb://***" in str(caught.value)
    assert "fake-user" not in str(caught.value)
    assert "fake-password" not in str(caught.value)
    assert "fake-user" not in caplog.text
    assert "fake-password" not in caplog.text
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_create_prompt(
    store,
    registered_model_repository,
    registered_model_record_factory,
):
    stored_record = registered_model_record_factory(
        name=PROMPT_NAME,
        description=PROMPT_DESCRIPTION,
        tags={IS_PROMPT_TAG_KEY: "true", **PROMPT_TAGS},
    )
    registered_model_repository.create.return_value = stored_record

    prompt = store.create_prompt(
        name=PROMPT_NAME,
        description=PROMPT_DESCRIPTION,
        tags=PROMPT_TAGS,
    )

    assert prompt.name == PROMPT_NAME
    assert prompt.description == PROMPT_DESCRIPTION
    assert prompt.creation_timestamp == stored_record.creation_timestamp
    assert prompt.tags == PROMPT_TAGS
    registered_model_repository.create.assert_called_once_with(
        name=PROMPT_NAME,
        creation_timestamp=stored_record.creation_timestamp,
        description=PROMPT_DESCRIPTION,
        tags={IS_PROMPT_TAG_KEY: "true", **PROMPT_TAGS},
        deployment_job_id=None,
    )


def test_get_prompt(
    store,
    registered_model_repository,
    registered_model_record_factory,
    registered_model_details_factory,
):
    stored_record = registered_model_record_factory(
        name=PROMPT_NAME,
        description=PROMPT_DESCRIPTION,
        tags={IS_PROMPT_TAG_KEY: "true", **PROMPT_TAGS},
    )
    registered_model_repository.find_by_name_with_latest_versions.return_value = (
        registered_model_details_factory(registered_model=stored_record)
    )

    prompt = store.get_prompt(PROMPT_NAME)

    assert prompt.name == PROMPT_NAME
    assert prompt.description == PROMPT_DESCRIPTION
    assert prompt.creation_timestamp == stored_record.creation_timestamp
    assert prompt.tags == PROMPT_TAGS
    registered_model_repository.find_by_name_with_latest_versions.assert_called_once_with(
        PROMPT_NAME
    )


def test_get_prompt_returns_none_for_registered_model(
    store,
    registered_model_repository,
    registered_model_record_factory,
    registered_model_details_factory,
):
    model_record = registered_model_record_factory(
        name=PROMPT_NAME,
        tags=PROMPT_TAGS,
    )
    registered_model_repository.find_by_name_with_latest_versions.return_value = (
        registered_model_details_factory(registered_model=model_record)
    )

    assert store.get_prompt(PROMPT_NAME) is None


def test_get_prompt_returns_none_when_name_does_not_exist(
    store,
    registered_model_repository,
):
    registered_model_repository.find_by_name_with_latest_versions.return_value = None

    assert store.get_prompt(PROMPT_NAME) is None


def test_delete_prompt(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
):
    prompt_record = registered_model_record_factory(
        name=PROMPT_NAME,
        tags={IS_PROMPT_TAG_KEY: "true"},
    )
    registered_model_repository.delete.return_value = prompt_record

    store.delete_prompt(PROMPT_NAME)

    registered_model_repository.delete.assert_called_once_with(PROMPT_NAME)
    model_version_repository.delete_all_for_registered_model.assert_called_once_with(
        registered_model_id=prompt_record.model_id,
    )


def test_delete_missing_prompt_raises_mlflow_exception(
    store,
    registered_model_repository,
    model_version_repository,
):
    registered_model_repository.delete.side_effect = RepositoryNotFoundError(PROMPT_NAME)

    with pytest.raises(MlflowException, match=f"name={PROMPT_NAME} not found"):
        store.delete_prompt(PROMPT_NAME)

    model_version_repository.delete_all_for_registered_model.assert_not_called()


def test_create_prompt_rejects_name_owned_by_model(
    store,
    registered_model_repository,
    registered_model_record_factory,
):
    registered_model_repository.create.side_effect = RepositoryAlreadyExistsError(PROMPT_NAME)
    registered_model_repository.find_by_name.return_value = registered_model_record_factory(
        name=PROMPT_NAME,
    )

    with pytest.raises(
        MlflowException,
        match="name is already taken by a registered model",
    ):
        store.create_prompt(PROMPT_NAME)


def test_create_prompt_version(
    store,
    registered_model_repository,
    model_version_repository,
    tracking_client,
    registered_model_record_factory,
    model_version_record_factory,
    registered_model_details_factory,
):
    prompt_record = registered_model_record_factory(
        name=PROMPT_NAME,
        tags={IS_PROMPT_TAG_KEY: "true", **PROMPT_TAGS},
    )
    version_record = model_version_record_factory(
        registered_model_id=prompt_record.model_id,
        description=VERSION_DESCRIPTION,
        tags={
            IS_PROMPT_TAG_KEY: "true",
            PROMPT_TEXT_TAG_KEY: PROMPT_TEMPLATE,
            **VERSION_TAGS,
        },
    )
    registered_model_repository.find_by_name.return_value = prompt_record
    registered_model_repository.allocate_next_version.return_value = 1
    registered_model_repository.find_by_name_with_latest_versions.return_value = (
        registered_model_details_factory(
            registered_model=prompt_record,
            latest_versions=(version_record,),
        )
    )
    model_version_repository.create.return_value = version_record

    prompt_version = store.create_prompt_version(
        name=PROMPT_NAME,
        template=PROMPT_TEMPLATE,
        description=VERSION_DESCRIPTION,
        tags=VERSION_TAGS,
    )

    assert prompt_version.name == PROMPT_NAME
    assert prompt_version.version == 1
    assert prompt_version.template == PROMPT_TEMPLATE
    assert prompt_version.commit_message == VERSION_DESCRIPTION
    assert prompt_version.tags == VERSION_TAGS
    registered_model_repository.allocate_next_version.assert_called_once_with(
        model_id=prompt_record.model_id,
        last_updated_timestamp=version_record.creation_timestamp,
    )

    create_arguments = model_version_repository.create.call_args.kwargs
    assert create_arguments["registered_model_id"] == prompt_record.model_id
    assert create_arguments["version"] == 1
    assert create_arguments["creation_timestamp"] == version_record.creation_timestamp
    assert create_arguments["description"] == VERSION_DESCRIPTION
    assert create_arguments["source"] == "prompt-template"
    assert create_arguments["storage_location"] == "prompt-template"
    assert create_arguments["run_id"] is None
    assert create_arguments["run_link"] is None
    assert create_arguments["model_id"] is None
    assert create_arguments["tags"][IS_PROMPT_TAG_KEY] == "true"
    assert create_arguments["tags"][PROMPT_TEXT_TAG_KEY] == PROMPT_TEMPLATE
    assert create_arguments["tags"]["author"] == "Ada"
    assert tracking_client.mock_calls == []


def test_get_prompt_version_by_number(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
    registered_model_details_factory,
):
    prompt_record = registered_model_record_factory(
        name=PROMPT_NAME,
        tags={IS_PROMPT_TAG_KEY: "true", **PROMPT_TAGS},
        aliases={"production": 2},
    )
    version_record = model_version_record_factory(
        registered_model_id=prompt_record.model_id,
        version=2,
        tags={
            IS_PROMPT_TAG_KEY: "true",
            PROMPT_TEXT_TAG_KEY: PROMPT_TEMPLATE,
            **VERSION_TAGS,
        },
    )
    registered_model_repository.find_by_name_with_latest_versions.return_value = (
        registered_model_details_factory(registered_model=prompt_record)
    )
    registered_model_repository.find_by_name.return_value = prompt_record
    model_version_repository.find_by_version.return_value = version_record

    prompt_version = store.get_prompt_version(PROMPT_NAME, 2)

    assert prompt_version.name == PROMPT_NAME
    assert prompt_version.version == 2
    assert prompt_version.template == PROMPT_TEMPLATE
    assert prompt_version.tags == VERSION_TAGS
    assert prompt_version.aliases == ["production"]
    model_version_repository.find_by_version.assert_called_once_with(
        registered_model_id=prompt_record.model_id,
        version=2,
    )


def test_get_prompt_version_by_alias(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
    registered_model_details_factory,
):
    prompt_record = registered_model_record_factory(
        name=PROMPT_NAME,
        tags={IS_PROMPT_TAG_KEY: "true"},
        aliases={"production": 3},
    )
    version_record = model_version_record_factory(
        registered_model_id=prompt_record.model_id,
        version=3,
        tags={
            IS_PROMPT_TAG_KEY: "true",
            PROMPT_TEXT_TAG_KEY: PROMPT_TEMPLATE,
        },
    )
    registered_model_repository.find_by_name_with_latest_versions.return_value = (
        registered_model_details_factory(registered_model=prompt_record)
    )
    registered_model_repository.find_by_name.return_value = prompt_record
    model_version_repository.find_by_version.return_value = version_record

    prompt_version = store.get_prompt_version(PROMPT_NAME, "production")

    assert prompt_version.version == 3
    assert prompt_version.aliases == ["production"]
    registered_model_repository.find_by_name.assert_called_once_with(PROMPT_NAME)
    model_version_repository.find_by_version.assert_called_once_with(
        registered_model_id=prompt_record.model_id,
        version=3,
    )


def test_get_non_prompt_version_returns_none(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
    registered_model_details_factory,
):
    prompt_record = registered_model_record_factory(
        name=PROMPT_NAME,
        tags={IS_PROMPT_TAG_KEY: "true"},
    )
    model_version = model_version_record_factory(
        registered_model_id=prompt_record.model_id,
        tags={"author": "Ada"},
    )
    registered_model_repository.find_by_name_with_latest_versions.return_value = (
        registered_model_details_factory(registered_model=prompt_record)
    )
    registered_model_repository.find_by_name.return_value = prompt_record
    model_version_repository.find_by_version.return_value = model_version

    assert store.get_prompt_version(PROMPT_NAME, 1) is None


def test_missing_prompt_version_tag_target_raises_mlflow_exception(
    store,
    registered_model_repository,
):
    registered_model_repository.find_by_name.return_value = None

    with pytest.raises(MlflowException, match="Model Version"):
        store.set_prompt_version_tag(PROMPT_NAME, 2, "reviewed", "true")


def test_set_prompt_alias_requires_existing_version(
    store,
    registered_model_repository,
    model_version_repository,
):
    model_version_repository.exists_for_registered_model.return_value = False

    with pytest.raises(MlflowException, match="Model Version"):
        store.set_prompt_alias(PROMPT_NAME, "production", 7)

    registered_model_repository.set_alias_by_name.assert_not_called()


def test_delete_prompt_alias_translates_missing_prompt(
    store,
    registered_model_repository,
):
    registered_model_repository.delete_alias_by_name.side_effect = RepositoryNotFoundError(
        PROMPT_NAME
    )

    with pytest.raises(MlflowException, match="Registered Model"):
        store.delete_prompt_alias(PROMPT_NAME, "production")
