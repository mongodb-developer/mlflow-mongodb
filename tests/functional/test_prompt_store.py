"""Functional tests for the prompt API against a real MongoDB database."""

import pytest
from mlflow.exceptions import MlflowException

from mlflow_mongodb import MongoDBModelRegistryStore
from mlflow_mongodb.repositories import ModelVersionRepository

PROMPT_NAME = "mongodb-functional-prompt"
PROMPT_DESCRIPTION = "Prompt persisted by the MongoDB functional suite"
PROMPT_TEMPLATE_V1 = "Answer {{question}} using the supplied context."
PROMPT_TEMPLATE_V2 = "Answer {{question}} concisely using the supplied context."


def test_prompt_metadata_lifecycle(store: MongoDBModelRegistryStore):
    created = store.create_prompt(
        name=PROMPT_NAME,
        description=PROMPT_DESCRIPTION,
        tags={"team": "platform"},
    )

    assert created.name == PROMPT_NAME
    assert created.description == PROMPT_DESCRIPTION
    assert created.tags == {"team": "platform"}

    stored = store.get_prompt(PROMPT_NAME)
    assert stored is not None
    assert stored.name == PROMPT_NAME
    assert stored.description == PROMPT_DESCRIPTION
    assert stored.tags == {"team": "platform"}

    store.set_prompt_tag(PROMPT_NAME, "environment", "functional-test")
    assert store.get_prompt(PROMPT_NAME).tags == {
        "team": "platform",
        "environment": "functional-test",
    }

    store.delete_prompt_tag(PROMPT_NAME, "team")
    assert store.get_prompt(PROMPT_NAME).tags == {"environment": "functional-test"}

    store.delete_prompt(PROMPT_NAME)
    assert store.get_prompt(PROMPT_NAME) is None


def test_prompt_version_lifecycle(store: MongoDBModelRegistryStore):
    store.create_prompt(PROMPT_NAME, description=PROMPT_DESCRIPTION)

    version_one = store.create_prompt_version(
        name=PROMPT_NAME,
        template=PROMPT_TEMPLATE_V1,
        description="Initial prompt",
        tags={"author": "Ada"},
    )
    version_two = store.create_prompt_version(
        name=PROMPT_NAME,
        template=PROMPT_TEMPLATE_V2,
        description="Make the response concise",
        tags={"author": "Grace"},
    )

    assert version_one.version == 1
    assert version_one.template == PROMPT_TEMPLATE_V1
    assert version_one.commit_message == "Initial prompt"
    assert version_one.tags == {"author": "Ada"}
    assert version_two.version == 2
    assert version_two.template == PROMPT_TEMPLATE_V2
    assert version_two.commit_message == "Make the response concise"
    assert version_two.tags == {"author": "Grace"}

    stored_version = store.get_prompt_version(PROMPT_NAME, 1)
    assert stored_version is not None
    assert stored_version.template == PROMPT_TEMPLATE_V1
    assert stored_version.tags == {"author": "Ada"}

    store.delete_prompt_version(PROMPT_NAME, 1)
    with pytest.raises(MlflowException, match="not found"):
        store.get_prompt_version(PROMPT_NAME, 1)

    assert store.get_prompt(PROMPT_NAME) is not None
    assert store.get_prompt_version(PROMPT_NAME, 2).template == PROMPT_TEMPLATE_V2


def test_prompt_version_tags_and_aliases(store: MongoDBModelRegistryStore):
    store.create_prompt(PROMPT_NAME)
    version = store.create_prompt_version(PROMPT_NAME, PROMPT_TEMPLATE_V1)

    store.set_prompt_version_tag(PROMPT_NAME, version.version, "reviewed", "true")
    assert store.get_prompt_version(PROMPT_NAME, version.version).tags == {"reviewed": "true"}

    store.delete_prompt_version_tag(PROMPT_NAME, version.version, "reviewed")
    assert store.get_prompt_version(PROMPT_NAME, version.version).tags == {}

    store.set_prompt_alias(PROMPT_NAME, "production", version.version)
    aliased = store.get_prompt_version(PROMPT_NAME, "production")
    assert aliased is not None
    assert aliased.version == version.version
    assert aliased.aliases == ["production"]

    store.delete_prompt_alias(PROMPT_NAME, "production")
    with pytest.raises(MlflowException, match="alias production not found"):
        store.get_prompt_version(PROMPT_NAME, "production")


def test_deleting_prompt_removes_its_versions(store: MongoDBModelRegistryStore):
    store.create_prompt(PROMPT_NAME)
    store.create_prompt_version(PROMPT_NAME, PROMPT_TEMPLATE_V1)
    registered_model = store._registered_model_repository.find_by_name(PROMPT_NAME)
    assert registered_model is not None

    store.delete_prompt(PROMPT_NAME)

    assert store.get_prompt(PROMPT_NAME) is None
    assert (
        store._database[ModelVersionRepository.COLLECTION_NAME].count_documents({
            "registered_model_id": registered_model.model_id
        })
        == 0
    )
