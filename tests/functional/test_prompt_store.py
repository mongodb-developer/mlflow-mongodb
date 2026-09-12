"""Functional tests for the prompt API against a real MongoDB database."""

import mlflow
import pytest
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY

from mlflow_mongodb import MongoDBModelRegistryStore

PROMPT_NAME = "mongodb-functional-prompt"
PROMPT_DESCRIPTION = "Prompt persisted by the MongoDB functional suite"
PROMPT_TEMPLATE_V1 = "Answer {{question}} using the supplied context."
PROMPT_TEMPLATE_V2 = "Answer {{question}} concisely using the supplied context."
SUPPORTS_PROMPT_VERSION_SEARCH = not mlflow.__version__.startswith("3.1.")


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
        store._database[store._settings.model_versions_collection_name].count_documents(
            {"registered_model_id": registered_model.model_id}
        )
        == 0
    )


def test_model_searches_exclude_prompts_unless_prompt_filter_is_explicit(
    store: MongoDBModelRegistryStore,
):
    store.create_registered_model(
        "search-model",
        tags=[RegisteredModelTag("fruit", "apple")],
    )
    store.create_model_version(
        "search-model",
        "s3://models/search-model/1",
        tags=[ModelVersionTag("fruit", "apple")],
    )
    store.create_prompt("search-prompt-one")
    store.create_prompt("search-prompt-two", tags={"fruit": "apple"})
    store.create_prompt_version("search-prompt-one", "Prompt one")
    store.create_prompt_version(
        "search-prompt-two",
        "Prompt two, version one",
        tags={"fruit": "apple"},
    )
    store.create_prompt_version(
        "search-prompt-two",
        "Prompt two, version two",
        tags={"fruit": "orange"},
    )

    assert [model.name for model in store.search_registered_models()] == ["search-model"]
    assert [model.name for model in store.search_registered_models("tags.fruit = 'apple'")] == [
        "search-model"
    ]
    assert store.search_registered_models("name = 'search-prompt-one'") == []

    prompts = store.search_registered_models(
        f"tags.`{IS_PROMPT_TAG_KEY}` = 'true'",
    )
    assert {prompt.name for prompt in prompts} == {
        "search-prompt-one",
        "search-prompt-two",
    }
    matching_prompts = store.search_registered_models(
        f"tags.`{IS_PROMPT_TAG_KEY}` = 'true' AND tags.fruit = 'apple'",
    )
    assert [prompt.name for prompt in matching_prompts] == ["search-prompt-two"]

    model_versions = store.search_model_versions()
    assert [(version.name, version.version) for version in model_versions] == [("search-model", 1)]
    assert [version.name for version in store.search_model_versions("tags.fruit = 'apple'")] == [
        "search-model"
    ]

    prompt_versions = store.search_model_versions(
        f"tags.`{IS_PROMPT_TAG_KEY}` = 'true'",
    )
    assert {(version.name, version.version) for version in prompt_versions} == {
        ("search-prompt-one", 1),
        ("search-prompt-two", 1),
        ("search-prompt-two", 2),
    }
    matching_prompt_versions = store.search_model_versions(
        f"tags.`{IS_PROMPT_TAG_KEY}` = 'true' AND tags.fruit = 'apple'",
    )
    assert [(version.name, version.version) for version in matching_prompt_versions] == [
        ("search-prompt-two", 1)
    ]


@pytest.mark.skipif(
    not SUPPORTS_PROMPT_VERSION_SEARCH,
    reason="MLflow 3.1 does not implement search_prompt_versions for OSS registries",
)
def test_search_prompt_versions_orders_paginates_and_validates_prompt_type(
    store: MongoDBModelRegistryStore,
):
    store.create_prompt("search-prompt-versions")
    for version_number in range(1, 4):
        store.create_prompt_version(
            "search-prompt-versions",
            f"Hello {{{{name}}}} v{version_number}",
        )

    results = store.search_prompt_versions("search-prompt-versions")
    assert [version.version for version in results] == [3, 2, 1]
    assert all(version.name == "search-prompt-versions" for version in results)

    first_page = store.search_prompt_versions("search-prompt-versions", max_results=2)
    assert [version.version for version in first_page] == [3, 2]
    assert first_page.token is not None

    second_page = store.search_prompt_versions(
        "search-prompt-versions",
        max_results=2,
        page_token=first_page.token,
    )
    assert [version.version for version in second_page] == [1]
    assert second_page.token is None

    with pytest.raises(MlflowException, match="not found"):
        store.search_prompt_versions("missing-prompt")

    store.create_registered_model("registered-model-not-prompt")
    with pytest.raises(MlflowException, match="registered as a model, not a prompt"):
        store.search_prompt_versions("registered-model-not-prompt")


def test_models_and_prompts_cannot_reuse_each_others_names(
    store: MongoDBModelRegistryStore,
):
    store.create_registered_model("name-owned-by-model")
    store.create_prompt("name-owned-by-prompt")

    with pytest.raises(MlflowException, match="Registered Model.*already exists"):
        store.create_registered_model("name-owned-by-model")
    with pytest.raises(MlflowException, match="Prompt.*already exists"):
        store.create_prompt("name-owned-by-prompt")
    with pytest.raises(MlflowException, match="already taken by a registered model"):
        store.create_prompt("name-owned-by-model")
    with pytest.raises(MlflowException, match="already taken by a prompt"):
        store.create_registered_model("name-owned-by-prompt")
