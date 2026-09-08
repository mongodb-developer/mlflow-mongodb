"""Functional search and pagination contracts backed by MongoDB."""

from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY, PROMPT_TEXT_TAG_KEY

from mlflow_mongodb import MongoDBModelRegistryStore


def _create_search_data(store: MongoDBModelRegistryStore):
    store.create_registered_model(
        "alpha-model",
        tags=[RegisteredModelTag("team", "Platform")],
    )
    store.create_registered_model(
        "beta-model",
        tags=[RegisteredModelTag("team", "platform-tools")],
    )
    store.create_registered_model(
        "gamma-model",
        tags=[RegisteredModelTag("team", "analytics")],
    )
    store.create_prompt(
        "prompt-model",
        tags={"team": "Platform"},
    )

    alpha_one = store.create_model_version(
        "alpha-model",
        source="s3://models/alpha/1",
        run_id="run-a",
        tags=[ModelVersionTag("team", "Risk")],
    )
    alpha_two = store.create_model_version(
        "alpha-model",
        source="s3://models/alpha/2",
        run_id="run-b",
        tags=[ModelVersionTag("team", "Platform")],
    )
    beta_one = store.create_model_version(
        "beta-model",
        source="s3://models/beta/1",
        run_id="run-c",
        tags=[ModelVersionTag("team", "risk")],
    )
    store.create_prompt_version("prompt-model", "Summarize {{text}}")

    store.transition_model_version_stage(
        "alpha-model",
        alpha_one.version,
        "Production",
        archive_existing_versions=False,
    )
    store.transition_model_version_stage(
        "alpha-model",
        alpha_two.version,
        "Staging",
        archive_existing_versions=False,
    )
    return alpha_one, alpha_two, beta_one


def test_search_registered_models_filters_orders_joins_and_paginates(
    store: MongoDBModelRegistryStore,
):
    _create_search_data(store)
    filter_string = "name LIKE '%-model' AND tags.team ILIKE 'plat%'"

    first_page = store.search_registered_models(
        filter_string,
        max_results=1,
        order_by=["name DESC"],
    )
    second_page = store.search_registered_models(
        filter_string,
        max_results=1,
        order_by=["name DESC"],
        page_token=first_page.token,
    )

    assert [model.name for model in first_page] == ["beta-model"]
    assert first_page.token is not None
    assert [model.name for model in second_page] == ["alpha-model"]
    assert second_page.token is None
    assert {
        (version.version, version.current_stage) for version in second_page[0].latest_versions
    } == {(1, "Production"), (2, "Staging")}


def test_search_model_versions_filters_orders_excludes_prompts_and_paginates(
    store: MongoDBModelRegistryStore,
):
    _create_search_data(store)

    all_model_versions = store.search_model_versions(order_by=["name ASC", "version_number ASC"])
    assert [(version.name, version.version) for version in all_model_versions] == [
        ("alpha-model", 1),
        ("alpha-model", 2),
        ("beta-model", 1),
    ]

    source_matches = store.search_model_versions(
        "source_path LIKE 's3://models/%'",
        order_by=["name ASC", "version_number ASC"],
    )
    assert [(version.name, version.version) for version in source_matches] == [
        ("alpha-model", 1),
        ("alpha-model", 2),
        ("beta-model", 1),
    ]

    filter_string = "run_id IN ('run-a', 'run-c') AND tags.team ILIKE 'RISK'"
    first_page = store.search_model_versions(
        filter_string,
        max_results=1,
        order_by=["name DESC"],
    )
    second_page = store.search_model_versions(
        filter_string,
        max_results=1,
        order_by=["name DESC"],
        page_token=first_page.token,
    )

    assert [(version.name, version.version) for version in first_page] == [("beta-model", 1)]
    assert first_page.token is not None
    assert [(version.name, version.version) for version in second_page] == [("alpha-model", 1)]
    assert second_page.token is None


def test_search_model_versions_normalizes_quoted_numeric_operand(
    store: MongoDBModelRegistryStore,
):
    _create_search_data(store)

    matches = store.search_model_versions("version_number = '2'")

    assert [(version.name, version.version) for version in matches] == [("alpha-model", 2)]


def test_wildcard_filters_match_multiline_tags_and_prompt_text(
    store: MongoDBModelRegistryStore,
):
    multiline_value = "first\nsecond"
    store.create_registered_model(
        "multiline-model",
        tags=[RegisteredModelTag("note", multiline_value)],
    )
    store.create_prompt("multiline-prompt")
    store.create_prompt_version("multiline-prompt", multiline_value)

    models = store.search_registered_models("tags.note LIKE 'first%second'")
    prompt_versions = store.search_model_versions(
        f"tags.`{PROMPT_TEXT_TAG_KEY}` ILIKE 'FIRST%SECOND' AND tags.`{IS_PROMPT_TAG_KEY}` = 'true'"
    )

    assert [model.name for model in models] == ["multiline-model"]
    assert [(version.name, version.version) for version in prompt_versions] == [
        ("multiline-prompt", 1)
    ]
