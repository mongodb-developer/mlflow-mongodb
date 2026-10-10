"""Broader MLflow model-registry search contracts backed by MongoDB."""

import pytest
from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModelTag,
)
from mlflow.utils.search_utils import SearchModelVersionUtils

from mlflow_mongodb import MongoDBModelRegistryStore
from mlflow_mongodb.model_registry.repositories import (
    ModelVersionFilter,
    ModelVersionOrder,
    RegisteredModelFilter,
    RegisteredModelOrder,
)

SUPPORTS_MODEL_NAME_LIST_FILTERS = "name" in getattr(
    SearchModelVersionUtils, "LIST_SUPPORTED_KEYS", ()
)


def _search_model_version_numbers(store, filter_string):
    return {version.version for version in store.search_model_versions(filter_string)}


def test_model_version_repository_rejects_unsupported_comparator(
    store: MongoDBModelRegistryStore,
):
    name = "search-model-version-unsupported-comparator"
    store.create_registered_model(name)
    store.create_model_version(name, "models/unsupported-comparator", run_id="excluded-run")

    with pytest.raises(ValueError, match="^Unsupported model-version comparator: NOT IN$"):
        store._model_version_repository.search(
            filters=(ModelVersionFilter("attribute", "run_id", "NOT IN", ("excluded-run",)),),
            order_by=(ModelVersionOrder("version_number", True),),
            exclude_prompts=True,
            offset=0,
            max_results=100,
        )


def test_registered_model_repository_rejects_unsupported_comparator(
    store: MongoDBModelRegistryStore,
):
    name = "search-registered-model-unsupported-comparator"
    store.create_registered_model(name)

    with pytest.raises(ValueError, match="^Unsupported registered-model comparator: >$"):
        store._registered_model_repository.search(
            filters=(RegisteredModelFilter("attribute", "name", ">", name),),
            order_by=(RegisteredModelOrder("name", True),),
            offset=0,
            max_results=100,
        )


def test_search_model_versions_supports_portable_attribute_filters(
    store: MongoDBModelRegistryStore,
):
    name = "search-model-version-attributes"
    store.create_registered_model(name)
    run_id_one = "first-run-lowercase"
    run_id_two = "second-run-lowercase"
    run_id_three = "third-run-lowercase"
    store.create_model_version(name, "A/B", run_id=run_id_one)
    store.create_model_version(name, "A/C", run_id=run_id_two)
    store.create_model_version(name, "A/D", run_id=run_id_two)
    store.create_model_version(name, "A/D", run_id=run_id_three)

    assert _search_model_version_numbers(store, f"name = '{name}'") == {1, 2, 3, 4}
    assert _search_model_version_numbers(store, "version_number = 2") == {2}
    assert _search_model_version_numbers(store, "version_number <= 3") == {1, 2, 3}
    assert _search_model_version_numbers(store, f"run_id = '{run_id_one}'") == {1}
    assert _search_model_version_numbers(store, f"run_id = '{run_id_two}'") == {2, 3}
    assert _search_model_version_numbers(store, f"run_id IN ('{run_id_one}', '{run_id_two}')") == {
        1,
        2,
        3,
    }
    assert _search_model_version_numbers(
        store,
        f"run_id IN ('{run_id_one.upper()}', '{run_id_two}')",
    ) == {2, 3}
    assert _search_model_version_numbers(store, f"run_id LIKE '{run_id_two[:10]}%'") == {2, 3}
    assert _search_model_version_numbers(store, f"run_id ILIKE '{run_id_two[:10].upper()}%'") == {
        2,
        3,
    }
    assert _search_model_version_numbers(store, "source_path = 'A/D'") == {3, 4}
    assert _search_model_version_numbers(store, "source_path = 'A'") == set()
    assert _search_model_version_numbers(store, "source_path = ''") == set()


@pytest.mark.skipif(
    not SUPPORTS_MODEL_NAME_LIST_FILTERS,
    reason="The installed MLflow parser does not support model-name list filters",
)
def test_search_model_versions_supports_name_list_filters(store: MongoDBModelRegistryStore):
    first_name = "search-model-name-list-first"
    second_name = "search-model-name-list-second"
    excluded_name = "search-model-name-list-excluded"
    for name in (first_name, second_name, excluded_name):
        store.create_registered_model(name)

    first_version = store.create_model_version(first_name, "models/first/1")
    second_version = store.create_model_version(first_name, "models/first/2")
    other_model_version = store.create_model_version(second_name, "models/second/1")
    store.create_model_version(excluded_name, "models/excluded/1")

    results = store.search_model_versions(f"name IN ('{first_name}', '{second_name}')")

    assert {(version.name, version.version) for version in results} == {
        (first_name, first_version.version),
        (first_name, second_version.version),
        (second_name, other_model_version.version),
    }
    assert store.search_model_versions("name IN ('missing-model')") == []


def test_search_model_versions_attribute_inequality_excludes_null_and_missing_values(
    store: MongoDBModelRegistryStore,
):
    name = "search-model-version-attribute-inequality"
    store.create_registered_model(name)
    versions = {
        state: store.create_model_version(name, f"models/{state}", run_id=run_id)
        for state, run_id in (
            ("equal", "excluded-run"),
            ("different", "included-run"),
            ("null", None),
            ("missing", "removed-run"),
        )
    }

    # The store writes an explicit null for omitted run IDs; unset one to test a missing field.
    collection = store._database[store._settings.model_versions_collection_name]
    result = collection.update_one({"run_id": "removed-run"}, {"$unset": {"run_id": ""}})
    assert result.modified_count == 1

    assert _search_model_version_numbers(store, f"name = '{name}'") == {
        version.version for version in versions.values()
    }
    assert _search_model_version_numbers(
        store, f"name = '{name}' AND run_id != 'excluded-run'"
    ) == {versions["different"].version}


def test_search_model_versions_reflects_model_version_changes(
    store: MongoDBModelRegistryStore,
):
    name = "search-model-version-changes"
    store.create_registered_model(name)
    run_id_one = "search-model-version-changes-run-one"
    run_id_two = "search-model-version-changes-run-two"
    run_id_three = "search-model-version-changes-run-three"
    store.create_model_version(name, "A/B", run_id=run_id_one)
    store.create_model_version(name, "A/C", run_id=run_id_two)
    store.create_model_version(name, "A/D", run_id=run_id_two)
    version_four = store.create_model_version(name, "A/D", run_id=run_id_three)

    store.delete_model_version(name, version_four.version)
    assert _search_model_version_numbers(store, "") == {1, 2, 3}
    assert _search_model_version_numbers(store, None) == {1, 2, 3}
    assert _search_model_version_numbers(store, "source_path = 'A/D'") == {3}

    store.transition_model_version_stage(name, 1, "Production", False)
    store.update_model_version(name, 1, "Online prediction model")
    result = store.search_model_versions(f"run_id = '{run_id_one}'")
    assert len(result) == 1
    assert isinstance(result[0], ModelVersion)
    assert result[0].current_stage == "Production"
    assert result[0].source == "A/B"
    assert result[0].description == "Online prediction model"


def test_search_model_versions_by_tag_keeps_same_key_conditions_on_one_tag(
    store: MongoDBModelRegistryStore,
):
    name = "search-model-version-tags"
    store.create_registered_model(name)
    store.create_model_version(
        name,
        "A/B",
        tags=[ModelVersionTag("t1", "abc"), ModelVersionTag("t2", "xyz")],
    )
    store.create_model_version(
        name,
        "A/C",
        tags=[ModelVersionTag("t1", "abc"), ModelVersionTag("t2", "x123")],
    )

    assert _search_model_version_numbers(store, f"name = '{name}' AND tag.t2 = 'xyz'") == {1}
    assert _search_model_version_numbers(store, "name = 'wrong-name' AND tag.t2 = 'xyz'") == set()
    assert _search_model_version_numbers(store, "tag.`t2` = 'xyz'") == {1}
    assert _search_model_version_numbers(store, "tag.t3 = 'xyz'") == set()
    assert _search_model_version_numbers(store, "tag.t2 != 'xy'") == {1, 2}
    assert _search_model_version_numbers(store, "tag.t2 LIKE 'xy%'") == {1}
    assert _search_model_version_numbers(store, "tag.t2 LIKE 'xY%'") == set()
    assert _search_model_version_numbers(store, "tag.t2 ILIKE 'xY%'") == {1}
    assert _search_model_version_numbers(store, "tag.T2 = 'xyz'") == set()
    assert _search_model_version_numbers(store, "tag.t1 = 'abc' AND tag.t2 LIKE 'x%'") == {1, 2}
    assert _search_model_version_numbers(store, "tag.t2 LIKE 'x%' AND tag.t2 != 'xyz'") == {2}


def test_search_model_versions_orders_and_paginates_stably(
    store: MongoDBModelRegistryStore,
    monkeypatch,
):
    clock = {"now": 1_700_003_000_000}
    monkeypatch.setattr(
        "mlflow_mongodb.model_registry.store.get_current_time_millis",
        lambda: clock["now"],
    )
    names = [
        "OrderMV1",
        "OrderMV2",
        "OrderMV3",
        "OrderMV4",
        "OrderMV1",
        "OrderMV4",
    ]
    for name in set(names):
        store.create_registered_model(name)

    created = []
    for index, name in enumerate(names):
        clock["now"] += 1_000
        created.append(
            store.create_model_version(
                name,
                source="A" if index < 3 else "B",
                run_id=f"order-run-{index}",
            )
        )

    default_order = store.search_model_versions()
    assert [version.name for version in default_order] == names[::-1]
    assert [version.version for version in default_order] == [2, 2, 1, 1, 1, 1]

    by_name = store.search_model_versions(order_by=["name DESC"])
    assert [version.name for version in by_name] == sorted(names, reverse=True)
    assert [version.version for version in by_name] == [2, 1, 1, 1, 2, 1]

    by_version = store.search_model_versions(order_by=["version_number DESC"])
    assert [version.name for version in by_version] == [
        "OrderMV1",
        "OrderMV4",
        "OrderMV1",
        "OrderMV2",
        "OrderMV3",
        "OrderMV4",
    ]
    assert [version.version for version in by_version] == [2, 2, 1, 1, 1, 1]

    by_creation = store.search_model_versions(order_by=["creation_timestamp DESC"])
    assert [(version.name, version.version) for version in by_creation] == [
        (version.name, version.version) for version in created[::-1]
    ]

    clock["now"] += 1_000
    store.update_model_version("OrderMV1", 1, "latest update")
    by_last_update = store.search_model_versions(order_by=["last_updated_timestamp ASC"])
    assert (by_last_update[-1].name, by_last_update[-1].version) == ("OrderMV1", 1)

    page_name = "PageMV"
    store.create_registered_model(page_name)
    page_versions = []
    for index in range(12):
        clock["now"] += 1_000
        page_versions.append(store.create_model_version(page_name, f"page-source-{index}"))
    expected = page_versions[::-1]

    returned = []
    token = None
    for page_size in (3, 4, 10):
        page = store.search_model_versions(
            f"name = '{page_name}'",
            max_results=page_size,
            page_token=token,
        )
        returned.extend(page)
        token = page.token
    assert [(version.name, version.version) for version in returned] == [
        (version.name, version.version) for version in expected
    ]
    assert token is None


def test_search_registered_models_supports_portable_name_filters(
    store: MongoDBModelRegistryStore,
):
    prefix = "search-registered-model-"
    names = [f"{prefix}{suffix}" for suffix in ("RM1", "RM2", "RM3", "RM4A", "RM4ab")]
    for name in names:
        store.create_registered_model(name)

    assert {model.name for model in store.search_registered_models(None)} == set(names)
    assert {model.name for model in store.search_registered_models(f"name = '{names[0]}'")} == {
        names[0]
    }
    assert {
        model.name for model in store.search_registered_models(f"name = '{names[0]}-missing'")
    } == set()
    assert {
        model.name for model in store.search_registered_models(f"name LIKE '{prefix}%'")
    } == set(names)
    assert {model.name for model in store.search_registered_models("name LIKE '%RM%'")} == set(
        names
    )
    assert {model.name for model in store.search_registered_models("name LIKE '_earch%'")} == set(
        names
    )
    assert {
        model.name for model in store.search_registered_models(f"name LIKE '{prefix}RM4A%'")
    } == {names[3]}
    assert {
        model.name
        for model in store.search_registered_models(f"name ILIKE '{prefix.upper()}RM4A%'")
    } == set(names[3:])
    assert {model.name for model in store.search_registered_models("name ILIKE '%%'")} == set(names)


def test_search_registered_models_excludes_deleted_models(
    store: MongoDBModelRegistryStore,
):
    names = [f"search-registered-delete-{suffix}" for suffix in ("RM1", "RM2", "RM3")]
    for name in names:
        store.create_registered_model(name)

    store.delete_registered_model(names[-1])

    assert {model.name for model in store.search_registered_models(None)} == set(names[:-1])
    assert {
        model.name for model in store.search_registered_models(f"name = '{names[-1]}'")
    } == set()


def test_search_registered_models_filters_by_tags(
    store: MongoDBModelRegistryStore,
):
    first_name = "search-registered-tags-first"
    second_name = "search-registered-tags-second"
    store.create_registered_model(
        first_name,
        tags=[RegisteredModelTag("t1", "abc"), RegisteredModelTag("t2", "xyz")],
    )
    store.create_registered_model(
        second_name,
        tags=[
            RegisteredModelTag("t1", "abcd"),
            RegisteredModelTag("t2", "xyz123"),
            RegisteredModelTag("t3", "XYZ"),
        ],
    )

    assert {model.name for model in store.search_registered_models("tag.t3 = 'XYZ'")} == {
        second_name
    }
    assert {
        model.name
        for model in store.search_registered_models(f"name = '{first_name}' AND tag.t1 = 'abc'")
    } == {first_name}
    assert {model.name for model in store.search_registered_models("tag.t1 LIKE 'ab%'")} == {
        first_name,
        second_name,
    }
    assert {model.name for model in store.search_registered_models("tag.t1 ILIKE 'aB%'")} == {
        first_name,
        second_name,
    }
    assert {
        model.name
        for model in store.search_registered_models("tag.t1 LIKE 'ab%' AND tag.t2 LIKE 'xy%'")
    } == {first_name, second_name}
    assert {model.name for model in store.search_registered_models("tag.t3 = 'XYz'")} == set()
    assert {model.name for model in store.search_registered_models("tag.T3 = 'XYZ'")} == set()
    assert {model.name for model in store.search_registered_models("tag.t1 != 'abc'")} == {
        second_name
    }
    assert {
        model.name
        for model in store.search_registered_models("tag.t1 != 'abcd' AND tag.t1 LIKE 'ab%'")
    } == {first_name}


def test_search_registered_models_orders_and_paginates_stably(
    store: MongoDBModelRegistryStore,
    monkeypatch,
):
    clock = {"now": 1_700_004_000_000}
    monkeypatch.setattr(
        "mlflow_mongodb.model_registry.store.get_current_time_millis",
        lambda: clock["now"],
    )

    timestamped_names = []
    for timestamp, name in (
        (1, "OrderRM1"),
        (1, "OrderRM2"),
        (2, "OrderRM3"),
        (2, "OrderRM4"),
    ):
        clock["now"] = timestamp
        timestamped_names.append(store.create_registered_model(name).name)

    def names(order_by=None):
        return [
            model.name
            for model in store.search_registered_models(
                "name LIKE 'OrderRM%'",
                max_results=100,
                order_by=order_by,
            )
        ]

    assert names() == timestamped_names
    assert names(["name DESC"]) == timestamped_names[::-1]
    assert names(["last_updated_timestamp ASC", "name DESC"]) == [
        "OrderRM2",
        "OrderRM1",
        "OrderRM4",
        "OrderRM3",
    ]
    assert names(["timestamp DESC"]) == [
        "OrderRM3",
        "OrderRM4",
        "OrderRM1",
        "OrderRM2",
    ]
    assert names(["timestamp asc", "name desc"]) == [
        "OrderRM2",
        "OrderRM1",
        "OrderRM4",
        "OrderRM3",
    ]

    clock["now"] = 1_700_004_100_000
    expected_names = [
        store.create_registered_model(f"PageRM{index:03}").name for index in range(12)
    ]
    returned_names = []
    token = None
    for page_size in (3, 4, 10):
        page = store.search_registered_models(
            "name LIKE 'PageRM%'",
            max_results=page_size,
            page_token=token,
        )
        returned_names.extend(model.name for model in page)
        token = page.token
    assert returned_names == expected_names
    assert token is None
