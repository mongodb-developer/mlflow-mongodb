"""Experiment tag mutations and search visibility against MongoDB."""

import pytest
from mlflow.entities import ExperimentTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.utils.validation import (
    MAX_EXPERIMENT_TAG_KEY_LENGTH,
    MAX_EXPERIMENT_TAG_VAL_LENGTH,
)

from mlflow_mongodb.tracking.store import MongoDBTrackingStore


def _stored_experiment(store: MongoDBTrackingStore, experiment_id: str):
    return store._database[store._settings.experiments_collection_name].find_one(
        {"_id": int(experiment_id)}
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [("team", "risk"), ("team.name", "$risk"), ("team name", "$$ROOT"), ("empty", "")],
)
def test_set_experiment_tag_persists_literals_and_preserves_metadata(store, clock, key, value):
    experiment_id = store.create_experiment("tag-create", tags=[ExperimentTag("keep", "yes")])
    before = _stored_experiment(store, experiment_id)
    clock["now"] += 1000

    assert store.set_experiment_tag(experiment_id, ExperimentTag(key, value)) is None

    expected_tags = {"keep": "yes", key: value}
    assert store.get_experiment(experiment_id).tags == expected_tags
    assert store.get_experiment_by_name("tag-create").tags == expected_tags
    assert _stored_experiment(store, experiment_id) == {
        **before,
        "tags": [{"k": "keep", "v": "yes"}, {"k": key, "v": value}],
    }
    results = store.search_experiments(filter_string=f"tags.`{key}` = '{value}'")
    assert [experiment.experiment_id for experiment in results] == [experiment_id]


@pytest.mark.parametrize("value", ["new", "$risk", "$$ROOT", ""])
def test_replace_experiment_tag_is_unique_and_idempotent(store, clock, value):
    experiment_id = store.create_experiment(
        "tag-replace",
        tags=[ExperimentTag("team.name", "old"), ExperimentTag("keep", "yes")],
    )
    before = _stored_experiment(store, experiment_id)
    clock["now"] += 1000

    store.set_experiment_tag(experiment_id, ExperimentTag("team.name", value))
    replaced = _stored_experiment(store, experiment_id)
    assert replaced == {
        **before,
        "tags": [{"k": "keep", "v": "yes"}, {"k": "team.name", "v": value}],
    }
    assert store.get_experiment(experiment_id).tags == {"keep": "yes", "team.name": value}
    assert store.search_experiments(filter_string="tags.`team.name` = 'old'") == []
    assert [
        experiment.experiment_id
        for experiment in store.search_experiments(filter_string=f"tags.`team.name` = '{value}'")
    ] == [experiment_id]

    clock["now"] += 1000
    store.set_experiment_tag(experiment_id, ExperimentTag("team.name", value))

    assert _stored_experiment(store, experiment_id) == replaced
    assert store.get_experiment(experiment_id).tags == {"keep": "yes", "team.name": value}


def test_set_experiment_tag_on_empty_array(store):
    experiment_id = store.create_experiment("empty-tags")

    store.set_experiment_tag(experiment_id, ExperimentTag("team", "$risk"))

    assert store.get_experiment(experiment_id).tags == {"team": "$risk"}
    assert _stored_experiment(store, experiment_id)["tags"] == [{"k": "team", "v": "$risk"}]


@pytest.mark.parametrize("key", ["team", "team.name", "team name"])
def test_delete_experiment_tag_preserves_other_tags_and_metadata(store, clock, key):
    experiment_id = store.create_experiment(
        "tag-delete", tags=[ExperimentTag(key, "$risk"), ExperimentTag("keep", key)]
    )
    before = _stored_experiment(store, experiment_id)
    clock["now"] += 1000

    assert store.delete_experiment_tag(experiment_id, key) is None

    assert store.get_experiment(experiment_id).tags == {"keep": key}
    assert store.get_experiment_by_name("tag-delete").tags == {"keep": key}
    assert _stored_experiment(store, experiment_id) == {**before, "tags": [{"k": "keep", "v": key}]}
    assert store.search_experiments(filter_string=f"tags.`{key}` = '$risk'") == []


def test_delete_last_experiment_tag_leaves_empty_array(store):
    experiment_id = store.create_experiment("last-tag", tags=[ExperimentTag("team", "risk")])

    store.delete_experiment_tag(experiment_id, "team")

    assert store.get_experiment(experiment_id).tags == {}
    assert _stored_experiment(store, experiment_id)["tags"] == []


@pytest.mark.parametrize("tags", [[], [ExperimentTag("keep", "missing")]])
def test_delete_missing_tag_reports_not_found_without_mutation(store, tags):
    experiment_id = store.create_experiment("missing-tag", tags=tags)
    before = _stored_experiment(store, experiment_id)
    before_entity = store.get_experiment(experiment_id)

    with pytest.raises(MlflowException, match="No tag with name: missing") as caught:
        store.delete_experiment_tag(experiment_id, "missing")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert _stored_experiment(store, experiment_id) == before
    assert store.get_experiment(experiment_id).tags == before_entity.tags


def test_deleting_experiment_tag_twice_reports_not_found(store):
    experiment_id = store.create_experiment("repeat-delete", tags=[ExperimentTag("team", "risk")])
    store.delete_experiment_tag(experiment_id, "team")
    assert store.get_experiment(experiment_id).tags == {}
    before = _stored_experiment(store, experiment_id)
    assert before["tags"] == []

    with pytest.raises(MlflowException) as caught:
        store.delete_experiment_tag(experiment_id, "team")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert _stored_experiment(store, experiment_id) == before


@pytest.mark.parametrize("experiment_id", [None, "missing", 123456789])
def test_set_tag_on_missing_experiment_does_not_create_document(store, experiment_id):
    with pytest.raises(MlflowException, match="No Experiment with id") as caught:
        store.set_experiment_tag(experiment_id, ExperimentTag("team", "risk"))

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert store._database[store._settings.experiments_collection_name].count_documents({}) == 1


@pytest.mark.parametrize("experiment_id", [None, "missing", 123456789])
def test_delete_tag_on_missing_experiment_does_not_create_document(store, experiment_id):
    with pytest.raises(MlflowException, match="No Experiment with id") as caught:
        store.delete_experiment_tag(experiment_id, "team")

    assert caught.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert store._database[store._settings.experiments_collection_name].count_documents({}) == 1


def test_set_tag_on_deleted_experiment_is_rejected_without_mutation(store):
    experiment_id = store.create_experiment("deleted-tag-set", tags=[ExperimentTag("team", "old")])
    store.delete_experiment(experiment_id)
    before = _stored_experiment(store, experiment_id)

    with pytest.raises(MlflowException, match="must be in the 'active' state") as caught:
        store.set_experiment_tag(experiment_id, ExperimentTag("team", "new"))

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert _stored_experiment(store, experiment_id) == before
    assert store.get_experiment(experiment_id).tags == {"team": "old"}


@pytest.mark.parametrize("key", ["team", "missing"])
def test_delete_tag_on_deleted_experiment_checks_lifecycle_before_tag_presence(store, key):
    experiment_id = store.create_experiment(
        "deleted-tag-delete", tags=[ExperimentTag("team", "old")]
    )
    store.delete_experiment(experiment_id)
    before = _stored_experiment(store, experiment_id)

    with pytest.raises(MlflowException, match="must be in the 'active' state") as caught:
        store.delete_experiment_tag(experiment_id, key)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert _stored_experiment(store, experiment_id) == before
    assert store.get_experiment(experiment_id).tags == {"team": "old"}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (None, "value"),
        ("$invalid", "value"),
        ("a/../b", "value"),
        ("k" * (MAX_EXPERIMENT_TAG_KEY_LENGTH + 1), "value"),
        ("valid", "v" * (MAX_EXPERIMENT_TAG_VAL_LENGTH + 1)),
    ],
)
def test_invalid_tag_is_rejected_without_mutation(store, key, value):
    experiment_id = store.create_experiment("invalid-tag", tags=[ExperimentTag("keep", "yes")])
    before = _stored_experiment(store, experiment_id)

    with pytest.raises(MlflowException) as caught:
        store.set_experiment_tag(experiment_id, ExperimentTag(key, value))

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert _stored_experiment(store, experiment_id) == before
    assert store.get_experiment(experiment_id).tags == {"keep": "yes"}


def test_invalid_tag_validation_precedes_missing_experiment_lookup(store):
    with pytest.raises(MlflowException) as caught:
        store.set_experiment_tag("missing", ExperimentTag("$invalid", "value"))

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert store._database[store._settings.experiments_collection_name].count_documents({}) == 1


def test_experiment_tag_mutations_do_not_affect_other_experiments(store):
    first = store.create_experiment("first-tags", tags=[ExperimentTag("team", "first")])
    second = store.create_experiment("second-tags", tags=[ExperimentTag("team", "second")])
    untouched = _stored_experiment(store, second)

    store.set_experiment_tag(first, ExperimentTag("team", "updated"))

    assert store.get_experiment(first).tags == {"team": "updated"}
    assert _stored_experiment(store, first)["tags"] == [{"k": "team", "v": "updated"}]
    assert _stored_experiment(store, second) == untouched

    store.delete_experiment_tag(first, "team")

    assert store.get_experiment(first).tags == {}
    assert _stored_experiment(store, first)["tags"] == []
    assert store.get_experiment(second).tags == {"team": "second"}
    assert _stored_experiment(store, second) == untouched
