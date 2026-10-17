from concurrent.futures import ThreadPoolExecutor

import pytest
from mlflow.entities import ExperimentTag, LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.utils.uri import resolve_uri_if_local

from mlflow_mongodb.tracking.store import MongoDBTrackingStore


def _reopen(store, artifact_uri=None):
    reopened = MongoDBTrackingStore(store.store_uri, artifact_uri=artifact_uri)
    reopened.__dict__["_database"] = store._database
    return reopened


@pytest.mark.parametrize("lookup", ["id", "name", "search"])
@pytest.mark.parametrize("artifact_uri", [None, "s3://default-artifacts"])
def test_fresh_database_has_default_experiment(store, clock, lookup, artifact_uri):
    collection = store._database[store._settings.experiments_collection_name]
    collection.delete_many({})
    reopened = _reopen(store, artifact_uri)

    if lookup == "id":
        experiment = reopened.get_experiment("0")
    elif lookup == "name":
        experiment = reopened.get_experiment_by_name("Default")
    else:
        (experiment,) = reopened.search_experiments()

    assert experiment.experiment_id == "0"
    assert experiment.name == "Default"
    assert experiment.lifecycle_stage == LifecycleStage.ACTIVE
    assert experiment.artifact_location == resolve_uri_if_local(f"{artifact_uri or './mlruns'}/0")
    assert experiment.creation_time == clock["now"]
    assert experiment.last_update_time == clock["now"]
    assert experiment.tags == {}
    assert collection.count_documents({}) == 1
    assert MlflowClient(tracking_uri=store.store_uri).get_experiment("0").name == "Default"


def test_reopening_preserves_modified_and_deleted_default_experiment(store, clock):
    store.rename_experiment("0", "renamed-default")
    store.set_experiment_tag("0", ExperimentTag("team", "risk"))
    store.delete_experiment("0")
    collection = store._database[store._settings.experiments_collection_name]
    before = collection.find_one({"_id": 0})
    clock["now"] += 1000

    experiment = _reopen(store, "s3://different-artifacts").get_experiment("0")

    assert experiment.lifecycle_stage == LifecycleStage.DELETED
    assert collection.find_one({"_id": 0}) == before


def test_concurrent_stores_initialize_one_default_experiment(store):
    collection = store._database[store._settings.experiments_collection_name]
    collection.delete_many({})

    def open_default(_):
        return _reopen(store, store.artifact_uri).get_experiment("0")

    with ThreadPoolExecutor(max_workers=8) as executor:
        experiments = list(executor.map(open_default, range(16)))

    assert {experiment.experiment_id for experiment in experiments} == {"0"}
    assert len({experiment.creation_time for experiment in experiments}) == 1
    assert collection.count_documents({}) == 1


def test_default_experiment_name_is_reserved_by_existing_record(store):
    with pytest.raises(MlflowException, match="already exists"):
        store.create_experiment("Default")

    assert store.get_experiment_by_name("Default").experiment_id == "0"


def test_default_initialization_reports_conflicting_legacy_name(store):
    collection = store._database[store._settings.experiments_collection_name]
    document = collection.find_one({"_id": 0})
    collection.delete_many({})
    document["_id"] = 42
    collection.insert_one(document)

    with pytest.raises(MlflowException, match="Unable to initialize experiment repository"):
        _reopen(store).get_experiment("0")

    assert collection.find_one({"_id": 0}) is None
    assert collection.find_one({"_id": 42}) == document
