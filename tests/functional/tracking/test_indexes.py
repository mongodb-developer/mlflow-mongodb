from mlflow_mongodb.tracking.repositories import ExperimentRepository, RunRepository
from mlflow_mongodb.tracking.store import MongoDBTrackingStore


def test_experiment_repository_indexes(mongodb_store: MongoDBTrackingStore):
    indexes = {
        index["name"]: {
            "key": list(index["key"].items()),
            "unique": index.get("unique", False),
        }
        for index in mongodb_store._database[
            mongodb_store._settings.experiments_collection_name
        ].list_indexes()
        if index["name"] != "_id_"
    }

    assert indexes == {
        ExperimentRepository.UNIQUE_NAME_INDEX: {
            "key": [("name", 1)],
            "unique": True,
        },
        "experiments_lifecycle_creation_id": {
            "key": [("lifecycle_stage", 1), ("creation_time", -1), ("_id", 1)],
            "unique": False,
        },
    }


def test_run_repository_indexes(mongodb_store: MongoDBTrackingStore):
    database = mongodb_store._database
    settings = mongodb_store._settings
    run_indexes = database[settings.runs_collection_name].index_information()
    metric_indexes = database[settings.run_metrics_collection_name].index_information()

    assert run_indexes[RunRepository.EXPERIMENT_INDEX]["key"] == [("experiment_id", 1)]
    assert not run_indexes[RunRepository.EXPERIMENT_INDEX].get("unique", False)
    assert metric_indexes["run_metrics_run_k_timestamp"]["key"] == [
        ("run_id", 1),
        ("k", 1),
        ("timestamp", 1),
    ]
    assert not metric_indexes["run_metrics_run_k_timestamp"].get("unique", False)


def test_logged_model_repository_indexes(mongodb_store: MongoDBTrackingStore):
    settings = mongodb_store._settings
    indexes = mongodb_store._database[settings.logged_models_collection_name].index_information()
    assert indexes["_id_"]["key"] == [("_id", 1)]
    expected = {
        "logged_models_experiment_creation_id": [
            ("experiment_id", 1),
            ("creation_timestamp", -1),
            ("_id", 1),
        ],
        "logged_models_experiment_tags": [("experiment_id", 1), ("tags.k", 1), ("tags.v", 1)],
        "logged_models_experiment_params": [("experiment_id", 1), ("params.k", 1), ("params.v", 1)],
    }
    assert set(indexes) == {"_id_", *expected}
    for name, keys in expected.items():
        assert indexes[name]["key"] == keys
        assert not indexes[name].get("unique", False)
    metric_indexes = mongodb_store._database[
        settings.run_metrics_collection_name
    ].index_information()
    for name, keys in {
        "run_metrics_model_k_latest": [
            ("model_id", 1),
            ("k", 1),
            ("timestamp", -1),
            ("step", -1),
            ("run_id", 1),
            ("_id", 1),
        ],
        "run_metrics_model_dataset": [("model_id", 1), ("dataset_name", 1), ("dataset_digest", 1)],
    }.items():
        assert metric_indexes[name]["key"] == keys
        assert not metric_indexes[name].get("unique", False)
