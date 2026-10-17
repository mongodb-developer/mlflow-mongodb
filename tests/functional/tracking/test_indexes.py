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
