from mlflow_mongodb.tracking.repositories import ExperimentRepository
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
