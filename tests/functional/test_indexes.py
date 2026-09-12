"""Functional assertions for production MongoDB index definitions."""

from mlflow_mongodb import MongoDBModelRegistryStore
from mlflow_mongodb.repositories import (
    ModelVersionRepository,
    RegisteredModelRepository,
)


def _application_indexes(store, collection_name):
    return {
        index["name"]: {
            "key": list(index["key"].items()),
            "unique": index.get("unique", False),
        }
        for index in store._database[collection_name].list_indexes()
        if index["name"] != "_id_"
    }


def test_registered_model_repository_indexes(mongodb_store: MongoDBModelRegistryStore):
    assert _application_indexes(
        mongodb_store,
        mongodb_store._settings.registered_models_collection_name,
    ) == {
        RegisteredModelRepository.UNIQUE_NAME_INDEX: {
            "key": [("name", 1)],
            "unique": True,
        },
        RegisteredModelRepository.TAGS_INDEX: {
            "key": [("tags.key", 1), ("tags.value", 1)],
            "unique": False,
        },
    }


def test_model_version_repository_indexes(mongodb_store: MongoDBModelRegistryStore):
    assert _application_indexes(
        mongodb_store,
        mongodb_store._settings.model_versions_collection_name,
    ) == {
        ModelVersionRepository.UNIQUE_VERSION_INDEX: {
            "key": [("registered_model_id", 1), ("version", 1)],
            "unique": True,
        },
        ModelVersionRepository.LATEST_VERSION_INDEX: {
            "key": [
                ("registered_model_id", 1),
                ("current_stage", 1),
                ("version", -1),
            ],
            "unique": False,
        },
        ModelVersionRepository.TAGS_INDEX: {
            "key": [("tags.key", 1), ("tags.value", 1)],
            "unique": False,
        },
    }
