"""Unit tests for shared MongoDB collection settings."""

import pytest

from mlflow_mongodb.infrastructure.settings import MongoDBSettings


def test_empty_environment_uses_default_collection_names():
    settings = MongoDBSettings.from_environment({})

    assert settings.registered_models_collection_name == "registered_models"
    assert settings.model_versions_collection_name == "model_versions"
    assert settings.experiments_collection_name == "experiments"
    assert settings.experiment_id_counters_collection_name == "experiment_id_counters"


@pytest.mark.parametrize(
    ("variable", "field"),
    [
        ("MLFLOW_MONGODB_REGISTERED_MODELS_COLLECTION", "registered_models_collection_name"),
        ("MLFLOW_MONGODB_MODEL_VERSIONS_COLLECTION", "model_versions_collection_name"),
        ("MLFLOW_MONGODB_EXPERIMENTS_COLLECTION", "experiments_collection_name"),
        (
            "MLFLOW_MONGODB_EXPERIMENT_ID_COUNTERS_COLLECTION",
            "experiment_id_counters_collection_name",
        ),
        ("MLFLOW_MONGODB_TRACES_COLLECTION", "traces_collection_name"),
        ("MLFLOW_MONGODB_SPANS_COLLECTION", "spans_collection_name"),
        ("MLFLOW_MONGODB_ASSESSMENTS_COLLECTION", "assessments_collection_name"),
    ],
)
def test_environment_override_preserves_other_default(variable, field):
    settings = MongoDBSettings.from_environment({variable: "custom_collection"})
    expected = MongoDBSettings(**{field: "custom_collection"})

    assert settings == expected


def test_omitted_environment_reads_process_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_MONGODB_REGISTERED_MODELS_COLLECTION", "custom_models")
    monkeypatch.setenv("MLFLOW_MONGODB_MODEL_VERSIONS_COLLECTION", "custom_versions")
    monkeypatch.setenv("MLFLOW_MONGODB_EXPERIMENTS_COLLECTION", "custom_experiments")
    monkeypatch.setenv("MLFLOW_MONGODB_EXPERIMENT_ID_COUNTERS_COLLECTION", "custom_counters")

    assert MongoDBSettings.from_environment() == MongoDBSettings(
        registered_models_collection_name="custom_models",
        model_versions_collection_name="custom_versions",
        experiments_collection_name="custom_experiments",
        experiment_id_counters_collection_name="custom_counters",
    )


def test_explicit_environment_ignores_process_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_MONGODB_REGISTERED_MODELS_COLLECTION", "custom_models")
    monkeypatch.setenv("MLFLOW_MONGODB_MODEL_VERSIONS_COLLECTION", "custom_versions")

    assert MongoDBSettings.from_environment({}) == MongoDBSettings()


@pytest.mark.parametrize(
    "invalid_name",
    ["", "invalid\x00name", "invalid$name", "system.models"],
)
@pytest.mark.parametrize(
    "setting_field",
    [
        "registered_models_collection_name",
        "model_versions_collection_name",
        "experiments_collection_name",
        "experiment_id_counters_collection_name",
        "traces_collection_name",
        "spans_collection_name",
        "assessments_collection_name",
    ],
)
def test_mongodb_settings_reject_invalid_collection_names(setting_field, invalid_name):
    with pytest.raises(ValueError, match="Invalid MongoDB collection name"):
        MongoDBSettings(**{setting_field: invalid_name})
