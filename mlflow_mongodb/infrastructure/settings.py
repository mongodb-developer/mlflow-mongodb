"""Runtime settings for the MongoDB MLflow plugin."""

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class MongoDBSettings:
    """Configuration for MongoDB collection names."""

    experiments_collection_name: str = "experiments"
    runs_collection_name: str = "runs"
    run_metrics_collection_name: str = "run_metrics"
    logged_models_collection_name: str = "logged_models"
    registered_models_collection_name: str = "registered_models"
    model_versions_collection_name: str = "model_versions"

    def __post_init__(self) -> None:
        for field_name in (
            "experiments_collection_name",
            "runs_collection_name",
            "run_metrics_collection_name",
            "logged_models_collection_name",
            "registered_models_collection_name",
            "model_versions_collection_name",
        ):
            collection_name = getattr(self, field_name)
            if (
                not collection_name
                or "\x00" in collection_name
                or "$" in collection_name
                or collection_name.startswith("system.")
            ):
                raise ValueError(f"Invalid MongoDB collection name: {collection_name!r}")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "MongoDBSettings":
        """Load collection-name overrides from environment variables."""
        environment = os.environ if environ is None else environ
        return cls(
            experiments_collection_name=environment.get(
                "MLFLOW_MONGODB_EXPERIMENTS_COLLECTION",
                cls.experiments_collection_name,
            ),
            runs_collection_name=environment.get(
                "MLFLOW_MONGODB_RUNS_COLLECTION",
                cls.runs_collection_name,
            ),
            run_metrics_collection_name=environment.get(
                "MLFLOW_MONGODB_RUN_METRICS_COLLECTION",
                cls.run_metrics_collection_name,
            ),
            logged_models_collection_name=environment.get(
                "MLFLOW_MONGODB_LOGGED_MODELS_COLLECTION",
                cls.logged_models_collection_name,
            ),
            registered_models_collection_name=environment.get(
                "MLFLOW_MONGODB_REGISTERED_MODELS_COLLECTION",
                cls.registered_models_collection_name,
            ),
            model_versions_collection_name=environment.get(
                "MLFLOW_MONGODB_MODEL_VERSIONS_COLLECTION",
                cls.model_versions_collection_name,
            ),
        )
