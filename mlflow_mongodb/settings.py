"""Runtime settings for the MongoDB MLflow plugin."""

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class MongoDBSettings:
    """Configuration for MongoDB collection names."""

    registered_models_collection_name: str = "registered_models"
    model_versions_collection_name: str = "model_versions"

    def __post_init__(self) -> None:
        for field_name in (
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
            registered_models_collection_name=environment.get(
                "MLFLOW_MONGODB_REGISTERED_MODELS_COLLECTION",
                cls.registered_models_collection_name,
            ),
            model_versions_collection_name=environment.get(
                "MLFLOW_MONGODB_MODEL_VERSIONS_COLLECTION",
                cls.model_versions_collection_name,
            ),
        )
