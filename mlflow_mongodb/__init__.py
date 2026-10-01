"""MongoDB model registry store plugin for MLflow."""

from mlflow_mongodb._version import __version__
from mlflow_mongodb.model_registry.store import MongoDBModelRegistryStore

__all__ = ["MongoDBModelRegistryStore", "__version__"]
