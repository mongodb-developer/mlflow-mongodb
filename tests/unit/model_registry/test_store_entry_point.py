"""Local functional checks for MLflow entry-point discovery."""

import pytest
from mlflow.tracking._model_registry.utils import _get_store  # ruff: ignore[import-private-name]
from mlflow.tracking._tracking_service.utils import (  # ruff: ignore[import-private-name]
    _get_store as _get_tracking_store,
)

from mlflow_mongodb.model_registry.store import MongoDBModelRegistryStore
from mlflow_mongodb.tracking.store import MongoDBTrackingStore


@pytest.mark.parametrize("scheme", ["mongodb", "mongodb+srv"])
def test_supported_scheme_resolves_to_plugin_store(scheme):
    store_uri = f"{scheme}://localhost:27017/mlflow"
    store = _get_store(
        store_uri=store_uri,
        tracking_uri="file:///tmp/mlruns",
    )

    assert isinstance(store, MongoDBModelRegistryStore)
    assert store.store_uri == store_uri


@pytest.mark.parametrize("scheme", ["mongodb", "mongodb+srv"])
def test_supported_scheme_resolves_to_tracking_store(scheme):
    store_uri = f"{scheme}://localhost:27017/mlflow"
    artifact_uri = "file:///tmp/mlruns"
    store = _get_tracking_store(store_uri=store_uri, artifact_uri=artifact_uri)

    assert isinstance(store, MongoDBTrackingStore)
    assert store.store_uri == store_uri
    assert store.artifact_uri == artifact_uri
