"""Local functional checks for MLflow entry-point discovery."""

import pytest
from mlflow.tracking._model_registry.utils import _get_store  # ruff: ignore[import-private-name]

from mlflow_mongodb import MongoDBModelRegistryStore


@pytest.mark.parametrize("scheme", ["mongodb", "mongodb+srv"])
def test_supported_scheme_resolves_to_plugin_store(scheme):
    store_uri = f"{scheme}://localhost:27017/mlflow"
    store = _get_store(
        store_uri=store_uri,
        tracking_uri="file:///tmp/mlruns",
    )

    assert isinstance(store, MongoDBModelRegistryStore)
    assert store.store_uri == store_uri
