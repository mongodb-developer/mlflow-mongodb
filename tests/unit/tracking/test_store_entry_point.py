import pytest
from mlflow.tracking._tracking_service.utils import _get_store  # ruff: ignore[import-private-name]

from mlflow_mongodb.tracking.store import MongoDBTrackingStore


@pytest.mark.parametrize("scheme", ["mongodb", "mongodb+srv"])
def test_supported_scheme_resolves_to_tracking_store(scheme):
    store_uri = f"{scheme}://localhost:27017/mlflow"
    artifact_uri = "s3://experiment-artifacts"

    store = _get_store(store_uri=store_uri, artifact_uri=artifact_uri)

    assert isinstance(store, MongoDBTrackingStore)
    assert store.store_uri == store_uri
    assert store.artifact_uri == artifact_uri
    assert "_mongo_client" not in store.__dict__
