"""Functional tests for registered-model aliases backed by MongoDB."""

import pytest
from mlflow.exceptions import MlflowException
from mlflow.utils import validation as mlflow_validation

from mlflow_mongodb import MongoDBModelRegistryStore

MODEL_NAME = "mongodb-functional-alias-model"
SUPPORTS_LATEST_ALIAS_LOOKUP = hasattr(
    mlflow_validation,
    "_validate_model_alias_name_reserved",
)


def _create_model_versions(store: MongoDBModelRegistryStore):
    store.create_registered_model(MODEL_NAME)
    first_version = store.create_model_version(
        MODEL_NAME,
        source="s3://models/mongodb-functional-alias-model/1",
    )
    second_version = store.create_model_version(
        MODEL_NAME,
        source="s3://models/mongodb-functional-alias-model/2",
    )
    return first_version, second_version


@pytest.mark.skipif(
    not SUPPORTS_LATEST_ALIAS_LOOKUP,
    reason="MLflow before 3.3 rejects the reserved latest alias during basic validation",
)
def test_latest_alias_resolves_without_being_stored(store: MongoDBModelRegistryStore):
    _, expected_version = _create_model_versions(store)

    resolved_version = store.get_model_version_by_alias(MODEL_NAME, "LaTeSt")

    assert resolved_version.version == expected_version.version
    assert resolved_version.aliases == []
    assert store.get_registered_model(MODEL_NAME).aliases == {}


@pytest.mark.skipif(
    not SUPPORTS_LATEST_ALIAS_LOOKUP,
    reason="MLflow before 3.3 rejects the reserved latest alias during basic validation",
)
def test_latest_alias_raises_when_model_has_no_versions(store: MongoDBModelRegistryStore):
    store.create_registered_model(MODEL_NAME)

    with pytest.raises(MlflowException, match="Latest version not found"):
        store.get_model_version_by_alias(MODEL_NAME, "latest")


@pytest.mark.skipif(
    SUPPORTS_LATEST_ALIAS_LOOKUP,
    reason="MLflow 3.3 and newer support latest as a virtual lookup alias",
)
def test_mlflow_before_3_3_rejects_latest_alias_lookup(store: MongoDBModelRegistryStore):
    _create_model_versions(store)

    with pytest.raises(MlflowException, match="latest.*reserved"):
        store.get_model_version_by_alias(MODEL_NAME, "latest")


def test_latest_alias_cannot_be_stored(store: MongoDBModelRegistryStore):
    first_version, _ = _create_model_versions(store)

    with pytest.raises(MlflowException, match="latest.*reserved"):
        store.set_registered_model_alias(MODEL_NAME, "LaTeSt", first_version.version)

    assert store.get_registered_model(MODEL_NAME).aliases == {}
