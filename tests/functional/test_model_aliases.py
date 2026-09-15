"""Functional tests for registered-model aliases backed by MongoDB."""

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.utils import validation as mlflow_validation

from mlflow_mongodb import MongoDBModelRegistryStore

MODEL_NAME = "mongodb-functional-alias-model"
SUPPORTS_LATEST_ALIAS_LOOKUP = hasattr(
    mlflow_validation,
    "_validate_model_alias_name_reserved",
)


@pytest.fixture
def model_versions(store: MongoDBModelRegistryStore):
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
def test_latest_alias_resolves_without_being_stored(
    store: MongoDBModelRegistryStore,
    model_versions,
):
    _, expected_version = model_versions

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
    not SUPPORTS_LATEST_ALIAS_LOOKUP,
    reason="MLflow before 3.3 rejects the reserved latest alias during basic validation",
)
def test_latest_alias_raises_when_model_does_not_exist(store: MongoDBModelRegistryStore):
    with pytest.raises(MlflowException, match="Registered Model.*not found") as missing_error:
        store.get_model_version_by_alias(f"{MODEL_NAME}-missing", "latest")

    assert missing_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


@pytest.mark.skipif(
    SUPPORTS_LATEST_ALIAS_LOOKUP,
    reason="MLflow 3.3 and newer support latest as a virtual lookup alias",
)
def test_mlflow_before_3_3_rejects_latest_alias_lookup(store: MongoDBModelRegistryStore):
    with pytest.raises(MlflowException, match="latest.*reserved"):
        store.get_model_version_by_alias(MODEL_NAME, "latest")


def test_latest_alias_cannot_be_stored(
    store: MongoDBModelRegistryStore,
    model_versions,
):
    first_version, _ = model_versions

    with pytest.raises(MlflowException, match="latest.*reserved"):
        store.set_registered_model_alias(MODEL_NAME, "LaTeSt", first_version.version)

    assert store.get_registered_model(MODEL_NAME).aliases == {}


def test_stored_alias_lifecycle_is_reflected_on_models_and_versions(
    store: MongoDBModelRegistryStore,
    model_versions,
):
    first_version, second_version = model_versions

    store.set_registered_model_alias(MODEL_NAME, "candidate", str(second_version.version))

    assert store.get_registered_model(MODEL_NAME).aliases == {"candidate": second_version.version}
    assert store.get_model_version(MODEL_NAME, first_version.version).aliases == []
    assert store.get_model_version(MODEL_NAME, second_version.version).aliases == ["candidate"]
    assert store.get_model_version_by_alias(MODEL_NAME, "candidate").version == (
        second_version.version
    )

    store.set_registered_model_alias(MODEL_NAME, "candidate", first_version.version)
    assert store.get_registered_model(MODEL_NAME).aliases == {"candidate": first_version.version}
    assert store.get_model_version(MODEL_NAME, first_version.version).aliases == ["candidate"]
    assert store.get_model_version(MODEL_NAME, second_version.version).aliases == []

    store.delete_registered_model_alias(MODEL_NAME, "candidate")
    assert store.get_registered_model(MODEL_NAME).aliases == {}
    assert store.get_model_version(MODEL_NAME, first_version.version).aliases == []

    with pytest.raises(
        MlflowException,
        match="alias candidate not found",
    ) as missing_alias_error:
        store.get_model_version_by_alias(MODEL_NAME, "candidate")
    assert missing_alias_error.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_deleting_alias_target_removes_alias(
    store: MongoDBModelRegistryStore,
    model_versions,
):
    _, second_version = model_versions
    store.set_registered_model_alias(MODEL_NAME, "candidate", second_version.version)

    store.delete_model_version(MODEL_NAME, second_version.version)

    assert store.get_registered_model(MODEL_NAME).aliases == {}
    with pytest.raises(MlflowException, match="alias candidate not found"):
        store.get_model_version_by_alias(MODEL_NAME, "candidate")


def test_alias_operations_validate_targets(store: MongoDBModelRegistryStore):
    store.create_registered_model(MODEL_NAME)

    with pytest.raises(MlflowException, match="Model Version") as missing_version_error:
        store.set_registered_model_alias(MODEL_NAME, "candidate", 99)
    assert missing_version_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    store.delete_registered_model(MODEL_NAME)
    with pytest.raises(MlflowException, match="Registered Model") as missing_model_error:
        store.delete_registered_model_alias(MODEL_NAME, "candidate")
    assert missing_model_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_deleting_registered_model_makes_its_aliases_unresolvable(
    store: MongoDBModelRegistryStore,
    model_versions,
):
    _, second_version = model_versions
    store.set_registered_model_alias(MODEL_NAME, "candidate", second_version.version)

    store.delete_registered_model(MODEL_NAME)

    with pytest.raises(MlflowException, match="Registered Model.*not found") as missing_error:
        store.get_model_version_by_alias(MODEL_NAME, "candidate")
    assert missing_error.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
