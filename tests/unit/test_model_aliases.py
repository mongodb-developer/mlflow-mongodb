"""Unit tests for registered-model alias behavior across MLflow versions."""

import pytest
from mlflow.exceptions import MlflowException
from mlflow.utils import validation as mlflow_validation

MODEL_NAME = "fraud-detector"
SUPPORTS_LATEST_ALIAS_LOOKUP = hasattr(
    mlflow_validation,
    "_validate_model_alias_name_reserved",
)


@pytest.mark.skipif(
    not SUPPORTS_LATEST_ALIAS_LOOKUP,
    reason="MLflow before 3.3 rejects the reserved latest alias during basic validation",
)
def test_get_latest_alias_returns_latest_version_without_stored_alias(
    store,
    registered_model_repository,
    model_version_repository,
    registered_model_record_factory,
    model_version_record_factory,
    registered_model_details_factory,
):
    registered_model = registered_model_record_factory(name=MODEL_NAME)
    latest_version = model_version_record_factory(
        registered_model_id=registered_model.model_id,
        version=2,
        source=f"s3://models/{MODEL_NAME}/2",
    )
    registered_model_repository.find_latest_version_by_name.return_value = (
        registered_model_details_factory(
            registered_model=registered_model,
            latest_versions=(latest_version,),
        )
    )

    model_version = store.get_model_version_by_alias(MODEL_NAME, "LaTeSt")

    assert model_version.version == 2
    assert model_version.aliases == []
    registered_model_repository.find_latest_version_by_name.assert_called_once_with(MODEL_NAME)
    registered_model_repository.find_by_name.assert_not_called()
    model_version_repository.find_latest_by_stages.assert_not_called()
    model_version_repository.find_by_version.assert_not_called()


@pytest.mark.skipif(
    not SUPPORTS_LATEST_ALIAS_LOOKUP,
    reason="MLflow before 3.3 rejects the reserved latest alias during basic validation",
)
def test_get_latest_alias_raises_when_model_has_no_versions(
    store,
    registered_model_repository,
    registered_model_record_factory,
    registered_model_details_factory,
):
    registered_model_repository.find_latest_version_by_name.return_value = (
        registered_model_details_factory(
            registered_model=registered_model_record_factory(name=MODEL_NAME)
        )
    )

    with pytest.raises(MlflowException, match="Latest version not found"):
        store.get_model_version_by_alias(MODEL_NAME, "latest")


def test_latest_alias_cannot_be_stored(
    store,
    registered_model_repository,
    model_version_repository,
):
    with pytest.raises(MlflowException, match="latest.*reserved"):
        store.set_registered_model_alias(MODEL_NAME, "LaTeSt", 1)

    model_version_repository.exists_for_registered_model.assert_not_called()
    registered_model_repository.set_alias_by_name.assert_not_called()
