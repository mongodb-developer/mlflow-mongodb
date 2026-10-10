from unittest.mock import MagicMock, call

import pytest
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure

from mlflow_mongodb.infrastructure.errors import (
    RepositoryAlreadyExistsError,
    RepositoryPersistenceError,
)
from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.repositories import ExperimentRepository
from mlflow_mongodb.tracking.repositories.experiments import ExperimentFilter, ExperimentOrder


@pytest.fixture
def collection():
    return MagicMock()


@pytest.fixture
def repository(collection):
    database = MagicMock()
    database.__getitem__.return_value = collection
    settings = MongoDBSettings(experiments_collection_name="test_experiments")

    repository = ExperimentRepository(database, settings=settings)

    database.__getitem__.assert_called_once_with("test_experiments")
    collection.create_index.assert_has_calls(
        [
            call([("name", ASCENDING)], unique=True, name="experiments_name_unique"),
            call(
                [("lifecycle_stage", ASCENDING), ("creation_time", DESCENDING), ("_id", ASCENDING)],
                name="experiments_lifecycle_creation_id",
            ),
        ]
    )
    return repository


@pytest.mark.parametrize("comparator", ["RLIKE", "BETWEEN"])
def test_value_condition_rejects_unsupported_experiment_comparator(comparator):
    with pytest.raises(ValueError) as caught:
        ExperimentRepository._value_condition(comparator, "risk")

    assert str(caught.value) == f"Unsupported experiment comparator: {comparator}"


@pytest.mark.parametrize(
    ("key_pattern", "expected_error"),
    [
        ({"name": 1}, RepositoryAlreadyExistsError),
        ({"_id": 1}, RepositoryPersistenceError),
    ],
)
def test_duplicate_experiment_name_and_id_have_distinct_errors(
    repository, collection, key_pattern, expected_error
):
    duplicate = DuplicateKeyError("Private duplicate details", details={"keyPattern": key_pattern})
    collection.insert_one.side_effect = duplicate

    with pytest.raises(expected_error) as caught:
        repository.create(
            experiment_id="42",
            name="example-experiment",
            artifact_location="s3://artifacts/42",
            lifecycle_stage="active",
            creation_timestamp=1000,
            tags={"team": "risk"},
        )

    assert caught.value.__cause__ is duplicate
    collection.insert_one.assert_called_once_with(
        {
            "_id": "42",
            "name": "example-experiment",
            "artifact_location": "s3://artifacts/42",
            "lifecycle_stage": "active",
            "creation_time": 1000,
            "last_update_time": 1000,
            "tags": [{"k": "team", "v": "risk"}],
        }
    )


def test_search_translates_deferred_cursor_failure(repository, collection):
    backend_error = OperationFailure("Private cursor details")
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.__iter__.side_effect = backend_error
    collection.find.return_value = cursor

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.search(
            lifecycle_stages=["active"],
            filters=(),
            order_by=(ExperimentOrder("creation_time", False),),
            offset=4,
            limit=3,
        )

    assert caught.value.__cause__ is backend_error
    collection.find.assert_called_once_with({"$and": [{"lifecycle_stage": {"$in": ["active"]}}]})
    cursor.sort.assert_called_once_with([("creation_time", DESCENDING)])
    cursor.skip.assert_called_once_with(4)
    cursor.limit.assert_called_once_with(3)
    cursor.__iter__.assert_called_once()


@pytest.mark.parametrize(
    ("comparator", "expected_clause"),
    [
        ("IS NULL", {"tags": {"$not": {"$elemMatch": {"k": "team"}}}}),
        ("IS NOT NULL", {"tags": {"$elemMatch": {"k": "team"}}}),
    ],
)
def test_search_tag_existence_filters_build_key_only_clauses(
    repository, collection, comparator, expected_clause
):
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.__iter__.return_value = iter([])
    collection.find.return_value = cursor

    records = repository.search(
        lifecycle_stages=["active"],
        filters=(ExperimentFilter("tag", "team", comparator, None),),
        order_by=(ExperimentOrder("creation_time", False),),
        offset=0,
        limit=10,
    )

    assert records == []
    collection.find.assert_called_once_with(
        {"$and": [{"lifecycle_stage": {"$in": ["active"]}}, expected_clause]}
    )
    cursor.sort.assert_called_once_with([("creation_time", DESCENDING)])
    cursor.skip.assert_called_once_with(0)
    cursor.limit.assert_called_once_with(10)
    cursor.__iter__.assert_called_once()


def test_repository_initialization_translates_index_failure():
    backend_error = OperationFailure("Private index details")
    collection = MagicMock()
    collection.create_index.side_effect = backend_error
    database = MagicMock()
    database.__getitem__.return_value = collection

    with pytest.raises(RepositoryPersistenceError) as caught:
        ExperimentRepository(database)

    assert caught.value.__cause__ is backend_error
    database.__getitem__.assert_called_once_with("experiments")
    collection.create_index.assert_called_once_with(
        [("name", ASCENDING)], unique=True, name="experiments_name_unique"
    )


def test_record_conversion_errors_are_not_reported_as_database_failures(repository, collection):
    collection.find_one.return_value = {"_id": "42"}

    with pytest.raises(KeyError, match="name"):
        repository.find_by_id("42")

    collection.find_one.assert_called_once_with({"_id": "42"})


def test_missing_experiment_is_returned_without_persistence_error(repository, collection):
    collection.find_one.return_value = None

    assert repository.find_by_id("missing") is None

    collection.find_one.assert_called_once_with({"_id": "missing"})
