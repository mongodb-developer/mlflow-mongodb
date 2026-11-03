from unittest.mock import MagicMock, call

import pytest
from pymongo import ASCENDING, DESCENDING, ReturnDocument
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
def counter_collection():
    return MagicMock()


@pytest.fixture
def repository(collection, counter_collection):
    database = MagicMock()
    database.__getitem__.side_effect = {
        "test_experiments": collection,
        "experiment_id_counters": counter_collection,
    }.__getitem__
    settings = MongoDBSettings(experiments_collection_name="test_experiments")

    repository = ExperimentRepository(database, settings=settings)

    database.__getitem__.assert_has_calls(
        [call("test_experiments"), call("experiment_id_counters")]
    )
    collection.create_index.assert_has_calls(
        [
            call([("name", ASCENDING)], unique=True, name="experiments_name_unique"),
            call(
                [("lifecycle_stage", ASCENDING), ("creation_time", DESCENDING), ("_id", ASCENDING)],
                name="experiments_lifecycle_creation_id",
            ),
        ]
    )
    counter_collection.update_one.assert_called_once_with(
        {"_id": "test_experiments"},
        {"$setOnInsert": {"value": 0}},
        upsert=True,
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
            experiment_id=42,
            name="example-experiment",
            artifact_location="s3://artifacts/42",
            lifecycle_stage="active",
            creation_timestamp=1000,
            tags={"team": "risk"},
        )

    assert caught.value.__cause__ is duplicate
    collection.insert_one.assert_called_once_with(
        {
            "_id": 42,
            "name": "example-experiment",
            "artifact_location": "s3://artifacts/42",
            "lifecycle_stage": "active",
            "creation_time": 1000,
            "last_update_time": 1000,
            "tags": [{"k": "team", "v": "risk"}],
        }
    )


@pytest.mark.parametrize(("counter_value", "expected_id"), [(1, 1), (2**31, 2**31)])
def test_allocate_experiment_id_increments_the_configured_counter_atomically(
    repository, counter_collection, counter_value, expected_id
):
    counter_collection.find_one_and_update.return_value = {"value": counter_value}

    experiment_id = repository.allocate_experiment_id()

    assert experiment_id == expected_id
    assert type(experiment_id) is int
    counter_collection.find_one_and_update.assert_called_once_with(
        {"_id": "test_experiments"},
        {"$inc": {"value": 1}},
        return_document=ReturnDocument.AFTER,
    )


def test_allocate_experiment_id_rejects_missing_counter(repository, counter_collection):
    counter_collection.find_one_and_update.return_value = None

    with pytest.raises(RepositoryPersistenceError, match="Unable to allocate experiment ID"):
        repository.allocate_experiment_id()


def test_allocate_experiment_id_rejects_value_at_int64_ceiling(repository, counter_collection):
    counter_collection.find_one_and_update.return_value = {"value": 2**63 - 1}

    with pytest.raises(RepositoryPersistenceError, match="Unable to allocate experiment ID"):
        repository.allocate_experiment_id()


def test_allocate_experiment_id_translates_counter_failure(repository, counter_collection):
    backend_error = OperationFailure("Private counter details")
    counter_collection.find_one_and_update.side_effect = backend_error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.allocate_experiment_id()

    assert caught.value.__cause__ is backend_error


def test_allocated_id_is_not_rolled_back_when_experiment_insert_fails(
    repository, counter_collection, collection
):
    counter_collection.find_one_and_update.return_value = {"value": 1}
    collection.insert_one.side_effect = DuplicateKeyError(
        "Private duplicate details", details={"keyPattern": {"name": 1}}
    )

    experiment_id = repository.allocate_experiment_id()
    with pytest.raises(RepositoryAlreadyExistsError):
        repository.create(
            experiment_id=experiment_id,
            name="example-experiment",
            artifact_location=f"s3://artifacts/{experiment_id}",
            lifecycle_stage="active",
            creation_timestamp=1000,
            tags={},
        )

    counter_collection.find_one_and_update.assert_called_once()
    counter_collection.update_one.assert_called_once()


def test_create_experiment_persists_allocated_id_and_artifact_location(repository, collection):
    experiment_id = repository.create(
        experiment_id=1,
        name="example-experiment",
        artifact_location="s3://default-artifacts/1",
        lifecycle_stage="active",
        creation_timestamp=1000,
        tags={},
    )

    assert experiment_id == "1"
    collection.insert_one.assert_called_once_with(
        {
            "_id": 1,
            "name": "example-experiment",
            "artifact_location": "s3://default-artifacts/1",
            "lifecycle_stage": "active",
            "creation_time": 1000,
            "last_update_time": 1000,
            "tags": [],
        }
    )


def test_default_experiment_does_not_increment_counter(repository, counter_collection, collection):
    repository.ensure_default_experiment(
        artifact_location="s3://artifacts/0", creation_timestamp=1000
    )

    counter_collection.find_one_and_update.assert_not_called()
    assert collection.update_one.call_args.args[0] == {"_id": 0}


def test_search_translates_partial_cursor_failure_and_closes_cursor(repository, collection):
    backend_error = OperationFailure("Private cursor details")

    def partial_results():
        yield {"_id": 42}
        raise backend_error

    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.__iter__.side_effect = partial_results
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
    cursor.close.assert_called_once_with()


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
    database.__getitem__.assert_has_calls([call("experiments"), call("experiment_id_counters")])
    collection.create_index.assert_called_once_with(
        [("name", ASCENDING)], unique=True, name="experiments_name_unique"
    )


def test_repository_initialization_translates_counter_initialization_failure():
    backend_error = OperationFailure("Private counter initialization details")
    experiment_collection = MagicMock()
    counter_collection = MagicMock()
    counter_collection.update_one.side_effect = backend_error
    database = MagicMock()
    database.__getitem__.side_effect = {
        "experiments": experiment_collection,
        "experiment_id_counters": counter_collection,
    }.__getitem__

    with pytest.raises(RepositoryPersistenceError) as caught:
        ExperimentRepository(database)

    assert caught.value.__cause__ is backend_error
    counter_collection.update_one.assert_called_once_with(
        {"_id": "experiments"},
        {"$setOnInsert": {"value": 0}},
        upsert=True,
    )


def test_record_conversion_errors_are_not_reported_as_database_failures(repository, collection):
    collection.find_one.return_value = {"_id": 42}

    with pytest.raises(KeyError, match="name"):
        repository.find_by_id(42)

    collection.find_one.assert_called_once_with({"_id": 42})


def test_record_conversion_preserves_numeric_experiment_id(repository, collection):
    collection.find_one.return_value = {
        "_id": 42,
        "name": "example-experiment",
        "artifact_location": "s3://artifacts/42",
        "lifecycle_stage": "active",
        "creation_time": 1000,
        "last_update_time": 1000,
        "tags": [],
    }

    record = repository.find_by_id(42)

    assert record.experiment_id == 42
    assert collection.find_one.call_args.args[0] == {"_id": 42}


def test_missing_experiment_is_returned_without_persistence_error(repository, collection):
    collection.find_one.return_value = None

    assert repository.find_by_id(99) is None

    collection.find_one.assert_called_once_with({"_id": 99})


def test_default_experiment_uses_insert_only_upsert(repository, collection):
    repository.ensure_default_experiment(
        artifact_location="s3://artifacts/0", creation_timestamp=1000
    )

    collection.update_one.assert_called_once_with(
        {"_id": 0},
        {
            "$setOnInsert": {
                "name": "Default",
                "artifact_location": "s3://artifacts/0",
                "lifecycle_stage": "active",
                "creation_time": 1000,
                "last_update_time": 1000,
                "tags": [],
            }
        },
        upsert=True,
    )


@pytest.mark.parametrize("existing_default", [None, {"_id": 0}])
def test_default_experiment_duplicate_only_succeeds_if_default_exists(
    repository, collection, existing_default
):
    duplicate = DuplicateKeyError("Private duplicate details")
    collection.update_one.side_effect = duplicate
    collection.find_one.return_value = existing_default

    if existing_default is None:
        with pytest.raises(RepositoryPersistenceError) as caught:
            repository.ensure_default_experiment(
                artifact_location="s3://artifacts/0", creation_timestamp=1000
            )
        assert caught.value.__cause__ is duplicate
    else:
        repository.ensure_default_experiment(
            artifact_location="s3://artifacts/0", creation_timestamp=1000
        )

    collection.find_one.assert_called_once_with({"_id": 0})


def test_default_experiment_translates_database_failure(repository, collection):
    backend_error = OperationFailure("Private initialization details")
    collection.update_one.side_effect = backend_error

    with pytest.raises(RepositoryPersistenceError) as caught:
        repository.ensure_default_experiment(
            artifact_location="s3://artifacts/0", creation_timestamp=1000
        )

    assert caught.value.__cause__ is backend_error
