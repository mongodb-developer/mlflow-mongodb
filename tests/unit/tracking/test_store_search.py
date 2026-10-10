import logging

import pytest
from mlflow.entities import Experiment, LifecycleStage, ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils

import mlflow_mongodb.tracking.store as tracking_store
from mlflow_mongodb.infrastructure.errors import (
    RepositoryInvalidAttributeError,
    RepositoryUnsupportedComparatorError,
    RepositoryUnsupportedFieldTypeError,
)
from mlflow_mongodb.tracking.repositories.experiments import ExperimentFilter, ExperimentOrder


@pytest.mark.parametrize(
    ("filter_string", "expected_filter"),
    [
        ("name LIKE 'risk%'", ExperimentFilter("attribute", "name", "LIKE", "risk%")),
        ("tags.team ILIKE 'Plat%'", ExperimentFilter("tag", "team", "ILIKE", "Plat%")),
        ("creation_time >= 1000", ExperimentFilter("attribute", "creation_time", ">=", 1000.0)),
        ("last_update_time < 2000", ExperimentFilter("attribute", "last_update_time", "<", 2000.0)),
        ("tags.`team.name` = 'risk'", ExperimentFilter("tag", "team.name", "=", "risk")),
    ],
)
def test_parse_experiment_filters(store, filter_string, expected_filter):
    assert store._parse_experiment_filters(filter_string) == (expected_filter,)


@pytest.mark.skipif(
    "IS NULL" not in SearchExperimentsUtils.VALID_TAG_COMPARATORS,
    reason="The installed MLflow version does not support experiment tag existence filters",
)
@pytest.mark.parametrize("comparator", ["IS NULL", "IS NOT NULL"])
def test_search_passes_tag_existence_filters_to_repository(
    store, experiment_repository, comparator
):
    experiment_repository.search.return_value = []

    page = store.search_experiments(filter_string=f"tags.team {comparator}", max_results=2)

    assert page == []
    assert page.token is None
    experiment_repository.search.assert_called_once_with(
        lifecycle_stages=[LifecycleStage.ACTIVE],
        filters=(ExperimentFilter("tag", "team", comparator, None),),
        order_by=(ExperimentOrder("creation_time", False), ExperimentOrder("experiment_id", True)),
        offset=0,
        limit=3,
    )


@pytest.mark.parametrize("filter_string", [None, ""])
def test_parse_empty_experiment_filters(store, filter_string):
    assert store._parse_experiment_filters(filter_string) == ()


@pytest.mark.parametrize(
    "filter_string",
    [
        "name > 'risk'",
        "tags.team > 'risk'",
        "creation_time LIKE '1000'",
        "unknown = 'risk'",
        "name = unquoted",
    ],
)
def test_search_rejects_invalid_filters_before_persistence(
    store, experiment_repository, filter_string
):
    with pytest.raises(MlflowException) as caught:
        store.search_experiments(filter_string=filter_string)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.search.assert_not_called()


def test_search_translates_and_logs_unsupported_filter_field_type(
    store, experiment_repository, monkeypatch, caplog
):
    monkeypatch.setattr(
        SearchExperimentsUtils,
        "parse_search_filter",
        lambda _filter_string: [
            {"type": "unexpected", "key": "name", "comparator": "=", "value": "risk"}
        ],
    )

    with pytest.raises(MlflowException) as caught:
        store.search_experiments(filter_string="name = 'risk'")

    assert caught.value.message == "Invalid token type: unexpected"
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    validation_error = caught.value.__context__
    assert isinstance(validation_error, RepositoryUnsupportedFieldTypeError)
    assert validation_error.field_type == "unexpected"
    experiment_repository.search.assert_not_called()

    records = [record for record in caplog.records if record.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].msg == "Unable to validate experiment filter: %s"
    assert records[0].args == (validation_error,)
    assert records[0].exc_info is None


def test_search_translates_and_logs_invalid_filter_attribute(
    store, experiment_repository, monkeypatch, caplog
):
    monkeypatch.setattr(
        SearchExperimentsUtils,
        "parse_search_filter",
        lambda _filter_string: [
            {"type": "attribute", "key": "unknown", "comparator": "=", "value": "risk"}
        ],
    )

    with pytest.raises(MlflowException) as caught:
        store.search_experiments(filter_string="unknown = 'risk'")

    assert caught.value.message == "Invalid attribute name: unknown"
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    validation_error = caught.value.__context__
    assert isinstance(validation_error, RepositoryInvalidAttributeError)
    assert validation_error.key == "unknown"
    experiment_repository.search.assert_not_called()

    records = [record for record in caplog.records if record.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].msg == "Unable to validate experiment filter: %s"
    assert records[0].args == (validation_error,)
    assert records[0].exc_info is None


@pytest.mark.parametrize("attribute", ["creation_time", "last_update_time"])
def test_search_translates_and_logs_invalid_numeric_filter_comparator(
    store, experiment_repository, monkeypatch, caplog, attribute
):
    monkeypatch.setattr(
        SearchExperimentsUtils,
        "parse_search_filter",
        lambda _filter_string: [
            {"type": "attribute", "key": attribute, "comparator": "LIKE", "value": "1000"}
        ],
    )

    with pytest.raises(MlflowException) as caught:
        store.search_experiments(filter_string=f"{attribute} LIKE '1000'")

    assert caught.value.message == (
        "Invalid comparator 'LIKE' not one of "
        f"'{SearchExperimentsUtils.VALID_STRING_ATTRIBUTE_COMPARATORS}"
    )
    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    validation_error = caught.value.__context__
    assert isinstance(validation_error, RepositoryUnsupportedComparatorError)
    assert validation_error.field_type == "attribute"
    assert validation_error.key == attribute
    assert validation_error.comparator == "LIKE"
    assert validation_error.field_specific is True
    experiment_repository.search.assert_not_called()

    records = [record for record in caplog.records if record.name == tracking_store.logger.name]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].msg == "Unable to validate experiment filter: %s"
    assert records[0].args == (validation_error,)
    assert records[0].exc_info is None


@pytest.mark.parametrize(
    ("order_by", "expected"),
    [
        (None, (ExperimentOrder("creation_time", False), ExperimentOrder("experiment_id", True))),
        ([], (ExperimentOrder("creation_time", False), ExperimentOrder("experiment_id", True))),
        (["name ASC"], (ExperimentOrder("name", True), ExperimentOrder("experiment_id", False))),
        (["experiment_id ASC"], (ExperimentOrder("experiment_id", True),)),
    ],
)
def test_parse_experiment_order_adds_deterministic_tiebreaker(store, order_by, expected):
    assert store._parse_experiment_order(order_by) == expected


@pytest.mark.parametrize("order_by", [["tags.team ASC"], ["unknown ASC"], ["name DESC extra"]])
def test_search_rejects_invalid_order_before_persistence(store, experiment_repository, order_by):
    with pytest.raises(MlflowException) as caught:
        store.search_experiments(order_by=order_by)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.search.assert_not_called()


@pytest.mark.parametrize(
    "max_results", [True, False, 0, -1, 1.5, "1", SEARCH_MAX_RESULTS_THRESHOLD + 1]
)
def test_search_validates_page_size_before_persistence(store, experiment_repository, max_results):
    with pytest.raises(MlflowException) as caught:
        store.search_experiments(max_results=max_results)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.search.assert_not_called()


@pytest.mark.parametrize("page_token", ["not-a-page-token", SearchUtils.create_page_token(-1)])
def test_search_validates_page_token_before_persistence(store, experiment_repository, page_token):
    with pytest.raises(MlflowException) as caught:
        store.search_experiments(page_token=page_token)

    assert caught.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    experiment_repository.search.assert_not_called()


@pytest.mark.parametrize(
    ("view_type", "lifecycle_stages"),
    [
        (ViewType.ACTIVE_ONLY, [LifecycleStage.ACTIVE]),
        (ViewType.DELETED_ONLY, [LifecycleStage.DELETED]),
        (ViewType.ALL, [LifecycleStage.ACTIVE, LifecycleStage.DELETED]),
    ],
)
@pytest.mark.parametrize("has_more", [False, True])
def test_search_passes_criteria_and_returns_paged_entities(
    store, experiment_repository, experiment_record_factory, view_type, lifecycle_stages, has_more
):
    records = [
        experiment_record_factory(
            experiment_id=str(index),
            name=f"risk-{index}",
            lifecycle_stage=lifecycle_stages[0],
            tags={"team": "risk"},
        )
        for index in range(3 if has_more else 2)
    ]
    experiment_repository.search.return_value = records

    page = store.search_experiments(
        view_type=view_type,
        max_results=2,
        filter_string="name LIKE 'risk%' AND tags.team = 'risk'",
        order_by=["name ASC"],
        page_token=SearchUtils.create_page_token(4),
    )

    assert isinstance(page, PagedList)
    assert all(isinstance(experiment, Experiment) for experiment in page)
    assert [experiment.experiment_id for experiment in page] == ["0", "1"]
    assert page[0].tags == {"team": "risk"}
    assert page[0].lifecycle_stage == lifecycle_stages[0]
    assert page[0].artifact_location == records[0].artifact_location
    assert page[0].creation_time == records[0].creation_time
    assert page[0].last_update_time == records[0].last_update_time
    assert page.token == (SearchUtils.create_page_token(6) if has_more else None)
    experiment_repository.search.assert_called_once_with(
        lifecycle_stages=lifecycle_stages,
        filters=(
            ExperimentFilter("attribute", "name", "LIKE", "risk%"),
            ExperimentFilter("tag", "team", "=", "risk"),
        ),
        order_by=(ExperimentOrder("name", True), ExperimentOrder("experiment_id", False)),
        offset=4,
        limit=3,
    )


def test_search_returns_empty_page(store, experiment_repository):
    experiment_repository.search.return_value = []

    page = store.search_experiments(max_results=2)

    assert page == []
    assert page.token is None
    experiment_repository.search.assert_called_once_with(
        lifecycle_stages=[LifecycleStage.ACTIVE],
        filters=(),
        order_by=(ExperimentOrder("creation_time", False), ExperimentOrder("experiment_id", True)),
        offset=0,
        limit=3,
    )
