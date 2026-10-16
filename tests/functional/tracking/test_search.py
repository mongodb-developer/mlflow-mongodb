import pytest
from mlflow.entities import ExperimentTag, ViewType
from mlflow.utils.search_utils import SearchExperimentsUtils

from mlflow_mongodb.tracking.repositories.experiments import ExperimentFilter, ExperimentOrder
from mlflow_mongodb.tracking.store import MongoDBTrackingStore


@pytest.fixture
def search_experiments(store: MongoDBTrackingStore, clock):
    experiment_ids = {}
    for timestamp, name, tags in (
        (1000, "RiskA", [ExperimentTag("team", "risk"), ExperimentTag("purpose", "evaluation")]),
        (2000, "RiskB", [ExperimentTag("team", "platform"), ExperimentTag("purpose", "risk")]),
        (3000, "Other", []),
    ):
        clock["now"] = timestamp
        experiment_ids[name] = store.create_experiment(name, tags=tags)
    return experiment_ids


@pytest.mark.parametrize(("field_type", "key"), [("attribute", "name"), ("tag", "team")])
def test_experiment_repository_rejects_unsupported_comparator(
    store: MongoDBTrackingStore, field_type, key
):
    store.create_experiment("unsupported-comparator", tags=[ExperimentTag("team", "risk")])

    with pytest.raises(ValueError, match="^Unsupported experiment comparator: RLIKE$"):
        store._experiment_repository.search(
            lifecycle_stages=["active"],
            filters=(ExperimentFilter(field_type, key, "RLIKE", "risk"),),
            order_by=(ExperimentOrder("creation_time", False),),
            offset=0,
            limit=10,
        )


@pytest.mark.parametrize(
    ("filter_string", "expected_names"),
    [
        (None, {"RiskA", "RiskB", "Other"}),
        ("", {"RiskA", "RiskB", "Other"}),
        ("name = 'RiskA'", {"RiskA"}),
        ("name != 'RiskA'", {"RiskB", "Other"}),
        ("name LIKE 'Risk%'", {"RiskA", "RiskB"}),
        ("name LIKE 'risk%'", set()),
        ("name ILIKE 'risk%'", {"RiskA", "RiskB"}),
        ("name LIKE 'Risk_'", {"RiskA", "RiskB"}),
        ("tags.team = 'risk'", {"RiskA"}),
        ("tags.team != 'risk'", {"RiskB"}),
        ("tags.team ILIKE 'RIS%'", {"RiskA"}),
        ("tags.Team = 'risk'", set()),
        ("tags.team = 'risk' AND tags.purpose = 'risk'", set()),
        ("tags.team LIKE 'r%' AND tags.team != 'risk'", set()),
        ("creation_time >= 2000", {"RiskB", "Other"}),
        ("last_update_time < 2000", {"RiskA"}),
    ],
)
def test_search_experiment_attribute_and_tag_filters(
    store: MongoDBTrackingStore, search_experiments, filter_string, expected_names
):
    results = store.search_experiments(filter_string=filter_string)

    assert {experiment.name for experiment in results} == expected_names
    assert {experiment.experiment_id for experiment in results} == {
        search_experiments[name] for name in expected_names
    }


@pytest.fixture
def tag_existence_experiments(store: MongoDBTrackingStore):
    experiment_ids = {
        name: store.create_experiment(name, tags=tags)
        for name, tags in (
            ("missing-tags", []),
            ("empty-tags", []),
            ("unrelated-tag", [ExperimentTag("purpose", "team")]),
            ("present-tag", [ExperimentTag("team", "risk")]),
            ("empty-value", [ExperimentTag("team", "")]),
            ("null-value", [ExperimentTag("team", "risk")]),
        )
    }
    collection = store._database[store._settings.experiments_collection_name]
    missing_tags = collection.update_one(
        {"_id": experiment_ids["missing-tags"]}, {"$unset": {"tags": ""}}
    )
    assert missing_tags.modified_count == 1
    null_value = collection.update_one(
        {"_id": experiment_ids["null-value"]}, {"$set": {"tags": [{"k": "team", "v": None}]}}
    )
    assert null_value.modified_count == 1
    return experiment_ids


@pytest.mark.parametrize(
    ("comparator", "expected_names"),
    [
        ("IS NULL", {"missing-tags", "empty-tags", "unrelated-tag"}),
        ("IS NOT NULL", {"present-tag", "empty-value", "null-value"}),
    ],
)
def test_repository_search_tag_existence_filters(
    store: MongoDBTrackingStore, tag_existence_experiments, comparator, expected_names
):
    records = store._experiment_repository.search(
        lifecycle_stages=["active"],
        filters=(ExperimentFilter("tag", "team", comparator, None),),
        order_by=(ExperimentOrder("creation_time", False),),
        offset=0,
        limit=10,
    )

    assert {record.name for record in records} == expected_names
    assert {record.experiment_id for record in records} == {
        tag_existence_experiments[name] for name in expected_names
    }


@pytest.mark.skipif(
    "IS NULL" not in SearchExperimentsUtils.VALID_TAG_COMPARATORS,
    reason="The installed MLflow version does not support experiment tag existence filters",
)
@pytest.mark.parametrize(
    ("comparator", "expected_names"),
    [
        ("IS NULL", {"missing-tags", "empty-tags", "unrelated-tag"}),
        ("IS NOT NULL", {"present-tag", "empty-value", "null-value"}),
    ],
)
def test_search_experiment_tag_existence_filters(
    store: MongoDBTrackingStore, tag_existence_experiments, comparator, expected_names
):
    experiments = store.search_experiments(filter_string=f"tags.team {comparator}")

    assert {experiment.name for experiment in experiments} == expected_names
    assert {experiment.experiment_id for experiment in experiments} == {
        tag_existence_experiments[name] for name in expected_names
    }


@pytest.mark.parametrize(
    ("view_type", "expected_names"),
    [
        (ViewType.ACTIVE_ONLY, {"RiskB", "Other"}),
        (ViewType.DELETED_ONLY, {"RiskA"}),
        (ViewType.ALL, {"RiskA", "RiskB", "Other"}),
    ],
)
def test_search_experiment_lifecycle_views(
    store: MongoDBTrackingStore, search_experiments, view_type, expected_names
):
    store.delete_experiment(search_experiments["RiskA"])

    results = store.search_experiments(view_type=view_type)

    assert {experiment.name for experiment in results} == expected_names


@pytest.mark.parametrize(
    ("order_by", "expected_names"),
    [
        (None, ["Other", "RiskB", "RiskA"]),
        (["creation_time ASC"], ["RiskA", "RiskB", "Other"]),
        (["name DESC"], ["RiskB", "RiskA", "Other"]),
        (["last_update_time ASC"], ["RiskA", "RiskB", "Other"]),
    ],
)
@pytest.mark.usefixtures("search_experiments")
def test_search_experiment_ordering(store: MongoDBTrackingStore, order_by, expected_names):
    results = store.search_experiments(order_by=order_by)

    assert [experiment.name for experiment in results] == expected_names


@pytest.mark.parametrize("page_size", [1, 2, 3, 10])
def test_search_experiments_paginates_timestamp_ties_without_duplicates(
    store: MongoDBTrackingStore, clock, page_size
):
    clock["now"] = 1000
    experiment_ids = [store.create_experiment(f"experiment-{index}") for index in range(7)]
    expected_ids = sorted(experiment_ids)
    returned_ids = []
    token = None

    for _ in range(len(experiment_ids) + 1):
        page = store.search_experiments(max_results=page_size, page_token=token)
        assert len(page) <= page_size
        returned_ids.extend(experiment.experiment_id for experiment in page)
        token = page.token
        if token is None:
            break
    else:
        pytest.fail("Experiment pagination did not terminate")

    assert returned_ids == expected_ids
    assert len(set(returned_ids)) == len(experiment_ids)
    assert token is None


def test_search_experiments_reflects_rename_and_updated_timestamp(
    store: MongoDBTrackingStore, clock, search_experiments
):
    clock["now"] = 4000
    store.rename_experiment(search_experiments["RiskA"], "Renamed")

    results = store.search_experiments(
        filter_string="name = 'Renamed' AND last_update_time >= 4000"
    )

    assert [experiment.experiment_id for experiment in results] == [search_experiments["RiskA"]]
    assert results[0].tags == {"team": "risk", "purpose": "evaluation"}
    assert store.search_experiments(filter_string="name = 'RiskA'") == []
