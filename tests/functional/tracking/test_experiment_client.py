from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from mlflow_mongodb.tracking.store import MongoDBTrackingStore


def test_experiment_lifecycle_through_mlflow_client(store: MongoDBTrackingStore):
    client = MlflowClient(tracking_uri=store.store_uri)
    experiment_id = client.create_experiment(
        "client-experiment",
        artifact_location="s3://experiment-artifacts/client",
        tags={"team": "risk"},
    )

    experiment = client.get_experiment(experiment_id)
    assert experiment.name == "client-experiment"
    assert experiment.tags == {"team": "risk"}
    assert experiment.artifact_location == "s3://experiment-artifacts/client"
    assert store.get_experiment(experiment_id).name == "client-experiment"
    assert client.get_experiment_by_name("client-experiment").experiment_id == experiment_id

    client.rename_experiment(experiment_id, "client-renamed")
    assert client.get_experiment_by_name("client-experiment") is None
    assert client.get_experiment_by_name("client-renamed").experiment_id == experiment_id

    client.delete_experiment(experiment_id)
    assert client.get_experiment(experiment_id).lifecycle_stage == "deleted"
    assert [experiment.experiment_id for experiment in client.search_experiments()] == ["0"]
    assert [
        experiment.experiment_id
        for experiment in client.search_experiments(view_type=ViewType.DELETED_ONLY)
    ] == [experiment_id]

    client.restore_experiment(experiment_id)
    assert client.get_experiment(experiment_id).lifecycle_stage == "active"
    assert [experiment.experiment_id for experiment in client.search_experiments()] == [
        experiment_id,
        "0",
    ]


def test_experiment_tag_lifecycle_through_mlflow_client(store: MongoDBTrackingStore):
    client = MlflowClient(tracking_uri=store.store_uri)
    experiment_id = client.create_experiment("client-tags", tags={"keep": "yes"})
    collection = store._database[store._settings.experiments_collection_name]

    client.set_experiment_tag(experiment_id, "team.name", "old")

    assert client.get_experiment(experiment_id).tags == {"keep": "yes", "team.name": "old"}
    assert collection.find_one({"_id": experiment_id})["tags"] == [
        {"k": "keep", "v": "yes"},
        {"k": "team.name", "v": "old"},
    ]

    client.set_experiment_tag(experiment_id, "team.name", "$new")

    assert client.get_experiment(experiment_id).tags == {"keep": "yes", "team.name": "$new"}
    assert collection.find_one({"_id": experiment_id})["tags"] == [
        {"k": "keep", "v": "yes"},
        {"k": "team.name", "v": "$new"},
    ]
    assert client.search_experiments(filter_string="tags.`team.name` = 'old'") == []
    assert [
        experiment.experiment_id
        for experiment in client.search_experiments(filter_string="tags.`team.name` = '$new'")
    ] == [experiment_id]

    client.delete_experiment_tag(experiment_id, "team.name")

    assert client.get_experiment(experiment_id).tags == {"keep": "yes"}
    assert collection.find_one({"_id": experiment_id})["tags"] == [{"k": "keep", "v": "yes"}]
    assert client.search_experiments(filter_string="tags.`team.name` = '$new'") == []
