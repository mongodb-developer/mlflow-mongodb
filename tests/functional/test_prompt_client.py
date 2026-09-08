"""MLflow client smoke test for the MongoDB model-registry plugin."""

import threading

from mlflow import MlflowClient

from mlflow_mongodb import MongoDBModelRegistryStore

PROMPT_NAME = "mongodb-client-functional-prompt"
PROMPT_TEMPLATE = "Summarize {{document}} in three sentences."


def _wait_for_prompt_linking_threads() -> None:
    for thread in threading.enumerate():
        if thread.name.startswith("link_prompt_to_experiment_thread"):
            thread.join(timeout=5)
            if thread.is_alive():
                raise TimeoutError(f"Thread {thread.name} did not complete within 5 seconds")


def test_mlflow_client_prompt_lifecycle(
    store: MongoDBModelRegistryStore,
    mongodb_uri: str,
    tmp_path,
):
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=mongodb_uri)

    created = client.register_prompt(
        name=PROMPT_NAME,
        template=PROMPT_TEMPLATE,
        commit_message="Initial version",
        tags={"owner": "platform"},
    )
    _wait_for_prompt_linking_threads()

    assert created.name == PROMPT_NAME
    assert created.version == 1
    assert created.template == PROMPT_TEMPLATE
    assert created.commit_message == "Initial version"

    stored = client.get_prompt_version(PROMPT_NAME, 1)
    assert stored is not None
    assert stored.template == PROMPT_TEMPLATE
    assert stored.tags == {"owner": "platform"}

    client.set_prompt_alias(PROMPT_NAME, "production", 1)
    assert client.get_prompt_version(PROMPT_NAME, "production").version == 1

    client.delete_prompt_alias(PROMPT_NAME, "production")
    client.delete_prompt_version(PROMPT_NAME, "1")
    client.delete_prompt(PROMPT_NAME)
    assert client.get_prompt(PROMPT_NAME) is None
