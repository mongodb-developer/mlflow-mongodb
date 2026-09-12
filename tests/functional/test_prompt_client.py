"""MLflow client smoke test for the MongoDB model-registry plugin."""

from mlflow import MlflowClient

PROMPT_NAME = "mongodb-client-functional-prompt"
PROMPT_TEMPLATE = "Summarize {{document}} in three sentences."


def test_mlflow_client_prompt_lifecycle(
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
