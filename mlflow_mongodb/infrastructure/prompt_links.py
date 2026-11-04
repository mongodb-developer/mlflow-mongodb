"""Resolve caller-supplied prompt references before writing associations."""

from collections.abc import Callable, Sequence

from mlflow.entities.model_registry import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracing.constant import TraceTagKey

LINKED_PROMPTS_TAG_KEY = getattr(TraceTagKey, "LINKED_PROMPTS", "mlflow.linkedPrompts")


def resolve_prompt_version(
    name: str,
    version: str,
    get_prompt_version: Callable[[str, str], PromptVersion | None],
) -> PromptVersion:
    """Load a prompt version or report a missing public reference."""
    stored = get_prompt_version(name, version)
    if stored is None:
        raise MlflowException(
            f"Prompt version (name={name}, version={version}) not found.",
            RESOURCE_DOES_NOT_EXIST,
        )
    return stored


def resolve_prompt_versions(
    prompt_versions: Sequence[PromptVersion],
    get_prompt_version: Callable[[str, str], PromptVersion | None],
) -> list[PromptVersion]:
    """Load each distinct prompt version and reject unresolved references."""
    resolved = {}
    for prompt in prompt_versions:
        identity = (prompt.name, str(prompt.version))
        if identity in resolved:
            continue
        resolved[identity] = resolve_prompt_version(*identity, get_prompt_version)
    return list(resolved.values())
