"""Internal MongoDB update builders for repository documents."""

from collections.abc import Mapping
from typing import Any


def build_replace_array_element_pipeline(
    *,
    array_field: str,
    key_field: str,
    key: str,
    element: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a pipeline that replaces array elements matching a key.

    The pipeline treats a missing or null array as empty, removes all existing
    elements whose ``key_field`` equals ``key``, and appends ``element``.
    Caller-provided values are wrapped with ``$literal`` so they are treated
    as data rather than aggregation expressions.
    """
    return [
        {
            "$set": {
                # Rebuild the array so replacing a key cannot create a
                # duplicate entry, while preserving unrelated elements.
                array_field: {
                    "$concatArrays": [
                        {
                            "$filter": {
                                "input": {"$ifNull": [f"${array_field}", []]},
                                "as": "stored_element",
                                "cond": {
                                    "$ne": [
                                        f"$$stored_element.{key_field}",
                                        {"$literal": key},
                                    ]
                                },
                            }
                        },
                        {"$literal": [dict(element)]},
                    ]
                }
            }
        }
    ]


def build_remove_array_element_update(
    *,
    array_field: str,
    key_field: str,
    key: str,
) -> dict[str, Any]:
    """Build a MongoDB ``$pull`` update that removes matching array elements."""
    return {"$pull": {array_field: {key_field: key}}}
