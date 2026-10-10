"""Internal MongoDB update builders for repository documents."""

from collections.abc import Mapping
from typing import Any


def build_merge_array_expression(
    field: str,
    records: list[Mapping[str, Any]],
    identity: str,
) -> dict[str, Any]:
    """Build a MongoDB expression that merges embedded records by identity."""
    incoming = {"$literal": [dict(record) for record in records]}
    keys = {"$literal": [record[identity] for record in records]}
    remaining_records = {
        "$filter": {
            "input": {"$ifNull": [field, []]},
            "as": "stored",
            "cond": {"$not": [{"$in": [f"$$stored.{identity}", keys]}]},
        }
    }
    return {"$concatArrays": [remaining_records, incoming]}


def build_replace_array_element_pipeline(
    *,
    array_field: str,
    key_field: str,
    element: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a pipeline that replaces an array element by its identity field.

    The identity value is read from ``element[key_field]``. The pipeline treats
    a missing or null array as empty, removes all existing elements with the
    same identity value, and appends ``element``. Caller-provided values are
    wrapped with ``$literal`` so they are treated as data rather than
    aggregation expressions.
    """
    return [
        {
            "$set": {
                array_field: build_merge_array_expression(f"${array_field}", [element], key_field)
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
