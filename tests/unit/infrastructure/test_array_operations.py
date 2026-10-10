"""Unit tests for MongoDB array update construction."""

import pytest

from mlflow_mongodb.infrastructure.array_operations import (
    build_remove_array_element_update,
    build_replace_array_element_pipeline,
)


@pytest.mark.parametrize(("array_field", "key_field"), [("tags", "k"), ("aliases", "alias")])
def test_replace_pipeline_treats_expression_like_values_as_literal_data(array_field, key_field):
    key = "$name"
    element = {key_field: key, "v": {"$concat": ["$secret", "suffix"]}}

    pipeline = build_replace_array_element_pipeline(
        array_field=array_field, key_field=key_field, element=element
    )

    filtered, appended = pipeline[0]["$set"][array_field]["$concatArrays"]
    assert filtered["$filter"]["input"] == {"$ifNull": [f"${array_field}", []]}
    assert filtered["$filter"]["cond"] == {
        "$not": [{"$in": [f"$$stored.{key_field}", {"$literal": [key]}]}]
    }
    assert appended == {"$literal": [element]}


def test_replace_pipeline_copies_element_without_mutating_input():
    element = {"k": "team", "v": "platform"}

    pipeline = build_replace_array_element_pipeline(
        array_field="tags", key_field="k", element=element
    )
    element["v"] = "changed"

    appended = pipeline[0]["$set"]["tags"]["$concatArrays"][1]["$literal"]
    assert appended == [{"k": "team", "v": "platform"}]


@pytest.mark.parametrize(("array_field", "key_field"), [("tags", "k"), ("aliases", "alias")])
def test_remove_update_targets_only_matching_key(array_field, key_field):
    assert build_remove_array_element_update(
        array_field=array_field, key_field=key_field, key="$name"
    ) == {"$pull": {array_field: {key_field: "$name"}}}
