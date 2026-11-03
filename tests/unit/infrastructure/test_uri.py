"""URI redaction retains schemes without exposing connection details."""

import pytest

from mlflow_mongodb.infrastructure.errors import redact_uri


@pytest.mark.parametrize(
    "uri,expected",
    [
        (
            "mongodb://fake-user:fake-password@host/database?password=secret#token",
            "mongodb://***",
        ),
        (
            "mongodb+srv://fake-user:fake-password@cluster/database",
            "mongodb+srv://***",
        ),
        (
            "unsupported-mongodb://fake-user:fake-password@host/database",
            "unsupported-mongodb://***",
        ),
        ("https://host/path?token=secret", "https://***"),
        ("MONGODB://fake-user:fake-password@host", "mongodb://***"),
        ("mongodb://fake-user:fake%40password@host", "mongodb://***"),
        ("mongodb://fake-user:fake-password@first,second/database", "mongodb://***"),
    ],
)
def test_redact_uri_keeps_only_scheme(uri, expected):
    assert redact_uri(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        None,
        "",
        "fake-user:fake-password@host/database",
        "fake-user@host/database",
        "mongodb://fake-user:fake-password@[broken/database",
        "mongodb:///fake-user:fake-password@host",
        "mongodb://",
        " mongo\ndb://fake-user:fake-password@host",
    ],
)
def test_redact_uri_invalid_input_returns_fixed_placeholder(uri):
    assert redact_uri(uri) == "<invalid or redacted URI>"


def test_redact_uri_handles_scheme_starting_with_digit():
    uri = "1mongodb://fake-user:fake-password@host"

    # Parser behavior varies by Python version; both outcomes hide connection details.
    assert redact_uri(uri) in {
        "1mongodb://***",
        "<invalid or redacted URI>",
    }
