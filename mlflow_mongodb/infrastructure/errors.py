"""Shared infrastructure failures translated at the store boundary."""

from collections.abc import Collection


class RepositoryAlreadyExistsError(Exception):
    """Raised when a resource with the requested unique identity already exists."""


class RepositoryNotFoundError(Exception):
    """Raised when a resource is not stored in the expected lifecycle stage."""


class RepositoryNotActiveError(Exception):
    """Raised when an operation requires an active resource."""


class RepositoryPersistenceError(Exception):
    """Raised when a database operation cannot be completed."""


class RepositoryUnsupportedFieldTypeError(Exception):
    """Raised when a filter targets an unsupported field type."""

    def __init__(self, field_type: str):
        self.field_type = field_type
        super().__init__(f"Unsupported search field type: {field_type}")


class RepositoryInvalidAttributeError(Exception):
    """Raised when a filter targets an unknown attribute."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Invalid search attribute: {key}")


class RepositoryUnsupportedComparatorError(Exception):
    """Raised when a comparator is unsupported for a filter field."""

    def __init__(
        self,
        field_type: str,
        key: str,
        comparator: str,
        allowed: Collection[str],
        *,
        field_specific: bool = False,
    ):
        self.field_type = field_type
        self.key = key
        self.comparator = comparator
        self.allowed = tuple(sorted(allowed))
        self.field_specific = field_specific
        super().__init__(f"Unsupported comparator {comparator} for {field_type}.{key}")


class RepositoryInvalidRegexError(Exception):
    """Raised when a filter contains an invalid regular expression."""
