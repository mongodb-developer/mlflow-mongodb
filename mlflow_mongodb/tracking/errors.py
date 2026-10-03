"""Tracking domain exceptions and repository persistence errors."""


class RepositoryPersistenceError(Exception):
    """Raised when a repository database operation fails."""


class RepositoryAlreadyExistsError(Exception):
    """Raised when a resource with the requested unique identity already exists."""


class RepositoryNotFoundError(Exception):
    """Raised when a resource is not stored in the expected lifecycle stage."""


class RepositoryNotActiveError(Exception):
    """Raised when an operation requires an active resource."""
