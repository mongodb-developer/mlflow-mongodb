"""Shared search-filter validation and MongoDB value-condition construction."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from mlflow_mongodb.infrastructure.errors import (
    RepositoryInvalidAttributeError,
    RepositoryInvalidRegexError,
    RepositoryUnsupportedComparatorError,
    RepositoryUnsupportedFieldTypeError,
)

COMPARISON_OPERATORS = MappingProxyType(
    {"=": "$eq", "!=": "$ne", "<": "$lt", "<=": "$lte", ">": "$gt", ">=": "$gte"}
)


def build_value_condition(
    comparator: str,
    value: str | int | float | tuple[str, ...],
) -> str | int | float | tuple[str, ...] | dict[str, Any] | re.Pattern[str]:
    """Translate a supported comparator and value into a MongoDB condition."""
    if comparator == "=":
        return value
    if comparator in COMPARISON_OPERATORS:
        return {COMPARISON_OPERATORS[comparator]: value}
    if comparator in ("IN", "NOT IN"):
        return {"$in" if comparator == "IN" else "$nin": list(value)}
    if comparator in ("LIKE", "ILIKE"):
        return like_regex(str(value), comparator)
    raise ValueError(f"Unsupported comparator: {comparator}")


def like_regex(value: str, comparator: str) -> re.Pattern[str]:
    """Build the registry LIKE pattern, including its final-newline behavior."""
    pattern = re.escape(value).replace("%", ".*").replace("_", ".")
    if not value.startswith("%"):
        pattern = f"^{pattern}"
    if not value.endswith("%"):
        pattern = f"{pattern}$"
    flags = re.DOTALL | (re.IGNORECASE if comparator == "ILIKE" else 0)
    return re.compile(pattern, flags)


@dataclass(frozen=True)
class SearchFilterClause:
    """The common clause fields, without converting the parsed value."""

    field_type: str
    key: str
    comparator: str
    value: Any


class SearchFilterValidator(ABC):
    """Validate field/operator rules, required domain rules, then regex syntax."""

    def __init__(
        self,
        *,
        field_types: Collection[str],
        attribute_keys: Collection[str] | None = None,
        comparators: Mapping[str, Collection[str]] | None = None,
        field_comparators: Mapping[tuple[str, str], Collection[str]] | None = None,
        regex_comparators: Collection[str] = (),
        uppercase_comparators: bool = True,
    ):
        self._field_types = frozenset(field_types)
        self._attribute_keys = frozenset(attribute_keys) if attribute_keys is not None else None
        self._comparators = {
            field_type: frozenset(allowed) for field_type, allowed in (comparators or {}).items()
        }
        self._field_comparators = {
            field: frozenset(allowed) for field, allowed in (field_comparators or {}).items()
        }
        self._regex_comparators = frozenset(regex_comparators)
        self._uppercase_comparators = uppercase_comparators

    @abstractmethod
    def _validate_domain_rules(self, clause: SearchFilterClause) -> None:
        """Implement the domain validation required after field/operator validation."""
        raise NotImplementedError

    def validate(self, parsed: Mapping[str, Any]) -> SearchFilterClause:
        """Return normalized clause fields or a condition-specific shared error."""
        comparator = parsed["comparator"]
        clause = SearchFilterClause(
            parsed["type"],
            parsed["key"],
            comparator.upper() if self._uppercase_comparators else comparator,
            parsed["value"],
        )
        if clause.field_type not in self._field_types:
            raise RepositoryUnsupportedFieldTypeError(clause.field_type)
        if (
            clause.field_type == "attribute"
            and self._attribute_keys is not None
            and clause.key not in self._attribute_keys
        ):
            raise RepositoryInvalidAttributeError(clause.key)

        field = (clause.field_type, clause.key)
        allowed = self._field_comparators.get(field, self._comparators.get(clause.field_type))
        if allowed is not None and clause.comparator not in allowed:
            raise RepositoryUnsupportedComparatorError(
                clause.field_type,
                clause.key,
                clause.comparator,
                allowed,
                field_specific=field in self._field_comparators,
            )
        self._validate_domain_rules(clause)
        if clause.comparator in self._regex_comparators:
            try:
                re.compile(clause.value)
            except re.error as exc:
                raise RepositoryInvalidRegexError(
                    f"Invalid search filter regular expression: {exc}"
                ) from exc
        return clause


class ConfiguredSearchFilterValidator(SearchFilterValidator):
    """Supply the required domain hook through configured validation callbacks."""

    def __init__(
        self,
        *,
        rules: Sequence[Callable[[SearchFilterClause], None]] = (),
        **configuration: Any,
    ):
        super().__init__(**configuration)
        self._rules = tuple(rules)

    def _validate_domain_rules(self, clause: SearchFilterClause) -> None:
        for rule in self._rules:
            rule(clause)
