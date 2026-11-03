"""Retry limits, backoff, and exception propagation."""

from unittest.mock import Mock, call

import pytest

from mlflow_mongodb.infrastructure.retry import retry_on_exception


@pytest.mark.parametrize(("attempts", "backoff"), [(0, ()), (-1, ()), (1, (0,)), (3, (-0.1,))])
def test_invalid_configuration_is_rejected(attempts, backoff):
    with pytest.raises(ValueError):
        retry_on_exception(ValueError, attempts=attempts, backoff_seconds=backoff)


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_success_returns_result_and_sleeps_only_between_failed_attempts(monkeypatch, failures):
    sleep = Mock()
    monkeypatch.setattr("mlflow_mongodb.infrastructure.retry.time.sleep", sleep)
    function = Mock(side_effect=[ValueError("retry")] * failures + ["result"])
    wrapped = retry_on_exception(ValueError, attempts=3, backoff_seconds=(0.1, 0.2))(function)

    assert wrapped("argument", flag=True) == "result"
    assert function.call_args_list == [call("argument", flag=True)] * (failures + 1)
    assert sleep.call_args_list == [call(delay) for delay in (0.1, 0.2)[:failures]]


def test_short_backoff_sequence_does_not_limit_attempts(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("mlflow_mongodb.infrastructure.retry.time.sleep", sleep)
    function = Mock(side_effect=[ValueError(), ValueError(), "done"])

    assert retry_on_exception(ValueError, attempts=3, backoff_seconds=(0.1,))(function)() == "done"
    assert function.call_count == 3
    sleep.assert_called_once_with(0.1)


def test_exhaustion_reraises_the_final_exception(monkeypatch):
    monkeypatch.setattr("mlflow_mongodb.infrastructure.retry.time.sleep", Mock())
    first, last = ValueError("first"), ValueError("last")
    function = Mock(side_effect=[first, last])

    with pytest.raises(ValueError) as caught:
        retry_on_exception(ValueError, attempts=2)(function)()

    assert caught.value is last
    assert function.call_count == 2


def test_exhaustion_factory_receives_last_error_and_preserves_cause():
    error = ValueError("last")
    replacement = RuntimeError("exhausted")
    factory = Mock(return_value=replacement)
    function = Mock(side_effect=error)

    with pytest.raises(RuntimeError) as caught:
        retry_on_exception(ValueError, attempts=2, on_exhausted=factory)(function)()

    assert caught.value is replacement
    assert caught.value.__cause__ is error
    factory.assert_called_once_with(error)


def test_unrelated_exception_propagates_without_retry_or_backoff(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("mlflow_mongodb.infrastructure.retry.time.sleep", sleep)
    error = TypeError("programming error")
    function = Mock(side_effect=error)
    factory = Mock()

    with pytest.raises(TypeError) as caught:
        retry_on_exception(ValueError, attempts=3, backoff_seconds=(1, 2), on_exhausted=factory)(
            function
        )()

    assert caught.value is error
    function.assert_called_once_with()
    sleep.assert_not_called()
    factory.assert_not_called()


def test_wrapper_preserves_function_metadata():
    def operation():
        """Operation documentation."""

    wrapped = retry_on_exception(ValueError, attempts=1)(operation)

    assert wrapped.__name__ == operation.__name__
    assert wrapped.__doc__ == operation.__doc__
    assert wrapped.__wrapped__ is operation
