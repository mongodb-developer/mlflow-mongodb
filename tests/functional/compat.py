"""Installed MLflow capabilities used by functional tests."""

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException

try:
    Feedback(value=None)
except MlflowException:
    SUPPORTS_NULL_FEEDBACK = False
else:
    SUPPORTS_NULL_FEEDBACK = True
