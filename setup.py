from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent


setup(
    name="mlflow-mongodb",
    version="0.1.0",
    description="MongoDB model registry store plugin for MLflow",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/WaVEV/mlflow-mongodb",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.10",
    install_requires=[
        # Temporary workaround for https://github.com/mlflow/mlflow/pull/25559.
        # Remove once the upstream WSGI code is fixed or MLflow lifts the version block.
        "anyio>=3.6.2,!=4.15.0,<5",
        "mlflow>=3.1.0",
        "pymongo[srv]>=4.0.0,<5",
    ],
    extras_require={
        "dev": [
            "check-jsonschema==0.37.4",
            "pre-commit==4.6.1",
            "pytest>=8.0.0",
            "pytest-cov",
            "ruff==0.16.0",
        ],
    },
    entry_points={
        "mlflow.model_registry_store": [
            "mongodb=mlflow_mongodb.model_registry_store:MongoDBModelRegistryStore",
            "mongodb+srv=mlflow_mongodb.model_registry_store:MongoDBModelRegistryStore",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
