# MongoDB Backend for MLflow

> [WARNING]
> This project is under active development and has not reached a stable release. APIs, storage
> formats, and configuration may change at any time without notice. **It is not intended for
> production use.** See [LICENSE](LICENSE) for the full disclaimer of warranties.

An MLflow plugin that stores tracking data (experiments, runs, traces, and logged models) and
model registry data (registered models, versions, aliases, tags, and prompts) in MongoDB. The
tracking store and registry store can be configured independently, so each can point at MongoDB,
another supported backend, or a mix of the two.

## Installation

Python 3.10 or newer, MLflow 3.5 or newer within the 3.x series, and a MongoDB deployment
are required.

```bash
python -m pip install "mlflow-mongodb>=0.1.0.dev0"
```

## Running MLflow

The tracking store and registered-model store are configured independently, and each can use
MongoDB or any other MLflow-supported backend. For example, the following configuration keeps
experiments and runs in SQLite while storing registered models and their versions in MongoDB:

```bash
export TRACKING_STORE_URI="sqlite:///tracking.db"
export REGISTRY_STORE_URI="mongodb://localhost:27017/mlflow_registry"
export MLFLOW_BIND_HOST="127.0.0.1"
export MLFLOW_BIND_PORT="5000"

mlflow server \
  --backend-store-uri "${TRACKING_STORE_URI}" \
  --registry-store-uri "${REGISTRY_STORE_URI}" \
  --host "${MLFLOW_BIND_HOST}" \
  --port "${MLFLOW_BIND_PORT}"
```

To store tracking data in MongoDB as well, point `TRACKING_STORE_URI` at a MongoDB URI instead:

```bash
export TRACKING_STORE_URI="mongodb://localhost:27017/mlflow_tracking"
```

Open `http://127.0.0.1:5000` to use the MLflow UI. Clients can connect to the server with:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

The MongoDB URI must include the database name; `mlflow_registry` is the database name in
the example above. Authentication, replica-set, TLS, and other connection options can be
provided using standard MongoDB URI syntax. For example:

```text
mongodb://username:password@mongo.example.com:27017/mlflow_registry?authSource=admin
```

When using the MongoDB tracking backend, MLflow 3.5 clients upload full trace payloads to
MLflow's existing artifact storage. Configure a reachable experiment artifact location,
such as S3 or an MLflow artifact proxy. The MongoDB store returns the trace artifact URI;
it does not implement its own artifact provider or upload the payload itself.

On MLflow 3.5 servers, trace retrieval uses artifacts while spans are also persisted in
MongoDB. On MLflow 3.6 and newer servers, modern clients use MongoDB-backed trace retrieval.
A 3.5 client cannot read a trace written only to MongoDB by a modern client unless an
artifact payload also exists. Cross-version reads of that kind are not guaranteed.

Aggregated trace metric queries require MLflow 3.9 or newer; available metrics and dimensions
depend on the installed MLflow version. Older versions preserve native MLflow spans without
newer OTel translations, and derive cost summaries only when MLflow provides its cost aggregator.
These limitations do not disable run metric logging.

## Documentation

New to MLflow? Start with the [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/ml/getting-started/quickstart/).
The following guides provide additional context for using this plugin:

- [Model Registry Tutorial](https://mlflow.org/docs/latest/ml/model-registry/tutorial) — register and manage model versions.
- [Backend Stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/) — understand where MLflow stores tracking data.
- [Tracking Server Configuration](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) — configure tracking and registry stores independently.
- [MongoDB Connection Strings](https://www.mongodb.com/docs/manual/reference/connection-string/) — configure authentication, TLS, replica sets, and other URI options.

## Versioning

This project follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`), same as
[MLflow](https://mlflow.org/) itself. It's currently on major version `0`, meaning the API isn't
yet stable and may change between releases. `main` always carries a `.devN` version (e.g.
`0.1.0.dev0`) to signal it's unreleased and ahead of the latest published version.

## Contributor setup

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/mongodb-developer/mlflow-mongodb.git
cd mlflow-mongodb

python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If you downloaded a source archive instead, extract it and run the `venv` and `pip` commands
from the extracted repository root. The final `.` in `pip install -e .` means “install the
project in the current directory,” and `-e` keeps the installation linked to that source
checkout.

Install the development dependencies and Git hooks with:

```bash
python -m pip install -e ".[dev]"
pre-commit install --install-hooks
```

Run all formatting, linting, and repository checks with:

```bash
pre-commit run --all-files
```

## Functional tests

The functional tests use a real MongoDB 8.0 or newer server. Install the project in editable
mode with its development dependencies before running them:

```bash
python -m pip install -e ".[dev]"
```

Set `MONGODB_URI` to a dedicated test database. The database name must contain `test` as a
distinct hyphen- or underscore-separated segment; for example:

```bash
export MONGODB_URI="mongodb://localhost:27017/mlflow_functional_test"
python -m pytest tests/functional
```

Authentication, replica-set, TLS, and other standard MongoDB URI options can be included in
`MONGODB_URI`. Use a replica set when running tests that exercise MongoDB transactions.

The functional suite deletes documents from its application collections after every test so
their indexes can be reused. At the end of the test session, it drops the database selected by
`MONGODB_URI`. Never point `MONGODB_URI` at a database containing data that must be preserved.
