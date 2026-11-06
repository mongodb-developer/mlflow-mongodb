# MongoDB Backend for MLflow

An MLflow plugin that stores tracking data (experiments, runs, traces, and logged models) and
model registry data (registered models, versions, aliases, tags, and prompts) in MongoDB. The
tracking store and registry store are configured independently, so either one can use MongoDB
while the other uses another MLflow backend.

Licensed under the [Apache License 2.0](LICENSE).

## Installation

```bash
python -m pip install mlflow-mongodb
```

### Supported versions

| Component | Minimum | Tested up to |
| --------- | ------- | ------------ |
| Python    | 3.11    | 3.14         |
| MLflow    | 3.5     | 3.17         |
| PyMongo   | 4.0     | 4.19         |
| MongoDB   | 7.0     | 8.0          |

MLflow 4.x is not supported. Every pull request and release is tested against both the minimum
and the latest of these combinations.

## Running MLflow

The following configuration stores both tracking data and registry data in MongoDB:

```bash
export TRACKING_STORE_URI="mongodb://localhost:27017/mlflow"
export REGISTRY_STORE_URI="mongodb://localhost:27017/mlflow"
export MLFLOW_BIND_HOST="127.0.0.1"
export MLFLOW_BIND_PORT="5000"

mlflow server \
  --backend-store-uri "${TRACKING_STORE_URI}" \
  --registry-store-uri "${REGISTRY_STORE_URI}" \
  --host "${MLFLOW_BIND_HOST}" \
  --port "${MLFLOW_BIND_PORT}"
```

Open `http://127.0.0.1:5000` to use the MLflow UI. Clients can connect to the server with:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

The MongoDB URI must include the database name; `mlflow` is the database name in the example
above. Authentication, replica-set, TLS, and other connection options can be provided using
standard MongoDB URI syntax. Both `mongodb://` and `mongodb+srv://` schemes are supported, so a
MongoDB Atlas connection string works unchanged. For example:

```text
mongodb://username:password@mongo.example.com:27017/mlflow?authSource=admin
mongodb+srv://username:password@cluster0.example.mongodb.net/mlflow
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

## Reporting issues

Bug reports and feature requests are tracked in
[GitHub Issues](https://github.com/mongodb-developer/mlflow-mongodb/issues). When reporting a
bug, include the MLflow, PyMongo, MongoDB, and `mlflow-mongodb` versions, the server command you
ran, and a minimal reproduction. Do not include connection strings or credentials.

Security vulnerabilities must not be filed as public issues. Report them through
[MongoDB's vulnerability reporting process](https://www.mongodb.com/docs/manual/tutorial/create-a-vulnerability-report/)
instead.

## Versioning

This project follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`), same as
[MLflow](https://mlflow.org/) itself. It's currently on major version `0`, meaning the API isn't
yet stable and may change between releases. `main` always carries a `.devN` version (e.g.
`0.1.0.dev0`) to signal it's unreleased and ahead of the latest published version.

## Verifying releases

Releases are published to [PyPI](https://pypi.org/project/mlflow-mongodb/) through PyPI's
trusted publishing, so no long-lived upload credentials exist. Each
[GitHub release](https://github.com/mongodb-developer/mlflow-mongodb/releases) also carries the
wheel and source distribution, a detached GPG signature (`.sig`) for each, and a `SHA256SUMS`
file.

The signatures are made with MongoDB's Garasign service using this key:

|             |                                                     |
| ----------- | --------------------------------------------------- |
| Key         | `ai-builder-experience <packaging@mongodb.com>`     |
| Fingerprint | `6E33 09C7 3676 2845 73EC 0E52 5994 2DBC 12A6 CD89` |
| Public key  | <https://pgp.mongodb.com/ai-builder-experience.pub> |

To verify a release, download the distribution, its `.sig`, and `SHA256SUMS` from the release
page, import the key, and check the checksum and signature:

```bash
curl -sSLO https://pgp.mongodb.com/ai-builder-experience.pub
gpg --import ai-builder-experience.pub
sha256sum -c SHA256SUMS --ignore-missing
gpg --verify mlflow_mongodb-<version>-py3-none-any.whl.sig mlflow_mongodb-<version>-py3-none-any.whl
```

Expect `Good signature from "ai-builder-experience <packaging@mongodb.com>"` and the fingerprint
above. Compare the fingerprint against this README, not only against the downloaded key, so a
swapped key cannot verify itself.

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

## Running tests

Install the project in editable mode with its development dependencies first:

```bash
python -m pip install -e ".[dev]"
```

The unit tests need no database:

```bash
python -m pytest tests/unit
```

The functional tests use a real MongoDB 7.0 or newer server; a standalone server is fine, and CI
runs them against a single-node replica set. Set `MONGODB_URI` to a dedicated test database; for
example:

```bash
export MONGODB_URI="mongodb://localhost:27017/mlflow_functional_test"
python -m pytest tests/functional
```

Authentication, replica-set, TLS, and other standard MongoDB URI options can be included in
`MONGODB_URI`.

The functional suite deletes documents from its application collections after every test so
their indexes can be reused. At the end of the test session, it drops the database selected by
`MONGODB_URI`. Never point `MONGODB_URI` at a database containing data that must be preserved.
