# MLForge

[![CI](https://github.com/Assilzt/mlforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Assilzt/mlforge/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

**Transparent, reproducible tabular ML baseline workbench.**

MLForge is an open-source MVP for turning a CSV or Excel table into a documented machine-learning baseline. It profiles the data, detects a classification or regression task, compares a small set of scikit-learn pipelines with cross-validation, evaluates the winner once on a hold-out set, and exports a portable model, predictions, and JSON report.

The project is intentionally transparent: the current MVP does not claim to solve every AutoML problem. It focuses on a reliable core that can be inspected, tested, and extended.

## What is implemented

- CSV and Excel input through the CLI and a small FastAPI API.
- Column profiling: inferred type, missing values, cardinality, duplicates, small-sample and quality warnings.
- Automatic classification/regression detection with a manual override.
- Leakage-conscious protocol: hold-out split first, cross-validation on the training split only, then one final test evaluation.
- Preprocessing with numeric imputation/scaling and categorical imputation/one-hot encoding.
- Candidate models: Logistic Regression, Random Forest, Extra Trees, Ridge, depending on task.
- Metrics: accuracy, weighted F1 and ROC-AUC for binary classification; R², MAE and RMSE for regression.
- Reproducibility: fixed seed and recorded Python/library versions.
- Export of `model.joblib`, `predictions.csv`, and `report.json`.
- Minimal FastAPI endpoints for health, profiling, training, and result downloads.
- Pytest coverage for profiling, training exports, and API health.
- GitHub Actions CI with Ruff and pytest on Python 3.11 and 3.12.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

mlforge train sample_data/classification.csv --target churn --out result
mlforge predict result/model.joblib sample_data/classification.csv --out result/predictions.csv
```

Run the API:

```bash
uvicorn backend.app.main:app --reload
```

Then open the interactive API docs at `http://127.0.0.1:8000/docs`.

## Python API

```python
import pandas as pd
from mlforge_core import profile_dataframe, train_dataframe

df = pd.read_csv("data.csv")
print(profile_dataframe(df))
report = train_dataframe(df, target="churn", out_dir="result")
print(report["winner"], report["metrics"])
```

## Output

A training run creates:

```text
result/
├── model.joblib       # pure scikit-learn Pipeline
├── predictions.csv    # untouched hold-out predictions
└── report.json        # profile, warnings, leaderboard, metrics, seed, versions
```

The exported pipeline contains standard scikit-learn preprocessing and estimators. It does not import MLForge custom classes at prediction time.

## Repository structure

```text
mlforge_core/           # standalone profiling, training, prediction, CLI
backend/app/            # thin FastAPI layer
frontend/               # minimal no-build demo page
sample_data/            # small example tables
tests/                  # core and API tests
.github/workflows/      # CI
```

## Important limitations of this MVP

- Tabular data only; no images, audio, free-text modeling, time series, clustering, or SHAP yet.
- Text and ID-like feature columns are excluded from the baseline and reported as warnings.
- The current API is a single-user development API with no authentication or production file-store isolation.
- The MVP exports a portable model/report, but does not yet generate standalone `train.py` and `predict.py` bundle scripts.
- The sample datasets are tiny and are for smoke testing only; meaningful evaluation requires a representative dataset.

## Roadmap

- [ ] Standalone bundle with generated `train.py`, `predict.py`, schema, requirements, and README.
- [ ] Stronger leakage detection and permutation feature importance.
- [ ] Datetime feature extraction and configurable model/time budgets.
- [ ] Dashboard and chart-data endpoints.
- [ ] SQLite metadata layer, background jobs, upload limits, and safe multi-user deployment.
- [ ] Docker packaging and a production-ready frontend.

## Development

```bash
ruff check .
pytest -q
```

## License

MIT

## Author

**Assil Zitouni** · Data Science, USTHB · [github.com/Assilzt](https://github.com/Assilzt)
