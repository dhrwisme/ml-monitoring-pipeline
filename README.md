# ML Monitoring Pipeline

An end-to-end MLOps project: forecasting short-horizon national electricity
consumption in France, with the full production lifecycle around it —
ingestion, feature engineering, training, containerized serving, CI/CD,
deployment, and drift-triggered retraining.

## Motivation

Most of the ML work I've done so far — coursework, a thesis, prior data
science roles — ends at a trained model and a notebook. This project exists
to close the other half: the engineering discipline that makes a model
usable in production — version control workflow, automated testing, CI/CD,
containerization, cloud deployment, and monitoring. The forecasting problem
itself is intentionally kept simple (a single well-understood model,
XGBoost) so the actual complexity budget goes into the pipeline around it,
not the model.

## What it does

Predicts French national electricity consumption 15 minutes ahead, using
public data from RTE (the French grid operator) via their éCO2mix API.
The forecasting task is a proxy — the same pipeline shape (ingest → feature
engineer → train → serve → monitor → retrain) applies to most operational
time-series forecasting problems, not just this one.

## Data source

[RTE éCO2mix](https://www.rte-france.com/eco2mix), accessed via the public
[ODRÉ Opendatasoft API](https://odre.opendatasoft.com/), no authentication
required. Data is at 15-minute resolution and includes national consumption,
generation by source, and RTE's own day-ahead forecasts.

## Status

🚧 In progress — Phase 2 (data pipeline). Data ingestion, feature
engineering, model training and evaluation are validated end-to-end in a
notebook (`notebooks/01_explore_rte_data.ipynb`); the next step is
extracting that logic into a tested, reusable pipeline (`src/`).

## Approach

- **Target:** consumption 15 minutes ahead (`shift(-1)` on the raw series),
  not RTE's own day-ahead forecast — those are a different horizon and not
  a fair baseline for this task.
- **Features:** lagged consumption (1, 4, 96, 672 steps back), rolling
  mean/std, and calendar features (hour, day of week, weekend, month).
- **Baseline:** naive persistence (predict the most recent known value),
  the correct same-horizon comparison for a 15-minute-ahead forecast.
- **Model:** XGBoost regressor, kept deliberately simple — the point of
  this project is the pipeline, not squeezing out accuracy. SHAP is used
  for interpretability rather than treating the model as a black box.
- **Validation:** chronological train/test split (no shuffling — this is
  a time series), verified with no date overlap between splits.

## Architecture (target end state)
RTE API → ingestion script → feature store (CSV/SQLite for now)
→ training script → model artifact
→ FastAPI serving container → Azure Container Apps
→ monitoring (prediction drift) → retraining trigger

CI runs tests and builds the container on every PR; deployment to Azure
happens on merge to `main`.

## Repo structure
notebooks/ exploratory work — ingestion, features, training, evaluation
src/ (coming next) the same logic as importable, tested modules
data/ raw and processed data (gitignored)
models/ trained model artifacts (gitignored)


## Why this dataset

RTE's API is free, requires no key, and updates in near real time, which
makes it usable for a genuinely live monitoring/retraining demo rather than
a static historical dataset with nothing left to happen to it.