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

Deployed. Ingestion, feature engineering, training, and evaluation are
implemented as tested modules in `src/`, gated by CI on every pull request.
The model is served via FastAPI, containerized with Docker, and deployed
live on Render:

https://ml-monitoring-pipeline.onrender.com

(Free tier — the instance spins down after 15 minutes of inactivity; the
first request after idle can take ~50 seconds to wake it back up.)

A live monitoring dashboard is up at
[/dashboard](https://ml-monitoring-pipeline.onrender.com/dashboard),
showing rolling accuracy against RTE's actual (since-published) readings
and feature drift against the training baseline. Retraining is still
triggered manually on drift, not automated — see Architecture below.

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

## Architecture
RTE API → src/ingest.py → src/features.py → src/train.py → model artifact
→ FastAPI (src/api.py) → Docker → Render (deployed)
→ every /predict logged to Postgres → GitHub Actions (every 15 min) calls
  POST /monitor/reconcile, joining in RTE's actual readings once published
→ GET /monitor/stats (rolling MAE, drift report) → /dashboard (polls stats,
  renders it) → retraining trigger [manual today, not automated]

**Why Postgres, not SQLite:** Render's free web services have no
persistent disk — anything written to local container disk is lost on
every restart, and the free instance restarts routinely (it spins down
after 15 minutes idle). A monitoring dashboard whose data resets
constantly isn't actually monitoring anything, so prediction logging
lives in a free-tier external Postgres (Neon or Supabase) instead. See
`.env.example` for the `DATABASE_URL` / `RECONCILE_TOKEN` variables this
requires, both as local `.env` values and as real environment
variables/secrets on Render and in the repo's GitHub Actions secrets.

**Why a GitHub Actions cron, not an in-process scheduler:** the same
free-tier spin-down means a scheduler running inside the FastAPI process
wouldn't fire reliably — it's asleep whenever there's no traffic, which is
exactly when a background job would need to run. A scheduled GitHub
Actions workflow (`.github/workflows/reconcile.yml`) hits
`POST /monitor/reconcile` from outside the service every 15 minutes
instead, which also has the side effect of keeping the instance warm.

CI (GitHub Actions) runs the test suite — including Postgres integration
tests against a `postgres:16` service container — on every pull request.
Deployment to Render is automatic on merge to `main`, via Render's native
GitHub integration.

**Note on the Azure pivot:** this was originally designed for Azure
Container Apps. Partway through, Azure Student verification became
unavailable (lost access to the required school email), so the project
moved to Render instead — free tier, no credit card, and Docker-based
GitHub auto-deploy that fit the remaining timeline. Given more time, Azure
Container Apps remains the more production-representative target.

**Note on the model artifact:** `models/xgb_v1.json` is committed directly
to git rather than pulled from a model registry. This is a deliberate
simplification for this project's scope and timeline — a longer-running
system would store it in a registry (e.g. MLflow) or produce it fresh as
part of the deploy pipeline.

## Repo structure
notebooks/  exploratory work — ingestion, features, training, evaluation
src/        ingestion, feature engineering, training, serving, and
            monitoring (prediction logging, actuals reconciliation, drift)
            as tested, importable modules
static/     dashboard.html — served at /dashboard, polls /monitor/stats
tests/      unit tests (leakage check, split-integrity check, drift/match
            logic, Postgres integration tests)
.github/    CI (tests on every PR) and the reconcile cron workflow
Dockerfile  container definition for the FastAPI service
data/       raw and processed data (gitignored)
models/     trained model artifact and feature baseline (tracked in git —
            see Architecture note)


## Why this dataset

RTE's API is free, requires no key, and updates in near real time, which
makes it usable for a genuinely live monitoring/retraining demo rather than
a static historical dataset with nothing left to happen to it.