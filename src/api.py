"""FastAPI service serving 15-minute-ahead consumption forecasts, plus
prediction logging, actuals reconciliation, and a monitoring dashboard.
"""

import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.features import FEATURE_COLUMNS
from src.monitor import compute_stats, init_db, log_prediction, reconcile_actuals

load_dotenv()

init_db()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_v1.json"
DASHBOARD_PATH = PROJECT_ROOT / "static" / "dashboard.html"
RECONCILE_TOKEN = os.environ.get("RECONCILE_TOKEN")

app = FastAPI(title="RTE Consumption Forecast API")

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)


class PredictionRequest(BaseModel):
    consommation: float
    lag_1: float
    lag_4: float
    lag_96: float
    lag_672: float
    roll_mean_4: float
    roll_std_4: float
    roll_mean_96: float
    hour: int
    day_of_week: int
    is_weekend: int
    month: int
    # Timestamp the features above describe (i.e. when `consommation` was
    # last known-true). Optional for backward compatibility -- omitting it
    # falls back to server-receipt time, which approximates but does not
    # guarantee correct reconciliation against RTE's actuals 15 minutes
    # later. Callers with a real data timestamp should pass it.
    as_of: Optional[datetime] = None


class PredictionResponse(BaseModel):
    predicted_consommation_15min_ahead: float


class ReconcileResponse(BaseModel):
    checked: int
    matched: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    payload = request.model_dump()
    row = pd.DataFrame([payload])[FEATURE_COLUMNS]
    prediction = float(model.predict(row)[0])
    log_prediction(payload, prediction, as_of=request.as_of)
    return PredictionResponse(predicted_consommation_15min_ahead=prediction)


@app.post("/monitor/reconcile", response_model=ReconcileResponse)
def reconcile(x_reconcile_token: Optional[str] = Header(default=None)):
    """Join RTE's since-published actuals onto past predictions. Called on a
    schedule by a GitHub Actions workflow, not meant for public traffic --
    gated by a shared-secret header so random callers can't trigger repeated
    outbound calls to RTE's API on this service's behalf.
    """
    if not RECONCILE_TOKEN or not x_reconcile_token or not secrets.compare_digest(
        x_reconcile_token, RECONCILE_TOKEN
    ):
        raise HTTPException(status_code=401, detail="invalid or missing reconcile token")
    return reconcile_actuals()


@app.get("/monitor/stats")
def stats():
    """Everything the dashboard needs: rolling accuracy, pending-reconciliation
    count, recent predicted-vs-actual points, and the drift report."""
    return compute_stats()


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_PATH.read_text()
