"""FastAPI service serving 15-minute-ahead consumption forecasts."""

from pathlib import Path

from src.monitor import init_db, log_prediction
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import FEATURE_COLUMNS

init_db()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_v1.json"

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


class PredictionResponse(BaseModel):
    predicted_consommation_15min_ahead: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    row = pd.DataFrame([request.model_dump()])[FEATURE_COLUMNS]
    prediction = float(model.predict(row)[0])
    log_prediction(request.model_dump(), prediction)
    return PredictionResponse(predicted_consommation_15min_ahead=prediction)
