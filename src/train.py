"""Train the XGBoost consumption-forecasting model and evaluate against baselines."""

from pathlib import Path

from src.monitor import save_baseline
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.features import FEATURE_COLUMNS, build_features
from src.ingest import DEFAULT_OUTPUT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_v1.json"
SPLIT_RATIO = 0.85


def chronological_split(df: pd.DataFrame, split_ratio: float = SPLIT_RATIO):
    """Split a time-ordered DataFrame into train/test without shuffling."""
    split_idx = int(len(df) * split_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model: xgb.XGBRegressor, X_train, y_train, X_test, y_test) -> dict:
    """Compute model and persistence-baseline metrics."""
    y_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    persistence_pred = X_test["consommation"]

    return {
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_rmse": root_mean_squared_error(y_test, y_pred),
        "train_mae": mean_absolute_error(y_train, train_pred),
        "persistence_mae": mean_absolute_error(y_test, persistence_pred),
    }


def save_model(model: xgb.XGBRegressor, output_path: Path = MODEL_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
    return output_path.resolve()


if __name__ == "__main__":
    raw = pd.read_csv(DEFAULT_OUTPUT, parse_dates=["date_heure"])
    features = build_features(raw)

    train_df, test_df = chronological_split(features)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target"]

    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_train, y_train, X_test, y_test)

    improvement = (1 - metrics["test_mae"] / metrics["persistence_mae"]) * 100
    print(f"Test  MAE: {metrics['test_mae']:.2f}  RMSE: {metrics['test_rmse']:.2f}")
    print(f"Train MAE: {metrics['train_mae']:.2f}")
    print(f"Persistence baseline MAE: {metrics['persistence_mae']:.2f} "
          f"(model improves by {improvement:.1f}%)")

    saved_path = save_model(model)
    print(f"Model saved to: {saved_path}")

    baseline_path = save_baseline(X_train)
    print(f"Feature baseline saved to: {baseline_path}")
