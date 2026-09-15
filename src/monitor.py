"""Prediction logging and drift detection for the deployed model."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "predictions.db"
BASELINE_PATH = PROJECT_ROOT / "models" / "feature_baseline.json"
DRIFT_THRESHOLD_STDS = 2.0


def save_baseline(X_train: pd.DataFrame, output_path: Path = BASELINE_PATH) -> Path:
    """Save per-feature mean/std from training data -- the reference point drift is measured against."""
    baseline = {
        col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std())}
        for col in X_train.columns
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)
    return output_path.resolve()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the predictions table if it doesn't exist. Safe to call on every startup."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    columns_sql = ", ".join(f"{col} REAL" for col in FEATURE_COLUMNS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            {columns_sql},
            prediction REAL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(features: dict, prediction: float, db_path: Path = DB_PATH) -> None:
    """Append one prediction request (inputs + output) to the log."""
    conn = sqlite3.connect(db_path)
    cols = ["timestamp"] + FEATURE_COLUMNS + ["prediction"]
    placeholders = ",".join("?" for _ in cols)
    values = [datetime.now(timezone.utc).isoformat()] + [features[c] for c in FEATURE_COLUMNS] + [prediction]
    conn.execute(f"INSERT INTO predictions ({','.join(cols)}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()


def load_recent_predictions(limit: int = 100, db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df


def load_baseline(baseline_path: Path = BASELINE_PATH) -> dict:
    with open(baseline_path) as f:
        return json.load(f)


def check_drift(recent: pd.DataFrame, baseline: dict, threshold_stds: float = DRIFT_THRESHOLD_STDS) -> list:
    """Compare recent feature means against the training baseline. Returns one dict per feature."""
    report = []
    for feature, stats in baseline.items():
        if feature not in recent.columns or recent.empty:
            continue
        recent_mean = float(recent[feature].mean())
        deviation = abs(recent_mean - stats["mean"])
        drifted = stats["std"] > 0 and deviation > threshold_stds * stats["std"]
        report.append({
            "feature": feature,
            "recent_mean": recent_mean,
            "baseline_mean": stats["mean"],
            "baseline_std": stats["std"],
            "drifted": drifted,
        })
    return report


if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"No predictions logged yet at {DB_PATH}. Make some /predict requests first, then re-run this.")
        raise SystemExit(0)

    baseline = load_baseline()
    recent = load_recent_predictions()
    print(f"Checked {len(recent)} recent predictions against the training baseline.\n")

    any_drift = False
    for row in check_drift(recent, baseline):
        flag = "DRIFT" if row["drifted"] else "ok"
        print(f"[{flag:5}] {row['feature']:15} recent_mean={row['recent_mean']:.1f}  "
              f"baseline_mean={row['baseline_mean']:.1f}  baseline_std={row['baseline_std']:.1f}")
        any_drift = any_drift or row["drifted"]

    print("\nDrift detected -- in a production system this would open a retraining PR for human review."
          if any_drift else "\nNo drift detected.")