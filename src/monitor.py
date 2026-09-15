"""Prediction logging, ground-truth reconciliation, and drift detection.

Storage is Postgres (via DATABASE_URL), not SQLite -- Render's free tier has
no persistent disk, so anything written to local container disk is lost on
every restart/spin-down. A prediction log that resets constantly can't
support the "recent accuracy" and "drift over time" stats this module
exists to produce, so persistence has to live outside the web service.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2
import psycopg2.extras
import requests

from src.features import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "models" / "feature_baseline.json"
DRIFT_THRESHOLD_STDS = 2.0

# The model predicts consumption this far ahead of the input features
# (see src/features.py: target = consommation.shift(-1) on 15-minute data).
HORIZON = timedelta(minutes=15)

# eco2mix-national-tr updates near-real-time but not instantly; don't try to
# reconcile a prediction until its target time is safely in the past.
RECONCILE_LAG = timedelta(minutes=10)

# How far a matched actual reading is allowed to sit from the exact target
# timestamp before it's rejected as "no matching reading found" rather than
# silently paired with a nearby-but-wrong value.
MATCH_TOLERANCE = timedelta(minutes=10)

EXPORT_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/exports/json"


def get_connection():
    """Open a new Postgres connection from DATABASE_URL.

    A fresh connection per call (rather than a shared pool) mirrors the
    original sqlite3 pattern here and is intentionally simple -- traffic to
    this API is low enough that pooling isn't worth the added moving part
    yet. Revisit if request volume grows.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. This module requires Postgres -- see "
            ".env.example for local setup."
        )
    return psycopg2.connect(database_url)


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


def load_baseline(baseline_path: Path = BASELINE_PATH) -> dict:
    with open(baseline_path) as f:
        return json.load(f)


def init_db(conn=None) -> None:
    """Create the predictions table/indexes if they don't exist. Safe to call on every startup."""
    own_conn = conn is None
    conn = conn or get_connection()
    feature_cols_sql = ", ".join(f"{col} DOUBLE PRECISION" for col in FEATURE_COLUMNS)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY,
                requested_at TIMESTAMPTZ NOT NULL,
                target_time TIMESTAMPTZ NOT NULL,
                {feature_cols_sql},
                prediction DOUBLE PRECISION NOT NULL,
                actual DOUBLE PRECISION,
                reconciled_at TIMESTAMPTZ
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_target_time
            ON predictions (target_time)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_unreconciled
            ON predictions (target_time) WHERE actual IS NULL
        """)
    conn.commit()
    if own_conn:
        conn.close()


def log_prediction(
    features: dict,
    prediction: float,
    as_of: Optional[datetime] = None,
    conn=None,
) -> None:
    """Append one prediction request (inputs + output) to the log.

    `as_of` is the timestamp the input features describe -- i.e. when
    `consommation` and the lag/rolling features were last known-true. It
    anchors target_time (as_of + HORIZON), which is what reconciliation
    matches against RTE's actuals later. Callers that don't supply it get
    server-receipt time, which is a reasonable approximation for a client
    calling /predict against live data but not exact -- prefer passing it
    explicitly when the caller knows the true data timestamp.
    """
    as_of = as_of or datetime.now(timezone.utc)
    target_time = as_of + HORIZON

    own_conn = conn is None
    conn = conn or get_connection()
    cols = ["requested_at", "target_time"] + FEATURE_COLUMNS + ["prediction"]
    placeholders = ",".join(["%s"] * len(cols))
    values = [as_of, target_time] + [features[c] for c in FEATURE_COLUMNS] + [prediction]
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO predictions ({','.join(cols)}) VALUES ({placeholders})",
            values,
        )
    conn.commit()
    if own_conn:
        conn.close()


def match_nearest(
    target_time: datetime, actuals: dict, tolerance: timedelta = MATCH_TOLERANCE
) -> Optional[float]:
    """Pure matching logic: nearest actual reading to target_time, or None if
    nothing falls within tolerance. Kept free of any DB/network I/O so it's
    cheap to unit test directly.
    """
    if not actuals:
        return None
    closest_time = min(actuals.keys(), key=lambda t: abs(t - target_time))
    if abs(closest_time - target_time) <= tolerance:
        return actuals[closest_time]
    return None


def fetch_actuals_window(start: datetime, end: datetime) -> dict:
    """Fetch RTE's actual (since-published) consumption readings in [start, end]."""
    where = (
        f"date_heure >= '{start.isoformat()}' AND date_heure <= '{end.isoformat()}' "
        "AND consommation IS NOT NULL"
    )
    params = {"where": where, "order_by": "date_heure asc", "limit": 200}
    response = requests.get(EXPORT_URL, params=params, timeout=30)
    response.raise_for_status()
    rows = response.json()
    return {pd.to_datetime(r["date_heure"]): r["consommation"] for r in rows}


def reconcile_actuals(conn=None, limit: int = 500) -> dict:
    """Join RTE's since-published actuals onto predictions whose target_time
    has safely passed. Idempotent -- only touches rows where actual IS NULL,
    so re-running (e.g. a retried GitHub Actions run) is harmless.
    """
    own_conn = conn is None
    conn = conn or get_connection()

    cutoff = datetime.now(timezone.utc) - RECONCILE_LAG
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, target_time FROM predictions
            WHERE actual IS NULL AND target_time <= %s
            ORDER BY target_time ASC
            LIMIT %s
            """,
            (cutoff, limit),
        )
        pending = cur.fetchall()

        if not pending:
            if own_conn:
                conn.close()
            return {"checked": 0, "matched": 0}

        window_start = min(t for _, t in pending) - MATCH_TOLERANCE
        window_end = max(t for _, t in pending) + MATCH_TOLERANCE
        actuals = fetch_actuals_window(window_start, window_end)

        matched = 0
        for pred_id, target_time in pending:
            actual_value = match_nearest(target_time, actuals)
            if actual_value is not None:
                cur.execute(
                    "UPDATE predictions SET actual = %s, reconciled_at = %s WHERE id = %s",
                    (actual_value, datetime.now(timezone.utc), pred_id),
                )
                matched += 1

    conn.commit()
    if own_conn:
        conn.close()
    return {"checked": len(pending), "matched": matched}


def load_recent_predictions(limit: int = 200, conn=None) -> pd.DataFrame:
    own_conn = conn is None
    conn = conn or get_connection()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT %s", (limit,)
        )
        rows = cur.fetchall()
    if own_conn:
        conn.close()
    return pd.DataFrame(rows)


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


def compute_stats(conn=None, recent_limit: int = 200, chart_points: int = 100) -> dict:
    """Aggregate everything the dashboard needs in one call: rolling accuracy
    against reconciled actuals, pending-reconciliation count, recent
    predicted-vs-actual points for the chart, and the drift report.
    """
    own_conn = conn is None
    conn = conn or get_connection()

    recent = load_recent_predictions(limit=recent_limit, conn=conn)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM predictions WHERE actual IS NULL")
        pending_count = cur.fetchone()[0]

    matched = recent[recent["actual"].notna()].copy() if not recent.empty else recent
    if not matched.empty:
        matched["error"] = matched["actual"] - matched["prediction"]
        mae = float(matched["error"].abs().mean())
        n_matched = int(len(matched))
    else:
        mae = None
        n_matched = 0

    chart = (
        matched.sort_values("target_time")
        .tail(chart_points)[["target_time", "prediction", "actual"]]
        if not matched.empty
        else pd.DataFrame(columns=["target_time", "prediction", "actual"])
    )
    chart_points_out = [
        {
            "target_time": row["target_time"].isoformat(),
            "prediction": float(row["prediction"]),
            "actual": float(row["actual"]),
        }
        for _, row in chart.iterrows()
    ]

    try:
        baseline = load_baseline()
        drift = check_drift(recent, baseline)
    except FileNotFoundError:
        drift = []

    if own_conn:
        conn.close()

    return {
        "rolling_mae": mae,
        "n_matched": n_matched,
        "pending_reconciliation": int(pending_count),
        "recent_points": chart_points_out,
        "drift": drift,
    }


if __name__ == "__main__":
    result = reconcile_actuals()
    print(f"Reconciliation: checked {result['checked']}, matched {result['matched']}")

    stats = compute_stats()
    if stats["rolling_mae"] is not None:
        print(f"Rolling MAE ({stats['n_matched']} matched predictions): {stats['rolling_mae']:.1f}")
    else:
        print("No reconciled predictions yet.")
    print(f"Pending reconciliation: {stats['pending_reconciliation']}")

    any_drift = False
    for row in stats["drift"]:
        flag = "DRIFT" if row["drifted"] else "ok"
        print(f"[{flag:5}] {row['feature']:15} recent_mean={row['recent_mean']:.1f}  "
              f"baseline_mean={row['baseline_mean']:.1f}  baseline_std={row['baseline_std']:.1f}")
        any_drift = any_drift or row["drifted"]

    print("\nDrift detected -- in a production system this would open a retraining PR for human review."
          if any_drift else "\nNo drift detected.")
