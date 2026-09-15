"""Tests for src/monitor.py.

Pure-logic tests (check_drift, match_nearest) run everywhere, no DB needed.
The rest are integration tests against a real Postgres -- CI provides one as
a service container (see .github/workflows/ci.yml); locally they're skipped
unless DATABASE_URL is set and reachable, so `pytest` still works out of
the box without a local Postgres running.
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.monitor import check_drift, match_nearest


def test_no_drift_when_recent_matches_baseline():
    baseline = {"consommation": {"mean": 45000.0, "std": 2000.0}}
    recent = pd.DataFrame({"consommation": [44800, 45200, 45100]})

    report = check_drift(recent, baseline)

    assert len(report) == 1
    assert report[0]["drifted"] is False


def test_drift_flagged_when_recent_mean_far_from_baseline():
    baseline = {"consommation": {"mean": 45000.0, "std": 2000.0}}
    recent = pd.DataFrame({"consommation": [60000, 61000, 59500]})  # >2 std away

    report = check_drift(recent, baseline)

    assert report[0]["drifted"] is True


def test_match_nearest_within_tolerance():
    target = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    actuals = {
        datetime(2026, 1, 1, 11, 45, tzinfo=timezone.utc): 100.0,
        datetime(2026, 1, 1, 12, 3, tzinfo=timezone.utc): 200.0,  # closest
        datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc): 300.0,
    }

    result = match_nearest(target, actuals, tolerance=timedelta(minutes=10))

    assert result == 200.0


def test_match_nearest_outside_tolerance_returns_none():
    target = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    actuals = {datetime(2026, 1, 1, 12, 25, tzinfo=timezone.utc): 200.0}

    result = match_nearest(target, actuals, tolerance=timedelta(minutes=10))

    assert result is None


def test_match_nearest_empty_actuals_returns_none():
    target = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    assert match_nearest(target, {}, tolerance=timedelta(minutes=10)) is None


# --- Integration tests: real Postgres required ---------------------------

@pytest.fixture()
def db_conn():
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set -- skipping Postgres integration tests")
    from src.monitor import get_connection, init_db

    try:
        conn = get_connection()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not connect to DATABASE_URL: {exc}")

    init_db(conn=conn)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE predictions RESTART IDENTITY")
    conn.commit()

    yield conn
    conn.close()


def _sample_features():
    return {
        "consommation": 45000.0, "lag_1": 44900.0, "lag_4": 44500.0,
        "lag_96": 43000.0, "lag_672": 42000.0, "roll_mean_4": 44700.0,
        "roll_std_4": 200.0, "roll_mean_96": 44000.0,
        "hour": 14, "day_of_week": 2, "is_weekend": 0, "month": 9,
    }


def test_log_prediction_round_trips(db_conn):
    from src.monitor import load_recent_predictions, log_prediction

    as_of = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    log_prediction(_sample_features(), prediction=45100.0, as_of=as_of, conn=db_conn)

    recent = load_recent_predictions(conn=db_conn)

    assert len(recent) == 1
    row = recent.iloc[0]
    assert row["prediction"] == 45100.0
    assert row["consommation"] == 45000.0
    assert row["actual"] is None
    # target_time must be exactly as_of + the model's 15-minute horizon
    assert pd.Timestamp(row["target_time"]) == pd.Timestamp(as_of) + pd.Timedelta(minutes=15)


def test_reconcile_actuals_matches_and_is_idempotent(db_conn, monkeypatch):
    import src.monitor as monitor

    as_of = datetime.now(timezone.utc) - timedelta(minutes=30)
    monitor.log_prediction(_sample_features(), prediction=45100.0, as_of=as_of, conn=db_conn)
    target_time = as_of + monitor.HORIZON

    monkeypatch.setattr(
        monitor, "fetch_actuals_window",
        lambda start, end: {target_time: 45050.0},
    )

    result = monitor.reconcile_actuals(conn=db_conn)
    assert result == {"checked": 1, "matched": 1}

    recent = monitor.load_recent_predictions(conn=db_conn)
    assert recent.iloc[0]["actual"] == 45050.0

    # re-running must not re-match an already-reconciled row
    result_again = monitor.reconcile_actuals(conn=db_conn)
    assert result_again == {"checked": 0, "matched": 0}


def test_reconcile_actuals_skips_rows_not_yet_due(db_conn, monkeypatch):
    import src.monitor as monitor

    # target_time = now + HORIZON, i.e. minutes in the future -- nowhere
    # near the RECONCILE_LAG cutoff yet.
    monitor.log_prediction(_sample_features(), prediction=45100.0, conn=db_conn)

    called = {"n": 0}

    def _fail_if_called(start, end):
        called["n"] += 1
        return {}

    monkeypatch.setattr(monitor, "fetch_actuals_window", _fail_if_called)

    result = monitor.reconcile_actuals(conn=db_conn)

    assert result == {"checked": 0, "matched": 0}
    assert called["n"] == 0


def test_compute_stats_reports_rolling_mae(db_conn, monkeypatch):
    import src.monitor as monitor

    as_of = datetime.now(timezone.utc) - timedelta(minutes=30)
    monitor.log_prediction(_sample_features(), prediction=45000.0, as_of=as_of, conn=db_conn)
    target_time = as_of + monitor.HORIZON
    monkeypatch.setattr(monitor, "fetch_actuals_window", lambda start, end: {target_time: 45200.0})
    monitor.reconcile_actuals(conn=db_conn)

    stats = monitor.compute_stats(conn=db_conn)

    assert stats["n_matched"] == 1
    assert stats["rolling_mae"] == pytest.approx(200.0)
    assert stats["pending_reconciliation"] == 0
    assert len(stats["recent_points"]) == 1
