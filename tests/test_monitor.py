"""Tests for src/monitor.py's drift detection logic."""

import pandas as pd

from src.monitor import check_drift


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