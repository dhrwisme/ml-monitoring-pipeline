"""Tests for src/features.py."""

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_features


def _make_synthetic_raw(n_rows: int = 700) -> pd.DataFrame:
    """Build a small synthetic dataset shaped like the real eco2mix data.

    n_rows must exceed 672 (the longest lag) plus a buffer, or build_features
    will drop every row and the tests below become meaningless.
    """
    date_heure = pd.date_range("2026-01-01", periods=n_rows, freq="15min", tz="UTC")
    consommation = np.arange(n_rows, dtype=float) * 10  # predictable arithmetic sequence
    return pd.DataFrame({
        "date_heure": date_heure,
        "consommation": consommation,
        "prevision_j": consommation,
        "prevision_j1": consommation,
    })


def test_target_is_next_reading_not_current():
    """target[i] must equal consommation at the ORIGINAL row i+1 -- this is the
    leakage check. Since consommation increases by exactly 10 each step,
    target[i] must equal consommation[i] + 10.
    """
    features = build_features(_make_synthetic_raw())
    assert (features["target"] == features["consommation"] + 10).all()


def test_no_nans_survive_dropna():
    features = build_features(_make_synthetic_raw())
    assert features.isnull().sum().sum() == 0


def test_all_feature_columns_present():
    features = build_features(_make_synthetic_raw())
    for col in FEATURE_COLUMNS:
        assert col in features.columns