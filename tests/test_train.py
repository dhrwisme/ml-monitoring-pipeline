"""Tests for src/train.py."""

import pandas as pd

from src.train import chronological_split


def test_split_sizes_and_no_temporal_overlap():
    df = pd.DataFrame({
        "date_heure": pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC"),
        "value": range(100),
    })
    train_df, test_df = chronological_split(df, split_ratio=0.85)

    assert len(train_df) == 85
    assert len(test_df) == 15
    # every train timestamp must be strictly earlier than every test timestamp
    assert train_df["date_heure"].max() < test_df["date_heure"].min()