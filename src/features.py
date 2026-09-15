"""Build model-ready features from raw eco2mix data."""

import pandas as pd

FEATURE_COLUMNS = [
    "consommation", "lag_1", "lag_4", "lag_96", "lag_672",
    "roll_mean_4", "roll_std_4", "roll_mean_96",
    "hour", "day_of_week", "is_weekend", "month",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model-ready feature DataFrame from raw eco2mix data.

    Target: consumption 15 minutes ahead (shift(-1)) -- no leakage, since
    it uses only information that would already exist at prediction time.
    Drops rows with NaNs introduced at the start/end of the series by
    lagging, rolling windows, and the forward shift.
    """
    df_features = df[["date_heure", "consommation", "prevision_j", "prevision_j1"]].copy()

    df_features["target"] = df_features["consommation"].shift(-1)

    for lag in [1, 4, 96, 672]:
        df_features[f"lag_{lag}"] = df_features["consommation"].shift(lag)

    # never center=True here -- that would average in future values
    df_features["roll_mean_4"] = df_features["consommation"].rolling(4).mean()
    df_features["roll_std_4"] = df_features["consommation"].rolling(4).std()
    df_features["roll_mean_96"] = df_features["consommation"].rolling(96).mean()

    df_features["hour"] = df_features["date_heure"].dt.hour
    df_features["day_of_week"] = df_features["date_heure"].dt.dayofweek
    df_features["is_weekend"] = df_features["day_of_week"].isin([5, 6]).astype(int)
    df_features["month"] = df_features["date_heure"].dt.month

    df_features = df_features.dropna().reset_index(drop=True)

    return df_features


if __name__ == "__main__":
    from src.ingest import DEFAULT_OUTPUT

    raw = pd.read_csv(DEFAULT_OUTPUT, parse_dates=["date_heure"])
    features = build_features(raw)
    print(f"Raw: {len(raw)} rows -> Features: {len(features)} rows "
          f"({len(raw) - len(features)} dropped to lag/rolling/shift NaNs)")
    print(features[["date_heure", "consommation", "target"]].head(3))