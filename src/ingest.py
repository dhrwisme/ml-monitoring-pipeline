"""Ingest RTE eco2mix data and save it as a local CSV."""

from pathlib import Path

import pandas as pd
import requests

EXPORT_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/exports/json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "eco2mix_historical.csv"


def fetch_eco2mix_data(where: str = "consommation IS NOT NULL") -> pd.DataFrame:
    """Pull the full eco2mix history from the ODRE exports/json endpoint."""
    params = {"where": where, "order_by": "date_heure asc"}
    response = requests.get(EXPORT_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    df["date_heure"] = pd.to_datetime(df["date_heure"])
    return df


def save_raw_data(df: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Save df to CSV, creating parent directories as needed. Returns the resolved path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path.resolve()


if __name__ == "__main__":
    df = fetch_eco2mix_data()
    saved_path = save_raw_data(df)
    print(f"Saved {len(df)} rows to: {saved_path}")