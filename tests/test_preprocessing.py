from __future__ import annotations

import pandas as pd

from src.preprocessing import generate_sequences, split_chronologically, validate_and_prepare_dataframe


def test_validate_and_prepare_dataframe_handles_duplicates_and_missing_dates():
    df = pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"],
            "TMAX": [220, 220, 221, 225, 226],
            "NAME": ["JFK_Airport", "JFK_Airport", "JFK_Airport", "JFK_Airport", "JFK_Airport"],
        }
    )

    cleaned = validate_and_prepare_dataframe(df)
    assert cleaned["date"].is_monotonic_increasing
    assert cleaned["tmax_c"].notna().all()
    assert cleaned.duplicated(subset=["date", "location"]).sum() == 0
    assert cleaned["date"].nunique() == len(cleaned)


def test_generate_sequences_creates_lookback_windows():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    values = [20.0, 21.0, 22.0, 23.0, 22.5, 24.0, 25.0, 24.5, 26.0, 27.0]
    df = pd.DataFrame({"date": dates, "location": "JFK_Airport", "tmax_c": values, "latitude": 40.64, "longitude": -73.78})

    X, y, dates_out, locations = generate_sequences(df, lookback=3)
    assert X.shape[1] == 3
    assert X.shape[2] == 1
    assert len(y) == len(dates_out) == len(locations)
    assert X.shape[0] > 0


def test_split_chronologically_preserves_time_order():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    temps = list(range(30))
    df = pd.DataFrame({"date": dates, "location": "JFK_Airport", "tmax_c": temps, "latitude": 40.64, "longitude": -73.78})

    train_df, val_df, test_df = split_chronologically(df, validation_fraction=0.2, test_fraction=0.2)
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()
