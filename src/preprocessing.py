from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def validate_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    rename_map = {
        "DATE": "date",
        "date": "date",
        "tmax": "tmax_c",
        "TMAX": "tmax_c",
        "temperature_max_c": "tmax_c",
        "maximum_temperature_c": "tmax_c",
        "location": "location",
        "LOCATION": "location",
        "NAME": "location",
        "latitude": "latitude",
        "LATITUDE": "latitude",
        "longitude": "longitude",
        "LONGITUDE": "longitude",
    }
    cleaned = cleaned.rename(columns=rename_map)

    if "date" not in cleaned.columns:
        raise ValueError("The dataset must contain a date column.")
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    if "tmax_c" not in cleaned.columns:
        if "TAVG" in cleaned.columns:
            cleaned["tmax_c"] = pd.to_numeric(cleaned["TAVG"], errors="coerce") / 10.0
        else:
            raise ValueError("Temperature data is missing the expected maximum-temperature column.")
    cleaned["tmax_c"] = pd.to_numeric(cleaned["tmax_c"], errors="coerce")

    if "location" not in cleaned.columns:
        cleaned["location"] = "default_location"
    cleaned["location"] = cleaned["location"].fillna("default_location").astype(str)

    if "latitude" not in cleaned.columns:
        cleaned["latitude"] = np.nan
    if "longitude" not in cleaned.columns:
        cleaned["longitude"] = np.nan

    cleaned = cleaned.drop_duplicates(subset=["date", "location"], keep="last")
    cleaned = cleaned.sort_values(["location", "date"]).reset_index(drop=True)
    cleaned = cleaned[cleaned["date"].notna()].copy()
    cleaned = cleaned[~pd.isna(cleaned["tmax_c"])].copy()

    cleaned["tmax_c"] = cleaned["tmax_c"].astype(float)

    expanded_frames = []
    for location, group in cleaned.groupby("location", sort=True):
        base = group.sort_values("date").set_index("date")
        start = base.index.min()
        end = base.index.max()
        full_index = pd.date_range(start=start, end=end, freq="D")
        loc_frame = base.reindex(full_index).reset_index().rename(columns={"index": "date"})
        loc_frame = loc_frame.set_index("date")
        loc_frame["location"] = location
        loc_frame["latitude"] = base["latitude"].iloc[0] if "latitude" in base.columns and not base["latitude"].dropna().empty else np.nan
        loc_frame["longitude"] = base["longitude"].iloc[0] if "longitude" in base.columns and not base["longitude"].dropna().empty else np.nan
        loc_frame["tmax_c"] = pd.to_numeric(loc_frame["tmax_c"], errors="coerce")
        loc_frame["tmax_c"] = loc_frame["tmax_c"].interpolate(method="time", limit_direction="both")
        loc_frame = loc_frame.reset_index().rename(columns={"date": "date"})
        expanded_frames.append(loc_frame)

    expanded = pd.concat(expanded_frames, ignore_index=True)
    expanded = expanded.dropna(subset=["tmax_c"]).sort_values(["location", "date"]).reset_index(drop=True)
    return expanded


def split_chronologically(df: pd.DataFrame, validation_fraction: float = 0.15, test_fraction: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction + test_fraction < 1:
        raise ValueError("validation_fraction + test_fraction must be between 0 and 1.")

    all_dates = sorted(df["date"].dropna().unique())
    if len(all_dates) < 3:
        raise ValueError("At least three daily timestamps are required for training, validation, and testing.")

    total_days = len(all_dates)
    train_end = int(total_days * (1 - (validation_fraction + test_fraction)))
    val_end = train_end + int(total_days * validation_fraction)

    train_cutoff = all_dates[train_end]
    val_cutoff = all_dates[val_end]

    train_df = df[df["date"] < train_cutoff].copy()
    val_df = df[(df["date"] >= train_cutoff) & (df["date"] < val_cutoff)].copy()
    test_df = df[df["date"] >= val_cutoff].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("The chronological split produced an empty train, validation, or test set.")
    return train_df, val_df, test_df


def generate_sequences(df: pd.DataFrame, lookback: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if lookback <= 0:
        raise ValueError("lookback must be a positive integer.")

    locations = []
    sequences = []
    targets = []
    target_dates = []

    for location, group in df.groupby("location", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        values = group["tmax_c"].to_numpy(dtype=float)
        dates = group["date"].to_numpy()
        for i in range(len(values) - lookback):
            window = values[i : i + lookback]
            if np.any(np.isnan(window)):
                continue
            target_value = values[i + lookback]
            if np.isnan(target_value):
                continue
            sequences.append(window)
            targets.append(target_value)
            target_dates.append(dates[i + lookback])
            locations.append(location)

    if not sequences:
        raise ValueError("No valid sequences were generated for the configured lookback window.")

    X = np.asarray(sequences, dtype=np.float32).reshape(len(sequences), lookback, 1)
    y = np.asarray(targets, dtype=np.float32).reshape(len(targets), 1)
    dates_arr = np.asarray(target_dates, dtype=object)
    location_arr = np.asarray(locations, dtype=object)
    return X, y, dates_arr, location_arr


def prepare_training_data(df: pd.DataFrame, lookback: int = 30, validation_fraction: float = 0.15, test_fraction: float = 0.15) -> dict[str, Any]:
    train_df, val_df, test_df = split_chronologically(df, validation_fraction=validation_fraction, test_fraction=test_fraction)
    X_train, y_train, train_dates, _ = generate_sequences(train_df, lookback=lookback)
    X_val, y_val, val_dates, _ = generate_sequences(val_df, lookback=lookback)
    X_test, y_test, test_dates, _ = generate_sequences(test_df, lookback=lookback)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(X_train.reshape(-1, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(y_train.reshape(-1, 1))

    X_train_scaled = feature_scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape[0], lookback, 1)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape[0], lookback, 1)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape[0], lookback, 1)

    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train_scaled,
        "y_train": y_train_scaled,
        "X_val": X_val_scaled,
        "y_val": y_val_scaled,
        "X_test": X_test_scaled,
        "y_test": y_test_scaled,
        "dates_train": train_dates,
        "dates_val": val_dates,
        "dates_test": test_dates,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "train_start": train_df["date"].min(),
        "train_end": train_df["date"].max(),
        "val_start": val_df["date"].min(),
        "val_end": val_df["date"].max(),
        "test_start": test_df["date"].min(),
        "test_end": test_df["date"].max(),
    }


def save_config_json(config: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, default=str, indent=2)
