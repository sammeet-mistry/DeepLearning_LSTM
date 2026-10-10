from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DEFAULT_STATION_ID = "IN-DELHI"
DEFAULT_LOCATION = "New_Delhi"
DEFAULT_LATITUDE = 28.6139
DEFAULT_LONGITUDE = 77.2090
NOAA_API_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
OPEN_METEO_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def build_open_meteo_url(latitude: float, longitude: float, start_date: str, end_date: str) -> str:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max",
        "timezone": "auto",
    }
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"{OPEN_METEO_API_URL}?{query}"


def download_noaa_temperature_data(
    station_id: str = DEFAULT_STATION_ID,
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-31",
    location_name: str = DEFAULT_LOCATION,
) -> pd.DataFrame:
    url = build_open_meteo_url(DEFAULT_LATITUDE, DEFAULT_LONGITUDE, start_date, end_date)
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if "daily" not in payload or "temperature_2m_max" not in payload["daily"]:
        raise ValueError("Open-Meteo did not return a valid daily temperature_2m_max series.")

    df = pd.DataFrame({
        "date": payload["daily"]["time"],
        "location": location_name,
        "station_id": station_id,
        "latitude": DEFAULT_LATITUDE,
        "longitude": DEFAULT_LONGITUDE,
        "tmax_c": pd.to_numeric(payload["daily"]["temperature_2m_max"], errors="coerce"),
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[["date", "location", "station_id", "latitude", "longitude", "tmax_c"]].copy()
    return df.dropna(subset=["date", "tmax_c"]).sort_values("date").reset_index(drop=True)


def load_temperature_dataset(
    data_path: Optional[str | os.PathLike[str]] = None,
    station_id: str = DEFAULT_STATION_ID,
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-31",
    location_name: str = DEFAULT_LOCATION,
) -> pd.DataFrame:
    data_file = Path(data_path) if data_path is not None else Path("data") / "new_delhi_temperature.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
        if {"date", "tmax_c"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)

    data_file.parent.mkdir(parents=True, exist_ok=True)
    df = download_noaa_temperature_data(station_id, start_date, end_date, location_name)
    df.to_csv(data_file, index=False)
    return df


if __name__ == "__main__":
    data = load_temperature_dataset()
    print(f"Loaded {len(data)} daily records from {data['date'].min()} to {data['date'].max()}.")
