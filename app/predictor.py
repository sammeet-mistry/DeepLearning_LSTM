from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from src.data_loader import load_temperature_dataset
from src.predict import predict_temperature
from src.preprocessing import validate_and_prepare_dataframe


class ClimatePredictor:
    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model_config = self._load_model_config()
        self.architecture = ["Input", "LSTM(32)", "Dropout(0.2)", "Dense(1)"]
        if self.model_config:
            self.model_config["location"] = self.model_config.get("location", "New_Delhi")

    def _load_model_config(self) -> dict:
        config_path = self.model_dir / "training_config.json"
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @property
    def model_available(self) -> bool:
        return (self.model_dir / "best_model.keras").exists() and bool(self.model_config)

    def build_sequence_from_data(self, location: str) -> list[float]:
        data = load_temperature_dataset(Path("data") / "noaa_jfk_daily_temperature.csv")
        cleaned = validate_and_prepare_dataframe(data)
        location_data = cleaned[cleaned["location"] == location].sort_values("date").reset_index(drop=True)
        if location_data.empty:
            raise ValueError(f"No historical observations were found for location '{location}'.")
        sequence = location_data["tmax_c"].dropna().astype(float).tolist()
        lookback = int(self.model_config.get("lookback", 30))
        if len(sequence) < lookback:
            raise ValueError(f"Location '{location}' does not have enough historical daily observations to satisfy the required lookback of {lookback} days.")
        return sequence[-lookback:]

    def predict(self, location: str, history: list[float] | None = None) -> dict:
        if history is None:
            history = self.build_sequence_from_data(location)
        result = predict_temperature(history, model_dir=self.model_dir, location=location)
        return result

    def metrics(self) -> dict:
        metrics_path = Path("outputs") / "evaluation_summary.json"
        if not metrics_path.exists():
            return {}
        with metrics_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def model_info(self) -> dict:
        info = {
            "location": self.model_config.get("location", "New_Delhi"),
            "lookback": self.model_config.get("lookback", 30),
            "model_path": str(self.model_dir / "best_model.keras"),
            "train_start": self.model_config.get("train_start", "unknown"),
            "train_end": self.model_config.get("train_end", "unknown"),
            "validation_start": self.model_config.get("val_start", None),
            "validation_end": self.model_config.get("val_end", None),
            "test_start": self.model_config.get("test_start", None),
            "test_end": self.model_config.get("test_end", None),
            "training_timestamp_utc": self.model_config.get("training_timestamp_utc", None),
            "architecture": self.architecture,
        }
        return info
