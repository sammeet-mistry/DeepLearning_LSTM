from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

MODEL_DIR = ROOT / "models"


def load_artifacts(model_dir: str | Path = MODEL_DIR) -> tuple[tf.keras.Model, dict, dict]:
    model_dir = Path(model_dir)
    required = [model_dir / "best_model.keras", model_dir / "training_config.json", model_dir / "feature_scaler.pkl", model_dir / "target_scaler.pkl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    model = tf.keras.models.load_model(model_dir / "best_model.keras")
    with (model_dir / "training_config.json").open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    with (model_dir / "feature_scaler.pkl").open("rb") as fh:
        feature_scaler = pickle.load(fh)
    with (model_dir / "target_scaler.pkl").open("rb") as fh:
        target_scaler = pickle.load(fh)
    return model, config, {"feature_scaler": feature_scaler, "target_scaler": target_scaler}


def predict_temperature(sequence: list[float], model_dir: str | Path = MODEL_DIR, location: str = "New_Delhi") -> dict:
    if not sequence or len(sequence) < 1:
        raise ValueError("The input sequence must include at least one observed value.")

    arr = np.asarray(sequence, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("The sequence contains NaN or infinite values.")

    model, config, scalers = load_artifacts(model_dir)
    lookback = int(config["lookback"])
    if len(arr) < lookback:
        raise ValueError(f"The input sequence length ({len(arr)}) is shorter than the required lookback window ({lookback}).")

    recent = arr[-lookback:]
    scaled_history = scalers["feature_scaler"].transform(recent.reshape(-1, 1)).reshape(1, lookback, 1)
    scaled_prediction = model.predict(scaled_history, verbose=0)[0, 0]
    predicted_temp = float(scalers["target_scaler"].inverse_transform(np.array([[scaled_prediction]], dtype=float))[0, 0])

    forecast_date = str((np.datetime64('today') + np.timedelta64(1, 'D')))
    return {
        "location": location,
        "forecast_date": forecast_date,
        "predicted_temperature_c": round(predicted_temp, 2),
        "model_id": str(model_dir / "best_model.keras"),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_sequence_length": len(recent),
        "warnings": [],
    }


if __name__ == "__main__":
    sample = [22.5, 23.1, 24.0, 24.7, 25.1, 25.4, 26.0, 26.6, 27.1, 28.0, 28.4, 29.1, 29.4, 29.8, 30.2, 30.6, 30.1, 29.5, 28.8, 27.9, 27.2, 26.5, 25.9, 25.2, 24.1, 24.5, 23.8, 24.0, 24.6, 25.1]
    print(predict_temperature(sample))
