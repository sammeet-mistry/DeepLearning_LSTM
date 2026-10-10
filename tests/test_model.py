from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM

from src.predict import predict_temperature


def test_predict_temperature_returns_valid_output(tmp_path: Path):
    model = Sequential([
        Input(shape=(8, 1)),
        LSTM(8),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model_path = tmp_path / "best_model.keras"
    model.save(model_path)

    feature_scaler = pickle.dumps
    # This directory intentionally stores the model and scalers required by inference.
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / "best_model.keras")

    # Create the minimal scaler artifacts expected by the inference module.
    from sklearn.preprocessing import MinMaxScaler

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(np.arange(80, dtype=np.float32).reshape(-1, 1))
    target_scaler = MinMaxScaler()
    target_scaler.fit(np.array([[25.0], [30.0]], dtype=np.float32))

    with (model_dir / "feature_scaler.pkl").open("wb") as fh:
        pickle.dump(feature_scaler, fh)
    with (model_dir / "target_scaler.pkl").open("wb") as fh:
        pickle.dump(target_scaler, fh)
    with (model_dir / "training_config.json").open("w", encoding="utf-8") as fh:
        fh.write('{"lookback": 8, "location": "JFK_Airport"}')

    sequence = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]
    result = predict_temperature(sequence, model_dir=model_dir)

    assert "predicted_temperature_c" in result
    assert isinstance(result["predicted_temperature_c"], float)
    assert result["input_sequence_length"] == 8
