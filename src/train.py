from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

from src.data_loader import load_temperature_dataset
from src.preprocessing import prepare_training_data, save_config_json, validate_and_prepare_dataframe

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "new_delhi_temperature.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


def build_lstm_model(lookback: int, units: int = 32, dropout_rate: float = 0.2, learning_rate: float = 0.001) -> Sequential:
    model = Sequential(
        [
            Input(shape=(lookback, 1)),
            LSTM(units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units // 2),
            Dropout(dropout_rate / 2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def save_training_history(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_model(
    data_path: str | Path = DATA_PATH,
    lookback: int = 30,
    units: int = 32,
    dropout_rate: float = 0.2,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.001,
    random_seed: int = 42,
) -> dict:
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    raw_df = load_temperature_dataset(data_path)
    cleaned = validate_and_prepare_dataframe(raw_df)
    prepared = prepare_training_data(cleaned, lookback=lookback, validation_fraction=0.15, test_fraction=0.15)

    model = build_lstm_model(lookback=lookback, units=units, dropout_rate=dropout_rate, learning_rate=learning_rate)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "best_model.keras"

    early_stopping = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0)

    history = model.fit(
        prepared["X_train"],
        prepared["y_train"],
        validation_data=(prepared["X_val"], prepared["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
    )

    with (MODEL_DIR / "feature_scaler.pkl").open("wb") as fh:
        pickle.dump(prepared["feature_scaler"], fh)
    with (MODEL_DIR / "target_scaler.pkl").open("wb") as fh:
        pickle.dump(prepared["target_scaler"], fh)

    config = {
        "lookback": lookback,
        "units": units,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "random_seed": random_seed,
        "train_start": str(prepared["train_start"]),
        "train_end": str(prepared["train_end"]),
        "val_start": str(prepared["val_start"]),
        "val_end": str(prepared["val_end"]),
        "test_start": str(prepared["test_start"]),
        "test_end": str(prepared["test_end"]),
        "training_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "location": "New_Delhi",
    }
    save_config_json(config, MODEL_DIR / "training_config.json")
    save_training_history(history, OUTPUT_DIR / "training_loss.png")
    return {"model": model, "history": history, "config": config, "prepared": prepared}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_model()
    print("Training complete. Model artifacts were saved in the models directory.")


if __name__ == "__main__":
    main()
