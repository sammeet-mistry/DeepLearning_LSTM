from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_loader import load_temperature_dataset
from src.preprocessing import prepare_training_data, save_config_json, validate_and_prepare_dataframe

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"mae": mae, "rmse": rmse}


def baseline_predictions(history: np.ndarray) -> np.ndarray:
    return np.asarray(history[:, -1], dtype=float)


def evaluate_model(model_dir: str | Path = MODEL_DIR) -> dict:
    model_dir = Path(model_dir)
    model = tf.keras.models.load_model(model_dir / "best_model.keras")

    with (model_dir / "feature_scaler.pkl").open("rb") as fh:
        feature_scaler = pickle.load(fh)
    with (model_dir / "target_scaler.pkl").open("rb") as fh:
        target_scaler = pickle.load(fh)
    with (model_dir / "training_config.json").open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    data = load_temperature_dataset(ROOT / "data" / "new_delhi_temperature.csv")
    cleaned = validate_and_prepare_dataframe(data)
    prepared = prepare_training_data(cleaned, lookback=int(config["lookback"]))

    predictions_scaled = model.predict(prepared["X_test"], verbose=0).reshape(-1)
    predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(-1)
    actual = target_scaler.inverse_transform(prepared["y_test"].reshape(-1, 1)).reshape(-1)
    baseline = baseline_predictions(prepared["X_test"].reshape(len(prepared["X_test"]), -1))

    model_metrics = compute_metrics(actual, predictions)
    baseline_metrics = compute_metrics(actual, baseline)

    test_dates = prepared["dates_test"]
    rows = []
    for idx, date in enumerate(test_dates):
        rows.append(
            {
                "date": str(date),
                "actual_temperature_c": float(actual[idx]),
                "predicted_temperature_c": float(predictions[idx]),
                "baseline_temperature_c": float(baseline[idx]),
                "absolute_error": float(abs(actual[idx] - predictions[idx])),
                "baseline_error": float(abs(actual[idx] - baseline[idx])),
            }
        )

    metrics_summary = {
        "model": model_metrics,
        "baseline": baseline_metrics,
        "lookback": int(config["lookback"]),
        "train_start": config["train_start"],
        "train_end": config["train_end"],
        "test_start": config["test_start"],
        "test_end": config["test_end"],
        "location": config.get("location", "New_Delhi"),
        "evaluated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUTS_DIR / "evaluation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_summary, fh, indent=2)

    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    actual_plot = actual
    pred_plot = predictions
    axes[0].plot(test_dates, actual_plot, label="Actual", color="black")
    axes[0].plot(test_dates, pred_plot, label="LSTM prediction", color="tab:blue")
    axes[0].set_title("Actual vs predicted temperature")
    axes[0].legend()
    axes[0].set_ylabel("Temperature (°C)")

    residuals = actual_plot - pred_plot
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].scatter(test_dates, residuals, color="tab:orange")
    axes[1].set_title("Residuals")
    axes[1].set_ylabel("Residual (°C)")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    return metrics_summary


if __name__ == "__main__":
    summary = evaluate_model()
    print(json.dumps(summary, indent=2))
