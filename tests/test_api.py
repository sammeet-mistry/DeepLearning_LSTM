from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.predictor import ClimatePredictor


class FakePredictor:
    model_available = True

    def predict(self, location: str, history=None):
        return {
            "location": location,
            "forecast_date": "2025-03-02",
            "predicted_temperature_c": 27.5,
            "model_id": "fake-model",
            "generated_at_utc": "2025-03-01T00:00:00Z",
            "input_sequence_length": 30,
            "warnings": [],
        }

    def metrics(self):
        return {
            "model": {"mae": 2.0, "rmse": 2.5},
            "baseline": {"mae": 2.2, "rmse": 2.8},
            "lookback": 30,
            "train_start": "2024-01-01",
            "train_end": "2024-12-31",
            "test_start": "2025-01-01",
            "test_end": "2025-01-31",
            "location": "JFK_Airport",
            "evaluated_at_utc": "2025-03-01T00:00:00Z",
        }

    def model_info(self):
        return {
            "location": "JFK_Airport",
            "lookback": 30,
            "model_path": "models/best_model.keras",
            "train_start": "2024-01-01",
            "train_end": "2024-12-31",
            "validation_start": "2025-01-01",
            "validation_end": "2025-01-15",
            "test_start": "2025-01-16",
            "test_end": "2025-01-31",
            "training_timestamp_utc": "2025-03-01T00:00:00Z",
            "architecture": ["Input", "LSTM(32)", "Dropout(0.2)", "Dense(1)"],
        }


def test_health_endpoint():
    app = create_app(FakePredictor())
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_validates_request():
    app = create_app(FakePredictor())
    client = TestClient(app)
    response = client.post("/api/predict", json={"location": "JFK_Airport", "history": [20.0] * 30})
    assert response.status_code == 200
    assert response.json()["location"] == "JFK_Airport"

    bad_response = client.post("/api/predict", json={"location": "JFK_Airport", "history": []})
    assert bad_response.status_code == 422
