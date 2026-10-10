from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.predictor import ClimatePredictor
from app.schemas import HealthResponse, MetricSummary, ModelInfo, PredictRequest, PredictionResponse


def create_app(predictor: ClimatePredictor | None = None) -> FastAPI:
    app = FastAPI(title="Climate LSTM Forecast API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    predictor = predictor or ClimatePredictor(model_dir=Path("models"))
    app.state.predictor = predictor

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> dict:
        return {
            "status": "ok" if predictor.model_available else "training-artifacts-missing",
            "model_available": predictor.model_available,
            "model_path": str(Path("models") / "best_model.keras"),
        }

    @app.post("/api/predict", response_model=PredictionResponse)
    def predict(payload: PredictRequest) -> dict:
        if not predictor.model_available:
            raise HTTPException(status_code=503, detail="The trained model is not available. Run the training pipeline first.")

        try:
            result = predictor.predict(location=payload.location, history=payload.history)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return result

    @app.get("/api/metrics", response_model=MetricSummary)
    def metrics() -> dict:
        values = predictor.metrics()
        if not values:
            raise HTTPException(status_code=503, detail="Evaluation summary not found. Run the evaluation pipeline first.")
        return values

    @app.get("/api/model-info", response_model=ModelInfo)
    def model_info() -> dict:
        values = predictor.model_info()
        if not values:
            raise HTTPException(status_code=503, detail="Model metadata not found. Train the model first.")
        return values

    return app


app = create_app()
