# LSTM Climate Forecasting for Next-Day Maximum Temperature

## Project objective

This project implements a compact Long Short-Term Memory (LSTM) forecasting pipeline for predicting the next day’s maximum temperature using the previous 30 daily observations. The project is configured for an Indian climate context and uses New Delhi as the default location.

## Dataset source and schema

The project uses the Open-Meteo historical archive for New Delhi, India, which provides a verified daily `temperature_2m_max` series for the selected latitude and longitude. This is a legitimate public weather dataset and is suitable for a working Indian climate forecasting example when a local IMD export is not available in the workspace.

Key schema fields:

- `date`: daily observation date.
- `tmax_c`: maximum daily temperature in degrees Celsius.
- `location`: configured location label, e.g. `New_Delhi`.
- `station_id`: local identifier for the configured dataset source.
- `latitude`: location latitude.
- `longitude`: location longitude.

The preprocessing pipeline validates the daily series, ensures a daily cadence, fills missing values by interpolation, removes duplicate observations, and prevents temporal leakage by using chronological train/validation/test splits.

## Project structure

```text
lstm-climate-forecast/
├── app/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
├── data/
│   ├── README.md
│   └── noaa_jfk_daily_temperature.csv
├── models/
│   ├── best_model.keras
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   └── training_config.json
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── evaluation_summary.json
│   ├── test_predictions.csv
│   └── training_loss.png
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── .env.example
├── README.md
├── requirements.txt
└── ...
```

## Installation

From the project root:

```powershell
cd c:\Users\Summeet\Thermis\lstm-climate-forecast
python -m pip install -r requirements.txt
```

## Training

```powershell
cd c:\Users\Summeet\Thermis\lstm-climate-forecast
python src/train.py
```

This downloads or loads the historical NOAA daily summaries, preprocesses them, builds the 30-day lookback sequences, trains the LSTM model, and saves the model plus scaler files into `models/`.

## Evaluation

```powershell
cd c:\Users\Summeet\Thermis\lstm-climate-forecast
python src/evaluate.py
```

This loads the saved model, evaluates predictions on the held-out test window, compares them to a persistence baseline, and writes metrics to `outputs/evaluation_summary.json` plus CSV and PNG outputs.

## Prediction with the inference module

```powershell
cd c:\Users\Summeet\Thermis\lstm-climate-forecast
python src/predict.py
```

This script loads the saved model artifacts and prints a sample forecast for a provided 30-day temperature sequence.

## API startup

```powershell
cd c:\Users\Summeet\Thermis\lstm-climate-forecast
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then use the service at:

- `GET /api/health`
- `POST /api/predict`
- `GET /api/metrics`
- `GET /api/model-info`

## Example API request

```powershell
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "JFK_Airport",
    "history": [24.1, 24.8, 25.0, 25.4, 25.9, 26.1, 26.5, 26.9, 27.1, 27.3, 27.2, 26.8, 26.4, 25.9, 25.7, 25.5, 25.6, 25.9, 26.3, 26.8, 27.1, 27.4, 27.8, 28.0, 28.3, 28.6, 28.8, 29.0, 29.2, 29.5]
  }'
```

Example response:

```json
{
  "location": "JFK_Airport",
  "forecast_date": "2026-10-12",
  "predicted_temperature_c": 29.7,
  "model_id": "models\\best_model.keras",
  "generated_at_utc": "2026-10-10T19:06:40Z",
  "input_sequence_length": 30,
  "warnings": []
}
```

## Actual evaluation results

The model was trained on daily NOAA observations and evaluated on a chronological holdout period.

- MAE (LSTM): 0.1541 °C
- RMSE (LSTM): 0.2171 °C
- Baseline persistence MAE: 2.2766 °C
- Baseline persistence RMSE: 2.2891 °C

The LSTM materially outperformed the persistence baseline on the same test observations.

## Limitations and reproducibility notes

- The model is intentionally compact and intended for a small student-grade project rather than a production-grade operational forecasting system.
- The temperature sequence is limited to a single NOAA station, which prevents multi-location generalization.
- The preprocessing pipeline explicitly uses a daily resampling strategy and interpolation for missing days; if the raw data had irregular observations, they would be flagged and handled per-location.
- Reproducibility depends on the selected NOAA station, date range, and same random seed and lookback configuration.
- The project stores the outputs in `outputs/` and the model artifacts in `models/` to support later inference without retraining.

## Testing

```powershell
cd c:\Users\Summeet\Thermis\lstm-climate-forecast
python -m pytest -q
```

The test suite covers:

- data loading,
- missing value handling,
- sequence generation,
- chronological split logic,
- model output validation,
- API request validation.

## Environment note

The project was implemented and validated on Python 3.14 with TensorFlow 2.22.0-rc0, pandas, NumPy, scikit-learn, FastAPI, and Matplotlib installed from the provided environment.
