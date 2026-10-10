# Dataset Source

This project uses NOAA NCEI daily summaries for a single meteorological station: `USW00012839` (John F. Kennedy International Airport, New York, USA).

The raw observations are retrieved from the public NOAA Daily Summaries API, which provides daily climate values including maximum temperature (`TMAX`). This is a legitimate historical dataset and is used only for model training and evaluation.

## Schema

The downloaded records include at least the following fields:

- `DATE`: ISO date string for the daily observation.
- `TMAX`: maximum daily temperature in tenths of a degree Celsius.
- `TAVG`: average daily temperature in tenths of a degree Celsius.
- `TMIN`: minimum daily temperature in tenths of a degree Celsius.
- `LATITUDE`: station latitude.
- `LONGITUDE`: station longitude.
- `NAME`: station name.

## Notes

- Temperature values are converted from tenths of degrees Celsius to full degrees Celsius before model training.
- Missing values and duplicate dates are handled in the preprocessing stage.
- The system intentionally prevents leakage by splitting the time series chronologically.

## File

The project stores the processed daily station data in `data/noaa_jfk_daily_temperature.csv` after the first successful data-load step.
