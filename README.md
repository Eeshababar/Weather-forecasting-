## Weather Forecasting with Stacked Ensembles & Classical Time Series
Daily forecasting for targets like maxtp (max temperature) using a leakage-safe stacked ensemble (Ridge + Random Forest + Gradient Boosting) alongside classical AR/ARIMA/SARIMAX diagnostics.
Includes robust CSV header detection, rich time-features (lags, rolling stats, EWM, Fourier), time-aware evaluation, and optional seasonality/stationarity checks (ADF/KPSS, STL).

## Feature
Robust CSV ingestion with automatic header detection (and delimiter sniffing).
Leakage-safe feature engineering:
Calendar: day-of-week, day-of-year, weekend flag
Lags: 1, 2, 3, 7, 14 (+ 21, 28 for longer memory)
Rolling mean/std with .shift(1) (prevents look-ahead)
EWM (spans 7 & 30) for responsive trend signals
Fourier annual seasonality (K=1..3)
Two horizons out of the box: t+1 and t+7 (adds polynomial features for the harder 7-day horizon).
Stacked ensemble:
Base learners: Ridge, RandomForestRegressor, GradientBoostingRegressor
Meta-learner: LinearRegression (no intercept → pure weighted blend)
Time-aware evaluation: chronological train/test split; inner validation slice for learning stack weights.
Classical TS tools (via notebooks): ACF/PACF, ADF/KPSS, STL/seasonal_decompose, AR/SARIMAX.

## Repository Contents
Notebook names reflect what’s in this project; place them under notebooks/ in your repo.
notebooks/Autoregression_and_ARIMA.ipynb
ACF/PACF, ADF/KPSS, STL, seasonal decomposition, AR & SARIMAX exploration, and walk-forward evaluation.
Helper functions include: create_features, plot_acf_pacf, fit_ar, arma_aic_grid, rolling_forecast_with_fixed_coefs, metrics, _safe_mape.
notebooks/Weather_Forecasting_using_Time_series_analysis.ipynb
Focused on stationarity checks (ADF/KPSS) and STL decomposition with basic time-aware splits and feature creation.


