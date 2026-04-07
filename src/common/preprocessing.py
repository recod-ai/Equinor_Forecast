# flake8: noqa: E402
"""
This module provides a suite of functions for running time series forecasting
experiments using the Darts library, with a focus on sliding-window validation
and iterative forecasting for deep learning models.
"""

# ==================================================================================================================================================
#                      --- Module Function Roadmap ---
# ==================================================================================================================================================
# | Function Name                          | Key Role                     | Purpose                                                                 |
# |----------------------------------------|------------------------------|-------------------------------------------------------------------------|
# | `process_deep_encoder_data_source`     | **Primary Entry Point**      | Orchestrates the entire forecasting process for a given data source.    |
# | `process_deep_encoder_well`            | **Core Well Processor**      | Handles data prep, training, and evaluation for a single well.          |
# | `run_sliding_window_forecasting`       | Backtesting Strategy         | Implements a sliding window approach for robust backtesting of models.  |
# | `fast_iterative_forecast`              | Forecasting Strategy         | Generates forecasts by reusing predictions from prior steps             |
# | `train_deep_encoder_model`             | Model Training Wrapper       | (Imported) A wrapper function that handles the model training logic.    |
# | `prepare_time_series` / `_full`        | Data Preparation             | Converts pandas DataFrames into Darts `TimeSeries` objects for modeling.|
# | `split_train_validation`               | Data Preparation             | Splits a `TimeSeries` into training and validation sets.                |
# ==================================================================================================================================================

# --- Standard Library Imports ---
from __future__ import annotations
import gc
from functools import reduce
from typing import Dict, List, Union

# --- Third-Party Imports ---
import pandas as pd
from darts import TimeSeries

# --- Local Application Imports ---
from common.forecasting import iterative_forecast_deep_encoder
from common.models import train_deep_encoder_model
from config import HYPERPARAM_DESCRIPTIONS
from evaluation.evaluation import evaluate_and_plot_results

# --- Darts Compatibility Import ---
# Handle `concatenate` for different Darts versions
try:
    # Available in Darts >= 0.25
    from darts.utils.utils import concatenate
except ImportError:
    # Fallback for older Darts versions
    def concatenate(series_seq: List[TimeSeries]) -> TimeSeries:
        """Concatenates a sequence of TimeSeries along the time axis."""
        if not series_seq:
            raise ValueError("The sequence of series to concatenate is empty.")
        return reduce(lambda a, b: a.append(b), series_seq)

try:
    # Available in Darts ≥ 0.25
    from darts.utils.utils import concatenate  # type: ignore
except ImportError:
    # Compatibility for older versions
    def concatenate(series_seq):
        """
        Concatenates a sequence of TimeSeries along the time axis.
        Equivalent to utils.concatenate() from newer versions.
        """
        series_seq = list(series_seq)
        if not series_seq:
            raise ValueError("The sequence of series is empty.")
        return reduce(lambda a, b: a.append(b), series_seq)


def prepare_time_series(
    dataframe: pd.DataFrame, target: str, covariates: list, train_size: int, horizon: int
) -> tuple:
    """Prepare training and testing TimeSeries with covariates."""
    # Set date index starting at Jan 1, 2020
    dataframe.index = pd.date_range(start="2020-01-01", periods=len(dataframe), freq="D")
    
    # Split data into train and test sets
    train_df, test_df = dataframe.iloc[:train_size], dataframe.iloc[train_size:]
    
    # Convert target and covariate columns to TimeSeries
    train_series = TimeSeries.from_series(train_df[target])
    test_series = TimeSeries.from_series(test_df[target])
    train_cov = TimeSeries.from_dataframe(train_df[covariates])
    test_cov = TimeSeries.from_dataframe(test_df[covariates])
    
    # Combine covariates from train and test
    full_covariates = train_cov.append(test_cov)
    return train_series, test_series, full_covariates


def prepare_time_series_full(
    dataframe: pd.DataFrame, target: str, covariates: list, train_size: int, horizon: int
) -> tuple:
    """Prepare full TimeSeries for target and covariates."""
    # Set date index starting at Jan 1, 2020
    dataframe.index = pd.date_range(start="2020-01-01", periods=len(dataframe), freq="D")
    
    full_series = TimeSeries.from_series(dataframe[target])
    full_covariates = TimeSeries.from_dataframe(dataframe[covariates])
    return full_series, full_covariates


def split_train_validation(series: TimeSeries, covariates: TimeSeries, validation_ratio: float = 0.6) -> tuple:
    """Split a TimeSeries and its covariates into train and validation parts."""
    train_series, val_series = series.split_after(validation_ratio)
    train_cov, val_cov = covariates.split_after(validation_ratio)
    return train_series, val_series, train_cov, val_cov


def run_sliding_window_forecasting(
    full_series: TimeSeries,
    full_covariates: TimeSeries,
    initial_train_size: int,
    forecast_horizon: int,
    validation_ratio: float,
    stride: int,
    model_type: str,
) -> tuple:
    """Run sliding window forecasting on the full TimeSeries, liberando memória em cada janela."""
    forecast_segments = []
    iteration = 1

    for window_start in range(initial_train_size, len(full_series) - forecast_horizon, stride):
        window_end = window_start + forecast_horizon

        # Define train and test sets for current window
        train_series = full_series[iteration:window_start]
        test_series = full_series[window_start:window_end]
        cov_train = full_covariates[:window_start]

        # Split training data into train/validation sets
        train, val, train_cov, val_cov = split_train_validation(train_series, cov_train, validation_ratio)

        # Print report every 100 iterations
        if iteration % 100 == 0:
            num_train_days = len(train_series)
            train_days = int(validation_ratio * num_train_days)
            val_days = num_train_days - train_days
            print(f"Iteration {iteration}:")
            print(f"  Training: Days {iteration} to {window_start}")
            print(f"  Test: Days {window_start} to {window_end} (Horizon: {forecast_horizon} days)")
            print(f"  Split: {train_days} train, {val_days} validation days")

        # --- Treinamento ---
        model = train_deep_encoder_model(
            train_series=train,
            train_covariates=train_cov,
            val_series=val,
            val_covariates=val_cov,
            model_type=model_type,
            output_chunk_length=forecast_horizon,
        )

        # --- Previsão ---
        forecast = model.predict(n=forecast_horizon, series=train_series, verbose=False, n_jobs=1)
        forecast_aligned = forecast.slice_intersect(test_series)

        # Salva só o necessário
        if window_start == initial_train_size:
            forecast_segments.append(forecast_aligned)
        else:
            forecast_segments.append(forecast_aligned[-stride:])

        # --- Libera objetos grandes logo após uso ---
        del model
        del train_series, test_series, cov_train
        del train, val, train_cov, val_cov
        del forecast, forecast_aligned
        gc.collect()  # Força coleta de lixo imediatamente

        iteration += stride

    # Concatenar os resultados finais
    if forecast_segments:
        full_forecast = concatenate(forecast_segments, ignore_time_axis=True)
        alignment_start = initial_train_size
        alignment_end = initial_train_size + len(full_forecast)
        aligned_test_series = full_series[alignment_start:alignment_end]
        full_forecast = full_forecast.slice_intersect(aligned_test_series)
    else:
        full_forecast = TimeSeries()
        aligned_test_series = TimeSeries()

    # Opcional: libera lista temporária de forecasts do loop
    del forecast_segments
    gc.collect()

    return full_forecast, aligned_test_series



def process_well(
    well: str,
    model_type: str,
    data_source: dict,
    initial_train_size: int,
    forecast_horizon: int,
    sampling_rate: int,
    metrics_accumulator,
    validation_ratio: float,
    stride: int,
    preloaded_data: pd.DataFrame = None,
) -> None:
    """Process a single well: prepare series, forecast, and evaluate results."""
    print(f"Processing well: {well}")
    df = preloaded_data
    df = df.dropna()[data_source['features']]

    target = data_source["target_column"]
    covariates = [col for col in data_source["features"] if col != target]

    full_series, full_covariates = prepare_time_series_full(
        dataframe=df, target=target, covariates=covariates, train_size=initial_train_size, horizon=forecast_horizon
    )
    print(f"Initial training size: {initial_train_size}")

    # Run sliding window forecasting
    full_forecast, aligned_test_series = run_sliding_window_forecasting(
        full_series, full_covariates, initial_train_size, forecast_horizon, validation_ratio, stride, model_type
    )

    # Prepare data for plotting
    test_series_plot = [aligned_test_series.values().flatten().tolist()]
    forecast_series_plot = [full_forecast.values().flatten().tolist()]

    # Calculate cumulative sum of training series (excluding last point)
    train_cum_sum = full_series[:initial_train_size].pd_series().cumsum()[:-1].iloc[-1]

    print(train_cum_sum)

    # Evaluate and plot results
    evaluate_and_plot_results(
        test_series=test_series_plot,
        forecast_series=forecast_series_plot,
        dataset=data_source["name"],
        well_name=well,
        lag_window=7,
        horizon=forecast_horizon,
        train_cumulative_sum=train_cum_sum,
        sampling_rate=sampling_rate,
        metrics_accumulator=metrics_accumulator,
        method=model_type,
        plot_cumulative=True,
    )


def process_data_source(
    data_source: dict,
    model_type: str,
    initial_train_size: int,
    forecast_horizon: int,
    sampling_rate: int,
    metrics_accumulator,
    validation_ratio: float,
    stride: int,
    preloaded_data,
) -> None:
    """Process an entire data source by iterating over all wells."""
    print(f"Processing data source: {data_source['name']}")
    if isinstance(preloaded_data, dict):
        # Iterate over each well and its DataFrame
        for well, df in preloaded_data.items():
            process_well(
                well, model_type, data_source, initial_train_size, forecast_horizon,
                sampling_rate, metrics_accumulator, validation_ratio, stride, preloaded_data=df
            )
    else:
        # Assume a single DataFrame for the first well
        process_well(
            data_source["wells"][0], model_type, data_source, initial_train_size, forecast_horizon,
            sampling_rate, metrics_accumulator, validation_ratio, stride, preloaded_data=preloaded_data
        )

        


def fast_iterative_forecast(
    model,
    train_series: TimeSeries,
    test_series: TimeSeries,
    full_covariates: TimeSeries,
    input_chunk_length: int,
    output_chunk_length: int,
) -> TimeSeries:
    """
    First iteration → full forecast horizon.
    Others         → only the last point.
    """
    series_total = train_series.append(test_series)

    # 1) Generate ALL forecasts (full horizon)
    forecasts = model.historical_forecasts(
        series=series_total,
        # past_covariates=full_covariates,
        start=len(train_series) - output_chunk_length + 1,
        forecast_horizon=output_chunk_length,
        stride=1,
        retrain=False,
        last_points_only=False,  # Returns the full horizon
        verbose=False,
    )


    # 2) Keep the 1st full horizon and only the last point for the others
    combined = [forecasts[0]] + [fc[-1:] for fc in forecasts[1:]]

    return concatenate(combined).slice_intersect(test_series)


def process_deep_encoder_well(
    well: str,
    data_source: dict,
    preloaded_data: pd.DataFrame,
    train_size: int,
    forecast_horizon: int,
    lag_window: int,
    sampling_rate: int,
    metrics_accumulator: list,
    model_type: str,          # <-- CHANGED: Now a single model string
    model_config_size: str,   # <-- NEW: The configuration profile to use
) -> None:
    """
    Processes a single well for a SINGLE model and a SINGLE configuration.
    This is the core function called by the papermill notebook.
    """
    # 1. Select the hyperparameters for this specific run
    if model_type not in HYPERPARAM_DESCRIPTIONS or model_config_size not in HYPERPARAM_DESCRIPTIONS[model_type]:
        print(f"Skipping {model_type} with config {model_config_size}: Not defined in config.py")
        return
        
    hyperparams = HYPERPARAM_DESCRIPTIONS[model_type][model_config_size]

    # 2. Data preparation (your existing logic)
    df = preloaded_data.copy()
    if data_source.get("variable_mapping"):
        df = df.rename(columns=data_source["variable_mapping"])
    features = [f for f in data_source["features"] if f in df.columns]
    df = df.dropna()[features]
    target_column = data_source["target_column"]
    covariate_columns = [c for c in features if c != target_column]
    train_series, test_series, full_covariates = prepare_time_series(
        dataframe=df, target=target_column, covariates=covariate_columns,
        train_size=train_size, horizon=forecast_horizon
    )

    # debug_sample_count = 10
    # if debug_sample_count is not None and debug_sample_count > 0:
    #     print(f"  [DEBUG MODE] Truncating test series to {debug_sample_count} samples.")
    #     test_series = test_series[:debug_sample_count]
    
    train, val = train_series.split_after(0.6)
    train_cov, val_cov = (full_covariates[:train_size].split_after(0.6)[0], full_covariates[:train_size].split_after(0.6)[1])

    print(f"\nWell: {well} | Model: {model_type} | Config: {model_config_size}")
    print(f"  > Hyperparameters: {hyperparams}")
    print(f"Training series length: {len(train)}")
    print(f"Validation series length: {len(val)}")

    # 3. Train the single specified model with the selected config
    model = train_deep_encoder_model(
        train_series=train,
        train_covariates=train_cov,
        val_series=val,
        val_covariates=val_cov,
        model_type=model_type,
        output_chunk_length=forecast_horizon,
        hyperparam_config=hyperparams, # <-- Pass the selected hyperparams
    )

    # 4. Fast forecast and evaluation (your existing logic)
    print(f"  > Forecasting with model: {model_type} for well: {well}")
    full_forecast = fast_iterative_forecast(
        model=model, train_series=train_series, test_series=test_series,
        full_covariates=full_covariates, input_chunk_length=lag_window,
        output_chunk_length=forecast_horizon
    )
    train_cum_sum = pd.Series(train_series.values().flatten()).cumsum()[:-1].iloc[-1]
    test_vals = [test_series.values().flatten().tolist()]
    pred_vals = [full_forecast.values().flatten().tolist()]

    # IMPORTANT: Pass the config size to your evaluation function
    evaluate_and_plot_results(
        test_series=test_vals, forecast_series=pred_vals,
        dataset=data_source["name"], well_name=well,
        lag_window=lag_window, horizon=forecast_horizon,
        train_cumulative_sum=train_cum_sum, sampling_rate=sampling_rate,
        metrics_accumulator=metrics_accumulator,
        method=model_type,
        model_config_size=model_config_size, # <-- Pass the new parameter
    )

    del model
    gc.collect()


def process_deep_encoder_data_source(
    data_source: dict,
    train_size: int,
    forecast_horizon: int,
    lag_window: int,
    sampling_rate: int,
    metrics_accumulator: list,
    preloaded_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    model_type: str,            
    model_config_size: str,   
) -> None:
    """
    Processes all wells from a data source for a SINGLE model and configuration.
    """
    print(f"\nProcessing data source: {data_source['name']} for {model_type} ({model_config_size})")
    wells = data_source['wells']
    
    # This loop remains, as a single job might still process multiple wells for the same model/config
    for well in wells:
        well_data = preloaded_data.get(well) if isinstance(preloaded_data, dict) else preloaded_data
        if well_data is not None:
            process_deep_encoder_well(
                well=well, data_source=data_source, preloaded_data=well_data,
                train_size=train_size, forecast_horizon=forecast_horizon,
                lag_window=lag_window, sampling_rate=sampling_rate,
                metrics_accumulator=metrics_accumulator,
                model_type=model_type,
                model_config_size=model_config_size,
            )
        else:
            print(f"Warning: Data for well '{well}' not found in preloaded_data.")
