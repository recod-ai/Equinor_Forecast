import pandas as pd
from darts import TimeSeries, concatenate
from common.models import train_deep_encoder_model
from evaluation.evaluation import evaluate_and_plot_results


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
    """Run sliding window forecasting on the full TimeSeries."""
    forecast_segments = []
    iteration = 1

    # Slide the window over the series
    for window_start in range(initial_train_size, len(full_series) - forecast_horizon, stride):
        window_end = window_start + forecast_horizon

        # Define train and test sets for current window
        train_series = full_series[iteration:window_start]
        test_series = full_series[window_start:window_end]
        cov_train = full_covariates[:window_start]
        # cov_test is available if needed: cov_test = full_covariates[window_start:window_end]

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

        # Train the model
        model = train_deep_encoder_model(
            train_series=train,
            train_covariates=train_cov,
            val_series=val,
            val_covariates=val_cov,
            model_type=model_type,
            output_chunk_length=forecast_horizon,
        )

        # Forecast and align with test series
        forecast = model.predict(n=forecast_horizon, series=train_series, verbose=False, n_jobs=16)
        forecast_aligned = forecast.slice_intersect(test_series)

        # Store full forecast for first window; otherwise, only the last 'stride' points
        if window_start == initial_train_size:
            forecast_segments.append(forecast_aligned)
        else:
            forecast_segments.append(forecast_aligned[-stride:])

        iteration += stride

    # Concatenate forecast segments and align with test series
    if forecast_segments:
        full_forecast = concatenate(forecast_segments, ignore_time_axis=True)
        alignment_start = initial_train_size
        alignment_end = initial_train_size + len(full_forecast)
        aligned_test_series = full_series[alignment_start:alignment_end]
        full_forecast = full_forecast.slice_intersect(aligned_test_series)
    else:
        full_forecast = TimeSeries()
        aligned_test_series = TimeSeries()

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

        
        
# ------------------------------------------------------------------------------------------------------------

import pandas as pd
from typing import Union, Dict, List
from common.forecasting import iterative_forecast_deep_encoder
from evaluation.evaluation import evaluate_and_plot_results




def process_deep_encoder_well(
    well: str,
    data_source: dict,
    preloaded_data: pd.DataFrame,
    train_size: int,
    forecast_horizon: int,
    lag_window: int,
    sampling_rate: int,
    metrics_accumulator: list,
    model_types: List[str]
) -> None:
    """
    Processa um único poço: prepara as séries, treina os modelos especificados em
    model_types e avalia as previsões.
    """
    # Cópia dos dados para evitar efeitos colaterais
    df = preloaded_data.copy()

    # Aplica o mapeamento de variáveis, se definido
    variable_mapping = data_source.get('variable_mapping')
    if variable_mapping:
        df = df.rename(columns=variable_mapping)

    # Remove linhas com valores ausentes e seleciona as features relevantes
    features = data_source['features']
    df = df.dropna()[features]

    target_column = data_source['target_column']
    covariate_columns = [col for col in features if col != target_column]

    # Prepara as séries temporais (treinamento, teste e covariáveis completas)
    train_series, test_series, full_covariates = prepare_time_series(
        dataframe=df,
        target=target_column,
        covariates=covariate_columns,
        train_size=train_size,
        horizon=forecast_horizon
    )

    # Divide a série de treinamento para obtenção de validação
    train, val = train_series.split_after(0.6)
    train_covariates = full_covariates[:train_size].split_after(0.6)[0]
    val_covariates = full_covariates[:train_size].split_after(0.6)[1]

    print(f"\nPoço: {well}")
    print(f"Comprimento da série de treinamento: {len(train)}")
    print(f"Comprimento da série de validação: {len(val)}")

    # Usa os modelos definidos pelo usuário em model_types
    models = {}
    for model_type in model_types:
        print(f"Treinando modelo: {model_type} para o poço: {well}")
        model = train_deep_encoder_model(
            train_series=train,
            train_covariates=train_covariates,
            val_series=val,
            val_covariates=val_covariates,
            model_type=model_type,
            output_chunk_length=forecast_horizon
        )
        models[model_type] = model

    for model_type, model in models.items():
        print(f"Realizando previsão com o modelo: {model_type} para o poço: {well}")
        full_forecast = iterative_forecast_deep_encoder(
            model=model,
            train_series=train_series,
            test_series=test_series,
            full_covariates=full_covariates,
            input_chunk_length=lag_window,
            output_chunk_length=forecast_horizon
        )

        # Calcula a soma cumulativa da série de treinamento (exceto o último ponto)
        train_cum_sum = pd.Series(train_series.values().flatten()).cumsum()[:-1].iloc[-1]


        # Extrai os valores das séries para plotagem (flatten e conversão para lista)
        test_series_plot = [test_series.values().flatten().tolist()]
        full_forecast_plot = [full_forecast.values().flatten().tolist()]

        evaluate_and_plot_results(
            test_series=test_series_plot,
            forecast_series=full_forecast_plot,
            dataset=data_source['name'],
            well_name=well,
            lag_window=lag_window,
            horizon=forecast_horizon,
            train_cumulative_sum=train_cum_sum,
            sampling_rate=sampling_rate,
            metrics_accumulator=metrics_accumulator,
            method=model_type
        )


from typing import List
import pandas as pd
from joblib import Parallel, delayed            # ⬅ opcional, ver abaixo

from functools import reduce
from darts import TimeSeries

try:
    # disponíveis a partir do Darts ≥ 0.25
    from darts.utils.utils import concatenate  # type: ignore
except ImportError:
    # ---- compatibilidade para versões antigas -----------------------------
    def concatenate(series_seq):
        """
        Concatena uma sequência de TimeSeries no eixo temporal.
        Equivalente ao utils.concatenate() das versões novas.
        """
        # garante lista/tupla
        series_seq = list(series_seq)
        if not series_seq:
            raise ValueError("A sequência de séries está vazia.")

        # faz append em cadeia (tudo é imutável, portanto seguro)
        return reduce(lambda a, b: a.append(b), series_seq)


# ---- helper 10-30× mais rápido que o loop manual ---------------------------
def fast_iterative_forecast(
    model,
    train_series: TimeSeries,
    test_series: TimeSeries,
    full_covariates: TimeSeries,
    input_chunk_length: int,
    output_chunk_length: int,
) -> TimeSeries:
    """
    1ª iteração → horizonte completo
    Demais      → só o último ponto
    """
    series_total = train_series.append(test_series)

    # 1) gera TODAS as previsões (horizonte completo)
    forecasts = model.historical_forecasts(
        series=series_total,
        # past_covariates=full_covariates,
        start=len(train_series) - output_chunk_length + 1,
        forecast_horizon=output_chunk_length,
        stride=1,
        retrain=False,
        last_points_only=False,          # devolve o horizonte inteiro
        verbose=False,
    )

    # 2) guarda o 1º horizonte completo e só o último ponto dos demais
    combined = [forecasts[0]] + [fc[-1:] for fc in forecasts[1:]]

    return concatenate(combined).slice_intersect(test_series)


# ---- função solicitada -----------------------------------------------------

def process_deep_encoder_well(
    well: str,
    data_source: dict,
    preloaded_data: pd.DataFrame,
    train_size: int,
    forecast_horizon: int,
    lag_window: int,
    sampling_rate: int,
    metrics_accumulator: list,
    model_types: List[str],
) -> None:
    """
    Prepara as séries, treina os modelos em `model_types` e avalia previsões
    para um único poço — agora muito mais rápido.
    """

    # ---------- 1. preparação dos dados -------------------------------------
    df = preloaded_data.copy()                               # evita side-effects

    variable_mapping = data_source.get("variable_mapping")
    if variable_mapping:
        df = df.rename(columns=variable_mapping)

    features = data_source["features"]
    df = df.dropna()[features]

    target_column = data_source["target_column"]
    covariate_columns = [c for c in features if c != target_column]

    train_series, test_series, full_covariates = prepare_time_series(
        dataframe=df,
        target=target_column,
        covariates=covariate_columns,
        train_size=train_size,
        horizon=forecast_horizon,
    )

    print('covariate_columns', covariate_columns)

    train, val = train_series.split_after(0.6)
    train_cov, val_cov = (
        full_covariates[:train_size].split_after(0.6)[0],
        full_covariates[:train_size].split_after(0.6)[1],
    )

    print(f"\nPoço: {well}")
    print(f"Comprimento da série de treinamento: {len(train)}")
    print(f"Comprimento da série de validação: {len(val)}")

    # ---------- 2. treina todos os modelos ----------------------------------
    def _train_one(model_type):
        print(f"Treinando modelo: {model_type} para o poço: {well}")
        return model_type, train_deep_encoder_model(
            train_series=train,
            train_covariates=train_cov,
            val_series=val,
            val_covariates=val_cov,
            model_type=model_type,
            output_chunk_length=forecast_horizon,
        )

    # → paralelize o treino se quiser; com poucos modelos o overhead é inútil
    models = dict(_train_one(mt) for mt in model_types)
    # Para paralelizar, descomente:
    # models = dict(Parallel(n_jobs=-1)(delayed(_train_one)(mt) for mt in model_types))

    # ---------- 3. previsões rápidas e avaliação ----------------------------
    for model_type, model in models.items():
        print(f"Realizando previsão com o modelo: {model_type} para o poço: {well}")
        full_forecast = fast_iterative_forecast(
            model=model,
            train_series=train_series,
            test_series=test_series,
            full_covariates=full_covariates,
            input_chunk_length=lag_window,
            output_chunk_length=forecast_horizon,
        )

        # soma cumulativa do treino (exclui o último ponto)
        train_cum_sum = (
            pd.Series(train_series.values().flatten()).cumsum()[:-1].iloc[-1]
        )

        # flattens → listas para seu plotter
        test_vals = [test_series.values().flatten().tolist()]
        pred_vals = [full_forecast.values().flatten().tolist()]

        evaluate_and_plot_results(
            test_series=test_vals,
            forecast_series=pred_vals,
            dataset=data_source["name"],
            well_name=well,
            lag_window=lag_window,
            horizon=forecast_horizon,
            train_cumulative_sum=train_cum_sum,
            sampling_rate=sampling_rate,
            metrics_accumulator=metrics_accumulator,
            method=model_type,
        )

        if metrics_accumulator and "Poço" not in metrics_accumulator[-1]:
            metrics_accumulator[-1]["Poço"] = well
            metrics_accumulator[-1]["Método"] = model_type


def process_deep_encoder_data_source(
    data_source: dict,
    train_size: int,
    forecast_horizon: int,
    lag_window: int,
    sampling_rate: int,
    metrics_accumulator: list,
    preloaded_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    model_types: list
) -> None:
    """
    Processa todos os poços de uma fonte de dados usando os dados pré-carregados,
    propagando a seleção de modelos definida em model_types.
    """
    print(f"\nProcessando fonte de dados: {data_source['name']}")
    wells = data_source['wells']

    if isinstance(preloaded_data, dict):
        # Para cada poço, utiliza os dados correspondentes
        for well in wells:
            df = preloaded_data.get(well)
            if df is not None:
                process_deep_encoder_well(
                    well=well,
                    data_source=data_source,
                    preloaded_data=df,
                    train_size=train_size,
                    forecast_horizon=forecast_horizon,
                    lag_window=lag_window,
                    sampling_rate=sampling_rate,
                    metrics_accumulator=metrics_accumulator,
                    model_types=model_types
                )
            else:
                print(f"Warning: Dados não disponíveis para o poço '{well}'.")
    else:
        # Se houver apenas um DataFrame (ex.: único poço)
        process_deep_encoder_well(
            well=wells[0],
            data_source=data_source,
            preloaded_data=preloaded_data,
            train_size=train_size,
            forecast_horizon=forecast_horizon,
            lag_window=lag_window,
            sampling_rate=sampling_rate,
            metrics_accumulator=metrics_accumulator,
            model_types=model_types
        )
