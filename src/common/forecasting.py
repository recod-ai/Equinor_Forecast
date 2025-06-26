from darts import concatenate, TimeSeries
import pandas as pd

def iterative_forecast_deep_encoder(
    model: object, 
    train_series: TimeSeries, 
    test_series: TimeSeries, 
    full_covariates: TimeSeries,
    input_chunk_length: int, 
    output_chunk_length: int
) -> TimeSeries:
    """
    Realiza a validação walk-forward utilizando o modelo deep encoder treinado com covariáveis passadas.

    Args:
        model (object): Modelo deep encoder treinado (e.g., TiDEModel, NHiTSModel).
        train_series (TimeSeries): Série temporal de treinamento.
        test_series (TimeSeries): Série temporal de teste para previsão.
        full_covariates (TimeSeries): Todas as covariáveis (treino + teste).
        input_chunk_length (int): Número de passos de tempo passados usados como entrada para o modelo.
        output_chunk_length (int): Número de passos de tempo futuros que o modelo prevê.

    Returns:
        TimeSeries: Objeto TimeSeries concatenado contendo todos os valores previstos.
    """
    # Cria uma cópia da série de treinamento para evitar modificar os dados originais
    current_series = train_series.copy()
    
    # Inicializa uma lista para armazenar as previsões individuais
    forecasts = []

    # Número total de passos na série de teste
    n_test = len(test_series)
    
    # Itera sobre cada passo de tempo na série de teste
    for t in range(n_test):
        # Define o início e fim da janela de covariáveis para a previsão atual        
        forecast_start_time = current_series.end_time() - pd.Timedelta(days=input_chunk_length)
        forecast_end_time = current_series.end_time() + pd.Timedelta(days=output_chunk_length)
        
        # Extrair as covariáveis correspondentes ao intervalo de previsão
        covariate_slice = full_covariates.slice(forecast_start_time, forecast_end_time)

        # Faz a previsão para o próximo horizonte
        forecast = model.predict(
            n=output_chunk_length, 
            series=current_series, 
            # past_covariates=covariate_slice,
            verbose=False
        )

        # Extrai a previsão (a previsão mais futura)
        if t != 0:
            last_pred = forecast[-1:]
        else:
            last_pred = forecast

        # Adiciona a previsão à lista de previsões
        forecasts.append(last_pred)

        # Recupera o valor real da série de teste para o próximo passo
        actual_next = test_series[t]

        # Adiciona o valor real à série atual para a próxima iteração
        current_series = current_series.append(actual_next)

    # Concatena todas as previsões individuais em uma única série temporal
    full_forecast = concatenate(forecasts)

    # Alinha a série prevista com o intervalo de tempo da série de teste
    full_forecast = full_forecast.slice_intersect(test_series)

    return full_forecast