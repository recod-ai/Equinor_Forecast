from typing import Any, List, Dict, Callable
import numpy as np
from sklearn.metrics import r2_score


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import r2_score, mean_absolute_error


def plot_time_series(df, serie_name, well=None):
    """
    Plot a time series with an artistic and creative design using Plotly.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the time series data.
        serie_name (str): Name of the time series.
        well (str): Well identifier.

    Returns:
        None
    """
    import numpy as np
    import plotly.graph_objects as go

    array = np.array(df).flatten()

    # Use the index as x-axis
    x = np.arange(len(array))

    # Create plot figure
    fig = go.Figure()

    # Add a filled line trace with a gradient effect
    fig.add_trace(go.Scatter(
        x=x,
        y=array,
        mode='lines',
        name=serie_name,
        line=dict(color='#206A92', width=3),
    ))

    title = f"{serie_name} - Serie: {well}"

    # Update layout with artistic elements
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=36, color='#2E2E2E')
        ),
        xaxis=dict(
            title="Days",
            title_font=dict(size=30, color='#2E2E2E'),
            tickfont=dict(size=20, color='#2E2E2E'),
            gridcolor='rgba(200,200,200,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            title="Production",
            title_font=dict(size=30, color='#2E2E2E'),
            tickfont=dict(size=20, color='#2E2E2E'),
            gridcolor='rgba(200,200,200,0.2)',
            zeroline=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        width=1200,
        height=600,
        showlegend=False
    )

    # Add custom hover information
    fig.update_traces(
        hovertemplate='<b>Days:</b> %{x}<br><b>Value:</b> %{y:.2f}'
    )


    # Show the figure
    fig.show()


def compute_metrics(y_test_list: List[np.ndarray], y_pred_list: List[np.ndarray], wells: List[str]) -> pd.DataFrame:
    """
    Computa as métricas MAE e SMAPE para cada poço.

    Parâmetros:
    - y_test_list (List[np.ndarray]): Lista de valores reais por poço.
    - y_pred_list (List[np.ndarray]): Lista de valores previstos por poço.
    - wells (List[str]): Lista de nomes dos poços.

    Retorna:
    - pd.DataFrame: DataFrame contendo MAE e SMAPE para cada poço.
    """
    data = []
    for well, y_test, y_pred in zip(wells, y_test_list, y_pred_list):
        _, smape_score, mae = evaluate(y_test, y_pred)
        data.append({'Poço': well, 'MAE': mae, 'SMAPE': smape_score})
    metrics_df = pd.DataFrame(data)
    return metrics_df


def compute_metrics_to_df(
    y_test: np.ndarray, 
    y_pred: np.ndarray, 
    well: str, 
    method: str, 
    metric_type: str = "Series"
) -> Dict[str, Any]:
    """
    Computa as métricas R², SMAPE e MAE para uma série de dados.
    
    Parâmetros:
    - y_test (np.ndarray): Valores reais.
    - y_pred (np.ndarray): Valores previstos.
    - well (str): Nome do poço.
    - method (str): Nome do método/modelo utilizado.
    - metric_type (str): Tipo da métrica ("Series" ou "Cumulative").
    
    Retorna:
    - Dict[str, Any]: Dicionário contendo as métricas calculadas.
    """
    
    metrics = evaluate_return_dict(y_test, y_pred)
    metrics.update({
        'Well': well, 
        'Method': method, 
        'Category': metric_type
    })
    return metrics


# Função evaluate_and_plot_results com o parâmetro plot_cumulative adicionado
def evaluate_and_plot_results(
    test_series: 'TimeSeries', 
    forecast_series: 'TimeSeries',
    dataset: str,
    well_name: str, 
    lag_window: int, 
    horizon: int,
    train_cumulative_sum: float,
    sampling_rate: int,
    metrics_accumulator: List[Dict[str, Any]],
    method: str,
    plot_cumulative: bool = True  # Parâmetro adicionado
):
    """
    Avalia e plota os resultados da previsão.
    
    Args:
        test_series (TimeSeries): Série de teste real.
        forecast_series (TimeSeries): Série de previsão.
        well_name (str): Nome do poço.
        lag_window (int): Tamanho da janela de lag.
        horizon (int): Horizonte de previsão.
        train_cumulative_sum (float): Soma cumulativa da série de treinamento.
        sampling_rate (int): Taxa de amostragem para plotagem.
        metrics_accumulator (List[Dict[str, Any]]): Lista para acumular as métricas.
        method (str): Método/Algoritmo utilizado.
        plot_cumulative (bool): Se True, plota a soma cumulativa.
    
    Returns:
        None
    """
    
    actual = test_series
    predicted = forecast_series
    
    # Avaliar e plotar a série principal
    print(f"Avaliando e plotando série principal para o poço: {well_name}")
    metrics_series = compute_metrics_to_df(
        y_test=actual, 
        y_pred=predicted, 
        well=well_name, 
        method=method, 
        metric_type="Series"
    )
    if metrics_series and not plot_cumulative:
        metrics_accumulator.append(metrics_series)
    plot_results(
        y_test_list=actual,
        y_pred_list=predicted,
        wells=[well_name],
        window_size=lag_window,
        forecast_steps=horizon,
        dataset=dataset,
    )
    
    if plot_cumulative:
        # Calcular as somas acumuladas
        
        test_series = [item for sublist in test_series for item in sublist]
        forecast_series = [item for sublist in forecast_series for item in sublist]
        
        actual_cumsum = pd.concat([pd.Series([train_cumulative_sum]), pd.Series(test_series)]).cumsum()
        predicted_cumsum = pd.concat([pd.Series([train_cumulative_sum]), pd.Series(forecast_series)]).cumsum()
        
        # Downsample das séries para plotagem se necessário
        actual_cumsum = actual_cumsum[::sampling_rate].values
        predicted_cumsum = predicted_cumsum[::sampling_rate].values
        
        # Avaliar e plotar a soma cumulativa
        print(f"\nAvaliando e plotando soma cumulativa para o poço: {well_name}")
        metrics_cumsum = compute_metrics_to_df(
            y_test=actual_cumsum, 
            y_pred=predicted_cumsum, 
            well=well_name, 
            method=method, 
            metric_type="Cumulative"
        )
        if metrics_cumsum:
            metrics_accumulator.append(metrics_cumsum)
        plot_results(
            y_test_list=[actual_cumsum],
            y_pred_list=[predicted_cumsum],
            wells=[well_name],
            window_size=lag_window,
            forecast_steps=horizon,
            dataset=dataset,
        )


def display_metrics(metrics: List[Dict[str, Any]]):
    """
    Exibe as métricas acumuladas em um DataFrame de forma organizada e estilizada.
    
    Parâmetros:
    - metrics (List[Dict[str, Any]]): Lista de dicionários contendo as métricas.
    
    Retorna:
    - None
    """
    if not metrics:
        print("\nNenhuma métrica para exibir.")
        return
    
    # Cria o DataFrame a partir da lista de dicionários
    metrics_df = pd.DataFrame(metrics)
    
    # Reorganiza as colunas para melhor visualização, se existirem
    expected_columns = ['Poço', 'Método', 'Tipo', 'R²', 'SMAPE', 'MAE']
    existing_columns = [col for col in expected_columns if col in metrics_df.columns]
    metrics_df = metrics_df[existing_columns]
    
    # Estiliza o DataFrame para uma visualização mais elegante (funciona melhor em Jupyter Notebooks)
    styled_metrics = (
        metrics_df
        .style
        .format({
            'R²': "{:.4f}",
            'SMAPE': "{:.2f}%",
            'MAE': "{:.4f}"
        })
        .set_properties(**{
            'font-size': '12pt',
            'text-align': 'center'
        })
        .background_gradient(cmap='coolwarm', subset=['R²', 'SMAPE', 'MAE'])
        .set_table_styles([
            {'selector': 'th', 'props': [('font-size', '14pt'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('font-size', '12pt'), ('text-align', 'center')]}
        ])
    )
    
    # Tenta exibir o DataFrame estilizado; se não for possível (ambiente não suporta), exibe como tabela simples
    try:
        from IPython.display import display
        display(styled_metrics)
    except:
        print(metrics_df)
    
    return metrics_df


def plot_results(
    y_test_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    wells: List[str],
    window_size: int,
    forecast_steps: int,
    dataset: str
) -> None:
    """
    Plota as previsões para cada poço.

    Parâmetros:
    - y_test_list (List[np.ndarray]): Lista de valores reais por poço.
    - y_pred_list (List[np.ndarray]): Lista de valores previstos por poço.
    - wells (List[str]): Lista de nomes dos poços.
    - window_size (int): Tamanho da janela utilizada no modelo.
    - forecast_steps (int): Número de passos de previsão.
    - series_name (str): Nome da série sendo prevista.

    Retorna:
    - None
    """
    for well, y_test, y_pred in zip(wells, y_test_list, y_pred_list):
        plot_predictions(
            y_test,
            y_pred,
            dataset,
            well,
            set_name=f"Test Set for {well}",
            smape=smape(y_test, y_pred),
            mae=mean_absolute_error(y_test, y_pred),
            window_size=window_size,
            forecast_steps=forecast_steps,
            percentage_split=None
        )

def evaluate(y_test, y_pred):
    """
    Avalia o modelo no conjunto de teste usando R², SMAPE e MAE.

    Parâmetros:
    - y_test: Valores reais (array ou lista).
    - y_pred: Valores previstos pelo modelo (array ou lista).

    Retorna:
    - r2_score_test: O R² do conjunto de teste.
    - smape_score_test: O SMAPE do conjunto de teste.
    - mae_score_test: O MAE (Mean Absolute Error) do conjunto de teste.
    """
    r2_score_test = r2_score(y_test, y_pred)
    smape_score_test = smape(y_test, y_pred)
    mae_score_test = mean_absolute_error(y_test, y_pred)
    
    print(f"R² on the test set: {r2_score_test:.4f}")
    print(f"SMAPE on the test set: {smape_score_test:.4f}")
    print(f"MAE on the test set: {mae_score_test:.4f}")
    
    return r2_score_test, smape_score_test, mae_score_test


def evaluate_return_dict(y_test, y_pred) -> Dict[str, float]:
    """
    Avalia o modelo no conjunto de teste usando R², SMAPE e MAE.
    
    Parâmetros:
    - y_test: Valores reais (array ou lista).
    - y_pred: Valores previstos (array ou lista).
    
    Retorna:
    - Dict[str, float]: Dicionário contendo as métricas calculadas.
    """
    # Verifica e converte os dados para np.ndarray, se necessário
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Remove dimensões extras, se necessário
    if y_test.ndim > 1 and y_test.shape[0] == 1:
        y_test = y_test.flatten()
    if y_pred.ndim > 1 and y_pred.shape[0] == 1:
        y_pred = y_pred.flatten()
    
    r2 = r2_score(y_test, y_pred)
    smape_score = smape(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"R² on the test set: {r2:.4f}")
    print(f"SMAPE on the test set: {smape_score:.4f}%")
    print(f"MAE on the test set: {mae:.4f}")
    
    return {'R²': r2, 'SMAPE': smape_score, 'MAE': mae}


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Parâmetros:
    - y_true (np.ndarray): Valores reais.
    - y_pred (np.ndarray): Valores previstos.
    
    Retorna:
    - float: Valor do SMAPE em porcentagem.
    """
    
        # Verifica e converte os dados para np.ndarray, se necessário
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # Evita divisão por zero
    return np.mean(diff) * 100

def evaluate_and_plot_if_needed(
    y_test_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    wells: List[str],
    window_size: int,
    forecast_steps: int,
    series_name: str,
) -> None:
    """
    Evaluates and plots the predictions at specified intervals.

    Parameters:
    - control_iteration (int): The current iteration index.
    - y_test_list (list of np.ndarray): List containing true target values for each well.
    - y_pred_list (list of np.ndarray): List containing predicted values for each well.
    - wells (list of str): List of well identifiers.
    - window_size (int): The window size used in the model.
    - forecast_steps (int): The number of steps ahead being forecasted.
    - series_name (str): The name of the target series being predicted.
    - evaluation_interval (int, optional): The interval at which to evaluate and plot. Defaults to 140.
    - first_evaluation (int, optional): The iteration at which to perform the first evaluation. Defaults to 14.

    Returns:
    - None
    """
    # Evaluate and plot at specified intervals
    for i, (well, y_test, y_pred) in enumerate(zip(wells, y_test_list, y_pred_list)):
        # Check if there are predictions to evaluate
        if not y_test or not y_pred:
            print(f"No predictions to evaluate for well {well}")
            continue

        # Evaluate the predictions for the current well
        _, smape_score, mae = evaluate(y_test, y_pred)

        # Plot the predictions with evaluation metrics
        plot_predictions(
            y_test,
            y_pred,
            dataset,
            well,
            set_name=f"Test Set {i + 1}",
            smape=smape_score,
            mae=mae,
            window_size=window_size,
            forecast_steps=forecast_steps,
            percentage_split=None
        )
            
            
def plot_predictions(y, y_pred, dataset, well, set_name="Test Set", smape=0, mae=0, window_size=3, forecast_steps=28, percentage_split=None):
    """
    Plot a time series with predicted values using an innovative and eye-catching style.
    """
    import numpy as np
    import plotly.graph_objects as go

    # Check if y and y_pred have the same length
    if len(y) != len(y_pred):
        raise ValueError("y and y_pred must have the same length")
    
    # Use the index as x-axis
    x = np.arange(len(y))

    # Create plot figure
    fig = go.Figure()

    # Add filled area between actual and predicted values
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Actual Values',
        line=dict(color='#206A92', width=6),
        fill=None
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y_pred,
        mode='lines',
        name='Predicted Values',
        line=dict(color='yellowgreen', width=6, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(154,205,50,0.2)'  # Semi-transparent fill between lines
    ))

    title = set_name

    # Update layout with a modern design
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(
            color='#2E2E2E',
            size=36
        )),
        xaxis_title="Days",
        yaxis_title="Value",
        showlegend=True,
        legend=dict(orientation="h", x=0.5, y=1.1, font=dict(size=28), xanchor='center'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        width=1200,
        height=600,
    )

    # Add annotations for SMAPE, R², and additional information
    annotations = [
        (f"SMAPE: {smape:.2f}%", "#206A92"),
        (f"MAE: {mae:.2f}", "yellowgreen")
    ]
    
    if window_size is not None:
        annotations.append((f"Windows: {window_size}", "#2E2E2E"))
    if forecast_steps is not None:
        annotations.append((f"Steps: {forecast_steps}", "#2E2E2E"))
    if percentage_split is not None:
        annotations.append((f"Train: {percentage_split*100:.0f}%", "#2E2E2E"))

    for i, (text, color) in enumerate(annotations):
        fig.add_annotation(
            x=0.02,
            y=0.98 - i * 0.12,
            text=text,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=28, color=color),
            align='left',
            bordercolor='rgba(0,0,0,0)',
            bgcolor='rgba(255,255,255,0.8)'
        )

    # Update axes with a sleek style
    fig.update_xaxes(
        title_font=dict(size=30, color='#2E2E2E'),
        tickfont=dict(size=26, color='#2E2E2E'),
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    )
    fig.update_yaxes(
        title_font=dict(size=30, color='#2E2E2E'),
        tickfont=dict(size=26, color='#2E2E2E'),
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    )

    # Enhance interactivity with custom hover information
    fig.update_traces(
        hovertemplate='<b>Time Step:</b> %{x}<br><b>Value:</b> %{y:.2f}'
    )

    # Show the figure
    fig.show()
    
from typing import Optional   
# Função genérica de plotagem
def plot_series(
    well: str,
    title: str,
    series: List,
    series_names: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    line_styles: Optional[List[str]] = None,
    x: Optional[np.ndarray] = None
):
    """
    Plota uma ou várias séries temporais de forma genérica e elegante.
    
    Args:
        well (str): Nome do poço para incluir no título do gráfico.
        title (str): Título principal do gráfico.
        *series (iterable): Séries temporais a serem plotadas.
        series_names (List[str], optional): Lista de nomes para as séries. Se não especificada, serão usados 'Serie 1', 'Serie 2', etc.
        colors (List[str], optional): Lista de cores para as séries. Se não especificado, será usada uma paleta padrão.
        line_styles (List[str], optional): Lista de estilos de linha para as séries. Opções comuns: 'solid', 'dash', 'dot'.
        x (np.ndarray, optional): Valores para o eixo x. Se não especificado, será usado np.arange com base no comprimento da primeira série.
    """
    # Validação das séries
    if not series:
        raise ValueError("Nenhuma série fornecida para plotagem.")
    
    num_series = len(series)
    
    # Definir nomes das séries
    if series_names:
        if len(series_names) != num_series:
            raise ValueError("O número de nomes fornecidos não corresponde ao número de séries.")
        names = series_names
    else:
        names = [f"Serie {i+1}" for i in range(num_series)]
    
    # Definir cores padrão se não especificadas
    default_colors = [
        '#206A92', 'yellowgreen', '#FF5733', '#33FFCE',
        '#8E44AD', '#E67E22', '#2ECC71', '#3498DB',
        '#E74C3C', '#1ABC9C'
    ]
    if colors:
        if len(colors) < num_series:
            raise ValueError("Número de cores fornecidas é menor que o número de séries.")
        plot_colors = colors
    else:
        plot_colors = default_colors[:num_series]
    
    # Definir estilos de linha padrão
    if line_styles:
        if len(line_styles) < num_series:
            raise ValueError("Número de estilos de linha fornecidos é menor que o número de séries.")
        styles = line_styles
    else:
        styles = ['solid'] * num_series
    
    # Definir valores do eixo x
    if x is not None:
        if len(x) != len(series[0]):
            raise ValueError("O comprimento de 'x' deve corresponder ao comprimento das séries.")
        x_values = x
    else:
        x_values = np.arange(len(series[0]))
    
    # Criar a figura
    fig = go.Figure()
    
    # Adicionar cada série como um trace
    for i in range(num_series):
        fig.add_trace(go.Scatter(
            x=x_values,
            y=series[i],
            mode='lines',
            name=names[i],
            line=dict(color=plot_colors[i], width=4, dash=styles[i]),
            fill=None
        ))
    
    # Atualizar o layout com um design moderno
    fig.update_layout(
        title=dict(text=f"{title}: Serie(s) - Poço: {well}", x=0.5, font=dict(
            color='#2E2E2E',
            size=36
        )),
        xaxis_title="Dias",
        yaxis_title="Valor",
        showlegend=True,
        legend=dict(orientation="h", x=0.5, y=1.1, font=dict(size=28), xanchor='center'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        width=1200,
        height=600,
    )
    
    # Atualizar os eixos com um estilo elegante
    fig.update_xaxes(
        title_font=dict(size=30, color='#2E2E2E'),
        tickfont=dict(size=20, color='#2E2E2E'),
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    )
    fig.update_yaxes(
        title_font=dict(size=30, color='#2E2E2E'),
        tickfont=dict(size=20, color='#2E2E2E'),
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    )
    
    # Melhorar a interatividade com informações de hover personalizadas
    fig.update_traces(
        hovertemplate='<b>Time Step:</b> %{x}<br><b>Value:</b> %{y:.2f}'
    )
    
    # Mostrar a figura
    fig.show()
    

    
from IPython.display import display
import plotly.graph_objects as go
    
    
def analyze_correlations(
    df: pd.DataFrame,
    series_compared: str = 'BORE_OIL_VOL',
    top_n: int = 5
) -> List[str]:
    """
    Analisa e exibe a correlação da série especificada com as demais variáveis do DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        series_compared (str, optional): Nome da série a ser comparada. Defaults to 'BORE_OIL_VOL'.
        top_n (int, optional): Número de features mais correlacionadas a serem retornadas. Defaults to 5.
    
    Returns:
        List[str]: Lista com os nomes das `top_n` features mais correlacionadas.
    """
    # Verifica se a série especificada existe no DataFrame
    if series_compared not in df.columns:
        raise ValueError(f"A série '{series_compared}' não existe no DataFrame.")
    
    # Calcula a matriz de correlação
    correlation_matrix = df.corr()
    
    # Extrai as correlações da série especificada com as demais
    series_corr = correlation_matrix[series_compared].drop(labels=[series_compared])
    
    # Ordena as correlações pelo valor absoluto, de forma decrescente
    series_corr_sorted = series_corr.abs().sort_values(ascending=False)
    
    # Seleciona as top_n features mais correlacionadas
    top_features = series_corr_sorted.head(top_n).index.tolist()
    
    # Cria um DataFrame para exibir as correlações ordenadas com coluna 'Feature'
    display_corr = correlation_matrix[[series_compared]].drop(labels=[series_compared])
    display_corr = display_corr.loc[top_features].reset_index()
    display_corr.columns = ['Feature', 'Correlation']
    
    # Estiliza o DataFrame para uma visualização mais elegante (funciona melhor em Jupyter Notebooks)
    styled_corr = (
        display_corr
        .style
        .format({'Correlation': "{:.2f}"})
        .set_properties(**{
            'font-size': '16pt',
            'text-align': 'center'
        })
        .background_gradient(cmap='coolwarm', subset=['Correlation'])
        .set_table_styles([
            {'selector': 'th', 'props': [('font-size', '18pt'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('font-size', '16pt'), ('text-align', 'center')]}
        ])
    )
    
    # Tenta exibir o DataFrame estilizado; se não for possível (ambiente não suporta), exibe como tabela simples
    try:
        display(styled_corr)
    except:
        print(display_corr)
    
    return top_features


from typing import Tuple

def evaluate_and_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, well: str,
                       set_name: str, additional_params: dict):
    from evaluation.evaluation import evaluate
    from forecast_pipeline.ensemble_output import EnsembleOutput
    from forecast_pipeline.plotting import plot_predictions_wrapper
    r2, smape, mae = evaluate(y_true, y_pred)
    plot_series = plot_predictions_wrapper  # alias; usa wrapper novo
    plot_series(
        EnsembleOutput(pred_test=y_pred, pred_val=y_pred),
        truth=y_true,
        kind="P50",
        well=well,
    )
    return r2, smape, mae


def evaluate_and_plot_all_wells(
    dataset:str,
    wells: List[str],
    y_test_list: List[List[float]],
    y_pred_list: List[List[float]],
    window_size: int,
    forecast_steps: int,
    metrics_accumulator: List,
    method: str
) -> None:
    """
    Evaluates and plots results for all wells.

    Parameters:
    - wells (List[str]): List of well names.
    - y_test_list (List[List[float]]): List of test data per well.
    - y_pred_list (List[List[float]]): List of prediction data per well.
    - window_size (int): Size of the sliding window.
    - forecast_steps (int): Forecast steps.
    - metrics_accumulator (List): Accumulator for metrics.
    - method (str): Method name for labeling.
    """
    for i, well_name in enumerate(wells):
        y_test = [y_test_list[i]]
        y_pred = [y_pred_list[i]]

        evaluate_and_plot_results(
            test_series=y_test,
            forecast_series=y_pred,
            dataset=dataset,
            well_name=well_name,
            lag_window=window_size,
            horizon=forecast_steps,
            train_cumulative_sum=0.0,  # Adjust as needed
            sampling_rate=1,           # Adjust as needed
            metrics_accumulator=metrics_accumulator,
            method=method,
            plot_cumulative=False
        )
    print(f"\nMetrics per Well ({method}):")
    display_metrics(metrics_accumulator)
    

# File: src/evaluation/evaluation.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# File: src/evaluation/evaluation.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def history_evaluation(history, metrics=None):
    """
    Plot the training evolution from a Keras History object using Plotly.

    This function generates an interactive plot that displays the evolution
    of training metrics (e.g., loss, accuracy) over epochs, comparing training
    and validation curves. The visual style is inspired by the `plot_predictions`
    function.

    Parameters:
        history : History
            The History object returned from model.fit().
        metrics : list of str, optional
            List of metric names to plot (e.g., ['loss', 'accuracy']). If None,
            defaults to ['loss'] if available; otherwise, all training metrics
            (keys not starting with 'val_') will be used.

    Returns:
        fig : plotly.graph_objects.Figure
            The Plotly figure containing the training evolution plot.
    """
    # Extract the history dictionary
    history_dict = history.history
    
    # Determine which metrics to plot
    if metrics is None:
        if 'loss' in history_dict:
            metrics = ['loss']
        else:
            # Use all training metrics (exclude keys starting with 'val_')
            metrics = [key for key in history_dict.keys() if not key.startswith('val_')]
    
    num_plots = len(metrics)
    
    # Create subplots: one row per metric
    fig = make_subplots(
        rows=num_plots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f"{m.capitalize()} Evolution" for m in metrics]
    )
    
    epochs = len(history_dict[metrics[0]])
    x = np.arange(1, epochs + 1)

    # Define visual style parameters
    training_color = '#206A92'
    validation_color = 'yellowgreen'
    fill_color = 'rgba(154,205,50,0.2)'  # Semi-transparent fill

    for i, metric in enumerate(metrics):
        row = i + 1

        # Add training metric trace
        fig.add_trace(go.Scatter(
            x=x,
            y=history_dict[metric],
            mode='lines',
            name=f'Training {metric.capitalize()}',
            line=dict(color=training_color, width=6)
        ), row=row, col=1)

        # Add validation metric trace if available
        val_metric = f'val_{metric}'
        if val_metric in history_dict:
            fig.add_trace(go.Scatter(
                x=x,
                y=history_dict[val_metric],
                mode='lines',
                name=f'Validation {metric.capitalize()}',
                line=dict(color=validation_color, width=6, dash='dot'),
                fill='tonexty',
                fillcolor=fill_color
            ), row=row, col=1)

            # Add annotation for best validation performance
            if metric.lower() == 'loss':
                best_epoch = np.argmin(history_dict[val_metric]) + 1
                best_val = np.min(history_dict[val_metric])
                annotation_text = f"Min Val {metric.capitalize()}: {best_val:.4f} (Epoch {best_epoch})"
            elif metric.lower() in ['accuracy', 'acc']:
                best_epoch = np.argmax(history_dict[val_metric]) + 1
                best_val = np.max(history_dict[val_metric])
                annotation_text = f"Max Val {metric.capitalize()}: {best_val:.4f} (Epoch {best_epoch})"
            else:
                annotation_text = None

            if annotation_text is not None:
                fig.add_annotation(
                    x=0.98,
                    y=0.98,
                    xref=f'x{row} domain' if row > 1 else 'x domain',
                    yref=f'y{row} domain' if row > 1 else 'y domain',
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=28, color=validation_color),
                    align='right',
                    bordercolor='rgba(0,0,0,0)',
                    bgcolor='rgba(255,255,255,0.8)'
                )

    # Update overall layout with a modern design.
    # The legend is now positioned just below the title.
    fig.update_layout(
        title=dict(text="Training Evolution", x=0.5, font=dict(color='#2E2E2E', size=36)),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, y=0.9, font=dict(size=28), xanchor='center'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        width=1200,
        height=600 * num_plots,
        margin=dict(t=120, b=100)
    )

    # Update x and y axes with a sleek style for all subplots
    fig.update_xaxes(
        title_text="Epochs",
        title_font=dict(size=30, color='#2E2E2E'),
        tickfont=dict(size=26, color='#2E2E2E'),
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    )
    fig.update_yaxes(
        title_font=dict(size=30, color='#2E2E2E'),
        tickfont=dict(size=26, color='#2E2E2E'),
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    )

    # Enhance interactivity with a custom hover template
    fig.update_traces(
        hovertemplate='<b>Epoch:</b> %{x}<br><b>Value:</b> %{y:.4f}'
    )

    fig.show()
    return fig


def evaluate_model(y_test_scaled, y_pred_scaled, scaler_y,
                   lag_window, horizon, train_size, config,
                   eval_title, set_name):
    """
    Inversely scales predictions and actual values, then evaluates and plots the results.
    """
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    evaluate_and_plot(
        y_true=y_test_inv,
        y_pred=y_pred_inv,
        title=eval_title,
        well=config["wells"][0],
        set_name=set_name,
        additional_params={'window_size': lag_window, 'forecast_steps': horizon, 'percentage_split': train_size}
    )
    return y_test_inv, y_pred_inv


def evaluate_cumulative(y_test_inv, y_pred_inv, y_train_cumsum, lag_window, horizon, config, set_name='Cumulative'):
    """
    Computes cumulative sums for predictions and actual values and plots the evaluation.
    """
    plot_time_series
    # Cálculo da soma acumulada do teste, ajustando pelo último valor do treino
    y_true_cumsum = np.cumsum(y_test_inv) + y_train_cumsum

    # Cálculo da soma acumulada das previsões, ajustando pelo último valor do treino
    y_pred_cumsum = np.cumsum(y_pred_inv) + y_train_cumsum

    
    evaluate_and_plot(
        y_true=y_true_cumsum,
        y_pred=y_pred_cumsum,
        title='Cumulative',
        well=config["wells"][0],
        set_name=set_name,
        additional_params={'window_size': lag_window, 'forecast_steps': horizon, 'percentage_split': None}
    )
    return y_true_cumsum, y_pred_cumsum



# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from typing import Optional, List
from themes import themes

def plot_forecast_windows(
    X: np.ndarray,
    y: np.ndarray,
    input_length: int,
    output_length: int,
    indices: Optional[List[int]] = None,
    n_samples_to_plot: int = 3,
    palette: Optional[List[str]] = None,
    theme: str = "cyberpunk",
    title: Optional[str] = "Time Series Forecast Visualization",
    x_label: Optional[str] = "Days",
    y_label: Optional[str] = "Production",
    font_size: int = 20,
    width: int = 1300,
    height: int = 500,
):
    """
    Visualiza janelas históricas e de forecast de um dataset gerado por janela deslizante usando Plotly.
    
    Para cada índice selecionado, a função plota:
      - Dados históricos (janela de input) com linha sólida.
      - Dados de forecast com linha tracejada.
      - Um marcador+linha conectando o final do input com o início do forecast.
      - Uma linha vertical pontilhada dividindo visualmente input e forecast.
    
    Args:
        X (np.ndarray): Janelas históricas com shape (num_samples, input_length).
        y (np.ndarray): Janelas de forecast com shape (num_samples, output_length).
        input_length (int): Comprimento da janela de input.
        output_length (int): Comprimento da janela de forecast.
        indices (List[int], opcional): Índices específicos a serem plotados. Se None, serão selecionados índices espaçados.
        n_samples_to_plot (int, opcional): Número de amostras a plotar se `indices` não for fornecido.
        palette (List[str], opcional): Lista de cores [input, forecast]. Se não fornecida, as cores serão definidas pelo tema.
        theme (str, opcional): Tema a ser usado ('cyberpunk', 'minimal', 'dark').
        title (str, opcional): Título geral da figura.
        x_label (str, opcional): Rótulo para o eixo x.
        y_label (str, opcional): Rótulo para o eixo y.
        font_size (int, opcional): Tamanho base da fonte.
        width (int, opcional): Largura total da figura.
        height (int, opcional): Altura total da figura.
    
    Raises:
        ValueError: Se o número de colunas em X ou y não corresponder a input_length ou output_length.
    """
     
    # Valida dimensões dos dados
    if X.shape[1] != input_length:
        raise ValueError(f"Expected X to have {input_length} columns, but got {X.shape[1]}.")
    if y.shape[1] != output_length:
        raise ValueError(f"Expected y to have {output_length} columns, but got {y.shape[1]}.")

    num_samples = X.shape[0]
    if indices is None:
        indices = np.linspace(0, num_samples - 1, n_samples_to_plot, dtype=int).tolist()
    num_plots = len(indices)
    
    # Configura as cores a partir do parâmetro palette (caso fornecido) ou do tema selecionado
    if palette and isinstance(palette, list) and len(palette) >= 2:
        theme_colors = {
            "input": palette[0],
            "output": palette[1],
            "accent": palette[0],  # Você pode ajustar o 'accent' se desejar
            "bg": themes.get(theme, themes["minimal"])["bg"],
            "grid": themes.get(theme, themes["minimal"])["grid"],
            "text": themes.get(theme, themes["minimal"])["text"],
        }
    else:
        theme_colors = themes.get(theme, themes["minimal"])
    
    # Cria subplots lado a lado
    fig = make_subplots(
        rows=1, cols=num_plots,
        subplot_titles=[f"Sample Index: {idx}" for idx in indices],
        horizontal_spacing=0.08
    )
    
    # Loop para adicionar os traces em cada subplot
    for i, idx in enumerate(indices):
        historical = X[idx, :]
        forecast = y[idx, :]
        
        x_historical = np.arange(input_length)
        x_forecast = np.arange(input_length, input_length + output_length)
        
        show_legend = (i == 0)
        
        # Traço dos dados históricos (input) com linha sólida
        fig.add_trace(
            go.Scatter(
                x=x_historical,
                y=historical,
                mode='lines',
                name='Historical',
                line=dict(color=theme_colors["input"], width=4, dash='solid'),
                hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.2f}',
                showlegend=show_legend
            ),
            row=1, col=i+1
        )
        
        # Traço dos dados de forecast com linha tracejada
        fig.add_trace(
            go.Scatter(
                x=x_forecast,
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color=theme_colors["output"], width=4, dash='dash'),
                hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.2f}',
                showlegend=show_legend
            ),
            row=1, col=i+1
        )
        
        # Trace para o marcador de transição entre input e forecast
        fig.add_trace(
            go.Scatter(
                x=[input_length-1, input_length],
                y=[historical[-1], forecast[0]],
                mode='markers+lines',
                marker=dict(color=theme_colors["input"], size=8),
                line=dict(color=theme_colors["input"], width=2, dash='dot'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Linha vertical pontilhada dividindo input e forecast
        fig.add_vline(
            x=input_length - 0.5,
            line=dict(color=theme_colors["bg"], width=3, dash='dash'),
            row=1, col=i+1
        )
    
    # Atualiza o layout geral utilizando as cores do tema
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=font_size+15, color=theme_colors["text"])
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(
            orientation="h", x=0.5, y=-0.15, xanchor='center',
            font=dict(size=font_size+8, color=theme_colors["text"])
        ),
        plot_bgcolor=theme_colors["bg"],
        paper_bgcolor=theme_colors["bg"],
        width=width,
        height=height,
        font=dict(size=font_size, color=theme_colors["text"]),
        margin=dict(l=60, r=60, t=80, b=80)
    )
    
    # Customiza os eixos de cada subplot
    for j in range(1, num_plots + 1):
        fig.update_xaxes(
            title_font=dict(size=font_size, color=theme_colors["text"]),
            tickfont=dict(size=font_size, color=theme_colors["text"]),
            gridcolor=theme_colors["grid"],
            zeroline=False,
            row=1, col=j
        )
        fig.update_yaxes(
            title_font=dict(size=font_size, color=theme_colors["text"]),
            tickfont=dict(size=font_size, color=theme_colors["text"]),
            gridcolor=theme_colors["grid"],
            zeroline=False,
            row=1, col=j
        )
    
    fig.show()
    
    
def display_data_overview(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, theme: str = "minimal"):
    """
    Displays a formatted and styled table with dataset shapes, sample counts, data types,
    and training/testing proportions.
    
    Parameters:
      X_train_scaled (np.ndarray): Training features.
      X_test_scaled (np.ndarray): Testing features.
      y_train_scaled (np.ndarray): Training targets.
      y_test_scaled (np.ndarray): Testing targets.
      theme (str): One of the predefined themes to style the table. Options include:
                   "cyberpunk", "minimal", "dark", "surrealist", "futurist", 
                   "neoclassical", "deconstructivism".
    
    Returns:
      pd.DataFrame: A DataFrame containing the overview metrics (also printed).
    """
    import pandas as pd
    
    # Use the provided theme or fallback to "minimal"
    theme = themes.get(theme, themes["minimal"])
    
    # Calculate sample counts and proportions
    n_train = X_train_scaled.shape[0]
    n_test = X_test_scaled.shape[0]
    total_samples = n_train + n_test
    train_pct = (n_train / total_samples) * 100
    test_pct = (n_test / total_samples) * 100
    
    # Create metrics as a list of dictionaries for training and testing sets.
    metrics = [
        {
            "Set": "Training",
            "X Shape": str(X_train_scaled.shape),
            "y Shape": str(y_train_scaled.shape),
            "Samples": n_train,
            "Data Format": type(X_train_scaled).__name__,
            "Proportion": f"{train_pct:.2f} %"
        },
        {
            "Set": "Testing",
            "X Shape": str(X_test_scaled.shape),
            "y Shape": str(y_test_scaled.shape),
            "Samples": n_test,
            "Data Format": type(X_test_scaled).__name__,
            "Proportion": f"{test_pct:.2f} %"
        }
    ]
    
    overview_df = pd.DataFrame(metrics)
    
    # Style the DataFrame using the chosen theme.
    styled_overview = (
        overview_df
        .style
        .format({"Proportion": "{}"})  # Already formatted in the dictionary.
        .set_properties(**{
            "font-size": "12pt",
            "text-align": "center",
            "color": theme["text"],
            "background-color": theme["bg"]
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [("background-color", theme["accent"]),
                          ("color", theme["text"]),
                          ("font-size", "14pt"),
                          ("text-align", "center"),
                          ("padding", "8px")]
            },
            {
                "selector": "td",
                "props": [("border", f"1px solid {theme['grid']}"),
                          ("padding", "8px")]
            }
        ])
        .background_gradient(cmap="coolwarm", subset=["Samples"])
    )
    
    # Attempt to display the styled DataFrame (works best in Jupyter Notebooks)
    try:
        from IPython.display import display
        display(styled_overview)
    except Exception as e:
        print(overview_df)
    
    return overview_df




'''
------------------------------------------------
SEQ-TO-SEQ Functions
------------------------------------------------
''' 
import logging                                        

def evaluate_model_seq(y_test_scaled: np.ndarray, 
                       y_pred_scaled: np.ndarray, 
                       scaler_y, 
                       input_length: int, 
                       output_length: int, 
                       train_size: float, 
                       config: Dict[str, Any],
                       eval_title: str = "",
                       set_name: str = "",
                       aggregation_method: str = 'mean',
                       quantiles: Optional[List[float]] = None,
                       plot: bool = True
                      ) -> Dict[str, Any]:
    """
    Evaluates seq-to-seq forecasts by computing global metrics.
    
    For the ground truth, the original series is reconstructed using reconstruct_true_series,
    whereas predictions are aggregated using aggregate_predictions.
    
    Returns a dictionary with:
      - 'agg_y_test': Reconstructed ground truth series.
      - 'agg_y_pred': Aggregated prediction series.
      - 'global_metrics': Metrics (R², SMAPE, MAE) computed on the aggregated series.
    """
    from common.seq_preprocessing import aggregate_predictions, reconstruct_true_series
    # Inverse-transform full arrays (shape: (n_samples, horizon))
    y_test_inv_full = scaler_y.inverse_transform(y_test_scaled)
    y_pred_inv_full = scaler_y.inverse_transform(y_pred_scaled)
    
    # Reconstruct the original ground truth series.
    agg_y_test = reconstruct_true_series(y_test_inv_full)
    # agg_y_pred = reconstruct_true_series(y_pred_inv_full)
    agg_y_pred = aggregate_predictions(y_pred_inv_full)
    
    # Compute global metrics on the reconstructed/aggregated series.
    r2, smape, mae = evaluate(agg_y_test, agg_y_pred)
    # r2, smape, mae = evaluate(y_test_inv_full, y_pred_inv_full)
    # logging.info(f"Normal SMAPE: {smape}")
    
    global_metrics = {'R²': r2, 'SMAPE': smape, 'MAE': mae, 'Category': 'Global'}
    
    if plot:
    
        # Plot aggregated forecasts.
        evaluate_and_plot(
            y_true=agg_y_test,
            y_pred=agg_y_pred,
            title=eval_title + " - Aggregated",
            well=config["wells"][0],
            set_name=set_name,
            additional_params={'window_size': input_length, 'forecast_steps': output_length, 'percentage_split': train_size}
        )
    
    return {
        'agg_y_test': agg_y_test,
        'agg_y_pred': agg_y_pred,
        'global_metrics': global_metrics
    }


def evaluate_cumulative_seq(agg_y_test: np.ndarray, 
                            agg_y_pred: np.ndarray, 
                            y_train_original: np.ndarray,
                            scaler_target,
                            input_length: int, 
                            output_length: int, 
                            config: Dict[str, Any],
                            set_name: str = "Cumulative",
                            plot: bool = True, ) -> Dict[str, np.ndarray]:
    """
    Computes cumulative sums for the aggregated ground truth and prediction series.
    
    For the training series, the original series is reconstructed.
    """
    # Reconstruct training series.
    from common.seq_preprocessing import reconstruct_true_series
    train_series = reconstruct_true_series(y_train_original)
    
    # Compute cumulative sums, adjusted by the last value of the training series.
    y_train_cumsum = np.cumsum(train_series)
    
    print('y_train_cumsum', y_train_cumsum[-1])
    
    y_test_cumsum = np.cumsum(agg_y_test) + y_train_cumsum[-1]
    y_pred_cumsum = np.cumsum(agg_y_pred) + y_train_cumsum[-1]
    
    if plot:
    
        evaluate_and_plot(
            y_true=y_test_cumsum,
            y_pred=y_pred_cumsum,
            title='Cumulative Forecast',
            well=config["wells"][0],
            set_name=set_name,
            additional_params={'window_size': input_length, 'forecast_steps': output_length, 'percentage_split': None}
        )
    
    return {
        'y_test_cumsum': y_test_cumsum,
        'y_pred_cumsum': y_pred_cumsum
    }

    
    
def compute_metrics_to_df_seq(y_test: np.ndarray, 
                              y_pred: np.ndarray, 
                              well: str, 
                              method: str,
                              category:str) -> Dict[str, Any]:
    """
    Computes evaluation metrics (R², SMAPE, MAE) for aggregated seq-to-seq forecasts.
    
    y_test and y_pred should be aggregated series.
    """
    metrics = evaluate_return_dict(y_test, y_pred)
    metrics.update({
        'Well': well,
        'Method': method,
        'Category': category
    })
    return metrics



from themes import themes  # Assumes themes.py exports a dictionary named `themes`

def organize_metrics(metrics_list, default_values):
    """
    Given a list of metric dictionaries, fill in missing keys with default_values.
    """
    organized = []
    for m in metrics_list:
        for key, default in default_values.items():
            if key not in m:
                m[key] = default
        organized.append(m)
    return pd.DataFrame(organized)

def organize_and_display_metrics(global_metrics_list, aggregated_metrics_list, cumulative_metrics_list, slice_metrics_list=None, theme_name="minimal"):
    """
    Organizes global, aggregated, cumulative, and (optionally) slice metrics into a single DataFrame.
    Applies a styled design based on a selected theme.
    
    Parameters:
      global_metrics_list (List[Dict]): Global metrics.
      aggregated_metrics_list (List[Dict]): Aggregated metrics.
      cumulative_metrics_list (List[Dict]): Cumulative metrics.
      slice_metrics_list (Optional[List[Dict]]): Slice metrics (if available).
      theme_name (str): Key for selecting a theme from themes.py.
    
    Returns:
      pd.DataFrame: The combined DataFrame (displayed with styling if possible).
    """
    # Define default values for missing keys.
    defaults = {
        "Category": "N/A", 
        "Well": "N/A", 
        "Method": "N/A",  
        "R²": np.nan, 
        "SMAPE": np.nan, 
        "MAE": np.nan
    }
    
    # Organize the different metric groups.
    df_global = organize_metrics(global_metrics_list, defaults)
    df_aggregated = organize_metrics(aggregated_metrics_list, defaults)
    df_cumulative = organize_metrics(cumulative_metrics_list, defaults)
    
    frames = [df_global, df_aggregated, df_cumulative]
    
    # If slice metrics are provided, organize them as well.
    if slice_metrics_list is not None and len(slice_metrics_list) > 0:
        df_slice = pd.DataFrame(slice_metrics_list)
        # Fill in missing columns with default values.
        for col, default in defaults.items():
            if col not in df_slice.columns:
                df_slice[col] = default
        # Optionally, add a label for the slice metrics.
        df_slice["Category"] = "Slice"
        frames.append(df_slice)
    
    # Combine all metrics.
    combined_df = pd.concat(frames, ignore_index=True)
    
    # Define the order of columns.
    desired_order = ["Category", "Well", "Method", "R²", "SMAPE", "MAE"]
    combined_df = combined_df[[col for col in desired_order if col in combined_df.columns]]
    
    # Select the theme, defaulting to 'minimal' if the provided key is not found.
    chosen_theme = themes.get(theme_name, themes["minimal"])
    
    # Create a styled DataFrame using the selected theme.
    styled_metrics = (
        combined_df.style
        .format({
            "R²": "{:.4f}",
            "SMAPE": "{:.2f}%",
            "MAE": "{:.4f}"
        })
        .set_properties(**{
            'font-size': '12pt',
            'text-align': 'center',
            'color': chosen_theme["text"],
            'background-color': chosen_theme["bg"]
        })
        .background_gradient(cmap='coolwarm', subset=['R²', 'SMAPE', 'MAE'])
        .set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', chosen_theme["accent"]),
                    ('color', chosen_theme["bg"]),
                    ('font-size', '14pt'),
                    ('text-align', 'center'),
                    ('border', f'1px solid {chosen_theme["grid"]}')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('font-size', '12pt'),
                    ('text-align', 'center'),
                    ('border', f'1px solid {chosen_theme["grid"]}')
                ]
            },
            {
                'selector': 'table',
                'props': [
                    ('background-color', chosen_theme["bg"]),
                    ('border-collapse', 'collapse')
                ]
            }
        ])
    )
    
    # Try to display the styled DataFrame.
    try:
        from IPython.display import display
        display(styled_metrics)
    except Exception as e:
        print(combined_df)
    
    return combined_df



from typing import Sequence, Union, Optional, Any

# --- More Sophisticated Palette ---
COLOR_PRIMARY = '#0077B6'  # Strong Blue (Star Command Blue)
COLOR_SECONDARY = '#F94144' # Strong Red (Imperial Red)
COLOR_ACCENT_FILL = 'rgba(249, 65, 68, 0.1)' # Very subtle red fill
COLOR_TEXT = '#2c3e50'      # Midnight Blue (Slightly softer dark)
COLOR_GRID = 'rgba(189, 195, 199, 0.4)' # Silver Sand / Light Gray
FONT_FAMILY = "Lato, Arial, sans-serif" # Modern, clean font (Lato preferred if available)

def plot_comparison(
    y1: Union[Sequence[float], np.ndarray],
    y2: Union[Sequence[float], np.ndarray],
    feature: str,
    well_info: str,
    series1_name: Optional[str] = None,
    series2_name: Optional[str] = None,
):
    """
    Plots a disruptive yet clean comparative time series chart using Plotly.

    Emphasizes contrast between two series using distinct colors, line styles,
    and a subtle fill highlighting the difference. Aims for presentation
    sophistication with refined typography and layout.

    Args:
        y1: First time series data (e.g., baseline, model 1).
        y2: Second time series data (e.g., scenario, model 2).
        feature: Name of the feature being plotted (for title).
        well_info: Descriptive label for the comparison context (e.g., "Well A vs Well B", "Baseline vs Scenario X").
        series1_name: Optional explicit name for the first series (legend/hover).
                      If None, attempts to parse from well_info or defaults to "Series 1".
        series2_name: Optional explicit name for the second series (legend/hover).
                      If None, attempts to parse from well_info or defaults to "Series 2".
    """
    # --- Data Preparation ---
    min_length = min(len(y1), len(y2))
    if min_length == 0:
        print(f"Warning: Insufficient data for feature '{feature}' in comparison '{well_info}'. Skipping plot.")
        return None # Return None to indicate no plot was generated

    y1 = np.array(y1[:min_length]) # Ensure numpy array for potential future calcs
    y2 = np.array(y2[:min_length])
    x_values = np.arange(min_length)

    # --- Determine Series Names ---
    s1_name = "Series 1"
    s2_name = "Series 2"
    if series1_name:
        s1_name = series1_name
    if series2_name:
        s2_name = series2_name
    elif " vs " in well_info: # Try parsing from well_info if names not provided
        try:
            parsed_names = well_info.split(" vs ")
            if len(parsed_names) == 2:
                s1_name = parsed_names[0]
                s2_name = parsed_names[1]
        except Exception:
            pass # Keep defaults if parsing fails

    # --- Plotting ---
    fig = go.Figure()

    # Series 1: Primary color, solid, slightly thicker line
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y1,
        mode='lines',
        name=s1_name,
        line=dict(color=COLOR_PRIMARY, width=3.5), # Slightly thicker primary line
        fill=None,
        hovertemplate=f'<b>{s1_name}</b><br>' +
                      'Time Step: %{x}<br>' +
                      'Value: %{y:.3f}<extra></extra>' # Extra removes trace name repetition
    ))

    # Series 2: Secondary color, dashed line, subtle fill 'tonexty'
    # The fill highlights the area *between* series 1 and series 2
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y2,
        mode='lines',
        name=s2_name,
        line=dict(color=COLOR_SECONDARY, width=2.5, dash='dash'), # Thinner, dashed line
        fill='tonexty', # Fill the area between this trace and the previous one
        fillcolor=COLOR_ACCENT_FILL, # Use a very subtle fill color
        hovertemplate=f'<b>{s2_name}</b><br>' +
                      'Time Step: %{x}<br>' +
                      'Value: %{y:.3f}<extra></extra>'
    ))

    # --- Layout and Styling ---
    fig.update_layout(
        title=dict(
            text=f"<b>Comparison: {feature}</b><br><span style='font-size: 0.7em; color: {COLOR_TEXT}'>{well_info}</span>",
            x=0.05, # Align title left for a more modern feel
            y=0.95,
            xanchor='left',
            yanchor='top',
            font=dict(size=24, color=COLOR_TEXT, family=FONT_FAMILY)
        ),
        xaxis_title="Time Step",
        yaxis_title=feature, # Use feature name directly for Y-axis if appropriate
        font=dict(family=FONT_FAMILY, size=14, color=COLOR_TEXT),
        showlegend=True,
        legend=dict(
            orientation="h", # Horizontal legend
            yanchor="bottom",
            y=-0.25,          # Position below X-axis title
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.6)', # Semi-transparent background
            bordercolor=COLOR_GRID,
            borderwidth=1
        ),
        plot_bgcolor='white', # Clean white plot background
        paper_bgcolor='white', # Clean white paper background
        margin=dict(l=80, r=40, t=100, b=100), # Adjust margins for titles/legend
        hovermode='x unified', # Show hover info for all traces at a given x
        width=1200, # Keep specified size, make parameter if needed
        height=600,
    )

    # --- Axis Styling ---
    fig.update_xaxes(
        title_font=dict(size=16),
        tickfont=dict(size=12),
        gridcolor=COLOR_GRID,
        gridwidth=1,
        zeroline=False,
        showline=True, # Show axis line
        linewidth=1,
        linecolor=COLOR_GRID,
        mirror=True, # Show lines on top/right as well
        ticks='outside', # Ticks outside plot area
    )
    fig.update_yaxes(
        title_font=dict(size=16),
        tickfont=dict(size=12),
        gridcolor=COLOR_GRID,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=COLOR_GRID,
        mirror=True,
        ticks='outside',
        # Automatically format ticks nicely
        tickformat=".2f" if np.abs(y1).max() < 10 and np.abs(y2).max() < 10 else ".1f" if np.abs(y1).max() < 100 and np.abs(y2).max() < 100 else ""
    )

    # Consider adding range slider for long series if appropriate
    # fig.update_xaxes(rangeslider_visible=True)

    fig.show()
    
    
def safe_numeric_format(value: Any, fmt: str = "{:.2f}") -> str:
    """Safely formats a value as numeric, returns original if conversion fails."""
    try:
        # Handle potential pandas types or strings that represent numbers
        numeric_val = pd.to_numeric(value)
        if pd.isna(numeric_val):
            return "-" # Represent NaN consistently
        return fmt.format(numeric_val)
    except (ValueError, TypeError):
        return str(value) # Return original as string if not numeric


def display_welch_t_test_results(
    test_results: List[Dict[str, Any]],
    title: str = "Welch's T-Test Results",
    p_value_threshold: float = 0.05
):
    """
    Displays Welch's T-Test results with a sophisticated and elegant style.

    Features:
    - Highlights significant results subtly (p-value < threshold).
    - Clean typography and alignment (text left, numbers right).
    - Minimalist borders and clear spacing.
    - Consistent color scheme.

    Args:
        test_results: List of dictionaries, each with keys like
                      'Feature', 'Statistic', 'P-value', optionally 'Significance'.
        title: Title displayed above the table.
        p_value_threshold: Threshold for considering a p-value significant.

    Returns:
        The pandas Styler object, displayed in the environment, or None if no results.
    """
    if not test_results:
        print(f"Info: No test results provided for '{title}'. Nothing to display.")
        return None

    df = pd.DataFrame(test_results)

    # Ensure essential columns exist, handle potential missing 'Significance'
    required_cols = ["Feature", "Statistic", "P-value"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input data must contain columns: {', '.join(required_cols)}")

    if "Significance" not in df.columns:
         # Create Significance based on p-value if not present
         df['P-value_numeric'] = pd.to_numeric(df['P-value'], errors='coerce')
         df["Significance"] = np.where(
             df['P-value_numeric'] < p_value_threshold,
             "Significant", # Use a clearer label
             "Not Significant"
         )
         df = df.drop(columns=['P-value_numeric']) # Drop helper column

    # Add flag for styling, ensuring P-value is numeric for comparison
    df["_is_significant"] = pd.to_numeric(df["P-value"], errors='coerce') < p_value_threshold

    # Define formatting for specific columns
    formatters = {
        "Statistic": lambda x: safe_numeric_format(x, "{:,.2f}"),
        "P-value": lambda x: safe_numeric_format(x, "{:.4f}"),
    }

    # Identify column types for alignment
    numeric_cols = ["Statistic", "P-value"] # Specify numeric cols expected
    text_cols = [col for col in df.columns if col not in numeric_cols + ["_is_significant"]] # All others are text

    # --- Styling ---
    styled = (
        df.style
          .hide(axis="columns", subset=["_is_significant"]) # Use hide() for cleaner hiding
          .format(formatters, na_rep="-")
          .set_properties(
              subset=numeric_cols, **{
                  "text-align": "right",
                  "font-variant-numeric": "tabular-nums",
              }
           )
          .set_properties(
              subset=text_cols, **{"text-align": "left"}
           )
          .set_properties(**{ # General cell props
              "font-family": FONT_FAMILY,
              "font-size": "10.5pt",
              "padding": "8px 12px",
              "border": "none", # Remove default borders first
              "border-bottom": f"1px solid {COLOR_BORDER_LIGHT}", # Add only bottom border
              "color": COLOR_TEXT,
          })
          .set_table_styles([
              { # Table-wide
                  "selector": "table",
                  "props": [
                      ("width", "100%"),
                      ("border-collapse", "collapse"),
                      ("background-color", COLOR_BACKGROUND),
                      # Optional: Add a subtle top/bottom border to the whole table
                      # ("border-top", f"2px solid {COLOR_PRIMARY}"),
                      # ("border-bottom", f"2px solid {COLOR_PRIMARY}")
                  ]
              },
              { # Header row
                  "selector": "th.col_heading",
                  "props": [
                      ("background-color", COLOR_BACKGROUND), # Lighter header background
                      ("color", COLOR_PRIMARY), # Use primary color for text
                      ("font-family", FONT_FAMILY),
                      ("font-size", "11pt"),
                      ("font-weight", "600"),
                      ("text-align", "center"), # Center headers still fine
                      ("padding", "10px 12px"),
                      ("border", "none"),
                      ("border-bottom", f"2px solid {COLOR_PRIMARY}") # Stronger bottom border for header
                  ]
              },
              { # Header for index (usually blank)
                   "selector": "th.index_name", "props": [("display", "none")]
              },
              { # Data rows (tr) general - remove default row hover if any
                  "selector": "tr", "props": [("background-color", "transparent")]
              },
              { # Remove bottom border from last data row
                  "selector": "tbody tr:last-child td",
                  "props": [("border-bottom", "none")]
              },
              { # Caption (Title)
                  "selector": "caption",
                  "props": [
                      ("caption-side", "top"),
                      ("font-size", "16pt"),
                      ("font-weight", "bold"),
                      ("color", COLOR_TEXT),
                      ("padding", "15px 0"),
                      ("text-align", "left"),
                  ]
              }
          ])
          # Apply subtle highlight to significant rows using background color
          .apply(
              lambda row: [f'background-color: {COLOR_HIGHLIGHT_BG}' if row["_is_significant"] else '' for _ in row],
              axis=1
          )
          # Optionally: Highlight the specific P-value cell more strongly
          # .applymap(
          #      lambda val, is_sig: f'font-weight: bold; color: {COLOR_SECONDARY}' if is_sig else '',
          #      subset=pd.IndexSlice[:, ['P-value']],
          #      is_sig=df['_is_significant']
          #  )
          .set_caption(title)
          .set_table_attributes('style="margin-bottom: 20px;"') # Add space below table
    )

    display(styled)
    return styled


import pandas as pd
import numpy as np
from IPython.display import display, HTML
from typing import List, Dict, Any, Optional, Union

# --- Using the previously established palette ---
COLOR_PRIMARY = '#0077B6'  # Strong Blue
COLOR_SECONDARY = '#F94144' # Strong Red
COLOR_ACCENT_FILL = 'rgba(249, 65, 68, 0.1)' # Very subtle red fill
COLOR_TEXT = '#2c3e50'      # Midnight Blue
COLOR_TEXT_LIGHT = '#FFFFFF' # White
COLOR_GRID = 'rgba(189, 195, 199, 0.4)' # Silver Sand / Light Gray
COLOR_BACKGROUND = '#FFFFFF' # White
COLOR_HIGHLIGHT_BG = 'rgba(249, 65, 68, 0.08)' # Even more subtle red for highlighting rows/cells
COLOR_BORDER_LIGHT = '#EAECEE' # Very light border color

FONT_FAMILY = "Lato, Arial, sans-serif"


def display_descriptive_statistics(
    df: pd.DataFrame,
    title: str = "Descriptive Statistics",
    precision: int = 2 # Default to 2 decimal places for stats
):
    """
    Displays descriptive statistics with a sophisticated and elegant style.

    Features:
    - Clean typography and alignment (text left, numbers right).
    - Subtle borders and generous padding.
    - Defined color scheme for headers and text.
    - Consistent numeric formatting.

    Args:
        df: DataFrame containing descriptive statistics (e.g., from df.describe()).
        title: Title displayed above the table.
        precision: Number of decimal places for numeric formatting.

    Returns:
        The pandas Styler object, displayed in the environment, or None if df is empty.
    """
    if df.empty:
        print(f"Info: DataFrame for '{title}' is empty. Nothing to display.")
        return None

    # Determine numeric columns for right-alignment and formatting
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    format_string = f"{{:,.{precision}f}}" # e.g., "{:,.2f}"

    styled_df = (
        df.style
        .format(format_string, subset=numeric_cols, na_rep="-") # Format numerics, use '-' for NaN
        .set_properties(
            subset=numeric_cols, **{ # Right-align numbers
                "text-align": "right",
                "font-variant-numeric": "tabular-nums" # Helps align numbers vertically
            }
        )
        .set_properties(
            subset=text_cols, **{"text-align": "left"} # Left-align text columns
         )
        .set_properties( **{ # General cell properties
              "font-family": FONT_FAMILY,
              "font-size": "10.5pt", # Slightly larger base font
              "padding": "6px 8px", # More padding
              "border": f"1px solid {COLOR_BORDER_LIGHT}", # Lighter border for all cells initially
              "color": COLOR_TEXT,
          })
        .set_table_styles([
            { # Table-wide styles
                "selector": "table",
                "props": [
                    ("width", "100%"), # Make table responsive width
                    ("border-collapse", "collapse"),
                    ("background-color", COLOR_BACKGROUND),
                    ("border", f"1px solid {COLOR_BORDER_LIGHT}") # Outer border
                ]
            },
            { # Header row styles
                "selector": "th.col_heading", # Column headers
                "props": [
                    ("background-color", COLOR_PRIMARY),
                    ("color", COLOR_TEXT_LIGHT),
                    ("font-family", FONT_FAMILY),
                    ("font-size", "11pt"),
                    ("font-weight", "600"),
                    ("text-align", "center"), # Center-align headers
                    ("padding", "10px 12px"),
                    ("border", f"1px solid {COLOR_PRIMARY}") # Header border matches background
                ]
            },
             { # Index column header (top-left cell)
                "selector": "th.index_name",
                "props": [
                    ("background-color", COLOR_BACKGROUND), # Match background
                    ("color", COLOR_TEXT),
                    ("font-weight", "normal"),
                    ("border", f"1px solid {COLOR_BORDER_LIGHT}"),
                    ("text-align", "left"),
                ]
            },
            { # Index column cells (statistic names like 'mean', 'std')
                "selector": "th.row_heading",
                "props": [
                    ("background-color", COLOR_BACKGROUND),
                    ("color", COLOR_TEXT),
                    ("font-weight", "600"), # Make stat names bold
                    ("text-align", "left"),
                    ("padding-left", "15px"), # Indent stat names slightly
                    ("border", f"1px solid {COLOR_BORDER_LIGHT}")
                ]
            },
            { # Data cells (td)
                "selector": "td",
                "props": [
                     # Already set in general properties, override if needed
                     ("border-bottom", f"1px solid {COLOR_BORDER_LIGHT}"),
                     ("border-top", "none"), # Remove duplicate top border
                     ("border-left", "none"),
                     ("border-right", "none"),
                ]
            },
            { # Remove bottom border from last row for cleaner look
                "selector": "tr:last-child td",
                "props": [("border-bottom", "none")]
            },
            { # Table caption (Title)
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "16pt"),
                    ("font-weight", "bold"),
                    ("color", COLOR_TEXT),
                    ("padding", "15px 0"),
                    ("text-align", "left"), # Align title left
                ]
            }
        ])
        .set_caption(title)
        .set_table_attributes('style="margin-bottom: 20px;"') # Add space below table
    )

    display(styled_df)
    return styled_df