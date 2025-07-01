import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def create_window_dataset(
    train_df, window_size, target_series, forecast_steps
):
    """
    Creates a dataset for time series prediction with lagged features for the target
    series at the end of the feature matrix and the rest of the features unchanged.

    Args:
    train_df (pd.DataFrame): DataFrame containing the time series data.
    window_size (int): Size of the sliding window for features.
    target_series (str): Name of the target series to predict.
    forecast_steps (int): Number of steps to forecast into the future.

    Returns:
    tuple: A tuple containing two NumPy arrays:
      - X: Features (static features followed by lagged target)
      - y: Target values (values to predict)
    """

    # Extract the target series with a variable shift
    y = train_df[target_series].shift(-forecast_steps).to_numpy()[:-forecast_steps]  # Exclude last values

    # Separate the target series and other features
    target_values = train_df[target_series].to_numpy()
    other_features = train_df.drop(columns=[target_series]).to_numpy()

    # Calculate the number of windows
    num_windows = len(y) - window_size + 1

    # Initialize arrays to store lagged target features and combined features
    X = np.zeros((num_windows, other_features.shape[1] + window_size))
    y_output = np.zeros((num_windows, 1))

    # Fill X with static other features and lagged values of the target series
    for i in range(num_windows):
        # Add static values of other features at the current timestep
        X[i, :other_features.shape[1]] = other_features[i + window_size - 1]

        # Lagged values of the target series
        X[i, other_features.shape[1]:] = target_values[i:i + window_size]

        # Target value after the window
        y_output[i] = y[i + window_size - 1]

    return X, y_output.ravel()



def normalize_by_window_mean(X_train, X_test, y_train, y_test, window_size, well_index):
    """
    This function normalizes the target variable by the window mean of the last 7 days
    and adds the window mean as a feature to the input data.

    Args:
      X_train (np.ndarray): Training features with shape (n_samples, n_features)
      X_test (np.ndarray): Testing features with shape (n_samples, n_features)
      y_train (np.ndarray): Training target variable with shape (n_samples,)
      y_test (np.ndarray): Testing target variable with shape (n_samples,)

    Returns:
      X_train_new (np.ndarray): Training features with window mean added (n_samples, n_features+1)
      X_test_new (np.ndarray): Testing features with window mean added (n_samples, n_features+1)
      y_train_norm (np.ndarray): Normalized training target variable (n_samples,)
      y_test_norm (np.ndarray): Normalized testing target variable (n_samples,)
    """
    
    # Calculate window mean for the last n days of each row
    window_mean_train = np.mean(X_train[:, -window_size:], axis=1, keepdims=True)
    window_mean_test = np.mean(X_test[:, -window_size:], axis=1, keepdims=True)
    
    inclination_train = X_train[:, -1]/X_train[:, -window_size]
    inclination_test = X_test[:, -1]/X_test[:, -window_size]
    
    inclination_train = np.expand_dims(inclination_train, axis=1)
    inclination_test = np.expand_dims(inclination_test, axis=1)

    # X_train = np.concatenate((X_train, inclination_train), axis=1)
    # X_test = np.concatenate((X_test, inclination_test), axis=1)
    

    # Add window mean as a new feature
    X_train_new = np.concatenate((X_train, window_mean_train), axis=1)
    X_test_new = np.concatenate((X_test, window_mean_test), axis=1)
    

    y_train = y_train/window_mean_train.squeeze()
    y_test = y_test/window_mean_test.squeeze()

    return X_train_new, X_test_new, y_train, y_test



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def split_sets(
    df,
    window_size,
    series_name,
    forecast_steps,
    control_iteration,
    train_windows,
    fine_tuning_windows,
    well_index,
    max_train=None,
    scaler=None,
    cum_sum=None
):
    """
    Splits the dataset into training and testing sets, adjusting for the training window length.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the well data.
    - window_size (int): The size of the window for creating sequences.
    - series_name (str): The name of the target series in the DataFrame.
    - forecast_steps (int): The number of steps to forecast.
    - control_iteration (int): The current iteration index.
    - train_windows (int): The window length for training the base model (well 0).
    - fine_tuning_windows (int): The window length for fine-tuning on other wells.
    - well_index (int): The index of the well (0 for base model, others for fine-tuning).
    - scaler (StandardScaler, optional): The scaler used for normalization; if None, a new scaler is created.

    Returns:
    - X_train (np.ndarray): The training feature set.
    - X_test (np.ndarray): The testing feature set.
    - y_train (np.ndarray): The training target set.
    - y_test (np.ndarray): The testing target set.
    - scaler (StandardScaler): The scaler used for normalization.
    """
    from sklearn.preprocessing import StandardScaler

    total_length = len(df)

    # Determine window length based on the well index
    if well_index == 0:
        window_length = train_windows  # Base model uses train_windows
    else:
        window_length = fine_tuning_windows  # Other wells use fine_tuning_windows

    # Calculate start and end indices for the data window
    start_idx = control_iteration
    end_idx = control_iteration + window_length + 2*forecast_steps

    # Ensure indices are within the bounds of the DataFrame
    if end_idx > total_length:
        end_idx = total_length
        start_idx = total_length - window_length - 2*forecast_steps
        

    # Extract the data window for the current iteration
    df_window = df.iloc[start_idx:end_idx].copy()
    
    # Criar uma nova coluna no dataframe com a série filtrada
    if not cum_sum:
        filtered_values = apply_custom_kalman_filter(df_window[series_name].values, process_var=1e-6, measurement_var=1e-3)
        df_window.loc[:, series_name] = filtered_values
        

    # Create datasets using a windowed approach
    X, y = create_window_dataset(
        df_window,
        window_size=window_size,
        target_series=series_name,
        forecast_steps=forecast_steps
    )
    
    
    # Split into training and testing sets
    X_train, y_train = X[:window_length-window_size+1], y[:window_length-window_size+1]  # All except the last sample for training
    X_test, y_test = X[-1:], y[-1:]    # Last sample for testing
    

    if control_iteration == 0:
        print('Length of X_train:', len(X_train))
        inspect_datasets(X_train, y_train, X_test, y_test, window_size, num_original_features=X_train.shape[-1])
    
    
    if cum_sum:
        # print("Normalizing by the WINDOW average.")
        X_train, X_test, y_train, y_test = normalize_by_window_mean(
            X_train, X_test, y_train, y_test, window_size, well_index
        )
    
    else:
        max_train = StandardScaler()
        max_train.fit(y_train.reshape(-1, 1))

        # Transform features using the scaler
        y_train = max_train.transform(y_train.reshape(-1, 1)).flatten()
        y_test = max_train.transform(y_test.reshape(-1, 1)).flatten()
    
      
    # Initialize and fit the scaler if it's None (for the base model)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_train)
        
    # Transform features using the scaler
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
        

    if control_iteration == 0:
        print('Length of X_train:', len(X_train))
        inspect_datasets(X_train, y_train, X_test, y_test, window_size, num_original_features=X_train.shape[-1])
    

    return X_train, X_test, y_train, y_test, max_train, scaler


def filter_data_for_iteration(
    df_list,
    window_size,
    series_name,
    forecast_steps,
    control_iteration,
    active_wells,
    train_windows,
    fine_tuning_windows,
    scaler=None,
    cum_sum = None
):
    """
    Filters and prepares the data for the current iteration, only for active wells.

    Parameters:
    - df_list (list of pd.DataFrame): List of DataFrames for each well.
    - window_size (int): The size of the window for creating sequences.
    - series_name (str): The name of the target series in the DataFrame.
    - forecast_steps (int): The number of steps to forecast.
    - control_iteration (int): The current iteration index.
    - active_wells (list of int): List of indices of active wells.
    - train_windows (int): The window length for training the base model (well 0).
    - fine_tuning_windows (int): The window length for fine-tuning on other wells.
    - scaler (StandardScaler, optional): The scaler used for normalization; if None, it will be initialized on well 0.

    Returns:
    - results (list): A list of tuples containing (X_train, X_test, y_train, y_test, scaler) for each active well.
    """
    results = []

            
    # Process other active wells, reusing the scaler trained on well 0
    for i in active_wells:
        # Check if there is enough data for the current iteration
        remaining_data = len(df_list[i]) - fine_tuning_windows - 2*forecast_steps
        # print(f'Len df i: {len(df_list[i])}, {i}')
        # print(f'remaining_data: {remaining_data}')
        if remaining_data > control_iteration:
            X_train, X_test, y_train, y_test, max_train, scaler = split_sets(
                df_list[i],
                window_size,
                series_name,
                forecast_steps,
                control_iteration,
                train_windows,
                fine_tuning_windows,
                well_index=i,   # Specify the current well index
                max_train = None,
                scaler= None,
                cum_sum=cum_sum
            )
            results.append((X_train, X_test, y_train, y_test, max_train, scaler))
        else:
            print(f"Not enough data for well {i} at iteration {control_iteration}. Skipping.")
            

    return results


def prepare_train_test_sets(sets, model_type, well_idx=0):
    """
    Prepares the training and testing sets for model training.

    Parameters:
    - sets (list): A list where each element is a tuple or list containing:
        - X_train: Training features.
        - X_test: Testing features.
        - y_train: Training targets.
        - y_test: Testing targets.
        - scaler: Scaler object used for data normalization.
    - model_type (str): The type of model ('GBR' for Gradient Boosting Regressor or 'DL' for Deep Learning).
    - well_idx (int): The index of the well to use for the base model's training data (default is 0).

    Returns:
    - X_train (np.ndarray): Training features for the base model.
    - y_train (np.ndarray): Training targets for the base model.
    - X_tests (list of np.ndarray): List of testing features for all wells.
    - y_tests (list of np.ndarray): List of testing targets for all wells.
    - scalers (list): List of scaler objects for each well.
    """
    # Extract base well's training data
    X_train = sets[well_idx][0]
    y_train = sets[well_idx][2]

    # Reshape X_train according to the model type
    if model_type == 'XGB':
        X_train = X_train.reshape(X_train.shape[0], -1)
    elif model_type == 'DL':
        X_train = X_train.reshape(X_train.shape[0], 1, -1)
    else:
        X_train = X_train.reshape(X_train.shape[0], -1)

    # Initialize lists for testing data and scalers
    X_tests = []
    y_tests = []
    scalers = []
    max_trains = []

    for s in sets:
        X_test, y_test, max_train, scaler = s[1], s[3], s[-2], s[-1]

        # Reshape X_test according to the model type
        if model_type == 'GBR':
            X_test = X_test.reshape(X_test.shape[0], -1)
        elif model_type == 'DL':
            X_test = X_test.reshape(X_test.shape[0], 1, -1)

        X_tests.append(X_test)
        y_tests.append(y_test)
        scalers.append(scaler)
        max_trains.append(max_train)

    return X_train, y_train, X_tests, y_tests, max_trains, scalers

# Função para organizar wells e df_list de acordo com o tamanho dos DataFrames
def organize_wells_by_df_size(wells, df_list):
    # Combina os wells e seus respectivos DataFrames e organiza pela quantidade de linhas no DataFrame (len)
    sorted_wells_and_dfs = sorted(zip(wells, df_list), key=lambda x: len(x[1]), reverse=True)
    
    # Descompacta os resultados organizados
    sorted_wells, sorted_dfs = zip(*sorted_wells_and_dfs)
    
    return list(sorted_wells), list(sorted_dfs)

def initialize_prediction_lists(num_wells):
    """
    Initializes empty lists to store y_test and y_pred values for each well.

    Parameters:
    - num_wells (int): The number of wells.

    Returns:
    - y_test_list (list): A list containing empty lists for each well's y_test values.
    - y_pred_list (list): A list containing empty lists for each well's y_pred values.
    """
    # Initialize empty lists for each well
    y_test_list = [[] for _ in range(num_wells)]
    y_pred_list = [[] for _ in range(num_wells)]
    return y_test_list, y_pred_list


def calculate_total_iterations(df_list):
    """
    Calculates the total number of iterations based on the maximum length of the data among wells.

    Parameters:
    - df_list (list of pd.DataFrame): A list of DataFrames for each well.

    Returns:
    - total_iterations (int): The maximum number of rows across all DataFrames.
    """
    # Find the maximum length among all DataFrames
    total_iterations = max(len(df) for df in df_list)
    return total_iterations


def plot_datasets(y_train, y_test, window_size, index=None):
    """
    Plota as sequências de alvos de treinamento e teste ao longo do tempo usando um estilo artístico.

    Parâmetros:
    - y_train (np.ndarray): Alvos de treinamento.
    - y_test (np.ndarray): Alvos de teste.
    - window_size (int): Tamanho da janela usada para características defasadas.
    - index (pd.Index ou np.ndarray, opcional): Índice ou timestamps associados aos dados.

    Retorna:
    - None
    """
    import numpy as np
    import plotly.graph_objects as go
    import pandas as pd

    # Ajustar o índice se fornecido
    if index is not None:
        train_indices = index[window_size:-1]
        test_indices = index[-1:]
    else:
        train_indices = np.arange(window_size, window_size + len(y_train))
        test_indices = np.array([window_size + len(y_train)])

    # Criar traços para treinamento e teste
    fig = go.Figure()

    # Traço dos alvos de treinamento
    fig.add_trace(go.Scatter(
        x=train_indices,
        y=y_train,
        mode='lines',
        name='Training Target',
        line=dict(color='#206A92', width=3),
    ))

    # Traço dos alvos de teste
    fig.add_trace(go.Scatter(
        x=test_indices,
        y=y_test,
        mode='markers',
        name='Testing Target',
        marker=dict(color='red', size=10),
    ))

    # Atualizar layout com elementos artísticos
    fig.update_layout(
        title=dict(
            text='Target Values Over Time',
            x=0.5,
            font=dict(size=36, color='#2E2E2E')
        ),
        xaxis=dict(
            title='Time',
            title_font=dict(size=30, color='#2E2E2E'),
            tickfont=dict(size=20, color='#2E2E2E'),
            gridcolor='rgba(200,200,200,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            title='Target',
            title_font=dict(size=30, color='#2E2E2E'),
            tickfont=dict(size=20, color='#2E2E2E'),
            gridcolor='rgba(200,200,200,0.2)',
            zeroline=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        width=1200,
        height=600,
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            y=1.05,
            xanchor='center',
            font=dict(size=20)
        )
    )

    # Adicionar informações customizadas ao hover
    fig.update_traces(
        hovertemplate='<b>Time:</b> %{x}<br><b>Target:</b> %{y:.2f}'
    )

    # Exibir o gráfico
    fig.show()




def inspect_datasets(X_train, y_train, X_test, y_test, window_size, num_original_features, feature_names=None, index=None):
    """
    Inspeciona os conjuntos de dados de treinamento e teste para verificar a organização correta e ausência de sobreposição.
    Também chama a função de plotagem para visualizar as sequências de dados.

    Parâmetros:
    - X_train (np.ndarray): Características de treinamento.
    - y_train (np.ndarray): Alvos de treinamento.
    - X_test (np.ndarray): Características de teste.
    - y_test (np.ndarray): Alvos de teste.
    - window_size (int): Tamanho da janela usada para características defasadas.
    - num_original_features (int): Número de características originais antes da defasagem.
    - feature_names (List[str], opcional): Nomes das características originais (antes da defasagem).
    - index (pd.Index ou np.ndarray, opcional): Índice ou timestamps associados aos dados.

    Retorna:
    - None
    """
    import pandas as pd
    import numpy as np

    # Imprimir as formas dos conjuntos de dados
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}\n")

    num_features = X_train.shape[1]
    expected_num_features = num_original_features * window_size

    if feature_names is None:
        feature_names = [f'Feature_{i+1}' for i in range(num_original_features)]

    # Gerar nomes de características defasadas
    lagged_feature_names = [
        f'{f_name}_lag_{lag}'
        for f_name in feature_names
        for lag in range(window_size, 0, -1)
    ]

    # Verificar se há características adicionais
    num_additional_features = num_features - expected_num_features
    if num_additional_features > 0:
        lagged_feature_names.extend(
            [f'Additional_Feature_{i+1}' for i in range(num_additional_features)]
        )

    if len(lagged_feature_names) != num_features:
        print("Descompasso entre os nomes gerados e o número de colunas. Ajustando os nomes das características.")
        lagged_feature_names = [f'Feature_{i}' for i in range(num_features)]

    # Criar DataFrames para melhor visualização
    X_train_df = pd.DataFrame(X_train, columns=lagged_feature_names)
    X_test_df = pd.DataFrame(X_test, columns=lagged_feature_names)

    # Ajustar o índice se fornecido
    if index is not None:
        X_train_df.index = index[window_size:-1]
        X_test_df.index = index[-1:]
        y_train_series = pd.Series(y_train, index=index[window_size:-1], name='Target')
        y_test_series = pd.Series(y_test, index=index[-1:], name='Target')
    else:
        train_indices = np.arange(window_size, window_size + len(X_train))
        test_indices = np.array([window_size + len(X_train)])
        X_train_df.index = train_indices
        X_test_df.index = test_indices
        y_train_series = pd.Series(y_train, index=train_indices, name='Target')
        y_test_series = pd.Series(y_test, index=test_indices, name='Target')

    # Exibir os últimos exemplos dos dados de treinamento
    print("Últimos exemplos dos dados de treinamento:")
    print(pd.concat([X_train_df.tail(), y_train_series.tail()], axis=1))
    print()

    # Exibir os dados de teste
    print("Dados de teste:")
    print(pd.concat([X_test_df, y_test_series], axis=1))
    print()

    # Verificar sobreposição entre conjuntos de treinamento e teste
    overlap = np.intersect1d(X_train_df.index, X_test_df.index)
    if len(overlap) > 0:
        print("Aviso: Há sobreposição entre os conjuntos de treinamento e teste nos seguintes índices:")
        print(overlap)
    else:
        print("Nenhuma sobreposição detectada entre os conjuntos de treinamento e teste.")

    # Chamar a função de plotagem
    plot_data_sequences(y_train, y_test, window_size, index)

    
def plot_data_sequences(y_train, y_test, window_size, index=None):
    """
    Plota as sequências de alvos de treinamento e teste ao longo do tempo usando Plotly, seguindo o estilo dos plots anteriores.

    Parâmetros:
    - y_train (np.ndarray): Alvos de treinamento.
    - y_test (np.ndarray): Alvos de teste.
    - window_size (int): Tamanho da janela usada para características defasadas.
    - index (pd.Index ou np.ndarray, opcional): Índice ou timestamps associados aos dados.

    Retorna:
    - None
    """
    import numpy as np
    import plotly.graph_objects as go
    import pandas as pd

    # Ajustar o índice se fornecido
    if index is not None:
        train_indices = index[window_size:-1]
        test_indices = index[-1:]
    else:
        train_indices = np.arange(window_size, window_size + len(y_train))
        test_indices = np.array([window_size + len(y_train)])

    # Criar traços para treinamento e teste
    fig = go.Figure()

    # Traço dos alvos de treinamento com preenchimento semi-transparente
    fig.add_trace(go.Scatter(
        x=train_indices,
        y=y_train,
        mode='lines',
        name='Training Target',
        line=dict(color='#206A92', width=5),
        fill='tozeroy',
        fillcolor='rgba(32,106,146,0.1)',  # Transparência de 10%
    ))

    # Traço dos alvos de teste
    fig.add_trace(go.Scatter(
        x=test_indices,
        y=y_test,
        mode='markers',
        name='Testing Target',
        marker=dict(color='red', size=20),
    ))

    # Atualizar layout com elementos artísticos
    fig.update_layout(
        title=dict(
            text='Target Values Over Time',
            x=0.5,
            font=dict(size=55, color='#2E2E2E')
        ),
        xaxis=dict(
            title='Time',
            title_font=dict(size=50, color='#2E2E2E'),
            tickfont=dict(size=40, color='#2E2E2E'),
            gridcolor='rgba(200,200,200,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            title='Target',
            title_font=dict(size=50, color='#2E2E2E'),
            tickfont=dict(size=40, color='#2E2E2E'),
            gridcolor='rgba(200,200,200,0.2)',
            zeroline=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        width=1200,
        height=600,
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            y=1.05,
            xanchor='center',
            font=dict(size=40)
        )
    )

    # Adicionar informações customizadas ao hover
    fig.update_traces(
        hovertemplate='<b>Time:</b> %{x}<br><b>Target:</b> %{y:.2f}'
    )

    # Exibir o gráfico
    fig.show()
    
    
    

from pykalman import KalmanFilter
import numpy as np
import pandas as pd
from typing import List


# def apply_custom_kalman_filter(data, process_var=1e-5, measurement_var=1e-3):
#     # Definir a matriz de transição e a variância do processo
#     transition_matrix = [[1]]  # assume uma série unidimensional, onde o valor depende do anterior
#     observation_matrix = [[1]]  # mede o estado atual

#     # Inicializar o filtro de Kalman com variâncias de processo e medição
#     kf = KalmanFilter(
#         transition_matrices=transition_matrix,
#         observation_matrices=observation_matrix,
#         transition_covariance=process_var * np.eye(1),
#         observation_covariance=measurement_var * np.eye(1),
#         initial_state_mean=0
#     )

#     # Aplicar o filtro de Kalman nos dados
#     data_kalman, _ = kf.filter(data)
#     return data_kalman.flatten()  # Retorna como uma lista

def apply_kalman_filter(
    df: pd.DataFrame,
    features_to_filter: List[str],
    process_var: float = 1e-5,
    measurement_var: float = 1e-3
) -> pd.DataFrame:
    """
    Aplica um filtro de Kalman personalizado nas features especificadas do DataFrame.
    
    Para cada feature em `features_to_filter`, cria uma nova coluna com o sufixo '_Original' 
    que armazena a série original e substitui a série original pela série filtrada.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        features_to_filter (List[str]): Lista de nomes das features a serem filtradas.
        process_var (float, optional): Variância do processo para o filtro de Kalman. Defaults to 1e-5.
        measurement_var (float, optional): Variância da medição para o filtro de Kalman. Defaults to 1e-3.
    
    Returns:
        pd.DataFrame: DataFrame com as séries filtradas e originais preservadas.
    """
    for feature in features_to_filter:
        if feature not in df.columns:
            raise ValueError(f"A feature '{feature}' não existe no DataFrame.")

        # Preserva a série original
        original_feature = f"{feature}_Original"
        df[original_feature] = df[feature].copy()

        # Aplica o filtro de Kalman
        df[feature] = apply_custom_kalman_filter(df[feature], process_var, measurement_var)

    return df

def apply_custom_kalman_filter(data, process_var=1e-4, measurement_var=1e-3, epsilon=1e-6):
    """
    Aplica um Filtro de Kalman personalizado com restrição de positividade.

    Parâmetros:
    - data: array-like, dados a serem filtrados.
    - process_var: float, variância do processo.
    - measurement_var: float, variância da medição.
    - epsilon: float, pequena constante para evitar log(0).

    Retorna:
    - data_kalman: array-like, dados filtrados com positividade garantida.
    """
    # Garantir que todos os dados sejam positivos adicionando epsilon se necessário
    data_positive = np.array(data) + epsilon
    data_positive[data_positive <= 0] = epsilon  # Substituir valores não positivos por epsilon

    # Aplicar a transformação logarítmica
    # log_data = np.log(data_positive)
    
    log_data = data_positive

    # Definir a matriz de transição e a variância do processo
    transition_matrix = [[1]]  # Série unidimensional
    observation_matrix = [[1]]  # Observa o estado atual

    # Inicializar o filtro de Kalman com as variâncias de processo e medição
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=process_var * np.eye(1),
        observation_covariance=measurement_var * np.eye(1),
        initial_state_mean=log_data[0],
        initial_state_covariance=1
    )

    # Aplicar o filtro de Kalman nos dados transformados
    state_means, state_covariances = kf.filter(log_data)

    data_kalman = state_means.flatten()

    return data_kalman


def filter_data(X_train_aug, X_test, y_train_aug, option=1,
                process_var=1e-4, measurement_var=1e-3, epsilon=1e-6):
    """
    Applies a custom Kalman filter to the provided data according to the selected option.

    Parameters:
    - X_train_aug: pd.DataFrame, training features.
    - X_test: pd.DataFrame, testing features.
    - y_train_aug: array-like or pd.Series, training target.
    - option: int, either:
         1 -> Filter ALL columns of X_train_aug and X_test, plus y_train_aug.
         2 -> Filter only columns containing 'lag' in their names in X_train_aug and X_test, plus y_train_aug.
    - process_var: float, process variance for the Kalman filter.
    - measurement_var: float, measurement variance for the Kalman filter.
    - epsilon: float, a small constant to ensure positivity.

    Returns:
    - X_train_filtered: pd.DataFrame, filtered training features.
    - X_test_filtered: pd.DataFrame, filtered testing features.
    - y_train_filtered: np.array, filtered training target.
    """
    
    # Create copies of the DataFrames so the originals remain unaltered
    X_train_filtered = X_train_aug.copy()
    X_test_filtered = X_test.copy()

    # Option 1: Filter ALL columns
    if option == 1:
        cols_to_filter_train = X_train_filtered.columns
        cols_to_filter_test = X_test_filtered.columns
    # Option 2: Filter only columns with 'lag' in their name
    elif option == 2:
        cols_to_filter_train = [col for col in X_train_filtered.columns if 'lag' in col]
        cols_to_filter_test = [col for col in X_test_filtered.columns if 'lag' in col]
    else:
        raise ValueError("Invalid option. Please choose 1 or 2.")

    # Apply filter to the selected columns for X_train_aug
    for col in cols_to_filter_train:
        X_train_filtered[col] = apply_custom_kalman_filter(
            X_train_filtered[col].values, process_var, measurement_var, epsilon)

    # Apply filter to the selected columns for X_test
    for col in cols_to_filter_test:
        X_test_filtered[col] = apply_custom_kalman_filter(
            X_test_filtered[col].values, process_var, measurement_var, epsilon)

    # Always filter y_train_aug (converted to a numpy array if necessary)
    y_train_filtered = apply_custom_kalman_filter(
        y_train_aug, process_var, measurement_var, epsilon)
    
    
    return X_train_filtered, X_test_filtered, pd.DataFrame(y_train_filtered)