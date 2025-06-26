import pandas as pd
import numpy as np
import pywt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from common.common import split_time_series, augment_with_synthetic_samples, augment_phys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from forecast_pipeline.config import CANON_FEATURES
from forecast_pipeline.config import DEFAULT_DATASET


import numpy as np
from typing import Any, Dict, List, Optional
import logging

# -------------------------------------------------
# Seq-to-Seq pipeline
# -------------------------------------------------

def create_sliding_window_seq_to_seq(df: pd.DataFrame, target_col: str, input_length: int, output_length: int, stride: int = 1):
    """
    Creates sliding window samples for sequence-to-sequence learning.
    
    For each window, X contains input_length rows with all features,
    while y contains output_length rows from the target column.
    """
    data = df.values
    target_idx = df.columns.get_loc(target_col)
    X, y = [], []
    total_length = data.shape[0]
    for i in range(0, total_length - input_length - output_length + 1, stride):
        X_window = data[i: i + input_length, :]  # input window: all features
        y_window = data[i + input_length: i + input_length + output_length, target_idx]  # forecast window: target only
        X.append(X_window)
        y.append(y_window)
    return np.array(X), np.array(y)


"""
The difference occurs because window creation requires that each window contain a continuous block of rows with a size equal to **input_length + output_length**. This means that not every row in the original dataset can be the starting point of a complete window.

Here's how it works:

- **Formula:** For a dataframe with \( n \) rows, the number of possible windows is:
\[
\text{number of windows} = n - (\text{input\_length} + \text{output\_length}) + 1
\]

- **Training set:**
\( n = 1135 \)
\( \text{input\_length} = 7 \)
\( \text{output\_length} = 30 \)
So, the number of windows is:
\[
1135 - (7 + 30) + 1 = 1135 - 37 + 1 = 1099
\]

- **Test set:**
\( n = 1701 \)
Likewise:
\[
1701 - 37 + 1 = 1665
\]

Adding: \( 1099 + 1665 = 2764 \) windows.

Therefore, of the original 2836 records, 72 "lost" records (36 in each set) cannot start a complete window of 7+30 lines. This behavior is expected when using the sliding window method, since the records at the end of each set do not have enough subsequent lines to form the complete window.
"""


import joblib
import os

def save_scaler(scaler: StandardScaler, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)
    print(f"Scaler successfully saved at {filepath}.")



class RobustStandardScaler(StandardScaler):
    def fit(self, X, y=None):
        super().fit(X, y)
        # guard band
        self.scale_ = np.where(self.scale_ < 1e-3, 1e-3, self.scale_)
        return self

if DEFAULT_DATASET == "VOLVE":


    def prepare_data_seq(
        df: pd.DataFrame,
        target_col: str,
        input_length: int,
        output_length: int,
        test_size: float = 0.6,
        val_size: float = 0.1,
        data_aug: bool = True
    ):
        """
        Prepara os dados para seq-to-seq usando início do teste como validação (sem embaralhar):
    
        1. Divide cronologicamente em (treino) e (teste).
        2. Cria janelas deslizantes para treino e teste.
        3. (Opcional) Data augmentation no conjunto de treino.
        4. Separa características não-alvo e alvo, achata, escala e remonta janelas.
        5. Calcula y escalado para treino e teste.
        6. Usa os primeiros val_size% de TESTE como validação.
    
        Retorna:
          X_train, X_val, X_test,
          y_train, y_val, y_test,
          scaler_target,
          y_train_original
        """
        # 1. Split temporally em treino+val e teste
        df_temp, df_test = split_time_series(df, test_size) 
    
        # 2. Sliding windows
        X_temp, y_temp = create_sliding_window_seq_to_seq(
            df_temp, target_col, input_length, output_length
        )
        X_test_all, y_test_all = create_sliding_window_seq_to_seq(
            df_test, target_col, input_length, output_length
        )
    
        # Keep original y_train
        y_train_original = y_temp.copy()
    
        # 3. Data augmentation (apenas em treino)
        if data_aug:
            X_temp, y_temp = augment_with_synthetic_samples(X_temp, y_temp)
    
        # if data_aug:
        #     X_temp, y_temp = augment_phys(
        #         X_temp, y_temp,
        #         feature_indices={
        #                 "PI":0,
        #                 "CE":1,
        #                 "BORE_GAS_VOL":2,
        #                 "AVG_DOWNHOLE_PRESSURE":3,
        #                 "AVG_WHP_P":4,
        #                 "Tempo_Inicio_Prod":5,
        #                 "Taxa_Declinio":6,
        #                 "BORE_OIL_VOL":7,
        #         },
        #         scale_factors=(1.5, 2, 3, 5, 7, 9, 11, 13, 15, 19),
        #         use_scale=True,
        #         use_jitter=False,
        #         use_decline_warp=False,
        #         n_bootstrap=0,           # duas cópias bootstrap
        #         sigma_pressure=8.0,
        #         random_state=42
        #     )
    
    
        # 4. Separa features / target em X
        X_temp_feats = X_temp[:, :, :-1]
        X_temp_targ  = X_temp[:, :, -1:].reshape(-1, 1)
        X_test_feats = X_test_all[:, :, :-1]
        X_test_targ  = X_test_all[:, :, -1:].reshape(-1, 1)
    
        # Achata para escalar
        n_temp, win_len, n_feat = X_temp_feats.shape
        n_test = X_test_feats.shape[0]
        X_temp_flat = X_temp_feats.reshape(-1, n_feat)
        X_test_flat = X_test_feats.reshape(-1, n_feat)
    
        # 5a. Escala features
        scaler_X = StandardScaler()
        # scaler_X = RobustScaler()
        X_temp_scaled_feats = scaler_X.fit_transform(X_temp_flat)
        X_test_scaled_feats = scaler_X.transform(X_test_flat)
        X_temp_scaled_feats = X_temp_scaled_feats.reshape(n_temp, win_len, n_feat)
        X_test_scaled_feats = X_test_scaled_feats.reshape(n_test, win_len, n_feat)
        save_scaler(scaler_X, 'scalers/scaler_X.pkl')
    
        # 5b. Escala target em X
        scaler_target = StandardScaler()
        # scaler_target = RobustScaler()
        X_temp_scaled_targ = scaler_target.fit_transform(X_temp_targ)
        X_test_scaled_targ = scaler_target.transform(X_test_targ)
        X_temp_scaled_targ = X_temp_scaled_targ.reshape(n_temp, win_len, 1)
        X_test_scaled_targ = X_test_scaled_targ.reshape(n_test, win_len, 1)
        save_scaler(scaler_target, 'scalers/scaler_target.pkl')
    
        # 6. Reconstroi janelas escaladas
        X_temp_scaled = np.concatenate([X_temp_scaled_feats, X_temp_scaled_targ], axis=-1)
        X_test_scaled = np.concatenate([X_test_scaled_feats, X_test_scaled_targ], axis=-1)
    
        # 7. Escala y
        y_temp_flat = y_temp.flatten().reshape(-1, 1)
        y_temp_scaled = scaler_target.transform(y_temp_flat).flatten().reshape(y_temp.shape)
        y_test_flat = y_test_all.flatten().reshape(-1, 1)
        y_test_scaled = scaler_target.transform(y_test_flat).flatten().reshape(y_test_all.shape)
    
        # 8. Usa início de TESTE como validação
        n_test_windows = X_test_scaled.shape[0]
        n_val = int(n_test_windows * val_size)
        X_val = X_test_scaled[:n_val]
        y_val = y_test_scaled[:n_val]
        X_test = X_test_scaled[n_val:]
        y_test = y_test_scaled[n_val:]
    
        # Conjunto de treino completo é X_temp_scaled
        X_train = X_temp_scaled
        y_train = y_temp_scaled
    
        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler_target,
            y_train_original
        )


elif DEFAULT_DATASET == "UNISIM_IV":

    def prepare_data_seq(
        df: pd.DataFrame,
        target_col: str,
        input_length: int,
        output_length: int,
        test_size: float = 0.6,
        val_size: float = 0.1,
        data_aug: bool = True
    ):
        # ------------------------------------------- 1. split
        df_temp, df_test = split_time_series(df, test_size)
    
        X_temp_raw, y_temp_raw = create_sliding_window_seq_to_seq(
            df_temp, target_col, input_length, output_length
        )
        X_test_raw, y_test_raw = create_sliding_window_seq_to_seq(
            df_test, target_col, input_length, output_length
        )
    
        # ------------------------------------------- 2. FIT SCALERS *ANTES* DA DA
        # achata apenas o TRAIN real
        n_temp, win_len, n_feat = X_temp_raw.shape
        X_temp_flat = X_temp_raw.reshape(-1, n_feat)
    
        scaler_X = RobustStandardScaler().fit(X_temp_flat)
        scaler_y = StandardScaler().fit(y_temp_raw.flatten().reshape(-1, 1))
    
        save_scaler(scaler_X, 'scalers/scaler_X.pkl')          # caminhos únicos por case
        save_scaler(scaler_y, 'scalers/scaler_target.pkl')
    
        # ------------------------------------------- 3. APPLY SCALERS
        def scale_X(X):
            sh = X.shape
            return scaler_X.transform(X.reshape(-1, n_feat)).reshape(sh)
    
        def scale_y(y):
            return scaler_y.transform(y.reshape(-1, 1)).reshape(y.shape)
    
        X_temp_scaled = scale_X(X_temp_raw)
        y_temp_scaled = scale_y(y_temp_raw)
        X_test_scaled = scale_X(X_test_raw)
        y_test_scaled = scale_y(y_test_raw)
    
        # ------------------------------------------- 4. DATA AUG (agora no espaço z-score)
        # if data_aug:
        #     X_temp_scaled, y_temp_scaled = augment_with_synthetic_samples(
        #         X_temp_scaled, y_temp_scaled, scales=[1.5,2,3,5,7,9,]   # ≤4 fatores já é suficiente
        #     )
    
        # ------------------------------------------- 5. reshape target dentro de X …
        # (mesma lógica que já tinha, mas usando arrays _scaled_)
        X_temp_feats = X_temp_scaled[:, :, :-1]
        X_temp_targ  = X_temp_scaled[:, :, -1:]
        X_test_feats = X_test_scaled[:, :, :-1]
        X_test_targ  = X_test_scaled[:, :, -1:]
    
        # concatena alvo nas features
        X_train = np.concatenate([X_temp_feats, X_temp_targ], axis=-1)
        X_test  = np.concatenate([X_test_feats, X_test_targ], axis=-1)
    
        # ------------------------------------------- 6. val split
        n_val = int(X_test.shape[0] * val_size)
        X_val, y_val = X_test[:n_val], y_test_scaled[:n_val]
        X_test, y_test = X_test[n_val:], y_test_scaled[n_val:]
    
        return (X_train, X_val, X_test,
                y_temp_scaled, y_val, y_test,
                scaler_y,          # para inversão de previsões
                y_temp_raw)        # curva original p/ métrica




# -------------------------------------------------
# Stub / placeholder functions for training & evaluation
# (Replace these with your actual implementations.)
# -------------------------------------------------

def prepare_inputs_seq(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_kind, selected_features, main_feature):
    """
    Prepares inputs for training a seq-to-seq model.
    
    For example, one might package the training arrays in a dictionary.
    """
    train_kwargs = {
        "X_train": X_train_scaled,
        "y_train": y_train_scaled
    }
    prediction_input = X_test_scaled
    
    return train_kwargs, prediction_input

def denormalize_target_column(X_scaled: np.ndarray, scaler_target) -> np.ndarray:
    """
    Denormalizes the target column (last feature) in the input array X_scaled.
    
    Parameters:
        X_scaled (np.ndarray): Scaled input data of shape (n_samples, win_len, n_features).
        scaler_target: A fitted scaler (e.g., StandardScaler) for the target variable.
        
    Returns:
        np.ndarray: X_scaled with the last column (target feature) denormalized.
    """
    # Create a copy to avoid modifying the original array
    X_denorm = X_scaled.copy()
    n_samples, win_len, _ = X_scaled.shape
    # Reshape the target column to 2D, apply inverse transformation, then reshape back
    X_denorm[:, :, -1] = scaler_target.inverse_transform(
        X_denorm[:, :, -1].reshape(-1, 1)
    ).reshape(n_samples, win_len)
    return X_denorm

def denormalize_targets(y_scaled: np.ndarray, scaler_target) -> np.ndarray:
    """
    Denormalizes the target sequences y_scaled using the provided scaler.
    
    Parameters:
        y_scaled (np.ndarray): Scaled target data.
        scaler_target: A fitted scaler (e.g., StandardScaler) for the target variable.
        
    Returns:
        np.ndarray: Denormalized target sequences.
    """
    return scaler_target.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)

def plot_denormalized_forecast(X_scaled: np.ndarray, y_scaled: np.ndarray, scaler_target, 
                               input_length: int, output_length: int, theme: str = 'black') -> None:
    """
    Denormalizes the scaled input and target data and plots the forecast windows.
    
    This function:
      1. Denormalizes the target column of X_scaled.
      2. Denormalizes the target sequences y_scaled.
      3. Plots the forecast windows using the denormalized data.
    
    Parameters:
        X_scaled (np.ndarray): Scaled input data of shape (n_samples, win_len, n_features).
        y_scaled (np.ndarray): Scaled target sequences.
        scaler_target: A fitted scaler (e.g., StandardScaler) for the target variable.
        input_length (int): Length of the input window.
        output_length (int): Length of the forecast window.
        theme (str): Plot theme to be passed to the plotting function.
    
    Returns:
        None
    """
    from evaluation.evaluation import plot_forecast_windows
    X_denorm = denormalize_target_column(X_scaled, scaler_target)
    y_denorm = denormalize_targets(y_scaled, scaler_target)
    
    # Plot using the last column of X_denorm which corresponds to the target variable
    plot_forecast_windows(X_denorm[:, :, -1], y_denorm, input_length, output_length, theme=theme)
    
# Filtering and soft target functions
def filter_signal(signal: np.ndarray, method: str = "exponential_smoothing", **kwargs) -> np.ndarray:
    """
    Filters a 1D signal using the specified method.
    
    Supported methods:
      - "exponential_smoothing": Uses Holt’s exponential smoothing.
      - "wavelet": Uses wavelet denoising (with 'db1' and level=3).
      - "kalman": Uses a Kalman filter (requires pykalman).
    """
    if method == "exponential_smoothing":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        smoothing_level = kwargs.get("smoothing_level", 0.2)
        model = ExponentialSmoothing(signal, trend=None, seasonal=None, initialization_method="estimated")
        fit = model.fit(smoothing_level=smoothing_level)
        return fit.fittedvalues
    elif method == "wavelet":
        import pywt
        coeffs = pywt.wavedec(signal, 'db1', level=3)
        threshold = kwargs.get("threshold", np.std(signal) * 0.5)
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        filtered = pywt.waverec(coeffs, 'db1')
        return filtered[:len(signal)]
    elif method == "kalman":
        from pykalman import KalmanFilter
        process_var = kwargs.get("process_var", 1e-5)
        measurement_var = kwargs.get("measurement_var", 1e-3)
        kf = KalmanFilter(initial_state_mean=signal[0],
                          n_dim_obs=1,
                          transition_covariance=process_var * np.eye(1),
                          observation_covariance=measurement_var * np.eye(1))
        state_means, _ = kf.smooth(signal)
        return state_means.flatten()
    else:
        raise ValueError(f"Unknown filtering method: {method}")

def apply_filter_to_X_and_y(X: np.ndarray, y: np.ndarray, method: str = "exponential_smoothing", **kwargs):
    """
    Applies filtering to the target series contained in X (last column) and to y.
    """
    X_filtered = np.copy(X)
    for i in range(X.shape[0]):
        signal = X[i, :, -1]
        X_filtered[i, :, -1] = filter_signal(signal, method=method, **kwargs)
    
    y_filtered = np.copy(y)
    for i in range(y.shape[0]):
        y_filtered[i, :] = filter_signal(y[i, :], method=method, **kwargs)
    
    return X_filtered, y_filtered

def generate_soft_targets(y: np.ndarray, method: str = "exponential_smoothing", **kwargs) -> np.ndarray:
    """
    Generates soft targets from raw target sequences by filtering.
    """
    y_soft = np.empty_like(y)
    for i in range(y.shape[0]):
        y_soft[i, :] = filter_signal(y[i, :], method=method, **kwargs)
    return y_soft


import numpy as np
from typing import List, Optional, Dict

def reconstruct_true_series(y_full: np.ndarray) -> np.ndarray:
    """
    Reconstruct original series from overlapping windows (stride=1).
    First window gives its full horizon; then each next window
    contributes only its last element.
    """
    n_samples, horizon = y_full.shape
    L = n_samples + horizon - 1
    out = np.empty(L)
    out[:horizon] = y_full[0, :]
    for i in range(1, n_samples):
        out[i + horizon - 1] = y_full[i, -1]
    return out

import numpy as np
from typing import List, Optional, Dict

def aggregate_predictions(
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Reconstrói a série a partir de janelas seq2seq,
    mas, em vez de usar só o último ponto de cada janela,
    usa a média dos seus últimos `tail_size` pontos.

    Parameters
    ----------
    predictions : np.ndarray, shape (n_windows, horizon)
        Saída do modelo para cada janela.
    tail_size : int
        Quantos passos finais de cada janela incluir na média.

    Returns
    -------
    out : np.ndarray, shape (n_windows + horizon - 1,)
        Série reconstruída, mantendo o mesmo lag de
        reconstruct_true_series, mas suavizada na cauda.
    """
    tail_size = 7
    n_windows, horizon = predictions.shape
    L = n_windows + horizon - 1
    out = np.empty(L)

    # 1) Copia a primeira janela inteira (comportamento original)
    out[:horizon] = predictions[0, :]

    # 2) Para cada janela i>0, coloca no índice correto a média
    #    das últimas tail_size previsões DESSA janela
    for i in range(1, n_windows):
        start = horizon - tail_size
        tail_vals = predictions[i, start:horizon]
        out[i + horizon - 1] = tail_vals.mean()

    return out