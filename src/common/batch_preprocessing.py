import os
import time
from typing import Any, List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import logging

from common.common import split_time_series, augment_with_synthetic_samples

# --- Helper Functions ---

def generate_lagged_features(
    df: pd.DataFrame,
    target_col: str,
    lag_window: int,
    horizon: int = 30
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates lagged features for time series regression.

    Parameters:
      df: DataFrame containing the time series.
      target_col: The name of the target feature.
      lag_window: The number of lags to include.
      horizon: The forecast horizon (to shift the target).

    Returns:
      A tuple of (DataFrame with lagged features, Series of original target values).
    """
    df_lagged = df.copy()
    # Create lag features for each lag in [0, lag_window)
    for lag in range(lag_window):
        df_lagged[f"{target_col}_lag_{lag}"] = df_lagged[target_col].shift(lag)
    # Create future target by shifting by -horizon
    df_lagged[f"{target_col}_target"] = df_lagged[target_col].shift(-horizon)
    # Preserve a copy of the original target for later evaluation
    # (Assuming there is a column 'BORE_OIL_VOL_Original'; if not, create one)
    if f"{target_col}_Original" not in df_lagged.columns:
        df_lagged[f"{target_col}_Original"] = df_lagged[target_col]
    # Shift the original target as well
    df_lagged[f"{target_col}_Original"] = df_lagged[f"{target_col}_Original"].shift(-horizon)
    # Drop rows with NaN values (introduced by shifting)
    df_lagged = df_lagged.dropna()
    target_original = df_lagged[f"{target_col}_Original"].copy()
    # Remove the original target column from features to avoid leakage
    df_lagged = df_lagged.drop(columns=[f"{target_col}_Original"])
    return df_lagged, target_original


def load_and_preprocess_data(DataSource, config, selected_features, well):
    # Load data using the provided configuration
    data_source_obj = DataSource(config)
    loader = data_source_obj.get_loader()
    df_loaded = loader.load()

    # If data is a dict, extract the data for the specific well
    if isinstance(df_loaded, dict):
        df_loaded = df_loaded.get(well, pd.DataFrame())
    if df_loaded.empty:
        raise ValueError(f"No data loaded for well {well}. Check your configuration and data source.")
    return df_loaded[selected_features].copy()


def prepare_data(df, main_feature, lag_window, horizon, test_size, data_aug=True):
    """
    Creates lagged features, splits data into train/test, and scales the datasets.
    
    Returns:
      - X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y
    """
    # Prepare datasets without deep learning augmentation (for_dl=False)
    X_train_aug, X_test, y_train_aug, y_test, y_train_cumsum = prepare_datasets(
        df=df,
        target_col=main_feature,
        lag_window=lag_window,
        horizon=horizon,
        test_size=test_size,
        data_aug=data_aug
    )
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, _, scaler_y = scale_and_reshape_data(
        X_train_aug, X_test, y_train_aug, y_test
    )
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y, y_train_cumsum


def prepare_inputs(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
                   feature_kind, selected_features, main_feature):
    """
    Prepares the inputs for the model based on the feature type (Branch or Normal).

    Returns:
      - train_kwargs: dict with training data.
      - prediction_input: input to be used for model prediction.
    """
    if feature_kind == 'Branch':
        logging.info("Using Multi-Branch inputs")
        branch_inputs_train, branch_inputs_test = prepare_branch_inputs(
            X_train_scaled, X_test_scaled, selected_features, main_feature
        )
        train_kwargs = {
            'branch_inputs_train': branch_inputs_train,
            'branch_inputs_test': branch_inputs_test,
            'y_train_scaled': y_train_scaled,
            'y_test_scaled': y_test_scaled
        }
        prediction_input = branch_inputs_test
    else:
        logging.info("Using Normal (Single) input")
        normal_inputs_train, normal_inputs_test, _, _ = prepare_normal_inputs(
            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
        )
        train_kwargs = {
            'X_train': normal_inputs_train,
            'y_train': y_train_scaled
        }
        prediction_input = normal_inputs_test

    return train_kwargs, prediction_input


def prepare_datasets(
    df: pd.DataFrame,
    target_col: str,
    lag_window: int,
    horizon: int = 30,
    test_size: float = 0.7,
    data_aug: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepares datasets for time series regression by generating lagged features,
    splitting the data, and augmenting the training set.

    Parameters:
      df: Input DataFrame.
      target_col: Name of the target feature.
      lag_window: Window size for lag features.
      horizon: Forecast horizon.
      test_size: Proportion of data for testing.

    Returns:
      A tuple:
        (X_train_original, X_train_augmented, X_test,
         y_train_original, y_train_augmented, y_test, target_original)
    """
    # Generate lagged features and obtain the original target for evaluation
    df_lagged, target_original = generate_lagged_features(df, target_col, lag_window, horizon)
    
    # Split the lagged dataset into training and testing sets
    df_train, df_test = split_time_series(df_lagged, target_original, test_size)
    print('df_train.shape, df_test.shape', df_train.shape, df_test.shape)
    
    # Determine feature columns (all except the target columns)
    feature_cols = [col for col in df_lagged.columns if col not in [target_col, f"{target_col}_target"]]
    X_train_original = df_train[feature_cols].copy()
    y_train_original = df_train[f"{target_col}_target"].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test[f"{target_col}_target"].copy()
    
    # Cálculo da soma acumulada do treino
    y_train_cumsum = np.cumsum(np.array(y_train_original))
    
    # Create synthetic (augmented) samples from training data
    X_train_aug, y_train_aug = augment_with_synthetic_samples(X_train_original, y_train_original)
    
    if not data_aug:
        X_train_aug, y_train_aug = X_train_original, y_train_original
    
    return X_train_aug, X_test, y_train_aug, y_test, y_train_cumsum[-1]




def scale_and_reshape_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Scales the training and testing features and targets using StandardScaler,
    then reshapes the feature arrays to have a time-step dimension.

    Parameters:
        X_train: Training features as a DataFrame.
        X_test: Testing features as a DataFrame.
        y_train: Training target as a Series.
        y_test: Testing target as a Series.

    Returns:
        A tuple containing:
            - X_train_scaled: Scaled and reshaped training features (samples, 1, features).
            - X_test_scaled: Scaled and reshaped testing features (samples, 1, features).
            - y_train_scaled: Scaled training targets (flattened array).
            - y_test_scaled: Scaled testing targets (flattened array).
            - scaler_X: Fitted scaler for features.
            - scaler_y: Fitted scaler for targets.
    """
    # Create scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Convert targets to numpy arrays (flatten)
    y_train_arr = y_train.values.flatten()
    y_test_arr = y_test.values.flatten()

    # Scale features
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale targets (reshape to 2D for the scaler, then flatten)
    y_train_scaled = scaler_y.fit_transform(y_train_arr.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test_arr.reshape(-1, 1)).flatten()

    # Reshape feature arrays to include a time-step dimension (assumed to be 1)
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y





# src/darts_common/preprocessing.py
from itertools import combinations
import numpy as np
import pandas as pd
from typing import List, Tuple

def prepare_normal_inputs(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For generic (single‐input) models, return the normalized training and testing arrays.
    """
    return X_train_scaled, X_test_scaled, y_train, y_test

def prepare_branch_inputs(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    feature_names: List[str],
    target_feature: str,
    combination_size: int = 3
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Creates branch inputs for multi-branch models.
    
    This function selects combinations of features (excluding the target) and for each combination
    appends the target feature. Then, for each combination, it extracts the corresponding features
    from X_train_scaled and X_test_scaled and reshapes them to have a time-step dimension.
    
    Parameters:
        X_train_scaled: Normalized training data with shape (samples, 1, num_features).
        X_test_scaled: Normalized testing data with shape (samples, 1, num_features).
        feature_names: List of feature names in order.
        target_feature: The target feature name.
        combination_size: Number of non-target features to combine.
        
    Returns:
        A tuple (branch_inputs_train, branch_inputs_test) where each is a list of arrays.
    """
    # All features except target
    non_target_features = [f for f in feature_names if f != target_feature]
    comb = list(combinations(non_target_features, combination_size))
    feature_combinations = [tuple(list(c) + [target_feature]) for c in comb]
    
    branch_inputs_train = []
    branch_inputs_test = []
    
    print(f'feature_combinations: {feature_combinations}')
    
    for features in feature_combinations:
        # Get indices for the selected features
        indices = [feature_names.index(f) for f in features]
        # Remove the time-step dimension (assumed to be 1) and then select the columns
        X_train_subset = X_train_scaled[:, 0, indices]
        X_test_subset = X_test_scaled[:, 0, indices]
        # Restore time-step dimension (1 timestep)
        X_train_subset = X_train_subset.reshape((X_train_subset.shape[0], 1, X_train_subset.shape[1]))
        X_test_subset = X_test_subset.reshape((X_test_subset.shape[0], 1, X_test_subset.shape[1]))
        branch_inputs_train.append(X_train_subset)
        branch_inputs_test.append(X_test_subset)
    
    return branch_inputs_train, branch_inputs_test
