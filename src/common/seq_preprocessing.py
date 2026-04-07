#src/common/seq_preprocessing.py
from __future__ import annotations

# ===================================================================================================================================================
#                      --- Module Function Roadmap ---
# ===================================================================================================================================================
# | Function Name                          | Key Role                     | Purpose                                                                 |
# |----------------------------------------|------------------------------|-------------------------------------------------------------------------|
# | `prepare_data_seq`                     | Main Entry Point (Seq2Seq)   | The primary function to process a DataFrame for Seq2Seq models.         |
# | `prepare_data_for_darts`               | Main Entry Point (Darts)     | The primary function to process a DataFrame for Darts-based models.     |
# | `create_sliding_window_seq_to_seq`     | Core Windowing Logic         | Converts a time series into overlapping (X, y) windows for training.    |
# | `aggregate_predictions`                | Post-processing              | Reconstructs a 1D forecast from a set of overlapping prediction windows.|
# | `check_seq2seq_data_leakage`           | Auditing & Validation        | Verifies that no future information has leaked into the training data.  |
# | `report_split_sample_counts`           | Auditing & Validation        | Prints a detailed report on how the time series was partitioned.        |
# | `audit_seq2seq_continuity_from_lengths`| Auditing & Validation        | Checks for temporal gaps between the train, validation, and test sets.  |
# | `inspect_set_boundaries`               | Debugging & Inspection       | Prints the data values at the split boundaries to visualize continuity. |
# | `calculate_split_boundaries`           | Core Splitting Logic         | Computes the exact indices for splitting the data chronologically.      |
# | `filter_signal`                        | Signal Processing            | Applies filters (e.g., EWMA, Kalman) to a 1D signal to smooth it.       |
# | `generate_soft_targets`                | Signal Processing            | Creates a smoothed version of the target `y` arrays for training.       |
# | `analyze_scaler_effects`               | Diagnostics                  | Prints a report comparing different data scaling strategies.            |
# | `save_scaler`                          | Utility                      | Saves a fitted scaler object (e.g., StandardScaler) to a file.          |
# ===================================================================================================================================================

"""
    STRATEGY: Chronological Split with Causal Context Injection
    ===========================================================

    CONCEPT
    -------
    This implementation solves the "Cold Start" problem inherent in sliding window 
    generation for time series. Instead of discarding the first `L` samples of 
    Validation/Test sets (due to lack of history), it injects the tail of the 
    previous set strictly as *input context* (X features), ensuring the target 
    (y) remains causally isolated.

    ILLUSTRATIVE EXAMPLE
    --------------------
    Parameters: 
      - Lookback (L) = 2
      - Horizon (H) = 1
      - Split Boundary = Between Index 9 and 10

    +-------+-------+----------+---------------------------+-------------------------------+
    | Index | Value | Set      | Feature Window (X)        | Target (y)                    |
    | (t)   | y(t)  | Assignment| [t-L : t]                 | [t : t+H]                     |
    +-------+-------+----------+---------------------------+-------------------------------+
    | ...   | ...   | TRAIN    | ...                       | ...                           |
    | 09    | 90    | TRAIN    | [70, 80]                  | [90]                          |
    +-------+-------+----------+---------------------------+-------------------------------+
    |                  HARD SPLIT BOUNDARY (No y overlap)                                  |
    +-------+-------+----------+---------------------------+-------------------------------+
    | 10    | 100   | VAL      | [80, 90]  <-- Injected    | [100]                         |
    | 11    | 110   | VAL      | [90, 100]                 | [110]                         |
    +-------+-------+----------+---------------------------+-------------------------------+

    ARCHITECTURAL ELEGANCE & SAFETY
    -------------------------------
    1. ZERO DATA LEAKAGE (Safety):
       - Strict separation of targets: y_train indices < Split < y_val indices.
       - The model predicts y(10) (Val) using X(10) (derived from Train data). 
       - This is valid inference, not leakage: predicting the future requires knowledge 
         of the immediate past.

    2. SAMPLE PRESERVATION (Metric Integrity):
       - Naive implementations drop the first L samples of validation, skewing metrics 
         on small datasets.
       - This strategy maintains `len(y_val) == len(df_val)`, ensuring every sample 
         allocated to validation is actually evaluated.

    3. "SIDECAR" PATTERN FOR STATEFUL FILTERS:
       - The function attaches a `_split_ctx` payload to the scaler object.
       - This carries the exact boundary values (scaled) needed to "warm start" 
         stateful post-processors (e.g., recursive filters like HP or EWMA) 
         without requiring the monolithic training set during inference/deployment.
"""

# --- Standard Library Imports ---

import hashlib
import logging
import os
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, Union

# --- Third-Party Imports ---
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import statsmodels.api as sm
from pykalman import KalmanFilter
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from statsmodels.tsa.api import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Local Application Imports ---
from common.common import augment_with_synthetic_samples, split_time_series
from forecast_pipeline.config import CANON_FEATURES, DEFAULT_DATASET
from forecast_pipeline.logging_utils import (
    box_log,
    get_logger,
    log_context,
    log_da_usage,
    phase,
)

# --- Optional Imports for Rich Display ---
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define fallback dummies for type safety and basic functionality
    Console, Panel, Table = (object, object, object)

def create_sliding_window_seq_to_seq(
    df: pd.DataFrame,
    target_col: str,
    input_length: int,
    output_length: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for seq2seq learning.

    X shape: (N, L, F) with **target as the last channel**.
    y shape: (N, H)   with future target values.

    The function reorders columns so that the target feature is last in X.
    """
    if input_length <= 0 or output_length <= 0:
        raise ValueError("input_length and output_length must be > 0")
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in dataframe")

    # Reorder columns so target is last channel in X
    feature_cols = [c for c in df.columns if c != target_col]
    ordered_cols = feature_cols + [target_col]
    data = df[ordered_cols].values  # (T, F) where F = len(feature_cols) + 1 (target last)

    T, F = data.shape
    if T < input_length + output_length:
        # Not enough samples to form a single window
        return np.empty((0, input_length, F), dtype=float), np.empty((0, output_length), dtype=float)

    X_list, y_list = [], []
    target_idx = F - 1
    for start in range(0, T - input_length - output_length + 1, stride):
        x_window = data[start : start + input_length, :]                                  # (L, F)
        y_window = data[start + input_length : start + input_length + output_length, target_idx]  # (H,)
        X_list.append(x_window)
        y_list.append(y_window)

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    return X, y


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


def prepare_data_for_darts(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float, # This is the validation size as a fraction of the *total* dataset
    scaler_type: str = 'robust'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, Any, pd.Series, pd.Series]:
    """
    Performs a clean, continuous chronological split suitable for Darts.

    This function:
    1. Splits the data into train, validation, and test sets without gaps.
    2. Keeps unscaled copies of the target variable for evaluation.
    3. Fits separate scalers for the target and features *only* on the training data.
    4. Transforms all three data splits.
    5. Returns all necessary artifacts in a single, well-defined tuple.

    Args:
        df: The input DataFrame.
        target_col: The name of the target column.
        test_size: Fraction of data for the test set (e.g., 0.5).
        val_size: Fraction of data for the validation set (e.g., 0.1).
        scaler_type: 'robust' or 'standard'.

    Returns:
        A tuple containing:
        (df_train_scaled, df_val_scaled, df_test_scaled,
         scaler_X, scaler_target,
         y_train_unscaled, y_test_unscaled)
    """
    # 1. Chronological Split (No Gaps)
    n = len(df)
    if not (0 < test_size < 1 and 0 < val_size < 1 and (test_size + val_size) < 1):
        raise ValueError("Invalid test/val sizes. Must be > 0 and sum < 1.")
        
    test_start_index = int(n * (1 - test_size))
    val_start_index = int(n * (1 - test_size - val_size))

    df_train = df.iloc[:val_start_index]
    df_val = df.iloc[val_start_index:test_start_index]
    df_test = df.iloc[test_start_index:]

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError("Data split resulted in one or more empty DataFrames.")

    # 2. Keep Unscaled Targets for Final Evaluation
    # These will be passed through to fulfill the pipeline's contract.
    y_train_unscaled = df_train[target_col].copy()
    y_test_unscaled = df_test[target_col].copy()

    # 3. Fit Separate Scalers for Target and Features (Covariates)
    feature_cols = [c for c in df.columns if c != target_col]
    
    if scaler_type == 'robust':
        scaler_X = RobustScaler()
        scaler_target = RobustScaler()
    else:
        scaler_X = StandardScaler()
        scaler_target = StandardScaler()

       
        
    # Fit scalers ONLY on the training data
    scaler_target.fit(df_train[[target_col]])
    if feature_cols:
        scaler_X.fit(df_train[feature_cols])

    # 4. Transform All Data Splits
    # Create copies to hold the scaled data
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()
    
    # Apply the transformations
    df_train_scaled[target_col] = scaler_target.transform(df_train[[target_col]])
    df_val_scaled[target_col] = scaler_target.transform(df_val[[target_col]])
    df_test_scaled[target_col] = scaler_target.transform(df_test[[target_col]])
    
    if feature_cols:
        df_train_scaled[feature_cols] = scaler_X.transform(df_train[feature_cols])
        df_val_scaled[feature_cols] = scaler_X.transform(df_val[feature_cols])
        df_test_scaled[feature_cols] = scaler_X.transform(df_test[feature_cols])
    
    return (
        df_train_scaled,
        df_val_scaled,
        df_test_scaled,
        scaler_X,
        scaler_target,
        y_train_unscaled,
        y_test_unscaled
    )

def _scale_feature_and_target_sets(
    X_train_feats, X_val_feats, X_test_feats,
    y_train, y_val, y_test,
    scaler_type: Literal['robust', 'standard'] = 'robust'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    """Fits scalers on training data and transforms all sets."""
    if scaler_type == 'robust':
        scaler_X = RobustScaler()
        scaler_target = RobustScaler()
    else: # Default to standard
        scaler_X = StandardScaler()
        scaler_target = StandardScaler()
        
    # Fit only on training data
    n_train, win_len, n_feat = X_train_feats.shape
    scaler_X.fit(X_train_feats.reshape(-1, n_feat))
    scaler_target.fit(y_train.reshape(-1, 1))

    # Helper to scale 3D feature arrays
    def scale_3d_array(arr_3d, scaler):
        n_s, win_l, n_f = arr_3d.shape
        return scaler.transform(arr_3d.reshape(-1, n_f)).reshape(n_s, win_l, n_f)

    # Scale all feature sets
    X_train_scaled = scale_3d_array(X_train_feats, scaler_X)
    X_val_scaled = scale_3d_array(X_val_feats, scaler_X)
    X_test_scaled = scale_3d_array(X_test_feats, scaler_X)

    # Scale all target sets
    y_train_scaled = scaler_target.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val_scaled = scaler_target.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test_scaled = scaler_target.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_target



# ==============================================================================
# 2. REFACTORED PUBLIC FUNCTIONS 
# ==============================================================================

def check_seq2seq_data_leakage(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
    *,
    input_length: int,
    output_length: int,
    backend: str,
    scaler_target=None,
    round_decimals: int = 8,
    atol_boundary: float = 1e-8,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Comprehensive and backend-aware leakage check for seq2seq windowed data.
    """
    log = logger or get_logger("job.data_prep")

    def _box(title: str, lines: list[str]) -> None:
        width = 102
        top    = "┌" + "─" * (width - 2) + "┐"
        midsep = "├" + "─" * (width - 2) + "┤"
        bot    = "└" + "─" * (width - 2) + "┘"
        def pad(s): return "│ " + s.ljust(width - 4) + " │"
        log.info(top); log.info(pad(title)); log.info(midsep)
        for s in lines: log.info(pad(s))
        log.info(bot)

    def _hash_rows(a: np.ndarray) -> set[str]:
        if a.ndim not in [2, 3]:
            raise ValueError("Hashing requires 2D or 3D array.")
        arr_flat = a.reshape(a.shape[0], -1)
        arr_rounded = np.round(arr_flat, round_decimals)
        return {hashlib.sha1(row.tobytes()).hexdigest() for row in arr_rounded}

    def _count_intersections(A: np.ndarray, B: np.ndarray) -> int:
        if A.size == 0 or B.size == 0: return 0
        return len(_hash_rows(A) & _hash_rows(B))

    H = int(output_length)
    backend = str(backend).lower()

    _box("DATA LEAKAGE CHECK – INPUT SUMMARY", [
        f"Backend: {backend.upper()}",
        f"L={input_length}, H={H}",
        f"X_train={X_train.shape}, y_train={y_train.shape}",
        f"X_val  ={X_val.shape},   y_val  ={y_val.shape}",
        f"X_test ={X_test.shape},  y_test ={y_test.shape}",
    ])

    # 1) Exact duplicate target windows
    dup_y_tr_va = _count_intersections(y_train, y_val)
    dup_y_tr_te = _count_intersections(y_train, y_test)
    dup_y_va_te = _count_intersections(y_val, y_test)
    any_exact_dup = any([dup_y_tr_va, dup_y_tr_te, dup_y_va_te])

    _box("CHECK 1: EXACT DUPLICATE TARGET WINDOWS (y)", [
        "Identical target windows across splits should be zero.",
        f"Train vs Val:  {dup_y_tr_va}",
        f"Train vs Test: {dup_y_tr_te}",
        f"Val   vs Test: {dup_y_va_te}",
        f"VERDICT: {'FAIL' if any_exact_dup else 'PASS'}",
    ])

    # 2) Boundary overlap (backend-aware)
    def reconstruct_series(yw: np.ndarray) -> np.ndarray:
        if yw.size == 0: return np.array([])
        n, h = yw.shape
        series = np.full(n + h - 1, np.nan)
        series[:h] = yw[0, :]
        for i in range(1, n):
            series[i + h - 1] = yw[i, -1]
        return series

    ytr_phys = scaler_target.inverse_transform(y_train) if scaler_target is not None else y_train
    yva_phys = scaler_target.inverse_transform(y_val)   if scaler_target is not None else y_val

    tr_series = reconstruct_series(ytr_phys)
    va_series = reconstruct_series(yva_phys)

    overlap_len = max(0, H - 1)
    boundary_overlap = False
    if overlap_len > 0 and len(tr_series) >= overlap_len and len(va_series) >= overlap_len:
        diff = np.max(np.abs(tr_series[-overlap_len:] - va_series[:overlap_len]))
        boundary_overlap = bool(diff <= atol_boundary)

    if backend == "seq2":
        verdict_overlap = "FAIL" if boundary_overlap else "PASS"
        explain = "Seq2 requires strict separation at split boundary."
    else:
        verdict_overlap = "PASS (EXPECTED)"
        explain = "Darts uses train tail as context; this is expected and safe."

    _box("CHECK 2: BOUNDARY OVERLAP", [
        f"Overlap Detected: {'YES' if boundary_overlap else 'NO'}",
        f"Backend: {backend.upper()} — {explain}",
        f"VERDICT: {verdict_overlap}",
    ])

    # Final verdict (Seq2 must have no overlap; Darts can)
    final_ok = (not any_exact_dup) and (backend != "seq2" or not boundary_overlap)
    final = "PASS (No data leakage detected)" if final_ok else "FAIL (Potential data leakage detected)"
    _box("FINAL VERDICT", [final])

    return {"final_verdict": final, "details": {}}



# Let's restart with a cleaner, more robust approach.
# The core idea is to pass the unambiguous lengths directly.

def report_split_sample_counts(
    train_len: int,
    val_len: int,
    test_len: int,
    *,
    input_length: int,
    output_length: int,
    backend: str,
    logger: Optional[logging.Logger] = None,
    title: str = "UNIFIED SPLIT SUMMARY"
) -> None:
    """
    Logs a universal, backend-aware summary of chronological data splits.

    This function focuses on the unambiguous length of the time series partitions,
    which is identical for both Darts and Seq2 backends after the unified split logic.
    For Seq2, it also reports the resulting number of training windows for transparency.

    Args:
        train_len (int): Length of the training time series partition.
        val_len (int): Length of the validation time series partition.
        test_len (int): Length of the test time series partition.
        input_length (int): The 'L' hyperparameter.
        output_length (int): The 'H' (horizon) hyperparameter.
        backend (str): The name of the backend ('Seq2' or 'Darts') to tailor the report.
        logger: Optional logger instance.
        title: The title for the summary box.
    """
    log = logger or get_logger("job.data_prep")
    backend_name = str(backend).upper()
    total_len = train_len + val_len + test_len

    def pct(x: int, denom: int) -> float:
        return (100.0 * x / denom) if denom > 0 else 0.0

    lines = [
        f"Backend: {backend_name}",
        f"Parameters: input_length(L)={input_length}, output_length(H)={output_length}",
        "",
        "--- Time Series Partition Lengths ---",
        f"Train series length:      {train_len} ({pct(train_len, total_len):.2f}%)",
        f"Validation series length: {val_len} ({pct(val_len, total_len):.2f}%)",
        f"Test series length:       {test_len} ({pct(test_len, total_len):.2f}%)",
        f"Total series length:      {total_len}",
    ]

    # The number of training samples is now the same for both backends,
    # as it's derived from the identical train_len.
    num_train_samples = max(0, train_len - input_length - output_length + 1)
    
    lines.extend([
        "",
        "--- Training Samples Generated ---",
        f"Number of training windows (X, y): {num_train_samples}",
    ])

    # The number of evaluation tasks (forecast origins) is also equivalent.
    num_val_forecasts = val_len
    num_test_forecasts = test_len
    lines.extend([
        "",
        "--- Evaluation Tasks (Forecast Origins) ---",
        f"Validation forecast origins: {num_val_forecasts}",
        f"Test forecast origins:       {num_test_forecasts}",
    ])

    # --- Formatting the box (copied from your original function) ---
    width = 102
    top    = "┌" + "─" * (width - 2) + "┐"
    midsep = "├" + "─" * (width - 2) + "┤"
    bot    = "└" + "─" * (width - 2) + "┘"

    def pad(s: str) -> str:
        return "│ " + s.ljust(width - 4) + " │"

    log.info(top)
    log.info(pad(title))
    log.info(midsep)
    for s in lines:
        log.info(pad(s))
    log.info(bot)


def audit_seq2seq_continuity_from_lengths(
    *,
    total_windows: int,
    B_train: int,
    B_val: int,
    B_test: int,
    input_length: int,
    output_length: int,
    gap_used: int,  # use H-1 para seu split atual
    y_train_rec: np.ndarray | None = None,
    y_val_rec:   np.ndarray | None = None,
    y_test_rec:  np.ndarray | None = None,
    logger: logging.Logger | None = None,
):
    """
    Audita continuidade temporal e 'perda' de janelas em split seq2seq baseado em comprimentos.
    Assume o padrão de fatia:
      train: [0 : B_tr)
      val:   [B_tr + gap : B_tr + gap + B_val)
      test:  [B_tr + gap + B_val + gap : ... + B_test)

    Onde cada janela i cobre alvos [i + input_length, i + input_length + H - 1].
    """
    log = logger or logging.getLogger("audit-coverage")
    H = int(output_length)
    gap = int(gap_used)

    def box(title, lines):
        width = max(len(title), *(len(s) for s in lines)) + 4
        top = "┌" + "─"*(width-2) + "┐"
        mid = "├" + "─"*(width-2) + "┤"
        bot = "└" + "─"*(width-2) + "┘"
        log.info(top)
        log.info(f"│ {title.ljust(width-4)} │")
        log.info(mid)
        for s in lines:
            log.info(f"│ {s.ljust(width-4)} │")
        log.info(bot)

    # ---------------- window index ranges (reconstruídos dos comprimentos) ----------------
    tr_a, tr_b = 0, B_train
    va_a, va_b = B_train + gap, B_train + gap + B_val
    te_a, te_b = B_train + gap + B_val + gap, B_train + B_val + B_test + 2*gap

    dropped = total_windows - (B_train + B_val + B_test)
    gaps_idx_tr_val = max(0, va_a - tr_b)
    gaps_idx_val_te = max(0, te_a - va_b)
    overlaps_idx = max(0, tr_b - (B_train)) + max(0, va_b - (B_train + gap + B_val))  # sempre 0 aqui

    box("WINDOW RANGES (BY INDEX)", [
        f"total_windows={total_windows} | gap_used={gap} | H={H}",
        f"train: [{tr_a}:{tr_b}) -> {B_train}",
        f"val  : [{va_a}:{va_b}) -> {B_val}",
        f"test : [{te_a}:{te_b}) -> {B_test}",
        f"dropped windows (redundant): {dropped} ({dropped/total_windows:.2%})",
        f"gaps between buckets (index): train→val={gaps_idx_tr_val}, val→test={gaps_idx_val_te}",
        f"overlap between buckets (index): {overlaps_idx}",
    ])

    # ---------------- target-time coverage ----------------
    def target_span(i):  # janela i cobre:
        return (i + input_length, i + input_length + H - 1)

    def block_span(a, b):
        if b <= a:
            return None
        start, _ = target_span(a)
        _, end   = target_span(b - 1)
        return (start, end)

    tr_span = block_span(tr_a, tr_b)
    va_span = block_span(va_a, va_b)
    te_span = block_span(te_a, te_b)

    def span_str(s):
        return "n/a" if s is None else f"[{s[0]}, {s[1]}] (len={s[1]-s[0]+1})"

    # deltas entre blocos (por timestamps de alvo)
    def gap_between(a, b):
        if a is None or b is None: return None
        return b[0] - a[1] - 1  # 0 => contíguo

    gap_t_tr_val = gap_between(tr_span, va_span)
    gap_t_val_te = gap_between(va_span, te_span)

    box("TARGET-TIME COVERAGE (START/END PER BLOCK)", [
        f"train targets: {span_str(tr_span)}",
        f"val   targets: {span_str(va_span)}",
        f"test  targets: {span_str(te_span)}",
        f"target-time gap (steps): train→val={gap_t_tr_val}, val→test={gap_t_val_te}",
        f"expected contiguous if gap_used == H-1 → {gap == (H-1)}",
    ])

    # ---------------- expected plot length (se reconstruir e 'aparar' H-1) ----------------
    # Se você concatena: len = (B_tr + H - 1) + B_val + B_test
    expected_concat_len = (B_train + H - 1) + B_val + B_test
    lines = [f"Expected concatenated length (trimmed): {expected_concat_len} = (B_tr+H-1)+B_val+B_test"]
    if y_train_rec is not None and y_val_rec is not None and y_test_rec is not None:
        # trimmed: usar val/test sem o prefixo de (H-1)
        val_trim = y_val_rec[H-1:] if len(y_val_rec) >= H else np.array([], dtype=float)
        test_trim = y_test_rec[H-1:] if len(y_test_rec) >= H else np.array([], dtype=float)
        actual_len = len(y_train_rec) + len(val_trim) + len(test_trim)
        lines.append(f"Actual concatenated length (provided arrays): {actual_len}")
        diff = actual_len - expected_concat_len
        lines.append(f"Difference (actual - expected): {diff}")
    box("PLOT LENGTH CHECK", lines)

    # ---------------- verdict ----------------
    contiguous = (gap_t_tr_val == 0 or gap_t_tr_val is None) and (gap_t_val_te == 0 or gap_t_val_te is None)
    verdict = "PASS (continuous and leakage-safe)" if contiguous else "WARN (not contiguous in target-time)"
    box("VERDICT", [
        f"Contiguous by target timestamps: {'YES' if contiguous else 'NO'}",
        f"Dropped windows are REDUNDANT (do not reduce coverage): {dropped} windows",
        f"=> {verdict}"
    ])

    return {
        "window_ranges": {"train": (tr_a,tr_b), "val": (va_a,va_b), "test": (te_a,te_b)},
        "target_spans":  {"train": tr_span, "val": va_span, "test": te_span},
        "target_gaps":   {"train_val": gap_t_tr_val, "val_test": gap_t_val_te},
        "expected_plot_len": expected_concat_len,
        "dropped_windows": dropped,
        "contiguous": contiguous,
    }



def inspect_set_boundaries(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    target_col_index: int = -1 # Assume target is the last feature in X
):
    """

    Prints a detailed, elegant report of the data values at the boundaries
    between the train, validation, and test sets to help visualize the
    sliding window structure and any gaps.
    """
    if not RICH_AVAILABLE:
        print("--- Data Boundary Inspection ---")
        # Add simple print statements here as a fallback
        return

    console = Console()
    console.print(Panel("[bold cyan]🕵️ Data Boundary Inspection 🕵️[/]", expand=False))

    # --- Train -> Validation Boundary ---
    train_val_table = Table(title="[bold]Train -> Validation Boundary[/]", box=None, show_header=True, header_style="magenta")
    train_val_table.add_column("Data Point", style="dim")
    train_val_table.add_column("X (Input Window)", justify="right")
    train_val_table.add_column("y (Target Window)", justify="right")

    # Last sample of the training set
    last_X_train_target = X_train[-1, :, target_col_index]
    last_y_train = y_train[-1, :]
    
    # First sample of the validation set
    first_X_val_target = X_val[0, :, target_col_index]
    first_y_val = y_val[0, :]
    
    train_val_table.add_row(
        "Last Train Sample", 
        f"[green]{np.round(last_X_train_target[-3:], 2)}[/]", 
        f"[bold green]{np.round(last_y_train[:3], 2)}[/]"
    )
    train_val_table.add_row(
        "First Validation Sample", 
        f"[yellow]{np.round(first_X_val_target[-3:], 2)}[/]", 
        f"[bold yellow]{np.round(first_y_val[:3], 2)}[/]"
    )
    console.print(train_val_table)
    print(f"Observation: The first input of the validation set (`X_val`) should chronologically follow the last input of the training set (`X_train`). The target (`y_train`) is the future of the input (`X_train`).")

    # --- Validation -> Test Boundary ---
    val_test_table = Table(title="[bold]Validation -> Test Boundary[/]", box=None, show_header=True, header_style="magenta")
    val_test_table.add_column("Data Point", style="dim")
    val_test_table.add_column("X (Input Window)", justify="right")
    val_test_table.add_column("y (Target Window)", justify="right")

    # Last sample of the validation set
    last_X_val_target = X_val[-1, :, target_col_index]
    last_y_val = y_val[-1, :]
    
    # First sample of the test set
    first_X_test_target = X_test[0, :, target_col_index]
    first_y_test = y_test[0, :]
    
    val_test_table.add_row(
        "Last Validation Sample", 
        f"[yellow]{np.round(last_X_val_target[-3:], 2)}[/]", 
        f"[bold yellow]{np.round(last_y_val[:3], 2)}[/]"
    )
    val_test_table.add_row(
        "First Test Sample", 
        f"[red]{np.round(first_X_test_target[-3:], 2)}[/]", 
        f"[bold red]{np.round(first_y_test[:3], 2)}[/]"
    )
    console.print(val_test_table)
    print(f"Observation: An explicit 'gap' of {y_train.shape[1] - 1} timesteps exists between the end of the validation set and the start of the test set to prevent any lookahead bias.")



@dataclass(frozen=True)
class SplitBoundaries:
    """Chronological split boundaries (absolute indices over the raw dataframe)."""
    train_end_idx: int
    val_start_idx: int
    val_end_idx: int
    test_start_idx: int

def calculate_split_boundaries(
    total_len: int,
    test_size: float,
    val_size: float,
) -> SplitBoundaries:
    """
    Compute chronological split indices with simple, reproducible rounding.
    Layout (0-based, end-exclusive semantics for slicing):
        [0 : train_end) | [val_start : val_end) | [test_start : total_len)

    Notes:
    - train_end == val_start
    - val_end    == test_start
    - Guarantees monotonicity and clamps to valid ranges.
    """
    if total_len <= 0:
        raise ValueError("total_len must be positive")
    if not (0.0 <= test_size < 1.0) or not (0.0 <= val_size < 1.0):
        raise ValueError("test_size and val_size must be in [0,1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    # Use floor for stable, deterministic boundaries
    n_test = int(np.floor(total_len * test_size))
    n_val  = int(np.floor(total_len * val_size))
    n_train = total_len - n_val - n_test
    n_train = max(0, n_train)  # clamp

    train_end = n_train
    val_start = train_end
    val_end   = val_start + n_val
    test_start = val_end

    # Monotonic clamps
    train_end  = min(max(train_end, 0), total_len)
    val_start  = min(max(val_start, 0), total_len)
    val_end    = min(max(val_end, val_start), total_len)
    test_start = min(max(test_start, val_end), total_len)

    return SplitBoundaries(
        train_end_idx=train_end,
        val_start_idx=val_start,
        val_end_idx=val_end,
        test_start_idx=test_start,
    )

def analyze_scaler_effects(
    *,
    method_name: str,
    scaler_target: Any,
    scaler_X: Any,
    df_train_raw: pd.DataFrame,
    y_train_windows_unscaled: np.ndarray,
    y_train_scaled: np.ndarray,
    target_col: str,
    feature_cols: list
):
    """
    Performs and logs a detailed statistical analysis of scaler effects.

    This function provides a quantitative baseline to compare different
    normalization strategies by inspecting:
    1. The parameters (mean, scale) learned by the fitted scalers.
    2. The "pure" statistical properties of the original 1D training series.
    3. The "weighted" statistical properties of the overlapping training windows.
    4. The resulting distribution of the scaled training data.

    Args:
        method_name (str): A descriptive name for the method being analyzed
                           (e.g., "Original (Window-Fit)").
        scaler_target (object): The fitted scaler for the target variable.
        scaler_X (object): The fitted scaler for the feature variables.
        df_train_raw (pd.DataFrame): The unscaled, continuous training DataFrame.
        y_train_windows_unscaled (np.ndarray): The unscaled training target windows (N, H).
        y_train_scaled (np.ndarray): The scaled training target windows.
        target_col (str): The name of the target column.
        feature_cols (list): A list of feature column names.
    """
    
    def _get_params(scaler):
        """Safely extracts mean and scale from StandardScaler or RobustScaler."""
        if hasattr(scaler, 'mean_'):  # StandardScaler
            return scaler.mean_[0], scaler.scale_[0]
        elif hasattr(scaler, 'center_'):  # RobustScaler
            return scaler.center_[0], scaler.scale_[0]
        return np.nan, np.nan

    print("\n" + "="*70)
    print(f"         SCALER EFFECTS ANALYSIS for: {method_name}")
    print("="*70 + "\n")

    # --- 1. Target Scaler Analysis ---
    print(f"--- Target Scaler Analysis (target_col: '{target_col}') ---")
    
    # Get parameters from the scaler that was passed in
    scaler_mean, scaler_scale = _get_params(scaler_target)
    
    # Calculate reference statistics
    pure_mean = df_train_raw[target_col].mean()
    pure_std = df_train_raw[target_col].std()
    
    weighted_mean = y_train_windows_unscaled.mean()
    weighted_std = y_train_windows_unscaled.std()
    
    # Check which reference statistic matches the scaler's parameters
    is_pure_match = np.allclose(scaler_mean, pure_mean) and np.allclose(scaler_scale, pure_std)
    is_weighted_match = np.allclose(scaler_mean, weighted_mean) and np.allclose(scaler_scale, weighted_std)

    print("[Scaler Parameters]")
    print(f"  - Mean.......: {scaler_mean:<15.4f}")
    print(f"  - Scale (Std): {scaler_scale:<15.4f}\n")

    print("[Reference Statistics]")
    print(f"  - 1D Series (Pure):    Mean={pure_mean:<10.4f} | Std={pure_std:<10.4f} {'<-- MATCH!' if is_pure_match else ''}")
    print(f"  - Windowed (Weighted): Mean={weighted_mean:<10.4f} | Std={weighted_std:<10.4f} {'<-- MATCH!' if is_weighted_match else ''}\n")

    # Analyze the resulting scaled data
    scaled_mean = y_train_scaled.mean()
    scaled_std = y_train_scaled.std()
    scaled_min = y_train_scaled.min()
    scaled_max = y_train_scaled.max()

    print("[Resulting Scaled Data Distribution (y_train_scaled)]")
    print(f"  - Mean.......: {scaled_mean:<15.4f}")
    print(f"  - Std........: {scaled_std:<15.4f}")
    print(f"  - Min........: {scaled_min:<15.4f}")
    print(f"  - Max........: {scaled_max:<15.4f}\n")

    # --- 2. Feature Scaler Analysis ---
    if feature_cols:
        print(f"--- Feature Scaler Analysis ---")
        for i, col_name in enumerate(feature_cols):
            feature_pure_mean = df_train_raw[col_name].mean()
            feature_pure_std = df_train_raw[col_name].std()
            
            # Note: For features, we assume the scaler was fit on the 1D series
            # as it's the most common and robust method.
            if hasattr(scaler_X, 'mean_'):
                scaler_feat_mean = scaler_X.mean_[i]
                scaler_feat_scale = scaler_X.scale_[i]
            elif hasattr(scaler_X, 'center_'):
                scaler_feat_mean = scaler_X.center_[i]
                scaler_feat_scale = scaler_X.scale_[i]
            else:
                scaler_feat_mean, scaler_feat_scale = np.nan, np.nan

            is_feat_match = np.allclose(scaler_feat_mean, feature_pure_mean) and np.allclose(scaler_feat_scale, feature_pure_std)
            
            print(f"Feature: '{col_name}'")
            print(f"  - Scaler Params : Mean={scaler_feat_mean:<10.4f} | Scale={scaler_feat_scale:<10.4f}")
            print(f"  - 1D Series Stats: Mean={feature_pure_mean:<10.4f} | Std={feature_pure_std:<10.4f} {'<-- MATCH!' if is_feat_match else ''}")
    
    print("="*70 + "\n")




# Helper function moved outside for better modularity and clarity
def _scale_target_in_x_helper(X: np.ndarray, scaler_target: Any) -> np.ndarray:
    """
    Scales the last feature of an X window (assumed to be the target variable)
    using a pre-fitted target scaler.
    """
    flat = X[:, :, -1:].reshape(-1, 1)
    scaled = scaler_target.transform(flat)
    return scaled.reshape(X.shape[0], X.shape[1], 1)



def _make_left_contexts_for_split_full(
    df_raw: pd.DataFrame,
    *,
    input_length: int,   # L
    output_length: int,  # H
    split_start_idx: int,
    max_left: Optional[int] = None,
    feature_order_like_train: Optional[List[str]] = None,
    target_col: Optional[str] = None,
) -> np.ndarray:
    """
    Build LEFT contexts tight to the split boundary. Each window's last forecast
    aligns at split_start_idx + i, where i = 0..K-1.
    """
    log = get_logger("job.data_prep")
    L, H, T = int(input_length), int(output_length), int(len(df_raw))

    if T == 0:
        log.warning("Cannot build LEFT contexts: input dataframe is empty (T=0).")
        return np.empty((0, L, 0), dtype=float)

    # Determine column order, ensuring target is last if provided
    cols = feature_order_like_train.copy() if feature_order_like_train else list(df_raw.columns)
    if target_col and target_col in cols:
        cols = [c for c in cols if c != target_col] + [target_col]
    dfw = df_raw[cols]
    F = dfw.shape[1]

    K_target = H if (max_left is None) else max(0, int(max_left))
    if K_target == 0:
        log.info("Skipping LEFT build: max_left=0 was requested.")
        return np.empty((0, L, F), dtype=float)

    # Calculate window start indices
    # For forecast[-1] at split_start_idx + i, window start s = split_start_idx + i - (L + H - 1)
    s0 = split_start_idx - (L + H - 1)
    s_min, s_max = 0, T - L
    i_lo = max(0, s_min - s0)            # Ensure s >= 0
    i_hi = min(K_target - 1, s_max - s0) # Ensure s <= T - L

    if i_hi < i_lo:
        log.warning("No LEFT contexts fit for split at index %d.", split_start_idx)
        return np.empty((0, L, F), dtype=float)

    # Build the LEFT windows
    num_left_windows = i_hi - i_lo + 1
    vals = dfw.values
    X_left = np.empty((num_left_windows, L, F), dtype=float)
    for k, i in enumerate(range(i_lo, i_lo + num_left_windows)):
        s = s0 + i
        X_left[k] = vals[s : s + L, :]

    # Log summary in a box
    box_log(
        log,
        f"LEFT Context Build Summary (Split @ {split_start_idx})",
        [
            f"Input Series (T, F):   ({T}, {F})",
            f"Window Params (L, H):  ({L}, {H})",
            f"Target Windows (K):    {K_target}",
            f"Calculated i range:    [{i_lo}, {i_hi}]",
            f"Resulting Shape:       {X_left.shape}",
        ],
    )

    return X_left

# ------------------------------- small helpers -------------------------------

def _split_frames(df: pd.DataFrame, train_end: int, val_start: int, val_end: int, test_start: int
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw partitions as dataframes: (train, val, test)."""
    return df.iloc[:train_end], df.iloc[val_start:val_end], df.iloc[test_start:]


def _build_seq2_windows(
    df_train: pd.DataFrame,
    df_val_ctx: pd.DataFrame,
    df_val_only: pd.DataFrame,
    df_test_ctx: pd.DataFrame,
    df_test_only: pd.DataFrame,
    target_col: str,
    L: int,
    H: int,
):
    """
    Build sliding windows:
      - train strictly inside train
      - val/test derived from their *contexts* but cropped to the split segment
    """
    X_train, y_train = create_sliding_window_seq_to_seq(df_train, target_col, L, H)

    # Validation windows from context cropped to val-only count
    X_val_all, y_val_all = create_sliding_window_seq_to_seq(df_val_ctx, target_col, L, H)
    n_val = max(0, len(df_val_only) - H + 1)
    X_val, y_val = X_val_all[-n_val:], y_val_all[-n_val:]

    # Test windows from context cropped to test-only count
    X_test_all, y_test_all = create_sliding_window_seq_to_seq(df_test_ctx, target_col, L, H)
    n_test = max(0, len(df_test_only) - H + 1)
    X_test, y_test = X_test_all[-n_test:], y_test_all[-n_test:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _scale_feature_target_sets(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    scaler_type: str = "robust"
):
    """
    Fit scalers on train only; scale feature channels and target arrays.
    Recompose X_* to include the scaled target channel at the last position.
    """
    # Split off features (all but last) for scaling
    X_feats = [X[:, :, :-1] for X in (X_train, X_val, X_test)]
    (X_train_feats_scaled, X_val_feats_scaled, X_test_feats_scaled,
     y_train_scaled,       y_val_scaled,       y_test_scaled,
     scaler_X, scaler_target) = _scale_feature_and_target_sets(
        *X_feats, y_train, y_val, y_test, scaler_type=scaler_type
    )

    # Scale target inside X using the same target scaler and recompose
    X_train_t = _scale_target_in_x_helper(X_train, scaler_target)
    X_val_t   = _scale_target_in_x_helper(X_val,   scaler_target)
    X_test_t  = _scale_target_in_x_helper(X_test,  scaler_target)

    X_train_scaled = np.concatenate([X_train_feats_scaled, X_train_t], axis=-1)
    X_val_scaled   = np.concatenate([X_val_feats_scaled,   X_val_t],   axis=-1)
    X_test_scaled  = np.concatenate([X_test_feats_scaled,  X_test_t],  axis=-1)

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            scaler_X, scaler_target)


def _scale_left_with_train_scalers(
    X_left_raw: np.ndarray,
    scaler_X,
    scaler_target,
    input_length: int,
    n_total_feats: int,
) -> np.ndarray:
    """Scale LEFT contexts using *train-fitted* scalers, preserving last channel as target."""
    if X_left_raw.size == 0:
        return np.empty((0, input_length, n_total_feats), dtype=float)
    n_feats = n_total_feats - 1  # exclude target channel
    feats = X_left_raw[:, :, :-1].reshape(-1, n_feats)
    feats_s = scaler_X.transform(feats).reshape(X_left_raw.shape[0], X_left_raw.shape[1], n_feats)
    t_s = _scale_target_in_x_helper(X_left_raw, scaler_target)  # (N, L, 1)
    return np.concatenate([feats_s, t_s], axis=-1)


def _true_left_prefix_scaled_1d(
    df_full: pd.DataFrame,
    target_col: str,
    split_start_idx: int,
    H: int,
    scaler_target
) -> np.ndarray:
    """
    Return up to H-1 target points immediately before the split, scaled (1D).
    Empty array if not applicable.
    """
    if split_start_idx <= 0 or H <= 1:
        return np.empty((0,), dtype=float)
    left = max(0, split_start_idx - (H - 1))
    raw = df_full.iloc[left:split_start_idx][target_col].to_numpy().reshape(-1, 1)
    if raw.size == 0:
        return np.empty((0,), dtype=float)
    return scaler_target.transform(raw).reshape(-1)  # 1D


def _attach_left_sidecar(
    scaler_target,
    *,
    X_val_left_scaled: np.ndarray,
    X_test_left_scaled: np.ndarray,
    y_val_left_true_scaled_1d: np.ndarray,
    y_test_left_true_scaled_1d: np.ndarray,
    H: int,
    train_end: int,
    val_end: int,
    test_start: int,
):
    """Attach LEFT contexts + TRUE prefixes to scaler_target._split_ctx (backward compatible keys included)."""
    sidecar = {
        "H": int(H),
        "X_val_left_scaled":  X_val_left_scaled,
        "X_test_left_scaled": X_test_left_scaled,
        "val_left_count":  int(X_val_left_scaled.shape[0]),
        "test_left_count": int(X_test_left_scaled.shape[0]),
        # TRUE prefixes (scaled) + alias keys for older consumers
        "y_val_left_true_scaled_1d":  y_val_left_true_scaled_1d,
        "y_test_left_true_scaled_1d": y_test_left_true_scaled_1d,
        "y_val_left_true_scaled":  y_val_left_true_scaled_1d,   # alias
        "y_test_left_true_scaled": y_test_left_true_scaled_1d,  # alias
        "val_left_true_len":  int(y_val_left_true_scaled_1d.size),
        "test_left_true_len": int(y_test_left_true_scaled_1d.size),
        # boundaries
        "boundaries": {
            "train_end": int(train_end),
            "val_end":   int(val_end),
            "test_start": int(test_start),
        },
        "note": "Split-aware left-of-split windows + TRUE prefixes for warm-start; all scaled with train-only scalers.",
    }
    setattr(scaler_target, "_split_ctx", sidecar)


import os
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional, Tuple, List, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------
# Palette helper (uses your style / keys)
# ---------------------------------------------------------------------
def _get_color_palette(name: str) -> Dict[str, str]:
    palettes = {
        "default": {
            "train_from_x": "#3949AB",
            "train_from_y": "#0277BD",
            "actual_post_train": "#0277BD",
            "validation": "#E53935",
            "test_initial": "#4CAF50",
            "test_rolling": "#4CAF50",
            "fill_train": "rgba(232, 234, 246, 0.5)",
            "fill_val": "rgba(255, 235, 238, 0.5)",
            "fill_test": "rgba(232, 245, 233, 0.5)",
            "text": "#333",
            "grid": "#EAEAEA",
            "test_border": "#4CAF50",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
        },
        "metallic_azure": {
            "train_from_x": "#546E7A",
            "train_from_y": "#90A4AE",
            "actual_post_train": "#1E88E5",
            "validation": "#00B8D4",
            "test_initial": "#263238",
            "test_rolling": "#4FC3F7",
            "fill_train": "rgba(84, 110, 122, 0.12)",
            "fill_val": "rgba(0, 184, 212, 0.10)",
            "fill_test": "rgba(79, 195, 247, 0.12)",
            "text": "#222",
            "grid": "#E5E7EB",
            "test_border": "#4FC3F7",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
        },
        "abyssal_expedition_light": {
            "train_from_x": "#0D47A1",
            "train_from_y": "#1976D2",
            "actual_post_train": "#00796B",
            "validation": "#00B8D4",
            "test_initial": "#AA00FF",
            "test_rolling": "#F9A825",
            "fill_train": "rgba(25,118,210,0.10)",
            "fill_val": "rgba(0,184,212,0.10)",
            "fill_test": "rgba(249,168,37,0.12)",
            "text": "#222",
            "grid": "#EAEAEA",
            "test_border": "#F9A825",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
        },
    }

    if name not in palettes:
        raise ValueError(f"Unknown palette '{name}'. Available options are: {list(palettes.keys())}")

    pal = palettes[name]
    pal.setdefault("plot_bgcolor", "white")
    pal.setdefault("paper_bgcolor", "white")
    return pal


# ---------------------------------------------------------------------
# Core: causal smoother + mass correction
# ---------------------------------------------------------------------
def _causal_ema_mass_correct_1d(
    y: np.ndarray,
    *,
    alpha: float,
    k_integral: float,
    init_s: Optional[float] = None,
    init_e: float = 0.0,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Causal EMA smoothing with integral (mass) correction.
    - Smoother (causal): s_t = alpha*y_t + (1-alpha)*s_{t-1}
    - Mass correction:   out_t = s_t + k * E_{t-1}
    - Error integral:    E_t = E_{t-1} + (y_t - out_t)

    Returns:
      out: filtered series
      s_last: last EMA state
      e_last: last integral error state
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    out = np.empty(n, dtype=float)

    a = float(np.clip(alpha, 1e-6, 1.0))
    k = float(np.clip(k_integral, 0.0, 1.0))

    # Initialize state
    s = float(init_s) if (init_s is not None and np.isfinite(init_s)) else float(y[0]) if n > 0 else 0.0
    e = float(init_e) if np.isfinite(init_e) else 0.0

    for i in range(n):
        yi = float(y[i])
        if not np.isfinite(yi):
            # If missing, hold last output (conservative, causal)
            out[i] = out[i - 1] if i > 0 else s
            continue

        # Causal EMA
        s = a * yi + (1.0 - a) * s

        # Integral correction (keeps cumulative close)
        oi = s + k * e

        # Optional clamping (use carefully; may change total mass)
        if clamp_min is not None:
            oi = max(float(clamp_min), oi)
        if clamp_max is not None:
            oi = min(float(clamp_max), oi)

        out[i] = oi

        # Update integral error
        e = e + (yi - oi)

    return out, float(s), float(e)


def _apply_causal_mass_filter_df(
    df_in: pd.DataFrame,
    *,
    columns: Sequence[str],
    alpha: float,
    k_integral: float,
    states_in: Optional[Dict[str, Dict[str, float]]] = None,
    clamp_nonnegative: bool = False,
    nonnegative_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Applies causal EMA+mass correction column-wise, optionally warm-starting from `states_in`.

    states format:
      states[col] = {"s": last_s, "e": last_e}
    """
    df = df_in.copy()
    states_out: Dict[str, Dict[str, float]] = {}

    nonneg_set = set(nonnegative_columns or [])
    for col in columns:
        if col not in df.columns:
            continue

        y = df[col].astype(float).to_numpy()
        st = (states_in or {}).get(col, {})
        init_s = st.get("s", None)
        init_e = st.get("e", 0.0)

        clamp_min = 0.0 if (clamp_nonnegative and (col in nonneg_set or len(nonneg_set) == 0)) else None

        y_f, s_last, e_last = _causal_ema_mass_correct_1d(
            y,
            alpha=alpha,
            k_integral=k_integral,
            init_s=init_s,
            init_e=init_e,
            clamp_min=clamp_min,
        )
        df[col] = y_f
        states_out[col] = {"s": s_last, "e": e_last}

    return df, states_out


# ---------------------------------------------------------------------
# Storytelling plot (target channel only)
# ---------------------------------------------------------------------
def _plot_story_causal_mass_smoothing(
    *,
    df_raw: pd.DataFrame,
    df_filt: pd.DataFrame,
    target_col: str,
    split_indices: Dict[str, int],
    palette: str,
    title: str,
    subtitle: str,
    width: int,
    height: int,
    show: bool,
    save_path: Optional[str],
) -> go.Figure:
    colors = _get_color_palette(palette)
    FONT_FAMILY = "Inter, Arial, sans-serif"

    y_raw = df_raw[target_col].astype(float).to_numpy()
    y_f = df_filt[target_col].astype(float).to_numpy()

    x = np.arange(len(y_raw), dtype=int)
    train_end = int(split_indices["train_end"])
    val_end = int(split_indices["val_end"])
    test_start = int(split_indices["test_start"])

    # Cumsums (handle NaNs as 0 for mass-balance visualization clarity)
    y_raw0 = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_f0 = np.nan_to_num(y_f, nan=0.0, posinf=0.0, neginf=0.0)
    c_raw = np.cumsum(y_raw0)
    c_f = np.cumsum(y_f0)
    c_diff = c_f - c_raw

    # Drift metrics
    def _pct(a: float, b: float) -> float:
        den = abs(b) + 1e-12
        return 100.0 * (a - b) / den

    drift_total_pct = _pct(c_f[-1], c_raw[-1]) if len(c_raw) else 0.0
    drift_total_abs = float(c_diff[-1]) if len(c_diff) else 0.0
    max_abs_cdiff = float(np.nanmax(np.abs(c_diff))) if len(c_diff) else 0.0

    # Per-split drift (end of each region)
    def _drift_at(idx: int) -> Tuple[float, float]:
        idx = int(np.clip(idx, 0, len(c_raw) - 1))
        return float(c_diff[idx]), _pct(c_f[idx], c_raw[idx])

    drift_train_abs, drift_train_pct = _drift_at(train_end - 1)
    drift_val_abs, drift_val_pct = _drift_at(val_end - 1)

    # Figure with 3 rows (story)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.42, 0.34, 0.24],
        subplot_titles=(
            "1) Instantaneous signal (Raw vs Filtered)",
            "2) Cumulative sum (Raw vs Filtered)",
            "3) Cumulative difference (Filtered - Raw)",
        ),
    )

    # Background regions (Train / Val / Test)
    def _add_regions(yref: str = "paper"):
        fig.add_vrect(x0=0, x1=train_end - 1, fillcolor=colors["fill_train"], layer="below", line_width=0)
        fig.add_vrect(x0=train_end - 1, x1=val_end - 1, fillcolor=colors["fill_val"], layer="below", line_width=0)
        fig.add_vrect(x0=val_end - 1, x1=len(x) - 1, fillcolor=colors["fill_test"], layer="below", line_width=0)

        fig.add_annotation(
            text="<b>Train</b>", x=int(train_end * 0.5), yref=yref, y=0.99, showarrow=False,
            font=dict(size=16, color=colors["text"], family=FONT_FAMILY),
        )
        fig.add_annotation(
            text="<b>Validation</b>", x=int(train_end + (val_end - train_end) * 0.5), yref=yref, y=0.99,
            showarrow=False, font=dict(size=16, color=colors["validation"], family=FONT_FAMILY),
        )
        fig.add_annotation(
            text="<b>Test</b>", x=int(val_end + (len(x) - val_end) * 0.5), yref=yref, y=0.99,
            showarrow=False, font=dict(size=16, color=colors["test_border"], family=FONT_FAMILY),
        )

    _add_regions()

    # Row 1: Raw vs filtered
    fig.add_trace(
        go.Scatter(
            x=x, y=y_raw, mode="lines", name="Raw",
            line=dict(color="rgba(120,120,120,0.55)", width=1.5),
            hovertemplate="t=%{x}<br>raw=%{y:.4g}<extra></extra>",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y_f, mode="lines", name="Filtered (causal + mass correction)",
            line=dict(color=colors["actual_post_train"], width=2.8),
            hovertemplate="t=%{x}<br>filtered=%{y:.4g}<extra></extra>",
        ),
        row=1, col=1
    )

    # Row 2: Cumulative
    fig.add_trace(
        go.Scatter(
            x=x, y=c_raw, mode="lines", name="Cumulative Raw",
            line=dict(color="rgba(30,30,30,0.75)", width=2, dash="dash"),
            hovertemplate="t=%{x}<br>cum_raw=%{y:.6g}<extra></extra>",
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=c_f, mode="lines", name="Cumulative Filtered",
            line=dict(color=colors["test_rolling"], width=2.2),
            hovertemplate="t=%{x}<br>cum_filt=%{y:.6g}<extra></extra>",
        ),
        row=2, col=1
    )

    # Row 3: Cumulative difference
    fig.add_trace(
        go.Scatter(
            x=x, y=c_diff, mode="lines", name="CumDiff (Filtered - Raw)",
            line=dict(color=colors["validation"], width=2.0),
            hovertemplate="t=%{x}<br>cum_diff=%{y:.6g}<extra></extra>",
        ),
        row=3, col=1
    )
    fig.add_hline(y=0.0, line_width=1, line_color=colors["grid"], row=3, col=1)

    # Storytelling annotation box (requirements)
    story_lines = [
        "✅ causal (ML-safe for forecasting)",
        "✅ cumulative “sticks” by construction (drift controlled)",
        "",
        f"<b>Total drift</b>: {drift_total_abs:.6g}  ({drift_total_pct:.4f}%)",
        f"<b>Train-end drift</b>: {drift_train_abs:.6g}  ({drift_train_pct:.4f}%)",
        f"<b>Val-end drift</b>: {drift_val_abs:.6g}  ({drift_val_pct:.4f}%)",
        f"<b>Max |cum diff|</b>: {max_abs_cdiff:.6g}",
    ]
    fig.add_annotation(
        text="<br>".join(story_lines),
        align="left",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.01, y=0.02,
        xanchor="left", yanchor="bottom",
        bordercolor="#ddd",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.92)",
        font=dict(size=13, color=colors["text"], family="Courier New, monospace"),
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:14px'>{subtitle}</span>",
            x=0.5, y=0.95, xanchor="center", yanchor="top"
        ),
        font=dict(family=FONT_FAMILY, color=colors["text"], size=14),
        plot_bgcolor=colors["plot_bgcolor"],
        paper_bgcolor=colors["paper_bgcolor"],
        width=width,
        height=height,
        margin=dict(t=120, b=70, l=80, r=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
    )

    # Axes styling
    for r in (1, 2, 3):
        fig.update_xaxes(gridcolor=colors["grid"], zeroline=False, row=r, col=1)
        fig.update_yaxes(gridcolor=colors["grid"], zeroline=False, row=r, col=1)

    fig.update_xaxes(title_text="Time index", row=3, col=1)
    fig.update_yaxes(title_text=target_col, row=1, col=1)
    fig.update_yaxes(title_text=f"Cumulative sum({target_col})", row=2, col=1)
    fig.update_yaxes(title_text="CumDiff", row=3, col=1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path, scale=2)  # requires kaleido installed
    if show:
        fig.show(config={"toImageButtonOptions": {"format": "png", "filename": "qc_causal_mass_smoothing", "scale": 3}})

    return fig


# ---------------------------------------------------------------------
# Single entry function (plug-and-play)
# ---------------------------------------------------------------------
def maybe_filter_splits_causal_mass_preserving(
    *,
    df_full: pd.DataFrame,
    train_end: int,
    val_start: int,
    val_end: int,
    test_start: int,
    enabled: bool = True,
    filter_all_features: bool = True,
    feature_order: Optional[List[str]] = None,
    # Filter params
    ema_alpha: float = 0.18,
    k_integral: float = 0.06,
    clamp_nonnegative: bool = False,
    nonnegative_columns: Optional[Sequence[str]] = None,
    # Plot/QC
    target_col: Optional[str] = None,
    target_index: int = -1,  # used only if target_col is None
    palette: str = "abyssal_expedition_light",
    plot: bool = True,
    show: bool = True,
    width: int = 1100,
    height: int = 920,
    output_dir: Optional[str] = None,
    file_prefix: str = "QC_CausalMass",
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Plug-and-play entry point:

    - Causal smoothing + mass correction applied AFTER split boundaries exist.
    - Filters either all features or only selected ones.
    - Produces a self-contained storytelling QC plot for the target channel only.
    - Returns:
        df_train_f, df_val_f, df_test_f, df_full_f, meta

    IMPORTANT:
      If you want downstream contexts (df_val_ctx / df_test_ctx) to use filtered data,
      use df_full_f slices instead of the original df_full slices.
    """
    log = logger or logging.getLogger(__name__)

    meta: Dict[str, Any] = {
        "enabled": bool(enabled),
        "method": "causal_ema_mass_correction",
        "ema_alpha": float(ema_alpha),
        "k_integral": float(k_integral),
        "filter_all_features": bool(filter_all_features),
        "clamp_nonnegative": bool(clamp_nonnegative),
        "target_col": None,
        "plot_saved_to": None,
    }

    # If disabled, passthrough
    if not enabled:
        df_train = df_full.iloc[:train_end].copy()
        df_val = df_full.iloc[val_start:val_end].copy()
        df_test = df_full.iloc[test_start:].copy()
        meta["target_col"] = target_col
        log.info("[CausalMass] Skipped (disabled).")
        return df_train, df_val, df_test, df_full.copy(), meta

    # Determine columns to filter
    cols = list(feature_order) if feature_order else list(df_full.columns)
    if not filter_all_features:
        # default: only filter target (if provided), else last column by index
        if target_col is not None:
            cols = [target_col]
        else:
            cols = [df_full.columns[int(target_index)]]

    # Resolve target column for QC plot
    if target_col is None:
        target_col = df_full.columns[int(target_index)]
    meta["target_col"] = str(target_col)

    # Split raw frames (for return + QC comparisons)
    df_train_raw = df_full.iloc[:train_end].copy()
    df_val_raw = df_full.iloc[val_start:val_end].copy()
    df_test_raw = df_full.iloc[test_start:].copy()

    # Filtering strategy (safe + elegant):
    # Filter the FULL dataframe causally in chronological order (no look-ahead),
    # so boundaries are naturally warm-started.
    log.info(
        "[CausalMass] Applying causal EMA + mass correction on %d columns (alpha=%.4f, k=%.4f).",
        len(cols), float(ema_alpha), float(k_integral)
    )

    df_full_f = df_full.copy()
    # Apply column-wise with shared chronological pass; since each column is independent,
    # we can do it column by column but over full time, which still is causal.
    df_full_f, _ = _apply_causal_mass_filter_df(
        df_full_f,
        columns=cols,
        alpha=float(ema_alpha),
        k_integral=float(k_integral),
        states_in=None,
        clamp_nonnegative=bool(clamp_nonnegative),
        nonnegative_columns=nonnegative_columns,
    )

    # Slice filtered splits
    df_train_f = df_full_f.iloc[:train_end].copy()
    df_val_f = df_full_f.iloc[val_start:val_end].copy()
    df_test_f = df_full_f.iloc[test_start:].copy()

    # QC plot (target only)
    if plot and (target_col in df_full.columns):
        split_idx = {"train_end": int(train_end), "val_end": int(val_end), "test_start": int(test_start)}
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"{file_prefix}_{target_col}.png")
            meta["plot_saved_to"] = save_path

        subtitle = (
            f"Method: causal EMA + integral (mass) correction | "
            f"alpha={float(ema_alpha):.3f}, k={float(k_integral):.3f} | "
            f"Filtered columns: {'ALL' if filter_all_features else 'TARGET'}"
        )
        # _plot_story_causal_mass_smoothing(
        #     df_raw=df_full,
        #     df_filt=df_full_f,
        #     target_col=str(target_col),
        #     split_indices=split_idx,
        #     palette=palette,
        #     title=f"Causal, Mass-Preserving Smoothing QC — {target_col}",
        #     subtitle=subtitle,
        #     width=int(width),
        #     height=int(height),
        #     show=bool(show),
        #     save_path=save_path,
        # )

    return df_train_f, df_val_f, df_test_f, df_full_f, meta



# ---------------------------------- main API ---------------------------------

def prepare_data_seq_volve(
    df: pd.DataFrame,
    target_col: str,
    input_length: int,
    output_length: int,
    test_size: float = 0.5,
    val_size: float = 0.1,
    data_aug_params: Optional[Dict[str, Any]] = None
):
    """
    Split-first + sidecar 'left-of-split'.
    Keeps original return contract; injects LEFT windows (scaled) *and* TRUE warm prefixes (scaled)
    into `scaler_target._split_ctx` for validation/test splits.

    Sidecar adds (backward-compatible):
      - "X_val_left_scaled", "X_test_left_scaled", "val_left_count", "test_left_count"
      - "y_val_left_true_scaled_1d", "y_test_left_true_scaled_1d" (and aliases without _1d)
      - "val_left_true_len", "test_left_true_len"
      - "boundaries": {train_end, val_end, test_start}
    """
    # Light-touch logging via project utilities (kept minimal).
    log = get_logger("job.data_prep")

    # 1) Split boundaries
    with phase(log, "Seq2: compute split boundaries", L=input_length, H=output_length, test_size=test_size, val_size=val_size):
        boundaries = calculate_split_boundaries(total_len=len(df), test_size=test_size, val_size=val_size)
        train_end = boundaries.train_end_idx
        val_start, val_end = boundaries.val_start_idx, boundaries.val_end_idx
        test_start = boundaries.test_start_idx
        log.info(
            "Split boundaries -> train_end=%d | val=[%d,%d) | test_start=%d | total_len=%d",
            train_end, val_start, val_end, test_start, len(df)
        )

    # 2) Raw partitions
    df_train, df_val, df_test = _split_frames(df, train_end, val_start, val_end, test_start)

    # df_train, df_val, df_test, df_filt, smooth_meta = maybe_filter_splits_causal_mass_preserving(
    #     df_full=df,
    #     train_end=train_end,
    #     val_start=val_start,
    #     val_end=val_end,
    #     test_start=test_start,
    #     enabled=True,
    #     filter_all_features=False,          # as you requested
    #     ema_alpha=0.18,
    #     k_integral=0.06,
    #     clamp_nonnegative=False,           # you can set True if desired
    #     target_col=target_col,             # BORE_OIL_VOL
    #     palette="abyssal_expedition_light",
    #     plot=True,
    #     show=True,
    #     output_dir=None,         # if you have it
    #     logger=log,
    # )


    # 3) Sliding windows
    with phase(log, "Seq2: build sliding windows"):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = _build_seq2_windows(
            df_train=df_train,
            df_val_ctx=df.iloc[:val_end],
            df_val_only=df_val,
            df_test_ctx=df.iloc[train_end:],
            df_test_only=df_test,
            target_col=target_col,
            L=input_length,
            H=output_length,
        )

        # (X_train, y_train), (X_val, y_val), (X_test, y_test) = _build_seq2_windows(
        #     df_train=df_train,
        #     df_val_ctx=df_filt.iloc[:val_end],
        #     df_val_only=df_val,
        #     df_test_ctx=df_filt.iloc[train_end:],
        #     df_test_only=df_test,
        #     target_col=target_col,
        #     L=input_length,
        #     H=output_length,
        # )


        box_log(
            log, "Sliding Window Shapes",
            [
                f"X_train: {getattr(X_train, 'shape', None)} | y_train: {getattr(y_train, 'shape', None)}",
                f"X_val:   {getattr(X_val, 'shape', None)} | y_val:   {getattr(y_val, 'shape', None)}",
                f"X_test:  {getattr(X_test, 'shape', None)} | y_test:  {getattr(y_test, 'shape', None)}",
            ],
        )

    

    # 4) LEFT raw (tight to split boundaries)
    feature_order_like_train = list(df.columns)
    with phase(log, "Seq2: build LEFT (val/test)"):
        X_val_left_raw = _make_left_contexts_for_split_full(
            df_raw=df,
            input_length=input_length,
            output_length=output_length,
            split_start_idx=val_start,
            max_left=output_length,
            feature_order_like_train=feature_order_like_train,
            target_col=target_col,
        )
        X_test_left_raw = _make_left_contexts_for_split_full(
            df_raw=df,
            input_length=input_length,
            output_length=output_length,
            split_start_idx=test_start,
            max_left=output_length,
            feature_order_like_train=feature_order_like_train,
            target_col=target_col,
        )
        log.info("[LeftBuild] Raw LEFT shapes -> val=%s | test=%s",
                 tuple(getattr(X_val_left_raw, "shape", ())), tuple(getattr(X_test_left_raw, "shape", ())))

    # 5) Split report (compact)
    with phase(log, "Seq2: split report"):
        report_split_sample_counts(
            train_len=len(df_train),
            val_len=len(df_val),
            test_len=len(df_test),
            input_length=input_length,
            output_length=output_length,
            backend='Seq2',
            logger=log
        )

    # 6) Data augmentation (train only, optional)
    y_train_original = y_train.copy()
    if data_aug_params and data_aug_params.get("data_sample", 1.0) < 1.0:
        with phase(log, "Seq2: data augmentation (train)"):
            frac = float(data_aug_params["data_sample"])
            log_da_usage(log, used=True, reason=f"downsample to {frac:.2f}")
            X_train, y_train = augment_with_synthetic_samples(X_train, y_train, data_sample=frac)
            log.info("After DA -> X_train=%s y_train=%s", tuple(X_train.shape), tuple(y_train.shape))
    else:
        log_da_usage(log, used=False)

    # 7) Scaling (fit on train only)
    with phase(log, "Seq2: scale features/targets"):
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train_scaled,  y_val_scaled,  y_test_scaled,
         scaler_X, scaler_target) = _scale_feature_target_sets(
            X_train, X_val, X_test, y_train, y_val, y_test, scaler_type="robust"
        )

    box_log(
        log, "Scaled Tensor Shapes",
        [
            f"X_train_scaled: {getattr(X_train_scaled, 'shape', None)} | y_train_scaled: {getattr(y_train_scaled, 'shape', None)}",
            f"X_val_scaled:   {getattr(X_val_scaled,   'shape', None)} | y_val_scaled:   {getattr(y_val_scaled,   'shape', None)}",
            f"X_test_scaled:  {getattr(X_test_scaled,  'shape', None)} | y_test_scaled:  {getattr(y_test_scaled,  'shape', None)}",
        ],
    )

    # 8) Scale LEFT using the train scalers + build TRUE prefixes (scaled)
    with phase(log, "Seq2: scale LEFT with train scalers"):
        n_total_feats = X_train_scaled.shape[-1]
        X_val_left_scaled  = _scale_left_with_train_scalers(X_val_left_raw,  scaler_X, scaler_target, input_length, n_total_feats)
        X_test_left_scaled = _scale_left_with_train_scalers(X_test_left_raw, scaler_X, scaler_target, input_length, n_total_feats)
        log.info("[LeftBuild] Scaled LEFT -> val=%s | test=%s",
                 tuple(X_val_left_scaled.shape), tuple(X_test_left_scaled.shape))

        # TRUE prefixes (scaled, 1D)
        with phase(log, "Seq2: build TRUE LEFT prefixes (scaled)"):
            y_val_left_true_scaled_1d  = _true_left_prefix_scaled_1d(df, target_col, val_start,  output_length, scaler_target)
            y_test_left_true_scaled_1d = _true_left_prefix_scaled_1d(df, target_col, test_start, output_length, scaler_target)
            log.info("[TrueLeft] prefix lengths -> val=%d/%d test=%d/%d",
                     y_val_left_true_scaled_1d.size, max(0, output_length - 1),
                     y_test_left_true_scaled_1d.size, max(0, output_length - 1))

    # 9) Leakage checks
    with phase(log, "Seq2: leakage checks"):
        _ = check_seq2seq_data_leakage(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            input_length=input_length,
            output_length=output_length,
            backend='seq2',
            scaler_target=scaler_target,
            logger=log,
        )

    # 10) Sidecar attach (no contract change)
    with phase(log, "Seq2: attach LEFT sidecar"):
        _attach_left_sidecar(
            scaler_target,
            X_val_left_scaled=X_val_left_scaled,
            X_test_left_scaled=X_test_left_scaled,
            y_val_left_true_scaled_1d=y_val_left_true_scaled_1d,
            y_test_left_true_scaled_1d=y_test_left_true_scaled_1d,
            H=output_length,
            train_end=train_end,
            val_end=val_end,
            test_start=test_start,
        )
        log.info(
            "Attached LEFT sidecar (val_left=%d, test_left=%d, true_prefixes: val=%d, test=%d).",
            int(X_val_left_scaled.shape[0]),
            int(X_test_left_scaled.shape[0]),
            int(y_val_left_true_scaled_1d.size),
            int(y_test_left_true_scaled_1d.size),
        )

    # 11) Return — keep original contract
    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_scaled, y_val_scaled, y_test_scaled,
        scaler_X, scaler_target, y_train_original,
    )



# --- Implementation for UNISIM_IV ---
# This function now internally uses the same logic as the VOLVE version.
def prepare_data_seq_unisim_iv(
    df: pd.DataFrame,
    target_col: str,
    input_length: int,
    output_length: int,
    test_size: float = 0.5, # Default updated to 0.5
    val_size: float = 0.1,
    data_aug_params: Optional[Dict[str, Any]] = None
) -> Tuple:
    """
    Wrapper for UNISIM_IV that now calls the standard chronological workflow
    to ensure consistency with the VOLVE pipeline.
    """
    # This function now becomes a simple alias to the main, robust implementation.
    # We are unifying the logic to be consistent.
    logging.info("INFO: Using standardized chronological split for UNISIM_IV data preparation.")
    return prepare_data_seq_volve(
        df, target_col, input_length, output_length,
        test_size, val_size, data_aug_params=data_aug_params
    )

if DEFAULT_DATASET == "VOLVE":
    prepare_data_seq = prepare_data_seq_volve
elif DEFAULT_DATASET == "UNISIM_IV":
    prepare_data_seq = prepare_data_seq_unisim_iv
else:
    # Fallback to the standard method
    prepare_data_seq = prepare_data_seq_volve


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


def _second_diff_std(x: np.ndarray) -> float:
    """Desvio-padrão da 2ª diferença (curvatura discreta). Segura para séries curtas."""
    x = np.asarray(x, float)
    if x.size < 5:
        return 0.0
    d2 = np.diff(x, n=2)
    return float(np.std(d2)) if d2.size > 0 else 0.0

def _lambda_adapt_from_curv(
    hist_smooth_tail: np.ndarray,
    base_curv: float,
    hp_lambda: float,
    lam_low: float,
    lam_high: float,
    curv_win: int,
    gamma: float = 1.0,
) -> float:
    """
    Converte curvatura local → λ efetivo, respeitando [lam_low, lam_high].
    c_ref = base_curv. Razão r = clamp(c/base_curv, 1/8..8).
    λ_eff = clamp(hp_lambda * r**(-gamma), lam_low..lam_high).
    """
    tail = hist_smooth_tail[-int(curv_win):] if curv_win > 0 else hist_smooth_tail
    c = _second_diff_std(tail)
    if base_curv <= 1e-12:
        return float(np.clip(hp_lambda, lam_low, lam_high))
    r = np.clip(c / base_curv, 1/8.0, 8.0)
    lam_eff = hp_lambda * (r ** (-gamma))
    lam_eff = float(np.clip(lam_eff, lam_low, lam_high))
    return lam_eff

def _hampel_clip_point(
    last_raw: float,
    hist_smooth_tail: np.ndarray,
    clip_win: int = 15,
    clip_k: float = 3.0,
) -> float:
    """
    Hampel/median de 1 ponto: recorta o 'last_raw' usando mediana e MAD do histórico SUAVIZADO
    em janela curta (não cria atraso, só protege contra outlier).
    """
    tail = hist_smooth_tail[-int(clip_win):] if clip_win > 0 else hist_smooth_tail
    if tail.size < 5:
        return float(last_raw)
    med = np.median(tail)
    mad = np.median(np.abs(tail - med))
    if mad <= 1e-12:
        return float(last_raw)
    sigma = 1.4826 * mad
    lo, hi = med - clip_k * sigma, med + clip_k * sigma
    return float(np.clip(last_raw, lo, hi))


def _hp_composite_hist_base(
    predictions: np.ndarray, hp_lambda: float,
    *,  # NOVOS (opcionais)
    lambda_adapt: bool = False,
    lam_low: float = None, lam_high: float = None, curv_win: int = 30, gamma: float = 1.0,
    robust_clip: bool = False, clip_win: int = 15, clip_k: float = 3.0,
) -> np.ndarray:
    if sm is None:
        raise ImportError("statsmodels é necessário para HP filter.")
    N, H = predictions.shape
    L = N + H - 1

    # 1) base: filtra a primeira fita
    _, smoothed_history = sm.tsa.filters.hpfilter(predictions[0, :], lamb=hp_lambda)
    out = np.empty(L, dtype=float)
    out[:H] = smoothed_history

    # baseline de curvatura p/ λ adaptativo
    base_curv = _second_diff_std(smoothed_history)
    lam_low  = float(lam_low  if lam_low  is not None else hp_lambda / 5.0)
    lam_high = float(lam_high if lam_high is not None else hp_lambda * 5.0)

    # 2) passos seguintes
    for i in range(1, N):
        t = H + i - 1
        next_raw = float(predictions[i, -1])

        # robustez (opcional) usando histórico já suavizado
        if robust_clip:
            next_raw = _hampel_clip_point(next_raw, out[:t], clip_win=clip_win, clip_k=clip_k)

        # λ adaptativo (opcional)
        lam_eff = ( _lambda_adapt_from_curv(out[:t], base_curv, hp_lambda, lam_low, lam_high, curv_win, gamma)
                    if lambda_adapt else hp_lambda )

        composite = np.append(out[:t], next_raw)     # histórico SUAVIZADO + ponto bruto (clipado)
        _, smoothed = sm.tsa.filters.hpfilter(composite, lamb=lam_eff)
        out[t] = smoothed[-1]

    return out

def _hp_composite_raw_base(
    predictions: np.ndarray, hp_lambda: float,
    *,  # NOVOS (opcionais)
    lambda_adapt: bool = False,
    lam_low: float = None, lam_high: float = None, curv_win: int = 30, gamma: float = 1.0,
    robust_clip: bool = False, clip_win: int = 15, clip_k: float = 3.0,
) -> np.ndarray:
    if sm is None:
        raise ImportError("statsmodels é necessário para HP filter.")
    N, H = predictions.shape
    L = N + H - 1

    raw = np.empty(L, dtype=float)
    out = np.empty(L, dtype=float)

    # 1) bootstrap bruto com a primeira fita inteira
    raw[:H] = predictions[0, :]
    _, sm_first = sm.tsa.filters.hpfilter(raw[:H], lamb=hp_lambda)
    out[:H] = sm_first

    # baseline de curvatura p/ λ adaptativo (na base suavizada)
    base_curv = _second_diff_std(sm_first)
    lam_low  = float(lam_low  if lam_low  is not None else hp_lambda / 5.0)
    lam_high = float(lam_high if lam_high is not None else hp_lambda * 5.0)

    # 2) passos seguintes
    for i in range(1, N):
        t = H + i - 1
        last_raw = float(predictions[i, -1])

        # robustez (opcional) baseada no histórico SUAVIZADO (out[:t]) para não atrasar
        if robust_clip:
            last_raw = _hampel_clip_point(last_raw, out[:t], clip_win=clip_win, clip_k=clip_k)

        raw[t] = last_raw

        # λ adaptativo (opcional) usando a curvatura do histórico suavizado atual
        lam_eff = ( _lambda_adapt_from_curv(out[:t], base_curv, hp_lambda, lam_low, lam_high, curv_win, gamma)
                    if lambda_adapt else hp_lambda )

        _, sm_hist = sm.tsa.filters.hpfilter(raw[:t+1], lamb=lam_eff)
        out[t] = sm_hist[-1]

    return out



# ---------------------------------------------------------------------
# NOVO: warm deslizante que remove a “fita” na fronteira do split
# ---------------------------------------------------------------------
def _hp_hist_warm_deslizante(
    predictions: np.ndarray,
    hp_lambda: float,
    *,
    warm_prefix_true_1d: np.ndarray = None,   # 1D true prefix (physical units)
    warm_left_windows_2d: np.ndarray = None,  # (K,H) predicted LEFT windows (physical units)
    lambda_adapt: bool = False,
    lam_low: float = None, lam_high: float = None, curv_win: int = 30, gamma: float = 1.0,
    robust_clip: bool = False, clip_win: int = 15, clip_k: float = 3.0,
) -> np.ndarray:
    """
    HP composite (history base) with sliding warm.
    Warm source priority: TRUE prefix (1D) > predicted LEFT windows (2D).
    """
    if sm is None:
        raise ImportError("statsmodels is required for HP filter.")
    import numpy as np

    N, H = predictions.shape
    L = N + H - 1
    out = np.empty(L, dtype=float)

    tail_len = max(0, H - 1)

    # choose warm tail
    if warm_prefix_true_1d is not None and warm_prefix_true_1d.size > 0:
        src = warm_prefix_true_1d[-tail_len:]
        warm_src = "truth"
    elif warm_left_windows_2d is not None and warm_left_windows_2d.size > 0:
        # reuse last elements of LEFT windows
        k = min(tail_len, warm_left_windows_2d.shape[0])
        src = warm_left_windows_2d[-k:, -1] if k > 0 else np.empty((0,), dtype=float)
        warm_src = "pred"
    else:
        src = np.empty((0,), dtype=float)
        warm_src = "none"

    logging.info("[AggWarm] hp_hist_warm: source=%s H=%d warm_tail_len=%d", warm_src, H, src.size)

    # publish warm tail as smoothed trend
    if src.size > 0:
        _, smoothed = sm.tsa.filters.hpfilter(src, lamb=hp_lambda)
        out[:src.size] = smoothed

    # baseline curvature for adaptive lambda
    base_curv = _second_diff_std(out[:max(src.size, 10)]) if src.size > 0 else 0.0
    lam_low  = float(lam_low  if lam_low  is not None else hp_lambda / 5.0)
    lam_high = float(lam_high if lam_high is not None else hp_lambda * 5.0)

    # assimilate one point per window (last element), refiltering composite
    for i in range(N):
        t = src.size + i
        next_raw = float(predictions[i, -1])

        if robust_clip:
            next_raw = _hampel_clip_point(next_raw, out[:t], clip_win=clip_win, clip_k=clip_k)

        lam_eff = (_lambda_adapt_from_curv(out[:t], base_curv, hp_lambda, lam_low, lam_high, curv_win, gamma)
                   if lambda_adapt else hp_lambda)

        composite = np.append(out[:t], next_raw)
        _, smoothed = sm.tsa.filters.hpfilter(composite, lamb=lam_eff)
        out[t] = smoothed[-1]

    return out




def _hp_raw_warm_deslizante(
    predictions: np.ndarray,
    hp_lambda: float,
    *,
    warm_prefix_true_1d: np.ndarray = None,   # 1D true prefix (physical units)
    warm_left_windows_2d: np.ndarray = None,  # (K,H) predicted LEFT windows (physical units)
    lambda_adapt: bool = False,
    lam_low: float = None, lam_high: float = None, curv_win: int = 30, gamma: float = 1.0,
    robust_clip: bool = False, clip_win: int = 15, clip_k: float = 3.0,
) -> np.ndarray:
    """
    HP composite (raw-history base) with sliding warm.
    Warm source priority: TRUE prefix (1D) > predicted LEFT windows (2D).
    """
    if sm is None:
        raise ImportError("statsmodels is required for HP filter.")
    import numpy as np

    N, H = predictions.shape
    L = N + H - 1
    raw = np.empty(L, dtype=float)
    out = np.empty(L, dtype=float)

    tail_len = max(0, H - 1)

    # choose warm tail
    if warm_prefix_true_1d is not None and warm_prefix_true_1d.size > 0:
        src = warm_prefix_true_1d[-tail_len:]
        warm_src = "truth"
    elif warm_left_windows_2d is not None and warm_left_windows_2d.size > 0:
        k = min(tail_len, warm_left_windows_2d.shape[0])
        src = warm_left_windows_2d[-k:, -1] if k > 0 else np.empty((0,), dtype=float)
        warm_src = "pred"
    else:
        src = np.empty((0,), dtype=float)
        warm_src = "none"

    logging.info("[AggWarm] hp_raw_warm: source=%s H=%d warm_tail_len=%d", warm_src, H, src.size)

    # bootstrap: raw & smoothed from the warm tail
    if src.size > 0:
        raw[:src.size] = src
        _, sm_first = sm.tsa.filters.hpfilter(raw[:src.size], lamb=hp_lambda)
        out[:src.size] = sm_first
    else:
        # nothing to warm with — consistent with the downgrade in the caller
        pass

    base_curv = _second_diff_std(out[:max(src.size, 10)]) if src.size > 0 else 0.0
    lam_low  = float(lam_low  if lam_low  is not None else hp_lambda / 5.0)
    lam_high = float(lam_high if lam_high is not None else hp_lambda * 5.0)

    for i in range(N):
        t = src.size + i
        last_raw = float(predictions[i, -1])

        if robust_clip:
            last_raw = _hampel_clip_point(last_raw, out[:t], clip_win=clip_win, clip_k=clip_k)

        raw[t] = last_raw
        lam_eff = (_lambda_adapt_from_curv(out[:t], base_curv, hp_lambda, lam_low, lam_high, curv_win, gamma)
                   if lambda_adapt else hp_lambda)

        _, sm_hist = sm.tsa.filters.hpfilter(raw[:t+1], lamb=lam_eff)
        out[t] = sm_hist[-1]

    return out


# -------------------------------------------------
# Reconstrução clássica (já existente)
# -------------------------------------------------
def reconstruct_true_series(windows_2d: np.ndarray) -> np.ndarray:
    if windows_2d.ndim != 2:
        raise ValueError("reconstruct_true_series espera shape (N, H).")
    N, H = windows_2d.shape
    L = N + H - 1
    out = np.empty(L, dtype=float)
    out[:H] = windows_2d[0, :]
    for i in range(1, N):
        t = H + i - 1
        out[t] = windows_2d[i, -1]
    return out




def _causal_ewma_step(prev: float, x: float, alpha: float) -> float:
    """Single-step causal EWMA with fallback for NaN prev."""
    if np.isnan(prev):
        return float(x)
    return float(alpha * x + (1.0 - alpha) * prev)


def _causal_holt_step(level: float, trend: float, x: float, alpha: float, beta: float):
    """
    One-step additive Holt's linear trend (causal).
    Returns (new_level, new_trend, y_out).
    """
    new_level = alpha * x + (1 - alpha) * (level + trend)
    new_trend = beta * (new_level - level) + (1 - beta) * trend
    y_out = new_level  # publish level as filtered value
    return float(new_level), float(new_trend), float(y_out)


def _hp_trend_last(series: np.ndarray, hp_lambda: float) -> float:
    """Return last trend value of HP filter applied to 1D series."""
    if sm is None:
        raise ImportError("statsmodels is required for HP filter.")
    cycle, trend = sm.tsa.filters.hpfilter(np.asarray(series, float), lamb=float(hp_lambda))
    return float(trend[-1])


# -------------------------------------------------
# Caminho novo: reconstruct_warm com filtro causal
# -------------------------------------------------
def _reconstruct_warm_laststep(
    predictions: np.ndarray,                  # (N, H) for current split
    warm_left_windows_2d: Optional[np.ndarray],  # (K, H) left-side windows (optional if TRUE prefix provided)
    *,
    warm_true_prefix_1d: Optional[np.ndarray] = None,  # <<< NEW (optional) TRUE prefix, prioritized
    filter_kind: str = "hp",                  # "hp" (default) | "none" | "raw" | "ewma" | "holt"
    hp_lambda: float = 16000.0,
    ewma_alpha: float = 0.25,
    holt_alpha: float = 0.25,
    holt_beta: float = 0.10,
) -> np.ndarray:
    """
    Engineer's approach (warm-first, then stepwise split):
      - Publish the first H-1 points from a warm sequence:
          * Prefer TRUE prefix tail (if provided), else use the last offsets of LEFT windows.
          * Optionally apply a causal/low-pass filter on this warm sequence.
      - For each day in the split, take only the last offset of predictions[i] and
        append it after filtering causally over the published history.

    Backward compatible: if `warm_true_prefix_1d` is not provided, behavior is identical
    to the previous version (uses LEFT last offsets).
    """
    if predictions.ndim != 2:
        raise ValueError("_reconstruct_warm_laststep: `predictions` must be (N, H).")
    N, H = predictions.shape
    if N == 0 or H == 0:
        return np.empty((0,), dtype=float)

    # Determine warm tail length = min(H-1, len(available warm))
    Hm1 = max(0, H - 1)

    # Build preferred warm sequence:
    # 1) TRUE prefix tail (preferred)  2) LEFT last offsets  3) empty if none
    warm_seq = np.empty((0,), dtype=float)

    if warm_true_prefix_1d is not None and warm_true_prefix_1d.size > 0:
        # take the last H-1 points from the true prefix (already 1D)
        tail = warm_true_prefix_1d.reshape(-1)[-Hm1:] if Hm1 > 0 else np.empty((0,), dtype=float)
        warm_seq = np.asarray(tail, float)

    elif warm_left_windows_2d is not None:
        if warm_left_windows_2d.ndim != 2:
            raise ValueError("_reconstruct_warm_laststep: `warm_left_windows_2d` must be (K, H).")
        K, H_left = warm_left_windows_2d.shape
        if H_left != H:
            raise ValueError(f"_reconstruct_warm_laststep: LEFT H={H_left} differs from split H={H}.")
        tail_len = min(Hm1, max(0, K))
        warm_seq = warm_left_windows_2d[-tail_len:, -1] if tail_len > 0 else np.empty((0,), dtype=float)

    fk = (filter_kind or "hp").lower()
    # Accept "none" as alias of "raw"
    if fk == "none":
        fk = "raw"

    logging.info(
        "[ReconstructWarm] filter=%s  H=%d  tail_len=%d  hp_lambda=%.1f",
        fk, H, int(warm_seq.size), float(hp_lambda)
    )

    # Output length: L = (H-1 warm) + N last-steps
    L = (warm_seq.size if warm_seq.size > 0 else 0) + N
    out = np.empty(L, dtype=float)

    # 1) Warm publish (first H-1 points)
    if warm_seq.size > 0:
        if fk == "raw":
            out[:warm_seq.size] = warm_seq
        elif fk == "ewma":
            y = np.nan
            for t in range(warm_seq.size):
                y = _causal_ewma_step(y, warm_seq[t], ewma_alpha)
                out[t] = y
        elif fk == "holt":
            level = float(warm_seq[0])
            trend = 0.0
            out[0] = level
            for t in range(1, warm_seq.size):
                level, trend, y = _causal_holt_step(level, trend, warm_seq[t], holt_alpha, holt_beta)
                out[t] = y
        elif fk == "hp":
            if sm is None:
                raise ImportError("statsmodels is required for HP filter.")
            _, trend = sm.tsa.filters.hpfilter(warm_seq, lamb=float(hp_lambda))
            out[:warm_seq.size] = trend
        else:
            logging.warning("[ReconstructWarm] unknown filter_kind='%s'; using 'raw'.", fk)
            out[:warm_seq.size] = warm_seq

    # 2) Split days (append step-by-step the last offset of each prediction row)
    #    Apply the same causal rule on top of the history produced so far.
    for i in range(N):
        t = (warm_seq.size if warm_seq.size > 0 else 0) + i
        x_new = float(predictions[i, -1])

        if fk == "raw":
            out[t] = x_new
        elif fk == "ewma":
            prev = out[t-1] if t > 0 else x_new
            out[t] = _causal_ewma_step(prev, x_new, ewma_alpha)
        elif fk == "holt":
            if t == 0:
                level = x_new
                trend = 0.0
                out[t] = level
            else:
                prev_level = out[t-1]
                prev_trend = (out[t-1] - out[t-2]) if t >= 2 else 0.0
                level, trend, y = _causal_holt_step(prev_level, prev_trend, x_new, holt_alpha, holt_beta)
                out[t] = y
        elif fk == "hp":
            # HP is non-causal, mas mantemos o "jeito engenheiro" de aplicar até o último instante
            out[t] = _hp_trend_last(np.append(out[:t], x_new), hp_lambda=float(hp_lambda)) if t > 0 else x_new
        else:
            out[t] = x_new  # fallback

    return out



_VALID_RW_FILTERS = {"raw", "hp", "ewma", "holt"}

def _parse_policy(policy: str, reconstruct_warm_filter: str) -> Tuple[str, Optional[str]]:
    p = (policy or "reconstruct").strip().lower()
    filt = (reconstruct_warm_filter or "raw").strip().lower()
    if filt == "none":
        filt = "raw"

    if p.startswith("reconstruct_warm_"):
        suffix = p.split("reconstruct_warm_", 1)[1].strip()
        suffix = "raw" if suffix == "none" else suffix
        if suffix not in _VALID_RW_FILTERS:
            raise ValueError(
                f"Invalid reconstruct_warm filter '{suffix}'. "
                f"Expected one of: {sorted(_VALID_RW_FILTERS)}."
            )
        return "reconstruct_warm", suffix

    if p == "reconstruct_warm":
        if filt not in _VALID_RW_FILTERS:
            raise ValueError(
                f"Invalid reconstruct_warm_filter '{filt}'. "
                f"Expected one of: {sorted(_VALID_RW_FILTERS)}."
            )
        return "reconstruct_warm", filt

    return p, None


def _ensure_2d_predictions(pred: np.ndarray) -> Tuple[int, int]:
    if not isinstance(pred, np.ndarray):
        raise TypeError("`predictions` must be a numpy.ndarray.")
    if pred.ndim != 2:
        raise ValueError("`predictions` must have shape (N, H).")
    N, H = pred.shape
    if N == 0 or H == 0:
        raise ValueError("`predictions` cannot be empty.")
    return N, H


def _has_arr(x: Optional[np.ndarray]) -> bool:
    return isinstance(x, np.ndarray) and x.size > 0


def _maybe_downgrade(policy: str, have_left: bool, have_true_prefix: bool) -> str:
    """If a *_warm policy is requested but zero warm context is available, downgrade."""
    warm_available = have_left or have_true_prefix
    if warm_available:
        return policy
    if policy == "hp_hist_warm":
        logging.warning("[Agg] hp_hist_warm requested without LEFT/TRUE; downgrading to hp_hist.")
        return "hp_hist"
    if policy == "hp_raw_warm":
        logging.warning("[Agg] hp_raw_warm requested without LEFT/TRUE; downgrading to hp_raw.")
        return "hp_raw"
    if policy == "reconstruct_warm":
        logging.warning("[Agg] reconstruct_warm requested without LEFT/TRUE; downgrading to reconstruct.")
        return "reconstruct"
    return policy


def _dispatch_no_warm(base_policy: str, predictions: np.ndarray, *, hp_lambda: float) -> np.ndarray:
    if base_policy == "reconstruct":
        return reconstruct_true_series(predictions)
    if base_policy == "hp_hist":
        return _hp_composite_hist_base(predictions, hp_lambda=hp_lambda)
    if base_policy == "hp_raw":
        return _hp_composite_raw_base(predictions, hp_lambda=hp_lambda)
    raise ValueError(
        f"Unknown non-warm policy '{base_policy}'. "
        f"Expected one of: 'reconstruct', 'hp_hist', 'hp_raw'."
    )


def _dispatch_warm_hp(
    base_policy: str,
    predictions: np.ndarray,
    warm_left_windows_2d: Optional[np.ndarray],
    *,
    hp_lambda: float
) -> np.ndarray:
    if base_policy == "hp_hist_warm":
        if not _has_arr(warm_left_windows_2d):
            logging.warning("[Agg] hp_hist_warm without LEFT; fallback to hp_hist.")
            return _hp_composite_hist_base(predictions, hp_lambda=hp_lambda)
        return _hp_hist_warm_deslizante(
            predictions,
            hp_lambda=hp_lambda,
            warm_left_windows_2d=warm_left_windows_2d
        )

    if base_policy == "hp_raw_warm":
        if not _has_arr(warm_left_windows_2d):
            logging.warning("[Agg] hp_raw_warm without LEFT; fallback to hp_raw.")
            return _hp_composite_raw_base(predictions, hp_lambda=hp_lambda)
        return _hp_raw_warm_deslizante(
            predictions,
            hp_lambda=hp_lambda,
            warm_left_windows_2d=warm_left_windows_2d
        )

    raise ValueError("Unknown warm HP policy ...")


def _dispatch_reconstruct_warm(
    predictions: np.ndarray,
    *,
    warm_left_windows_2d: Optional[np.ndarray],
    warm_true_prefix_1d: Optional[np.ndarray],
    filter_kind: str,
    hp_lambda: float,
    ewma_alpha: float,
    holt_alpha: float,
    holt_beta: float,
) -> np.ndarray:
    """
    Prefer TRUE prefix if available; otherwise use LEFT windows.
    Falls back gracefully if backend signature doesn't support TRUE prefix.
    """
    # Prefer TRUE prefix path
    if _has_arr(warm_true_prefix_1d):
        try:
            # Ideal path: backend aceita TRUE prefix explicitamente
            return _reconstruct_warm_laststep(
                predictions,
                warm_left_windows_2d=warm_left_windows_2d,   # pode ser None
                warm_true_prefix_1d=warm_true_prefix_1d,     # <<< priorizado
                filter_kind=filter_kind,
                hp_lambda=hp_lambda,
                ewma_alpha=ewma_alpha,
                holt_alpha=holt_alpha,
                holt_beta=holt_beta,
            )
        except TypeError:
            # Versões antigas não aceitam TRUE prefix — vamos tentar com LEFT se houver
            logging.warning("[Agg] backend does not accept TRUE prefix; falling back to LEFT-only warm.")
            if _has_arr(warm_left_windows_2d):
                return _reconstruct_warm_laststep(
                    predictions,
                    warm_left_windows_2d=warm_left_windows_2d,
                    filter_kind=filter_kind,
                    hp_lambda=hp_lambda,
                    ewma_alpha=ewma_alpha,
                    holt_alpha=holt_alpha,
                    holt_beta=holt_beta,
                )
            # Sem LEFT também? então degrade para reconstruct simples
            logging.warning("[Agg] no LEFT available; fallback to reconstruct.")
            return reconstruct_true_series(predictions)

    # Sem TRUE prefix, tenta com LEFT
    if _has_arr(warm_left_windows_2d):
        return _reconstruct_warm_laststep(
            predictions,
            warm_left_windows_2d=warm_left_windows_2d,
            filter_kind=filter_kind,
            hp_lambda=hp_lambda,
            ewma_alpha=ewma_alpha,
            holt_alpha=holt_alpha,
            holt_beta=holt_beta,
        )

    # Sem nenhum warm: degrade
    logging.warning("[Agg] reconstruct_warm without LEFT/TRUE; fallback to reconstruct.")
    return reconstruct_true_series(predictions)


def aggregate_predictions(
    predictions: np.ndarray,
    *,
    policy: str = "reconstruct",
    hp_lambda: float = 8000.0,
    warm_left_windows_2d: np.ndarray = None,    # for *_warm
    warm_true_prefix_1d: np.ndarray = None,     # TRUE prefix support (preferred when present)
    # knobs for reconstruct_warm (used if policy is "reconstruct_warm" without suffix)
    reconstruct_warm_filter: str = "raw",       # valid: "raw"|"none"|"hp"|"ewma"|"holt"
    ewma_alpha: float = 0.25,
    holt_alpha: float = 0.25,
    holt_beta: float = 0.10,
) -> np.ndarray:
    """
    Unified aggregator with TRUE prefix support and 'auto' policy.

    Supported `policy` values:
      - "reconstruct"
      - "hp_hist", "hp_raw"
      - "hp_hist_warm", "hp_raw_warm"
      - "reconstruct_warm_raw" | "reconstruct_warm_hp" | "reconstruct_warm_ewma" | "reconstruct_warm_holt"
      - legacy: "reconstruct_warm" (uses `reconstruct_warm_filter` arg)
      - NEW: "auto" (tries a sensible ordered list and returns the first that succeeds)

    Notes:
      - "raw" and "none" are synonyms for the causal filter.
      - Any *_warm policy can use either TRUE prefix (preferred) or LEFT windows.
      - Returns a 1D aggregated series (np.ndarray).
    """
    # --- helpers from existing codebase (assumed available) ---
    # _ensure_2d_predictions, _parse_policy, _has_arr, _maybe_downgrade,
    # _dispatch_no_warm, _dispatch_warm_hp, _dispatch_reconstruct_warm

    hp_lambda = 8000
    _ensure_2d_predictions(predictions)

    # normalize inputs
    policy_norm = (policy or "reconstruct").strip().lower()
    if reconstruct_warm_filter is not None:
        rwf_norm = reconstruct_warm_filter.strip().lower()
        if rwf_norm == "none":
            rwf_norm = "raw"  # treat "none" as the causal/no-filter path
        reconstruct_warm_filter = rwf_norm

    # ------------------------------------------------------------------
    # NEW: AUTO — try a prioritized list and return the first that works
    # ------------------------------------------------------------------
    if policy_norm == "auto":
        # Order can be customized; TRUE prefix (if provided) will be preferred inside dispatch
        candidates = [
            "reconstruct_warm_ewma",
            "reconstruct_warm_hp",
            "reconstruct_warm_raw",
            "reconstruct",           # simple, very robust fallback
            "hp_hist_warm",
            "hp_raw_warm",
            "hp_hist",
            "hp_raw",
        ]

        last_error = None
        for cand in candidates:
            try:
                if cand.startswith("reconstruct_warm_"):
                    # Map suffix to filter knob and use the unified warm dispatcher
                    suffix = cand.split("reconstruct_warm_", 1)[1]  # raw|hp|ewma|holt
                    return aggregate_predictions(
                        predictions,
                        policy="reconstruct_warm",
                        hp_lambda=hp_lambda,
                        warm_left_windows_2d=warm_left_windows_2d,
                        warm_true_prefix_1d=warm_true_prefix_1d,
                        reconstruct_warm_filter=suffix,
                        ewma_alpha=ewma_alpha,
                        holt_alpha=holt_alpha,
                        holt_beta=holt_beta,
                    )
                else:
                    # Direct call with the legacy policy
                    return aggregate_predictions(
                        predictions,
                        policy=cand,
                        hp_lambda=hp_lambda,
                        warm_left_windows_2d=warm_left_windows_2d,
                        warm_true_prefix_1d=warm_true_prefix_1d,
                        reconstruct_warm_filter=reconstruct_warm_filter,
                        ewma_alpha=ewma_alpha,
                        holt_alpha=holt_alpha,
                        holt_beta=holt_beta,
                    )
            except Exception as ex:
                last_error = ex
                continue

        raise ValueError(
            "AUTO aggregation failed to evaluate any candidate. "
            f"Last error: {last_error!r}. Candidates tried: {candidates}"
        )

    # -------------------------------------------------------
    # Direct suffix policies: reconstruct_warm_<raw|hp|...>
    # -------------------------------------------------------
    if policy_norm.startswith("reconstruct_warm_"):
        suffix = policy_norm.split("reconstruct_warm_", 1)[1]
        if suffix not in {"raw", "hp", "ewma", "holt"}:
            raise ValueError(
                f"Unknown reconstruct_warm suffix '{suffix}'. "
                "Expected one of: 'raw', 'hp', 'ewma', 'holt'."
            )
        reconstruct_warm_filter = suffix
        policy_norm = "reconstruct_warm"

    # Parse/validate the base policy with the existing utility
    base_policy, rw_filter = _parse_policy(policy_norm, reconstruct_warm_filter)

    have_left = _has_arr(warm_left_windows_2d)
    have_true = _has_arr(warm_true_prefix_1d)
    effective_policy = _maybe_downgrade(base_policy, have_left, have_true)

    # Non-warm paths
    if effective_policy in {"reconstruct", "hp_hist", "hp_raw"}:
        return _dispatch_no_warm(effective_policy, predictions, hp_lambda=hp_lambda)

    # Warm HP paths (LEFT only)
    if effective_policy in {"hp_hist_warm", "hp_raw_warm"}:
        return _dispatch_warm_hp(
            effective_policy,
            predictions,
            warm_left_windows_2d,
            hp_lambda=hp_lambda
        )

    # Reconstruct warm (TRUE prefix preferred; otherwise LEFT)
    if effective_policy == "reconstruct_warm":
        # rw_filter already validated by _parse_policy (raw|hp|ewma|holt)
        return _dispatch_reconstruct_warm(
            predictions,
            warm_left_windows_2d=warm_left_windows_2d,
            warm_true_prefix_1d=warm_true_prefix_1d,
            filter_kind=rw_filter,
            hp_lambda=hp_lambda,
            ewma_alpha=ewma_alpha,
            holt_alpha=holt_alpha,
            holt_beta=holt_beta,
        )

    raise ValueError(
        f"Unknown aggregation policy '{policy}'. "
        f"Expected one of: 'reconstruct', 'hp_hist', 'hp_raw', "
        f"'hp_hist_warm', 'hp_raw_warm', 'reconstruct_warm', or "
        f"'reconstruct_warm_<raw|hp|ewma|holt>'."
    )


