import pandas as pd
import numpy as np
import pywt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from common.common import split_time_series, augment_with_synthetic_samples, augment_phys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from forecast_pipeline.config import CANON_FEATURES
from forecast_pipeline.config import DEFAULT_DATASET


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Literal
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


# ==============================================================================
# 1. NEW: MODULAR, INTERNAL HELPER FUNCTIONS
# ==============================================================================
# These helpers contain the core logic and will be shared by both final functions.

def _split_data_chronologically(df: pd.DataFrame, test_size: float, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Performs a standard chronological train-validation-test split."""
    if not (0 < test_size < 1 and 0 < val_size < 1 and (test_size + val_size) < 1):
        raise ValueError("test_size/val_size must be > 0 and their sum < 1.")
    
    n = len(df)
    test_start_index = int(n * (1 - test_size))
    val_start_index = int(n * (1 - test_size - val_size))

    df_train = df.iloc[:val_start_index]
    df_val = df.iloc[val_start_index:test_start_index]
    df_test = df.iloc[test_start_index:]
    return df_train, df_val, df_test

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
# 2. REFACTORED PUBLIC FUNCTIONS (Preserving the original interface)
# ==============================================================================
import logging
import numpy as np
from typing import Optional, Dict
import hashlib

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
    scaler_target=None,
    round_decimals: int = 8,
    atol_boundary: float = 1e-8,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """
    Comprehensive leakage check for seq2seq windows.

    - Detects exact duplicate windows across splits (X full, X target-channel, y).
    - Checks boundary overlap: whether the first (H-1) targets of VAL/TEST equal
      the last (H-1) targets of the preceding block (TRAIN/VAL), after optional
      inverse transform.
    - Logs a clean, boxed report via logging.info.

    Returns a dict with all counts and booleans.
    """
    log = logger or logging.getLogger("leakage-check")

    def _box(title: str, lines: list[str]) -> None:
        width = max(len(title), *(len(s) for s in lines)) + 4
        top    = "┌" + "─" * (width - 2) + "┐"
        midsep = "├" + "─" * (width - 2) + "┤"
        bot    = "└" + "─" * (width - 2) + "┘"
        def pad(s): 
            return "│ " + s.ljust(width - 4) + " │"
        log.info(top)
        log.info(pad(title))
        log.info(midsep)
        for s in lines:
            log.info(pad(s))
        log.info(bot)

    def _sha1_rows(a: np.ndarray) -> set[str]:
        """
        Hash each window (row) after rounding; works for 2D (B, T*C) or 3D (B,T,C).
        """
        if a.ndim == 3:
            flat = a.reshape(a.shape[0], -1)
        elif a.ndim == 2:
            flat = a
        else:
            raise ValueError("Expected 2D or 3D array for hashing.")
        flat = np.round(flat, round_decimals)
        # tobytes is fastest; hash row by row
        out = set()
        for i in range(flat.shape[0]):
            out.add(hashlib.sha1(flat[i].tobytes()).hexdigest())
        return out

    def _dup_intersection(A: np.ndarray, B: np.ndarray) -> int:
        return len(_sha1_rows(A) & _sha1_rows(B))

    # --------- shapes / basic info ----------
    Btr = y_train.shape[0]; Bva = y_val.shape[0]; Bte = y_test.shape[0]
    H   = int(output_length)
    ov  = max(H - 1, 0)

    _box("DATA LEAKAGE CHECK – INPUT SUMMARY", [
        f"input_length={input_length} | output_length(H)={H} | overlap(H-1)={ov}",
        f"X_train shape: {getattr(X_train, 'shape', None)}",
        f"X_val   shape: {getattr(X_val,   'shape', None)}",
        f"X_test  shape: {getattr(X_test,  'shape', None)}",
        f"y_train shape: {y_train.shape}",
        f"y_val   shape: {y_val.shape}",
        f"y_test  shape: {y_test.shape}",
    ])

    # --------- duplicate windows across splits ----------
    # 1) Entire X window (all features + channels)
    dup_X_tr_va  = _dup_intersection(X_train, X_val)
    dup_X_tr_te  = _dup_intersection(X_train, X_test)
    dup_X_va_te  = _dup_intersection(X_val,   X_test)

    # 2) Target-in-X channel only (last channel)
    def _last_channel(X): 
        if X.ndim != 3: 
            raise ValueError("X must be 3D (B,T,C).")
        return X[:, :, -1]
    tr_t = _last_channel(X_train); va_t = _last_channel(X_val); te_t = _last_channel(X_test)
    dup_xt_tr_va = _dup_intersection(tr_t, va_t)
    dup_xt_tr_te = _dup_intersection(tr_t, te_t)
    dup_xt_va_te = _dup_intersection(va_t, te_t)

    # 3) y windows
    dup_y_tr_va = _dup_intersection(y_train, y_val)
    dup_y_tr_te = _dup_intersection(y_train, y_test)
    dup_y_va_te = _dup_intersection(y_val,   y_test)

    _box("EXACT DUPLICATES ACROSS SPLITS", [
        f"[X full]  Train vs Val : {dup_X_tr_va}",
        f"[X full]  Train vs Test: {dup_X_tr_te}",
        f"[X full]  Val   vs Test: {dup_X_va_te}",
        f"[X tgt ]  Train vs Val : {dup_xt_tr_va}",
        f"[X tgt ]  Train vs Test: {dup_xt_tr_te}",
        f"[X tgt ]  Val   vs Test: {dup_xt_va_te}",
        f"[y     ]  Train vs Val : {dup_y_tr_va}",
        f"[y     ]  Train vs Test: {dup_y_tr_te}",
        f"[y     ]  Val   vs Test: {dup_y_va_te}",
    ])

    # --------- boundary overlap check (H-1) on targets ----------
    # Optionally inverse-transform targets to physical scale
    def _inv_y(y):
        if scaler_target is None:
            return y.astype(float, copy=False)
        y2 = scaler_target.inverse_transform(y)  # y is (B,H)
        return y2

    ytr_phys = _inv_y(y_train)
    yva_phys = _inv_y(y_val)
    yte_phys = _inv_y(y_test)

    # reconstruct series from (B,H)
    def reconstruct_true_series_local(y_full: np.ndarray) -> np.ndarray:
        n_samples, horizon = y_full.shape
        L = n_samples + horizon - 1
        out = np.empty(L, dtype=float)
        out[:horizon] = y_full[0, :]
        for i in range(1, n_samples):
            out[i + horizon - 1] = y_full[i, -1]
        return out

    tr_rec  = reconstruct_true_series_local(ytr_phys)
    va_rec  = reconstruct_true_series_local(yva_phys)
    te_rec  = reconstruct_true_series_local(yte_phys)

    overlap_train_val = np.max(np.abs(tr_rec[-ov:] - va_rec[:ov])) if (len(tr_rec) >= ov and len(va_rec) >= ov and ov > 0) else np.nan
    overlap_val_test  = np.max(np.abs(va_rec[-ov:] - te_rec[:ov])) if (len(va_rec) >= ov and len(te_rec) >= ov and ov > 0) else np.nan

    leak_boundary_tr_val = bool(np.isfinite(overlap_train_val) and overlap_train_val <= atol_boundary)
    leak_boundary_val_te = bool(np.isfinite(overlap_val_test)  and overlap_val_test  <= atol_boundary)

    _box("BOUNDARY OVERLAP (H-1 targets)", [
        f"Train→Val | max|Δ| over H-1 = {overlap_train_val if np.isfinite(overlap_train_val) else 'n/a'}  (≤ {atol_boundary}? {'YES' if leak_boundary_tr_val else 'NO'})",
        f"Val→Test  | max|Δ| over H-1 = {overlap_val_test  if np.isfinite(overlap_val_test)  else 'n/a'}  (≤ {atol_boundary}? {'YES' if leak_boundary_val_te else 'NO'})",
    ])

    # --------- verdict ----------
    any_exact_dup = any([
        dup_X_tr_va, dup_X_tr_te, dup_X_va_te,
        dup_xt_tr_va, dup_xt_tr_te, dup_xt_va_te,
        dup_y_tr_va, dup_y_tr_te, dup_y_va_te
    ])
    any_boundary_dup = leak_boundary_tr_val or leak_boundary_val_te
    verdict = "PASS (no leakage detected)" if (not any_exact_dup and not any_boundary_dup) else "FAIL (potential leakage)"

    _box("VERDICT", [
        f"Exact duplicates across splits: {'YES' if any_exact_dup else 'NO'}",
        f"Boundary duplication (H-1): {'YES' if any_boundary_dup else 'NO'}",
        f"=> {verdict}"
    ])

    return {
        "dup_X":   {"tr_val": dup_X_tr_va, "tr_test": dup_X_tr_te, "val_test": dup_X_va_te},
        "dup_Xt":  {"tr_val": dup_xt_tr_va, "tr_test": dup_xt_tr_te, "val_test": dup_xt_va_te},
        "dup_y":   {"tr_val": dup_y_tr_va, "tr_test": dup_y_tr_te, "val_test": dup_y_va_te},
        "overlap": {"train_val_maxabs": overlap_train_val, "val_test_maxabs": overlap_val_test},
        "leak_boundary": {"train_val": leak_boundary_tr_val, "val_test": leak_boundary_val_te},
        "verdict": verdict
    }

def report_split_sample_counts(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    *,
    total_windows: int,
    gap_train_val: int = 0,
    gap_val_test: Optional[int] = None,   # if None, will try to infer
    output_length: Optional[int] = None,  # used only to auto-set gap_val_test=output_length-1
    logger: Optional[logging.Logger] = None,
    title: str = "SPLIT SUMMARY (SAMPLES ONLY)"
) -> None:
    """
    Logs a boxed summary with the number of samples (windows) in Train/Val/Test
    and their percentages relative to the total number of windows (BEFORE any
    synthetic augmentation). Also reports gap windows skipped by design.

    Notes
    -----
    - 'Samples' = windows (first dimension of X_*).
    - To reflect your current policy (keep train→val, skip val→test),
      pass gap_train_val=0 and gap_val_test=output_length-1.
    """
    log = logger or logging.getLogger("split-summary")

    n_tr = int(getattr(X_train, "shape", [0])[0] if X_train is not None else 0)
    n_va = int(getattr(X_val,   "shape", [0])[0] if X_val   is not None else 0)
    n_te = int(getattr(X_test,  "shape", [0])[0] if X_test  is not None else 0)

    used = n_tr + n_va + n_te

    # If user didn't pass gap_val_test, try to set from output_length or infer
    if gap_val_test is None:
        if output_length is not None:
            gap_val_test = max(0, int(output_length) - 1)
        else:
            # Best-effort inference assuming only two gaps (train→val and val→test)
            gap_val_test = max(0, total_windows - used - max(0, gap_train_val))

    gap_total = max(0, int(gap_train_val)) + max(0, int(gap_val_test))
    dropped   = max(0, total_windows - used)  # what actually ended up unused

    def pct(x: int, denom: int) -> float:
        return (100.0 * x / denom) if denom > 0 else 0.0

    lines = [
        f"Total windows (no augmentation): {total_windows}",
        f"Train samples: {n_tr} ({pct(n_tr, total_windows):.2f}%)",
        f"Validation samples: {n_va} ({pct(n_va, total_windows):.2f}%)",
        f"Test samples: {n_te} ({pct(n_te, total_windows):.2f}%)",
        f"Gap windows (skipped by design): train→val={gap_train_val}, val→test={gap_val_test}, total={gap_total}",
        f"Unused windows (dropped): {dropped} ({pct(dropped, total_windows):.2f}%)",
        f"Check: used/total = {used}/{total_windows} ({pct(used, total_windows):.2f}%)",
    ]

    width = max(len(title), *(len(s) for s in lines)) + 4
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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def inspect_set_boundaries(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    target_idx: int = -1,
    well_name: str = "Unknown Well",
    num_points: int = 50
) -> None:
    """
    Prints and plots the data at the boundaries between train/val/test sets
    to visually inspect for continuity and gaps.

    This function looks at the target variable from the last feature dimension.

    Args:
        X_train: The unscaled training data array of windows.
        X_val: The unscaled validation data array of windows.
        X_test: The unscaled test data array of windows.
        target_idx: The index of the target variable in the last dimension of X arrays.
        well_name: An identifier for the plot title.
        num_points: How many data points to show on each side of a boundary.
    """
    print("---" * 20)
    print(f"🔎 Inspecting Data Continuity at Set Boundaries for [Well: {well_name}]")
    print("---" * 20)

    # --- 1. Extract the relevant time series segments ---
    # We take the *entire* last sample from train/val and first sample from val/test
    last_train_sample = X_train[-1, :, target_idx]
    first_val_sample = X_val[0, :, target_idx]
    last_val_sample = X_val[-1, :, target_idx]
    first_test_sample = X_test[0, :, target_idx]

    # --- 2. Print the last 5 elements for numerical inspection ---
    print(f"\n[Numeric Check] Last 5 values of each boundary sample:")
    print(f"  Last Train Sample (...end): {np.round(last_train_sample[-5:], 2)}")
    print(f"  First Val Sample (start...):  {np.round(first_val_sample[:5], 2)}")
    print("-" * 20)
    print(f"  Last Val Sample (...end):   {np.round(last_val_sample[-5:], 2)}")
    print(f"  First Test Sample (start...): {np.round(first_test_sample[:5], 2)}")

    # --- 3. Plot the transitions for visual inspection ---
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharey=True)
    fig.suptitle(f"Data Continuity Check for Well: {well_name}", fontsize=16)

    # Plot 1: Train -> Validation Transition
    ax1 = axes[0]
    train_segment = last_train_sample[-num_points:]
    val_segment = first_val_sample[:num_points]
    combined_series1 = np.concatenate([train_segment, val_segment])
    x_axis1 = np.arange(len(combined_series1))

    ax1.plot(x_axis1[:num_points], train_segment, 'o-', label=f'End of Last Train Sample')
    ax1.plot(x_axis1[num_points:], val_segment, 'o-', label=f'Start of First Val Sample')
    ax1.axvline(x=num_points - 0.5, color='r', linestyle='--', label='Train/Val Boundary')
    ax1.set_title("Train → Validation Transition")
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: Validation -> Test Transition
    ax2 = axes[1]
    val_segment_end = last_val_sample[-num_points:]
    test_segment = first_test_sample[:num_points]
    combined_series2 = np.concatenate([val_segment_end, test_segment])
    x_axis2 = np.arange(len(combined_series2))

    ax2.plot(x_axis2[:num_points], val_segment_end, 'o-', color='green', label=f'End of Last Val Sample')
    ax2.plot(x_axis2[num_points:], test_segment, 'o-', color='purple', label=f'Start of First Test Sample')
    ax2.axvline(x=num_points - 0.5, color='r', linestyle='--', label='Val/Test Boundary (with Gap)')
    ax2.set_title("Validation → Test Transition")
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

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
    Refatorado: cria TODAS as janelas antes de dividir.
    Evita gaps e sobreposições incorretas entre conjuntos.
    """

    # 1) Cria TODAS as janelas
    X_all, y_all = create_sliding_window_seq_to_seq(
        df, target_col, input_length, output_length
    )
    total_windows = len(X_all)

    print('total_windows', total_windows)

    # 2) Define quantidades para Test e Val (em nº de janelas)
    n_test = int(total_windows * test_size)
    n_train_val = total_windows - n_test
    n_val = int(n_train_val * val_size)
    n_train = n_train_val - n_val

    gap = output_length - 1
    # 3) Divide cronologicamente (ordem natural)
    X_train, y_train = X_all[:n_train-gap], y_all[:n_train-gap]
    X_val,   y_val   = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test,  y_test  = X_all[n_train + n_val + gap:], y_all[n_train + n_val+gap:]
    y_train_original = y_train.copy()

    report_split_sample_counts(
        X_train, X_val, X_test,
        total_windows=total_windows  # len(X_all)
    )

    # =========================================================================
    # NEW: CALL THE INSPECTION FUNCTION HERE
    # =========================================================================
    # Call the diagnostic function to visualize the data at the boundaries
    # before any scaling is applied.
    # inspect_set_boundaries(X_train, X_val, X_test, well_name='None')
    # =========================================================================

    # 4) Data augmentation no treino (opcional)
    if data_aug_params and data_aug_params.get("data_sample", 1.0) < 1.0:
        X_train, y_train = augment_with_synthetic_samples(
            X_train, y_train, data_sample=data_aug_params["data_sample"]
        )

    import matplotlib.pyplot as plt
    plt.plot(y_train[:, -1])
    plt.show()
    
    # 5) Escalonamento
    X_feats = [X[:, :, :-1] for X in (X_train, X_val, X_test)]
    X_train_feats_scaled, X_val_feats_scaled, X_test_feats_scaled, \
        y_train_scaled, y_val_scaled, y_test_scaled, \
        scaler_X, scaler_target = _scale_feature_and_target_sets(
            *X_feats,
            y_train, y_val, y_test,
            scaler_type="robust"
        )

    def scale_target_in_x(X):
        flat = X[:, :, -1:].reshape(-1, 1)
        scaled = scaler_target.transform(flat)
        return scaled.reshape(X.shape[0], X.shape[1], 1)

    X_train_t = scale_target_in_x(X_train)
    X_val_t   = scale_target_in_x(X_val)
    X_test_t  = scale_target_in_x(X_test)

    X_train_scaled = np.concatenate([X_train_feats_scaled, X_train_t], axis=-1)
    X_val_scaled   = np.concatenate([X_val_feats_scaled,   X_val_t],   axis=-1)
    X_test_scaled  = np.concatenate([X_test_feats_scaled,  X_test_t],  axis=-1)

    res = check_seq2seq_data_leakage(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_scaled, y_val_scaled, y_test_scaled,
        input_length=input_length, output_length=output_length,
        scaler_target=scaler_target,     # passe None se já estiverem desscalados
        round_decimals=8, atol_boundary=1e-8
    )


    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_scaled, y_val_scaled, y_test_scaled,
        scaler_X, scaler_target, y_train_original,
    )


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

# if DEFAULT_DATASET == "VOLVE":


#     def prepare_data_seq(
#         df: pd.DataFrame,
#         target_col: str,
#         input_length: int,
#         output_length: int,
#         test_size: float = 0.5,
#         val_size: float = 0.1,
#         data_aug_params: bool = True,
#     ):
#         """
#         Prepares seq-to-seq data using traditional chronological split:
#         - Train: from start to (1 - test_size - val_size)
#         - Validation: next val_size
#         - Test: last test_size
    
#         Returns:
#           X_train, X_val, X_test,
#           y_train, y_val, y_test,
#           scaler_target,
#           y_train_original
#         """
    
#         n = len(df)
#         test_start = int(n * (1 - test_size))
#         val_start = int(n * (1 - test_size - val_size))
    
#         df_train = df.iloc[:val_start]
#         df_val   = df.iloc[val_start:test_start]
#         df_test  = df.iloc[test_start:]
    
#         # 2. Sliding windows
#         X_train, y_train = create_sliding_window_seq_to_seq(df_train, target_col, input_length, output_length)
#         X_val, y_val     = create_sliding_window_seq_to_seq(df_val,   target_col, input_length, output_length)
#         X_test, y_test   = create_sliding_window_seq_to_seq(df_test,  target_col, input_length, output_length)
    
#         # Keep original y_train
#         y_train_original = y_train.copy()
    
#         # 3. Data augmentation (train only)
#         if data_aug_params:
#             aug_params = data_aug_params or {}
#             X_train, y_train = augment_with_synthetic_samples(
#                 X_train, y_train, data_sample=aug_params.get('data_sample', 0.25)
#             )
    
#         # 4. Separate features / target in X
#         X_train_feats = X_train[:, :, :-1]
#         X_train_targ  = X_train[:, :, -1:].reshape(-1, 1)
#         X_val_feats   = X_val[:, :, :-1]
#         X_val_targ    = X_val[:, :, -1:].reshape(-1, 1)
#         X_test_feats  = X_test[:, :, :-1]
#         X_test_targ   = X_test[:, :, -1:].reshape(-1, 1)
    
#         # Flatten for scaling
#         n_train, win_len, n_feat = X_train_feats.shape
#         n_val   = X_val_feats.shape[0]
#         n_test  = X_test_feats.shape[0]
    
#         X_train_flat = X_train_feats.reshape(-1, n_feat)
#         X_val_flat   = X_val_feats.reshape(-1, n_feat)
#         X_test_flat  = X_test_feats.reshape(-1, n_feat)
    
#         # 5a. Scale features
#         scaler_X = StandardScaler()
#         # scaler_X = RobustScaler()
#         X_train_scaled_feats = scaler_X.fit_transform(X_train_flat)
#         X_val_scaled_feats   = scaler_X.transform(X_val_flat)
#         X_test_scaled_feats  = scaler_X.transform(X_test_flat)
    
#         X_train_scaled_feats = X_train_scaled_feats.reshape(n_train, win_len, n_feat)
#         X_val_scaled_feats   = X_val_scaled_feats.reshape(n_val, win_len, n_feat)
#         X_test_scaled_feats  = X_test_scaled_feats.reshape(n_test, win_len, n_feat)
    
#         # 5b. Scale target
#         scaler_target = StandardScaler()
#         # scaler_target = RobustScaler()
#         X_train_scaled_targ = scaler_target.fit_transform(X_train_targ)
#         X_val_scaled_targ   = scaler_target.transform(X_val_targ)
#         X_test_scaled_targ  = scaler_target.transform(X_test_targ)
    
#         X_train_scaled_targ = X_train_scaled_targ.reshape(n_train, win_len, 1)
#         X_val_scaled_targ   = X_val_scaled_targ.reshape(n_val, win_len, 1)
#         X_test_scaled_targ  = X_test_scaled_targ.reshape(n_test, win_len, 1)
    
#         # 6. Rebuild scaled windows
#         X_train_scaled = np.concatenate([X_train_scaled_feats, X_train_scaled_targ], axis=-1)
#         X_val_scaled   = np.concatenate([X_val_scaled_feats, X_val_scaled_targ], axis=-1)
#         X_test_scaled  = np.concatenate([X_test_scaled_feats, X_test_scaled_targ], axis=-1)
    
#         # 7. Scale y
#         y_train_flat = y_train.flatten().reshape(-1, 1)
#         y_train_scaled = scaler_target.transform(y_train_flat).flatten().reshape(y_train.shape)
#         y_val_flat = y_val.flatten().reshape(-1, 1)
#         y_val_scaled = scaler_target.transform(y_val_flat).flatten().reshape(y_val.shape)
#         y_test_flat = y_test.flatten().reshape(-1, 1)
#         y_test_scaled = scaler_target.transform(y_test_flat).flatten().reshape(y_test.shape)
    
#         return (
#             X_train_scaled, X_val_scaled, X_test_scaled,
#             y_train_scaled, y_val_scaled, y_test_scaled,
#             scaler_X, scaler_target,
#             y_train_original
#         )


# elif DEFAULT_DATASET == "UNISIM_IV":

#     def prepare_data_seq(
#         df: pd.DataFrame,
#         target_col: str,
#         input_length: int,
#         output_length: int,
#         test_size: float = 0.5,
#         val_size: float = 0.1,
#         data_aug_params: bool = True
#     ):
#         # ------------------------------------------- 1. split
#         df_temp, df_test = split_time_series(df, test_size)
    
#         X_temp_raw, y_temp_raw = create_sliding_window_seq_to_seq(
#             df_temp, target_col, input_length, output_length
#         )
#         X_test_raw, y_test_raw = create_sliding_window_seq_to_seq(
#             df_test, target_col, input_length, output_length
#         )
    
#         # ------------------------------------------- 2. FIT SCALERS *ANTES* DA DA
#         # achata apenas o TRAIN real
#         n_temp, win_len, n_feat = X_temp_raw.shape
#         X_temp_flat = X_temp_raw.reshape(-1, n_feat)
    
#         scaler_X = RobustStandardScaler().fit(X_temp_flat)
#         scaler_y = StandardScaler().fit(y_temp_raw.flatten().reshape(-1, 1))
    
#         save_scaler(scaler_X, 'scalers/scaler_X.pkl')          # caminhos únicos por case
#         save_scaler(scaler_y, 'scalers/scaler_target.pkl')
    
#         # ------------------------------------------- 3. APPLY SCALERS
#         def scale_X(X):
#             sh = X.shape
#             return scaler_X.transform(X.reshape(-1, n_feat)).reshape(sh)
    
#         def scale_y(y):
#             return scaler_y.transform(y.reshape(-1, 1)).reshape(y.shape)
    
#         X_temp_scaled = scale_X(X_temp_raw)
#         y_temp_scaled = scale_y(y_temp_raw)
#         X_test_scaled = scale_X(X_test_raw)
#         y_test_scaled = scale_y(y_test_raw)
    
#         # ------------------------------------------- 4. DATA AUG (agora no espaço z-score)
#         # if data_aug_params:
#         #     X_temp_scaled, y_temp_scaled = augment_with_synthetic_samples(
#         #         X_temp_scaled, y_temp_scaled, scales=[1.5,2,3,5,7,9,]   # ≤4 fatores já é suficiente
#         #     )
    
#         # ------------------------------------------- 5. reshape target dentro de X …
#         # (mesma lógica que já tinha, mas usando arrays _scaled_)
#         X_temp_feats = X_temp_scaled[:, :, :-1]
#         X_temp_targ  = X_temp_scaled[:, :, -1:]
#         X_test_feats = X_test_scaled[:, :, :-1]
#         X_test_targ  = X_test_scaled[:, :, -1:]
    
#         # concatena alvo nas features
#         X_train = np.concatenate([X_temp_feats, X_temp_targ], axis=-1)
#         X_test  = np.concatenate([X_test_feats, X_test_targ], axis=-1)
    
#         # ------------------------------------------- 6. val split
#         n_val = int(X_test.shape[0] * val_size)
#         X_val, y_val = X_test[:n_val], y_test_scaled[:n_val]
#         X_test, y_test = X_test[n_val:], y_test_scaled[n_val:]
    
#         return (X_train, X_val, X_test,
#                 y_temp_scaled, y_val, y_test,
#                 scaler_X, scaler_y,          # para inversão de previsões
#                 y_temp_raw)        # curva original p/ métrica




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
    tail_size = 1
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