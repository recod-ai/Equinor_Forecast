# common/eval_darts.py
import numpy as np
from typing import Any, Dict, Tuple
import logging

# ===== Ribbons → single timeline (LATEST policy) =====
def latest_from_ribbons(ribbons: np.ndarray) -> np.ndarray:
    ribbons = np.asarray(ribbons)
    if ribbons.ndim == 1:
        ribbons = ribbons.reshape(-1, 1)
    n, H = ribbons.shape
    out = np.empty(n + H - 1, dtype=ribbons.dtype)
    out[:H] = ribbons[0]
    for i in range(1, n):
        out[i + H - 1] = ribbons[i, -1]
    return out

# ===== Inverse scaling helpers =====
def _inv_ts_1d(ts_darts, scaler) -> np.ndarray:
    return scaler.inverse_transform(ts_darts.values()).ravel()

def _inv_ribbons_2d(ribbons: np.ndarray, scaler) -> np.ndarray:
    ribbons = np.asarray(ribbons)
    if ribbons.ndim == 1:
        ribbons = ribbons.reshape(-1, 1)
    return scaler.inverse_transform(ribbons.reshape(-1, 1)).reshape(ribbons.shape[0], -1)


def series_lengths(train_kwargs: Dict[str, Any], prediction_input: Dict[str, Any]) -> Tuple[int, int, int]:
    main_col = train_kwargs["main_col"]
    T_train = len(train_kwargs["X_train"][main_col])
    T_val   = len(train_kwargs["X_val"][main_col])
    T_test  = len(prediction_input["ts_test"][main_col])
    return T_train, T_val, T_test

def log_split_lengths(train_kwargs: Dict[str, Any], prediction_input: Dict[str, Any], title: str = "SPLIT LENGTHS") -> None:
    T_train, T_val, T_test = series_lengths(train_kwargs, prediction_input)
    logging.info(
        f"{title}\n"
        f"+------------------+------+\n"
        f"| key              | val  |\n"
        f"+------------------+------+\n"
        f"| train.len        | {T_train:<4d} |\n"
        f"| val.len          | {T_val:<4d} |\n"
        f"| test.len         | {T_test:<4d} |\n"
        f"| total.len        | {T_train + T_val + T_test:<4d} |\n"
        f"+------------------+------+\n"
    )

def summarize_ribbons_shape(pred_val_ribbons: np.ndarray, pred_test_ribbons: np.ndarray) -> str:
    nV, Hv = np.asarray(pred_val_ribbons).shape
    nT, Ht = np.asarray(pred_test_ribbons).shape
    return f"val_ribbons=(n={nV}, H={Hv}) | test_ribbons=(n={nT}, H={Ht})"

