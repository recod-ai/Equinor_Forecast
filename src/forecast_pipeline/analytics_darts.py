# src/forecast_pipeline/analytics_darts.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse your helpers already used in the notebook
from common.eval_darts import latest_from_ribbons, _inv_ts_1d, _inv_ribbons_2d
from forecast_pipeline.analytics import evaluate_job
from forecast_pipeline.plotting import plot_darts_integrated


def _scale_1d(arr: np.ndarray, scaler) -> np.ndarray:
    """Scale a 1D array with a scikit-learn scaler; returns shape (N,)."""
    arr = np.asarray(arr).reshape(-1, 1)
    return scaler.transform(arr).ravel()

# --- keep existing imports ---

def _ensure_train_windows(y_train_original, train_kwargs):
    """
    Return train as 2-D 'windows' (N, H) for the legacy evaluator.
    If None or 1-D, rebuild from Darts X_train main column as (N, 1).
    """
    import numpy as np
    if y_train_original is not None and getattr(y_train_original, "ndim", 1) == 2:
        return y_train_original

    scaler   = train_kwargs["scaler_target"]
    main_col = train_kwargs["main_col"]
    ts_train = train_kwargs["X_train"][main_col]     # Darts TimeSeries
    series   = _inv_ts_1d(ts_train, scaler)          # 1-D physical units
    return np.asarray(series, dtype=float).reshape(-1, 1)


def _ensure_legacy_params(params: dict, train_kwargs: dict) -> dict:
    """
    Ensure legacy keys expected by plotting/evaluator exist:
      - lag_window  (from input_chunk_length)
      - horizon     (from output_chunk_length)
    Leaves existing keys untouched.
    """
    p = dict(params) if params is not None else {}

    def _get(key):
        return (
            p.get(key)
            or (p.get("params") or {}).get(key)
            or train_kwargs.get(key)
            or (train_kwargs.get("params") or {}).get(key)
        )

    L = p.get("lag_window")  or _get("input_chunk_length")
    H = p.get("horizon")     or _get("output_chunk_length")

    if L is None or H is None:
        raise KeyError("Could not infer lag_window/horizon from params/train_kwargs.")

    p.setdefault("lag_window",  int(L))
    p.setdefault("horizon",     int(H))
    p.setdefault("architecture_name", p.get("architecture_name", "Darts"))
    return p




def evaluate_job_from_darts(
    *,
    train_kwargs: Dict[str, Any],
    prediction_input: Dict[str, Any],
    pred_val_ribbons: np.ndarray,
    pred_test_ribbons: np.ndarray,
    y_train_original: np.ndarray,
    params: Dict[str, Any],
    well: str,
    config: Optional[Dict[str, Any]] = None,
    plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any],
           pd.DataFrame, pd.DataFrame, Dict[str, Any],
           Dict[str, Any]]:
    """
    Adapter that:
      1) Takes Darts historical_forecasts ribbons (val/test),
      2) Reconstructs point series via 'latest' policy,
      3) Aligns/slices to exact split lengths in ORIGINAL units,
      4) Re-scales them so we can call the existing `evaluate_job` unmodified,
      5) Returns the exact tuple that `evaluate_job` returns (tolerant to 6|7).

    This intentionally routes through the *point-forecast* branch of `evaluate_job`
    (architecture_name 'Darts_*' is NOT in SEQ2SEQ_ARCHS), so no pipeline semantics change.
    """
    config = config or {}

    main_col = train_kwargs["main_col"]
    scaler   = train_kwargs["scaler_target"]

    # Build a *train series in physical units* from X_train (main column)
    ts_train_s = train_kwargs["X_train"][main_col]   # Darts TimeSeries
    train_series_phys = _inv_ts_1d(ts_train_s, scaler)  # 1-D (len = train)

    # Coerce to "windows" shape expected by evaluate_job's train reconstruction
    # (each window has horizon=1 → reconstruction is identity)
    y_train_windows_phys = np.asarray(train_series_phys, dtype=float).reshape(-1, 1)

    # True series (original units)
    ts_train = train_kwargs["X_train"][main_col]
    ts_val   = train_kwargs["X_val"][main_col]
    ts_test  = prediction_input["ts_test"][main_col]

    y_train_true = _inv_ts_1d(ts_train, scaler)
    y_val_true   = _inv_ts_1d(ts_val,   scaler)
    y_test_true  = _inv_ts_1d(ts_test,  scaler)

    # Ribbons → original-unit predictions → slice to splits
    val_rib_u  = _inv_ribbons_2d(pred_val_ribbons,  scaler)
    test_rib_u = _inv_ribbons_2d(pred_test_ribbons, scaler)

    yhat_val_full  = latest_from_ribbons(val_rib_u)
    yhat_test_full = latest_from_ribbons(test_rib_u)

    yhat_val  = np.asarray(yhat_val_full, dtype=float)[: len(y_val_true)]
    yhat_test = np.asarray(yhat_test_full, dtype=float)[: len(y_test_true)]

    # Re-scale both truth and preds so `evaluate_job` can invert & compute as usual
    y_val_scaled       = _scale_1d(y_val_true, scaler)
    y_val_pred_scaled  = _scale_1d(yhat_val,   scaler)
    y_test_scaled      = _scale_1d(y_test_true, scaler)
    y_test_pred_scaled = _scale_1d(yhat_test,   scaler)

    # Inject legacy keys expected by plotting/evaluator (lag_window/horizon)
    params_legacy = _ensure_legacy_params(params, train_kwargs)

    # Delegate to your existing evaluation (point-forecast path)
    ev = evaluate_job(
        y_test_scaled=y_test_scaled,
        y_test_pred=y_test_pred_scaled,
        y_val_scaled=y_val_scaled,
        y_val_pred=y_val_pred_scaled,
        scaler_target=scaler,
        y_train_original=y_train_windows_phys,
        params=params_legacy,
        config=config,
        well=well,
        plot=plot,
        ensemble_out=None,
        x_train_main_windows=None,
    )

    # ✅ tolerant unpack (6 or 7)
    series_artifacts: Dict[str, Any] = {}
    if isinstance(ev, (tuple, list)) and len(ev) == 7:
        (agg_test_df, cum_test_df, gm_test,
         agg_val_df,  cum_val_df,  gm_val,
         series_artifacts) = ev
    elif isinstance(ev, (tuple, list)) and len(ev) == 6:
        (agg_test_df, cum_test_df, gm_test,
         agg_val_df,  cum_val_df,  gm_val) = ev
        series_artifacts = {}
    else:
        raise ValueError(
            f"Unexpected evaluate_job return: type={type(ev)} len={len(ev) if hasattr(ev,'__len__') else 'n/a'}"
        )

    if plot:
        # Title like: "Darts: TiDE — P15 (tide_shallow_fast)"
        arch = params_legacy.get("architecture_name", "")
        model_name = arch.split("_", 1)[1] if arch.startswith("Darts_") and "_" in arch else (arch or "Darts")
        profile = (
            params_legacy.get("profile")
            or params_legacy.get("profile_name")
            or (params_legacy.get("params") or {}).get("profile")
        )
        suffix = f" ({profile})" if profile else ""
        title_prefix = f"Darts: {model_name} — {well}{suffix}"

        plot_darts_integrated(
            train_kwargs=train_kwargs,
            prediction_input=prediction_input,
            pred_val_ribbons=pred_val_ribbons,
            pred_test_ribbons=pred_test_ribbons,
            title_prefix=title_prefix,
        )

    # ✅ return 7 always (matches unpack_eval7 + Series Store / self-heal)
    return (
        agg_test_df, cum_test_df, gm_test,
        agg_val_df,  cum_val_df,  gm_val,
        series_artifacts,
    )

