# src/forecast_pipeline/arps_offline.py


# =============================================================================
# Offline Analytic ARPS Post-Processing (PINN → Analytic Curve)
# =============================================================================
#
# High-level purpose
# ------------------
# This module is a deterministic post-processing step that takes the PINN
# prediction(s) and converts them into a simple, physically-plausible analytic
# decline curve (ARPS-like exponential). It is meant to:
#   1) stabilize long-horizon extrapolation,
#   2) provide a robust "baseline" analytic shape,
#   3) preserve strict evaluation rules (no target leakage),
#   4) optionally produce an ensemble ("spaghetti members") for visualization
#      and robustness, then reduce it to a single final curve.
#
#
# Core idea in plain English
# --------------------------
# The PINN produces a predicted ribbon/trajectory in *scaled space*.
# We temporarily invert it back to PHYSICAL units to estimate a geometry/shape
# parameter (here called `b`, the decline rate/curvature control).
#
# Crucially:
#   - `b` is allowed to be inferred from the PINN prediction shape.
#   - The anchor `q0` (the starting level) must come from HISTORY (X-only),
#     in PHYSICAL units, when running offline_analytic mode.
#     We never anchor from the PINN head to avoid leakage / self-anchoring.
#
#
# Spaghetti / ensemble members (how multiple b values are used)
# -------------------------------------------------------------
# When "spaghetti" mode is enabled, the module does NOT train multiple PINNs.
# Instead, it builds many analytic ARPS trajectories by sweeping a set of
# candidate `b` values on the Validation split (past-only context):
#
#   Step 1) Choose a candidate set of b values
#           - either a grid (e.g., evenly spaced) or sampled values in a range.
#           - size is configurable; think tens to hundreds of candidates.
#
#   Step 2) For each candidate b_k:
#           - hold q0 fixed from HISTORY (physical anchor),
#           - render an analytic curve y_k(t) in PHYSICAL units,
#           - (optionally) enforce monotonic constraints / guard rails.
#
#   Step 3) Outlier trimming (robustness):
#           - compute a scalar summary per member (e.g., AUC in PHYSICAL units),
#           - drop extreme members (too aggressive / too flat / unstable).
#
#   Step 4) Reduce ensemble → single curve:
#           - aggregate the remaining members pointwise (median is typical),
#           - obtain a robust final analytic curve,
#           - rescale back to pipeline space for compatibility.
#
# Result: you get both (a) a robust single curve and (b) optional members for
# integrated visualization / uncertainty bands — without changing the PINN.
#
#
# Coupling between Validation and Test (sharing b across splits)
# -------------------------------------------------------------
# This module can optionally "couple" splits so that Test uses a `b` informed by
# Validation (still respecting causality):
#
#   - val_only:
#       * estimate b from VAL (or build b candidates on VAL),
#       * apply that same b to TEST (VAL stays as-is or analytic, per config).
#
#   - val_plus_test:
#       * compute b_val and b_test (or summaries),
#       * combine them with a configured weighting into b_final,
#       * apply b_final consistently to both VAL and TEST predictions.
#
# The anchor q0 is always resolved from history windows (X-only) for each split.
#
#
# What metrics are computed "on top of who" (evaluation target)
# -------------------------------------------------------------
# Metrics are computed over the finalized predictions produced by the pipeline
# for each split (VAL and/or TEST), compared against the ground-truth targets
# for those rows:
#
#   - If offline_analytic is enabled:
#       predictions = analytic curve output (possibly coupled / spaghetti-reduced)
#       reference   = y_true in that split (VAL or TEST)
#
# The module itself focuses on producing the analytic predictions; the pipeline
# around it typically computes metrics between (pred vs y_true) per row/sample.
#
#
# Public entry points (conceptual responsibilities)
# -------------------------------------------------
# - _fit_arps_core_from_pinn:
#     * invert PINN predictions to PHYSICAL (fit-time only),
#     * estimate/collect candidate b values, filter and reduce to b*,
#     * render analytic curve using (q0_history_phys, b*),
#     * return curve in scaled space + metadata.
#
# - _apply_arps_from_pinn:
#     * given (q0_phys, b), render analytic curve (no fitting),
#     * used heavily by spaghetti members to generate many trajectories cheaply.
#
# - _maybe_coupling_offline_analytic:
#     * orchestrates VAL/TEST coupling rules,
#     * resolves q0 anchors from history-only fields in outs,
#     * optionally runs spaghetti generation → trimming → aggregation.
#
# - analytic_exponential_extrapolation_batch:
#     * legacy-friendly batch helper for extrapolation using the same core logic.
#
#
# Key invariants (design constraints)
# ----------------------------------
# - No leakage:
#     * Val/Test targets never influence q0 or fitting decisions.
#     * Any cross-boundary sharing is X-only context (past observations) needed
#       to build windows; never uses future labels.
#
# - Physical-first decisions:
#     * trimming (AUC), q0 validity checks, and robustness logic happen in
#       PHYSICAL units; outputs are then scaled back for pipeline consistency.
#
# - Deterministic post-processing:
#     * this module does not retrain the PINN; it reshapes/extrapolates outputs.
# =============================================================================


# =============================================================================
# Offline Analytic ARPS Post-Processing (PINN → Analytic Curve)
# =============================================================================
#
# What this module does
# ---------------------
# This file implements a *deterministic* post-processing layer that turns a
# model's (PINN) predicted ribbon into a physically-plausible extrapolation
# using an ARPS-style exponential form:
#
#     q(t) = q0 * exp(b * t)
#
# with optional monotonic enforcement and coupling between VAL/TEST splits.
#
# The key design principle is **no leakage**:
#   - Geometry parameter `b` may be estimated from PINN predictions.
#   - Anchor `q0` must come from history (X) in PHYSICAL units when running
#     offline_analytic mode (Scenario 2). We never "anchor from the PINN head".
#
#
# Public surface (exported in __all__)
# -----------------------------------
# 1) _fit_arps_core_from_pinn
#    - Input:  y_pinn_scaled (N, H_train), scaler_y, q0_override_phys (required),
#              plus fit knobs (fit_window, region, outlier filtering).
#    - Output: y_analytic_scaled (1, H_target) and analytic_params dict.
#    - Responsibility:
#        * Inverse PINN ribbon to PHYSICAL (only for fitting).
#        * Fit candidate b's across ensemble members, guard them, optionally
#          filter outliers, then reduce to b*.
#        * Build the analytic curve in PHYSICAL using (q0_override_phys, b*),
#          then scale back to pipeline space.
#
# 2) _apply_arps_from_pinn
#    - Input:  pinn ribbon (usually 1D), scaler_y, H_target, b, q0_override_phys.
#    - Output: y_scaled (1, H_target)
#    - Responsibility:
#        * "Render" the analytic curve given (q0, b) without fitting.
#        * In spaghetti mode, accepts rebuild_log_ctx to aggregate per-member
#          rebuild stats (avoids log spam).
#
# 3) _maybe_coupling_offline_analytic
#    - Input:  outs_val / outs_test dicts, latent_cfg knobs, split lengths.
#    - Output: (new_outs_val, new_outs_test) with pred replaced when enabled.
#    - Responsibility:
#        * Gate on latent_cfg.mode == "offline_analytic" and coupling mode.
#        * Resolve history anchors q0_val_phys / q0_test_phys from outs.
#        * Couple splits by sharing `b`:
#            - val_only: use b_val on TEST (VAL stays base)
#            - val_plus_test: compute weighted b_final and apply to both
#        * Optional "spaghetti": build multiple members using a b-candidate set,
#          optionally trim trajectories by AUC in PHYSICAL space, aggregate in
#          PHYSICAL space, rescale to keep pipeline compatibility.
#
# 4) analytic_exponential_extrapolation_batch
#    - Legacy-compatible batch extrapolator for (N, H_train) → (N, H_target).
#    - Uses the same math core (fit b per row; build curve; rescale).
#
# 5) _inverse_2d_with_scaler / _transform_2d_with_scaler
#    - Robust helpers for scalers trained either per-timestep (n_features=H) or
#      on flattened targets (n_features=1). These are the only places where we
#      rely on scaler semantics.
#
# 6) PIPELINE_PRESETS + describe_pipeline_presets + build_pipeline_config_overrides
#    - Convenience "recipes" to wire the pipeline into one of a few stable
#      operational modes (OFF, FULL_SEQUENCE, OFFLINE_ANALYTIC_*).
#
#
# Data flow / contracts (mental model)
# -----------------------------------
#
#  A) History anchor path (q0)
#     ------------------------
#     X_any (dataset/ndarray/tensor) ──► extract_first_window (L,F)
#                                      └─► compute_history_q0_phys
#                                            - selects a channel (default last)
#                                            - optional inverse of that channel
#                                            - robust smoothing (Hampel, HP)
#                                            - tail reducer (median/mean/last)
#                                            - returns q0_phys + meta
#
#     q0_phys is later stored in outs["q0_anchor_phys"] and/or outs["anchor"]["q0_phys"].
#
#  B) Geometry fitting path (b)
#     --------------------------
#     PINN prediction ribbon (scaled) ──► _fit_arps_core_from_pinn
#                                         - inverse to phys for fitting
#                                         - fit b candidates from ribbon shape
#                                         - outlier filtering + reducer
#                                         - guard b for monotonic constraints
#                                         - build analytic curve in phys
#                                         - rescale to return y_analytic_scaled
#
#  C) Coupling + spaghetti path (optional)
#     ------------------------------------
#     outs_val/outs_test + analytic_params ──► _maybe_coupling_offline_analytic
#       - resolves q0 anchors from outs (history-only)
#       - shares b across splits (val_only or weighted val_plus_test)
#       - if spaghetti:
#            * build many curves via _apply_arps_from_pinn (with rebuild_log_ctx)
#            * trim members by AUC in PHYSICAL
#            * aggregate members in PHYSICAL, rescale back
#            * optionally attach pred_members + meta for plotting/integrated view
#
#
# Key invariants
# --------------
# - "Scenario 2" strictness:
#     * When offline_analytic is active, q0 is NOT estimated from the PINN head.
#       Missing/invalid q0_history should result in a skip (coupling) or a
#       hard failure (fit/apply) depending on the caller's contract.
#
# - Scale discipline:
#     * Fit/trim/aggregate decisions that depend on magnitudes (AUC, q0 validity,
#       robust smoothing) are done in PHYSICAL units.
#     * Returned predictions are scaled to match the rest of the pipeline.
#
# - Logging discipline:
#     * Spaghetti mode aggregates rebuild stats into compact summaries.
#
# Quick navigation
# ----------------
# - If you want to change how q0 is computed: compute_history_q0_phys
# - If you want to change b estimation / filtering: _fit_arps_core_from_pinn
# - If you want to change how coupling behaves: _maybe_coupling_offline_analytic
# - If scalers behave unexpectedly: _inverse_2d_with_scaler / _transform_2d_with_scaler
# =============================================================================

# =============================================================================
# Methodological note: Offline calibration with past-only context (Scenario B)
# =============================================================================
# This module follows an *offline* evaluation protocol that mirrors how the
# forecaster runs *online* (causal), while preserving every row assigned to
# Validation/Test.
#
# Problem: sliding windows need history
# -------------------------------------
# Window-based forecasting creates each sample as:
#   - X(t): lookback window of length L ending at time t-1 (past observations)
#   - y(t): horizon of length H starting at time t (future targets)
#
# With a strict chronological split, the first L samples of Val/Test would not
# have enough history if we only allowed data *inside* the split to build X.
# Many pipelines solve this by dropping those samples, which:
#   - hides boundary behavior (the hardest region),
#   - distorts metrics on small datasets,
#   - and changes the effective evaluation interval.
#
# Our choice: "offline calibration with past-only context"
# --------------------------------------------------------
# We enforce a hard chronological split on *targets* (y), but we allow the tail
# of the previous split to be used as *input context* (X-only) for constructing
# the first windows of the next split.
#
# Two hard rules (no leakage)
# ---------------------------
# 1) Target isolation (hard boundary):
#    Validation/Test targets never include training indices.
#    If `split_idx` is the first index of Validation:
#      y_train uses indices <  split_idx
#      y_val   uses indices >= split_idx
#    Same rule applies for Test.
#
# 2) Context injection is input-only (X-only):
#    The only information allowed to cross the split boundary is *past*
#    observations required to build X windows. No future targets, labels, or any
#    function of them may be used to build features, fit parameters, or set
#    initial states for Val/Test.
#
# Illustrative example (L=2, H=1)
# -------------------------------
# Hard boundary between index 9 (Train) and 10 (Val):
#
#   Train:
#     t=9    X(9)  = [y(7), y(8)]     y(9)  = [y(9)]
#
#   Val (first sample keeps full history via injected context):
#     t=10   X(10) = [y(8), y(9)]  <- past tail (Train context)
#           y(10) = [y(10)]       <- strictly in Validation
#
# This is causal and valid: at deployment time, y(9) is known when predicting
# y(10). We are not using the future to predict the past.
#
# Why we do this (guarantees)
# ---------------------------
# - Strictly-causal evaluation:
#   Predictions at time t only use information available up to t-1.
#
# - Sample preservation / metric integrity:
#   We avoid dropping the first L rows of Val/Test. Therefore:
#     len(y_val)  == number_of_rows_assigned_to_validation
#     len(y_test) == number_of_rows_assigned_to_test
#   Metrics reflect the full intended evaluation interval.
#
# - Clear interpretation (papers/review):
#   The protocol matches "calibrate offline on past → deploy on future" with a
#   hard y-boundary and X-only context, consistent with production causality.
#
# Implementation note (allowed vs forbidden)
# ------------------------------------------
# Allowed cross-boundary signals:
#   - raw past observations (history),
#   - deterministic transforms whose parameters were fit on Train only,
#   - used solely to construct X windows.
#
# Forbidden under this protocol:
#   - anything that uses Val/Test targets (directly or indirectly) to shape
#     features, parameters, or initial states.
# =============================================================================


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
from dataclasses import dataclass
import numpy as np


__all__ = [
    "_fit_arps_core_from_pinn",
    "_apply_arps_from_pinn",
    "_maybe_coupling_offline_analytic",
    "analytic_exponential_extrapolation_batch",
    "_inverse_2d_with_scaler",
    "_transform_2d_with_scaler",
    "PIPELINE_PRESETS",
    "describe_pipeline_presets",
    "build_pipeline_config_overrides",
]

# ============================================================
# Logger
# ============================================================

def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    return logger if logger is not None else logging.getLogger(__name__)



def extract_first_window(X_any: Any, *, logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    """
    Extract the first window as a 2D numpy array (L, F) from:
      - np.ndarray: (N, L, F) or (L, F)
      - tf.Tensor / tf.RaggedTensor: (N, L, F) or (L, F)
      - tf.data.Dataset / iterable: yields X or (X, y) or (X, y, w)
        - batched: (B, L, F) -> returns first item
        - unbatched: (L, F) -> returns directly

    Returns:
      np.ndarray of shape (L, F) or None if not possible.
    """
    log = _get_logger(logger)

    def _to_numpy(x: Any) -> np.ndarray:
        # Prefer TensorFlow's .numpy() when available, else fall back to np.asarray.
        try:
            if hasattr(x, "numpy"):
                return np.asarray(x.numpy())
        except Exception:
            pass
        return np.asarray(x)

    def _as_first_window_2d(a: np.ndarray) -> Optional[np.ndarray]:
        a = np.asarray(a)
        if a.ndim == 3:
            x0 = np.asarray(a[0])
            return x0 if x0.ndim == 2 else None
        if a.ndim == 2:
            return np.asarray(a)
        return None

    try:
        # 1) Direct array-like (numpy / tensor / ragged)
        if isinstance(X_any, np.ndarray):
            return _as_first_window_2d(X_any)

        # TensorFlow Tensor/RaggedTensor (only if TF is installed)
        try:
            import tensorflow as tf  # type: ignore
        except Exception:
            tf = None

        if tf is not None and isinstance(X_any, (tf.Tensor, tf.RaggedTensor)):
            return _as_first_window_2d(_to_numpy(X_any))

        # 2) Iterable / Dataset: pull the first element
        try:
            it = iter(X_any)
            first = next(it)
        except Exception:
            return None

        # Dataset may yield (X, y) or (X, y, w)
        if isinstance(first, (tuple, list)) and first:
            first = first[0]

        return _as_first_window_2d(_to_numpy(first))

    except Exception as ex:
        log.debug("extract_first_window failed: %s", str(ex))
        return None


def compute_history_q0_phys(
    *,
    X_any: Any,
    scaler_X=None,
    scaler_target=None,
    window: int = 30,
    kind: str = "median",
    min_q0_phys: float = 1e-6,
    channel: int = -1,
    x_in_scaler_space: Optional[bool] = None,

    # ---- adaptive detection ----
    detect_window: int = 80,
    spike_sigma_k: float = 3.0,          # spike if > p50 + K*sigma_robust
    spike_frac_lo: float = 0.10,         # <lo => "normal"
    spike_frac_hi: float = 0.30,         # >hi => "spiky"
    asym_lo: float = 1.5,                # <lo => "normal-ish"
    asym_hi: float = 3.0,                # >hi => "spiky-ish"

    # ---- smoothing candidates (kept as fallback/debug) ----
    median_win: int = 9,
    envelope_win: int = 15,
    envelope_q: float = 0.15,
    spiky_tail_q: Optional[float] = None,

    # ---- how to combine candidates (kept) ----
    spiky_mode: str = "blend",
    hard_spiky_alpha: float = 0.15,

    # ---- optional safety cap ----
    cap_hi_sigma: Optional[float] = None,

    logger: Optional[logging.Logger] = None,
    debug: bool = False,

    # ---- NEW: robust trend anchor knobs (defaults are “safe”) ----
    trend_window: Optional[int] = None,      # if None: uses detect_window (capped by series length)
    trend_iters: int = 5,                    # IRLS iterations
    clip_sigma_high: float = 1.2,            # aggressive only on positive residuals (1.2–2.0)
    clip_sigma_low: float = 8.0,             # very permissive on negative side
    weight_floor: float = 0.05,              # avoid degeneracy
    soft_k: float = 2.0,                     # softness of downweight above threshold
    use_log_space: bool = False,             # if True and y>0: fit in log-space
) -> Tuple[Optional[float], Dict[str, Any]]:

    log = logger or logging.getLogger(__name__)
    meta: Dict[str, Any] = {
        "ok": False,
        "reason": None,
        "L": None,
        "F": None,
        "channel_resolved": None,
        "x_in_scaler_space": None,
        "inverse_applied": False,
        "inverse_strategy": None,
        "inverse_error": None,
        "detect_window": int(detect_window),
        "spike_frac": None,
        "asym": None,
        "alpha": None,
        "q0_normal": None,
        "q0_spiky": None,
        "q0": None,
        "window_eff": None,
        "spiky_mode_used": None,
        "cap_hi": None,

        # NEW meta
        "q0_trend": None,
        "trend_ok": False,
        "trend_window_eff": None,
        "trend_iters_used": 0,
        "trend_points_eff": None,
        "trend_slope": None,
        "trend_intercept": None,
        "trend_sigma_low": None,
        "trend_log_space": None,
    }

    def fail(reason: str):
        meta["reason"] = reason
        return None, meta

    def norm_kind(k: Any) -> str:
        return (str(k).strip().lower() if k is not None else "median") or "median"

    def reduce_1d(v: np.ndarray, k: str) -> float:
        v = np.asarray(v, float).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        kk = norm_kind(k)
        if kk == "last":
            return float(v[-1])
        if kk == "mean":
            return float(np.mean(v))
        return float(np.median(v))

    def resolve_scaled(flag: Optional[bool]) -> bool:
        return bool(flag) if flag is not None else bool((scaler_X is not None) or (scaler_target is not None))

    def inverse_target(col1d: np.ndarray) -> np.ndarray:
        if scaler_target is None:
            raise ValueError("scaler_target is None (required to inverse last channel of X).")
        return scaler_target.inverse_transform(np.asarray(col1d, float).reshape(-1, 1)).reshape(-1)

    def inverse_feature(col1d: np.ndarray, *, col_idx: int, n_feats: int) -> np.ndarray:
        if scaler_X is None:
            raise ValueError("scaler_X is None (required to inverse feature channel).")
        x = np.asarray(col1d, float).reshape(-1)
        if x.size == 0:
            return x

        center = getattr(scaler_X, "center_", None)
        scale = getattr(scaler_X, "scale_", None)
        if center is not None and scale is not None:
            center = np.asarray(center, float).reshape(-1)
            scale = np.asarray(scale, float).reshape(-1)
            if center.size == n_feats and scale.size == n_feats and 0 <= col_idx < n_feats:
                return x * float(scale[col_idx]) + float(center[col_idx])

        Xtmp = np.zeros((x.size, n_feats), float)
        Xtmp[:, col_idx] = x
        inv = scaler_X.inverse_transform(Xtmp)
        return np.asarray(inv[:, col_idx], float)

    def rolling_stat(y: np.ndarray, win: int, *, mode: str, q: float = 0.5) -> np.ndarray:
        y = np.asarray(y, float).reshape(-1)
        n = int(y.size)
        if n == 0:
            return y.copy()
        win = int(win)
        if win < 3:
            return y.copy()
        if win % 2 == 0:
            win += 1
        k = win // 2

        out = np.empty_like(y)
        q = float(np.clip(q, 0.01, 0.99))
        for i in range(n):
            a, b = max(0, i - k), min(n, i + k + 1)
            w = y[a:b]
            w = w[np.isfinite(w)]
            if w.size == 0:
                out[i] = y[i]
            else:
                if mode == "median":
                    out[i] = float(np.median(w))
                else:
                    out[i] = float(np.quantile(w, q))
        return out

    def robust_sigma(v: np.ndarray) -> float:
        v = np.asarray(v, float).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size < 5:
            return float("nan")
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if np.isfinite(mad) and mad > 0:
            return 1.4826 * mad
        p25, p75 = np.percentile(v, [25, 75])
        iqr = float(p75 - p25)
        return (iqr / 1.349) if (np.isfinite(iqr) and iqr > 0) else float("nan")

    def blend_weight(spike_frac: float, asym: float) -> float:
        if not np.isfinite(spike_frac):
            a_f = 0.5
        else:
            den = max(float(spike_frac_hi - spike_frac_lo), 1e-12)
            a_f = (float(spike_frac_hi) - float(spike_frac)) / den
            a_f = float(np.clip(a_f, 0.0, 1.0))

        if not np.isfinite(asym):
            a_a = 0.5
        else:
            den = max(float(asym_hi - asym_lo), 1e-12)
            a_a = (float(asym_hi) - float(asym)) / den
            a_a = float(np.clip(a_a, 0.0, 1.0))

        return float(min(a_f, a_a))

    # ---- Robust asymmetric trend fit (key) ----
    def one_sided_sigma(resid: np.ndarray) -> float:
        r = np.asarray(resid, float).reshape(-1)
        r = r[np.isfinite(r)]
        if r.size < 6:
            return float("nan")
        med = float(np.median(r))
        low = r[r <= med]
        if low.size < 6:
            low = r
        med_low = float(np.median(low))
        mad = float(np.median(np.abs(low - med_low)))
        if np.isfinite(mad) and mad > 1e-12:
            return 1.4826 * mad
        # fallback: use lower spread (q50-q10)
        q10, q50 = np.percentile(low, [10, 50])
        s = float((q50 - q10) / 1.2816)  # normal approx
        return s if np.isfinite(s) and s > 1e-12 else float("nan")

    def wls_fit(t: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
        # Solve min sum w*(y-(m t + c))^2
        t = np.asarray(t, float).reshape(-1)
        y = np.asarray(y, float).reshape(-1)
        w = np.asarray(w, float).reshape(-1)
        w = np.clip(w, 0.0, None)
        sw = float(np.sum(w))
        if sw <= 0 or t.size < 2:
            return 0.0, float(np.median(y))
        wt = w * t
        wy = w * y
        s_tt = float(np.sum(wt * t))
        s_t  = float(np.sum(wt))
        s_y  = float(np.sum(wy))
        s_ty = float(np.sum(wt * y))
        det = s_tt * sw - s_t * s_t
        if abs(det) < 1e-18:
            # near-singular -> constant fit
            return 0.0, float(s_y / sw)
        m = (s_ty * sw - s_t * s_y) / det
        c = (s_tt * s_y - s_t * s_ty) / det
        return float(m), float(c)

    def robust_asym_trend_q0(y_tail: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        info = {"ok": False}
        y_tail = np.asarray(y_tail, float).reshape(-1)
        y_tail = y_tail[np.isfinite(y_tail)]
        n = int(y_tail.size)
        if n < 6:
            return float("nan"), info

        # local time axis normalized to [-1, 0]
        t = np.linspace(-1.0, 0.0, n, dtype=float)

        # optional log-space
        is_log = bool(use_log_space) and np.all(y_tail > 0)
        info["log_space"] = is_log
        y = np.log(y_tail + 1e-12) if is_log else y_tail.copy()

        # init weights
        w = np.ones(n, dtype=float)

        it_used = 0
        for it in range(max(1, int(trend_iters))):
            # fit
            m, c = wls_fit(t, y, w)
            y_hat = m * t + c
            resid = y - y_hat

            sig = one_sided_sigma(resid[w > 0])
            if not np.isfinite(sig) or sig <= 1e-12:
                sig = float(np.std(resid[w > 0])) if np.any(w > 0) else float("nan")
            if not np.isfinite(sig) or sig <= 1e-12:
                # can't scale -> stop
                it_used = it + 1
                info.update({"m": m, "c": c, "sigma_low": float("nan"), "iters_used": it_used})
                break

            rmed = float(np.median(resid[w > 0]))
            upper = rmed + float(clip_sigma_high) * sig
            lower = rmed - float(clip_sigma_low) * sig

            # soft downweight for positive outliers only
            w_new = np.ones(n, dtype=float)
            # extreme negatives: keep (or mildly downweight if beyond lower)
            neg_bad = resid < lower
            if np.any(neg_bad):
                w_new[neg_bad] = 1.0  # keep them (they define the "floor")

            pos_bad = resid > upper
            if np.any(pos_bad):
                z = (resid[pos_bad] - upper) / max(sig, 1e-12)
                # smooth decay: exp(-(z/soft_k)^2), with floor
                w_new[pos_bad] = np.exp(- (z / max(float(soft_k), 1e-6)) ** 2)

            # enforce weight floor to avoid singular fits
            wf = float(np.clip(weight_floor, 0.0, 0.49))
            w_new = np.clip(w_new, wf, 1.0)

            # stop if weights stabilized
            if np.max(np.abs(w_new - w)) < 1e-3:
                w = w_new
                it_used = it + 1
                info.update({"m": m, "c": c, "sigma_low": float(sig), "iters_used": it_used})
                break

            w = w_new
            it_used = it + 1
            info.update({"m": m, "c": c, "sigma_low": float(sig), "iters_used": it_used})

        # final fit with last weights
        m, c = wls_fit(t, y, w)
        y0 = c  # since t=0 -> intercept

        q0 = float(np.exp(y0)) if is_log else float(y0)

        info.update({
            "ok": np.isfinite(q0),
            "m": float(m),
            "c": float(c),
            "q0": float(q0) if np.isfinite(q0) else float("nan"),
            "w_eff": float(np.sum(w)),
            "n": int(n),
        })
        return q0, info

    # --- 1) Extract first window (same dependency as your code) ---
    x0 = extract_first_window(X_any, logger=log)
    if x0 is None:
        return fail("cannot_extract_first_window")
    x0 = np.asarray(x0, float)
    if x0.ndim != 2 or x0.size == 0:
        return fail("x0_not_2d_or_empty")
    L, F = map(int, x0.shape)
    meta["L"], meta["F"] = L, F

    # --- 2) Channel ---
    ch = int(channel)
    if ch < 0:
        ch = F + ch
    if not (0 <= ch < F):
        ch = F - 1
    meta["channel_resolved"] = int(ch)

    # --- 3) Scaler space ---
    xin = resolve_scaled(x_in_scaler_space)
    meta["x_in_scaler_space"] = bool(xin)

    # --- 4) Series + inverse to physical ---
    try:
        series = np.asarray(x0[:, ch], float).reshape(-1)
        series = series[np.isfinite(series)]
        if series.size == 0:
            return fail("empty_series")

        if xin:
            if ch == (F - 1):
                series = inverse_target(series)
                meta["inverse_strategy"] = "target_channel_with_scaler_target"
            else:
                n_feats = F - 1
                if n_feats <= 0:
                    raise ValueError("F-1 <= 0 (no feature channels to inverse).")
                n_in = getattr(scaler_X, "n_features_in_", n_feats)
                if n_in not in (None, n_feats):
                    raise ValueError(f"scaler_X.n_features_in_={n_in} incompatible with n_feats={n_feats}")
                series = inverse_feature(series, col_idx=int(ch), n_feats=int(n_feats))
                meta["inverse_strategy"] = "feature_channel_with_scaler_X"
            meta["inverse_applied"] = True

        series = np.asarray(series, float)
        series = series[np.isfinite(series)]
        if series.size == 0:
            return fail("empty_series_after_inverse")
    except Exception as ex:
        meta["inverse_error"] = f"{type(ex).__name__}: {ex}"
        return fail("inverse_failed")

    # --- 5) Detect regime on tail (kept) ---
    W = max(10, min(int(detect_window), int(series.size)))
    tail_det = np.asarray(series[-W:], float)
    tail_det = tail_det[np.isfinite(tail_det)]
    if tail_det.size < 5:
        return fail("too_short_for_detection")

    p10, p50, p90 = np.percentile(tail_det, [10, 50, 90])
    sig = robust_sigma(tail_det)

    if np.isfinite(sig) and sig > 0:
        thr = float(p50) + float(spike_sigma_k) * float(sig)
        spike_frac = float(np.mean(tail_det > thr))
    else:
        spike_frac = float(np.mean(tail_det > float(p90)))

    denom = max(float(p50 - p10), 1e-12)
    asym = float((p90 - p50) / denom)
    alpha = blend_weight(spike_frac, asym)

    meta["spike_frac"] = float(spike_frac)
    meta["asym"] = float(asym)
    meta["alpha"] = float(alpha)

    # --- 6) Old candidates (kept as fallback/debug) ---
    s_med = rolling_stat(series, win=int(median_win), mode="median")
    q_env = float(np.clip(float(envelope_q), 0.01, 0.49))
    s_env = rolling_stat(series, win=int(envelope_win), mode="quantile", q=q_env)

    w = max(1, int(window) if window is not None else 10)
    w_eff = max(1, min(w, int(min(s_med.size, s_env.size))))
    meta["window_eff"] = int(w_eff)

    q0_normal = float(reduce_1d(s_med[-w_eff:], kind))
    if spiky_tail_q is not None:
        tq = float(np.clip(float(spiky_tail_q), 0.01, 0.49))
        tail_env = np.asarray(s_env[-w_eff:], float)
        tail_env = tail_env[np.isfinite(tail_env)]
        q0_spiky = float(np.quantile(tail_env, tq)) if tail_env.size else float("nan")
    else:
        q0_spiky = float(reduce_1d(s_env[-w_eff:], kind))

    meta["q0_normal"] = float(q0_normal) if np.isfinite(q0_normal) else None
    meta["q0_spiky"] = float(q0_spiky) if np.isfinite(q0_spiky) else None

    # --- 7) NEW primary: robust asymmetric trend q0 ---
    tw = int(trend_window) if trend_window is not None else int(detect_window)
    tw = max(10, min(tw, int(series.size)))
    meta["trend_window_eff"] = int(tw)
    y_tail = np.asarray(series[-tw:], float)
    q0_trend, tinfo = robust_asym_trend_q0(y_tail)

    meta["trend_ok"] = bool(tinfo.get("ok", False) and np.isfinite(q0_trend))
    meta["q0_trend"] = float(q0_trend) if (np.isfinite(q0_trend)) else None
    meta["trend_iters_used"] = int(tinfo.get("iters_used", 0) or 0)
    meta["trend_points_eff"] = float(tinfo.get("w_eff", float("nan"))) if "w_eff" in tinfo else None
    meta["trend_slope"] = float(tinfo.get("m", float("nan"))) if "m" in tinfo else None
    meta["trend_intercept"] = float(tinfo.get("c", float("nan"))) if "c" in tinfo else None
    meta["trend_sigma_low"] = float(tinfo.get("sigma_low", float("nan"))) if "sigma_low" in tinfo else None
    meta["trend_log_space"] = bool(tinfo.get("log_space", False))

    # --- 8) Combine (trend-first, continuous; fallback to old logic if trend fails) ---
    # trend is the spiky-safe anchor; normal candidate keeps you accurate on clean series.
    if meta["trend_ok"] and np.isfinite(q0_normal):
        q0 = float(alpha) * float(q0_normal) + (1.0 - float(alpha)) * float(q0_trend)
        meta["spiky_mode_used"] = "trend_blend"
    elif meta["trend_ok"]:
        q0 = float(q0_trend)
        meta["spiky_mode_used"] = "trend_only"
    else:
        # fallback to your previous combine behavior
        mode = str(spiky_mode).strip().lower()
        if float(alpha) <= float(hard_spiky_alpha):
            mode = "envelope" if mode == "blend" else mode

        if mode == "envelope":
            q0 = q0_spiky if np.isfinite(q0_spiky) else q0_normal
        elif mode == "min":
            a = q0_normal if np.isfinite(q0_normal) else float("inf")
            b = q0_spiky if np.isfinite(q0_spiky) else float("inf")
            q0 = float(min(a, b))
            if not np.isfinite(q0) or q0 == float("inf"):
                q0 = q0_spiky if np.isfinite(q0_spiky) else q0_normal
        else:
            if not np.isfinite(q0_normal):
                q0 = q0_spiky
            elif not np.isfinite(q0_spiky):
                q0 = q0_normal
            else:
                q0 = float(alpha) * float(q0_normal) + (1.0 - float(alpha)) * float(q0_spiky)

        meta["spiky_mode_used"] = f"fallback_{mode}"

    # --- 9) Optional safety cap (kept; caps above low-freq tail) ---
    if cap_hi_sigma is not None and np.isfinite(cap_hi_sigma):
        if np.isfinite(sig) and sig > 0:
            cap_hi = float(p50) + float(cap_hi_sigma) * float(sig)
            meta["cap_hi"] = float(cap_hi)
            if np.isfinite(q0):
                q0 = float(min(q0, cap_hi))

    meta["q0"] = float(q0) if np.isfinite(q0) else None

    # --- 10) Validate ---
    min_q0 = max(0.0, float(min_q0_phys) if np.isfinite(min_q0_phys) else 1e-6)
    if (not np.isfinite(q0)) or (q0 <= min_q0):
        meta["reason"] = "q0_invalid_or_tiny"
        return None, meta

    meta["ok"] = True
    meta["reason"] = "ok"

    if debug:
        log.info(
            "history_q0_ok q0=%.6g mode=%s alpha=%.2f spike_frac=%.2f asym=%.2f "
            "q0_normal=%s q0_spiky=%s q0_trend=%s cap_hi=%s log=%s",
            float(q0), str(meta.get("spiky_mode_used")), float(alpha), float(spike_frac), float(asym),
            str(meta.get("q0_normal")), str(meta.get("q0_spiky")), str(meta.get("q0_trend")),
            str(meta.get("cap_hi")), str(meta.get("trend_log_space")),
        )

    return float(q0), meta




def _inverse_2d_with_scaler(y_scaled: np.ndarray, scaler) -> np.ndarray:
    """
    Invert scaling for 2D ribbons (N, H) regardless of how the scaler was fitted.

    Cases:
      - scaler.n_features_in_ == H:  scaler fitted per timestep -> direct inverse_transform.
      - scaler.n_features_in_ == 1:  scaler fitted on flattened targets -> flatten, inverse, reshape.
    """
    y_scaled = np.asarray(y_scaled, dtype=float)
    if y_scaled.ndim != 2:
        raise ValueError(f"_inverse_2d_with_scaler expects 2D array, got shape={y_scaled.shape}")

    if scaler is None:
        return y_scaled.copy()

    n_features = getattr(scaler, "n_features_in_", None)
    if n_features is None:
        n_features = y_scaled.shape[1]

    if int(n_features) == int(y_scaled.shape[1]):
        return scaler.inverse_transform(y_scaled)

    if int(n_features) == 1:
        flat = y_scaled.reshape(-1, 1)
        inv_flat = scaler.inverse_transform(flat)
        return inv_flat.reshape(y_scaled.shape)

    raise ValueError(
        f"_inverse_2d_with_scaler: incompatible shapes. "
        f"y_scaled has {y_scaled.shape[1]} cols, scaler has n_features_in_={n_features}."
    )


def _transform_2d_with_scaler(y_phys: np.ndarray, scaler) -> np.ndarray:
    """
    Apply scaling for 2D ribbons (N, H) regardless of how the scaler was fitted.
    Mirrors _inverse_2d_with_scaler logic.
    """
    y_phys = np.asarray(y_phys, dtype=float)
    if y_phys.ndim != 2:
        raise ValueError(f"_transform_2d_with_scaler expects 2D array, got shape={y_phys.shape}")

    if scaler is None:
        return y_phys.copy()

    n_features = getattr(scaler, "n_features_in_", None)
    if n_features is None:
        n_features = y_phys.shape[1]

    if int(n_features) == int(y_phys.shape[1]):
        return scaler.transform(y_phys)

    if int(n_features) == 1:
        flat = y_phys.reshape(-1, 1)
        tr_flat = scaler.transform(flat)
        return tr_flat.reshape(y_phys.shape)

    raise ValueError(
        f"_transform_2d_with_scaler: incompatible shapes. "
        f"y_phys has {y_phys.shape[1]} cols, scaler has n_features_in_={n_features}."
    )



# ============================================================
# Internal config + normalization
# ============================================================
FIT_WINDOW = 80
@dataclass(frozen=True)
class OfflineAnalyticCfg:
    fit_window: int = FIT_WINDOW
    fit_region: str = "head"          # "head" | "tail"
    anchor_kind: str = "median"       # "last" | "mean" | "median"
    anchor_window: int = 10

    force_monotonic: bool = True
    max_decline_per_step: Optional[float] = 0.15

    # multi-window policies (default keeps current behavior)
    b_fit_policy: str = "first"       # "first"|"last"|"all"|"first_k"|"last_k"
    q0_anchor_policy: str = "first"   # same domain as above
    b_fit_k: Optional[int] = None
    q0_anchor_k: Optional[int] = None

    b_reducer: str = "median"         # "median"|"mean"|"trimmed_mean"
    q0_reducer: str = "median"        # "median"|"mean"

    # coupling extras
    coupling_spaghetti: bool = False
    coupling_spaghetti_k: Optional[int] = None
    coupling_spaghetti_reducer: str = "median"  # "mean"|"median"


def _norm_str(x: Any, default: str) -> str:
    s = str(x).strip().lower() if x is not None else default
    return s or default


def _normalize_cfg(
    *,
    fit_window: int,
    force_monotonic: bool,
    max_decline_per_step: Optional[float],
    fit_region: str,
    anchor_kind: str,
    anchor_window: int,
    b_fit_policy: Optional[str] = None,
    q0_anchor_policy: Optional[str] = None,
    b_fit_k: Optional[int] = None,
    q0_anchor_k: Optional[int] = None,
    b_reducer: Optional[str] = None,
    q0_reducer: Optional[str] = None,
    coupling_spaghetti: Optional[bool] = None,
    coupling_spaghetti_k: Optional[int] = None,
    coupling_spaghetti_reducer: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> OfflineAnalyticCfg:
    log = _get_logger(logger)

    def _choice(name: str, value: Any, allowed: set, default: str) -> str:
        v = _norm_str(value, default)
        if v not in allowed:
            log.warning("arps_offline: unknown %s=%r; using %r.", name, value, default)
            return default
        return v

    region = _choice("fit_region", fit_region, {"head", "tail"}, "head")
    ak = _choice("anchor_kind", anchor_kind, {"last", "mean", "median"}, "median")

    fw = max(5, int(fit_window) if fit_window is not None else FIT_WINDOW)
    aw = max(1, int(anchor_window) if anchor_window is not None else 10)

    valid_pol = {"first", "last", "all", "first_k", "last_k"}
    bpol = _choice("b_fit_policy", b_fit_policy, valid_pol, "first")
    qpol = _choice("q0_anchor_policy", q0_anchor_policy, valid_pol, "first")

    bred = _choice("b_reducer", b_reducer, {"mean", "median", "trimmed_mean"}, "median")
    qred = _choice("q0_reducer", q0_reducer, {"mean", "median"}, "median")

    cs = bool(coupling_spaghetti) if coupling_spaghetti is not None else False
    csr = _choice("coupling_spaghetti_reducer", coupling_spaghetti_reducer, {"mean", "median"}, "median")

    return OfflineAnalyticCfg(
        fit_window=fw,
        fit_region=region,
        anchor_kind=ak,
        anchor_window=aw,
        force_monotonic=bool(force_monotonic),
        max_decline_per_step=float(max_decline_per_step) if max_decline_per_step is not None else None,
        b_fit_policy=bpol,
        q0_anchor_policy=qpol,
        b_fit_k=b_fit_k,
        q0_anchor_k=q0_anchor_k,
        b_reducer=bred,
        q0_reducer=qred,
        coupling_spaghetti=cs,
        coupling_spaghetti_k=coupling_spaghetti_k,
        coupling_spaghetti_reducer=csr,
    )



def _reduce(values: np.ndarray, kind: str) -> float:
    kind = _norm_str(kind, "median")
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")

    if kind == "mean":
        return float(np.mean(v))
    if kind == "trimmed_mean":
        if v.size <= 2:
            return float(np.mean(v))
        vs = np.sort(v)
        trim = max(1, int(0.1 * vs.size))
        core = vs[trim:-trim] if vs.size > 2 * trim else vs
        return float(np.mean(core))
    return float(np.median(v))


# ============================================================
# Math core
# ============================================================

def _guard_b(
    b: float,
    *,
    force_monotonic: bool,
    max_decline_per_step: Optional[float],
) -> float:
    if not np.isfinite(b):
        b = 0.0

    if force_monotonic and b > 0:
        b = -abs(b)

    if max_decline_per_step is not None and max_decline_per_step > 0:
        b_min = float(np.log(max(1.0 - float(max_decline_per_step), 1e-6)))
        if b < b_min:
            b = b_min

    return float(b)


def _enforce_monotone_nonincreasing(x: np.ndarray) -> np.ndarray:
    # correct: cumulative minimum (never increases)
    x = np.asarray(x, dtype=float)
    return np.minimum.accumulate(x)


def _select_fit_region(series_phys: np.ndarray, fit_region: str, fit_window: int) -> np.ndarray:
    s = np.asarray(series_phys, dtype=float)
    fit_window = max(2, min(int(fit_window), s.size))
    return s[:fit_window] if fit_region == "head" else s[-fit_window:][::-1]


def _fit_b_single_window(
    series_phys: np.ndarray,
    *,
    fit_region: str,
    fit_window: int,
    anchor_kind: str,
    anchor_window: int,
) -> Tuple[float, float, bool]:
    """
    Fit b on region using OLS-through-origin on log(region/q0) ~ b*t.
    Returns (b, q0_used_for_fit, used_fallback).
    """
    eps = 1e-8
    s = np.asarray(series_phys, dtype=float)
    s = np.where(np.isfinite(s), s, 0.0)
    s = np.maximum(s, eps)

    region = _select_fit_region(s, fit_region, fit_window)
    region = np.maximum(region, eps)
    M = int(region.size)
    if M < 2:
        return 0.0, float(max(s[-1], eps)), True

    m = max(1, min(int(anchor_window), M))
    ak = _norm_str(anchor_kind, "median")
    if ak == "mean":
        q0 = float(np.mean(region[:m]))
    elif ak == "median":
        q0 = float(np.median(region[:m]))
    else:
        q0 = float(region[0])

    if (not np.isfinite(q0)) or q0 <= eps:
        return 0.0, float(max(s[-1], eps)), True

    rel = np.maximum(region / q0, eps)
    log_rel = np.log(rel)

    t = np.arange(M, dtype=float)
    denom = float(np.sum(t * t))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0, q0, True

    num = float(np.sum(t * log_rel))
    if not np.isfinite(num):
        return 0.0, q0, True

    return float(num / denom), q0, False


def _build_arps_curve_phys(
    *,
    q0: float,
    b: float,
    H_target: int,
    force_monotonic: bool,
) -> np.ndarray:
    t = np.arange(int(H_target), dtype=float)
    q = float(q0) * np.exp(float(b) * t)
    q = np.where(np.isfinite(q), q, float(q0))
    q = np.maximum(q, 0.0)
    if force_monotonic:
        q = _enforce_monotone_nonincreasing(q)
    return q


# ============================================================
# Public: analytic_exponential_extrapolation_batch (compat)
# ============================================================

def analytic_exponential_extrapolation_batch(
    y_pred_scaled: np.ndarray,
    scaler_y,
    *,
    target_length: int,
    fit_window: int = FIT_WINDOW,
    force_monotonic: bool = True,
    max_decline_per_step: Optional[float] = None,
    fit_region: str = "tail",
    anchor_kind: str = "last",
    anchor_window: int = 15,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    log = _get_logger(logger)

    y_pred_scaled = np.asarray(y_pred_scaled, dtype=float)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(1, -1)
    if y_pred_scaled.ndim != 2:
        raise ValueError(f"analytic_exponential_extrapolation_batch expects 2D input, got shape={y_pred_scaled.shape}")

    N, H_train = y_pred_scaled.shape
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError(f"target_length must be > 0, got {target_length}")

    cfg = _normalize_cfg(
        fit_window=fit_window,
        force_monotonic=force_monotonic,
        max_decline_per_step=max_decline_per_step,
        fit_region=fit_region,
        anchor_kind=anchor_kind,
        anchor_window=anchor_window,
        logger=log,
    )

    try:
        y_phys = _inverse_2d_with_scaler(y_pred_scaled, scaler_y)
    except Exception as ex:
        log.exception("analytic_exponential_extrapolation_batch inverse failed (%s); using scaled as phys.", str(ex))
        y_phys = y_pred_scaled.copy()

    b_all = np.zeros(N, dtype=float)
    fallback_count = 0
    out_phys = np.zeros((N, target_length), dtype=float)

    for i in range(N):
        b_i, q0_fit, fb = _fit_b_single_window(
            y_phys[i],
            fit_region=cfg.fit_region,
            fit_window=min(cfg.fit_window, H_train),
            anchor_kind=cfg.anchor_kind,
            anchor_window=cfg.anchor_window,
        )
        b_i = _guard_b(b_i, force_monotonic=cfg.force_monotonic, max_decline_per_step=cfg.max_decline_per_step)
        if fb:
            fallback_count += 1

        b_all[i] = b_i
        out_phys[i, :] = _build_arps_curve_phys(
            q0=q0_fit, b=b_i, H_target=target_length, force_monotonic=cfg.force_monotonic
        )

    try:
        out_scaled = _transform_2d_with_scaler(out_phys, scaler_y)
    except Exception as ex:
        log.exception("analytic_exponential_extrapolation_batch transform failed (%s); returning phys.", str(ex))
        out_scaled = out_phys.copy()

    debug = {
        "H_train": int(H_train),
        "target_length": int(target_length),
        "fit_window": int(min(cfg.fit_window, H_train)),
        "fit_region": cfg.fit_region,
        "anchor_kind": cfg.anchor_kind,
        "anchor_window": int(cfg.anchor_window),
        "b_all": b_all,
        "b_mean": float(np.nanmean(b_all)) if b_all.size else float("nan"),
        "b_std": float(np.nanstd(b_all)) if b_all.size else float("nan"),
        "fallback_count": int(fallback_count),
    }
    return out_scaled, debug


# ============================================================
# Public: _fit_arps_core_from_pinn (compat + policies)
# ============================================================

def _fit_arps_core_from_pinn(
    *,
    split_name: str,
    base_split: str,
    y_pinn_scaled: np.ndarray,
    scaler_y,
    H_target: int,
    fit_window: int = FIT_WINDOW,
    force_monotonic: bool = True,
    max_decline_per_step: Optional[float] = 0.15,
    fit_region: str = "head",
    anchor_kind: str = "median",
    anchor_window: int = 10,
    logger=None,
    # selection/reduction knobs (4.1/4.2)
    b_fit_policy: Optional[str] = None,     # "all" | "first_k"
    q0_anchor_policy: Optional[str] = None, # kept for backward compat (ignored now)
    b_fit_k: Optional[int] = None,
    q0_anchor_k: Optional[int] = None,      # kept for backward compat (ignored now)
    b_reducer: Optional[str] = None,        # "median" | "mean"
    q0_reducer: Optional[str] = None,       # kept for backward compat (ignored now)
    # outlier filtering (default off)
    b_outlier_filter: Optional[str] = None,     # "none"|"trim"|"mad"|"iqr"
    b_outlier_trim_pct: Optional[float] = None, # e.g., 0.10
    b_outlier_mad_z: Optional[float] = None,    # e.g., 3.5
    b_outlier_iqr_k: Optional[float] = None,    # e.g., 1.5
    b_outlier_min_keep: Optional[int] = None,   # e.g., 5
    # Scenario 2: REQUIRED real/history anchor in PHYSICAL units
    q0_override_phys: Optional[float] = None,
    # NEW: minimum allowed q0 (physical)
    min_q0_phys: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    ARPS fit on PINN ribbons (scaled). STRICT: q0 must come from q0_override_phys (physical).
    If missing/invalid -> raises ValueError("missing_history_anchor ...").
    """
    log = _get_logger(logger)

    # -------------------------
    # Small helpers
    # -------------------------
    def norm(x: Any, default: str) -> str:
        return (str(x).strip().lower() if x is not None else default) or default

    def to_int(x: Any, default: Optional[int]) -> Optional[int]:
        try:
            return default if x is None else int(x)
        except Exception:
            return default

    def to_float(x: Any, default: Optional[float]) -> Optional[float]:
        try:
            return default if x is None else float(x)
        except Exception:
            return default

    def reduce_1d(v: np.ndarray, reducer: str) -> float:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        r = norm(reducer, "median")
        return float(np.mean(v)) if r in ("mean", "avg", "average") else float(np.median(v))

    def filter_outliers(
        b: np.ndarray,
        *,
        method: str,
        trim_pct: float,
        mad_z: float,
        iqr_k: float,
        min_keep: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        b = np.asarray(b, dtype=float)
        b = b[np.isfinite(b)]

        m = norm(method, "none")
        meta: Dict[str, Any] = {
            "method": m,
            "n_raw": int(b.size),
            "n_clean": int(b.size),
            "dropped": 0,
            "used_fallback": False,
        }
        if b.size == 0 or m in ("none", "off", "false", "0"):
            return b, meta

        min_keep = max(1, int(min_keep))

        if m == "trim":
            p = float(trim_pct)
            p = 0.0 if not np.isfinite(p) else max(0.0, min(0.49, p))
            if b.size < 3 or p <= 0.0:
                return b, meta
            bs = np.sort(b)
            k = int(np.floor(p * bs.size))
            core = bs[k:-k] if bs.size > 2 * k else bs
            if core.size < min_keep:
                meta["used_fallback"] = True
                return b, meta
            meta["n_clean"] = int(core.size)
            meta["dropped"] = int(bs.size - core.size)
            return core, meta

        if m == "mad":
            z = float(mad_z)
            z = 3.5 if (not np.isfinite(z) or z <= 0.0) else z
            med = float(np.median(b))
            mad = float(np.median(np.abs(b - med)))
            if mad <= 0.0 or not np.isfinite(mad):
                return b, meta
            scale = 1.4826 * mad
            score = np.abs(b - med) / max(scale, 1e-12)
            core = b[score <= z]
            if core.size < min_keep:
                meta["used_fallback"] = True
                return b, meta
            meta["n_clean"] = int(core.size)
            meta["dropped"] = int(b.size - core.size)
            return core, meta

        if m == "iqr":
            k = float(iqr_k)
            k = 1.5 if (not np.isfinite(k) or k <= 0.0) else k
            q1 = float(np.percentile(b, 25))
            q3 = float(np.percentile(b, 75))
            iqr = q3 - q1
            if iqr <= 0.0 or not np.isfinite(iqr):
                return b, meta
            lo, hi = q1 - k * iqr, q3 + k * iqr
            core = b[(b >= lo) & (b <= hi)]
            if core.size < min_keep:
                meta["used_fallback"] = True
                return b, meta
            meta["n_clean"] = int(core.size)
            meta["dropped"] = int(b.size - core.size)
            return core, meta

        meta["method"] = "none"
        return b, meta

    # -------------------------
    # Inputs
    # -------------------------
    y = np.asarray(y_pinn_scaled, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.size == 0:
        raise ValueError(f"_fit_arps_core_from_pinn split={split_name}: expected 2D non-empty, got shape={y.shape}")

    N, H_train = map(int, y.shape)
    Ht = int(H_target)
    if Ht <= 0:
        raise ValueError(f"_fit_arps_core_from_pinn split={split_name}: H_target<=0")

    fit_window_eff = max(1, min(int(fit_window), H_train))
    anchor_window_eff = max(1, int(anchor_window))

    # b policies (q0 policies kept only for compat metadata)
    b_fit_policy_eff = norm(b_fit_policy, "all")
    b_fit_k_eff = to_int(b_fit_k, N)
    b_fit_k_eff = max(1, min(int(b_fit_k_eff), N))
    b_reducer_eff = norm(b_reducer, "median")

    # -------------------------
    # STRICT q0 anchor (physical)
    # -------------------------
    q0_used = to_float(q0_override_phys, float("nan"))
    min_q0 = to_float(min_q0_phys, 1e-8)
    min_q0 = 1e-8 if (min_q0 is None or not np.isfinite(min_q0)) else float(min_q0)
    min_q0 = max(0.0, float(min_q0))

    if (q0_used is None) or (not np.isfinite(q0_used)) or (q0_used <= min_q0):
        raise ValueError(
            f"_fit_arps_core_from_pinn split={split_name}: missing_history_anchor "
            f"(q0_override_phys={q0_override_phys!r}, min_q0_phys={min_q0})"
        )

    q0_star = float(max(float(q0_used), max(min_q0, 1e-12)))
    q0_source = "override_phys"

    # -------------------------
    # Inverse to physical for fitting b
    # -------------------------
    try:
        y_phys = _inverse_2d_with_scaler(y, scaler_y)
    except Exception as ex:
        log.exception(
            "_fit_arps_core_from_pinn split=%s inverse failed (%s); using scaled as phys.",
            split_name, str(ex),
        )
        y_phys = y.copy()

    # -------------------------
    # Choose members to fit b
    # -------------------------
    use_first_k = b_fit_policy_eff in ("first_k", "first", "k_first")
    idx_use = range(b_fit_k_eff) if use_first_k else range(N)

    # -------------------------
    # Fit b candidates
    # -------------------------
    b_list: List[float] = []
    fallback_count = 0

    for idx in idx_use:
        b_i, _q0fit_i, fb = _fit_b_single_window(
            y_phys[int(idx)],
            fit_region=fit_region,
            fit_window=fit_window_eff,
            anchor_kind=anchor_kind,
            anchor_window=anchor_window_eff,
        )
        b_i = _guard_b(
            float(b_i),
            force_monotonic=bool(force_monotonic),
            max_decline_per_step=max_decline_per_step,
        )
        b_list.append(float(b_i))
        fallback_count += int(bool(fb))

    if not b_list:
        b_list = [0.0]

    b_raw = np.asarray(b_list, dtype=float)

    # -------------------------
    # Optional outlier removal
    # -------------------------
    method = norm(b_outlier_filter, "none")
    trim_pct = float(to_float(b_outlier_trim_pct, 0.10) or 0.10)
    mad_z = float(to_float(b_outlier_mad_z, 3.5) or 3.5)
    iqr_k = float(to_float(b_outlier_iqr_k, 1.5) or 1.5)
    min_keep = int(to_int(b_outlier_min_keep, 3) or 3)

    b_clean, out_meta = filter_outliers(
        b_raw,
        method=method,
        trim_pct=trim_pct,
        mad_z=mad_z,
        iqr_k=iqr_k,
        min_keep=min_keep,
    )

    b_for_reduce = b_clean if b_clean.size else b_raw
    b_star = float(reduce_1d(b_for_reduce, b_reducer_eff))
    b_star = _guard_b(
        float(b_star),
        force_monotonic=bool(force_monotonic),
        max_decline_per_step=max_decline_per_step,
    )

    b_std_raw = float(np.nanstd(b_raw)) if b_raw.size else float("nan")
    b_std = float(np.nanstd(b_for_reduce)) if b_for_reduce.size else float("nan")

    # -------------------------
    # Build analytic curve (phys) then scale back
    # -------------------------
    q_phys = _build_arps_curve_phys(
        q0=float(q0_star),
        b=float(b_star),
        H_target=Ht,
        force_monotonic=bool(force_monotonic),
    ).reshape(1, -1)

    try:
        y_analytic_scaled = _transform_2d_with_scaler(q_phys, scaler_y)
    except Exception as ex:
        log.exception(
            "_fit_arps_core_from_pinn split=%s transform failed (%s); returning phys.",
            split_name, str(ex),
        )
        y_analytic_scaled = q_phys

    # -------------------------
    # Params/meta (compat keys preserved)
    # -------------------------
    params: Dict[str, Any] = {
        "split": str(base_split),
        "H_train": int(H_train),
        "H_target": int(Ht),
        "fit_region": str(fit_region),
        "anchor_kind": str(anchor_kind),
        "anchor_window": int(anchor_window_eff),
        "fit_window": int(fit_window_eff),
        "force_monotonic": bool(force_monotonic),
        "max_decline_per_step": (float(max_decline_per_step) if max_decline_per_step is not None else None),

        "b": float(b_star),
        "b_std": float(b_std),
        "b_std_raw": float(b_std_raw),
        "b_candidates": [float(x) for x in b_raw.tolist()] if b_raw.size else [],
        "b_fit_policy": str(b_fit_policy_eff),
        "b_fit_k": int(b_fit_k_eff),
        "b_reducer": str(b_reducer_eff),
        "b_outlier": dict(out_meta or {}),
        "fallback_count": int(fallback_count),

        "q0": float(q0_star),
        "q0_source": str(q0_source),
        "q0_override_phys": float(q0_used),

        # explicitly ignored knobs (compat)
        "q0_anchor_policy_ignored": str(norm(q0_anchor_policy, "first_k")),
        "q0_anchor_k_ignored": int(to_int(q0_anchor_k, 1) or 1),
        "q0_reducer_ignored": str(norm(q0_reducer, "median")),
        "min_q0_phys": float(min_q0),
    }

    log.info(
        "offline_analytic_fit split=%s base_split=%s "
        "q0_source=%s q0=%.6g min_q0=%.6g "
        "fit_region=%s fit_window=%d anchor_kind=%s anchor_window=%d "
        "b_fit_policy=%s b_fit_k=%d b_reducer=%s "
        "b=%.6f b_std=%.6f fallback_count=%d outlier_method=%s",
        str(split_name),
        str(base_split),
        str(q0_source),
        float(q0_star),
        float(min_q0),
        str(fit_region),
        int(fit_window_eff),
        str(anchor_kind),
        int(anchor_window_eff),
        str(b_fit_policy_eff),
        int(b_fit_k_eff),
        str(b_reducer_eff),
        float(b_star),
        float(b_std),
        int(fallback_count),
        str((out_meta or {}).get("method", "none")),
    )

    return np.asarray(y_analytic_scaled, dtype=float), params

# ============================================================
# Public: _apply_arps_from_pinn (compat)
# ============================================================

def _apply_arps_from_pinn(
    y_pinn_scaled: np.ndarray,
    scaler_y,
    *,
    H_target: int,
    b: float,
    anchor_kind: str,          # kept for backward-compat (unused)
    anchor_window: int,        # kept for backward-compat (unused)
    force_monotonic: bool,
    max_decline_per_step: Optional[float],
    logger,
    split_label: str,
    q0_override: Optional[float] = None,
    q0_override_phys: Optional[float] = None,
    require_q0_override: bool = True,
    min_q0_phys: float = 1e-6,
    # NEW: collector para não spammar logs em spaghetti
    rebuild_log_ctx: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    import numpy as np

    log = _get_logger(logger)

    H_target = int(H_target)
    if H_target <= 0:
        raise ValueError(f"_apply_arps_from_pinn split={split_label}: H_target<=0")

    # Resolve q0 (physical)
    q0_phys = None
    q0_source = "none"
    try:
        if q0_override_phys is not None and np.isfinite(float(q0_override_phys)):
            q0_phys = float(q0_override_phys)
            q0_source = "override_phys"
        elif q0_override is not None and np.isfinite(float(q0_override)):
            q0_phys = float(q0_override)
            q0_source = "override_legacy"
    except Exception:
        q0_phys = None
        q0_source = "none"

    try:
        min_q0 = float(min_q0_phys)
    except Exception:
        min_q0 = 1e-6
    min_q0 = max(0.0, float(min_q0))

    if q0_phys is None or (not np.isfinite(q0_phys)) or (q0_phys <= min_q0):
        msg = (
            f"_apply_arps_from_pinn split={split_label}: missing/invalid history anchor "
            f"(q0={q0_phys}, min_q0={min_q0}). Refusing to anchor from PINN."
        )
        if require_q0_override:
            raise ValueError(msg)
        log.warning(msg)
        return np.full((1, H_target), np.nan, dtype=float)

    # Guard b
    b_used = _guard_b(
        float(b),
        force_monotonic=bool(force_monotonic),
        max_decline_per_step=max_decline_per_step,
    )

    # Build curve (phys)
    q_phys = _build_arps_curve_phys(
        q0=float(q0_phys),
        b=float(b_used),
        H_target=int(H_target),
        force_monotonic=bool(force_monotonic),
    ).reshape(1, -1)

    # NEW: se estamos em loop (spaghetti), agrega e NÃO loga 150 linhas
    if isinstance(rebuild_log_ctx, dict):
        rebuild_log_ctx.setdefault("split", str(split_label))
        rebuild_log_ctx.setdefault("q0", float(q0_phys))
        rebuild_log_ctx.setdefault("q0_source", str(q0_source))
        rebuild_log_ctx.setdefault("len", int(H_target))
        rebuild_log_ctx.setdefault("n_calls", 0)
        rebuild_log_ctx.setdefault("b_used", [])
        rebuild_log_ctx["n_calls"] += 1
        rebuild_log_ctx["b_used"].append(float(b_used))
    else:
        # caso “single curve”, ok manter 1 linha
        log.info(
            "arps_rebuild split=%s q0=%.6g q0_source=%s b_used=%.6f len=%d",
            str(split_label),
            float(q0_phys),
            str(q0_source),
            float(b_used),
            int(H_target),
        )

    # Scale back
    try:
        return _transform_2d_with_scaler(q_phys, scaler_y)
    except Exception as ex:
        log.exception(
            "_apply_arps_from_pinn split=%s transform failed (%s); returning phys.",
            split_label, str(ex)
        )
        return q_phys




def _maybe_coupling_offline_analytic(
    *,
    outs_val: dict,
    outs_test: dict,
    scaler_y,
    split_recon_lengths: dict,
    latent_cfg: dict,
    logger,
):
    """
    Couple VAL/TEST in offline_analytic mode by sharing ARPS geometry parameter b.

    - coupling_mode="val_only": apply VAL b onto TEST (VAL stays base)
      - optional spaghetti members + AUC trimming + physical aggregation
    - coupling_mode!="val_only": val_plus_test -> weighted b_final applied to both

    Backward-compatible refactor:
      - Adds optional "theta sampling" for spaghetti:
          cfg["coupling_theta_sampling"] = True
        to sample b around b_val (theta-hat) instead of (or fallback to) b_candidates.
    """
    import numpy as np
    from typing import Any, Dict, Optional, Tuple, List

    cfg = latent_cfg or {}
    split_recon_lengths = split_recon_lengths or {}

    # -----------------------
    # Tiny utilities
    # -----------------------
    def _mode(c: Dict[str, Any]) -> str:
        return str(c.get("mode", c.get("latent_mode", "off"))).strip().lower()

    def _as_1d(a: Any) -> Optional[np.ndarray]:
        if a is None:
            return None
        x = np.asarray(a, dtype=float)
        return None if x.size == 0 else x.reshape(-1)

    def _as_2d_row(a: Any) -> Optional[np.ndarray]:
        if a is None:
            return None
        x = np.asarray(a, dtype=float)
        if x.size == 0:
            return None
        if x.ndim == 1:
            return x.reshape(1, -1)
        if x.ndim == 2:
            return x
        return x.reshape(1, -1)

    def _f(x: Any, default=np.nan) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _i(x: Any, default=0) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def _min_q0(c: Dict[str, Any]) -> float:
        v = c.get("history_anchor_min_q0_phys", c.get("q0_min_phys", 1e-6))
        v = _f(v, 1e-6)
        return max(0.0, float(v)) if np.isfinite(v) else 1e-6

    def _resolve_q0_phys(outs: Dict[str, Any], split: str, min_q0_phys: float) -> Tuple[Optional[float], str]:
        for path, getter in (
            ("outs.q0_anchor_phys", lambda o: o.get("q0_anchor_phys", None)),
            ("outs.anchor.q0_phys", lambda o: (o.get("anchor") or {}).get("q0_phys", None) if isinstance(o.get("anchor"), dict) else None),
        ):
            try:
                v = getter(outs)
                if v is None:
                    continue
                q0 = float(v)
                if np.isfinite(q0) and q0 > float(min_q0_phys):
                    return float(q0), path
            except Exception:
                pass

        logger.info("arps_anchor_resolve split=%s status=missing_history_anchor min_q0=%s", str(split), str(min_q0_phys))
        return None, "none"

    def _reduce_members_phys(values_phys_2d: np.ndarray, reducer: str) -> np.ndarray:
        r = str(reducer).strip().lower()
        if r in ("median", "p50"):
            return np.nanmedian(values_phys_2d, axis=0)
        if r in ("mean", "avg", "average"):
            return np.nanmean(values_phys_2d, axis=0)
        raise ValueError(f"Unknown reducer '{reducer}' (use 'median' or 'mean').")

    def _auc_phys(members_scaled_2d: np.ndarray) -> np.ndarray:
        m = np.asarray(members_scaled_2d, dtype=float)
        if m.ndim != 2 or m.size == 0:
            return np.asarray([], dtype=float)
        m_phys = _inverse_2d_with_scaler(m, scaler_y)
        return np.nansum(np.asarray(m_phys, dtype=float), axis=1)

    def _trim_by_auc(members_scaled_2d: np.ndarray, trim_pct: float, min_keep: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        m = np.asarray(members_scaled_2d, dtype=float)
        K = int(m.shape[0]) if m.ndim == 2 else 0
        min_keep = max(1, int(min_keep))
        trim_pct = float(trim_pct)
        trim_pct = 0.0 if (not np.isfinite(trim_pct)) else max(0.0, min(0.49, trim_pct))

        if K <= 0:
            return m, {"filter": "auc_trim", "K_in": 0, "K_out": 0, "reason": "empty"}

        auc = _auc_phys(m)
        if auc.size != K or (not np.any(np.isfinite(auc))):
            return m, {"filter": "auc_trim", "K_in": K, "K_out": K, "trim_pct": trim_pct, "min_keep": min_keep, "reason": "no_finite_auc"}

        if K <= min_keep or trim_pct <= 0.0:
            return m, {"filter": "auc_trim", "K_in": K, "K_out": K, "trim_pct": trim_pct, "min_keep": min_keep, "reason": "no_trim"}

        idx = np.argsort(auc, kind="mergesort")
        k = int(np.floor(trim_pct * K))
        lo, hi = k, K - k

        if (hi - lo) < min_keep:
            extra = max(0, K - min_keep)
            drop_lo = extra // 2
            drop_hi = extra - drop_lo
            lo, hi = drop_lo, K - drop_hi

        lo = max(0, min(lo, K))
        hi = max(lo, min(hi, K))
        keep = idx[lo:hi]
        out = m[keep, :]

        return out, {
            "filter": "auc_trim",
            "K_in": K,
            "K_out": int(out.shape[0]),
            "trim_pct": trim_pct,
            "min_keep": min_keep,
            "keep_lo": lo,
            "keep_hi": hi,
        }

    def _aggregate_members_phys_then_rescale(members_scaled_2d: np.ndarray, reducer: str) -> np.ndarray:
        m = np.asarray(members_scaled_2d, dtype=float)
        if m.ndim != 2 or m.size == 0:
            return np.asarray(m).reshape(1, -1)
        m_phys = np.asarray(_inverse_2d_with_scaler(m, scaler_y), dtype=float)
        agg_phys = _reduce_members_phys(m_phys, reducer).reshape(1, -1)
        return np.asarray(_transform_2d_with_scaler(agg_phys, scaler_y), dtype=float).reshape(1, -1)

    def _resolve_traj_trim_cfg(c: Dict[str, Any], spaghetti_enabled: bool) -> Tuple[str, float, int]:
        raw = (
            c.get("traj_outlier_filter")
            or c.get("trajectory_outlier_filter")
            or c.get("coupling_traj_outlier_filter")
            or c.get("coupling_traj_filter")
        )
        filt = ("auc_trim" if spaghetti_enabled else "none") if raw is None else str(raw).strip().lower()
        trim = _f(c.get("traj_outlier_trim_pct", c.get("trajectory_outlier_trim_pct", 0.10)), 0.10)
        keep = _i(c.get("traj_outlier_min_keep", c.get("trajectory_outlier_min_keep", 20)), 20)
        return str(filt), float(trim), int(keep)

    def _emit_rebuild_summary(split: str, ctx: Optional[Dict[str, Any]], members_scaled: Optional[np.ndarray], trim_meta: Optional[Dict[str, Any]]) -> None:
        if not isinstance(ctx, dict):
            return
        b = np.asarray(ctx.get("b_used", []), dtype=float)
        b = b[np.isfinite(b)]
        n_calls = int(ctx.get("n_calls", 0))
        K_out = int(members_scaled.shape[0]) if isinstance(members_scaled, np.ndarray) and members_scaled.ndim == 2 else 0

        q0 = ctx.get("q0", None)
        q0_src = ctx.get("q0_source", None)
        Ht = ctx.get("len", None)

        if b.size:
            logger.info(
                "arps_rebuild_summary split=%s len=%s q0=%s src=%s K=%d->%d b[min,p50,mean,max]=[%.6g,%.6g,%.6g,%.6g] std=%.6g",
                str(split), str(Ht), (None if q0 is None else float(q0)), str(q0_src),
                int(n_calls), int(K_out),
                float(np.min(b)), float(np.median(b)), float(np.mean(b)), float(np.max(b)), float(np.std(b)),
            )
        else:
            logger.info("arps_rebuild_summary split=%s len=%s q0=%s src=%s K=%d->%d", str(split), str(Ht), str(q0), str(q0_src), int(n_calls), int(K_out))

        if isinstance(trim_meta, dict) and trim_meta.get("filter") == "auc_trim":
            logger.info(
                "arps_rebuild_filter split=%s filter=auc_trim trim_pct=%.3f min_keep=%d keep=%d/%d",
                str(split),
                float(trim_meta.get("trim_pct", 0.0)),
                int(trim_meta.get("min_keep", 0)),
                int(trim_meta.get("K_out", K_out)),
                int(trim_meta.get("K_in", n_calls)),
            )

    # -----------------------
    # NEW (backward-compatible): resolve b_used for spaghetti
    # -----------------------
    def _resolve_b_used_for_spaghetti(
        *,
        ap_val: Dict[str, Any],
        b_val: float,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns (b_used, meta). Backward-compatible defaults:
          - If theta sampling disabled: prefer b_candidates if present else [b_val].
          - If theta sampling enabled: sample around b_val (Normal) with optional clipping.
        """
        b_candidates = ap_val.get("b_candidates", ap_val.get("b_all"))
        b_candidates = np.asarray(b_candidates, dtype=float) if b_candidates is not None else np.asarray([], dtype=float)
        b_candidates = b_candidates[np.isfinite(b_candidates)]

        K_req = _i(cfg.get("coupling_spaghetti_k", 0), 0)
        if K_req <= 0:
            K_req = int(b_candidates.size) if b_candidates.size > 0 else 1

        theta_sampling = bool(cfg.get("coupling_theta_sampling", False))
        dist = str(cfg.get("coupling_theta_sampling_dist", "normal")).strip().lower()

        if not theta_sampling:
            if b_candidates.size == 0:
                return np.asarray([float(b_val)], dtype=float), {"method": "legacy_single_b", "K_req": int(K_req), "K_eff": 1}
            K_eff = max(1, min(int(K_req), int(b_candidates.size)))
            return np.asarray(b_candidates[:K_eff], dtype=float), {
                "method": "legacy_candidates",
                "K_req": int(K_req),
                "K_eff": int(K_eff),
                "candidates_total": int(b_candidates.size),
            }

        # ---- sampling path (Normal only for now) ----
        if dist not in ("normal", "gaussian"):
            dist = "normal"

        # Optional deterministic seed (ONLY if user sets it)
        seed = cfg.get("coupling_theta_sampling_seed", None)
        if seed is not None:
            try:
                np.random.seed(int(seed))
            except Exception:
                pass

        sigma_abs = _f(cfg.get("coupling_theta_sampling_sigma", np.nan), np.nan)
        rel_sigma = cfg.get("coupling_theta_sampling_rel_sigma", None)
        sigma = None

        if np.isfinite(sigma_abs):
            sigma = float(sigma_abs)
        else:
            try:
                rel_sigma_f = float(rel_sigma) if rel_sigma is not None else np.nan
            except Exception:
                rel_sigma_f = np.nan
            if np.isfinite(rel_sigma_f):
                sigma = abs(float(b_val)) * float(rel_sigma_f)

        if sigma is None or (not np.isfinite(sigma)) or sigma <= 0:
            sigma = max(1e-12, abs(float(b_val)) * 0.01) if np.isfinite(b_val) else 0.0

        K_eff = max(1, int(K_req))

        if (not np.isfinite(b_val)) or sigma <= 0.0:
            samples = np.full((K_eff,), float(b_val), dtype=float)
        else:
            samples = np.random.normal(loc=float(b_val), scale=float(sigma), size=int(K_eff)).astype(float)

        cmin = cfg.get("coupling_theta_sampling_clip_min", None)
        cmax = cfg.get("coupling_theta_sampling_clip_max", None)
        try:
            cmin_f = float(cmin) if cmin is not None else None
        except Exception:
            cmin_f = None
        try:
            cmax_f = float(cmax) if cmax is not None else None
        except Exception:
            cmax_f = None

        # --- PATCH: robust sentinel handling for theta sampling clipping ---
        # Rationale:
        #   clip_max=0 should NOT clamp b to 0 (kills ensemble); treat as "disable".
        #   Also guard against invalid ranges (clip_min >= clip_max).
        if cmax_f is not None:
            if (not np.isfinite(cmax_f)) or (float(cmax_f) <= 0.0):
                logger.warning(
                    "arps_theta_sampling: clip_max=%r is non-positive/invalid; disabling clip_max to avoid collapsing samples.",
                    cmax,
                )
                cmax_f = None
        
        if cmin_f is not None:
            if (not np.isfinite(cmin_f)):
                logger.warning(
                    "arps_theta_sampling: clip_min=%r is invalid; disabling clip_min.",
                    cmin,
                )
                cmin_f = None
        
        # If both are set but inconsistent, disable both (safest backward-compatible behavior)
        if (cmin_f is not None) and (cmax_f is not None):
            if float(cmin_f) >= float(cmax_f):
                logger.warning(
                    "arps_theta_sampling: invalid clip range clip_min=%r >= clip_max=%r; disabling clipping.",
                    cmin_f, cmax_f,
                )
                cmin_f = None
                cmax_f = None
        
        # Apply clipping only if valid
        if cmin_f is not None:
            samples = np.maximum(samples, float(cmin_f))
        if cmax_f is not None:
            samples = np.minimum(samples, float(cmax_f))


        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            samples = np.asarray([float(b_val)], dtype=float)

        return np.asarray(samples, dtype=float), {
            "method": "theta_sampling",
            "dist": dist,
            "seed": (None if seed is None else int(seed)),
            "sigma": float(sigma),
            "K_req": int(K_req),
            "K_eff": int(samples.size),
            "clip_min": (None if cmin_f is None else float(cmin_f)),
            "clip_max": (None if cmax_f is None else float(cmax_f)),
            "fallback_candidates_total": int(b_candidates.size),
        }

    def _build_members(
        *,
        split: str,
        pinn_scaled_1d: np.ndarray,
        H_target: int,
        b_used: np.ndarray,
        anchor_kind: str,
        anchor_window: int,
        force_monotonic: bool,
        max_decline_per_step: Optional[float],
        q0_phys: float,
        q0_src: str,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        H_target = int(H_target)
        if H_target <= 0:
            return None, {"split": str(split), "len": int(H_target), "n_calls": 0, "b_used": []}

        b_arr = np.asarray(b_used, dtype=float)
        b_arr = b_arr[np.isfinite(b_arr)]
        if b_arr.size == 0:
            return None, {"split": str(split), "len": int(H_target), "n_calls": 0, "b_used": []}

        ctx: Dict[str, Any] = {
            "split": str(split),
            "len": int(H_target),
            "q0": float(q0_phys),
            "q0_source": str(q0_src),
            "n_calls": 0,
            "b_used": [],
        }
        members: List[np.ndarray] = []

        for b_i in b_arr:
            y_i = _apply_arps_from_pinn(
                pinn_scaled_1d,
                scaler_y,
                H_target=H_target,
                b=float(b_i),
                anchor_kind=anchor_kind,
                anchor_window=int(anchor_window),
                force_monotonic=bool(force_monotonic),
                max_decline_per_step=max_decline_per_step,
                logger=logger,
                split_label=str(split),
                q0_override_phys=float(q0_phys),
                rebuild_log_ctx=ctx,
            )
            members.append(np.asarray(y_i, dtype=float).reshape(-1))

        return (np.vstack(members) if members else None), ctx

    # -----------------------
    # Guards
    # -----------------------
    if outs_val is None or outs_test is None:
        return outs_val, outs_test

    coupling_mode = str(cfg.get("arps_coupling_mode", "none")).strip().lower()
    if _mode(cfg) != "offline_analytic" or coupling_mode == "none":
        return outs_val, outs_test

    if scaler_y is None:
        logger.warning("arps_coupling: scaler_y is None; skipping coupling.")
        return outs_val, outs_test

    H_val = _i(split_recon_lengths.get("val", 0), 0)
    H_test = _i(split_recon_lengths.get("test", 0), 0)
    if H_val <= 0 or H_test <= 0:
        logger.warning("arps_coupling: invalid H_val=%d or H_test=%d; skipping coupling.", H_val, H_test)
        return outs_val, outs_test

    ap_val = outs_val.get("analytic_params")
    ap_test = outs_test.get("analytic_params")
    if not isinstance(ap_val, dict) or not isinstance(ap_test, dict):
        logger.warning("arps_coupling: missing analytic_params; skipping coupling.")
        return outs_val, outs_test

    b_val = _f(ap_val.get("b", np.nan))
    b_test = _f(ap_test.get("b", np.nan))
    if not np.isfinite(b_val) or not np.isfinite(b_test):
        logger.warning("arps_coupling: non-finite b (b_val=%r, b_test=%r); skipping coupling.", b_val, b_test)
        return outs_val, outs_test

    pinn_val = _as_1d(outs_val.get("pred_pinn_coupling", outs_val.get("pred")))
    pinn_test = _as_1d(outs_test.get("pred_pinn_coupling", outs_test.get("pred")))
    if pinn_val is None or pinn_test is None:
        logger.warning("arps_coupling: missing PINN ribbons; skipping coupling.")
        return outs_val, outs_test

    y_val_base = _as_2d_row(outs_val.get("pred"))
    y_test_base = _as_2d_row(outs_test.get("pred"))
    if y_val_base is None or y_test_base is None:
        logger.warning("arps_coupling: missing outs_*['pred']; skipping coupling.")
        return outs_val, outs_test

    # knobs
    anchor_kind = str(cfg.get("analytic_anchor_kind", "median")).strip().lower()
    anchor_window = _i(cfg.get("arps_anchor_window", cfg.get("analytic_anchor_window", 15)), 15)
    force_monotonic = bool(cfg.get("analytic_force_monotonic", True))

    max_decline = _f(cfg.get("analytic_max_decline_per_step", np.nan), np.nan)
    max_decline = None if (not np.isfinite(max_decline)) else float(max_decline)

    coupling_spaghetti = bool(cfg.get("coupling_spaghetti", False))
    spaghetti_reducer = str(cfg.get("coupling_spaghetti_reducer", cfg.get("coupling_spaghetti_agg", "median"))).strip().lower()

    emit_members = bool(cfg.get("emit_spaghetti_members") or cfg.get("plot", False) or cfg.get("debug_arps", False))
    traj_filter, traj_trim_pct, traj_min_keep = _resolve_traj_trim_cfg(cfg, spaghetti_enabled=coupling_spaghetti)
    do_auc_trim = traj_filter in ("auc_trim", "trim_auc", "auc")

    min_q0_phys = _min_q0(cfg)
    q0_val_phys, q0_val_src = _resolve_q0_phys(outs_val, "val", min_q0_phys)
    q0_test_phys, q0_test_src = _resolve_q0_phys(outs_test, "test", min_q0_phys)

    logger.info(
        "arps_anchor_resolved history_only q0_test_phys=%s src_test=%s q0_val_phys=%s src_val=%s min_q0=%s",
        (None if q0_test_phys is None else float(q0_test_phys)), str(q0_test_src),
        (None if q0_val_phys is None else float(q0_val_phys)), str(q0_val_src),
        str(min_q0_phys),
    )

    if coupling_mode == "val_only":
        if q0_test_phys is None:
            logger.info("arps_coupling_skip mode=val_only reason=missing_test_q0_history_anchor")
            return outs_val, outs_test
    else:
        if q0_val_phys is None or q0_test_phys is None:
            logger.info("arps_coupling_skip mode=val_plus_test reason=missing_val_or_test_q0_history_anchor")
            return outs_val, outs_test

    # -----------------------
    # Coupling modes
    # -----------------------
    if coupling_mode == "val_only":
        logger.info("arps_coupling mode=val_only b_val=%.6f", float(b_val))

        # default: VAL unchanged
        y_val_new = y_val_base
        y_test_new: Optional[np.ndarray] = None

        if coupling_spaghetti:
            b_used, b_meta = _resolve_b_used_for_spaghetti(ap_val=ap_val, b_val=float(b_val), cfg=cfg)

            logger.info(
                "arps_coupling_spaghetti: method=%s K_req=%s K_eff=%s reducer=%s traj_filter=%s trim_pct=%.3f min_keep=%d",
                str(b_meta.get("method")),
                str(b_meta.get("K_req")),
                str(b_meta.get("K_eff")),
                str(spaghetti_reducer),
                str(traj_filter),
                float(traj_trim_pct),
                int(traj_min_keep),
            )

            if b_meta.get("method") == "theta_sampling":
                logger.info(
                    "arps_coupling_theta_sampling enabled dist=%s sigma=%s seed=%s clip_min=%s clip_max=%s",
                    str(b_meta.get("dist")),
                    str(b_meta.get("sigma")),
                    str(b_meta.get("seed")),
                    str(b_meta.get("clip_min")),
                    str(b_meta.get("clip_max")),
                )

            test_members, test_ctx = _build_members(
                split="test",
                pinn_scaled_1d=pinn_test,
                H_target=H_test,
                b_used=b_used,
                anchor_kind=anchor_kind,
                anchor_window=anchor_window,
                force_monotonic=force_monotonic,
                max_decline_per_step=max_decline,
                q0_phys=float(q0_test_phys),
                q0_src=str(q0_test_src),
            )

            val_members, val_ctx = (None, {})
            if q0_val_phys is not None:
                val_members, val_ctx = _build_members(
                    split="val",
                    pinn_scaled_1d=pinn_val,
                    H_target=H_val,
                    b_used=b_used,
                    anchor_kind=anchor_kind,
                    anchor_window=anchor_window,
                    force_monotonic=force_monotonic,
                    max_decline_per_step=max_decline,
                    q0_phys=float(q0_val_phys),
                    q0_src=str(q0_val_src),
                )

            test_trim_meta = val_trim_meta = None
            if do_auc_trim:
                if test_members is not None:
                    test_members, test_trim_meta = _trim_by_auc(test_members, float(traj_trim_pct), int(traj_min_keep))
                if val_members is not None:
                    val_members, val_trim_meta = _trim_by_auc(val_members, float(traj_trim_pct), int(traj_min_keep))

            _emit_rebuild_summary("test", test_ctx, test_members, test_trim_meta)
            if val_members is not None and isinstance(val_ctx, dict):
                _emit_rebuild_summary("val", val_ctx, val_members, val_trim_meta)

            # Aggregate in PHYSICAL and rescale (or fallback to single curve if members missing)
            if test_members is None:
                y_test_new = _as_2d_row(_apply_arps_from_pinn(
                    pinn_test, scaler_y,
                    H_target=H_test, b=float(b_val),
                    anchor_kind=anchor_kind, anchor_window=anchor_window,
                    force_monotonic=force_monotonic, max_decline_per_step=max_decline,
                    logger=logger, split_label="test",
                    q0_override_phys=float(q0_test_phys),
                ))
            else:
                y_test_new = _aggregate_members_phys_then_rescale(test_members, spaghetti_reducer)

            if val_members is not None:
                y_val_new = _aggregate_members_phys_then_rescale(val_members, spaghetti_reducer)

            new_val, new_test = dict(outs_val), dict(outs_test)
            new_val["pred_analytic_coupled"] = y_val_new
            new_test["pred_analytic_coupled"] = y_test_new
            new_val["pred"] = y_val_new
            new_test["pred"] = y_test_new

            # stamp anchors
            new_val.setdefault("anchor", {})
            new_test.setdefault("anchor", {})
            if isinstance(new_val["anchor"], dict):
                new_val["anchor"].update({"q0_phys": q0_val_phys, "q0_source": q0_val_src})
            if isinstance(new_test["anchor"], dict):
                new_test["anchor"].update({"q0_phys": q0_test_phys, "q0_source": q0_test_src})

            # optional payload for integrated view / plotting
            if emit_members:
                if test_members is not None:
                    new_test["pred_members"] = np.asarray(test_members, dtype=float)
                    new_test["pred_members_meta"] = {
                        "split": "test",
                        "reducer": spaghetti_reducer,
                        "reduce_space": "physical",
                        "traj_outlier": (dict(test_trim_meta) if test_trim_meta else None),
                        "b_used_meta": dict(b_meta),
                    }
                    new_test["integrated_view_test_members_scaled"] = new_test["pred_members"]
                    new_test["integrated_view_test_members_meta"] = dict(new_test["pred_members_meta"])
                if val_members is not None:
                    new_val["pred_members"] = np.asarray(val_members, dtype=float)
                    new_val["pred_members_meta"] = {
                        "split": "val",
                        "reducer": spaghetti_reducer,
                        "reduce_space": "physical",
                        "traj_outlier": (dict(val_trim_meta) if val_trim_meta else None),
                        "b_used_meta": dict(b_meta),
                    }
                    new_val["integrated_view_val_members_scaled"] = new_val["pred_members"]
                    new_val["integrated_view_val_members_meta"] = dict(new_val["pred_members_meta"])

            logger.info(
                "arps_coupling_apply mode=val_only_spaghetti reducer=%s len_val=%d len_test=%d emit_members=%s traj_filter=%s",
                str(spaghetti_reducer), int(H_val), int(H_test), "yes" if emit_members else "no", str(traj_filter),
            )
            return new_val, new_test

        # Single-curve coupling: TEST rebuilt using b_val, VAL unchanged
        y_test_new = _as_2d_row(_apply_arps_from_pinn(
            pinn_test, scaler_y,
            H_target=H_test, b=float(b_val),
            anchor_kind=anchor_kind, anchor_window=anchor_window,
            force_monotonic=force_monotonic, max_decline_per_step=max_decline,
            logger=logger, split_label="test",
            q0_override_phys=float(q0_test_phys),
        ))

        new_val, new_test = dict(outs_val), dict(outs_test)
        new_val["pred_analytic_coupled"] = y_val_base
        new_test["pred_analytic_coupled"] = y_test_new
        new_val["pred"] = y_val_base
        new_test["pred"] = y_test_new
        return new_val, new_test

    # val_plus_test: weighted b_final applied to BOTH
    w_val, w_test = float(max(H_val, 1)), float(max(H_test, 1))
    b_final = (float(b_val) * w_val + float(b_test) * w_test) / (w_val + w_test)

    logger.info(
        "arps_coupling mode=val_plus_test b_val=%.6f b_test=%.6f -> b_final=%.6f (w_val=%.1f w_test=%.1f)",
        float(b_val), float(b_test), float(b_final), w_val, w_test,
    )

    y_val_new = _as_2d_row(_apply_arps_from_pinn(
        pinn_val, scaler_y,
        H_target=H_val, b=float(b_final),
        anchor_kind=anchor_kind, anchor_window=anchor_window,
        force_monotonic=force_monotonic, max_decline_per_step=max_decline,
        logger=logger, split_label="val",
        q0_override_phys=float(q0_val_phys),
    ))
    y_test_new = _as_2d_row(_apply_arps_from_pinn(
        pinn_test, scaler_y,
        H_target=H_test, b=float(b_final),
        anchor_kind=anchor_kind, anchor_window=anchor_window,
        force_monotonic=force_monotonic, max_decline_per_step=max_decline,
        logger=logger, split_label="test",
        q0_override_phys=float(q0_test_phys),
    ))

    new_val, new_test = dict(outs_val), dict(outs_test)
    new_val["pred_analytic_coupled"] = y_val_new
    new_test["pred_analytic_coupled"] = y_test_new
    new_val["pred"] = y_val_new
    new_test["pred"] = y_test_new
    return new_val, new_test




def show_pipeline_presets(*, as_style: bool = True) -> Any:
    """
    Notebook-friendly viewer for PIPELINE_PRESETS (no truncation).
    Returns the DataFrame (or Styler) and also displays it.
    """
    import pandas as pd
    from IPython.display import display

    df = pd.DataFrame(describe_pipeline_presets())

    # Local (non-global) display options
    with pd.option_context(
        "display.max_colwidth", None,
        "display.max_columns", 50,
        "display.width", 200,
        "display.expand_frame_repr", False,
    ):
        if as_style:
            sty = df.style.set_properties(**{"white-space": "pre-wrap"})
            display(sty)
            return sty
        else:
            display(df)
            return df



def build_pipeline_config_overrides(
    *,
    preset: str = "OFF",
    job_knobs: Optional[Dict[str, Any]] = None,
    run_params: Optional[Dict[str, Any]] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build CONFIG_OVERRIDES in a safe, consistent way.

    Contract:
      - Always starts from OFF baseline.
      - Applies the chosen preset.
      - Applies notebook-controlled knobs (seed/plot/lag_window/horizon/test_size/val_size).
      - Applies run_params (ensemble_size/max_workers).
      - Applies extra_overrides last (if you really need a one-off tweak).

    Also enforces:
      - job_defaults.latent_mode == job_defaults.latent_cfg["mode"] (single source of truth).
      - If latent_cfg.plot is not explicitly set, it inherits job_defaults.plot.
        (This makes downstream gates that read latent_cfg.plot behave consistently.)
    """
    if preset not in PIPELINE_PRESETS:
        raise ValueError(f"Unknown preset={preset!r}. Available: {sorted(PIPELINE_PRESETS)}")

    # Minimal stable skeleton
    cfg: Dict[str, Any] = {
        "job_defaults": {
            "seed": 42,
            "plot": True,
            "lag_window": 100,
            "horizon": 300,
            "test_size": 0.45,
            "val_size": 0.2,
            "latent_mode": "off",
            "eval_mode": "seq",
            "fullseq_mode": "deploy_split_k",
            "fullseq_k": 1,
            "latent_cfg": {"mode": "off"},
        },
        "run_params": {"ensemble_size": 1, "max_workers": 1},
    }

    # OFF baseline first, then selected preset
    _deep_update(cfg, PIPELINE_PRESETS["OFF"].overrides)
    _deep_update(cfg, PIPELINE_PRESETS[preset].overrides)

    # Notebook-controlled knobs (only these)
    if job_knobs:
        _deep_update(cfg["job_defaults"], job_knobs)

    # Run params (only ensemble knobs)
    if run_params:
        _deep_update(cfg["run_params"], run_params)

    # Optional last-mile overrides (rare)
    if extra_overrides:
        _deep_update(cfg, extra_overrides)

    # Enforce coherence: one truth for mode + propagate plot intent into latent_cfg (if missing)
    jd = cfg["job_defaults"]

    lc = jd.get("latent_cfg")
    if not isinstance(lc, dict):
        lc = {}

    mode = str(lc.get("mode", jd.get("latent_mode", "off"))).strip().lower()
    lc["mode"] = mode

    # IMPORTANT: downstream emission gates check latent_cfg.plot, not job_defaults.plot.
    # Only inherit if not explicitly set at latent_cfg level.
    if "plot" not in lc:
        lc["plot"] = bool(jd.get("plot", False))

    jd["latent_cfg"] = lc
    jd["latent_mode"] = mode

    return cfg



def rank_position_summary(members_2d: Any, pred_1d: Any) -> Dict[str, Any]:
    import numpy as np
    M = np.asarray(members_2d)  # (n_members, T)
    p = np.asarray(pred_1d)     # (T,)
    if M.ndim != 2 or p.ndim != 1 or M.shape[1] != p.shape[0] or M.shape[0] < 2:
        return {"rank_pos": "na"}

    with np.errstate(invalid="ignore"):
        pos = np.nanmean(M <= p[None, :], axis=0)  # (T,)

    def q(x): return float(np.nanquantile(pos, x)) if np.isfinite(pos).any() else None

    return {
        "T": int(pos.shape[0]),
        "rank_med": q(0.50),
        "rank_q10": q(0.10),
        "rank_q90": q(0.90),
        "rank_bad%": float(np.nanmean((pos < 0.35) | (pos > 0.65)) * 100.0),
    }


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict update (src wins)."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = deepcopy(v)
    return dst


@dataclass(frozen=True)
class PipelinePreset:
    name: str
    description: str
    overrides: Dict[str, Any]


# ---------------------------------------------------------------------
# Focused presets (4 modes)
# ---------------------------------------------------------------------

# 1) Baseline: raw model outputs, no post-processing.
_OFF = PipelinePreset(
    name="OFF",
    description=(
        "Baseline. No post-processing. Predictions are raw model outputs. "
        "USES ONLY PINN - Legacy Mode"
    ),
    overrides={
        "job_defaults": {
            "latent_mode": "off",
            "eval_mode": "seq",                 # keep classic evaluation
            "fullseq_mode": "deploy_split_k",
            "fullseq_k": 1,
            "latent_cfg": {"mode": "off"},
        }
    },
)

# 2) Full-sequence adapter (PhysicsDecoder): rebuild to full-length by design.
_FULL_SEQUENCE = PipelinePreset(
    name="FULL_SEQUENCE_PHYSICSDECODER",
    description=(
        "Full-sequence latent adapter (PhysicsDecoder). "
        "Reconstructs full-length series using split_recon_lengths."
    ),
    overrides={
        "job_defaults": {
            "latent_mode": "full_sequence",
            "eval_mode": "fullseq_k",
            "fullseq_mode": "deploy_split_k",
            "fullseq_k": 1,
            "latent_cfg": {"mode": "full_sequence"},
        }
    },
)


# 4) Offline-analytic, coupled, ensemble-curve + spaghetti: members emitted for integrated view
#    PLUS: trajectory outlier trim by AUC (10% each side), keep >= 20.
_PINN_PLUS_ANALYTIC = PipelinePreset(
    name="PINN_PLUS_ANALYTIC",
    description=(
        "Offline-analytic ARPS with VAL->TEST coupling (val_only) and spaghetti members "
        "for integrated-view plotting. Members are trimmed by trajectory AUC "
        "(drop 10% lowest/highest, keep >= 20). Final curve is aggregated from trimmed members."
    ),
    overrides={
        "job_defaults": {
            "latent_mode": "offline_analytic",
            "eval_mode": "agg",
            "plot": True,                       # ensures emission gate works even without debug_arps
            "latent_cfg": {
                "mode": "offline_analytic",
                "analytic_override_pred": True,
                "arps_coupling_mode": "val_only",
            
                # Use all VAL windows -> b candidates -> members
                "analytic_use_only_first_window": False,
                "b_fit_policy": "all",
                "default_b_fit_k": -1,
                "b_reducer": "median",
            
                # Spaghetti members
                "coupling_spaghetti": True,
                "coupling_spaghetti_k": -1,  #-1 all samples
                "coupling_spaghetti_agg": "median",
            
                # Trajectory outlier trimming (AUC-based)
                "traj_outlier_filter": "auc_trim",
                "traj_outlier_trim_pct": 0.2,
                "traj_outlier_min_keep": 20,
            
                # Anchor (keep consistent with your logs)
                "analytic_anchor_kind": "median",
                "analytic_anchor_window": 15,
            
                # NEW: deterministic history anchor rules
                "history_anchor_x_in_scaler_space": True,
                "history_anchor_channel": -1,
                "history_anchor_source": "x_target_channel",
            
                "debug_arps": True,
            },
        }
    },
)

# -----------------------------------------------------------------------------
# Mode 4 — Pure ARPS (Canonical) + per-trial parameter sampling (spaghetti)
# -----------------------------------------------------------------------------
_ARPS_ENSEMBLE_SPAGHETTI = PipelinePreset(
    name="ARPS_ENSEMBLE_SPAGHETTI",
    description=(
        "Pure ARPS canonical fit with per-trial parameter sampling around theta-hat "
        "to generate spaghetti members. Optional AUC trimming and median (p50) aggregation. "
        "Scores in VAL, audits in TEST. (No offline_analytic / no coupling.)"
    ),
    overrides={
        "job_defaults": {
            # IMPORTANT: pure ARPS path (no offline_analytic)
            "latent_mode": "off",
            "eval_mode": "agg",
            "plot": True,

            # Dedicated block for mode-4 only (consumed by _run_single_job_arps)
            "arps_ensemble": {
                "enabled": True,
                "seed": 42,

                # -------------------------
                # members + aggregation
                # -------------------------
                "k": 150,                    # you wanted 150
                "agg": "median",             # "median" | "mean"
                "emit_members": True,        # keep members for integrated view / Series Store
                "emit_members_scaled": True, # ensure members_scaled exist in ensemble_out (heavy)
                "emit_members_for_integrated_view": True,

                # -------------------------
                # NEW: qi anchoring (history-based q0)
                # -------------------------
                # Uses compute_history_q0_phys(X_any=train_kwargs["X_train"], scaler_X, scaler_target, ...)
                "qi_anchor": "history_q0_phys",
                "q0_window": 30,
                "q0_kind": "median",
                "q0_min": 1e-6,
                "q0_channel": -1,            # last channel by default (same convention as compute_history_q0_phys)

                # -------------------------
                # parameter sampling knobs
                # -------------------------
                # Master switch: when False, disables ALL perturbations (qi/D/b).
                "theta_sampling": True,

                # Your compatibility vocabulary (informational today)
                "theta_sampling_dist": "normal",       # informational (sampler fixed-form today)

                # IMPORTANT: In your implementation, theta_sampling_sigma maps to b_abs_sigma if b_abs_sigma not set.
                # For hyperbolic, if you forget b_abs_sigma, code will fallback to b_abs_sigma_default_hyperbolic (0.05 by default).
                "theta_sampling_sigma": None,          # leave None to rely on b_abs_sigma below (clearer)

                "theta_sampling_rel_sigma": None,      # optional future use
                "theta_sampling_clip": {"b_min": 1e-3, "b_max": 5.0},

                # Concrete sampler keys (USED NOW)
                "qi_rel_sigma": 0.0,       # lognormal rel noise on qi
                "D_rel_sigma": 0.30,        # lognormal rel noise on D
                "b_abs_sigma": 0.08,        # <-- give real diversity in hyperbolic (0.003 is usually too tiny)
                "b_clip_min": 1e-3,
                "b_clip_max": 5.0,

                # Optional: if you want the code to auto-pick b sigma for hyperbolic when b_abs_sigma=None
                # "b_abs_sigma_default_hyperbolic": 0.05,

                # -------------------------
                # trimming (AUC-based)
                # -------------------------
                "traj_outlier_filter": "auc_trim",
                "traj_outlier_trim_pct": 0.2,
                "traj_outlier_min_keep": 20,

                # -------------------------
                # misc
                # -------------------------
                "debug_arps": False,
            },
        }
    },
)





PIPELINE_PRESETS = {
    p.name: p
    for p in [
        _OFF,
        _FULL_SEQUENCE,
        _PINN_PLUS_ANALYTIC,
        _ARPS_ENSEMBLE_SPAGHETTI,   # NEW
    ]
}


def describe_pipeline_presets() -> List[Dict[str, str]]:
    """Human-friendly preset catalog (data only)."""
    return [
        {"preset": name, "description": spec.description}
        for name, spec in PIPELINE_PRESETS.items()
    ]