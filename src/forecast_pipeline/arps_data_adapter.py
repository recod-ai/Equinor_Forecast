# src/forecast_pipeline/adapters/arps_data_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureMap:
    """
    Canonical mapping for ARPS drivers inside X windows.
    Column names must match the DataFrame feature names BEFORE the target was moved to last.
    Indices are resolved from `feature_names` (DataFrame order with target removed).
    """
    pi: str = "PI"
    pwf: str = "AVG_DOWNHOLE_PRESSURE"
    time: str = "Tempo_Inicio_Prod"
    target: str = "BORE_OIL_VOL"  # for unit checks only


@dataclass
class ArpsSplitDrivers:
    """
    Drivers for a single split (train/val/test), inverse-scaled to physical units.

    Shapes:
        X_pi_hist, X_pwf_hist, X_t_hist: (N, L)
        left_target_windows_scaled: (K, W) or None  (scaled with scaler_target),
            where W = min(available_left_length, H). We do NOT require W == H.
    """
    X_pi_hist: np.ndarray
    X_pwf_hist: np.ndarray
    X_t_hist: np.ndarray
    left_target_windows_scaled: Optional[np.ndarray]
    horizon: int


# ---------------------------------------------------------------------
# Index & inverse-transform helpers
# ---------------------------------------------------------------------
def _resolve_indices(feature_names: List[str], fmap: FeatureMap) -> Dict[str, int]:
    """
    Resolve column indices inside the 'features part' of X (i.e., excluding the target channel).
    """
    try:
        pi_idx   = feature_names.index(fmap.pi)
        pwf_idx  = feature_names.index(fmap.pwf)
        time_idx = feature_names.index(fmap.time)
        return {"pi": pi_idx, "pwf": pwf_idx, "time": time_idx}
    except ValueError as e:
        raise KeyError(
            "FeatureMap could not resolve indices from feature_names.\n"
            f"Got feature_names={feature_names}\nExpected at least: "
            f"[{fmap.pi}, {fmap.pwf}, {fmap.time}]"
        ) from e


def _inverse_feature_channel(x_scaled: np.ndarray, scaler_X, channel_idx: int) -> np.ndarray:
    """
    Inverse-transform one feature channel from X_scaled using the global feature scaler_X.

    x_scaled: (N, L, F_total) where F_total = (#features_without_target + 1 target channel)
    channel_idx: index inside the 'features_without_target' block (i.e., before the last channel).
    Returns (N, L) in physical units.
    """
    feats_scaled = x_scaled[..., :-1]  # (N, L, F_feat)
    c = feats_scaled[..., channel_idx].reshape(-1, 1)  # (N*L, 1)

    nl, nf = c.shape[0], feats_scaled.shape[-1]
    zero = np.zeros((nl, nf), dtype=feats_scaled.dtype)
    zero[:, channel_idx:channel_idx+1] = c

    inv = scaler_X.inverse_transform(zero)[:, channel_idx]  # (N*L,)
    return inv.reshape(x_scaled.shape[0], x_scaled.shape[1])


def _extract_time_channel(x_scaled: np.ndarray, scaler_X, time_idx: int) -> np.ndarray:
    """
    Time is treated like a standard feature and inverse-transformed via scaler_X.
    Returns integer-like time if it was originally integer (best-effort).
    """
    t = _inverse_feature_channel(x_scaled, scaler_X, time_idx)
    if np.all(np.isfinite(t)):
        t_rounded = np.rint(t)
        if np.nanmean(np.abs(t - t_rounded)) < 1e-6:
            t = t_rounded
    return t


# ---------------------------------------------------------------------
# LEFT sidecar extraction (split-aware)
# ---------------------------------------------------------------------
def _get_left_target_windows_scaled(
    scaler_target,
    split: str,
    expected_h: int
) -> Optional[np.ndarray]:
    """
    Build warm-left target windows (scaled with scaler_target) for the given split ('val'|'test').

    Reads the sidecar attached by data prep: `scaler_target._split_ctx`.
    Returns shape (K, W) with W = min(available_left_length, expected_h),
    or None if not available.

    Notes:
    - Sidecar stores LEFT as X windows (K, L, F_total). We extract the target channel
      (last feature) and keep the trailing W columns.
    - We no longer warn when LEFT length < H; this is expected when L < H.
      Only warn if LEFT is missing or malformed.
    """
    ctx = getattr(scaler_target, "_split_ctx", None)
    if not ctx:
        logger.info("[ARPS.Adapter] No _split_ctx on scaler_target; warm-left unavailable.")
        return None

    h_ctx = int(ctx.get("H", 0)) or int(expected_h or 0)
    if h_ctx <= 0:
        logger.info("[ARPS.Adapter] _split_ctx has no valid 'H'; warm-left unavailable.")
        return None

    if expected_h and (h_ctx != expected_h):
        logger.warning("[ARPS.Adapter] Sidecar H (%d) != expected H (%d). Proceeding with sidecar H.", h_ctx, expected_h)

    key = "X_val_left_scaled" if split == "val" else "X_test_left_scaled"
    left_X = ctx.get(key, None)
    if left_X is None or getattr(left_X, "size", 0) == 0:
        logger.info("[ARPS.Adapter] Sidecar has no '%s'; warm-left unavailable.", key)
        return None

    # Expect (K, L, F_total), target channel = last
    if left_X.ndim != 3 or left_X.shape[-1] < 1:
        logger.warning("[ARPS.Adapter] Unexpected LEFT shape %s. Skipping warm-left.", tuple(getattr(left_X, "shape", ())))
        return None

    target_scaled = left_X[..., -1]  # (K, L_available)
    K, L_avail = target_scaled.shape
    if L_avail == 0:
        logger.info("[ARPS.Adapter] LEFT is empty after extraction; warm-left unavailable.")
        return None

    W = int(min(L_avail, h_ctx))
    if L_avail < h_ctx:
        logger.info("[ARPS.Adapter] LEFT width (%d) < H (%d); using W=%d (no warm fallback needed).", L_avail, h_ctx, W)

    warm = target_scaled[:, -W:]  # (K, W)
    return warm.astype(float, copy=False)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def extract_arps_drivers_for_split(
    X_split_scaled: np.ndarray,
    scaler_X,
    scaler_target,
    feature_names: List[str],
    fmap: FeatureMap,
    horizon: int,
    split: str,
) -> ArpsSplitDrivers:
    """
    Extract PI, Pwf, and time (inverse-scaled) from a split's X windows and
    build warm-left target windows from scaler_target's sidecar (scaled).

    - Does NOT reconstruct continuous series; keeps sliding windows contract.
    - LEFT width may be <= H (typical when L < H).
    """
    if X_split_scaled.ndim != 3:
        raise ValueError("X_split_scaled must be rank-3 (N, L, F_total).")
    if X_split_scaled.shape[-1] < 2:
        raise ValueError("Last dim must include feature block and the target channel.")

    idx = _resolve_indices(feature_names, fmap)

    X_pi_hist  = _inverse_feature_channel(X_split_scaled, scaler_X, idx["pi"])
    X_pwf_hist = _inverse_feature_channel(X_split_scaled, scaler_X, idx["pwf"])
    X_t_hist   = _extract_time_channel(X_split_scaled, scaler_X, idx["time"])

    warm_left = _get_left_target_windows_scaled(scaler_target, split=split, expected_h=horizon)

    return ArpsSplitDrivers(
        X_pi_hist=X_pi_hist,
        X_pwf_hist=X_pwf_hist,
        X_t_hist=X_t_hist,
        left_target_windows_scaled=warm_left,
        horizon=int(horizon),
    )


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------
def assert_shapes_and_alignment(drivers: ArpsSplitDrivers, X_split_scaled: np.ndarray):
    """
    Basic alignment checks:
      - PI/Pwf/time history must match (N, L) of X.
      - LEFT, if present, must have shape (K, W) with 1 <= W <= H (no requirement of W==H).
    """
    n, L, _ = X_split_scaled.shape
    assert drivers.X_pi_hist.shape  == (n, L)
    assert drivers.X_pwf_hist.shape == (n, L)
    assert drivers.X_t_hist.shape   == (n, L)
    if drivers.left_target_windows_scaled is not None:
        K, W = drivers.left_target_windows_scaled.shape
        assert 1 <= W <= drivers.horizon, f"LEFT width ({W}) must satisfy 1 <= W <= H ({drivers.horizon})."


def check_units_sanity(pi: np.ndarray, pwf: np.ndarray, target_name: str, dataset_name: str):
    """
    Heuristic checks; logs warnings if values look off (only magnitude sanity).
    """
    with np.errstate(all="ignore"):
        med_pwf = float(np.nanmedian(pwf))
        med_pi  = float(np.nanmedian(pi))
    # psi heuristic
    if not (5.0 <= med_pwf <= 15000.0):
        logger.warning("[ARPS.Adapter][%s] Unusual Pwf median %.2f psi.", dataset_name, med_pwf)
    # PI in stb/d/psi (very rough). Keep as INFO to avoid noisy logs on synthetic/sparse wells.
    if not (1e-5 <= med_pi <= 100.0):
        logger.info("[ARPS.Adapter][%s] PI median %.6f out of typical bounds (stb/d/psi).", dataset_name, med_pi)


def mean_roundtrip_error(channel_scaled: np.ndarray, channel_idx: int, scaler_X) -> float:
    """
    Compute mean absolute error for scale -> inverse -> scale on one feature channel.
    """
    feats = channel_scaled[..., :-1]  # remove target channel
    c = feats[..., channel_idx]       # (N, L)

    # forward: inverse
    inv = _inverse_feature_channel(
        np.concatenate([feats, channel_scaled[..., -1:]], axis=-1),
        scaler_X,
        channel_idx,
    )

    # back: re-scale just that channel via transform on zeros+channel
    nl = inv.size
    nf = feats.shape[-1]
    zero = np.zeros((nl, nf), dtype=feats.dtype)
    zero[:, channel_idx] = inv.reshape(-1)
    re = scaler_X.transform(zero)[:, channel_idx].reshape(c.shape)

    err = float(np.nanmean(np.abs(c - re)))
    return err
