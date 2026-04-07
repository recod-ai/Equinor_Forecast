#src/plotting/ensemble_stats.py
# --- Standard Library Imports ---
# Used for built-in functionalities like logging and static type checking.
import logging
from typing import Dict, Optional, Sequence, Tuple

# --- Third-Party Library Imports ---
# Core external libraries for numerical operations, data manipulation, and plotting.
import numpy as np
import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger(__name__)

def _fan_levels_default() -> Tuple[float, float, float, float]:
    """Default fan coverages (central): 50/70/90/95%."""
    return (0.50, 0.9)

def _fan_from_gaussian(mean: pd.Series, scale: pd.Series,
                       levels: Sequence[float]) -> Dict[float, Tuple[pd.Series, pd.Series]]:
    """
    Constrói bandas assumindo normalidade ao redor da média com 'scale' (STD ou SEM).
    Determinístico: usa _z_for_coverage(level) para cada nível.
    """
    m = mean.to_numpy(copy=False, dtype=float)
    s = pd.Series(scale).to_numpy(copy=False, dtype=float)
    out: Dict[float, Tuple[pd.Series, pd.Series]] = {}
    for cov in levels:
        z = _z_for_coverage(float(cov))
        delta = z * s
        lo = pd.Series(m - delta, index=mean.index)
        hi = pd.Series(m + delta, index=mean.index)
        out[float(cov)] = (lo, hi)
    return out



_Z_BY_COVERAGE = {
    0.50: 0.674, 0.68: 1.000, 0.80: 1.282, 0.90: 1.645,
    0.95: 1.960, 0.98: 2.326, 0.99: 2.576
}

def _z_for_coverage(coverage: float) -> float:
    """
    Aproxima z de cobertura central (dois lados) de forma determinística.
    Usa tabela conhecida e interpola linearmente entre os pontos.
    """
    c = float(coverage)
    if c in _Z_BY_COVERAGE:
        return _Z_BY_COVERAGE[c]
    xs = np.array(sorted(_Z_BY_COVERAGE.keys()), dtype=float)
    ys = np.array([_Z_BY_COVERAGE[x] for x in xs], dtype=float)
    c_clamped = float(np.clip(c, xs[0], xs[-1]))
    return float(np.interp(c_clamped, xs, ys))


# src/plotting/ensemble_stats.py
def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """
    Canonical SMAPE (mean of pointwise values), consistent with:
      2*|y - ŷ| / (|y| + |ŷ|) * 100
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.abs(yt) + np.abs(yp) + eps
    num = np.abs(yt - yp)
    return float(np.mean(200.0 * num / denom))


def _merge_ytrue_on_t(df: pd.DataFrame, full_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'ytrue' is present by joining on 't'. If already present, keep as-is.
    """
    if "ytrue" in df.columns:
        return df
    if "t" not in df.columns or "t" not in full_history_df.columns or "ytrue" not in full_history_df.columns:
        return df  # fail-safe: return untouched (metrics will no-op)
    return df.merge(full_history_df[["t", "ytrue"]], on="t", how="left")


# --- keep imports above as-is ---
from typing import Dict, Optional, Sequence, Tuple  # make sure Optional is imported

def _compute_coverage_and_sharpness(
    g: pd.DataFrame,
    *,
    ci_mode: str,
    coverage: float,
    mean_col: str,
    std_col: str,
    n_members_col: str,
    qlow_col: str = None,
    qhigh_col: str = None,
    override_k: Optional[float] = None,   # NEW
) -> Tuple[float, float]:
    """
    Compute Coverage@coverage and Sharpness@coverage for a group (single split).
    - Coverage: fraction of points where ytrue ∈ [lower, upper]
    - Sharpness: median width (upper - lower)  [same unit as y]

    If override_k is provided and ci_mode in {'std','sem'}, use that multiplier
    instead of the usual z(coverage). For 'quantile', override_k is ignored.
    """
    cov = float(coverage)
    k_default = _z_for_coverage(cov)
    k_use = float(override_k) if (override_k is not None and str(ci_mode).lower() in {"std", "sem"}) else k_default

    lower, upper = _compute_bounds_for_mode(
        g,
        ci_mode=ci_mode,
        ci_k=k_use,
        ci_quantiles=(0.5 - cov/2.0, 0.5 + cov/2.0),
        std_col=mean_col.replace("mean", "std") if std_col is None else std_col,
        mean_col=mean_col,
        n_members_col=n_members_col,
        qlow_col=qlow_col,
        qhigh_col=qhigh_col,
    )

    # Coverage needs ytrue
    if "ytrue" in g.columns:
        y = g["ytrue"].to_numpy(dtype=float)
        inside = (y >= lower) & (y <= upper)
        cov_rate = float(np.nanmean(inside)) if inside.size else np.nan
    else:
        cov_rate = np.nan

    width = (upper - lower)
    sharpness = float(np.nanmedian(width)) if width.size else np.nan
    return cov_rate, sharpness



def calibrate_k_on_validation(
    df_final_vt: pd.DataFrame,
    full_history_df: pd.DataFrame,
    *,
    ci_mode: str = "sem",
    target_coverage: float = 0.90,
    mean_col: str = "yhat_final_mean",
    std_col: str = "std_final",
    n_members_col: str = "n_members",
    qlow_col: str = "yhat_q_low_final",
    qhigh_col: str = "yhat_q_high_final",
    k_grid: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Calibrate a multiplicative factor k_eff on the Validation split so that
    Coverage@target_coverage is as close as possible to target_coverage.

    Returns {
        'k_eff': float,
        'cov_val': float,    # achieved coverage on Val using k_eff
        'sharp_val': float,  # median width using k_eff
        'n_points': int,
    } plus 'grid' if needed for debugging.
    """
    # Prepare grid
    if k_grid is None:
        # Wide but reasonable search; includes identity (=1.0)
        k_grid = np.linspace(0.25, 3.50, 66)

    # Ensure ytrue is present
    df = _merge_ytrue_on_t(df_final_vt.copy(), full_history_df)
    if "split" not in df.columns or mean_col not in df.columns or df.empty:
        return {"k_eff": 1.0, "cov_val": np.nan, "sharp_val": np.nan, "n_points": 0}

    # Filter to validation rows
    g = df[df["split"].astype(str).str.lower().str.startswith("val")].copy()
    if g.empty or "ytrue" not in g.columns:
        log.warning("[calibration] No validation data (or ytrue missing). Falling back to k_eff=1.0")
        return {"k_eff": 1.0, "cov_val": np.nan, "sharp_val": np.nan, "n_points": 0}

    # For quantile mode, we do NOT rescale (quantiles are already empirical)
    if str(ci_mode).lower() == "quantile":
        cov, shp = _compute_coverage_and_sharpness(
            g, ci_mode="quantile", coverage=target_coverage,
            mean_col=mean_col, std_col=std_col, n_members_col=n_members_col,
            qlow_col=qlow_col, qhigh_col=qhigh_col
        )
        return {"k_eff": 1.0, "cov_val": cov, "sharp_val": shp, "n_points": int(g.shape[0])}

    # Grid-search on k multiplier
    best_k = 1.0
    best_err = float("inf")
    best_cov = np.nan
    best_shp = np.nan

    for k in k_grid:
        cov, shp = _compute_coverage_and_sharpness(
            g,
            ci_mode=ci_mode,
            coverage=target_coverage,
            mean_col=mean_col,
            std_col=std_col,
            n_members_col=n_members_col,
            qlow_col=qlow_col,
            qhigh_col=qhigh_col,
            override_k=_z_for_coverage(target_coverage) * float(k),
        )
        err = abs(cov - target_coverage) if np.isfinite(cov) else float("inf")
        # Tie-breaker: prefer sharper (narrower) bands
        if (err < best_err) or (np.isclose(err, best_err) and (np.isfinite(shp) and shp < best_shp)):
            best_err = err
            best_k = float(k)
            best_cov = cov
            best_shp = shp

    return {
        "k_eff": best_k,
        "cov_val": best_cov,
        "sharp_val": best_shp,
        "n_points": int(g.shape[0]),
    }



def _compute_bounds_for_mode(
    g: "pd.DataFrame",
    *,
    ci_mode: str,
    ci_k: float,
    ci_quantiles: Tuple[float, float],
    std_col: str,
    mean_col: str,
    n_members_col: str,
    qlow_col: Optional[str] = None,
    qhigh_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (lower, upper) arrays for the group g based on ci_mode.
    Fallbacks:
      - "sem" requires std_col and n_members_col; if n_members missing, falls back to "std".
      - "quantile" requires qlow_col/qhigh_col in g; if absent, falls back to "std".
    """
    mean = g[mean_col].to_numpy()

    mode = str(ci_mode or "std").lower()
    if mode not in {"std", "sem", "quantile"}:
        mode = "std"

    if mode == "quantile":
        ql, qh = qlow_col, qhigh_col
        if ql in g.columns and qh in g.columns:
            lower = g[ql].to_numpy()
            upper = g[qh].to_numpy()
            return lower, upper
        else:
            log.warning("[plot] quantile mode requested but quantile columns not found; falling back to 'std'.")
            mode = "std"

    if mode == "sem":
        if std_col in g.columns and n_members_col in g.columns:
            std = g[std_col].to_numpy()
            n   = g[n_members_col].astype(float).clip(lower=1.0).to_numpy()
            sem = std / np.sqrt(n)
            k   = float(ci_k)
            return mean - k * sem, mean + k * sem
        else:
            if std_col not in g.columns:
                log.warning("[plot] sem mode: '%s' not in columns; falling back to 'std'.", std_col)
            if n_members_col not in g.columns:
                log.warning("[plot] sem mode: '%s' not in columns; falling back to 'std'.", n_members_col)
            mode = "std"

    # default: STD mode
    if std_col not in g.columns:
        # No dispersion info: return degenerate band at the mean
        log.warning("[plot] std mode: '%s' not in columns; plotting mean line only.", std_col)
        return mean, mean
    std = g[std_col].to_numpy()
    k   = float(ci_k)
    return mean - k * std, mean + k * std


def _choose_scale_for_mode(g: pd.DataFrame, ci_mode: str, n_members_col: str, std_col: str) -> pd.Series:
    """Return the dispersion scale to use in gaussian fallback (STD or SEM)."""
    scale = g[std_col].astype(float)
    if ci_mode == "sem":
        n = g[n_members_col].astype(float).clip(lower=1.0) if n_members_col in g.columns else None
        if n is not None:
            scale = scale / np.sqrt(n)
    return scale


def summarize_split_metrics_std_or_sem(
    df_final_vt: pd.DataFrame,
    full_history_df: pd.DataFrame,
    *,
    ci_mode: str,
    coverage: float = 0.90,
    mean_col: str = "yhat_final_mean",
    std_col: str = "std_final",
    n_members_col: str = "n_members",
    override_k: Optional[float] = None,   # NEW
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Return two dicts with metrics for Validation and Test:
      - SMAPE (median %)
      - Cov@X (fraction)
      - Sharp@X (units of y)
    Uses override_k (if given) so metrics reflect calibrated bands.
    """
    if df_final_vt is None or df_final_vt.empty:
        return {}, {}

    # ensure ytrue is present
    df = _merge_ytrue_on_t(df_final_vt.copy(), full_history_df)
    if "split" not in df.columns or mean_col not in df.columns:
        return {}, {}

    def _one(split_key: str) -> Dict[str, float]:
        g = df[df["split"].astype(str).str.lower().str.startswith(split_key)].copy()
        if g.empty:
            return {}

        smp = np.nan
        if "ytrue" in g.columns:
            smp = _smape(g["ytrue"].to_numpy(), g[mean_col].to_numpy())

        cov, shp = _compute_coverage_and_sharpness(
            g,
            ci_mode=ci_mode,
            coverage=coverage,
            mean_col=mean_col,
            std_col=std_col,
            n_members_col=n_members_col,
            qlow_col="yhat_q_low_final",
            qhigh_col="yhat_q_high_final",
            override_k=override_k,  # propagate calibration
        )

        return {
            "SMAPE": smp,
            f"Cov@{int(coverage*100)}": cov,
            f"Sharp@{int(coverage*100)}": shp,
        }

    return _one("val"), _one("test")


def summarize_intra_metrics_by_arch(
    df_intra_vt: pd.DataFrame,
    full_history_df: pd.DataFrame,
    *,
    ci_mode: str,
    coverage: float = 0.90,
    mean_col: str = "yhat_family_mean",
    std_col: str = "std_family",
    n_members_col: str = "n_members",
    override_k: Optional[float] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute metrics per-arch and per split (Validation/Test).
    Returns: {arch: {"val": {...}, "test": {...}}}
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    if df_intra_vt is None or df_intra_vt.empty or "arch" not in df_intra_vt.columns:
        return out

    df = _merge_ytrue_on_t(df_intra_vt.copy(), full_history_df)
    for arch, g_arch in df.groupby("arch"):
        arch_res: Dict[str, Dict[str, float]] = {}

        for split_key, split_name in [("val", "val"), ("test", "test")]:
            g = g_arch[g_arch["split"].astype(str).str.lower().str.startswith(split_key)]
            if g.empty:
                continue

            smp = np.nan
            if "ytrue" in g.columns:
                smp = _smape(g["ytrue"].to_numpy(), g[mean_col].to_numpy())

            cov, shp = _compute_coverage_and_sharpness(
                g,
                ci_mode=ci_mode,
                coverage=coverage,
                mean_col=mean_col,
                std_col=std_col,
                n_members_col=n_members_col,
                qlow_col="q_lo_family",
                qhigh_col="q_hi_family",
                override_k=override_k,
            )

            arch_res[split_name] = {
                "SMAPE": smp,
                f"Cov@{int(coverage*100)}": cov,
                f"Sharp@{int(coverage*100)}": shp,
            }

        if arch_res:
            out[str(arch)] = arch_res

    return out


def compute_residuals_vt(df_vt: pd.DataFrame, mean_col: str) -> pd.Series:
    """
    Return residuals (ytrue - mean) for VAL/TEST rows (assumes 'ytrue' present).
    If ytrue missing, returns empty series.
    """
    if "ytrue" not in df_vt.columns or df_vt.empty:
        return pd.Series(dtype=float)
    return (df_vt["ytrue"] - df_vt[mean_col]).astype(float)