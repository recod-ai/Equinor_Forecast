# src/forecast_pipeline/arps_canonical.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, Callable
import numpy as np

try:
    # Optional dependency; we fallback to grid/closed-form if not available
    import scipy.optimize as _opt
except Exception:  # pragma: no cover
    _opt = None

ArpsVariant = Literal["hyperbolic", "harmonic", "exponential"]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ArpsParams:
    """
    Fitted ARPS parameters (backward-compatible).
    - For piecewise models, `piecewise=True` and `left/right` carry sub-models.
    - `variant` is kept (often the right-most segment variant in piecewise mode).
    """
    variant: ArpsVariant
    qi: float
    D: float
    b: float = 1.0  # used only in hyperbolic; 1.0 = harmonic

    # --- Optional metadata (do not break callers that ignore them) ---
    weighting: str = "none"  # {"none","1_over_q2","time_decay"}
    loss: str = "wls"        # {"wls","huber","cauchy","quantile"}
    solver: str = "grid"     # {"grid","lbfgs","trust-constr"} (lbfgs needs SciPy)
    burn_in_fraction: float = 0.0

    # Piecewise (one change-point)
    piecewise: bool = False
    cp_index: Optional[int] = None
    left: Optional["ArpsParams"] = None
    right: Optional["ArpsParams"] = None


# =============================================================================
# Model registry (extensibility for alternative declines in the future)
# =============================================================================

# Predictors must be callables like: fn(t: np.ndarray, theta: ArpsParams) -> np.ndarray
_MODEL_REGISTRY: Dict[str, Callable[[np.ndarray, ArpsParams], np.ndarray]] = {}


def register_decline_model(name: str, predict_fn: Callable[[np.ndarray, ArpsParams], np.ndarray]) -> None:
    """Register a new decline model (future extension)."""
    _MODEL_REGISTRY[name] = predict_fn


# =============================================================================
# Base ARPS models and forecasting
# =============================================================================

def _q_exp(t: np.ndarray, qi: float, D: float) -> np.ndarray:
    return qi * np.exp(-D * t)


def _q_hyp(t: np.ndarray, qi: float, D: float, b: float) -> np.ndarray:
    # q(t) = qi / (1 + b*D*t)^(1/b)
    return qi / np.power(1.0 + b * D * t, 1.0 / b)


def arps_forecast(t: np.ndarray, theta: ArpsParams) -> np.ndarray:
    """
    Canonical ARPS forecast. If `theta.piecewise`, this returns the prediction
    of the *top-level* theta object (not composing both segments). For out-of-
    sample forecasting you should call `forecast_canonical_from_train`, which
    automatically uses the last (right) regime when piecewise=True.
    """
    # Allow plugged models via registry (future-proof)
    if theta.variant in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[theta.variant](t, theta)

    if theta.variant == "exponential":
        return _q_exp(t, theta.qi, theta.D)
    elif theta.variant == "harmonic":
        return _q_hyp(t, theta.qi, theta.D, 1.0)
    else:  # "hyperbolic"
        return _q_hyp(t, theta.qi, theta.D, theta.b)


# =============================================================================
# Helpers: weighting and robust losses
# =============================================================================

def _weights(kind: str, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    if kind == "1_over_q2":
        return 1.0 / np.maximum(q, 1e-6) ** 2
    elif kind == "time_decay":
        T = max(float(t[-1] - t[0]), 1.0)
        lam = 2.0 / T
        return np.exp(-lam * (t - t[0]))
    return np.ones_like(q)


def _pinball(e: np.ndarray, tau: float) -> np.ndarray:
    # Quantile/Pinball loss (tau in (0,1))
    return np.where(e >= 0.0, tau * e, (tau - 1.0) * e)


def _rho(resid: np.ndarray, *, loss: str, delta: float, tau: float) -> np.ndarray:
    """
    Robust loss function ρ(e):
      - wls:   e^2
      - huber: 0.5*e^2 if |e|<=δ else δ(|e| - 0.5δ)
      - cauchy: δ^2 * log(1 + (e/δ)^2)
      - quantile: pinball(e, tau)
    """
    if loss == "wls":
        return resid**2
    if loss == "huber":
        a = np.abs(resid)
        return np.where(a <= delta, 0.5 * resid**2, delta * (a - 0.5 * delta))
    if loss == "cauchy":
        return (delta**2) * np.log1p((resid / max(delta, 1e-12)) ** 2)
    if loss == "quantile":
        return _pinball(resid, tau=tau)
    # sane default
    return resid**2


def _robust_obj(
    q_hat: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
    *,
    loss: str,
    delta: float,
    tau: float,
    scale_y: float,
) -> float:
    """
    Weighted robust objective: sum_i w_i * rho( (qhat_i - q_i) / scale_y ).
    'scale_y' stabilizes the loss magnitude across wells/datasets.
    """
    e = (q_hat - q) / max(scale_y, 1e-12)
    return float(np.sum(w * _rho(e, loss=loss, delta=delta, tau=tau)))


def _scale_of(y: np.ndarray) -> float:
    """Robust scale estimate for the series (median absolute deviation proxy)."""
    med = np.median(y)
    return float(np.median(np.abs(y - med)) + 1e-12)


# =============================================================================
# Legacy closed-form fits (kept for robustness/fallback)
# =============================================================================

def _fit_exp_linear(t: np.ndarray, q: np.ndarray, w: Optional[np.ndarray]) -> Tuple[float, float]:
    """
    Exponential fit via weighted linear regression: log q = log qi - D t.
    Returns (qi, D). Used as fallback or when loss='wls' & solver='grid'.
    """
    mask = (q > 0) & np.isfinite(q)
    t, q = t[mask], q[mask]
    if w is None:
        w = np.ones_like(q)
    else:
        w = w[mask]

    y = np.log(q)
    X = np.stack([np.ones_like(t), -t], axis=1)
    # WLS via normal equations with diagonal weights
    W = np.diag(w / (np.max(w) + 1e-12))
    beta, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
    log_qi, D = beta[0], beta[1]
    qi = float(np.exp(log_qi))
    D = float(max(D, 1e-12))
    return qi, D


def _fit_hyperbolic_grid_b(
    t: np.ndarray,
    q: np.ndarray,
    w: Optional[np.ndarray],
    b_grid: np.ndarray,
) -> Tuple[float, float, float]:
    """
    For each fixed b, linearize q^{-b} = qi^{-b} (1 + b D t) = a + c t,
    solve via WLS and choose the b that minimizes SSE in the q domain.
    Returns (qi, D, b). Fallback when SciPy is not available.
    """
    mask = (q > 0) & np.isfinite(q)
    t, q = t[mask], q[mask]
    if w is None:
        w = np.ones_like(q)
    else:
        w = w[mask]

    best = (np.inf, None, None, None)  # (sse, qi, D, b)
    for b in b_grid:
        # y = q^{-b}
        y = np.power(q, -b)
        X = np.stack([np.ones_like(t), t], axis=1)
        W = np.diag(w / (np.max(w) + 1e-12))
        beta, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
        a, c = beta[0], beta[1]

        if a <= 0:
            continue
        qi = np.power(a, -1.0 / b)

        denom = (a * b)
        if denom <= 0:
            continue
        D = c / denom
        if not np.isfinite(D) or D <= 0:
            continue

        q_hat = _q_hyp(t, qi, D, b)
        sse = float(np.sum((q_hat - q) ** 2))
        if sse < best[0]:
            best = (sse, qi, D, b)

    if best[1] is None:
        # fallback: exponential
        qi_e, D_e = _fit_exp_linear(t, q, w)
        return qi_e, D_e, 1.0
    return best[1], best[2], best[3]


# =============================================================================
# Continuous optimization (SciPy) for hyperbolic/harmonic/exp
# =============================================================================

def _fit_by_lbfgs(
    t: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
    *,
    variant: ArpsVariant,
    loss: str,
    delta: float,
    tau: float,
) -> Tuple[float, float, float]:
    """
    Fit parameters by minimizing the robust objective with L-BFGS-B.
    Works for:
      - exponential: (qi, D), b ignored
      - harmonic:    (qi, D) with b=1.0
      - hyperbolic:  (qi, D, b) with bounds
    Returns (qi, D, b).
    """
    if _opt is None:
        raise RuntimeError("SciPy is not available for L-BFGS-B optimization.")

    scale_y = _scale_of(q)

    # Optimize in log-space for positivity; b also in log.
    def _eval_obj_exp(x):
        qi = np.exp(x[0])
        D = np.exp(x[1])
        qhat = _q_exp(t, qi, D)
        return _robust_obj(qhat, q, w, loss=loss, delta=delta, tau=tau, scale_y=scale_y)

    def _eval_obj_harm(x):
        qi = np.exp(x[0])
        D = np.exp(x[1])
        qhat = _q_hyp(t, qi, D, 1.0)
        return _robust_obj(qhat, q, w, loss=loss, delta=delta, tau=tau, scale_y=scale_y)

    def _eval_obj_hyp(x):
        qi = np.exp(x[0])
        D = np.exp(x[1])
        b = np.exp(x[2])
        qhat = _q_hyp(t, qi, D, b)
        return _robust_obj(qhat, q, w, loss=loss, delta=delta, tau=tau, scale_y=scale_y)

    # Initial guesses from simple moments (safe, scale-invariant-ish)
    qi0 = max(float(np.percentile(q, 95)), 1e-8)
    D0 = 1e-3
    b0 = 1.0

    if variant == "exponential":
        x0 = np.array([np.log(qi0), np.log(D0)], dtype=float)
        bounds = [(np.log(1e-8), np.log(1e8)), (np.log(1e-12), np.log(10.0))]
        res = _opt.minimize(_eval_obj_exp, x0, method="L-BFGS-B", bounds=bounds)
        qi = float(np.exp(res.x[0]))
        D = float(np.exp(res.x[1]))
        return qi, max(D, 1e-12), 1.0

    if variant == "harmonic":
        x0 = np.array([np.log(qi0), np.log(D0)], dtype=float)
        bounds = [(np.log(1e-8), np.log(1e8)), (np.log(1e-12), np.log(10.0))]
        res = _opt.minimize(_eval_obj_harm, x0, method="L-BFGS-B", bounds=bounds)
        qi = float(np.exp(res.x[0]))
        D = float(np.exp(res.x[1]))
        return qi, max(D, 1e-12), 1.0

    # hyperbolic
    x0 = np.array([np.log(qi0), np.log(D0), np.log(b0)], dtype=float)
    bounds = [
        (np.log(1e-8), np.log(1e8)),   # qi
        (np.log(1e-12), np.log(10.0)), # D
        (np.log(1e-3), np.log(5.0)),   # b
    ]
    res = _opt.minimize(_eval_obj_hyp, x0, method="L-BFGS-B", bounds=bounds)
    qi = float(np.exp(res.x[0]))
    D = float(np.exp(res.x[1]))
    b = float(np.exp(res.x[2]))
    return qi, max(D, 1e-12), min(max(b, 1e-3), 5.0)


# =============================================================================
# BIC scoring and piecewise (1 change-point)
# =============================================================================

def _bic_from_rss(n: int, rss: float, k_params: int) -> float:
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + k_params * np.log(max(n, 2))


def _segment_fit_and_rss(
    t: np.ndarray,
    q: np.ndarray,
    *,
    variant: ArpsVariant,
    weighting: str,
    loss: str,
    loss_delta: float,
    quantile_tau: float,
    solver: str,
    b_grid: Optional[np.ndarray],
) -> Tuple[ArpsParams, float]:
    """
    Fit a single segment and return (theta, RSS) where RSS is a weighted SSE proxy.
    """
    w = _weights(weighting, q, t)
    scale_y = _scale_of(q)

    # 🔒 STRICT GUARANTEE here as well (for piecewise path)
    if str(solver).lower() == "lbfgs" and _opt is None:
        raise RuntimeError(
            "Requested solver='lbfgs' inside piecewise fit but SciPy is not available. "
            "Install SciPy or set solver='grid'/'trust-constr'."
        )

    # --- Fit parameters ---
    if str(solver).lower() == "lbfgs" and _opt is not None:
        qi, D, b = _fit_by_lbfgs(
            t, q, w,
            variant=variant, loss=loss, delta=loss_delta, tau=quantile_tau
        )
    else:
        # Fallback (legacy behavior): grid/closed-form
        if variant == "exponential":
            qi, D = _fit_exp_linear(t, q, w)
            b = 1.0
        elif variant == "harmonic":
            qi, D, _ = _fit_hyperbolic_grid_b(t, q, w, np.array([1.0]))
            b = 1.0
        else:  # hyperbolic
            if b_grid is None:
                b_grid = np.unique(np.concatenate([
                    np.linspace(0.1, 2.0, 20),
                    np.array([1.0])
                ]))
            qi, D, b = _fit_hyperbolic_grid_b(t, q, w, b_grid)

    theta = ArpsParams(
        variant=variant, qi=float(max(qi, 1e-8)), D=float(max(D, 1e-12)),
        b=float(min(max(b, 1e-3), 5.0)),
        weighting=weighting, loss=loss, solver=solver
    )

    # --- Compute RSS in the original q-domain (weighted SSE) ---
    qhat = arps_forecast(t, theta)
    rss = float(np.sum(w * ((qhat - q) / max(scale_y, 1e-12))**2))
    return theta, rss


def _fit_piecewise_one_cp(
    t: np.ndarray,
    q: np.ndarray,
    *,
    base_variant: ArpsVariant,
    weighting: str,
    loss: str,
    loss_delta: float,
    quantile_tau: float,
    solver: str,
    b_grid: Optional[np.ndarray],
    min_seg_len: int,
    delta_bic_threshold: float = 2.0,
) -> ArpsParams:
    """
    One change-point piecewise ARPS:
      - Try cp in [min_seg_len, n-min_seg_len)
      - Fit left/right segments independently
      - Choose cp with minimum BIC; accept only if beats single-model BIC by Δ > threshold.
    Returns either a piecewise ArpsParams or a single-segment ArpsParams.
    """
    n = int(q.size)
    assert 2 * min_seg_len < n, "Not enough points for piecewise fitting."

    # Single-model fit (baseline BIC)
    theta_single, rss_single = _segment_fit_and_rss(
        t, q, variant=base_variant, weighting=weighting, loss=loss,
        loss_delta=loss_delta, quantile_tau=quantile_tau, solver=solver, b_grid=b_grid
    )
    k_single = 3 if base_variant == "hyperbolic" else 2
    bic_single = _bic_from_rss(n, rss_single, k_single)

    # Search best change-point
    best = (np.inf, None, None, None)  # (bic, cp, left, right)
    for cp in range(min_seg_len, n - min_seg_len):
        tL, qL = t[:cp], q[:cp]
        tR, qR = t[cp:], q[cp:]

        left_theta,  left_rss  = _segment_fit_and_rss(
            tL, qL, variant=base_variant, weighting=weighting, loss=loss,
            loss_delta=loss_delta, quantile_tau=quantile_tau, solver=solver, b_grid=b_grid
        )
        right_theta, right_rss = _segment_fit_and_rss(
            tR, qR, variant=base_variant, weighting=weighting, loss=loss,
            loss_delta=loss_delta, quantile_tau=quantile_tau, solver=solver, b_grid=b_grid
        )

        rss_tot = left_rss + right_rss
        k_tot = (3 if base_variant == "hyperbolic" else 2) * 2 + 1  # +1 for cp
        bic_tot = _bic_from_rss(n, rss_tot, k_tot)

        if bic_tot < best[0]:
            best = (bic_tot, cp, left_theta, right_theta)

    best_bic, cp_idx, left, right = best
    if (best_bic + delta_bic_threshold) < bic_single:
        # Accept piecewise
        # Use right variant as top-level marker (handy for logs)
        top_variant = right.variant if right is not None else base_variant
        return ArpsParams(
            variant=top_variant, qi=right.qi, D=right.D, b=right.b,
            weighting=weighting, loss=loss, solver=solver,
            piecewise=True, cp_index=int(cp_idx), left=left, right=right
        )

    # Otherwise, return single model
    return theta_single


# =============================================================================
# Public API (stable contract)
# =============================================================================

def fit_arps_canonical(
    q_train_phys: np.ndarray,
    *,
    variant: ArpsVariant = "hyperbolic",
    weighting: str = "none",
    b_grid: Optional[np.ndarray] = None,
    # New knobs (all optional, defaults keep legacy behavior)
    loss: Literal["wls", "huber", "cauchy", "quantile"] = "wls",
    loss_delta: float = 1.0,            # Huber/Cauchy scale
    quantile_tau: float = 0.5,          # for loss="quantile"
    burn_in_fraction: float = 0.0,      # 0.0 .. 0.2 reasonable
    solver: Literal["grid", "lbfgs", "trust-constr"] = "grid",
    piecewise: bool = False,
    min_segment_len: Optional[int] = None,   # if None, auto 10% of n (>=10)
    piecewise_min_delta_bic: float = 2.0,    # how much better the BIC must be
) -> ArpsParams:
    """
    Fits ARPS on the continuous (physical) TRAIN set. Time is absolute: t=0..n-1.

    Enhancements:
      - Robust losses (WLS/Huber/Cauchy/Quantile)
      - Optional burn-in trimming (discard early fraction)
      - Continuous optimization for hyperbolic/harmonic/exp (LBFGS if SciPy available)
      - Optional one-change-point piecewise fit selected by BIC

    The returned ArpsParams remains backward-compatible. When piecewise=True,
    `left`/`right` carry the sub-models and `cp_index` marks the change-point.
    """
    q = np.asarray(q_train_phys, dtype=float).copy()
    n = int(q.size)
    if n < 5:
        # too short: exponential fallback
        qi = float(max(q[0], 1e-6))
        return ArpsParams(variant="exponential", qi=qi, D=1e-3, b=1.0)

    # --- Burn-in trimming (no filtering; simply discards the leading fraction) ---
    burn_in_fraction = float(max(0.0, min(burn_in_fraction, 0.2)))
    start = int(np.floor(burn_in_fraction * n))
    t_full = np.arange(n, dtype=float)
    t = t_full[start:]
    q = q[start:]
    n_eff = int(q.size)

    if n_eff < 5:
        # fallback again if trimming left too few points
        qi = float(max(q_train_phys[-1], 1e-6))
        return ArpsParams(variant="exponential", qi=qi, D=1e-3, b=1.0)

    # --- Piecewise handling ---
    if piecewise:
        # Auto min segment length: 10% of effective n (at least 10)
        min_seg_len = int(min_segment_len or max(10, int(0.1 * n_eff)))
        if 2 * min_seg_len < n_eff:
            return _fit_piecewise_one_cp(
                t, q,
                base_variant=variant,
                weighting=weighting,
                loss=loss, loss_delta=loss_delta, quantile_tau=quantile_tau,
                solver=solver, b_grid=b_grid,
                min_seg_len=min_seg_len,
                delta_bic_threshold=piecewise_min_delta_bic
            )
        # not enough points for piecewise → fall through to single model

    # --- Single segment fit ---
    w = _weights(weighting, q, t)

    # 🔒 STRICT GUARANTEE: never silently degrade a requested solver
    if str(solver).lower() == "lbfgs" and _opt is None:
        raise RuntimeError(
            "Requested solver='lbfgs' but SciPy is not available. "
            "Install SciPy or set solver='grid'/'trust-constr'."
        )

    # Solver selection
    if str(solver).lower() == "lbfgs":
        qi, D, b = _fit_by_lbfgs(
            t, q, w,
            variant=variant, loss=loss, delta=loss_delta, tau=quantile_tau
        )
    else:
        # Legacy fallback: closed-form/grid
        if variant == "exponential":
            qi, D = _fit_exp_linear(t, q, w)
            b = 1.0
        elif variant == "harmonic":
            qi, D, _ = _fit_hyperbolic_grid_b(t, q, w, np.array([1.0]))
            b = 1.0
        else:  # hyperbolic
            if b_grid is None:
                b_grid = np.unique(np.concatenate([
                    np.linspace(0.1, 2.0, 20),
                    np.array([1.0])
                ]))
            qi, D, b = _fit_hyperbolic_grid_b(t, q, w, b_grid)

    # Sanitize and return
    return ArpsParams(
        variant=variant,
        qi=float(max(qi, 1e-8)),
        D=float(max(D, 1e-12)),
        b=float(min(max(b, 1e-3), 5.0)),
        weighting=weighting,
        loss=loss,
        solver=solver,
        burn_in_fraction=burn_in_fraction,
        piecewise=False,
    )



def forecast_canonical_from_train(
    theta: ArpsParams,
    train_len: int,
    val_len: int,
    test_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecasts continuously from the end of the training set using absolute time:
      t_val  = [train_len, ..., train_len+val_len-1]
      t_test = [train_len+val_len, ..., train_len+val_len+test_len-1]

    If `theta.piecewise=True`, we use the *right-most* regime parameters
    (post change-point) for out-of-sample times, which is the most standard
    assumption in production decline analysis.
    """
    # If piecewise, use the right-most segment for extrapolation
    base = theta.right if (theta.piecewise and theta.right is not None) else theta

    t_val  = np.arange(train_len, train_len + val_len, dtype=float)
    t_test = np.arange(train_len + val_len, train_len + val_len + test_len, dtype=float)

    yv = arps_forecast(t_val,  base)
    yt = arps_forecast(t_test, base)
    return yv, yt


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ArpsParams:
    """
    Fitted ARPS parameters (backward-compatible).
    - For piecewise models, `piecewise=True` and `left/right` carry sub-models.
    - `variant` is kept (often the right-most segment variant in piecewise mode).
    """
    variant: ArpsVariant
    qi: float
    D: float
    b: float = 1.0  # used only in hyperbolic; 1.0 = harmonic

    # --- Optional metadata (do not break callers that ignore them) ---
    weighting: str = "none"  # {"none","1_over_q2","time_decay"}
    loss: str = "wls"        # {"wls","huber","cauchy","quantile"}
    solver: str = "grid"     # {"grid","lbfgs","trust-constr"} (lbfgs needs SciPy)
    burn_in_fraction: float = 0.0

    # Piecewise (one change-point)
    piecewise: bool = False
    cp_index: Optional[int] = None
    left: Optional["ArpsParams"] = None
    right: Optional["ArpsParams"] = None


# =============================================================================
# Model registry (extensibility for alternative declines in the future)
# =============================================================================

# Predictors must be callables like: fn(t: np.ndarray, theta: ArpsParams) -> np.ndarray
_MODEL_REGISTRY: Dict[str, Callable[[np.ndarray, ArpsParams], np.ndarray]] = {}


def register_decline_model(name: str, predict_fn: Callable[[np.ndarray, ArpsParams], np.ndarray]) -> None:
    """Register a new decline model (future extension)."""
    _MODEL_REGISTRY[name] = predict_fn


# =============================================================================
# Base ARPS models and forecasting
# =============================================================================

def _q_exp(t: np.ndarray, qi: float, D: float) -> np.ndarray:
    return qi * np.exp(-D * t)


def _q_hyp(t: np.ndarray, qi: float, D: float, b: float) -> np.ndarray:
    # q(t) = qi / (1 + b*D*t)^(1/b)
    return qi / np.power(1.0 + b * D * t, 1.0 / b)


def arps_forecast(t: np.ndarray, theta: ArpsParams) -> np.ndarray:
    """
    Canonical ARPS forecast. If `theta.piecewise`, this returns the prediction
    of the *top-level* theta object (not composing both segments). For out-of-
    sample forecasting you should call `forecast_canonical_from_train`, which
    automatically uses the last (right) regime when piecewise=True.
    """
    # Allow plugged models via registry (future-proof)
    if theta.variant in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[theta.variant](t, theta)

    if theta.variant == "exponential":
        return _q_exp(t, theta.qi, theta.D)
    elif theta.variant == "harmonic":
        return _q_hyp(t, theta.qi, theta.D, 1.0)
    else:  # "hyperbolic"
        return _q_hyp(t, theta.qi, theta.D, theta.b)


# =============================================================================
# Helpers: weighting and robust losses
# =============================================================================

def _weights(kind: str, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    if kind == "1_over_q2":
        return 1.0 / np.maximum(q, 1e-6) ** 2
    elif kind == "time_decay":
        T = max(float(t[-1] - t[0]), 1.0)
        lam = 2.0 / T
        return np.exp(-lam * (t - t[0]))
    return np.ones_like(q)


def _pinball(e: np.ndarray, tau: float) -> np.ndarray:
    # Quantile/Pinball loss (tau in (0,1))
    return np.where(e >= 0.0, tau * e, (tau - 1.0) * e)


def _rho(resid: np.ndarray, *, loss: str, delta: float, tau: float) -> np.ndarray:
    """
    Robust loss function ρ(e):
      - wls:   e^2
      - huber: 0.5*e^2 if |e|<=δ else δ(|e| - 0.5δ)
      - cauchy: δ^2 * log(1 + (e/δ)^2)
      - quantile: pinball(e, tau)
    """
    if loss == "wls":
        return resid**2
    if loss == "huber":
        a = np.abs(resid)
        return np.where(a <= delta, 0.5 * resid**2, delta * (a - 0.5 * delta))
    if loss == "cauchy":
        return (delta**2) * np.log1p((resid / max(delta, 1e-12)) ** 2)
    if loss == "quantile":
        return _pinball(resid, tau=tau)
    # sane default
    return resid**2


def _robust_obj(
    q_hat: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
    *,
    loss: str,
    delta: float,
    tau: float,
    scale_y: float,
) -> float:
    """
    Weighted robust objective: sum_i w_i * rho( (qhat_i - q_i) / scale_y ).
    'scale_y' stabilizes the loss magnitude across wells/datasets.
    """
    e = (q_hat - q) / max(scale_y, 1e-12)
    return float(np.sum(w * _rho(e, loss=loss, delta=delta, tau=tau)))


def _scale_of(y: np.ndarray) -> float:
    """Robust scale estimate for the series (median absolute deviation proxy)."""
    med = np.median(y)
    return float(np.median(np.abs(y - med)) + 1e-12)


# =============================================================================
# Legacy closed-form fits (kept for robustness/fallback)
# =============================================================================

def _fit_exp_linear(t: np.ndarray, q: np.ndarray, w: Optional[np.ndarray]) -> Tuple[float, float]:
    """
    Exponential fit via weighted linear regression: log q = log qi - D t.
    Returns (qi, D). Used as fallback or when loss='wls' & solver='grid'.
    """
    mask = (q > 0) & np.isfinite(q)
    t, q = t[mask], q[mask]
    if w is None:
        w = np.ones_like(q)
    else:
        w = w[mask]

    y = np.log(q)
    X = np.stack([np.ones_like(t), -t], axis=1)
    # WLS via normal equations with diagonal weights
    W = np.diag(w / (np.max(w) + 1e-12))
    beta, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
    log_qi, D = beta[0], beta[1]
    qi = float(np.exp(log_qi))
    D = float(max(D, 1e-12))
    return qi, D


def _fit_hyperbolic_grid_b(
    t: np.ndarray,
    q: np.ndarray,
    w: Optional[np.ndarray],
    b_grid: np.ndarray,
) -> Tuple[float, float, float]:
    """
    For each fixed b, linearize q^{-b} = qi^{-b} (1 + b D t) = a + c t,
    solve via WLS and choose the b that minimizes SSE in the q domain.
    Returns (qi, D, b). Fallback when SciPy is not available.
    """
    mask = (q > 0) & np.isfinite(q)
    t, q = t[mask], q[mask]
    if w is None:
        w = np.ones_like(q)
    else:
        w = w[mask]

    best = (np.inf, None, None, None)  # (sse, qi, D, b)
    for b in b_grid:
        # y = q^{-b}
        y = np.power(q, -b)
        X = np.stack([np.ones_like(t), t], axis=1)
        W = np.diag(w / (np.max(w) + 1e-12))
        beta, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
        a, c = beta[0], beta[1]

        if a <= 0:
            continue
        qi = np.power(a, -1.0 / b)

        denom = (a * b)
        if denom <= 0:
            continue
        D = c / denom
        if not np.isfinite(D) or D <= 0:
            continue

        q_hat = _q_hyp(t, qi, D, b)
        sse = float(np.sum((q_hat - q) ** 2))
        if sse < best[0]:
            best = (sse, qi, D, b)

    if best[1] is None:
        # fallback: exponential
        qi_e, D_e = _fit_exp_linear(t, q, w)
        return qi_e, D_e, 1.0
    return best[1], best[2], best[3]


# =============================================================================
# Continuous optimization (SciPy) for hyperbolic/harmonic/exp
# =============================================================================

def _fit_by_lbfgs(
    t: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
    *,
    variant: ArpsVariant,
    loss: str,
    delta: float,
    tau: float,
) -> Tuple[float, float, float]:
    """
    Fit parameters by minimizing the robust objective with L-BFGS-B.
    Works for:
      - exponential: (qi, D), b ignored
      - harmonic:    (qi, D) with b=1.0
      - hyperbolic:  (qi, D, b) with bounds
    Returns (qi, D, b).
    """
    if _opt is None:
        raise RuntimeError("SciPy is not available for L-BFGS-B optimization.")

    scale_y = _scale_of(q)

    # Optimize in log-space for positivity; b also in log.
    def _eval_obj_exp(x):
        qi = np.exp(x[0])
        D = np.exp(x[1])
        qhat = _q_exp(t, qi, D)
        return _robust_obj(qhat, q, w, loss=loss, delta=delta, tau=tau, scale_y=scale_y)

    def _eval_obj_harm(x):
        qi = np.exp(x[0])
        D = np.exp(x[1])
        qhat = _q_hyp(t, qi, D, 1.0)
        return _robust_obj(qhat, q, w, loss=loss, delta=delta, tau=tau, scale_y=scale_y)

    def _eval_obj_hyp(x):
        qi = np.exp(x[0])
        D = np.exp(x[1])
        b = np.exp(x[2])
        qhat = _q_hyp(t, qi, D, b)
        return _robust_obj(qhat, q, w, loss=loss, delta=delta, tau=tau, scale_y=scale_y)

    # Initial guesses from simple moments (safe, scale-invariant-ish)
    qi0 = max(float(np.percentile(q, 95)), 1e-8)
    D0 = 1e-3
    b0 = 1.0

    if variant == "exponential":
        x0 = np.array([np.log(qi0), np.log(D0)], dtype=float)
        bounds = [(np.log(1e-8), np.log(1e8)), (np.log(1e-12), np.log(10.0))]
        res = _opt.minimize(_eval_obj_exp, x0, method="L-BFGS-B", bounds=bounds)
        qi = float(np.exp(res.x[0]))
        D = float(np.exp(res.x[1]))
        return qi, max(D, 1e-12), 1.0

    if variant == "harmonic":
        x0 = np.array([np.log(qi0), np.log(D0)], dtype=float)
        bounds = [(np.log(1e-8), np.log(1e8)), (np.log(1e-12), np.log(10.0))]
        res = _opt.minimize(_eval_obj_harm, x0, method="L-BFGS-B", bounds=bounds)
        qi = float(np.exp(res.x[0]))
        D = float(np.exp(res.x[1]))
        return qi, max(D, 1e-12), 1.0

    # hyperbolic
    x0 = np.array([np.log(qi0), np.log(D0), np.log(b0)], dtype=float)
    bounds = [
        (np.log(1e-8), np.log(1e8)),   # qi
        (np.log(1e-12), np.log(10.0)), # D
        (np.log(1e-3), np.log(5.0)),   # b
    ]
    res = _opt.minimize(_eval_obj_hyp, x0, method="L-BFGS-B", bounds=bounds)
    qi = float(np.exp(res.x[0]))
    D = float(np.exp(res.x[1]))
    b = float(np.exp(res.x[2]))
    return qi, max(D, 1e-12), min(max(b, 1e-3), 5.0)


# =============================================================================
# BIC scoring and piecewise (1 change-point)
# =============================================================================

def _bic_from_rss(n: int, rss: float, k_params: int) -> float:
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + k_params * np.log(max(n, 2))


def _segment_fit_and_rss(
    t: np.ndarray,
    q: np.ndarray,
    *,
    variant: ArpsVariant,
    weighting: str,
    loss: str,
    loss_delta: float,
    quantile_tau: float,
    solver: str,
    b_grid: Optional[np.ndarray],
) -> Tuple[ArpsParams, float]:
    """
    Fit a single segment and return (theta, RSS) where RSS is a weighted SSE proxy.
    """
    w = _weights(weighting, q, t)
    scale_y = _scale_of(q)

    # 🔒 STRICT GUARANTEE here as well (for piecewise path)
    if str(solver).lower() == "lbfgs" and _opt is None:
        raise RuntimeError(
            "Requested solver='lbfgs' inside piecewise fit but SciPy is not available. "
            "Install SciPy or set solver='grid'/'trust-constr'."
        )

    # --- Fit parameters ---
    if str(solver).lower() == "lbfgs" and _opt is not None:
        qi, D, b = _fit_by_lbfgs(
            t, q, w,
            variant=variant, loss=loss, delta=loss_delta, tau=quantile_tau
        )
    else:
        # Fallback (legacy behavior): grid/closed-form
        if variant == "exponential":
            qi, D = _fit_exp_linear(t, q, w)
            b = 1.0
        elif variant == "harmonic":
            qi, D, _ = _fit_hyperbolic_grid_b(t, q, w, np.array([1.0]))
            b = 1.0
        else:  # hyperbolic
            if b_grid is None:
                b_grid = np.unique(np.concatenate([
                    np.linspace(0.1, 2.0, 20),
                    np.array([1.0])
                ]))
            qi, D, b = _fit_hyperbolic_grid_b(t, q, w, b_grid)

    theta = ArpsParams(
        variant=variant, qi=float(max(qi, 1e-8)), D=float(max(D, 1e-12)),
        b=float(min(max(b, 1e-3), 5.0)),
        weighting=weighting, loss=loss, solver=solver
    )

    # --- Compute RSS in the original q-domain (weighted SSE) ---
    qhat = arps_forecast(t, theta)
    rss = float(np.sum(w * ((qhat - q) / max(scale_y, 1e-12))**2))
    return theta, rss


def _fit_piecewise_one_cp(
    t: np.ndarray,
    q: np.ndarray,
    *,
    base_variant: ArpsVariant,
    weighting: str,
    loss: str,
    loss_delta: float,
    quantile_tau: float,
    solver: str,
    b_grid: Optional[np.ndarray],
    min_seg_len: int,
    delta_bic_threshold: float = 2.0,
) -> ArpsParams:
    """
    One change-point piecewise ARPS:
      - Try cp in [min_seg_len, n-min_seg_len)
      - Fit left/right segments independently
      - Choose cp with minimum BIC; accept only if beats single-model BIC by Δ > threshold.
    Returns either a piecewise ArpsParams or a single-segment ArpsParams.
    """
    n = int(q.size)
    assert 2 * min_seg_len < n, "Not enough points for piecewise fitting."

    # Single-model fit (baseline BIC)
    theta_single, rss_single = _segment_fit_and_rss(
        t, q, variant=base_variant, weighting=weighting, loss=loss,
        loss_delta=loss_delta, quantile_tau=quantile_tau, solver=solver, b_grid=b_grid
    )
    k_single = 3 if base_variant == "hyperbolic" else 2
    bic_single = _bic_from_rss(n, rss_single, k_single)

    # Search best change-point
    best = (np.inf, None, None, None)  # (bic, cp, left, right)
    for cp in range(min_seg_len, n - min_seg_len):
        tL, qL = t[:cp], q[:cp]
        tR, qR = t[cp:], q[cp:]

        left_theta,  left_rss  = _segment_fit_and_rss(
            tL, qL, variant=base_variant, weighting=weighting, loss=loss,
            loss_delta=loss_delta, quantile_tau=quantile_tau, solver=solver, b_grid=b_grid
        )
        right_theta, right_rss = _segment_fit_and_rss(
            tR, qR, variant=base_variant, weighting=weighting, loss=loss,
            loss_delta=loss_delta, quantile_tau=quantile_tau, solver=solver, b_grid=b_grid
        )

        rss_tot = left_rss + right_rss
        k_tot = (3 if base_variant == "hyperbolic" else 2) * 2 + 1  # +1 for cp
        bic_tot = _bic_from_rss(n, rss_tot, k_tot)

        if bic_tot < best[0]:
            best = (bic_tot, cp, left_theta, right_theta)

    best_bic, cp_idx, left, right = best
    if (best_bic + delta_bic_threshold) < bic_single:
        # Accept piecewise
        # Use right variant as top-level marker (handy for logs)
        top_variant = right.variant if right is not None else base_variant
        return ArpsParams(
            variant=top_variant, qi=right.qi, D=right.D, b=right.b,
            weighting=weighting, loss=loss, solver=solver,
            piecewise=True, cp_index=int(cp_idx), left=left, right=right
        )

    # Otherwise, return single model
    return theta_single


# =============================================================================
# Public API (stable contract)
# =============================================================================

def fit_arps_canonical(
    q_train_phys: np.ndarray,
    *,
    variant: ArpsVariant = "hyperbolic",
    weighting: str = "none",
    b_grid: Optional[np.ndarray] = None,
    # New knobs (all optional, defaults keep legacy behavior)
    loss: Literal["wls", "huber", "cauchy", "quantile"] = "wls",
    loss_delta: float = 1.0,            # Huber/Cauchy scale
    quantile_tau: float = 0.5,          # for loss="quantile"
    burn_in_fraction: float = 0.0,      # 0.0 .. 0.2 reasonable
    solver: Literal["grid", "lbfgs", "trust-constr"] = "grid",
    piecewise: bool = False,
    min_segment_len: Optional[int] = None,   # if None, auto 10% of n (>=10)
    piecewise_min_delta_bic: float = 2.0,    # how much better the BIC must be
) -> ArpsParams:
    """
    Fits ARPS on the continuous (physical) TRAIN set. Time is absolute: t=0..n-1.

    Enhancements:
      - Robust losses (WLS/Huber/Cauchy/Quantile)
      - Optional burn-in trimming (discard early fraction)
      - Continuous optimization for hyperbolic/harmonic/exp (LBFGS if SciPy available)
      - Optional one-change-point piecewise fit selected by BIC

    The returned ArpsParams remains backward-compatible. When piecewise=True,
    `left`/`right` carry the sub-models and `cp_index` marks the change-point.
    """
    q = np.asarray(q_train_phys, dtype=float).copy()
    n = int(q.size)
    if n < 5:
        # too short: exponential fallback
        qi = float(max(q[0], 1e-6))
        return ArpsParams(variant="exponential", qi=qi, D=1e-3, b=1.0)

    # --- Burn-in trimming (no filtering; simply discards the leading fraction) ---
    burn_in_fraction = float(max(0.0, min(burn_in_fraction, 0.2)))
    start = int(np.floor(burn_in_fraction * n))
    t_full = np.arange(n, dtype=float)
    t = t_full[start:]
    q = q[start:]
    n_eff = int(q.size)

    if n_eff < 5:
        # fallback again if trimming left too few points
        qi = float(max(q_train_phys[-1], 1e-6))
        return ArpsParams(variant="exponential", qi=qi, D=1e-3, b=1.0)

    # --- Piecewise handling ---
    if piecewise:
        # Auto min segment length: 10% of effective n (at least 10)
        min_seg_len = int(min_segment_len or max(10, int(0.1 * n_eff)))
        if 2 * min_seg_len < n_eff:
            return _fit_piecewise_one_cp(
                t, q,
                base_variant=variant,
                weighting=weighting,
                loss=loss, loss_delta=loss_delta, quantile_tau=quantile_tau,
                solver=solver, b_grid=b_grid,
                min_seg_len=min_seg_len,
                delta_bic_threshold=piecewise_min_delta_bic
            )
        # not enough points for piecewise → fall through to single model

    # --- Single segment fit ---
    w = _weights(weighting, q, t)

    # 🔒 STRICT GUARANTEE: never silently degrade a requested solver
    if str(solver).lower() == "lbfgs" and _opt is None:
        raise RuntimeError(
            "Requested solver='lbfgs' but SciPy is not available. "
            "Install SciPy or set solver='grid'/'trust-constr'."
        )

    # Solver selection
    if str(solver).lower() == "lbfgs":
        qi, D, b = _fit_by_lbfgs(
            t, q, w,
            variant=variant, loss=loss, delta=loss_delta, tau=quantile_tau
        )
    else:
        # Legacy fallback: closed-form/grid
        if variant == "exponential":
            qi, D = _fit_exp_linear(t, q, w)
            b = 1.0
        elif variant == "harmonic":
            qi, D, _ = _fit_hyperbolic_grid_b(t, q, w, np.array([1.0]))
            b = 1.0
        else:  # hyperbolic
            if b_grid is None:
                b_grid = np.unique(np.concatenate([
                    np.linspace(0.1, 2.0, 20),
                    np.array([1.0])
                ]))
            qi, D, b = _fit_hyperbolic_grid_b(t, q, w, b_grid)

    # Sanitize and return
    return ArpsParams(
        variant=variant,
        qi=float(max(qi, 1e-8)),
        D=float(max(D, 1e-12)),
        b=float(min(max(b, 1e-3), 5.0)),
        weighting=weighting,
        loss=loss,
        solver=solver,
        burn_in_fraction=burn_in_fraction,
        piecewise=False,
    )



def forecast_canonical_from_train(
    theta: ArpsParams,
    train_len: int,
    val_len: int,
    test_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecasts continuously from the end of the training set using absolute time:
      t_val  = [train_len, ..., train_len+val_len-1]
      t_test = [train_len+val_len, ..., train_len+val_len+test_len-1]

    If `theta.piecewise=True`, we use the *right-most* regime parameters
    (post change-point) for out-of-sample times, which is the most standard
    assumption in production decline analysis.
    """
    # If piecewise, use the right-most segment for extrapolation
    base = theta.right if (theta.piecewise and theta.right is not None) else theta

    t_val  = np.arange(train_len, train_len + val_len, dtype=float)
    t_test = np.arange(train_len + val_len, train_len + val_len + test_len, dtype=float)

    yv = arps_forecast(t_val,  base)
    yt = arps_forecast(t_test, base)
    return yv, yt


# =============================================================================
# NEW: Ensemble utilities (pure ARPS path only; non-breaking)
# =============================================================================

from typing import Any, Dict, List, Optional, Tuple

from typing import Any, Dict, List, Optional, Tuple

def sample_arps_params(
    theta_hat: ArpsParams,
    *,
    K: int,
    seed: Optional[int],
    qi_rel_sigma: float = 0.05,
    D_rel_sigma: float = 0.10,
    b_abs_sigma: float = 0.10,
    b_clip_min: float = 1e-3,
    b_clip_max: float = 5.0,
    qi0_override: Optional[float] = None,
    D0_override: Optional[float] = None,
    b0_override: Optional[float] = None,

    # ---- NEW: anchor q at boundary instead of qi at t=0 ----
    q_anchor_phys: Optional[float] = None,   # q at t=t_anchor
    t_anchor: Optional[int] = None,          # typically train_len
) -> Tuple[List[ArpsParams], Dict[str, Any]]:
    K = int(max(1, K))
    rng = np.random.default_rng(int(seed) if seed is not None else None)

    qi0 = float(qi0_override) if (qi0_override is not None and np.isfinite(qi0_override)) else float(getattr(theta_hat, "qi", 1e-8))
    D0  = float(D0_override)  if (D0_override  is not None and np.isfinite(D0_override))  else float(getattr(theta_hat, "D",  1e-12))
    b0  = float(b0_override)  if (b0_override  is not None and np.isfinite(b0_override))  else float(getattr(theta_hat, "b",  1.0))

    qi0 = float(max(qi0, 1e-8))
    D0  = float(max(D0,  1e-12))

    qi_rel_sigma = float(max(0.0, qi_rel_sigma))
    D_rel_sigma  = float(max(0.0, D_rel_sigma))
    b_abs_sigma  = float(max(0.0, b_abs_sigma))

    # sample multipliers
    qi_mult = np.exp(rng.normal(loc=0.0, scale=qi_rel_sigma, size=K))
    D_mult  = np.exp(rng.normal(loc=0.0, scale=D_rel_sigma,  size=K))

    qi_s = qi0 * qi_mult
    D_s  = D0  * D_mult

    variant = getattr(theta_hat, "variant", "exponential")

    # b sampling (only for hyperbolic)
    if variant == "hyperbolic":
        if b_abs_sigma > 0:
            b_s = rng.normal(loc=b0, scale=b_abs_sigma, size=K).astype(float)
        else:
            b_s = np.full((K,), b0, dtype=float)
        b_s = np.clip(b_s, float(b_clip_min), float(b_clip_max))
    elif variant == "harmonic":
        b_s = np.full((K,), 1.0, dtype=float)
    else:
        b_s = np.full((K,), 1.0, dtype=float)

    # ---- NEW: enforce q(t_anchor)=q_anchor_phys by adjusting qi per member ----
    q_anchor_ok = (q_anchor_phys is not None) and np.isfinite(q_anchor_phys) and (float(q_anchor_phys) > 0) and (t_anchor is not None) and (int(t_anchor) >= 0)
    if q_anchor_ok:
        qa = float(q_anchor_phys)
        ta = float(int(t_anchor))
        if variant == "exponential":
            # q(t)=qi*exp(-D*t) => qi = q(t)*exp(D*t)
            qi_s = qa * np.exp(D_s * ta)
        elif variant in ("harmonic", "hyperbolic"):
            # q(t)=qi/(1+b*D*t)^(1/b)  (harmonic is b=1)
            # => qi = q(t) * (1+b*D*t)^(1/b)
            denom = 1.0 + (b_s * D_s * ta)
            denom = np.maximum(denom, 1e-12)
            qi_s = qa * np.power(denom, 1.0 / np.maximum(b_s, 1e-6))
        # safety
        qi_s = np.maximum(qi_s, 1e-8)

    samples: List[ArpsParams] = []
    for i in range(K):
        samples.append(ArpsParams(
            variant=variant,
            qi=float(max(qi_s[i], 1e-8)),
            D=float(max(D_s[i], 1e-12)),
            b=float(b_s[i]),
            weighting=getattr(theta_hat, "weighting", "none"),
            loss=getattr(theta_hat, "loss", "wls"),
            solver=getattr(theta_hat, "solver", "grid"),
            burn_in_fraction=getattr(theta_hat, "burn_in_fraction", 0.0),
            piecewise=False,
        ))

    meta = {
        "seed": None if seed is None else int(seed),
        "K": int(K),
        "variant": variant,
        "anchors_effective": {
            "qi0_base": float(qi0),
            "D0": float(D0),
            "b0": float(b0),
            "q_anchor_phys": None if q_anchor_phys is None else float(q_anchor_phys),
            "t_anchor": None if t_anchor is None else int(t_anchor),
            "q_anchor_applied": bool(q_anchor_ok),
        },
        "sampler": {
            "qi_rel_sigma": float(qi_rel_sigma),
            "D_rel_sigma": float(D_rel_sigma),
            "b_abs_sigma": float(b_abs_sigma),
            "b_clip_min": float(b_clip_min),
            "b_clip_max": float(b_clip_max),
        },
    }
    return samples, meta




def forecast_members_canonical_from_train(
    thetas: List[ArpsParams],
    *,
    train_len: int,
    val_len: int,
    test_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      members_val_phys:  (K, val_len)
      members_test_phys: (K, test_len)
    """
    K = int(len(thetas))
    if K <= 0:
        return np.zeros((0, int(val_len)), dtype=float), np.zeros((0, int(test_len)), dtype=float)

    t_val  = np.arange(int(train_len), int(train_len) + int(val_len), dtype=float)
    t_test = np.arange(int(train_len) + int(val_len), int(train_len) + int(val_len) + int(test_len), dtype=float)

    mv = np.zeros((K, int(val_len)), dtype=float)
    mt = np.zeros((K, int(test_len)), dtype=float)

    for i, th in enumerate(thetas):
        base = th.right if (getattr(th, "piecewise", False) and th.right is not None) else th
        mv[i, :] = arps_forecast(t_val, base).reshape(-1)
        mt[i, :] = arps_forecast(t_test, base).reshape(-1)

    return mv, mt


def _auc_rows(x_2d: np.ndarray) -> np.ndarray:
    x = np.asarray(x_2d, dtype=float)
    if x.ndim != 2 or x.size == 0:
        return np.asarray([], dtype=float)
    return np.nansum(x, axis=1)


def trim_members_by_auc(
    members_2d_phys: np.ndarray,
    *,
    trim_pct: float,
    min_keep: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Keep central band by AUC, symmetric trimming.
    """
    m = np.asarray(members_2d_phys, dtype=float)
    K = int(m.shape[0]) if m.ndim == 2 else 0
    min_keep = int(max(1, min_keep))
    trim_pct = float(0.0 if (not np.isfinite(trim_pct)) else max(0.0, min(0.49, trim_pct)))

    if K <= 0:
        return m, {"filter": "auc_trim", "K_in": 0, "K_out": 0, "reason": "empty"}

    auc = _auc_rows(m)
    if auc.size != K or (not np.any(np.isfinite(auc))):
        return m, {"filter": "auc_trim", "K_in": K, "K_out": K, "reason": "no_finite_auc"}

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

    meta = {
        "filter": "auc_trim",
        "K_in": int(K),
        "K_out": int(out.shape[0]),
        "trim_pct": float(trim_pct),
        "min_keep": int(min_keep),
        "keep_lo": int(lo),
        "keep_hi": int(hi),
    }
    return out, meta


def reduce_members_phys(
    members_2d_phys: np.ndarray,
    *,
    reducer: str,
) -> np.ndarray:
    m = np.asarray(members_2d_phys, dtype=float)
    r = str(reducer).strip().lower()
    if r in ("median", "p50"):
        return np.nanmedian(m, axis=0)
    if r in ("mean", "avg", "average"):
        return np.nanmean(m, axis=0)
    raise ValueError(f"Unknown reducer '{reducer}' (use 'median' or 'mean').")


def bands_phys(
    members_2d_phys: np.ndarray,
    *,
    percentiles: List[int],
) -> Dict[str, np.ndarray]:
    m = np.asarray(members_2d_phys, dtype=float)
    out: Dict[str, np.ndarray] = {}
    for p in percentiles:
        out[f"p{int(p)}"] = np.nanpercentile(m, float(p), axis=0)
    return out


def build_arps_ensemble_out_from_theta_hat(
    *,
    theta_hat: ArpsParams,
    train_len: int,
    val_len: int,
    test_len: int,
    scaler_target,
    cfg: Dict[str, Any],
    logger=None,

    # ---- NEW (non-breaking): allow history-based q0 anchor ----
    X_any: Any = None,
    scaler_X=None,
) -> Dict[str, Any]:
    """
    Build an 'ensemble_out'-like payload for the pure ARPS path.

    NEW behavior (simple, opt-in by cfg):
      - qi_anchor: "theta_hat" (default) or "history_q0_phys"
      - If "history_q0_phys", attempts to call compute_history_q0_phys(X_any, scaler_X, scaler_target, ...)
        and uses that as qi0_override for sampling.

    FIX:
      - theta_sampling=False now disables ALL perturbations (qi/D/b)
    """
    cfg = dict(cfg or {})

    # -----------------------------
    # helpers
    # -----------------------------
    def _to_bool(x, default=False):
        try:
            if isinstance(x, bool):
                return x
            if x is None:
                return default
            s = str(x).strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
            return default
        except Exception:
            return default

    def _to_int(x, default: int):
        try:
            if x is None:
                return int(default)
            return int(x)
        except Exception:
            return int(default)

    def _to_float(x, default: float):
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _norm_agg(x: Any) -> str:
        s = str(x or "").strip().lower()
        if s in ("p50", "median"):
            return "median"
        if s in ("mean", "avg", "average"):
            return "mean"
        return "median"

    def _log(logger, level: str, msg: str, *args):
        if logger is None:
            return
        try:
            fn = getattr(logger, level, None)
            if callable(fn):
                fn(msg, *args)
        except Exception:
            pass

    # -----------------------------
    # 0) Enabled gate
    # -----------------------------
    enabled = _to_bool(cfg.get("enabled", True), default=True)
    if not enabled:
        return {"val": {}, "test": {}, "meta": {"arps_ensemble": {"enabled": False}}}

    seed = cfg.get("seed", None)

    print(cfg)

    # K
    K = cfg.get("k", None)
    if K is None:
        K = cfg.get("K", cfg.get("members", cfg.get("coupling_spaghetti_k", 50)))
    K = int(max(1, _to_int(K, 50)))

    # agg/reducer
    reducer = cfg.get("agg", None)
    if reducer is None:
        reducer = cfg.get("reducer", "median")
    reducer = _norm_agg(reducer)

    # ---- sampling master switch ----
    theta_sampling = _to_bool(cfg.get("theta_sampling", True), default=True)

    # default interpretation is (qi,D lognormal rel; b normal abs)
    qi_rel_sigma = _to_float(cfg.get("qi_rel_sigma", 0.05), 0.05)
    D_rel_sigma  = _to_float(cfg.get("D_rel_sigma", 0.10), 0.10)

    # NEW theta sampling sigma (historical name in your snippet)
    theta_sampling_sigma = cfg.get("theta_sampling_sigma", None)

    # legacy explicit
    b_abs_sigma = cfg.get("b_abs_sigma", None)
    if b_abs_sigma is None:
        b_abs_sigma = cfg.get("b_sigma", None)

    # ---- IMPORTANT: theta_sampling=False disables ALL perturbations ----
    if not theta_sampling:
        qi_rel_sigma = 0.0
        D_rel_sigma  = 0.0
        b_abs_sigma  = 0.0

    # otherwise: resolve b sigma intelligently
    if theta_sampling:
        if b_abs_sigma is None:
            if theta_sampling_sigma is not None:
                b_abs_sigma = _to_float(theta_sampling_sigma, 0.05)
            else:
                # smart default: if hyperbolic, ensure non-zero shape diversity
                if getattr(theta_hat, "variant", None) == "hyperbolic":
                    b_abs_sigma = _to_float(cfg.get("b_abs_sigma_default_hyperbolic", 0.05), 0.05)
                else:
                    b_abs_sigma = 0.0

    b_abs_sigma = _to_float(b_abs_sigma, 0.0)

    # clipping: NEW dict or legacy scalars
    clip = cfg.get("theta_sampling_clip", None)
    if isinstance(clip, dict):
        b_clip_min = cfg.get("b_clip_min", clip.get("b_min", 1e-3))
        b_clip_max = cfg.get("b_clip_max", clip.get("b_max", 5.0))
    else:
        b_clip_min = cfg.get("b_clip_min", 1e-3)
        b_clip_max = cfg.get("b_clip_max", 5.0)

    b_clip_min = _to_float(b_clip_min, 1e-3)
    if b_clip_max is None:
        b_clip_max_eff = 1e9
    else:
        b_clip_max_eff = _to_float(b_clip_max, 5.0)

    # trimming
    outlier_filter = cfg.get("traj_outlier_filter", cfg.get("filter", "auc_trim"))
    outlier_filter = str(outlier_filter or "").strip().lower() or "auc_trim"
    trim_pct = _to_float(cfg.get("traj_outlier_trim_pct", cfg.get("trim_pct", 0.2)), 0.2)
    min_keep = _to_int(cfg.get("traj_outlier_min_keep", cfg.get("min_keep", 20)), 20)

    # emission
    emit_members = _to_bool(cfg.get("emit_members", True), default=True)
    emit_members_scaled = _to_bool(cfg.get("emit_members_scaled", emit_members), default=emit_members)
    emit_members_for_integrated_view = _to_bool(cfg.get("emit_members_for_integrated_view", True), default=True)

    percentiles = cfg.get("band_percentiles", [10, 50, 90])
    try:
        percentiles = [int(p) for p in list(percentiles)]
    except Exception:
        percentiles = [10, 50, 90]

    # -----------------------------
    # 1) qi anchor selection (NEW)
    # -----------------------------
    qi_anchor = str(cfg.get("qi_anchor", "theta_hat") or "theta_hat").strip().lower()
    qi0_override: Optional[float] = None
    q0_meta: Dict[str, Any] = {"mode": qi_anchor, "ok": False}

    if qi_anchor in ("history_q0_phys", "q0_phys", "history"):
        # best-effort import to avoid hard-coupling
        try:
            from forecast_pipeline.arps_offline import compute_history_q0_phys  # adjust if your project path differs
        except Exception:
            compute_history_q0_phys = None

        if compute_history_q0_phys is None:
            q0_meta.update({"ok": False, "reason": "compute_history_q0_phys_import_failed"})
        elif X_any is None or scaler_target is None:
            q0_meta.update({"ok": False, "reason": "missing_X_any_or_scaler_target"})
        else:
            try:
                q0_window = _to_int(cfg.get("q0_window", cfg.get("q0_phys_window", 30)), 30)
                q0_kind   = str(cfg.get("q0_kind", cfg.get("q0_phys_kind", "median")) or "median")
                q0_min    = _to_float(cfg.get("q0_min", cfg.get("min_q0_phys", 1e-6)), 1e-6)
                q0_channel = _to_int(cfg.get("q0_channel", cfg.get("channel", -1)), -1)

                qi_est = compute_history_q0_phys(
                    X_any=X_any,
                    scaler_X=scaler_X,
                    scaler_target=scaler_target,
                    window=int(q0_window),
                    kind=str(q0_kind),
                    min_q0_phys=float(q0_min),
                    channel=int(q0_channel),
                )
                
                # ---- robust unwrap (handles tuple/list/np scalars) ----
                if isinstance(qi_est, (tuple, list)):
                    # common patterns: (q0, meta) or (q0, something)
                    qi_est = qi_est[0] if len(qi_est) > 0 else np.nan
                
                # numpy scalar -> python float
                try:
                    qi_est = float(np.asarray(qi_est).reshape(-1)[0])
                except Exception:
                    qi_est = float(qi_est)

                if np.isfinite(qi_est) and qi_est > 0:
                    qi0_override = float(max(qi_est, 1e-8))
                    q0_meta.update({
                        "ok": True,
                        "q0_window": int(q0_window),
                        "q0_kind": str(q0_kind),
                        "q0_min": float(q0_min),
                        "q0_channel": int(q0_channel),
                        "qi0_override": float(qi0_override),
                    })
                else:
                    q0_meta.update({"ok": False, "reason": "non_finite_or_non_positive_q0", "value": qi_est})
            except Exception as ex:
                q0_meta.update({"ok": False, "reason": "compute_failed", "error": str(ex)})

        if not q0_meta.get("ok", False):
            _log(logger, "warning", "ARPS_ensemble qi_anchor=history_q0_phys failed; fallback to theta_hat.qi. meta=%s", q0_meta)

    # -----------------------------
    # 2) Sample thetas (physical params)
    # -----------------------------

    q_anchor_mode = str(cfg.get("qi_anchor", "theta_hat") or "theta_hat").strip().lower()

    q_anchor_phys = None
    t_anchor = None
    
    if q_anchor_mode in ("history_q0_phys", "q0_phys", "history"):
        # qi0_override já foi estimado (qi0_override = q do histórico, perto do boundary)
        if qi0_override is not None and np.isfinite(qi0_override) and qi0_override > 0:
            q_anchor_phys = float(qi0_override)
            t_anchor = int(train_len)   # boundary do forecast
            # IMPORTANT: não usar qi0_override como qi(t=0) nesse modo
            qi0_override_for_sampler = None
        else:
            qi0_override_for_sampler = qi0_override
    else:
        qi0_override_for_sampler = qi0_override

    
    thetas, sampler_meta = sample_arps_params(
        theta_hat,
        K=K,
        seed=seed,
        qi_rel_sigma=float(qi_rel_sigma),
        D_rel_sigma=float(D_rel_sigma),
        b_abs_sigma=float(b_abs_sigma),
        b_clip_min=float(b_clip_min),
        b_clip_max=float(b_clip_max_eff),
        qi0_override=qi0_override_for_sampler,
        q_anchor_phys=q_anchor_phys,
        t_anchor=t_anchor,
    )


    # -----------------------------
    # 3) Forecast members (physical)
    # -----------------------------
    mv_phys, mt_phys = forecast_members_canonical_from_train(
        thetas,
        train_len=int(train_len),
        val_len=int(val_len),
        test_len=int(test_len),
    )

    # -----------------------------
    # 4) Trimming (physical)
    # -----------------------------
    trim_meta_val = {"filter": "none"}
    trim_meta_test = {"filter": "none"}
    mv_keep = mv_phys
    mt_keep = mt_phys

    if outlier_filter in ("auc_trim", "auc", "trim_auc"):
        mv_keep, trim_meta_val = trim_members_by_auc(mv_phys, trim_pct=trim_pct, min_keep=min_keep)
        mt_keep, trim_meta_test = trim_members_by_auc(mt_phys, trim_pct=trim_pct, min_keep=min_keep)
    elif outlier_filter in ("none", "off", "false", ""):
        pass
    else:
        trim_meta_val = {"filter": outlier_filter, "reason": "unknown_filter_keep_all"}
        trim_meta_test = {"filter": outlier_filter, "reason": "unknown_filter_keep_all"}

    # -----------------------------
    # 5) Aggregate + bands (physical)
    # -----------------------------
    agg_val_phys  = reduce_members_phys(mv_keep, reducer=reducer).reshape(-1)
    agg_test_phys = reduce_members_phys(mt_keep, reducer=reducer).reshape(-1)

    bands_val_phys  = bands_phys(mv_keep, percentiles=percentiles)
    bands_test_phys = bands_phys(mt_keep, percentiles=percentiles)

    # -----------------------------
    # 6) Convert to scaled
    # -----------------------------
    def _scale_1d(arr_1d: np.ndarray) -> np.ndarray:
        a = np.asarray(arr_1d, dtype=float).reshape(-1, 1)
        return scaler_target.transform(a).reshape(-1)

    def _scale_2d(arr_2d: np.ndarray) -> np.ndarray:
        m = np.asarray(arr_2d, dtype=float)
        K2, H2 = m.shape
        flat = m.reshape(-1, 1)
        out = scaler_target.transform(flat).reshape(K2, H2)
        return out

    agg_val_scaled  = _scale_1d(agg_val_phys)
    agg_test_scaled = _scale_1d(agg_test_phys)

    bands_val_scaled  = {k: _scale_1d(v) for k, v in bands_val_phys.items()}
    bands_test_scaled = {k: _scale_1d(v) for k, v in bands_test_phys.items()}

    out: Dict[str, Any] = {
        "val": {
            "agg_scaled": agg_val_scaled,
            "bands_scaled": bands_val_scaled,
        },
        "test": {
            "agg_scaled": agg_test_scaled,
            "bands_scaled": bands_test_scaled,
        },
        "meta": {
            "split_recon_lengths": {"val": int(val_len), "test": int(test_len)},
            "arps_ensemble": {
                "enabled": True,
                "k": int(K),
                "agg": reducer,
                "theta_sampling": bool(theta_sampling),
                "theta_sampling_sigma": float(theta_sampling_sigma) if theta_sampling_sigma is not None else None,
                "qi_anchor": qi_anchor,
                "q0_meta": q0_meta,
                "sampler": sampler_meta,
                "trim_val": trim_meta_val,
                "trim_test": trim_meta_test,
                "cfg_effective": {
                    "seed": seed,
                    "k": int(K),
                    "agg": reducer,
                    "theta_sampling": bool(theta_sampling),
                    "qi_rel_sigma": float(qi_rel_sigma),
                    "D_rel_sigma": float(D_rel_sigma),
                    "b_abs_sigma": float(b_abs_sigma),
                    "b_clip_min": float(b_clip_min),
                    "b_clip_max": None if b_clip_max is None else float(_to_float(b_clip_max, b_clip_max_eff)),
                    "traj_outlier_filter": outlier_filter,
                    "traj_outlier_trim_pct": float(trim_pct),
                    "traj_outlier_min_keep": int(min_keep),
                    "band_percentiles": list(percentiles),
                    "qi_anchor": qi_anchor,
                },
            },
        },
    }

    # members_scaled (optional heavy)
    if emit_members_scaled:
        mv_keep_scaled = _scale_2d(mv_keep)
        mt_keep_scaled = _scale_2d(mt_keep)
        out["val"]["members_scaled"] = mv_keep_scaled
        out["test"]["members_scaled"] = mt_keep_scaled

        if emit_members_for_integrated_view:
            out["meta"]["integrated_view_val_members_scaled"] = mv_keep_scaled
            out["meta"]["integrated_view_test_members_scaled"] = mt_keep_scaled

    _log(
        logger,
        "info",
        "ARPS_ensemble built k=%d agg=%s theta_sampling=%s qi_anchor=%s qi0_override=%s val_members=%s test_members=%s",
        int(K),
        reducer,
        bool(theta_sampling),
        qi_anchor,
        None if qi0_override is None else float(qi0_override),
        None if out["val"].get("members_scaled") is None else tuple(out["val"]["members_scaled"].shape),
        None if out["test"].get("members_scaled") is None else tuple(out["test"]["members_scaled"].shape),
    )

    return out


