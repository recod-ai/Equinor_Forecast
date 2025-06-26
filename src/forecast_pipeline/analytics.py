"""forecast_pipeline.analytics
-------------------------------------------------
Funções utilitárias para cenários, envelopes, Monte
Carlo e acumulação.  E‑vitam que cada notebook
reimplemente a mesma lógica.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

try:
    # scipy é opcional para evitar dependência dura no core; se não houver,
    # usamos valores de z-score tabelados.
    from scipy.stats import norm
except ModuleNotFoundError:  # fallback mínimo
    norm = None  # type: ignore

__all__ = [
    "scenario_curve",
    "make_envelope",
    "mc_sample",
    "cumulate",
]


# ---------------------------------------------------------------------
# helpers --------------------------------------------------------------
# ---------------------------------------------------------------------
_Z_CACHE = {}


def _z_score(p: float) -> float:
    """Retorna z tal que Φ(z) = p (p between 0 and 1)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    if p in _Z_CACHE:
        return _Z_CACHE[p]
    if norm is not None:
        z = float(norm.ppf(p))
    else:  # tabela curta para p típicos
        table = {0.90: 1.2815516, 0.10: -1.2815516,
                 0.95: 1.6448536, 0.05: -1.6448536,
                 0.975: 1.9599639, 0.025: -1.9599639}
        if p not in table:
            raise RuntimeError("scipy missing and p not in fallback table")
        z = table[p]
    _Z_CACHE[p] = z
    return z


# ---------------------------------------------------------------------
# API pública ----------------------------------------------------------
# ---------------------------------------------------------------------

def scenario_curve(mu: np.ndarray, sigma: Optional[np.ndarray], p: float) -> np.ndarray:
    """Calcula a curva do percentil *p*.

    *Se ``sigma`` existir*:  μ + z σ.
    Caso contrário assume que ``mu`` é um *stack* de curvas (S,B,H)
    e devolve o percentil empírico ao longo do eixo 0.
    """
    z = _z_score(p)

    if sigma is not None:
        return mu + z * sigma

    # sem σ → usamos percentil dos snapshots
    if mu.ndim < 3:
        raise ValueError("Without sigma, mu must be shape (S,B,H)")
    return np.percentile(mu, p * 100.0, axis=0)


from common.seq_preprocessing import reconstruct_true_series

def _to_series(arr: np.ndarray) -> np.ndarray:
    """Converte (B,H) → série (L,) se preciso; caso contrário devolve 1-D."""
    return reconstruct_true_series(arr) if arr.ndim == 2 else arr.ravel()

def make_envelope(mu: np.ndarray,
                  sigma: Optional[np.ndarray],
                  p_lower: float,
                  p_upper: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna curvas inferior / superior (percentis) **já no mesmo comprimento**
    da série agregada.
    """
    # converte janelas → série longa
    mu_series = _to_series(mu)
    sig_series = _to_series(sigma) if sigma is not None else None

    lo = scenario_curve(mu_series, sig_series, p_lower)
    up = scenario_curve(mu_series, sig_series, p_upper)
    return lo, up



def mc_sample(mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
    """Gera *n* amostras Monte Carlo ~(μ, σ²).

    Retorna array (n,B,H).
    """
    if sigma is None:
        raise ValueError("mc_sample requires sigma array")
    if n <= 0:
        raise ValueError("n must be positive")
    eps = np.random.randn(n, *mu.shape)  # (n,B,H)
    return mu + eps * sigma


def cumulate(sequence: np.ndarray) -> np.ndarray:
    """Cumsum ao longo do último eixo (H)."""
    return np.cumsum(sequence, axis=-1)
