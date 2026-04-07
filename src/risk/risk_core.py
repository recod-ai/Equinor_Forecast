# src/risk/risk_core.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("risk")

# Try to import alignment helpers for the aligned path (Stage 4 parity).
# The risk-simple path below does NOT require these.
try:
    from common.phase_viz_support import (
        align_with_t_factory,
        resolve_horizon_indices,
        slice_by_t,
        parse_split_selector,
        fallback_full_history_from_series,
        fallback_boundaries,
    )
    _HAVE_ALIGN = True
except Exception:
    _HAVE_ALIGN = False


from common.log_utils import log_block

def _log_risk_window_summary(*, well: str, arch: str, selector: str, horizon_days: int,
                             n_members: int, n_points_total: Optional[int] = None,
                             using_simple: bool = True,
                             offsets: Optional[dict] = None,
                             t_range: Optional[tuple] = None) -> None:
    title = "Risk Window — Effective Slice"
    lines = [
        f"Well/Arch: {well} / {arch}",
        f"Selector: {selector}   Horizon: {horizon_days}d",
        f"Path: {'simple(split+idx)' if using_simple else 'aligned(t)'}",
        f"Members: {n_members}" + (f"   Points: {n_points_total}" if n_points_total is not None else ""),
    ]
    if offsets:
        lines.append(f"Offsets: VAL_OFF={offsets.get('val')}  TEST_OFF={offsets.get('test')}")
    if t_range:
        lines.append(f"t-range after merge: {t_range[0]} → {t_range[1]}")
    log_block(title, lines, width=100)



# ============================================================================
#  A) SIMPLE RISK CORE (no 't' required) — recommended for CDF plots
# ============================================================================

def accumulate_members_simple(
    series_df: pd.DataFrame,
    *,
    well: str,
    arch: str,
    selector: str,            # "train" | "val" | "test" | "val+test"
    horizon_days: int,        # -1 = all
    member_id_col: str = "job_hash",
    yhat_col: str = "yhat",
) -> pd.DataFrame:
    """
    Accumulate per-member predictions directly from series_df using only split + idx
    (no 't' or alignment needed). This is the simplest, most robust path for risk CDF.

    Rules:
      - 'train'     : take TRAIN up to horizon_days (if >=0) else all TRAIN
      - 'val'       : take VAL up to horizon_days (if >=0) else all VAL
      - 'test'      : take TEST up to horizon_days (if >=0) else all TEST
      - 'val+test'  : sum(ALL VAL) + (TEST window per horizon_days)

    Returns: DataFrame ['job_hash','accum','n_points'].
    """
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["job_hash", "accum", "n_points"])

    need = {"split", "idx", member_id_col, yhat_col}
    if not need.issubset(series_df.columns):
        return pd.DataFrame(columns=["job_hash", "accum", "n_points"])

    s = series_df.copy()

    if "well" in s.columns:
        s = s[s["well"].astype(str) == str(well)]
    if "arch" in s.columns:
        s = s[s["arch"].astype(str).str.lower().eq(str(arch).lower())]
    if s.empty:
        return pd.DataFrame(columns=["job_hash", "accum", "n_points"])

    s["split"] = s["split"].astype(str)

    def _take(df: pd.DataFrame, split_name: str, H: int) -> pd.DataFrame:
        g = df[df["split"].str.lower().str.startswith(split_name)].copy()
        if g.empty:
            return g
        g = g.sort_values("idx")
        if H is not None and H >= 0:
            # Keep the first H rows per member_id within this split (by idx order)
            g = g.groupby(member_id_col, sort=False).head(H)
        return g

    sel = selector.lower().strip()
    H = int(horizon_days)

    parts = []
    if sel == "train":
        parts.append(_take(s, "train", H))
    elif sel == "val":
        parts.append(_take(s, "val", H))
    elif sel == "test":
        parts.append(_take(s, "test", H))
    elif sel in ("val+test", "validation+test", "valtest"):
        parts.append(_take(s, "val", -1))  # always full validation
        parts.append(_take(s, "test", H))  # horizon applies to test
    else:
        # Default: treat as 'val+test'
        parts.append(_take(s, "val", -1))
        parts.append(_take(s, "test", H))

    w = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if w.empty:
        log.warning(f"[risk] accumulate_members_simple: empty slice for well={well}, arch={arch}, "
                    f"selector={selector}, H={horizon_days}")
        return pd.DataFrame(columns=["job_hash", "accum", "n_points"])

    # LOG: contagem por split + head(H) efetivo
    cnt = w["split"].astype(str).str.lower().value_counts().to_dict()
    log.info(f"[risk] accumulate_members_simple: window splits={cnt} (selector={selector}, H={horizon_days})")

    agg = (
        w.groupby(member_id_col, sort=False)
         .agg(accum=(yhat_col, "sum"), n_points=("idx", "count"))
         .reset_index()
    )
    agg[member_id_col] = agg[member_id_col].astype(str)
    return agg.rename(columns={member_id_col: "job_hash"})[["job_hash", "accum", "n_points"]]


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically-stable softmax."""
    if temperature <= 0:
        temperature = 1.0
    z = (x - np.max(x)) / float(temperature)
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def weights_for_members(
    df_members: pd.DataFrame,
    *,
    strategy: str = "uniform",
    temperature: float = 0.5,
) -> pd.DataFrame:
    """
    Compute weights per member (job_hash).

    strategies:
      - "uniform": 1 / N for all.
      - "distance_softmax": Validation-only distance to the family mean. We compute
            per-time mean (using 't' if present, else 'idx'), then per-member mean
            absolute deviation; weights = softmax(-dist / temperature).

    Returns: ['job_hash','weight'] with sum(weight) ~= 1.0.
    """
    if df_members is None or df_members.empty or "job_hash" not in df_members.columns:
        return pd.DataFrame(columns=["job_hash", "weight"])

    uniq = df_members["job_hash"].astype(str).unique().tolist()
    if len(uniq) == 0:
        return pd.DataFrame(columns=["job_hash", "weight"])

    if str(strategy).lower() == "distance_softmax":
        if "split" not in df_members.columns:
            log.warning("[risk] 'distance_softmax' requires 'split'; falling back to uniform.")
        else:
            val = df_members[df_members["split"].astype(str).str.lower().str.startswith("val")]
            if not val.empty and "yhat" in val.columns:
                time_key = "t" if "t" in val.columns else ("idx" if "idx" in val.columns else None)
                if time_key is not None:
                    mu_t = val.groupby(time_key, sort=False)["yhat"].mean().rename("mu").reset_index()
                    v = val.merge(mu_t, on=time_key, how="left")
                    v["abs_err"] = (v["yhat"] - v["mu"]).abs()
                    dist = v.groupby("job_hash", sort=False)["abs_err"].mean().rename("dist").reset_index()
                    w = _softmax(-dist["dist"].to_numpy(dtype=float), temperature=float(temperature))
                    out = dist.assign(weight=w)[["job_hash", "weight"]]
                    s = float(out["weight"].sum())
                    if s > 0:
                        out["weight"] = out["weight"] / s
                    return out

    # Uniform fallback
    w = np.full(len(uniq), 1.0 / float(len(uniq)))
    return pd.DataFrame({"job_hash": uniq, "weight": w})


def weighted_quantiles(
    values: np.ndarray | Sequence[float],
    weights: Optional[np.ndarray | Sequence[float]] = None,
    qs: Sequence[float] = (0.1, 0.5, 0.9),
) -> Dict[float, float]:
    """
    Compute weighted quantiles for 0<=q<=1. If weights=None, falls back to unweighted.
    Stable: sorts by value, builds weighted CDF, and interpolates.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {float(q): np.nan for q in qs}

    if weights is None:
        return {float(q): float(np.quantile(x, q)) for q in qs}

    w = np.asarray(weights, dtype=float)
    if w.shape != x.shape:
        raise ValueError("weights and values must have the same shape")

    w = np.clip(w, a_min=0.0, a_max=None)
    sw = w.sum()
    if sw <= 0:
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / sw

    idx = np.argsort(x, kind="mergesort")
    x = x[idx]
    w = w[idx]
    cdf = np.cumsum(w)

    out: Dict[float, float] = {}
    for q in qs:
        qf = float(q)
        pos = np.searchsorted(cdf, qf, side="left")
        if pos == 0:
            out[qf] = float(x[0])
        elif pos >= len(x):
            out[qf] = float(x[-1])
        else:
            x0, x1 = x[pos - 1], x[pos]
            c0, c1 = cdf[pos - 1], cdf[pos]
            t = 0.0 if c1 == c0 else (qf - c0) / (c1 - c0)
            out[qf] = float(x0 + (x1 - x0) * t)
    return out


# ============================================================================
#  B) ALIGNED PATH (with 't') — kept for parity with Stage 4 (optional)
# ============================================================================

def build_member_series(
    series_df: pd.DataFrame,
    well: str,
    arch: str,
    *,
    full_history_df: Optional[pd.DataFrame] = None,
    boundaries: Optional[Dict[str, int]] = None,
    member_id_col: str = "job_hash",
    yhat_col: str = "yhat",
) -> pd.DataFrame:
    """
    Return per-member series aligned to the common time axis 't' using Stage 4 logic.
    Output: ['well','arch',member_id_col,'split','t',yhat_col]
    Robust to missing 't': builds 'global_idx' from (idx, split) with VAL/TEST offsets.
    """
    if not isinstance(series_df, pd.DataFrame) or series_df.empty:
        return pd.DataFrame(columns=["well", "arch", member_id_col, "split", "t", yhat_col])

    s = series_df.copy()
    if "well" in s.columns:
        s = s[s["well"].astype(str) == str(well)]
    if "arch" in s.columns:
        s = s[s["arch"].astype(str).str.lower().eq(str(arch).lower())]
    if s.empty:
        return pd.DataFrame(columns=["well", "arch", member_id_col, "split", "t", yhat_col])

    need_any_idx = ("idx" in s.columns) or ("global_idx" in s.columns) or ("t" in s.columns)
    if (member_id_col not in s.columns) or (yhat_col not in s.columns) or (not need_any_idx):
        return pd.DataFrame(columns=["well", "arch", member_id_col, "split", "t", yhat_col])

    s["split"] = s["split"].astype(str) if "split" in s.columns else "test"

    # If no alignment helpers are available, best effort: just keep what's there.
    if not _HAVE_ALIGN:
        keep = [c for c in ["well", "arch", member_id_col, "split", "t", yhat_col] if c in s.columns]
        out = s[keep].copy()
        return out.sort_values("t") if "t" in out.columns else out

    # Prepare full history & boundaries
    fh = full_history_df
    if fh is None or fh.empty or ("t" not in fh.columns):
        fh = fallback_full_history_from_series(series_df, well)
    b = boundaries
    if not isinstance(b, dict) or not {"train_end", "val_end", "test_start"} <= set(b.keys()):
        b = fallback_boundaries(well, fh)

    # If 't' equals 'idx' (numeric & equal), drop 't' to force proper alignment
    if "t" in s.columns and "idx" in s.columns:
        try:
            if pd.api.types.is_numeric_dtype(s["t"]) and pd.api.types.is_numeric_dtype(s["idx"]) and (s["t"] == s["idx"]).all():
                s = s.drop(columns=["t"])
        except Exception:
            pass

    # Build global_idx (when needed) using the same offsets as Stage 4
        # Build global_idx (when needed) using the same offsets as Stage 4
    if "t" not in s.columns:
        tr_end = int(b.get("train_end", 0))
        va_end = int(b.get("val_end", tr_end))
        te_sta = int(b.get("test_start", va_end + 1))

        VAL_OFFSET  = tr_end + 1
        TEST_OFFSET = max(va_end, te_sta) + 1

        # Split normalizado (tolerante a espaços/maiúsculas)
        split_norm = s["split"].astype(str).str.strip().str.lower()

        if "global_idx" not in s.columns and "idx" in s.columns:
            idx_i = s["idx"].astype(int)

            # Classificação robusta de split
            is_val  = split_norm.str.startswith("val")
            is_test = split_norm.str.startswith("test")

            s["global_idx"] = np.where(
                is_val,  idx_i + VAL_OFFSET,
                np.where(is_test, idx_i + TEST_OFFSET, idx_i)
            )

        if "global_idx" in s.columns:
            # LOG: estatísticas pré-merge
            gi_min, gi_max = int(s["global_idx"].min()), int(s["global_idx"].max())
            log.info(f"[risk] build_member_series: global_idx range before clamp: {gi_min}..{gi_max}  "
                     f"(VAL_OFF={VAL_OFFSET}, TEST_OFF={TEST_OFFSET})")

            # Clamp defensivo ao eixo do histórico
            hi = len(fh) - 1
            s["global_idx"] = s["global_idx"].clip(lower=0, upper=hi)

            # Mapa de t; prefira LEFT join + drop de NaN explícito para diagnosticar perdas
            t_map = fh.reset_index().rename(columns={"index": "global_idx"})[["global_idx", "t"]]

            before = len(s)
            s = s.merge(t_map, on="global_idx", how="left")
            lost = before - len(s.dropna(subset=["t"]))
            if lost > 0:
                log.warning(f"[risk] build_member_series: {lost} row(s) lost due to missing 't' after merge.")

            s = s.dropna(subset=["t"]).copy()

            # LOG: estatísticas pós-merge
            t_min = float(s["t"].min()) if not s.empty else np.nan
            t_max = float(s["t"].max()) if not s.empty else np.nan
            log.info(f"[risk] build_member_series: t range after merge: {t_min} → {t_max}")


    keep = [c for c in ["well", "arch", member_id_col, "split", "t", yhat_col] if c in s.columns]
    s = s[keep].copy()
    if "t" not in s.columns or s.empty:
        return pd.DataFrame(columns=["well", "arch", member_id_col, "split", "t", yhat_col])

    return s.sort_values("t")


def accumulate_window(
    df_members: pd.DataFrame,
    *,
    t_start: Any,
    t_end: Any,
    method: str = "sum",
) -> pd.DataFrame:
    """
    Accumulate predictions per member within [t_start, t_end] (inclusive).
    method:
      - "sum": simple sum (assumes regular cadence).
      - "trapz": trapezoidal integral over (t, yhat) for irregular sampling.
    Returns: ['job_hash','accum','n_points'].
    """
    if df_members is None or df_members.empty or "t" not in df_members.columns:
        return pd.DataFrame(columns=["job_hash", "accum", "n_points"])

    # Slice window by t
    if _HAVE_ALIGN:
        w = slice_by_t(df_members, t_start, t_end)
    else:
        # Best-effort slice if helpers are unavailable
        w = df_members[(df_members["t"] >= t_start) & (df_members["t"] <= t_end)].copy() if "t" in df_members.columns else df_members.copy()

    if w.empty:
        return pd.DataFrame(columns=["job_hash", "accum", "n_points"])

    out_rows = []
    for jh, g in w.groupby("job_hash", sort=False):
        g = g.sort_values("t")
        y = g["yhat"].to_numpy(dtype=float)
        if method == "trapz":
            x = pd.to_datetime(g["t"]).view(np.int64) if np.issubdtype(g["t"].dtype, np.datetime64) else g["t"].to_numpy(dtype=float)
            accum = float(np.trapz(y, x))
        else:
            accum = float(np.nansum(y))
        out_rows.append({"job_hash": str(jh), "accum": accum, "n_points": int(len(g))})

    return pd.DataFrame(out_rows)


def compute_risk_stats(
    df_members_aligned: pd.DataFrame,
    *,
    full_history_df: pd.DataFrame,
    boundaries: Dict[str, Any],
    selector: str,
    horizon_days: int,
    weighting: str = "uniform",
    distance_temp: float = 0.5,
) -> Dict[str, Any]:
    """
    Risk stats for a single (selector, horizon) window using the ALIGNED path (with 't').
    Kept for parity with Stage 4; the simple path (accumulate_members_simple) is recommended for CDF.
    """
    if (df_members_aligned is None) or df_members_aligned.empty or (not _HAVE_ALIGN):
        return {
            "q10": np.nan, "q50": np.nan, "q90": np.nan,
            "mean": np.nan, "min": np.nan, "max": np.nan,
            "n_members": 0, "n_points_window": 0,
            "t_start": None, "t_end": None,
        }

    # Resolve [start_idx, end_idx] on the t-axis and map to actual t values
    i0, i1 = resolve_horizon_indices(full_history_df, boundaries, selector, horizon_days)  # type: ignore[arg-type]
    t_axis = full_history_df["t"].reset_index(drop=True)
    if i0 > i1 or i0 >= len(t_axis):
        return {
            "q10": np.nan, "q50": np.nan, "q90": np.nan,
            "mean": np.nan, "min": np.nan, "max": np.nan,
            "n_members": int(df_members_aligned["job_hash"].nunique()),
            "n_points_window": 0,
            "t_start": None, "t_end": None,
        }
    t0, t1 = t_axis.iloc[i0], t_axis.iloc[i1]

    # Accumulate per member in the window
    acc = accumulate_window(df_members_aligned, t_start=t0, t_end=t1, method="sum")
    if acc.empty:
        return {
            "q10": np.nan, "q50": np.nan, "q90": np.nan,
            "mean": np.nan, "min": np.nan, "max": np.nan,
            "n_members": int(df_members_aligned["job_hash"].nunique()),
            "n_points_window": 0,
            "t_start": t0, "t_end": t1,
        }

    # Compute weights
    wts = weights_for_members(df_members_aligned, strategy=weighting, temperature=distance_temp)
    acc = acc.merge(wts, on="job_hash", how="left")
    if "weight" not in acc.columns or acc["weight"].isna().all():
        acc["weight"] = 1.0 / float(len(acc))
    else:
        s = float(acc["weight"].sum())
        if s > 0:
            acc["weight"] = acc["weight"] / s

    q = weighted_quantiles(acc["accum"].to_numpy(), acc["weight"].to_numpy(), qs=(0.1, 0.5, 0.9))
    mean = float(np.sum(acc["accum"].to_numpy() * acc["weight"].to_numpy()))

    return {
        "q10": q.get(0.1, np.nan),
        "q50": q.get(0.5, np.nan),
        "q90": q.get(0.9, np.nan),
        "mean": mean,
        "min": float(np.min(acc["accum"].to_numpy())) if len(acc) else np.nan,
        "max": float(np.max(acc["accum"].to_numpy())) if len(acc) else np.nan,
        "n_members": int(len(acc)),
        "n_points_window": int(acc["n_points"].max()) if len(acc) else 0,
        "t_start": t0,
        "t_end": t1,
    }
