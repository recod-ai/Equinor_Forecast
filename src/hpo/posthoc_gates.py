"""
hpo.posthoc_gates

Gating / thresholding utilities used by the post-hoc distribution filter.

This module groups all threshold computation (quantiles, MAD bounds, valcum gate),
predicate composition, and vectorized application of gates. It is intentionally
stateless: it operates on DataFrames + PostHocConfig and returns derived tables
or annotated DataFrames, keeping selection/audit logic elsewhere.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from .posthoc_config import PostHocConfig



def apply_valcum_gate(df: pd.DataFrame, cfg: PostHocConfig) -> pd.DataFrame:
    """
    Per-well quantile band gate on val_smape_cum with adaptive q_low.

    Writes:
      - valcum_pass (bool)
      - _valcum_q_low_used, _valcum_q_high_used, _valcum_mode_used (audit breadcrumbs)
    """
    import logging
    import numpy as np
    import pandas as pd

    log = logging.getLogger(__name__)

    if df is None or df.empty or "val_smape_cum" not in df.columns:
        out = df.copy() if df is not None else pd.DataFrame()
        out["valcum_pass"] = True
        return out

    gate = dict(getattr(cfg, "valcum_gate", None) or {})

    mode        = str(gate.get("mode", "smooth")).lower()
    base_q_low  = float(gate.get("q_low", 0.08))
    q_high      = float(gate.get("q_high", 0.80))
    min_q_low   = float(gate.get("min_q_low", 0.02))
    cap_q_low   = float(gate.get("cap_q_low", 0.10))
    min_gap     = float(gate.get("min_gap", 0.05))
    smooth_ref  = float(gate.get("smooth_rcv_ref", 0.80))
    tail_bump   = float(gate.get("tail_bump", 0.01))
    tail_thresh = float(gate.get("tail_thresh", 1.60))
    well_col    = str(gate.get("well_col", getattr(cfg, "well_col", "well")))

    # Guards
    cap_q_low = max(min_q_low, min(cap_q_low, q_high - min_gap))
    base_q_low = float(np.clip(base_q_low, min_q_low, cap_q_low))

    out = df.copy()

    def _stats(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return dict(n=0, q05=np.nan, q25=np.nan, q50=np.nan, q75=np.nan, q95=np.nan,
                        iqr=np.nan, rcv=np.nan, tail=np.nan)
        q05, q25, q50, q75, q95 = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        iqr = float(q75 - q25)
        rcv = float(iqr / (abs(q50) + 1e-9))                 # robust CV
        tail = float((q95 - q50) / ((q50 - q05) + 1e-9))     # right-tail strength
        return dict(n=int(s.size), q05=q05, q25=q25, q50=q50, q75=q75, q95=q95, iqr=iqr, rcv=rcv, tail=tail)

    def _choose_q_low_rule(n: int, rcv: float, tail: float, base: float) -> float:
        ql = base
        if n >= 150 and rcv <= 0.45 and tail <= 1.40:
            ql = min(ql, 0.06)
        if n >= 300 and rcv <= 0.30 and tail <= 1.20:
            ql = min(ql, 0.05)
        if (rcv >= 1.00) or (tail >= 2.50):
            ql = max(ql, 0.09)
        elif (rcv >= 0.60) or (tail >= 1.80):
            ql = max(ql, 0.07)
        if n < 150:
            ql = max(ql, 0.08)
        return float(np.clip(ql, min_q_low, cap_q_low))

    def _choose_q_low_smooth(n: int, rcv: float, tail: float, base: float) -> float:
        if n <= 0 or not np.isfinite(rcv):
            ql = base
        else:
            alpha = float(np.clip(rcv / max(smooth_ref, 1e-9), 0.0, 1.0))
            ql = 0.05 + 0.05 * alpha
            if np.isfinite(tail) and tail > tail_thresh:
                ql += tail_bump
            if n < 150:
                ql = max(ql, 0.07)
        return float(np.clip(ql, min_q_low, cap_q_low))

    qlow_by_well: dict = {}

    for well_name, s in out.groupby(well_col, sort=False)["val_smape_cum"]:
        st = _stats(s)

        if mode == "strict_override":
            ql, chosen = base_q_low, "strict"
        elif mode == "rule":
            ql, chosen = _choose_q_low_rule(st["n"], st["rcv"], st["tail"], base_q_low), "rule"
        else:
            ql, chosen = _choose_q_low_smooth(st["n"], st["rcv"], st["tail"], base_q_low), "smooth"

        # Never touch q_high
        ql = min(ql, q_high - min_gap)
        qlow_by_well[well_name] = ql

        capped = (ql >= cap_q_low - 1e-12) or (ql <= min_q_low + 1e-12)
        log.info(
            "valcum_gate: well=%s mode=%s q_low=%.3f q_high=%.2f n=%d rCV=%s tail=%s%s",
            well_name, chosen, ql, q_high, int(st["n"]),
            (f"{st['rcv']:.3f}" if np.isfinite(st["rcv"]) else "nan"),
            (f"{st['tail']:.3f}" if np.isfinite(st["tail"]) else "nan"),
            (" [capped]" if capped else ""),
        )

    # Apply per-well mask (this fixes the s.name bug)
    out["valcum_pass"] = False
    for well_name, idx in out.groupby(well_col, sort=False).groups.items():
        s = pd.to_numeric(out.loc[idx, "val_smape_cum"], errors="coerce")
        ql = float(qlow_by_well.get(well_name, base_q_low))
        lo = s.quantile(ql)
        hi = s.quantile(q_high)
        out.loc[idx, "valcum_pass"] = s.between(lo, hi).fillna(False).astype(bool).values

    out["valcum_pass"] = out["valcum_pass"].astype(bool)
    out["_valcum_q_low_used"] = out[well_col].map(qlow_by_well)
    out["_valcum_q_high_used"] = float(q_high)
    out["_valcum_mode_used"] = str(mode)

    return out


def quantile_thresholds(df: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
    """
    Computes primary quantile cutoffs for each well and metric.

    - Skips missing metrics safely.
    - Anti-leak: skips test_* metrics.
    - Accepts empty/None cfg.primary_quantile (returns empty table).
    """
    import logging
    import pandas as pd

    well_col = getattr(cfg, "well_col", "well")
    out_cols = [well_col, "metric", "cutoff", "quantile"]

    if df is None or df.empty:
        return pd.DataFrame(columns=out_cols)

    pq = getattr(cfg, "primary_quantile", None) or {}
    if not isinstance(pq, dict) or not pq:
        return pd.DataFrame(columns=out_cols)

    results = []
    for metric, q in pq.items():
        metric = str(metric)
        if metric.startswith("test_"):
            logging.info("[quantile] ignoring test metric: %s", metric)
            continue
        if metric not in df.columns:
            logging.warning("[quantile] metric '%s' not in df; skipping", metric)
            continue

        q = float(q)

        s = pd.to_numeric(df[metric], errors="coerce")
        cutoffs = (
            pd.concat([df[[well_col]], s.rename(metric)], axis=1)
              .groupby(well_col, dropna=False)[metric]
              .quantile(q)
              .reset_index()
              .rename(columns={metric: "cutoff"})
        )
        cutoffs["metric"] = metric
        cutoffs["quantile"] = q
        results.append(cutoffs)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=out_cols)

def mad_guards(df: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
    """
    Computes robust upper/lower bounds using (optionally) one-sided and/or log MAD.

    - Anti-leak: skips test_* metrics.
    - Skips missing metric columns.
    - Uses cfg.mad_guard options: enabled/alpha/metrics/log/side.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import median_abs_deviation

    well_col = getattr(cfg, "well_col", "well")
    guard = getattr(cfg, "mad_guard", None) or {}
    if not guard.get("enabled", False):
        return pd.DataFrame(columns=[well_col, "metric", "lower_bound", "upper_bound"])

    alpha = float(guard.get("alpha", 1.0))
    metrics_for_mad = guard.get("metrics", getattr(cfg, "metrics", [])) or []
    metrics_for_mad = [m for m in metrics_for_mad if not str(m).startswith("test_")]

    use_log = bool(guard.get("log", True))
    side = str(guard.get("side", "right")).lower()  # "right" | "left" | "both"

    lib_isb = getattr(cfg, "lower_is_better", {}) or {}

    def _safe_mad(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if x.empty:
            return 0.0
        m = float(median_abs_deviation(x, scale="normal"))
        return m if (np.isfinite(m) and m > 0) else 1e-12

    def _mad_bounds(s: pd.Series, lower_is_better: bool) -> tuple[float, float]:
        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            return (-np.inf, np.inf)

        med = float(x.median())

        if side == "right":
            x_use = x[x >= med]
        elif side == "left":
            x_use = x[x <= med]
        else:
            x_use = x
        if x_use.empty:
            x_use = x

        if use_log and (x_use > -0.999999).all():
            x_tr = np.log1p(x_use.to_numpy())
            med_tr = float(np.median(x_tr))
            mad_tr = _safe_mad(pd.Series(x_tr))

            if lower_is_better:
                upper = np.expm1(med_tr + alpha * mad_tr)
                return (-np.inf, float(upper))
            lower = np.expm1(med_tr - alpha * mad_tr)
            return (float(lower), np.inf)

        mad = _safe_mad(x_use)
        if lower_is_better:
            return (-np.inf, float(med + alpha * mad))
        return (float(med - alpha * mad), np.inf)

    records = []
    for well, g in df.groupby(well_col, sort=False, dropna=False):
        for metric in metrics_for_mad:
            metric = str(metric)
            if metric not in g.columns:
                continue
            lower_is_better = bool(lib_isb.get(metric, True))
            lo, hi = _mad_bounds(g[metric], lower_is_better)
            records.append({well_col: well, "metric": metric, "lower_bound": lo, "upper_bound": hi})

    out = pd.DataFrame.from_records(records)
    return out if not out.empty else pd.DataFrame(columns=[well_col, "metric", "lower_bound", "upper_bound"])

def ensure_mad_for_metric(
    df: pd.DataFrame,
    mdf: pd.DataFrame | None,
    well_col: str,
    metric: str,
    lower_is_better: bool,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Ensure there is a MAD row per (well, metric).
    - If missing or non-finite in mdf, compute robust bounds and inject.
    - Returns a normalized mdf with columns: [well_col, metric, lower_bound, upper_bound].
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import median_abs_deviation

    metric = str(metric)

    def _safe_mad(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if x.empty:
            return 0.0
        m = float(median_abs_deviation(x, scale="normal"))
        if not np.isfinite(m) or m <= 0:
            return 1e-12
        return m

    if df is None or df.empty or metric not in df.columns or well_col not in df.columns:
        return mdf if mdf is not None else pd.DataFrame(columns=[well_col, "metric", "lower_bound", "upper_bound"])

    dfm = df[[well_col, metric]].copy()
    dfm[metric] = pd.to_numeric(dfm[metric], errors="coerce")
    dfm = dfm.dropna(subset=[metric])
    if dfm.empty:
        return mdf if mdf is not None else pd.DataFrame(columns=[well_col, "metric", "lower_bound", "upper_bound"])

    rows = []
    for w, g in dfm.groupby(well_col, sort=False, dropna=False):
        med = float(g[metric].median())
        mad = _safe_mad(g[metric])
        if lower_is_better:
            lo, hi = -np.inf, med + alpha * mad
        else:
            lo, hi = med - alpha * mad, np.inf

        # sanitize
        if lower_is_better and not np.isfinite(hi):
            hi = med
        if (not lower_is_better) and not np.isfinite(lo):
            lo = med

        rows.append({well_col: w, "metric": metric, "lower_bound": float(lo), "upper_bound": float(hi)})

    inj = pd.DataFrame(rows, columns=[well_col, "metric", "lower_bound", "upper_bound"])

    if mdf is None or mdf.empty:
        return inj

    # Normalize mdf cols
    m = mdf.copy()
    for c in [well_col, "metric", "lower_bound", "upper_bound"]:
        if c not in m.columns:
            m[c] = np.nan

    key = [well_col, "metric"]
    m = m.merge(inj, on=key, how="outer", suffixes=("_old", "_new"))

    def pick(old, new):
        old = pd.to_numeric(old, errors="coerce")
        new = pd.to_numeric(new, errors="coerce")
        return np.where(np.isfinite(old), old, new)

    m["lower_bound"] = pick(m["lower_bound_old"], m["lower_bound_new"])
    m["upper_bound"] = pick(m["upper_bound_old"], m["upper_bound_new"])

    return m[[well_col, "metric", "lower_bound", "upper_bound"]]



def thresholds_table(qdf: pd.DataFrame, mdf: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
    """
    Merge quantile and MAD thresholds into a single table for reporting.

    Robustness:
    - Uses cfg.well_col, not hardcoded 'well'.
    - Works if either side is empty.
    """
    import pandas as pd

    well_col = getattr(cfg, "well_col", "well")

    qdf = qdf if qdf is not None else pd.DataFrame()
    mdf = mdf if mdf is not None else pd.DataFrame()

    if qdf.empty and mdf.empty:
        return pd.DataFrame(columns=[well_col, "metric", "cutoff", "quantile", "lower_bound", "upper_bound"])
    if mdf.empty:
        return qdf.copy()
    if qdf.empty:
        return mdf.copy()

    return pd.merge(qdf, mdf, on=[well_col, "metric"], how="left")

def compose_predicates(
    qdf: pd.DataFrame,
    mdf: pd.DataFrame,
    cfg: "PostHocConfig",
) -> Dict[Tuple[str, str], Callable[[float], bool]]:
    """
    Build per-(well, metric) predicates from the UNION:
      - quantile cutoff + MAD bounds when both exist
      - quantile only when MAD missing
      - MAD only when quantile missing

    Anti-leak: ignores test_* metrics even if present in cfg.metrics.
    """
    import numpy as np
    import pandas as pd

    predicates: Dict[Tuple[str, str], Callable[[float], bool]] = {}

    well_col = getattr(cfg, "well_col", "well")
    lib_isb = getattr(cfg, "lower_is_better", {}) or {}

    # effective metrics (no test_*)
    cfg_metrics = [m for m in (getattr(cfg, "metrics", []) or []) if not str(m).startswith("test_")]

    # Filter qdf/mdf to metrics of interest
    if qdf is not None and not qdf.empty:
        qdf = qdf[qdf["metric"].isin(cfg_metrics)]
    if mdf is not None and not mdf.empty:
        mdf = mdf[mdf["metric"].isin(cfg_metrics)]

    q_pivot = (
        qdf.pivot(index=well_col, columns="metric", values="cutoff")
        if qdf is not None and not qdf.empty else pd.DataFrame()
    )
    m_lower = (
        mdf.pivot(index=well_col, columns="metric", values="lower_bound")
        if mdf is not None and not mdf.empty else pd.DataFrame()
    )
    m_upper = (
        mdf.pivot(index=well_col, columns="metric", values="upper_bound")
        if mdf is not None and not mdf.empty else pd.DataFrame()
    )

    wells = list(set(q_pivot.index) | set(m_lower.index) | set(m_upper.index))
    metrics = list((set(q_pivot.columns) | set(m_lower.columns) | set(m_upper.columns)) & set(cfg_metrics))

    for well in wells:
        for metric in metrics:
            lower_is_better = bool(lib_isb.get(metric, True))

            # Neutral quantile cutoff if missing:
            # - lower is better => +inf (no restriction)
            q = (np.inf if lower_is_better else -np.inf)
            if (well in q_pivot.index) and (metric in q_pivot.columns):
                q = q_pivot.loc[well, metric]

            # Neutral MAD bounds if missing
            m_lo, m_hi = -np.inf, np.inf
            if (well in m_lower.index) and (metric in m_lower.columns):
                m_lo = m_lower.loc[well, metric]
            if (well in m_upper.index) and (metric in m_upper.columns):
                m_hi = m_upper.loc[well, metric]

            if lower_is_better:
                hi = min(q, m_hi) if np.isfinite(q) else m_hi
                predicates[(well, metric)] = (lambda v, hi=hi: True if not np.isfinite(hi) else (v <= hi))
            else:
                lo = max(q, m_lo) if np.isfinite(q) else m_lo
                predicates[(well, metric)] = (lambda v, lo=lo: True if not np.isfinite(lo) else (v >= lo))

    return predicates

def apply_multi_metric_gates(
    df: pd.DataFrame,
    predicates: Dict[Tuple[str, str], Callable[[float], bool]],
    cfg: "PostHocConfig",
) -> pd.DataFrame:
    """
    Apply per-(well, metric) predicates.

    - Anti-leak: ignores test_* metrics.
    - Vectorized per-well; no row-wise apply(axis=1).
    - NaN is strict: fails the gate when a predicate exists.
    - Always writes passes_gates as bool.
    """
    import pandas as pd

    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        out["passes_gates"] = pd.Series(dtype=bool)
        return out

    well_col = getattr(cfg, "well_col", "well")
    cfg_metrics = getattr(cfg, "metrics", []) or []
    metrics = [m for m in cfg_metrics if (m in df.columns and not str(m).startswith("test_"))]

    out = df.copy()

    # If we can't gate, pass everything deterministically
    if (well_col not in out.columns) or (not metrics) or (not predicates):
        out["passes_gates"] = True
        out["passes_gates"] = out["passes_gates"].astype(bool)
        return out

    passes = pd.Series(True, index=out.index)

    # Group once by well (fast) and apply each metric's predicate only within the group
    for w, idx in out.groupby(well_col, sort=False).groups.items():
        # idx is an Index of row positions for this well
        for metric in metrics:
            pred = predicates.get((w, metric))
            if pred is None:
                continue  # no restriction for this (well, metric)

            s = out.loc[idx, metric]
            # strict NaN handling: NaN => fail if predicate exists
            metric_ok = s.map(pred)
            metric_ok = metric_ok.fillna(False).astype(bool)

            passes.loc[idx] &= metric_ok.values

    out["passes_gates"] = passes.astype(bool)
    return out

def log_gate_diagnostics(df: pd.DataFrame,
                         qdf: pd.DataFrame | None,
                         mdf: pd.DataFrame | None,
                         cfg: "PostHocConfig",
                         max_examples: int = 3) -> None:
    """
    Loga, por (well, metric):
      - quantile_cutoff
      - mad_lower/mad_upper
      - limites efetivos aplicados (lo/hi) segundo lower_is_better
      - contagens: pass_quantile, pass_mad, pass_effective, passes_gates (se existir), valcum_pass (se existir)
      - exemplos de conflitos (até max_examples de cada tipo)

    Uso: chame logo após calcular qdf/mdf e (opcionalmente) após marcar passes_gates/valcum_pass.
    """
    if df is None or df.empty:
        logging.info("[gates] DF vazio; nada a logar.")
        return

    well_col = getattr(cfg, "well_col", "well")
    mets = list(getattr(cfg, "metrics", []))
    lib = dict(getattr(cfg, "lower_is_better", {}) or {})
    # pivôs seguros
    if qdf is not None and not qdf.empty:
        qdf = qdf[qdf["metric"].isin(mets)]
        q_piv = qdf.pivot(index=well_col, columns="metric", values="cutoff")
    else:
        q_piv = pd.DataFrame()

    if mdf is not None and not mdf.empty:
        mdf = mdf[mdf["metric"].isin(mets)]
        lo_piv = mdf.pivot(index=well_col, columns="metric", values="lower_bound")
        hi_piv = mdf.pivot(index=well_col, columns="metric", values="upper_bound")
    else:
        lo_piv = pd.DataFrame()
        hi_piv = pd.DataFrame()

    wells = sorted(set(df[well_col].unique()))
    for w in wells:
        sub = df[df[well_col] == w]
        if sub.empty:
            continue
        for m in mets:
            if m not in sub.columns:
                continue
            lower_is_better = bool(lib.get(m, True))

            # pega quantil e MAD (ou padrões neutros)
            q_cut = (
                q_piv.loc[w, m] if (m in q_piv.columns and w in q_piv.index)
                else (np.inf if not lower_is_better else -np.inf)
            )
            m_lo = (
                lo_piv.loc[w, m] if (m in lo_piv.columns and w in lo_piv.index)
                else -np.inf
            )
            m_hi = (
                hi_piv.loc[w, m] if (m in hi_piv.columns and w in hi_piv.index)
                else np.inf
            )

            # máscaras por regra
            x = pd.to_numeric(sub[m], errors="coerce")
            if lower_is_better:
                pass_q = x <= q_cut if np.isfinite(q_cut) else pd.Series(True, index=sub.index)
                pass_m = x <= m_hi if np.isfinite(m_hi) else pd.Series(True, index=sub.index)
                eff_lo, eff_hi = -np.inf, (min(q_cut, m_hi) if np.isfinite(q_cut) else m_hi)
                pass_eff = x <= eff_hi if np.isfinite(eff_hi) else pd.Series(True, index=sub.index)
            else:
                pass_q = x >= q_cut if np.isfinite(q_cut) else pd.Series(True, index=sub.index)
                pass_m = x >= m_lo if np.isfinite(m_lo) else pd.Series(True, index=sub.index)
                eff_lo, eff_hi = (max(q_cut, m_lo) if np.isfinite(q_cut) else m_lo), np.inf
                pass_eff = x >= eff_lo if np.isfinite(eff_lo) else pd.Series(True, index=sub.index)

            # contagens
            n = len(sub)
            c_q   = int(pass_q.sum())
            c_m   = int(pass_m.sum())
            c_eff = int(pass_eff.sum())
            c_pg  = int(sub.get("passes_gates", pd.Series(True, index=sub.index)).sum()) if "passes_gates" in sub.columns else None
            c_vc  = int(sub.get("valcum_pass",  pd.Series(True, index=sub.index)).sum())   if "valcum_pass"  in sub.columns else None

            # conflito: passou quantil mas caiu no MAD
            ex_q_not_m = sub[pass_q & (~pass_m)]
            # conflito: passou MAD mas caiu no quantil
            ex_m_not_q = sub[pass_m & (~pass_q)]

            # origem das restrições
            has_q  = np.isfinite(q_cut)
            has_lo = np.isfinite(m_lo)
            has_hi = np.isfinite(m_hi)
            if has_q and (has_lo or has_hi):
                src = "both"
            elif has_q:
                src = "quantile"
            elif has_lo or has_hi:
                src = "mad"
            else:
                src = "none"

            logging.info(
                "[gates] well=%s metric=%s lib=%s src=%s "
                "q=%s mad=(%s,%s) eff=(%s,%s) "
                "n=%d pass_q=%d pass_m=%d pass_eff=%d%s%s",
                str(w), m, lower_is_better, src,
                (f"{q_cut:.6g}" if np.isfinite(q_cut) else "±inf"),
                (f"{m_lo:.6g}"  if np.isfinite(m_lo) else "-inf"),
                (f"{m_hi:.6g}"  if np.isfinite(m_hi) else "+inf"),
                ("-inf" if not np.isfinite(eff_lo) else f"{eff_lo:.6g}"),
                ("+inf" if not np.isfinite(eff_hi) else f"{eff_hi:.6g}"),
                n, c_q, c_m, c_eff,
                (f" passes_gates={c_pg}" if c_pg is not None else ""),
                (f" valcum_pass={c_vc}"  if c_vc is not None else "")
            )

            if len(ex_q_not_m) > 0:
                logging.info("[gates]   examples pass_quantile_but_fail_mad (top %d):", max_examples)
                cols_show = [well_col, m]
                logging.info("\n%s", ex_q_not_m[cols_show].head(max_examples).to_string(index=False))
            if len(ex_m_not_q) > 0:
                logging.info("[gates]   examples pass_mad_but_fail_quantile (top %d):", max_examples)
                cols_show = [well_col, m]
                logging.info("\n%s", ex_m_not_q[cols_show].head(max_examples).to_string(index=False))