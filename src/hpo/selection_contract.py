# src/hpo/selection_contract.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Canonical schema (fixed output contract)
# -----------------------------------------------------------------------------

CANONICAL_SUMMARY_COLUMNS: List[str] = [
    "dataset",
    "well",
    "architecture",
    "score_col",
    "scoring_strategy",
    "selection_source",
    "chosen_job_hash",
    "chosen_trial_id",
    "chosen_strategy_display",
    "chosen_val_smape_agg",
    "chosen_val_smape_cum",
    "chosen_score_value",
    "chosen_test_smape_agg",
    "pool_size",
    "pool_best_val_smape_agg",
    "pool_best_test_smape_agg",
    "regret_test",
    "ratio_test",
    "chosen_test_percentile",
    # ✅ NEW (audit-only)
    "val_test_spearman",
]


# Minimal diagnostics schema for contract stability.
# Neighborhood selector will populate richer diagnostics later.
CANONICAL_DIAGNOSTICS_COLUMNS: List[str] = [
    "dataset",
    "well",
    "architecture",
    "selector_mode",
    "selection_path",
    "selection_source",
    "score_col",
    "scoring_strategy",
    "pool_method",
    "pool_size",
    "notes",
]

# src/hpo/selection_contract.py

CANONICAL_THRESHOLDS_COLUMNS: List[str] = [
    "well",
    "metric",
    "cutoff",
    "quantile",
    "lower_bound",
    "upper_bound",
]


# -----------------------------------------------------------------------------
# Schema helpers
# -----------------------------------------------------------------------------

def _ensure_cols(df: Optional[pd.DataFrame], cols: Sequence[str]) -> pd.DataFrame:
    """Ensure a DataFrame contains all requested columns (fill missing with NaN)."""
    if df is None:
        df = pd.DataFrame()
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def ensure_thresholds_schema(thresholds_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = _ensure_cols(thresholds_df, CANONICAL_THRESHOLDS_COLUMNS)
    return out.reindex(columns=CANONICAL_THRESHOLDS_COLUMNS)

def empty_thresholds() -> pd.DataFrame:
    """Return an empty thresholds DataFrame with a stable schema."""
    return ensure_thresholds_schema(pd.DataFrame())


def ensure_summary_schema(summary_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Enforce the canonical summary schema:
      - adds missing columns with NaN
      - reorders columns
    """
    out = _ensure_cols(summary_df, CANONICAL_SUMMARY_COLUMNS)
    return out.reindex(columns=CANONICAL_SUMMARY_COLUMNS)


def ensure_diagnostics_schema(diagnostics_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Enforce the canonical diagnostics schema:
      - adds missing columns with NaN
      - reorders columns
    """
    out = _ensure_cols(diagnostics_df, CANONICAL_DIAGNOSTICS_COLUMNS)
    return out.reindex(columns=CANONICAL_DIAGNOSTICS_COLUMNS)


def empty_summary() -> pd.DataFrame:
    """Return an empty canonical summary DataFrame."""
    return ensure_summary_schema(pd.DataFrame())


def empty_diagnostics() -> pd.DataFrame:
    """Return an empty canonical diagnostics DataFrame."""
    return ensure_diagnostics_schema(pd.DataFrame())


def empty_top_performers() -> pd.DataFrame:
    """Return an empty top_performers DataFrame."""
    return pd.DataFrame()


def _ensure_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure meta is always present and has a minimal stable shape.
    NOTE: We do not enforce all target fields here yet (patches later),
    but we guarantee a dict exists and core keys are present.
    """
    out: Dict[str, Any] = dict(meta or {})
    out.setdefault("selector_mode", "unknown")
    out.setdefault("selection_path", "unknown")
    out.setdefault("selection_col", out.get("score_col", "unknown"))
    out.setdefault("pool_method", "unknown")
    out.setdefault("score_direction", "unknown")
    out.setdefault("anti_leak_policy", "unknown")
    return out


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if np.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def _arch_group_value(x: Any) -> str:
    """
    Coarse grouping label used ONLY for audit pooling (best/regret/spearman),
    so Darts subtypes don't fragment the "family" pool.
    Does NOT affect selection itself.
    """
    s = "" if x is None else str(x)
    s0 = s.strip().lower().replace(" ", "")
    if "darts" in s0:
        return "DARTS"
    if "arps" in s0:
        return "ARPS"
    if "seq2" in s0 or "pinn" in s0:
        return "PINN"
    return s.strip() or "UNKNOWN"


# -----------------------------------------------------------------------------
# Canonical summary builder
# -----------------------------------------------------------------------------

# def build_canonical_summary(
#     *,
#     master_df: pd.DataFrame,
#     top_performers: pd.DataFrame,
#     score_col: str,
#     scoring_strategy: str,
#     selection_source: str = "run_distribution_filter",
#     group_cols: Optional[Sequence[str]] = None,
#     # Metrics (audit-only)
#     val_metric: str = "val_smape_agg",
#     val_cum_metric: str = "val_smape_cum",
#     test_metric: str = "test_smape_agg",
# ) -> pd.DataFrame:
#     """
#     Build a canonical per-group summary table with a fixed schema.

#     IMPORTANT:
#     - Pure post-processing step; must NOT change selection.
#     - TEST metrics are audit-only (regret/ratio/percentile + spearman).
#     """
#     import numpy as np
#     import pandas as pd
#     from typing import Any, Dict, List, Optional, Sequence

#     if master_df is None:
#         master_df = pd.DataFrame()
#     if top_performers is None:
#         top_performers = pd.DataFrame()

#     # -------------------------
#     # Grouping resolution
#     # -------------------------
#     if group_cols is None:
#         group_cols = []
#         for c in ("dataset", "well", "architecture"):
#             if c in master_df.columns or c in top_performers.columns:
#                 group_cols.append(c)
#         if not group_cols:
#             group_cols = ["well"] if "well" in (master_df.columns.tolist() + top_performers.columns.tolist()) else []

#     if top_performers.empty or not group_cols:
#         return empty_summary()

#     # Identity columns in champions
#     job_hash_col = _first_existing_col(top_performers, ["job_hash", "trial_hash", "trial_id"])
#     trial_id_col = _first_existing_col(top_performers, ["trial", "trial_id", "number", "optuna_trial_number"])
#     strat_col = _first_existing_col(top_performers, ["strategy_display", "physics_strategy", "variant", "strategy"])

#     # Ensure group cols exist
#     for c in group_cols:
#         if c not in master_df.columns:
#             master_df = master_df.copy()
#             master_df[c] = "unknown"
#         if c not in top_performers.columns:
#             top_performers = top_performers.copy()
#             top_performers[c] = "unknown"

#     # -------------------------
#     # Helpers (local, compact)
#     # -------------------------
#     def _num(s: Any) -> float:
#         return _safe_float(s)

#     def _spearman_from_two_cols(pool: pd.DataFrame, a: str, b: str) -> float:
#         """Spearman rank correlation of a vs b in pool. Returns NaN if not enough data."""
#         if a not in pool.columns or b not in pool.columns:
#             return float("nan")
#         x = pd.to_numeric(pool[a], errors="coerce")
#         y = pd.to_numeric(pool[b], errors="coerce")
#         m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
#         if int(m.sum()) < 3:
#             return float("nan")
#         rx = x[m].rank(method="average", ascending=True)
#         ry = y[m].rank(method="average", ascending=True)
#         try:
#             return float(rx.corr(ry, method="pearson"))
#         except Exception:
#             return float("nan")

#     # -------------------------
#     # Reattach TEST to champions if missing (LEGACY fix)
#     # -------------------------
#     tp = top_performers.copy()

#     need_test = (test_metric not in tp.columns) or tp[test_metric].isna().all()
#     have_test_in_master = (test_metric in master_df.columns) and master_df[test_metric].notna().any()

#     if need_test and have_test_in_master:
#         # Prefer stable join keys (in order)
#         key_sets = [
#             [*group_cols, "job_hash"],
#             [*group_cols, "trial_hash"],
#             [*group_cols, "trial_id"],
#             [*group_cols, "optuna_trial_number"],
#             [*group_cols, "trial"],
#             [*group_cols, "number"],
#         ]
#         join_keys: List[str] = []
#         for ks in key_sets:
#             if all((k in tp.columns) and (k in master_df.columns) for k in ks):
#                 join_keys = ks
#                 break

#         # Fallback: join by group + (strategy + hparams) when available (still relatively safe for your HPO)
#         if not join_keys:
#             fallback_keys = [*group_cols]
#             for k in ["physics_strategy", "epochs", "batch_size", "learning_rate", "data_sample", "lag_window", "horizon"]:
#                 if k in tp.columns and k in master_df.columns:
#                     fallback_keys.append(k)
#             if len(fallback_keys) > len(group_cols):
#                 join_keys = fallback_keys

#         if join_keys:
#             lookup_cols = list(dict.fromkeys(join_keys + [test_metric]))
#             test_lookup = master_df[lookup_cols].dropna(subset=[test_metric]).drop_duplicates(subset=join_keys, keep="first")
#             tp = tp.merge(test_lookup, on=join_keys, how="left", suffixes=("", "_audit"))
#         # else: no safe join => keep NaN; pool stats will still be computed from master_df.

#     # Group pools from master (audit independent from survivors)
#     g_master = master_df.groupby(list(group_cols), dropna=False, sort=False)

#     rows: List[Dict[str, Any]] = []

#     for _, chosen_row in tp.iterrows():
#         key = tuple(chosen_row.get(c, "unknown") for c in group_cols)
#         try:
#             pool = g_master.get_group(key)
#         except Exception:
#             pool = master_df

#         pool_size = int(len(pool))

#         chosen_val = _num(chosen_row.get(val_metric))
#         chosen_val_cum = _num(chosen_row.get(val_cum_metric))
#         chosen_score_value = _num(chosen_row.get(score_col))
#         chosen_test = _num(chosen_row.get(test_metric)) if test_metric in chosen_row.index else float("nan")

#         pool_best_val = _num(pool[val_metric].min()) if val_metric in pool.columns else float("nan")

#         pool_best_test = float("nan")
#         regret_test = float("nan")
#         ratio_test = float("nan")
#         chosen_test_percentile = float("nan")

#         if test_metric in pool.columns and pool[test_metric].notna().any():
#             pool_best_test = _num(pd.to_numeric(pool[test_metric], errors="coerce").min())
#             if np.isfinite(chosen_test) and np.isfinite(pool_best_test) and pool_best_test > 0:
#                 regret_test = chosen_test - pool_best_test
#                 ratio_test = chosen_test / pool_best_test

#             # Percentile of chosen test within pool (lower is better)
#             try:
#                 s = pd.to_numeric(pool[test_metric], errors="coerce")
#                 if np.isfinite(chosen_test):
#                     ranks = s.rank(pct=True, ascending=True, method="average")
#                     idx = (s - chosen_test).abs().idxmin()
#                     chosen_test_percentile = _num(ranks.loc[idx])
#             except Exception:
#                 chosen_test_percentile = float("nan")

#         # Spearman(VAL,TEST) within this group pool
#         val_test_spearman = _spearman_from_two_cols(pool, val_metric, test_metric)

#         out: Dict[str, Any] = {
#             # Identity
#             "dataset": chosen_row.get("dataset", "unknown"),
#             "well": chosen_row.get("well", "unknown"),
#             "architecture": chosen_row.get("architecture", chosen_row.get("architecture_name", "unknown")),
#             # Meta
#             "score_col": score_col,
#             "scoring_strategy": scoring_strategy,
#             "selection_source": selection_source,
#             # Chosen identity
#             "chosen_job_hash": chosen_row.get(job_hash_col) if job_hash_col else np.nan,
#             "chosen_trial_id": chosen_row.get(trial_id_col) if trial_id_col else np.nan,
#             "chosen_strategy_display": chosen_row.get(strat_col) if strat_col else np.nan,
#             # Chosen metrics
#             "chosen_val_smape_agg": chosen_val,
#             "chosen_val_smape_cum": chosen_val_cum,
#             "chosen_score_value": chosen_score_value,
#             "chosen_test_smape_agg": chosen_test,
#             # Pool audit
#             "pool_size": pool_size,
#             "pool_best_val_smape_agg": pool_best_val,
#             "pool_best_test_smape_agg": pool_best_test,
#             "regret_test": regret_test,
#             "ratio_test": ratio_test,
#             "chosen_test_percentile": chosen_test_percentile,
#             # New
#             "val_test_spearman": val_test_spearman,
#         }
#         rows.append(out)

#     summary = pd.DataFrame(rows)
#     return ensure_summary_schema(summary)


def build_canonical_summary(
    *,
    master_df: pd.DataFrame,
    top_performers: pd.DataFrame,
    score_col: str,
    scoring_strategy: str,
    selection_source: str = "run_distribution_filter",
    group_cols: Optional[Sequence[str]] = None,
    # Metrics (audit-only)
    val_metric: str = "val_smape_agg",
    val_cum_metric: str = "val_smape_cum",
    test_metric: str = "test_smape_agg",
) -> pd.DataFrame:
    """
    Build a canonical per-group summary table with a fixed schema.

    IMPORTANT:
    - Pure post-processing step; must NOT change selection.
    - TEST metrics are audit-only (regret/ratio/percentile + spearman).
    - Uses a coarse architecture grouping ONLY for audit pooling to avoid
      Darts_* fragmentation (family-level pool stats).
    """
    if master_df is None:
        master_df = pd.DataFrame()
    if top_performers is None:
        top_performers = pd.DataFrame()

    # -------------------------
    # Grouping resolution
    # -------------------------
    if group_cols is None:
        group_cols = []
        for c in ("dataset", "well", "architecture"):
            if c in master_df.columns or c in top_performers.columns:
                group_cols.append(c)
        if not group_cols:
            group_cols = ["well"] if ("well" in master_df.columns or "well" in top_performers.columns) else []

    if top_performers.empty or not group_cols:
        return empty_summary()

    # Identity columns in champions
    job_hash_col = _first_existing_col(top_performers, ["job_hash", "trial_hash", "trial_id"])
    trial_id_col = _first_existing_col(top_performers, ["trial", "trial_id", "number", "optuna_trial_number"])
    strat_col = _first_existing_col(top_performers, ["strategy_display", "physics_strategy", "variant", "strategy", "profile"])

    # Ensure group cols exist (avoid KeyError during key creation)
    for c in group_cols:
        if c not in master_df.columns:
            master_df = master_df.copy()
            master_df[c] = "unknown"
        if c not in top_performers.columns:
            top_performers = top_performers.copy()
            top_performers[c] = "unknown"

    def _num(x: Any) -> float:
        return _safe_float(x)

    def _spearman_from_two_cols(pool: pd.DataFrame, a: str, b: str) -> float:
        if a not in pool.columns or b not in pool.columns:
            return float("nan")
        x = pd.to_numeric(pool[a], errors="coerce")
        y = pd.to_numeric(pool[b], errors="coerce")
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 3:
            return float("nan")
        rx = x[m].rank(method="average", ascending=True)
        ry = y[m].rank(method="average", ascending=True)
        try:
            return float(rx.corr(ry, method="pearson"))
        except Exception:
            return float("nan")

    tp = top_performers.copy()
    md = master_df.copy()

    # -------------------------
    # Audit-only coarse architecture grouping
    # -------------------------
    # Determine which source column we can map from
    arch_src_master = "architecture" if "architecture" in md.columns else ("architecture_name" if "architecture_name" in md.columns else None)
    arch_src_tp = "architecture" if "architecture" in tp.columns else ("architecture_name" if "architecture_name" in tp.columns else None)

    if arch_src_master:
        md["__arch_group__"] = md[arch_src_master].map(_arch_group_value)
    else:
        md["__arch_group__"] = "unknown"

    if arch_src_tp:
        tp["__arch_group__"] = tp[arch_src_tp].map(_arch_group_value)
    else:
        tp["__arch_group__"] = "unknown"

    # Build pool grouping columns: same as group_cols but replace architecture(/name) with __arch_group__
    pool_group_cols = list(group_cols)
    pool_group_cols = ["__arch_group__" if c in ("architecture", "architecture_name") else c for c in pool_group_cols]

    # IMPORTANT: groupby MUST match the key we later build (pool_group_cols)
    g_master = md.groupby(pool_group_cols, dropna=False, sort=False)

    # -------------------------
    # If champions don't carry TEST, try to reattach safely (audit-only)
    # (This does not impact pooling stats; pool stats are computed from master_df anyway.)
    # -------------------------
    need_test = (test_metric not in tp.columns) or tp[test_metric].isna().all()
    have_test_in_master = (test_metric in md.columns) and md[test_metric].notna().any()

    if need_test and have_test_in_master:
        key_sets = [
            [*group_cols, "job_hash"],
            [*group_cols, "trial_hash"],
            [*group_cols, "trial_id"],
            [*group_cols, "optuna_trial_number"],
            [*group_cols, "trial"],
            [*group_cols, "number"],
        ]
        join_keys: List[str] = []
        for ks in key_sets:
            if all((k in tp.columns) and (k in md.columns) for k in ks):
                join_keys = ks
                break

        if not join_keys:
            fallback_keys = [*group_cols]
            for k in ["physics_strategy", "profile", "epochs", "n_epochs", "batch_size", "learning_rate", "data_sample", "lag_window", "horizon"]:
                if k in tp.columns and k in md.columns:
                    fallback_keys.append(k)
            if len(fallback_keys) > len(group_cols):
                join_keys = fallback_keys

        if join_keys:
            lookup_cols = list(dict.fromkeys(join_keys + [test_metric]))
            test_lookup = md[lookup_cols].dropna(subset=[test_metric]).drop_duplicates(subset=join_keys, keep="first")
            tp = tp.merge(test_lookup, on=join_keys, how="left", suffixes=("", "_audit"))

    # -------------------------
    # Build rows
    # -------------------------
    rows: List[Dict[str, Any]] = []

    for _, chosen_row in tp.iterrows():
        key = tuple(chosen_row.get(c, "unknown") for c in pool_group_cols)
        try:
            pool = g_master.get_group(key)
        except Exception:
            # If grouping keys mismatch, don't silently globalize; fall back to filtering by available cols
            pool = md
            try:
                for c, v in zip(pool_group_cols, key):
                    if c in pool.columns:
                        pool = pool[pool[c].astype(str) == str(v)]
            except Exception:
                pool = md  # final fallback

        pool_size = int(len(pool))

        chosen_val = _num(chosen_row.get(val_metric))
        chosen_val_cum = _num(chosen_row.get(val_cum_metric))
        chosen_score_value = _num(chosen_row.get(score_col))
        chosen_test = _num(chosen_row.get(test_metric)) if test_metric in chosen_row.index else float("nan")

        pool_best_val = _num(pd.to_numeric(pool[val_metric], errors="coerce").min()) if (val_metric in pool.columns) else float("nan")

        pool_best_test = float("nan")
        regret_test = float("nan")
        ratio_test = float("nan")
        chosen_test_percentile = float("nan")

        if test_metric in pool.columns and pool[test_metric].notna().any():
            s_test = pd.to_numeric(pool[test_metric], errors="coerce")
            pool_best_test = _num(s_test.min())
            if np.isfinite(chosen_test) and np.isfinite(pool_best_test) and pool_best_test > 0:
                regret_test = chosen_test - pool_best_test
                ratio_test = chosen_test / pool_best_test

            try:
                if np.isfinite(chosen_test):
                    ranks = s_test.rank(pct=True, ascending=True, method="average")
                    idx = (s_test - chosen_test).abs().idxmin()
                    chosen_test_percentile = _num(ranks.loc[idx])
            except Exception:
                chosen_test_percentile = float("nan")

        val_test_spearman = _spearman_from_two_cols(pool, val_metric, test_metric)

        out: Dict[str, Any] = {
            "dataset": chosen_row.get("dataset", "unknown"),
            "well": chosen_row.get("well", "unknown"),
            "architecture": chosen_row.get("architecture", chosen_row.get("architecture_name", "unknown")),
            "score_col": score_col,
            "scoring_strategy": scoring_strategy,
            "selection_source": selection_source,
            "chosen_job_hash": chosen_row.get(job_hash_col) if job_hash_col else np.nan,
            "chosen_trial_id": chosen_row.get(trial_id_col) if trial_id_col else np.nan,
            "chosen_strategy_display": chosen_row.get(strat_col) if strat_col else np.nan,
            "chosen_val_smape_agg": chosen_val,
            "chosen_val_smape_cum": chosen_val_cum,
            "chosen_score_value": chosen_score_value,
            "chosen_test_smape_agg": chosen_test,
            "pool_size": pool_size,
            "pool_best_val_smape_agg": pool_best_val,
            "pool_best_test_smape_agg": pool_best_test,
            "regret_test": regret_test,
            "ratio_test": ratio_test,
            "chosen_test_percentile": chosen_test_percentile,
            "val_test_spearman": val_test_spearman,
        }
        rows.append(out)

    summary = pd.DataFrame(rows)
    return ensure_summary_schema(summary)



# -----------------------------------------------------------------------------
# Contract normalizer
# -----------------------------------------------------------------------------

def make_selection_result_contract(
    *,
    master_df: pd.DataFrame,
    filter_result: Dict[str, Any],
    score_col: str,
    scoring_strategy: str,
    selection_source: str = "run_distribution_filter",
    meta: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[pd.DataFrame] = None,
    thresholds: Optional[pd.DataFrame] = None,
    top_performers: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Normalize the selection output contract.

    Guarantees required keys always exist:
      - top_performers: DataFrame
      - summary: canonical summary DataFrame (fixed schema)
      - diagnostics: canonical diagnostics DataFrame (fixed schema)
      - thresholds: DataFrame (may be empty)
      - meta: dict

    Also preserves legacy keys from filter_result when present.

    Enhancement:
      - If diagnostics provides regret/pool audit fields, promote them into canonical summary
        (pure post-processing; does not affect selection).
    """
    base: Dict[str, Any] = dict(filter_result or {}) if isinstance(filter_result, dict) else {}

    # Resolve frames with precedence: explicit arg > filter_result > empty
    tp = top_performers
    if tp is None:
        tp = base.get("top_performers", None)
    tp = tp if isinstance(tp, pd.DataFrame) else empty_top_performers()

    th = thresholds
    if th is None:
        th = base.get("thresholds", None)
    th = th if isinstance(th, pd.DataFrame) else empty_thresholds()
    th = ensure_thresholds_schema(th)

    diag = diagnostics
    if diag is None:
        diag = base.get("diagnostics", None)
    diag = diag if isinstance(diag, pd.DataFrame) else empty_diagnostics()

    # Keep a copy of "rich" diagnostics before canonical truncation (so we can promote extra cols)
    diag_rich = diag.copy() if isinstance(diag, pd.DataFrame) else pd.DataFrame()

    # Canonicalize diagnostics (stable schema)
    diag = ensure_diagnostics_schema(diag)

    summary = build_canonical_summary(
        master_df=master_df if master_df is not None else pd.DataFrame(),
        top_performers=tp,
        score_col=score_col,
        scoring_strategy=scoring_strategy,
        selection_source=selection_source,
    )
    summary = ensure_summary_schema(summary)

    # -------------------------------------------------------------------------
    # Promote regret/pool audit fields from diagnostics into summary (if provided)
    # -------------------------------------------------------------------------
    try:
        if isinstance(diag_rich, pd.DataFrame) and not diag_rich.empty and isinstance(summary, pd.DataFrame) and not summary.empty:
            # We merge by group identity
            key_cols = [c for c in ["dataset", "well", "architecture"] if c in summary.columns and c in diag_rich.columns]
            if key_cols:
                # Candidate columns to promote
                promote_cols = [
                    "pool_size",
                    "pool_best_val_smape_agg",
                    "pool_best_test_smape_agg",
                    "regret_test",
                    "ratio_test",
                    "chosen_test_percentile",
                ]
                available = [c for c in promote_cols if c in diag_rich.columns]
                if available:
                    diag_small = diag_rich[key_cols + available].drop_duplicates(subset=key_cols, keep="first").copy()
                    merged = summary.merge(diag_small, on=key_cols, how="left", suffixes=("", "__diag"))

                    # For each promoted field:
                    # - if summary is NaN, fill from diagnostics
                    # - otherwise keep existing
                    for c in available:
                        c_diag = f"{c}__diag"
                        if c_diag in merged.columns:
                            merged[c] = merged[c].combine_first(merged[c_diag])
                            merged.drop(columns=[c_diag], inplace=True, errors="ignore")

                    summary = ensure_summary_schema(merged)
    except Exception:
        # never fail contract because of promotion
        pass

    out: Dict[str, Any] = dict(base)
    out["top_performers"] = tp
    out["thresholds"] = th
    out["summary"] = summary
    out["diagnostics"] = diag

    # Meta (stable)
    meta_out = _ensure_meta(meta or out.get("meta", None))
    meta_out.setdefault("score_col", score_col)
    meta_out.setdefault("scoring_strategy", scoring_strategy)
    meta_out.setdefault("selection_source", selection_source)
    from hpo.score_ordering import resolve_ordering
    ordering = resolve_ordering(score_col, lower_is_better=None, default_ascending=True)
    meta_out.setdefault("score_direction", "ascending" if ordering.ascending else "descending")
    out["meta"] = meta_out

    # Keep compatibility keys even if they don't exist
    if "survivors" not in out or not isinstance(out.get("survivors"), pd.DataFrame):
        out["survivors"] = base.get("survivors", pd.DataFrame()) if isinstance(base.get("survivors"), pd.DataFrame) else pd.DataFrame()
    if "gate_counts" not in out or not isinstance(out.get("gate_counts"), pd.DataFrame):
        out["gate_counts"] = base.get("gate_counts", pd.DataFrame()) if isinstance(base.get("gate_counts"), pd.DataFrame) else pd.DataFrame()
    if "pool_counts" not in out or not isinstance(out.get("pool_counts"), pd.DataFrame):
        out["pool_counts"] = base.get("pool_counts", pd.DataFrame()) if isinstance(base.get("pool_counts"), pd.DataFrame) else pd.DataFrame()

    logger.info(
        "[contract] built selection contract | top_performers=%d | summary_rows=%d | thresholds_rows=%d | diagnostics_rows=%d | score_col=%s | scoring_strategy=%s | selection_source=%s",
        len(out["top_performers"]),
        len(out["summary"]),
        len(out["thresholds"]),
        len(out["diagnostics"]),
        score_col,
        scoring_strategy,
        selection_source,
    )

    return out



# ----------------------------
# Patch 3: regret diagnostics for LEGACY (audit-only)
# ----------------------------
def _sig(df: pd.DataFrame) -> pd.Series:
    """Stable-ish signature to join survivors/top_performers back to audit_df for test audit."""
    if df is None or df.empty:
        return pd.Series([], dtype=object)

    base_cols = [c for c in ["dataset", "well", "architecture"] if c in df.columns]
    base = df[base_cols].astype(str).agg("|".join, axis=1) if base_cols else pd.Series(["global"] * len(df), index=df.index, dtype=object)

    trial_col = next((c for c in ["optuna_trial_number", "trial", "trial_id", "number"] if c in df.columns), None)
    if trial_col:
        t = pd.to_numeric(df[trial_col], errors="coerce")
        t = t.apply(lambda x: f"trial:{int(x)}" if pd.notna(x) and np.isfinite(float(x)) else "trial:na")
        return base + "|" + t.astype(str)

    if "job_hash" in df.columns:
        return base + "|job:" + df["job_hash"].astype(str)

    if "experiment_id" in df.columns:
        return base + "|exp:" + df["experiment_id"].astype(str)

    return pd.Series([np.nan] * len(df), index=df.index, dtype=object)

# def build_legacy_regret_diagnostics(
#     *,
#     audit_df: pd.DataFrame,
#     survivors_df: pd.DataFrame,          # selection-view pool (no test_*)
#     top_performers: pd.DataFrame,        # chosen rows
#     selector_mode: str,
#     selection_path: str,
#     score_col: str,
#     scoring_strategy: str,
#     pool_method: str = "survivors",
#     test_metric: str = "test_smape_agg",
#     val_metric: str = "val_smape_agg",
# ) -> pd.DataFrame:
#     """
#     Computes audit-only regret stats on TEST, where pool := survivors.
#     Does not affect selection. Returned DF may include extra cols; contract will canonicalize.
#     """
#     if audit_df is None:
#         audit_df = pd.DataFrame()
#     if survivors_df is None or survivors_df.empty:
#         return pd.DataFrame()
#     if top_performers is None or top_performers.empty:
#         return pd.DataFrame()
#     if test_metric not in audit_df.columns:
#         return pd.DataFrame()

#     # group identity
#     group_cols = [c for c in ["dataset", "well", "architecture"] if c in survivors_df.columns]
#     if not group_cols:
#         group_cols = [c for c in ["well"] if c in survivors_df.columns]
#     if not group_cols:
#         return pd.DataFrame()

#     aud = audit_df.copy()
#     aud["__sig__"] = _sig(aud)

#     pool = survivors_df.copy()
#     pool["__sig__"] = _sig(pool)

#     tp = top_performers.copy()
#     tp["__sig__"] = _sig(tp)

#     # attach TEST into pool and tp (audit-only)
#     lookup = aud[["__sig__", test_metric]].dropna(subset=["__sig__"]).drop_duplicates(subset=["__sig__"], keep="first")
#     pool = pool.merge(lookup, on="__sig__", how="left")
#     tp = tp.merge(lookup, on="__sig__", how="left", suffixes=("", "__audit"))

#     # if tp already has test_metric keep it, else use __audit
#     if test_metric not in tp.columns and f"{test_metric}__audit" in tp.columns:
#         tp[test_metric] = tp[f"{test_metric}__audit"]
#     tp.drop(columns=[f"{test_metric}__audit"], inplace=True, errors="ignore")

#     def fnum(x: Any) -> float:
#         try:
#             v = float(x)
#             return v if np.isfinite(v) else float("nan")
#         except Exception:
#             return float("nan")

#     g_pool = pool.groupby(group_cols, dropna=False, sort=False)
#     rows = []

#     for _, ch in tp.iterrows():
#         key = tuple(ch.get(c, "unknown") for c in group_cols)
#         try:
#             p = g_pool.get_group(key)
#         except Exception:
#             p = pool

#         pool_size = int(len(p))
#         chosen_test = fnum(ch.get(test_metric))
#         pool_best_test = float("nan")
#         regret = float("nan")
#         ratio = float("nan")
#         pct = float("nan")

#         s = pd.to_numeric(p.get(test_metric, pd.Series([], dtype=float)), errors="coerce").to_numpy(dtype=float)
#         s = s[np.isfinite(s)]
#         if len(s):
#             pool_best_test = float(np.nanmin(s))
#             if np.isfinite(chosen_test):
#                 regret = float(chosen_test - pool_best_test)
#                 ratio = float(chosen_test / (pool_best_test + 1e-12))
#                 pct = float(np.mean(s <= chosen_test))

#         pool_best_val = float("nan")
#         if val_metric in p.columns and p[val_metric].notna().any():
#             pool_best_val = fnum(pd.to_numeric(p[val_metric], errors="coerce").min())

#         rows.append(
#             {
#                 "dataset": ch.get("dataset", "unknown"),
#                 "well": ch.get("well", "unknown"),
#                 "architecture": ch.get("architecture", ch.get("architecture_name", "unknown")),
#                 "selector_mode": selector_mode,
#                 "selection_path": selection_path,
#                 "selection_source": "regret_audit_from_survivors",
#                 "score_col": score_col,
#                 "scoring_strategy": scoring_strategy,
#                 "pool_method": pool_method,
#                 "pool_size": pool_size,
#                 "notes": "",

#                 # promoted into canonical summary (via contract patch)
#                 "pool_best_val_smape_agg": pool_best_val,
#                 "pool_best_test_smape_agg": pool_best_test,
#                 "regret_test": regret,
#                 "ratio_test": ratio,
#                 "chosen_test_percentile": pct,
#             }
#         )

#     return pd.DataFrame(rows)


def build_legacy_regret_diagnostics(
    *,
    audit_df: pd.DataFrame,
    survivors_df: pd.DataFrame,          # selection-view pool (no test_*)
    top_performers: pd.DataFrame,        # chosen rows
    selector_mode: str,
    selection_path: str,
    score_col: str,
    scoring_strategy: str,
    pool_method: str = "survivors",
    test_metric: str = "test_smape_agg",
    val_metric: str = "val_smape_agg",
) -> pd.DataFrame:
    """
    Computes audit-only regret stats on TEST, where pool := survivors.
    Does not affect selection.

    Uses coarse architecture grouping ONLY for audit pooling to avoid Darts_* fragmentation.
    Keeps dataset+well grouping whenever possible.
    """
    if audit_df is None:
        audit_df = pd.DataFrame()
    if survivors_df is None or survivors_df.empty:
        return pd.DataFrame()
    if top_performers is None or top_performers.empty:
        return pd.DataFrame()
    if test_metric not in audit_df.columns:
        return pd.DataFrame()

    # group identity (prefer dataset+well+architecture if present)
    group_cols = [c for c in ["dataset", "well", "architecture"] if c in survivors_df.columns]
    if not group_cols:
        group_cols = [c for c in ["dataset", "well"] if c in survivors_df.columns]
    if not group_cols:
        group_cols = [c for c in ["well"] if c in survivors_df.columns]
    if not group_cols:
        return pd.DataFrame()

    aud = audit_df.copy()
    aud["__sig__"] = _sig(aud)

    pool = survivors_df.copy()
    pool["__sig__"] = _sig(pool)

    tp = top_performers.copy()
    tp["__sig__"] = _sig(tp)

    # Coarse arch group for audit pooling (only if architecture is in group)
    if "architecture" in group_cols or "architecture_name" in pool.columns or "architecture_name" in tp.columns:
        pool["__arch_group__"] = pool.get("architecture", pool.get("architecture_name", "unknown")).map(_arch_group_value)
        tp["__arch_group__"] = tp.get("architecture", tp.get("architecture_name", "unknown")).map(_arch_group_value)
    else:
        pool["__arch_group__"] = "unknown"
        tp["__arch_group__"] = "unknown"

    pool_group_cols = ["__arch_group__" if c == "architecture" else c for c in group_cols]

    # attach TEST into pool and tp (audit-only) via signature
    lookup = aud[["__sig__", test_metric]].dropna(subset=["__sig__"]).drop_duplicates(subset=["__sig__"], keep="first")
    pool = pool.merge(lookup, on="__sig__", how="left")
    tp = tp.merge(lookup, on="__sig__", how="left", suffixes=("", "__audit"))

    if test_metric not in tp.columns and f"{test_metric}__audit" in tp.columns:
        tp[test_metric] = tp[f"{test_metric}__audit"]
    tp.drop(columns=[f"{test_metric}__audit"], inplace=True, errors="ignore")

    def fnum(x: Any) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else float("nan")
        except Exception:
            return float("nan")

    g_pool = pool.groupby(pool_group_cols, dropna=False, sort=False)
    rows: List[Dict[str, Any]] = []

    for _, ch in tp.iterrows():
        key = tuple(ch.get(c, "unknown") for c in pool_group_cols)
        try:
            p = g_pool.get_group(key)
        except Exception:
            p = pool

        pool_size = int(len(p))
        chosen_test = fnum(ch.get(test_metric))

        pool_best_test = float("nan")
        regret = float("nan")
        ratio = float("nan")
        pct = float("nan")

        s = pd.to_numeric(p.get(test_metric, pd.Series([], dtype=float)), errors="coerce").to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if len(s):
            pool_best_test = float(np.nanmin(s))
            if np.isfinite(chosen_test):
                regret = float(chosen_test - pool_best_test)
                ratio = float(chosen_test / (pool_best_test + 1e-12))
                pct = float(np.mean(s <= chosen_test))

        pool_best_val = float("nan")
        if val_metric in p.columns and p[val_metric].notna().any():
            pool_best_val = fnum(pd.to_numeric(p[val_metric], errors="coerce").min())

        rows.append(
            {
                "dataset": ch.get("dataset", "unknown"),
                "well": ch.get("well", "unknown"),
                "architecture": ch.get("architecture", ch.get("architecture_name", "unknown")),
                "selector_mode": selector_mode,
                "selection_path": selection_path,
                "selection_source": "regret_audit_from_survivors",
                "score_col": score_col,
                "scoring_strategy": scoring_strategy,
                "pool_method": pool_method,
                "pool_size": pool_size,
                "notes": "",
                "pool_best_val_smape_agg": pool_best_val,
                "pool_best_test_smape_agg": pool_best_test,
                "regret_test": regret,
                "ratio_test": ratio,
                "chosen_test_percentile": pct,
            }
        )

    return pd.DataFrame(rows)
