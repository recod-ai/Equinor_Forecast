"""
hpo.posthoc_filtering

Robust post-hoc filtering + champion selection orchestration for HPO leaderboards.

Contracts:
- Anti-leak: drop any `test_*` columns before gating/selection.
- Legacy lock: selection pool is always "survivors" (passes_gates).
- Decision column: `cfg.metric_to_optimize` (semantic aliases may be added).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, List


import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from hpo.posthoc_config import PostHocConfig
from hpo.score_ordering import resolve_ordering
from hpo.score_semantics import ensure_semantic_score_columns, resolve_score_semantics

from .posthoc_gates import (
    quantile_thresholds, mad_guards, thresholds_table, ensure_mad_for_metric,
    compose_predicates, apply_multi_metric_gates, apply_valcum_gate,
)
from .posthoc_selection import select_champions_by_strategy


try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    VIZ_LIBS_INSTALLED = True
except Exception:
    VIZ_LIBS_INSTALLED = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.columns import Columns
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False



def _find_arch_col(df: pd.DataFrame, arch_col: Optional[str] = None) -> str:
    """Helper para encontrar a coluna de arquitetura de forma robusta."""
    if arch_col and arch_col in df.columns:
        return arch_col
    for cand in ("architecture", "architecture_name", "arch"):
        if cand in df.columns:
            return cand
    raise ValueError("Nenhuma coluna de arquitetura encontrada em: ['architecture', 'architecture_name', 'arch']")


def run_distribution_filter(master_df: pd.DataFrame, cfg: PostHocConfig) -> Dict[str, Any]:
    """
    Orchestrates the robust filtering + champion selection pipeline.

    Explicit semantics:
    - "survivors": rows that passed 'passes_gates' (diagnostic only)
    - "df_for_selection": rows actually available for selection (LEGACY => survivors only)
    - selection column = cfg.metric_to_optimize (resolved earlier in pipeline)

    Anti-leak policy:
    - Selection must never see any 'test_*' columns. Those are audit-only and must be handled outside.

    Legacy lock:
    - pool_method is effectively ALWAYS "survivors" in legacy.
      If cfg.pool_method is "top_pct" or "val_band", we warn and override.
    """

    from hpo.score_semantics import resolve_score_semantics, ensure_semantic_score_columns

    logger = logging.getLogger(__name__)

    # Resolve score_col early for meta consistency even on early-return
    score_col = getattr(cfg, "metric_to_optimize", "weighted_score") or "weighted_score"
    sem = resolve_score_semantics(str(score_col), default_robust_kind="legacy")

    if master_df is None or master_df.empty:
        logger.info("[filter] Empty master_df. Nothing to filter.")
        return {
            "top_performers": pd.DataFrame(),
            "thresholds": pd.DataFrame(),
            "survivors": pd.DataFrame(),
            "gate_counts": pd.DataFrame(),
            "pool_counts": pd.DataFrame(),
            "df_gated": pd.DataFrame(),
            "meta": {"score_semantics": sem.__dict__},
        }

    logger.info("[filter] Starting distribution filtering pipeline...")

    # -------------------------------------------------------------------------
    # 0) Anti-leak hard: drop any TEST columns before *any* selection logic
    # -------------------------------------------------------------------------
    test_cols = [c for c in master_df.columns if str(c).startswith("test_")]
    if test_cols:
        logger.info("[filter][anti_leak] Dropping %d test_* columns from selection input.", len(test_cols))
    master_df_sel = master_df.drop(columns=test_cols, errors="ignore")

    # -------------------------------------------------------------------------
    # 1) Prepare Data
    # -------------------------------------------------------------------------
    df = sanitize_and_validate(master_df_sel, cfg)
    df = deduplicate_by_signature(df, cfg)

    score_col = getattr(cfg, "metric_to_optimize", "weighted_score") or "weighted_score"
    sem = resolve_score_semantics(str(score_col), default_robust_kind="legacy")

    if str(score_col) == "robust_score":
        try:
            df, cfg = plug_robust_score(df, cfg)
            score_col = "robust_score"
            sem = resolve_score_semantics(str(score_col), default_robust_kind="legacy")
        except Exception as e:
            logger.warning("[filter][plug] plug_robust_score failed (%s). Falling back to 'weighted_score'.", e)
            try:
                cfg.metric_to_optimize = "weighted_score"
            except Exception:
                pass
            score_col = "weighted_score"
            sem = resolve_score_semantics(str(score_col), default_robust_kind="legacy")

    # Ensure semantic alias columns exist (no behavior changes; extra columns only)
    df = ensure_semantic_score_columns(df, sem)# -------------------------------------------------------------------------
    df = sanitize_and_validate(master_df_sel, cfg)
    
    # If decision wants robust_score, ensure it exists BEFORE dedup,
    # otherwise dedup will fallback to val/weighted and may drop the true robust winner.
    score_col = getattr(cfg, "metric_to_optimize", "weighted_score") or "weighted_score"
    if str(score_col) == "robust_score":
        try:
            df, cfg = plug_robust_score(df, cfg)
            score_col = "robust_score"
        except Exception as e:
            logger.warning("[filter][plug] plug_robust_score failed (%s). Falling back to 'weighted_score'.", e)
            try:
                cfg.metric_to_optimize = "weighted_score"
            except Exception:
                pass
            score_col = "weighted_score"
    
    # Now dedup with the correct selection_col available
    df = deduplicate_by_signature(df, cfg)
    
    sem = resolve_score_semantics(str(score_col), default_robust_kind="legacy")
    df = ensure_semantic_score_columns(df, sem)

    n_wells = df[cfg.well_col].nunique() if cfg.well_col in df.columns else 0
    logger.info("[filter] Analyzing %d trials across %d wells.", len(df), n_wells)

    # -------------------------------------------------------------------------
    # 2) Compute Thresholds (Quantiles + MAD)
    # -------------------------------------------------------------------------
    qdf = quantile_thresholds(df, cfg)
    mdf = mad_guards(df, cfg)
    th_tab = thresholds_table(qdf, mdf, cfg)

    if (cfg.mad_guard or {}).get("enabled", False):
        mdf = ensure_mad_for_metric(
            df=df,
            mdf=mdf,
            well_col=cfg.well_col,
            metric=score_col,
            lower_is_better=bool((cfg.lower_is_better or {}).get(score_col, True)),
            alpha=float((cfg.mad_guard or {}).get("alpha", 1.0)),
        )

    logger.info("[filter] Computed quantile and MAD thresholds.")

    # -------------------------------------------------------------------------
    # 3) Apply Gates (multi-metric + valcum)
    # -------------------------------------------------------------------------
    preds = compose_predicates(qdf, mdf, cfg)
    df_gated = apply_multi_metric_gates(df, preds, cfg)
    df_gated = apply_valcum_gate(df_gated, cfg)

    # Survivors = diagnostic (passes_gates only)
    if "passes_gates" in df_gated.columns:
        survivors_mask = df_gated["passes_gates"].astype(bool)
    else:
        survivors_mask = pd.Series(True, index=df_gated.index)

    survivors = df_gated.loc[survivors_mask].copy()

    # Gate counts diagnostics (total vs kept_by_gates)
    total_counts = (
        df.groupby(cfg.well_col, dropna=False)[cfg.well_col]
          .count()
          .rename("total_in_pool")
          .reset_index()
    )
    kept_counts = (
        survivors.groupby(cfg.well_col, dropna=False)[cfg.well_col]
                 .count()
                 .rename("kept_by_gates")
                 .reset_index()
    )
    gate_counts = (
        total_counts.merge(kept_counts, on=cfg.well_col, how="left")
                    .fillna({"kept_by_gates": 0})
    )
    if "kept_by_gates" in gate_counts.columns:
        gate_counts["kept_by_gates"] = gate_counts["kept_by_gates"].astype(int)

    for _, row in gate_counts.iterrows():
        well = row[cfg.well_col]
        kept = int(row.get("kept_by_gates", 0))
        tot = int(row.get("total_in_pool", 0))
        logger.info("[filter][survivors] well=%s kept_by_gates=%d/%d", well, kept, tot)

    # -------------------------------------------------------------------------
    # 4) Optional Pareto mark (annotates only)
    # -------------------------------------------------------------------------
    df_for_selection = df_gated
    if getattr(cfg, "apply_pareto", False):
        df_for_selection = pareto_mark(df_for_selection, cfg)
        logger.info("[filter] Marked non-dominated (Pareto) solutions.")

    # -------------------------------------------------------------------------
    # 4.5) LEGACY LOCK: pool_method is effectively ALWAYS survivors
    # -------------------------------------------------------------------------
    requested_pool = str(getattr(cfg, "pool_method", "survivors") or "survivors").lower()
    if requested_pool != "survivors":
        logger.warning(
            "[filter][LEGACY_LOCK] pool_method='%s' requested, but legacy selection only supports 'survivors'. "
            "Overriding to 'survivors'.",
            requested_pool,
        )
    pool_method = "survivors"

    # Mark pool rows (for debugging/diagnostics)
    df_for_selection = df_for_selection.copy()
    df_for_selection["__pool_included__"] = False

    if "passes_gates" in df_for_selection.columns:
        df_for_selection.loc[df_for_selection["passes_gates"].astype(bool), "__pool_included__"] = True
    else:
        df_for_selection["__pool_included__"] = True

    # pool_counts: per-well pool size actually used
    pool_counts_rows = []
    for well_name, block in df_for_selection.groupby(cfg.well_col, sort=False, dropna=False):
        pool_size = int(block["__pool_included__"].astype(bool).sum())
        pool_counts_rows.append(
            {
                cfg.well_col: well_name,
                "pool_method": pool_method,
                "pool_size": pool_size,
            }
        )
    pool_counts = pd.DataFrame(pool_counts_rows)

    for _, r in pool_counts.iterrows():
        logger.info(
            "[filter][pool] well=%s method=%s pool_size=%d",
            r[cfg.well_col],
            r["pool_method"],
            int(r["pool_size"]),
        )

    # Keep only pool rows for selection
    df_for_selection = df_for_selection[df_for_selection["__pool_included__"].astype(bool)].copy()

    # -------------------------------------------------------------------------
    # 5) Select Final Champions
    # -------------------------------------------------------------------------
    top_performers = select_champions_by_strategy(df_for_selection, cfg)

    if top_performers is None or top_performers.empty:
        logger.info("[filter] No champions selected after filtering. Gates/pool may be too strict.")
        return {
            "top_performers": pd.DataFrame(),
            "thresholds": th_tab,
            "survivors": survivors,
            "gate_counts": gate_counts,
            "pool_counts": pool_counts,
            "df_gated": df_gated,
            "meta": {"score_semantics": sem.__dict__},
        }

    # Final picks diagnostics
    if cfg.well_col in top_performers.columns:
        picked_counts = (
            top_performers.groupby(cfg.well_col, dropna=False)[cfg.well_col]
                          .count()
                          .rename("picked_final")
                          .reset_index()
        )
        diag_final = (
            gate_counts.merge(picked_counts, on=cfg.well_col, how="left")
                       .fillna({"picked_final": 0})
        )
        diag_final["picked_final"] = diag_final["picked_final"].astype(int)

        for _, row in diag_final.iterrows():
            well = row[cfg.well_col]
            picked = int(row.get("picked_final", 0))
            kept_gates = int(row.get("kept_by_gates", 0))
            tot = int(row.get("total_in_pool", 0))
            logger.info(
                "[filter][final] well=%s picked_final=%d (from survivors=%d) total=%d",
                well, picked, kept_gates, tot
            )
    else:
        logger.info("[filter][final] picked_final=%d (no per-well column '%s')", len(top_performers), cfg.well_col)

    logger.info("[filter] Selected %d final champions.", len(top_performers))

    # -------------------------------------------------------------------------
    # 6) Visuals
    # -------------------------------------------------------------------------
    if VIZ_LIBS_INSTALLED and getattr(cfg, "plot", False):
        logger.info("[filter] Generating visual dashboards...")
        for well_name, group in df.groupby(cfg.well_col, dropna=False):
            well_champions = top_performers[top_performers[cfg.well_col] == well_name]
            if not well_champions.empty:
                well_thresholds = th_tab[th_tab[cfg.well_col] == well_name]
                plot_well_dashboard(group, well_thresholds, well_champions, well_name, cfg)

    return {
        "top_performers": top_performers,
        "thresholds": th_tab,
        "survivors": survivors,
        "gate_counts": gate_counts,
        "pool_counts": pool_counts,
        "df_gated": df_gated,
        "meta": {"score_semantics": sem.__dict__},
    }

def sanitize_and_validate(master_df: pd.DataFrame, cfg: PostHocConfig) -> pd.DataFrame:
    """
    Ensures required columns exist, drops NaNs in metrics, and coerces dtypes.

    Robustness:
    - If cfg.arch_col is missing, it will try common fallbacks ("architecture_name", "arch", "architecture")
      and use the first available for validation purposes.
    """
    df = master_df.copy()

    well_col = getattr(cfg, "well_col", "well")
    arch_col = getattr(cfg, "arch_col", "architecture")

    if arch_col not in df.columns:
        for cand in ("architecture", "architecture_name", "arch"):
            if cand in df.columns:
                arch_col = cand
                break
        else:
            arch_col = None

    required_cols = [well_col] + ( [arch_col] if arch_col else [] ) + list(getattr(cfg, "metrics", []))
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in master_df: {missing_cols}")

    # Drop rows with NaNs in any key metrics
    metrics = list(getattr(cfg, "metrics", []))
    df.dropna(subset=metrics, inplace=True)

    # Coerce metrics to numeric and drop failures
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df.dropna(subset=metrics, inplace=True)

    return df




def deduplicate_by_signature(df: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
    import logging
    import hashlib

    from hpo.score_ordering import resolve_ordering

    if df is None or df.empty:
        return df

    # Detect family to choose sensible defaults
    fam = "arps" if (
        ("variant" in df.columns) or
        ("architecture_name" in df.columns and df["architecture_name"].astype(str).str.contains("Arps", na=False).any())
    ) else "generic"

    if fam == "arps":
        DEFAULT_SIGNATURE_COLS = [
            "variant", "solver", "loss", "burn_in_fraction", "piecewise",
            "weighting", "quantile_tau", "b_grid_kind", "b_grid_size",
            "b_min", "b_max", "lag_window", "horizon",
        ]
    else:
        DEFAULT_SIGNATURE_COLS = [
            "physics_strategy", "data_sample", "learning_rate", "lag_window",
            "batch_size", "epochs", "architecture_profile",
        ]

    # ---- Robust fetch of signature cols ----
    sig_attr = getattr(cfg, "hpo_signature_cols", DEFAULT_SIGNATURE_COLS)
    if sig_attr is None:
        signature_cols = DEFAULT_SIGNATURE_COLS
    elif isinstance(sig_attr, (list, tuple)):
        signature_cols = list(sig_attr)
    elif isinstance(sig_attr, str):
        signature_cols = [sig_attr]
    else:
        logging.warning("[dedup] Unexpected type for cfg.hpo_signature_cols=%r; using defaults.", type(sig_attr))
        signature_cols = DEFAULT_SIGNATURE_COLS

    present = [c for c in signature_cols if c in df.columns]
    missing = [c for c in signature_cols if c not in df.columns]
    if missing:
        logging.info("[dedup] Ignoring missing signature columns: %s", missing)

    if not present:
        logging.info("[dedup] No signature columns present; skipping deduplication.")
        return df

    out = df.copy()

    # ---- resolve selection/sort column: prefer cfg.selection_col, then cfg.metric_to_optimize, then fall back ----
    selection_col = getattr(cfg, "selection_col", None) or getattr(cfg, "metric_to_optimize", None)

    if not selection_col or selection_col not in out.columns:
        for candidate in ("weighted_score", "robust_score", "val_smape_agg", "val_smape_cum"):
            if candidate in out.columns:
                selection_col = candidate
                break

    if not selection_col:
        logging.warning("[dedup] No suitable selection column found; returning frame unchanged.")
        return out

    # ---- unify direction via single source of truth ----
    lib = getattr(cfg, "lower_is_better", {}) or {}
    ordering = resolve_ordering(selection_col, lower_is_better=lib, default_ascending=True)
    asc = ordering.ascending

    logging.info(
        "[dedup] Using selection_col=%s | ascending=%s | higher_is_better=%s | reason=%s | signature_cols=%s",
        selection_col,
        asc,
        ordering.higher_is_better,
        ordering.reason,
        present,
    )

    # ---- Build stable signature hash (stringified values, NaN-safe) ----
    out["signature_cols_used"] = ",".join(present)

    def _row_sig_str(r: pd.Series) -> str:
        parts = []
        for c in present:
            v = r.get(c)
            if pd.isna(v):
                parts.append(f"{c}=<NA>")
            else:
                parts.append(f"{c}={str(v)}")
        return "|".join(parts)

    sig_str = out.apply(_row_sig_str, axis=1)
    out["signature_hash"] = sig_str.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())

    # ---- Keep best per signature_hash according to ordering ----
    # We only need to sort by selection_col (direction-aware), but stable sort helps reproducibility.
    out = out.sort_values(by=[selection_col], ascending=[asc], kind="mergesort")
    out = out.drop_duplicates(subset=["signature_hash"], keep="first")

    return out

def plug_robust_score(
    df: pd.DataFrame,
    cfg: "PostHocConfig",
    *,
    arch_col: str | None = None,
    weights_by_arch: dict | None = None,
) -> tuple[pd.DataFrame, "PostHocConfig"]:
    """
    Legacy plug-in for robust_score.

    Backward compatible:
      - Ensures 'robust_score' exists (legacy meaning)
      - Ensures semantic alias 'legacy_robust_score' exists
      - Injects 'robust_score' into cfg as the decision column (unchanged behavior)
    """
    out = df.copy()

    if "robust_score" not in out.columns:
        out = add_robust_score(out, weights_by_arch=weights_by_arch, arch_col=arch_col)

    # ✅ semantics alias (Patch 2)
    if "legacy_robust_score" not in out.columns:
        out["legacy_robust_score"] = out["robust_score"]

    cfg = plug_score_column_into_cfg(
        out, cfg,
        score_col="robust_score",
        ensure_in_metrics=True,
        default_quantile=float((getattr(cfg, "primary_quantile", {}) or {}).get("robust_score", 0.6)),
        default_lower_is_better=True,
    )
    return out, cfg

def plug_score_column_into_cfg(
    df: pd.DataFrame,
    cfg: "PostHocConfig",
    *,
    score_col: str,
    ensure_in_metrics: bool = True,
    default_quantile: float = 0.6,
    default_lower_is_better: bool = True,
) -> "PostHocConfig":
    """
    Plug-and-play:
      - garante que cfg.metric_to_optimize aponta para score_col
      - garante score_col em cfg.metrics
      - garante score_col em cfg.lower_is_better
      - garante cfg.primary_quantile não vazio e inclui score_col (se fizer sentido)
    """
    # 1) metric_to_optimize
    try:
        cfg.metric_to_optimize = score_col
    except Exception:
        pass

    # 2) lower_is_better
    lib = getattr(cfg, "lower_is_better", None)
    if not isinstance(lib, dict):
        lib = {}
    lib.setdefault(score_col, default_lower_is_better)
    try:
        cfg.lower_is_better = lib
    except Exception:
        pass

    # 3) cfg.metrics
    if ensure_in_metrics:
        mets = list(getattr(cfg, "metrics", []) or [])
        if score_col not in mets:
            mets.append(score_col)
        try:
            cfg.metrics = mets
        except Exception:
            pass

    # 4) primary_quantile
    pq = dict(getattr(cfg, "primary_quantile", {}) or {})
    if not pq:
        pq = {score_col: float(default_quantile)}
    else:
        # garante pelo menos um filtro pelo score (se ainda não existir)
        pq.setdefault(score_col, float(default_quantile))
    try:
        cfg.primary_quantile = pq
    except Exception:
        pass

    return cfg

def add_weighted_score(
    df: pd.DataFrame,
    metric_weights: dict,
    lower_is_better: dict,
    group_keys: list[str] | None = None
) -> pd.DataFrame:
    """
    Normaliza métricas e calcula weighted_score.
    Agrupa por chaves existentes; tolera ausência de 'dataset' (ex.: ARPS).
    - Se group_keys=None -> tenta ["dataset","well"]
    - Usa só as chaves que existirem no DF
    - Se nada sobrar, tenta ["well"]; se ainda assim não der, aplica globalmente
    """
    if df is None or df.empty:
        return df

    # defina alvo padrão e interseção com colunas presentes
    default_keys = ["dataset", "well"]
    wanted = group_keys if group_keys is not None else default_keys
    gkeys = [k for k in wanted if k in df.columns]

    # fallback: tenta só 'well'
    if not gkeys and "well" in df.columns:
        gkeys = ["well"]

    df_scored = df.copy()

    def _score_block(block: pd.DataFrame) -> pd.DataFrame:
        out = block.copy()
        # normaliza cada métrica em ranks 0..1 (menor=melhor -> rank direto; maior=melhor -> 1-rank)
        for metric, is_lower_better in lower_is_better.items():
            if metric not in out.columns:
                continue
            ranks = out[metric].rank(pct=True, ascending=True)  # rank baixo = valor baixo
            out[f"{metric}_norm"] = ranks if is_lower_better else (1 - ranks)
        # score ponderado
        out["weighted_score"] = 0.0
        for metric, w in metric_weights.items():
            col = f"{metric}_norm"
            if col in out.columns:
                out["weighted_score"] += float(w) * out[col]
        return out

    # aplica com ou sem groupby, conforme chaves disponíveis
    if gkeys:
        return df_scored.groupby(gkeys, group_keys=False).apply(_score_block).reset_index(drop=True)
    else:
        # sem chaves de agrupamento: aplica no DF inteiro
        return _score_block(df_scored)

def load_master_leaderboard(campaign_name: str, results_dir: str | Path,
                            include_validation_alt: bool = True) -> Optional[pd.DataFrame]:
    """
    Lê leaderboards de uma campanha. Tolera CSVs vazios/corrompidos.
    include_validation_alt: se True, também procura por validation_{campaign}_top_*/leaderboard.csv

    Backward-compatible and extended:
      - Supports nested layout:
          <results_dir>/<campaign_name>/results/<family>/<run>/leaderboard.csv
      - Also supports when results_dir already IS ".../<campaign_name>/results"
      - Keeps legacy patterns from the original implementation.
    """
    import os
    from pathlib import Path
    import pandas as pd
    from pandas.errors import EmptyDataError, ParserError

    results_path = Path(results_dir)

    # --- Collect candidate files from multiple compatible layouts ---
    files: list[Path] = []

    # (A) NEW: canonical nested layout where results_dir points to the campaign group root
    #     <results_dir>/<campaign_name>/results/*/*/leaderboard.csv
    nested_root_a = results_path / campaign_name / "results"
    if nested_root_a.is_dir():
        files += list(nested_root_a.glob("*/*/leaderboard.csv"))

    # (B) NEW: results_dir may already be ".../<campaign_name>/results"
    #     <results_dir>/*/*/leaderboard.csv  (and verify path contains the campaign_name)
    if results_path.name == "results":
        files_b = list(results_path.glob("*/*/leaderboard.csv"))
        if files_b:
            # keep only those whose path contains campaign_name somewhere above
            filtered_b = []
            cn = str(campaign_name)
            for f in files_b:
                # Check any ancestor equals campaign_name
                if any(p.name == cn for p in f.parents):
                    filtered_b.append(f)
            files += filtered_b

    # (C) Legacy multi-cycle layout (kept from original):
    files += list((results_path).glob(f"{campaign_name}_cycle_*/leaderboard.csv"))

    # (D) Legacy "simple validation" layout:
    p_simple = results_path / campaign_name / "leaderboard.csv"
    if p_simple.exists():
        files.append(p_simple)

    # (E) Legacy/alt validation tops:
    if include_validation_alt:
        files += list((results_path).glob(f"validation_{campaign_name}_top_*/leaderboard.csv"))

    # De-duplicate & keep only actual files
    files = sorted(set(f for f in files if f.is_file()))

    # If nothing found, mirror the original user-facing message and return None
    if not files:
        print(f"⚠️ WARNING: No leaderboards found for campaign '{campaign_name}' in '{results_path}'.")
        return None

    # --- Read all files, tolerate empties/corrupt, aggregate ---
    valid_dfs = []
    skipped = []

    for f in files:
        try:
            if os.path.getsize(f) == 0:
                skipped.append((str(f), "empty file (0 bytes)"))
                continue

            df = pd.read_csv(f)

            if df is None or getattr(df, "shape", (0, 0))[1] == 0:
                skipped.append((str(f), "no columns after read_csv"))
                continue

            # If a 'campaign' column is not present, add it using the run-folder name.
            # This is safe and helps Stage 1 harvesting; doesn't break existing code that ignores extra cols.
            if "campaign" not in df.columns:
                # In nested layout, the run name is the parent directory of the CSV
                df = df.copy()
                df["campaign"] = f.parent.name

            valid_dfs.append(df)

        except (EmptyDataError, ParserError) as e:
            skipped.append((str(f), f"{type(e).__name__}: {e}"))
        except Exception as e:
            skipped.append((str(f), f"Unexpected: {type(e).__name__}: {e}"))

    if not valid_dfs:
        print(
            f"⚠️ WARNING: Found {len(files)} leaderboard file(s) for campaign '{campaign_name}', "
            f"but none were readable. Skipped: {skipped}"
        )
        return None

    master_df = pd.concat(valid_dfs, ignore_index=True)

    if skipped:
        print(f"ℹ️ Skipped {len(skipped)} file(s) for '{campaign_name}':")
        for path, reason in skipped:
            print(f"   - {path} → {reason}")

    msg = "Aggregated" if len(valid_dfs) > 1 else "Loaded"
    print(f"✅ {msg} {len(valid_dfs)} leaderboard(s) for campaign '{campaign_name}' with {len(master_df)} total trials.")
    return master_df

def _parse_campaign_name(campaign_name: str) -> Dict[str, str]:
    """
    Robustly parse dataset / well / (optional) architecture from campaign names.

    Works for examples like:
      - validation_UNISIM_IV_P15_Seq2PIN
      - UNISIM_IV_P15_Seq2Context
      - robust_smoke_darts_UNISIM_IV_P15          (no arch suffix)
      - UNISIM_IV_P15                              (no arch suffix)

    Returns {"dataset": "...", "well": "...", "architecture": "..."} with "unknown" if not matched.
    """
    pattern = re.compile(
        r'^(?:validation_|robust_smoke_darts_)?'                # known optional prefixes
        r'(?P<dataset>VOLVE|UNISIM_IV|UNISIM)_'                 # dataset
        r'(?P<well>.+?)'                                        # well (non-greedy)
        r'(?:_(?P<architecture>Seq2Context|Seq2Trend|Seq2PIN))?'# optional architecture suffix
        r'(?:_top_\d+)?$'                                       # optional _top_N
    )

    m = pattern.match(campaign_name)
    if not m:
        return {"dataset": "unknown", "well": "unknown", "architecture": "unknown"}

    parsed = m.groupdict()
    parsed["well"] = (parsed.get("well") or "").strip()

    # Optional normalization for known aliases
    well_normalization_map = {"F14": "15-9-F-14", "F12": "15-9-F-12"}
    if parsed["well"] in well_normalization_map:
        parsed["well"] = well_normalization_map[parsed["well"]]

    # Fill missing architecture with "unknown"
    if not parsed.get("architecture"):
        parsed["architecture"] = "unknown"

    return parsed

def plot_well_dashboard(well_df, thresholds, champions_df, well_name, cfg) -> Any:
    """Builds a visual dashboard for a single well."""
    if not VIZ_LIBS_INSTALLED:
        warnings.warn("Visualization libraries (plotly) not installed. Skipping plot generation.")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"{cfg.metrics[0]} Distribution",
            f"{cfg.metrics[1]} vs {cfg.metrics[2]}",
            f"{cfg.metrics[1]} Distribution",
            f"{cfg.metrics[2]} Distribution"
        ),
        specs=[[{}, {"type": "scatter"}], [{}, {}]]
    )

    colors = ['#636EFA', '#EF553B', '#00CC96']

    # Subplots for metric distributions
    for i, metric in enumerate(cfg.metrics):
        row, col = (1, 1) if i == 0 else (2, 1) if i == 1 else (2, 2)
        
        # Histogram
        fig.add_trace(go.Histogram(x=well_df[metric], name=metric, marker_color=colors[i], nbinsx=30), row=row, col=col)
        
        # Threshold lines
        metric_thresholds = thresholds[thresholds['metric'] == metric]
        if not metric_thresholds.empty:
            cutoff = metric_thresholds['cutoff'].iloc[0]
            upper_bound = metric_thresholds['upper_bound'].iloc[0]
            fig.add_vline(x=cutoff, line_width=2, line_dash="dash", line_color="firebrick",
                          annotation_text=f"Q({metric_thresholds['quantile'].iloc[0]})", row=row, col=col)
            if np.isfinite(upper_bound):
                fig.add_vline(x=upper_bound, line_width=2, line_dash="dot", line_color="goldenrod",
                              annotation_text="MAD Guard", row=row, col=col)

        # Champion markers
        fig.add_trace(go.Scatter(
            x=champions_df[metric], y=np.zeros(len(champions_df)),
            mode='markers', name='Champions',
            marker=dict(symbol='star', color='gold', size=15, line=dict(color='black', width=1))
        ), row=row, col=col)

    # Scatter plot for Pareto metrics
    metric_x, metric_y = cfg.pareto_metrics[0], cfg.pareto_metrics[1]
    for arch, group in well_df.groupby(cfg.arch_col):
        fig.add_trace(go.Scatter(
            x=group[metric_x], y=group[metric_y],
            mode='markers', name=arch,
            marker=dict(opacity=0.7)
        ), row=1, col=2)
    
    # Highlight champions on scatter
    fig.add_trace(go.Scatter(
        x=champions_df[metric_x], y=champions_df[metric_y],
        mode='markers', name='Champions',
        marker=dict(symbol='star', color='gold', size=15, line=dict(color='black', width=1))
    ), row=1, col=2)
    
    fig.update_layout(
        title_text=f"<b>Analysis Dashboard for Well: {well_name}</b>",
        showlegend=True, height=800
    )
    fig.update_xaxes(title_text=metric_x, row=1, col=2)
    fig.update_yaxes(title_text=metric_y, row=1, col=2)
    
    fig.show()

def pareto_mark(df: pd.DataFrame, cfg: PostHocConfig) -> pd.DataFrame:
    """
    Marks rows that are on the Pareto front (per well) for cfg.pareto_metrics.

    Plug-and-play + robust:
    - Direction-aware: respects cfg.lower_is_better per metric.
      (Metrics with higher_is_better are internally negated so we always minimize.)
    - NaN-safe: rows with any NaN in pareto metrics are marked as not Pareto.
    - Defensive: if pareto_metrics invalid/missing, creates 'is_pareto' and returns.

    Writes:
      - is_pareto: bool
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        out["is_pareto"] = False
        return out

    well_col = getattr(cfg, "well_col", "well")
    pareto_cols = list(getattr(cfg, "pareto_metrics", []) or [])

    out = df.copy()
    out["is_pareto"] = False

    # Basic guards
    if not pareto_cols or well_col not in out.columns:
        return out

    # Keep only columns present
    present = [c for c in pareto_cols if c in out.columns]
    if len(present) < 2:
        return out

    lib_isb = dict(getattr(cfg, "lower_is_better", {}) or {})

    # Pre-compute direction multipliers: +1 for lower-is-better, -1 for higher-is-better
    # After transform, we ALWAYS minimize.
    mult = np.array([1.0 if bool(lib_isb.get(m, True)) else -1.0 for m in present], dtype=float)

    def _pareto_min(points: np.ndarray) -> np.ndarray:
        """
        Return mask of non-dominated points under minimization.

        A point j is dominated if there exists i such that:
          points[i] <= points[j] in all dims AND points[i] < points[j] in at least one dim.
        """
        n = points.shape[0]
        is_eff = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_eff[i]:
                continue
            p = points[i]
            # Any point that is >= p in all dims and > p in some dim is dominated by p (minimization)
            dominated = np.all(points >= p, axis=1) & np.any(points > p, axis=1)
            dominated[i] = False
            is_eff[dominated] = False
        return is_eff

    for _, g in out.groupby(well_col, sort=False, dropna=False):
        if g.empty:
            continue

        block = g[present].apply(pd.to_numeric, errors="coerce")
        valid = block.notna().all(axis=1)
        if not bool(valid.any()):
            continue

        pts = (block.loc[valid].to_numpy(dtype=float) * mult)  # direction-aware -> minimization
        mask_valid = _pareto_min(pts)

        # write only on valid rows; invalid remain False
        out.loc[block.index[valid], "is_pareto"] = mask_valid

    out["is_pareto"] = out["is_pareto"].astype(bool)
    return out

def neighbor_signature_cols(arch_name: str, df_columns: List[str], *, arch_col: str) -> List[str]:
    base_cols = ["well", arch_col]
    
    if "Seq2PIN" in arch_name:
        sig_cols = base_cols + ["physics_strategy", "aggregation_method", "lag_window", "horizon"]
    elif "Arps_Canonical" in arch_name:
        sig_cols = base_cols + ["variant", "lag_window", "horizon"]
    elif "Darts" in arch_name:
        profile_col = "profile" if "profile" in df_columns else "physics_strategy"
        sig_cols = base_cols + [profile_col, "lag_window", "horizon", "input_chunk_length", "output_chunk_length"]
    else:
        sig_cols = base_cols

    return [col for col in sig_cols if col in df_columns]

def compute_neighbor_iqr(
    df: pd.DataFrame,
    metric: str = "val_smape_cum",
    arch_col: Optional[str] = None,
    min_group: int = 4
) -> pd.Series:
    """
    Calcula o IQR de uma métrica em grupos de vizinhos "próximos" (mesma assinatura),
    normalizado por (well, architecture). Penaliza grupos pequenos.
    """
    if df.empty or metric not in df.columns:
        return pd.Series(index=df.index, dtype=float).fillna(0.0)

    arch_col_found = _find_arch_col(df, arch_col)
    output_iqr_norm = pd.Series(np.nan, index=df.index, name="neighbor_iqr_cum_norm")

    for (well, arch), group in df.groupby(["well", arch_col_found]):
        if group.empty:
            continue
        
        sig_cols = neighbor_signature_cols(arch, list(df.columns), arch_col=arch_col_found)
        if not all(c in group.columns for c in sig_cols):
            continue

        iqr_series = group.groupby(sig_cols, dropna=False)[metric].transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
        group_sizes = group.groupby(sig_cols, dropna=False)[metric].transform('size')
        
        # Rank alto (próximo de 1.0) significa IQR alto (ruim)
        iqr_norm = iqr_series.rank(pct=True, ascending=True).fillna(0.5).clip(0, 1)
        
        # penaliza => puxa pra 1.0 (ruim)
        iqr_norm.loc[group_sizes < min_group] = np.maximum(iqr_norm.loc[group_sizes < min_group], 0.75)

        output_iqr_norm.loc[group.index] = iqr_norm

    return output_iqr_norm.fillna(0.0)

def compute_rank_gap(
    df: pd.DataFrame,
    by: List[str] = ["well", "architecture"],
    agg_col: str = "val_smape_agg",
    cum_col: str = "val_smape_cum"
) -> pd.Series:
    """
    Penaliza o desalinhamento entre os ranks de val_smape_agg e val_smape_cum.
    """
    if df.empty or agg_col not in df.columns or cum_col not in df.columns:
        return pd.Series(index=df.index, dtype=float).fillna(0.0)

    by_cols = by[:]
    if "architecture" not in by_cols:
        arch_col_found = _find_arch_col(df)
        if arch_col_found not in by_cols:
            by_cols.append(arch_col_found)
    
    valid_by = [c for c in by_cols if c in df.columns]

    def _compute_gap(group: pd.DataFrame) -> pd.Series:
        r_agg = group[agg_col].rank(pct=True, ascending=True)
        r_cum = group[cum_col].rank(pct=True, ascending=True)
        return abs(r_agg - r_cum)

    gap = df.groupby(valid_by, group_keys=False).apply(_compute_gap) if valid_by else _compute_gap(df)
    
    return gap.rename("rank_gap_norm").fillna(0.0)

def add_robust_score(
    df: pd.DataFrame,
    weights_by_arch: Optional[Dict[str, Dict]] = None,
    arch_col: Optional[str] = None
) -> pd.DataFrame:
    """
    LEGACY robust score (min-max + penalties).
    Fix: For DARTS, compute/normalize robust_score across all DARTS profiles per well
    so scores are comparable across profiles.
    """
    out = df.copy()
    arch_col_found = _find_arch_col(out, arch_col)

    if "neighbor_iqr_cum_norm" not in out.columns:
        out["neighbor_iqr_cum_norm"] = compute_neighbor_iqr(out, arch_col=arch_col_found)
    if "rank_gap_norm" not in out.columns:
        out["rank_gap_norm"] = compute_rank_gap(out, by=["well", arch_col_found])

    default_weights = {
        "Seq2PIN":        {"w_agg": 1.00, "w_cum": 0.25, "alpha": 0.15, "beta": 0.10},
        "Arps_Canonical": {"w_agg": 1.00, "w_cum": 0.25, "alpha": 0.15, "beta": 0.10},
        "Darts":          {"w_agg": 1.00, "w_cum": 0.25, "alpha": 0.15, "beta": 0.10},
        "default":        {"w_agg": 1.00, "w_cum": 0.25, "alpha": 0.15, "beta": 0.10},
    }
    if weights_by_arch:
        default_weights.update(weights_by_arch)

    out["robust_score"] = np.nan

    def min_max_scale(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        s_min, s_max = s.min(), s.max()
        if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max == s_min:
            return pd.Series(0.5, index=s.index)
        return (s - s_min) / (s_max - s_min)

    # ---- KEY FIX: build a "family group" so DARTS profiles share the same scale per well ----
    def _family_key(a: str) -> str:
        s = "" if a is None else str(a)
        return "Darts" if ("Darts" in s or "darts" in s.lower()) else s

    out["__robust_family__"] = out[arch_col_found].astype(str).map(_family_key)

    # Group by (well, robust_family) instead of (well, architecture)
    for (well, fam), group in out.groupby(["well", "__robust_family__"], sort=False, dropna=False):
        if group.empty:
            continue

        w = default_weights["Darts"] if fam == "Darts" else default_weights.get(fam, default_weights["default"])

        norm_agg = min_max_scale(group["val_smape_agg"])
        norm_cum = min_max_scale(group["val_smape_cum"])
        norm_iqr = min_max_scale(group["neighbor_iqr_cum_norm"])
        norm_gap = min_max_scale(group["rank_gap_norm"])

        score = (
            w["w_agg"] * norm_agg +
            w["w_cum"] * norm_cum +
            w["alpha"] * norm_iqr +
            w["beta"]  * norm_gap
        )

        out.loc[group.index, "robust_score"] = score

    if "legacy_robust_score" not in out.columns:
        out["legacy_robust_score"] = out["robust_score"]

    out = out.drop(columns=["__robust_family__"], errors="ignore")
    return out


def print_campaign_summary(campaign_queue: List[Dict], max_concurrent: int = 1):
    """
    Prints a beautifully formatted summary of all campaigns about to be run,
    arranging them in a grid layout.

    Args:
        campaign_queue (List[Dict]): The list of campaign jobs to be executed.
        max_concurrent (int): The maximum number of parallel jobs.
    """
    if not RICH_AVAILABLE:
        # Simple fallback if rich is not installed
        print("--- Campaign Batch Summary ---")
        print(f"Queued Campaigns: {len(campaign_queue)}")
        print(f"Max Parallel Runs: {max_concurrent}")
        for job in campaign_queue:
            print(f"- {job['name']}")
        print("--------------------------")
        return

    # --- Rich-formatted output ---
    console = Console()
    
    # Print the main title panel (unchanged)
    console.print(
        Panel(
            f"[bold]Queued Campaigns:[/] {len(campaign_queue)}\n[bold]Max Parallel Runs:[/] {max_concurrent}",
            title="🚀 HPO Batch Execution Plan 🚀",
            style="bold blue",
            expand=False
        )
    )

    # --- 1. Iterate and Collect Panels ---
    campaign_panels = []
    for job in campaign_queue:
        try:
            from config_loader import load_campaign_config
            config = load_campaign_config(job['config_path'])

            # Create the summary table for one campaign
            table = Table.grid(expand=True, padding=(0, 1))
            table.add_column(style="magenta", justify="right")
            table.add_column(style="green")
            
            arch = config.job_defaults.architecture_name
            dataset = config.run_scope.dataset_name
            wells = ", ".join(config.run_scope.wells)
            trials = sum(config.hpo_params.trials_per_cycle_schedule)

            table.add_row("[b]Arch:[/b]", arch)
            table.add_row("[b]Data:[/b]", f"{dataset} ({wells})")
            table.add_row("[b]Trials:[/b]", str(trials))

            # Create a Panel for this campaign
            panel = Panel(
                table,
                title=f"[cyan bold]{job['name']}[/]",
                title_align="left",
                border_style="dim cyan",
                # Add some padding to space out the panels
                padding=(1, 2)
            )
            # Add the created Panel to our list
            campaign_panels.append(panel)
        except Exception as e:
            campaign_panels.append(Panel(f"[bold red]Error:\n{e}", title=job['name']))
    
    # --- 2. Group and Render ---
    if campaign_panels:
        # Create a Columns layout with our list of panels.
        # `expand=True` makes the columns fill the available width.
        # `equal=True` tries to make columns the same width.
        console.print(Columns(campaign_panels, equal=True, expand=True))