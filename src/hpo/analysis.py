# src/hpo/analysis.py

# ==============================================================================
# SELECTION PRESETS (dashboard): what each mode does + knobs + intuition
#
# This project supports 4 explicit selection modes. In ALL cases:
#   - VAL-only for filtering/selection (anti-leak): any `test_*` is NOT used to choose champions.
#   - TEST is audit-only after picks: regret/ratio/spearman/pctl, etc.
#   - Deterministic: stable sorting + tie-breaks → same inputs => same champions.
#
# Vocabulary:
#   - selector_mode   = WHICH path runs (LEGACY vs NEIGHBOR).
#   - scoring_strategy= HOW score columns are produced during aggregation (e.g., weighted_score vs robust_score).
#   - metric_to_optimize / selection_col = the decision column (single source of truth; ordering resolved once).
#
# ------------------------------------------------------------------------------
# 1) LEGACY_WEIGHTED
# ------------------------------------------------------------------------------
# What it does:
#   - Aggregates campaign results -> master DF
#   - Drops `test_*` (anti-leak)
#   - Deduplicates by HP signature (same configuration counts once; keep best)
#   - Applies VAL-only gates per well (quantiles + MAD guard + optional val_smape_cum band gate)
#   - Pool = survivors (only those that passed gates)
#   - Selects champions by stable sort on `weighted_score` (lower is better)
#
# Knobs you can tune (legacy):
#   - Weighted score composition: METRIC_WEIGHTS (which VAL metrics matter + weights)
#   - Gate aggressiveness: primary_quantile, mad_guard, valcum_gate, relax_pool, apply_pareto (annotation-only)
#   - Diversity / selection shape: top_strategies_per_well, per_strategy_k, selection_strategy, hpo_signature_cols
#
# Intuition:
#   "Filter out suspicious/outliers first, then pick the best composite VAL score from the survivors."
#
# ------------------------------------------------------------------------------
# 2) LEGACY_ROBUSTCOL
# ------------------------------------------------------------------------------
# What it does:
#   - Same legacy pipeline as above (anti-leak -> dedup -> gates -> pool=survivors)
#   - Decision column is `robust_score` (legacy robust score variant)
#   - Selects champions by stable sort on `robust_score` (lower is better)
#
# Knobs:
#   - All legacy knobs (gates/relax/dedup/selection_strategy)
#   - Robust score semantics: resolved via score_semantics; produced by aggregation under scoring_strategy="robust_score"
#
# Intuition:
#   "Still trust gates, but rank survivors with a robustness-aware score instead of a pure weighted sum."
#
# ------------------------------------------------------------------------------
# 3) NEIGHBOR_TOP_PCT
# ------------------------------------------------------------------------------
# What it does (neighborhood path):
#   - Aggregates results -> master DF
#   - Drops `test_*` (anti-leak)
#   - For each (dataset, well, architecture) group:
#       (a) Build a VAL-only pool using top-percentile: pool_method="top_pct"
#       (b) Compute a local robust score using neighborhood behavior (kNN in HP-space)
#       (c) Pick the best candidate by that robust local score
#   - Then computes TEST audit metrics for reporting (regret/ratio/spearman/etc.)
#
# Knobs you can tune (neighborhood_overrides):
#   - Pool building: pool_cfg (top_pct, drop, take, min_candidates)
#   - Local robustness: robust_cfg (k, min_strat, alpha/beta/gamma, luck_q, distance weights, etc.)
#   - Grouping context: group_cols (e.g., dataset+well+architecture vs other granularities)
#
# Intuition:
#   "Not only be good in VAL, but be good *and stable* compared to similar HP neighbors (reduce 'lucky' picks)."
#
# ------------------------------------------------------------------------------
# 4) NEIGHBOR_VAL_BAND
# ------------------------------------------------------------------------------
# What it does:
#   - Same neighborhood path as above, but pool_method="val_band"
#   - Pool is defined by a VAL band (near-best / within a VAL range), then the same kNN robust local scoring
#   - TEST remains audit-only after selection
#
# Knobs:
#   - Same neighborhood knobs (pool_cfg + robust_cfg + group_cols)
#   - The key difference is the pool criterion (band vs percentile) as implemented in pick_pool_idx
#
# Intuition:
#   "Compare only the truly competitive candidates (within a plausible VAL band), then choose the most robust locally."
#
# ------------------------------------------------------------------------------
# Quick rule-of-thumb:
#   - LEGACY_*    : gates first -> pool=survivors -> rank by {weighted_score|robust_score}
#   - NEIGHBOR_*  : pool first (top_pct or val_band) -> local kNN robust scoring -> pick -> TEST audits after
# ==============================================================================


# flake8: noqa: E402
"""Hyperparameter Optimization (HPO) analysis and reporting utilities."""

# --- Standard Library Imports ---
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

# --- Third-Party Imports ---
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import optuna.visualization as vis
from box import Box
from plotly.subplots import make_subplots
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder

# --- Local Application Imports ---
# Absolute imports from other packages
from forecast_pipeline.analytics import (
    _style_champions_table,
    analyze_best_per_architecture,
    analyze_holistically,
)
from forecast_pipeline.plotting import (
    plot_campaign_strategy_performance,
    plot_champions_well,
    plot_hyperparameter_importance_per_well,
    plot_performance_by_architecture,
)

# Relative imports from the 'hpo' package
from .posthoc_filtering import (
    _parse_campaign_name,
    load_master_leaderboard,
    run_distribution_filter,
    add_weighted_score,
    compute_rank_gap,
    compute_neighbor_iqr,
    add_robust_score,
)
from .posthoc_config import make_default_config, SelectionRunConfig
from .optuna_utils import (
    analyze_single_campaign,
    generate_profile_from_dataframe,
    get_hyperparameter_columns,
    get_study,
    get_top_n_configs,
    visualize_study,
)
from .validation import (
    _format_pp,
    _style_equivalence,
    build_validation_delta_table,
    create_validation_report,
    run_validation_comparison,
    style_validation_report,
)

from .selection_contract import (
    make_selection_result_contract,
    build_legacy_regret_diagnostics,
)

# --- Optional Imports for Rich Display ---
try:
    from IPython.display import HTML, display
except ImportError:
    display = print  # Fallback for non-IPython environments
    HTML = str

try:
    from rich.console import Console
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    class Console:  # Fallback console
        def print(self, text, *args, **kwargs):
            print(text)

    Markdown = str


# ==============================================================================
# 1. MULTI-CAMPAIGN ANALYSIS FUNCTIONS
# ==============================================================================


# You can place this helper function at the top of the analysis.py file as well
def log_pipeline_path(title: str, steps: List[str]):
    """Logs the selected pipeline path in a formatted box."""
    logger = logging.getLogger(__name__)
    max_len = max(len(title), max(len(s) for s in steps)) + 4
    
    logger.info("┌─" + "─" * max_len + "─┐")
    logger.info(f"│ {title.ljust(max_len)} │")
    logger.info("├─" + "─" * max_len + "─┤")
    for step in steps:
        logger.info(f"│ {step.ljust(max_len)} │")
    logger.info("└─" + "─" * max_len + "─┘")


def aggregate_all_campaign_results(
    results_dir: str | Path,
    metric_weights: Dict,
    lower_is_better: Dict,
    campaign_names: Optional[List[str]] = None,
    scoring_strategy: str = "robust_score"  # "robust_score" or "weighted_score"
) -> pd.DataFrame:
    """
    Aggregates leaderboards from multiple campaigns and calculates scores based on the
    chosen scoring strategy ('robust_score' or 'weighted_score').
    """
    results_path = Path(results_dir)
    all_master_leaderboards: List[pd.DataFrame] = []

    if campaign_names:
        search_list = campaign_names
    else:
        all_dirs = [d for d in results_path.iterdir() if d.is_dir()]
        search_list = sorted(list(set([d.name.split('_cycle_')[0] for d in all_dirs])))
        print(f"Discovered {len(search_list)} unique campaigns to aggregate.")

    search_list = [
        c for c in search_list
        if c and not c.startswith(".") and not c.startswith("validation") and c != ".ipynb_checkpoints"
    ]

    for name in search_list:
        df = load_master_leaderboard(name, results_path)
        if df is None or df.empty:
            continue

        df = df.copy()
        df["campaign"] = name
        
        md = _parse_campaign_name(name)
        df["campaign_dataset"] = md.get("dataset", "unknown")
        df["campaign_well"] = md.get("well", "unknown")
        df["campaign_architecture"] = md.get("architecture", "unknown")

        def _needs_fill(col: str) -> bool:
            if col not in df.columns: return True
            s = df[col]
            if s.isna().all(): return True
            if s.dtype == object and (s.fillna("").str.strip() == "").all(): return True
            return False

        if _needs_fill("dataset") and md["dataset"] != "unknown":
            df["dataset"] = md["dataset"]
        if _needs_fill("well") and md["well"] not in ("", "unknown"):
            df["well"] = md["well"]
        if _needs_fill("architecture"):
            if "architecture_name" in df.columns and df["architecture_name"].notna().any():
                df["architecture"] = df["architecture_name"]
            elif md["architecture"] != "unknown":
                df["architecture"] = md["architecture"]

        if "strategy_display" not in df.columns:
            if "physics_strategy" in df.columns:
                df["strategy_display"] = df["physics_strategy"].astype(str)
            elif "variant" in df.columns:
                df["strategy_display"] = df["variant"].astype(str)
            else:
                fallback_arch_col = next((c for c in ("architecture", "architecture_name") if c in df.columns), None)
                if fallback_arch_col:
                    df["strategy_display"] = df[fallback_arch_col].astype(str)
                else:
                    df["strategy_display"] = "unknown"

        all_master_leaderboards.append(df)

    if not all_master_leaderboards:
        return pd.DataFrame()

    master_df = pd.concat(all_master_leaderboards, ignore_index=True)

    # --- START OF REFACTORED SCORING LOGIC ---

    if scoring_strategy == "robust_score":
        # PATH 1: Robust Score Flow
        log_pipeline_path(
            title="SCORING PATH: ROBUST SCORE",
            steps=[
                "1. Calculate stability/consistency components.",
                "2. Combine into 'robust_score'.",
                "3. 'weighted_score' will NOT be calculated."
            ]
        )
        
        leak_cols = [c for c in master_df.columns if c.startswith("test_")]
        if leak_cols:
            logging.info("Anti-leakage check: Ignoring test metrics: %s", leak_cols)

        arch_col = next((c for c in ("architecture", "architecture_name") if c in master_df.columns), None)
        
        if arch_col:
            master_df["rank_gap_norm"] = compute_rank_gap(master_df, by=["well", arch_col])
            master_df["neighbor_iqr_cum_norm"] = compute_neighbor_iqr(master_df, metric="val_smape_cum", arch_col=arch_col)
            master_df = add_robust_score(master_df, arch_col=arch_col)
        else:
            logging.warning("Architecture column not found. Skipping robust_score calculation.")
            for col in ["rank_gap_norm", "neighbor_iqr_cum_norm", "robust_score"]:
                if col not in master_df.columns:
                    master_df[col] = np.nan
    
    elif scoring_strategy == "weighted_score":
        # PATH 2: Legacy Weighted Score Flow
        log_pipeline_path(
            title="SCORING PATH: WEIGHTED SCORE",
            steps=[
                "1. Normalize metrics based on 'metric_weights'.",
                "2. Combine into 'weighted_score'.",
                "3. Robust components will NOT be calculated."
            ]
        )
        master_df = add_weighted_score(
            master_df,
            metric_weights,
            lower_is_better,
            group_keys=["dataset", "well"]
        )
    else:
        raise ValueError(f"Unknown scoring_strategy: '{scoring_strategy}'. Must be 'robust_score' or 'weighted_score'.")

    # --- END OF REFACTORED SCORING LOGIC ---
    
    print(f"✅ Successfully aggregated and scored {len(master_df)} total trials.")
    return master_df



def analyze_across_campaigns(
    master_df: pd.DataFrame,
    metric_to_optimize: str = "weighted_score",
):
    """
    Performs high-level analysis on an aggregated DataFrame of results from multiple campaigns.

    NOTE:
    - Ordering is resolved via a single source of truth (resolve_ordering).
    - For weighted_score: higher is better.
    - For robust_score: lower is better.
    """
    from hpo.score_ordering import resolve_ordering

    if master_df is None or master_df.empty:
        print("Master DataFrame is empty. Nothing to analyze.")
        return

    if metric_to_optimize not in master_df.columns:
        print(f"❌ ERROR: Metric to optimize '{metric_to_optimize}' not found in DataFrame.")
        print("      Consider passing 'metric_weights' to 'aggregate_all_campaign_results' to calculate it.")
        return

    ordering = resolve_ordering(metric_to_optimize, lower_is_better=None, default_ascending=True)
    print(
        f"\n[analysis] metric_to_optimize={metric_to_optimize} | "
        f"ascending={ordering.ascending} | higher_is_better={ordering.higher_is_better} | reason={ordering.reason}"
    )

    print(f"\n{'='*20} 🌐 Holistic Campaign Analysis {'='*20}")

    # --- 1. Best Overall Models ---
    print(f"\n--- 🏆 Top 20 Overall Performers (by '{metric_to_optimize}') ---")
    display_cols = ["campaign", "architecture", "dataset", "well", metric_to_optimize, "val_smape_cum", "val_smape_agg"]
    available_cols = [c for c in display_cols if c in master_df.columns]

    top20 = (
        master_df.nsmallest(20, metric_to_optimize)
        if ordering.ascending
        else master_df.nlargest(20, metric_to_optimize)
    )
    display(top20[available_cols])

    # --- 2. Architecture Performance ---
    print(f"\n--- 🏛️ Average Performance by Architecture ---")
    # Group by campaign and take the best row per campaign according to ordering
    grp_campaign = master_df.groupby("campaign")[metric_to_optimize]
    best_idx = grp_campaign.idxmin() if ordering.ascending else grp_campaign.idxmax()
    best_per_campaign = master_df.loc[best_idx]

    arch_perf = (
        best_per_campaign.groupby("architecture")[metric_to_optimize]
        .mean()
        .sort_values(ascending=ordering.ascending)
    )
    print(arch_perf.to_string())

    # --- 3. Per-Well Analysis ---
    print(f"\n--- 🛢️ Best Architecture per Well ---")
    grp_well = master_df.groupby("well")[metric_to_optimize]
    best_idx_well = grp_well.idxmin() if ordering.ascending else grp_well.idxmax()
    best_model_per_well = master_df.loc[best_idx_well]

    well_summary = (
        best_model_per_well[["well", "architecture", metric_to_optimize]]
        .sort_values(by="well")
        .set_index("well")
    )
    print(well_summary.to_string())

    print(f"\n{'='*20} ✅ Holistic Analysis Complete {'='*20}")


def run_validation_analysis(
    original_campaign_name: str,
    validation_campaign_name: str,
    config: Box # Pass the config for paths
):
    """
    High-level workflow to load results from an HPO run and its corresponding
    validation run, then generate and display a comparison report.
    """
    print(f"\n{'='*20}  Vergleichsanalyse: {original_campaign_name} vs. {validation_campaign_name} {'='*20}")
    
    hpo_studies_dir = config.infra.hpo_studies_dir
    results_dir = config.infra.experiments_output_dir

    # 1. Load data for both campaigns
    hpo_leaderboard = load_master_leaderboard(original_campaign_name, results_dir)
    validation_leaderboard = load_master_leaderboard(validation_campaign_name, results_dir)
    study = get_study(original_campaign_name, hpo_studies_dir)
    
    if any(df is None or df.empty for df in [hpo_leaderboard, validation_leaderboard]) or study is None:
        print("--- Validation analysis aborted due to missing data for one or both campaigns. ---")
        return

    # 2. Get hyperparameters and generate the report
    hyper_cols = get_hyperparameter_columns(study)
    if not hyper_cols:
        print("Could not determine hyperparameter columns from study. Cannot generate report.")
        return

    report_df = create_validation_report(hpo_leaderboard, validation_leaderboard, hyper_cols)
    
    # 3. Style and display
    if not report_df.empty:
        print("\n--- ✅ HPO vs. Validation Performance Comparison ---")
        styled_report = style_validation_report(report_df)
        display(HTML(styled_report.to_html()))
    else:
        print("--- Comparison report is empty. No matching configurations found between runs. ---")

SelectorMode = Literal[
    "LEGACY_WEIGHTED",
    "LEGACY_ROBUSTCOL",
    "NEIGHBOR_TOP_PCT",
    "NEIGHBOR_VAL_BAND",
]


def _generate_selection_plots(
    master_df: pd.DataFrame,
    metric_to_optimize: str,
    *,
    plot_architecture_performance: bool = False,
    plot_hparam_importance_per_well: bool = False,
    plot_champions_per_well: bool = False,
    plot_summary_bars: bool = False
) -> None:
    """
    Thin wrapper around your existing plotting functions.
    All toggles are False by default.
    """
    if master_df is None or master_df.empty:
        logging.info("[plots] master_df is empty — skipping plots.")
        return

    if plot_architecture_performance:
        logging.info("[plots] plot_performance_by_architecture")
        plot_performance_by_architecture(master_df, metric=metric_to_optimize)

    if plot_champions_per_well:
        logging.info("[plots] plot_champions_per_well")
        plot_champions_well(master_df, metric=metric_to_optimize)

    if plot_hparam_importance_per_well:
        logging.info("[plots] plot_hyperparameter_importance_per_well")
        plot_hyperparameter_importance_per_well(master_df, metric_col=metric_to_optimize)

    if plot_summary_bars:
        logging.info("[plots] plot_summary_bars")
        # AUTO detecta seq2 vs darts
        plot_campaign_strategy_performance(master_df, metric=metric_to_optimize)


# def run_selection_pipeline(cfg: "SelectionRunConfig") -> Dict[str, Any]:
#     """
#     End-to-end HPO selection runner.

#     Semantics:
#     - scoring_strategy: how scores are PRODUCED during aggregation ("weighted_score" | "robust_score")
#     - selection_col: column USED to order/select champions
#     - selector_mode: dispatches LEGACY_* vs NEIGHBOR_*
#     - anti-leak: selection NEVER sees test_* columns (audit can).

#     IMPORTANT (canonical grouping):
#     - Adds architecture_family ∈ {PINN, ARPS, DARTS, ...} and architecture_subtype (original label).
#     - Neighborhood selection defaults to grouping by architecture_family (not raw 'architecture'),
#       preventing Darts_* subtypes from being treated as different "architectures" for "best/regret" logic.
#     """
#     import logging
#     import re
#     from pathlib import Path
#     from typing import Any, Dict, List, Optional, Tuple

#     import pandas as pd

#     from hpo.score_ordering import resolve_ordering
#     from hpo.selection_contract import make_selection_result_contract

#     logger = logging.getLogger(__name__)

#     # ----------------------------
#     # Utils (small, local, compact)
#     # ----------------------------
#     def as_df(x: Any) -> pd.DataFrame:
#         return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

#     def mkdir(p: Path) -> Path:
#         p.mkdir(parents=True, exist_ok=True)
#         return p

#     def mode_and_path(raw: Any) -> Tuple[str, str]:
#         mode = str(raw or "LEGACY_WEIGHTED").upper().strip()
#         mode_map = {
#             "LEGACY_WEIGHTED": "LEGACY_GATES",
#             "LEGACY_ROBUSTCOL": "LEGACY_GATES",
#             "NEIGHBOR_TOP_PCT": "NEIGHBORHOOD_ROBUST",
#             "NEIGHBOR_VAL_BAND": "NEIGHBORHOOD_ROBUST",
#         }
#         path = mode_map.get(mode)
#         if path is None:
#             logger.warning("[selection] Unknown selector_mode='%s'. Falling back to LEGACY_WEIGHTED.", mode)
#             mode = "LEGACY_WEIGHTED"
#             path = mode_map[mode]
#         return mode, path

#     def split_audit_selection(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
#         test_cols = [c for c in df.columns if str(c).startswith("test_")]
#         sel = df.drop(columns=test_cols, errors="ignore") if test_cols else df
#         return df, sel, test_cols

#     def resolve_selection_col(mode: str, sel_df: pd.DataFrame) -> str:
#         target = getattr(cfg, "metric_to_optimize", None) or getattr(cfg, "scoring_strategy", None) or "weighted_score"

#         if mode == "LEGACY_ROBUSTCOL" and (not getattr(cfg, "metric_to_optimize", None) or cfg.metric_to_optimize == "weighted_score"):
#             target = "robust_score"
#         if mode == "LEGACY_WEIGHTED" and not getattr(cfg, "metric_to_optimize", None):
#             target = "weighted_score"

#         if sel_df is not None and not sel_df.empty and target not in sel_df.columns:
#             produced = getattr(cfg, "scoring_strategy", None)
#             for cand in [produced, "robust_score", "weighted_score", "val_smape_agg", "val_smape_cum"]:
#                 if cand and cand in sel_df.columns:
#                     logger.warning("[selection] selection_col='%s' missing. Falling back to '%s'.", target, cand)
#                     target = cand
#                     break
#             else:
#                 logger.warning("[selection] selection_col='%s' missing; cols_sample=%s", target, list(sel_df.columns)[:30])
#         return str(target)

#     def persist_master(audit_df: pd.DataFrame) -> str:
#         p = (cfg.reports_dir / "hpo_master_leaderboard.csv").resolve()
#         audit_df.to_csv(p, index=False)
#         logger.info("[selection] Saved master leaderboard: %s", p)
#         return str(p)

#     # ----------------------------
#     # PATCH (GLOBAL): canonical architecture family
#     # ----------------------------
#     def _arch_family(x: object) -> str:
#         s = "" if x is None else str(x).strip()
#         s0 = s.lower().replace(" ", "")
#         if "darts" in s0:
#             return "DARTS"
#         if "arps" in s0:
#             return "ARPS"
#         if "seq2" in s0 or "pinn" in s0:
#             return "PINN"
#         # fallback: keep a clean token
#         return re.sub(r"[^0-9a-zA-Z_]+", "", s).upper() or "UNKNOWN"

#     def _ensure_arch_family_cols(df: pd.DataFrame) -> pd.DataFrame:
#         if df is None or df.empty:
#             return df
#         out = df.copy()

#         # choose best available label column
#         src = None
#         for cand in ["architecture", "architecture_name"]:
#             if cand in out.columns:
#                 src = cand
#                 break

#         if src:
#             out["architecture_subtype"] = out[src].astype(str)
#             out["architecture_family"] = out[src].map(_arch_family)
#         else:
#             out["architecture_subtype"] = "UNKNOWN"
#             out["architecture_family"] = "UNKNOWN"

#         # Keep backwards compatibility: if some downstream expects "architecture" to be coarse,
#         # we DO NOT overwrite it here (too risky). We only add new cols.
#         return out

#     def _normalize_group_cols(cols: List[str], df: pd.DataFrame) -> List[str]:
#         """
#         Ensure group columns exist and map legacy 'architecture' grouping to 'architecture_family'
#         when available (prevents Darts_* fragmentation).
#         """
#         if not cols:
#             return cols
#         out = []
#         for c in cols:
#             if c == "architecture" and "architecture_family" in df.columns:
#                 out.append("architecture_family")
#             else:
#                 out.append(c)
#         # drop non-existing safely (prevents crashes if a user passes old keys)
#         out2 = [c for c in out if c in df.columns]
#         return out2 or out

#     def _default_join_keys(audit_df: pd.DataFrame, tp: pd.DataFrame) -> List[str]:
#         """
#         Prefer stable identifiers when present; fall back to (dataset, well, subtype) style.
#         """
#         key_candidates = [
#             "job_hash", "trial_hash", "trial_id", "trial", "number", "optuna_trial_number", "experiment_id",
#             "dataset", "well",
#             # try both, because some parts use architecture_name
#             "architecture", "architecture_name",
#             "architecture_subtype", "architecture_family",
#             "physics_strategy", "strategy_display", "variant", "profile",
#         ]
#         return [c for c in key_candidates if c in audit_df.columns and c in tp.columns]

#     def maybe_write_validation_profile(*, top_performers: pd.DataFrame, audit_df: pd.DataFrame, test_cols: List[str]) -> Optional[str]:
#         if top_performers is None or top_performers.empty:
#             logger.warning("[selection] No champions to export; skipping validation profile.")
#             return None

#         tp = top_performers.copy()

#         # (audit-only) reattach test_* if possible
#         if test_cols:
#             join_keys = _default_join_keys(audit_df, tp)
#             if join_keys:
#                 test_lookup = audit_df[join_keys + test_cols].drop_duplicates(subset=join_keys, keep="first")
#                 tp = tp.merge(test_lookup, on=join_keys, how="left")
#             else:
#                 logger.warning("[selection][artifact] Could not reattach test_* cols (no join keys).")

#         out_path = (cfg.reports_dir.resolve() / f"{cfg.validation_run_name}.csv").resolve()
#         logger.info("[selection] Generating validation profile: %s", out_path)

#         _ = generate_profile_from_dataframe(
#             candidates_df=tp,
#             output_profile_path=out_path,
#             fixed_params={"seed": cfg.validation_seed, "plot": True},
#         )
#         return str(out_path)

#     def empty_contract(*, audit_df: pd.DataFrame, selection_col: str, mode: str, selection_path: str) -> Dict[str, Any]:
#         out = make_selection_result_contract(
#             master_df=audit_df,
#             filter_result={"top_performers": pd.DataFrame(), "thresholds": pd.DataFrame()},
#             score_col=selection_col,
#             scoring_strategy=cfg.scoring_strategy,
#             selection_source="empty",
#             meta={
#                 "selector_mode": mode,
#                 "selection_path": selection_path,
#                 "selection_col": selection_col,
#                 "scoring_strategy": cfg.scoring_strategy,
#                 "anti_leak_policy": "selection_df_drop_test_prefix",
#             },
#         )
#         out.update(
#             {
#                 "master_df": pd.DataFrame(),
#                 "leaderboard_path": None,
#                 "validation_profile_path": None,
#                 "selection_metric_used": selection_col,
#                 "scoring_strategy_used": cfg.scoring_strategy,
#             }
#         )
#         return out

#     # ----------------------------
#     # dirs
#     # ----------------------------
#     mkdir(cfg.reports_dir)
#     posthoc_dir = mkdir(cfg.reports_dir / "posthoc_analysis")

#     # ----------------------------
#     # mode / path
#     # ----------------------------
#     mode, selection_path = mode_and_path(getattr(cfg, "selector_mode", None))
#     logger.info(
#         "[selection] selector_mode=%s selection_path=%s scoring_strategy=%s metric_to_optimize=%s",
#         mode, selection_path, getattr(cfg, "scoring_strategy", None), getattr(cfg, "metric_to_optimize", None),
#     )

#     # ----------------------------
#     # aggregate
#     # ----------------------------
#     logger.info("[selection] Aggregating results from %s", cfg.results_dir)
#     master_df = aggregate_all_campaign_results(
#         results_dir=cfg.results_dir,
#         metric_weights=cfg.metric_weights,
#         lower_is_better=cfg.lower_is_better,
#         campaign_names=None,
#         scoring_strategy=cfg.scoring_strategy,
#     )
#     audit_df = as_df(master_df)
#     audit_df, selection_df, test_cols = split_audit_selection(audit_df)

#     # apply canonical cols AFTER split so both get them
#     audit_df = _ensure_arch_family_cols(audit_df)
#     selection_df = _ensure_arch_family_cols(selection_df)

#     logger.info(
#         "[selection][anti_leak] audit_cols=%d selection_cols=%d dropped_test_cols=%d",
#         int(audit_df.shape[1]), int(selection_df.shape[1]), int(len(test_cols)),
#     )

#     selection_col = resolve_selection_col(mode, selection_df)
#     logger.info("[selection] selection_col=%s (DECISION) | scoring_strategy=%s | selector_mode=%s", selection_col, cfg.scoring_strategy, mode)

#     # plots are audit-only (safe)
#     _generate_selection_plots(
#         audit_df,
#         selection_col,
#         plot_architecture_performance=getattr(cfg, "plot_architecture_performance", False),
#         plot_hparam_importance_per_well=getattr(cfg, "plot_hparam_importance_per_well", False),
#         plot_champions_per_well=getattr(cfg, "plot_champions_per_well", False),
#         plot_summary_bars=getattr(cfg, "plot_summary_bars", False),
#     )

#     if audit_df.empty:
#         logger.warning("[selection] No results found. Did you run campaigns?")
#         return empty_contract(audit_df=audit_df, selection_col=selection_col, mode=mode, selection_path="EMPTY")

#     leaderboard_path = persist_master(audit_df)

#     # ----------------------------
#     # dispatch: NEIGHBOR
#     # ----------------------------
#     if selection_path == "NEIGHBORHOOD_ROBUST":
#         from hpo.neighborhood_selection import run_neighborhood_selection

#         over = dict(getattr(cfg, "neighborhood_overrides", None) or {})

#         pool_method = str(over.get("pool_method") or ("top_pct" if mode == "NEIGHBOR_TOP_PCT" else "val_band")).lower()
#         pool_cfg = dict(over.get("pool_cfg", {}) or {})
#         robust_cfg = dict(over.get("robust_cfg", {}) or {})

#         # IMPORTANT: default to architecture_family to avoid Darts_* subtype fragmentation
#         default_group_cols = ["dataset", "well", "architecture_family"] if "architecture_family" in audit_df.columns else ["dataset", "well", "architecture"]
#         group_cols = list(over.get("group_cols") or default_group_cols)
#         group_cols = _normalize_group_cols(group_cols, audit_df)

#         logger.info(
#             "[selection][dispatch] NEIGHBORHOOD_ROBUST | pool_method=%s | group_cols=%s | pool_cfg=%s | robust_cfg=%s",
#             pool_method, group_cols, pool_cfg, robust_cfg,
#         )

#         champions_df, pool_diag = run_neighborhood_selection(
#             audit_df=audit_df,
#             pool_method=pool_method,
#             pool_cfg=pool_cfg,
#             robust_cfg=robust_cfg,
#             group_cols=group_cols,
#         )

#         ordering = resolve_ordering(selection_col, lower_is_better=cfg.lower_is_better, default_ascending=True)

#         out = make_selection_result_contract(
#             master_df=audit_df,
#             filter_result={
#                 "top_performers": champions_df,
#                 "thresholds": pd.DataFrame(),
#                 "diagnostics": pool_diag,
#             },
#             score_col=selection_col,
#             scoring_strategy=cfg.scoring_strategy,
#             selection_source="run_neighborhood_selection",
#             meta={
#                 "selector_mode": mode,
#                 "selection_path": "NEIGHBORHOOD_ROBUST",
#                 "selection_col": selection_col,
#                 "scoring_strategy": cfg.scoring_strategy,
#                 "pool_method": pool_method,
#                 "score_direction": "ascending" if ordering.ascending else "descending",
#                 "ordering_reason": ordering.reason,
#                 "anti_leak_policy": "selection_df_drop_test_prefix",
#                 "group_cols": group_cols,
#                 "arch_canonicalization": {
#                     "enabled": True,
#                     "family_col": "architecture_family",
#                     "subtype_col": "architecture_subtype",
#                 },
#             },
#         )

#         validation_profile_path = maybe_write_validation_profile(
#             top_performers=champions_df,
#             audit_df=audit_df,
#             test_cols=test_cols,
#         )

#         out.update(
#             {
#                 "master_df": audit_df,
#                 "leaderboard_path": leaderboard_path,
#                 "validation_profile_path": validation_profile_path,
#                 "selection_metric_used": selection_col,
#                 "scoring_strategy_used": cfg.scoring_strategy,
#             }
#         )
#         return out

#     # ----------------------------
#     # LEGACY gates path
#     # ----------------------------
#     ph_over = dict(getattr(cfg, "posthoc_overrides", None) or {})
#     ph_over.setdefault("out_dir", posthoc_dir)

#     if getattr(cfg, "arch_filter", None):
#         ph_over["arch_filter"] = cfg.arch_filter

#     requested_pool = str(ph_over.get("pool_method", "survivors")).lower()
#     if requested_pool in {"top_pct", "val_band"}:
#         logger.warning("[selection][LEGACY_LOCK] pool_method='%s' not allowed in legacy. Forcing 'survivors'.", requested_pool)
#     ph_over["pool_method"] = "survivors"

#     ph_over["metric_to_optimize"] = selection_col

#     lib_isb = dict(getattr(cfg, "lower_is_better", None) or {})
#     lib_isb.update(dict(ph_over.get("lower_is_better", {}) or {}))
#     ordering = resolve_ordering(selection_col, lower_is_better=lib_isb, default_ascending=True)

#     ph_over.setdefault("lower_is_better", {})
#     ph_over["lower_is_better"][selection_col] = ordering.ascending

#     if cfg.scoring_strategy == "robust_score" and selection_col == "robust_score":
#         robust_q = (ph_over.get("primary_quantile", {}) or {}).get("val_smape_agg", 0.6)
#         ph_over["primary_quantile"] = {"robust_score": robust_q}
#         ph_over.setdefault("metrics", [])
#         if "robust_score" not in ph_over["metrics"]:
#             ph_over["metrics"].append("robust_score")
#     else:
#         ph_over.setdefault("primary_quantile", {"val_smape_agg": 0.6})

#     ph_cfg = make_default_config(**ph_over)

#     logger.info(
#         "[selection] Running legacy filters | primary_quantiles=%s | selection_col=%s | pool_method=%s",
#         getattr(ph_cfg, "primary_quantile", None), selection_col, "survivors",
#     )

#     # anti-leak: selection uses selection_df (no test_*)
#     filter_result = run_distribution_filter(selection_df, ph_cfg)

#     # regret diagnostics from survivors (audit-only)
#     legacy_regret_diag = build_legacy_regret_diagnostics(
#         audit_df=audit_df,
#         survivors_df=as_df(filter_result.get("survivors", pd.DataFrame())),
#         top_performers=as_df(filter_result.get("top_performers", pd.DataFrame())),
#         selector_mode=mode,
#         selection_path="LEGACY_GATES",
#         score_col=selection_col,
#         scoring_strategy=cfg.scoring_strategy,
#         pool_method="survivors",
#         test_metric="test_smape_agg",
#         val_metric="val_smape_agg",
#     )

#     out = make_selection_result_contract(
#         master_df=audit_df,
#         filter_result=filter_result,
#         diagnostics=legacy_regret_diag if isinstance(legacy_regret_diag, pd.DataFrame) and not legacy_regret_diag.empty else None,
#         score_col=selection_col,
#         scoring_strategy=cfg.scoring_strategy,
#         selection_source="run_distribution_filter",
#         meta={
#             "selector_mode": mode,
#             "selection_path": "LEGACY_GATES",
#             "selection_col": selection_col,
#             "scoring_strategy": cfg.scoring_strategy,
#             "pool_method": "survivors",
#             "score_direction": "ascending" if ordering.ascending else "descending",
#             "ordering_reason": ordering.reason,
#             "anti_leak_policy": "selection_df_drop_test_prefix",
#             "arch_canonicalization": {
#                 "enabled": True,
#                 "family_col": "architecture_family",
#                 "subtype_col": "architecture_subtype",
#             },
#         },
#     )

#     top_performers = out.get("top_performers", pd.DataFrame())
#     thresholds = out.get("thresholds", pd.DataFrame())

#     # validation profile (artifact only; may reattach test_*)
#     validation_profile_path = None
#     if top_performers is None or top_performers.empty:
#         logger.warning("[selection] No champions survived the gates/pool.")
#     else:
#         if test_cols:
#             join_keys = _default_join_keys(audit_df, top_performers)
#             if join_keys:
#                 test_lookup = audit_df[join_keys + test_cols].drop_duplicates(subset=join_keys, keep="first")
#                 top_performers = top_performers.merge(test_lookup, on=join_keys, how="left")
#             else:
#                 logger.warning("[selection][artifact] Could not restore test_* cols for validation artifact (no join keys).")

#         out_path = (cfg.reports_dir.resolve() / f"{cfg.validation_run_name}.csv").resolve()
#         logger.info("[selection] Generating validation profile: %s", out_path)
#         _ = generate_profile_from_dataframe(
#             candidates_df=top_performers,
#             output_profile_path=out_path,
#             fixed_params={"seed": cfg.validation_seed, "plot": True},
#         )
#         validation_profile_path = str(out_path)

#     out.update(
#         {
#             "master_df": audit_df,
#             "thresholds": thresholds,
#             "leaderboard_path": leaderboard_path,
#             "validation_profile_path": validation_profile_path,
#             "selection_metric_used": selection_col,
#             "scoring_strategy_used": cfg.scoring_strategy,
#             "meta": {
#                 "selector_mode": mode,
#                 "selection_path": "LEGACY_GATES",
#                 "selection_col": selection_col,
#                 "scoring_strategy": cfg.scoring_strategy,
#                 "pool_method": "survivors",
#                 "score_direction": "ascending" if ordering.ascending else "descending",
#                 "ordering_reason": ordering.reason,
#                 "anti_leak_policy": "selection_df_drop_test_prefix",
#                 "arch_canonicalization": {
#                     "enabled": True,
#                     "family_col": "architecture_family",
#                     "subtype_col": "architecture_subtype",
#                 },
#                 "config_resolved": {
#                     "metric_to_optimize": getattr(ph_cfg, "metric_to_optimize", None),
#                     "primary_quantile": getattr(ph_cfg, "primary_quantile", None),
#                     "mad_guard": getattr(ph_cfg, "mad_guard", None),
#                     "valcum_gate": getattr(ph_cfg, "valcum_gate", None),
#                     "apply_pareto": getattr(ph_cfg, "apply_pareto", None),
#                     "arch_filter": getattr(ph_cfg, "arch_filter", None),
#                     "pool_method_requested": requested_pool,
#                 },
#             },
#         }
#     )
#     return out

def run_selection_pipeline(cfg: "SelectionRunConfig") -> Dict[str, Any]:
    """
    End-to-end HPO selection runner.

    Keeps legacy behavior intact, but adds a robust, modular PRI routing layer:

    Core semantics (unchanged):
      - scoring_strategy: how scores are PRODUCED during aggregation ("weighted_score" | "robust_score")
      - metric_to_optimize: which column is USED to order/select champions
      - selector_mode: dispatches LEGACY_* vs NEIGHBOR_*
      - anti-leak: selection NEVER sees test_* columns (audit can).

    PRI routing (new, plug-and-play):
      - If cfg.enable_pri_routing is True, load PRI policy JSON and route selector_mode per group.
      - Policy schema supported (your current PRI JSON):
          policy: [{"campaign":..., "well":..., "architecture":..., "pri_policy_mode":...}, ...]
        plus typical alternatives: recommended_mode/mode/selector_mode/etc.
      - Key mismatch hardened:
          * If policy uses meta.architecture_value="seq2" and rows have architecture="seq2"
            while leaderboard has architecture subtype (e.g. "Seq2PIN"), routing automatically
            drops 'architecture' from the join key and routes by (campaign, well).
      - Robust score availability:
          * If any routed mode needs robust_score, a second aggregation run is performed with
            scoring_strategy="robust_score" and robust_score is merged back into audit/selection DFs
            using stable keys (job_hash/trial_id/...); otherwise falls back gracefully.

    Canonical architecture grouping (kept):
      - Adds architecture_family ∈ {PINN, ARPS, DARTS, ...} and architecture_subtype
      - Neighborhood defaults to grouping by architecture_family to avoid Darts_* fragmentation.
    """
    import json
    import logging
    import re
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any, Dict, List, Mapping, Optional, Tuple, Iterable

    import pandas as pd

    from hpo.score_ordering import resolve_ordering
    from hpo.selection_contract import make_selection_result_contract

    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------------------
    def as_df(x: Any) -> pd.DataFrame:
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

    def mkdir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _safe_list(x: Any) -> List[Any]:
        return list(x) if isinstance(x, (list, tuple)) else ([] if x is None else [x])

    def _as_mode(x: Any) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        return s.upper() if s else None

    def mode_and_path(raw: Any) -> Tuple[str, str]:
        mode = str(raw or "LEGACY_WEIGHTED").upper().strip()
        mode_map = {
            "LEGACY_WEIGHTED": "LEGACY_GATES",
            "LEGACY_ROBUSTCOL": "LEGACY_GATES",
            "NEIGHBOR_TOP_PCT": "NEIGHBORHOOD_ROBUST",
            "NEIGHBOR_VAL_BAND": "NEIGHBORHOOD_ROBUST",
        }
        path = mode_map.get(mode)
        if path is None:
            logger.warning("[selection] Unknown selector_mode='%s'. Falling back to LEGACY_WEIGHTED.", mode)
            mode = "LEGACY_WEIGHTED"
            path = mode_map[mode]
        return mode, path

    def split_audit_selection(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        test_cols = [c for c in df.columns if str(c).startswith("test_")]
        sel = df.drop(columns=test_cols, errors="ignore") if test_cols else df
        return df, sel, test_cols

    # ------------------------------------------------------------------------------
    # Architecture canonicalization
    # ------------------------------------------------------------------------------
    def _arch_family(x: object) -> str:
        s = "" if x is None else str(x).strip()
        s0 = s.lower().replace(" ", "")
        if "darts" in s0:
            return "DARTS"
        if "arps" in s0:
            return "ARPS"
        if "seq2" in s0 or "pinn" in s0:
            return "PINN"
        return re.sub(r"[^0-9a-zA-Z_]+", "", s).upper() or "UNKNOWN"

    def ensure_arch_family_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        src = None
        for cand in ["architecture", "architecture_name"]:
            if cand in out.columns:
                src = cand
                break
        if src:
            out["architecture_subtype"] = out[src].astype(str)
            out["architecture_family"] = out[src].map(_arch_family)
        else:
            out["architecture_subtype"] = "UNKNOWN"
            out["architecture_family"] = "UNKNOWN"
        return out

    def normalize_group_cols(cols: List[str], df: pd.DataFrame) -> List[str]:
        """
        Map 'architecture' grouping to 'architecture_family' when available, to avoid subtype fragmentation.
        """
        if not cols:
            return cols
        out = []
        for c in cols:
            if c == "architecture" and "architecture_family" in df.columns:
                out.append("architecture_family")
            else:
                out.append(c)
        out2 = [c for c in out if c in df.columns]
        return out2 or out

    # ------------------------------------------------------------------------------
    # Selection column resolution (global vs per-mode)
    # ------------------------------------------------------------------------------
    def resolve_selection_col(mode: str, sel_df: pd.DataFrame) -> str:
        """
        Decide which column to use for ordering. Keeps current semantics:
          - LEGACY_ROBUSTCOL prefers robust_score
          - LEGACY_WEIGHTED prefers weighted_score
          - Otherwise uses cfg.metric_to_optimize or cfg.scoring_strategy fallback
        """
        target = getattr(cfg, "metric_to_optimize", None) or getattr(cfg, "scoring_strategy", None) or "weighted_score"

        if mode == "LEGACY_ROBUSTCOL" and (not getattr(cfg, "metric_to_optimize", None) or cfg.metric_to_optimize == "weighted_score"):
            target = "robust_score"
        if mode == "LEGACY_WEIGHTED" and not getattr(cfg, "metric_to_optimize", None):
            target = "weighted_score"

        if sel_df is not None and not sel_df.empty and target not in sel_df.columns:
            produced = getattr(cfg, "scoring_strategy", None)
            for cand in [produced, "robust_score", "weighted_score", "val_smape_agg", "val_smape_cum"]:
                if cand and cand in sel_df.columns:
                    logger.warning("[selection] selection_col='%s' missing. Falling back to '%s'.", target, cand)
                    target = cand
                    break
            else:
                logger.warning("[selection] selection_col='%s' missing; cols_sample=%s", target, list(sel_df.columns)[:30])
        return str(target)

    # ------------------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------------------
    def persist_master(audit_df: pd.DataFrame) -> str:
        p = (Path(cfg.reports_dir) / "hpo_master_leaderboard.csv").resolve()
        audit_df.to_csv(p, index=False)
        logger.info("[selection] Saved master leaderboard: %s", p)
        return str(p)

    def default_join_keys(audit_df: pd.DataFrame, tp: pd.DataFrame) -> List[str]:
        """
        Prefer stable identifiers when present; fall back to (campaign, well, architecture subtype) style.
        """
        key_candidates = [
            "job_hash", "trial_hash", "trial_id", "trial", "number", "optuna_trial_number", "experiment_id",
            "dataset", "campaign", "well",
            "architecture", "architecture_name",
            "architecture_subtype", "architecture_family",
            "physics_strategy", "strategy_display", "variant", "profile",
        ]
        return [c for c in key_candidates if c in audit_df.columns and c in tp.columns]

    def maybe_write_validation_profile(*, top_performers: pd.DataFrame, audit_df: pd.DataFrame, test_cols: List[str]) -> Optional[str]:
        if top_performers is None or top_performers.empty:
            logger.warning("[selection] No champions to export; skipping validation profile.")
            return None

        tp = top_performers.copy()
        if test_cols:
            join_keys = default_join_keys(audit_df, tp)
            if join_keys:
                test_lookup = audit_df[join_keys + test_cols].drop_duplicates(subset=join_keys, keep="first")
                tp = tp.merge(test_lookup, on=join_keys, how="left")
            else:
                logger.warning("[selection][artifact] Could not reattach test_* cols (no join keys).")

        out_path = (Path(cfg.reports_dir).resolve() / f"{cfg.validation_run_name}.csv").resolve()
        logger.info("[selection] Generating validation profile: %s", out_path)
        _ = generate_profile_from_dataframe(
            candidates_df=tp,
            output_profile_path=out_path,
            fixed_params={"seed": cfg.validation_seed, "plot": True},
        )
        return str(out_path)

    def empty_contract(*, audit_df: pd.DataFrame, selection_col: str, mode: str, selection_path: str) -> Dict[str, Any]:
        out = make_selection_result_contract(
            master_df=audit_df,
            filter_result={"top_performers": pd.DataFrame(), "thresholds": pd.DataFrame()},
            score_col=selection_col,
            scoring_strategy=cfg.scoring_strategy,
            selection_source="empty",
            meta={
                "selector_mode": mode,
                "selection_path": selection_path,
                "selection_col": selection_col,
                "scoring_strategy": cfg.scoring_strategy,
                "anti_leak_policy": "selection_df_drop_test_prefix",
            },
        )
        out.update(
            {
                "master_df": pd.DataFrame(),
                "leaderboard_path": None,
                "validation_profile_path": None,
                "selection_metric_used": selection_col,
                "scoring_strategy_used": cfg.scoring_strategy,
            }
        )
        return out

    # ------------------------------------------------------------------------------
    # PRI policy loading + routing model
    # ------------------------------------------------------------------------------
    @dataclass(frozen=True)
    class PriRoutingPlan:
        enabled: bool
        policy_path: Optional[str]
        group_cols: List[str]
        routing_map: Dict[Tuple[Any, ...], str]
        meta: Dict[str, Any]

    def resolve_pri_policy_path() -> Path:
        explicit = getattr(cfg, "pri_policy_path", None)
        if explicit:
            return Path(explicit).expanduser()
        return (Path(cfg.reports_dir) / "pri__hpo_master_leaderboard__selection_policy.json").resolve()

    def pick_first_list(d: Mapping[str, Any]) -> Optional[List[Any]]:
        for k in ["policy", "groups", "items", "rows", "decisions", "entries", "records"]:
            v = d.get(k)
            if isinstance(v, list):
                return v
        for k in ["data", "result", "output"]:
            v = d.get(k)
            if isinstance(v, dict):
                vv = pick_first_list(v)
                if vv is not None:
                    return vv
        return None

    def safe_routing_group_cols(df: pd.DataFrame) -> List[str]:
        """
        Prefer (campaign, well). This avoids mismatches where policy 'architecture' is a constant ("seq2")
        while leaderboard 'architecture' is subtype ("Seq2PIN"). Still supports fallback to well-only.
        """
        if df is None or df.empty:
            return []
        if "campaign" in df.columns and "well" in df.columns:
            return ["campaign", "well"]
        if "well" in df.columns:
            return ["well"]
        return []

    def build_pri_routing_plan(*, audit_df: pd.DataFrame) -> PriRoutingPlan:
        enabled = bool(getattr(cfg, "enable_pri_routing", False))
        if not enabled:
            return PriRoutingPlan(False, None, [], {}, {})

        p = resolve_pri_policy_path()
        if not p.exists():
            logger.warning("[selection][pri] policy not found: %s | routing disabled", p)
            return PriRoutingPlan(False, str(p), [], {}, {})

        try:
            policy = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[selection][pri] failed to parse policy JSON: %s | err=%s | routing disabled", p, e)
            return PriRoutingPlan(False, str(p), [], {}, {})

        if not isinstance(policy, dict):
            logger.warning("[selection][pri] policy JSON is not a dict: %s | routing disabled", p)
            return PriRoutingPlan(False, str(p), [], {}, {})

        meta = dict(policy.get("meta", {}) or {})
        items = pick_first_list(policy)
        if not isinstance(items, list) or not items:
            logger.warning("[selection][pri] policy list missing/empty: %s | routing disabled", p)
            return PriRoutingPlan(False, str(p), [], {}, meta)

        # Start with safe join cols based on data
        desired_cols = safe_routing_group_cols(audit_df)
        if not desired_cols:
            logger.warning("[selection][pri] routing enabled, but no usable group cols in data | routing disabled")
            return PriRoutingPlan(False, str(p), [], {}, meta)

        # If policy schema claims group_cols and includes architecture, we may still drop it.
        policy_group_cols = list(policy.get("group_cols") or [])
        eff_cols = list(desired_cols)

        # Heuristic: if policy rows have architecture == meta.architecture_value (constant), treat architecture as constant and drop it.
        arch_val = str(meta.get("architecture_value") or "").strip().lower()
        if arch_val and ("architecture" in policy_group_cols):
            same, total = 0, 0
            for r in items:
                if isinstance(r, dict) and "architecture" in r:
                    total += 1
                    if str(r.get("architecture")).strip().lower() == arch_val:
                        same += 1
            if total > 0 and (same / total) >= 0.90:
                # prefer (campaign, well) join; keep backwards compat by never requiring architecture
                eff_cols = [c for c in eff_cols if c != "architecture"]

        routing_map: Dict[Tuple[Any, ...], str] = {}

        # Supported mode keys (includes your PRI field!)
        mode_keys = [
            "pri_policy_mode",
            "recommended_mode",
            "mode",
            "selector_mode",
            "selection_mode",
            "chosen_mode",
        ]

        for row in items:
            if not isinstance(row, dict):
                continue

            mode_val = None
            for mk in mode_keys:
                if mk in row:
                    mode_val = _as_mode(row.get(mk))
                    break
            if not mode_val:
                continue

            # build key by eff_cols
            ok = True
            key_parts = []
            for c in eff_cols:
                if c not in row:
                    ok = False
                    break
                key_parts.append(row.get(c))
            if not ok:
                continue

            routing_map[tuple(key_parts)] = mode_val

        if not routing_map:
            logger.warning("[selection][pri] routing enabled, but policy map is empty after parsing. policy=%s | routing disabled", p)
            return PriRoutingPlan(False, str(p), eff_cols, {}, meta)

        logger.info(
            "[selection][pri] routing enabled | policy=%s | group_cols=%s | n_groups_in_policy=%d",
            p, eff_cols, len(routing_map),
        )
        return PriRoutingPlan(True, str(p), eff_cols, routing_map, meta)

    def write_routing_sidecars(*, reports_dir: Path, applied_df: pd.DataFrame, summary: Dict[str, Any]) -> None:
        try:
            p1 = (reports_dir / "pri_routing__applied.csv").resolve()
            applied_df.to_csv(p1, index=False)
            logger.info("[selection][pri] wrote routing applied: %s", p1)
        except Exception as e:
            logger.warning("[selection][pri] failed writing routing applied CSV: %s", e)

        try:
            p2 = (reports_dir / "pri_routing__summary.json").resolve()
            p2.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            logger.info("[selection][pri] wrote routing summary: %s", p2)
        except Exception as e:
            logger.warning("[selection][pri] failed writing routing summary JSON: %s", e)

    # ------------------------------------------------------------------------------
    # Robust-score merge for routing
    # ------------------------------------------------------------------------------
    def ensure_robust_score_if_needed(
        *,
        need_robust: bool,
        audit_df: pd.DataFrame,
        selection_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not need_robust:
            return audit_df, selection_df
        if "robust_score" in audit_df.columns and "robust_score" in selection_df.columns:
            return audit_df, selection_df

        logger.info("[selection][pri] robust_score required by routing; computing robust aggregation for merge...")

        robust_master = aggregate_all_campaign_results(
            results_dir=cfg.results_dir,
            metric_weights=cfg.metric_weights,
            lower_is_better=cfg.lower_is_better,
            campaign_names=None,
            scoring_strategy="robust_score",
        )
        robust_audit = as_df(robust_master)
        robust_audit, robust_sel, _ = split_audit_selection(robust_audit)

        # Prefer stable join keys
        preferred = ["job_hash", "trial_hash", "trial_id", "optuna_trial_number", "number"]
        join_keys = [k for k in preferred if k in audit_df.columns and k in robust_audit.columns]

        if not join_keys:
            # fallback keys (riskier but often sufficient)
            fallback = ["campaign", "well", "architecture", "physics_strategy", "variant", "profile"]
            join_keys = [k for k in fallback if k in audit_df.columns and k in robust_audit.columns]

        if not join_keys:
            logger.warning("[selection][pri] cannot merge robust_score (no join keys). Routing to ROBUSTCOL may fallback.")
            return audit_df, selection_df

        take_cols = list(dict.fromkeys(join_keys + ["robust_score"]))
        robust_lookup = robust_audit[take_cols].drop_duplicates(subset=join_keys, keep="first")

        audit2 = audit_df.merge(robust_lookup, on=join_keys, how="left")
        sel2 = selection_df.merge(robust_lookup, on=join_keys, how="left")

        missing = int(audit2["robust_score"].isna().sum()) if "robust_score" in audit2.columns else -1
        logger.info("[selection][pri] merged robust_score | join_keys=%s | missing_after_merge=%d", join_keys, missing)
        return audit2, sel2

    # ------------------------------------------------------------------------------
    # Core selection runners (legacy / neighborhood) operating on subsets
    # ------------------------------------------------------------------------------
    def run_neighbor(
        *,
        audit_sub: pd.DataFrame,
        mode_override: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        from hpo.neighborhood_selection import run_neighborhood_selection

        mo, _ = mode_and_path(mode_override)
        over = dict(getattr(cfg, "neighborhood_overrides", None) or {})

        pool_method = str(over.get("pool_method") or ("top_pct" if mo == "NEIGHBOR_TOP_PCT" else "val_band")).lower()
        pool_cfg = dict(over.get("pool_cfg", {}) or {})
        robust_cfg = dict(over.get("robust_cfg", {}) or {})

        default_group_cols = ["dataset", "well", "architecture_family"] if "architecture_family" in audit_sub.columns else ["dataset", "well", "architecture"]
        group_cols = list(over.get("group_cols") or default_group_cols)
        group_cols = normalize_group_cols(group_cols, audit_sub)

        logger.info(
            "[selection][dispatch] NEIGHBORHOOD_ROBUST | mode=%s pool_method=%s group_cols=%s | subset_rows=%d",
            mo, pool_method, group_cols, int(audit_sub.shape[0]),
        )

        champs, diag = run_neighborhood_selection(
            audit_df=audit_sub,
            pool_method=pool_method,
            pool_cfg=pool_cfg,
            robust_cfg=robust_cfg,
            group_cols=group_cols,
        )
        meta = {"selector_mode": mo, "selection_path": "NEIGHBORHOOD_ROBUST", "pool_method": pool_method, "group_cols": group_cols}
        return as_df(champs), as_df(diag), meta

    def run_legacy(
        *,
        selection_sub: pd.DataFrame,
        audit_sub: pd.DataFrame,
        mode_override: str,
        out_dir: Path,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], str]:
        mo, _ = mode_and_path(mode_override)
        sel_col_local = resolve_selection_col(mo, selection_sub)

        ph_over = dict(getattr(cfg, "posthoc_overrides", None) or {})
        ph_over.setdefault("out_dir", out_dir)

        if getattr(cfg, "arch_filter", None):
            ph_over["arch_filter"] = cfg.arch_filter

        requested_pool = str(ph_over.get("pool_method", "survivors")).lower()
        if requested_pool in {"top_pct", "val_band"}:
            logger.warning("[selection][LEGACY_LOCK] pool_method='%s' not allowed in legacy. Forcing 'survivors'.", requested_pool)
        ph_over["pool_method"] = "survivors"

        ph_over["metric_to_optimize"] = sel_col_local

        lib_isb = dict(getattr(cfg, "lower_is_better", None) or {})
        lib_isb.update(dict(ph_over.get("lower_is_better", {}) or {}))
        ordering = resolve_ordering(sel_col_local, lower_is_better=lib_isb, default_ascending=True)

        ph_over.setdefault("lower_is_better", {})
        ph_over["lower_is_better"][sel_col_local] = ordering.ascending

        if cfg.scoring_strategy == "robust_score" and sel_col_local == "robust_score":
            robust_q = (ph_over.get("primary_quantile", {}) or {}).get("val_smape_agg", 0.6)
            ph_over["primary_quantile"] = {"robust_score": robust_q}
            ph_over.setdefault("metrics", [])
            if "robust_score" not in ph_over["metrics"]:
                ph_over["metrics"].append("robust_score")
        else:
            ph_over.setdefault("primary_quantile", {"val_smape_agg": 0.6})

        ph_cfg = make_default_config(**ph_over)

        logger.info(
            "[selection][dispatch] LEGACY | mode=%s selection_col=%s | subset_rows=%d | out_dir=%s",
            mo, sel_col_local, int(selection_sub.shape[0]), getattr(ph_cfg, "out_dir", None),
        )

        filter_result = run_distribution_filter(selection_sub, ph_cfg)

        legacy_regret_diag = build_legacy_regret_diagnostics(
            audit_df=audit_sub,
            survivors_df=as_df(filter_result.get("survivors", pd.DataFrame())),
            top_performers=as_df(filter_result.get("top_performers", pd.DataFrame())),
            selector_mode=mo,
            selection_path="LEGACY_GATES",
            score_col=sel_col_local,
            scoring_strategy=cfg.scoring_strategy,
            pool_method="survivors",
            test_metric="test_smape_agg",
            val_metric="val_smape_agg",
        )

        meta = {
            "selector_mode": mo,
            "selection_path": "LEGACY_GATES",
            "selection_col": sel_col_local,
            "pool_method": "survivors",
            "score_direction": "ascending" if ordering.ascending else "descending",
            "ordering_reason": ordering.reason,
            "pool_method_requested": requested_pool,
        }
        return (
            as_df(filter_result.get("top_performers", pd.DataFrame())),
            as_df(filter_result.get("thresholds", pd.DataFrame())),
            as_df(legacy_regret_diag),
            meta,
            sel_col_local,
        )

    # ------------------------------------------------------------------------------
    # Subsetting helper
    # ------------------------------------------------------------------------------
    def subset_by_keys(df: pd.DataFrame, group_cols: List[str], keys: List[Tuple[Any, ...]]) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if not keys:
            return df.iloc[0:0].copy()
        key_df = pd.DataFrame(keys, columns=group_cols)
        return df.merge(key_df.drop_duplicates(), on=group_cols, how="inner")

    # ------------------------------------------------------------------------------
    # Dirs
    # ------------------------------------------------------------------------------
    mkdir(Path(cfg.reports_dir))
    posthoc_dir = mkdir(Path(cfg.reports_dir) / "posthoc_analysis")

    # ------------------------------------------------------------------------------
    # Global mode/path (default behavior)
    # ------------------------------------------------------------------------------
    mode, selection_path = mode_and_path(getattr(cfg, "selector_mode", None))
    logger.info(
        "[selection] selector_mode=%s selection_path=%s scoring_strategy=%s metric_to_optimize=%s",
        mode, selection_path, getattr(cfg, "scoring_strategy", None), getattr(cfg, "metric_to_optimize", None),
    )

    # ------------------------------------------------------------------------------
    # Aggregate once (base)
    # ------------------------------------------------------------------------------
    logger.info("[selection] Aggregating results from %s", cfg.results_dir)
    master_df = aggregate_all_campaign_results(
        results_dir=cfg.results_dir,
        metric_weights=cfg.metric_weights,
        lower_is_better=cfg.lower_is_better,
        campaign_names=None,
        scoring_strategy=cfg.scoring_strategy,
    )
    audit_df = as_df(master_df)
    audit_df, selection_df, test_cols = split_audit_selection(audit_df)

    audit_df = ensure_arch_family_cols(audit_df)
    selection_df = ensure_arch_family_cols(selection_df)

    logger.info(
        "[selection][anti_leak] audit_cols=%d selection_cols=%d dropped_test_cols=%d",
        int(audit_df.shape[1]), int(selection_df.shape[1]), int(len(test_cols)),
    )

    # Global selection col used for consolidated plotting context (unchanged)
    selection_col = resolve_selection_col(mode, selection_df)
    logger.info("[selection] selection_col=%s (DECISION) | scoring_strategy=%s | selector_mode=%s", selection_col, cfg.scoring_strategy, mode)

    # Audit-only plots (safe)
    _generate_selection_plots(
        audit_df,
        selection_col,
        plot_architecture_performance=getattr(cfg, "plot_architecture_performance", False),
        plot_hparam_importance_per_well=getattr(cfg, "plot_hparam_importance_per_well", False),
        plot_champions_per_well=getattr(cfg, "plot_champions_per_well", False),
        plot_summary_bars=getattr(cfg, "plot_summary_bars", False),
    )

    if audit_df.empty:
        logger.warning("[selection] No results found. Did you run campaigns?")
        return empty_contract(audit_df=audit_df, selection_col=selection_col, mode=mode, selection_path="EMPTY")

    leaderboard_path = persist_master(audit_df)

    # ------------------------------------------------------------------------------
    # PRI routing (optional)
    # ------------------------------------------------------------------------------
    pri_plan = build_pri_routing_plan(audit_df=audit_df)

    if pri_plan.enabled:
        # Determine which groups exist in data and which mode each will use
        group_cols = pri_plan.group_cols
        routing_map = pri_plan.routing_map

        groups: List[Tuple[Tuple[Any, ...], str]] = []
        for key_vals, _dfg in audit_df.groupby(group_cols, dropna=False):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            key = tuple(key_vals)
            chosen = routing_map.get(key) or mode  # fallback per group
            groups.append((key, str(chosen).upper().strip()))

        counts: Dict[str, int] = {}
        applied_rows: List[Dict[str, Any]] = []
        for key, chosen in groups:
            counts[chosen] = counts.get(chosen, 0) + 1
            row = {c: (key[i] if i < len(key) else None) for i, c in enumerate(group_cols)}
            row.update({"chosen_mode": chosen})
            applied_rows.append(row)

        applied_df = pd.DataFrame(applied_rows) if applied_rows else pd.DataFrame(columns=group_cols + ["chosen_mode"])
        summary = {
            "pri_routing_enabled": True,
            "policy_path": pri_plan.policy_path,
            "group_cols": group_cols,
            "n_groups_in_data": int(len(groups)),
            "n_groups_in_policy": int(len(routing_map)),
            "counts_by_mode": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "policy_meta": pri_plan.meta,
        }
        write_routing_sidecars(reports_dir=Path(cfg.reports_dir), applied_df=applied_df, summary=summary)

        # If any routed mode needs robust_score, ensure it's available
        any_needs_robust = any(m == "LEGACY_ROBUSTCOL" for m in counts.keys())
        audit_df, selection_df = ensure_robust_score_if_needed(
            need_robust=any_needs_robust,
            audit_df=audit_df,
            selection_df=selection_df,
        )

        # Partition keys by chosen mode (run per-slice)
        mode_to_keys: Dict[str, List[Tuple[Any, ...]]] = {}
        for key, chosen in groups:
            mode_to_keys.setdefault(chosen, []).append(key)

        all_top: List[pd.DataFrame] = []
        all_thr: List[pd.DataFrame] = []
        all_diag: List[pd.DataFrame] = []
        meta_modes: List[Dict[str, Any]] = []

        for chosen_mode, keys in mode_to_keys.items():
            chosen_mode_u = str(chosen_mode).upper().strip()
            _, chosen_path = mode_and_path(chosen_mode_u)

            audit_sub = subset_by_keys(audit_df, group_cols, keys)
            selection_sub = subset_by_keys(selection_df, group_cols, keys)

            if audit_sub is None or audit_sub.empty:
                continue

            if chosen_path == "NEIGHBORHOOD_ROBUST":
                champs, diag, meta_local = run_neighbor(audit_sub=audit_sub, mode_override=chosen_mode_u)
                if not champs.empty:
                    champs = champs.copy()
                    champs["pri_routed_mode"] = chosen_mode_u
                    all_top.append(champs)
                if not diag.empty:
                    diag = diag.copy()
                    diag["pri_routed_mode"] = chosen_mode_u
                    all_diag.append(diag)
                meta_modes.append(meta_local)
                continue

            # LEGACY slice: isolate out_dir per routed mode to avoid overwrite
            out_dir = mkdir(Path(cfg.reports_dir) / "posthoc_analysis" / f"pri_routing__{chosen_mode_u.lower()}")
            tp, thr, diag, meta_local, _sel_col_local = run_legacy(
                selection_sub=selection_sub,
                audit_sub=audit_sub,
                mode_override=chosen_mode_u,
                out_dir=out_dir,
            )
            if not tp.empty:
                tp = tp.copy()
                tp["pri_routed_mode"] = chosen_mode_u
                all_top.append(tp)
            if not thr.empty:
                thr = thr.copy()
                thr["pri_routed_mode"] = chosen_mode_u
                all_thr.append(thr)
            if not diag.empty:
                diag = diag.copy()
                diag["pri_routed_mode"] = chosen_mode_u
                all_diag.append(diag)

            meta_modes.append(meta_local)

        champions_df = pd.concat(all_top, ignore_index=True) if all_top else pd.DataFrame()
        thresholds_df = pd.concat(all_thr, ignore_index=True) if all_thr else pd.DataFrame()
        diagnostics_df = pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame()

        validation_profile_path = maybe_write_validation_profile(
            top_performers=champions_df,
            audit_df=audit_df,
            test_cols=test_cols,
        )

        out = make_selection_result_contract(
            master_df=audit_df,
            filter_result={
                "top_performers": champions_df,
                "thresholds": thresholds_df,
                "diagnostics": diagnostics_df if not diagnostics_df.empty else None,
            },
            score_col=selection_col,  # global context for consolidated plots
            scoring_strategy=cfg.scoring_strategy,
            selection_source="pri_routing_dispatch",
            meta={
                "pri_routing_enabled": True,
                "pri_policy_path": pri_plan.policy_path,
                "pri_group_cols": group_cols,
                "pri_counts_by_mode": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                "pri_policy_meta": pri_plan.meta,
                "global_selector_mode": mode,
                "global_selection_path": selection_path,
                "global_selection_col_for_plots": selection_col,
                "anti_leak_policy": "selection_df_drop_test_prefix",
                "arch_canonicalization": {"enabled": True, "family_col": "architecture_family", "subtype_col": "architecture_subtype"},
                "per_mode_meta": meta_modes,
            },
        )

        out.update(
            {
                "master_df": audit_df,
                "thresholds": thresholds_df,
                "leaderboard_path": leaderboard_path,
                "validation_profile_path": validation_profile_path,
                "selection_metric_used": selection_col,
                "scoring_strategy_used": cfg.scoring_strategy,
            }
        )
        return out

    # ------------------------------------------------------------------------------
    # No PRI routing: preserve old behavior (unchanged semantics)
    # ------------------------------------------------------------------------------
    if selection_path == "NEIGHBORHOOD_ROBUST":
        champs, diag, meta_local = run_neighbor(audit_sub=audit_df, mode_override=mode)

        ordering = resolve_ordering(selection_col, lower_is_better=cfg.lower_is_better, default_ascending=True)

        out = make_selection_result_contract(
            master_df=audit_df,
            filter_result={"top_performers": champs, "thresholds": pd.DataFrame(), "diagnostics": diag},
            score_col=selection_col,
            scoring_strategy=cfg.scoring_strategy,
            selection_source="run_neighborhood_selection",
            meta={
                "selector_mode": meta_local.get("selector_mode", mode),
                "selection_path": "NEIGHBORHOOD_ROBUST",
                "selection_col": selection_col,
                "scoring_strategy": cfg.scoring_strategy,
                "pool_method": meta_local.get("pool_method"),
                "score_direction": "ascending" if ordering.ascending else "descending",
                "ordering_reason": ordering.reason,
                "anti_leak_policy": "selection_df_drop_test_prefix",
                "group_cols": meta_local.get("group_cols"),
                "arch_canonicalization": {"enabled": True, "family_col": "architecture_family", "subtype_col": "architecture_subtype"},
            },
        )

        validation_profile_path = maybe_write_validation_profile(
            top_performers=champs,
            audit_df=audit_df,
            test_cols=test_cols,
        )

        out.update(
            {
                "master_df": audit_df,
                "leaderboard_path": leaderboard_path,
                "validation_profile_path": validation_profile_path,
                "selection_metric_used": selection_col,
                "scoring_strategy_used": cfg.scoring_strategy,
            }
        )
        return out

    # LEGACY path (unchanged)
    ph_over = dict(getattr(cfg, "posthoc_overrides", None) or {})
    ph_over.setdefault("out_dir", posthoc_dir)

    if getattr(cfg, "arch_filter", None):
        ph_over["arch_filter"] = cfg.arch_filter

    requested_pool = str(ph_over.get("pool_method", "survivors")).lower()
    if requested_pool in {"top_pct", "val_band"}:
        logger.warning("[selection][LEGACY_LOCK] pool_method='%s' not allowed in legacy. Forcing 'survivors'.", requested_pool)
    ph_over["pool_method"] = "survivors"

    ph_over["metric_to_optimize"] = selection_col

    lib_isb = dict(getattr(cfg, "lower_is_better", None) or {})
    lib_isb.update(dict(ph_over.get("lower_is_better", {}) or {}))
    ordering = resolve_ordering(selection_col, lower_is_better=lib_isb, default_ascending=True)

    ph_over.setdefault("lower_is_better", {})
    ph_over["lower_is_better"][selection_col] = ordering.ascending

    if cfg.scoring_strategy == "robust_score" and selection_col == "robust_score":
        robust_q = (ph_over.get("primary_quantile", {}) or {}).get("val_smape_agg", 0.6)
        ph_over["primary_quantile"] = {"robust_score": robust_q}
        ph_over.setdefault("metrics", [])
        if "robust_score" not in ph_over["metrics"]:
            ph_over["metrics"].append("robust_score")
    else:
        ph_over.setdefault("primary_quantile", {"val_smape_agg": 0.6})

    ph_cfg = make_default_config(**ph_over)

    logger.info(
        "[selection] Running legacy filters | primary_quantiles=%s | selection_col=%s | pool_method=%s",
        getattr(ph_cfg, "primary_quantile", None), selection_col, "survivors",
    )

    filter_result = run_distribution_filter(selection_df, ph_cfg)

    legacy_regret_diag = build_legacy_regret_diagnostics(
        audit_df=audit_df,
        survivors_df=as_df(filter_result.get("survivors", pd.DataFrame())),
        top_performers=as_df(filter_result.get("top_performers", pd.DataFrame())),
        selector_mode=mode,
        selection_path="LEGACY_GATES",
        score_col=selection_col,
        scoring_strategy=cfg.scoring_strategy,
        pool_method="survivors",
        test_metric="test_smape_agg",
        val_metric="val_smape_agg",
    )

    out = make_selection_result_contract(
        master_df=audit_df,
        filter_result=filter_result,
        diagnostics=legacy_regret_diag if not as_df(legacy_regret_diag).empty else None,
        score_col=selection_col,
        scoring_strategy=cfg.scoring_strategy,
        selection_source="run_distribution_filter",
        meta={
            "selector_mode": mode,
            "selection_path": "LEGACY_GATES",
            "selection_col": selection_col,
            "scoring_strategy": cfg.scoring_strategy,
            "pool_method": "survivors",
            "score_direction": "ascending" if ordering.ascending else "descending",
            "ordering_reason": ordering.reason,
            "anti_leak_policy": "selection_df_drop_test_prefix",
            "arch_canonicalization": {"enabled": True, "family_col": "architecture_family", "subtype_col": "architecture_subtype"},
        },
    )

    top_performers = out.get("top_performers", pd.DataFrame())
    thresholds = out.get("thresholds", pd.DataFrame())

    validation_profile_path = None
    if top_performers is None or top_performers.empty:
        logger.warning("[selection] No champions survived the gates/pool.")
    else:
        if test_cols:
            join_keys = default_join_keys(audit_df, top_performers)
            if join_keys:
                test_lookup = audit_df[join_keys + test_cols].drop_duplicates(subset=join_keys, keep="first")
                top_performers = top_performers.merge(test_lookup, on=join_keys, how="left")
            else:
                logger.warning("[selection][artifact] Could not restore test_* cols for validation artifact (no join keys).")

        out_path = (Path(cfg.reports_dir).resolve() / f"{cfg.validation_run_name}.csv").resolve()
        logger.info("[selection] Generating validation profile: %s", out_path)
        _ = generate_profile_from_dataframe(
            candidates_df=top_performers,
            output_profile_path=out_path,
            fixed_params={"seed": cfg.validation_seed, "plot": True},
        )
        validation_profile_path = str(out_path)

    out.update(
        {
            "master_df": audit_df,
            "thresholds": thresholds,
            "leaderboard_path": leaderboard_path,
            "validation_profile_path": validation_profile_path,
            "selection_metric_used": selection_col,
            "scoring_strategy_used": cfg.scoring_strategy,
            "meta": {
                "selector_mode": mode,
                "selection_path": "LEGACY_GATES",
                "selection_col": selection_col,
                "scoring_strategy": cfg.scoring_strategy,
                "pool_method": "survivors",
                "score_direction": "ascending" if ordering.ascending else "descending",
                "ordering_reason": ordering.reason,
                "anti_leak_policy": "selection_df_drop_test_prefix",
                "arch_canonicalization": {"enabled": True, "family_col": "architecture_family", "subtype_col": "architecture_subtype"},
                "config_resolved": {
                    "metric_to_optimize": getattr(ph_cfg, "metric_to_optimize", None),
                    "primary_quantile": getattr(ph_cfg, "primary_quantile", None),
                    "mad_guard": getattr(ph_cfg, "mad_guard", None),
                    "valcum_gate": getattr(ph_cfg, "valcum_gate", None),
                    "apply_pareto": getattr(ph_cfg, "apply_pareto", None),
                    "arch_filter": getattr(ph_cfg, "arch_filter", None),
                    "pool_method_requested": requested_pool,
                },
            },
        }
    )
    return out




# -----------------------------------------------------------------------------
# select_champions_from_df (refatorado p/ logs + compact block; lógica inalterada)
# -----------------------------------------------------------------------------
def select_champions_from_df(
    master_df: pd.DataFrame,
    selection_cfg_overrides: Optional[Dict[str, Any]] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    lower_is_better: Optional[Dict[str, bool]] = None,
    scoring_strategy: str = "weighted_score",
    ensure_topk_per_group: bool = True,
    enable_familywise_filter: bool = True,
    enable_backfill_from_family: bool = True,
) -> pd.DataFrame:
    """
    Champion selector with strict K control (unchanged logic).
    Compact logging mode collapses per-well/family chatter into a summary block.
    """
    import os
    import logging
    import numpy as np
    import pandas as pd

    # --- logging facade (uses your existing utils) ---------------------------
    from common.log_utils import (
        info, warn, ok, err,
        log_block, summarize_stage1,
        is_compact_logging, effective_log_width,
        vlog, silence_loggers, parse_silenced_from_env
    )

    log = logging.getLogger("champion_select")
    COMPACT = is_compact_logging(None)  # env/CFG-driven; no cfg object here
    WIDTH = effective_log_width(None, fallback=100)

    # additional opt-in silencers via env
    extra_silencers = parse_silenced_from_env()  # PHASE_SILENCE_LOGGERS
    # default silencers when in compact mode (safe guesses; ignore if absent)
    default_noisy = ["hpo.distribution_filter", "distribution_filter", "robust_filter"]
    silent_names = (default_noisy + extra_silencers) if COMPACT else extra_silencers

    # ---- input checks -------------------------------------------------------
    if master_df is None or master_df.empty:
        warn("[select] Input DataFrame is empty.")
        return pd.DataFrame()

    df0 = master_df.copy()
    has_well = "well" in df0.columns
    has_job  = "job_hash" in df0.columns

    if not has_well:
        warn("[select] Column 'well' missing; selection may misbehave.")
    if not has_job:
        warn("[select] Column 'job_hash' missing; Top-K clamp/backfill may misbehave.")

    # leaderboard lines can be very chatty; keep them only in verbose mode
    if "campaign" in df0.columns and has_well and not COMPACT:
        if has_job:
            by_cw = df0.groupby(["campaign","well"], as_index=False)["job_hash"].nunique()
            for _, r in by_cw.iterrows():
                info("✅ Loaded leaderboard for campaign '%s' | well=%s with %d trials.",
                     r["campaign"], r["well"], int(r["job_hash"]))
        else:
            by_cw = df0.groupby(["campaign","well"], as_index=False).size().rename(columns={"size":"n"})
            for _, r in by_cw.iterrows():
                info("✅ Loaded leaderboard for campaign '%s' | well=%s with %d trials.",
                     r["campaign"], r["well"], int(r["n"]))

    # ---- 1) score once ------------------------------------------------------
    df_scored = df0.copy()
    if scoring_strategy == "weighted_score":
        eff_weights = metric_weights or {"val_smape_agg": 1.0}
        eff_dir = {"val_smape_agg": True, "weighted_score": True}
        if isinstance(lower_is_better, dict):
            eff_dir.update(lower_is_better)
        df_scored = add_weighted_score(df_scored, metric_weights=eff_weights, lower_is_better=eff_dir)
        score_col = "weighted_score"
    elif scoring_strategy == "robust_score":
        if "rank_gap_norm" not in df_scored.columns:
            df_scored["rank_gap_norm"] = compute_rank_gap(df_scored)
        if "neighbor_iqr_cum_norm" not in df_scored.columns:
            df_scored["neighbor_iqr_cum_norm"] = compute_neighbor_iqr(df_scored)
        df_scored = add_robust_score(df_scored)
        score_col = "robust_score"
    else:
        raise ValueError(f"Unknown scoring_strategy: {scoring_strategy}")

    # ---- 2) build PostHoc config & wire K everywhere -----------------------
    cfg = make_default_config(**(selection_cfg_overrides or {}))

    # ensure metric_to_optimize and direction for the score (unchanged)
    try:
        if not getattr(cfg, "metric_to_optimize", None):
            setattr(cfg, "metric_to_optimize", score_col)
    except Exception:
        pass
    lib = getattr(cfg, "lower_is_better", None)
    if not isinstance(lib, dict):
        lib = {}
    lib.setdefault(score_col, True)
    try:
        setattr(cfg, "lower_is_better", lib)
    except Exception:
        pass
    asc = bool((getattr(cfg, "lower_is_better", {}) or {}).get(score_col, True))

    def _pos(x):
        try:
            x = int(x)
            return x if x > 0 else None
        except Exception:
            return None

    K_override = _pos((selection_cfg_overrides or {}).get("top_strategies_per_well") if isinstance(selection_cfg_overrides, dict) else None)
    K_cfg1 = _pos(getattr(cfg, "top_strategies_per_well", None))
    K_cfg2 = _pos(getattr(cfg, "top_k_per_well", None))  # legacy
    K = next((v for v in (K_override, K_cfg1, K_cfg2) if v is not None), None)
    if K is not None:
        for attr in ("top_strategies_per_well", "top_n_strategies", "top_k_per_well"):
            try: setattr(cfg, attr, int(K))
            except Exception: pass
        try:
            pk = getattr(cfg, "per_strategy_k", None)
            if pk is None or int(pk) < 1:
                setattr(cfg, "per_strategy_k", int(K))
            else:
                setattr(cfg, "per_strategy_k", max(int(pk), 1))
        except Exception:
            pass
        try:
            if getattr(cfg, "min_arch_diversity", None) is not None:
                setattr(cfg, "min_arch_diversity", 0)
        except Exception:
            pass

        info("[select] K wired: K=%s → top_strategies_per_well=%s, top_n_strategies=%s, top_k_per_well=%s, per_strategy_k=%s, min_arch_diversity=%s",
             K,
             getattr(cfg, "top_strategies_per_well", None),
             getattr(cfg, "top_n_strategies", None),
             getattr(cfg, "top_k_per_well", None),
             getattr(cfg, "per_strategy_k", None),
             getattr(cfg, "min_arch_diversity", None),
             )
    else:
        info("[select] No K resolved; selector will not clamp internally.")

    # ---- 3) family helpers --------------------------------------------------
    def detect_family_whole(df: pd.DataFrame) -> str:
        cols = set(df.columns)
        if {"variant","solver","weighting","loss"}.issubset(cols):           return "arps"
        if {"profile","n_epochs"}.issubset(cols):                            return "darts"
        if {"physics_strategy","epochs"}.issubset(cols):                     return "seq2"
        return "generic"

    def split_by_family(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        fam_all = detect_family_whole(df)
        if fam_all != "generic":
            return {fam_all: df}
        fam_col = next((c for c in ["arch","architecture","architecture_name","family"] if c in df.columns), None)
        if fam_col is None:
            return {"generic": df}
        s = df[fam_col].astype(str).str.lower()
        fams = {
            "arps":  df[s.str.contains("arps",  na=False)],
            "seq2":  df[(s.str.contains("seq2", na=False) | s.str.contains("pinn", na=False)) & ~s.str.contains("arps", na=False)],
            "darts": df[s.str.contains("darts", na=False) & ~s.str.contains("arps", na=False) & ~(s.str.contains("seq2", na=False) | s.str.contains("pinn", na=False))],
        }
        leftovers = df[~(s.str.contains("arps", na=False) |
                         s.str.contains("darts", na=False) |
                         s.str.contains("seq2", na=False) |
                         s.str.contains("pinn", na=False))]
        out = {k:v for k,v in fams.items() if not v.empty}
        if not leftovers.empty: out["generic"] = leftovers
        return out

    def log_pool_stats(df: pd.DataFrame, tag: str):
        """
        Loga um resumo do pool:
          - total de linhas
          - nº de wells
          - preview por (well[, arch]) com contagem de job_hash.
        Em modo compact, mantém o log em uma linha só.
        """
        if df is None or df.empty:
            info("----- %s ----- total_rows=0 | wells=0", tag)
            return

        total = len(df)
        uniq_w = df["well"].nunique() if "well" in df.columns else "-"

        # constrói um preview por (well, arch) se as colunas existirem
        preview = ""
        if has_well and has_job:
            df_tmp = df.copy()
            grp_cols = ["well"]
            if "arch" in df_tmp.columns:
                grp_cols.append("arch")
            elif "architecture" in df_tmp.columns:
                grp_cols.append("architecture")
            elif "architecture_name" in df_tmp.columns:
                grp_cols.append("architecture_name")

            try:
                cnt = (
                    df_tmp.groupby(grp_cols)["job_hash"]
                          .nunique()
                          .reset_index()
                )
                rows = []
                for _, r in cnt.head(8).iterrows():
                    if len(grp_cols) == 2:
                        rows.append(f"{r['well']}/{str(r[grp_cols[1]])}={int(r['job_hash'])}")
                    else:
                        rows.append(f"{r['well']}={int(r['job_hash'])}")
                preview = " | " + ", ".join(rows) if rows else ""
            except Exception:
                preview = ""

        if COMPACT:
            info("----- %s ----- total_rows=%s | wells=%s%s",
                 tag, f"{total:,}", uniq_w, preview)
            return

        # modo verboso (já existia, mantido)
        if not has_well:
            info("[select] %s: total rows=%d", tag, total)
            return

        if has_job:
            cnt_w = df.groupby("well")["job_hash"].nunique()
        else:
            cnt_w = df.groupby("well").size()

        info("----- %s -----", tag)
        for w, n in cnt_w.items():
            info("well=%s → %d trials", w, int(n))


    # ---- 4) family×well filtering with hard K from SURVIVORS ---------------
    log_pool_stats(df_scored, "POOL BEFORE FILTER")

    pools = split_by_family(df_scored) if enable_familywise_filter else {"all": df_scored}
    picked_blocks: list[pd.DataFrame] = []

    # compact-mode: silence noisy third-party loggers while filtering
    with silence_loggers(silent_names, level=logging.WARNING):
        for fam, fam_df in pools.items():
            if fam_df.empty:
                continue
            iterator = fam_df.groupby("well") if has_well else [("all", fam_df)]
            for well_name, g in iterator:
                try:
                    res = run_distribution_filter(g, cfg)
                except Exception as e:
                    warn("[select] Filtering failed for fam=%s well=%s: %s", fam, well_name, e)
                    continue

                survivors = res.get("survivors")
                gate_counts = res.get("gate_counts")

                # diagnostics → only in verbose mode
                if not COMPACT:
                    if gate_counts is not None and not getattr(gate_counts, "empty", True):
                        row = gate_counts.iloc[0]
                        info("[survivors][fam=%s] well=%s: kept_by_gates=%s/%s",
                             fam, row.get(getattr(cfg, "well_col", "well"), well_name),
                             int(row.get("kept_by_gates", 0)), int(row.get("total_in_pool", 0)))
                    elif survivors is not None:
                        info("[survivors][fam=%s] well=%s: kept_by_gates=%d/%d", fam, well_name, len(survivors), len(g))

                # Hard decision: ignore any 'top_performers' returned upstream.
                if survivors is None or getattr(survivors, "empty", True):
                    if not COMPACT:
                        info("[final][fam=%s] well=%s: picked_final=0/%d (from survivors), total=0/%d",
                             fam, well_name, 0, len(g))
                    continue

                take_K = K if K is not None else len(survivors)
                use_col = score_col if score_col in survivors.columns else "val_smape_agg"
                picked = survivors.sort_values(use_col, ascending=asc).head(take_K).copy()

                if not COMPACT:
                    info("[final][fam=%s] well=%s: picked_final=%d/%d (from survivors), total=%d/%d",
                         fam, well_name, len(picked), len(survivors), len(picked), len(g))

                picked_blocks.append(picked)

    champions = pd.concat(picked_blocks, ignore_index=True) if picked_blocks else pd.DataFrame()
    if champions.empty:
        warn("[select] No champions after family×well filtering.")
        return champions

    log_pool_stats(champions, "POOL AFTER FILTER")

    # ---- 5) Optional Top-K per (well, arch) clamp ---------------------------
    if not ensure_topk_per_group:
        info("[select] Final clamp disabled (ensure_topk_per_group=False).")
        return champions.copy()

    well_col = "well" if "well" in champions.columns else None
    arch_col = next((c for c in ["arch","architecture","architecture_name","family"] if c in champions.columns), None)
    if well_col is None:
        warn("[select] 'well' column missing; skipping Top-K clamp.")
        return champions.copy()
    if arch_col is None:
        arch_col = "_arch_family"
        champions = champions.copy()
        champions[arch_col] = "generic"

    # resolve K again (same precedence)
    K2 = K
    if K2 is None:
        K2 = next((v for v in (
            _pos(getattr(cfg, "top_strategies_per_well", None)),
            _pos(getattr(cfg, "top_n_strategies", None)),
            _pos(getattr(cfg, "top_k_per_well", None)),
        ) if v is not None), None)

    champs_sorted = champions.sort_values([well_col, arch_col, score_col], ascending=[True, True, asc])
    if K2 is None:
        info("[select] No Top-K clamp applied at the end (no K resolved).")
        out = champs_sorted.copy()
    else:
        out = (
            champs_sorted
            .groupby([well_col, arch_col], as_index=False, sort=False, group_keys=False)
            .head(int(K2))
            .copy()
        )

        # ---- 6) Optional backfill from the same family ----------------------
        if enable_backfill_from_family and K2 is not None:
            picked_hashes = set(out["job_hash"].astype(str)) if "job_hash" in out.columns else set()
            pool_arch_col = arch_col if arch_col in df_scored.columns else (
                next((c for c in ["arch","architecture","architecture_name","family"] if c in df_scored.columns), "_arch_family")
            )
            if pool_arch_col == "_arch_family" and "_arch_family" not in df_scored.columns:
                s = df_scored.get("architecture_name", df_scored.get("arch", "")).astype(str).str.lower()
                fam_guess = np.where(s.str.contains("arps", na=False), "arps",
                              np.where(s.str.contains("darts", na=False), "darts",
                              np.where(s.str.contains("seq2", na=False) | s.str.contains("pinn", na=False), "seq2", "generic")))
                df_scored["_arch_family"] = fam_guess
                pool_arch_col = "_arch_family"

            pool_sorted = df_scored.sort_values([well_col, pool_arch_col, score_col], ascending=[True, True, asc])
            needed = []
            for (w, a), gpool in pool_sorted.groupby([well_col, pool_arch_col]):
                have = out[(out[well_col] == w) & (out[arch_col] == a)]
                missing = int(K2) - len(have)
                if missing <= 0:
                    continue
                cand = gpool if "job_hash" not in gpool.columns else gpool[~gpool["job_hash"].astype(str).isin(picked_hashes)]
                take = cand.head(missing)
                if not take.empty:
                    take = take.reindex(columns=out.columns, fill_value=np.nan)
                    needed.append(take)
                    if "job_hash" in take.columns:
                        picked_hashes.update(take["job_hash"].astype(str))
            if needed:
                out = pd.concat([out] + needed, ignore_index=True)

        try:
            n_wells = out[well_col].nunique()
            n_archs = out[arch_col].nunique()
            info("[select] Enforced Top-%d per (well, arch): %d wells × %d archs → %d rows.",
                 int(K2), n_wells, n_archs, len(out))
        except Exception:
            pass

    # ---- Compact summary block ---------------------------------------------
    if COMPACT:
        lines = summarize_stage1(champions_df=out, posthoc={
            "top_strategies_per_well": K if K is not None else getattr(cfg, "top_strategies_per_well", None),
            "selection_strategy": getattr(cfg, "selection_strategy", None),
            "valcum_gate": getattr(cfg, "valcum_gate", {}),
        }, score_col=score_col)
        log_block("Stage 1 — Robust Champion Harvester (Summary)", lines, level=logging.INFO, width=WIDTH)

    return out




