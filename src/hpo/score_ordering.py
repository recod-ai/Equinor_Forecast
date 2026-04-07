# src/hpo/score_ordering.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd  # needed for analyze_across_campaigns typing

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ordering:
    """Defines how to sort a metric to put the best rows first."""
    metric: str
    ascending: bool  # True => lower is better
    higher_is_better: bool
    reason: str


# src/hpo/score_ordering.py

def resolve_ordering(
    metric_name: Optional[str],
    lower_is_better: Optional[Dict[str, bool]] = None,
    *,
    default_ascending: bool = True,
) -> Ordering:
    """
    Resolve ordering for a metric.

    Rules:
      1) If `lower_is_better` specifies the metric: use it (source of truth).
      2) Otherwise apply hard-coded invariants for known critical metrics.
      3) Else fallback to `default_ascending` (warn).

    NOTE:
      - This is intentionally defensive: it prevents accidental inversions when callers
        forget to pass `lower_is_better` (e.g., meta builders, analysis helpers).
    """
    m = (metric_name or "").strip()
    if not m:
        logger.warning("[ordering] Empty metric_name; using default_ascending=%s", default_ascending)
        return Ordering(
            metric="unknown",
            ascending=default_ascending,
            higher_is_better=not default_ascending,
            reason="empty_metric",
        )

    lib = lower_is_better or {}

    # 1) Config is the source of truth (when present)
    if m in lib:
        asc = bool(lib[m])
        return Ordering(metric=m, ascending=asc, higher_is_better=not asc, reason="cfg_lower_is_better")

    # 2) Hard-coded invariants (only when NOT specified by config)
    # Your pipeline expects these to be stable across the codebase.
    if m == "weighted_score":
        # Per your decision: weighted_score is LOWER-IS-BETTER
        return Ordering(metric=m, ascending=True, higher_is_better=False, reason="hard_rule_weighted_score")

    if m == "robust_score":
        # robust_score is LOWER-IS-BETTER
        return Ordering(metric=m, ascending=True, higher_is_better=False, reason="hard_rule_robust_score")

    # 3) Safe fallback
    logger.warning(
        "[ordering] Unknown metric '%s' not found in lower_is_better; defaulting ascending=%s.",
        m,
        default_ascending,
    )
    return Ordering(metric=m, ascending=default_ascending, higher_is_better=not default_ascending, reason="default_fallback")



def analyze_across_campaigns(
    master_df: pd.DataFrame,
    metric_to_optimize: str = "weighted_score",
    *,
    lower_is_better: Optional[Dict[str, bool]] = None,
):
    """
    Performs high-level analysis on an aggregated DataFrame of results from multiple campaigns.

    NOTE:
    - Ordering is resolved via a single source of truth (resolve_ordering).
    - Pass `lower_is_better` to avoid accidental inversions (especially for weighted_score).
    """
    if master_df is None or master_df.empty:
        logger.info("[analysis] Master DataFrame is empty. Nothing to analyze.")
        return

    if metric_to_optimize not in master_df.columns:
        logger.error("[analysis] Metric to optimize '%s' not found in DataFrame.", metric_to_optimize)
        return

    ordering = resolve_ordering(metric_to_optimize, lower_is_better=lower_is_better, default_ascending=True)
    logger.info(
        "[analysis] metric_to_optimize=%s | ascending=%s | higher_is_better=%s | reason=%s",
        metric_to_optimize,
        ordering.ascending,
        ordering.higher_is_better,
        ordering.reason,
    )

    # --- 1. Best Overall Models ---
    display_cols = ["campaign", "architecture", "dataset", "well", metric_to_optimize, "val_smape_cum", "val_smape_agg"]
    available_cols = [c for c in display_cols if c in master_df.columns]

    top20 = (
        master_df.nsmallest(20, metric_to_optimize)
        if ordering.ascending
        else master_df.nlargest(20, metric_to_optimize)
    )

    # Keep "display" optional (works in notebooks, harmless elsewhere)
    try:
        display(top20[available_cols])  # type: ignore[name-defined]
    except Exception:
        logger.info("[analysis] Top20 preview:\n%s", top20[available_cols].head(20).to_string(index=False))

    # --- 2. Architecture Performance ---
    grp_campaign = master_df.groupby("campaign")[metric_to_optimize]
    best_idx = grp_campaign.idxmin() if ordering.ascending else grp_campaign.idxmax()
    best_per_campaign = master_df.loc[best_idx]

    arch_perf = (
        best_per_campaign.groupby("architecture")[metric_to_optimize]
        .mean()
        .sort_values(ascending=ordering.ascending)
    )
    logger.info("[analysis] Mean best-per-campaign by architecture:\n%s", arch_perf.to_string())

    # --- 3. Per-Well Analysis ---
    grp_well = master_df.groupby("well")[metric_to_optimize]
    best_idx_well = grp_well.idxmin() if ordering.ascending else grp_well.idxmax()
    best_model_per_well = master_df.loc[best_idx_well]

    well_summary = (
        best_model_per_well[["well", "architecture", metric_to_optimize]]
        .sort_values(by="well")
        .set_index("well")
    )
    logger.info("[analysis] Best architecture per well:\n%s", well_summary.to_string())
