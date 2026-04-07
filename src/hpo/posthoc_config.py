"""
hpo.posthoc_config

Post-hoc configuration layer for the HPO analysis pipeline.

This module defines `PostHocConfig` and `make_default_config`, which normalize user overrides,
resolve the single source of truth for the decision/ordering column (selection_col vs metric_to_optimize),
and apply lightweight sanity checks to keep the downstream post-hoc filtering/selection deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class PostHocConfig:
    """Configuration for the post-hoc distribution filter."""

    # Metrics and their optimization direction
    metrics: List[str] = field(default_factory=lambda: ["weighted_score", "val_smape_agg", "val_smape_cum"])
    lower_is_better: Dict[str, bool] = field(default_factory=lambda: {
        "weighted_score": True,
        "val_smape_agg": True,
        "val_smape_cum": True,
    })

    # ------------------------------------------------------------------
    # Explicit: column used to ORDER/SELECT (decision column)
    # - selection_col takes priority
    # - metric_to_optimize is kept as legacy alias
    # ------------------------------------------------------------------
    selection_col: Optional[str] = None
    metric_to_optimize: str = "weighted_score"

    selection_strategy: str = "median_of_the_best"

    # Filtering parameters
    primary_quantile: Dict[str, float] = field(default_factory=lambda: {
        "weighted_score": 0.6,
        "val_smape_agg": 0.6,
        "val_smape_cum": 0.6,
    })
    mad_guard: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "alpha": 2.0})
    apply_pareto: bool = True
    pareto_metrics: List[str] = field(default_factory=lambda: ["val_smape_agg", "val_smape_cum"])

    # ------------------------------------------------------------------
    # Pool policy (kept for compatibility; legacy selection forces survivors upstream)
    # ------------------------------------------------------------------
    pool_method: str = "survivors"  # "survivors" | "top_pct" | "val_band"
    pool_cfg: Dict[str, Any] = field(default_factory=dict)

    # Final curation parameters
    top_k_per_well: int = 1
    min_arch_diversity: int = 5
    min_samples_per_well: int = 20
    plot: bool = False

    # Column names and identifiers
    well_col: str = "well"
    arch_col: str = "architecture"
    id_cols: List[str] = field(default_factory=lambda: [
        "physics_strategy",
        "data_sample",
        "learning_rate",
        "lag_window",
        "batch_size",
        "epochs",
    ])

    strategy_col: str = "physics_strategy"
    top_strategies_per_well: int = 3
    per_strategy_k: int = 1
    arch_filter: Optional[Union[str, List[str]]] = None

    out_dir: Path = Path("reports/posthoc")
    viz_backend: str = "plotly"

    hpo_signature_cols: Optional[List[str]] = None

    valcum_gate: Dict[str, float] = field(default_factory=lambda: {"q_low": 0.30, "q_high": 0.85})
    relax_pool: bool = True


def make_default_config(**kwargs) -> PostHocConfig:
    """
    Creates a PostHocConfig instance and applies robust sanity checks.

    Key behaviors:
      - Filters unknown kwargs (prevents unexpected keyword crashes).
      - Resolves selection column with priority: selection_col > metric_to_optimize.
      - Ensures lower_is_better and primary_quantile are consistent with the resolved selection metric.
    """
    metric_weights = kwargs.pop("metric_weights", None)

    # ------------------------------------------------------------------
    # 1) Filter unknown kwargs (plug-and-play stability)
    # ------------------------------------------------------------------
    allowed = {f.name for f in fields(PostHocConfig)}
    unknown = sorted([k for k in kwargs.keys() if k not in allowed])
    if unknown:
        logging.warning("[posthoc_cfg] Dropping unknown PostHocConfig kwargs: %s", unknown)
        for k in unknown:
            kwargs.pop(k, None)

    cfg = PostHocConfig(**kwargs)

    # ------------------------------------------------------------------
    # 2) Resolve selection column (single source of truth)
    # ------------------------------------------------------------------
    requested = (getattr(cfg, "selection_col", None) or "").strip() or None
    legacy = (getattr(cfg, "metric_to_optimize", None) or "").strip() or "weighted_score"
    metric_to_optimize = requested or legacy

    # Keep both fields coherent
    cfg.metric_to_optimize = metric_to_optimize
    if requested is None:
        cfg.selection_col = None  # explicit: not set by user

    # ------------------------------------------------------------------
    # 3) Sanity checks (keep your current intent)
    # ------------------------------------------------------------------
    default_lib = {
        "weighted_score": True,
        "val_smape_agg": True,
        "val_smape_cum": True,
        "robust_score": True,  # allow robust_score as a score metric (lower is better)
    }
    cfg.lower_is_better = {**default_lib, **(cfg.lower_is_better or {})}

    # Ensure the primary metric has direction
    cfg.lower_is_better.setdefault(metric_to_optimize, True)

    # Auto-prune from metric_weights (unchanged behavior)
    if isinstance(metric_weights, dict):
        zero_keys = {k for k, v in metric_weights.items() if v is not None and float(v) == 0.0}
        if zero_keys:
            cfg.metrics = [m for m in (cfg.metrics or []) if m not in zero_keys]
            for k in list((cfg.primary_quantile or {}).keys()):
                if k in zero_keys:
                    cfg.primary_quantile.pop(k, None)

    # Ensure primary metric in cfg.metrics
    if metric_to_optimize not in (cfg.metrics or []):
        cfg.metrics.append(metric_to_optimize)

    # Ensure primary_quantile contains the primary metric
    if not (cfg.primary_quantile or {}):
        cfg.primary_quantile = {metric_to_optimize: 0.6}
    else:
        cfg.primary_quantile.setdefault(metric_to_optimize, 0.6)

    # Pareto metrics sanitization (unchanged)
    cfg.pareto_metrics = [m for m in (cfg.pareto_metrics or []) if m in (cfg.metrics or [])]
    if len(cfg.pareto_metrics) < 2:
        cfg.apply_pareto = False

    logging.info(
        "[posthoc_cfg] resolved | selection_col=%s metric_to_optimize=%s lower_is_better[%s]=%s",
        getattr(cfg, "selection_col", None),
        metric_to_optimize,
        metric_to_optimize,
        bool((cfg.lower_is_better or {}).get(metric_to_optimize, True)),
    )

    return cfg


@dataclass
class SelectionRunConfig:
    """
    All knobs for the HPO selection pipeline.

    Key semantics:
    - scoring_strategy: controls how scores are PRODUCED during aggregation.
      Examples: "weighted_score", "robust_score"
    - metric_to_optimize: controls which column is USED to order/filter/select.
      Typically equals the score column, but can be different.
    - selector_mode: explicit selection mode (prevents hybrid semantics).
    """
    # Required
    results_dir: Path
    reports_dir: Path
    metric_weights: Dict[str, float]
    lower_is_better: Dict[str, bool]

    # Score production during aggregation
    scoring_strategy: Literal["weighted_score", "robust_score"] = "weighted_score"

    # Selection ordering column (must exist after aggregation)
    metric_to_optimize: str = "weighted_score"

    # Post-hoc filter overrides (merged into make_default_config)
    posthoc_overrides: Dict[str, Any] = field(default_factory=dict)

    # Optional: restrict to a single architecture for strategy-aware selection
    arch_filter: Optional[str] = None

    # Optional plots (default: OFF)
    plot_architecture_performance: bool = False
    plot_hparam_importance_per_well: bool = False
    plot_champions_per_well: bool = False
    plot_summary_bars: bool = False

    # Validation profile generation
    validation_run_name: str = "final_validation_of_champions"
    validation_seed: int = 123

    # ------------------------------------------------------------------
    # Patch 3: explicit dispatcher knobs (no neighborhood implementation yet)
    # ------------------------------------------------------------------
    selector_mode: SelectorMode = "LEGACY_WEIGHTED"

    # Reserved for NEIGHBOR_* configs (Patch 5/6). Kept here to avoid breaking API later.
    neighborhood_overrides: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # PRI routing (VAL-only meta-selector). Safe defaults keep legacy behavior.
    # ------------------------------------------------------------------
    enable_pri_routing: bool = False
    pri_policy_path: Optional[str] = None

