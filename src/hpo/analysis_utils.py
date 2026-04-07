# # src/hpo/analysis_utils.py

# # 1. Standard library imports
# import logging
# import os
# import re
# import warnings
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# # 2. Third-party imports
# import numpy as np
# import pandas as pd
# from box import Box
# from pandas.errors import EmptyDataError, ParserError
# from scipy.stats import median_abs_deviation
# from statsmodels.distributions.empirical_distribution import ECDF

# # 3. Optional imports for rich text and visualization
# try:
#     from rich.columns import Columns
#     from rich.console import Console
#     from rich.panel import Panel
#     from rich.style import Style
#     from rich.table import Table
#     RICH_AVAILABLE = True
# except ImportError:
#     RICH_AVAILABLE = False

# try:
#     import plotly.graph_objects as go
#     from plotly.subplots import make_subplots
#     VIZ_LIBS_INSTALLED = True
# except ImportError:
#     VIZ_LIBS_INSTALLED = False


# # 🔽 NOVO: use a fachada de logs unificada
# from common.log_utils import (
#     info, warn, ok, err,
#     log_block, summarize_stage1,
#     is_compact_logging, effective_log_width,
# )



# def add_weighted_score(
#     df: pd.DataFrame,
#     metric_weights: dict,
#     lower_is_better: dict,
#     group_keys: list[str] | None = None
# ) -> pd.DataFrame:
#     """
#     Normaliza métricas e calcula weighted_score.
#     Agrupa por chaves existentes; tolera ausência de 'dataset' (ex.: ARPS).
#     - Se group_keys=None -> tenta ["dataset","well"]
#     - Usa só as chaves que existirem no DF
#     - Se nada sobrar, tenta ["well"]; se ainda assim não der, aplica globalmente
#     """
#     if df is None or df.empty:
#         return df

#     # defina alvo padrão e interseção com colunas presentes
#     default_keys = ["dataset", "well"]
#     wanted = group_keys if group_keys is not None else default_keys
#     gkeys = [k for k in wanted if k in df.columns]

#     # fallback: tenta só 'well'
#     if not gkeys and "well" in df.columns:
#         gkeys = ["well"]

#     df_scored = df.copy()

#     def _score_block(block: pd.DataFrame) -> pd.DataFrame:
#         out = block.copy()
#         # normaliza cada métrica em ranks 0..1 (menor=melhor -> rank direto; maior=melhor -> 1-rank)
#         for metric, is_lower_better in lower_is_better.items():
#             if metric not in out.columns:
#                 continue
#             ranks = out[metric].rank(pct=True, ascending=True)  # rank baixo = valor baixo
#             out[f"{metric}_norm"] = ranks if is_lower_better else (1 - ranks)
#         # score ponderado
#         out["weighted_score"] = 0.0
#         for metric, w in metric_weights.items():
#             col = f"{metric}_norm"
#             if col in out.columns:
#                 out["weighted_score"] += float(w) * out[col]
#         return out

#     # aplica com ou sem groupby, conforme chaves disponíveis
#     if gkeys:
#         return df_scored.groupby(gkeys, group_keys=False).apply(_score_block).reset_index(drop=True)
#     else:
#         # sem chaves de agrupamento: aplica no DF inteiro
#         return _score_block(df_scored)

# def print_campaign_summary(campaign_queue: List[Dict], max_concurrent: int = 1):
#     """
#     Prints a beautifully formatted summary of all campaigns about to be run,
#     arranging them in a grid layout.

#     Args:
#         campaign_queue (List[Dict]): The list of campaign jobs to be executed.
#         max_concurrent (int): The maximum number of parallel jobs.
#     """
#     if not RICH_AVAILABLE:
#         # Simple fallback if rich is not installed
#         print("--- Campaign Batch Summary ---")
#         print(f"Queued Campaigns: {len(campaign_queue)}")
#         print(f"Max Parallel Runs: {max_concurrent}")
#         for job in campaign_queue:
#             print(f"- {job['name']}")
#         print("--------------------------")
#         return

#     # --- Rich-formatted output ---
#     console = Console()
    
#     # Print the main title panel (unchanged)
#     console.print(
#         Panel(
#             f"[bold]Queued Campaigns:[/] {len(campaign_queue)}\n[bold]Max Parallel Runs:[/] {max_concurrent}",
#             title="🚀 HPO Batch Execution Plan 🚀",
#             style="bold blue",
#             expand=False
#         )
#     )

#     # --- 1. Iterate and Collect Panels ---
#     campaign_panels = []
#     for job in campaign_queue:
#         try:
#             from config_loader import load_campaign_config
#             config = load_campaign_config(job['config_path'])

#             # Create the summary table for one campaign
#             table = Table.grid(expand=True, padding=(0, 1))
#             table.add_column(style="magenta", justify="right")
#             table.add_column(style="green")
            
#             arch = config.job_defaults.architecture_name
#             dataset = config.run_scope.dataset_name
#             wells = ", ".join(config.run_scope.wells)
#             trials = sum(config.hpo_params.trials_per_cycle_schedule)

#             table.add_row("[b]Arch:[/b]", arch)
#             table.add_row("[b]Data:[/b]", f"{dataset} ({wells})")
#             table.add_row("[b]Trials:[/b]", str(trials))

#             # Create a Panel for this campaign
#             panel = Panel(
#                 table,
#                 title=f"[cyan bold]{job['name']}[/]",
#                 title_align="left",
#                 border_style="dim cyan",
#                 # Add some padding to space out the panels
#                 padding=(1, 2)
#             )
#             # Add the created Panel to our list
#             campaign_panels.append(panel)
#         except Exception as e:
#             campaign_panels.append(Panel(f"[bold red]Error:\n{e}", title=job['name']))
    
#     # --- 2. Group and Render ---
#     if campaign_panels:
#         # Create a Columns layout with our list of panels.
#         # `expand=True` makes the columns fill the available width.
#         # `equal=True` tries to make columns the same width.
#         console.print(Columns(campaign_panels, equal=True, expand=True))



# ==============================================================================
# CONFIGURATION OBJECT
# ==============================================================================

# @dataclass
# class PostHocConfig:
#     """Configuration for the post-hoc distribution filter."""

#     # Metrics and their optimization direction
#     metrics: List[str] = field(default_factory=lambda: ["weighted_score", "val_smape_agg", "val_smape_cum"])
#     lower_is_better: Dict[str, bool] = field(default_factory=lambda: {
#         "weighted_score": True,
#         "val_smape_agg": True,
#         "val_smape_cum": True,
#     })

#     # ------------------------------------------------------------------
#     # Explicit: column used to ORDER/SELECT (decision column)
#     # - selection_col takes priority
#     # - metric_to_optimize is kept as legacy alias
#     # ------------------------------------------------------------------
#     selection_col: Optional[str] = None
#     metric_to_optimize: str = "weighted_score"

#     selection_strategy: str = "median_of_the_best"

#     # Filtering parameters
#     primary_quantile: Dict[str, float] = field(default_factory=lambda: {
#         "weighted_score": 0.6,
#         "val_smape_agg": 0.6,
#         "val_smape_cum": 0.6,
#     })
#     mad_guard: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "alpha": 2.0})
#     apply_pareto: bool = True
#     pareto_metrics: List[str] = field(default_factory=lambda: ["val_smape_agg", "val_smape_cum"])

#     # ------------------------------------------------------------------
#     # Pool policy (kept for compatibility; legacy selection forces survivors upstream)
#     # ------------------------------------------------------------------
#     pool_method: str = "survivors"  # "survivors" | "top_pct" | "val_band"
#     pool_cfg: Dict[str, Any] = field(default_factory=dict)

#     # Final curation parameters
#     top_k_per_well: int = 1
#     min_arch_diversity: int = 5
#     min_samples_per_well: int = 20
#     plot: bool = False

#     # Column names and identifiers
#     well_col: str = "well"
#     arch_col: str = "architecture"
#     id_cols: List[str] = field(default_factory=lambda: [
#         "physics_strategy",
#         "data_sample",
#         "learning_rate",
#         "lag_window",
#         "batch_size",
#         "epochs",
#     ])

#     strategy_col: str = "physics_strategy"
#     top_strategies_per_well: int = 3
#     per_strategy_k: int = 1
#     arch_filter: Optional[Union[str, List[str]]] = None

#     out_dir: Path = Path("reports/posthoc")
#     viz_backend: str = "plotly"

#     hpo_signature_cols: Optional[List[str]] = None

#     valcum_gate: Dict[str, float] = field(default_factory=lambda: {"q_low": 0.30, "q_high": 0.85})
#     relax_pool: bool = True



# def apply_valcum_gate(df: pd.DataFrame, cfg: PostHocConfig) -> pd.DataFrame:
#     """
#     Per-well quantile band gate on val_smape_cum with adaptive q_low.

#     Writes:
#       - valcum_pass (bool)
#       - _valcum_q_low_used, _valcum_q_high_used, _valcum_mode_used (audit breadcrumbs)
#     """
#     import logging
#     import numpy as np
#     import pandas as pd

#     log = logging.getLogger(__name__)

#     if df is None or df.empty or "val_smape_cum" not in df.columns:
#         out = df.copy() if df is not None else pd.DataFrame()
#         out["valcum_pass"] = True
#         return out

#     gate = dict(getattr(cfg, "valcum_gate", None) or {})

#     mode        = str(gate.get("mode", "smooth")).lower()
#     base_q_low  = float(gate.get("q_low", 0.08))
#     q_high      = float(gate.get("q_high", 0.80))
#     min_q_low   = float(gate.get("min_q_low", 0.02))
#     cap_q_low   = float(gate.get("cap_q_low", 0.10))
#     min_gap     = float(gate.get("min_gap", 0.05))
#     smooth_ref  = float(gate.get("smooth_rcv_ref", 0.80))
#     tail_bump   = float(gate.get("tail_bump", 0.01))
#     tail_thresh = float(gate.get("tail_thresh", 1.60))
#     well_col    = str(gate.get("well_col", getattr(cfg, "well_col", "well")))

#     # Guards
#     cap_q_low = max(min_q_low, min(cap_q_low, q_high - min_gap))
#     base_q_low = float(np.clip(base_q_low, min_q_low, cap_q_low))

#     out = df.copy()

#     def _stats(s: pd.Series) -> dict:
#         s = pd.to_numeric(s, errors="coerce").dropna()
#         if s.empty:
#             return dict(n=0, q05=np.nan, q25=np.nan, q50=np.nan, q75=np.nan, q95=np.nan,
#                         iqr=np.nan, rcv=np.nan, tail=np.nan)
#         q05, q25, q50, q75, q95 = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
#         iqr = float(q75 - q25)
#         rcv = float(iqr / (abs(q50) + 1e-9))                 # robust CV
#         tail = float((q95 - q50) / ((q50 - q05) + 1e-9))     # right-tail strength
#         return dict(n=int(s.size), q05=q05, q25=q25, q50=q50, q75=q75, q95=q95, iqr=iqr, rcv=rcv, tail=tail)

#     def _choose_q_low_rule(n: int, rcv: float, tail: float, base: float) -> float:
#         ql = base
#         if n >= 150 and rcv <= 0.45 and tail <= 1.40:
#             ql = min(ql, 0.06)
#         if n >= 300 and rcv <= 0.30 and tail <= 1.20:
#             ql = min(ql, 0.05)
#         if (rcv >= 1.00) or (tail >= 2.50):
#             ql = max(ql, 0.09)
#         elif (rcv >= 0.60) or (tail >= 1.80):
#             ql = max(ql, 0.07)
#         if n < 150:
#             ql = max(ql, 0.08)
#         return float(np.clip(ql, min_q_low, cap_q_low))

#     def _choose_q_low_smooth(n: int, rcv: float, tail: float, base: float) -> float:
#         if n <= 0 or not np.isfinite(rcv):
#             ql = base
#         else:
#             alpha = float(np.clip(rcv / max(smooth_ref, 1e-9), 0.0, 1.0))
#             ql = 0.05 + 0.05 * alpha
#             if np.isfinite(tail) and tail > tail_thresh:
#                 ql += tail_bump
#             if n < 150:
#                 ql = max(ql, 0.07)
#         return float(np.clip(ql, min_q_low, cap_q_low))

#     qlow_by_well: dict = {}

#     for well_name, s in out.groupby(well_col, sort=False)["val_smape_cum"]:
#         st = _stats(s)

#         if mode == "strict_override":
#             ql, chosen = base_q_low, "strict"
#         elif mode == "rule":
#             ql, chosen = _choose_q_low_rule(st["n"], st["rcv"], st["tail"], base_q_low), "rule"
#         else:
#             ql, chosen = _choose_q_low_smooth(st["n"], st["rcv"], st["tail"], base_q_low), "smooth"

#         # Never touch q_high
#         ql = min(ql, q_high - min_gap)
#         qlow_by_well[well_name] = ql

#         capped = (ql >= cap_q_low - 1e-12) or (ql <= min_q_low + 1e-12)
#         log.info(
#             "valcum_gate: well=%s mode=%s q_low=%.3f q_high=%.2f n=%d rCV=%s tail=%s%s",
#             well_name, chosen, ql, q_high, int(st["n"]),
#             (f"{st['rcv']:.3f}" if np.isfinite(st["rcv"]) else "nan"),
#             (f"{st['tail']:.3f}" if np.isfinite(st["tail"]) else "nan"),
#             (" [capped]" if capped else ""),
#         )

#     # Apply per-well mask (this fixes the s.name bug)
#     out["valcum_pass"] = False
#     for well_name, idx in out.groupby(well_col, sort=False).groups.items():
#         s = pd.to_numeric(out.loc[idx, "val_smape_cum"], errors="coerce")
#         ql = float(qlow_by_well.get(well_name, base_q_low))
#         lo = s.quantile(ql)
#         hi = s.quantile(q_high)
#         out.loc[idx, "valcum_pass"] = s.between(lo, hi).fillna(False).astype(bool).values

#     out["valcum_pass"] = out["valcum_pass"].astype(bool)
#     out["_valcum_q_low_used"] = out[well_col].map(qlow_by_well)
#     out["_valcum_q_high_used"] = float(q_high)
#     out["_valcum_mode_used"] = str(mode)

#     return out



# def make_default_config(**kwargs) -> "PostHocConfig":
#     """
#     Creates a PostHocConfig instance and applies robust sanity checks.

#     Key behaviors:
#       - Filters unknown kwargs (prevents unexpected keyword crashes).
#       - Resolves selection column with priority: selection_col > metric_to_optimize.
#       - Ensures lower_is_better and primary_quantile are consistent with the resolved selection metric.
#     """
#     import logging
#     from dataclasses import fields

#     metric_weights = kwargs.pop("metric_weights", None)

#     # ------------------------------------------------------------------
#     # 1) Filter unknown kwargs (plug-and-play stability)
#     # ------------------------------------------------------------------
#     allowed = {f.name for f in fields(PostHocConfig)}
#     unknown = sorted([k for k in kwargs.keys() if k not in allowed])
#     if unknown:
#         logging.warning(
#             "[posthoc_cfg] Dropping unknown PostHocConfig kwargs: %s",
#             unknown,
#         )
#         for k in unknown:
#             kwargs.pop(k, None)

#     cfg = PostHocConfig(**kwargs)

#     # ------------------------------------------------------------------
#     # 2) Resolve selection column (single source of truth)
#     # ------------------------------------------------------------------
#     requested = (getattr(cfg, "selection_col", None) or "").strip() or None
#     legacy = (getattr(cfg, "metric_to_optimize", None) or "").strip() or "weighted_score"
#     metric_to_optimize = requested or legacy

#     # Keep both fields coherent
#     cfg.metric_to_optimize = metric_to_optimize
#     if requested is None:
#         cfg.selection_col = None  # explicit: not set by user

#     # ------------------------------------------------------------------
#     # 3) Sanity checks (keep your current intent)
#     # ------------------------------------------------------------------
#     default_lib = {
#         "weighted_score": True,
#         "val_smape_agg": True,
#         "val_smape_cum": True,
#         "robust_score": True,  # allow robust_score as a score metric (lower is better)
#     }
#     cfg.lower_is_better = {**default_lib, **(cfg.lower_is_better or {})}

#     # Ensure the primary metric has direction
#     cfg.lower_is_better.setdefault(metric_to_optimize, True)

#     # Auto-prune from metric_weights (unchanged behavior)
#     if isinstance(metric_weights, dict):
#         zero_keys = {k for k, v in metric_weights.items() if v is not None and float(v) == 0.0}
#         if zero_keys:
#             cfg.metrics = [m for m in (cfg.metrics or []) if m not in zero_keys]
#             for k in list((cfg.primary_quantile or {}).keys()):
#                 if k in zero_keys:
#                     cfg.primary_quantile.pop(k, None)

#     # Ensure primary metric in cfg.metrics
#     if metric_to_optimize not in (cfg.metrics or []):
#         cfg.metrics.append(metric_to_optimize)

#     # Ensure primary_quantile contains the primary metric
#     if not (cfg.primary_quantile or {}):
#         cfg.primary_quantile = {metric_to_optimize: 0.6}
#     else:
#         cfg.primary_quantile.setdefault(metric_to_optimize, 0.6)

#     # Pareto metrics sanitization (unchanged)
#     cfg.pareto_metrics = [m for m in (cfg.pareto_metrics or []) if m in (cfg.metrics or [])]
#     if len(cfg.pareto_metrics) < 2:
#         cfg.apply_pareto = False

#     logging.info(
#         "[posthoc_cfg] resolved | selection_col=%s metric_to_optimize=%s lower_is_better[%s]=%s",
#         getattr(cfg, "selection_col", None),
#         metric_to_optimize,
#         metric_to_optimize,
#         bool((cfg.lower_is_better or {}).get(metric_to_optimize, True)),
#     )

#     return cfg




# ==============================================================================
# IMPLEMENTATION FUNCTIONS (1-15)
# ==============================================================================

# def sanitize_and_validate(master_df: pd.DataFrame, cfg: PostHocConfig) -> pd.DataFrame:
#     """
#     Ensures required columns exist, drops NaNs in metrics, and coerces dtypes.

#     Robustness:
#     - If cfg.arch_col is missing, it will try common fallbacks ("architecture_name", "arch", "architecture")
#       and use the first available for validation purposes.
#     """
#     df = master_df.copy()

#     well_col = getattr(cfg, "well_col", "well")
#     arch_col = getattr(cfg, "arch_col", "architecture")

#     if arch_col not in df.columns:
#         for cand in ("architecture", "architecture_name", "arch"):
#             if cand in df.columns:
#                 arch_col = cand
#                 break
#         else:
#             arch_col = None

#     required_cols = [well_col] + ( [arch_col] if arch_col else [] ) + list(getattr(cfg, "metrics", []))
#     missing_cols = [col for col in required_cols if col not in df.columns]
#     if missing_cols:
#         raise ValueError(f"Missing required columns in master_df: {missing_cols}")

#     # Drop rows with NaNs in any key metrics
#     metrics = list(getattr(cfg, "metrics", []))
#     df.dropna(subset=metrics, inplace=True)

#     # Coerce metrics to numeric and drop failures
#     for metric in metrics:
#         df[metric] = pd.to_numeric(df[metric], errors="coerce")
#     df.dropna(subset=metrics, inplace=True)

#     return df




# def deduplicate_by_signature(df: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
#     import logging
#     import hashlib

#     from hpo.score_ordering import resolve_ordering

#     if df is None or df.empty:
#         return df

#     # Detect family to choose sensible defaults
#     fam = "arps" if (
#         ("variant" in df.columns) or
#         ("architecture_name" in df.columns and df["architecture_name"].astype(str).str.contains("Arps", na=False).any())
#     ) else "generic"

#     if fam == "arps":
#         DEFAULT_SIGNATURE_COLS = [
#             "variant", "solver", "loss", "burn_in_fraction", "piecewise",
#             "weighting", "quantile_tau", "b_grid_kind", "b_grid_size",
#             "b_min", "b_max", "lag_window", "horizon",
#         ]
#     else:
#         DEFAULT_SIGNATURE_COLS = [
#             "physics_strategy", "data_sample", "learning_rate", "lag_window",
#             "batch_size", "epochs", "architecture_profile",
#         ]

#     # ---- Robust fetch of signature cols ----
#     sig_attr = getattr(cfg, "hpo_signature_cols", DEFAULT_SIGNATURE_COLS)
#     if sig_attr is None:
#         signature_cols = DEFAULT_SIGNATURE_COLS
#     elif isinstance(sig_attr, (list, tuple)):
#         signature_cols = list(sig_attr)
#     elif isinstance(sig_attr, str):
#         signature_cols = [sig_attr]
#     else:
#         logging.warning("[dedup] Unexpected type for cfg.hpo_signature_cols=%r; using defaults.", type(sig_attr))
#         signature_cols = DEFAULT_SIGNATURE_COLS

#     present = [c for c in signature_cols if c in df.columns]
#     missing = [c for c in signature_cols if c not in df.columns]
#     if missing:
#         logging.info("[dedup] Ignoring missing signature columns: %s", missing)

#     if not present:
#         logging.info("[dedup] No signature columns present; skipping deduplication.")
#         return df

#     out = df.copy()

#     # ---- resolve selection/sort column: prefer cfg.selection_col, then cfg.metric_to_optimize, then fall back ----
#     selection_col = getattr(cfg, "selection_col", None) or getattr(cfg, "metric_to_optimize", None)

#     if not selection_col or selection_col not in out.columns:
#         for candidate in ("weighted_score", "robust_score", "val_smape_agg", "val_smape_cum"):
#             if candidate in out.columns:
#                 selection_col = candidate
#                 break

#     if not selection_col:
#         logging.warning("[dedup] No suitable selection column found; returning frame unchanged.")
#         return out

#     # ---- unify direction via single source of truth ----
#     lib = getattr(cfg, "lower_is_better", {}) or {}
#     ordering = resolve_ordering(selection_col, lower_is_better=lib, default_ascending=True)
#     asc = ordering.ascending

#     logging.info(
#         "[dedup] Using selection_col=%s | ascending=%s | higher_is_better=%s | reason=%s | signature_cols=%s",
#         selection_col,
#         asc,
#         ordering.higher_is_better,
#         ordering.reason,
#         present,
#     )

#     # ---- Build stable signature hash (stringified values, NaN-safe) ----
#     out["signature_cols_used"] = ",".join(present)

#     def _row_sig_str(r: pd.Series) -> str:
#         parts = []
#         for c in present:
#             v = r.get(c)
#             if pd.isna(v):
#                 parts.append(f"{c}=<NA>")
#             else:
#                 parts.append(f"{c}={str(v)}")
#         return "|".join(parts)

#     sig_str = out.apply(_row_sig_str, axis=1)
#     out["signature_hash"] = sig_str.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())

#     # ---- Keep best per signature_hash according to ordering ----
#     # We only need to sort by selection_col (direction-aware), but stable sort helps reproducibility.
#     out = out.sort_values(by=[selection_col], ascending=[asc], kind="mergesort")
#     out = out.drop_duplicates(subset=["signature_hash"], keep="first")

#     return out



# def compose_predicates(
#     qdf: pd.DataFrame,
#     mdf: pd.DataFrame,
#     cfg: "PostHocConfig",
# ) -> Dict[Tuple[str, str], Callable[[float], bool]]:
#     """
#     Build per-(well, metric) predicates from the UNION:
#       - quantile cutoff + MAD bounds when both exist
#       - quantile only when MAD missing
#       - MAD only when quantile missing

#     Anti-leak: ignores test_* metrics even if present in cfg.metrics.
#     """
#     import numpy as np
#     import pandas as pd

#     predicates: Dict[Tuple[str, str], Callable[[float], bool]] = {}

#     well_col = getattr(cfg, "well_col", "well")
#     lib_isb = getattr(cfg, "lower_is_better", {}) or {}

#     # effective metrics (no test_*)
#     cfg_metrics = [m for m in (getattr(cfg, "metrics", []) or []) if not str(m).startswith("test_")]

#     # Filter qdf/mdf to metrics of interest
#     if qdf is not None and not qdf.empty:
#         qdf = qdf[qdf["metric"].isin(cfg_metrics)]
#     if mdf is not None and not mdf.empty:
#         mdf = mdf[mdf["metric"].isin(cfg_metrics)]

#     q_pivot = (
#         qdf.pivot(index=well_col, columns="metric", values="cutoff")
#         if qdf is not None and not qdf.empty else pd.DataFrame()
#     )
#     m_lower = (
#         mdf.pivot(index=well_col, columns="metric", values="lower_bound")
#         if mdf is not None and not mdf.empty else pd.DataFrame()
#     )
#     m_upper = (
#         mdf.pivot(index=well_col, columns="metric", values="upper_bound")
#         if mdf is not None and not mdf.empty else pd.DataFrame()
#     )

#     wells = list(set(q_pivot.index) | set(m_lower.index) | set(m_upper.index))
#     metrics = list((set(q_pivot.columns) | set(m_lower.columns) | set(m_upper.columns)) & set(cfg_metrics))

#     for well in wells:
#         for metric in metrics:
#             lower_is_better = bool(lib_isb.get(metric, True))

#             # Neutral quantile cutoff if missing:
#             # - lower is better => +inf (no restriction)
#             q = (np.inf if lower_is_better else -np.inf)
#             if (well in q_pivot.index) and (metric in q_pivot.columns):
#                 q = q_pivot.loc[well, metric]

#             # Neutral MAD bounds if missing
#             m_lo, m_hi = -np.inf, np.inf
#             if (well in m_lower.index) and (metric in m_lower.columns):
#                 m_lo = m_lower.loc[well, metric]
#             if (well in m_upper.index) and (metric in m_upper.columns):
#                 m_hi = m_upper.loc[well, metric]

#             if lower_is_better:
#                 hi = min(q, m_hi) if np.isfinite(q) else m_hi
#                 predicates[(well, metric)] = (lambda v, hi=hi: True if not np.isfinite(hi) else (v <= hi))
#             else:
#                 lo = max(q, m_lo) if np.isfinite(q) else m_lo
#                 predicates[(well, metric)] = (lambda v, lo=lo: True if not np.isfinite(lo) else (v >= lo))

#     return predicates




# def quantile_thresholds(df: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
#     """
#     Computes primary quantile cutoffs for each well and metric.

#     - Skips missing metrics safely.
#     - Anti-leak: skips test_* metrics.
#     - Accepts empty/None cfg.primary_quantile (returns empty table).
#     """
#     import logging
#     import pandas as pd

#     well_col = getattr(cfg, "well_col", "well")
#     out_cols = [well_col, "metric", "cutoff", "quantile"]

#     if df is None or df.empty:
#         return pd.DataFrame(columns=out_cols)

#     pq = getattr(cfg, "primary_quantile", None) or {}
#     if not isinstance(pq, dict) or not pq:
#         return pd.DataFrame(columns=out_cols)

#     results = []
#     for metric, q in pq.items():
#         metric = str(metric)
#         if metric.startswith("test_"):
#             logging.info("[quantile] ignoring test metric: %s", metric)
#             continue
#         if metric not in df.columns:
#             logging.warning("[quantile] metric '%s' not in df; skipping", metric)
#             continue

#         q = float(q)

#         s = pd.to_numeric(df[metric], errors="coerce")
#         cutoffs = (
#             pd.concat([df[[well_col]], s.rename(metric)], axis=1)
#               .groupby(well_col, dropna=False)[metric]
#               .quantile(q)
#               .reset_index()
#               .rename(columns={metric: "cutoff"})
#         )
#         cutoffs["metric"] = metric
#         cutoffs["quantile"] = q
#         results.append(cutoffs)

#     return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=out_cols)



# def log_gate_diagnostics(df: pd.DataFrame,
#                          qdf: pd.DataFrame | None,
#                          mdf: pd.DataFrame | None,
#                          cfg: "PostHocConfig",
#                          max_examples: int = 3) -> None:
#     """
#     Loga, por (well, metric):
#       - quantile_cutoff
#       - mad_lower/mad_upper
#       - limites efetivos aplicados (lo/hi) segundo lower_is_better
#       - contagens: pass_quantile, pass_mad, pass_effective, passes_gates (se existir), valcum_pass (se existir)
#       - exemplos de conflitos (até max_examples de cada tipo)

#     Uso: chame logo após calcular qdf/mdf e (opcionalmente) após marcar passes_gates/valcum_pass.
#     """
#     if df is None or df.empty:
#         logging.info("[gates] DF vazio; nada a logar.")
#         return

#     well_col = getattr(cfg, "well_col", "well")
#     mets = list(getattr(cfg, "metrics", []))
#     lib = dict(getattr(cfg, "lower_is_better", {}) or {})
#     # pivôs seguros
#     if qdf is not None and not qdf.empty:
#         qdf = qdf[qdf["metric"].isin(mets)]
#         q_piv = qdf.pivot(index=well_col, columns="metric", values="cutoff")
#     else:
#         q_piv = pd.DataFrame()

#     if mdf is not None and not mdf.empty:
#         mdf = mdf[mdf["metric"].isin(mets)]
#         lo_piv = mdf.pivot(index=well_col, columns="metric", values="lower_bound")
#         hi_piv = mdf.pivot(index=well_col, columns="metric", values="upper_bound")
#     else:
#         lo_piv = pd.DataFrame()
#         hi_piv = pd.DataFrame()

#     wells = sorted(set(df[well_col].unique()))
#     for w in wells:
#         sub = df[df[well_col] == w]
#         if sub.empty:
#             continue
#         for m in mets:
#             if m not in sub.columns:
#                 continue
#             lower_is_better = bool(lib.get(m, True))

#             # pega quantil e MAD (ou padrões neutros)
#             q_cut = (
#                 q_piv.loc[w, m] if (m in q_piv.columns and w in q_piv.index)
#                 else (np.inf if not lower_is_better else -np.inf)
#             )
#             m_lo = (
#                 lo_piv.loc[w, m] if (m in lo_piv.columns and w in lo_piv.index)
#                 else -np.inf
#             )
#             m_hi = (
#                 hi_piv.loc[w, m] if (m in hi_piv.columns and w in hi_piv.index)
#                 else np.inf
#             )

#             # máscaras por regra
#             x = pd.to_numeric(sub[m], errors="coerce")
#             if lower_is_better:
#                 pass_q = x <= q_cut if np.isfinite(q_cut) else pd.Series(True, index=sub.index)
#                 pass_m = x <= m_hi if np.isfinite(m_hi) else pd.Series(True, index=sub.index)
#                 eff_lo, eff_hi = -np.inf, (min(q_cut, m_hi) if np.isfinite(q_cut) else m_hi)
#                 pass_eff = x <= eff_hi if np.isfinite(eff_hi) else pd.Series(True, index=sub.index)
#             else:
#                 pass_q = x >= q_cut if np.isfinite(q_cut) else pd.Series(True, index=sub.index)
#                 pass_m = x >= m_lo if np.isfinite(m_lo) else pd.Series(True, index=sub.index)
#                 eff_lo, eff_hi = (max(q_cut, m_lo) if np.isfinite(q_cut) else m_lo), np.inf
#                 pass_eff = x >= eff_lo if np.isfinite(eff_lo) else pd.Series(True, index=sub.index)

#             # contagens
#             n = len(sub)
#             c_q   = int(pass_q.sum())
#             c_m   = int(pass_m.sum())
#             c_eff = int(pass_eff.sum())
#             c_pg  = int(sub.get("passes_gates", pd.Series(True, index=sub.index)).sum()) if "passes_gates" in sub.columns else None
#             c_vc  = int(sub.get("valcum_pass",  pd.Series(True, index=sub.index)).sum())   if "valcum_pass"  in sub.columns else None

#             # conflito: passou quantil mas caiu no MAD
#             ex_q_not_m = sub[pass_q & (~pass_m)]
#             # conflito: passou MAD mas caiu no quantil
#             ex_m_not_q = sub[pass_m & (~pass_q)]

#             # origem das restrições
#             has_q  = np.isfinite(q_cut)
#             has_lo = np.isfinite(m_lo)
#             has_hi = np.isfinite(m_hi)
#             if has_q and (has_lo or has_hi):
#                 src = "both"
#             elif has_q:
#                 src = "quantile"
#             elif has_lo or has_hi:
#                 src = "mad"
#             else:
#                 src = "none"

#             logging.info(
#                 "[gates] well=%s metric=%s lib=%s src=%s "
#                 "q=%s mad=(%s,%s) eff=(%s,%s) "
#                 "n=%d pass_q=%d pass_m=%d pass_eff=%d%s%s",
#                 str(w), m, lower_is_better, src,
#                 (f"{q_cut:.6g}" if np.isfinite(q_cut) else "±inf"),
#                 (f"{m_lo:.6g}"  if np.isfinite(m_lo) else "-inf"),
#                 (f"{m_hi:.6g}"  if np.isfinite(m_hi) else "+inf"),
#                 ("-inf" if not np.isfinite(eff_lo) else f"{eff_lo:.6g}"),
#                 ("+inf" if not np.isfinite(eff_hi) else f"{eff_hi:.6g}"),
#                 n, c_q, c_m, c_eff,
#                 (f" passes_gates={c_pg}" if c_pg is not None else ""),
#                 (f" valcum_pass={c_vc}"  if c_vc is not None else "")
#             )

#             if len(ex_q_not_m) > 0:
#                 logging.info("[gates]   examples pass_quantile_but_fail_mad (top %d):", max_examples)
#                 cols_show = [well_col, m]
#                 logging.info("\n%s", ex_q_not_m[cols_show].head(max_examples).to_string(index=False))
#             if len(ex_m_not_q) > 0:
#                 logging.info("[gates]   examples pass_mad_but_fail_quantile (top %d):", max_examples)
#                 cols_show = [well_col, m]
#                 logging.info("\n%s", ex_m_not_q[cols_show].head(max_examples).to_string(index=False))


# def mad_guards(df: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
#     """
#     Computes robust upper/lower bounds using (optionally) one-sided and/or log MAD.

#     - Anti-leak: skips test_* metrics.
#     - Skips missing metric columns.
#     - Uses cfg.mad_guard options: enabled/alpha/metrics/log/side.
#     """
#     import numpy as np
#     import pandas as pd
#     from scipy.stats import median_abs_deviation

#     well_col = getattr(cfg, "well_col", "well")
#     guard = getattr(cfg, "mad_guard", None) or {}
#     if not guard.get("enabled", False):
#         return pd.DataFrame(columns=[well_col, "metric", "lower_bound", "upper_bound"])

#     alpha = float(guard.get("alpha", 1.0))
#     metrics_for_mad = guard.get("metrics", getattr(cfg, "metrics", [])) or []
#     metrics_for_mad = [m for m in metrics_for_mad if not str(m).startswith("test_")]

#     use_log = bool(guard.get("log", True))
#     side = str(guard.get("side", "right")).lower()  # "right" | "left" | "both"

#     lib_isb = getattr(cfg, "lower_is_better", {}) or {}

#     def _safe_mad(x: pd.Series) -> float:
#         x = pd.to_numeric(x, errors="coerce").dropna()
#         if x.empty:
#             return 0.0
#         m = float(median_abs_deviation(x, scale="normal"))
#         return m if (np.isfinite(m) and m > 0) else 1e-12

#     def _mad_bounds(s: pd.Series, lower_is_better: bool) -> tuple[float, float]:
#         x = pd.to_numeric(s, errors="coerce").dropna()
#         if x.empty:
#             return (-np.inf, np.inf)

#         med = float(x.median())

#         if side == "right":
#             x_use = x[x >= med]
#         elif side == "left":
#             x_use = x[x <= med]
#         else:
#             x_use = x
#         if x_use.empty:
#             x_use = x

#         if use_log and (x_use > -0.999999).all():
#             x_tr = np.log1p(x_use.to_numpy())
#             med_tr = float(np.median(x_tr))
#             mad_tr = _safe_mad(pd.Series(x_tr))

#             if lower_is_better:
#                 upper = np.expm1(med_tr + alpha * mad_tr)
#                 return (-np.inf, float(upper))
#             lower = np.expm1(med_tr - alpha * mad_tr)
#             return (float(lower), np.inf)

#         mad = _safe_mad(x_use)
#         if lower_is_better:
#             return (-np.inf, float(med + alpha * mad))
#         return (float(med - alpha * mad), np.inf)

#     records = []
#     for well, g in df.groupby(well_col, sort=False, dropna=False):
#         for metric in metrics_for_mad:
#             metric = str(metric)
#             if metric not in g.columns:
#                 continue
#             lower_is_better = bool(lib_isb.get(metric, True))
#             lo, hi = _mad_bounds(g[metric], lower_is_better)
#             records.append({well_col: well, "metric": metric, "lower_bound": lo, "upper_bound": hi})

#     out = pd.DataFrame.from_records(records)
#     return out if not out.empty else pd.DataFrame(columns=[well_col, "metric", "lower_bound", "upper_bound"])

# def apply_multi_metric_gates(
#     df: pd.DataFrame,
#     predicates: Dict[Tuple[str, str], Callable[[float], bool]],
#     cfg: "PostHocConfig",
# ) -> pd.DataFrame:
#     """
#     Apply per-(well, metric) predicates.

#     - Anti-leak: ignores test_* metrics.
#     - Vectorized per-well; no row-wise apply(axis=1).
#     - NaN is strict: fails the gate when a predicate exists.
#     - Always writes passes_gates as bool.
#     """
#     import pandas as pd

#     if df is None or df.empty:
#         out = df.copy() if df is not None else pd.DataFrame()
#         out["passes_gates"] = pd.Series(dtype=bool)
#         return out

#     well_col = getattr(cfg, "well_col", "well")
#     cfg_metrics = getattr(cfg, "metrics", []) or []
#     metrics = [m for m in cfg_metrics if (m in df.columns and not str(m).startswith("test_"))]

#     out = df.copy()

#     # If we can't gate, pass everything deterministically
#     if (well_col not in out.columns) or (not metrics) or (not predicates):
#         out["passes_gates"] = True
#         out["passes_gates"] = out["passes_gates"].astype(bool)
#         return out

#     passes = pd.Series(True, index=out.index)

#     # Group once by well (fast) and apply each metric's predicate only within the group
#     for w, idx in out.groupby(well_col, sort=False).groups.items():
#         # idx is an Index of row positions for this well
#         for metric in metrics:
#             pred = predicates.get((w, metric))
#             if pred is None:
#                 continue  # no restriction for this (well, metric)

#             s = out.loc[idx, metric]
#             # strict NaN handling: NaN => fail if predicate exists
#             metric_ok = s.map(pred)
#             metric_ok = metric_ok.fillna(False).astype(bool)

#             passes.loc[idx] &= metric_ok.values

#     out["passes_gates"] = passes.astype(bool)
#     return out


from .posthoc_config import PostHocConfig


# def pareto_mark(df: pd.DataFrame, cfg: PostHocConfig) -> pd.DataFrame:
#     """
#     Marks rows that are on the Pareto front (per well) for cfg.pareto_metrics.

#     Plug-and-play + robust:
#     - Direction-aware: respects cfg.lower_is_better per metric.
#       (Metrics with higher_is_better are internally negated so we always minimize.)
#     - NaN-safe: rows with any NaN in pareto metrics are marked as not Pareto.
#     - Defensive: if pareto_metrics invalid/missing, creates 'is_pareto' and returns.

#     Writes:
#       - is_pareto: bool
#     """
#     if df is None or df.empty:
#         out = df.copy() if df is not None else pd.DataFrame()
#         out["is_pareto"] = False
#         return out

#     well_col = getattr(cfg, "well_col", "well")
#     pareto_cols = list(getattr(cfg, "pareto_metrics", []) or [])

#     out = df.copy()
#     out["is_pareto"] = False

#     # Basic guards
#     if not pareto_cols or well_col not in out.columns:
#         return out

#     # Keep only columns present
#     present = [c for c in pareto_cols if c in out.columns]
#     if len(present) < 2:
#         return out

#     lib_isb = dict(getattr(cfg, "lower_is_better", {}) or {})

#     # Pre-compute direction multipliers: +1 for lower-is-better, -1 for higher-is-better
#     # After transform, we ALWAYS minimize.
#     mult = np.array([1.0 if bool(lib_isb.get(m, True)) else -1.0 for m in present], dtype=float)

#     def _pareto_min(points: np.ndarray) -> np.ndarray:
#         """
#         Return mask of non-dominated points under minimization.

#         A point j is dominated if there exists i such that:
#           points[i] <= points[j] in all dims AND points[i] < points[j] in at least one dim.
#         """
#         n = points.shape[0]
#         is_eff = np.ones(n, dtype=bool)
#         for i in range(n):
#             if not is_eff[i]:
#                 continue
#             p = points[i]
#             # Any point that is >= p in all dims and > p in some dim is dominated by p (minimization)
#             dominated = np.all(points >= p, axis=1) & np.any(points > p, axis=1)
#             dominated[i] = False
#             is_eff[dominated] = False
#         return is_eff

#     for _, g in out.groupby(well_col, sort=False, dropna=False):
#         if g.empty:
#             continue

#         block = g[present].apply(pd.to_numeric, errors="coerce")
#         valid = block.notna().all(axis=1)
#         if not bool(valid.any()):
#             continue

#         pts = (block.loc[valid].to_numpy(dtype=float) * mult)  # direction-aware -> minimization
#         mask_valid = _pareto_min(pts)

#         # write only on valid rows; invalid remain False
#         out.loc[block.index[valid], "is_pareto"] = mask_valid

#     out["is_pareto"] = out["is_pareto"].astype(bool)
#     return out




# def _to_list(x):
#     if x is None:
#         return None
#     if isinstance(x, (list, tuple, set)):
#         return list(x)
#     return [x]




# #
# # Helper Data Structure and Functions (com as correções e novas lógicas)
# #
# import warnings
# from dataclasses import dataclass, field
# from typing import List, Dict, Any, Optional

# import pandas as pd
# import numpy as np


# @dataclass
# class SelectionConfig:
#     """Holds all parsed configuration for the champion selection process."""
#     well_col: str
#     arch_col: Optional[str]
#     strat_col: str
#     score_col: str
#     lower_is_better: bool
#     apply_pareto: bool
#     top_n_strategies: int
#     per_strategy_k: int
#     selection_strategy: str
#     min_samples_per_well: int
#     arch_filter: Optional[List[str]] = None
#     id_cols: List[str] = field(default_factory=list)
#     relax_pool: bool = True

#     # ✅ Needed because _parse_config already passes these
#     pool_method: str = "survivors"
#     pool_cfg: Dict[str, Any] = field(default_factory=dict)


# # --- replace this whole function ---
# def _filter_pool_with_relaxation(pool: pd.DataFrame, config: SelectionConfig, *, allow_relax: bool = False) -> pd.DataFrame:
#     """
#     Aplica filtros em níveis:
#       L0: passes_gates & valcum_pass & (pareto se ativo)
#       L1: somente passes_gates
#       L2: somente valcum_pass
#       L3: sem nenhum dos dois

#     Retorna o primeiro nível não-vazio e marca __gate_level__ para diagnóstico.
#     """
#     import pandas as pd

#     if pool is None or pool.empty:
#         return pool

#     def _mark(df: pd.DataFrame, level_name: str) -> pd.DataFrame:
#         out = df.copy()
#         if not out.empty:
#             out["__gate_level__"] = level_name
#         return out

#     def level_df(df: pd.DataFrame, level_name: str) -> pd.DataFrame:
#         out = df

#         if level_name in ("L0", "L1"):
#             if "passes_gates" in out.columns:
#                 out = out[out["passes_gates"].astype(bool)]

#         if level_name in ("L0", "L2"):
#             if "valcum_pass" in out.columns:
#                 out = out[out["valcum_pass"].astype(bool)]

#         if level_name == "L0" and bool(getattr(config, "apply_pareto", False)):
#             if "is_pareto" in out.columns:
#                 out = out[out["is_pareto"].astype(bool)]

#         return _mark(out, level_name)

#     l0 = level_df(pool, "L0")
#     if not allow_relax:
#         return l0

#     if not l0.empty:
#         return l0

#     l1 = level_df(pool, "L1")
#     if not l1.empty:
#         return l1

#     l2 = level_df(pool, "L2")
#     if not l2.empty:
#         return l2

#     return _mark(pool, "L3")




# def _parse_config(df: pd.DataFrame, cfg: "PostHocConfig") -> SelectionConfig:
#     """
#     Parses the raw config object into a structured SelectionConfig.

#     IMPORTANT:
#     - `score_col` is the DECISION/ORDERING column.
#     - Direction MUST be resolved by a single source of truth:
#         resolve_ordering(score_col, lower_is_better=cfg.lower_is_better, default_ascending=True)
#       This avoids "silent inversions" AND avoids hardcoding assumptions that may
#       not match how the score was constructed (e.g., your weighted_score is lower-is-better).
#     """
#     import logging
#     from hpo.score_ordering import resolve_ordering  # local import to avoid circular issues

#     well_col = getattr(cfg, "well_col", "well")
#     strat_col = getattr(cfg, "strategy_col", "physics_strategy")

#     arch_col = getattr(cfg, "arch_col", None)
#     if arch_col not in df.columns:
#         for cand in ("architecture", "architecture_name", "arch"):
#             if cand in df.columns:
#                 arch_col = cand
#                 break
#         else:
#             arch_col = None

#     # -----------------------------------------------------------
#     # Explicit: selection_col = decision/ordering column
#     # Priority: cfg.selection_col > cfg.metric_to_optimize
#     # -----------------------------------------------------------
#     requested_selection_col = getattr(cfg, "selection_col", None) or getattr(cfg, "metric_to_optimize", "weighted_score")

#     # Deterministic fallback if missing
#     if requested_selection_col not in df.columns:
#         fallback_candidates = [
#             "robust_score",
#             "weighted_score",
#             "val_smape_agg",
#             "val_smape_cum",
#         ]
#         fallback = next((c for c in fallback_candidates if c in df.columns), None)
#         if fallback is None:
#             raise ValueError(
#                 f"[selection_cfg] Requested selection_col='{requested_selection_col}' not found and no fallback is available. "
#                 f"Available cols sample={list(df.columns)[:40]}"
#             )

#         logging.warning(
#             "[selection_cfg] selection_col='%s' not found. Falling back to '%s'. "
#             "Fix config wiring: set PostHocConfig.selection_col (preferred) or metric_to_optimize (legacy).",
#             requested_selection_col, fallback
#         )
#         score_col = fallback
#     else:
#         score_col = requested_selection_col

#     # ---------------------------------------------------------------------
#     # Resolve direction via single source of truth (no hardcoded overrides)
#     # ---------------------------------------------------------------------
#     lib_isb = getattr(cfg, "lower_is_better", {}) or {}
#     ordering = resolve_ordering(score_col, lower_is_better=lib_isb, default_ascending=True)
#     isb = ordering.ascending  # True => lower is better

#     top_n = getattr(cfg, "top_strategies_per_well", getattr(cfg, "top_k_per_well", 1))

#     arch_filter = getattr(cfg, "arch_filter", None)
#     if arch_filter is not None and not isinstance(arch_filter, (list, tuple, set)):
#         arch_filter = [arch_filter]

#     pool_method = str(getattr(cfg, "pool_method", "survivors")).lower()

#     # Auditable log
#     logging.info(
#         "[selection_cfg] selection_col=%s (requested=%s) lower_is_better=%s higher_is_better=%s reason=%s "
#         "strat_col=%s arch_col=%s top_n=%s per_k=%s pool=%s",
#         score_col,
#         requested_selection_col,
#         bool(isb),
#         bool(not isb),
#         ordering.reason,
#         strat_col,
#         arch_col,
#         int(top_n),
#         int(getattr(cfg, "per_strategy_k", 1)),
#         pool_method,
#     )

#     return SelectionConfig(
#         well_col=well_col,
#         arch_col=arch_col,
#         strat_col=strat_col,
#         score_col=score_col,
#         lower_is_better=bool(isb),
#         apply_pareto=bool(getattr(cfg, "apply_pareto", False)),
#         top_n_strategies=int(top_n),
#         per_strategy_k=int(getattr(cfg, "per_strategy_k", 1)),
#         selection_strategy=getattr(cfg, "selection_strategy", "best_of_the_best"),
#         min_samples_per_well=int(getattr(cfg, "min_samples_per_well", 0)),
#         arch_filter=arch_filter,
#         id_cols=list(getattr(cfg, "id_cols", [])),
#         relax_pool=bool(getattr(cfg, "relax_pool", True)),
#         pool_method=pool_method,
#         pool_cfg=dict(getattr(cfg, "pool_cfg", {}) or {}),
#     )




# def _pick_k_exemplars(g_sorted: pd.DataFrame, config: SelectionConfig) -> pd.DataFrame:
#     """Picks the top K rows based on the selection strategy, returning a safe copy."""
#     if config.selection_strategy == "median_of_the_best":
#         n = len(g_sorted)
#         take = min(config.per_strategy_k, n)
#         start = max(0, (n // 2) - (take // 2))
#         return g_sorted.iloc[start:start + take].copy() # Cópia explícita
#     else: # "best_of_the_best"
#         return g_sorted.head(config.per_strategy_k).copy() # Cópia explícita

# def _finalize_champion_df(df: pd.DataFrame, config: SelectionConfig) -> pd.DataFrame:
#     """Sorts, cleans, and validates the final DataFrame of champions."""
#     if df.empty:
#         return df
    
#     sort_cols = [c for c in (config.well_col, config.arch_col, config.strat_col, config.score_col) if c and c in df.columns]
#     sort_asc = [config.lower_is_better if c == config.score_col else True for c in sort_cols]
#     out = df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
#     out = out.drop(columns=["__rep_score__", "__source__"], errors="ignore")
#     return out


# def select_champions_by_strategy(df: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
#     import logging
#     import warnings
#     from typing import List

#     import pandas as pd

#     from hpo.score_ordering import resolve_ordering
#     from hpo.sort_utils import infer_tiebreak_cols, stable_sort

#     if df is None or df.empty:
#         return pd.DataFrame() if df is None else df.copy()

#     config = _parse_config(df, cfg)
#     df_prepared = df.copy()

#     # selection_col is the ORDERING column (single source of truth)
#     selection_col = (
#         getattr(config, "score_col", None)
#         or getattr(cfg, "selection_col", None)
#         or getattr(cfg, "metric_to_optimize", "weighted_score")
#     )

#     if selection_col not in df_prepared.columns:
#         fallback_candidates = ["val_smape_agg", "val_smape_cum", "weighted_score", "robust_score"]
#         fallback = next((c for c in fallback_candidates if c in df_prepared.columns), None)
#         if fallback is None:
#             raise KeyError(
#                 f"[selection] selection_col='{selection_col}' not found and no fallback metrics available. "
#                 f"Available cols sample={list(df_prepared.columns)[:30]}"
#             )
#         logging.warning(
#             "[selection] selection_col='%s' missing. Falling back to '%s'.", selection_col, fallback
#         )
#         selection_col = fallback
#         try:
#             config.score_col = fallback
#         except Exception:
#             pass

#     lib_isb = dict(getattr(cfg, "lower_is_better", {}) or {})
#     ordering = resolve_ordering(selection_col, lower_is_better=lib_isb, default_ascending=True)

#     tiebreak_cols = infer_tiebreak_cols(df_prepared)

#     logging.info(
#         "[selection] selection_col=%s | ascending=%s | higher_is_better=%s | reason=%s | "
#         "selection_strategy=%s | top_n=%s | per_strategy_k=%s | relax_pool=%s",
#         selection_col,
#         ordering.ascending,
#         ordering.higher_is_better,
#         ordering.reason,
#         getattr(config, "selection_strategy", "unknown"),
#         getattr(config, "top_n_strategies", None),
#         getattr(config, "per_strategy_k", None),
#         getattr(config, "relax_pool", None),
#     )

#     # Optional arch filter
#     if getattr(config, "arch_filter", None) and getattr(config, "arch_col", None):
#         df_prepared = df_prepared[df_prepared[config.arch_col].isin(config.arch_filter)]
#         if df_prepared.empty:
#             warnings.warn("No rows left after applying arch_filter.")
#             return df_prepared

#     final_blocks: List[pd.DataFrame] = []
#     processed_indices = set()

#     # ---------------------------
#     # (A) DARTS path
#     # ---------------------------
#     if getattr(config, "arch_col", None) and "physics_strategy" in df_prepared.columns:
#         darts_mask = df_prepared[config.arch_col].astype(str).str.contains("Darts", case=False, na=False)
#         df_darts = df_prepared[darts_mask]
#         if not df_darts.empty:
#             for _, g in df_darts.groupby(config.well_col, sort=False, dropna=False):
#                 pool = _filter_pool_with_relaxation(g, config, allow_relax=bool(getattr(config, "relax_pool", False)))
#                 if pool.empty:
#                     continue

#                 best_per_ps = (
#                     stable_sort(pool, selection_col, ordering, tiebreak_cols=tiebreak_cols)
#                     .groupby("physics_strategy", sort=False, dropna=False)
#                     .head(int(getattr(config, "per_strategy_k", 1)))
#                 )
#                 picked = stable_sort(best_per_ps, selection_col, ordering, tiebreak_cols=tiebreak_cols).head(
#                     int(getattr(config, "top_n_strategies", 1))
#                 )

#                 if not picked.empty:
#                     picked = picked.drop_duplicates(subset=[config.well_col, "physics_strategy"], keep="first")
#                     final_blocks.append(picked)
#                     processed_indices.update(picked.index)

#             processed_indices.update(df_darts.index)

#     # ----------------------------------------
#     # (B) Other architectures
#     # ----------------------------------------
#     df_rest = df_prepared.drop(index=list(processed_indices)) if processed_indices else df_prepared
#     if getattr(config, "arch_col", None) and not df_rest.empty:
#         for _, df_arch in df_rest.groupby(config.arch_col, sort=False, dropna=False):
#             diversity_col = None
#             try:
#                 from .analysis_utils import diversity_col_for_arch
#                 diversity_col = diversity_col_for_arch(df_arch)
#             except Exception:
#                 diversity_col = None

#             for _, group_df in df_arch.groupby([config.well_col, config.arch_col], sort=False, dropna=False):
#                 pool = _filter_pool_with_relaxation(
#                     group_df, config, allow_relax=bool(getattr(config, "relax_pool", False))
#                 )
#                 if pool.empty:
#                     continue

#                 if diversity_col and diversity_col in pool.columns:
#                     best_per_div = (
#                         stable_sort(pool, selection_col, ordering, tiebreak_cols=tiebreak_cols)
#                         .groupby(diversity_col, sort=False, dropna=False)
#                         .head(int(getattr(config, "per_strategy_k", 1)))
#                     )
#                     picked = stable_sort(best_per_div, selection_col, ordering, tiebreak_cols=tiebreak_cols).head(
#                         int(getattr(config, "top_n_strategies", 1))
#                     )
#                 else:
#                     sorted_pool = stable_sort(pool, selection_col, ordering, tiebreak_cols=tiebreak_cols)

#                     if getattr(config, "selection_strategy", "") == "median_of_the_best":
#                         n = min(int(getattr(config, "top_n_strategies", 1)), len(sorted_pool))
#                         if n <= 0:
#                             picked = sorted_pool.head(0)
#                         else:
#                             start = max(0, (len(sorted_pool) // 2) - (n // 2))
#                             picked = sorted_pool.iloc[start : start + n].copy()
#                     else:
#                         picked = sorted_pool.head(int(getattr(config, "top_n_strategies", 1))).copy()

#                 if not picked.empty:
#                     final_blocks.append(picked)
#                     processed_indices.update(picked.index)

#     if not final_blocks:
#         return pd.DataFrame()

#     champions_df = pd.concat(final_blocks, ignore_index=True)
#     return _finalize_champion_df(champions_df, config)





# def thresholds_table(qdf: pd.DataFrame, mdf: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
#     """
#     Merge quantile and MAD thresholds into a single table for reporting.

#     Robustness:
#     - Uses cfg.well_col, not hardcoded 'well'.
#     - Works if either side is empty.
#     """
#     import pandas as pd

#     well_col = getattr(cfg, "well_col", "well")

#     qdf = qdf if qdf is not None else pd.DataFrame()
#     mdf = mdf if mdf is not None else pd.DataFrame()

#     if qdf.empty and mdf.empty:
#         return pd.DataFrame(columns=[well_col, "metric", "cutoff", "quantile", "lower_bound", "upper_bound"])
#     if mdf.empty:
#         return qdf.copy()
#     if qdf.empty:
#         return mdf.copy()

#     return pd.merge(qdf, mdf, on=[well_col, "metric"], how="left")



# def plot_well_dashboard(well_df, thresholds, champions_df, well_name, cfg) -> Any:
#     """Builds a visual dashboard for a single well."""
#     if not VIZ_LIBS_INSTALLED:
#         warnings.warn("Visualization libraries (plotly) not installed. Skipping plot generation.")
#         return None

#     fig = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=(
#             f"{cfg.metrics[0]} Distribution",
#             f"{cfg.metrics[1]} vs {cfg.metrics[2]}",
#             f"{cfg.metrics[1]} Distribution",
#             f"{cfg.metrics[2]} Distribution"
#         ),
#         specs=[[{}, {"type": "scatter"}], [{}, {}]]
#     )

#     colors = ['#636EFA', '#EF553B', '#00CC96']

#     # Subplots for metric distributions
#     for i, metric in enumerate(cfg.metrics):
#         row, col = (1, 1) if i == 0 else (2, 1) if i == 1 else (2, 2)
        
#         # Histogram
#         fig.add_trace(go.Histogram(x=well_df[metric], name=metric, marker_color=colors[i], nbinsx=30), row=row, col=col)
        
#         # Threshold lines
#         metric_thresholds = thresholds[thresholds['metric'] == metric]
#         if not metric_thresholds.empty:
#             cutoff = metric_thresholds['cutoff'].iloc[0]
#             upper_bound = metric_thresholds['upper_bound'].iloc[0]
#             fig.add_vline(x=cutoff, line_width=2, line_dash="dash", line_color="firebrick",
#                           annotation_text=f"Q({metric_thresholds['quantile'].iloc[0]})", row=row, col=col)
#             if np.isfinite(upper_bound):
#                 fig.add_vline(x=upper_bound, line_width=2, line_dash="dot", line_color="goldenrod",
#                               annotation_text="MAD Guard", row=row, col=col)

#         # Champion markers
#         fig.add_trace(go.Scatter(
#             x=champions_df[metric], y=np.zeros(len(champions_df)),
#             mode='markers', name='Champions',
#             marker=dict(symbol='star', color='gold', size=15, line=dict(color='black', width=1))
#         ), row=row, col=col)

#     # Scatter plot for Pareto metrics
#     metric_x, metric_y = cfg.pareto_metrics[0], cfg.pareto_metrics[1]
#     for arch, group in well_df.groupby(cfg.arch_col):
#         fig.add_trace(go.Scatter(
#             x=group[metric_x], y=group[metric_y],
#             mode='markers', name=arch,
#             marker=dict(opacity=0.7)
#         ), row=1, col=2)
    
#     # Highlight champions on scatter
#     fig.add_trace(go.Scatter(
#         x=champions_df[metric_x], y=champions_df[metric_y],
#         mode='markers', name='Champions',
#         marker=dict(symbol='star', color='gold', size=15, line=dict(color='black', width=1))
#     ), row=1, col=2)
    
#     fig.update_layout(
#         title_text=f"<b>Analysis Dashboard for Well: {well_name}</b>",
#         showlegend=True, height=800
#     )
#     fig.update_xaxes(title_text=metric_x, row=1, col=2)
#     fig.update_yaxes(title_text=metric_y, row=1, col=2)
    
#     fig.show()

# ==============================================================================
# ORCHESTRATION FUNCTION
# ==============================================================================

# def ensure_mad_for_metric(
#     df: pd.DataFrame,
#     mdf: pd.DataFrame | None,
#     well_col: str,
#     metric: str,
#     lower_is_better: bool,
#     alpha: float = 1.0,
# ) -> pd.DataFrame:
#     """
#     Ensure there is a MAD row per (well, metric).
#     - If missing or non-finite in mdf, compute robust bounds and inject.
#     - Returns a normalized mdf with columns: [well_col, metric, lower_bound, upper_bound].
#     """
#     import numpy as np
#     import pandas as pd
#     from scipy.stats import median_abs_deviation

#     metric = str(metric)

#     def _safe_mad(x: pd.Series) -> float:
#         x = pd.to_numeric(x, errors="coerce").dropna()
#         if x.empty:
#             return 0.0
#         m = float(median_abs_deviation(x, scale="normal"))
#         if not np.isfinite(m) or m <= 0:
#             return 1e-12
#         return m

#     if df is None or df.empty or metric not in df.columns or well_col not in df.columns:
#         return mdf if mdf is not None else pd.DataFrame(columns=[well_col, "metric", "lower_bound", "upper_bound"])

#     dfm = df[[well_col, metric]].copy()
#     dfm[metric] = pd.to_numeric(dfm[metric], errors="coerce")
#     dfm = dfm.dropna(subset=[metric])
#     if dfm.empty:
#         return mdf if mdf is not None else pd.DataFrame(columns=[well_col, "metric", "lower_bound", "upper_bound"])

#     rows = []
#     for w, g in dfm.groupby(well_col, sort=False, dropna=False):
#         med = float(g[metric].median())
#         mad = _safe_mad(g[metric])
#         if lower_is_better:
#             lo, hi = -np.inf, med + alpha * mad
#         else:
#             lo, hi = med - alpha * mad, np.inf

#         # sanitize
#         if lower_is_better and not np.isfinite(hi):
#             hi = med
#         if (not lower_is_better) and not np.isfinite(lo):
#             lo = med

#         rows.append({well_col: w, "metric": metric, "lower_bound": float(lo), "upper_bound": float(hi)})

#     inj = pd.DataFrame(rows, columns=[well_col, "metric", "lower_bound", "upper_bound"])

#     if mdf is None or mdf.empty:
#         return inj

#     # Normalize mdf cols
#     m = mdf.copy()
#     for c in [well_col, "metric", "lower_bound", "upper_bound"]:
#         if c not in m.columns:
#             m[c] = np.nan

#     key = [well_col, "metric"]
#     m = m.merge(inj, on=key, how="outer", suffixes=("_old", "_new"))

#     def pick(old, new):
#         old = pd.to_numeric(old, errors="coerce")
#         new = pd.to_numeric(new, errors="coerce")
#         return np.where(np.isfinite(old), old, new)

#     m["lower_bound"] = pick(m["lower_bound_old"], m["lower_bound_new"])
#     m["upper_bound"] = pick(m["upper_bound_old"], m["upper_bound_new"])

#     return m[[well_col, "metric", "lower_bound", "upper_bound"]]



# def run_distribution_filter(master_df: pd.DataFrame, cfg: PostHocConfig) -> Dict[str, Any]:
#     """
#     Orchestrates the robust filtering + champion selection pipeline.

#     Explicit semantics:
#     - "survivors": rows that passed 'passes_gates' (diagnostic only)
#     - "df_for_selection": rows actually available for selection (LEGACY => survivors only)
#     - selection column = cfg.metric_to_optimize (resolved earlier in pipeline)

#     Anti-leak policy:
#     - Selection must never see any 'test_*' columns. Those are audit-only and must be handled outside.

#     Legacy lock:
#     - pool_method is effectively ALWAYS "survivors" in legacy.
#       If cfg.pool_method is "top_pct" or "val_band", we warn and override.
#     """
#     import logging
#     import numpy as np
#     import pandas as pd

#     from hpo.score_semantics import resolve_score_semantics, ensure_semantic_score_columns

#     logger = logging.getLogger(__name__)

#     # Resolve score_col early for meta consistency even on early-return
#     score_col = getattr(cfg, "metric_to_optimize", "weighted_score") or "weighted_score"
#     sem = resolve_score_semantics(str(score_col), default_robust_kind="legacy")

#     if master_df is None or master_df.empty:
#         logger.info("[filter] Empty master_df. Nothing to filter.")
#         return {
#             "top_performers": pd.DataFrame(),
#             "thresholds": pd.DataFrame(),
#             "survivors": pd.DataFrame(),
#             "gate_counts": pd.DataFrame(),
#             "pool_counts": pd.DataFrame(),
#             "df_gated": pd.DataFrame(),
#             "meta": {"score_semantics": sem.__dict__},
#         }

#     logger.info("[filter] Starting distribution filtering pipeline...")

#     # -------------------------------------------------------------------------
#     # 0) Anti-leak hard: drop any TEST columns before *any* selection logic
#     # -------------------------------------------------------------------------
#     test_cols = [c for c in master_df.columns if str(c).startswith("test_")]
#     if test_cols:
#         logger.info("[filter][anti_leak] Dropping %d test_* columns from selection input.", len(test_cols))
#     master_df_sel = master_df.drop(columns=test_cols, errors="ignore")

#     # -------------------------------------------------------------------------
#     # 1) Prepare Data
#     # -------------------------------------------------------------------------
#     df = sanitize_and_validate(master_df_sel, cfg)
#     df = deduplicate_by_signature(df, cfg)

#     score_col = getattr(cfg, "metric_to_optimize", "weighted_score") or "weighted_score"
#     sem = resolve_score_semantics(str(score_col), default_robust_kind="legacy")

#     if str(score_col) == "robust_score":
#         try:
#             df, cfg = plug_robust_score(df, cfg)
#             score_col = "robust_score"
#             sem = resolve_score_semantics(str(score_col), default_robust_kind="legacy")
#         except Exception as e:
#             logger.warning("[filter][plug] plug_robust_score failed (%s). Falling back to 'weighted_score'.", e)
#             try:
#                 cfg.metric_to_optimize = "weighted_score"
#             except Exception:
#                 pass
#             score_col = "weighted_score"
#             sem = resolve_score_semantics(str(score_col), default_robust_kind="legacy")

#     # Ensure semantic alias columns exist (no behavior changes; extra columns only)
#     df = ensure_semantic_score_columns(df, sem)

#     n_wells = df[cfg.well_col].nunique() if cfg.well_col in df.columns else 0
#     logger.info("[filter] Analyzing %d trials across %d wells.", len(df), n_wells)

#     # -------------------------------------------------------------------------
#     # 2) Compute Thresholds (Quantiles + MAD)
#     # -------------------------------------------------------------------------
#     qdf = quantile_thresholds(df, cfg)
#     mdf = mad_guards(df, cfg)
#     th_tab = thresholds_table(qdf, mdf, cfg)

#     if (cfg.mad_guard or {}).get("enabled", False):
#         mdf = ensure_mad_for_metric(
#             df=df,
#             mdf=mdf,
#             well_col=cfg.well_col,
#             metric=score_col,
#             lower_is_better=bool((cfg.lower_is_better or {}).get(score_col, True)),
#             alpha=float((cfg.mad_guard or {}).get("alpha", 1.0)),
#         )

#     logger.info("[filter] Computed quantile and MAD thresholds.")

#     # -------------------------------------------------------------------------
#     # 3) Apply Gates (multi-metric + valcum)
#     # -------------------------------------------------------------------------
#     preds = compose_predicates(qdf, mdf, cfg)
#     df_gated = apply_multi_metric_gates(df, preds, cfg)
#     df_gated = apply_valcum_gate(df_gated, cfg)

#     # Survivors = diagnostic (passes_gates only)
#     if "passes_gates" in df_gated.columns:
#         survivors_mask = df_gated["passes_gates"].astype(bool)
#     else:
#         survivors_mask = pd.Series(True, index=df_gated.index)

#     survivors = df_gated.loc[survivors_mask].copy()

#     # Gate counts diagnostics (total vs kept_by_gates)
#     total_counts = (
#         df.groupby(cfg.well_col, dropna=False)[cfg.well_col]
#           .count()
#           .rename("total_in_pool")
#           .reset_index()
#     )
#     kept_counts = (
#         survivors.groupby(cfg.well_col, dropna=False)[cfg.well_col]
#                  .count()
#                  .rename("kept_by_gates")
#                  .reset_index()
#     )
#     gate_counts = (
#         total_counts.merge(kept_counts, on=cfg.well_col, how="left")
#                     .fillna({"kept_by_gates": 0})
#     )
#     if "kept_by_gates" in gate_counts.columns:
#         gate_counts["kept_by_gates"] = gate_counts["kept_by_gates"].astype(int)

#     for _, row in gate_counts.iterrows():
#         well = row[cfg.well_col]
#         kept = int(row.get("kept_by_gates", 0))
#         tot = int(row.get("total_in_pool", 0))
#         logger.info("[filter][survivors] well=%s kept_by_gates=%d/%d", well, kept, tot)

#     # -------------------------------------------------------------------------
#     # 4) Optional Pareto mark (annotates only)
#     # -------------------------------------------------------------------------
#     df_for_selection = df_gated
#     if getattr(cfg, "apply_pareto", False):
#         df_for_selection = pareto_mark(df_for_selection, cfg)
#         logger.info("[filter] Marked non-dominated (Pareto) solutions.")

#     # -------------------------------------------------------------------------
#     # 4.5) LEGACY LOCK: pool_method is effectively ALWAYS survivors
#     # -------------------------------------------------------------------------
#     requested_pool = str(getattr(cfg, "pool_method", "survivors") or "survivors").lower()
#     if requested_pool != "survivors":
#         logger.warning(
#             "[filter][LEGACY_LOCK] pool_method='%s' requested, but legacy selection only supports 'survivors'. "
#             "Overriding to 'survivors'.",
#             requested_pool,
#         )
#     pool_method = "survivors"

#     # Mark pool rows (for debugging/diagnostics)
#     df_for_selection = df_for_selection.copy()
#     df_for_selection["__pool_included__"] = False

#     if "passes_gates" in df_for_selection.columns:
#         df_for_selection.loc[df_for_selection["passes_gates"].astype(bool), "__pool_included__"] = True
#     else:
#         df_for_selection["__pool_included__"] = True

#     # pool_counts: per-well pool size actually used
#     pool_counts_rows = []
#     for well_name, block in df_for_selection.groupby(cfg.well_col, sort=False, dropna=False):
#         pool_size = int(block["__pool_included__"].astype(bool).sum())
#         pool_counts_rows.append(
#             {
#                 cfg.well_col: well_name,
#                 "pool_method": pool_method,
#                 "pool_size": pool_size,
#             }
#         )
#     pool_counts = pd.DataFrame(pool_counts_rows)

#     for _, r in pool_counts.iterrows():
#         logger.info(
#             "[filter][pool] well=%s method=%s pool_size=%d",
#             r[cfg.well_col],
#             r["pool_method"],
#             int(r["pool_size"]),
#         )

#     # Keep only pool rows for selection
#     df_for_selection = df_for_selection[df_for_selection["__pool_included__"].astype(bool)].copy()

#     # -------------------------------------------------------------------------
#     # 5) Select Final Champions
#     # -------------------------------------------------------------------------
#     top_performers = select_champions_by_strategy(df_for_selection, cfg)

#     if top_performers is None or top_performers.empty:
#         logger.info("[filter] No champions selected after filtering. Gates/pool may be too strict.")
#         return {
#             "top_performers": pd.DataFrame(),
#             "thresholds": th_tab,
#             "survivors": survivors,
#             "gate_counts": gate_counts,
#             "pool_counts": pool_counts,
#             "df_gated": df_gated,
#             "meta": {"score_semantics": sem.__dict__},
#         }

#     # Final picks diagnostics
#     if cfg.well_col in top_performers.columns:
#         picked_counts = (
#             top_performers.groupby(cfg.well_col, dropna=False)[cfg.well_col]
#                           .count()
#                           .rename("picked_final")
#                           .reset_index()
#         )
#         diag_final = (
#             gate_counts.merge(picked_counts, on=cfg.well_col, how="left")
#                        .fillna({"picked_final": 0})
#         )
#         diag_final["picked_final"] = diag_final["picked_final"].astype(int)

#         for _, row in diag_final.iterrows():
#             well = row[cfg.well_col]
#             picked = int(row.get("picked_final", 0))
#             kept_gates = int(row.get("kept_by_gates", 0))
#             tot = int(row.get("total_in_pool", 0))
#             logger.info(
#                 "[filter][final] well=%s picked_final=%d (from survivors=%d) total=%d",
#                 well, picked, kept_gates, tot
#             )
#     else:
#         logger.info("[filter][final] picked_final=%d (no per-well column '%s')", len(top_performers), cfg.well_col)

#     logger.info("[filter] Selected %d final champions.", len(top_performers))

#     # -------------------------------------------------------------------------
#     # 6) Visuals
#     # -------------------------------------------------------------------------
#     if VIZ_LIBS_INSTALLED and getattr(cfg, "plot", False):
#         logger.info("[filter] Generating visual dashboards...")
#         for well_name, group in df.groupby(cfg.well_col, dropna=False):
#             well_champions = top_performers[top_performers[cfg.well_col] == well_name]
#             if not well_champions.empty:
#                 well_thresholds = th_tab[th_tab[cfg.well_col] == well_name]
#                 plot_well_dashboard(group, well_thresholds, well_champions, well_name, cfg)

#     return {
#         "top_performers": top_performers,
#         "thresholds": th_tab,
#         "survivors": survivors,
#         "gate_counts": gate_counts,
#         "pool_counts": pool_counts,
#         "df_gated": df_gated,
#         "meta": {"score_semantics": sem.__dict__},
#     }






# def load_master_leaderboard(campaign_name: str, results_dir: str | Path,
#                             include_validation_alt: bool = True) -> Optional[pd.DataFrame]:
#     """
#     Lê leaderboards de uma campanha. Tolera CSVs vazios/corrompidos.
#     include_validation_alt: se True, também procura por validation_{campaign}_top_*/leaderboard.csv

#     Backward-compatible and extended:
#       - Supports nested layout:
#           <results_dir>/<campaign_name>/results/<family>/<run>/leaderboard.csv
#       - Also supports when results_dir already IS ".../<campaign_name>/results"
#       - Keeps legacy patterns from the original implementation.
#     """
#     import os
#     from pathlib import Path
#     import pandas as pd
#     from pandas.errors import EmptyDataError, ParserError

#     results_path = Path(results_dir)

#     # --- Collect candidate files from multiple compatible layouts ---
#     files: list[Path] = []

#     # (A) NEW: canonical nested layout where results_dir points to the campaign group root
#     #     <results_dir>/<campaign_name>/results/*/*/leaderboard.csv
#     nested_root_a = results_path / campaign_name / "results"
#     if nested_root_a.is_dir():
#         files += list(nested_root_a.glob("*/*/leaderboard.csv"))

#     # (B) NEW: results_dir may already be ".../<campaign_name>/results"
#     #     <results_dir>/*/*/leaderboard.csv  (and verify path contains the campaign_name)
#     if results_path.name == "results":
#         files_b = list(results_path.glob("*/*/leaderboard.csv"))
#         if files_b:
#             # keep only those whose path contains campaign_name somewhere above
#             filtered_b = []
#             cn = str(campaign_name)
#             for f in files_b:
#                 # Check any ancestor equals campaign_name
#                 if any(p.name == cn for p in f.parents):
#                     filtered_b.append(f)
#             files += filtered_b

#     # (C) Legacy multi-cycle layout (kept from original):
#     files += list((results_path).glob(f"{campaign_name}_cycle_*/leaderboard.csv"))

#     # (D) Legacy "simple validation" layout:
#     p_simple = results_path / campaign_name / "leaderboard.csv"
#     if p_simple.exists():
#         files.append(p_simple)

#     # (E) Legacy/alt validation tops:
#     if include_validation_alt:
#         files += list((results_path).glob(f"validation_{campaign_name}_top_*/leaderboard.csv"))

#     # De-duplicate & keep only actual files
#     files = sorted(set(f for f in files if f.is_file()))

#     # If nothing found, mirror the original user-facing message and return None
#     if not files:
#         print(f"⚠️ WARNING: No leaderboards found for campaign '{campaign_name}' in '{results_path}'.")
#         return None

#     # --- Read all files, tolerate empties/corrupt, aggregate ---
#     valid_dfs = []
#     skipped = []

#     for f in files:
#         try:
#             if os.path.getsize(f) == 0:
#                 skipped.append((str(f), "empty file (0 bytes)"))
#                 continue

#             df = pd.read_csv(f)

#             if df is None or getattr(df, "shape", (0, 0))[1] == 0:
#                 skipped.append((str(f), "no columns after read_csv"))
#                 continue

#             # If a 'campaign' column is not present, add it using the run-folder name.
#             # This is safe and helps Stage 1 harvesting; doesn't break existing code that ignores extra cols.
#             if "campaign" not in df.columns:
#                 # In nested layout, the run name is the parent directory of the CSV
#                 df = df.copy()
#                 df["campaign"] = f.parent.name

#             valid_dfs.append(df)

#         except (EmptyDataError, ParserError) as e:
#             skipped.append((str(f), f"{type(e).__name__}: {e}"))
#         except Exception as e:
#             skipped.append((str(f), f"Unexpected: {type(e).__name__}: {e}"))

#     if not valid_dfs:
#         print(
#             f"⚠️ WARNING: Found {len(files)} leaderboard file(s) for campaign '{campaign_name}', "
#             f"but none were readable. Skipped: {skipped}"
#         )
#         return None

#     master_df = pd.concat(valid_dfs, ignore_index=True)

#     if skipped:
#         print(f"ℹ️ Skipped {len(skipped)} file(s) for '{campaign_name}':")
#         for path, reason in skipped:
#             print(f"   - {path} → {reason}")

#     msg = "Aggregated" if len(valid_dfs) > 1 else "Loaded"
#     print(f"✅ {msg} {len(valid_dfs)} leaderboard(s) for campaign '{campaign_name}' with {len(master_df)} total trials.")
#     return master_df



# def _parse_campaign_name(campaign_name: str) -> Dict[str, str]:
#     """
#     Robustly parse dataset / well / (optional) architecture from campaign names.

#     Works for examples like:
#       - validation_UNISIM_IV_P15_Seq2PIN
#       - UNISIM_IV_P15_Seq2Context
#       - robust_smoke_darts_UNISIM_IV_P15          (no arch suffix)
#       - UNISIM_IV_P15                              (no arch suffix)

#     Returns {"dataset": "...", "well": "...", "architecture": "..."} with "unknown" if not matched.
#     """
#     pattern = re.compile(
#         r'^(?:validation_|robust_smoke_darts_)?'                # known optional prefixes
#         r'(?P<dataset>VOLVE|UNISIM_IV|UNISIM)_'                 # dataset
#         r'(?P<well>.+?)'                                        # well (non-greedy)
#         r'(?:_(?P<architecture>Seq2Context|Seq2Trend|Seq2PIN))?'# optional architecture suffix
#         r'(?:_top_\d+)?$'                                       # optional _top_N
#     )

#     m = pattern.match(campaign_name)
#     if not m:
#         return {"dataset": "unknown", "well": "unknown", "architecture": "unknown"}

#     parsed = m.groupdict()
#     parsed["well"] = (parsed.get("well") or "").strip()

#     # Optional normalization for known aliases
#     well_normalization_map = {"F14": "15-9-F-14", "F12": "15-9-F-12"}
#     if parsed["well"] in well_normalization_map:
#         parsed["well"] = well_normalization_map[parsed["well"]]

#     # Fill missing architecture with "unknown"
#     if not parsed.get("architecture"):
#         parsed["architecture"] = "unknown"

#     return parsed

# ==============================================================================
# NOVAS FUNÇÕES DE ROBUSTEZ (PLUG-AND-PLAY) - VERSÃO REFINADA
# ==============================================================================

# def _find_arch_col(df: pd.DataFrame, arch_col: Optional[str] = None) -> str:
#     """Helper para encontrar a coluna de arquitetura de forma robusta."""
#     if arch_col and arch_col in df.columns:
#         return arch_col
#     for cand in ("architecture", "architecture_name", "arch"):
#         if cand in df.columns:
#             return cand
#     raise ValueError("Nenhuma coluna de arquitetura encontrada em: ['architecture', 'architecture_name', 'arch']")

# def neighbor_signature_cols(arch_name: str, df_columns: List[str]) -> List[str]:
#     """
#     Define, por arquitetura, quais colunas formam a "assinatura do vizinho".
#     Retorna apenas colunas que existem no DataFrame.
#     """
#     arch_name = arch_name or "unknown"
#     base_cols = ["well", "architecture"]
    
#     if "Seq2PIN" in arch_name:
#         sig_cols = base_cols + ["physics_strategy", "aggregation_method", "lag_window", "horizon"]
#     elif "Arps_Canonical" in arch_name:
#         sig_cols = base_cols + ["variant", "lag_window", "horizon"]
#     elif "Darts" in arch_name:
#         profile_col = "profile" if "profile" in df_columns else "physics_strategy"
#         sig_cols = base_cols + [profile_col, "lag_window", "horizon", "input_chunk_length", "output_chunk_length"]
#     else:
#         sig_cols = base_cols

#     return [col for col in sig_cols if col in df_columns]

# def compute_neighbor_iqr(
#     df: pd.DataFrame,
#     metric: str = "val_smape_cum",
#     arch_col: Optional[str] = None,
#     min_group: int = 4
# ) -> pd.Series:
#     """
#     Calcula o IQR de uma métrica em grupos de vizinhos "próximos" (mesma assinatura),
#     normalizado por (well, architecture). Penaliza grupos pequenos.
#     """
#     if df.empty or metric not in df.columns:
#         return pd.Series(index=df.index, dtype=float).fillna(0.0)

#     arch_col_found = _find_arch_col(df, arch_col)
#     output_iqr_norm = pd.Series(np.nan, index=df.index, name="neighbor_iqr_cum_norm")

#     for (well, arch), group in df.groupby(["well", arch_col_found]):
#         if group.empty:
#             continue
        
#         sig_cols = neighbor_signature_cols(arch, list(df.columns))
#         if not all(c in group.columns for c in sig_cols):
#             continue

#         iqr_series = group.groupby(sig_cols, dropna=False)[metric].transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
#         group_sizes = group.groupby(sig_cols, dropna=False)[metric].transform('size')
        
#         # Rank alto (próximo de 1.0) significa IQR alto (ruim)
#         iqr_norm = iqr_series.rank(pct=True, ascending=True).fillna(0.5).clip(0, 1)
        
#         # penaliza => puxa pra 1.0 (ruim)
#         iqr_norm.loc[group_sizes < min_group] = np.maximum(iqr_norm.loc[group_sizes < min_group], 0.75)

#         output_iqr_norm.loc[group.index] = iqr_norm

#     return output_iqr_norm.fillna(0.0)

# def compute_rank_gap(
#     df: pd.DataFrame,
#     by: List[str] = ["well", "architecture"],
#     agg_col: str = "val_smape_agg",
#     cum_col: str = "val_smape_cum"
# ) -> pd.Series:
#     """
#     Penaliza o desalinhamento entre os ranks de val_smape_agg e val_smape_cum.
#     """
#     if df.empty or agg_col not in df.columns or cum_col not in df.columns:
#         return pd.Series(index=df.index, dtype=float).fillna(0.0)

#     by_cols = by[:]
#     if "architecture" not in by_cols:
#         arch_col_found = _find_arch_col(df)
#         if arch_col_found not in by_cols:
#             by_cols.append(arch_col_found)
    
#     valid_by = [c for c in by_cols if c in df.columns]

#     def _compute_gap(group: pd.DataFrame) -> pd.Series:
#         r_agg = group[agg_col].rank(pct=True, ascending=True)
#         r_cum = group[cum_col].rank(pct=True, ascending=True)
#         return abs(r_agg - r_cum)

#     gap = df.groupby(valid_by, group_keys=False).apply(_compute_gap) if valid_by else _compute_gap(df)
    
#     return gap.rename("rank_gap_norm").fillna(0.0)

# # Em src/hpo/analysis_utils.py

# def add_robust_score(
#     df: pd.DataFrame,
#     weights_by_arch: Optional[Dict[str, Dict]] = None,
#     arch_col: Optional[str] = None
# ) -> pd.DataFrame:
#     """
#     LEGACY robust score (min-max + penalties).
#     Backward compatible:
#       - Produces 'robust_score' (existing behavior)
#       - Also produces 'legacy_robust_score' (semantic alias)
#     """
#     out = df.copy()
#     arch_col_found = _find_arch_col(out, arch_col)

#     if "neighbor_iqr_cum_norm" not in out.columns:
#         out["neighbor_iqr_cum_norm"] = compute_neighbor_iqr(out, arch_col=arch_col_found)
#     if "rank_gap_norm" not in out.columns:
#         out["rank_gap_norm"] = compute_rank_gap(out, by=["well", arch_col_found])

#     default_weights = {
#         "Seq2PIN":        {"w_agg": 1.00, "w_cum": 0.25, "alpha": 0.15, "beta": 0.10},
#         "Arps_Canonical": {"w_agg": 1.00, "w_cum": 0.25, "alpha": 0.15, "beta": 0.10},
#         "Darts":          {"w_agg": 1.00, "w_cum": 0.25, "alpha": 0.15, "beta": 0.10},
#         "default":        {"w_agg": 1.00, "w_cum": 0.25, "alpha": 0.15, "beta": 0.10},
#     }
#     if weights_by_arch:
#         default_weights.update(weights_by_arch)

#     out["robust_score"] = np.nan

#     def min_max_scale(s: pd.Series) -> pd.Series:
#         s_min, s_max = s.min(), s.max()
#         if s_max == s_min:
#             return pd.Series(0.5, index=s.index)
#         return (s - s_min) / (s_max - s_min)

#     for (well, arch), group in out.groupby(["well", arch_col_found], sort=False, dropna=False):
#         if group.empty:
#             continue

#         w = next((v for k, v in default_weights.items() if k in str(arch)), default_weights["default"])

#         norm_agg = min_max_scale(group["val_smape_agg"])
#         norm_cum = min_max_scale(group["val_smape_cum"])
#         norm_iqr = min_max_scale(group["neighbor_iqr_cum_norm"])
#         norm_gap = min_max_scale(group["rank_gap_norm"])

#         score = (
#             w["w_agg"] * norm_agg +
#             w["w_cum"] * norm_cum +
#             w["alpha"] * norm_iqr +
#             w["beta"]  * norm_gap
#         )

#         out.loc[group.index, "robust_score"] = score

#     # ✅ semantics alias (Patch 2)
#     if "legacy_robust_score" not in out.columns:
#         out["legacy_robust_score"] = out["robust_score"]

#     return out

# def diversity_col_for_arch(df_arch: pd.DataFrame) -> Optional[str]:
#     """
#     Informa qual coluna representa a diversidade interna para um DataFrame de uma arquitetura.
#     """
#     if df_arch.empty:
#         return None
    
#     arch_col = _find_arch_col(df_arch)
#     arch_name = df_arch[arch_col].iloc[0] if not df_arch[arch_col].empty else ""

#     if "Seq2" in arch_name:
#         return "physics_strategy" if "physics_strategy" in df_arch.columns else None
#     elif "Arps" in arch_name:
#         return "variant" if "variant" in df_arch.columns else None
#     elif "Darts" in arch_name:
#         # ⚠️ Mude aqui: Darts segue a MESMA lógica do Seq2 → diversidade por physics_strategy
#         return "physics_strategy" if "physics_strategy" in df_arch.columns else None
#     return None

    
# def plug_score_column_into_cfg(
#     df: pd.DataFrame,
#     cfg: "PostHocConfig",
#     *,
#     score_col: str,
#     ensure_in_metrics: bool = True,
#     default_quantile: float = 0.6,
#     default_lower_is_better: bool = True,
# ) -> "PostHocConfig":
#     """
#     Plug-and-play:
#       - garante que cfg.metric_to_optimize aponta para score_col
#       - garante score_col em cfg.metrics
#       - garante score_col em cfg.lower_is_better
#       - garante cfg.primary_quantile não vazio e inclui score_col (se fizer sentido)
#     """
#     # 1) metric_to_optimize
#     try:
#         cfg.metric_to_optimize = score_col
#     except Exception:
#         pass

#     # 2) lower_is_better
#     lib = getattr(cfg, "lower_is_better", None)
#     if not isinstance(lib, dict):
#         lib = {}
#     lib.setdefault(score_col, default_lower_is_better)
#     try:
#         cfg.lower_is_better = lib
#     except Exception:
#         pass

#     # 3) cfg.metrics
#     if ensure_in_metrics:
#         mets = list(getattr(cfg, "metrics", []) or [])
#         if score_col not in mets:
#             mets.append(score_col)
#         try:
#             cfg.metrics = mets
#         except Exception:
#             pass

#     # 4) primary_quantile
#     pq = dict(getattr(cfg, "primary_quantile", {}) or {})
#     if not pq:
#         pq = {score_col: float(default_quantile)}
#     else:
#         # garante pelo menos um filtro pelo score (se ainda não existir)
#         pq.setdefault(score_col, float(default_quantile))
#     try:
#         cfg.primary_quantile = pq
#     except Exception:
#         pass

#     return cfg


# def plug_robust_score(
#     df: pd.DataFrame,
#     cfg: "PostHocConfig",
#     *,
#     arch_col: str | None = None,
#     weights_by_arch: dict | None = None,
# ) -> tuple[pd.DataFrame, "PostHocConfig"]:
#     """
#     Legacy plug-in for robust_score.

#     Backward compatible:
#       - Ensures 'robust_score' exists (legacy meaning)
#       - Ensures semantic alias 'legacy_robust_score' exists
#       - Injects 'robust_score' into cfg as the decision column (unchanged behavior)
#     """
#     out = df.copy()

#     if "robust_score" not in out.columns:
#         out = add_robust_score(out, weights_by_arch=weights_by_arch, arch_col=arch_col)

#     # ✅ semantics alias (Patch 2)
#     if "legacy_robust_score" not in out.columns:
#         out["legacy_robust_score"] = out["robust_score"]

#     cfg = plug_score_column_into_cfg(
#         out, cfg,
#         score_col="robust_score",
#         ensure_in_metrics=True,
#         default_quantile=float((getattr(cfg, "primary_quantile", {}) or {}).get("robust_score", 0.6)),
#         default_lower_is_better=True,
#     )
#     return out, cfg




# # --- Stage 1: Champion Harvester ---
# def select_champions_for_campaign(
#     campaign_name: str,
#     leaderboard_root: Path,
#     selection_cfg_overrides: Optional[Dict[str, Any]] = None,
#     metric_weights: Optional[Dict[str, float]] = None,
#     lower_is_better: Optional[Dict[str, bool]] = None,
# ) -> pd.DataFrame:
#     """
#     Harvests champions for a single campaign using the proven selection pipeline.

#     Returns a tidy DataFrame with (at least) the columns:
#       ['campaign','dataset','well','architecture','job_hash',
#        'val_smape_agg','val_smape_cum','weighted_score','robust_score']

#     Behavior:
#       - loads all leaderboard.csv for the campaign (multi-cycle tolerant)
#       - computes weighted_score (always) and robust_score (if available)
#       - runs the robust distribution filter to pick champions
#       - parses/patches dataset/well if missing (from campaign_name)
#       - normalizes the architecture column to 'architecture'
#       - returns empty DataFrame if nothing is found/selected
#     """
#     # Defaults if caller doesn't pass overrides
#     _metric_weights = metric_weights or {"val_smape_agg": 1.0, "val_smape_cum": 0.25}
#     _lower_is_better = lower_is_better or {"val_smape_agg": True, "val_smape_cum": True}

#     # 1) Load leaderboards
#     df = load_master_leaderboard(campaign_name, leaderboard_root)
#     if df is None or df.empty:
#         logging.info("[harvest] No leaderboard entries found for campaign='%s'", campaign_name)
#         return pd.DataFrame(columns=[
#             "campaign","dataset","well","architecture","job_hash",
#             "val_smape_agg","val_smape_cum","weighted_score","robust_score"
#         ])

#     # 2) Score trials (weighted_score always; robust_score if functions/cols permit)
#     try:
#         df = add_weighted_score(df, metric_weights=_metric_weights, lower_is_better=_lower_is_better)
#     except Exception as e:
#         logging.warning("[harvest] add_weighted_score failed (%s); continuing without it.", e)

#     try:
#         df = add_robust_score(df)  # uses defaults inside; safe if extra features missing
#     except Exception as e:
#         logging.info("[harvest] add_robust_score skipped (%s); continuing.", e)

#     # 3) Run robust gating + selection
#     cfg = make_default_config(**(selection_cfg_overrides or {}))
#     run = run_distribution_filter(df, cfg)
#     champs = (run or {}).get("top_performers")
#     if champs is None or champs.empty:
#         logging.info("[harvest] No champions selected for campaign='%s'.", campaign_name)
#         return pd.DataFrame(columns=[
#             "campaign","dataset","well","architecture","job_hash",
#             "val_smape_agg","val_smape_cum","weighted_score","robust_score"
#         ])

#     # 4) Ensure campaign/dataset/well/architecture/job_hash are present & clean
#     champs = champs.copy()
#     champs["campaign"] = campaign_name

#     # normalize architecture column -> 'architecture'
#     arch_col = None
#     for cand in ("architecture", "architecture_name", "arch"):
#         if cand in champs.columns:
#             arch_col = cand
#             break
#     if arch_col is None:
#         # create a placeholder if truly missing
#         champs["architecture"] = "unknown"
#     elif arch_col != "architecture":
#         champs.rename(columns={arch_col: "architecture"}, inplace=True)

#     # parse dataset/well from campaign name if needed
#     need_dataset = "dataset" not in champs.columns or champs["dataset"].isna().all()
#     need_well = "well" not in champs.columns or champs["well"].isna().all()
#     if need_dataset or need_well:
#         parsed = _parse_campaign_name(campaign_name)  # existing helper
#         if "dataset" not in champs.columns:
#             champs["dataset"] = parsed.get("dataset", "unknown")
#         champs["dataset"] = champs["dataset"].fillna(parsed.get("dataset", "unknown"))
#         if "well" not in champs.columns:
#             champs["well"] = parsed.get("well", "unknown")
#         champs["well"] = champs["well"].fillna(parsed.get("well", "unknown"))

#     # locate job_hash (fallbacks just in case)
#     if "job_hash" not in champs.columns:
#         for alt in ("job", "hash", "job_id", "trial_hash"):
#             if alt in champs.columns:
#                 champs["job_hash"] = champs[alt]
#                 break
#     if "job_hash" not in champs.columns:
#         # fail-safe: produce empty result with correct schema
#         logging.warning("[harvest] No job_hash column found after fallbacks; returning empty.")
#         return pd.DataFrame(columns=[
#             "campaign","dataset","well","architecture","job_hash",
#             "val_smape_agg","val_smape_cum","weighted_score","robust_score"
#         ])

#     # 5) Build minimal, essential schema (keep what exists)
#     wanted = [
#         "campaign","dataset","well","architecture","job_hash",
#         "val_smape_agg","val_smape_cum","weighted_score","robust_score",
#     ]
#     present = [c for c in wanted if c in champs.columns]
#     out = champs[present].drop_duplicates().reset_index(drop=True)

#     # Sort for determinism
#     sort_cols = [c for c in ["well","architecture","val_smape_agg","val_smape_cum","weighted_score","robust_score"] if c in out.columns]
#     out = out.sort_values(sort_cols, ascending=True).reset_index(drop=True)

#     return out



# -----------------------------------------------------------------------------
# select_champions_from_df (refatorado p/ logs + compact block; lógica inalterada)
# -----------------------------------------------------------------------------
# def select_champions_from_df(
#     master_df: pd.DataFrame,
#     selection_cfg_overrides: Optional[Dict[str, Any]] = None,
#     metric_weights: Optional[Dict[str, float]] = None,
#     lower_is_better: Optional[Dict[str, bool]] = None,
#     scoring_strategy: str = "weighted_score",
#     ensure_topk_per_group: bool = True,
#     enable_familywise_filter: bool = True,
#     enable_backfill_from_family: bool = True,
# ) -> pd.DataFrame:
#     """
#     Champion selector with strict K control (unchanged logic).
#     Compact logging mode collapses per-well/family chatter into a summary block.
#     """
#     import os
#     import logging
#     import numpy as np
#     import pandas as pd

#     # --- logging facade (uses your existing utils) ---------------------------
#     from common.log_utils import (
#         info, warn, ok, err,
#         log_block, summarize_stage1,
#         is_compact_logging, effective_log_width,
#         vlog, silence_loggers, parse_silenced_from_env
#     )

#     log = logging.getLogger("champion_select")
#     COMPACT = is_compact_logging(None)  # env/CFG-driven; no cfg object here
#     WIDTH = effective_log_width(None, fallback=100)

#     # additional opt-in silencers via env
#     extra_silencers = parse_silenced_from_env()  # PHASE_SILENCE_LOGGERS
#     # default silencers when in compact mode (safe guesses; ignore if absent)
#     default_noisy = ["hpo.distribution_filter", "distribution_filter", "robust_filter"]
#     silent_names = (default_noisy + extra_silencers) if COMPACT else extra_silencers

#     # ---- input checks -------------------------------------------------------
#     if master_df is None or master_df.empty:
#         warn("[select] Input DataFrame is empty.")
#         return pd.DataFrame()

#     df0 = master_df.copy()
#     has_well = "well" in df0.columns
#     has_job  = "job_hash" in df0.columns

#     if not has_well:
#         warn("[select] Column 'well' missing; selection may misbehave.")
#     if not has_job:
#         warn("[select] Column 'job_hash' missing; Top-K clamp/backfill may misbehave.")

#     # leaderboard lines can be very chatty; keep them only in verbose mode
#     if "campaign" in df0.columns and has_well and not COMPACT:
#         if has_job:
#             by_cw = df0.groupby(["campaign","well"], as_index=False)["job_hash"].nunique()
#             for _, r in by_cw.iterrows():
#                 info("✅ Loaded leaderboard for campaign '%s' | well=%s with %d trials.",
#                      r["campaign"], r["well"], int(r["job_hash"]))
#         else:
#             by_cw = df0.groupby(["campaign","well"], as_index=False).size().rename(columns={"size":"n"})
#             for _, r in by_cw.iterrows():
#                 info("✅ Loaded leaderboard for campaign '%s' | well=%s with %d trials.",
#                      r["campaign"], r["well"], int(r["n"]))

#     # ---- 1) score once ------------------------------------------------------
#     df_scored = df0.copy()
#     if scoring_strategy == "weighted_score":
#         eff_weights = metric_weights or {"val_smape_agg": 1.0}
#         eff_dir = {"val_smape_agg": True, "weighted_score": True}
#         if isinstance(lower_is_better, dict):
#             eff_dir.update(lower_is_better)
#         df_scored = add_weighted_score(df_scored, metric_weights=eff_weights, lower_is_better=eff_dir)
#         score_col = "weighted_score"
#     elif scoring_strategy == "robust_score":
#         if "rank_gap_norm" not in df_scored.columns:
#             df_scored["rank_gap_norm"] = compute_rank_gap(df_scored)
#         if "neighbor_iqr_cum_norm" not in df_scored.columns:
#             df_scored["neighbor_iqr_cum_norm"] = compute_neighbor_iqr(df_scored)
#         df_scored = add_robust_score(df_scored)
#         score_col = "robust_score"
#     else:
#         raise ValueError(f"Unknown scoring_strategy: {scoring_strategy}")

#     # ---- 2) build PostHoc config & wire K everywhere -----------------------
#     cfg = make_default_config(**(selection_cfg_overrides or {}))

#     # ensure metric_to_optimize and direction for the score (unchanged)
#     try:
#         if not getattr(cfg, "metric_to_optimize", None):
#             setattr(cfg, "metric_to_optimize", score_col)
#     except Exception:
#         pass
#     lib = getattr(cfg, "lower_is_better", None)
#     if not isinstance(lib, dict):
#         lib = {}
#     lib.setdefault(score_col, True)
#     try:
#         setattr(cfg, "lower_is_better", lib)
#     except Exception:
#         pass
#     asc = bool((getattr(cfg, "lower_is_better", {}) or {}).get(score_col, True))

#     def _pos(x):
#         try:
#             x = int(x)
#             return x if x > 0 else None
#         except Exception:
#             return None

#     K_override = _pos((selection_cfg_overrides or {}).get("top_strategies_per_well") if isinstance(selection_cfg_overrides, dict) else None)
#     K_cfg1 = _pos(getattr(cfg, "top_strategies_per_well", None))
#     K_cfg2 = _pos(getattr(cfg, "top_k_per_well", None))  # legacy
#     K = next((v for v in (K_override, K_cfg1, K_cfg2) if v is not None), None)
#     if K is not None:
#         for attr in ("top_strategies_per_well", "top_n_strategies", "top_k_per_well"):
#             try: setattr(cfg, attr, int(K))
#             except Exception: pass
#         try:
#             pk = getattr(cfg, "per_strategy_k", None)
#             if pk is None or int(pk) < 1:
#                 setattr(cfg, "per_strategy_k", int(K))
#             else:
#                 setattr(cfg, "per_strategy_k", max(int(pk), 1))
#         except Exception:
#             pass
#         try:
#             if getattr(cfg, "min_arch_diversity", None) is not None:
#                 setattr(cfg, "min_arch_diversity", 0)
#         except Exception:
#             pass

#         info("[select] K wired: K=%s → top_strategies_per_well=%s, top_n_strategies=%s, top_k_per_well=%s, per_strategy_k=%s, min_arch_diversity=%s",
#              K,
#              getattr(cfg, "top_strategies_per_well", None),
#              getattr(cfg, "top_n_strategies", None),
#              getattr(cfg, "top_k_per_well", None),
#              getattr(cfg, "per_strategy_k", None),
#              getattr(cfg, "min_arch_diversity", None),
#              )
#     else:
#         info("[select] No K resolved; selector will not clamp internally.")

#     # ---- 3) family helpers --------------------------------------------------
#     def detect_family_whole(df: pd.DataFrame) -> str:
#         cols = set(df.columns)
#         if {"variant","solver","weighting","loss"}.issubset(cols):           return "arps"
#         if {"profile","n_epochs"}.issubset(cols):                            return "darts"
#         if {"physics_strategy","epochs"}.issubset(cols):                     return "seq2"
#         return "generic"

#     def split_by_family(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
#         fam_all = detect_family_whole(df)
#         if fam_all != "generic":
#             return {fam_all: df}
#         fam_col = next((c for c in ["arch","architecture","architecture_name","family"] if c in df.columns), None)
#         if fam_col is None:
#             return {"generic": df}
#         s = df[fam_col].astype(str).str.lower()
#         fams = {
#             "arps":  df[s.str.contains("arps",  na=False)],
#             "seq2":  df[(s.str.contains("seq2", na=False) | s.str.contains("pinn", na=False)) & ~s.str.contains("arps", na=False)],
#             "darts": df[s.str.contains("darts", na=False) & ~s.str.contains("arps", na=False) & ~(s.str.contains("seq2", na=False) | s.str.contains("pinn", na=False))],
#         }
#         leftovers = df[~(s.str.contains("arps", na=False) |
#                          s.str.contains("darts", na=False) |
#                          s.str.contains("seq2", na=False) |
#                          s.str.contains("pinn", na=False))]
#         out = {k:v for k,v in fams.items() if not v.empty}
#         if not leftovers.empty: out["generic"] = leftovers
#         return out

#     def log_pool_stats(df: pd.DataFrame, tag: str):
#         """
#         Loga um resumo do pool:
#           - total de linhas
#           - nº de wells
#           - preview por (well[, arch]) com contagem de job_hash.
#         Em modo compact, mantém o log em uma linha só.
#         """
#         if df is None or df.empty:
#             info("----- %s ----- total_rows=0 | wells=0", tag)
#             return

#         total = len(df)
#         uniq_w = df["well"].nunique() if "well" in df.columns else "-"

#         # constrói um preview por (well, arch) se as colunas existirem
#         preview = ""
#         if has_well and has_job:
#             df_tmp = df.copy()
#             grp_cols = ["well"]
#             if "arch" in df_tmp.columns:
#                 grp_cols.append("arch")
#             elif "architecture" in df_tmp.columns:
#                 grp_cols.append("architecture")
#             elif "architecture_name" in df_tmp.columns:
#                 grp_cols.append("architecture_name")

#             try:
#                 cnt = (
#                     df_tmp.groupby(grp_cols)["job_hash"]
#                           .nunique()
#                           .reset_index()
#                 )
#                 rows = []
#                 for _, r in cnt.head(8).iterrows():
#                     if len(grp_cols) == 2:
#                         rows.append(f"{r['well']}/{str(r[grp_cols[1]])}={int(r['job_hash'])}")
#                     else:
#                         rows.append(f"{r['well']}={int(r['job_hash'])}")
#                 preview = " | " + ", ".join(rows) if rows else ""
#             except Exception:
#                 preview = ""

#         if COMPACT:
#             info("----- %s ----- total_rows=%s | wells=%s%s",
#                  tag, f"{total:,}", uniq_w, preview)
#             return

#         # modo verboso (já existia, mantido)
#         if not has_well:
#             info("[select] %s: total rows=%d", tag, total)
#             return

#         if has_job:
#             cnt_w = df.groupby("well")["job_hash"].nunique()
#         else:
#             cnt_w = df.groupby("well").size()

#         info("----- %s -----", tag)
#         for w, n in cnt_w.items():
#             info("well=%s → %d trials", w, int(n))


#     # ---- 4) family×well filtering with hard K from SURVIVORS ---------------
#     log_pool_stats(df_scored, "POOL BEFORE FILTER")

#     pools = split_by_family(df_scored) if enable_familywise_filter else {"all": df_scored}
#     picked_blocks: list[pd.DataFrame] = []

#     # compact-mode: silence noisy third-party loggers while filtering
#     with silence_loggers(silent_names, level=logging.WARNING):
#         for fam, fam_df in pools.items():
#             if fam_df.empty:
#                 continue
#             iterator = fam_df.groupby("well") if has_well else [("all", fam_df)]
#             for well_name, g in iterator:
#                 try:
#                     res = run_distribution_filter(g, cfg)
#                 except Exception as e:
#                     warn("[select] Filtering failed for fam=%s well=%s: %s", fam, well_name, e)
#                     continue

#                 survivors = res.get("survivors")
#                 gate_counts = res.get("gate_counts")

#                 # diagnostics → only in verbose mode
#                 if not COMPACT:
#                     if gate_counts is not None and not getattr(gate_counts, "empty", True):
#                         row = gate_counts.iloc[0]
#                         info("[survivors][fam=%s] well=%s: kept_by_gates=%s/%s",
#                              fam, row.get(getattr(cfg, "well_col", "well"), well_name),
#                              int(row.get("kept_by_gates", 0)), int(row.get("total_in_pool", 0)))
#                     elif survivors is not None:
#                         info("[survivors][fam=%s] well=%s: kept_by_gates=%d/%d", fam, well_name, len(survivors), len(g))

#                 # Hard decision: ignore any 'top_performers' returned upstream.
#                 if survivors is None or getattr(survivors, "empty", True):
#                     if not COMPACT:
#                         info("[final][fam=%s] well=%s: picked_final=0/%d (from survivors), total=0/%d",
#                              fam, well_name, 0, len(g))
#                     continue

#                 take_K = K if K is not None else len(survivors)
#                 use_col = score_col if score_col in survivors.columns else "val_smape_agg"
#                 picked = survivors.sort_values(use_col, ascending=asc).head(take_K).copy()

#                 if not COMPACT:
#                     info("[final][fam=%s] well=%s: picked_final=%d/%d (from survivors), total=%d/%d",
#                          fam, well_name, len(picked), len(survivors), len(picked), len(g))

#                 picked_blocks.append(picked)

#     champions = pd.concat(picked_blocks, ignore_index=True) if picked_blocks else pd.DataFrame()
#     if champions.empty:
#         warn("[select] No champions after family×well filtering.")
#         return champions

#     log_pool_stats(champions, "POOL AFTER FILTER")

#     # ---- 5) Optional Top-K per (well, arch) clamp ---------------------------
#     if not ensure_topk_per_group:
#         info("[select] Final clamp disabled (ensure_topk_per_group=False).")
#         return champions.copy()

#     well_col = "well" if "well" in champions.columns else None
#     arch_col = next((c for c in ["arch","architecture","architecture_name","family"] if c in champions.columns), None)
#     if well_col is None:
#         warn("[select] 'well' column missing; skipping Top-K clamp.")
#         return champions.copy()
#     if arch_col is None:
#         arch_col = "_arch_family"
#         champions = champions.copy()
#         champions[arch_col] = "generic"

#     # resolve K again (same precedence)
#     K2 = K
#     if K2 is None:
#         K2 = next((v for v in (
#             _pos(getattr(cfg, "top_strategies_per_well", None)),
#             _pos(getattr(cfg, "top_n_strategies", None)),
#             _pos(getattr(cfg, "top_k_per_well", None)),
#         ) if v is not None), None)

#     champs_sorted = champions.sort_values([well_col, arch_col, score_col], ascending=[True, True, asc])
#     if K2 is None:
#         info("[select] No Top-K clamp applied at the end (no K resolved).")
#         out = champs_sorted.copy()
#     else:
#         out = (
#             champs_sorted
#             .groupby([well_col, arch_col], as_index=False, sort=False, group_keys=False)
#             .head(int(K2))
#             .copy()
#         )

#         # ---- 6) Optional backfill from the same family ----------------------
#         if enable_backfill_from_family and K2 is not None:
#             picked_hashes = set(out["job_hash"].astype(str)) if "job_hash" in out.columns else set()
#             pool_arch_col = arch_col if arch_col in df_scored.columns else (
#                 next((c for c in ["arch","architecture","architecture_name","family"] if c in df_scored.columns), "_arch_family")
#             )
#             if pool_arch_col == "_arch_family" and "_arch_family" not in df_scored.columns:
#                 s = df_scored.get("architecture_name", df_scored.get("arch", "")).astype(str).str.lower()
#                 fam_guess = np.where(s.str.contains("arps", na=False), "arps",
#                               np.where(s.str.contains("darts", na=False), "darts",
#                               np.where(s.str.contains("seq2", na=False) | s.str.contains("pinn", na=False), "seq2", "generic")))
#                 df_scored["_arch_family"] = fam_guess
#                 pool_arch_col = "_arch_family"

#             pool_sorted = df_scored.sort_values([well_col, pool_arch_col, score_col], ascending=[True, True, asc])
#             needed = []
#             for (w, a), gpool in pool_sorted.groupby([well_col, pool_arch_col]):
#                 have = out[(out[well_col] == w) & (out[arch_col] == a)]
#                 missing = int(K2) - len(have)
#                 if missing <= 0:
#                     continue
#                 cand = gpool if "job_hash" not in gpool.columns else gpool[~gpool["job_hash"].astype(str).isin(picked_hashes)]
#                 take = cand.head(missing)
#                 if not take.empty:
#                     take = take.reindex(columns=out.columns, fill_value=np.nan)
#                     needed.append(take)
#                     if "job_hash" in take.columns:
#                         picked_hashes.update(take["job_hash"].astype(str))
#             if needed:
#                 out = pd.concat([out] + needed, ignore_index=True)

#         try:
#             n_wells = out[well_col].nunique()
#             n_archs = out[arch_col].nunique()
#             info("[select] Enforced Top-%d per (well, arch): %d wells × %d archs → %d rows.",
#                  int(K2), n_wells, n_archs, len(out))
#         except Exception:
#             pass

#     # ---- Compact summary block ---------------------------------------------
#     if COMPACT:
#         lines = summarize_stage1(champions_df=out, posthoc={
#             "top_strategies_per_well": K if K is not None else getattr(cfg, "top_strategies_per_well", None),
#             "selection_strategy": getattr(cfg, "selection_strategy", None),
#             "valcum_gate": getattr(cfg, "valcum_gate", {}),
#         }, score_col=score_col)
#         log_block("Stage 1 — Robust Champion Harvester (Summary)", lines, level=logging.INFO, width=WIDTH)

#     return out


