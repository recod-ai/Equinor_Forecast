"""
hpo.posthoc_selection

Champion selection utilities for the post-hoc HPO pipeline.

This module is responsible for turning a gated candidate pool into final
per-well champions. It parses PostHocConfig into a SelectionConfig, applies
(optional) pool relaxation levels, performs stable sorting with deterministic
tie-breaks, and selects exemplars per strategy/architecture without touching
any TEST-only audit columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .posthoc_config import PostHocConfig


@dataclass
class SelectionConfig:
    """Holds all parsed configuration for the champion selection process."""
    well_col: str
    arch_col: Optional[str]
    strat_col: str
    score_col: str
    lower_is_better: bool
    apply_pareto: bool
    top_n_strategies: int
    per_strategy_k: int
    selection_strategy: str
    min_samples_per_well: int
    arch_filter: Optional[List[str]] = None
    id_cols: List[str] = field(default_factory=list)
    relax_pool: bool = True

    # ✅ Needed because _parse_config already passes these
    pool_method: str = "survivors"
    pool_cfg: Dict[str, Any] = field(default_factory=dict)


# --- replace this whole function ---
def _filter_pool_with_relaxation(pool: pd.DataFrame, config: SelectionConfig, *, allow_relax: bool = False) -> pd.DataFrame:
    """
    Aplica filtros em níveis:
      L0: passes_gates & valcum_pass & (pareto se ativo)
      L1: somente passes_gates
      L2: somente valcum_pass
      L3: sem nenhum dos dois

    Retorna o primeiro nível não-vazio e marca __gate_level__ para diagnóstico.
    """
    import pandas as pd

    if pool is None or pool.empty:
        return pool

    def _mark(df: pd.DataFrame, level_name: str) -> pd.DataFrame:
        out = df.copy()
        if not out.empty:
            out["__gate_level__"] = level_name
        return out

    def level_df(df: pd.DataFrame, level_name: str) -> pd.DataFrame:
        out = df

        if level_name in ("L0", "L1"):
            if "passes_gates" in out.columns:
                out = out[out["passes_gates"].astype(bool)]

        if level_name in ("L0", "L2"):
            if "valcum_pass" in out.columns:
                out = out[out["valcum_pass"].astype(bool)]

        if level_name == "L0" and bool(getattr(config, "apply_pareto", False)):
            if "is_pareto" in out.columns:
                out = out[out["is_pareto"].astype(bool)]

        return _mark(out, level_name)

    l0 = level_df(pool, "L0")
    if not allow_relax:
        return l0

    if not l0.empty:
        return l0

    l1 = level_df(pool, "L1")
    if not l1.empty:
        return l1

    l2 = level_df(pool, "L2")
    if not l2.empty:
        return l2

    return _mark(pool, "L3")




def _parse_config(df: pd.DataFrame, cfg: "PostHocConfig") -> SelectionConfig:
    """
    Parses the raw config object into a structured SelectionConfig.

    IMPORTANT:
    - `score_col` is the DECISION/ORDERING column.
    - Direction MUST be resolved by a single source of truth:
        resolve_ordering(score_col, lower_is_better=cfg.lower_is_better, default_ascending=True)
      This avoids "silent inversions" AND avoids hardcoding assumptions that may
      not match how the score was constructed (e.g., your weighted_score is lower-is-better).
    """
    import logging
    from hpo.score_ordering import resolve_ordering  # local import to avoid circular issues

    well_col = getattr(cfg, "well_col", "well")
    strat_col = getattr(cfg, "strategy_col", "physics_strategy")

    arch_col = getattr(cfg, "arch_col", None)
    if arch_col not in df.columns:
        for cand in ("architecture", "architecture_name", "arch"):
            if cand in df.columns:
                arch_col = cand
                break
        else:
            arch_col = None

    # -----------------------------------------------------------
    # Explicit: selection_col = decision/ordering column
    # Priority: cfg.selection_col > cfg.metric_to_optimize
    # -----------------------------------------------------------
    requested_selection_col = getattr(cfg, "selection_col", None) or getattr(cfg, "metric_to_optimize", "weighted_score")

    # Deterministic fallback if missing
    if requested_selection_col not in df.columns:
        fallback_candidates = [
            "robust_score",
            "weighted_score",
            "val_smape_agg",
            "val_smape_cum",
        ]
        fallback = next((c for c in fallback_candidates if c in df.columns), None)
        if fallback is None:
            raise ValueError(
                f"[selection_cfg] Requested selection_col='{requested_selection_col}' not found and no fallback is available. "
                f"Available cols sample={list(df.columns)[:40]}"
            )

        logging.warning(
            "[selection_cfg] selection_col='%s' not found. Falling back to '%s'. "
            "Fix config wiring: set PostHocConfig.selection_col (preferred) or metric_to_optimize (legacy).",
            requested_selection_col, fallback
        )
        score_col = fallback
    else:
        score_col = requested_selection_col

    # ---------------------------------------------------------------------
    # Resolve direction via single source of truth (no hardcoded overrides)
    # ---------------------------------------------------------------------
    lib_isb = getattr(cfg, "lower_is_better", {}) or {}
    ordering = resolve_ordering(score_col, lower_is_better=lib_isb, default_ascending=True)
    isb = ordering.ascending  # True => lower is better

    top_n = getattr(cfg, "top_strategies_per_well", getattr(cfg, "top_k_per_well", 1))

    arch_filter = getattr(cfg, "arch_filter", None)
    if arch_filter is not None and not isinstance(arch_filter, (list, tuple, set)):
        arch_filter = [arch_filter]

    pool_method = str(getattr(cfg, "pool_method", "survivors")).lower()

    # Auditable log
    logging.info(
        "[selection_cfg] selection_col=%s (requested=%s) lower_is_better=%s higher_is_better=%s reason=%s "
        "strat_col=%s arch_col=%s top_n=%s per_k=%s pool=%s",
        score_col,
        requested_selection_col,
        bool(isb),
        bool(not isb),
        ordering.reason,
        strat_col,
        arch_col,
        int(top_n),
        int(getattr(cfg, "per_strategy_k", 1)),
        pool_method,
    )

    return SelectionConfig(
        well_col=well_col,
        arch_col=arch_col,
        strat_col=strat_col,
        score_col=score_col,
        lower_is_better=bool(isb),
        apply_pareto=bool(getattr(cfg, "apply_pareto", False)),
        top_n_strategies=int(top_n),
        per_strategy_k=int(getattr(cfg, "per_strategy_k", 1)),
        selection_strategy=getattr(cfg, "selection_strategy", "best_of_the_best"),
        min_samples_per_well=int(getattr(cfg, "min_samples_per_well", 0)),
        arch_filter=arch_filter,
        id_cols=list(getattr(cfg, "id_cols", [])),
        relax_pool=bool(getattr(cfg, "relax_pool", True)),
        pool_method=pool_method,
        pool_cfg=dict(getattr(cfg, "pool_cfg", {}) or {}),
    )




def _pick_k_exemplars(g_sorted: pd.DataFrame, config: SelectionConfig) -> pd.DataFrame:
    """Picks the top K rows based on the selection strategy, returning a safe copy."""
    if config.selection_strategy == "median_of_the_best":
        n = len(g_sorted)
        take = min(config.per_strategy_k, n)
        start = max(0, (n // 2) - (take // 2))
        return g_sorted.iloc[start:start + take].copy() # Cópia explícita
    else: # "best_of_the_best"
        return g_sorted.head(config.per_strategy_k).copy() # Cópia explícita

def _finalize_champion_df(df: pd.DataFrame, config: SelectionConfig) -> pd.DataFrame:
    """Sorts, cleans, and validates the final DataFrame of champions."""
    if df.empty:
        return df
    
    sort_cols = [c for c in (config.well_col, config.arch_col, config.strat_col, config.score_col) if c and c in df.columns]
    sort_asc = [config.lower_is_better if c == config.score_col else True for c in sort_cols]
    out = df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
    out = out.drop(columns=["__rep_score__", "__source__"], errors="ignore")
    return out


def select_champions_by_strategy(df: pd.DataFrame, cfg: "PostHocConfig") -> pd.DataFrame:
    import logging
    import warnings
    from typing import List

    import pandas as pd

    from hpo.score_ordering import resolve_ordering
    from hpo.sort_utils import infer_tiebreak_cols, stable_sort

    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    config = _parse_config(df, cfg)
    df_prepared = df.copy()

    # selection_col is the ORDERING column (single source of truth)
    selection_col = (
        getattr(config, "score_col", None)
        or getattr(cfg, "selection_col", None)
        or getattr(cfg, "metric_to_optimize", "weighted_score")
    )

    if selection_col not in df_prepared.columns:
        fallback_candidates = ["val_smape_agg", "val_smape_cum", "weighted_score", "robust_score"]
        fallback = next((c for c in fallback_candidates if c in df_prepared.columns), None)
        if fallback is None:
            raise KeyError(
                f"[selection] selection_col='{selection_col}' not found and no fallback metrics available. "
                f"Available cols sample={list(df_prepared.columns)[:30]}"
            )
        logging.warning(
            "[selection] selection_col='%s' missing. Falling back to '%s'.", selection_col, fallback
        )
        selection_col = fallback
        try:
            config.score_col = fallback
        except Exception:
            pass

    lib_isb = dict(getattr(cfg, "lower_is_better", {}) or {})
    ordering = resolve_ordering(selection_col, lower_is_better=lib_isb, default_ascending=True)

    tiebreak_cols = infer_tiebreak_cols(df_prepared)

    logging.info(
        "[selection] selection_col=%s | ascending=%s | higher_is_better=%s | reason=%s | "
        "selection_strategy=%s | top_n=%s | per_strategy_k=%s | relax_pool=%s",
        selection_col,
        ordering.ascending,
        ordering.higher_is_better,
        ordering.reason,
        getattr(config, "selection_strategy", "unknown"),
        getattr(config, "top_n_strategies", None),
        getattr(config, "per_strategy_k", None),
        getattr(config, "relax_pool", None),
    )

    # Optional arch filter
    if getattr(config, "arch_filter", None) and getattr(config, "arch_col", None):
        df_prepared = df_prepared[df_prepared[config.arch_col].isin(config.arch_filter)]
        if df_prepared.empty:
            warnings.warn("No rows left after applying arch_filter.")
            return df_prepared

    final_blocks: List[pd.DataFrame] = []
    processed_indices = set()

    # ---------------------------
    # (A) DARTS path
    # ---------------------------
    if getattr(config, "arch_col", None) and "physics_strategy" in df_prepared.columns:
        darts_mask = df_prepared[config.arch_col].astype(str).str.contains("Darts", case=False, na=False)
        df_darts = df_prepared[darts_mask]
        if not df_darts.empty:
            for _, g in df_darts.groupby(config.well_col, sort=False, dropna=False):
                pool = _filter_pool_with_relaxation(g, config, allow_relax=bool(getattr(config, "relax_pool", False)))
                if pool.empty:
                    continue

                best_per_ps = (
                    stable_sort(pool, selection_col, ordering, tiebreak_cols=tiebreak_cols)
                    .groupby("physics_strategy", sort=False, dropna=False)
                    .head(int(getattr(config, "per_strategy_k", 1)))
                )
                picked = stable_sort(best_per_ps, selection_col, ordering, tiebreak_cols=tiebreak_cols).head(
                    int(getattr(config, "top_n_strategies", 1))
                )

                if not picked.empty:
                    picked = picked.drop_duplicates(subset=[config.well_col, "physics_strategy"], keep="first")
                    final_blocks.append(picked)
                    processed_indices.update(picked.index)

            processed_indices.update(df_darts.index)

    # ----------------------------------------
    # (B) Other architectures
    # ----------------------------------------
    df_rest = df_prepared.drop(index=list(processed_indices)) if processed_indices else df_prepared
    if getattr(config, "arch_col", None) and not df_rest.empty:
        for _, df_arch in df_rest.groupby(config.arch_col, sort=False, dropna=False):
            diversity_col = None
            try:
                diversity_col = diversity_col_for_arch(df_arch)
            except Exception:
                diversity_col = None

            for _, group_df in df_arch.groupby([config.well_col, config.arch_col], sort=False, dropna=False):
                pool = _filter_pool_with_relaxation(
                    group_df, config, allow_relax=bool(getattr(config, "relax_pool", False))
                )
                if pool.empty:
                    continue

                if diversity_col and diversity_col in pool.columns:
                    best_per_div = (
                        stable_sort(pool, selection_col, ordering, tiebreak_cols=tiebreak_cols)
                        .groupby(diversity_col, sort=False, dropna=False)
                        .head(int(getattr(config, "per_strategy_k", 1)))
                    )
                    picked = stable_sort(best_per_div, selection_col, ordering, tiebreak_cols=tiebreak_cols).head(
                        int(getattr(config, "top_n_strategies", 1))
                    )
                else:
                    sorted_pool = stable_sort(pool, selection_col, ordering, tiebreak_cols=tiebreak_cols)

                    if getattr(config, "selection_strategy", "") == "median_of_the_best":
                        n = min(int(getattr(config, "top_n_strategies", 1)), len(sorted_pool))
                        if n <= 0:
                            picked = sorted_pool.head(0)
                        else:
                            start = max(0, (len(sorted_pool) // 2) - (n // 2))
                            picked = sorted_pool.iloc[start : start + n].copy()
                    else:
                        picked = sorted_pool.head(int(getattr(config, "top_n_strategies", 1))).copy()

                if not picked.empty:
                    final_blocks.append(picked)
                    processed_indices.update(picked.index)

    if not final_blocks:
        return pd.DataFrame()

    champions_df = pd.concat(final_blocks, ignore_index=True)
    return _finalize_champion_df(champions_df, config)

def diversity_col_for_arch(df_arch: pd.DataFrame) -> Optional[str]:
    """
    Informa qual coluna representa a diversidade interna para um DataFrame de uma arquitetura.
    """
    if df_arch.empty:
        return None
    
    arch_col = _find_arch_col(df_arch)
    arch_name = df_arch[arch_col].iloc[0] if not df_arch[arch_col].empty else ""

    if "Seq2" in arch_name:
        return "physics_strategy" if "physics_strategy" in df_arch.columns else None
    elif "Arps" in arch_name:
        return "variant" if "variant" in df_arch.columns else None
    elif "Darts" in arch_name:
        # ⚠️ Mude aqui: Darts segue a MESMA lógica do Seq2 → diversidade por physics_strategy
        return "physics_strategy" if "physics_strategy" in df_arch.columns else None
    return None

def _find_arch_col(df: pd.DataFrame, arch_col: Optional[str] = None) -> str:
    """Helper para encontrar a coluna de arquitetura de forma robusta."""
    if arch_col and arch_col in df.columns:
        return arch_col
    for cand in ("architecture", "architecture_name", "arch"):
        if cand in df.columns:
            return cand
    raise ValueError("Nenhuma coluna de arquitetura encontrada em: ['architecture', 'architecture_name', 'arch']")