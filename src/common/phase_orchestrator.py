# src/common/phase_orchestrator.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable  
from contextlib import contextmanager

import logging
import pandas as pd
import numpy as np
from copy import deepcopy

from common.log_utils import (
    stage_banner, ok, warn, info, err,
    log_block, summarize_preflight, summarize_stage1, summarize_stage2, summarize_stage25,
    is_compact_logging, effective_log_width, install_compact_filters,
)
from common.phase_viz_support import _resolve_paths, _discover_and_load_leaderboards, normalize_well_key


# ---------------------------
# Imports that rely on sys.path
# ---------------------------
def _import_runtime_deps():
    # Logging configured upstream in your env; keep imports minimal and non-intrusive
    from forecast_pipeline.io_utils import configure_logging  # noqa: F401
    # Selection & scoring
    
    # New locations (post-refactor) — adjust paths ONLY if your repo names differ
    from hpo.posthoc_config import make_default_config  # noqa: F401
    
    # Selection entrypoint (legacy+neighbor dispatcher / public API)
    from hpo.analysis import select_champions_from_df   # noqa: F401
    
    # Score builders
    from hpo.posthoc_filtering import add_weighted_score, add_robust_score, compute_rank_gap, compute_neighbor_iqr
    
    # Series
    from common.series_store_reader import read_series_by_champions  # noqa: F401
    # Ensemble ops
    from hpo.ensemble_ops import build_intra_family_ensemble, build_inter_family_ensemble  # noqa: F401
    # Plotting
    from plotting.ensemble_plots import plot_conjugated_ensemble  # noqa: F401
    # Viz helpers
    from common.phase_viz_support import (  # noqa: F401
        build_full_history, infer_boundaries
    )
    return dict(
        read_series_by_champions=read_series_by_champions,
        build_intra_family_ensemble=build_intra_family_ensemble,
        build_inter_family_ensemble=build_inter_family_ensemble,
        plot_conjugated_ensemble=plot_conjugated_ensemble,
        select_champions_from_df=select_champions_from_df,
        make_default_config=make_default_config,
        add_weighted_score=add_weighted_score, add_robust_score=add_robust_score,
        compute_rank_gap=compute_rank_gap, compute_neighbor_iqr=compute_neighbor_iqr,
        stage_banner=stage_banner, ok=ok, warn=warn, info=info, err=err,
        configure_logging=configure_logging,
    )


@contextmanager
def _quiet_existing_info_logs(enabled: bool):
    """
    Temporarily raises all currently-registered loggers to WARNING,
    so noisy downstream INFO chatter does not leak in compact mode.
    """
    if not enabled:
        yield
        return

    manager = logging.root.manager
    loggers: list[logging.Logger] = [logging.getLogger()]
    seen = {id(loggers[0])}

    for name, obj in manager.loggerDict.items():
        if isinstance(obj, logging.Logger):
            lg = logging.getLogger(name)
            if id(lg) not in seen:
                loggers.append(lg)
                seen.add(id(lg))

    saved_levels = {lg: lg.level for lg in loggers}

    try:
        for lg in loggers:
            if lg.getEffectiveLevel() < logging.WARNING:
                lg.setLevel(logging.WARNING)
        yield
    finally:
        for lg, level in saved_levels.items():
            lg.setLevel(level)


@dataclass
class Phase4Config:
    scoring_strategy: str = "weighted_score"
    campaigns_to_ensemble: Dict[str, str] = field(
        default_factory=lambda: {"T_current": "HPO_210_Lag_100_Horizon_300"}
    )
    n_champions_per_group: int = 2
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {"val_smape_agg": 5.0, "val_smape_cum": 1.0}
    )
    lower_is_better: Dict[str, bool] = field(
        default_factory=lambda: {
            "val_smape_agg": True,
            "val_smape_cum": True,
            "weighted_score": True,
            "robust_score": True,
        }
    )
    posthoc_overrides: Dict[str, Any] = field(
        default_factory=lambda: dict(
            top_strategies_per_well=2,
            per_strategy_k=1,
            selection_strategy="best_of_the_best",
            apply_pareto=False,
            mad_guard={"enabled": True, "alpha": 0.5},
            valcum_gate={"q_low": 0.1, "q_high": 0.9},
        )
    )
    wells_to_analyze: List[str] = field(
        default_factory=lambda: ["P11", "P12", "P13", "P14", "P15", "P16", "15/9-F-12", "15/9-F-14"]
    )

    project_root: Optional[Path] = None
    src_marker: str = "src"
    series_store_root: Optional[Path] = None

    enable_plots: bool = True
    palette: str = "metallic_azure"
    show_family_traces: bool = False
    show_champion_traces: bool = False

    # logging
    log_mode: str = "compact"   # "compact" | "verbose"
    log_width: int = 100

    # NEW: compact hygiene
    suppress_internal_info_in_compact: bool = True
    show_preflight_population_scan: bool = False
    show_stage1_table_preview: bool = False

    enable_risk_plots: bool = True
    risk_splits: List[str] = field(default_factory=lambda: ["val+test"])
    risk_horizons_days: List[int] = field(default_factory=lambda: [300, 600, 900, -1])
    risk_weighting: str = "uniform"
    risk_distance_temp: float = 0.5
    risk_palette: str = "default"
    risk_show_tables: bool = True
    risk_artifacts_dir: Optional[Path] = None


# ---------------------------
# Arch label → series partition
# ---------------------------
_ARCH_PARTITION_MAP = {
    "Arps_Canonical": "arps",
    "Arps": "arps",
    "Seq2PIN": "seq2",
    "Seq2PINN": "seq2",
    "Seq2": "seq2",
    "Darts": "darts",
    "DARTS": "darts",
}

def _well_alias_keys(w: Optional[str]) -> list[str]:
    """
    Retorna possíveis aliases de um poço:
      '15/9-F-14' -> ['15/9-F-14', '15-9-F-14']
      '15-9-F-14' -> ['15-9-F-14', '15/9-F-14']  (tentativa conservadora)
    """
    if w is None:
        return []
    s = str(w).strip()
    if not s:
        return []
    out = {s}
    if "/" in s:
        out.add(s.replace("/", "-"))
    # heurística simples: se é algo tipo 15-9-F-14, tenta primeira barra
    if "-" in s and "15-9-" in s:
        out.add(s.replace("-", "/", 1))
    return list(out)


def _normalize_arch_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # prefer 'arch'
    if "arch" not in out.columns:
        if "architecture" in out.columns:
            out.rename(columns={"architecture": "arch"}, inplace=True)
        elif "architecture_name" in out.columns:
            out.rename(columns={"architecture_name": "arch"}, inplace=True)
    if "arch" in out.columns:
        out["arch"] = out["arch"].astype(str).map(lambda x: _ARCH_PARTITION_MAP.get(x, x.lower()))
    return out


def _normalize_well_for_fs(well: Optional[str]) -> Optional[str]:
    """
    Normalize a well identifier to be filesystem-safe.
    For example: '15/9-F-14' → '15-9-F-14'.
    """
    if well is None:
        return None
    return str(well).replace("/", "-")

def _scan_campaign_parquets(series_root: Path, group: str) -> pd.DataFrame:
    """
    Escaneia group={group}/arch=*/**/job=*.parquet e devolve um DF com contagem por (well, arch, campaign).

    Isso NÃO carrega os Parquets, só conta arquivos no disco.
    """
    root = (series_root / f"group={group}").resolve()
    if not root.exists():
        warn(f"[Preflight] series_store group path not found: {root}")
        return pd.DataFrame(columns=["well", "arch", "campaign", "n_parquets"])

    rows: list[dict[str, Any]] = []
    # Caminho típico: arch=arps/dataset=VOLVE/well=15-9-F-14/campaign=validation_seq2/job=<hash>.parquet
    for p in root.glob("arch=*/**/job=*.parquet"):
        rel = p.relative_to(root)
        arch = None
        well = None
        campaign = None
        for part in rel.parts:
            if part.startswith("arch="):
                arch = part.split("=", 1)[1]
            elif part.startswith("well="):
                well = part.split("=", 1)[1]
            elif part.startswith("campaign="):
                campaign = part.split("=", 1)[1]
        rows.append({"well": well, "arch": arch, "campaign": campaign})

    if not rows:
        return pd.DataFrame(columns=["well", "arch", "campaign", "n_parquets"])

    df = pd.DataFrame(rows)
    grp = (
        df.groupby(["well", "arch", "campaign"], as_index=False)
          .size()
          .rename(columns={"size": "n_parquets"})
    )
    return grp



# ---------------------------
# Enforce Top-K per (well, arch)
# ---------------------------
def _enforce_topk_per_well_arch(df: pd.DataFrame, k: int, score_col: str, *, lower_is_better: bool = True) -> pd.DataFrame:
    """
    Guarantee at most k champions per (well, arch) by sorting on the active score column.
    Uses lower_is_better to decide ascending/descending.
    """
    if df is None or df.empty:
        return df

    use_col = score_col if score_col in df.columns else "val_smape_agg"
    ascending = True if lower_is_better else False

    grouped = []
    for (w, a), g in df.groupby(["well", "arch"], dropna=False):
        g = g.sort_values(use_col, ascending=ascending).head(k)
        grouped.append(g)
    out = pd.concat(grouped, ignore_index=True) if grouped else pd.DataFrame()
    return out


def _compute_primary_per_well(
    champions_df: pd.DataFrame,
    score_col: str = "weighted_score",
) -> Dict[str, str]:
    """
    Escolhe exatamente um job_hash primário por well (melhor score).
    Cai para 'val_smape_agg' se score_col não existir (lower é melhor).
    """
    logger = logging.getLogger("train_restrict")

    if champions_df is None or champions_df.empty:
        logger.warning("[train_restrict] _compute_primary_per_well: champions_df is empty.")
        return {}

    logger.info(
        "[train_restrict] _compute_primary_per_well: champions_df shape=%s, columns=%s",
        champions_df.shape, list(champions_df.columns)
    )

    if "well" not in champions_df.columns or "job_hash" not in champions_df.columns:
        logger.error(
            "[train_restrict] _compute_primary_per_well: missing required columns 'well' or 'job_hash'."
        )
        return {}

    use_col = score_col if score_col in champions_df.columns else "val_smape_agg"
    if use_col != score_col:
        logger.warning(
            "[train_restrict] _compute_primary_per_well: score_col '%s' not found, falling back to '%s'.",
            score_col, use_col
        )
    else:
        logger.info("[train_restrict] _compute_primary_per_well: using score column '%s'.", use_col)

    counts_per_well = champions_df["well"].value_counts(dropna=False)
    logger.info(
        "[train_restrict] _compute_primary_per_well: candidates per well:\n%s",
        counts_per_well.to_string()
    )

    tmp = (
        champions_df
        .copy()
        .sort_values(["well", use_col], ascending=[True, True])
    )

    primary = tmp.groupby("well", as_index=True)["job_hash"].first().to_dict()
    logger.info(
        "[train_restrict] _compute_primary_per_well: selected %d primary jobs for %d wells.",
        len(primary),
        champions_df["well"].nunique(dropna=False),
    )

    if primary:
        sample_items = list(primary.items())[:8]
        logger.info(
            "[train_restrict] _compute_primary_per_well: sample primary_by_well=%s",
            sample_items,
        )
    else:
        logger.warning("[train_restrict] _compute_primary_per_well: primary mapping is empty.")

    return primary



def _n_members_by_family(champs: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-(well, arch) ensemble size (n_members).

    We deliberately count HOW MANY champion rows went into each
    (well, arch) family, not how many *unique* job_hash values.
    This matches the intended 'n' you want to display in the plots.

    Output columns:
        well, arch, n_members
    """
    if champs is None or champs.empty:
        warn("[Stage 3] _n_members_by_family called with empty champions_df.")
        return pd.DataFrame(columns=["well", "arch", "n_members"])

    if not {"well", "arch", "job_hash"}.issubset(champs.columns):
        missing = {"well", "arch", "job_hash"} - set(champs.columns)
        warn(
            "[Stage 3] _n_members_by_family: champions_df missing columns %s; "
            "fallback n_members=1.",
            sorted(missing),
        )
        return pd.DataFrame(columns=["well", "arch", "n_members"])

    g = (
        champs
        .groupby(["well", "arch"], as_index=False)
        .agg(n_members=("job_hash", "size"))  # <-- key change: SIZE, not nunique
    )

    # normalize arch to the same tokens used in the plots (e.g. 'arps', 'seq2')
    g["arch"] = (
        g["arch"].astype(str).str.lower().replace({
            "arps_canonical": "arps",
            "arps": "arps",
            "seq2pin": "seq2",
            "seq2_pinn": "seq2",
            "seq2": "seq2",
        })
    )

    return g


def _preflight_block(paths: Dict[str, Path], cfg: Phase4Config) -> Dict[str, Any]:
    """
    Clean pre-flight:
      - one compact block in compact mode
      - optional parquet population scan
      - no duplicated INFO spam
    """
    series_root = paths["series_store_root"]
    compact = is_compact_logging(cfg)
    width = effective_log_width(cfg, fallback=100)

    bnd_path = series_root / "meta" / "boundaries.parquet"
    bnd_ok = bnd_path.exists()

    probe_well_raw = None
    if getattr(cfg, "wells_to_analyze", None):
        probe_well_raw = str(cfg.wells_to_analyze[0])

    probe_well_fs = _normalize_well_for_fs(probe_well_raw) or probe_well_raw
    hist_path = series_root / "history" / f"well={probe_well_fs}" / "history.parquet"
    hist_ok = hist_path.exists()

    if not compact:
        info(f"[Preflight] Using series_store_root = {series_root}")
        info(f"[Preflight] Expecting manifest at: {bnd_path}")
        if bnd_ok:
            info("✅ [Preflight] boundaries.parquet FOUND")
        else:
            warn("⚠️  [Preflight] boundaries.parquet NOT FOUND")

        info(f"[Preflight] Probing history at: {hist_path}")
        if hist_ok:
            info(f"✅ [Preflight] history for well={probe_well_raw} FOUND")
        else:
            warn(f"⚠️  [Preflight] history for well={probe_well_raw} NOT FOUND at this root")

    lines = [
        f"series_store_root: {series_root}",
        f"boundaries.parquet: {'FOUND' if bnd_ok else 'MISSING'}",
        f"history probe (well={probe_well_raw}): {'FOUND' if hist_ok else 'NOT FOUND'}",
    ]
    log_block("Pre-Flight", lines, level=logging.INFO, width=width)

    if getattr(cfg, "show_preflight_population_scan", False):
        try:
            group_name = None
            if getattr(cfg, "campaigns_to_ensemble", None):
                group_name = next(iter(cfg.campaigns_to_ensemble.values()))

            if group_name:
                pop = _scan_campaign_parquets(series_root, group_name)
                if not pop.empty:
                    if probe_well_fs:
                        mask = pop["well"].isin({probe_well_fs, probe_well_raw})
                        sub = pop.loc[mask].copy()
                    else:
                        sub = pop.copy()

                    if not sub.empty:
                        scan_lines = ["(counts are per well × arch × campaign)"]
                        for _, r in sub.sort_values(["well", "arch", "campaign"]).iterrows():
                            scan_lines.append(
                                f"well={r['well']} arch={r['arch']} "
                                f"campaign={r['campaign']} → n_parquets={int(r['n_parquets'])}"
                            )
                        log_block(
                            "Series Store Population (Parquet counts)",
                            scan_lines,
                            level=logging.INFO,
                            width=width,
                        )
        except Exception as e:
            warn(f"[Preflight] Parquet population scan failed softly: {e}")

    return {
        "boundaries_ok": bnd_ok,
        "history_ok": hist_ok,
        "probe_well": probe_well_raw,
        "probe_hist_path": hist_path,
    }



def _stage1_select_champions(
    cfg: Phase4Config,
    exp_root: Path,
    deps: Dict[str, Any],
) -> tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Stage 1 with compact-safe logging:
    - suppress noisy downstream INFO logs in compact mode
    - keep only one clean stage summary
    - optional table preview only when explicitly enabled
    """
    stage_banner(
        1,
        "Robust Champion Harvester",
        "Discover leaderboards → score → robust filtering → enforce Top-N per (well, arch)",
    )

    compact = is_compact_logging(cfg)
    width = effective_log_width(cfg, fallback=100)
    suppress_internal = compact and getattr(cfg, "suppress_internal_info_in_compact", True)

    all_champs: list[pd.DataFrame] = []
    stage_stats: list[dict[str, Any]] = []
    score_col = "robust_score" if cfg.scoring_strategy == "robust_score" else "weighted_score"

    posthoc = dict(cfg.posthoc_overrides or {})
    if "top_strategies_per_well" not in posthoc:
        posthoc["top_strategies_per_well"] = int(cfg.n_champions_per_group)

    if not compact:
        info(
            "[Stage 1] "
            f"top_strategies_per_well={posthoc.get('top_strategies_per_well')} | "
            f"score={score_col}"
        )

    select_champions_from_df = deps["select_champions_from_df"]

    for split_tag, group_name in cfg.campaigns_to_ensemble.items():
        if not compact:
            info(f"[Stage 1] Split='{split_tag}'  Group='{group_name}'")

        with _quiet_existing_info_logs(suppress_internal):
            master_df = _discover_and_load_leaderboards(exp_root, group_name)

        if master_df is None or master_df.empty:
            warn(f"[Stage 1] No leaderboards for '{group_name}'.")
            continue

        with _quiet_existing_info_logs(suppress_internal):
            champs = select_champions_from_df(
                master_df=master_df,
                selection_cfg_overrides=posthoc,
                metric_weights=cfg.metric_weights,
                lower_is_better=cfg.lower_is_better,
                scoring_strategy=cfg.scoring_strategy,
            )

        n_rows = int(len(master_df))
        n_wells = int(master_df["well"].nunique()) if "well" in master_df.columns else None

        if champs is None or champs.empty:
            stage_stats.append(
                dict(
                    split=split_tag,
                    group=group_name,
                    leaderboard_rows=n_rows,
                    wells=n_wells,
                    champions=0,
                )
            )
            warn(f"[Stage 1] No champions selected using '{cfg.scoring_strategy}' for '{group_name}'.")
            continue

        champs = _normalize_arch_column(champs)
        champs["split"] = split_tag
        all_champs.append(champs)

        stage_stats.append(
            dict(
                split=split_tag,
                group=group_name,
                leaderboard_rows=n_rows,
                wells=n_wells,
                champions=int(len(champs)),
            )
        )

    champions_df = pd.concat(all_champs, ignore_index=True) if all_champs else pd.DataFrame()

    if champions_df.empty:
        warn("[Stage 1] No champions harvested.")
        return champions_df, score_col, posthoc

    if not compact:
        ok(f"[Stage 1] Harvested {len(champions_df)} champions total.")

        from common.log_utils import (
            build_champion_summary_table,
            log_champion_summary_box,
            log_table_block,
        )

        log_champion_summary_box(
            champions_df,
            score_col=score_col,
            title="Stage 1 — Champions (Summary)",
            width=width,
        )

        if getattr(cfg, "show_stage1_table_preview", False):
            tbl = build_champion_summary_table(champions_df, score_col=score_col)
            if not tbl.empty:
                headers = ["well", "arch", "n", "score_min", "score_median", "score_mean"]
                rows = tbl[headers].head(12).values.tolist()
                log_table_block(
                    "Stage 1 — Champion Score Table (preview)",
                    headers,
                    rows,
                    width=width,
                )

                if getattr(cfg, "artifacts_dir", None):
                    outdir = Path(cfg.artifacts_dir)
                    outdir.mkdir(parents=True, exist_ok=True)
                    tbl.to_csv(outdir / "stage1_champion_scores.csv", index=False)

    lines = summarize_stage1(champions_df, posthoc, score_col=score_col)

    if stage_stats:
        lines.append("groups:")
        for s in stage_stats:
            group_short = str(s["group"])
            lines.append(
                f"  {s['split']}: rows={s['leaderboard_rows']:,} | "
                f"wells={s['wells']} | champs={s['champions']:,} | group={group_short}"
            )

    log_block(
        "Stage 1 — Robust Champion Harvester (Summary)",
        lines,
        level=logging.INFO,
        width=width,
    )

    return champions_df, score_col, posthoc




# ---------------------------
# Stage 2 — Series loader (helper)
# ---------------------------
def _stage2_load_series(
    cfg: Phase4Config,
    series_root: Path,
    champions_df: pd.DataFrame,
    deps: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    stage_banner(2, "Series Store Reader", "Load champion series (parquet) from series_store")

    compact = is_compact_logging(cfg)
    width = effective_log_width(cfg, fallback=100)
    suppress_internal = compact and getattr(cfg, "suppress_internal_info_in_compact", True)

    series_df: pd.DataFrame = pd.DataFrame()
    boundaries_df: pd.DataFrame = pd.DataFrame()
    full_history_by_well: Dict[str, pd.DataFrame] = {}

    if champions_df.empty:
        warn("[Stage 2] Skipped (no champions).")
        return series_df, boundaries_df, full_history_by_well

    split_to_group = dict(cfg.campaigns_to_ensemble)
    champs_local = champions_df.copy()
    champs_local["group"] = champs_local["split"].map(split_to_group)

    arch_counts = {}
    if {"arch", "job_hash"}.issubset(champs_local.columns):
        arch_counts = (
            champs_local.groupby("arch")["job_hash"]
            .nunique()
            .sort_index()
            .to_dict()
        )

    if not compact and arch_counts:
        pretty = ", ".join(f"{k}={v}" for k, v in arch_counts.items())
        info(f"[Stage 2] Champion counts by arch: {pretty}")

    with _quiet_existing_info_logs(suppress_internal):
        series_df, extras = deps["read_series_by_champions"](
            series_root, champs_local, return_extras=True
        )

    boundaries_df = extras.get("boundaries_df", pd.DataFrame())
    full_history_by_well = extras.get("full_history_by_well", {}) or {}

    if series_df is None or series_df.empty:
        warn("[Stage 2] No series loaded. Check series_store paths.")
        return pd.DataFrame(), boundaries_df, full_history_by_well

    uniq = series_df["job_hash"].nunique() if "job_hash" in series_df.columns else 0

    if compact:
        lines = summarize_stage2(
            series_df=series_df,
            boundaries_df=boundaries_df,
            full_history_by_well=full_history_by_well,
        )
        if arch_counts:
            pretty = " | ".join(f"{k}={v}" for k, v in arch_counts.items())
            lines.insert(0, f"champions by arch: {pretty}")

        log_block(
            "Stage 2 — Series Store Reader (Summary)",
            lines,
            level=logging.INFO,
            width=width,
        )
    else:
        ok(f"[Stage 2] Loaded {len(series_df)} rows, {uniq} unique champions.")
        cols = list(series_df.columns)
        preview = ", ".join(cols[:10]) + (" …" if len(cols) > 10 else "")
        info("Columns: " + preview)

    return series_df, boundaries_df, full_history_by_well


# ---------------------------
# Stage 2.5 — Train restrict to primary per well (helper)
# ---------------------------
def _stage25_restrict_train(
    cfg: Phase4Config,
    series_df: pd.DataFrame,
    champions_df: pd.DataFrame,
    score_col: str,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    plot_series_df = series_df.copy() if isinstance(series_df, pd.DataFrame) else pd.DataFrame()
    primary_by_well: Dict[str, str] = {}

    if plot_series_df.empty:
        warn("[Stage 2.5] plot_series_df not built (series_df empty).")
        return plot_series_df, primary_by_well

    try:
        from common.train_restrict import (  # type: ignore
            _compute_primary_per_well as _compute_primary_per_well_ext,
            _restrict_train_to_primary,
        )
        primary_by_well = _compute_primary_per_well_ext(champions_df, score_col=score_col)
    except Exception:
        warn("[Stage 2.5] train_restrict import failed; using local _compute_primary_per_well")
        primary_by_well = _compute_primary_per_well(champions_df, score_col=score_col)

    before_rows = len(plot_series_df)
    before_train = int(plot_series_df["split"].astype(str).str.lower().eq("train").sum()) \
        if "split" in plot_series_df.columns else None

    try:
        if '_restrict_train_to_primary' in locals():
            plot_series_df = _restrict_train_to_primary(plot_series_df, primary_by_well)
        else:
            warn("[Stage 2.5] _restrict_train_to_primary missing; keeping full TRAIN.")
    except Exception as e:
        warn(f"[Stage 2.5] train restriction failed: {e}")

    after_rows = len(plot_series_df)
    after_train = int(plot_series_df["split"].astype(str).str.lower().eq("train").sum()) \
        if "split" in plot_series_df.columns else None

    kept_wells = len({w for w, _ in primary_by_well.items()})
    ok(f"[Stage 2.5] Restricted TRAIN to 1 series per well (wells={kept_wells}).")
    if before_train is not None and after_train is not None:
        info(f"[Stage 2.5] Train rows: {before_train} → {after_train} (Δ={after_train - before_train}).")
    info(f"[Stage 2.5] Series rows: {before_rows} → {after_rows} (Δ={after_rows - before_rows}).")

    # Compact block (opcional)
    if getattr(cfg, "log_mode", "verbose") == "compact":
        lines = summarize_stage25(before_rows=before_rows,
                                  after_rows=after_rows,
                                  before_train=before_train,
                                  after_train=after_train,
                                  kept_wells=kept_wells,
                                  primary_by_well=primary_by_well)
        log_block("Stage 2.5 — Train Restriction (Summary)", lines,
                  level=logging.INFO, width=getattr(cfg, "log_width", 100))

    return plot_series_df, primary_by_well



# ---------------------------
# Stage 3 — Ensemble builder (helper)
# ---------------------------
def _stage3_build_ensembles(
    cfg: "Phase4Config",
    champions_df: pd.DataFrame,
    series_df: pd.DataFrame,
    deps: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build intra-family & inter-family ensembles and attach explicit hierarchy metadata.

    Compatibility:
      - keeps 'n_members' for downstream code/plots
      - adds:
          * intra_family_df['n_members_family']
          * final_ensemble_df['n_members_total']
          * both receive ['ensemble_level']
    """
    stage_banner(3, "Ensemble Computation", "Build intra-family & inter-family ensembles")

    intra_family_df = pd.DataFrame()
    final_ensemble_df = pd.DataFrame()

    if champions_df is None or champions_df.empty or series_df is None or series_df.empty:
        warn("[Stage 3] Skipped (missing champions or series).")
        return intra_family_df, final_ensemble_df

    info("[Stage 3] Building intra-family ensemble …")
    intra_family_df = deps["build_intra_family_ensemble"](champions_df, series_df)
    ok(f"[Stage 3] Intra-family shape: {intra_family_df.shape}")

    info("[Stage 3] Building inter-family (meta) ensemble …")
    final_ensemble_df = deps["build_inter_family_ensemble"](intra_family_df)
    ok(f"[Stage 3] Final ensemble shape: {final_ensemble_df.shape}")

    try:
        fam_counts = _n_members_by_family(champions_df)

        def _with_norm_well(df: pd.DataFrame, col: str = "well") -> pd.DataFrame:
            if df is None or df.empty or col not in df.columns:
                return df
            out = df.copy()
            out["_well_norm"] = out[col].astype(str).map(normalize_well_key)
            return out

        fam_counts = _with_norm_well(fam_counts, "well")
        intra_family_df = _with_norm_well(intra_family_df, "well")
        final_ensemble_df = _with_norm_well(final_ensemble_df, "well")

        # -------------------------
        # Intra-family: n_members per (well, arch)
        # -------------------------
        if not intra_family_df.empty:
            intra_family_df = intra_family_df.merge(
                fam_counts[["_well_norm", "arch", "n_members"]],
                on=["_well_norm", "arch"],
                how="left",
            )

            missing_intra = int(intra_family_df["n_members"].isna().sum())
            if missing_intra:
                warn(
                    "[Stage 3] Intra-family: n_members missing for %d/%d rows "
                    "(after alias-aware merge). Defaulting to 1.",
                    missing_intra, len(intra_family_df),
                )
                intra_family_df["n_members"] = intra_family_df["n_members"].fillna(1)

            intra_family_df["n_members"] = intra_family_df["n_members"].astype(int)
            intra_family_df["n_members_family"] = intra_family_df["n_members"]
            intra_family_df["ensemble_level"] = "family"
            intra_family_df = intra_family_df.drop(columns=["_well_norm"], errors="ignore")

        # -------------------------
        # Inter-family: total members per well
        # -------------------------
        if not final_ensemble_df.empty and not fam_counts.empty:
            inter_counts = (
                fam_counts
                .groupby("_well_norm", as_index=False)["n_members"]
                .sum()
                .rename(columns={"n_members": "n_members"})
            )

            final_ensemble_df = final_ensemble_df.merge(
                inter_counts[["_well_norm", "n_members"]],
                on="_well_norm",
                how="left",
            )

            missing_inter = int(final_ensemble_df["n_members"].isna().sum())
            if missing_inter:
                warn(
                    "[Stage 3] Inter-family: n_members missing for %d/%d rows "
                    "(after alias-aware merge). Defaulting to 1.",
                    missing_inter, len(final_ensemble_df),
                )
                final_ensemble_df["n_members"] = final_ensemble_df["n_members"].fillna(1)

            final_ensemble_df["n_members"] = final_ensemble_df["n_members"].astype(int)
            final_ensemble_df["n_members_total"] = final_ensemble_df["n_members"]
            final_ensemble_df["ensemble_level"] = "meta"
            final_ensemble_df = final_ensemble_df.drop(columns=["_well_norm"], errors="ignore")

        if not fam_counts.empty:
            ok(
                "[Stage 3] Family members per (well, arch):\n%s",
                fam_counts[["well", "arch", "n_members"]].to_string(index=False),
            )

    except Exception as e:
        warn(f"[Stage 3] Could not attach hierarchy metadata (soft failure): {type(e).__name__}: {e}")

    return intra_family_df, final_ensemble_df



# ---------------------------
# Stage 4 — Visualization & Reporting (helper)
# ---------------------------
def _stage4_visualize_and_report(
    cfg: Phase4Config,
    series_df: pd.DataFrame,
    plot_series_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    full_history_by_well: Dict[str, pd.DataFrame],
    intra_family_df: pd.DataFrame,
    final_ensemble_df: pd.DataFrame,
) -> None:
    """
    Stage 4 — Visualization & Reporting.

    Produces:
      1) Intra-only (Seq2PINN), if available
      2) Intra-only (ARPS), if available
      3) Inter (Final Ensemble) — conjugated view
      4) Ensemble Members — per-family spaghetti + bold FAMILY mean (NEW API)
    """
    if not getattr(cfg, "enable_plots", True):
        return

    stage_banner(
        4,
        "Visualization & Reporting",
        "Regions (Train/Val/Test), Ground Truth, Final Mean + band, optional overlays",
    )

    # ---- Imports
    try:
        from plotting.ensemble_plots import plot_conjugated_ensemble
        from plotting.ensemble_members import plot_ensemble_members_by_family  # nova API (com intra_family_df)
    except Exception as e:
        warn(f"[Stage 4] Plot modules not available: {e}")
        return

    try:
        from common.phase_viz_support import (  # type: ignore
            build_full_history,
            infer_boundaries,
            fallback_full_history_from_series,
            fallback_boundaries,
            lookup_manifest_bounds,
            filter_intra_by_arch,
            debug_frame,
            make_family_traces_visible,
            align_with_t_factory,
        )
    except Exception:
        from common.phase_viz_support import (  # type: ignore
            fallback_full_history_from_series,
            fallback_boundaries,
            lookup_manifest_bounds,
            filter_intra_by_arch,
            debug_frame,
            make_family_traces_visible,
            align_with_t_factory,
        )
        build_full_history = None
        infer_boundaries = None

    wells = cfg.wells_to_analyze or (
        sorted(final_ensemble_df["well"].unique()) if not final_ensemble_df.empty else []
    )
    if not wells:
        warn("[Stage 4] No wells to plot.")

    _boundaries_df = boundaries_df if isinstance(boundaries_df, pd.DataFrame) else pd.DataFrame()
    _full_hist_map = full_history_by_well or {}

    plotted = 0
    for w in wells:
        info(f"[Stage 4] Plotting well={w} …")
        try:
            # ---- Full history (manifest or fallback)
                        # tenta todos os aliases possíveis do poço no dicionário de histórico
            fh = None
            for key in _well_alias_keys(w):
                fh = _full_hist_map.get(key)
                if fh is not None and not getattr(fh, "empty", False):
                    break

            if fh is None or getattr(fh, "empty", True):
                if callable(build_full_history):
                    try:
                        fh = build_full_history(series_df, w, full_history_by_well=_full_hist_map)
                    except Exception:
                        fh = None

            if fh is None or getattr(fh, "empty", True):
                fh = fallback_full_history_from_series(series_df, w)
            if fh is None or fh.empty:
                warn(f"[Stage 4] Skipping well={w}: no full history.")
                continue

            # ---- Boundaries
            bounds = None
            for key in _well_alias_keys(w):
                bounds = lookup_manifest_bounds(_boundaries_df, key)
                if bounds is not None:
                    break

            if bounds is None and callable(infer_boundaries):
                try:
                    bounds = infer_boundaries(
                        final_ensemble_df, series_df, w,
                        boundaries_df=_boundaries_df, full_history_df=fh
                    )
                except Exception:
                    bounds = None
            if bounds is None:
                bounds = fallback_boundaries(w, fh)


            # def _split_counts_for_well(df: pd.DataFrame, well: str) -> tuple[int, int]:
            #     if df is None or df.empty or "well" not in df.columns or "split" not in df.columns:
            #         return 0, 0
            #     key = normalize_well_key(well)
            #     d = df[df["well"].astype(str).map(normalize_well_key).eq(key)].copy()
            #     if d.empty or "idx" not in d.columns:
            #         return 0, 0
            #     d["split"] = d["split"].astype(str).str.lower()
            #     # de-dup por (split, idx) p/ não somar seq2+arps juntos
            #     d = d.drop_duplicates(subset=["split", "idx"])
            #     n_val = int(d["split"].isin(["val", "validation"]).sum())
            #     n_test = int((d["split"] == "test").sum())
            #     return n_val, n_test
            
            # n_val, n_test = _split_counts_for_well(intra_family_df, w)
            # if (n_val + n_test) == 0:
            #     n_val, n_test = _split_counts_for_well(plot_series_df, w)
            
            # expected_total = int(bounds["train_end"]) + 1 + int(n_val) + int(n_test)
            
            # fh = maybe_pad_full_history_left_with_nans(
            #     fh,
            #     expected_total=expected_total,
            #     assume_t_is_index=True,
            #     tag=f"well={w}",
            # )
            
            # ---- Alignment helper (global_idx -> t) para ESTE poço
            _align_with_t = align_with_t_factory(fh, bounds)

            # ---- Align frames
            final_df_aligned  = _align_with_t(final_ensemble_df, "final_ensemble_df")
            intra_df_aligned  = _align_with_t(intra_family_df,   "intra_family_df")
            champs_df_aligned = _align_with_t(plot_series_df,    "plot_series_df")

            # ---- Per-family views
            intra_seq2 = filter_intra_by_arch(intra_df_aligned, "seq2")
            intra_arps = filter_intra_by_arch(intra_df_aligned, "arps")

            debug_frame("INTRA_SEQ2_in", intra_seq2, w)
            debug_frame("INTRA_ARPS_in", intra_arps, w)
            debug_frame("INTER_FINAL_in", final_df_aligned, w)

            # 1) Intra — Seq2
            if not intra_seq2.empty:
                fig_seq2 = plot_conjugated_ensemble(
                    final_ensemble_df=pd.DataFrame(),
                    intra_family_df=intra_seq2,
                    champion_series_df=champs_df_aligned,
                    full_history_df=fh,
                    boundaries=bounds,
                    well=w,
                    title=f"Intra-Family — PINN — {w}",
                    palette=getattr(cfg, "palette", "default"),
                    show_family_traces=True,
                    show_champion_traces=getattr(cfg, "show_champion_traces", False),
                    show_train_reconstruction=False,
                    show=False,
                )
                make_family_traces_visible(fig_seq2)
                try:
                    from IPython.display import display  # type: ignore
                    display(fig_seq2)
                except Exception:
                    fig_seq2.show()
                plotted += 1
            else:
                warn(f"[Stage 4] No intra Seq2 data for well={w}")

            # 2) Intra — ARPS
            if not intra_arps.empty:
                fig_arps = plot_conjugated_ensemble(
                    final_ensemble_df=pd.DataFrame(),
                    intra_family_df=intra_arps,
                    champion_series_df=champs_df_aligned,
                    full_history_df=fh,
                    boundaries=bounds,
                    well=w,
                    title=f"Intra-Family — ARPS — {w}",
                    palette=getattr(cfg, "palette", "default"),
                    show_family_traces=True,
                    show_champion_traces=getattr(cfg, "show_champion_traces", False),
                    show_train_reconstruction=False,
                    show=False,
                )
                make_family_traces_visible(fig_arps)
                try:
                    from IPython.display import display  # type: ignore
                    display(fig_arps)
                except Exception:
                    fig_arps.show()
                plotted += 1
            else:
                warn(f"[Stage 4] No intra ARPS data for well={w}")

            # 3) Inter — Final Ensemble (conjugated)
            fig_inter = plot_conjugated_ensemble(
                final_ensemble_df=final_df_aligned,
                intra_family_df=intra_df_aligned,
                champion_series_df=champs_df_aligned,
                full_history_df=fh,
                boundaries=bounds,
                well=w,
                title=f"Inter-Family — Final Ensemble — {w}",
                palette=getattr(cfg, "palette", "default"),
                show_family_traces=getattr(cfg, "show_family_traces", False),
                show_champion_traces=getattr(cfg, "show_champion_traces", False),
                show=True,
            )
            plotted += 1

            # 4) Ensemble Members — per-family (usa FAMILY mean; fallback para inter)
            if getattr(cfg, "show_ensemble_members_plot", True):
                final_w = final_df_aligned[
                    final_df_aligned.get("well", "").astype(str).eq(str(w))
                ].copy() if "well" in final_df_aligned.columns else final_df_aligned

                for fam_key, fam_label in (("seq2", "PINN"), ("arps", "ARPS")):
                    try:
                        # slice alinhado da família para usar a yhat_family_mean no members plot
                        intra_slice = intra_df_aligned.copy()
                        if "well" in intra_slice.columns:
                            intra_slice = intra_slice[intra_slice["well"].astype(str) == str(w)]
                        if "arch" in intra_slice.columns:
                            intra_slice = intra_slice[intra_slice["arch"].astype(str).str.lower().eq(fam_key)]

                        plot_ensemble_members_by_family(
                            series_df=series_df,           # RAW per-member series
                            final_ensemble_df=final_w,     # fallback inter (se faltar intra)
                            intra_family_df=intra_slice,   # <<< média da família para este arch
                            full_history_df=fh,
                            boundaries=bounds,
                            well=w,
                            arch_key=fam_key,              # "seq2" | "arps"
                            palette=getattr(cfg, "palette", "default"),
                            title=f"Ensemble Members — {w} — {fam_label}",
                            font_scale=1.1,
                            width=1100,
                            height=700,
                            max_members=120,
                            show=True,
                        )
                        plotted += 1
                    except Exception as e:
                        warn(f"[Stage 4] Members plot (family={fam_key}) failed for well={w}: {e}")

            ok(f"[Stage 4] Rendered well={w} (up to 5 figures)")
        except Exception as e:
            err(f"[Stage 4] Failed for well={w}: {type(e).__name__}: {e}")

    info(f"🏁 [Stage 4] Summary: plotted={plotted} figure(s)")







# --- small helpers (new) -----------------------------------------------------

def _normalize_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize filter dict (trim strings, accept None/missing keys)."""
    if not filters:
        return {}
    out = {}
    for k, v in filters.items():
        if isinstance(v, str):
            vv = v.strip()
            if vv:
                out[k] = vv
        elif v is not None:
            out[k] = v
    return out

def _filter_df_by_exact(df: pd.DataFrame, **eq) -> pd.DataFrame:
    """Filter a DataFrame by exact equality where columns exist; ignore missing cols."""
    if df is None or df.empty or not eq:
        return df
    mask = pd.Series(True, index=df.index)
    for col, val in eq.items():
        if col in df.columns and val is not None:
            mask &= (df[col].astype(str) == str(val))
    return df.loc[mask].copy()

def _restrict_history_map(history_by_well: Dict[str, pd.DataFrame], wells: Iterable[str]) -> Dict[str, pd.DataFrame]:
    if not history_by_well:
        return {}
    want = {str(w) for w in wells}
    return {w: df for w, df in history_by_well.items() if str(w) in want}

# ---------------------------
# Stage 4.1 — Risk (compute only, no disk I/O)
# ---------------------------
def _stage41_compute_risk_curves(
    cfg: Phase4Config,
    *,
    series_df: pd.DataFrame,
    final_ensemble_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    full_history_by_well: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Stage 4.1 — Risk (Simple Path)

    Current contract:
    - Risk is computed at FAMILY level from series_df members.
    - Windowing uses split+idx:
        * VAL = full
        * TEST = head(H)
    - final_ensemble_df is accepted for future extensions, but current risk
      output remains family-level on purpose.

    Returns:
        {well: risk_df}
    """
    import numpy as np
    import pandas as pd

    from common.log_utils import ok, warn, info, log_block
    from common.phase_viz_support import (
        lookup_manifest_bounds,
        infer_boundaries,
        fallback_boundaries,
        build_full_history,
        normalize_well_key,
    )

    try:
        from risk.risk_core import (
            build_member_series,
            weights_for_members,
            weighted_quantiles,
            accumulate_members_simple,
            _log_risk_window_summary,
        )
    except Exception as e:
        warn(f"[Stage 4.1] risk_core not available: {e}. Skipping risk computation.")
        return {}

    if not getattr(cfg, "enable_risk_plots", True):
        info("[Stage 4.1] Risk disabled by config.")
        return {}

    if series_df is None or series_df.empty:
        warn("[Stage 4.1] Skipped: series_df is empty.")
        return {}

    risk_splits = list(getattr(cfg, "risk_splits", ["val+test"]))
    risk_horizons_days = list(getattr(cfg, "risk_horizons_days", [300, 600, 900, -1]))
    risk_weighting = str(getattr(cfg, "risk_weighting", "uniform"))
    risk_temp = float(getattr(cfg, "risk_distance_temp", 0.5))

    out_map: Dict[str, pd.DataFrame] = {}

    # Wells in scope
    wells = []
    if getattr(cfg, "wells_to_analyze", None):
        target_keys = {normalize_well_key(w) for w in cfg.wells_to_analyze}
        if "well" in series_df.columns:
            wells = [
                w for w in sorted(series_df["well"].astype(str).dropna().unique().tolist())
                if normalize_well_key(w) in target_keys
            ]
    elif "well" in series_df.columns:
        wells = sorted(series_df["well"].astype(str).dropna().unique().tolist())

    if not wells:
        warn("[Stage 4.1] No wells found in scope.")
        return {}

    # Normalize history map lookup
    history_norm = {
        normalize_well_key(k): v for k, v in (full_history_by_well or {}).items()
    }

    for well in wells:
        try:
            norm_key = normalize_well_key(well)

            # Boundaries
            bounds = None
            try:
                bounds = lookup_manifest_bounds(boundaries_df, well)
            except Exception:
                bounds = None
            if bounds is None:
                try:
                    bounds = infer_boundaries(series_df, well)
                except Exception:
                    bounds = None
            if bounds is None:
                try:
                    fh_fallback = history_norm.get(norm_key)
                    bounds = fallback_boundaries(fh_fallback)
                except Exception:
                    bounds = None

            # Full history
            fh = history_norm.get(norm_key)
            if fh is None or getattr(fh, "empty", True):
                try:
                    fh = build_full_history(series_df, well)
                except Exception:
                    fh = pd.DataFrame()

            # Families available for this well
            if "arch" in series_df.columns:
                fams = (
                    series_df.loc[
                        series_df["well"].astype(str).map(normalize_well_key).eq(norm_key),
                        "arch",
                    ]
                    .astype(str)
                    .str.lower()
                    .unique()
                    .tolist()
                )
                fams = [f for f in sorted(fams) if f in ("seq2", "arps")]
            else:
                fams = ["seq2", "arps"]

            if not fams:
                warn(f"[Stage 4.1] No supported families for well={well}.")
                continue

            risk_rows: List[Dict[str, Any]] = []

            for arch in fams:
                # Member series used for weights
                df_members_for_weights = build_member_series(
                    series_df=series_df,
                    well=well,
                    arch=arch,
                    full_history_df=fh,
                    boundaries=bounds,
                )
                if df_members_for_weights is None or df_members_for_weights.empty:
                    warn(f"[Stage 4.1] No member series for well={well}, arch={arch}.")
                    continue

                wts = weights_for_members(
                    df_members=df_members_for_weights,
                    strategy=risk_weighting,
                    temperature=risk_temp,
                )

                for selector in risk_splits:
                    for H in risk_horizons_days:
                        acc = accumulate_members_simple(
                            series_df=series_df,
                            well=well,
                            arch=arch,
                            selector=str(selector),
                            horizon_days=int(H),
                            member_id_col="job_hash",
                            yhat_col="yhat",
                        )

                        if acc is None or acc.empty:
                            continue

                        acc = acc.merge(wts, on="job_hash", how="left")
                        if "weight" not in acc.columns or acc["weight"].isna().all():
                            n = acc["job_hash"].nunique()
                            acc["weight"] = 1.0 / max(1, n)
                        else:
                            s = float(acc["weight"].sum())
                            if s > 0:
                                acc["weight"] /= s
                            else:
                                n = acc["job_hash"].nunique()
                                acc["weight"] = 1.0 / max(1, n)

                        vals = acc["accum"].astype(float).to_numpy()
                        ws = acc["weight"].astype(float).to_numpy()
                        qs = weighted_quantiles(
                            values=vals,
                            weights=ws,
                            qs=(0.10, 0.50, 0.90),
                        )

                        n_eff = int(acc["job_hash"].nunique())
                        row = dict(
                            well=str(well),
                            arch=str(arch),
                            ensemble_level="family",
                            risk_source="series_df",
                            anchor_source="history_only_pending",   # placeholder until anchor metadata enters store
                            split_selector=str(selector),
                            horizon_days=int(H),
                            t_start=None,
                            t_end=None,
                            q10=float(qs.get(0.10, np.nan)),
                            q50=float(qs.get(0.50, np.nan)),
                            q90=float(qs.get(0.90, np.nan)),
                            mean=float(np.average(vals, weights=ws) if ws.size else np.nan),
                            min=float(np.min(vals) if vals.size else np.nan),
                            max=float(np.max(vals) if vals.size else np.nan),
                            n_members=n_eff,               # compatibility
                            n_members_effective=n_eff,     # explicit semantics
                            n_points_window=int(acc["n_points"].max()) if len(acc) else 0,
                        )
                        risk_rows.append(row)

                        _log_risk_window_summary(
                            well=str(well),
                            arch=str(arch),
                            selector=str(selector),
                            horizon_days=int(H),
                            n_members=row["n_members"],
                            n_points_total=row["n_points_window"],
                            using_simple=True,
                            offsets=None,
                            t_range=None,
                        )

            if not risk_rows:
                warn(f"[Stage 4.1] No risk rows for well={well}.")
                continue

            risk_df = pd.DataFrame(risk_rows).sort_values(
                ["well", "arch", "split_selector", "horizon_days"]
            )
            out_map[well] = risk_df

            lines = [
                f"Well: {well}",
                f"Families: {', '.join(fams)}",
                f"Rows: {len(risk_df)}",
                "Grid (selector × H): " + ", ".join(f"{s}/{h}" for s in risk_splits for h in risk_horizons_days),
                "Risk level: family-only",
                "Path: simple(split+idx) — VAL=full | TEST=head(H)",
            ]
            log_block("Stage 4.1 — Risk Summary (per well)", lines, width=100)
            ok(f"[Stage 4.1] Computed risk rows in-memory for well={well} (rows={len(risk_df)})")

        except Exception as e:
            warn(f"[Stage 4.1] Failed for well={well}: {e}")

    if out_map:
        ok(f"[Stage 4.1] Computed risk curves for {len(out_map)} well(s).")
    else:
        warn("[Stage 4.1] No risk output generated.")

    return out_map


def _flatten_risk_outputs_map(risk_outputs_by_well: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Flatten {well: risk_df} -> single DataFrame.
    Safe to reuse both in the main path and in replay notebooks.
    """
    if not risk_outputs_by_well:
        return pd.DataFrame()

    frames = []
    for well, df in risk_outputs_by_well.items():
        if df is None or df.empty:
            continue
        local = df.copy()
        if "well" not in local.columns:
            local["well"] = str(well)
        frames.append(local)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _clone_phase4_cfg(cfg: Phase4Config) -> Phase4Config:
    """
    Defensive copy so filters / compare arms do not mutate the caller config.
    """
    return deepcopy(cfg)


def _flatten_risk_outputs_map(risk_outputs_by_well: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Flatten {well: risk_df} -> single DataFrame.
    Safe to reuse both in the main path and in replay notebooks.
    """
    if not risk_outputs_by_well:
        return pd.DataFrame()

    frames = []
    for well, df in risk_outputs_by_well.items():
        if df is None or df.empty:
            continue
        local = df.copy()
        if "well" not in local.columns:
            local["well"] = str(well)
        frames.append(local)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_phase4_downstream_from_series(
    cfg: Phase4Config,
    *,
    champions_df: pd.DataFrame,
    series_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    full_history_by_well: Dict[str, pd.DataFrame],
    deps: Dict[str, Any],
    score_col: str,
) -> Dict[str, pd.DataFrame]:
    """
    Shared executor for the lower half of Phase 4.

    Flow:
      Stage 2.5 -> Stage 3 -> compact recap / rich tables -> Stage 4 -> Stage 4.1
    """
    from common.log_utils import (
        is_compact_logging,
        compact_section,
        log_phase4_compact_report,
        ok,
        warn,
    )
    from common.log_table import (
        build_table_from_plot_metrics,
        build_table_from_intra_metrics,
        render_rich_table,
    )

    # ---------- Stage 2.5 ----------
    with compact_section(
        enable=is_compact_logging(cfg),
        silence=["train_restrict"],
    ):
        plot_series_df, _primary_by_well = _stage25_restrict_train(
            cfg=cfg,
            series_df=series_df,
            champions_df=champions_df,
            score_col=score_col,
        )

    # ---------- Stage 3 ----------
    intra_family_df, final_ensemble_df = _stage3_build_ensembles(
        cfg=cfg,
        champions_df=champions_df,
        series_df=series_df,
        deps=deps,
    )

    # ---------- Compact recap ----------
    with compact_section(
        enable=is_compact_logging(cfg),
        silence=["ensemble_plot"],
    ):
        log_phase4_compact_report(
            final_ensemble_df=final_ensemble_df,
            intra_family_df=intra_family_df,
            full_history_by_well=full_history_by_well,
            boundaries_df=boundaries_df,
            cfg=cfg,
        )

    # ---------- Optional rich tables ----------
    if not is_compact_logging(cfg):
        results_inter = build_table_from_plot_metrics(
            final_ensemble_df=final_ensemble_df,
            full_history_by_well=full_history_by_well,
            boundaries_df=boundaries_df,
            ci_mode=getattr(cfg, "ci_mode", "sem"),
            coverage=getattr(cfg, "coverage_level", 0.90),
            use_alias_headers=True,
        )
        render_rich_table(
            results_inter,
            title="Inter-Family — Final Ensemble (Post-ensemble Metrics)",
            decimals=2,
            bar_width=18,
            bar_track=False,
            show_lines=True,
            header_style="bold black on white",
        )

        results_intra = build_table_from_intra_metrics(
            intra_family_df=intra_family_df,
            full_history_by_well=full_history_by_well,
            boundaries_df=boundaries_df,
            ci_mode=getattr(cfg, "ci_mode", "sem"),
            coverage=getattr(cfg, "coverage_level", 0.90),
            include_family_column=True,
            use_alias_headers=True,
        )
        render_rich_table(
            results_intra,
            title="Intra-Family — Family Means (Post-ensemble Metrics)",
            decimals=2,
            bar_width=18,
            bar_track=False,
            show_lines=True,
            header_style="bold black on white",
        )

    # ---------- Stage 4 ----------
    with compact_section(
        enable=is_compact_logging(cfg),
        silence=["ensemble_plot"],
    ):
        _stage4_visualize_and_report(
            cfg=cfg,
            series_df=series_df,
            plot_series_df=plot_series_df,
            boundaries_df=boundaries_df,
            full_history_by_well=full_history_by_well,
            intra_family_df=intra_family_df,
            final_ensemble_df=final_ensemble_df,
        )

    # ---------- Stage 4.1 — Risk ----------
    risk_outputs_by_well: Dict[str, pd.DataFrame] = {}
    risk_df = pd.DataFrame()

    try:
        risk_outputs_by_well = _stage41_compute_risk_curves(
            cfg=cfg,
            series_df=series_df,
            final_ensemble_df=final_ensemble_df,
            boundaries_df=boundaries_df,
            full_history_by_well=full_history_by_well,
        )
        risk_df = _flatten_risk_outputs_map(risk_outputs_by_well)
    except Exception as e:
        warn(f"[Phase 4] Risk stage failed softly: {e}")

    artifacts = dict(
        champions_df=champions_df,
        series_df=series_df,
        plot_series_df=plot_series_df,
        intra_family_df=intra_family_df,
        final_ensemble_df=final_ensemble_df,
        boundaries_df=boundaries_df,
        full_history_by_well=full_history_by_well,
        risk_outputs_by_well=risk_outputs_by_well,
        risk_df=risk_df,
    )
    ok("[Phase 4] Downstream stages complete.")
    return artifacts


def run_phase4_until_stage2(
    cfg: Phase4Config,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Public helper for notebooks / ablations.

    Runs Phase 4 only until the end of Stage 2:
      - preflight
      - Stage 1 champion selection
      - Stage 2 series loading

    Returns a small artifact bundle that can be replayed many times with
    different anchoring policies without reloading the store.
    """
    from common.log_utils import is_compact_logging, install_compact_filters, ok, info

    cfg_work = _clone_phase4_cfg(cfg)

    paths = _resolve_paths(cfg_work)
    deps = _import_runtime_deps()
    deps["configure_logging"]()

    if is_compact_logging(cfg_work):
        from common.log_utils import COMPACT_DEFAULT_PATTERNS
        install_compact_filters(COMPACT_DEFAULT_PATTERNS)

    # ---------- normalize filters ----------
    _filters = _normalize_filters(filters)
    if _filters:
        ds = _filters.get("dataset")
        wl = _filters.get("well")
        if wl and hasattr(cfg_work, "wells_to_analyze") and cfg_work.wells_to_analyze:
            cfg_work.wells_to_analyze = [str(wl)]
        info(
            f"[Phase 4] Applying filters: { {k: v for k, v in _filters.items()} }"
            + (" (note: dataset filtering happens post Stage 1)" if ds else "")
        )

    _ = _preflight_block(paths, cfg_work)

    champions_df, score_col, posthoc = _stage1_select_champions(
        cfg=cfg_work,
        exp_root=paths["experiment_root"],
        deps=deps,
    )

    series_df, boundaries_df, full_history_by_well = _stage2_load_series(
        cfg=cfg_work,
        series_root=paths["series_store_root"],
        champions_df=champions_df,
        deps=deps,
    )

    out = dict(
        cfg_used=cfg_work,
        champions_df=champions_df,
        score_col=score_col,
        posthoc=posthoc,
        series_df=series_df,
        boundaries_df=boundaries_df,
        full_history_by_well=full_history_by_well,
        paths=paths,
    )
    ok("[Phase 4] Stage-2 bundle ready for replay.")
    return out


def run_phase4_from_loaded_stage2(
    cfg: Phase4Config,
    *,
    champions_df: pd.DataFrame,
    series_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    full_history_by_well: Dict[str, pd.DataFrame],
    anchor_config: Optional[Any] = None,
    selected_policy_source: Optional[Any] = None,
    policy_overrides_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
    default_policy_name: Optional[str] = None,
    score_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Replay Phase 4 starting from already-loaded Stage 2 artifacts.

    Supported replay modes:
      - global policy: anchor_config="trend_strict"
      - selected map: selected_policy_source=".../selected_anchor_policies.csv"

    Flow:
      Stage 2.6 (optional) -> Stage 2.5 -> Stage 3 -> Stage 4 -> Stage 4.1
    """
    from common.log_utils import (
        is_compact_logging,
        install_compact_filters,
        ok,
        warn,
    )

    cfg_work = _clone_phase4_cfg(cfg)
    deps = _import_runtime_deps()
    deps["configure_logging"]()

    if is_compact_logging(cfg_work):
        from common.log_utils import COMPACT_DEFAULT_PATTERNS
        install_compact_filters(COMPACT_DEFAULT_PATTERNS)

    if series_df is None or series_df.empty:
        warn("[Phase 4 Replay] Skipped: series_df is empty.")
        return dict(
            champions_df=champions_df,
            series_df=pd.DataFrame(),
            plot_series_df=pd.DataFrame(),
            intra_family_df=pd.DataFrame(),
            final_ensemble_df=pd.DataFrame(),
            boundaries_df=boundaries_df,
            full_history_by_well=full_history_by_well,
            risk_outputs_by_well={},
            risk_df=pd.DataFrame(),
            anchor_config=anchor_config,
            selected_policy_source=selected_policy_source,
        )

    # ------------------------------------------------------------------
    # Stage 2.6 — optional history anchoring replay
    # ------------------------------------------------------------------
    anchor_cfg_resolved = None
    series_work = series_df.copy()

    try:
        if selected_policy_source is not None:
            from postproc_bias.anchoring import apply_history_anchoring_by_selected_policy

            series_work = apply_history_anchoring_by_selected_policy(
                series_df=series_work,
                boundaries_df=boundaries_df,
                full_history_by_well=full_history_by_well,
                selected_policy_source=selected_policy_source,
                policy_overrides_by_name=policy_overrides_by_name,
                default_policy_name=default_policy_name,
            )
            anchor_cfg_resolved = {
                "mode": "selected_map",
                "selected_policy_source": str(selected_policy_source),
                "default_policy_name": default_policy_name,
            }
            ok("[Phase 4 Replay] Stage 2.6 applied with selected-policy map.")

        elif anchor_config is not None:
            from postproc_bias.anchoring import (
                apply_history_anchoring,
                resolve_history_anchor_config,
                HistoryAnchorConfig,
            )

            if isinstance(anchor_config, str):
                anchor_cfg_resolved = resolve_history_anchor_config(anchor_config)

            elif isinstance(anchor_config, dict):
                policy_name = str(anchor_config.get("policy_name", "baseline_default"))
                overrides = {k: v for k, v in anchor_config.items() if k != "policy_name"}
                anchor_cfg_resolved = resolve_history_anchor_config(
                    policy_name=policy_name,
                    overrides=overrides or None,
                )

            elif isinstance(anchor_config, HistoryAnchorConfig):
                anchor_cfg_resolved = anchor_config

            else:
                raise TypeError(
                    "anchor_config must be one of: str | dict | HistoryAnchorConfig"
                )

            series_work = apply_history_anchoring(
                series_df=series_work,
                boundaries_df=boundaries_df,
                full_history_by_well=full_history_by_well,
                config=anchor_cfg_resolved,
            )
            ok(
                "[Phase 4 Replay] Stage 2.6 applied with policy=%s",
                getattr(anchor_cfg_resolved, "policy_name", "unknown"),
            )

    except Exception as e:
        warn(f"[Phase 4 Replay] Stage 2.6 failed softly: {e}")
        anchor_cfg_resolved = anchor_config if selected_policy_source is None else {
            "mode": "selected_map_failed",
            "selected_policy_source": str(selected_policy_source),
        }

    if not score_col:
        score_col = (
            "robust_score"
            if str(getattr(cfg_work, "scoring_strategy", "weighted_score")) == "robust_score"
            else "weighted_score"
        )

    artifacts = _run_phase4_downstream_from_series(
        cfg=cfg_work,
        champions_df=champions_df,
        series_df=series_work,
        boundaries_df=boundaries_df,
        full_history_by_well=full_history_by_well,
        deps=deps,
        score_col=score_col,
    )
    artifacts["anchor_config"] = anchor_cfg_resolved
    artifacts["selected_policy_source"] = selected_policy_source
    ok("[Phase 4 Replay] Replay complete.")
    return artifacts

# ---------------------------
# Orchestrator (refactored)
# ---------------------------
def run_phase4_pipeline(
    cfg: Phase4Config,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Phase 4 pipeline – compact-friendly, bias-free main path.

    Main path:
    - Stage 1: champion selection
    - Stage 2: series loading
    - Stage 2.5: train restriction for plotting
    - Stage 3: intra/inter ensemble
    - Stage 4: visualization/report
    - Stage 4.1: risk computation

    Returns:
        artifacts dict including flattened 'risk_df'
    """
    from common.log_utils import (
        is_compact_logging,
        install_compact_filters,
        info,
        ok,
    )

    cfg_work = _clone_phase4_cfg(cfg)

    paths = _resolve_paths(cfg_work)
    deps = _import_runtime_deps()
    deps["configure_logging"]()

    if is_compact_logging(cfg_work):
        from common.log_utils import COMPACT_DEFAULT_PATTERNS
        install_compact_filters(COMPACT_DEFAULT_PATTERNS)

    # ---------- normalize filters ----------
    _filters = _normalize_filters(filters)
    if _filters:
        ds = _filters.get("dataset")
        wl = _filters.get("well")
        if wl and hasattr(cfg_work, "wells_to_analyze") and cfg_work.wells_to_analyze:
            cfg_work.wells_to_analyze = [str(wl)]
        info(
            f"[Phase 4] Applying filters: { {k: v for k, v in _filters.items()} }"
            + (" (note: dataset filtering happens post Stage 1)" if ds else "")
        )

    exp_root = paths["experiment_root"]
    series_root = paths["series_store_root"]

    _ = _preflight_block(paths, cfg_work)

    champions_df, score_col, posthoc = _stage1_select_champions(
        cfg=cfg_work,
        exp_root=exp_root,
        deps=deps,
    )

    series_df, boundaries_df, full_history_by_well = _stage2_load_series(
        cfg=cfg_work,
        series_root=series_root,
        champions_df=champions_df,
        deps=deps,
    )

    artifacts = _run_phase4_downstream_from_series(
        cfg=cfg_work,
        champions_df=champions_df,
        series_df=series_df,
        boundaries_df=boundaries_df,
        full_history_by_well=full_history_by_well,
        deps=deps,
        score_col=score_col,
    )
    ok("[Phase 4] Pipeline complete.")
    return artifacts



def run_phase4_compare(
    cfg: Phase4Config,
    *,
    filters: Optional[Dict[str, Any]] = None,
    replay_policy_name: Optional[str] = None,
    replay_policy_overrides: Optional[Dict[str, Any]] = None,
    selected_policy_source: Optional[Any] = None,
    policy_overrides_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
    default_policy_name: Optional[str] = None,
    show_standard_plots: bool = False,
    show_replay_plots: bool = True,
) -> Dict[str, Any]:
    """
    Compare legacy Phase 4 vs replay Phase 4 on the exact same Stage-2 bundle.

    Supported replay modes:
      - global policy: replay_policy_name="trend_strict"
      - selected map: selected_policy_source=".../selected_anchor_policies.csv"
    """
    from common.log_utils import ok

    if replay_policy_name is None and selected_policy_source is None:
        replay_policy_name = "baseline_default"

    # Load Stage 2 once
    base_bundle = run_phase4_until_stage2(cfg, filters=filters)

    champions_df = base_bundle["champions_df"]
    series_df = base_bundle["series_df"]
    boundaries_df = base_bundle["boundaries_df"]
    full_history_by_well = base_bundle["full_history_by_well"]
    score_col = base_bundle.get("score_col")

    # standard arm
    cfg_standard = _clone_phase4_cfg(base_bundle["cfg_used"])
    cfg_standard.enable_plots = bool(show_standard_plots)

    deps_standard = _import_runtime_deps()
    deps_standard["configure_logging"]()

    standard_artifacts = _run_phase4_downstream_from_series(
        cfg=cfg_standard,
        champions_df=champions_df,
        series_df=series_df,
        boundaries_df=boundaries_df,
        full_history_by_well=full_history_by_well,
        deps=deps_standard,
        score_col=score_col,
    )
    standard_artifacts["anchor_config"] = None

    # replay arm
    cfg_replay = _clone_phase4_cfg(base_bundle["cfg_used"])
    cfg_replay.enable_plots = bool(show_replay_plots)

    if selected_policy_source is not None:
        replay_artifacts = run_phase4_from_loaded_stage2(
            cfg=cfg_replay,
            champions_df=champions_df,
            series_df=series_df,
            boundaries_df=boundaries_df,
            full_history_by_well=full_history_by_well,
            selected_policy_source=selected_policy_source,
            policy_overrides_by_name=policy_overrides_by_name,
            default_policy_name=default_policy_name,
            score_col=score_col,
        )
        replay_mode = "selected_map"
    else:
        if replay_policy_overrides:
            anchor_config = {"policy_name": str(replay_policy_name), **dict(replay_policy_overrides)}
        else:
            anchor_config = str(replay_policy_name)

        replay_artifacts = run_phase4_from_loaded_stage2(
            cfg=cfg_replay,
            champions_df=champions_df,
            series_df=series_df,
            boundaries_df=boundaries_df,
            full_history_by_well=full_history_by_well,
            anchor_config=anchor_config,
            score_col=score_col,
        )
        replay_mode = "global_policy"

    ok("[Phase 4 Compare] Standard and replay arms complete.")
    return dict(
        stage2_bundle=base_bundle,
        standard=standard_artifacts,
        replay=replay_artifacts,
        replay_mode=replay_mode,
        replay_policy_name=str(replay_policy_name) if replay_policy_name is not None else None,
        selected_policy_source=selected_policy_source,
    )
