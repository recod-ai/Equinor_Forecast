# src/common/log_utils.py
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import pandas as pd
import re
from dataclasses import dataclass

# -------------------------------------------------------------------------------------------------
# Public logging façade (emoji-friendly, minimal and consistent)
# -------------------------------------------------------------------------------------------------

def stage_banner(n: Union[int, str], title: str, subtitle: str = "", width: int = 92) -> None:
    """
    Draw a consistent stage banner. Pure logging—no side effects.
    """
    w = _safe_width(width)
    logging.info("\n" + "═" * w)
    logging.info(f"  🚀  Stage {n}: {title}")
    if subtitle:
        logging.info(f"      {subtitle}")
    logging.info("═" * w)

# src/common/log_utils.py

def ok(msg: str, *args, **kwargs) -> None:
    logging.info("✅ " + str(msg), *args, **kwargs)

def warn(msg: str, *args, **kwargs) -> None:
    logging.warning("⚠️  " + str(msg), *args, **kwargs)

def info(msg: str, *args, **kwargs) -> None:
    logging.info("ℹ️  " + str(msg), *args, **kwargs)

def err(msg: str, *args, **kwargs) -> None:
    logging.error("❌ " + str(msg), *args, **kwargs)


# -------------------------------------------------------------------------------------------------
# Compact-mode controls (no API change to callers)
# - You can drive this via cfg.log_mode / cfg.log_width OR env vars:
#     PHASE_LOG_MODE  ∈ {compact, verbose}   (default: verbose)
#     PHASE_LOG_WIDTH ∈ int                  (default: 100)
# -------------------------------------------------------------------------------------------------

# --- Verbosity helpers -------------------------------------------------------
from contextlib import contextmanager

def vlog(enabled: bool, message: str) -> None:
    """Verbose-only info log."""
    if enabled:
        logging.info(message)


# Verbose-only helpers (no-op if compact)
def info_v(msg: str, *args, **kwargs) -> None:
    if not is_compact_logging():
        info(msg, *args, **kwargs)

def debug_v(msg: str, *args, **kwargs) -> None:
    if not is_compact_logging():
        logging.debug(msg, *args, **kwargs)


@contextmanager
def silence_loggers(names: Sequence[str], level: int = logging.WARNING):
    """
    Temporarily raise the level of noisy third-party loggers.
    Usage:
        with silence_loggers(["hpo.distribution_filter", "robust_filter"], logging.WARNING):
            ... # code that triggers noisy logs
    """
    # keep (logger, old_level)
    saved = []
    try:
        for name in (names or []):
            lg = logging.getLogger(name)
            saved.append((lg, lg.level))
            lg.setLevel(level)
        yield
    finally:
        for lg, old in saved:
            try:
                lg.setLevel(old)
            except Exception:
                pass

def parse_silenced_from_env(env_var: str = "PHASE_SILENCE_LOGGERS") -> List[str]:
    """
    Accept comma/semicolon/space-separated list of logger names from env.
    Example: export PHASE_SILENCE_LOGGERS="hpo.distribution_filter, robust_filter, series_reader"
    """
    raw = os.environ.get(env_var, "") or ""
    parts = [p.strip() for p in re.split(r"[,\s;]+", raw) if p.strip()]
    return parts


def is_compact_logging(cfg: Optional[Any] = None) -> bool:
    """
    Returns True if compact blocks should be printed.
    Priority: cfg.log_mode -> env PHASE_LOG_MODE -> 'verbose'
    """
    mode = None
    if cfg is not None:
        mode = getattr(cfg, "log_mode", None)
    if not mode:
        mode = os.environ.get("PHASE_LOG_MODE", "compact")
    return str(mode).lower() == "compact"

def effective_log_width(cfg: Optional[Any] = None, fallback: int = 100) -> int:
    """
    Returns the desired block width.
    Priority: cfg.log_width -> env PHASE_LOG_WIDTH -> fallback
    """
    if cfg is not None:
        try:
            w = int(getattr(cfg, "log_width", fallback))
            return _safe_width(w)
        except Exception:
            pass
    try:
        return _safe_width(int(os.environ.get("PHASE_LOG_WIDTH", fallback)))
    except Exception:
        return _safe_width(fallback)

# -------------------------------------------------------------------------------------------------
# Block renderers (boxed/compact)
# -------------------------------------------------------------------------------------------------

def log_block(title: str, lines: Sequence[str], *, level: int = logging.INFO, width: int = 100) -> None:
    """
    Print a framed block with a centered title and body lines.
    """
    log = logging.getLogger()
    w = _safe_width(width)
    top = f"┌{'─' * (w - 2)}┐"
    mid = f"├{'─' * (w - 2)}┤"
    bot = f"└{'─' * (w - 2)}┘"

    def pad(s: str) -> str:
        s = _to_line(s, w - 4)
        return f"│ {s.ljust(w - 4)} │"

    log.log(level, top)
    log.log(level, pad(title.center(w - 4)))
    log.log(level, mid)
    for ln in (lines or []):
        log.log(level, pad(str(ln)))
    log.log(level, bot)

def log_kv_block(title: str, kv: Mapping[str, Any], *, level: int = logging.INFO, width: int = 100,
                 key_pad: Optional[int] = None) -> None:
    """
    Convenience renderer for short key=value summaries.
    """
    kv = kv or {}
    if not kv:
        log_block(title, ["(empty)"], level=level, width=width)
        return

    key_pad = key_pad or max(len(str(k)) for k in kv.keys())
    lines = [f"{str(k).rjust(key_pad)}: {kv[k]}" for k in kv]
    log_block(title, lines, level=level, width=width)

def log_table_block(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]],
                    *, level: int = logging.INFO, width: int = 100) -> None:
    """
    Simple monospace table (no external deps). Good for short matrices.
    """
    w = _safe_width(width)
    cols = [str(h) for h in headers]
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]

    # compute column widths
    col_w = [len(h) for h in cols]
    for r in rows:
        for i, c in enumerate(r):
            col_w[i] = max(col_w[i], len(c))

    def fmt_row(cells: Sequence[str]) -> str:
        parts = [c.ljust(col_w[i]) for i, c in enumerate(cells)]
        one = " | ".join(parts)
        return _to_line(one, w - 4)

    body = [fmt_row(cols)] + [fmt_row(["-" * cw for cw in col_w])] + [fmt_row(r) for r in rows]
    log_block(title, body, level=level, width=w)

# -------------------------------------------------------------------------------------------------
# Stage-specific summarizers already used in Phase 4 (kept generic enough)
# -------------------------------------------------------------------------------------------------

def summarize_preflight(paths: Dict[str, Path], cfg: Phase4Config, *, manifest_found: bool, probe_well: Optional[str], probe_found: Optional[bool]) -> list[str]:
    lines = []
    lines.append(f"series_store_root: {paths['series_store_root']}")
    lines.append(f"boundaries.parquet: {'FOUND' if manifest_found else 'NOT FOUND'}")
    if probe_well is not None and probe_found is not None:
        lines.append(f"history probe (well={probe_well}): {'FOUND' if probe_found else 'NOT FOUND'}")
    return lines

# ---------------------------
# Summarizers compact 
# ---------------------------
def summarize_stage1(champions_df: pd.DataFrame, posthoc: Dict[str, Any], score_col: str) -> list[str]:
    if champions_df is None or champions_df.empty:
        return ["champions: 0"]
    wells = champions_df["well"].nunique(dropna=False) if "well" in champions_df.columns else None
    total = len(champions_df)
    topk = posthoc.get("top_strategies_per_well", "?")
    arch_counts = {}
    if "arch" in champions_df.columns:
        arch_counts = champions_df["arch"].astype(str).value_counts().to_dict()
    arch_summary = " | ".join(f"{k}={fmt_int(v)}" for k, v in arch_counts.items()) if arch_counts else "n/a"

    lines = [
        f"score: {score_col} | top_k per (well, arch): {topk}",
        f"wells: {fmt_int(wells)} | champions: {fmt_int(total)} | arch split: {arch_summary}",
        f"selection strategy: {posthoc.get('selection_strategy', 'n/a')} | valcum: q∈[{posthoc.get('valcum_gate',{}).get('q_low','?')}, {posthoc.get('valcum_gate',{}).get('q_high','?')}]",
    ]
    return lines



def summarize_stage2(
    *,
    series_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    full_history_by_well: Dict[str, pd.DataFrame],
) -> list[str]:
    uniq_jobs = (series_df["job_hash"].nunique() if "job_hash" in getattr(series_df, "columns", []) else 0) if not series_df.empty else 0
    wells_hist = len(full_history_by_well or {})
    wells_bounds = 0 if boundaries_df is None or boundaries_df.empty else boundaries_df["well"].nunique()
    rows = 0 if series_df is None or series_df.empty else len(series_df)
    return [
        f"Rows loaded: {rows:,}",
        f"Unique champions (job_hash): {uniq_jobs}",
        f"Wells with history: {wells_hist}",
        f"Wells with boundaries: {wells_bounds}",
        "Columns: " + (", ".join(list(series_df.columns)[:10]) + (" …" if len(series_df.columns) > 10 else ""))
            if isinstance(series_df, pd.DataFrame) and not series_df.empty else "Columns: ",
    ]


def summarize_stage25(
    *,
    before_rows: Optional[int],
    after_rows: Optional[int],
    before_train: Optional[int],
    after_train: Optional[int],
    kept_wells: int,
    primary_by_well: Dict[str, str],
) -> list[str]:
    lines = [
        f"Wells with primary chosen: {kept_wells}",
        f"Rows: {before_rows:,} → {after_rows:,} (Δ={(after_rows or 0) - (before_rows or 0):,})",
    ]
    if before_train is not None and after_train is not None:
        lines.append(f"TRAIN rows: {before_train:,} → {after_train:,} (Δ={(after_train or 0) - (before_train or 0):,})")
    if primary_by_well:
        preview = ", ".join([f"{w}={j[:8]}" for w, j in list(primary_by_well.items())[:6]])
        if len(primary_by_well) > 6:
            preview += ", …"
        lines.append(f"Primary (preview): {preview}")
    return lines

# -------------------------------------------------------------------------------------------------
# Small formatting helpers (pure UI sugar)
# -------------------------------------------------------------------------------------------------
def fmt_int(x: Optional[int]) -> str:
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return str(x)

def _to_line(s: str, width: int) -> str:
    # non-breaking, trimmed line with ellipsis if needed
    if len(s) <= width:
        return s
    if width <= 1:
        return s[:width]
    return s[: max(0, width - 1)] + "…"

def _safe_width(w: int) -> int:
    try:
        w = int(w)
    except Exception:
        w = 80
    return max(40, min(200, w))  # keep it sane


# =============================================================================================
# Compact silencers (regex-based)
# =============================================================================================

@dataclass
class PatternSilencer(logging.Filter):
    """
    Filters out log records whose message matches ANY of the provided regex patterns.
    Case-insensitive by default. Safe to install multiple times; idempotent.
    """
    patterns: Sequence[re.Pattern]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for p in self.patterns:
            if p.search(msg):
                return False  # drop
        return True

# Default patterns that are noisy but not essential in compact mode
COMPACT_DEFAULT_PATTERNS: List[re.Pattern] = [
    # Stage 1 – robust gates spam
    re.compile(r"^Starting robust distribution filtering pipeline", re.I),
    re.compile(r"^\s*Analysing \d+ unique trials", re.I),
    re.compile(r"^\s*Computed quantile and MAD", re.I),
    re.compile(r"^valcum_gate:", re.I),
    re.compile(r"^\[survivors\]", re.I),
    re.compile(r"^\[pool\]", re.I),
    re.compile(r"^\[final\].*picked_final", re.I),
    re.compile(r"^✅ Loaded leaderboard for campaign", re.I),

    # Stage 2 – series reader internals
    re.compile(r"\[series_reader\]\s*Filled", re.I),
    re.compile(r"\[series_reader\]\s*Unique job_hash", re.I),

    # Stage 2.5 – train_restrict internals (keep only final 2 lines)
    re.compile(r"\[train_restrict\]\s*_compute_primary_per_well: champions_df shape", re.I),
    re.compile(r"\[train_restrict\]\s*_compute_primary_per_well: using score column", re.I),
    re.compile(r"\[train_restrict\]\s*_compute_primary_per_well: candidates per well", re.I),
    re.compile(r"\[train_restrict\]\s*_compute_primary_per_well: selected \d+ primary jobs", re.I),
    re.compile(r"\[train_restrict\]\s*_compute_primary_per_well: sample primary_by_well", re.I),
    re.compile(r"\[train_restrict\]\s*_restrict_train_to_primary: series_df shape", re.I),
    re.compile(r"\[train_restrict\]\s*_restrict_train_to_primary: split distribution before", re.I),
    re.compile(r"\[train_restrict\]\s*_restrict_train_to_primary: \d+ rows flagged as primary", re.I),
    re.compile(r"\[train_restrict\]\s*_restrict_train_to_primary: split distribution after", re.I),

    # Stage 4 – debug previews
    re.compile(r"^\[Stage 4\]\[dbg\]\s+", re.I),
    re.compile(r"^\[ensemble_plot\]\s+intra_family_df: well=.*→ shape=", re.I),
    re.compile(r"^\[ensemble_plot\]\s+champion_series_df: well=.*→ shape=", re.I),
    re.compile(r"^\[ensemble_plot\]\s+.*attached axis via global_idx", re.I),
    re.compile(r"^\[ensemble_plot\]\s+intra_family_df: VAL/TEST only", re.I),

    # Pandas/Series accidental prints
    re.compile(r"^\s*arch\s*$", re.I),
    re.compile(r"^\s*(arps|seq2)\s+\d+\s*$", re.I),
]



# Keep a single global instance reference so we can uninstall later if needed
__compact_filter_instance: Optional[PatternSilencer] = None

def install_compact_filters(patterns: Optional[Sequence[re.Pattern]] = None) -> None:
    """
    Install a regex-based filter on the root logger to suppress noisy INFO lines when in compact mode.
    Safe to call multiple times.
    """
    global __compact_filter_instance
    if __compact_filter_instance is not None:
        return  # already installed

    pats = list(patterns) if patterns is not None else COMPACT_DEFAULT_PATTERNS
    filt = PatternSilencer(patterns=pats)
    root = logging.getLogger()
    root.addFilter(filt)
    __compact_filter_instance = filt

def uninstall_compact_filters() -> None:
    """
    Remove the previously installed compact filter, if any.
    """
    global __compact_filter_instance
    if __compact_filter_instance is None:
        return
    root = logging.getLogger()
    try:
        root.removeFilter(__compact_filter_instance)
    except Exception:
        pass
    __compact_filter_instance = None


def build_champion_summary_table(
    champions_df: pd.DataFrame,
    score_col: str = "weighted_score",
) -> pd.DataFrame:
    """
    Returns a compact table: well, arch, n, score_min/median/mean.
    Safe if columns are missing.
    """
    import numpy as np
    import pandas as pd

    if champions_df is None or champions_df.empty:
        return pd.DataFrame(columns=["well", "arch", "n", "score_min", "score_median", "score_mean"])

    df = champions_df.copy()
    if "well" not in df.columns:
        df["well"] = "?"
    if "arch" not in df.columns:
        df["arch"] = (df.get("architecture_name") or "generic")
    if score_col not in df.columns:
        # fall back to a common metric if score is absent
        score_col = next((c for c in ["robust_score", "val_smape_agg", "val_smape_cum"] if c in df.columns), None)

    grp = df.groupby(["well", "arch"], dropna=False)
    if score_col is None:
        out = grp.size().reset_index(name="n")
        out["score_min"] = out["score_median"] = out["score_mean"] = np.nan
        return out.sort_values(["well", "arch"]).reset_index(drop=True)

    agg = grp[score_col].agg(["count", "min", "median", "mean"]).reset_index()
    agg.columns = ["well", "arch", "n", "score_min", "score_median", "score_mean"]
    return agg.sort_values(["well", "arch"]).reset_index(drop=True)



def log_champion_summary_box(
    champions_df: pd.DataFrame,
    *,
    score_col: str,
    title: str = "Stage 1 — Champions (Summary)",
    width: int = 100,
) -> None:
    """
    Prints a single compact box with total rows, wells, archs, score column
    and a short preview of top counts per (well, arch).
    """
    import pandas as pd

    lines = []
    rows = 0 if champions_df is None or champions_df.empty else len(champions_df)
    wells = champions_df["well"].nunique(dropna=False) if isinstance(champions_df, pd.DataFrame) and "well" in champions_df.columns else 0
    archs = champions_df["arch"].nunique(dropna=False) if isinstance(champions_df, pd.DataFrame) and "arch" in champions_df.columns else 0

    lines.append(f"Rows: {fmt_int(rows)} | Wells: {fmt_int(wells)} | Archs: {fmt_int(archs)}")
    lines.append(f"Score column: {score_col}")

    try:
        top_counts = (
            champions_df.groupby(["well", "arch"])
            .size().reset_index(name="n")
            .sort_values(["well", "arch", "n"], ascending=[True, True, False])
        )
        preview = ", ".join([f"{r['well']}/{r['arch']}={int(r['n'])}" for _, r in top_counts.head(8).iterrows()])
        if len(top_counts) > 8: preview += ", …"
        lines.append(f"Top counts (preview): {preview}")
    except Exception:
        pass

    log_block(title, lines, width=effective_log_width(None, fallback=width))


# ---- report helpers (plug & play) -------------------------------------------------------------

def log_phase4_compact_report(
    *,
    final_ensemble_df: pd.DataFrame,
    intra_family_df: pd.DataFrame,
    full_history_by_well: Dict[str, pd.DataFrame],
    boundaries_df: Optional[pd.DataFrame],
    cfg: Any,
) -> None:
    """
    One-shot compact recap:
      • Inter table (post-ensemble mean) with % errors
      • Intra table (family means) with % errors
      • 3-line KV footer (wells, members, medians)
    Uses the same summarizers the plot uses, so numbers match the figures.
    """
    from common.log_table import (
        build_table_from_plot_metrics,
        build_table_from_intra_metrics,
        render_rich_table,
    )
    width = effective_log_width(cfg, fallback=100)

    # Inter
    inter_tbl = build_table_from_plot_metrics(
        final_ensemble_df=final_ensemble_df,
        full_history_by_well=full_history_by_well,
        boundaries_df=boundaries_df,
        ci_mode=getattr(cfg, "ci_mode", "sem"),
        coverage=getattr(cfg, "coverage_level", 0.90),
    )
    render_rich_table(
        inter_tbl,
        title="Inter-Family — Final Ensemble (Post-ensemble Metrics)",
        decimals=2,
        bar_width=18,
        bar_track=False,
        show_lines=True,
        header_style="bold black on white",
    )

    # Intra
    intra_tbl = build_table_from_intra_metrics(
        intra_family_df=intra_family_df,
        full_history_by_well=full_history_by_well,
        boundaries_df=boundaries_df,
        ci_mode=getattr(cfg, "ci_mode", "sem"),
        coverage=getattr(cfg, "coverage_level", 0.90),
        include_family_column=True,
    )
    render_rich_table(
        intra_tbl,
        title="Intra-Family — Family Means (Post-ensemble Metrics)",
        decimals=2,
        bar_width=18,
        bar_track=False,
        show_lines=True,
        header_style="bold black on white",
    )

    # Footer: tiny KV
    try:
        wells = inter_tbl["Well"].nunique(dropna=False) if not inter_tbl.empty else 0
        mems  = int(pd.to_numeric(inter_tbl.get("Members", pd.Series()), errors="coerce").median()) if not inter_tbl.empty else 0
        vmed  = float(pd.to_numeric(inter_tbl.get("Validation Error (%)", pd.Series()).str.replace("%","", regex=False), errors="coerce").median()) if not inter_tbl.empty else float("nan")
        tmed  = float(pd.to_numeric(inter_tbl.get("Test Error (%)", pd.Series()).str.replace("%","", regex=False), errors="coerce").median()) if not inter_tbl.empty else float("nan")
        log_kv_block(
            "Phase 4 — Compact Recap",
            {
                "Wells": wells,
                "Median Members": mems,
                "Median Val Error (%)": f"{vmed:.2f}" if np.isfinite(vmed) else "–",
                "Median Test Error (%)": f"{tmed:.2f}" if np.isfinite(tmed) else "–",
            },
            width=width,
        )
    except Exception:
        pass


import numpy as np
import pandas as pd
from typing import Iterable, Sequence, Optional

def build_metric_table(
    df: pd.DataFrame,
    metrics: Sequence[str] = ("val_smape_agg","val_smape_cum","test_smape_agg","test_smape_cum"),
    groupby: Sequence[str] = ("well","arch"),
    agg: str = "median",
) -> pd.DataFrame:
    """
    Builds a compact table grouped by `groupby` with aggregator over `metrics`.
    Missing metrics are ignored. Returns stable columns: groupby + <metric>_<agg>.
    """
    if df is None or df.empty:
        cols = list(groupby) + [f"{m}_{agg}" for m in metrics]
        return pd.DataFrame(columns=cols)

    # keep only metrics that exist
    metrics_eff = [m for m in metrics if m in df.columns]
    if not metrics_eff:
        return pd.DataFrame(columns=list(groupby))

    g = df.groupby(list(groupby), dropna=False)
    agg_map = {m: agg for m in metrics_eff}
    out = g.agg(agg_map).reset_index()
    out = out.rename(columns={m: f"{m}_{agg}" for m in metrics_eff})
    # order columns nicely
    cols = list(groupby) + [f"{m}_{agg}" for m in metrics_eff]
    return out[cols].sort_values(list(groupby)).reset_index(drop=True)

def log_metric_table_block(
    title: str,
    metric_tbl: pd.DataFrame,
    *,
    max_rows: int = 12,
    width: int = 100,
) -> None:
    """
    Renders a short table block with the first `max_rows` rows.
    """
    from common.log_utils import log_table_block

    if metric_tbl is None or metric_tbl.empty:
        log_table_block(title, ["(empty)"], [], width=width)
        return

    headers = list(metric_tbl.columns)
    rows = metric_tbl.head(max_rows).values.tolist()
    log_table_block(title, headers, rows, width=width)

# --- Compact section (turnkey) ---------------------------------------------------------------
from contextlib import contextmanager

@contextmanager
def compact_section(
    enable: bool,
    *,
    silence: Sequence[str] = (),
    level: int = logging.WARNING,
    patterns: Optional[Sequence[re.Pattern]] = None,
):
    """
    Run a block with compact logging:
      - temporarily installs regex filters (drop noisy INFO lines),
      - temporarily bumps named loggers to `level` (default WARNING).
    Safe to nest; always restores previous state.
    """
    if not enable:
        # no-op
        yield
        return

    # 1) install regex filter (idempotent)
    install_compact_filters(patterns)

    # 2) raise levels on listed loggers
    saved = []
    try:
        for name in (silence or ()):
            lg = logging.getLogger(name)
            saved.append((lg, lg.level))
            lg.setLevel(level)
        yield
    finally:
        for lg, old in saved:
            try:
                lg.setLevel(old)
            except Exception:
                pass



# src/common/log_utils.py  (adições pequenas)

import hashlib
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence

def _hash_str(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def df_fingerprint(df: Optional[pd.DataFrame], key_cols: Sequence[str] = ("well","arch","split")) -> Dict[str, Any]:
    if df is None:
        return {"df": "None"}
    if df.empty:
        return {"rows": 0, "cols": 0}

    cols = list(df.columns)
    present_keys = [c for c in key_cols if c in cols]
    head_sig = ""
    try:
        head = df[present_keys].head(20).astype(str).to_csv(index=False)
        head_sig = _hash_str(head)
    except Exception:
        head_sig = "na"

    return {
        "rows": len(df),
        "cols": len(cols),
        "col0": cols[0] if cols else "na",
        "keys": ",".join(present_keys) if present_keys else "na",
        "head_sig": head_sig,
    }

def arr_fingerprint(a: Any) -> Dict[str, Any]:
    if a is None:
        return {"arr": "None"}
    a = np.asarray(a)
    return {
        "shape": tuple(a.shape),
        "dtype": str(a.dtype),
        "nan%": float(np.isnan(a).mean() * 100.0) if np.issubdtype(a.dtype, np.floating) else 0.0,
        "min": float(np.nanmin(a)) if a.size else None,
        "med": float(np.nanmedian(a)) if a.size else None,
        "max": float(np.nanmax(a)) if a.size else None,
    }
