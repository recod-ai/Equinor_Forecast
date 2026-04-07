# src/common/log_table.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

import numpy as np
import pandas as pd

# Plotting metrics/helpers (fonte única de verdade)
from plotting.ensemble_utils import only_val_test
from plotting.ensemble_stats import summarize_split_metrics_std_or_sem

# Rich (console tables)
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text



# ============================================================================================
# 1) Metrics helpers (used only if we need to derive post-ensemble metrics on the fly)
# ============================================================================================

def smape(y_true: pd.Series, y_pred: pd.Series, *, eps: float = 1e-12) -> float:
    """
    Symmetric MAPE in percent (%). Used only if we must derive metrics from raw series.
    If you already have 'smape_val'/'smape_test' columns computed elsewhere, you won't need this.
    """
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    m = (~y_true.isna()) & (~y_pred.isna())
    if not m.any():
        return np.nan
    yt = y_true[m].astype(float)
    yp = y_pred[m].astype(float)
    denom = (np.abs(yt) + np.abs(yp)).clip(lower=eps)
    return float(np.mean(np.abs(yp - yt) / denom) * 200.0)  # 2x for symmetric, ×100 to percent


# ============================================================================================
# 2) Builders: choose your source of truth
# ============================================================================================

def build_table_from_final_ensemble(
    final_ensemble_df: pd.DataFrame,
    *,
    split_col: str = "split",
    well_col: str = "well",
    ytrue_col: str = "ytrue",
    yhat_col: str = "yhat_final_mean",
    include_splits: Sequence[str] = ("val", "test"),
    attach_members: Optional[pd.DataFrame] = None,  # e.g. final_ensemble_df with 'n_members'
    members_key: str = "n_members",
) -> pd.DataFrame:
    """
    Build a per-well table computing SMAPE from the *post-ensemble* series (yhat_final_mean vs ytrue).
    This represents the final, aggregated performance.

    Returns columns: [well, n_rows, smape_val?, smape_test?, n_members?]
    """
    if final_ensemble_df is None or final_ensemble_df.empty:
        return pd.DataFrame(columns=[well_col, "n_rows", "smape_val", "smape_test"])

    need = {split_col, well_col, ytrue_col, yhat_col}
    missing = [c for c in need if c not in final_ensemble_df.columns]
    if missing:
        # Shape-stable empty table
        return pd.DataFrame(columns=[well_col, "n_rows", "smape_val", "smape_test"])

    df = final_ensemble_df.copy()

    # Compute SMAPE per well × split
    parts: List[pd.DataFrame] = []
    for sp in include_splits:
        sub = df[df[split_col] == sp]
        if sub.empty:
            parts.append(pd.DataFrame({well_col: [], f"smape_{sp}": []}))
            continue
        agg = (
            sub.groupby(well_col, dropna=False)
               .apply(lambda g: smape(g[ytrue_col], g[yhat_col]))
               .reset_index(name=f"smape_{sp}")
        )
        parts.append(agg)

    out = None
    for p in parts:
        out = p if out is None else out.merge(p, on=well_col, how="outer")

    # Row counts over selected splits
    counts = (
        df[df[split_col].isin(include_splits)]
        .groupby(well_col, dropna=False).size().reset_index(name="n_rows")
    )
    out = counts.merge(out, on=well_col, how="left") if out is not None else counts

    # Attach n_members (median per well), if available
    if attach_members is not None and not attach_members.empty and members_key in attach_members.columns:
        nm = (
            attach_members.groupby(well_col, dropna=False)[members_key]
            .median().reset_index(name=members_key)
        )
        out = out.merge(nm, on=well_col, how="left")

    # Order columns
    cols = [well_col, "n_rows"] + [c for c in out.columns if c.startswith("smape_")]
    if members_key in out.columns:
        cols.append(members_key)
    out = out[cols].sort_values(well_col).reset_index(drop=True)
    return out

# ============================================================================================
# 2.1) Builder usando o MESMO método do plot (consistência 100%)
# ============================================================================================



def _fallback_boundaries_from_hist(full_hist_df: pd.DataFrame) -> Dict[str, int]:
    if full_hist_df is None or full_hist_df.empty:
        return {"train_end": 0, "val_end": 0, "test_start": 1}
    n = len(full_hist_df)
    tr = n - 1
    return {"train_end": tr, "val_end": tr, "test_start": tr + 1}

def _manifest_bounds_for(
    boundaries_df: Optional[pd.DataFrame],
    well: str
) -> Optional[Dict[str, int]]:
    if boundaries_df is None or getattr(boundaries_df, "empty", True):
        return None
    if "well" not in boundaries_df.columns:
        return None
    sub = boundaries_df[boundaries_df["well"].astype(str) == str(well)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    out = {k: int(r.get(k)) for k in ("train_end", "val_end", "test_start") if k in r}
    if not out or any(pd.isna(v) for v in out.values()):
        return None
    return out

def _align_like_plot(
    df: pd.DataFrame,
    full_hist_df: pd.DataFrame,
    boundaries: Dict[str, int],
    *,
    split_col: str = "split"
) -> pd.DataFrame:
    """Replica o alinhamento do plot (global_idx → t)."""
    if df is None or df.empty:
        return df
    local = df.copy()
    if "idx" not in local.columns:
        return local  # nada a alinhar

    # offsets iguais aos do plot
    train_end  = int(boundaries.get("train_end", 0))
    val_end    = int(boundaries.get("val_end", train_end))
    test_start = int(boundaries.get("test_start", val_end + 1))
    VAL_OFFSET  = train_end + 1
    TEST_OFFSET = max(val_end, test_start) + 1

    split = local.get(split_col, "train").astype(str).str.lower() if split_col in local.columns else "train"
    local["global_idx"] = np.where(
        split.isin(["val", "validation"]), local["idx"].astype(int) + VAL_OFFSET,
        np.where(split.eq("test"),         local["idx"].astype(int) + TEST_OFFSET,
                                           local["idx"].astype(int))
    )

    # mapa de tempo a partir do histórico (mesmo eixo do plot)
    t_map = full_hist_df.reset_index().rename(columns={"index": "global_idx"})
    if "global_idx" not in t_map.columns:
        return local
    t_map = t_map[["global_idx", "t"]].drop_duplicates("global_idx")
    out = local.merge(t_map, on="global_idx", how="inner")
    return out

def build_table_from_plot_metrics(
    final_ensemble_df: pd.DataFrame,
    full_history_by_well: Dict[str, pd.DataFrame],
    *,
    boundaries_df: Optional[pd.DataFrame] = None,
    well_col: str = "well",
    split_col: str = "split",
    n_members_col: str = "n_members",
    ci_mode: str = "sem",
    coverage: float = 0.90,
    # presentation
    use_alias_headers: bool = True,
) -> pd.DataFrame:
    """
    Inter-family table (uses final_ensemble_df). Aligns like the plot and summarizes per well.
    Returns columns (raw): [well, n_rows, n_members, smape_val, smape_test]
    If use_alias_headers=True, columns are aliased to:
      Well | Rows | Members | Validation Error (%) | Test Error (%)
    """
    raw_cols = [well_col, "n_rows", "n_members", "smape_val", "smape_test"]
    if final_ensemble_df is None or final_ensemble_df.empty or well_col not in final_ensemble_df.columns:
        out = pd.DataFrame(columns=raw_cols)
        return _alias_columns(out) if use_alias_headers else out

    rows: List[Dict[str, Any]] = []
    for w, sub in final_ensemble_df.groupby(final_ensemble_df[well_col].astype(str)):
        fh = full_history_by_well.get(str(w))
        if fh is None or getattr(fh, "empty", True) or "t" not in fh.columns:
            rows.append({well_col: w, "n_rows": 0, "n_members": np.nan, "smape_val": np.nan, "smape_test": np.nan})
            continue

        b = _manifest_bounds_for(boundaries_df, str(w)) or _fallback_boundaries_from_hist(fh)
        aligned = _align_like_plot(sub, fh, b, split_col=split_col)
        vt = only_val_test(aligned, "final_ensemble_df")
        if vt.empty:
            rows.append({well_col: w, "n_rows": 0, "n_members": np.nan, "smape_val": np.nan, "smape_test": np.nan})
            continue

        val_m, test_m = summarize_split_metrics_std_or_sem(
            vt, fh,
            ci_mode=str(ci_mode), coverage=float(coverage),
            mean_col="yhat_final_mean", std_col="std_final", n_members_col=n_members_col,
            override_k=None,
        )
        n_members = float(
            np.nanmedian(pd.to_numeric(vt.get(n_members_col, np.nan), errors="coerce"))
        ) if n_members_col in vt.columns else np.nan

        rows.append({
            well_col: w,
            "n_rows": int(len(vt)),
            "n_members": n_members,
            "smape_val":   (None if not val_m  else val_m.get("SMAPE", np.nan)),
            "smape_test":  (None if not test_m else test_m.get("SMAPE", np.nan)),
        })

    out = pd.DataFrame(rows).sort_values(well_col).reset_index(drop=True)
    # prefer Int64 for better printing
    if "n_rows" in out.columns:
        out["n_rows"] = out["n_rows"].astype("Int64")

    return _alias_columns(out) if use_alias_headers else out

def build_table_from_intra_metrics(
    intra_family_df: pd.DataFrame,
    full_history_by_well: Dict[str, pd.DataFrame],
    *,
    boundaries_df: Optional[pd.DataFrame] = None,
    well_col: str = "well",
    arch_col: str = "arch",
    split_col: str = "split",
    n_members_col: str = "n_members",
    ci_mode: str = "sem",
    coverage: float = 0.90,
    include_family_column: bool = True,
    use_alias_headers: bool = True,
) -> pd.DataFrame:
    """
    Intra-family per-well metrics using family means (yhat_family_mean/std_family).
    Aligns like the plot and computes Validation/Test SMAPE for each (well, arch) group.
    Returns columns (raw): [well, arch?, n_rows, n_members, smape_val, smape_test]
    """
    base_cols = [well_col, "n_rows", "n_members", "smape_val", "smape_test"]
    raw_cols  = ([well_col, arch_col] + base_cols[1:]) if include_family_column else base_cols

    if intra_family_df is None or intra_family_df.empty or any(c not in intra_family_df.columns for c in [well_col, "yhat_family_mean"]):
        out = pd.DataFrame(columns=raw_cols)
        return _alias_columns_intra(out, include_family_column) if use_alias_headers else out

    rows: List[Dict[str, Any]] = []
    # group by well, then arch if desired
    level = [well_col, arch_col] if include_family_column and arch_col in intra_family_df.columns else [well_col]
    for keys, gwell in intra_family_df.groupby(level):
        if isinstance(keys, tuple):
            w, arch = keys
        else:
            w, arch = keys, None

        fh = full_history_by_well.get(str(w))
        if fh is None or getattr(fh, "empty", True) or "t" not in fh.columns:
            rec = {well_col: w, "n_rows": 0, "n_members": np.nan, "smape_val": np.nan, "smape_test": np.nan}
            if include_family_column: rec[arch_col] = arch
            rows.append(rec)
            continue

        b = _manifest_bounds_for(boundaries_df, str(w)) or _fallback_boundaries_from_hist(fh)
        aligned = _align_like_plot(gwell, fh, b, split_col=split_col)
        # Restrict to VAL/TEST
        vt = aligned[aligned[split_col].astype(str).str.lower().isin(["val", "validation", "test"])].copy()
        if vt.empty:
            rec = {well_col: w, "n_rows": 0, "n_members": np.nan, "smape_val": np.nan, "smape_test": np.nan}
            if include_family_column: rec[arch_col] = arch
            rows.append(rec)
            continue

        # Compute metrics with the same summarizer (but family columns)
        def _split(df, key): 
            return df[df[split_col].astype(str).str.lower().str.startswith(key)]
        val_m, test_m = {}, {}
        if not _split(vt, "val").empty or not _split(vt, "validation").empty:
            val_m, _ = summarize_split_metrics_std_or_sem(
                vt, fh,
                ci_mode=str(ci_mode), coverage=float(coverage),
                mean_col="yhat_family_mean", std_col="std_family", n_members_col=n_members_col,
                override_k=None,
            )
        if not _split(vt, "test").empty:
            _, test_m = summarize_split_metrics_std_or_sem(
                vt, fh,
                ci_mode=str(ci_mode), coverage=float(coverage),
                mean_col="yhat_family_mean", std_col="std_family", n_members_col=n_members_col,
                override_k=None,
            )

        n_members = float(
            np.nanmedian(pd.to_numeric(vt.get(n_members_col, np.nan), errors="coerce"))
        ) if n_members_col in vt.columns else np.nan

        rec = {
            well_col: w,
            "n_rows": int(len(vt)),
            "n_members": n_members,
            "smape_val":   (None if not val_m  else val_m.get("SMAPE", np.nan)),
            "smape_test":  (None if not test_m else test_m.get("SMAPE", np.nan)),
        }
        if include_family_column: rec[arch_col] = arch
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values([well_col, arch_col] if include_family_column else [well_col]).reset_index(drop=True)
    if "n_rows" in out.columns:
        out["n_rows"] = out["n_rows"].astype("Int64")

    return _alias_columns_intra(out, include_family_column) if use_alias_headers else out


def _alias_columns_intra(df: pd.DataFrame, include_family_column: bool) -> pd.DataFrame:
    alias = {
        "well": "Well",
        "arch": "Family",
        "n_rows": "Rows",
        "n_members": "Members",
        "smape_val": "Validation Error (%)",
        "smape_test": "Test Error (%)",
    }
    order = (["Well", "Family", "Rows", "Members", "Validation Error (%)", "Test Error (%)"]
             if include_family_column
             else ["Well", "Rows", "Members", "Validation Error (%)", "Test Error (%)"])
    dfa = df.rename(columns=alias)
    cols = [c for c in order if c in dfa.columns] + [c for c in dfa.columns if c not in order]
    return dfa[cols]



def _alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to presentation-friendly aliases."""
    alias = {
        "well": "Well",
        "n_rows": "Rows",
        "n_members": "Members",
        "smape_val": "Validation Error (%)",
        "smape_test": "Test Error (%)",
    }
    # Keep order nice
    order = ["Well", "Rows", "Members", "Validation Error (%)", "Test Error (%)"]
    dfa = df.rename(columns=alias)
    cols = [c for c in order if c in dfa.columns] + [c for c in dfa.columns if c not in order]
    return dfa[cols]





def build_table_from_existing_metrics(
    metrics_df: pd.DataFrame,
    *,
    well_col: str = "well",
    val_col: str = "val_smape_agg",
    test_col: str = "test_smape_agg",
    agg: str = "min",  # pick the best by well (min smape) if there are multiple rows per well
) -> pd.DataFrame:
    """
    Build a per-well table using already-computed columns (e.g. champions_df metrics).
    Useful when you do NOT want to recompute from series (pre-ensemble or already-aggregated metrics).

    Returns columns: [well, smape_val, smape_test]
    """
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(columns=[well_col, "smape_val", "smape_test"])

    need = {well_col, val_col, test_col}
    missing = [c for c in need if c not in metrics_df.columns]
    if missing:
        return pd.DataFrame(columns=[well_col, "smape_val", "smape_test"])

    df = metrics_df[[well_col, val_col, test_col]].copy()
    if agg:
        if agg == "min":
            df = df.groupby(well_col, dropna=False).min(numeric_only=True).reset_index()
        elif agg == "median":
            df = df.groupby(well_col, dropna=False).median(numeric_only=True).reset_index()
        elif agg == "mean":
            df = df.groupby(well_col, dropna=False).mean(numeric_only=True).reset_index()
        else:
            # fallback: first occurrence
            df = df.groupby(well_col, dropna=False).head(1).reset_index(drop=True)

    df = df.rename(columns={val_col: "smape_val", test_col: "smape_test"})
    # Optionally add a count of rows that participated in the agg
    df["n_rows"] = (
        metrics_df.groupby(well_col, dropna=False).size().reindex(df[well_col]).values
    )
    cols = [well_col, "n_rows", "smape_val", "smape_test"]
    return df[cols].sort_values(well_col).reset_index(drop=True)


# ============================================================================================
# 3) Rich rendering (pretty console)
# ============================================================================================

def _fmt_pct(x: Optional[float], decimals: int = 2) -> str:
    """Format percent with a fixed number of decimals."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "–"
    try:
        return f"{float(x):,.{decimals}f}%"
    except Exception:
        return "–"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _color_from_value(v: Optional[float], hi: float = 20.0) -> str:
    """
    Map error value to a smooth green→yellow→red gradient (0..hi).
    Uses truecolor when available; falls back to named colors gracefully.
    """
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "dim"
    x = _clamp(float(v) / float(hi), 0.0, 1.0)  # 0..1
    # piecewise gradient: green (0) → yellow (0.5) → red (1)
    if x <= 0.5:
        # green (46,204,113) → yellow (241,196,15)
        t = x / 0.5
        r = int(46  + t * (241 - 46))
        g = int(204 + t * (196 - 204))
        b = int(113 + t * ( 15 - 113))
    else:
        # yellow (241,196,15) → red (231,76,60)
        t = (x - 0.5) / 0.5
        r = int(241 + t * (231 - 241))
        g = int(196 + t * ( 76 - 196))
        b = int( 15 + t * ( 60 -  15))
    return f"rgb({r},{g},{b})"

def _heat_smape(value: Optional[float]) -> str:
    """Backward-compatible name used by older call sites; now delegates to gradient."""
    # assume 20% as high cap for coloring; you can tweak if seu domínio pede mais/menos
    return _color_from_value(value, hi=20.0)

# ============================================================================================
# 3) Rich rendering (pretty console) — single canonical renderer
# ============================================================================================

_BARS = "▁▂▃▄▅▆▇█"

def _smape_style(value: Optional[float]) -> str:
    """Mapeia SMAPE para estilos discreto (verde→amarelo→vermelho)."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "grey50"
    v = float(value)
    if v < 5:   return "bold green"
    if v < 10:  return "green"
    if v < 20:  return "yellow3"
    return "red"

def _fmt_pct_decimals(x: Optional[float], decimals: int) -> str:
    """Formata percentuais com casas decimais configuráveis; fallback para '–'."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "–"
    try:
        return f"{float(x):,.{decimals}f}%"
    except Exception:
        return "–"

def _mini_bar(
    value_pct: Optional[float],
    *,
    width: int = 16,
    show_track: bool = False,
    track_style: str = "grey23",
) -> Text:
    """
    Micro-bar composita para colunas de erro (0..100). Se show_track=True, aplica um trilho.
    """
    if value_pct is None or not np.isfinite(value_pct):
        return Text(" " * width, style="dim")

    v = float(np.clip(value_pct, 0.0, 100.0))
    n_fill = int(round(v / 100.0 * width))
    n_fill = max(0, min(n_fill, width))

    fill_style = _smape_style(v)
    FILL = "█"
    TRACK = " " if not show_track else "█"

    if show_track:
        bar = Text(TRACK * width, style=track_style)
        if n_fill > 0:
            bar.stylize(fill_style, 0, n_fill)
    else:
        bar = Text(FILL * n_fill + " " * (width - n_fill), style=fill_style)
    return bar

def render_rich_table(
    df: pd.DataFrame,
    *,
    title: str = "Phase 4 — Results",
    caption: Optional[str] = None,
    console: Optional[Console] = None,
    show_lines: bool = False,
    box_style = box.HEAVY,
    highlight_min_smape: bool = True,
    decimals: int = 2,
    bar_width: int = 16,
    bar_track: bool = False,
    header_style: str = "bold black on white",
    title_style: str = "bold underline",
    pad: Tuple[int, int] = (0, 1),
    metric_columns: Optional[Sequence[str]] = None,
    expand: bool = False,
    column_options: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Rich table renderer with:
      - explicit metric column selection
      - optional per-column rendering options
      - safer wrapping for long text fields (avoids '...')
    """
    console = console or Console()
    column_options = dict(column_options or {})

    if df is None or df.empty:
        table = Table(
            title=title,
            title_style=title_style,
            box=box_style,
            show_lines=show_lines,
            padding=pad,
            expand=expand,
        )
        table.add_column("Info", header_style=header_style, justify="left", style="dim")
        table.add_row("No data.")
        console.print(table)
        if caption:
            console.print(Text(caption, style="dim"))
        return

    if metric_columns is None:
        metric_cols = [c for c in df.columns if "Error (%)" in str(c)]
    else:
        metric_cols = [c for c in metric_columns if c in df.columns]

    non_metric = [c for c in df.columns if c not in metric_cols]

    table = Table(
        title=title,
        title_style=title_style,
        box=box_style,
        show_lines=show_lines,
        header_style=header_style,
        padding=pad,
        expand=expand,
    )

    # non-metric columns
    for c in non_metric:
        justify = "center" if c in ("Rows", "Members") else "left"
        style = "bold" if c in ("Well", "Family", "Selected Policy", "Best Test Policy") else ""

        opts = dict(column_options.get(str(c), {}))
        table.add_column(
            str(c),
            justify=opts.pop("justify", justify),
            style=opts.pop("style", style),
            no_wrap=opts.pop("no_wrap", False),
            overflow=opts.pop("overflow", "fold"),
            min_width=opts.pop("min_width", None),
            max_width=opts.pop("max_width", None),
        )

    # metric columns
    for c in metric_cols:
        opts = dict(column_options.get(str(c), {}))
        table.add_column(
            str(c),
            justify=opts.pop("justify", "center"),
            style=opts.pop("style", ""),
            no_wrap=opts.pop("no_wrap", True),
            overflow=opts.pop("overflow", "ellipsis"),
            min_width=opts.pop("min_width", None),
            max_width=opts.pop("max_width", None),
        )

    minima: Dict[str, float] = {}
    if highlight_min_smape and metric_cols:
        for c in metric_cols:
            col = df[c]
            vals = (
                pd.to_numeric(col, errors="coerce")
                if pd.api.types.is_numeric_dtype(col)
                else pd.to_numeric(col.astype(str).str.replace("%", "", regex=False), errors="coerce")
            )
            if vals.notna().any():
                minima[c] = float(vals.min())

    for _, row in df.iterrows():
        cells: List[Any] = []

        for c in non_metric:
            v = row[c]
            if c == "Rows" and pd.notna(v):
                try:
                    v = f"{int(v):,}"
                except Exception:
                    pass
            elif c == "Members" and pd.notna(v):
                try:
                    v = f"{float(v):.0f}"
                except Exception:
                    pass

            text = Text(str(v), style="")
            cells.append(text)

        for c in metric_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                raw_val = float(row[c]) if pd.notna(row[c]) else None
            else:
                try:
                    raw_val = float(str(row[c]).replace("%", ""))
                except Exception:
                    raw_val = None

            txt = _fmt_pct_decimals(raw_val, decimals)
            bar = _mini_bar(raw_val, width=bar_width, show_track=bar_track)
            style = _smape_style(raw_val)

            if c in minima and raw_val is not None and np.isfinite(raw_val) and abs(raw_val - minima[c]) < 1e-12:
                style = f"{style} reverse"

            cells.append(Text.assemble(bar, Text(" "), Text(txt, style=style)))

        table.add_row(*cells)

    console.print(table)
    if caption:
        console.print(Text(caption, style="dim"))



# ============================================================================================
# 4) Exports
# ============================================================================================

def export_table(df: pd.DataFrame, outdir: Path, name: str = "final_results") -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    csv_p = outdir / f"{name}.csv"
    df.to_csv(csv_p, index=False)
    paths["csv"] = csv_p
    try:
        # simple markdown for quick previews
        md_p = outdir / f"{name}.md"
        md = ["| " + " | ".join(map(str, df.columns)) + " |",
              "| " + " | ".join(["---"] * len(df.columns)) + " |"]
        for _, r in df.iterrows():
            md.append("| " + " | ".join(str(x) for x in r.values) + " |")
        md_p.write_text("\n".join(md), encoding="utf-8")
        paths["md"] = md_p
    except Exception:
        pass
    return paths



DEFAULT_COLS: List[str] = [
    "idx", "profile", "hidden_size", "num_encoder_layers", "num_decoder_layers",
    "decoder_output_dim", "dropout", "num_stacks", "num_blocks", "num_layers",
    "layer_widths", "shared_weights", "normalize", "const_init",
    "lags", "p", "d", "q", "seasonal", "seasonal_order", "m",
]

def _pick_columns(rows: List[Dict], cols: Iterable[str]) -> List[str]:
    present = set().union(*(r.keys() for r in rows)) if rows else set()
    return [c for c in cols if c in present]

def _format_table(rows: List[Dict], cols: List[str]) -> str:
    if not rows:
        return "(empty grid)"
    # widths
    w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    # header
    line = "+-" + "-+-".join("-" * w[c] for c in cols) + "-+"
    head = "| " + " | ".join(c.ljust(w[c]) for c in cols) + " |"
    parts = [line, head, line]
    # rows
    for r in rows:
        parts.append("| " + " | ".join(str(r.get(c, "")).ljust(w[c]) for c in cols) + " |")
    parts.append(line)
    return "\n".join(parts)

def log_grid_table(title: str, grid: List[Dict]) -> None:
    # add enumerated idx for readability
    rows = [{"idx": i, **row} for i, row in enumerate(grid, start=1)]
    cols = _pick_columns(rows, DEFAULT_COLS)
    table = _format_table(rows, cols)
    logging.info(f"{title}\n{table}")