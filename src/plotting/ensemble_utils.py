#src/plotting/ensemble_utils.py
# --- Standard Library Imports ---
# Used for built-in functionalities like iteration, logging, regular expressions,
# and static type checking.
import itertools
import logging
import re
from typing import Any, Dict, Optional, Tuple

# --- Third-Party Library Imports ---
# Core external libraries for numerical operations, data manipulation, and plotting.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as qcolors
log = logging.getLogger(__name__)


def _get_color_palette(name: str) -> Dict[str, str]:
    palettes = {
        "default": {
            "train_from_x": "#3949AB",
            "train_from_y": "#0277BD",
            "actual_post_train": "#0277BD",
            "validation": "#E53935",
            "test_initial": "#4CAF50",
            "test_rolling": "#4CAF50",
            "fill_train": "rgba(232, 234, 246, 0.5)",
            "fill_train_bk": "rgba(232, 234, 246, 0.15)",  # ← NOVO (background de treino)
            "fill_val":  "#9BA3AF",
            "fill_test": "#6B7280",
            "fill_val_bk": "rgba(255, 235, 238, 0.5)",
            "fill_test_bk": "rgba(232, 245, 233, 0.5)",
            "text": "#333",
            "grid": "#EAEAEA",
            "test_border": "#4CAF50",
        },
        "metallic_azure": {
            "train_from_x": "#546E7A",
            "train_from_y": "#90A4AE",
            "actual_post_train": "#1E88E5",
            "validation": "#00B8D4",
            "test_initial": "#263238",
            "test_rolling": "#4FC3F7",
            "fill_train": "rgba(84, 110, 122, 0.12)",
            "fill_train_bk": "rgba(84, 110, 122, 0.10)",   # ← NOVO
            "fill_val":   "rgba(0, 184, 212, 0.10)",
            "fill_test":  "rgba(79, 195, 247, 0.12)",
            "fill_val_bk":   "rgba(0, 184, 212, 0.10)",
            "fill_test_bk":  "rgba(79, 195, 247, 0.12)",
            "text": "#222",
            "grid": "#E5E7EB",
            "test_border": "#4FC3F7",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
        },
        "abyssal_expedition_light": {
            "train_from_x": "#0D47A1",
            "train_from_y": "#1976D2",
            "actual_post_train": "#00796B",
            "validation": "#00B8D4",
            "test_initial": "#AA00FF",
            "test_rolling": "#F9A825",
            "fill_train": "rgba(25,118,210,0.10)",
            "fill_train_bk": "rgba(25,118,210,0.08)",      # ← NOVO
            "fill_val": "rgba(0,184,212,0.10)",
            "fill_test": "rgba(249,168,37,0.12)",
            "fill_val_bk":   "rgba(0, 184, 212, 0.10)",
            "fill_test_bk":  "rgba(79, 195, 247, 0.12)",
            "text": "#222",
            "grid": "#EAEAEA",
            "test_border": "#F9A825",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
        },
    }
    if name not in palettes:
        raise ValueError(f"Unknown palette '{name}'. Available options are: {list(palettes.keys())}")
    palette = palettes[name]
    palette.setdefault("plot_bgcolor", "white")
    palette.setdefault("paper_bgcolor", "white")
    return palette


def _with_alpha(color: str, alpha: float) -> str:
    """
    Returns an rgba() string with the requested alpha from a '#RRGGBB' or 'rgb/rgba()' input.
    Alpha will be clamped to [0,1].
    """
    alpha = max(0.0, min(1.0, float(alpha)))

    if not isinstance(color, str) or not color:
        return f"rgba(0,0,0,{alpha})"

    color = color.strip()

    # Already rgb/rgba() → normalize alpha
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([0-9.]+))?\s*\)", color, re.I)
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"rgba({r},{g},{b},{alpha})"

    # Hex #RRGGBB
    m = re.match(r"#?([0-9a-fA-F]{6})", color)
    if m:
        hexv = m.group(1)
        r = int(hexv[0:2], 16)
        g = int(hexv[2:4], 16)
        b = int(hexv[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # Fallback (Plotly named colors etc.)
    return f"rgba(0,0,0,{alpha})"
    
def _apply_alpha(rgb_or_rgba: str, alpha: float) -> str:
    """
    Ensure color has the requested alpha. Accepts 'rgb(...)', 'rgba(...)' or '#rrggbb'.
    Delegates to _with_alpha for a single source of truth.
    """
    return _with_alpha(str(rgb_or_rgba), alpha)


def by_well(df: pd.DataFrame, well: str, name: str) -> pd.DataFrame:
    """
    Retorna somente as linhas de um poço, de forma robusta a variações
    de formato (15-9-F-14 vs 15/9-F-14, underscores, espaços, etc).

    name: rótulo só para logs (ex: 'final_ensemble_df').
    """
    from common.phase_viz_support import normalize_well_key

    if not isinstance(df, pd.DataFrame) or df.empty:
        log.info("[ensemble_plot] %s: empty for well=%s", name, well)
        return pd.DataFrame()

    if "well" not in df.columns:
        out = df.copy()
        log.info("[ensemble_plot] %s: no 'well' column → using full frame (shape=%s)", name, out.shape)
        return out

    target_key = normalize_well_key(well)
    well_keys = df["well"].map(normalize_well_key)

    mask = well_keys == target_key
    out = df[mask].copy()

    log.info(
        "[ensemble_plot] %s: well=%s norm_key=%s → shape=%s (matches=%d / %d)",
        name,
        well,
        target_key,
        out.shape,
        int(mask.sum()),
        len(df),
    )

    if "split" in out.columns:
        log.info(
            "[ensemble_plot] %s: split distribution:\n%s",
            name,
            out["split"].astype(str).value_counts(dropna=False).to_string()
        )

    return out



def ensure_time_axis(df: pd.DataFrame, name: str, full_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 't' time column exists by merging from full history using either
    'global_idx' (preferred) or 'idx' (legacy). If 't' already exists, keep it.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "t" in df.columns:
        log.info("[ensemble_plot] %s: 't' already present; keeping as is.", name)
        return df

    # Attach via global_idx (preferred)
    if "global_idx" in df.columns:
        tm = full_history_df.reset_index().rename(columns={"index": "global_idx"})[["global_idx", "t"]]
        out = df.merge(tm, on="global_idx", how="inner")
        log.info("[ensemble_plot] %s: attached axis via global_idx (%d -> %d).", name, len(df), len(out))
        return out

    # Fallback: attach via idx (legacy)
    if "idx" in df.columns:
        tm = full_history_df.reset_index().rename(columns={"index": "idx"})[["idx", "t"]]
        out = df.merge(tm, on="idx", how="inner")
        log.info("[ensemble_plot] %s: attached axis via idx (legacy) (%d -> %d).", name, len(df), len(out))
        return out

    return df

def coalesce_time_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    If both 't' and 'idx' are numeric and identical, drop 't' and re-merge is fine;
    otherwise just return the df. This keeps 't' meaningful (axis in history units).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    try:
        if "t" in df.columns and "idx" in df.columns:
            if pd.api.types.is_numeric_dtype(df["t"]) and pd.api.types.is_numeric_dtype(df["idx"]):
                if (df["t"] == df["idx"]).all():
                    # keep as-is; we already ensured 't' through ensure_time_axis
                    pass
    except Exception:
        pass
    return df

def only_val_test(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Filter to validation/test rows only (if 'split' exists)."""
    if df is None or df.empty or "split" not in df.columns:
        return df
    out = df[df["split"].astype(str).str.lower().isin(["val", "validation", "test"])].copy()
    log.info("[ensemble_plot] %s: VAL/TEST only → %s", name, out.shape)
    return out


def safe_int(value: Any, lo: int, hi: int) -> int:
    try:
        value = int(value)
    except Exception:
        value = lo
    return max(lo, min(value, hi))




def hover_y(fmt: str = ".3f") -> str:
    return f"<b>%{{y:{fmt}}}</b><extra>%{{fullData.name}}</extra>"


# -----------------------------------------------------------------------------
# Figure composition helpers (side-effect: add traces/layout to fig)
# -----------------------------------------------------------------------------
def add_region_backgrounds(fig: go.Figure, colors: Dict[str, str], t_axis: pd.Series,
                           tr_end_i: int, va_end_i: int) -> None:
    t_min, t_max = t_axis.iloc[0], t_axis.iloc[-1]
    x_tr_end, x_va_end = t_axis.iloc[tr_end_i], t_axis.iloc[va_end_i]
    fig.add_vrect(x0=t_min,   x1=x_tr_end, fillcolor=colors.get("fill_train_bk", _with_alpha("#9ebcda", 0.15)),
                  layer="below", line_width=0)
    fig.add_vrect(x0=x_tr_end, x1=x_va_end, fillcolor=colors.get("fill_val_bk", _with_alpha("#ef8a62", 0.12)),
                  layer="below", line_width=0)
    fig.add_vrect(x0=x_va_end, x1=t_max,    fillcolor=colors.get("fill_test_bk", _with_alpha("#67a9cf", 0.12)),
                  layer="below", line_width=0)


def add_ground_truth(fig: go.Figure, full_history_df: pd.DataFrame, colors: Dict[str, str]) -> None:
    fig.add_trace(go.Scatter(
        x=full_history_df["t"], y=full_history_df["ytrue"], name="Ground Truth",
        mode="lines",
        line=dict(color=colors.get("actual_post_train", "#1f77b4"), width=2.5),
        hovertemplate=hover_y()
    ))


def extract_quantile_columns(df: pd.DataFrame, prefix: str) -> Optional[Dict[float, pd.Series]]:
    """
    Detecta colunas de quantis para o prefixo (ex.: 'final', 'family', etc.).
    Aceita padrões:
      yhat_q05_{p}, q05_{p}, yhat_q10_{p}, yhat_q_low_{p}/yhat_q_high_{p}, q_lo_{p}/q_hi_{p}, etc.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols_lc = {c.lower(): c for c in df.columns}

    # pares explícitos e comuns
    pairs = [
        (f"yhat_q05_{prefix}", 0.05), (f"yhat_q10_{prefix}", 0.10),
        (f"yhat_q50_{prefix}", 0.50), (f"yhat_q70_{prefix}", 0.70),
        (f"yhat_q90_{prefix}", 0.90), (f"yhat_q95_{prefix}", 0.95),
        (f"q05_{prefix}", 0.05), (f"q10_{prefix}", 0.10),
        (f"q50_{prefix}", 0.50), (f"q70_{prefix}", 0.70),
        (f"q90_{prefix}", 0.90), (f"q95_{prefix}", 0.95),
        (f"q_lo_{prefix}", 0.05), (f"q_hi_{prefix}", 0.95),
        (f"yhat_q_low_{prefix}", 0.05), (f"yhat_q_high_{prefix}", 0.95),
    ]
    found: Dict[float, pd.Series] = {}
    for key, q in pairs:
        if key in cols_lc:
            found[q] = df[cols_lc[key]]

    # varredura regex adicional para yhat_q{NN}_{prefix}
    # ex.: yhat_q25_final → 0.25
    pat = re.compile(rf"^yhat_q(\d{{2}})_{re.escape(prefix)}$", re.I)
    for lc, orig in cols_lc.items():
        m = pat.match(lc)
        if m:
            q = int(m.group(1)) / 100.0
            found[q] = df[orig]

    if not found:
        return None
    return dict(sorted(found.items(), key=lambda kv: kv[0]))


def _fmt_metric_block(metrics: Dict[str, float], *, sharp_suffix: str = "") -> str:
    """
    Compact HTML for annotation box. Skips NaNs.
    - SMAPE: one decimal
    - Cov@: percentage with one decimal
    - Sharp@: integer with thousand separators (or 1 decimal if small)
    """
    if not metrics:
        return ""
    parts = []
    if "SMAPE" in metrics and np.isfinite(metrics["SMAPE"]):
        parts.append(f"SMAPE: {metrics['SMAPE']:.1f}%")
    for k, v in metrics.items():
        if k.startswith("Cov@") and np.isfinite(v):
            parts.append(f"{k}: {v*100:.1f}%")
        if k.startswith("Sharp@") and np.isfinite(v):
            val = metrics[k]
            if abs(val) >= 100:
                parts.append(f"{k}: {val:,.0f}{sharp_suffix}")
            else:
                parts.append(f"{k}: {val:,.1f}{sharp_suffix}")
    return "<br>".join(parts)

def add_uncertainty_metrics_annotations(
    fig: go.Figure,
    *,
    metrics_val: Dict[str, float],
    metrics_test: Dict[str, float],
    colors: Dict[str, str],
    font_scale: float,
    is_cumulative_plot: bool = False,
    sharp_suffix: str = "",
) -> None:
    """
    Add two metric boxes (Validation/Test) on the right side, styled like plot_integrated_view.
    """
    if not metrics_val and not metrics_test:
        return

    # Y anchors similar to your integrated_view defaults
    val_y_anchor, test_y_anchor = (0.9, 0.7) #if is_cumulative_plot else (0.98, 0.70)
    fs = 16 * font_scale

    if metrics_val:
        text_val = _fmt_metric_block(metrics_val, sharp_suffix=sharp_suffix)
        fig.add_annotation(
            text=f"<b>Validation</b><br>{text_val}",
            align="left", showarrow=False,
            xref="paper", yref="paper",
            x=0.7, y=val_y_anchor, xanchor="right", yanchor="top",
            bordercolor=colors.get("validation", "#d62728"),
            borderwidth=2, bgcolor="rgba(255,255,255,0.95)",
            font=dict(size=fs, color=colors.get("text", "#222"), family="Courier New, monospace")
        )

    if metrics_test:
        text_test = _fmt_metric_block(metrics_test, sharp_suffix=sharp_suffix)
        fig.add_annotation(
            text=f"<b>Test</b><br>{text_test}",
            align="left", showarrow=False,
            xref="paper", yref="paper",
            x=0.7, y=test_y_anchor, xanchor="right", yanchor="top",
            bordercolor=colors.get("test_border", colors.get("test_rolling", "#2ca02c")),
            borderwidth=2, bgcolor="rgba(255,255,255,0.95)",
            font=dict(size=fs, color=colors.get("text", "#222"), family="Courier New, monospace")
        )


def add_residuals_inset(fig: go.Figure, residuals: pd.Series,
                        *, xanchor: float = 0.79, yanchor: float = 0.20,
                        width_frac: float = 0.18, height_frac: float = 0.22,
                        nbins: int = 30) -> None:
    """
    Draw a small histogram inset of residuals (VAL+TEST) in the top-right area.
    """
    if residuals is None or residuals.empty:
        return
    # Define secondary axes domain
    x0 = xanchor
    x1 = min(0.995, xanchor + width_frac)
    y0 = 1.0 - height_frac
    y1 = 0.995
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=nbins, opacity=0.75, name="Residuals (Val+Test)",
        marker=dict(line=dict(width=0)), showlegend=False, xaxis="x2", yaxis="y2",
        hovertemplate="<b>Residual</b>: %{x:.2f}<br>Count: %{y}<extra></extra>"
    ))
    fig.update_layout(
        xaxis2=dict(domain=[x0, x1], anchor="y2", showgrid=False, zeroline=False, showticklabels=False),
        yaxis2=dict(domain=[y0, y1], anchor="x2", showgrid=False, zeroline=False, showticklabels=False),
        barmode="overlay"
    )