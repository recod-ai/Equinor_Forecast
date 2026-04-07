# src/plotting/risk_plots.py
from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common.phase_viz_support import _resolve_canonical_well
# ==============================================================================
# EXTERNAL DEPENDENCIES
# ==============================================================================
# --- Robust import block (plug-and-play) ---
try:
    from plotting.ensemble_plots import polish_layout
    from plotting.ensemble_utils import _get_color_palette
except Exception:  # fallback p/ import relativo quando rodando dentro de src/plotting
    from .ensemble_plots import polish_layout
    from .ensemble_utils import _get_color_palette

try:
    from risk.risk_core import (
        build_member_series,
        weights_for_members,
        accumulate_window,
        weighted_quantiles,
        resolve_horizon_indices,
    )
except Exception:
    # fallback se a lib "risk" estiver dentro do projeto
    from .risk_core import (
        build_member_series,
        weights_for_members,
        accumulate_window,
        weighted_quantiles,
        resolve_horizon_indices,
    )


# ==============================================================================
# FORMATTING & MATH UTILITIES
# ==============================================================================

def _fmt_compact(x: float) -> str:
    """Format large numbers into compact strings (e.g., 1.5M, 200K)."""
    if not np.isfinite(x):
        return "–"
    ax = abs(x)
    if ax >= 1e12:
        return f"{x/1e12:.2f}T"
    if ax >= 1e9:
        return f"{x/1e9:.2f}B"
    if ax >= 1e6:
        return f"{x/1e6:.2f}M"
    if ax >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:,.2f}"


def _fmt_stats_box(vals: np.ndarray, weights: Optional[np.ndarray]) -> str:
    """Generates a pastel-style HTML summary card."""
    n = int(vals.size)
    if n == 0:
        return "<b>No Data</b>"

    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

    if weights is None:
        mean = float(np.mean(vals))
    else:
        mean = float(np.average(vals, weights=weights))

    q = weighted_quantiles(vals, weights, qs=(0.10, 0.50, 0.90))
    p10 = q.get(0.10, np.nan)
    p50 = q.get(0.50, np.nan)
    p90 = q.get(0.90, np.nan)

    return (
        "<span style='font-size: 16px; font-weight: 800; color: #355c7d;'>Summary</span><br>"
        f"<span style='font-size: 14px; color: #666;'><b>Members:</b> {n:,}</span><br>"
        f"<span style='color: #e0e0e0;'>────────────────────</span><br>"
        f"<b style='font-size: 14px;'>Mean:</b> <span style='font-size: 14px;'>{_fmt_compact(mean)}</span><br>"
        f"<b style='font-size: 14px;'>P50:</b>  <span style='font-size: 14px;'>{_fmt_compact(p50)}</span><br>"
        f"<span style='color: #e0e0e0;'>────────────────────</span><br>"
        f"P10: {_fmt_compact(p10)}<br>"
        f"P90: {_fmt_compact(p90)}<br>"
        f"<span style='color: #999; font-size: 13px; font-style: italic;'>Range: {_fmt_compact(vmin)} — {_fmt_compact(vmax)}</span>"
    )


def _empirical_cdf(values: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_sorted, cdf_y) for an empirical CDF."""
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    v = v[mask]

    if v.size == 0:
        return np.array([]), np.array([])

    if weights is None:
        w = np.ones_like(v, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)[mask]
        if w.size != v.size or not np.isfinite(w).any():
            w = np.ones_like(v, dtype=float)

    w = np.clip(w, 0.0, np.inf)
    total_w = float(w.sum())
    w = (w / total_w) if total_w > 0.0 else (np.ones_like(v, dtype=float) / float(len(v)))

    order = np.argsort(v, kind="mergesort")
    xs = v[order]
    cdf = np.cumsum(w[order])
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0

    return xs, cdf


def _interpolate_cdf_at_x(xs: np.ndarray, cdf: np.ndarray, x_val: float) -> float:
    """Approximate CDF(x_val) for plotting markers."""
    if xs.size == 0 or not np.isfinite(x_val):
        return np.nan
    idx = np.searchsorted(xs, x_val, side="right") - 1
    if idx < 0:
        return float(cdf[0])
    if idx >= len(cdf):
        return float(cdf[-1])
    return float(cdf[idx])


# ==============================================================================
# THEME / STYLING ENGINE
# ==============================================================================

def _build_pastel_theme(colors: Dict[str, str]) -> Dict[str, str]:
    """
    Build a light, pastel theme based on the existing palette.
    No dark backgrounds or black-heavy colors.
    """
    primary = colors.get("test_rolling", "#55c1a7")      # base green
    secondary = colors.get("train_from_x", "#5c85d6")    # soft blue
    accent = colors.get("validation", "#f48c6c")         # soft coral

    return {
        "bg_plot": "#ffffff",
        "bg_paper": "#ffffff",
        "bg_card": "rgba(250, 252, 255, 0.98)",
        "grid": "#e9edf5",
        "axis": "#90a4ae",
        "text_main": "#37474f",
        "text_muted": "#78909c",
        "line_main": primary,
        "line_fill": "rgba(85, 193, 167, 0.18)",
        "line_band": "rgba(92, 133, 214, 0.15)",
        "q_p50": secondary,
        "q_tails": accent,
        "rug": "rgba(120, 144, 156, 0.45)",
    }


def _apply_clean_white_canvas(fig: go.Figure, colors: Dict[str, str], font_scale: float) -> None:
    """
    Enforces a white canvas and pastel styling (no dark backgrounds).
    """
    base_font_size = int(14 * font_scale)
    theme = _build_pastel_theme(colors)

    fig.update_layout(
        template=None,
        plot_bgcolor=theme["bg_plot"],
        paper_bgcolor=theme["bg_paper"],
        font=dict(
            family="Inter, Arial, Roboto, sans-serif",
            color=theme["text_main"],
            size=base_font_size,
        ),
        margin=dict(t=90, b=70, l=90, r=260),  # extra right margin for stats card
        dragmode="zoom",
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=theme["axis"],
            font_size=base_font_size,
            font_family="Inter, Arial, sans-serif",
        ),
    )

    grid_color = theme["grid"]
    axis_line_color = theme["axis"]

    common_axis = dict(
        showline=True,
        linewidth=1.5,
        linecolor=axis_line_color,
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
        mirror=False,
        ticks="outside",
        tickwidth=1.5,
        ticklen=6,
        tickcolor=axis_line_color,
        zeroline=False,
        tickfont=dict(size=base_font_size, color=theme["text_muted"]),
        automargin=True,
    )

    fig.update_xaxes(
        **common_axis,
        title_text="<b>Accumulated Production</b>",
        title_font=dict(size=int(base_font_size * 1.2), color=theme["text_main"]),
        tickformat=",.2s",
    )

    fig.update_yaxes(
        **common_axis,
        title_text="<b>Probability (CDF)</b>",
        title_font=dict(size=int(base_font_size * 1.2), color=theme["text_main"]),
        range=[-0.05, 1.02],
        tickformat=".0%",
        dtick=0.1,
    )


def _plot_config(width: int, height: int, filename: str = "risk_cdf_plot") -> Dict[str, Any]:
    """
    Central Plotly config: light mode bar, export to PNG in high resolution.
    """
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": height,
            "width": width,
            "scale": 3,
        },
    }


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_risk_cdf(
    *,
    accum_values: Iterable[float],
    weights: Optional[Iterable[float]] = None,
    title: str = "Risk Curve (CDF)",
    palette: str = "default",
    annotate: bool = True,
    font_scale: float = 1.30,
    width: int = 900,
    height: int = 600,
    show: bool = True,
) -> go.Figure:
    """
    Renders a pastel, story-driven Empirical CDF.

    Visual concept:
    - Soft pastel area under the CDF.
    - P10–P90 band as a translucent highlight.
    - Clean pastel quantile markers with dedicated hover tooltips.
    - Stats card floating on the right, like a side panel.
    - White canvas, subtle grid, zero dark backgrounds.
    """
    colors = _get_color_palette(palette)
    theme = _build_pastel_theme(colors)

    col_line = theme["line_main"]
    col_fill = theme["line_fill"]
    col_band = theme["line_band"]
    col_p50 = theme["q_p50"]
    col_tails = theme["q_tails"]
    col_rug = theme["rug"]

    vals = np.asarray(list(accum_values), dtype=float)
    ws = None if weights is None else np.asarray(list(weights), dtype=float)
    xs, cdf = _empirical_cdf(vals, ws)

    fig = go.Figure()

    # ==========================================================================
    # Empty state
    # ==========================================================================
    if xs.size == 0:
        _apply_clean_white_canvas(fig, colors, font_scale)
        fig.update_layout(width=width, height=height)
        fig.add_annotation(
            text=(
                "<b>No Data</b><br>"
                "<span style='font-size:12px;color:#9e9e9e;'>"
                "No accumulated production values available."
                "</span>"
            ),
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
        )

        fmt_title = f"<span style='font-size: 24px; font-weight: 700;'>{title}</span>"
        fig.update_layout(title=dict(text=fmt_title, x=0.02, xref="paper", xanchor="left", y=0.96))

        if show:
            fig.show(config=_plot_config(width, height))
        return fig

    # ==========================================================================
    # Quantiles
    # ==========================================================================
    q = weighted_quantiles(xs, None if ws is None else ws, qs=(0.10, 0.50, 0.90))
    q10, q50, q90 = q.get(0.10, np.nan), q.get(0.50, np.nan), q.get(0.90, np.nan)

    # ==========================================================================
    # 1) P10–P90 band (highlight)
    # ==========================================================================
    if np.isfinite(q10) and np.isfinite(q90) and q90 > q10:
        fig.add_vrect(
            x0=q10,
            x1=q90,
            fillcolor=col_band,
            layer="below",
            line_width=0,
        )
        fig.add_annotation(
            x=q10,
            y=1.01,
            xref="x",
            yref="paper",
            text="<span style='font-size:11px;color:#90a4ae;'>P10–P90 band</span>",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
        )

    # ==========================================================================
    # 2) Area under CDF (pastel)
    # ==========================================================================
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=cdf,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            fill="tozeroy",
            fillcolor=col_fill,
            showlegend=False,
        )
    )

    # ==========================================================================
    # 3) Main CDF line (on top of area)
    # ==========================================================================
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=cdf,
            mode="lines",
            name="CDF",
            line=dict(width=4, color=col_line, shape="hv"),
            hovertemplate="Accumulated: %{x:,.0f}<br>Probability: %{y:.1%}<extra></extra>",
        )
    )

    # ==========================================================================
    # 4) Rug plot (light density hint, near bottom)
    # ==========================================================================
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=np.full_like(xs, -0.03),
            mode="markers",
            marker=dict(
                symbol="line-ns-open",
                size=7,
                line=dict(width=1, color=col_rug),
            ),
            hoverinfo="skip",
            showlegend=False,
            cliponaxis=False,
        )
    )

    # ==========================================================================
    # 5) Quantile markers (lines + dots + hover)
    # ==========================================================================
    def _add_quantile_marker(val: float, label: str, col: str, dash: str = "dot") -> None:
        if not np.isfinite(val):
            return
        y_val = _interpolate_cdf_at_x(xs, cdf, val)

        # Vertical line
        fig.add_vline(
            x=val,
            line_width=2.0,
            line_dash=dash,
            line_color=col,
            opacity=0.85,
        )

        # Label at the top
        fig.add_annotation(
            x=val,
            y=1.02,
            xref="x",
            yref="paper",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color=col, size=int(13 * font_scale)),
            bgcolor="#ffffff",
            borderpad=2,
        )

        # Dot on the curve
        if np.isfinite(y_val):
            fig.add_trace(
                go.Scatter(
                    x=[val],
                    y=[y_val],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=col,
                        line=dict(color="#ffffff", width=1.5),
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"{label} quantile<br>"
                        "Accumulated: %{x:,.0f}<br>"
                        "Probability: %{y:.1%}<extra></extra>"
                    ),
                )
            )

    _add_quantile_marker(q10, "P10", col_tails, dash="dot")
    _add_quantile_marker(q50, "P50", col_p50, dash="dash")
    _add_quantile_marker(q90, "P90", col_tails, dash="dot")

    # ==========================================================================
    # 6) Stats card (side panel, top-right)
    # ==========================================================================
    if annotate:
        fig.add_annotation(
            text=_fmt_stats_box(xs, ws),
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.37,
            y=0.95,
            xanchor="right",
            yanchor="top",
            bgcolor=theme["bg_card"],
            bordercolor="#dde5f0",
            borderwidth=1,
            borderpad=2,
            width=140,
        )

    # ==========================================================================
    # 7) Layout polish (reusing your helper + pastel overrides)
    # ==========================================================================
    polish_layout(
        fig,
        colors,
        title_txt="",
        font_scale=font_scale,
        width=width,
        height=height,
        yaxis_log=False,
    )
    _apply_clean_white_canvas(fig, colors, font_scale)
    title_color = col_line  

    # Title styling (main + optional subtitle via <br>)
    if "<br>" in title:
        main, sub = title.split("<br>", 1)
        fmt_title = (
            f"<span style='font-size: 26px; font-weight: 700; color:{title_color};'>{main}</span><br>"
            f"<span style='font-size: 22px; color: #78909c;'>{sub}</span>"
        )
    else:
        fmt_title = f"<span style='font-size: 22px; font-weight: 700;'>{title}</span>"

    fig.update_layout(
        title=dict(
            text=fmt_title,
            x=0.02,
            xref="paper",
            xanchor="left",
            y=0.96,
        ),
        width=width,
        height=height,
    )

    if show:
        fig.show(config=_plot_config(width, height))

    return fig


from common.phase_viz_support import (
    lookup_manifest_bounds,
    infer_boundaries,
    fallback_boundaries,
    build_full_history,   # se você ainda quiser usar
    normalize_well_key,
)

from common.log_utils import log_block, warn  # se ainda não estiverem no topo


def plot_risk_cdf_for(
    *,
    series_df: pd.DataFrame,
    final_ensemble_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    full_history_by_well: Dict[str, pd.DataFrame],
    well: str,
    arch: str,
    selector: str,
    horizon_days: int,
    weighting: str = "uniform",
    temp: float = 0.5,
    palette: str = "default",
    title_prefix: str = "Risk Curve",
    width: int = 900,
    height: int = 600,
    show: bool = True,
) -> go.Figure:
    """
    Risk CDF (Simple Path): VAL=full + TEST=head(H) via split+idx.
    Usa build_member_series apenas para pesos quando distance_softmax.

    Agora alias-aware: '15/9-F-14' ⇔ '15-9-F-14' via normalize_well_key.
    """
    from risk.risk_core import (
        build_member_series,          # só para pesos distance_softmax
        accumulate_members_simple,    # janela efetiva
        weights_for_members,
    )

    # ------------------------------------------------------------------
    # 🔑 1) Resolver alias → canonical_well (o que aparece nos DataFrames)
    # ------------------------------------------------------------------
    wells_unicos = []

    if isinstance(series_df, pd.DataFrame) and "well" in series_df.columns:
        wells_unicos.extend(series_df["well"].dropna().astype(str).unique().tolist())
    if isinstance(boundaries_df, pd.DataFrame) and "well" in boundaries_df.columns:
        wells_unicos.extend(boundaries_df["well"].dropna().astype(str).unique().tolist())
    if isinstance(full_history_by_well, dict):
        wells_unicos.extend(map(str, full_history_by_well.keys()))

    canonical_well, norm_key = _resolve_canonical_well(well, wells_unicos)

    # ------------------------------------------------------------------
    # 2) Full history (opcional – só se existir mesmo)
    # ------------------------------------------------------------------
    fh = None
    if full_history_by_well:
        # tenta por chave normalizada primeiro
        fh_map_norm = {
            normalize_well_key(k): v for k, v in full_history_by_well.items()
        }
        fh = fh_map_norm.get(norm_key)

    if fh is None or getattr(fh, "empty", True):
        # fallback pra build_full_history, que também pode ser alias-aware
        try:
            fh = build_full_history(
                series_df, canonical_well, full_history_by_well=full_history_by_well
            )
        except TypeError:
            try:
                fh = build_full_history(
                    series_df, canonical_well, full_history_df_map=full_history_by_well
                )
            except Exception:
                fh = pd.DataFrame(columns=["t", "ytrue"])

    # ------------------------------------------------------------------
    # 3) Boundaries manifest (usando canonical_well, não o label com '/')
    # ------------------------------------------------------------------
    bounds = lookup_manifest_bounds(boundaries_df, canonical_well)
    if bounds is None:
        try:
            bounds = infer_boundaries(
                final_ensemble_df,
                series_df,
                canonical_well,
                boundaries_df=boundaries_df,
                full_history_df=fh,
            )
        except Exception:
            bounds = None
    if bounds is None:
        bounds = fallback_boundaries(canonical_well, fh)

    # ------------------------------------------------------------------
    # 4) Pesos por membro (distance_softmax ou uniforme)
    # ------------------------------------------------------------------
    df_members_for_weights = build_member_series(
        series_df=series_df,
        well=canonical_well,
        arch=str(arch).lower(),
        full_history_df=fh,
        boundaries=bounds,
    )

    if df_members_for_weights is None or df_members_for_weights.empty:
        warn(
            f"[risk_plot] No member series for well={canonical_well}, arch={arch} "
            f"(requested='{well}', norm_key='{norm_key}')."
        )
        # cai direto no plot "No Data"
        return plot_risk_cdf(
            accum_values=[],
            title=f"{title_prefix}<br>No Data: {well}",
            palette=palette,
            width=width,
            height=height,
            show=show,
        )

    wts = weights_for_members(
        df_members=df_members_for_weights,
        strategy=str(weighting),
        temperature=float(temp),
    )

    # ------------------------------------------------------------------
    # 5) Janela pela rota simples (igual Stage 4.1) — usando canonical_well
    # ------------------------------------------------------------------
    acc = accumulate_members_simple(
        series_df=series_df,
        well=canonical_well,
        arch=str(arch).lower(),
        selector=str(selector),
        horizon_days=int(horizon_days),   # aplica head(H) em TEST, VAL completo
        member_id_col="job_hash",
        yhat_col="yhat",
    )

    pts = 0 if acc is None or acc.empty else int(acc["n_points"].max())
    n_members = 0 if acc is None or acc.empty else int(acc["job_hash"].nunique())

    log_block(
        "Risk Window — Effective Slice (plot)",
        [
            f"Well/Arch: {well} / {str(arch).lower()} (canonical={canonical_well}, key={norm_key})",
            f"Selector: {selector}   Horizon: {horizon_days}d",
            "Path: simple(split+idx)",
            f"Members: {n_members}   Points (per member): {pts}",
        ],
        width=100,
    )

    if acc is None or acc.empty:
        return plot_risk_cdf(
            accum_values=[],
            title=f"{title_prefix}<br>No Data: {well}",
            palette=palette,
            width=width,
            height=height,
            show=show,
        )

    # ------------------------------------------------------------------
    # 6) Merge de pesos + normalização
    # ------------------------------------------------------------------
    acc = acc.merge(wts, on="job_hash", how="left")
    if "weight" not in acc.columns or acc["weight"].isna().all():
        n = acc["job_hash"].nunique()
        acc["weight"] = 1.0 / max(1, n)
    else:
        s = float(acc["weight"].sum())
        if s > 0:
            acc["weight"] /= s

    # ------------------------------------------------------------------
    # 7) Título bonitinho (mantém label original com '/')
    # ------------------------------------------------------------------
    raw_arch = str(arch).upper()
    arch_display = "PINN" if raw_arch == "SEQ2" else raw_arch
    h_display = "All History" if int(horizon_days) < 0 else f"Horizon: {int(horizon_days)}d"
    subtitle = f"Well: {well} | Architecture: {arch_display} | {selector.title()} | {h_display}"
    final_title = f"{title_prefix}<br>{subtitle}"

    return plot_risk_cdf(
        accum_values=acc["accum"].to_numpy(),
        weights=acc["weight"].to_numpy(),
        title=final_title,
        palette=palette,
        annotate=True,
        width=width,
        height=height,
        show=show,
    )


