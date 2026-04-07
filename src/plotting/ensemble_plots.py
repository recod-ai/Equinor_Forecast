# src/plotting/ensemble_plots.py

# --- Standard Library Imports ---
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as qcolors

# --- Local Application Imports ---
from .ensemble_stats import (
    _choose_scale_for_mode,
    _compute_bounds_for_mode,
    _compute_coverage_and_sharpness,
    _fan_from_gaussian,
    _fan_levels_default,
    compute_residuals_vt,
    summarize_split_metrics_std_or_sem,
)
from .ensemble_utils import (
    _get_color_palette,
    _with_alpha,
    add_ground_truth,
    add_region_backgrounds,
    add_uncertainty_metrics_annotations,
    by_well,
    coalesce_time_column,
    ensure_time_axis,
    extract_quantile_columns,
    hover_y,
    only_val_test,
    safe_int,
    add_residuals_inset
)


log = logging.getLogger(__name__)


def add_final_ensemble_traces(
    fig: go.Figure,
    df_final_vt: "pd.DataFrame",
    colors: Dict[str, str],
    *,
    band_alpha: float,
    band_inner_alpha: Optional[float],
    ci_mode: str,
    ci_k: float,
    ci_quantiles: Tuple[float, float],
    n_members_col: str,
    show_fan_chart: bool = False,
    fan_levels: Tuple[float, ...] = _fan_levels_default(),
    fan_alpha_base: float = 0.38,
    fan_alpha_step: float = 0.10,
    mode_tag_in_legend: bool = True,
    mark_last_test_point: bool = True,
    show_main_ci: bool = True,
) -> None:
    if df_final_vt is None or df_final_vt.empty:
        return

    need = {"t", "yhat_final_mean"}
    missing = need - set(df_final_vt.columns)
    if missing:
        raise KeyError(f"[plot] final_ensemble_df missing: {sorted(missing)}")

    mode_tag = {"std": "band", "sem": "Mean CI", "quantile": "Quantile band"}.get(ci_mode, "Band")

    # Inter legend title with optional n
    n_eff = None
    if n_members_col in df_final_vt.columns:
        try:
            n_eff = int(pd.to_numeric(df_final_vt[n_members_col], errors="coerce").dropna().max())
        except Exception:
            n_eff = None
    legend_title = "Inter" + (f" (n={n_eff})" if n_eff and n_eff > 0 else "")

    for split_name, g in df_final_vt.groupby("split", sort=False):
        g = g.sort_values("t").copy()
        label  = str(split_name).capitalize()
        is_test = "test" in str(split_name).lower()
        line_c = colors.get("test_initial", "#2ca02c") if is_test else colors.get("validation", "#d62728")
        fill_c = colors.get("fill_test", "rgba(76,175,80,0.12)") if is_test else colors.get("fill_val", "rgba(214,39,40,0.12)")

        # --- FAN CHART (if requested) ---
        if show_fan_chart:
            qcols = extract_quantile_columns(g, prefix="final")
            if qcols is not None and 0.5 in qcols and 0.95 in qcols:
                mean_series = g["yhat_final_mean"]
                avail = set(qcols.keys())
                if set(fan_levels).issubset(avail):
                    _draw_fan(
                        fig, g["t"], fan_levels, qcols,
                        base_fill=fill_c,
                        alpha_base=fan_alpha_base,
                        alpha_step=fan_alpha_step,
                        legendgroup="Inter",
                        label_prefix=f"Fan ({label})",
                        legend_title=legend_title,             # NEW
                    )
                else:
                    scale = _choose_scale_for_mode(g, ci_mode, n_members_col, std_col="std_final")
                    bands = _fan_from_gaussian(mean_series, scale, fan_levels)
                    _draw_fan_from_pairs(
                        fig, g["t"], bands,
                        base_fill=fill_c,
                        alpha_base=fan_alpha_base,
                        alpha_step=fan_alpha_step,
                        legendgroup="Inter",
                        label_prefix=f"Fan ({label})",
                        legend_title=legend_title,             # NEW
                    )
            else:
                scale = _choose_scale_for_mode(g, ci_mode, n_members_col, std_col="std_final")
                mean_series = g["yhat_final_mean"]
                bands = _fan_from_gaussian(mean_series, scale, fan_levels)
                _draw_fan_from_pairs(
                    fig, g["t"], bands,
                    base_fill=fill_c,
                    alpha_base=fan_alpha_base,
                    alpha_step=fan_alpha_step,
                    legendgroup="Inter",
                    label_prefix=f"Fan ({label})",
                    legend_title=legend_title,                 # NEW
                )

        # --- Bands ---
        lower, upper = _compute_bounds_for_mode(
            g,
            ci_mode=ci_mode, ci_k=ci_k, ci_quantiles=ci_quantiles,
            std_col="std_final", mean_col="yhat_final_mean",
            n_members_col=n_members_col,
            qlow_col="yhat_q_low_final", qhigh_col="yhat_q_high_final",
        )

        if band_inner_alpha is not None and ci_mode in ("std", "sem"):
            k_inner = min(ci_k, 1.0)
            inner_lo, inner_hi = _compute_bounds_for_mode(
                g, ci_mode=ci_mode, ci_k=k_inner, ci_quantiles=ci_quantiles,
                std_col="std_final", mean_col="yhat_final_mean", n_members_col=n_members_col
            )
            fig.add_trace(go.Scatter(
                x=g["t"], y=inner_lo, mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False, legendgroup="Inter"
            ))
            fig.add_trace(go.Scatter(
                x=g["t"], y=inner_hi, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=_with_alpha(fill_c, band_inner_alpha),
                name=f"Inner {mode_tag} ({label})" if mode_tag_in_legend else f"Inner Band ({label})",
                hovertemplate="<b>Inner band</b><extra></extra>",
                legendgroup="Inter", legendgrouptitle_text=legend_title
            ))

        if show_main_ci:
            fig.add_trace(go.Scatter(
                x=g["t"], y=lower, mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False, legendgroup="Inter"
            ))
            band_name = (
                f"{int(100*(ci_quantiles[1]-ci_quantiles[0]))}% {mode_tag} ({label})"
                if ci_mode == "quantile"
                else (f"95% {mode_tag} ({label})" if abs(ci_k - 1.96) < 1e-6
                      else f"{mode_tag} k={ci_k:g} ({label})")
            )
            fig.add_trace(go.Scatter(
                x=g["t"], y=upper, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=_with_alpha(fill_c, band_alpha),
                name=band_name, hovertemplate="<b>Band</b><extra></extra>",
                legendgroup="Inter", legendgrouptitle_text=legend_title
            ))

        fig.add_trace(go.Scatter(
            x=g["t"], y=g["yhat_final_mean"], mode="lines",
            name=f"Final Mean ({label})",
            line=dict(color=line_c, width=4),
            hovertemplate=hover_y(),
            legendgroup="Inter", legendgrouptitle_text=legend_title
        ))

        if mark_last_test_point and is_test and not g.empty:
            last_t = g["t"].iloc[-1]
            last_y = g["yhat_final_mean"].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_t], y=[last_y], mode="markers",
                name="Last Test Mean",
                marker=dict(size=9, line=dict(width=0.5, color="#333")),
                hovertemplate="<b>Last Test</b><br>t=%{x}<br>mean=%{y:.2f}<extra></extra>",
                showlegend=False, legendgroup="Inter"
            ))


def add_intra_family_traces(
    fig: go.Figure,
    df_intra_vt: "pd.DataFrame",
    colors: Dict[str, str],
    *,
    visible_default: bool,
    band_alpha: float,
    band_inner_alpha: Optional[float],
    ci_mode: str,
    ci_k: float,
    ci_quantiles: Tuple[float, float],
    n_members_col: str,
    ci_k_by_arch: Optional[Dict[str, float]] = None,
) -> None:
    if df_intra_vt is None or df_intra_vt.empty:
        return
    need = {"t", "yhat_family_mean", "arch"}
    missing = need - set(df_intra_vt.columns)
    if missing:
        raise KeyError(f"[plot] intra_family_df missing: {sorted(missing)}")

    for (arch, split_name), g in df_intra_vt.groupby(["arch", "split"], sort=False):
        g = g.sort_values("t").copy()
        arch = str(arch)
        label_split = str(split_name).capitalize()
        is_test = "test" in str(split_name).lower()
        line_c = colors.get("test_initial") if is_test else colors.get("validation")
        fill_c = colors.get("fill_test") if is_test else colors.get("fill_val")

        # legend group title with n
        n_eff = None
        if n_members_col in g.columns:
            try:
                n_eff = int(pd.to_numeric(g[n_members_col], errors="coerce").dropna().max())
            except Exception:
                n_eff = None
        group_id = f"Intra — {arch}"
        group_title = f"{group_id}" + (f" (n={n_eff})" if n_eff and n_eff > 0 else "")

        local_k = ci_k_by_arch.get(arch, ci_k) if ci_k_by_arch else ci_k

        lower, upper = _compute_bounds_for_mode(
            g,
            ci_mode=ci_mode, ci_k=local_k, ci_quantiles=ci_quantiles,
            std_col="std_family", mean_col="yhat_family_mean",
            n_members_col=n_members_col,
            qlow_col="q_lo_family", qhigh_col="q_hi_family",
        )

        if band_inner_alpha is not None and ci_mode in ("std", "sem"):
            k_inner = min(local_k, 1.0)
            inner_lo, inner_hi = _compute_bounds_for_mode(
                g, ci_mode=ci_mode, ci_k=k_inner, ci_quantiles=ci_quantiles,
                std_col="std_family", mean_col="yhat_family_mean", n_members_col=n_members_col
            )
            fig.add_trace(go.Scatter(
                x=g["t"], y=inner_lo, mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False,
                visible=True if visible_default else "legendonly",
                legendgroup=group_id, legendgrouptitle_text=group_title
            ))
            fig.add_trace(go.Scatter(
                x=g["t"], y=inner_hi, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=_with_alpha(fill_c, band_inner_alpha),
                name=f"Inner Band ({label_split})",
                hovertemplate="<b>Inner band</b><extra></extra>",
                visible=True if visible_default else "legendonly",
                legendgroup=group_id, legendgrouptitle_text=group_title
            ))

        fig.add_trace(go.Scatter(
            x=g["t"], y=lower, mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
            visible=True if visible_default else "legendonly",
            legendgroup=group_id, legendgrouptitle_text=group_title
        ))
        fig.add_trace(go.Scatter(
            x=g["t"], y=upper, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_with_alpha(fill_c, band_alpha),
            name=f"{arch} Band ({label_split})",
            hovertemplate="<b>Band</b><extra></extra>",
            visible=True if visible_default else "legendonly",
            legendgroup=group_id, legendgrouptitle_text=group_title
        ))

        fig.add_trace(go.Scatter(
            x=g["t"], y=g["yhat_family_mean"], mode="lines",
            name=f"{arch} Mean ({label_split})",
            line=dict(dash="dash", width=1.8, color=line_c),
            hovertemplate=hover_y(),
            visible=True if visible_default else "legendonly",
            legendgroup=group_id, legendgrouptitle_text=group_title
        ))







def _draw_fan(
    fig: go.Figure,
    x: "pd.Series",
    levels: Tuple[float, ...],
    qcols: Dict[float, "pd.Series"],
    *,
    base_fill: str,
    alpha_base: float,
    alpha_step: float,
    legendgroup: str,
    label_prefix: str,
    legend_title: Optional[str] = None,   # NEW
) -> None:
    # draw from outer to inner so fills stack nicely
    for i, q in enumerate(sorted(levels, reverse=True)):
        lo = qcols.get(1.0 - q, None)
        hi = qcols.get(q, None)
        if lo is None or hi is None:
            continue
        alpha = max(min(alpha_base + i * alpha_step, 1.0), 0.0)
        fig.add_trace(go.Scatter(
            x=x, y=lo, mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
            legendgroup=legendgroup, legendgrouptitle_text=legend_title
        ))
        fig.add_trace(go.Scatter(
            x=x, y=hi, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_with_alpha(base_fill, alpha),
            name=f"{label_prefix} {int(q*100)}%",
            hovertemplate="<b>Fan</b><extra></extra>",
            legendgroup=legendgroup, legendgrouptitle_text=legend_title
        ))


def _draw_fan_from_pairs(
    fig: go.Figure,
    x: "pd.Series",
    pairs: Dict[float, Tuple["pd.Series", "pd.Series"]],
    *,
    base_fill: str,
    alpha_base: float,
    alpha_step: float,
    legendgroup: str,
    label_prefix: str,
    legend_title: Optional[str] = None,   # NEW
) -> None:
    for i, q in enumerate(sorted(pairs.keys(), reverse=True)):
        lo, hi = pairs[q]
        alpha = max(min(alpha_base + i * alpha_step, 1.0), 0.0)
        fig.add_trace(go.Scatter(
            x=x, y=lo, mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
            legendgroup=legendgroup, legendgrouptitle_text=legend_title
        ))
        fig.add_trace(go.Scatter(
            x=x, y=hi, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=_with_alpha(base_fill, alpha),
            name=f"{label_prefix} {int(q*100)}%",
            hovertemplate="<b>Fan</b><extra></extra>",
            legendgroup=legendgroup, legendgrouptitle_text=legend_title
        ))



def add_champion_traces(fig: go.Figure, champs_vt: pd.DataFrame) -> None:
    if champs_vt is None or champs_vt.empty or not {"t", "yhat"}.issubset(champs_vt.columns):
        return
    group_cols = [c for c in ["job_hash", "split"] if c in champs_vt.columns] or ["well"]
    for _, g in champs_vt.groupby(group_cols):
        g = g.sort_values("t")
        fig.add_trace(go.Scatter(
            x=g["t"], y=g["yhat"], mode="lines",
            line=dict(color="rgba(80,80,80,0.45)", width=1),
            opacity=0.5, hoverinfo="skip", showlegend=False
        ))


def polish_layout(fig: go.Figure, colors: Dict[str, str], title_txt: str,
                  font_scale: float, width: int, height: int,
                  *, yaxis_log: bool = False, y_tickformat: str = ",~s") -> None:
    FONT_FAMILY = "Inter, Arial, sans-serif"
    legend_fs = 14 * font_scale
    axis_fs   = 16 * font_scale

    fig.update_layout(
        title=dict(text=title_txt, x=0.5, y=0.93, xanchor="center", yanchor="top"),
        xaxis_title="Time",
        yaxis_title="Production Rate",
        font=dict(family=FONT_FAMILY, color=colors.get("text", "#111"), size=axis_fs),
        xaxis=dict(gridcolor=colors.get("grid", "#EAEAEA"), zeroline=False, title_font_size=axis_fs + 2, tickfont_size=axis_fs),
        yaxis=dict(gridcolor=colors.get("grid", "#EAEAEA"), zeroline=False, title_font_size=axis_fs + 2, tickfont_size=axis_fs,
                   type=("log" if yaxis_log else "-"), tickformat=y_tickformat, separatethousands=True),
        legend=dict(
            orientation="v", x=1.02, xanchor="left", y=1.00, yanchor="top",
            bgcolor="rgba(255,255,255,0.92)", bordercolor=colors.get("grid", "#EAEAEA"),
            borderwidth=1, font=dict(size=int(14 * font_scale)), traceorder="normal",
            tracegroupgap=6, groupclick="toggleitem", # allow group toggles
        ),
        plot_bgcolor=colors.get("plot_bgcolor", "white"),
        paper_bgcolor=colors.get("paper_bgcolor", "white"),
        margin=dict(l=90, r=220, t=110, b=70),
        hovermode="x unified",
        width=width, height=height,
        colorway=[
            c for c in [
                colors.get("validation"),
                colors.get("test_rolling"),
                colors.get("train_from_y"),
                colors.get("actual_post_train"),
                colors.get("test_initial"),
                colors.get("train_from_x"),
            ] if c
        ] + qcolors.Plotly + qcolors.D3 + qcolors.Set2
    )


def _default_k_grid(base_k: float) -> np.ndarray:
    """
    Sensible k grid around a base multiplier (e.g., 1.96) spanning narrow to wide bands.
    """
    base = float(base_k) if np.isfinite(base_k) and base_k > 0 else 1.96
    # from 0.25*base to 3.0*base, denser near base
    left  = np.linspace(0.25 * base, 0.9 * base, 10, endpoint=False)
    mid   = np.linspace(0.9 * base, 1.2 * base, 13, endpoint=True)
    right = np.linspace(1.2 * base, 3.0 * base, 12, endpoint=True)
    grid  = np.unique(np.clip(np.concatenate([left, mid, right]), 1e-6, None))
    return grid

def _estimate_k_on_validation(
    g_val: pd.DataFrame,
    *,
    ci_mode: str,
    target_cov: float,
    mean_col: str,
    std_col: str,
    n_members_col: str,
    k_search_grid: Optional[Sequence[float]],
) -> Tuple[float, float, float]:
    """
    Grid-search k to match target coverage on Validation.
    Returns (k_eff, cov_at_k, sharp_at_k).
    """
    if g_val is None or g_val.empty:
        raise ValueError("Validation slice is empty; cannot calibrate bands.")
    if str(ci_mode).lower() not in {"std", "sem"}:
        # For quantile mode, k doesn't apply.
        return np.nan, np.nan, np.nan

    # Build grid
    grid = np.asarray(k_search_grid if k_search_grid is not None else _default_k_grid(1.96), dtype=float)
    grid = grid[np.isfinite(grid) & (grid > 0)]
    if grid.size == 0:
        grid = _default_k_grid(1.96)

    # Evaluate each k: minimize |coverage - target|; tiebreak on smaller sharpness
    best = None
    for k in grid:
        cov, sharp = _compute_coverage_and_sharpness(
            g_val,
            ci_mode=ci_mode,
            coverage=target_cov,
            mean_col=mean_col,
            std_col=std_col,
            n_members_col=n_members_col,
            qlow_col="yhat_q_low_final",
            qhigh_col="yhat_q_high_final",
            override_k=float(k),
        )
        if not np.isfinite(cov):
            continue
        loss = abs(cov - target_cov)
        key  = (loss, sharp if np.isfinite(sharp) else np.inf)
        if (best is None) or (key < best[0]):
            best = (key, float(k), float(cov), float(sharp if np.isfinite(sharp) else np.nan))

    if best is None:
        # Could happen if ytrue missing; return NaNs to signal no calibration
        return np.nan, np.nan, np.nan

    _, k_eff, cov_eff, sharp_eff = best
    return k_eff, cov_eff, sharp_eff


def plot_conjugated_ensemble(
    final_ensemble_df: pd.DataFrame,
    intra_family_df: pd.DataFrame,
    champion_series_df: pd.DataFrame,
    full_history_df: pd.DataFrame,
    boundaries: Dict[str, Any],
    well: str,
    title: Optional[str] = None,
    palette: str = "default",
    show_family_traces: bool = False,
    show_champion_traces: bool = False,
    *,
    show_train_reconstruction: bool = True,
    # --- visual controls ---
    font_scale: float = 1.2,
    width: int = 1100,
    height: int = 700,
    show: bool = True,
    # --- unified uncertainty/band controls (intra & inter) ---
    band_alpha: float = 0.7,                        # outer band opacity
    band_inner_alpha: Optional[float] = 0.85,       # inner band (pseudo-gradient)
    ci_mode: str = "std",                          # "std" | "sem" | "quantile"  (normalized below)
    ci_k: float = 1.96,                             # multiplier for "std"/"sem"
    ci_quantiles: Tuple[float, float] = (0.10, 0.90),  # for "quantile" mode
    n_members_col: str = "n_members",               # for "sem" (fallbacks if missing)
    # --- fan chart (50/70/90/95%), gaussian fallback if quantiles missing ---
    show_fan_chart: bool = True,
    fan_levels: Tuple[float, ...] = _fan_levels_default(),
    fan_alpha_base: float = 0.30,
    fan_alpha_step: float = 0.08,
    # --- extras: inset, log scale, formatting ---
    show_residuals_inset: bool = False,
    yaxis_log: bool = False,
    y_tickformat: str = ",~s",
    # --- metrics boxes ---
    show_uncertainty_metrics: bool = True,
    coverage_level: float = 0.90,
    # --- Inter calibration on Validation ---
    calibrate_on_val: bool = True,
    calibration_target: float = 0.90,
    k_search_grid: Optional[Sequence[float]] = None,
    # --- Intra calibration options ---
    calibrate_intra_on_val: bool = True,                 # per-arch calibration (preferred)
    apply_calibration_to_intra: bool = True,             # legacy: reuse Inter k for all Intra
) -> go.Figure:
    """
    Conjugated ensemble plot (Inter & Intra) with unified band logic, fan chart, and
    optional calibration on Validation.

    Calibrations:
      • Inter: If calibrate_on_val=True and ci_mode in {"std","sem"}, estimate k_eff on Validation
        to reach Coverage≈calibration_target. Apply the same k_eff to Inter Val/Test and metrics.

      • Intra: If calibrate_intra_on_val=True, estimate k_eff per family (arch) on Validation using
        yhat_family_mean/std_family and apply it to that family’s Val/Test and metrics.
        Otherwise, if apply_calibration_to_intra=True, reuse the Inter k for all families.
    """
    # -------------------------------------------------------------------------
    # Normalize ci_mode ONCE and use the normalized value everywhere
    # -------------------------------------------------------------------------
    ci_mode_norm = str(ci_mode or "std").lower()
    if ci_mode_norm not in {"std", "sem", "quantile"}:
        ci_mode_norm = "std"

    colors = _get_color_palette(palette)

    # --- Slice by well ---
    well_final = by_well(final_ensemble_df, well, "final_ensemble_df")
    well_intra = by_well(intra_family_df,  well, "intra_family_df")
    champs_all = by_well(champion_series_df, well, "champion_series_df")



    # --- Validate history ---
    if full_history_df is None or full_history_df.empty:
        raise ValueError(f"[plot] full_history_df is empty for well='{well}'.")
    fh = full_history_df.copy()
    if "t" not in fh.columns or "ytrue" not in fh.columns:
        raise KeyError("[plot] full_history_df must have t and ytrue.")
    fh = fh.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)

    # --- Attach/normalize time axis for all frames ---
    well_final = coalesce_time_column(ensure_time_axis(well_final, "final_ensemble_df", fh), "final_ensemble_df")
    well_intra = coalesce_time_column(ensure_time_axis(well_intra, "intra_family_df", fh),   "intra_family_df")
    champs_all = coalesce_time_column(ensure_time_axis(champs_all, "champion_series_df", fh), "champion_series_df")

    # --- Boundaries (index-space → axis-space) ---
    t_axis   = fh["t"].reset_index(drop=True)
    n_hist   = len(t_axis)
    tr_end_i = safe_int(boundaries.get("train_end", 0),            0, n_hist - 1)
    va_end_i = safe_int(boundaries.get("val_end", tr_end_i),       0, n_hist - 1)

    # --- Figure + regions ---
    fig = go.Figure()
    add_region_backgrounds(fig, colors, t_axis, tr_end_i, va_end_i)

    # Region titles
    title_fs = 18 * font_scale
    fig.add_annotation(text="<b>Train</b>", x=t_axis.iloc[:tr_end_i + 1].median(), yref="paper", y=0.98,
                       showarrow=False, font=dict(size=title_fs, color=colors.get("text", "#111")))
    fig.add_annotation(text="<b>Validation</b>", x=t_axis.iloc[tr_end_i:va_end_i + 1].median(), yref="paper", y=0.98,
                       showarrow=False, font=dict(size=title_fs, color=colors.get("validation", "#d62728")))
    fig.add_annotation(text="<b>Test</b>", x=t_axis.iloc[va_end_i:].median(), yref="paper", y=0.98,
                       showarrow=False, font=dict(size=title_fs, color=colors.get("test_border", colors.get("test_rolling", "#2ca02c"))))

    # --- Ground truth ---
    add_ground_truth(fig, fh, colors)
    # --- Optional: Train reconstruction from champions (primary) ---
    if show_train_reconstruction and not champs_all.empty:
        prim = champs_all[champs_all.get("is_primary", False)].copy()
        if prim.empty and "job_hash" in champs_all.columns and not champs_all.empty:
            fallback_job = champs_all["job_hash"].iloc[0]
            log.warning("[ensemble_plot] well=%s: no is_primary; fallback job=%s.", well, fallback_job)
            prim = champs_all[champs_all["job_hash"] == fallback_job].copy()
        if not prim.empty:
            if "split" in prim.columns:
                prim = prim[prim["split"].astype(str).str.lower().eq("train")]
            if "idx" in prim.columns and not prim.empty:
                prim = prim[prim["idx"].astype(int) <= tr_end_i]
        if {"t", "yhat"}.issubset(prim.columns) and not prim.empty:
            prim = prim.sort_values("t")
            fig.add_trace(go.Scatter(
                x=prim["t"], y=prim["yhat"],
                name="Train Reconstruction (Primary)",
                mode="lines",
                line=dict(color=colors.get("train_from_x", "#9467bd"), width=2.25),
                hovertemplate=hover_y()
            ))

    # ------------------------------------------------
    # Inter-family: calibrate on Validation
    # ------------------------------------------------
    well_final_vt = only_val_test(well_final, "final_ensemble_df")
    ci_k_used_inter = float(ci_k)
    calibration_note = ""



    if calibrate_on_val and (ci_mode_norm in {"std", "sem"}) and not well_final_vt.empty:
        g_val = well_final_vt[well_final_vt["split"].astype(str).str.lower().str.startswith("val")].copy()
        if not g_val.empty and "ytrue" in g_val.columns:
            try:
                k_eff, cov_eff, _ = _estimate_k_on_validation(
                    g_val,
                    ci_mode=ci_mode_norm,
                    target_cov=float(calibration_target),
                    mean_col="yhat_final_mean",
                    std_col="std_final",
                    n_members_col=n_members_col,
                    k_search_grid=k_search_grid,
                )
                if np.isfinite(k_eff):
                    ci_k_used_inter = float(k_eff)
                    calibration_note = (
                        f" · <span style='font-size:0.85em'>Inter bands calibrated on Val "
                        f"(target={int(calibration_target*100)}%, k={ci_k_used_inter:.3g}, cov={cov_eff*100:.1f}%)</span>"
                    )
            except Exception as e:
                log.exception("[ensemble_plot] Inter calibration failed; using ci_k default. Error: %s", e)
        else:
            log.warning("[ensemble_plot] Cannot calibrate Inter: Validation slice empty or missing ytrue.")

    # --- Draw Inter (VAL/TEST) with calibrated k ---
    if not well_final_vt.empty:
        add_final_ensemble_traces(
            fig, well_final_vt, colors,
            band_alpha=band_alpha, band_inner_alpha=band_inner_alpha,
            ci_mode=ci_mode_norm, ci_k=ci_k_used_inter, ci_quantiles=ci_quantiles, n_members_col=n_members_col,
            show_fan_chart=show_fan_chart, fan_levels=fan_levels,
            fan_alpha_base=fan_alpha_base, fan_alpha_step=fan_alpha_step,
            mode_tag_in_legend=True, mark_last_test_point=True,
            show_main_ci=True,  # keep fan as the main visual story
        )
    else:
        log.info("[ensemble_plot] well=%s: final_ensemble_df empty.", well)

    # ------------------------------------------------
    # Intra-family: per-arch calibration on Validation (preferred)
    # ------------------------------------------------
    well_intra_vt = only_val_test(well_intra, "intra_family_df")
    ci_k_by_arch: Dict[str, float] = {}


    if show_family_traces and not well_intra_vt.empty and (ci_mode_norm in {"std", "sem"}):
        if calibrate_intra_on_val:
            for arch, g_arch in well_intra_vt.groupby("arch", sort=False):
                g_val = g_arch[g_arch["split"].astype(str).str.lower().str.startswith("val")]
                if not g_val.empty and "ytrue" in g_val.columns:
                    try:
                        k_eff, _, _ = _estimate_k_on_validation(
                            g_val,
                            ci_mode=ci_mode_norm,
                            target_cov=float(calibration_target),
                            mean_col="yhat_family_mean",
                            std_col="std_family",
                            n_members_col=n_members_col,
                            k_search_grid=k_search_grid,
                        )
                        if np.isfinite(k_eff):
                            ci_k_by_arch[str(arch)] = float(k_eff)
                    except Exception as e:
                        log.exception("[ensemble_plot] Intra calibration failed for arch=%s. Error: %s", arch, e)
        elif apply_calibration_to_intra:
            # Legacy: reuse Inter k for all families
            for arch in well_intra_vt["arch"].astype(str).unique():
                ci_k_by_arch[arch] = float(ci_k_used_inter)

    # --- Draw Intra with per-arch calibrated k (when available) ---
    if show_family_traces and not well_intra_vt.empty:
        add_intra_family_traces(
            fig, well_intra_vt, colors,
            visible_default=well_final_vt.empty,  # if Inter missing, Intra starts visible
            band_alpha=band_alpha, band_inner_alpha=band_inner_alpha,
            ci_mode=ci_mode_norm, ci_k=float(ci_k), ci_quantiles=ci_quantiles, n_members_col=n_members_col,
            ci_k_by_arch=ci_k_by_arch if ci_k_by_arch else None,
        )

    # --- Champions (optional) ---
    if show_champion_traces and not champs_all.empty:
        champs_vt = only_val_test(champs_all, "champion_series_df (champ_traces)")
        add_champion_traces(fig, champs_vt)

    # --- Residuals inset (VAL+TEST vs final mean) ---
    if show_residuals_inset and not well_final_vt.empty:
        res = compute_residuals_vt(well_final_vt, mean_col="yhat_final_mean")
        try:
            add_residuals_inset(fig, res)  # assumed available in your utils
        except Exception:
            pass

    # --- Metrics (Inter) with calibrated Inter k ---
    if show_uncertainty_metrics and not well_final_vt.empty:
        val_m, test_m = summarize_split_metrics_std_or_sem(
            well_final_vt, fh,
            ci_mode=ci_mode_norm, coverage=coverage_level,
            mean_col="yhat_final_mean", std_col="std_final", n_members_col=n_members_col,
            override_k=ci_k_used_inter if ci_mode_norm in {"std", "sem"} else None,
        )
        add_uncertainty_metrics_annotations(
            fig,
            metrics_val=val_m, metrics_test=test_m,
            colors=colors, font_scale=font_scale,
            is_cumulative_plot=False,
            sharp_suffix="",
        )


    # --- Metrics (Intra) per arch, using each arch's calibrated k ---
    if show_uncertainty_metrics and show_family_traces and not well_intra_vt.empty:
        for arch_name, g_arch in well_intra_vt.groupby("arch", sort=False):
            override_k = None
            if ci_mode_norm in {"std", "sem"}:
                override_k = ci_k_by_arch.get(str(arch_name), float(ci_k))
            val_m_intra, test_m_intra = summarize_split_metrics_std_or_sem(
                g_arch, fh,
                ci_mode=ci_mode_norm, coverage=coverage_level,
                mean_col="yhat_family_mean", std_col="std_family", n_members_col=n_members_col,
                override_k=override_k,
            )
            # Use the SAME annotation component as Inter for visual consistency
            add_uncertainty_metrics_annotations(
                fig,
                metrics_val=val_m_intra, metrics_test=test_m_intra,
                colors=colors, font_scale=font_scale * 0.95,
                is_cumulative_plot=False,
                sharp_suffix="",
            )


    # --- Title & layout ---
    mode_tag = {"std": "Prediction band", "sem": "Mean CI", "quantile": "Quantile band"}.get(ci_mode_norm, "Band")
    title_txt = title or f"<b>Conjugated Ensemble — {well}</b> · <span style='font-size:0.85em'>{mode_tag}</span>{calibration_note}"
    polish_layout(fig, colors, title_txt, font_scale, width, height,
                  yaxis_log=yaxis_log, y_tickformat=y_tickformat)

    if show:
        fig.show(config={"toImageButtonOptions": {"format": "png", "filename": "conjugated_ensemble", "scale": 3}})
    log.info("[ensemble_plot] Completed for well=%s", well)
    return fig
