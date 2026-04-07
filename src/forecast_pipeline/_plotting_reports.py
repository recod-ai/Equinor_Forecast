# forecast_pipeline/_plotting_reports.py
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import f_classif

from rich.box import SQUARE
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from utils.utilities import (
    _detect_family_for_df,
    _normalize_for_champions,
    resolve_champions_columns_minimal,
)


def render_champions_view_auto(
    df: pd.DataFrame,
    *,
    per_well_k: int = 3,
    metric: str = "val_smape_agg",
    lower_is_better: bool = True,
    caption: str | None = None,
) -> "pd.io.formats.style.Styler":
    """
    Auto-configuring wrapper for render_champions_view.

    This function prepares the DataFrame by:
      - Normalizing columns (e.g., 'n_epochs' -> 'epochs').
      - Detecting the data family ('seq2', 'arps', etc.) to select relevant columns.
      - For the 'arps' family, it avoids forcing an 'architecture_name' column
        to provide a cleaner, more relevant view.
    It then calls the main render_champions_view with a generated configuration.
    """
    if df is None or df.empty:
        return pd.DataFrame({"info": ["No data"]}).style

    d = _normalize_for_champions(df)
    fam = _detect_family_for_df(d)
    cols = resolve_champions_columns_minimal(d, metric=metric)

    # For ARPS family, avoid forcing 'architecture_name' by providing no candidates
    arch_candidates = () if fam == "arps" else ("architecture_name", "architecture")

    return render_champions_view(
        d,
        columns=cols,
        per_well_k=per_well_k,
        sort_by=metric,
        lower_is_better=lower_is_better,
        arch_col_candidates=arch_candidates,
        caption=caption,
    )


def render_champions_view(
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    per_well_k: int = 3,
    sort_by: str = "val_smape_agg",
    lower_is_better: bool = True,
    well_col: str = "well",
    arch_col_candidates: List[str] = ("architecture_name", "architecture"),
    strategy_col: str = "physics_strategy",
    metric_cols_for_gradient: Optional[List[str]] = None,
    caption: Optional[str] = None,
) -> "pd.io.formats.style.Styler":
    """
    Build a story-driven, operator-friendly table of top-performing models.

    This function ranks models within each well based on a specified metric,
    selects the top 'k' performers, and applies rich styling for clear
    visual analysis. It includes color-coded categories, banded rows for
    readability, and metric-based background gradients.
    """
    if df is None or df.empty:
        return pd.DataFrame({"info": ["No data"]}).style

    # Robustly resolve the architecture column from the list of candidates
    arch_col = next((c for c in arch_col_candidates if c in df.columns), None)

    # If no columns are specified, build a default set
    if columns is None:
        default_cols = [well_col]
        if arch_col:
            default_cols.append(arch_col)
        default_cols.extend([strategy_col, "val_smape_agg", "horizon", "lag_window", "data_sample", "epochs", "batch_size", "learning_rate"])
        columns = default_cols
    
    # Ensure all requested columns actually exist in the DataFrame
    cols_available = [c for c in columns if c in df.columns]

    # The 'well' column is essential for grouping; add it if it was missed
    if well_col not in cols_available and well_col in df.columns:
        cols_available.insert(0, well_col)

    work = df.copy()

    # Rank entries within each well and select the top-k
    is_ascending = bool(lower_is_better)
    if sort_by not in work.columns:
        # Fallback sorting if the primary metric is not available
        sort_by = "weighted_score" if "weighted_score" in work.columns else next((c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])), None)
        if sort_by is None:
             raise ValueError("No valid numeric column found to sort by.")

    work["__rank__"] = (
        work.groupby(well_col)[sort_by]
            .rank(method="first", ascending=is_ascending)
            .astype(int)
    )
    if per_well_k and per_well_k > 0:
        work = work[work["__rank__"] <= per_well_k]

    work = work.sort_values([well_col, "__rank__", sort_by], ascending=[True, True, is_ascending])
    
    # Create the final display view with a clean 'rank' column
    view = work[cols_available + ["__rank__"]].copy()
    cols_final = ["__rank__"] + [c for c in view.columns if c != "__rank__"]
    view = view[cols_final].reset_index(drop=True).rename(columns={"__rank__": "rank"})

    # --- Styling Logic ---
    def _category_colors(values, *, s=65, l=38):
        """Generates a consistent color map for categorical values."""
        unique_vals = list(dict.fromkeys([str(v) for v in values]))
        n = max(1, len(unique_vals))
        return {val: f"hsl({int(360 * i / n)}, {s}%, {l}%)" for i, val in enumerate(unique_vals)}

    arch_map = _category_colors(view[arch_col]) if arch_col and arch_col in view.columns else {}
    strat_map = _category_colors(view[strategy_col], s=55, l=42) if strategy_col in view.columns else {}

    # Determine row banding for wells
    wells = view[well_col].tolist()
    well_first_indices = [i for i, (current, prev) in enumerate(zip(wells, [None] + wells)) if current != prev]
    band_flags = np.cumsum([1 if i in well_first_indices else 0 for i in range(len(view))]) % 2
    band_colors = ("#f9fafb", "#ffffff")

    def _style_rows_by_well(row):
        """Applies top borders and alternating background colors for each well group."""
        i = row.name
        style = "border-top: 3px solid #111827;" if i in well_first_indices else "border-top: 1px solid #e5e7eb;"
        style += f"background-color: {band_colors[band_flags[i]]};"
        if row.get("rank", 0) == 1:
            style += "font-weight: 600;"
        return [style] * len(row)

    def _style_category_col(series, cmap):
        """Applies a background color from a map to a categorical column."""
        return [f"background-color: {cmap.get(str(v))}; color: #fff; font-weight: 600;" if str(v) in cmap else "" for v in series]
    
    if metric_cols_for_gradient is None:
        metric_cols_for_gradient = [c for c in ["val_smape_agg", "val_smape_cum", "weighted_score", "robust_score"] if c in view.columns]
    
    caption = caption or f"Top {per_well_k} per well · sorted by {sort_by} ({'lower' if is_ascending else 'higher'} is better)"

    # Build and apply styles using the Styler object
    styler = view.style.set_caption(caption).set_table_styles(
        [
            {"selector": "caption", "props": "caption-side: top; font-size: 1.15rem; font-weight: 600; color: #111827;"},
            {"selector": "thead th", "props": "background-color: #111827; color: #ffffff; font-weight: 700; font-size: 0.95rem;"}
        ],
        overwrite=False
    ).apply(_style_rows_by_well, axis=1)

    if arch_col and arch_col in view.columns:
        styler = styler.apply(lambda s: _style_category_col(s, arch_map), subset=[arch_col])
    if strategy_col in view.columns:
        styler = styler.apply(lambda s: _style_category_col(s, strat_map), subset=[strategy_col])

    for mcol in metric_cols_for_gradient:
        if mcol in view.columns:
            cmap = "RdYlGn_r" if lower_is_better else "RdYlGn"
            styler = styler.background_gradient(subset=[mcol], cmap=cmap)

    if "rank" in view.columns:
        styler = styler.set_properties(subset=["rank"], **{"text-align": "center", "width": "4ch", "font-weight": "700", "color": "#111827"})

    styler = styler.set_properties(**{"font-size": "0.95rem"})

    # Apply number formatting for better readability
    styler = styler.format(_build_number_formatters(view))

    return styler

def _build_number_formatters(df: pd.DataFrame) -> Dict[str, str]:
    """Pretty numeric formats without scientific explosion; customize as needed."""
    fmts: Dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            if "rate" in col.lower() or "smape" in col.lower():
                fmts[col] = "{:.3f}"
            elif "learning_rate" in col:
                fmts[col] = "{:.3g}"  # concise for small LR
            else:
                fmts[col] = "{:.3g}"
        elif pd.api.types.is_integer_dtype(df[col]):
            fmts[col] = "{:d}"
    return fmts


# ==============================================================================
# 6. ENHANCED VISUAL ANALYSIS FUNCTIONS
# ==============================================================================
import plotly.express as px



def plot_champions_well(
    master_df: pd.DataFrame, 
    metric: str = "weighted_score", 
    group_by: str = "auto",
    font_scale: float = 1.8
):
    """
    Winning 'group' per well (lower is better), onde o 'group' é:
      - Seq2  : aggregation_method (fallback -> physics_strategy -> architecture)
      - Arps  : variant            (fallback -> architecture)
      - Outro : architecture (fallback -> architecture_name)
    """
    console = Console()
    console.print(Markdown(f"--- \n### 🥇 **Best Group Performance per Well**"))

    if master_df is None or master_df.empty or metric not in master_df.columns:
        print("No data/metric to plot.")
        return

    df = master_df.copy()

    # --- Lógica interna (sem alterações) ---
    def _has(col: str) -> bool:
        return col in df.columns
    def _arch_col() -> str | None:
        for c in ("architecture", "architecture_name"):
            if _has(c):
                return c
        return None
    def _is_seq2() -> bool:
        a = df.get("architecture", pd.Series([], dtype=str)).astype(str)
        an = df.get("architecture_name", pd.Series([], dtype=str)).astype(str)
        return a.str.contains("Seq2", na=False).any() or an.str.contains("Seq2", na=False).any()
    def _is_arps() -> bool:
        a = df.get("architecture", pd.Series([], dtype=str)).astype(str)
        an = df.get("architecture_name", pd.Series([], dtype=str)).astype(str)
        return a.str.contains("Arps", na=False).any() or an.str.contains("Arps", na=False).any()

    if group_by != "auto":
        if not _has(group_by):
            raise ValueError(f"group_by='{group_by}' não existe no DataFrame.")
        group_col = group_by
        group_label = group_by.replace("_", " ").title()
    else:
        if _is_seq2():
            if _has("aggregation_method"): group_col, group_label = "aggregation_method", "Aggregation Method"
            elif _has("physics_strategy"): group_col, group_label = "physics_strategy", "Physics Strategy"
            else: group_col, group_label = _arch_col(), "Architecture"
        elif _is_arps():
            if _has("variant"): group_col, group_label = "variant", "Variant"
            else: group_col, group_label = _arch_col(), "Architecture"
        else:
            group_col, group_label = _arch_col(), "Architecture"
        if group_col is None: raise ValueError("Não encontrei coluna para agrupar.")
    
    best_idx = df.groupby(["well", group_col])[metric].idxmin().dropna().astype(int)
    champions_df = df.loc[best_idx].copy()

    if champions_df.empty:
        print("No champions after grouping; check filters/gates upstream.")
        return

    well_order = champions_df.groupby("well")[metric].min().sort_values(ascending=True).index.tolist()
    palette = px.colors.qualitative.Safe

    subtitle_font_size = 14 * font_scale
    title_html = (
        f"<b>Winning {group_label} by Well</b>"
        f"<br><span style='font-size:{subtitle_font_size}px'></span>"
    )

    fig = px.bar(
        champions_df,
        x="well",
        y=metric,
        color=group_col,
        barmode="group",
        category_orders={"well": well_order},
        labels={"well": "Well", metric: f"Best {metric.replace('_', ' ').title()}", group_col: group_label},
        template="plotly_white",
        color_discrete_sequence=palette,
        # text=champions_df[metric].round(3), # <--- REMOVIDO: Esta linha adicionava o texto
    )

    fig.update_traces(
        # textposition="auto", # <--- REMOVIDO: Não é mais necessário
        # textfont=dict(size=20 * font_scale), # <--- REMOVIDO: Não é mais necessário
        marker_line_width=1,
        marker_line_color="white",
        hovertemplate="<b>Well:</b> %{x}<br>"
                      f"<b>{group_label}:</b> %{{legendgroup}}<br>"
                      f"<b>{metric.replace('_',' ').title()}:</b> %{{y:.4f}}"
                      "<extra></extra>",
    )

    capped_height = min(520, 720)
    fig.update_layout(
        autosize=True,
        height=capped_height,
        bargap=0.25,
        margin=dict(t=95, l=10, r=10, b=40),
        title=dict(
            text=title_html,
            x=0.5,
            xanchor="center",
            font=dict(size=20 * font_scale)
        ),
        font=dict(family="Arial", size=14 * font_scale),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=None),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        hoverlabel=dict(bgcolor="white", font_size=13 * font_scale),
    )
    
    fig.update_xaxes(
        title_text="",
        categoryorder="array", categoryarray=well_order,
        automargin=True,
        tickangle=-30 if len(well_order) > 10 else 0,
        showgrid=False,
        tickfont_size=12 * font_scale
    )
    fig.update_yaxes(
        title_text=f"Best {metric.replace('_',' ').title()}",
        automargin=True,
        showgrid=True, gridwidth=0.5, gridcolor="rgba(200,200,200,0.3)",
        zeroline=False,
        title_font_size=16 * font_scale,
        tickfont_size=14 * font_scale
    )

    high_res_config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'performance_by_architecture',
            'scale': 3
        }
    }
    fig.show(config=high_res_config)

    caption = f"""
        **Interpretation:** Head-to-head comparison of groups on each well.
        *   For each well, the **lowest bar** is the current champion (best score).
        *   Here, the “group” is **{group_label}** (auto-detected), para dar granularidade útil.
            """
    console.print(Markdown(caption))

def plot_performance_by_architecture(
    master_df: pd.DataFrame, 
    metric: str = "weighted_score", 
    font_scale: float = 1.0
):
    """
    Interactive, overflow-safe box plot comparing the distribution of the best (per-campaign) scores
    across architectures. Visual-only enhancements; data logic unchanged.
    """
    console = Console()
    console.print(Markdown(f"--- \n### 📊 **Performance Distribution by Architecture** (lower is better)"))

    # --- data logic: unchanged ---
    best_per_campaign = master_df.loc[master_df.groupby('campaign')[metric].idxmin()].copy()

    arch_order = (
        best_per_campaign
        .groupby("architecture")[metric]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )

    palette = px.colors.qualitative.Safe

    # ADICIONADO: Definir o título e o subtítulo aqui para usar no update_layout
    subtitle_font_size = 14 * font_scale
    title_html = (
        "<b>Performance Distribution by Architecture</b>"
        f"<br><span style='font-size:{subtitle_font_size}px'><i>(Best trial from each campaign)</i></span>"
    )

    fig = px.box(
        best_per_campaign,
        x="architecture",
        y=metric,
        color="architecture",
        category_orders={"architecture": arch_order},
        # title=... MODIFICADO: Título movido para fig.update_layout
        points="all",
        notched=True,
        template="plotly_white",
        labels={"architecture": "Architecture", metric: f"Best {metric.replace('_', ' ').title()}"},
        color_discrete_sequence=palette,
    )

    fig.update_traces(
        marker=dict(size=7, opacity=0.7, line=dict(width=0.5, color="white")),
        selector=dict(type="box"),
        hovertemplate="<b>%{x}</b><br>Best " + metric.replace('_',' ') + ": %{y:.4f}<extra></extra>",
    )

    capped_height = 480
    fig.update_layout(
        autosize=True,
        height=capped_height,
        margin=dict(t=90, l=10, r=10, b=20),
        # ADICIONADO: Título centralizado e com escala de fonte
        title=dict(
            text=title_html,
            x=0.5,
            xanchor="center",
            font=dict(size=20 * font_scale) # Tamanho da fonte do título principal
        ),
        # MODIFICADO: Escala de fonte global
        font=dict(family="Arial", size=14 * font_scale),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=None),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        # ADICIONADO: Escala de fonte para o hover
        hoverlabel=dict(bgcolor="white", font_size=13 * font_scale),
    )
    
    fig.update_xaxes(
        categoryorder="array", categoryarray=arch_order,
        automargin=True, tickangle=0, showgrid=False,
        # ADICIONADO: Escala de fonte para título e ticks do eixo
        title_font_size=16 * font_scale,
        tickfont_size=14 * font_scale
    )
    fig.update_yaxes(
        title_text=f"Best {metric.replace('_',' ').title()} (lower is better)",
        automargin=True,
        showgrid=True, gridwidth=0.5, gridcolor="rgba(200,200,200,0.3)",
        zeroline=False,
        # ADICIONADO: Escala de fonte para título e ticks do eixo
        title_font_size=16 * font_scale,
        tickfont_size=14 * font_scale
    )

    high_res_config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'performance_by_architecture',
            'scale': 3
        }
    }
    fig.show(config=high_res_config)

    caption = f"""
                **Interpretation:** Each box summarizes the distribution of *best-per-campaign* scores for an architecture.
                *   A **lower box** indicates better overall performance (remember: lower is better).
                *   A **shorter box** shows more consistency across campaigns.
                *   **Points** are the best scores from individual campaigns.
            """
    console.print(Markdown(caption))


def _resolve_arch_column(df: pd.DataFrame) -> str | None:
    """Best-effort para achar a coluna de arquitetura."""
    for cand in ("architecture", "architecture_name", "arch"):
        if cand in df.columns:
            return cand
    return None


def plot_campaign_strategy_performance(
    campaign_df: pd.DataFrame,
    metric: str = "weighted_score",
    strategy_col: str = "strategy_display",
    campaign_name: str | None = None,  # kept for API compatibility (not shown)
    font_scale: float = 2.2
):
    """
    Bar plot of strategy performance within one campaign.

    - Title: "<b>Average {Metric Pretty} {ARCH}</b>" (centered).
    - Inside each bar: "<Strategy>\\n{mean:.3f}" (moves outside if bar is short).
    - X-axis tick labels are hidden to avoid duplication with inside text.
    """

    # --------------------- 1) Validate ---------------------
    if campaign_df is None or campaign_df.empty:
        print("⚠️ Input DataFrame is empty. Nothing to plot.")
        return
    if metric not in campaign_df.columns:
        print(f"⚠️ Metric column '{metric}' not found.")
        return
    if strategy_col not in campaign_df.columns:
        print(f"⚠️ Strategy column '{strategy_col}' not found.")
        return

    df = campaign_df.copy()

    # --------------------- 2) Infer Architecture for title ---------------------
    def _infer_arch(d: pd.DataFrame) -> str:
        # prefer 'family' if present
        if "family" in d.columns:
            fams = d["family"].astype(str).str.lower().unique()
            if len(fams) == 1:
                f = fams[0]
                if "seq2" in f or "pinn" in f: return "PINN"
                if "arps" in f:               return "ARPS"
                if "darts" in f:              return "DARTS"
        # fallback: look at architecture_name
        if "architecture_name" in d.columns:
            a = d["architecture_name"].astype(str).str.lower()
            if a.str.startswith("seq2").all():   return "PINN"
            if a.str.startswith("arps").all():   return "ARPS"
            if a.str.startswith("darts").all():  return "DARTS"
        return "Model"

    arch_title = _infer_arch(df)

    # --------------------- 3) Aggregate (mean ± SEM) ---------------------------
    agg_df = (
        df.groupby(strategy_col, dropna=False)[metric]
          .agg(mean="mean", std="std", count="count")
          .reset_index()
    )
    if agg_df.empty:
        print("⚠️ Aggregation resulted in an empty DataFrame.")
        return

    agg_df["sem"] = agg_df["std"] / agg_df["count"].clip(lower=1).pow(0.5)

    # Strategy prettifier (only for display)
    PRETTY = {
        "pressure_ensemble": "Ensemble",
        "combined_exp_arps": "Exp–Arps",
        "exponential": "Exponential",
        "static": "Static",
        "arps": "Arps",
        "arps_decline": "Arps",
        "Linearregression": "Linear Reg"
    }
    def _pretty(s: str) -> str:
        if s is None: return "—"
        low = str(s).strip().lower()
        return PRETTY.get(low, str(s).replace("_", " ").title())

    agg_df["strategy_fmt"] = agg_df[strategy_col].map(_pretty)
    agg_df = agg_df.sort_values("mean", ascending=True).reset_index(drop=True)

    # --------------------- 4) Figure -------------------------------------------
    metric_pretty = metric.replace("_", " ").title()
    title_html = f"<b>Average {metric_pretty} {arch_title}</b>"

    fig = px.bar(
        agg_df,
        x="strategy_fmt",
        y="mean",
        error_y="sem",
        template="plotly_white",
        color="strategy_fmt",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    # Decide inside/outside per bar based on height
    max_mean = float(agg_df["mean"].max())
    threshold = max_mean * 0.12  # 12% of max height
    positions = ["inside" if m >= threshold else "outside" for m in agg_df["mean"]]

    fig.update_traces(
        width=0.6,  # Define a largura das barras (ex: 60% do espaço disponível)
        textposition=positions,
        texttemplate="%{x}<br>%{y:.3f}",
        insidetextanchor="middle",
        insidetextfont_color="white",
        marker_line_width=1,
        marker_line_color="white",
        hovertemplate="<b>%{x}</b><br>"
                      + f"Mean {metric_pretty}: " + "%{y:.4f}<br>"
                      + "SEM: %{error_y.array:.4f}"
                      + "<extra></extra>",
    )

    fig.update_layout(
        title=dict(text=title_html, x=0.5, xanchor="center", y=0.95, yanchor="top"),
        showlegend=False,
        font=dict(family="Inter, Arial, sans-serif", size=10 * font_scale),
        margin=dict(t=80, l=10, r=10, b=40),
        height=380,
        width = 700,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        bargap=0.05,  # Define o espaço entre as barras (ex: 10% da largura da barra)
    )

    fig.update_xaxes(
        title_text=None,
        categoryorder="array",
        categoryarray=agg_df["strategy_fmt"].tolist(),
        showgrid=False,
        showticklabels=False,  # <— hide axis labels; they’re inside bars
    )
    fig.update_yaxes(
        title_text=f"Average",
        showgrid=True, gridwidth=0.5, gridcolor="rgba(200,200,200,0.3)",
        zeroline=False,
    )

    fig.show(config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": "avg_strategy_performance", "scale": 3},
    })
    return fig

def plot_hyperparameter_importance_per_well(
    master_df: pd.DataFrame,
    metric_col: str = "weighted_score",
    family: str | None = None,
    top_k: int = 5,
    font_scale: float = 1.2,
    inside_frac: float = 0.18,
    show_report: bool = True, # Parâmetro para controlar o relatório
    show: bool = True,
):
    """
    Compact, publication-ready hyperparameter-importance grid.
    - Generates a rich-text numerical report of importances.
    - Robust against cases with insufficient data for importance calculation.
    - Compact aliases on y-axis.
    - Values inside bars when long enough; otherwise outside.
    """
    
    # --- 0) Sanity and Setup ---
    console = Console()

    if master_df is None or master_df.empty or metric_col not in master_df.columns:
        print("⚠️ No data/metric for hyperparameter importance.")
        return
    df = master_df.copy()

    # --- 1) Family detection ---
    def _detect_family(d: pd.DataFrame) -> str:
        txt = (d.get("architecture", pd.Series([], dtype=str)).fillna("").astype(str) + " "
               + d.get("architecture_name", pd.Series([], dtype=str)).fillna("").astype(str)).str.lower()
        joined = " ".join(txt.tolist())
        if ("seq2" in joined) or ("physics_strategy" in d.columns and "profile" not in d.columns and "variant" not in d.columns): return "seq2"
        if ("arps" in joined) or ("variant" in d.columns): return "arps"
        if ("darts" in joined) or ("profile" in d.columns): return "darts"
        return "seq2"

    fam = (family or _detect_family(df)).lower()
    family_display_name = fam.upper()
    if fam == "seq2":
        family_display_name = "PINN"

    if "epochs" not in df.columns and "n_epochs" in df.columns:
        df["epochs"] = df["n_epochs"]

    # --- 2) HP specs + COMPACT alias map ---
    FAMILY_HP = {
        "seq2": dict(numeric=["epochs", "batch_size", "learning_rate", "data_sample", "lag_window"], categorical=["physics_strategy", "aggregation_method"]),
        "arps": dict(numeric=["loss_delta", "quantile_tau", "burn_in_fraction", "piecewise_min_delta_bic", "b_min", "b_max"], categorical=["variant", "solver", "weighting", "loss", "piecewise"]),
        "darts": dict(numeric=["epochs", "batch_size", "learning_rate", "input_chunk_length", "output_chunk_length", "lag_window"], categorical=["profile"]),
    }
    COMPACT = {
        "learning_rate": "Learning Rate", "lag_window": "Lag", "batch_size": "Batch", "epochs": "Epochs", "data_sample": "DA", "aggregation_method": "Filter",
        "physics_strategy": "Physics", "input_chunk_length": "In", "output_chunk_length": "Out", "profile": "Profile", "variant": "Variant", "solver": "Solver",
        "weighting": "Weighting", "loss": "Loss", "piecewise": "Pwise", "loss_delta": "ΔLoss", "quantile_tau": "Quantile τ", "burn_in_fraction": "Burn",
        "piecewise_min_delta_bic": "ΔBIC", "b_min": "bₘᵢₙ", "b_max": "bₘₐₓ",
    }
    hp_spec = FAMILY_HP.get(fam, FAMILY_HP["seq2"])
    num_feats = [c for c in hp_spec["numeric"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=False) > 1]
    cat_feats = [c for c in hp_spec["categorical"] if c in df.columns and df[c].nunique(dropna=False) > 1]

    if not num_feats and not cat_feats: print(f"⚠️ No candidate hyperparameters present for family '{fam}'."); return
    if "well" not in df.columns: print("⚠️ No 'well' column found."); return
    wells = sorted(df["well"].dropna().unique())
    if len(wells) == 0: print("⚠️ No wells to plot."); return

    # --- FUNÇÃO INTERNA PARA GERAR O RELATÓRIO ---
    def _log_hp_importance_report(report_data: dict):
        table = Table(title=f"Hyperparameter Importance Report (Family: {family_display_name.upper()})", box=SQUARE, show_header=True, header_style="bold cyan")
        table.add_column("Well", style="dim", width=15)
        table.add_column("Hyperparameter (Alias)", style="bold", min_width=20)
        table.add_column("Importance (F-stat)", justify="right", style="magenta")
        has_data = False
        for well_name, imp_df in report_data.items():
            if not imp_df.empty:
                has_data = True
                table.add_section()
                for i, row in imp_df.iterrows():
                    well_display = well_name if i == 0 else ""
                    table.add_row(well_display, row["alias"], f"{row['importance']:.3f}")
        if has_data: console.print(table)
        else: console.print(f"⚠️ [yellow]No valid importance data found to generate report for family '{family_display_name}'.[/yellow]")

    # --- 3) Color map (Pré-passe) COM A CORREÇÃO DO BUG ---
    palette = px.colors.qualitative.Safe
    all_aliases = set()

    for well in wells:
        dw = df[df["well"] == well]
        if dw.empty: continue
        thresh = dw[metric_col].quantile(0.3)
        y = (dw[metric_col] <= thresh).astype(int)
        
        # --- INÍCIO DA CORREÇÃO DO BUG ---
        # Este bloco if/else garante que `imp` seja sempre criado.
        if y.nunique() < 2 or len(dw) < 4:
            imp = pd.DataFrame({"feature": []})
        else:
            X_num = dw[num_feats].astype(float) if num_feats else pd.DataFrame(index=dw.index)
            if cat_feats:
                X_cat = pd.get_dummies(dw[cat_feats].astype("category"), prefix_sep="=", drop_first=False, dtype=float)
                owner = {c: c.split("=")[0] for c in X_cat.columns}
            else:
                X_cat = pd.DataFrame(index=dw.index); owner = {}
            X = pd.concat([X_num, X_cat], axis=1)
            for c in X_num.columns: owner[c] = c
            nun = X.nunique(dropna=False); keep = nun[nun > 1].index.tolist()
            X = X[keep].fillna(X.mean(numeric_only=True))

            if X.shape[1] == 0:
                imp = pd.DataFrame({"feature": []})
            else:
                f_vals, _ = f_classif(X, y)
                enc = pd.DataFrame({"encoded": X.columns, "importance": f_vals})
                enc["feature"] = enc["encoded"].map(owner)
                imp = (enc.groupby("feature", as_index=False)["importance"].max().sort_values("importance", ascending=False).head(top_k))
        # --- FIM DA CORREÇÃO DO BUG ---
        
        aliases = [COMPACT.get(f, f) for f in imp["feature"].tolist()]
        all_aliases.update(aliases)

    all_aliases = sorted(list(all_aliases))
    rep = (len(all_aliases) // len(palette)) + 1
    colors = (palette * rep)[:len(all_aliases)]
    color_map = {lab: col for lab, col in zip(all_aliases, colors)}

    # --- 4) Figure scaffold ---
    n_cols = min(2, len(wells)); n_rows = int(np.ceil(len(wells) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"<b>Well: {w}</b>" for w in wells], horizontal_spacing=0.12, vertical_spacing=0.14)

    # --- 5) Per-well traces & COLETA DE DADOS PARA O RELATÓRIO ---
    report_data = {} # Dicionário para guardar os dados do relatório

    for i, well in enumerate(wells):
        dw = df[df["well"] == well].copy()
        if dw.empty: 
            report_data[well] = pd.DataFrame()
            continue
        thresh = dw[metric_col].quantile(0.3); y = (dw[metric_col] <= thresh).astype(int)
        
        if y.nunique() < 2 or len(dw) < 4:
            imp = pd.DataFrame({"feature": [], "importance": [], "alias": []})
        else:
            X_num = dw[num_feats].astype(float) if num_feats else pd.DataFrame(index=dw.index)
            if cat_feats:
                X_cat = pd.get_dummies(dw[cat_feats].astype("category"), prefix_sep="=", drop_first=False, dtype=float)
                owner = {c: c.split("=")[0] for c in X_cat.columns}
            else:
                X_cat = pd.DataFrame(index=dw.index); owner = {}
            X = pd.concat([X_num, X_cat], axis=1)
            for c in X_num.columns: owner[c] = c
            nun = X.nunique(dropna=False); keep = nun[nun > 1].index.tolist()
            X = X[keep].fillna(X.mean(numeric_only=True))
            if X.shape[1] == 0:
                imp = pd.DataFrame({"feature": [], "importance": [], "alias": []})
            else:
                f_vals, _ = f_classif(X, y)
                enc = pd.DataFrame({"encoded": X.columns, "importance": f_vals})
                enc["feature"] = enc["encoded"].map(owner)
                imp = (enc.groupby("feature", as_index=False)["importance"].max().sort_values("importance", ascending=False).head(top_k))
                imp["alias"] = imp["feature"].map(COMPACT).fillna(imp["feature"])
        
        report_data[well] = imp # Salva os dados de importância para o relatório

        row = i // n_cols + 1
        col = i % n_cols + 1
        vals = imp.get("importance", pd.Series([], dtype=float)).tolist()
        x_max = max(vals) if len(vals) else 1.0
        positions = ["inside" if (v >= inside_frac * x_max) else "outside" for v in vals]
        bar_colors = [color_map.get(alias, palette[idx % len(palette)]) for idx, alias in enumerate(imp.get("alias", []))]
        fig.add_trace(
            go.Bar(
                x=vals, y=imp.get("alias", []), orientation="h", marker=dict(color=bar_colors, line=dict(width=1.0, color="white")),
                text=[f"{v:.2f}" for v in vals], textposition=positions, insidetextfont=dict(color="white", size=int(12 * font_scale)),
                outsidetextfont=dict(color="black", size=int(12 * font_scale)), cliponaxis=False, hovertemplate="<b>%{y}</b><br>F-stat: %{x:.3f}<extra></extra>",
            ), row=row, col=col
        )
        fig.update_yaxes(row=row, col=col, categoryorder="total ascending", automargin=True, showgrid=False, ticks="")
        fig.update_xaxes(row=row, col=col, title_text="Importance (F-statistic) — higher is better", automargin=True, showgrid=False, zeroline=False)

    # --- 6) GERAÇÃO DO RELATÓRIO E EXIBIÇÃO DO GRÁFICO ---
    if show_report:
        _log_hp_importance_report(report_data)
    
    subtitle = f"Family: {family_display_name.upper()} • Top-{top_k} per well • target = Top-30% (lower {metric_col} is better)"
    fig.update_layout(
        template="plotly_white", showlegend=False, font=dict(family="Arial", size=int(14 * font_scale)),
        title=dict(text=(f"<b>Hyperparameter Importance by Well</b><br><span style='font-size:{int(14*font_scale)}px'><i>{subtitle}</i></span>"), x=0.5, y=0.98, xanchor="center", yanchor="top"),
        height=300 * n_rows + int(120 * font_scale), width=1100, margin=dict(t=110, l=70, r=110, b=50), uniformtext_mode="hide", bargap=0.22,
    )
    
    if show:
        high_res_config = {'toImageButtonOptions': {'format': 'png', 'filename': 'hp_importance_grid', 'scale': 3}}
        fig.show(config=high_res_config)