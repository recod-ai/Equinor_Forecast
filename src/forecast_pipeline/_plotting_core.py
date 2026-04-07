# forecast_pipeline/_plotting_core.py
from __future__ import annotations
import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .analytics import make_envelope, scenario_curve, _reconstruct_train_series_phys
from .ensemble_output import EnsembleOutput
from utils.utilities import _inverse_transform_1d, _looks_scaled

import plotly.graph_objects as go
import numpy as np
from typing import Dict, Optional, Tuple

# ------------------------------------------------------------------
# Low‑level helpers --------------------------------------------------
# ------------------------------------------------------------------

def _add_trace(fig: go.Figure, x: np.ndarray, y: np.ndarray, name: str,
               *, style: str = "solid", color: Optional[str] = None):
    line_style: dict = dict(width=4)
    if color:
        line_style["color"] = color
    if style == "dash":
        line_style["dash"] = "dash"
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=line_style))


# ------------------------------------------------------------------
# Public API ---------------------------------------------------------
# ------------------------------------------------------------------

# In forecast_pipeline/plotting.py

def plot_series(
    truth: np.ndarray,
    mean_curve: np.ndarray,
    *,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    envelope_color: str = "rgba(255,99,132,0.2)",
    samples: Optional[np.ndarray] = None,
    q_phys: Optional[np.ndarray] = None,
    res: Optional[np.ndarray] = None,
    # visual ---------------------------------------------------------
    title: str = "Forecast",
    well: str = "",
    width: int = 1200,
    height: int = 600,
    # --- metrics / extras ------------------------------------------
    r2: float | None = None, 
    smape: float | None = None,
    mae: float | None = None,
    window_size: int | None = None,
    forecast_steps: int | None = None,
    percentage_split: float | None = None,
    # --- NEW (PLUG-AND-PLAY) ---
    metric_labels: Dict[str, str] | None = None,
):
    """Plot actual vs prediction with optional envelope and metrics."""
    import numpy as np
    import plotly.graph_objects as go

    def _rv(a): return np.asarray(a).ravel()

    truth, mean_curve = _rv(truth), _rv(mean_curve)
    if lower is not None:  lower = _rv(lower)
    if upper is not None:  upper = _rv(upper)
    if q_phys is not None: q_phys = _rv(q_phys)
    if res is not None:    res    = _rv(res)

    x = np.arange(mean_curve.size)
    fig = go.Figure()

    # Main series
    _add_trace(fig, x, truth, name="Actual", color="#206A92")

    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=envelope_color,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Envelope",
        ))

    _add_trace(fig, x, mean_curve, name="Prediction", color="yellowgreen", style="dash")

    if q_phys is not None:
        _add_trace(fig, x, q_phys, name="Q_phys", color="#FF5733")
    if res is not None:
        _add_trace(fig, x, res, name="Residual", color="#8E44AD")

    # Fan chart (5–95%)
    if samples is not None:
        q5, q95 = np.percentile(samples, [5, 95], axis=0)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([_rv(q95), _rv(q5)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,0,0,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Fan 5–95%",
        ))

    # Layout and legend
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=36)),
        xaxis_title="Days",
        yaxis_title="Rate",
        plot_bgcolor="white",
        width=width,
        height=height,
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.25,
            xanchor="center",
            font=dict(size=20)
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    # --- START OF MODIFIED SECTION ---
    metrics_lines = []
    
    # Use metric_labels to get the correct display name for SMAPE, with a safe fallback
    metric_labels = metric_labels or {}  # Ensure it's a dict, not None
    smape_label = metric_labels.get("SMAPE", "SMAPE")  # Fallback to "SMAPE" if not provided

    if smape is not None:
        # Use the resolved label here
        metrics_lines.append(f"<span style='color:#206A92'>{smape_label}: {smape:.2f}%</span>")
    
    # The rest of the metric annotation logic is unchanged
    if mae is not None:
        metrics_lines.append(f"<span style='color:yellowgreen'>MAE: {mae:.2f}</span>")
    if forecast_steps is not None:
        metrics_lines.append(f"<span style='color:#2E2E2E'>Horizon: {forecast_steps}</span>")
    if percentage_split is not None:
        metrics_lines.append(f"<span style='color:#2E2E2E'>Train: {percentage_split*100:.0f}%</span>")
    if window_size is not None:
        metrics_lines.append(f"<span style='color:#2E2E2E'>Windows: {window_size}</span>")
    
    metrics_text = "<br>".join(metrics_lines)

    fig.add_annotation(
        x=0.8,
        y=0.5,
        xref="paper",
        yref="paper",
        text=metrics_text,
        showarrow=False,
        font=dict(size=22),
        bgcolor="rgba(255,255,255,0.85)",
        xanchor="left",
        align="left"
    )
    # --- END OF MODIFIED SECTION ---

    fig.update_xaxes(title_font=dict(size=26), tickfont=dict(size=22))
    fig.update_yaxes(title_font=dict(size=26), tickfont=dict(size=22))

    fig.show(renderer="png")



def plot_predictions_wrapper(
    ensemble,
    *,                               # só argumentos nomeados para evitar confusão
    truth: np.ndarray,
    kind: str = "P50",
    well: str = "",
    band: Tuple[float, float] | None = None,
    show_components: bool = False,
    mean_override: Optional[np.ndarray] = None,
    title=None,
    scaler=None,
    manual_envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    is_cum=None,
    **extra_plot_kwargs,             # smape, mae, window_size, forecast_steps, …
):
    """
    Decide qual cenário desenhar e invoca `plot_series`.

    Compatibilidade (PATCH):
      - Aceita `ensemble` como objeto (EnsembleOutput) OU dict (como no ARPS spaghetti).
      - Nunca assume `.q_phys` / `.res_test` / `.sigma_test` existirem.
      - Se `mean_override` for fornecido, ele é sempre a curva central.
    """
    kind = str(kind or "P50").upper()

    def _get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # -----------------------------
    # Curva central (prefer override)
    # -----------------------------
    if mean_override is not None:
        central = np.asarray(mean_override, dtype=float).ravel()
    else:
        # fallback: tente puxar do ensemble (objeto ou dict)
        pred_test = _get(ensemble, "pred_test", None)
        if pred_test is None and isinstance(ensemble, dict):
            # alguns payloads guardam curvas em "test"/"val"
            pred_test = (ensemble.get("test", {}) or {}).get("agg_scaled", None)
        if pred_test is None:
            # último fallback: usa NaNs para não quebrar plot
            central = np.full_like(np.asarray(truth, dtype=float).ravel(), np.nan)
        else:
            central = np.asarray(pred_test, dtype=float).ravel()

    lower = upper = None

    sigma_test = _get(ensemble, "sigma_test", None)

    if kind in ("P50", "MEAN"):
        mean_curve = central
        if band is not None:
            if sigma_test is not None:
                lower, upper = make_envelope(
                    np.asarray(_get(ensemble, "pred_test", central), dtype=float),
                    np.asarray(sigma_test, dtype=float),
                    *(band or (0.10, 0.90))
                )
            else:
                logging.info("Band solicitada, mas sigma indisponível – ribbon omitido")

    elif kind == "P90":
        if sigma_test is not None:
            mean_curve = scenario_curve(
                np.asarray(_get(ensemble, "pred_test", central), dtype=float),
                np.asarray(sigma_test, dtype=float),
                0.90
            )
        else:
            mean_curve = central

    elif kind == "P10":
        if sigma_test is not None:
            mean_curve = scenario_curve(
                np.asarray(_get(ensemble, "pred_test", central), dtype=float),
                np.asarray(sigma_test, dtype=float),
                0.10
            )
        else:
            mean_curve = central

    elif kind == "BAND":
        if band is None:
            band = (0.10, 0.90)
        if sigma_test is not None:
            lower, upper = make_envelope(
                np.asarray(_get(ensemble, "pred_test", central), dtype=float),
                np.asarray(sigma_test, dtype=float),
                *band
            )
        mean_curve = central

    else:
        raise ValueError(f"Unknown kind '{kind}'")

    # Envelope manual (cumulativo, etc.)
    if manual_envelope is not None:
        lower, upper = manual_envelope

    # -------------------------------------------------
    # desscale envelopes / componentes (central já vem pronto via mean_override)
    # -------------------------------------------------
    q_phys = _get(ensemble, "q_phys", None)
    res    = _get(ensemble, "res_test", _get(ensemble, "res", None))

    if scaler is not None and is_cum is False:
        def _inv(arr):
            return scaler.inverse_transform(np.asarray(arr, dtype=float).reshape(-1, 1)).ravel()

        if lower is not None and upper is not None:
            lower, upper = _inv(lower), _inv(upper)

        if show_components and q_phys is not None:
            q_phys = _inv(q_phys)
        if show_components and res is not None:
            res = _inv(res)

    plot_series(
        truth=truth,
        mean_curve=mean_curve,
        lower=lower,
        upper=upper,
        q_phys=q_phys if show_components else None,
        res=res if show_components else None,   # <-- FIX: era res_test (bug)
        title=title or f"Scenario {kind}",
        well=well,
        **extra_plot_kwargs,
    )




COLOR_PRIMARY = '#0077B6'  # Strong Blue (Star Command Blue)
COLOR_SECONDARY = '#F94144' # Strong Red (Imperial Red)
COLOR_ACCENT_FILL = 'rgba(249, 65, 68, 0.1)' # Very subtle red fill
COLOR_TEXT = '#2c3e50'      # Midnight Blue (Slightly softer dark)
COLOR_GRID = 'rgba(189, 195, 199, 0.4)' # Silver Sand / Light Gray
FONT_FAMILY = "Lato, Arial, sans-serif" # Modern, clean font (Lato preferred if available)

COLOR_PRIMARY     = "#32CD32"            # LimeGreen — destaca produção
COLOR_SECONDARY   = "#4A90E2"            # Soft Blue — para comparação ou gás
COLOR_ACCENT_FILL = "rgba(74, 144, 226, 0.1)"  # fill azul suave
COLOR_TEXT        = "#2C3E50"            # Dark Slate — boa legibilidade
COLOR_GRID        = "rgba(236, 240, 241, 0.5)" # Light Gray — grade sutil
FONT_FAMILY       = "Lato, Arial, sans-serif"

# --- em algum módulo util.py ou logo acima das funções ---
UNITS = {
    "BORE_OIL_VOL":          "stb d⁻¹",
    "BORE_GAS_VOL":          "scf d⁻¹",
    "BORE_WAT_VOL":          "stb d⁻¹",
    "AVG_DOWNHOLE_PRESSURE": "psi",
    "AVG_WHP_P":            "psi",
    "delta_P":              "psi",
    "PI":                   "stb d⁻¹ psi⁻¹",
    "CE":                   "stb d⁻¹ /% choke",
    "Taxa_Declinio":        "-",
}



def plot_by_well_advanced(
    df: Dict[str, pd.DataFrame],
    *,
    units: Mapping[str, str] = UNITS,
    columns: Optional[Iterable[str]] = None,
    well: str | None = None  
) -> None:
    """
    Plota séries temporais para um poço específico com visual profissional e aprimorado.
    """

    cols = columns or [c for c in df.columns if c != "Day"]

    for col in cols:
        if col not in df.columns:
            print(f"[warn] coluna '{col}' inexistente")
            continue

        mask = df[col].notna()
        y = df.loc[mask, col].values
        x = df.loc[mask, "Day"] if "Day" in df.columns else np.arange(len(y))

        unidade = f" ({units.get(col, '')})" if col in units else ""
        titulo  = f"{col}{unidade}"
        if well:
            titulo = f"{well} – {titulo}"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                symbol="circle",
                size=10,
                color=COLOR_PRIMARY,
                opacity=0.5,
                line=dict(color="white", width=1)
            ),
            name=col,
            line=dict(color=COLOR_PRIMARY, width=2),
            fill="tozeroy",
            fillcolor=COLOR_ACCENT_FILL,
            hovertemplate=(
                f"<b>{col}</b><br>"
                "Data: %{x}<br>"
                "Valor: %{y:.2f}<extra></extra>"
            )
        ))

        fig.update_layout(
            title=dict(
                text=f"<b>{col}</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=30, family=FONT_FAMILY, color=COLOR_TEXT)
            ),
            legend=dict(
                orientation="h",
                x=0.5,
                y=-0.2,
                xanchor="center",
                font=dict(size=14, family=FONT_FAMILY)
            ),
            xaxis_title="Day",
            yaxis_title=titulo,
            font=dict(family=FONT_FAMILY, size=14, color=COLOR_TEXT),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=80, r=80, t=100, b=100),
            hovermode="x unified",
            width=1400,
            height=800,
        )

        fig.update_xaxes(
            gridcolor=COLOR_GRID,
            zeroline=False,
            showline=True,
            linecolor=COLOR_GRID,
            ticks="outside",
            tickfont=dict(size=24),
            title_font=dict(size=26)
        )
        fig.update_yaxes(
            gridcolor=COLOR_GRID,
            zeroline=False,
            showline=True,
            linecolor=COLOR_GRID,
            ticks="outside",
            tickfont=dict(size=24),
            title_font=dict(size=26)
        )

        fig.show(renderer="png")


    
# --- Visual Style Definition ---
COLOR_ACTUAL = '#0077B6'          # Strong Blue for ground truth data
COLOR_VAL = '#F94144'              # Strong Red for validation predictions
COLOR_TEST = '#32CD32'             # LimeGreen for the primary test forecast (horizon)
COLOR_TEST_REMAINDER = '#84a98c'   # Greenish-Gray for the subsequent test forecast
COLOR_TEXT = '#2c3e50'             # Midnight Blue for high-readability text
COLOR_GRID = 'rgba(189, 195, 199, 0.4)' # Light Gray for a subtle grid
FONT_FAMILY = "Lato, Arial, sans-serif"   # Modern, professional font

# Subtle fill colors for background regions
COLOR_TRAIN_FILL = "rgba(0, 119, 182, 0.05)"
COLOR_VAL_FILL = "rgba(249, 65, 68, 0.05)"



def _format_metrics_for_annotation(metrics: Dict[str, float]) -> str:
    """Formats metrics into an aligned, multi-line string using a monospace font."""
    if not metrics:
        return ""
    
    text_lines = [f"<b>{k.upper()}:</b> {v:.2f}" for k, v in metrics.items()]
    return "<br>".join(text_lines)


def _get_color_palette(name: str) -> Dict[str, str]:
    """
    Retrieves a color palette dictionary by name for plot_integrated_view.
    This centralized function makes it easy to add, remove, or modify themes.

    New disruptive, story-driven palettes:
      - 'wildfire_regrowth' (Crisis → Intervention → Recovery)
      - 'abyssal_expedition' (Descent → Scan → Beacon)
    """
    palettes = {
        "default": {
            "train_from_x": "#3949AB",          # Darker blue for context
            "train_from_y": "#0277BD",          # Lighter blue for training tail
            "actual_post_train": "#0277BD",     # Consistent blue for ground truth
            "validation": "#E53935",            # Strong red for validation error
            # "test_initial": "#212121",          # Black for critical initial forecast
            "test_initial": "#4CAF50",          # Black for critical initial forecast
            "test_rolling": "#4CAF50",          # Green for subsequent forecast
            "fill_train": "rgba(232, 234, 246, 0.5)",
            "fill_val": "rgba(255, 235, 238, 0.5)",
            "fill_test": "rgba(232, 245, 233, 0.5)",
            "text": "#333",
            "grid": "#EAEAEA",
            "test_border": "#4CAF50",
        },

        # --- Cinzas & Azuis Metálicos (fundo claro) -------------------------------
        "metallic_azure": {
            "train_from_x": "#546E7A",          # aço azulado (Blue Grey 700)
            "train_from_y": "#90A4AE",          # alumínio azulado (Blue Grey 300)
            "actual_post_train": "#1E88E5",     # cobalto
            "validation": "#00B8D4",            # ciano metálico
            "test_initial": "#263238",          # gunmetal
            "test_rolling": "#4FC3F7",          # azul anodizado claro
            "fill_train": "rgba(84, 110, 122, 0.12)",
            "fill_val":   "rgba(0, 184, 212, 0.10)",
            "fill_test":  "rgba(79, 195, 247, 0.12)",
            "text": "#222",
            "grid": "#E5E7EB",
            "test_border": "#4FC3F7",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
        },
        
        # Light, disruptive: Abyssal Expedition (white canvas)
        "abyssal_expedition_light": {
            "train_from_x": "#0D47A1",
            "train_from_y": "#1976D2",
            "actual_post_train": "#00796B",   # deep teal for truth on white
            "validation": "#00B8D4",
            "test_initial": "#AA00FF",
            "test_rolling": "#F9A825",
            "fill_train": "rgba(25,118,210,0.10)",
            "fill_val": "rgba(0,184,212,0.10)",
            "fill_test": "rgba(249,168,37,0.12)",
            "text": "#222",
            "grid": "#EAEAEA",
            "test_border": "#F9A825",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
        },
    }

    if name not in palettes:
        raise ValueError(f"Unknown palette '{name}'. Available options are: {list(palettes.keys())}")

    # Set default background colors for palettes that don't define them
    palette = palettes[name]
    palette.setdefault("plot_bgcolor", "white")
    palette.setdefault("paper_bgcolor", "white")

    return palette


def plot_integrated_view(
    x_axis: np.ndarray,
    y_actual: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_test: np.ndarray,
    split_indices: Dict[str, int],
    metrics_val: Optional[Dict[str, float]],
    metrics_test: Optional[Dict[str, float]],
    title: str,
    yaxis_title: str = "Production Rate",
    well: str = "",
    horizon: Optional[int] = None,
    palette: str = "default",
    font_scale: float = 1.32,
    show: bool = True,
    width: int = 900,
    height: int = 600,
    stitch_val_test: bool = True,
    # NEW: optional spaghetti ribbons (K, T) where T == len(x_axis)
    y_pred_val_members: Optional[np.ndarray] = None,
    y_pred_test_members: Optional[np.ndarray] = None,
    max_members: int = 150,
    members_opacity: float = 0.15,
):
    """
    Dynamically responsive and overlap-free integrated plot for Train/Val/Test performance.

    Supports multiple color themes via the `palette` argument.

    NEW:
      - If y_pred_val_members / y_pred_test_members are provided, overlays spaghetti
        members behind the main curves (purely visual; does not affect metrics).

    Parameters
    ----------
    stitch_val_test : bool, default False
        If True, draws a small connecting line between the last Validation prediction
        point and the first Test prediction point (purely visual stitching, no data
        modification).

    y_pred_val_members / y_pred_test_members : Optional[np.ndarray]
        Arrays shaped (K, len(x_axis)) containing full-length member curves to overlay.
        These should already be aligned to x_axis.
    """

    
    # --- Color and Style Selection ---
    colors = _get_color_palette(palette)
    FONT_FAMILY = "Inter, Arial, sans-serif"

    fig = go.Figure()

    # --- Data Ranges ---
    train_end = int(split_indices["train_end"])
    val_end = int(split_indices["val_end"])
    y_min = np.nanmin(y_actual)
    y_max = np.nanmax(y_actual)
    y_padding = (y_max - y_min) * 0.1
    y_range = [y_min - y_padding * 0.2, y_max + y_padding]

    # --- Backgrounds and Region Titles ---
    fig.add_vrect(
        x0=x_axis[0],
        x1=x_axis[train_end - 1],
        fillcolor=colors["fill_train"],
        layer="below",
        line_width=0,
    )
    fig.add_vrect(
        x0=x_axis[train_end - 1],
        x1=x_axis[val_end - 1],
        fillcolor=colors["fill_val"],
        layer="below",
        line_width=0,
    )
    fig.add_vrect(
        x0=x_axis[val_end - 1],
        x1=x_axis[-1],
        fillcolor=colors["fill_test"],
        layer="below",
        line_width=0,
    )

    title_font_size = 18 * font_scale
    fig.add_annotation(
        text="<b>Train</b>",
        x=x_axis[train_end // 2],
        yref="paper",
        y=0.98,
        showarrow=False,
        font=dict(size=title_font_size, color=colors["text"]),
    )
    fig.add_annotation(
        text="<b>Validation</b>",
        x=x_axis[train_end + (val_end - train_end) // 2],
        yref="paper",
        y=0.98,
        showarrow=False,
        font=dict(size=title_font_size, color=colors["validation"]),
    )
    fig.add_annotation(
        text="<b>Test</b>",
        x=x_axis[val_end + (len(x_axis) - val_end) // 2],
        yref="paper",
        y=0.98,
        showarrow=False,
        font=dict(size=title_font_size, color=colors["test_border"]),
    )

    # --- Plot Traces ---
    has_valid_h = isinstance(horizon, (int, np.integer)) and int(horizon) > 0
    h = int(horizon) if has_valid_h else 0
    x_split_len = train_end - h
    split_train = has_valid_h and x_split_len > 0

    # Train (from X / from y)
    if split_train:
        fig.add_trace(
            go.Scatter(
                x=x_axis[:x_split_len],
                y=y_actual[:x_split_len],
                mode="lines",
                name="Train (from X)",
                line=dict(color=colors["train_from_x"], width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis[x_split_len - 1 : train_end],
                y=y_actual[x_split_len - 1 : train_end],
                mode="lines",
                name="Train (from y)",
                line=dict(color=colors["train_from_y"], width=2),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_axis[:train_end],
                y=y_actual[:train_end],
                mode="lines",
                name="Train",
                line=dict(color=colors["train_from_y"], width=2),
            )
        )

    # Actual Val + Test
    fig.add_trace(
        go.Scatter(
            x=x_axis[train_end - 1 :],
            y=y_actual[train_end - 1 :],
            mode="lines",
            name="Actual (Val+Test)",
            line=dict(color=colors["actual_post_train"], width=2.5),
        )
    )

    # ----------------------------
    # NEW: Spaghetti overlays first (so main curves draw on top)
    # ----------------------------
    def _overlay_members(members: Optional[np.ndarray], *, label: str, color: str) -> None:
        if members is None:
            logging.info("[integrated_debug] overlay skip %s: members=None", label)
            return
        try:
            mem = np.asarray(members, dtype=float)
            logging.info("[integrated_debug] overlay %s: mem.shape=%s x_axis=%d", label, str(mem.shape), len(x_axis))
    
            if mem.ndim == 1:
                mem = mem.reshape(1, -1)
            if mem.ndim != 2:
                logging.info("[integrated_debug] overlay skip %s: mem.ndim=%d", label, int(mem.ndim))
                return
            if mem.shape[1] != len(x_axis):
                logging.info("[integrated_debug] overlay skip %s: shape mismatch mem_T=%d x_axis=%d", label, int(mem.shape[1]), len(x_axis))
                return
    
            k_draw = min(int(max_members), int(mem.shape[0]))
            logging.info("[integrated_debug] overlay draw %s: k_draw=%d opacity=%.3f", label, int(k_draw), float(members_opacity))
    
            for i in range(k_draw):
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=mem[i],
                        mode="lines",
                        name=f"{label} member",
                        line=dict(color=color, width=1),
                        opacity=float(members_opacity),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        except Exception:
            logging.exception("[integrated_debug] overlay crash %s", label)
            return


    _overlay_members(y_pred_val_members, label="VAL", color=colors["validation"])
    _overlay_members(y_pred_test_members, label="TEST", color=colors["test_rolling"])

    # Validation prediction
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_pred_val,
            mode="lines",
            name="Validation Prediction",
            line=dict(color=colors["validation"], width=5, dash="dot"),
        )
    )

    # Test prediction (possibly split: initial vs rolling)
    te_x0 = val_end
    if has_valid_h and te_x0 < len(x_axis):
        split_point = min(len(x_axis), te_x0 + h)

        # Initial test segment
        fig.add_trace(
            go.Scatter(
                x=x_axis[te_x0:split_point],
                y=y_pred_test[te_x0:split_point],
                mode="lines",
                name="Test Prediction (Initial)",
                line=dict(color=colors["test_initial"], width=5, dash="dot"),
            )
        )

        # Rolling test segment
        if split_point < len(x_axis):
            fig.add_trace(
                go.Scatter(
                    x=x_axis[split_point - 1 :],
                    y=y_pred_test[split_point - 1 :],
                    mode="lines",
                    name="Test Prediction (Rolling)",
                    line=dict(color=colors["test_rolling"], width=5, dash="dot"),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_pred_test,
                mode="lines",
                name="Test Prediction",
                line=dict(color=colors["test_rolling"], width=2.5),
            )
        )

    # --- Optional visual stitching between Val and Test predictions ---
    if stitch_val_test:
        try:
            idx_val_last = val_end - 1
            idx_test_first = val_end

            if (
                0 <= idx_val_last < len(x_axis)
                and 0 <= idx_test_first < len(x_axis)
            ):
                y_val_last = y_pred_val[idx_val_last]
                y_test_first = y_pred_test[idx_test_first]

                if np.isfinite(y_val_last) and np.isfinite(y_test_first):
                    fig.add_trace(
                        go.Scatter(
                            x=[x_axis[idx_val_last], x_axis[idx_test_first]],
                            y=[y_val_last, y_test_first],
                            mode="lines",
                            name="Val-Test Stitch",
                            line=dict(color=colors["validation"], width=5, dash="dot"),
                            showlegend=False,
                        )
                    )
        except Exception:
            pass

    # --- Positioned Annotations for Metrics ---
    metrics_font_size = 16 * font_scale
    is_cumulative_plot = "cumulative" in yaxis_title.lower()
    val_y_anchor, test_y_anchor = (
        (0.65, 0.35) if is_cumulative_plot else (0.98, 0.7)
    )

    if metrics_val:
        text_val = _format_metrics_for_annotation(metrics_val)
        fig.add_annotation(
            text=f"<b>Validation</b><br>{text_val}",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.98,
            y=val_y_anchor,
            xanchor="right",
            yanchor="top",
            bordercolor=colors["validation"],
            borderwidth=2,
            bgcolor="rgba(255,255,255,0.95)",
            font=dict(
                size=metrics_font_size,
                color=colors["text"],
                family="Courier New, monospace",
            ),
        )

    if metrics_test:
        text_test = _format_metrics_for_annotation(metrics_test)
        fig.add_annotation(
            text=f"<b>Test</b><br>{text_test}",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.98,
            y=test_y_anchor,
            xanchor="right",
            yanchor="top",
            bordercolor=colors["test_border"],
            borderwidth=2,
            bgcolor="rgba(255,255,255,0.95)",
            font=dict(
                size=metrics_font_size,
                color=colors["text"],
                family="Courier New, monospace",
            ),
        )

    # --- Train Provenance Annotation ---
    if split_train:
        fig.add_annotation(
            text=(
                f"<b>Train Provenance</b><br>"
                f"<span style='color:{colors['train_from_x']}'>■</span> from X (context)  "
                f"<span style='color:{colors['train_from_y']}'>■</span> from y (tail, H={h})"
            ),
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.1,
            xanchor="left",
            yanchor="bottom",
            bordercolor="#ddd",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(
                size=14 * font_scale,
                color=colors["text"],
                family=FONT_FAMILY,
            ),
        )

    # --- Final Layout ---
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>", x=0.5, y=0.9, xanchor="center", yanchor="top"
        ),
        xaxis_title_text="Days",
        yaxis_title_text=yaxis_title,
        font=dict(
            family=FONT_FAMILY, color=colors["text"], size=16 * font_scale
        ),
        xaxis=dict(
            gridcolor=colors["grid"],
            zeroline=False,
            title_font_size=20 * font_scale,
            tickfont_size=16 * font_scale,
        ),
        yaxis=dict(
            gridcolor=colors["grid"],
            zeroline=False,
            range=y_range,
            title_font_size=20 * font_scale,
            tickfont_size=16 * font_scale,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=16 * font_scale),
            visible=False,
        ),
        plot_bgcolor=colors["plot_bgcolor"],
        paper_bgcolor=colors["paper_bgcolor"],
        width=width,
        height=height,
        margin=dict(t=120, b=150, l=100, r=40),
        hovermode="x unified",
    )

    if show:
        high_res_config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "integrated_view",
                "scale": 3,
            }
        }
        fig.show(config=high_res_config)

    return fig


def _plot_seq(
    *,
    truth: np.ndarray,
    pred: np.ndarray,
    metrics: Dict[str, float],
    label: str,
    scaler_target,
    params: Dict[str, Any],
    well: str,
    is_cum: bool = False,
    split: str = "test",
    plot: bool = True,
    ensemble_out: Optional["EnsembleOutput"] = None,
):
    if not plot:
        return

    smape_val = float(metrics.get("SMAPE")) if metrics.get("SMAPE") is not None else None
    mae_val   = float(metrics.get("MAE"))   if metrics.get("MAE")   is not None else None
    r2_val    = float(metrics.get("R²"))    if metrics.get("R²")    is not None else None

    # --- NEW: Define metric labels based on the context (cumulative or aggregated) ---
    metric_labels = {"SMAPE": "APE" if is_cum else "sMAPE"}

    percentage_split = 1 - params.get("test_size", 0.0) - params.get("val_size", 0.0)
    
    # The common keyword arguments dictionary now includes the metric_labels
    common_kw = dict(
        window_size=params["lag_window"],
        forecast_steps=params["horizon"],
        percentage_split=percentage_split,
        show_components=params.get("show_components", False),
        title=label,
        smape=smape_val,
        mae=mae_val,
        r2=r2_val,
        # --- NEW: Inject the labels dictionary into the kwargs ---
        metric_labels=metric_labels,
    )

    if ensemble_out is not None:
        from common.seq_preprocessing import reconstruct_true_series

        kind = params.get("__plot_kind__", "P50")
        band = params.get("band")
        manual_env: Optional[Tuple[np.ndarray, np.ndarray]] = None

        if is_cum and band is not None:
            mu_stack  = getattr(ensemble_out, "pred_val" if split == "val" else "pred_test", None)
            sigma_st  = getattr(ensemble_out, "sigma_val" if split == "val" else "sigma_test", None)
            if (mu_stack is not None) and (sigma_st is not None):
                mu_rate_scaled  = reconstruct_true_series(mu_stack)
                sig_rate_scaled = reconstruct_true_series(sigma_st)
                if _looks_scaled(mu_rate_scaled):
                    mu_rate_phys = _inverse_transform_1d(scaler_target, mu_rate_scaled)
                    scale = getattr(scaler_target, "scale_", [1.0])[0] if scaler_target is not None else 1.0
                    sig_rate_phys = sig_rate_scaled * scale
                else:
                    mu_rate_phys  = mu_rate_scaled
                    sig_rate_phys = sig_rate_scaled
                plo, phi = band
                low_rate  = scenario_curve(mu_rate_phys, sig_rate_phys, plo)
                high_rate = scenario_curve(mu_rate_phys, sig_rate_phys, phi)
                adj = truth[0] - mu_rate_phys[0]
                manual_env = (np.cumsum(low_rate) + adj, np.cumsum(high_rate) + adj)

        # The **common_kw now transparently passes metric_labels to the wrapper
        plot_predictions_wrapper(
            ensemble_out,
            truth=truth,
            kind=kind,
            well=well,
            band=band,
            mean_override=pred,
            manual_envelope=manual_env,
            is_cum=is_cum,
            scaler=None,
            **common_kw,
        )
    else:
        from evaluation.evaluation import evaluate_and_plot
        
        # We need to construct additional_params carefully to include the new labels
        additional_params = dict(
            window_size=common_kw["window_size"],
            forecast_steps=common_kw["forecast_steps"],
            percentage_split=common_kw["percentage_split"],
            # --- NEW: Inject labels into the nested dictionary for evaluate_and_plot path ---
            metric_labels=metric_labels,
        )

        evaluate_and_plot(
            y_true=truth,
            y_pred=pred,
            title=f"{label} – {'Cumulative' if is_cum else 'Aggregated'}",
            well=well,
            set_name=("val" if split == "val" else "test"),
            smape=smape_val,
            mae=mae_val,
            r2=r2_val,
            additional_params=additional_params,
        )

def _reconstruct_x_leadin_phys(
    x_train_windows: np.ndarray,   # Aceita (N, L, F) ou (N, L)
    scaler_target,
    target_feature_idx: int = -1,
) -> np.ndarray:
    """
    Reconstrói a série 1D do alvo a partir das janelas de X (stride=1).
    - Se for (N, L, F): seleciona o canal `target_feature_idx` → (N, L)
    - Se for (N, L): usa direto
    - Faz inverse_transform apenas se parecer escalado.
    Retorna comprimento L + (N - 1).
    """
    from common.seq_preprocessing import reconstruct_true_series

    if x_train_windows.ndim == 3:
        x_target_2d = x_train_windows[..., target_feature_idx]  # (N, L)
    elif x_train_windows.ndim == 2:
        x_target_2d = x_train_windows                           # (N, L) já é alvo
    else:
        raise ValueError(f"Esperado x_train_windows 2D ou 3D, veio {x_train_windows.ndim}D com shape={x_train_windows.shape}")

    if x_target_2d.ndim != 2:
        raise ValueError(f"Alvo em X precisa ser 2D (N, L), shape atual: {x_target_2d.shape}")

    series = reconstruct_true_series(x_target_2d)               # L + N - 1

    if _looks_scaled(series):
        series = _inverse_transform_1d(scaler_target, series)

    return series


# In forecast_pipeline/plotting.py

# def _plot_integrated_view_from_agg(
#     *,
#     agg_val_true: np.ndarray,
#     agg_val_pred: np.ndarray,
#     agg_test_true: np.ndarray,
#     agg_test_pred: np.ndarray,
#     y_train_original: np.ndarray,
#     x_train_windows: Optional[np.ndarray],
#     params: Dict[str, Any],
#     label: str,
#     well: str,
#     config: Dict[str, Any],
#     metrics_val_agg: Dict[str, float],
#     metrics_test_agg: Dict[str, float],
#     metrics_val_cum: Dict[str, float],
#     metrics_test_cum: Dict[str, float],
#     scaler_target,
#     target_feature_idx: int = -1,
#     plot: bool = True,
# ):
#     """
#     Assembles and plots the integrated view for both aggregated (rate) and cumulative series.
#     This refactored version creates display-specific metric dictionaries with the correct labels
#     ("sMAPE" for rates, "APE" for cumulative) before calling the plotting function,
#     ensuring the visual output is identical except for the metric label.
#     """
#     if not plot:
#         return

#     # =================================================================================
#     # PART 1: DATA ASSEMBLY (LOGIC IS IDENTICAL TO THE ORIGINAL)
#     # =================================================================================
#     L = int(params["lag_window"])
#     H = int(params["horizon"])
#     N = int(y_train_original.shape[0])

#     train_y_series = _reconstruct_train_series_phys(y_train_original, scaler_target)
#     assert x_train_windows is not None, "x_train_windows must be provided to plot the integrated view."
#     leadin_x = _reconstruct_x_leadin_phys(x_train_windows, scaler_target, target_feature_idx)
    
#     start_tail = N - 1
#     tail_y = train_y_series[start_tail : start_tail + H]
#     assert tail_y.shape[0] == H, f"Tail from y has unexpected length: {tail_y.shape[0]} vs H={H}"

#     train_integrated = np.concatenate([leadin_x, tail_y])
#     len_train = train_integrated.shape[0]
#     len_val, len_test = len(agg_val_true), len(agg_test_true)
#     total_len = len_train + len_val + len_test
#     x_axis = np.arange(total_len)

#     split_indices = {"train_end": len_train, "val_end": len_train + len_val}
#     y_actual_full = np.concatenate([train_integrated, agg_val_true, agg_test_true])

#     y_pred_val_full = np.full(total_len, np.nan)
#     y_pred_val_full[split_indices["train_end"]:split_indices["val_end"]] = agg_val_pred
#     y_pred_test_full = np.full(total_len, np.nan)
#     y_pred_test_full[split_indices["val_end"]:] = agg_test_pred

#     # =================================================================================
#     # PART 2: AGGREGATED (RATE) PLOT WITH CORRECT LABEL
#     # =================================================================================
    
#     # Create a new dictionary for display, mapping the internal "SMAPE" to the "sMAPE" label.
#     metrics_agg_val_display = {"sMAPE": metrics_val_agg.get("SMAPE"), "MAE": metrics_val_agg.get("MAE")}
#     metrics_agg_test_display = {"sMAPE": metrics_test_agg.get("SMAPE"), "MAE": metrics_test_agg.get("MAE")}
    
#     # Filter out None values to prevent them from showing up in the plot annotation.
#     metrics_agg_val_display = {k: v for k, v in metrics_agg_val_display.items() if v is not None}
#     metrics_agg_test_display = {k: v for k, v in metrics_agg_test_display.items() if v is not None}

#     # Call the plotting function. Its internal logic remains unchanged.
#     plot_integrated_view(
#         x_axis=x_axis, y_actual=y_actual_full,
#         y_pred_val=y_pred_val_full, y_pred_test=y_pred_test_full,
#         split_indices=split_indices,
#         metrics_val=metrics_agg_val_display,
#         metrics_test=metrics_agg_test_display,
#         title=label, yaxis_title="Production Rate", well=well, horizon=H
#     )

#     # =================================================================================
#     # PART 3: CUMULATIVE PLOT WITH CORRECT LABEL
#     # =================================================================================
    
#     # Cumulative series calculation is identical to the original.
#     y_actual_full_cum = np.cumsum(y_actual_full)

#     y_pred_val_cum = np.full(total_len, np.nan)
#     val_anchor = y_actual_full_cum[len_train - 1] if len_train > 0 else 0.0
#     y_pred_val_cum[len_train:split_indices["val_end"]] = val_anchor + np.cumsum(agg_val_pred)

#     y_pred_test_cum = np.full(total_len, np.nan)
#     test_anchor = y_actual_full_cum[split_indices["val_end"] - 1] if len_val > 0 else val_anchor
#     y_pred_test_cum[split_indices["val_end"]:] = test_anchor + np.cumsum(agg_test_pred)
    
#     # Create a new dictionary for display, mapping the internal "SMAPE" to the "APE" label.
#     metrics_cum_val_display = {"APE": metrics_val_cum.get("SMAPE"), "MAE": metrics_val_cum.get("MAE")}
#     metrics_cum_test_display = {"APE": metrics_test_cum.get("SMAPE"), "MAE": metrics_test_cum.get("MAE")}

#     # Filter out None values.
#     metrics_cum_val_display = {k: v for k, v in metrics_cum_val_display.items() if v is not None}
#     metrics_cum_test_display = {k: v for k, v in metrics_cum_test_display.items() if v is not None}

#     # Call the plotting function again for the cumulative view.
#     plot_integrated_view(
#         x_axis=x_axis, y_actual=y_actual_full_cum,
#         y_pred_val=y_pred_val_cum, y_pred_test=y_pred_test_cum,
#         split_indices=split_indices,
#         metrics_val=metrics_cum_val_display,
#         metrics_test=metrics_cum_test_display,
#         title=label, yaxis_title="Cumulative Sum", well=well, horizon=H
#     )


def _plot_integrated_view_from_agg(
    *,
    agg_val_true: np.ndarray,
    agg_val_pred: np.ndarray,
    agg_test_true: np.ndarray,
    agg_test_pred: np.ndarray,
    y_train_original: np.ndarray,
    x_train_windows: Optional[np.ndarray],
    params: Dict[str, Any],
    label: str,
    well: str,
    config: Dict[str, Any],
    metrics_val_agg: Dict[str, float],
    metrics_test_agg: Dict[str, float],
    metrics_val_cum: Dict[str, float],
    metrics_test_cum: Dict[str, float],
    scaler_target,
    target_feature_idx: int = -1,
    plot: bool = True,
):
    """
    Assembles and plots the integrated view for both aggregated (rate) and cumulative series.

    Plug-and-play update:
      - If params includes integrated_view_val_members / integrated_view_test_members,
        overlays spaghetti members in plot_integrated_view (visual only).
      - Metrics remain exactly the same.
    """
    if not plot:
        return

    # =================================================================================
    # PART 1: DATA ASSEMBLY (LOGIC IS IDENTICAL TO THE ORIGINAL)
    # =================================================================================
    L = int(params["lag_window"])
    H = int(params["horizon"])
    N = int(y_train_original.shape[0])

    train_y_series = _reconstruct_train_series_phys(y_train_original, scaler_target)
    assert x_train_windows is not None, "x_train_windows must be provided to plot the integrated view."
    leadin_x = _reconstruct_x_leadin_phys(x_train_windows, scaler_target, target_feature_idx)

    start_tail = N - 1
    tail_y = train_y_series[start_tail : start_tail + H]
    assert tail_y.shape[0] == H, f"Tail from y has unexpected length: {tail_y.shape[0]} vs H={H}"

    train_integrated = np.concatenate([leadin_x, tail_y])
    len_train = train_integrated.shape[0]
    len_val, len_test = len(agg_val_true), len(agg_test_true)
    total_len = len_train + len_val + len_test
    x_axis = np.arange(total_len)

    split_indices = {"train_end": len_train, "val_end": len_train + len_val}
    y_actual_full = np.concatenate([train_integrated, agg_val_true, agg_test_true])

    y_pred_val_full = np.full(total_len, np.nan)
    y_pred_val_full[split_indices["train_end"] : split_indices["val_end"]] = agg_val_pred

    y_pred_test_full = np.full(total_len, np.nan)
    y_pred_test_full[split_indices["val_end"] :] = agg_test_pred

    # =================================================================================
    # NEW: optional spaghetti members (expected shapes: (K, len_val) / (K, len_test))
    # =================================================================================
    def _to_2d(a: Any) -> Optional[np.ndarray]:
        if a is None:
            return None
        arr = np.asarray(a, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return None
        return arr

    def _embed_members_into_full(members_2d: Optional[np.ndarray], *, start: int, end: int) -> Optional[np.ndarray]:
        """
        Embed members into a (K, total_len) array with NaNs outside.
    
        Accepts:
          - (K, end-start): split-only members (VAL or TEST)
          - (K, total_len): already full-length members (passes through)
        """
        if members_2d is None:
            return None
        mem = np.asarray(members_2d, dtype=float)
        if mem.ndim == 1:
            mem = mem.reshape(1, -1)
        if mem.ndim != 2 or mem.size == 0:
            return None
    
        K, Hm = mem.shape
    
        # already full-length
        if Hm == total_len:
            return mem
    
        # split-only
        seg_len = int(end - start)
        if Hm != seg_len:
            # last resort: crop if longer; ignore if shorter
            if Hm > seg_len:
                mem = mem[:, :seg_len]
            else:
                return None
    
        full = np.full((K, total_len), np.nan, dtype=float)
        full[:, start:end] = mem
        return full


    # These are purely visual inputs. If missing/invalid, we ignore safely.
    val_members_local = _to_2d(params.get("integrated_view_val_members"))
    test_members_local = _to_2d(params.get("integrated_view_test_members"))

    logging.info(
        "[integrated_debug] params_members raw_val=%s raw_test=%s",
        None if params.get("integrated_view_val_members") is None else np.asarray(params.get("integrated_view_val_members")).shape,
        None if params.get("integrated_view_test_members") is None else np.asarray(params.get("integrated_view_test_members")).shape,
    )
    
    logging.info(
        "[integrated_debug] local_members val=%s test=%s total_len=%d len_train=%d len_val=%d len_test=%d train_end=%d val_end=%d",
        None if val_members_local is None else tuple(val_members_local.shape),
        None if test_members_local is None else tuple(test_members_local.shape),
        int(total_len), int(len_train), int(len_val), int(len_test),
        int(split_indices["train_end"]), int(split_indices["val_end"]),
    )


    y_pred_val_members_full = _embed_members_into_full(
        val_members_local,
        start=split_indices["train_end"],
        end=split_indices["val_end"],
    )
    y_pred_test_members_full = _embed_members_into_full(
        test_members_local,
        start=split_indices["val_end"],
        end=total_len,
    )

    logging.info(
        "[integrated_debug] full_members val=%s test=%s",
        None if y_pred_val_members_full is None else tuple(y_pred_val_members_full.shape),
        None if y_pred_test_members_full is None else tuple(y_pred_test_members_full.shape),
    )


    # =================================================================================
    # PART 2: AGGREGATED (RATE) PLOT WITH CORRECT LABEL
    # =================================================================================
    metrics_agg_val_display = {"sMAPE": metrics_val_agg.get("SMAPE"), "MAE": metrics_val_agg.get("MAE")}
    metrics_agg_test_display = {"sMAPE": metrics_test_agg.get("SMAPE"), "MAE": metrics_test_agg.get("MAE")}

    metrics_agg_val_display = {k: v for k, v in metrics_agg_val_display.items() if v is not None}
    metrics_agg_test_display = {k: v for k, v in metrics_agg_test_display.items() if v is not None}

    plot_integrated_view(
        x_axis=x_axis,
        y_actual=y_actual_full,
        y_pred_val=y_pred_val_full,
        y_pred_test=y_pred_test_full,
        split_indices=split_indices,
        metrics_val=metrics_agg_val_display,
        metrics_test=metrics_agg_test_display,
        title=label,
        yaxis_title="Production Rate",
        well=well,
        horizon=H,
        # NEW
        y_pred_val_members=y_pred_val_members_full,
        y_pred_test_members=y_pred_test_members_full,
    )

    # =================================================================================
    # PART 3: CUMULATIVE PLOT WITH CORRECT LABEL
    # =================================================================================
    y_actual_full_cum = np.cumsum(y_actual_full)

    y_pred_val_cum = np.full(total_len, np.nan)
    val_anchor = y_actual_full_cum[len_train - 1] if len_train > 0 else 0.0
    y_pred_val_cum[len_train : split_indices["val_end"]] = val_anchor + np.cumsum(agg_val_pred)

    y_pred_test_cum = np.full(total_len, np.nan)
    test_anchor = y_actual_full_cum[split_indices["val_end"] - 1] if len_val > 0 else val_anchor
    y_pred_test_cum[split_indices["val_end"] :] = test_anchor + np.cumsum(agg_test_pred)

    # NEW: cumulative members (same anchors as above)
    def _members_to_cum(members_full: Optional[np.ndarray], *, anchor: float, start: int, end: int) -> Optional[np.ndarray]:
        """
        Convert full-length rate members (K, total_len) to cumulative members (K, total_len),
        using the same anchoring strategy as the main curve.
        Only converts inside [start:end], keeps NaNs elsewhere.
        """
        if members_full is None:
            return None
        mem = np.asarray(members_full, dtype=float)
        if mem.ndim != 2 or mem.shape[1] != total_len:
            return None
        out = np.full_like(mem, np.nan)
        seg = mem[:, start:end]
        # cumsum with NaNs -> we want NaNs to stay NaNs; simplest is nan_to_num but that changes meaning.
        # Here we require finite segment; if NaNs exist inside segment, we skip that member.
        for i in range(mem.shape[0]):
            s = seg[i]
            if not np.all(np.isfinite(s)):
                continue
            out[i, start:end] = anchor + np.cumsum(s)
        return out

    y_pred_val_members_cum = _members_to_cum(
        y_pred_val_members_full, anchor=float(val_anchor),
        start=len_train, end=split_indices["val_end"]
    )
    y_pred_test_members_cum = _members_to_cum(
        y_pred_test_members_full, anchor=float(test_anchor),
        start=split_indices["val_end"], end=total_len
    )

    metrics_cum_val_display = {"APE": metrics_val_cum.get("SMAPE"), "MAE": metrics_val_cum.get("MAE")}
    metrics_cum_test_display = {"APE": metrics_test_cum.get("SMAPE"), "MAE": metrics_test_cum.get("MAE")}

    metrics_cum_val_display = {k: v for k, v in metrics_cum_val_display.items() if v is not None}
    metrics_cum_test_display = {k: v for k, v in metrics_cum_test_display.items() if v is not None}

    plot_integrated_view(
        x_axis=x_axis,
        y_actual=y_actual_full_cum,
        y_pred_val=y_pred_val_cum,
        y_pred_test=y_pred_test_cum,
        split_indices=split_indices,
        metrics_val=metrics_cum_val_display,
        metrics_test=metrics_cum_test_display,
        title=label,
        yaxis_title="Cumulative Sum",
        well=well,
        horizon=H,
        # NEW
        y_pred_val_members=y_pred_val_members_cum,
        y_pred_test_members=y_pred_test_members_cum,
    )


def plot_darts_integrated(
    *,
    train_kwargs: Dict[str, Any],
    prediction_input: Dict[str, Any],
    pred_val_ribbons: np.ndarray,    # ribbons from HF(start=len(train))
    pred_test_ribbons: np.ndarray,   # ribbons from HF(start=len(train)+len(val))
    title_prefix: str,
    well: str = "",                  # optional passthrough to integrated view
    yaxis_title: str = "Rate",
    font_scale: float = 1.2,
    show: bool = True,
    shade: Optional[Dict[str, Tuple[int, int]]] = None,
) -> go.Figure:
    """
    Adapter: convert Darts ribbons + TimeSeries into the inputs required by
    `plot_integrated_view`, then delegate the plotting.

    Keeps backward compatibility with existing calls.
    """
    from common.eval_darts import latest_from_ribbons, _inv_ts_1d, _inv_ribbons_2d
    from evaluation.evaluation import smape
    
    # ---- 1) Unpack essentials ----
    main_col = train_kwargs["main_col"]
    scaler   = train_kwargs["scaler_target"]
    ts_train = train_kwargs["X_train"][main_col]
    ts_val   = train_kwargs["X_val"][main_col]
    ts_test  = prediction_input["ts_test"][main_col]

    # ---- 2) Inverse-scale actuals and stitch full timeline ----
    y_train = _inv_ts_1d(ts_train, scaler)
    y_val   = _inv_ts_1d(ts_val,   scaler)
    y_test  = _inv_ts_1d(ts_test,  scaler)
    y_actual = np.concatenate([y_train, y_val, y_test], axis=0)

    T_tr, T_va, T_te = len(y_train), len(y_val), len(y_test)
    total_len = T_tr + T_va + T_te
    x_axis = np.arange(total_len, dtype=int)

    # ---- 3) Reconstruct latest-wins predictions and align on the full axis ----
    # Note: pred_val ribbons span VAL+TEST; we only plot the VAL portion here.
    y_pred_val = np.full(total_len, np.nan, dtype=float)
    y_pred_test = np.full(total_len, np.nan, dtype=float)

    if pred_val_ribbons is not None and np.size(pred_val_ribbons):
        val_rib_u = _inv_ribbons_2d(pred_val_ribbons, scaler)
        yhat_val_full = latest_from_ribbons(val_rib_u)         # len = T_va + T_te
        yhat_val_only = yhat_val_full[:T_va]                    # VAL portion only
        y_pred_val[T_tr:T_tr + T_va] = yhat_val_only

    if pred_test_ribbons is not None and np.size(pred_test_ribbons):
        test_rib_u = _inv_ribbons_2d(pred_test_ribbons, scaler)
        yhat_test_only = latest_from_ribbons(test_rib_u)[:T_te] # TEST only
        y_pred_test[T_tr + T_va:T_tr + T_va + T_te] = yhat_test_only

    # ---- 4) Metrics on ORIGINAL units (only inside their regions) ----
    smape_val  = smape(y_val,  y_pred_val[T_tr:T_tr + T_va])  if T_va > 0 else np.nan
    smape_test = smape(y_test, y_pred_test[T_tr + T_va:])     if T_te > 0 else np.nan
    metrics_val  = {"sMAPE (%)": smape_val}  if T_va > 0 else None
    metrics_test = {"sMAPE (%)": smape_test} if T_te > 0 else None

    # ---- 5) Boundaries + horizon for the richer plot ----
    split_indices = {
        "train_end": T_tr,           # exclusive
        "val_end":   T_tr + T_va,    # exclusive
    }
    # Prefer the configured horizon; otherwise infer from ribbons
    H = (
        int(train_kwargs.get("params", {}).get("output_chunk_length"))
        or (pred_test_ribbons.shape[1] if pred_test_ribbons is not None and pred_test_ribbons.ndim == 2 else None)
        or (pred_val_ribbons.shape[1]  if pred_val_ribbons  is not None and pred_val_ribbons.ndim  == 2 else None)
    )

    # ---- 6) Delegate to the refined view ----
    title = f"{title_prefix}"
    fig = plot_integrated_view(
        x_axis=x_axis,
        y_actual=y_actual,
        y_pred_val=y_pred_val,
        y_pred_test=y_pred_test,
        split_indices=split_indices,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        title=title,
        yaxis_title=yaxis_title,
        well=well or train_kwargs.get("well", ""),
        horizon=H,
        font_scale=font_scale,
        show=show,
        shade=shade,
    )
    return fig



# ---------------------------------------------------------------------
# Montagem de timeline integrada especificamente para ARPS (point-forecast)
# ---------------------------------------------------------------------
def assemble_arps_integrated_series(
    *,
    y_train_windows: np.ndarray,      # (N,H)
    x_train_windows: np.ndarray,      # (N,L, F) ou (N,L)
    scaler_target,
    agg_val_true: np.ndarray,         # 1D físico
    agg_test_true: np.ndarray,        # 1D físico
    L: int,
    H: int,
) -> Dict[str, Any]:
    """
    Constrói a série 'actual' completa (Train integrado + Val + Test) e os índices de split.
    Train integrado = lead-in de X (L + N - 1) + cauda de y (H).
    """
    # 1) Reconstruções físicas
    leadin_x = _reconstruct_x_leadin_phys(x_train_windows, scaler_target, -1)  # len = L + N - 1
    y_train_series = _reconstruct_train_series_phys(y_train_windows, scaler_target)  # len = N + H - 1
    N = int(y_train_windows.shape[0])

    # 2) Tail de y (H pontos), alinhada exatamente no fim de leadin_x
    start_tail = N - 1
    tail_y = y_train_series[start_tail : start_tail + H]
    if tail_y.shape[0] != H:
        raise RuntimeError(f"Tail de y com tamanho inesperado: {tail_y.shape[0]} != H={H}")

    # 3) Train integrado e splits
    train_integrated = np.concatenate([leadin_x, tail_y])        # len_train
    len_train = train_integrated.shape[0]
    len_val   = int(agg_val_true.shape[0])
    len_test  = int(agg_test_true.shape[0])

    # 4) Série ‘actual’ completa
    y_actual_full = np.concatenate([train_integrated, agg_val_true, agg_test_true])
    total_len = len_train + len_val + len_test
    x_axis    = np.arange(total_len, dtype=int)

    return {
        "x_axis": x_axis,
        "y_actual_full": y_actual_full,
        "len_train": len_train,
        "len_val": len_val,
        "len_test": len_test,
        "train_integrated": train_integrated,
        "leadin_x": leadin_x,
        "tail_y": tail_y,
        "split_indices": {"train_end": len_train, "val_end": len_train + len_val},
    }

# ---------------------------------------------------------------------
# Plot integrado (rate e cumulative) a partir de séries ARPS (point-forecast)
# Reusa a API pública 'plot_integrated_view' se já existir no projeto.
# ---------------------------------------------------------------------
def plot_arps_integrated_from_point(
    *,
    agg_val_true: np.ndarray, agg_val_pred: np.ndarray,
    agg_test_true: np.ndarray, agg_test_pred: np.ndarray,
    y_train_windows: np.ndarray, x_train_windows: np.ndarray,
    scaler_target,
    params: Dict[str, Any],
    label: str, well: str,
    metrics_val_agg: Dict[str, float], metrics_test_agg: Dict[str, float],
    metrics_val_cum: Dict[str, float], metrics_test_cum: Dict[str, float],
    plot_integrated_view_fn,   # inject plotting fn (avoid import cycles)
) -> None:
    """
    Builds and plots the integrated view (rate and cumulative) for ARPS path.

    Plug-and-play enhancement:
      - If params contains:
          params["integrated_view_val_members"]  shape (K, len_val)
          params["integrated_view_test_members"] shape (K, len_test)
        then overlays spaghetti members in plot_integrated_view_fn by passing:
          y_pred_val_members / y_pred_test_members (shape (K, total_len))
        for both RATE and CUMULATIVE plots.

    Safe behavior:
      - If members are missing or shape-mismatched, overlays are skipped silently (with debug logs).
    """
    import logging
    log = logging.getLogger(__name__)

    L = int(params["lag_window"])
    H = int(params["horizon"])

    built = assemble_arps_integrated_series(
        y_train_windows=y_train_windows,
        x_train_windows=x_train_windows,
        scaler_target=scaler_target,
        agg_val_true=agg_val_true,
        agg_test_true=agg_test_true,
        L=L, H=H,
    )

    x_axis         = built["x_axis"]
    y_actual_full  = built["y_actual_full"]
    split_indices  = built["split_indices"]
    len_train      = int(built["len_train"])
    len_val        = int(built["len_val"])
    total_len      = int(x_axis.size)

    # ----------------------------
    # 1) Region-aligned predictions (NaN outside)
    # ----------------------------
    y_pred_val_full  = np.full(total_len, np.nan, dtype=float)
    y_pred_test_full = np.full(total_len, np.nan, dtype=float)

    y_pred_val_full[split_indices["train_end"] : split_indices["val_end"]] = np.asarray(agg_val_pred, dtype=float)
    y_pred_test_full[split_indices["val_end"] : ]                          = np.asarray(agg_test_pred, dtype=float)

    # ----------------------------
    # 2) OPTIONAL: spaghetti members (RATE) from params
    #    expected shapes: (K,len_val) and (K,len_test)
    # ----------------------------
    def _to_2d(a: Any) -> Optional[np.ndarray]:
        if a is None:
            return None
        arr = np.asarray(a, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return None
        return arr

    def _embed_members_into_full(members_2d: Optional[np.ndarray], *, start: int, end: int) -> Optional[np.ndarray]:
        if members_2d is None:
            return None
        need = int(end - start)
        if members_2d.shape[1] != need:
            log.info(
                "[arps_integrated_debug] members shape mismatch: got=%s need=(K,%d) [start=%d end=%d]",
                tuple(members_2d.shape), need, start, end
            )
            return None
        full = np.full((members_2d.shape[0], total_len), np.nan, dtype=float)
        full[:, start:end] = members_2d
        return full

    val_members_local  = _to_2d(params.get("integrated_view_val_members"))
    test_members_local = _to_2d(params.get("integrated_view_test_members"))

    y_pred_val_members_full = _embed_members_into_full(
        val_members_local,
        start=int(split_indices["train_end"]),
        end=int(split_indices["val_end"]),
    )
    y_pred_test_members_full = _embed_members_into_full(
        test_members_local,
        start=int(split_indices["val_end"]),
        end=total_len,
    )

    log.info(
        "[arps_integrated_debug] members_full rate: val=%s test=%s",
        None if y_pred_val_members_full is None else tuple(y_pred_val_members_full.shape),
        None if y_pred_test_members_full is None else tuple(y_pred_test_members_full.shape),
    )

    # ----------------------------
    # 3) RATE integrated plot (+ spaghetti if available)
    # ----------------------------
    plot_integrated_view_fn(
        x_axis=x_axis,
        y_actual=y_actual_full,
        y_pred_val=y_pred_val_full,
        y_pred_test=y_pred_test_full,
        split_indices=split_indices,
        metrics_val={"sMAPE": metrics_val_agg.get("SMAPE"), "MAE": metrics_val_agg.get("MAE")},
        metrics_test={"sMAPE": metrics_test_agg.get("SMAPE"), "MAE": metrics_test_agg.get("MAE")},
        title=label,
        yaxis_title="Production Rate",
        well=well,
        horizon=H,
        font_scale=1.32,
        show=True,
        # NEW
        y_pred_val_members=y_pred_val_members_full,
        y_pred_test_members=y_pred_test_members_full,
    )

    # ----------------------------
    # 4) CUMULATIVE integrated plot (+ cumulative spaghetti)
    # ----------------------------
    y_actual_cum = np.cumsum(np.asarray(y_actual_full, dtype=float))

    y_pred_val_cum  = np.full(total_len, np.nan, dtype=float)
    y_pred_test_cum = np.full(total_len, np.nan, dtype=float)

    val_anchor  = float(y_actual_cum[len_train - 1]) if len_train > 0 else 0.0
    test_anchor = float(y_actual_cum[int(split_indices["val_end"]) - 1]) if len_val > 0 else val_anchor

    y_pred_val_cum[len_train : int(split_indices["val_end"])] = val_anchor + np.cumsum(np.asarray(agg_val_pred, dtype=float))
    y_pred_test_cum[int(split_indices["val_end"]) : ]         = test_anchor + np.cumsum(np.asarray(agg_test_pred, dtype=float))

    def _members_to_cum(
        members_full: Optional[np.ndarray],
        *,
        anchor: float,
        start: int,
        end: int
    ) -> Optional[np.ndarray]:
        if members_full is None:
            return None
        mem = np.asarray(members_full, dtype=float)
        if mem.ndim != 2 or mem.shape[1] != total_len:
            return None
        out = np.full_like(mem, np.nan)
        seg = mem[:, start:end]
        # only cum members with fully finite segment (avoid NaN-propagation surprises)
        for i in range(mem.shape[0]):
            s = seg[i]
            if not np.all(np.isfinite(s)):
                continue
            out[i, start:end] = anchor + np.cumsum(s)
        return out

    y_pred_val_members_cum = _members_to_cum(
        y_pred_val_members_full,
        anchor=val_anchor,
        start=len_train,
        end=int(split_indices["val_end"]),
    )
    y_pred_test_members_cum = _members_to_cum(
        y_pred_test_members_full,
        anchor=test_anchor,
        start=int(split_indices["val_end"]),
        end=total_len,
    )

    log.info(
        "[arps_integrated_debug] members_full cum:  val=%s test=%s",
        None if y_pred_val_members_cum is None else tuple(y_pred_val_members_cum.shape),
        None if y_pred_test_members_cum is None else tuple(y_pred_test_members_cum.shape),
    )

    plot_integrated_view_fn(
        x_axis=x_axis,
        y_actual=y_actual_cum,
        y_pred_val=y_pred_val_cum,
        y_pred_test=y_pred_test_cum,
        split_indices=split_indices,
        metrics_val={"APE": metrics_val_cum.get("SMAPE"), "MAE": metrics_val_cum.get("MAE")},
        metrics_test={"APE": metrics_test_cum.get("SMAPE"), "MAE": metrics_test_cum.get("MAE")},
        title=label,
        yaxis_title="Cumulative Sum",
        well=well,
        horizon=H,
        font_scale=1.32,
        show=True,
        # NEW
        y_pred_val_members=y_pred_val_members_cum,
        y_pred_test_members=y_pred_test_members_cum,
    )



def plot_error_distributions_story(
    df: pd.DataFrame,
    *,
    chosen_row: pd.Series,
    title: str,
    val_col: str = "val_smape_agg",
    test_col: str = "test_smape_agg",
    dataset: str = "",
    well: str = "",
    architecture: str = "",
    chosen_strategy: Optional[str] = None,
    chosen_trial: Optional[Any] = None,
    best_test: Optional[float] = None,
    regret_test: Optional[float] = None,
    ratio_test: Optional[float] = None,
    spearman_val_test: Optional[float] = None,
    pool_chosen_test_percentile: Optional[float] = None,
    palette: Optional[Dict[str, str]] = None,
    width: int = 1000, 
    height: int = 800,
    show: bool = True,
) -> "plotly.graph_objs._figure.Figure":
    
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise ImportError("Plotly is required. Install with: pip install plotly") from e

    # ---------- Palette Refinada (Charcoal Grey + User Colors) ----------
    colors = {
        "validation": "#E53935",        
        "test_rolling": "#4CAF50",     
        "charcoal": "#263238",         
        "fill_val": "rgba(255, 235, 238, 0.7)",
        "fill_test": "rgba(232, 245, 233, 0.7)",
        "text_main": "#263238",
        "text_dim": "#546E7A",
        "grid": "#ECEFF1",
        "panel_bg": "#FFFFFF",
        "panel_border": "#CFD8DC",
        "chosen": "#111827",
    }
    if palette: colors.update(palette)

    # ---------- Math Helpers ----------
    def _to_finite(series: pd.Series) -> np.ndarray:
        x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return x[np.isfinite(x)]

    def _robust_bandwidth(x: np.ndarray) -> float:
        n = len(x)
        if n < 2: return 1.0
        std = float(np.std(x, ddof=1))
        q75, q25 = np.percentile(x, [75, 25])
        iqr = float(q75 - q25)
        sigma = min(std, iqr / 1.349) if iqr > 0 else std
        sigma = sigma if (np.isfinite(sigma) and sigma > 1e-12) else 1.0
        return 1.06 * sigma * (n ** (-1 / 5))

    def _kde_1d(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
        z = (grid[:, None] - x[None, :]) / bw
        return np.exp(-0.5 * z * z).sum(axis=1) / (len(x) * bw * np.sqrt(2 * np.pi))

    def _density_at(grid: np.ndarray, dens: np.ndarray, x0: float) -> float:
        return float(np.interp(x0, grid, dens)) if np.isfinite(x0) else np.nan

    def _fmt(x: Optional[float], nd: int = 2) -> str:
        if x is None or not np.isfinite(float(x)): return "—"
        return f"{float(x):.{nd}f}"

    # ---------- Data Processing ----------
    v = _to_finite(df[val_col])
    t = _to_finite(df[test_col]) if test_col in df.columns else np.array([], dtype=float)
    
    chosen_val = float(pd.to_numeric(chosen_row.get(val_col, np.nan), errors="coerce"))
    chosen_test = float(pd.to_numeric(chosen_row.get(test_col, np.nan), errors="coerce")) if test_col in df.columns else np.nan
    best_val = float(np.nanmin(v)) if len(v) else np.nan
    
    allx = v if len(t) == 0 else np.concatenate([v, t])
    lo, hi = np.percentile(allx, [0.5, 99.5])
    pad = 0.15 * (hi - lo) if hi > lo else 5.0
    grid = np.linspace(lo - pad, hi + pad, 800)

    v_dn = _kde_1d(v, grid, _robust_bandwidth(v))
    v_dn /= (v_dn.max() + 1e-12)
    has_test = len(t) > 0
    t_dn = (_kde_1d(t, grid, _robust_bandwidth(t)) / (np.max(_kde_1d(t, grid, _robust_bandwidth(t))) + 1e-12)) if has_test else np.zeros_like(grid)

    # ---------- Layout Geometry (CLOSER DISTRIBUTION) ----------
    # Plot domain moved to start earlier (0.34) to close the gap
    x_plot_domain = [0.34, 0.98]
    y_val_base, y_test_base = 0.85, -0.3 
    h = 0.85

    fig = go.Figure()

    # --- Section Subtitles ---
    fig.add_annotation(xref="paper", yref="y", x=x_plot_domain[0], y=y_val_base + h + 0.12,
                       text="<b>VALIDATION DISTRIBUTION</b>", showarrow=False,
                       font=dict(size=16, color=colors["validation"]), xanchor="left")
    if has_test:
        fig.add_annotation(xref="paper", yref="y", x=x_plot_domain[0], y=y_test_base + h + 0.12,
                           text="<b>TEST DISTRIBUTION (AUDIT)</b>", showarrow=False,
                           font=dict(size=16, color=colors["test_rolling"]), xanchor="left")

    # --- Traces ---
    fig.add_trace(go.Scatter(x=grid, y=y_val_base + v_dn * h, mode="lines",
                             line=dict(color=colors["validation"], width=4),
                             fill="toself", fillcolor=colors["fill_val"], name="Validation"))
    if has_test:
        fig.add_trace(go.Scatter(x=grid, y=y_test_base + t_dn * h, mode="lines",
                                 line=dict(color=colors["test_rolling"], width=4),
                                 fill="toself", fillcolor=colors["fill_test"], name="Test"))

    # --- Markers ---
    def add_marker(y_base, d_norm, val, label, color, is_chosen=False):
        if not np.isfinite(val): return
        y_pt = y_base + _density_at(grid, d_norm, val) * h
        fig.add_trace(go.Scatter(x=[val], y=[y_pt], mode="markers+text",
                                 marker=dict(size=13 if is_chosen else 11, color=color, line=dict(width=2, color="white")),
                                 text=[f"<b>{label}</b>"], textposition="top center",
                                 textfont=dict(size=13, color=color), showlegend=False))

    add_marker(y_val_base, v_dn, chosen_val, "Chosen", colors["chosen"], True)
    add_marker(y_val_base, v_dn, best_val, "Best VAL", colors["validation"])
    if has_test:
        best_test_val = best_test if best_test is not None else np.nanmin(t)
        add_marker(y_test_base, t_dn, chosen_test, "Chosen", colors["chosen"], True)
        add_marker(y_test_base, t_dn, best_test_val, "Best TEST", colors["test_rolling"])

    # ---------- Information Panel (LEFT SIDE - INTEGRATED) ----------
    strat = chosen_strategy or chosen_row.get("physics_strategy", "—")
    trial = chosen_trial or chosen_row.get("optuna_trial_number", "—")
    
    panel_content = [
        f"<span style='font-size:14px; color:{colors['text_dim']}'>{dataset} · {well}</span>",
        f"<span style='font-size:18px; font-weight:bold; color:{colors['charcoal']}'>{architecture}</span>",
        "<br>",
        "<b>IDENTIFICATION</b>",
        f"Strategy: <span style='color:#0277BD'>{strat}</span>",
        f"Trial: <b>{trial}</b>",
        "<br>",
        "<b>PERFORMANCE (SMAPE %)</b>",
        f"<span style='color:{colors['validation']}'>■</span> Validation: <b>{_fmt(chosen_val)}%</b>",
        f"<span style='font-size:12px; color:#90A4AE'>Best in pool: {_fmt(best_val)}%</span>"
    ]

    if has_test:
        panel_content.extend([
            f"<br><span style='color:{colors['test_rolling']}'>■</span> Test (Audit): <b>{_fmt(chosen_test)}%</b>",
            f"<br><b>AUDIT METRICS</b>",
            f"Regret: <b>{_fmt(regret_test)}%</b>",
            f"Ratio: <b>{_fmt(ratio_test, 2)}x</b>",
            f"Spearman: <b>{_fmt(spearman_val_test, 3)}</b>"
        ])

    if "|" in title or "=" in title:
        panel_content.extend([
            "<br>",
            "<b>METADATA</b>",
            f"<span style='font-size:12px; color:{colors['text_dim']}'>{title.replace('|', '<br>')}</span>"
        ])
    
    # Anchor panel precisely to the left of the plot
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.9, 
        text="<br>".join(panel_content),
        showarrow=False, align="left", valign="top",
        font=dict(size=14, color=colors["text_main"], family="Arial, sans-serif"),
        bgcolor="rgba(255,255,255,1.0)", bordercolor=colors["panel_border"], borderwidth=1, borderpad=20
    )

    # ---------- Centralized Main Title (LOWERED) ----------
    fig.update_layout(
        title=dict(
            text="<b>Selection within Candidate Error Distribution</b>",
            x=0.5, y=0.9, 
            font=dict(size=28, color=colors["charcoal"]),
            xanchor='center'
        ),
        width=width, height=height,
        template="plotly_white",
        margin=dict(l=40, r=40, t=120, b=80),
        showlegend=False,
    )

    # ---------- Axes Styling (Charcoal) ----------
    fig.update_xaxes(
        title=dict(text="<b>Error (%)</b> — Lower is better", font=dict(size=18, color=colors["charcoal"])),
        domain=x_plot_domain, gridcolor=colors["grid"], tickfont=dict(size=14, color=colors["charcoal"]),
        zeroline=False, range=[grid.min(), grid.max()]
    )

    fig.update_yaxes(
        title=dict(text="<b>Relative Density</b>", font=dict(size=18, color=colors["charcoal"]), standoff=30),
        range=[-0.5, 2.1], 
        showticklabels=False, gridcolor=colors["grid"],
        zeroline=False
    )

    if show:
        fig.show()

    return fig


import re
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter, MaxNLocator

# (Opcional) alta resolução no Jupyter inline estático.
# Se quiser toolbar com botão de salvar, comente essas linhas e use:
# %matplotlib widget
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "retina")


# =============================================================================
# Storyboard Contract & Data Prep
# =============================================================================
@dataclass(frozen=True)
class _StoryboardColumns:
    test_size: str = "test_size"
    val_size: str = "val_size"
    hpo_trials: str = "HPO Trials"
    architecture: str = "Architecture"
    rate_test_mean: str = "Production Rate (Test) (MEAN)"
    cum_test_mean: str = "Cumulative Production (Test) (MEAN)"
    rate_test_iqr: str = "Production Rate (Test) (IQR)"
    cum_test_iqr: str = "Cumulative Production (Test) (IQR)"


def _get_theme(
    *,
    # ===== fontes principais =====
    note_fontsize_plot1: float = 12.5,
    note_fontsize_plot2: float = 12.5,
    badge_fontsize: float = 14.0,
    trial_label_fontsize: float = 13.0,
    split_title_fontsize: float = 16.0,
    axis_label_fontsize: float = 13.0,
    tick_fontsize: float = 11.5,
    panel_title_fontsize: float = 14.0,
    suptitle_fontsize: float = 17.0,
    legend_fontsize_plot1: float = 10.8,
    legend_fontsize_plot2: float = 10.8,
    # ===== layout de títulos / badges =====
    split_title_x: float = 0.14,       # move "Split 0.xx" p/ direita (Plot 2)
    badge_x: float = 0.012,
    badge_y: float = 1.025,
    # ===== tamanhos das figuras =====
    plot1_figsize: tuple = (14, 6.2),
    plot2_fig_width: float = 14.0,
    plot2_fig_row_height: float = 4.8,
    dpi: int = 120,
    # ===== legendas =====
    plot1_legend_ncol: int = 3,        # evita legendão em uma linha só
    plot2_legend_ncol: int = 3,
    # ===== notas / posicionamento =====
    plot1_left_note_loc: str = "top-left",
    plot1_right_note_loc: str = "bottom-right",  # move "more severe split →" p/ baixo (evita conflito)
    plot2_note_loc: str = "bottom-right",
) -> Dict[str, Any]:
    return {
        "bg": "#FFFFFF",
        "paper_bg": "#F9F9F9",
        "grid": "#E0E0E0",
        "text": "#2C3E50",
        "arch_colors": {
            "Arps Pure": "#E53935",
            "Arps Ensemble": "#4CAF50",
            "PINN Analytical": "#0277BD",
        },
        "fallback_colors": ["#8E24AA", "#F4511E", "#00897B", "#3949AB"],
        "fonts": {
            "note_plot1": note_fontsize_plot1,
            "note_plot2": note_fontsize_plot2,
            "badge": badge_fontsize,
            "trial_label": trial_label_fontsize,
            "split_title": split_title_fontsize,
            "axis_label": axis_label_fontsize,
            "tick": tick_fontsize,
            "panel_title": panel_title_fontsize,
            "suptitle": suptitle_fontsize,
            "legend_plot1": legend_fontsize_plot1,
            "legend_plot2": legend_fontsize_plot2,
        },
        "layout": {
            "split_title_x": split_title_x,
            "badge_x": badge_x,
            "badge_y": badge_y,
            "plot1_figsize": plot1_figsize,
            "plot2_fig_width": plot2_fig_width,
            "plot2_fig_row_height": plot2_fig_row_height,
            "dpi": dpi,
            "plot1_legend_ncol": plot1_legend_ncol,
            "plot2_legend_ncol": plot2_legend_ncol,
            "plot1_left_note_loc": plot1_left_note_loc,
            "plot1_right_note_loc": plot1_right_note_loc,
            "plot2_note_loc": plot2_note_loc,
        },
    }


def _metric_fmt(x: float, pos: int) -> str:
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:,.1f}"
    return f"{x:,.2f}"


def _arch_order(df: pd.DataFrame, arch_col: str) -> list[str]:
    preferred = ["Arps Pure", "Arps Ensemble", "PINN Analytical"]
    present = [a for a in preferred if a in set(df[arch_col].astype(str))]
    others = sorted(set(df[arch_col].astype(str)) - set(present))
    return present + others


def _build_cmap(df: pd.DataFrame, cols: _StoryboardColumns, theme: Dict[str, Any]) -> Dict[str, str]:
    order = _arch_order(df, cols.architecture)
    return {
        a: theme["arch_colors"].get(a, theme["fallback_colors"][i % len(theme["fallback_colors"])])
        for i, a in enumerate(order)
    }


def _safe_int_str(v: Any) -> str:
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)


def _add_badge(ax: plt.Axes, label: str, theme: Dict[str, Any]) -> None:
    ax.text(
        theme["layout"]["badge_x"], theme["layout"]["badge_y"], label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=theme["fonts"]["badge"], fontweight="bold", color=theme["text"],
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=theme["grid"], lw=0.9, alpha=0.95),
        zorder=20, clip_on=False
    )


def _add_note(
    ax: plt.Axes,
    text: str,
    theme: Dict[str, Any],
    *,
    loc: str = "top-right",
    fontsize: Optional[float] = None,
    alpha: float = 0.93,
) -> None:
    # posições explícitas (com folga interna)
    pos = {
        "top-right": (0.985, 0.94, "right", "top"),
        "bottom-right": (0.985, 0.04, "right", "bottom"),
        "top-left": (0.015, 0.94, "left", "top"),
        "bottom-left": (0.015, 0.04, "left", "bottom"),
    }
    x, y, ha, va = pos[loc]
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=fontsize or theme["fonts"]["note_plot1"],
        color=theme["text"], fontweight="500",
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=theme["grid"], lw=0.8, alpha=alpha),
        zorder=20,
    )


def _style_axes(ax: plt.Axes, theme: Dict[str, Any]) -> None:
    ax.set_facecolor(theme["bg"])
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.85, alpha=0.85)
    ax.grid(True, which="minor", color=theme["grid"], linewidth=0.45, alpha=0.35)
    ax.minorticks_on()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(theme["grid"])
        ax.spines[spine].set_linewidth(0.9)

    ax.tick_params(colors=theme["text"])
    ax.tick_params(axis="x", labelsize=theme["fonts"]["tick"])
    ax.tick_params(axis="y", labelsize=theme["fonts"]["tick"])

    for obj in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        obj.set_color(theme["text"])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(_metric_fmt))
    ax.margins(x=0.02)


def _prepare_data(df: pd.DataFrame, cols: _StoryboardColumns) -> pd.DataFrame:
    df = df.copy()

    for c in [cols.test_size, cols.val_size, cols.hpo_trials, cols.rate_test_mean, cols.cum_test_mean]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Extração vetorizada de IQR
    for c in [cols.rate_test_iqr, cols.cum_test_iqr]:
        if c in df.columns:
            ext = (
                df[c].astype(str)
                .str.extract(r"([+-]?\d+\.?\d*)\s*[–-]\s*([+-]?\d+\.?\d*)")
                .astype(float)
            )
            df[f"{c}_low"], df[f"{c}_high"] = ext[0], ext[1]

    if cols.test_size in df.columns:
        df["_split_label"] = df[cols.test_size].apply(lambda x: f"Split {x:.2f}" if pd.notna(x) else "Split ?")
    else:
        df["_split_label"] = "Split ?"

    return df


# =============================================================================
# Plot 1: Split Severity Curves
# =============================================================================
def _plot_split_severity(
    df: pd.DataFrame,
    cols: _StoryboardColumns,
    theme: Dict[str, Any],
    cmap: Dict[str, str],
    title_prefix: str,
) -> plt.Figure:
    metrics = [
        (cols.rate_test_mean, "Production Rate (Test) — Mean Error"),
        (cols.cum_test_mean, "Cumulative Production (Test) — Mean Error"),
    ]

    long_parts = []
    for col, name in metrics:
        if col in df.columns:
            tmp = df[[cols.test_size, cols.architecture, cols.hpo_trials, col]].copy()
            tmp = tmp.rename(columns={col: "val"})
            tmp["metric"] = name
            long_parts.append(tmp)

    if not long_parts:
        raise ValueError("No test mean columns found for Plot 1.")

    long_df = pd.concat(long_parts, ignore_index=True).dropna(subset=[cols.test_size, "val"])
    agg = (
        long_df.groupby([cols.test_size, cols.architecture, "metric"])["val"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    arch_order = _arch_order(df, cols.architecture)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=theme["layout"]["plot1_figsize"],
        dpi=theme["layout"]["dpi"],
        facecolor="white",
    )

    # Mais espaço embaixo para a legenda em caixa
    plt.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.24, wspace=0.16)

    panel_title_fs = max(theme["fonts"].get("panel_title", 12), 13)
    axis_label_fs = max(theme["fonts"].get("axis_label", 11), 12)
    tick_fs = max(theme["fonts"].get("tick", 10), 11)
    legend_fs = max(theme["fonts"].get("legend_plot1", 10), 11)

    for i, (ax, (_, m_label)) in enumerate(zip(axes, metrics)):
        m_df = long_df[long_df["metric"] == m_label].copy()
        m_agg = agg[agg["metric"] == m_label].copy()

        _style_axes(ax, theme)
        ax.set_facecolor("white")

        if m_df.empty:
            ax.set_title(m_label, fontsize=panel_title_fs, fontweight="bold", pad=10)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        # jitter horizontal determinístico por (arquitetura, split)
        g = m_df.groupby([cols.architecture, cols.test_size], dropna=False)[cols.test_size]
        m_df["x_jit"] = (
            m_df[cols.test_size]
            + (((g.cumcount() / (g.transform("size") - 1).replace(0, np.nan)) - 0.5).fillna(0.0) * 0.01)
        )

        for arch in arch_order:
            sub_agg = m_agg[m_agg[cols.architecture].astype(str) == str(arch)].sort_values(cols.test_size)
            sub_raw = m_df[m_df[cols.architecture].astype(str) == str(arch)]
            c = cmap.get(arch, "#999999")

            if not sub_agg.empty:
                ax.fill_between(
                    sub_agg[cols.test_size],
                    sub_agg["min"],
                    sub_agg["max"],
                    color=c,
                    alpha=0.07,   # um pouco mais suave
                    lw=0,
                    zorder=1,
                )

            if not sub_raw.empty:
                ax.scatter(
                    sub_raw["x_jit"],
                    sub_raw["val"],
                    s=48,
                    color=c,
                    alpha=0.24,   # reduz poluição visual
                    lw=0,
                    zorder=2,
                )

            if not sub_agg.empty:
                line, = ax.plot(
                    sub_agg[cols.test_size],
                    sub_agg["mean"],
                    color=c,
                    lw=2.8,
                    marker="o",
                    ms=6.8,
                    mfc=c,
                    mec="white",
                    mew=1.2,
                    zorder=4,
                    solid_capstyle="round",
                )
                line.set_path_effects([
                    pe.Stroke(linewidth=4.2, foreground="white", alpha=0.9),
                    pe.Normal(),
                ])

        ax.set_title(m_label, fontsize=panel_title_fs, fontweight="bold", pad=8)

        # improvement: xlabel redundante removido de cada painel;
        # usamos um supxlabel global na figura
        ax.set_xlabel("")

        # improvement: ylabel só no painel esquerdo
        if i == 0:
            ax.set_ylabel(
                "Error (lower is better)",
                fontsize=axis_label_fs,
                fontweight="bold",
            )
        else:
            ax.set_ylabel("")

        xticks = sorted(m_df[cols.test_size].dropna().unique())
        ax.set_xticks(xticks)
        ax.tick_params(axis="both", labelsize=tick_fs)

        # grade mais editorial: horizontal sutil
        ax.grid(True, axis="y", color=theme["grid"], alpha=0.25, lw=0.8)
        ax.grid(False, axis="x")

        # remove moldura excessiva
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # pequeno respiro vertical automático
        y_min = float(m_df["val"].min())
        y_max = float(m_df["val"].max())
        y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.05
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        _add_badge(ax, ["A", "B"][i], theme)

        left_note = f"{len(m_df)} campaign points\n{m_df[cols.test_size].nunique()} split levels"
        _add_note(
            ax,
            left_note,
            theme,
            loc=theme["layout"]["plot1_left_note_loc"],
            fontsize=theme["fonts"]["note_plot1"],
        )
        _add_note(
            ax,
            "more severe split →",
            theme,
            loc=theme["layout"]["plot1_right_note_loc"],
            fontsize=theme["fonts"]["note_plot1"],
        )

    legend_handles = [
        Line2D(
            [0], [0],
            color=cmap[a],
            marker="o",
            mfc=cmap[a],
            mec="white",
            mew=1.2,
            lw=2.8,
            ms=6.4,
            label=a,
        )
        for a in arch_order if a in cmap
    ]
    legend_handles += [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            mfc="#8a8a8a",
            alpha=0.30,
            ms=7,
            label="Campaign result (cloud)",
        ),
        Line2D(
            [0], [0],
            color="#8a8a8a",
            lw=2.8,
            marker="o",
            mec="white",
            mfc="#8a8a8a",
            label="Mean across wells",
        ),
    ]

    # Legenda inferior em caixa
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=min(len(legend_handles), max(3, theme["layout"].get("plot1_legend_ncol", 4))),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor=theme["grid"],
        facecolor="white",
        fontsize=legend_fs,
        borderpad=0.8,
        labelspacing=0.9,
        handlelength=2.2,
        handletextpad=0.8,
        columnspacing=1.4,
    )

    fig.suptitle(
        f"{title_prefix} — Results Across all Splits and HPO Trials",
        fontsize=theme["fonts"]["suptitle"],
        fontweight="black",
        color=theme["text"],
        y=0.965,
    )

    fig.supxlabel(
        "Test Split Severity (test_size)",
        fontsize=axis_label_fs,
        fontweight="bold",
        color=theme["text"],
        y=0.12,
    )

    return fig


# =============================================================================
# Plot 2: IQR Distribution
# =============================================================================
def _plot_iqr_distribution(
    df: pd.DataFrame,
    cols: _StoryboardColumns,
    theme: Dict[str, Any],
    cmap: Dict[str, str],
    title_prefix: str,
) -> plt.Figure:
    m_col, iqr_col = cols.cum_test_mean, cols.cum_test_iqr
    low_col, high_col = f"{iqr_col}_low", f"{iqr_col}_high"

    required = [cols.test_size, cols.hpo_trials, cols.architecture, m_col, low_col, high_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for Plot 2: {missing}")

    plot_df = df.dropna(subset=required).copy()
    if plot_df.empty:
        raise ValueError("No valid rows for Plot 2 after filtering.")

    splits = sorted(plot_df[cols.test_size].unique())
    if len(splits) == 0:
        raise ValueError("No split values found for Plot 2.")

    # versão horizontal 1×4; se houver menos splits, usa os disponíveis
    n_panels = len(splits)
    arch_order = _arch_order(df, cols.architecture)
    base_y = {arch: i * 1.10 for i, arch in enumerate(arch_order)}

    # range comum útil para todos os painéis
    x_min = float(plot_df[[m_col, low_col, high_col]].min().min())
    x_max = float(plot_df[[m_col, low_col, high_col]].max().max())
    x_span = x_max - x_min
    x_pad = x_span * 0.04 if x_span > 0 else 0.25

    # opcional: range mínimo comum mais justo
    x_lo = x_min - x_pad
    x_hi = x_max + x_pad

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(11.5, 2.55 * n_panels), theme["layout"]["plot2_fig_row_height"] * 0.95),
        dpi=theme["layout"]["dpi"],
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # compactação estrutural: pouco espaço entre colunas
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.82,
        bottom=0.23,
        wspace=0.035,
    )

    tick_fs = max(theme["fonts"].get("tick", 11.0), 11.0)
    ylab_fs = max(theme["fonts"].get("tick", 11.0) + 1.2, 12.0)
    split_fs = max(theme["fonts"].get("split_title", 11.0), 11.0)
    legend_fs = max(theme["fonts"].get("legend_plot2", 10.5), 10.8)

    for i, (ax, sp_val) in enumerate(zip(axes, splits)):
        sub = plot_df[plot_df[cols.test_size] == sp_val].copy()

        _style_axes(ax, theme)
        ax.set_facecolor("white")

        # grade só no eixo x e bem suave
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=theme["grid"], alpha=0.22, lw=0.75)

        for arch in arch_order:
            sub_a = sub[sub[cols.architecture].astype(str) == str(arch)].sort_values(cols.hpo_trials)
            if sub_a.empty:
                continue

            c = cmap.get(arch, "#999999")

            # fundo por arquitetura preservado, mas leve
            ax.axhspan(
                base_y[arch] - 0.30,
                base_y[arch] + 0.30,
                color=c,
                alpha=0.030,
                zorder=0,
            )

            offsets = np.linspace(-0.18, 0.18, len(sub_a)) if len(sub_a) > 1 else np.array([0.0])

            for (_, row), off in zip(sub_a.iterrows(), offsets):
                y = base_y[arch] + float(off)
                q1 = float(row[low_col])
                q3 = float(row[high_col])
                m = float(row[m_col])

                # IQR um pouco mais espesso
                ax.add_patch(
                    Rectangle(
                        (q1, y - 0.115),
                        max(q3 - q1, 1e-9),
                        0.23,
                        fc=c,
                        ec=c,
                        alpha=0.22,
                        lw=1.35,
                        zorder=2,
                    )
                )
                ax.plot([q1, q3], [y, y], color=c, lw=2.35, alpha=0.98, zorder=3)
                ax.plot([q1, q1], [y - 0.085, y + 0.085], color=c, lw=1.55, zorder=3)
                ax.plot([q3, q3], [y - 0.085, y + 0.085], color=c, lw=1.55, zorder=3)

                # mean maior
                ax.scatter(m, y, s=92, color=c, edgecolor="white", lw=1.2, zorder=4)

        ax.set_xlim(x_lo, x_hi)

        # títulos curtos no topo
        ax.set_title(
            f"{sp_val:.2f}",
            fontsize=split_fs,
            fontweight="bold",
            pad=6,
            color=theme["text"],
        )

        ax.set_yticks(list(base_y.values()))
        if i == 0:
            ax.set_yticklabels(arch_order, fontsize=ylab_fs, fontweight="600")
            ax.tick_params(axis="y", length=0, pad=6)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        ax.tick_params(axis="x", labelsize=tick_fs)

        # eixo x compartilhado: mantém ticks em todos, mas sem rótulo repetido
        # se quiser deixar ainda mais seco, mostrar labels só nos 2 painéis centrais
        ax.invert_yaxis()

        _add_badge(ax, chr(ord("A") + i), theme)

        # moldura mais limpa
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # legenda única, compacta, em uma linha
    iqr_gray = "#6f6f6f"
    legend_handles = [
        Line2D(
            [0], [0],
            color=cmap[a],
            lw=2.3,
            marker="o",
            mfc=cmap[a],
            mec="white",
            ms=6.8,
            label=a,
        )
        for a in arch_order if a in cmap
    ]
    legend_handles += [
        Patch(fc=iqr_gray, ec=iqr_gray, alpha=0.26, label="Empirical IQR"),
        Line2D([0], [0], marker="o", color="w", mfc="#666666", ms=7.4, label="Mean test error"),
    ]

    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.82,
        bottom=0.27,   # antes 0.23
        wspace=0.035,
    )
    
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),   # antes 0.06
        ncol=len(legend_handles),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor=theme["grid"],
        facecolor="white",
        fontsize=legend_fs,
        borderpad=0.42,
        labelspacing=0.45,
        handlelength=1.55,
        handletextpad=0.42,
        columnspacing=0.82,
    )

    fig.suptitle(
        "Predictive Spread by Split Severity",
        fontsize=theme["fonts"]["suptitle"],
        fontweight="black",
        color=theme["text"],
        y=0.95,
    )

    fig.supxlabel(
        "Cumulative Test Metric",
        fontsize=max(theme["fonts"].get("axis_label", 12.0), 12.5),
        fontweight="bold",
        color=theme["text"],
        y=0.125,
    )

    return fig


def _plot_iqr_distribution(
    df: pd.DataFrame,
    cols: _StoryboardColumns,
    theme: Dict[str, Any],
    cmap: Dict[str, str],
    title_prefix: str,
) -> plt.Figure:
    """
    Disruptive editorial alternative to the split-wise IQR panel plot.

    Visual grammar:
    - y-axis: split severity levels
    - x-axis: test metric (lower is better)
    - each architecture: one trajectory across split severities
    - at each split: horizontal thick bar = typical empirical IQR
    - point = typical center (median mean test error across runs)
    - direct labels at the trajectory end replace the legend

    Notes:
    - intended for full-width figure usage
    - designed to emphasize spread containment and drift under harsher splits
    """

    m_col, iqr_col = cols.cum_test_mean, cols.cum_test_iqr
    low_col, high_col = f"{iqr_col}_low", f"{iqr_col}_high"

    required = [cols.test_size, cols.architecture, cols.hpo_trials, m_col, low_col, high_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for spread trajectory map: {missing}")

    plot_df = df.dropna(subset=required).copy()
    if plot_df.empty:
        raise ValueError("No valid rows for spread trajectory map after filtering.")

    splits = sorted(plot_df[cols.test_size].dropna().unique())
    arch_order = _arch_order(df, cols.architecture)

    # ---------------------------------------------------------
    # Aggregate to one representative glyph per (architecture, split)
    # We summarize across rows/runs with robust medians.
    # ---------------------------------------------------------
    rows = []
    for arch in arch_order:
        for sp in splits:
            g = plot_df[
                (plot_df[cols.architecture].astype(str) == str(arch))
                & (plot_df[cols.test_size] == sp)
            ].copy()
            if g.empty:
                continue

            center = float(np.median(g[m_col]))
            q1 = float(np.median(g[low_col]))
            q3 = float(np.median(g[high_col]))

            if q1 > q3:
                q1, q3 = q3, q1

            # keep interval coherent with center if needed
            q1 = min(q1, center)
            q3 = max(q3, center)

            rows.append(
                {
                    "architecture": str(arch),
                    "split": float(sp),
                    "center": center,
                    "q1": q1,
                    "q3": q3,
                    "n": int(len(g)),
                }
            )

    agg = pd.DataFrame(rows)
    if agg.empty:
        raise ValueError("No aggregated rows available for spread trajectory map.")

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------
    split_to_y = {sp: i for i, sp in enumerate(splits)}
    n_arch = max(len(arch_order), 1)

    # small within-row offsets so trajectories do not overlap
    if n_arch == 1:
        arch_offsets = {arch_order[0]: 0.0}
    else:
        offsets = np.linspace(-0.22, 0.22, n_arch)
        arch_offsets = {arch: float(off) for arch, off in zip(arch_order, offsets)}

    agg["y"] = agg.apply(lambda r: split_to_y[r["split"]] + arch_offsets[r["architecture"]], axis=1)

    x_min = float(agg["q1"].min())
    x_max = float(agg["q3"].max())
    x_span = x_max - x_min
    x_pad_l = 0.05 * x_span if x_span > 0 else 0.25
    x_pad_r = 0.18 * x_span if x_span > 0 else 0.45  # extra room for direct labels

    fig, ax = plt.subplots(
        1,
        1,
        figsize=theme["layout"].get("plot2_figsize_disruptive", (11.8, 5.6)),
        dpi=theme["layout"]["dpi"],
        facecolor="white",
    )

    fig.subplots_adjust(left=0.12, right=0.92, top=0.86, bottom=0.15)

    _style_axes(ax, theme)
    ax.set_facecolor("white")

    # ---------------------------------------------------------
    # Background structure: split bands + separators
    # ---------------------------------------------------------
    for sp in splits:
        y0 = split_to_y[sp]
        ax.axhspan(y0 - 0.50, y0 + 0.50, color=theme["grid"], alpha=0.035, zorder=0)
        ax.axhline(y0, color=theme["grid"], lw=0.8, alpha=0.25, zorder=0)

    # ---------------------------------------------------------
    # Draw each architecture trajectory
    # ---------------------------------------------------------
    for arch in arch_order:
        sub = agg[agg["architecture"] == str(arch)].sort_values("split").copy()
        if sub.empty:
            continue

        c = cmap.get(arch, "#999999")

        # subtle architecture lane tint
        for _, r in sub.iterrows():
            ax.axhspan(
                r["y"] - 0.16,
                r["y"] + 0.16,
                color=c,
                alpha=0.035,
                zorder=0,
            )

        # connecting trajectory through centers
        ax.plot(
            sub["center"],
            sub["y"],
            linestyle="None",
        )


        # IQR bars + center points
        for _, r in sub.iterrows():
            ax.plot(
                [r["q1"], r["q3"]],
                [r["y"], r["y"]],
                color=c,
                lw=7.4,                      # thick editorial bar
                alpha=0.22,
                solid_capstyle="round",
                zorder=3,
            )
            ax.plot(
                [r["q1"], r["q3"]],
                [r["y"], r["y"]],
                color=c,
                lw=2.6,
                alpha=0.98,
                solid_capstyle="round",
                zorder=4,
            )
            ax.scatter(
                r["center"],
                r["y"],
                s=95,
                color=c,
                edgecolor="white",
                lw=1.25,
                zorder=5,
            )

        # direct label at the most severe split end
        last = sub.sort_values("split").iloc[-1]
        ax.annotate(
            arch,
            xy=(last["q3"], last["y"]),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=max(theme["fonts"].get("legend_plot2", 11.0), 11.5),
            fontweight="bold",
            color=c,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="white",
                ec="none",
                alpha=0.92,
            ),
            zorder=6,
            clip_on=False,
        )

    # ---------------------------------------------------------
    # Axes + labels
    # ---------------------------------------------------------
    ax.set_xlim(x_min - x_pad_l, x_max + x_pad_r)
    ax.set_ylim(-0.6, len(splits) - 1 + 0.6)
    ax.invert_yaxis()

    ax.set_yticks([split_to_y[sp] for sp in splits])
    ax.set_yticklabels([f"{sp:.2f}" for sp in splits], fontsize=max(theme["fonts"].get("tick", 11), 11))
    ax.set_ylabel(
        "Split severity",
        fontsize=max(theme["fonts"].get("axis_label", 12), 12),
        fontweight="bold",
    )

    ax.set_xlabel(
        "Cumulative test error (lower is better)",
        fontsize=max(theme["fonts"].get("axis_label", 12), 12.5),
        fontweight="bold",
    )

    ax.tick_params(axis="x", labelsize=max(theme["fonts"].get("tick", 11), 11))
    ax.tick_params(axis="y", length=0)

    # vertical grid only
    ax.grid(True, axis="x", color=theme["grid"], alpha=0.22, lw=0.8)
    ax.grid(False, axis="y")

    # clean frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------------------------------
    # Minimal headline + small subtitle
    # ---------------------------------------------------------
    fig.suptitle(
        f"Predictive Spread by Split Severity",
        fontsize=theme["fonts"]["suptitle"],
        fontweight="black",
        color=theme["text"],
        y=0.97,
    )

    ax.set_title(
        "Points = typical mean test error; horizontal bars = typical empirical IQR across runs",
        fontsize=max(theme["fonts"].get("note_plot2", 10.5), 10.5),
        color=theme["text"],
        pad=10,
    )

    return fig

# =============================================================================
# Main Entry Point (Plug-and-Play)
# =============================================================================
def plot_final_suite_storyboard(
    table_story: pd.DataFrame,
    title_prefix: str = "Final Suite Results",
    *,
    # ===== fontes (expostas para testes) =====
    note_fontsize_plot1: float = 12.5,
    note_fontsize_plot2: float = 12.5,
    badge_fontsize: float = 14.0,
    trial_label_fontsize: float = 13.0,
    split_title_fontsize: float = 16.0,
    axis_label_fontsize: float = 13.0,
    tick_fontsize: float = 11.5,
    panel_title_fontsize: float = 14.0,
    suptitle_fontsize: float = 17.0,
    legend_fontsize_plot1: float = 10.8,
    legend_fontsize_plot2: float = 10.8,
    # ===== tamanhos/layout =====
    plot1_figsize: tuple = (14, 6.2),
    plot2_fig_width: float = 14.0,
    plot2_fig_row_height: float = 4.8,
    dpi: int = 120,
    split_title_x: float = 0.14,
    plot1_legend_ncol: int = 3,
    plot2_legend_ncol: int = 3,
    plot1_left_note_loc: str = "top-left",
    plot1_right_note_loc: str = "bottom-right",
    plot2_note_loc: str = "bottom-right",
    # ===== controle =====
    show: bool = True,
) -> Dict[str, Any]:
    cols = _StoryboardColumns()
    theme = _get_theme(
        note_fontsize_plot1=note_fontsize_plot1,
        note_fontsize_plot2=note_fontsize_plot2,
        badge_fontsize=badge_fontsize,
        trial_label_fontsize=trial_label_fontsize,
        split_title_fontsize=split_title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_fontsize=tick_fontsize,
        panel_title_fontsize=panel_title_fontsize,
        suptitle_fontsize=suptitle_fontsize,
        legend_fontsize_plot1=legend_fontsize_plot1,
        legend_fontsize_plot2=legend_fontsize_plot2,
        split_title_x=split_title_x,
        plot1_figsize=plot1_figsize,
        plot2_fig_width=plot2_fig_width,
        plot2_fig_row_height=plot2_fig_row_height,
        dpi=dpi,
        plot1_legend_ncol=plot1_legend_ncol,
        plot2_legend_ncol=plot2_legend_ncol,
        plot1_left_note_loc=plot1_left_note_loc,
        plot1_right_note_loc=plot1_right_note_loc,
        plot2_note_loc=plot2_note_loc,
    )

    df = _prepare_data(table_story, cols)
    cmap = _build_cmap(df, cols, theme)

    fig1 = _plot_split_severity(df, cols, theme, cmap, title_prefix)
    if show:
        plt.show()

    fig2 = _plot_iqr_distribution(df, cols, theme, cmap, title_prefix)
    if show:
        plt.show()

    return {
        "fig_split_severity": fig1,
        "fig_iqr_distribution": fig2,
        "data_enriched": df,
        "theme": theme,
        "color_map": cmap,
    }


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter, MaxNLocator


def plot_unified_story_winner_dashboard(
    table2_unified_story: pd.DataFrame,
    *,
    palette: str = "default",
    title_prefix: str = "Final Suite Results",

    # ---- modo ----
    mode: str = "test",                # "test" | "test_val"

    # ---- colunas ----
    test_size_col: str = "test_size",
    arch_col: str = "Architecture",

    cum_test_col: str = "Cumulative Production (Test) (MEAN)",
    cum_val_col: str = "Cumulative Production (Validation) (MEAN)",
    rate_test_col: str = "Production Rate (Test) (MEAN)",
    rate_val_col: str = "Production Rate (Validation) (MEAN)",

    # ---- comportamento ----
    reduce_mode: str = "min",
    xlim_mode: str = "per_split",
    show_cloud: bool = True,
    cloud_mode: str | None = None,     # None => auto
    show: bool = True,

    # ---- layout ----
    ncols: int = 2,
    fig_width: float = 15.8,
    row_height: float = 5.0,
    dpi: int = 120,

    left: float = 0.07,
    right: float = 0.985,
    top: float = 0.86,
    bottom: float = 0.24,
    wspace: float = 0.18,
    hspace: float = 0.22,

    # ---- fontes ----
    suptitle_fs: float = 18.0,
    panel_title_fs: float = 18.0,
    badge_fs: float = 14.0,
    xtick_fs: float = 11.5,
    note_fs: float = 12.5,
    value_fs: float = 11.8,

    # ---- geometria ----
    arch_gap: float = 1.90,
    lane_gap: float = 0.22,
    band_h: float = 0.78,
    stem_alpha: float = 0.14,
    x_pad_frac: float = 0.12,
    label_dx: float = 7.0,

    # ---- rodapé: 2 cards ----
    footer_height: float = 0.16,       # altura total do rodapé
    footer_y: float = 0.02,            # posição Y do rodapé
    footer_gap: float = 0.012,         # espaço entre cards
    legend_card_frac: float = 0.45,    # fração do rodapé para a legenda (resto é winners)
    legend_fs: float = 13.0,
    legend_title_fs: float = 13.0,
    legend_ncol: int = 3,
    winners_title_fs: float = 13.0,
    winners_fs: float = 12.5,
) -> dict:
    """
    Dashboard por split + rodapé em 2 cards:
      (1) Legenda (esquerda)
      (2) Resumo Winners por split (direita)

    Requer: _get_color_palette(palette) existir no projeto.
    """
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if mode not in {"test", "test_val"}:
        raise ValueError("mode must be 'test' or 'test_val'")
    if reduce_mode not in {"min", "mean"}:
        raise ValueError("reduce_mode must be 'min' or 'mean'")
    if xlim_mode not in {"per_split", "global"}:
        raise ValueError("xlim_mode must be 'per_split' or 'global'")

    # cloud_mode auto
    if cloud_mode is None:
        cloud_mode = "test_only" if mode == "test" else "all"
    if cloud_mode not in {"test_only", "all"}:
        raise ValueError("cloud_mode must be 'test_only' or 'all'")

    # --- drop IQR ---
    drop_iqr = [
        "Production Rate (Validation) (IQR)",
        "Cumulative Production (Validation) (IQR)",
        "Production Rate (Test) (IQR)",
        "Cumulative Production (Test) (IQR)",
    ]
    df = table2_unified_story.copy(deep=True).drop(columns=drop_iqr, errors="ignore")

    required = [test_size_col, arch_col, cum_test_col, rate_test_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # numeric
    for c in [test_size_col, cum_test_col, cum_val_col, rate_test_col, rate_val_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[test_size_col, arch_col, cum_test_col, rate_test_col]).copy()
    if df.empty:
        raise ValueError("No valid rows after numeric cleaning.")

    # --- theme ---
    base = _get_color_palette(palette)
    theme = {
        "bg": base.get("plot_bgcolor", "white"),
        "paper_bg": base.get("paper_bgcolor", "white"),
        "grid": base.get("grid", "#E6E6E6"),
        "text": base.get("text", "#2C3E50"),
    }

    arch_colors = {
        "Arps Pure": base.get("validation", "#E53935"),
        "Arps Ensemble": base.get("test_rolling", "#4CAF50"),
        "PINN Analytical": base.get("actual_post_train", "#0277BD"),
    }
    fallback = [
        base.get("train_from_x", "#3949AB"),
        base.get("train_from_y", "#0277BD"),
        base.get("test_initial", "#4CAF50"),
        "#8E24AA", "#F4511E", "#00897B",
    ]

    preferred = ["Arps Pure", "Arps Ensemble", "PINN Analytical"]
    present = [a for a in preferred if a in set(df[arch_col].astype(str))]
    others = sorted(set(df[arch_col].astype(str)) - set(present))
    arch_order = present + others

    color_map = {}
    it = iter(fallback)
    for a in arch_order:
        color_map[a] = arch_colors.get(a, next(it, "#777777"))

    # --- reduce ---
    def _reduce(s: pd.Series) -> float:
        s = s.dropna()
        if s.empty:
            return np.nan
        return float(s.min()) if reduce_mode == "min" else float(s.mean())

    group_cols = [test_size_col, arch_col]
    agg = (
        df.groupby(group_cols, dropna=False)
        .apply(lambda g: pd.Series({
            "_cum_test": _reduce(g[cum_test_col]),
            "_cum_val": _reduce(g[cum_val_col]) if (mode == "test_val" and cum_val_col in g.columns) else np.nan,
            "_rate_test": _reduce(g[rate_test_col]),
            "_rate_val": _reduce(g[rate_val_col]) if (mode == "test_val" and rate_val_col in g.columns) else np.nan,
        }))
        .reset_index()
    )

    # winners per split (TEST only)
    win_c_idx = agg.groupby(test_size_col)["_cum_test"].idxmin()
    win_r_idx = agg.groupby(test_size_col)["_rate_test"].idxmin()
    winners = (
        pd.DataFrame({
            test_size_col: agg.loc[win_c_idx, test_size_col].values,
            "best_cumulative_test": agg.loc[win_c_idx, arch_col].values,
        })
        .merge(
            pd.DataFrame({
                test_size_col: agg.loc[win_r_idx, test_size_col].values,
                "best_rate_test": agg.loc[win_r_idx, arch_col].values,
            }),
            on=test_size_col, how="outer"
        )
        .sort_values(test_size_col)
        .reset_index(drop=True)
    )

    # --- axis formatting ---
    splits = sorted(df[test_size_col].dropna().unique())
    ns = len(splits)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(ns / ncols))

    def _fmt(x, _pos):
        ax = abs(x)
        if ax >= 1000: return f"{x:,.0f}"
        if ax >= 100: return f"{x:,.1f}"
        return f"{x:,.2f}"

    def _style_ax(ax):
        ax.set_facecolor(theme["bg"])
        ax.set_axisbelow(True)
        ax.grid(True, which="major", axis="x", color=theme["grid"], linewidth=0.85, alpha=0.85)
        ax.grid(True, which="minor", axis="x", color=theme["grid"], linewidth=0.45, alpha=0.30)
        ax.minorticks_on()
        ax.grid(False, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(theme["grid"])
        ax.spines["bottom"].set_color(theme["grid"])
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.tick_params(colors=theme["text"])
        ax.tick_params(axis="x", labelsize=xtick_fs)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.margins(x=0.02)

    def _xlim_for(sub_raw: pd.DataFrame):
        cols_for_xlim = [cum_test_col, rate_test_col]
        if mode == "test_val":
            if cum_val_col in sub_raw.columns: cols_for_xlim.append(cum_val_col)
            if rate_val_col in sub_raw.columns: cols_for_xlim.append(rate_val_col)
        xs = pd.concat([sub_raw[c].dropna() for c in cols_for_xlim if c in sub_raw.columns], axis=0)
        xmin, xmax = float(xs.min()), float(xs.max())
        pad = (xmax - xmin) * x_pad_frac if xmax > xmin else 1.0
        return (xmin - pad, xmax + pad)

    xlim_global = _xlim_for(df) if xlim_mode == "global" else None

    # lanes
    y_base = {a: i * arch_gap for i, a in enumerate(arch_order)}
    if mode == "test":
        lane_offsets = {"cum_test": -0.5 * lane_gap, "rate_test": +0.5 * lane_gap}
        lane_spec = {"cum_test": dict(marker="o", filled=True), "rate_test": dict(marker="s", filled=True)}
    else:
        lane_offsets = {
            "cum_test": -1.5 * lane_gap, "cum_val": -0.5 * lane_gap,
            "rate_test": +0.5 * lane_gap, "rate_val": +1.5 * lane_gap,
        }
        lane_spec = {
            "cum_test": dict(marker="o", filled=True),
            "cum_val": dict(marker="o", filled=False),
            "rate_test": dict(marker="s", filled=True),
            "rate_val": dict(marker="s", filled=False),
        }
    lane_to_col = {"cum_test": cum_test_col, "cum_val": cum_val_col, "rate_test": rate_test_col, "rate_val": rate_val_col}

    # --- figure ---
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, row_height * nrows),
        dpi=dpi,
        facecolor=theme["paper_bg"],
        sharey=True,
        sharex=False,
    )
    axes = np.array([axes]).flatten() if not isinstance(axes, np.ndarray) else axes.flatten()
    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

    badges = [chr(ord("A") + i) for i in range(26)]

    for i, (ax, sp) in enumerate(zip(axes, splits)):
        _style_ax(ax)
        sub_raw = df[df[test_size_col] == sp].copy()
        sub_agg = agg[agg[test_size_col] == sp].copy()

        xlim = _xlim_for(sub_raw) if xlim_mode == "per_split" else xlim_global
        ax.set_xlim(*xlim)

        # title
        ax.text(0.012, 1.02, badges[i], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=badge_fs, fontweight="bold", color=theme["text"],
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=theme["grid"], lw=0.9, alpha=0.95),
                zorder=20, clip_on=False)
        ax.text(0.10, 1.02, f"Split {sp:.2f}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=panel_title_fs, fontweight="black", color=theme["text"], zorder=20, clip_on=False)

        # bands
        for a in arch_order:
            y0 = y_base[a]
            c = color_map.get(a, "#777777")
            ax.axhspan(y0 - band_h, y0 + band_h, color=c, alpha=0.030, zorder=0)

        # cloud
        if show_cloud:
            for a in arch_order:
                c = color_map.get(a, "#777777")
                y0 = y_base[a]
                sub_a = sub_raw[sub_raw[arch_col].astype(str) == str(a)]
                if sub_a.empty:
                    continue
                n = len(sub_a)
                offs = np.linspace(-0.08, 0.08, n) if n > 1 else np.array([0.0])
                lanes = list(lane_spec.keys()) if cloud_mode == "all" else [k for k in lane_spec.keys() if k.endswith("test")]
                for lane in lanes:
                    col = lane_to_col[lane]
                    if col not in sub_a.columns:
                        continue
                    sub_vals = sub_a.dropna(subset=[col])
                    if sub_vals.empty:
                        continue
                    offs2 = offs if len(sub_vals) == n else (np.linspace(-0.08, 0.08, len(sub_vals)) if len(sub_vals) > 1 else np.array([0.0]))
                    yy = y0 + lane_offsets[lane] + offs2
                    spec = lane_spec[lane]
                    if spec["filled"]:
                        ax.scatter(sub_vals[col], yy, s=20, color=c, alpha=0.16, lw=0, marker=spec["marker"], zorder=2)
                    else:
                        ax.scatter(sub_vals[col], yy, s=18, facecolors="none", edgecolors=c, linewidths=1.0,
                                   alpha=0.14, marker=spec["marker"], zorder=2)

        # aggregated
        for a in arch_order:
            row = sub_agg[sub_agg[arch_col].astype(str) == str(a)]
            if row.empty:
                continue
            c = color_map.get(a, "#777777")
            y0 = y_base[a]
            for lane, spec in lane_spec.items():
                key = f"_{lane}"
                if key not in row.columns:
                    continue
                x = row[key].iloc[0]
                if pd.isna(x):
                    continue
                x = float(x)
                y = y0 + lane_offsets[lane]
                ax.plot([xlim[0], x], [y, y], color=c, alpha=stem_alpha, lw=6.0, solid_capstyle="round", zorder=1)

                if spec["filled"]:
                    m = ax.scatter([x], [y], s=92, color=c, edgecolor="white", lw=1.2, marker=spec["marker"], zorder=4)
                    m.set_path_effects([pe.Stroke(linewidth=2.6, foreground="white", alpha=0.9), pe.Normal()])
                else:
                    ax.scatter([x], [y], s=84, facecolors="white", edgecolors=c, lw=1.6, marker=spec["marker"], zorder=4)

                ax.annotate(
                    f"{x:.2f}",
                    (x, y), textcoords="offset points", xytext=(label_dx, 0),
                    ha="left", va="center",
                    fontsize=value_fs, color=theme["text"], fontweight="600",
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=theme["grid"], lw=0.7, alpha=0.95),
                    zorder=6,
                )

        # note
        ax.text(
            0.985, 0.94, "lower is better ←",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=note_fs,
            color=theme["text"],
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=theme["grid"], lw=0.8, alpha=0.93),
            zorder=20,
        )

        # remove y labels
        ax.set_yticks([y_base[a] for a in arch_order])
        ax.set_yticklabels([""] * len(arch_order))
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()

        if i // ncols == nrows - 1:
            ax.set_xlabel("Mean error (lower is better)", fontsize=13.0, fontweight="bold", color=theme["text"])

    for ax in axes[len(splits):]:
        ax.axis("off")

    # --- title ---
    fig.suptitle(
        f"{title_prefix} — Best Architecture per Split (Test Mean)",
        fontsize=suptitle_fs,
        fontweight="black",
        color=theme["text"],
        y=0.975,
    )

    # ---------------------------------------------------------------------
    # Footer cards: Legend (left) + Winners summary (right)
    # ---------------------------------------------------------------------
    footer_w = (right - left)
    gap = footer_gap * footer_w
    legend_w = footer_w * legend_card_frac - gap / 2
    winners_w = footer_w * (1 - legend_card_frac) - gap / 2

    legend_ax = fig.add_axes([left, footer_y, legend_w, footer_height])
    winners_ax = fig.add_axes([left + legend_w + gap, footer_y, winners_w, footer_height])

    for axc in (legend_ax, winners_ax):
        axc.set_facecolor("white")
        axc.set_xticks([]); axc.set_yticks([])
        for spine in axc.spines.values():
            spine.set_visible(True)
            spine.set_color(theme["grid"])
            spine.set_linewidth(0.9)

    # ---- Legend card ----
    arch_handles = [
        Line2D([0], [0], color=color_map[a], lw=4.0, marker="o", markersize=8,
               markerfacecolor=color_map[a], markeredgecolor="white", markeredgewidth=1.0, label=a)
        for a in arch_order
    ]
    sem_handles = []
    if show_cloud:
        sem_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666",
                                  alpha=0.18, markersize=9, label="Cloud"))

    sem_handles += [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="white",
               markeredgewidth=1.0, markersize=9, label="Cum • Test"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#666666", markeredgecolor="white",
               markeredgewidth=1.0, markersize=9, label="Rate • Test"),
    ]
    if mode == "test_val":
        sem_handles += [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#666666",
                   markeredgewidth=1.4, markersize=9, label="Cum • Val"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="#666666",
                   markeredgewidth=1.4, markersize=9, label="Rate • Val"),
        ]

    legend_ax.legend(
        handles=arch_handles + sem_handles,
        loc="center",
        ncol=min(legend_ncol, len(arch_handles + sem_handles)),
        frameon=False,
        fontsize=legend_fs,
        title=f"Legend (mode={mode})",
        title_fontsize=legend_title_fs,
        handletextpad=0.8,
        labelspacing=0.9,
        columnspacing=1.4,
    )

    # ---- Winners summary card ----
    winners_ax.text(
        0.5, 0.88, "Winners by split (Test)",
        ha="center", va="center",
        transform=winners_ax.transAxes,
        fontsize=winners_title_fs, fontweight="bold", color=theme["text"]
    )

    # Render as compact rows
    # Format: Split 0.40 | Cum: PINN | Rate: Arps Ensemble
    y0 = 0.6
    dy = 0.22 if len(winners) <= 3 else 0.18
    for i, row in winners.iterrows():
        sp = row[test_size_col]
        a_c = str(row["best_cumulative_test"])
        a_r = str(row["best_rate_test"])

        c_c = color_map.get(a_c, theme["text"])
        c_r = color_map.get(a_r, theme["text"])

        winners_ax.text(0.05, y0, f"Split {sp:.2f}", ha="left", va="center",
                        transform=winners_ax.transAxes, fontsize=winners_fs, fontweight="bold", color=theme["text"])

        winners_ax.text(0.35, y0, "Cum:", ha="right", va="center",
                        transform=winners_ax.transAxes, fontsize=winners_fs, color=theme["text"])
        winners_ax.text(0.37, y0, a_c, ha="left", va="center",
                        transform=winners_ax.transAxes, fontsize=winners_fs, fontweight="bold", color=c_c)

        winners_ax.text(0.65, y0, "Rate:", ha="right", va="center",
                        transform=winners_ax.transAxes, fontsize=winners_fs, color=theme["text"])
        winners_ax.text(0.67, y0, a_r, ha="left", va="center",
                        transform=winners_ax.transAxes, fontsize=winners_fs, fontweight="bold", color=c_r)

        y0 -= dy

    if show:
        plt.show()

    return {
        "fig": fig,
        "data_clean": df,
        "data_agg": agg,
        "winners": winners,
        "theme": theme,
        "color_map": color_map,
        "mode": mode,
    }