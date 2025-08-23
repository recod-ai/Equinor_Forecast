"""forecast_pipeline.plotting
------------------------------------------------
Camada de visualização unificada.

Funções:
  * plot_series – traça verdade + cenário(s) usando Plotly.
  * plot_predictions_wrapper – adaptação para o pipeline atual.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Mapping, Sequence
import numpy as np

# --- Optional imports for plotting backends ---
try:
    import plotly.io as pio
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    pio = None
    go = None
    make_subplots = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.offsetbox import AnchoredText
except Exception:  # pragma: no cover
    plt = None
    GridSpec = None
    AnchoredText = None

from typing import Optional
import numpy as np
import plotly.graph_objects as go

from forecast_pipeline.analytics import scenario_curve, make_envelope
from forecast_pipeline.ensemble_output import EnsembleOutput

__all__ = [
    "plot_series",
    "plot_predictions_wrapper",
]

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
    smape: float | None = None,
    mae: float | None = None,
    window_size: int | None = None,
    forecast_steps: int | None = None,
    percentage_split: float | None = None,
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

    # Layout and legend (legend moved down and enlarged)
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
            y=-0.25,  # pushed further down
            xanchor="center",
            font=dict(size=20)  # larger font
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    metrics_lines = []
    # Metrics annotations (bottom-right corner, ~20% from bottom)
    if smape is not None:
        metrics_lines.append(f"<span style='color:#206A92'>SMAPE: {smape:.2f}%</span>")
    if mae is not None:
        metrics_lines.append(f"<span style='color:yellowgreen'>MAE: {mae:.2f}</span>")
    if forecast_steps is not None:
        metrics_lines.append(f"<span style='color:#2E2E2E'>Horizon: {forecast_steps}</span>")
    if percentage_split is not None:
        metrics_lines.append(f"<span style='color:#2E2E2E'>Train: {percentage_split*100:.0f}%</span>")
    if window_size is not None:
        metrics_lines.append(f"<span style='color:#2E2E2E'>Windows: {window_size}</span>")
    
    # Join all lines into one HTML-formatted string with line breaks
    metrics_text = "<br>".join(metrics_lines)

    # Draw annotations stacked upwards from bottom-right
    fig.add_annotation(
        x=0.8,
        y=0.5,  # starts 20% from bottom, stacks up
        xref="paper",
        yref="paper",
        text=metrics_text,
        showarrow=False,
        font=dict(size=22),
        bgcolor="rgba(255,255,255,0.85)",
        xanchor="left",
        align="left"
    )

    fig.update_xaxes(title_font=dict(size=26), tickfont=dict(size=22))
    fig.update_yaxes(title_font=dict(size=26), tickfont=dict(size=22))

    fig.show(renderer="png") 




# ------------------------------------------------------------------
# Wrapper: decide cenário e chama plot_series ------------------------
# ------------------------------------------------------------------
import logging

def plot_predictions_wrapper(
    ensemble: EnsembleOutput,
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
    is_cum = None,
    **extra_plot_kwargs,             # smape, mae, window_size, forecast_steps, …
):
    """
    Decide qual cenário desenhar e invoca `plot_series`.

    Args
    ----
    ensemble : EnsembleOutput
        Saída agregada do ensemble.
    truth : np.ndarray
        Série ground-truth (mesma dimensão da curva que será plotada).
    kind : str
        "P50", "P90", "P10", "BAND" (envelope), ou "MEAN".
    mean_override : np.ndarray | None
        Se fornecido, usa esta curva como central (útil para séries reconstruídas).
    **extra_plot_kwargs
        Parâmetros extras encaminhados para `plot_series`
        (ex.: smape=…, mae=…, window_size=…).
    """
    kind = kind.upper()

    # Curva central — pode vir de override
    central = np.asarray(mean_override).ravel() if mean_override is not None \
              else ensemble.pred_test.ravel()

    lower = upper = None

    if kind in ("P50", "MEAN"):
        mean_curve = central
        # ➊ ribbon quando band pedido e σ existe
        if band is not None:
            if ensemble.sigma_test is not None:
                lower, upper = make_envelope(
                    ensemble.pred_test,
                    ensemble.sigma_test,
                    *(band or (0.10, 0.90))
                )
            else:
                logging.info("Band solicitada, mas sigma indisponível – ribbon omitido")



    elif kind == "P90":
        if ensemble.sigma_test is not None:
            mean_curve = scenario_curve(ensemble.pred_test, ensemble.sigma_test, 0.90)
        else:                                      # sem σ compatível
            mean_curve = central                   # degrada para central

    elif kind == "P10":
        if ensemble.sigma_test is not None:
            mean_curve = scenario_curve(ensemble.pred_test, ensemble.sigma_test, 0.10)
        else:
            mean_curve = central

    elif kind == "BAND":
        if band is None:
            band = (0.10, 0.90)
        if ensemble.sigma_test is not None:
            lower, upper = make_envelope(ensemble.pred_test, ensemble.sigma_test, *band)
        mean_curve = central
    else:
        raise ValueError(f"Unknown kind '{kind}'")

    # ------------------------------------------------------------------
    # Se vier envelope pronto (cumulativo), usa-o e ignora make_envelope
    # ------------------------------------------------------------------
    if manual_envelope is not None:
        lower, upper = manual_envelope         # sobrescreve (pode ser None,None)

    


    # -------------------------------------------------
    # desscale central / envelopes / componentes
    # -------------------------------------------------
    if scaler is not None and is_cum is False:
        def _inv(arr):
            return scaler.inverse_transform(arr.reshape(-1, 1)).ravel()
        if lower is not None and upper is not None:
            lower, upper = _inv(lower), _inv(upper)
        # componentes (se presentes)
        if show_components and ensemble.q_phys is not None:
            q_phys = _inv(ensemble.q_phys)
        else:
            q_phys = ensemble.q_phys
        if show_components and ensemble.res_test is not None:
            res = _inv(ensemble.res_test)
        else:
            res = ensemble.res_test
    else:
        q_phys = ensemble.q_phys
        res    = ensemble.res_test


    plot_series(
        truth=truth,
        mean_curve=mean_curve,
        lower=lower,
        upper=upper,
        q_phys=q_phys if show_components else None,
        res=res_test if show_components else None,
        title=title or f"Scenario {kind}",
        well=well,
        **extra_plot_kwargs,   # smape, mae, window_size, forecast_steps, …
    )

import plotly.graph_objects as go
import numpy as np
from typing import Dict, Optional


def plot_integrated_view(
    x_axis: np.ndarray,
    y_actual: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_test: np.ndarray,
    split_indices: Dict[str, int],
    metrics_val: Optional[Dict[str, float]],
    metrics_test: Optional[Dict[str, float]],
    title: str,
    yaxis_title: str = "Rate",
    well: str = "",
    font_scale: float = 1.2,
    show: bool = True,
    *,
    shade: Optional[Dict[str, Tuple[int, int]]] = None,
):
    """
    Integrated plot (Train/Val/Test).
    Se `shade` for fornecido, usa os intervalos dados para sombrear cada bloco
    (útil para incluir os H-1 pontos de overlap na borda de Val/Test).
    Caso contrário, usa os limites "clássicos" de split_indices.
    """
    import numpy as np
    import plotly.graph_objects as go

    # --- cores ---
    COLOR_ACTUAL = '#206A92'
    COLOR_VAL = '#ed6a5a'
    COLOR_TEST = 'yellowgreen'
    COLOR_TRAIN_FILL = "rgba(30, 60, 90, 0.05)"
    COLOR_VAL_FILL   = "rgba(237,106,90,0.05)"
    COLOR_TEST_FILL  = "rgba(100, 200, 120, 0.05)"  # sutil; transparente

    nmax = len(x_axis) - 1

    def _clamp(a: int) -> int:
        return max(0, min(int(a), nmax))

    def _pick_range(kind: str, default: Tuple[int, int]) -> Tuple[int, int]:
        if shade and kind in shade:
            x0, x1 = shade[kind]
            return _clamp(x0), _clamp(x1)
        return _clamp(default[0]), _clamp(default[1])

    fig = go.Figure()

    # --- ranges padrão (sem estender) ---
    train_default = (0, split_indices['train_end'])
    val_default   = (split_indices['train_end'], split_indices['val_end'])
    test_default  = (split_indices['val_end'], nmax)

    # --- aplica ranges (estendidos se shade for dado) ---
    tr_x0, tr_x1 = _pick_range("train", train_default)
    va_x0, va_x1 = _pick_range("val",   val_default)
    te_x0, te_x1 = _pick_range("test",  test_default)

    # --- áreas sombreadas ---
    fig.add_vrect(
        x0=tr_x0, x1=tr_x1,
        fillcolor=COLOR_TRAIN_FILL, layer="below", line_width=0,
        annotation_text="Train", annotation_position="top left",
        annotation_font_size=20 * font_scale, annotation_font_color="#223"
    )
    fig.add_vrect(
        x0=va_x0, x1=va_x1,
        fillcolor=COLOR_VAL_FILL, layer="below", line_width=0,
        annotation_text="Validation", annotation_position="top left",
        annotation_font_size=20 * font_scale, annotation_font_color="#b93d2e"
    )
    fig.add_vrect(
        x0=te_x0, x1=te_x1,
        fillcolor="rgba(255, 255, 255, 0)", layer="below", line_width=0,
        annotation_text="Test", annotation_position="top left",
        annotation_font_size=20 * font_scale, annotation_font_color="#33896a"
    )

    # --- séries ---
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_actual, mode='lines', name='Actual',
        line=dict(color=COLOR_ACTUAL, width=4),
        hovertemplate='Day: %{x}<br>Actual: %{y:,.2f}<extra></extra>',
        hoverlabel=dict(bgcolor='white', font_size=16*font_scale)
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_pred_val, mode='lines', name='Validation Prediction',
        line=dict(color=COLOR_VAL, width=3, dash='dash'),
        hovertemplate='Day: %{x}<br>Validation Pred: %{y:,.2f}<extra></extra>',
        hoverlabel=dict(bgcolor='white', font_size=16*font_scale)
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_pred_test, mode='lines', name='Test Prediction',
        line=dict(color=COLOR_TEST, width=3, dash='dash'),
        hovertemplate='Day: %{x}<br>Test Pred: %{y:,.2f}<extra></extra>',
        hoverlabel=dict(bgcolor='white', font_size=16*font_scale)
    ))

    # --- métricas ---
    y_anno = 0.83
    if metrics_val:
        text_val = "<br>".join([f"<b>{k}:</b> {v:.2f}" for k, v in metrics_val.items()])
        fig.add_annotation(
            text=f"<b>Validation</b><br>{text_val}",
            align='right', showarrow=False, xref='paper', yref='paper',
            x=0.98, y=y_anno, xanchor='right',
            bordercolor=COLOR_VAL, borderwidth=1,
            bgcolor="rgba(255,255,255,0.90)",
            font=dict(size=18*font_scale, color=COLOR_VAL)
        )
        y_anno -= 0.26
    if metrics_test:
        text_test = "<br>".join([f"<b>{k}:</b> {v:.2f}" for k, v in metrics_test.items()])
        fig.add_annotation(
            text=f"<b>Test</b><br>{text_test}",
            align='right', showarrow=False, xref='paper', yref='paper',
            x=0.98, y=y_anno, xanchor='right',
            bordercolor=COLOR_TEST, borderwidth=1,
            bgcolor="rgba(255,255,255,0.90)",
            font=dict(size=18*font_scale, color=COLOR_TEST)
        )

    # --- layout ---
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size: {19*font_scale}px; color: #555;'>Well: {well}</span>",
            x=0.5, y=0.97, font=dict(size=26*font_scale, family="Lato, Arial, sans-serif")
        ),
        xaxis_title="Days",
        yaxis_title=yaxis_title,
        xaxis=dict(title_font_size=20*font_scale, tickfont_size=16*font_scale,
                   showgrid=False, tickformat="d"),
        yaxis=dict(title_font_size=20*font_scale, tickfont_size=16*font_scale,
                   showgrid=False),
        legend=dict(
            x=0.5, y=-0.13, xanchor='center', yanchor='top',
            orientation='h', font=dict(size=17*font_scale),
            bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.12)', borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1600, height=800,
        margin=dict(t=120, r=220, b=80, l=80),
        hovermode='x unified'
    )

    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)

    if show:
        fig.show()
    return fig




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


    
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Mapping, Iterable

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

def plot_integrated_view(
    x_axis: np.ndarray,
    y_actual: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_test: np.ndarray,
    split_indices: Dict[str, int],
    metrics_val: Optional[Dict[str, float]],
    metrics_test: Optional[Dict[str, float]],
    title: str,
    yaxis_title: str = "Rate",
    well: str = "",
    horizon: Optional[int] = None, # New parameter for forecast horizon
    font_scale: float = 1.2,
    show: bool = True,
    *,
    shade: Optional[Dict[str, Tuple[int, int]]] = None,
):
    """
    Plots an integrated view (Train/Val/Test) with a professional visual style,
    highlighting the initial forecast horizon on the test set.
    """
    fig = go.Figure()

    nmax = len(x_axis) - 1

    def _clamp(a: int) -> int:
        return max(0, min(int(a), nmax))

    def _pick_range(kind: str, default: Tuple[int, int]) -> Tuple[int, int]:
        if shade and kind in shade:
            x0, x1 = shade[kind]
            return _clamp(x0), _clamp(x1)
        return _clamp(default[0]), _clamp(default[1])

    # --- Define data ranges ---
    train_default = (0, split_indices['train_end'])
    val_default   = (split_indices['train_end'], split_indices['val_end'])
    test_default  = (split_indices['val_end'], nmax)

    tr_x0, tr_x1 = _pick_range("train", train_default)
    va_x0, va_x1 = _pick_range("val",   val_default)
    te_x0, te_x1 = _pick_range("test",  test_default)

    # --- Shaded background areas ---
    fig.add_vrect(
        x0=tr_x0, x1=tr_x1, fillcolor=COLOR_TRAIN_FILL, layer="below", line_width=0,
        annotation_text="Train", annotation_position="top left",
        annotation_font_size=20 * font_scale, annotation_font_color="#223"
    )
    fig.add_vrect(
        x0=va_x0, x1=va_x1, fillcolor=COLOR_VAL_FILL, layer="below", line_width=0,
        annotation_text="Validation", annotation_position="top left",
        annotation_font_size=20 * font_scale, annotation_font_color=COLOR_VAL
    )
    fig.add_vrect(
        x0=te_x0, x1=te_x1, fillcolor="rgba(0,0,0,0)", layer="below", line_width=0,
        annotation_text="Test", annotation_position="top left",
        annotation_font_size=20 * font_scale, annotation_font_color=COLOR_TEST
    )

    # --- Data Traces ---
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_actual, mode='lines+markers', name='Actual',
        line=dict(color=COLOR_ACTUAL, width=3),
        marker=dict(symbol='circle', size=4, color=COLOR_ACTUAL, opacity=0.7),
        hovertemplate='Day: %{x}<br>Actual: %{y:,.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_pred_val, mode='lines', name='Validation Prediction',
        line=dict(color=COLOR_VAL, width=3, dash='dash'),
        hovertemplate='Day: %{x}<br>Validation Pred: %{y:,.2f}<extra></extra>',
    ))

    # --- Conditional Test Prediction Trace ---
    # If horizon is specified, split the test line into two colored parts
    if horizon and horizon > 0 and te_x0 < te_x1:
        horizon_split_point = min(te_x1, te_x0 + horizon)

        # Part 1: The initial forecast horizon (bright green)
        fig.add_trace(go.Scatter(
            x=x_axis[te_x0:horizon_split_point],
            y=y_pred_test[te_x0:horizon_split_point],
            mode='lines', name='Test Prediction',
            line=dict(color=COLOR_TEST, width=3, dash='dot'),
            hovertemplate='Day: %{x}<br>Test Pred: %{y:,.2f}<extra></extra>',
        ))
        # Part 2: The remainder of the test period (greenish-gray)
        if horizon_split_point < te_x1:
            fig.add_trace(go.Scatter(
                x=x_axis[horizon_split_point-1:te_x1+1], # Overlap by 1 point for continuous line
                y=y_pred_test[horizon_split_point-1:te_x1+1],
                mode='lines', name='Test Remainder', showlegend=False,
                line=dict(color=COLOR_TEST_REMAINDER, width=3, dash='dot'),
                hovertemplate='Day: %{x}<br>Test Pred: %{y:,.2f}<extra></extra>',
            ))
    else: # Default behavior: plot the entire test prediction in one color
        fig.add_trace(go.Scatter(
            x=x_axis, y=y_pred_test, mode='lines', name='Test Prediction',
            line=dict(color=COLOR_TEST, width=3, dash='dot'),
            hovertemplate='Day: %{x}<br>Test Pred: %{y:,.2f}<extra></extra>',
        ))

    # --- Metrics Annotations ---
    y_anno = 0.85
    if metrics_val:
        text_val = "<br>".join([f"<b>{k}:</b> {v:.2f}" for k, v in metrics_val.items()])
        fig.add_annotation(
            text=f"<b>Validation</b><br>{text_val}", align='right', showarrow=False,
            xref='paper', yref='paper', x=0.98, y=y_anno, xanchor='right',
            bordercolor=COLOR_VAL, borderwidth=2, bgcolor="rgba(255,255,255,0.92)",
            font=dict(size=18*font_scale, color=COLOR_TEXT, family=FONT_FAMILY)
        )
        y_anno -= 0.28
    if metrics_test:
        text_test = "<br>".join([f"<b>{k}:</b> {v:.2f}" for k, v in metrics_test.items()])
        fig.add_annotation(
            text=f"<b>Test</b><br>{text_test}", align='right', showarrow=False,
            xref='paper', yref='paper', x=0.98, y=y_anno, xanchor='right',
            bordercolor=COLOR_TEST, borderwidth=2, bgcolor="rgba(255,255,255,0.92)",
            font=dict(size=18*font_scale, color=COLOR_TEXT, family=FONT_FAMILY)
        )

    # --- Horizon Annotation / Footnote ---
    if horizon and horizon > 0:
        fig.add_annotation(
            text=f"<i>*The initial <b>{horizon}-day</b> forecast (green) shows the prediction from day one. "
                 f"The subsequent forecast (gray) simulates the model's performance over time.</i>",
            align='left', showarrow=False, xref='paper', yref='paper',
            x=0.01, y=-0.3, xanchor='left', yanchor='top',
            font=dict(size=14*font_scale, color=COLOR_TEXT, family=FONT_FAMILY)
        )

    # --- Layout and Theming ---
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size: {22*font_scale}px; color: #555;'>Well: {well}</span>",
            x=0.5, y=0.96, font=dict(size=30*font_scale, family=FONT_FAMILY, color=COLOR_TEXT)
        ),
        xaxis_title="Days", yaxis_title=yaxis_title,
        font=dict(family=FONT_FAMILY, color=COLOR_TEXT),
        xaxis=dict(
            title_font_size=22*font_scale, tickfont_size=18*font_scale,
            gridcolor=COLOR_GRID, zeroline=False, showgrid=False, showline=False
        ),
        yaxis=dict(
            title_font_size=22*font_scale, tickfont_size=18*font_scale,
            gridcolor=COLOR_GRID, zeroline=False, showgrid=False, showline=False
        ),
        legend=dict(
            x=0.5, y=-0.15, xanchor='center', yanchor='top',
            orientation='h', font=dict(size=20*font_scale),
            bgcolor='rgba(255,255,255,0.9)', bordercolor=COLOR_GRID, borderwidth=1
        ),
        plot_bgcolor='white', paper_bgcolor='white',
        width=1600, height=800,
        margin=dict(t=140, r=40, b=180, l=100), # Increased bottom margin for footnote
        hovermode='x unified',
        hoverlabel=dict(bgcolor='white', font_size=16*font_scale, font_family=FONT_FAMILY)
    )

    if show:
        fig.show(renderer="png")
    return fig

