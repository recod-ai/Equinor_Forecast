"""forecast_pipeline.plotting
------------------------------------------------
Camada de visualização unificada.

Funções:
  * plot_series – traça verdade + cenário(s) usando Plotly.
  * plot_predictions_wrapper – adaptação para o pipeline atual.
"""
from __future__ import annotations

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
    envelope_color: str = "rgba(255,99,132,0.2)",   # envelope vermelho-claro
    samples: Optional[np.ndarray] = None,           # (N,B,H)
    q_phys: Optional[np.ndarray] = None,
    res: Optional[np.ndarray] = None,
    # --- métricas / extras ------------------------------------------
    smape: float | None = None,
    mae: float | None = None,
    window_size: int | None = None,
    forecast_steps: int | None = None,
    percentage_split: float | None = None,
    # visual ---------------------------------------------------------
    title: str = "Forecast",
    well: str = "",
    width: int = 1200,
    height: int = 600,
):
    """Plota verdade, previsão, envelope + métricas."""
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

    # linhas principais
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

    # fan-chart amostral
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

    fig.update_layout(
        title=dict(text=f"{title}", x=0.5, font=dict(size=36)),
        xaxis_title="Steps",
        yaxis_title="Rate",
        plot_bgcolor="white",
        width=width,
        height=height,
        legend=dict(orientation="h", x=0.5, y=1.1, xanchor="center"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    # anotações de métricas
    ann = []
    if smape is not None: ann.append((f"SMAPE: {smape:.2f}%", "#206A92"))
    if mae is not None:   ann.append((f"MAE: {mae:.2f}", "yellowgreen"))
    if window_size is not None:
        ann.append((f"Windows: {window_size}", "#2E2E2E"))
    if forecast_steps is not None:
        ann.append((f"Steps: {forecast_steps}", "#2E2E2E"))
    if percentage_split is not None:
        ann.append((f"Train: {percentage_split*100:.0f}%", "#2E2E2E"))

    for i, (text, color) in enumerate(ann):
        fig.add_annotation(
            x=0.02, y=0.98 - i * 0.10,
            xref="paper", yref="paper",
            text=text,
            showarrow=False,
            font=dict(size=24, color=color),
            bgcolor="rgba(255,255,255,0.8)",
        )

    fig.update_xaxes(title_font=dict(size=26), tickfont=dict(size=22))
    fig.update_yaxes(title_font=dict(size=26), tickfont=dict(size=22))

    fig.show()



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

        fig.show()


    







