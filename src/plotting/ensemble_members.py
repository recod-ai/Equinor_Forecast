# src/plotting/ensemble_members.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .ensemble_utils import (
    _get_color_palette,
    _with_alpha,
    add_ground_truth,
    add_region_backgrounds,
    by_well,
    coalesce_time_column,
    ensure_time_axis,
    hover_y,
    only_val_test,
    safe_int,
)

# Reuse layout & backgrounds from your existing plot module
from .ensemble_plots import (
    polish_layout,
)


log = logging.getLogger(__name__)


# ---------------------------
# Public helper (plug & play)
# ---------------------------
# src/plotting/ensemble_members.py

from common.phase_viz_support import normalize_well_key


def build_members_frame_for_well(
    series_df: pd.DataFrame,
    well: str,
    *,
    member_id_col: str = "job_hash",
    yhat_col: str = "yhat",
) -> pd.DataFrame:
    """
    Extrai o frame por membro para um poço, de forma tolerante e alias-aware:
      - 'split' é opcional (preenche 'test' se ausente)
      - aceita 'idx' OU 'global_idx' OU 't' (pelo menos um)
      - casamento de well por normalize_well_key (p.ex. '15/9-F-14' ≡ '15-9-F-14')
    Retorna um DataFrame com colunas: ['well','split','idx?','global_idx?','t?','arch?', member_id_col, yhat_col]
    """
    if not isinstance(series_df, pd.DataFrame) or series_df.empty:
        return pd.DataFrame(columns=["well", "split", "t", member_id_col, yhat_col])

    cols = set(series_df.columns)
    must_have = {"well", member_id_col, yhat_col}
    if not must_have.issubset(cols):
        return pd.DataFrame(columns=["well", "split", "t", member_id_col, yhat_col])

    has_idx         = "idx" in cols
    has_gidx        = "global_idx" in cols
    has_t           = "t" in cols
    has_time_anchor = has_idx or has_gidx or has_t
    if not has_time_anchor:
        return pd.DataFrame(columns=["well", "split", "t", member_id_col, yhat_col])

    # -------------------------------
    # 🔑 Filtro por well (alias-aware)
    # -------------------------------
    target_key = normalize_well_key(well)
    sub = series_df.copy()

    mask = sub["well"].astype(str).map(lambda w: normalize_well_key(w) == target_key)
    sub = sub[mask]
    if sub.empty:
        return pd.DataFrame(columns=["well", "split", "t", member_id_col, yhat_col])

    # split opcional → default 'test'
    if "split" not in sub.columns:
        sub["split"] = "test"
    else:
        sub["split"] = sub["split"].astype(str)

    # Se t e idx forem numéricos e idênticos, descarta t redundante
    if has_t and has_idx:
        try:
            if pd.api.types.is_numeric_dtype(sub["t"]) and pd.api.types.is_numeric_dtype(sub["idx"]):
                if (sub["t"] == sub["idx"]).all():
                    sub = sub.drop(columns=["t"])
                    has_t = False
        except Exception:
            pass

    keep_order = ["well", "split", "idx", "global_idx", "t", "arch", member_id_col, yhat_col]
    keep_cols = [c for c in keep_order if c in sub.columns]
    out = sub[keep_cols].copy()

    return out


# ---------------------------
# Main plot
# ---------------------------

def plot_ensemble_members_by_family(
    *,
    series_df: pd.DataFrame,
    final_ensemble_df: pd.DataFrame,
    intra_family_df: pd.DataFrame = pd.DataFrame(),
    full_history_df: pd.DataFrame,
    boundaries: Dict[str, Any],
    well: str,
    arch_key: str,
    palette: str = "default",
    title: Optional[str] = None,
    font_scale: float = 1.2,
    width: int = 1100,
    height: int = 700,
    max_members: int = 60,
    show: bool = True,
) -> go.Figure:
    """
    Spaghetti de MEMBERS (apenas Val/Test) + Family Mean / Final Mean.
    Agora alias-aware para 'well'.
    """
    if full_history_df is None or full_history_df.empty:
        raise ValueError(f"[members] full_history_df is empty for well='{well}'.")
    if not isinstance(series_df, pd.DataFrame) or series_df.empty:
        raise ValueError("[members] series_df is empty; cannot draw members.")
    if "yhat" not in series_df.columns:
        raise KeyError("[members] series_df must contain 'yhat' for member traces.")

    colors = _get_color_palette(palette)
    fh = full_history_df.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)

    # ---- Offsets iguais ao Stage 4 (índice → eixo t) ----
    t_axis   = fh["t"].reset_index(drop=True)
    n_hist   = len(t_axis)
    tr_end_i = safe_int(boundaries.get("train_end", 0),            0, n_hist - 1)
    va_end_i = safe_int(boundaries.get("val_end", tr_end_i),       0, n_hist - 1)
    test_st  = safe_int(boundaries.get("test_start", va_end_i+1),  0, n_hist)
    VAL_OFFSET  = tr_end_i + 1
    TEST_OFFSET = max(va_end_i, test_st) + 1

    t_map_global = fh.reset_index().rename(columns={"index": "global_idx"})[["global_idx", "t"]]

    # ------------------------------------------------------------------
    # 🔑 Slice membros: alias-aware entre requested 'well' e coluna 'well'
    # ------------------------------------------------------------------
    s = series_df.copy()
    if "well" in s.columns:
        target_key = normalize_well_key(well)
        mask = s["well"].astype(str).map(lambda w: normalize_well_key(w) == target_key)
        s = s[mask]

    if "arch" in s.columns:
        s = s[s["arch"].astype(str).str.lower().eq(str(arch_key).lower())]

    # Normaliza split (sempre presente) e filtra VAL/TEST quando aplicável
    if "split" in s.columns:
        s["split"] = s["split"].astype(str).str.lower()
        s = s[s["split"].isin(["val", "validation", "test"])]
    else:
        s["split"] = "test"

    # Se existir 't' numérico igual a 'idx', descarta 't'
    if "t" in s.columns and "idx" in s.columns:
        try:
            if pd.api.types.is_numeric_dtype(s["t"]) and pd.api.types.is_numeric_dtype(s["idx"]) and (s["t"] == s["idx"]).all():
                s = s.drop(columns=["t"])
        except Exception:
            pass

    # Constrói global_idx via offsets
    if "global_idx" not in s.columns and "idx" in s.columns:
        split_ser = s["split"].astype(str).str.lower()
        idx_int = s["idx"].astype(int)
        s["global_idx"] = np.where(
            split_ser.isin(["val", "validation"]), idx_int + VAL_OFFSET,
            np.where(split_ser.eq("test"),         idx_int + TEST_OFFSET, idx_int)
        )

    if "global_idx" in s.columns:
        t_map_global = fh.reset_index().rename(columns={"index": "global_idx"})[["global_idx", "t"]]
        s = s.merge(t_map_global, on="global_idx", how="inner")

    s = ensure_time_axis(s, "members_by_family", fh)
    s = coalesce_time_column(s, "members_by_family")
    s = only_val_test(s, "members_by_family")

    # Cap de membros p/ responsividade
    if "job_hash" in s.columns:
        uniq = s["job_hash"].astype(str).unique().tolist()
        if len(uniq) > max_members:
            keep = set(uniq[:max_members])
            s = s[s["job_hash"].astype(str).isin(keep)]

    # ---- Family mean (preferida) ou fallback para final mean ----
    fam = pd.DataFrame()
    if isinstance(intra_family_df, pd.DataFrame) and not intra_family_df.empty:
        fam = intra_family_df.copy()
        if "well" in fam.columns:
            fam = fam[fam["well"].astype(str) == str(well)]
        if "arch" in fam.columns:
            fam = fam[fam["arch"].astype(str).str.lower().eq(str(arch_key).lower())]
        fam = ensure_time_axis(fam, "intra_family_df (members view)", fh)
        fam = coalesce_time_column(fam, "intra_family_df (members view)")
        fam = only_val_test(fam, "intra_family_df (members view)")

    inter = by_well(final_ensemble_df, well, "final_ensemble_df (members view)")
    inter = ensure_time_axis(inter, "final_ensemble_df (members view)", fh)
    inter = coalesce_time_column(inter, "final_ensemble_df (members view)")
    inter = only_val_test(inter, "final_ensemble_df (members view)")

    # ---- Figura & fundos ----
    fig = go.Figure()
    add_region_backgrounds(fig, colors, t_axis, tr_end_i, va_end_i)

    title_fs = 18 * font_scale
    fig.add_annotation(text="<b>Train</b>", x=t_axis.iloc[:tr_end_i + 1].median(),
                       yref="paper", y=0.98, showarrow=False,
                       font=dict(size=title_fs, color=colors.get("text", "#111")))
    fig.add_annotation(text="<b>Validation</b>", x=t_axis.iloc[tr_end_i:va_end_i + 1].median(),
                       yref="paper", y=0.98, showarrow=False,
                       font=dict(size=title_fs, color=colors.get("validation", "#d62728")))
    fig.add_annotation(text="<b>Test</b>", x=t_axis.iloc[va_end_i:].median(),
                       yref="paper", y=0.98, showarrow=False,
                       font=dict(size=title_fs, color=colors.get("test_border", colors.get("test_rolling", "#2ca02c"))))

    add_ground_truth(fig, fh, colors)

    # ---- Members (tons claros por split) ----
    if not s.empty and "t" in s.columns:
        member_val_color  = _with_alpha(colors.get("validation",   "#d62728"), 0.35)
        member_test_color = _with_alpha(colors.get("test_rolling", "#2ca02c"), 0.35)

        n_unique = int(s["job_hash"].nunique()) if "job_hash" in s.columns else None
        legend_title = f"Members{f' (n={n_unique})' if n_unique else ''}"

        seen_val, seen_test = False, False
        group_cols = [c for c in ["job_hash", "split"] if c in s.columns] or ["split"]
        for _, g in s.groupby(group_cols, sort=False):
            if g.empty or "t" not in g.columns:
                continue
            g = g.sort_values("t")
            split_name = str(g["split"].iloc[0]) if "split" in g.columns else "test"
            is_test = "test" in split_name.lower()
            col = member_test_color if is_test else member_val_color

            show_leg = False
            leg_name = None
            if is_test and not seen_test:
                show_leg, seen_test = True, True
                leg_name = "Member (Test)"
            if (not is_test) and (not seen_val):
                show_leg, seen_val = True, True
                leg_name = "Member (Val)"

            fig.add_trace(go.Scatter(
                x=g["t"], y=g["yhat"], mode="lines",
                name=(leg_name if leg_name else "Member"),
                line=dict(width=1.0, color=col, dash="dash"),
                hovertemplate=hover_y(),
                showlegend=show_leg,
                legendgroup="Members",
                legendgrouptitle_text=legend_title
            ))

    # ---- Family mean (preferida) OU Final mean (fallback) ----
    if not fam.empty and {"t", "yhat_family_mean", "split"}.issubset(fam.columns):
        for split_name, g in fam.groupby("split", sort=False):
            g = g.sort_values("t")
            is_test = "test" in str(split_name).lower()
            line_c  = colors.get("test_initial", "#2ca02c") if is_test else colors.get("validation", "#d62728")

            fig.add_trace(go.Scatter(
                x=g["t"], y=g["yhat_family_mean"], mode="lines",
                name=f"Family Mean ({'Test' if is_test else 'Val'})",
                line=dict(width=4, color=line_c),
                hovertemplate=hover_y(),
            ))
            if is_test and not g.empty:
                last_t = g["t"].iloc[-1]; last_y = g["yhat_family_mean"].iloc[-1]
                fig.add_trace(go.Scatter(
                    x=[last_t], y=[last_y], mode="markers",
                    name="Last Test Mean",
                    marker=dict(size=9, line=dict(width=0.5, color="#333")),
                    hovertemplate="<b>Last Test</b><br>t=%{x}<br>mean=%{y:.2f}<extra></extra>",
                    showlegend=False
                ))
    elif not inter.empty and {"t", "yhat_final_mean", "split"}.issubset(inter.columns):
        for split_name, g in inter.groupby("split", sort=False):
            g = g.sort_values("t")
            is_test = "test" in str(split_name).lower()
            line_c  = colors.get("test_initial", "#2ca02c") if is_test else colors.get("validation", "#d62728")

            fig.add_trace(go.Scatter(
                x=g["t"], y=g["yhat_final_mean"], mode="lines",
                name=f"Ensemble Mean ({'Test' if is_test else 'Val'})",
                line=dict(width=4, color=line_c),
                hovertemplate=hover_y(),
            ))
            if is_test and not g.empty:
                last_t = g["t"].iloc[-1]; last_y = g["yhat_final_mean"].iloc[-1]
                fig.add_trace(go.Scatter(
                    x=[last_t], y=[last_y], mode="markers",
                    name="Last Test Mean",
                    marker=dict(size=9, line=dict(width=0.5, color="#333")),
                    hovertemplate="<b>Last Test</b><br>t=%{x}<br>mean=%{y:.2f}<extra></extra>",
                    showlegend=False
                ))

    # ---- Layout ----
    family_label = "PINN" if str(arch_key).lower().startswith("seq2") else "ARPS"
    base_title = title or f"<b>Ensemble Members — {well}</b>"
    base_title = base_title.replace("</b>", f" — {family_label}</b>", 1)

    polish_layout(fig, colors, base_title, font_scale, width, height,
                  yaxis_log=False, y_tickformat=",~s")

    if show:
        fig.show(config={"toImageButtonOptions": {"format": "png", "filename": "ensemble_members", "scale": 3}})
    log.info("[members] Rendered well=%s arch=%s", well, arch_key)
    return fig
