# src/common/phase_viz_support.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Iterable, Callable  # ADICIONE Iterable, Callable
import logging
import pandas as pd
import sys
import numpy as np
from pathlib import Path
log = logging.getLogger("phase_viz")

# Wrapper para manter chamadas existentes sem quebrar
def warn(msg: str) -> None:
    log.warning(msg)


def _pick_group_dataset_for_well(series_df: pd.DataFrame, well: str) -> Tuple[Optional[str], Optional[str]]:
    sel = series_df[series_df["well"].astype(str) == str(well)] if "well" in series_df.columns else pd.DataFrame()
    group = None
    dataset = None
    if "group" in sel.columns and sel["group"].notna().any():
        m = sel["group"].mode(dropna=True)
        group = m.iloc[0] if len(m) else None
    if "dataset" in sel.columns and sel["dataset"].notna().any():
        m = sel["dataset"].mode(dropna=True)
        dataset = m.iloc[0] if len(m) else None
    return group, dataset


def _resolve_canonical_well(
    requested_well: str,
    wells: Iterable[str],
) -> Tuple[str, str]:
    """
    Dado um well solicitado (p.ex. '15/9-F-14') e uma lista de wells disponíveis
    (p.ex. ['15-9-F-14', 'P11', ...]), retorna:

        (canonical_well, norm_key)

    onde canonical_well é o que de fato aparece nos DataFrames (p.ex. '15-9-F-14'),
    e norm_key é a chave normalizada (p.ex. '159F14').

    Se não achar alias compatível, devolve o requested_well como canonical.
    """
    target_key = normalize_well_key(str(requested_well))
    canonical = None

    for w in wells:
        if normalize_well_key(str(w)) == target_key:
            canonical = str(w)
            break

    if canonical is None:
        canonical = str(requested_well)

    return canonical, target_key




def build_full_history(
    series_df: pd.DataFrame,
    well: str,
    *,
    full_history_by_well: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Busca histórico completo por aliases de poço (barra vs hífen, maiúsculas etc),
    usando normalize_well_key.
    """

    from common.log_utils import warn
    
    if not full_history_by_well:
        warn(f"[viz] full_history missing for well={well} (no map)")
        return pd.DataFrame(columns=["t", "ytrue"])

    target_key = normalize_well_key(well)

    # Encontra a chave compatível via normalização
    chosen_key = None
    for k in full_history_by_well.keys():
        if normalize_well_key(k) == target_key:
            chosen_key = k
            break

    if chosen_key is None:
        warn(f"[viz] full_history missing for well={well} (no alias match)")
        return pd.DataFrame(columns=["t", "ytrue"])

    df = full_history_by_well[chosen_key]
    if {"t", "ytrue"}.issubset(df.columns) and not df.empty:
        out = df[["t", "ytrue"]].copy()
        return out.sort_values("t")

    warn(f"[viz] full_history empty/invalid for well={well} (key={chosen_key})")
    return pd.DataFrame(columns=["t", "ytrue"])



def infer_boundaries(final_ensemble_df: pd.DataFrame,
                     series_df: pd.DataFrame,
                     well: str,
                     *,
                     boundaries_df: Optional[pd.DataFrame] = None,
                     full_history_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Política:
      1) Se houver linha no manifesto (boundaries_df), usa ela.
      2) Senão, infere por dados disponíveis:
         - train_end: max(idx) do split TRAIN, se houver; senão 0
         - val_end:   max(idx) do split VAL,   se houver; senão = train_end
         - test_start: min(idx) do split TEST, se houver; senão = val_end + 1
      3) Garante monotonia: train_end <= val_end < test_start
      4) Se soubermos o tamanho do histórico (full_history_df), aplica clamp para [0, n_hist-1]
    """
    log = logging.getLogger("phase_viz")

    def _to_int_safe(x, default=None):
        try:
            return int(x)
        except Exception:
            return default

    # Tamanho do histórico (se disponível)
    n_hist = None
    if isinstance(full_history_df, pd.DataFrame) and not full_history_df.empty:
        # full_history_df já está no eixo 't'; o plot faz clip contra n_hist-1
        # Mantemos consistência com a lógica do plot.
        try:
            n_hist = len(full_history_df.drop_duplicates(subset=["t"]))
        except Exception:
            n_hist = None

    # 1) Manifesto (se existir)
    if boundaries_df is not None and not boundaries_df.empty:
        group, dataset = _pick_group_dataset_for_well(series_df, well)
        m = (boundaries_df["well"] == well)

        if group is not None and "group" in boundaries_df.columns and boundaries_df["group"].notna().any():
            m &= (boundaries_df["group"] == group)
        if dataset is not None and "dataset" in boundaries_df.columns and boundaries_df["dataset"].notna().any():
            m &= (boundaries_df["dataset"] == dataset)

        rows = boundaries_df.loc[m]
        if not rows.empty:
            r = rows.iloc[-1]
            train_end = _to_int_safe(r.get("train_end"), 0)
            val_end   = _to_int_safe(r.get("val_end"), train_end)
            test_start= _to_int_safe(r.get("test_start"), (val_end if val_end is not None else 0) + 1)
            # normaliza
            if train_end is None: train_end = 0
            if val_end   is None: val_end   = train_end
            if test_start is None: test_start = val_end + 1

            # monotonia
            val_end = max(val_end, train_end)
            test_start = max(test_start, val_end + 1)

            # clamp
            if n_hist is not None and n_hist > 0:
                hi = n_hist - 1
                train_end = max(0, min(train_end, hi))
                val_end   = max(train_end, min(val_end, hi))
                # no plot fazemos clip de test_start para [0, hi]; mantemos igual
                test_start = max(0, min(test_start, hi))

            return dict(train_end=int(train_end), val_end=int(val_end), test_start=int(test_start))

    # 2) Fallback por inferência
    def _sel(df: pd.DataFrame, split_names: Iterable[str]) -> pd.DataFrame:
        if df is None or df.empty or "split" not in df.columns:
            return pd.DataFrame()
        return df[df["split"].astype(str).str.lower().isin([s.lower() for s in split_names])]

    # Preferimos índices de series_df (mais fieis ao espaço de idx)
    sdf_w = series_df[series_df["well"].astype(str) == str(well)] if ("well" in series_df.columns and not series_df.empty) else pd.DataFrame()
    fdf_w = final_ensemble_df[final_ensemble_df["well"].astype(str) == str(well)] if ("well" in final_ensemble_df.columns and not final_ensemble_df.empty) else pd.DataFrame()

    train_end = None
    val_end   = None
    test_start= None

    # TRAIN: max idx do TRAIN em series_df (ideal)
    tr = _sel(sdf_w, ["train"])
    if not tr.empty and "idx" in tr.columns:
        try:
            train_end = int(tr["idx"].max())
        except Exception:
            train_end = None

    # VAL: max idx do VAL (tenta series_df, senão final_ensemble_df)
    val_s = _sel(sdf_w, ["val", "validation"])
    if val_s.empty and not fdf_w.empty:
        val_s = _sel(fdf_w, ["val", "validation"])
    if not val_s.empty and "idx" in val_s.columns:
        try:
            val_end = int(val_s["idx"].max())
        except Exception:
            val_end = None

    # TEST: min idx do TEST (tenta series_df, senão final_ensemble_df)
    te = _sel(sdf_w, ["test"])
    if te.empty and not fdf_w.empty:
        te = _sel(fdf_w, ["test"])
    if not te.empty and "idx" in te.columns:
        try:
            test_start = int(te["idx"].min())
        except Exception:
            test_start = None

    # Defaults coerentes
    if train_end is None:
        train_end = 0
    if val_end is None:
        val_end = train_end
    if test_start is None:
        test_start = val_end + 1

    # Monotonia
    val_end = max(val_end, train_end)
    test_start = max(test_start, val_end + 1)

    # Clamp se soubermos n_hist
    if n_hist is not None and n_hist > 0:
        hi = n_hist - 1
        train_end = max(0, min(train_end, hi))
        val_end   = max(train_end, min(val_end, hi))
        test_start = max(0, min(test_start, hi))

    log.info("[viz] infer_boundaries fallback well=%s → train_end=%s, val_end=%s, test_start=%s (n_hist=%s)",
             well, train_end, val_end, test_start, n_hist)

    return dict(train_end=int(train_end), val_end=int(val_end), test_start=int(test_start))





# ---------------------------
# Path resolution
# ---------------------------
def _find_project_root(start_path: Path, marker: str = "src") -> Path:
    cur = start_path.resolve()
    while cur != cur.parent:
        if (cur / marker).is_dir():
            return cur
        cur = cur.parent
    raise FileNotFoundError(f"Could not find project root with marker '{marker}' from {start_path}")

def _resolve_paths(cfg: Phase4Config) -> Dict[str, Path]:
    root = cfg.project_root or _find_project_root(Path.cwd(), marker=cfg.src_marker)
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    series_root = (cfg.series_store_root if cfg.series_store_root is not None else (root / "series_store"))
    return {
        "project_root": root.resolve(),
        "src_path": src_path.resolve(),
        "experiment_root": (root / "src" / "experiment_configs").resolve(),
        "series_store_root": series_root.resolve(),
    }



# ---------------------------
# Leaderboard discovery (robust)
# ---------------------------

from common.log_utils import info, warn, log_block

def _discover_and_load_leaderboards(exp_root: Path, campaign_group: str) -> Optional[pd.DataFrame]:
    """
    Versão 'burra' e previsível:
      - Lê TODOS os leaderboard.csv em exp_root/<group>/results/**/leaderboard.csv,
        exceto os de validation (ex: results/*/validation*/leaderboard.csv).
      - Adiciona colunas de contexto a partir do path.
      - NÃO faz filtro por dataset/well aqui.
      - Quem filtra depois (por well/dataset) é a Phase 4 ou o consumidor.
    """
    results_root = exp_root / campaign_group / "results"
    if not results_root.is_dir():
        warn(f"[Stage 1] Results path not found: {results_root}")
        return None

    files = sorted(results_root.rglob("leaderboard.csv"))
    if not files:
        warn(f"[Stage 1] No leaderboards under: {results_root}")
        return None

    frames = []
    for f in files:
        try:
            rel = f.relative_to(results_root)
            parts = rel.parent.parts              # ex: ('seq2', 'VOLVE_15-9-F-14_Seq2PIN_cycle_1')
            arch_dir = parts[0] if parts else None
            campaign_dir = parts[-1] if parts else None
            campaign_dir_str = str(campaign_dir or "")

            # 🔴 NOVO: pular leaderboards de validation (ex: "validation_seq2")
            if campaign_dir_str.lower().startswith("validation"):
                info(
                    "[Stage 1] Skipping validation leaderboard '%s' (campaign_dir=%s)",
                    rel.as_posix(),
                    campaign_dir_str,
                )
                continue

            df = pd.read_csv(f).copy()

            # Contexto: coluna campaign / architecture
            if "campaign" not in df.columns:
                # aqui você mantém o que vier do CSV se já tiver 'campaign',
                # senão usa o nome do diretório como campaign
                df["campaign"] = campaign_dir_str

            if "architecture" not in df.columns and "arch" not in df.columns:
                # ex: 'seq2' / 'arps' ficam aqui; o select depois normaliza
                df["architecture"] = arch_dir

            frames.append(df)

            info(
                "[Stage 1] Loaded leaderboard '%s' rows=%d.",
                rel.as_posix(),
                len(df),
            )
        except Exception as e:
            warn(f"[Stage 1] Could not read {f}: {e}")

    if not frames:
        warn(f"[Stage 1] All leaderboards empty for group={campaign_group} (after validation filter).")
        return None

    master_df = pd.concat(frames, ignore_index=True)

    # Pequeno resumo estilo 'Leaderboard Population'
    try:
        cols = []
        if "well" in master_df.columns:
            cols.append("well")
        if "architecture" in master_df.columns:
            cols.append("architecture")
        elif "arch" in master_df.columns:
            cols.append("arch")
        if "campaign" in master_df.columns:
            cols.append("campaign")

        if cols:
            pop = (
                master_df.groupby(cols, as_index=False)
                         .size()
                         .rename(columns={"size": "n"})
            )
            lines = []
            for _, r in pop.sort_values(cols).iterrows():
                desc = ", ".join(f"{c}={r[c]}" for c in cols)
                lines.append(f"{desc} → {int(r['n'])}")
            if lines:
                lines.insert(0, "(counts are per " + " × ".join(cols) + ")")
                log_block("Stage 1 — Leaderboard Population (per well × arch × campaign)", lines)
    except Exception as e:
        warn(f"[Stage 1] population summary failed softly: {e}")


    # df_relicado = master_df.loc[master_df.index.repeat(5)].reset_index(drop=True)
    return master_df



# ---------------------------
# Fallbacks (safe defaults)
# ---------------------------
def fallback_full_history_from_series(series_df: pd.DataFrame, well: str) -> pd.DataFrame:
    """Build a minimal full-history frame (t,ytrue) from the raw series if needed."""
    if series_df is None or series_df.empty or "well" not in series_df.columns:
        return pd.DataFrame(columns=["t", "ytrue"])
    g = series_df.loc[series_df["well"].astype(str) == str(well)].copy()
    if g.empty or "ytrue" not in g.columns or g["ytrue"].isna().all():
        return pd.DataFrame(columns=["t", "ytrue"])
    if "t" not in g.columns:
        g["t"] = g.get("idx", pd.Series(range(len(g)), index=g.index))
    g = (
        g.loc[g["ytrue"].notna(), ["t", "ytrue"]]
        .drop_duplicates(subset=["t"])
        .sort_values("t")
    )
    return g


def fallback_boundaries(_: str, full_hist_df: pd.DataFrame) -> Dict[str, Any]:
    """Minimal boundaries when a manifest is not present."""
    if full_hist_df is None or full_hist_df.empty:
        return {"train_end": 0, "val_end": 0, "test_start": 1}
    n = len(full_hist_df)
    return {"train_end": n - 1, "val_end": n - 1, "test_start": n}


# ---------------------------
# Manifest lookup
# ---------------------------
def lookup_manifest_bounds(boundaries_df: pd.DataFrame, well: str) -> Optional[Dict[str, Any]]:
    """Look up train/val/test boundaries for a given well from a manifest dataframe."""
    if not isinstance(boundaries_df, pd.DataFrame) or boundaries_df.empty or "well" not in boundaries_df.columns:
        return None
    sub = boundaries_df[boundaries_df["well"].astype(str) == str(well)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    out = {k: r.get(k) for k in ("train_end", "val_end", "test_start")}
    if any(pd.isna(out[k]) for k in out):
        return None
    return out


# ---------------------------
# Intra-family filter
# ---------------------------
def filter_intra_by_arch(df: pd.DataFrame, arch_key: str) -> pd.DataFrame:
    """Return only rows for a normalized arch ('seq2', 'arps')."""
    if df is None or df.empty or "arch" not in df.columns:
        return pd.DataFrame()
    return df.loc[df["arch"].astype(str).str.lower().eq(arch_key)].copy()


# ---------------------------
# Debug helper
# ---------------------------
def debug_frame(tag: str, df: pd.DataFrame, well: str, max_rows: int = 3) -> None:
    if df is None or df.empty:
        logging.info("[PhaseViz][dbg] %s well=%s: EMPTY", tag, well); return
    splits = df["split"].astype(str).value_counts(dropna=False).to_dict() if "split" in df.columns else {}
    archs  = df["arch"].astype(str).value_counts(dropna=False).to_dict() if "arch" in df.columns else {}
    has_cols = {c: (c in df.columns) for c in
                ["t","idx","global_idx","yhat","ytrue","yhat_family_mean","yhat_final_mean","std_final"]}
    try:
        tmin = pd.to_datetime(df["t"]).min() if "t" in df.columns else None
        tmax = pd.to_datetime(df["t"]).max() if "t" in df.columns else None
    except Exception:
        tmin = df["t"].min() if "t" in df.columns else None
        tmax = df["t"].max() if "t" in df.columns else None
    logging.info("[PhaseViz][dbg] %s well=%s: shape=%s splits=%s archs=%s has=%s t=[%s .. %s]",
                 tag, well, df.shape, splits, archs, has_cols, tmin, tmax)


# ---------------------------
# Make dashed family means visible when intra-only
# ---------------------------
def make_family_traces_visible(fig) -> None:
    if fig is None: 
        return
    for tr in getattr(fig, "data", []):
        name = getattr(tr, "name", "") or ""
        line = getattr(tr, "line", None)
        if (" Mean (" in name) and (line is not None and getattr(line, "dash", None) == "dash") and getattr(tr, "visible", None) == "legendonly":
            tr.visible = True


# ---------------------------
# Alignment factory (global_idx -> t) with split offsets
# ---------------------------
def align_with_t_factory(full_history_df: pd.DataFrame, bounds: Dict[str, Any]) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
    """
    Returns a function that:
      - builds global_idx from (idx, split) using VAL/TEST offsets
      - merges 't' from full-history map
    so every frame lands on the same axis.
    """
    t_map = full_history_df.reset_index().rename(columns={"index": "global_idx"})[["global_idx", "t"]]
    tr_end = int(bounds.get("train_end", 0))
    va_end = int(bounds.get("val_end", tr_end))
    te_sta = int(bounds.get("test_start", va_end + 1))
    VAL_OFFSET  = tr_end + 1
    TEST_OFFSET = max(va_end, te_sta) + 1

    def _add_global_idx(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty or "idx" not in df.columns:
            return df
        d = df.copy()
        split = d["split"].astype(str).str.lower() if "split" in d.columns else "train"
        d["global_idx"] = np.where(
            split.isin(["val","validation"]), d["idx"].astype(int) + VAL_OFFSET,
            np.where(split.eq("test"),        d["idx"].astype(int) + TEST_OFFSET,
                                             d["idx"].astype(int))
        )
        return d

    def _align(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        local = df
        if "t" in local.columns and "idx" in local.columns:
            try:
                if pd.api.types.is_numeric_dtype(local["t"]) and pd.api.types.is_numeric_dtype(local["idx"]) and (local["t"] == local["idx"]).all():
                    local = local.drop(columns=["t"])
            except Exception:
                pass
        local = _add_global_idx(local, name)
        if "global_idx" not in local.columns:
            return local
        out = local.merge(t_map, on="global_idx", how="inner")
        log.info("[PhaseViz] Aligned %s: %d -> %d rows.", name, len(local), len(out))
        return out

    return _align


# =====================================================================
# NEW — Risk helpers (Package 1 scaffolding)
# =====================================================================

def parse_split_selector(selector: str) -> str:
    """
    Normalize a user-provided selector to one of:
      'train' | 'val' | 'test' | 'val+test'
    Accepts common aliases like 'validation', 'vt', 'val_test', etc.
    Defaults to 'val+test' if unknown.
    """
    if not selector:
        return "val+test"
    s = str(selector).strip().lower().replace(" ", "")
    aliases = {
        "train": {"train", "tr"},
        "val": {"val", "validation", "v"},
        "test": {"test", "te", "ts"},
        "val+test": {"val+test", "valtest", "vt", "v+t", "validation+test", "val_test"},
    }
    for key, names in aliases.items():
        if s in names:
            return key
    return "val+test"


def resolve_horizon_indices(
    full_history_df: pd.DataFrame,
    boundaries: Dict[str, Any],
    selector: str,
    horizon_days: int,
) -> Tuple[int, int]:
    """
    Compute [start_idx, end_idx] (inclusive) on the 't'-axis for a window defined by:
      - selector: 'train' | 'val' | 'test' | 'val+test'
      - horizon_days: number of days from start; -1 means "until the end of that window"
    Assumes full_history_df['t'] is already de-duplicated and sorted as in Stage 4.
    """
    if full_history_df is None or full_history_df.empty or "t" not in full_history_df.columns:
        return 0, -1  # caller should guard

    t_axis = full_history_df["t"].reset_index(drop=True)
    n_hist = len(t_axis)
    if n_hist == 0:
        return 0, -1

    tr_end = int(boundaries.get("train_end", 0))
    va_end = int(boundaries.get("val_end", tr_end))
    te_sta = int(boundaries.get("test_start", va_end + 1))

    # Clamp bounds to history
    hi = n_hist - 1
    tr_end = max(0, min(tr_end, hi))
    va_end = max(tr_end, min(va_end, hi))
    te_sta = max(0, min(te_sta, hi))

    sel = parse_split_selector(selector)

    if sel == "train":
        start_i = 0
        end_cap = tr_end
    elif sel == "val":
        start_i = tr_end + 1
        end_cap = va_end
    elif sel == "test":
        start_i = max(va_end + 1, te_sta)
        end_cap = hi
    else:  # 'val+test'
        start_i = tr_end + 1
        end_cap = hi

    # Guard against empty window when boundaries are degenerate
    start_i = max(0, min(start_i, hi))
    end_cap = max(start_i, min(end_cap, hi))

    if horizon_days is None or int(horizon_days) < 0:
        end_i = end_cap
    else:
        # horizon is a count of points from start (inclusive)
        end_i = min(end_cap, start_i + int(horizon_days) - 1)

    return int(start_i), int(end_i)


def slice_by_t(df: pd.DataFrame, t_start: Any, t_end: Any) -> pd.DataFrame:
    """
    Slice a frame by 't' inclusive. If 't' not present or df empty, returns df.
    Works with numeric or datetime-like 't'.
    """
    if df is None or df.empty or "t" not in df.columns:
        return df
    try:
        return df[(df["t"] >= t_start) & (df["t"] <= t_end)].copy()
    except Exception:
        # fallback without raising
        return df


import re
from typing import Optional

def normalize_well_key(w: Optional[object]) -> str:
    """
    Normaliza nome de poço para chave canônica:
    - Converte para string
    - Remove tudo que não for [0-9A-Za-z]
    - Uppercase

    Exemplos:
      '15-9-F-14'   -> '159F14'
      '15/9-F-14'   -> '159F14'
      ' 15_9 f14 '  -> '159F14'
    """
    if w is None:
        return ""
    s = str(w).strip()
    s = re.sub(r"[^0-9A-Za-z]", "", s)
    return s.upper()
