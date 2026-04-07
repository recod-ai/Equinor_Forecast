# src/hpo/final_suite.py

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# 0) Canonical schema (contract for cross-family comparison)
# =============================================================================

CANON_ID_COLS: Tuple[str, ...] = (
    "dataset",
    "well",
    "family",             # "seq2" | "arps" | "darts"
    "architecture_name",
    "strategy",           # Seq2: physics_strategy; Arps: variant; Darts: profile|architecture_name
)

CANON_VAL_METRICS: Tuple[str, ...] = (
    "val_smape_agg",
    "val_smape_cum",
)

CANON_TEST_METRICS: Tuple[str, ...] = (
    "test_smape_agg",
    "test_smape_cum",
)


CANON_OPTIONAL_AUX: Tuple[str, ...] = (
    "epochs", "n_epochs", "batch_size", "learning_rate",
    "lag_window", "horizon", "input_chunk_length", "output_chunk_length",
    "data_sample", "experiment_id",
    "aggregation_method",   # <<< ADICIONE ESTA LINHA
)


CANON_ALL_COLS: Tuple[str, ...] = CANON_ID_COLS + CANON_VAL_METRICS + CANON_TEST_METRICS + CANON_OPTIONAL_AUX


# =============================================================================
# 1) Presentation label map (decoupled; use only at export/render time)
# =============================================================================

def get_presentation_label_map() -> Dict[str, str]:
    return {
        "val_smape_agg":  "Production Rate (Validation)",
        "val_smape_cum":  "Cumulative Production (Validation)",
        "test_smape_agg": "Production Rate (Test)",
        "test_smape_cum": "Cumulative Production (Test)",
        "family":          "Model Family",
        "architecture_name":"Architecture",
        "strategy":        "Strategy",
        "well":            "Well",
        "dataset":         "Dataset",
        "reconstruction":  "Reconstruction",  
    }



# =============================================================================
# 2) Per-family canonicalizers
# =============================================================================
def _coalesce_suffix_columns(
    df: pd.DataFrame,
    base: str,
    *,
    prefer: str = "y",          # "y" (prefer _y over _x) or "x"
    drop_suffix_cols: bool = False,
) -> pd.DataFrame:
    """
    Coalesce merge-suffixed columns (base_x/base_y) into a single canonical `base`.

    Rule:
      base := base (if already present and non-null)
           else preferred suffix (_y by default) if present
           else the other suffix if present

    This is a non-breaking normalization step for post-merge CSVs.
    """
    if df is None or df.empty:
        return df

    bx, by = f"{base}_x", f"{base}_y"
    if (base not in df.columns) and (bx not in df.columns) and (by not in df.columns):
        return df

    out = df.copy()

    # ensure base exists
    if base not in out.columns:
        out[base] = np.nan

    # choose preference order
    first = by if prefer.lower().startswith("y") else bx
    second = bx if first == by else by

    def _as_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    out[base] = _as_num(out[base])

    if first in out.columns:
        out[base] = out[base].combine_first(_as_num(out[first]))
    if second in out.columns:
        out[base] = out[base].combine_first(_as_num(out[second]))

    if drop_suffix_cols:
        out = out.drop(columns=[c for c in (bx, by) if c in out.columns], errors="ignore")

    return out


def _normalize_merge_suffix_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common merge artifacts for test/val metrics:
      test_smape_agg_x/test_smape_agg_y -> test_smape_agg
      test_smape_cum_x/test_smape_cum_y -> test_smape_cum
      (optional) test_size_x/test_size_y -> test_size
    """
    out = df.copy()
    for base in ("test_smape_agg", "test_smape_cum", "test_size"):
        out = _coalesce_suffix_columns(out, base, prefer="y", drop_suffix_cols=False)
    return out


def _coerce_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric metrics exist and are numeric.

    - Required: val_smape_agg, val_smape_cum
    - Optional: test_smape_agg, test_smape_cum (created as NaN if absent)
    - Robust to merge artifacts: test_smape_agg_x/_y etc are coalesced into canonical columns.
    """
    out = df.copy()

    # Fix merge suffix artifacts BEFORE coercion
    out = _normalize_merge_suffix_metrics(out)

    # Required val metrics
    for col in ("val_smape_agg", "val_smape_cum"):
        if col not in out.columns:
            raise ValueError(f"Missing required validation metric '{col}'.")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Optional test metrics: create if absent
    for col in ("test_smape_agg", "test_smape_cum"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    return out



def _harmonize_epochs_and_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light, safe harmonization:
      - if 'epochs' missing and 'n_epochs' present -> create 'epochs' mirror.
      - if 'lag_window' missing and 'input_chunk_length' present -> create 'lag_window' mirror.
      - if 'horizon' missing and 'output_chunk_length' present -> create 'horizon' mirror.
    Never delete the original columns.
    """
    out = df.copy()
    if "epochs" not in out.columns and "n_epochs" in out.columns:
        out["epochs"] = out["n_epochs"]
    if "lag_window" not in out.columns and "input_chunk_length" in out.columns:
        out["lag_window"] = out["input_chunk_length"]
    if "horizon" not in out.columns and "output_chunk_length" in out.columns:
        out["horizon"] = out["output_chunk_length"]
    return out


def _final_canonical_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a view that contains the canonical columns (present ones).
    Missing optional columns are silently ignored; required ones must exist.
    """
    missing_required = [c for c in CANON_ID_COLS + CANON_VAL_METRICS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required canonical columns: {missing_required}")

    cols_present = [c for c in CANON_ALL_COLS if c in df.columns]
    return df.loc[:, cols_present].copy()


def canonicalize_seq2(df: pd.DataFrame) -> pd.DataFrame:
    """Map Seq2 validation CSV to canonical schema."""
    out = df.copy()

    # family
    out["family"] = "seq2"

    # architecture_name fallback
    if "architecture_name" not in out.columns and "architecture" in out.columns:
        out["architecture_name"] = out["architecture"]

    # strategy: physics_strategy is required for Seq2 canonicalization
    if "physics_strategy" not in out.columns:
        raise ValueError("Seq2 CSV must contain 'physics_strategy' to build canonical 'strategy'.")
    out["strategy"] = out["physics_strategy"]

    # metrics & harmonization
    out = _coerce_metrics(out)
    out = _harmonize_epochs_and_chunks(out)

    return _final_canonical_slice(out)


def canonicalize_arps(df: pd.DataFrame) -> pd.DataFrame:
    """Map Arps validation CSV to canonical schema."""
    out = df.copy()

    # family
    out["family"] = "arps"

    # architecture_name fallback
    if "architecture_name" not in out.columns and "architecture" in out.columns:
        out["architecture_name"] = out["architecture"]

    # strategy: variant
    if "variant" not in out.columns:
        raise ValueError("Arps CSV must contain 'variant' to build canonical 'strategy'.")
    out["strategy"] = out["variant"]

    # metrics & harmonization
    out = _coerce_metrics(out)
    out = _harmonize_epochs_and_chunks(out)

    return _final_canonical_slice(out)


def canonicalize_darts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["family"] = "darts"

    if "architecture_name" not in out.columns and "architecture" in out.columns:
        out["architecture_name"] = out["architecture"]

    if "physics_strategy" in out.columns and out["physics_strategy"].notna().any():
        out["strategy"] = out["physics_strategy"].astype(str)
    elif "profile" in out.columns and out["profile"].notna().any():
        out["strategy"] = out["profile"].astype(str)
    else:
        if "architecture_name" not in out.columns:
            raise ValueError("Darts CSV missing fields to derive 'strategy'.")
        out["strategy"] = out["architecture_name"].astype(str)

    out = _coerce_metrics(out)
    out = _harmonize_epochs_and_chunks(out)
    return _final_canonical_slice(out)



# =============================================================================
# 3) Loaders & assembly
# =============================================================================

def _load_csv(csv_path: Path | str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)


def load_family_validation_csv(csv_path: Path | str, family: str) -> pd.DataFrame:
    """
    Load and canonicalize a single family CSV.
    family ∈ {"seq2","arps","darts"}
    """
    raw = _load_csv(csv_path)
    family = family.lower().strip()
    if family == "seq2":
        return canonicalize_seq2(raw)
    if family == "arps":
        return canonicalize_arps(raw)
    if family == "darts":
        return canonicalize_darts(raw)
    raise ValueError(f"Unknown family '{family}'. Expected one of: seq2, arps, darts.")


def build_canonical_validation_df(
    seq2_csv: Optional[Path | str] = None,
    arps_csv: Optional[Path | str] = None,
    darts_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Convenience assembler: load the provided CSVs (if not None), canonicalize,
    and concatenate into a single cross-family DataFrame following the contract.
    """
    frames: List[pd.DataFrame] = []
    if seq2_csv:
        frames.append(load_family_validation_csv(seq2_csv, "seq2"))
    if arps_csv:
        frames.append(load_family_validation_csv(arps_csv, "arps"))
    if darts_csv:
        frames.append(load_family_validation_csv(darts_csv, "darts"))

    if not frames:
        raise ValueError("No CSVs provided. Pass at least one family CSV path.")

    df = pd.concat(frames, ignore_index=True)

    # Sanity: enforce required canonical columns exist post-concat
    missing_required = [c for c in CANON_ID_COLS + CANON_VAL_METRICS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Final canonical DF missing required columns: {missing_required}")

    # Stable ordering (optional)
    cols_present = [c for c in CANON_ALL_COLS if c in df.columns]
    df = df.loc[:, cols_present].copy()

    return df


# =============================================================================
# 4) Presentation view helpers (non-destructive)
# =============================================================================

def build_presentation_view(
    df_canon: pd.DataFrame,
    extra_cols: Optional[List[str]] = None,
    apply_label_map: bool = False,
) -> pd.DataFrame:
    """
    Build a presentation-friendly view:
      - choose a default column order (IDs + validation metrics + test metrics),
      - optionally append extra columns for context,
      - optionally rename using the presentation label map (for LaTeX/tables).
    """
    base_cols = list(CANON_ID_COLS) + list(CANON_VAL_METRICS) + list(CANON_TEST_METRICS)
    cols = [c for c in base_cols if c in df_canon.columns]
    if extra_cols:
        cols.extend([c for c in extra_cols if c in df_canon.columns and c not in cols])

    view = df_canon.loc[:, cols].copy()
    if apply_label_map:
        view = view.rename(columns=get_presentation_label_map())
    return view


# =============================================================================
# 5) Simple summary helpers (optional; useful for the next notebook steps)
# =============================================================================

def summarize_by_family(
    df_canon: pd.DataFrame,
    metrics: Tuple[str, ...] = ("val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum"),
) -> pd.DataFrame:
    """
    Compute mean and std of selected metrics grouped by family (ignores metrics not present).
    """
    use_metrics = [m for m in metrics if m in df_canon.columns]
    if not use_metrics:
        raise ValueError("None of the requested metrics exist in the canonical DataFrame.")
    agg = {m: ["mean", "std"] for m in use_metrics}
    out = df_canon.groupby("family", dropna=False).agg(agg)
    # Flatten MultiIndex columns: (metric, stat) -> "metric_mean" / "metric_std"
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()



def normalize_darts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize Darts validation results.
    strategy := physics_strategy (preferred) → profile → architecture_name
    """
    if df.empty:
        return df

    out = df.copy()
    out["family"] = "darts"

    if "architecture_name" not in out.columns and "architecture" in out.columns:
        out["architecture_name"] = out["architecture"]

    if "physics_strategy" in out.columns and out["physics_strategy"].notna().any():
        out["strategy"] = out["physics_strategy"].astype(str)
    elif "profile" in out.columns and out["profile"].notna().any():
        out["strategy"] = out["profile"].astype(str)
    else:
        if "architecture_name" not in out.columns:
            raise ValueError("Darts CSV missing fields to derive 'strategy' (physics_strategy/profile/architecture_name).")
        out["strategy"] = out["architecture_name"].astype(str)

    out = _coerce_metrics(out)
    out = _harmonize_epochs_and_chunks(out)
    return _final_canonical_slice(out)



def build_topn_per_well_view(
    df_canon: pd.DataFrame,
    top_n: int | None = 3,               # None => mostra tudo
    sort_metric: str = "val_smape_agg",  # menor é melhor
    include_test_metrics: bool = True,
    apply_label_map: bool = True,
    topn_per_architecture: bool = True,  # Top-N por arquitetura dentro do (Dataset, Well)
) -> pd.DataFrame:
    if df_canon.empty:
        return df_canon.copy()
    if sort_metric not in df_canon.columns:
        raise ValueError(f"[build_topn_per_well_view] sort_metric '{sort_metric}' not found.")

    work = df_canon.copy()

    # 1) Buckets de arquitetura e normalização de Strategy
    work["architecture"] = work["family"].map(_architecture_bucket_from_family)
    work["_arch_rank"] = work["architecture"].map(_arch_rank)
    if "strategy" in work.columns:
        work["strategy"] = work["strategy"].map(prettify_strategy)

    # 2) Reconstruction (precisa de aggregation_method para Seq2)
    if "aggregation_method" not in work.columns:
        work["aggregation_method"] = None
    work["reconstruction"] = work.apply(compute_reconstruction, axis=1)

    # 3) Ordenação por métrica
    work[sort_metric] = pd.to_numeric(work[sort_metric], errors="coerce")
    work = work.sort_values(
        ["dataset", "well", "_arch_rank", sort_metric],
        ascending=[True, True, True, True]
    )

    # 4) Top-N (opcional)
    if top_n is not None:
        if topn_per_architecture:
            work = (work
                    .groupby(["dataset", "well", "architecture"], group_keys=False)
                    .head(top_n))
        else:
            work = (work
                    .groupby(["dataset", "well"], group_keys=False)
                    .head(top_n))

    # 5) Promover wells que têm PINN para o topo (dentro de cada dataset)
    has_pinn = (work["architecture"] == "PINN")
    pinn_flag = (work.groupby(["dataset", "well"])["architecture"]
                    .transform(lambda s: int((s == "PINN").any())))
    # ordenar por: dataset, wells com PINN primeiro (desc), well, arquitetura (PINN→ARPS→DARTS), métrica
    work = work.assign(_has_pinn=pinn_flag)
    work = work.sort_values(
        ["dataset", "_has_pinn", "well", "_arch_rank", sort_metric],
        ascending=[True, False, True, True, True]
    )

    # 6) Seleção e ordem de colunas — “Reconstruction” logo após “Strategy”
    cols = [
        "dataset", "well", "architecture", "strategy", "reconstruction",
        "val_smape_agg", "val_smape_cum"
    ]
    if include_test_metrics:
        if "test_smape_agg" in work.columns: cols.append("test_smape_agg")
        if "test_smape_cum" in work.columns: cols.append("test_smape_cum")

    cols = [c for c in cols if c in work.columns]
    view = work.loc[:, cols].copy()

    # 7) Renomeios finais (Architecture -> label map) e aplicação do label map
    view = view.rename(columns={"architecture": "architecture_name"})
    if apply_label_map:
        view = view.rename(columns=get_presentation_label_map())

    # 8) Ordenação final de exibição: por Well, PINN→ARPS→DARTS, métrica
    lbl = get_presentation_label_map()
    col_arch = lbl.get("architecture_name", "Architecture")
    col_well = lbl.get("well", "Well")
    col_metric = lbl.get("val_smape_agg", "Production Rate (Validation)")

    order_map = {"PINN": 0, "ARPS": 1, "DARTS": 2}
    view["_arch_rank"] = view[col_arch].map(order_map).fillna(99)
    view = view.sort_values([col_well, "_arch_rank", col_metric],
                            ascending=[True, True, True]).drop(columns="_arch_rank")

    return view



def build_global_arch_strategy_stats(
    df_canon: pd.DataFrame,
    metrics: Tuple[str, ...] = ("val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum"),
    apply_label_map: bool = True,
) -> pd.DataFrame:
    """
    Table 2: global mean/std across ALL datasets/wells, grouped by:
      - Architecture bucket (ARPS/PINN/DARTS)
      - Strategy (prettified)

    Produz colunas metric_mean e metric_std para cada métrica disponível.
    """
    if df_canon.empty:
        return df_canon.copy()

    work = df_canon.copy()
    work["architecture"] = work["family"].map(_architecture_bucket_from_family)

    if "strategy" not in work.columns:
        raise ValueError("[build_global_arch_strategy_stats] Missing 'strategy' column in canonical DF.")
    work["strategy"] = work["strategy"].map(prettify_strategy)

    use_metrics = [m for m in metrics if m in work.columns]
    if not use_metrics:
        raise ValueError("None of the requested metrics exist in the canonical DataFrame.")

    agg = {m: ["mean", "std"] for m in use_metrics}
    out = (work
           .groupby(["architecture", "strategy"], dropna=False)
           .agg(agg))

    # Flatten MultiIndex columns
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.reset_index()

    # Order by val_smape_agg_mean if available (lower is better)
    sort_cols = [c for c in out.columns if c.startswith("val_smape_agg_mean")]
    if sort_cols:
        out = out.sort_values(sort_cols[0], ascending=True, na_position="last")

    # Apply labels for architecture/metrics if requested
    if apply_label_map:
        labels = get_presentation_label_map()
        rename_map = {}
        if "architecture" in out.columns:
            rename_map["architecture"] = labels.get("architecture_name", "Architecture")
        if "strategy" in out.columns:
            rename_map["strategy"] = labels.get("strategy", "Strategy")
        out = out.rename(columns=rename_map)

        # Também renomear cabeçalhos de métricas (nome base) se fizer sentido em relatório
        base_labels = {
            "val_smape_agg": labels.get("val_smape_agg", "Production Rate (Validation)"),
            "val_smape_cum": labels.get("val_smape_cum", "Cumulative Production (Validation)"),
            "test_smape_agg": labels.get("test_smape_agg", "Production Rate (Test)"),
            "test_smape_cum": labels.get("test_smape_cum", "Cumulative Production (Test)"),
        }
        final_cols = {}
        for c in out.columns:
            if c.endswith("_mean") or c.endswith("_std"):
                base = c.rsplit("_", 1)[0]
                stat = c.rsplit("_", 1)[1]
                if base in base_labels:
                    pretty_base = base_labels[base]
                    final_cols[c] = f"{pretty_base} ({stat.upper()})"
        out = out.rename(columns=final_cols)

    return out



# --- Strategy prettifier ------------------------------------------------------



# --- Strategy prettifier (mantém o que já tínhamos) --------------------------
_STRATEGY_OVERRIDES = {
    "pressure_ensemble": "Pressure Ensemble",
    "hyperbolic": "Hyperbolic",
    "harmonic": "Harmonic",
    "exponential": "Exponential",
    "combined_exp_arps": "Exp–Arps",
    "static": "Static",
    "arps": "Arps Decline",
    # Darts aliases:
    "nhits": "NHiTS",
    "nlinear": "NLinear",
    "tide": "TiDE",
    "tide_rin": "TiDE+RIN",
    "autoarima": "AutoARIMA",
    "arima": "ARIMA",
}

def prettify_strategy(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return s
    key = str(s).strip()
    low = key.lower()
    if low in _STRATEGY_OVERRIDES:
        return _STRATEGY_OVERRIDES[low]
    return key.replace("_", " ").title()


# --- Architecture bucket & order --------------------------------------------
def _architecture_bucket_from_family(family: str) -> str:
    fam = (family or "").lower().strip()
    if fam == "seq2":  return "PINN"
    if fam == "arps":  return "ARPS"
    if fam == "darts": return "DARTS"
    return fam.upper() or "UNKNOWN"

# PINN primeiro, depois ARPS, depois DARTS
_ARCH_ORDER = {"PINN": 0, "ARPS": 1, "DARTS": 2}
def _arch_rank(arch: str) -> int:
    return _ARCH_ORDER.get(arch, 99)


# --- Reconstruction (Seq2) ---------------------------------------------------
_RECON_OVERRIDES = {
    "reconstruct": "Default",
    "reconstruct_warm_raw": "Raw Signal",
    "reconstruct_warm_hp": "HP Filter",
    "reconstruct_warm_holt": "Holt Filter",
    "reconstruct_warm_ewma": "EWMA Filter",
}

def compute_reconstruction(row: pd.Series) -> str:
    fam = str(row.get("family", "")).lower()
    if fam == "seq2":
        method = str(row.get("aggregation_method", "")).strip().lower()
        return _RECON_OVERRIDES.get(method, method.replace("_", " ").title() if method else "Default")
    # ARPS/DARTS => sempre "Default"
    return "Default"



# --- Architecture-level summary (simple) --------------------------------------

ARCH_LABEL_BY_FAMILY = {
    "seq2": "PINN",
    "arps": "ARPS",
    "darts": "DARTS",
}

def build_architecture_summary(
    df_canon: pd.DataFrame,
    prefer_metric_for_order: str = "test_smape_agg",
    fallback_metric_for_order: str = "val_smape_agg",
    apply_label_map: bool = True,
) -> pd.DataFrame:
    """
    Aggregate metrics by Architecture only (PINN/DARTS/ARPS), computing MEAN/STD
    across *all* wells & datasets. Returns a presentation-ready dataframe ordered
    from best to worst by the chosen metric (smaller is better).

    Columns returned (labels if apply_label_map=True):
      Architecture,
      Production Rate (Validation) (MEAN/STD),
      Cumulative Production (Validation) (MEAN/STD),
      Production Rate (Test) (MEAN/STD),
      Cumulative Production (Test) (MEAN/STD)

    If the chosen order metric is missing (e.g., no test columns), we fallback.
    """
    if "family" not in df_canon.columns:
        raise ValueError("canonical DF must contain 'family'.")

    # map family -> Architecture label
    arch = df_canon["family"].str.lower().map(ARCH_LABEL_BY_FAMILY).fillna(df_canon["family"])
    df = df_canon.assign(Architecture=arch)

    metrics = [m for m in ["val_smape_agg","val_smape_cum","test_smape_agg","test_smape_cum"] if m in df.columns]
    if not metrics:
        raise ValueError("No metric columns found to summarize.")

    agg = {m: ["mean","std"] for m in metrics}
    out = df.groupby("Architecture", dropna=False).agg(agg)
    out.columns = [f"{m} ({s.upper()})" for m, s in out.columns]  # flatten with (MEAN)/(STD)
    out = out.reset_index()

    # reorder columns: Architecture first, then val, then test (each MEAN/STD)
    def _pair(m): 
        cols = []
        mean_c = f"{m} (MEAN)"
        std_c  = f"{m} (STD)"
        if mean_c in out.columns: cols.append(mean_c)
        if std_c in out.columns:  cols.append(std_c)
        return cols

    ordered_cols = ["Architecture"] + _pair("val_smape_agg") + _pair("val_smape_cum") + _pair("test_smape_agg") + _pair("test_smape_cum")
    ordered_cols = [c for c in ordered_cols if c in out.columns]
    out = out.loc[:, ordered_cols]

    # pick order metric (fallback if needed)
    order_metric = prefer_metric_for_order if f"{prefer_metric_for_order} (MEAN)" in out.columns else (
                   fallback_metric_for_order if f"{fallback_metric_for_order} (MEAN)" in out.columns else None)
    if order_metric:
        out = out.sort_values(by=f"{order_metric} (MEAN)", ascending=True, kind="mergesort")

    if apply_label_map:
        lab = get_presentation_label_map()
        rename_map = {
            "val_smape_agg (MEAN)": f"{lab['val_smape_agg']} (MEAN)",
            "val_smape_agg (STD)":  f"{lab['val_smape_agg']} (STD)",
            "val_smape_cum (MEAN)": f"{lab['val_smape_cum']} (MEAN)",
            "val_smape_cum (STD)":  f"{lab['val_smape_cum']} (STD)",
            "test_smape_agg (MEAN)":f"{lab['test_smape_agg']} (MEAN)",
            "test_smape_agg (STD)": f"{lab['test_smape_agg']} (STD)",
            "test_smape_cum (MEAN)":f"{lab['test_smape_cum']} (MEAN)",
            "test_smape_cum (STD)": f"{lab['test_smape_cum']} (STD)",
        }
        out = out.rename(columns=rename_map)

    return out


def build_architecture_summary_iqr(
    df_canon: pd.DataFrame,
    prefer_metric_for_order: str = "test_smape_agg",
    fallback_metric_for_order: str = "val_smape_agg",
    apply_label_map: bool = True,
) -> pd.DataFrame:
    """
    Versão robusta: agrega por Architecture (PINN/DARTS/ARPS),
    mantendo as 4 colunas de MEAN e substituindo as 4 colunas de STD por IQR (P25–P75).

    Retorna (se colunas existirem no df):
      Architecture,
      val_smape_agg (MEAN), val_smape_cum (MEAN), test_smape_agg (MEAN), test_smape_cum (MEAN),
      val_smape_agg (IQR),  val_smape_cum (IQR),  test_smape_agg (IQR),  test_smape_cum (IQR)

    Ordena do melhor para o pior pelo 'prefer_metric_for_order' (menor é melhor), 
    com fallback se a métrica estiver ausente.
    """
    # 1) Garantir Architecture a partir de 'family'
    if "family" not in df_canon.columns:
        raise ValueError("canonical DF must contain 'family'.")

    ARCH_LABEL_BY_FAMILY = {
        "seq2": "PINN",
        "arps": "ARPS",
        "darts": "DARTS",
    }

    arch_series = (
        df_canon["family"]
        .astype(str).str.lower()
        .map(ARCH_LABEL_BY_FAMILY)
        .fillna(df_canon["family"])
    )
    df = df_canon.assign(Architecture=arch_series)

    # 2) Selecionar métricas disponíveis
    metric_cols = [c for c in ["val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum"] if c in df.columns]
    if not metric_cols:
        raise ValueError("No metric columns found (expected one of val_smape_agg, val_smape_cum, test_smape_agg, test_smape_cum).")

    # 3) Agregar por Architecture: MEAN e IQR
    def _iqr_str(s: pd.Series) -> str:
        # P25–P75 como string arredondada (2 casas)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        return f"{q1:.2f}–{q3:.2f}"

    rows = []
    for arch, grp in df.groupby("Architecture", dropna=False):
        row = {"Architecture": arch}
        # Médias
        for m in metric_cols:
            row[f"{m} (MEAN)"] = float(grp[m].mean())
        # IQR
        for m in metric_cols:
            row[f"{m} (IQR)"] = _iqr_str(grp[m].astype(float))
        rows.append(row)

    out = pd.DataFrame(rows)

    # 4) Ordenar colunas: Architecture, depois Means (Val->Test), depois IQR (Val->Test)
    def _maybe(cols):
        return [c for c in cols if c in out.columns]

    ordered_cols = ["Architecture"] \
        + _maybe([f"val_smape_agg (MEAN)", f"val_smape_cum (MEAN)", f"test_smape_agg (MEAN)", f"test_smape_cum (MEAN)"]) \
        + _maybe([f"val_smape_agg (IQR)",  f"val_smape_cum (IQR)",  f"test_smape_agg (IQR)",  f"test_smape_cum (IQR)"])
    out = out.loc[:, ordered_cols]

    # 5) Ordenar linhas do melhor para o pior (menor média na métrica escolhida)
    order_metric = None
    if f"{prefer_metric_for_order} (MEAN)" in out.columns:
        order_metric = f"{prefer_metric_for_order} (MEAN)"
    elif f"{fallback_metric_for_order} (MEAN)" in out.columns:
        order_metric = f"{fallback_metric_for_order} (MEAN)"

    if order_metric:
        out = out.sort_values(by=order_metric, ascending=True, kind="mergesort")

    # 6) Aplicar label map (opcional) para nomes bonitos
    if apply_label_map:
        # usa seu mapeamento oficial já presente no projeto
        labels = get_presentation_label_map()  # ex.: {"val_smape_agg": "Production Rate (Validation)", ...}
        rename_map = {}
        for raw in ["val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum"]:
            if f"{raw} (MEAN)" in out.columns and raw in labels:
                rename_map[f"{raw} (MEAN)"] = f"{labels[raw]} (MEAN)"
            if f"{raw} (IQR)" in out.columns and raw in labels:
                rename_map[f"{raw} (IQR)"]  = f"{labels[raw]} (IQR)"
        out = out.rename(columns=rename_map)

    return out









from typing import Dict, Tuple, Optional, List, Any
from pathlib import Path
from collections import Counter
import logging
import re

import pandas as pd
import numpy as np

# Optional YAML support (prefer safe_load; fallback to regex if unavailable)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

log = logging.getLogger("hpo.final_suite")

# =============================================================================
# 0) Canonical schema (contract for cross-family comparison)
# =============================================================================

CANON_ID_COLS: Tuple[str, ...] = (
    "dataset",
    "well",
    "family",             # "seq2" | "arps" | "darts"
    "architecture_name",
    "strategy",           # Seq2: physics_strategy; Arps: variant; Darts: profile|architecture_name
)

CANON_VAL_METRICS: Tuple[str, ...] = (
    "val_smape_agg",
    "val_smape_cum",
)

CANON_TEST_METRICS: Tuple[str, ...] = (
    "test_smape_agg",
    "test_smape_cum",
)

CANON_OPTIONAL_AUX: Tuple[str, ...] = (
    "epochs", "n_epochs", "batch_size", "learning_rate",
    "lag_window", "horizon", "input_chunk_length", "output_chunk_length",
    "data_sample", "experiment_id",
    "aggregation_method",   # keep (used by Reconstruction)
)

CANON_ALL_COLS: Tuple[str, ...] = CANON_ID_COLS + CANON_VAL_METRICS + CANON_TEST_METRICS + CANON_OPTIONAL_AUX


# =============================================================================
# 1) Presentation label map (decoupled; use only at export/render time)
# =============================================================================

def get_presentation_label_map() -> Dict[str, str]:
    return {
        "val_smape_agg":  "Production Rate (Validation)",
        "val_smape_cum":  "Cumulative Production (Validation)",
        "test_smape_agg": "Production Rate (Test)",
        "test_smape_cum": "Cumulative Production (Test)",
        "family":          "Model Family",
        "architecture_name":"Architecture",
        "strategy":        "Strategy",
        "well":            "Well",
        "dataset":         "Dataset",
        "reconstruction":  "Reconstruction",
    }


# =============================================================================
# 2) Per-family canonicalizers
# =============================================================================

def _coalesce_suffix_columns(
    df: pd.DataFrame,
    base: str,
    *,
    prefer: str = "y",          # "y" (prefer _y over _x) or "x"
    drop_suffix_cols: bool = False,
) -> pd.DataFrame:
    """
    Coalesce merge-suffixed columns (base_x/base_y) into a single canonical `base`.

    Rule:
      base := base (if already present and non-null)
           else preferred suffix (_y by default) if present
           else the other suffix if present

    This is a non-breaking normalization step for post-merge CSVs.
    """
    if df is None or df.empty:
        return df

    bx, by = f"{base}_x", f"{base}_y"
    if (base not in df.columns) and (bx not in df.columns) and (by not in df.columns):
        return df

    out = df.copy()

    if base not in out.columns:
        out[base] = np.nan

    first = by if prefer.lower().startswith("y") else bx
    second = bx if first == by else by

    def _as_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    out[base] = _as_num(out[base])

    if first in out.columns:
        out[base] = out[base].combine_first(_as_num(out[first]))
    if second in out.columns:
        out[base] = out[base].combine_first(_as_num(out[second]))

    if drop_suffix_cols:
        out = out.drop(columns=[c for c in (bx, by) if c in out.columns], errors="ignore")

    return out


def _normalize_merge_suffix_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common merge artifacts for test/val metrics:
      test_smape_agg_x/test_smape_agg_y -> test_smape_agg
      test_smape_cum_x/test_smape_cum_y -> test_smape_cum
      (optional) test_size_x/test_size_y -> test_size
    """
    out = df.copy()
    for base in ("test_smape_agg", "test_smape_cum", "test_size"):
        out = _coalesce_suffix_columns(out, base, prefer="y", drop_suffix_cols=False)
    return out


def _coerce_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric metrics exist and are numeric.

    - Required: val_smape_agg, val_smape_cum
    - Optional: test_smape_agg, test_smape_cum (created as NaN if absent)
    - Robust to merge artifacts: test_smape_agg_x/_y etc are coalesced into canonical columns.
    """
    out = df.copy()

    out = _normalize_merge_suffix_metrics(out)

    for col in ("val_smape_agg", "val_smape_cum"):
        if col not in out.columns:
            raise ValueError(f"Missing required validation metric '{col}'.")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ("test_smape_agg", "test_smape_cum"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    return out


def _harmonize_epochs_and_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light, safe harmonization:
      - if 'epochs' missing and 'n_epochs' present -> create 'epochs' mirror.
      - if 'lag_window' missing and 'input_chunk_length' present -> create 'lag_window' mirror.
      - if 'horizon' missing and 'output_chunk_length' present -> create 'horizon' mirror.
    Never delete the original columns.
    """
    out = df.copy()
    if "epochs" not in out.columns and "n_epochs" in out.columns:
        out["epochs"] = out["n_epochs"]
    if "lag_window" not in out.columns and "input_chunk_length" in out.columns:
        out["lag_window"] = out["input_chunk_length"]
    if "horizon" not in out.columns and "output_chunk_length" in out.columns:
        out["horizon"] = out["output_chunk_length"]
    return out


def _final_canonical_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a view that contains the canonical columns (present ones).
    Missing optional columns are silently ignored; required ones must exist.
    """
    missing_required = [c for c in CANON_ID_COLS + CANON_VAL_METRICS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required canonical columns: {missing_required}")

    cols_present = [c for c in CANON_ALL_COLS if c in df.columns]
    return df.loc[:, cols_present].copy()


def canonicalize_seq2(df: pd.DataFrame) -> pd.DataFrame:
    """Map Seq2 validation CSV to canonical schema."""
    out = df.copy()

    out["family"] = "seq2"

    if "architecture_name" not in out.columns and "architecture" in out.columns:
        out["architecture_name"] = out["architecture"]

    if "physics_strategy" not in out.columns:
        raise ValueError("Seq2 CSV must contain 'physics_strategy' to build canonical 'strategy'.")
    out["strategy"] = out["physics_strategy"]

    out = _coerce_metrics(out)
    out = _harmonize_epochs_and_chunks(out)

    return _final_canonical_slice(out)


def canonicalize_arps(df: pd.DataFrame) -> pd.DataFrame:
    """Map Arps validation CSV to canonical schema."""
    out = df.copy()

    out["family"] = "arps"

    if "architecture_name" not in out.columns and "architecture" in out.columns:
        out["architecture_name"] = out["architecture"]

    if "variant" not in out.columns:
        raise ValueError("Arps CSV must contain 'variant' to build canonical 'strategy'.")
    out["strategy"] = out["variant"]

    out = _coerce_metrics(out)
    out = _harmonize_epochs_and_chunks(out)

    return _final_canonical_slice(out)


def canonicalize_darts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["family"] = "darts"

    if "architecture_name" not in out.columns and "architecture" in out.columns:
        out["architecture_name"] = out["architecture"]

    if "physics_strategy" in out.columns and out["physics_strategy"].notna().any():
        out["strategy"] = out["physics_strategy"].astype(str)
    elif "profile" in out.columns and out["profile"].notna().any():
        out["strategy"] = out["profile"].astype(str)
    else:
        if "architecture_name" not in out.columns:
            raise ValueError("Darts CSV missing fields to derive 'strategy'.")
        out["strategy"] = out["architecture_name"].astype(str)

    out = _coerce_metrics(out)
    out = _harmonize_epochs_and_chunks(out)
    return _final_canonical_slice(out)


# =============================================================================
# 3) Loaders & assembly
# =============================================================================

def _load_csv(csv_path: Path | str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)


def load_family_validation_csv(csv_path: Path | str, family: str) -> pd.DataFrame:
    """
    Load and canonicalize a single family CSV.
    family ∈ {"seq2","arps","darts"}
    """
    raw = _load_csv(csv_path)
    family = family.lower().strip()
    if family == "seq2":
        return canonicalize_seq2(raw)
    if family == "arps":
        return canonicalize_arps(raw)
    if family == "darts":
        return canonicalize_darts(raw)
    raise ValueError(f"Unknown family '{family}'. Expected one of: seq2, arps, darts.")


def build_canonical_validation_df(
    seq2_csv: Optional[Path | str] = None,
    arps_csv: Optional[Path | str] = None,
    darts_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Convenience assembler: load the provided CSVs (if not None), canonicalize,
    and concatenate into a single cross-family DataFrame following the contract.
    """
    frames: List[pd.DataFrame] = []
    if seq2_csv:
        frames.append(load_family_validation_csv(seq2_csv, "seq2"))
    if arps_csv:
        frames.append(load_family_validation_csv(arps_csv, "arps"))
    if darts_csv:
        frames.append(load_family_validation_csv(darts_csv, "darts"))

    if not frames:
        raise ValueError("No CSVs provided. Pass at least one family CSV path.")

    df = pd.concat(frames, ignore_index=True)

    missing_required = [c for c in CANON_ID_COLS + CANON_VAL_METRICS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Final canonical DF missing required columns: {missing_required}")

    cols_present = [c for c in CANON_ALL_COLS if c in df.columns]
    df = df.loc[:, cols_present].copy()

    return df


# =============================================================================
# 4) Presentation view helpers (non-destructive)
# =============================================================================

def build_presentation_view(
    df_canon: pd.DataFrame,
    extra_cols: Optional[List[str]] = None,
    apply_label_map: bool = False,
) -> pd.DataFrame:
    """
    Build a presentation-friendly view:
      - choose a default column order (IDs + validation metrics + test metrics),
      - optionally append extra columns for context,
      - optionally rename using the presentation label map (for LaTeX/tables).
    """
    base_cols = list(CANON_ID_COLS) + list(CANON_VAL_METRICS) + list(CANON_TEST_METRICS)
    cols = [c for c in base_cols if c in df_canon.columns]
    if extra_cols:
        cols.extend([c for c in extra_cols if c in df_canon.columns and c not in cols])

    view = df_canon.loc[:, cols].copy()
    if apply_label_map:
        view = view.rename(columns=get_presentation_label_map())
    return view


# =============================================================================
# 5) Simple summary helpers (optional; useful for the next notebook steps)
# =============================================================================

def summarize_by_family(
    df_canon: pd.DataFrame,
    metrics: Tuple[str, ...] = ("val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum"),
) -> pd.DataFrame:
    """
    Compute mean and std of selected metrics grouped by family (ignores metrics not present).
    """
    use_metrics = [m for m in metrics if m in df_canon.columns]
    if not use_metrics:
        raise ValueError("None of the requested metrics exist in the canonical DataFrame.")
    agg = {m: ["mean", "std"] for m in use_metrics}
    out = df_canon.groupby("family", dropna=False).agg(agg)
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def normalize_darts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize Darts validation results.
    strategy := physics_strategy (preferred) → profile → architecture_name
    """
    if df.empty:
        return df

    out = df.copy()
    out["family"] = "darts"

    if "architecture_name" not in out.columns and "architecture" in out.columns:
        out["architecture_name"] = out["architecture"]

    if "physics_strategy" in out.columns and out["physics_strategy"].notna().any():
        out["strategy"] = out["physics_strategy"].astype(str)
    elif "profile" in out.columns and out["profile"].notna().any():
        out["strategy"] = out["profile"].astype(str)
    else:
        if "architecture_name" not in out.columns:
            raise ValueError("Darts CSV missing fields to derive 'strategy' (physics_strategy/profile/architecture_name).")
        out["strategy"] = out["architecture_name"].astype(str)

    out = _coerce_metrics(out)
    out = _harmonize_epochs_and_chunks(out)
    return _final_canonical_slice(out)


def build_topn_per_well_view(
    df_canon: pd.DataFrame,
    top_n: int | None = 3,
    sort_metric: str = "val_smape_agg",
    include_test_metrics: bool = True,
    apply_label_map: bool = True,
    topn_per_architecture: bool = True,
) -> pd.DataFrame:
    if df_canon.empty:
        return df_canon.copy()
    if sort_metric not in df_canon.columns:
        raise ValueError(f"[build_topn_per_well_view] sort_metric '{sort_metric}' not found.")

    work = df_canon.copy()

    work["architecture"] = work["family"].map(_architecture_bucket_from_family)
    work["_arch_rank"] = work["architecture"].map(_arch_rank)
    if "strategy" in work.columns:
        work["strategy"] = work["strategy"].map(prettify_strategy)

    if "aggregation_method" not in work.columns:
        work["aggregation_method"] = None
    work["reconstruction"] = work.apply(compute_reconstruction, axis=1)

    work[sort_metric] = pd.to_numeric(work[sort_metric], errors="coerce")
    work = work.sort_values(
        ["dataset", "well", "_arch_rank", sort_metric],
        ascending=[True, True, True, True]
    )

    if top_n is not None:
        if topn_per_architecture:
            work = (work
                    .groupby(["dataset", "well", "architecture"], group_keys=False)
                    .head(top_n))
        else:
            work = (work
                    .groupby(["dataset", "well"], group_keys=False)
                    .head(top_n))

    pinn_flag = (work.groupby(["dataset", "well"])["architecture"]
                    .transform(lambda s: int((s == "PINN").any())))
    work = work.assign(_has_pinn=pinn_flag)
    work = work.sort_values(
        ["dataset", "_has_pinn", "well", "_arch_rank", sort_metric],
        ascending=[True, False, True, True, True]
    )

    cols = [
        "dataset", "well", "architecture", "strategy", "reconstruction",
        "val_smape_agg", "val_smape_cum"
    ]
    if include_test_metrics:
        if "test_smape_agg" in work.columns:
            cols.append("test_smape_agg")
        if "test_smape_cum" in work.columns:
            cols.append("test_smape_cum")

    cols = [c for c in cols if c in work.columns]
    view = work.loc[:, cols].copy()

    view = view.rename(columns={"architecture": "architecture_name"})
    if apply_label_map:
        view = view.rename(columns=get_presentation_label_map())

    lbl = get_presentation_label_map()
    col_arch = lbl.get("architecture_name", "Architecture")
    col_well = lbl.get("well", "Well")
    col_metric = lbl.get("val_smape_agg", "Production Rate (Validation)")

    order_map = {"PINN": 0, "ARPS": 1, "DARTS": 2}
    view["_arch_rank"] = view[col_arch].map(order_map).fillna(99)
    view = view.sort_values(
        [col_well, "_arch_rank", col_metric],
        ascending=[True, True, True]
    ).drop(columns="_arch_rank")

    return view


def build_global_arch_strategy_stats(
    df_canon: pd.DataFrame,
    metrics: Tuple[str, ...] = ("val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum"),
    apply_label_map: bool = True,
) -> pd.DataFrame:
    """
    Table 2 (alt): global mean/std across ALL datasets/wells, grouped by:
      - Architecture bucket (ARPS/PINN/DARTS)
      - Strategy (prettified)
    """
    if df_canon.empty:
        return df_canon.copy()

    work = df_canon.copy()
    work["architecture"] = work["family"].map(_architecture_bucket_from_family)

    if "strategy" not in work.columns:
        raise ValueError("[build_global_arch_strategy_stats] Missing 'strategy' column in canonical DF.")
    work["strategy"] = work["strategy"].map(prettify_strategy)

    use_metrics = [m for m in metrics if m in work.columns]
    if not use_metrics:
        raise ValueError("None of the requested metrics exist in the canonical DataFrame.")

    agg = {m: ["mean", "std"] for m in use_metrics}
    out = (work
           .groupby(["architecture", "strategy"], dropna=False)
           .agg(agg))

    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.reset_index()

    sort_cols = [c for c in out.columns if c.startswith("val_smape_agg_mean")]
    if sort_cols:
        out = out.sort_values(sort_cols[0], ascending=True, na_position="last")

    if apply_label_map:
        labels = get_presentation_label_map()
        rename_map = {}
        if "architecture" in out.columns:
            rename_map["architecture"] = labels.get("architecture_name", "Architecture")
        if "strategy" in out.columns:
            rename_map["strategy"] = labels.get("strategy", "Strategy")
        out = out.rename(columns=rename_map)

        base_labels = {
            "val_smape_agg": labels.get("val_smape_agg", "Production Rate (Validation)"),
            "val_smape_cum": labels.get("val_smape_cum", "Cumulative Production (Validation)"),
            "test_smape_agg": labels.get("test_smape_agg", "Production Rate (Test)"),
            "test_smape_cum": labels.get("test_smape_cum", "Cumulative Production (Test)"),
        }
        final_cols = {}
        for c in out.columns:
            if c.endswith("_mean") or c.endswith("_std"):
                base = c.rsplit("_", 1)[0]
                stat = c.rsplit("_", 1)[1]
                if base in base_labels:
                    pretty_base = base_labels[base]
                    final_cols[c] = f"{pretty_base} ({stat.upper()})"
        out = out.rename(columns=final_cols)

    return out


# --- Strategy prettifier ------------------------------------------------------

_STRATEGY_OVERRIDES = {
    "pressure_ensemble": "Pressure Ensemble",
    "hyperbolic": "Hyperbolic",
    "harmonic": "Harmonic",
    "exponential": "Exponential",
    "combined_exp_arps": "Exp–Arps",
    "static": "Static",
    "arps": "Arps Decline",
    # Darts aliases:
    "nhits": "NHiTS",
    "nlinear": "NLinear",
    "tide": "TiDE",
    "tide_rin": "TiDE+RIN",
    "autoarima": "AutoARIMA",
    "arima": "ARIMA",
}


def prettify_strategy(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return s
    key = str(s).strip()
    low = key.lower()
    if low in _STRATEGY_OVERRIDES:
        return _STRATEGY_OVERRIDES[low]
    return key.replace("_", " ").title()


def _architecture_bucket_from_family(family: str) -> str:
    fam = (family or "").lower().strip()
    if fam == "seq2":
        return "PINN"
    if fam == "arps":
        return "ARPS"
    if fam == "darts":
        return "DARTS"
    return fam.upper() or "UNKNOWN"


_ARCH_ORDER = {"PINN": 0, "ARPS": 1, "DARTS": 2}


def _arch_rank(arch: str) -> int:
    return _ARCH_ORDER.get(arch, 99)


_RECON_OVERRIDES = {
    "reconstruct": "Default",
    "reconstruct_warm_raw": "Raw Signal",
    "reconstruct_warm_hp": "HP Filter",
    "reconstruct_warm_holt": "Holt Filter",
    "reconstruct_warm_ewma": "EWMA Filter",
}


def compute_reconstruction(row: pd.Series) -> str:
    fam = str(row.get("family", "")).lower()
    if fam == "seq2":
        method = str(row.get("aggregation_method", "")).strip().lower()
        return _RECON_OVERRIDES.get(method, method.replace("_", " ").title() if method else "Default")
    return "Default"


ARCH_LABEL_BY_FAMILY = {
    "seq2": "PINN",
    "arps": "ARPS",
    "darts": "DARTS",
}


def build_architecture_summary(
    df_canon: pd.DataFrame,
    prefer_metric_for_order: str = "test_smape_agg",
    fallback_metric_for_order: str = "val_smape_agg",
    apply_label_map: bool = True,
) -> pd.DataFrame:
    """
    Aggregate metrics by Architecture only (PINN/DARTS/ARPS), computing MEAN/STD
    across *all* wells & datasets. Returns a presentation-ready dataframe ordered
    from best to worst by the chosen metric (smaller is better).
    """
    if "family" not in df_canon.columns:
        raise ValueError("canonical DF must contain 'family'.")

    arch = df_canon["family"].str.lower().map(ARCH_LABEL_BY_FAMILY).fillna(df_canon["family"])
    df = df_canon.assign(Architecture=arch)

    metrics = [m for m in ["val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum"] if m in df.columns]
    if not metrics:
        raise ValueError("No metric columns found to summarize.")

    agg = {m: ["mean", "std"] for m in metrics}
    out = df.groupby("Architecture", dropna=False).agg(agg)
    out.columns = [f"{m} ({s.upper()})" for m, s in out.columns]
    out = out.reset_index()

    def _pair(m: str) -> List[str]:
        cols: List[str] = []
        mean_c = f"{m} (MEAN)"
        std_c = f"{m} (STD)"
        if mean_c in out.columns:
            cols.append(mean_c)
        if std_c in out.columns:
            cols.append(std_c)
        return cols

    ordered_cols = ["Architecture"] + _pair("val_smape_agg") + _pair("val_smape_cum") + _pair("test_smape_agg") + _pair("test_smape_cum")
    ordered_cols = [c for c in ordered_cols if c in out.columns]
    out = out.loc[:, ordered_cols]

    order_metric = (
        prefer_metric_for_order if f"{prefer_metric_for_order} (MEAN)" in out.columns
        else (fallback_metric_for_order if f"{fallback_metric_for_order} (MEAN)" in out.columns else None)
    )
    if order_metric:
        out = out.sort_values(by=f"{order_metric} (MEAN)", ascending=True, kind="mergesort")

    if apply_label_map:
        lab = get_presentation_label_map()
        rename_map = {
            "val_smape_agg (MEAN)": f"{lab['val_smape_agg']} (MEAN)",
            "val_smape_agg (STD)":  f"{lab['val_smape_agg']} (STD)",
            "val_smape_cum (MEAN)": f"{lab['val_smape_cum']} (MEAN)",
            "val_smape_cum (STD)":  f"{lab['val_smape_cum']} (STD)",
            "test_smape_agg (MEAN)": f"{lab['test_smape_agg']} (MEAN)",
            "test_smape_agg (STD)":  f"{lab['test_smape_agg']} (STD)",
            "test_smape_cum (MEAN)": f"{lab['test_smape_cum']} (MEAN)",
            "test_smape_cum (STD)":  f"{lab['test_smape_cum']} (STD)",
        }
        out = out.rename(columns=rename_map)

    return out


def build_architecture_summary_iqr(
    df_canon: pd.DataFrame,
    prefer_metric_for_order: str = "test_smape_agg",
    fallback_metric_for_order: str = "val_smape_agg",
    apply_label_map: bool = True,
) -> pd.DataFrame:
    """
    Robust version: aggregates by Architecture (PINN/DARTS/ARPS),
    keeps MEAN columns and replaces STD columns with IQR (P25–P75) strings.
    """
    if "family" not in df_canon.columns:
        raise ValueError("canonical DF must contain 'family'.")

    arch_series = (
        df_canon["family"]
        .astype(str).str.lower()
        .map(ARCH_LABEL_BY_FAMILY)
        .fillna(df_canon["family"])
    )
    df = df_canon.assign(Architecture=arch_series)

    metric_cols = [c for c in ["val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum"] if c in df.columns]
    if not metric_cols:
        raise ValueError("No metric columns found (expected one of val_smape_agg, val_smape_cum, test_smape_agg, test_smape_cum).")

    def _iqr_str(s: pd.Series) -> str:
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        return f"{q1:.2f}–{q3:.2f}"

    rows: List[Dict[str, Any]] = []
    for arch, grp in df.groupby("Architecture", dropna=False):
        row: Dict[str, Any] = {"Architecture": arch}
        for m in metric_cols:
            row[f"{m} (MEAN)"] = float(grp[m].mean())
        for m in metric_cols:
            row[f"{m} (IQR)"] = _iqr_str(grp[m].astype(float))
        rows.append(row)

    out = pd.DataFrame(rows)

    def _maybe(cols: List[str]) -> List[str]:
        return [c for c in cols if c in out.columns]

    ordered_cols = ["Architecture"] \
        + _maybe([f"val_smape_agg (MEAN)", f"val_smape_cum (MEAN)", f"test_smape_agg (MEAN)", f"test_smape_cum (MEAN)"]) \
        + _maybe([f"val_smape_agg (IQR)",  f"val_smape_cum (IQR)",  f"test_smape_agg (IQR)",  f"test_smape_cum (IQR)"])
    out = out.loc[:, ordered_cols]

    order_metric = None
    if f"{prefer_metric_for_order} (MEAN)" in out.columns:
        order_metric = f"{prefer_metric_for_order} (MEAN)"
    elif f"{fallback_metric_for_order} (MEAN)" in out.columns:
        order_metric = f"{fallback_metric_for_order} (MEAN)"

    if order_metric:
        out = out.sort_values(by=order_metric, ascending=True, kind="mergesort")

    if apply_label_map:
        labels = get_presentation_label_map()
        rename_map: Dict[str, str] = {}
        for raw in ["val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum"]:
            if f"{raw} (MEAN)" in out.columns and raw in labels:
                rename_map[f"{raw} (MEAN)"] = f"{labels[raw]} (MEAN)"
            if f"{raw} (IQR)" in out.columns and raw in labels:
                rename_map[f"{raw} (IQR)"] = f"{labels[raw]} (IQR)"
        out = out.rename(columns=rename_map)

    return out


# =============================================================================
# 6) Final Suite runners (single & multi-campaign) + split readers
# =============================================================================

def _repo_root_from_any(repo_root: Path | str) -> Path:
    return repo_root if isinstance(repo_root, Path) else Path(str(repo_root)).expanduser().resolve()


def _reports_dir(repo_root: Path, exp: str) -> Path:
    return repo_root / "src" / "experiment_configs" / exp / "reports"


def _campaigns_dir(repo_root: Path, exp: str) -> Path:
    return repo_root / "src" / "experiment_configs" / "hpo_campaigns" / exp


def _family_csv_paths(reports_dir: Path) -> Dict[str, Optional[Path]]:
    def _p(fam: str) -> Optional[Path]:
        p = reports_dir / fam / "final_validation_of_champions.csv"
        return p if p.exists() else None

    return {"seq2": _p("seq2"), "arps": _p("arps"), "darts": _p("darts")}


def _yaml_safe_load_or_none(text: str) -> Optional[dict]:
    if yaml is None:
        return None
    try:
        obj = yaml.safe_load(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1].strip()
    return s


def _parse_yaml_via_regex(text: str) -> dict:
    """
    Minimal, stable YAML-ish parser for the fields we need when PyYAML is unavailable.
    It intentionally ignores nested structure and matches by key names.
    """
    def _m(pattern: str) -> Optional[str]:
        mm = re.search(pattern, text, flags=re.MULTILINE)
        return mm.group(1).strip() if mm else None

    campaign_name = _m(r"^\s*campaign_name\s*:\s*(.+?)\s*$")
    dataset_name = _m(r"^\s*dataset_name\s*:\s*(.+?)\s*$")
    wells_raw = _m(r"^\s*wells\s*:\s*\[(.*?)\]\s*$")
    test_size = _m(r"^\s*test_size\s*:\s*([0-9]*\.?[0-9]+)\s*$")
    val_size = _m(r"^\s*val_size\s*:\s*([0-9]*\.?[0-9]+)\s*$")

    wells: List[str] = []
    if wells_raw:
        parts = [p.strip() for p in wells_raw.split(",") if p.strip()]
        wells = [_strip_quotes(p) for p in parts]

    out: Dict[str, Any] = {
        "campaign_name": _strip_quotes(campaign_name) if campaign_name else None,
        "dataset_name": _strip_quotes(dataset_name) if dataset_name else None,
        "wells": wells if wells else None,
        "test_size": float(test_size) if test_size is not None else None,
        "val_size": float(val_size) if val_size is not None else None,
    }
    return out


def _extract_split_rows(
    exp: str,
    family: str,
    yaml_path: Path,
    obj: Optional[dict],
    raw_text: str,
) -> List[Dict[str, Any]]:
    """
    Normalize one YAML into 1+ rows (one per well) with the split fields.
    Skips if required fields are missing.
    """
    if obj is None:
        mini = _parse_yaml_via_regex(raw_text)
        campaign_name = mini.get("campaign_name")
        dataset_name = mini.get("dataset_name")
        wells = mini.get("wells") or []
        test_size = mini.get("test_size")
        val_size = mini.get("val_size")
    else:
        campaign_name = obj.get("campaign_name")
        run_scope = obj.get("run_scope", {}) or {}
        job_defaults = obj.get("job_defaults", {}) or {}
        dataset_name = run_scope.get("dataset_name")
        wells = run_scope.get("wells") or []
        test_size = job_defaults.get("test_size")
        val_size = job_defaults.get("val_size")

    if not dataset_name or not wells or test_size is None or val_size is None:
        log.warning(
            "[read_campaign_splits] Skipping YAML with missing fields: exp=%s family=%s file=%s "
            "(dataset=%s wells=%s test_size=%s val_size=%s)",
            exp, family, yaml_path, dataset_name, wells, test_size, val_size
        )
        return []

    # normalize wells as list[str]
    if isinstance(wells, str):
        wells = [wells]
    wells = [str(w).strip() for w in wells if str(w).strip()]

    rows: List[Dict[str, Any]] = []
    for well in wells:
        rows.append({
            "exp": exp,
            "family": family,
            "dataset": str(dataset_name).strip(),
            "well": str(well).strip(),
            "campaign_name": str(campaign_name).strip() if campaign_name else None,
            "test_size": float(test_size) if test_size is not None else np.nan,
            "val_size": float(val_size) if val_size is not None else np.nan,
            "yaml_path": str(yaml_path),
        })
    return rows


def read_campaign_splits(repo_root: Path | str, exp: str) -> pd.DataFrame:
    """
    Read split metadata from campaign YAMLs:
      src/experiment_configs/hpo_campaigns/{EXP}/{family}/*.yaml

    Returns a dataframe with columns:
      exp, family, dataset, well, campaign_name, test_size, val_size, yaml_path
    """
    root = _repo_root_from_any(repo_root)
    camp_dir = _campaigns_dir(root, exp)

    rows: List[Dict[str, Any]] = []
    for family in ("seq2", "arps", "darts"):
        fam_dir = camp_dir / family
        if not fam_dir.exists():
            continue
        for yp in sorted(fam_dir.glob("*.yaml")):
            try:
                text = yp.read_text(encoding="utf-8", errors="ignore")
                obj = _yaml_safe_load_or_none(text)
                rows.extend(_extract_split_rows(exp, family, yp, obj, text))
            except Exception as e:
                log.warning("[read_campaign_splits] Failed to parse YAML: exp=%s family=%s file=%s err=%s", exp, family, yp, e)

    cols = ["exp", "family", "dataset", "well", "campaign_name", "test_size", "val_size", "yaml_path"]
    return pd.DataFrame(rows, columns=cols)


def reduce_campaign_split(df_splits: pd.DataFrame, *, dominant_threshold: float = 0.80) -> Dict[str, Any]:
    """
    Reduce per-YAML split rows into:
      - split_mode='campaign' with single (test_size,val_size) if unique
      - split_mode='family' with per-family columns if not unique

    Note: If multiple pairs exist, we will warn. If a dominant pair covers >=dominant_threshold,
    we still keep split_mode='campaign' (but warn about inconsistency). Otherwise we fallback to family.
    """
    if df_splits is None or df_splits.empty:
        return {"split_mode": "unknown", "split_id": "unknown"}

    work = df_splits.copy()
    work["test_size"] = pd.to_numeric(work.get("test_size"), errors="coerce")
    work["val_size"] = pd.to_numeric(work.get("val_size"), errors="coerce")
    work = work.dropna(subset=["test_size", "val_size"])

    if work.empty:
        return {"split_mode": "unknown", "split_id": "unknown"}

    # make pairs stable for counting
    pairs = list(zip(work["test_size"].round(6), work["val_size"].round(6)))
    counts = Counter(pairs)
    unique_pairs = list(counts.keys())

    def _split_id(ts: float, vs: float) -> str:
        return f"t{ts:.2f}_v{vs:.2f}"

    if len(unique_pairs) == 1:
        ts, vs = unique_pairs[0]
        return {
            "split_mode": "campaign",
            "test_size": float(ts),
            "val_size": float(vs),
            "split_id": _split_id(float(ts), float(vs)),
        }

    # Conflict: warn and decide between dominant campaign split vs per-family fallback
    (dom_pair, dom_count) = counts.most_common(1)[0]
    dom_ratio = dom_count / max(1, len(pairs))
    ts_dom, vs_dom = dom_pair

    log.warning(
        "[reduce_campaign_split] Multiple split pairs detected (n=%d, unique=%d). Dominant=%s ratio=%.2f. "
        "Falling back to per-family if ratio < %.2f.",
        len(pairs), len(unique_pairs), _split_id(float(ts_dom), float(vs_dom)), dom_ratio, dominant_threshold
    )

    if dom_ratio >= dominant_threshold:
        return {
            "split_mode": "campaign",
            "test_size": float(ts_dom),
            "val_size": float(vs_dom),
            "split_id": _split_id(float(ts_dom), float(vs_dom)),
            "warning": "inconsistent_splits_dominant_used",
        }

    out: Dict[str, Any] = {"split_mode": "family", "split_id": "mixed"}
    for fam in ("seq2", "arps", "darts"):
        fam_df = work[work["family"].astype(str).str.lower() == fam]
        if fam_df.empty:
            out[f"test_size_{fam}"] = np.nan
            out[f"val_size_{fam}"] = np.nan
            continue
        fam_pairs = list(zip(fam_df["test_size"].round(6), fam_df["val_size"].round(6)))
        fam_counts = Counter(fam_pairs)
        (fam_pair, _) = fam_counts.most_common(1)[0]
        ts_f, vs_f = fam_pair
        out[f"test_size_{fam}"] = float(ts_f)
        out[f"val_size_{fam}"] = float(vs_f)

    out["warning"] = "inconsistent_splits_family_fallback"
    return out


def run_final_suite_single(
    repo_root: Path | str,
    exp: str,
    *,
    top_n: int = 2,
    sort_metric: str = "val_smape_agg",
    include_test_metrics: bool = True,
    apply_label_map: bool = True,
    topn_per_architecture: bool = True,
    prefer_metric_for_order: str = "test_smape_agg",
    fallback_metric_for_order: str = "val_smape_agg",
) -> Dict[str, Any]:
    """
    Single-campaign runner (keeps current behavior):
      - loads family CSVs from src/experiment_configs/{EXP}/reports/{family}/final_validation_of_champions.csv
      - builds canon, table1, table2
      - persists table1/table2 in reports/final_suite_outputs/
    """
    root = _repo_root_from_any(repo_root)
    reports_dir = _reports_dir(root, exp)
    csv_paths = _family_csv_paths(reports_dir)

    for fam, p in csv_paths.items():
        log.info("Using %s CSV: %s", fam, str(p) if p else "NOT FOUND")

    if not any(csv_paths.values()):
        raise RuntimeError(f"No family CSVs found for EXP={exp}. Generate at least one final_validation_of_champions.csv.")

    canon = build_canonical_validation_df(
        seq2_csv=str(csv_paths["seq2"]) if csv_paths["seq2"] else None,
        arps_csv=str(csv_paths["arps"]) if csv_paths["arps"] else None,
        darts_csv=str(csv_paths["darts"]) if csv_paths["darts"] else None,
    )

    numeric_cols = canon.select_dtypes(include=np.number).columns
    canon.loc[:, numeric_cols] = canon.loc[:, numeric_cols].round(2)

    table1 = build_topn_per_well_view(
        canon,
        top_n=top_n,
        sort_metric=sort_metric,
        include_test_metrics=include_test_metrics,
        apply_label_map=apply_label_map,
        topn_per_architecture=topn_per_architecture,
    )

    table2 = build_architecture_summary_iqr(
        canon,
        prefer_metric_for_order=prefer_metric_for_order,
        fallback_metric_for_order=fallback_metric_for_order,
        apply_label_map=apply_label_map,
    )

    numeric_cols2 = table2.select_dtypes(include=np.number).columns
    table2.loc[:, numeric_cols2] = table2.loc[:, numeric_cols2].round(2)

    out_dir = reports_dir / "final_suite_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep filenames compatible with your current notebook
    t1_path = out_dir / "table1_topN_per_well.csv"
    t2_path = out_dir / "table2_global_arch_strategy_stats.csv"
    table1.to_csv(t1_path, index=False)
    table2.to_csv(t2_path, index=False)

    log.info("Written: %s", str(t1_path))
    log.info("Written: %s", str(t2_path))

    return {
        "exp": exp,
        "reports_dir": reports_dir,
        "csv_paths": {k: (str(v) if v else None) for k, v in csv_paths.items()},
        "canon": canon,
        "table1": table1,
        "table2": table2,
        "out_dir": out_dir,
    }


def run_final_suite_multi(
    repo_root: Path | str,
    exps: List[str],
    *,
    top_n: int = 2,
    sort_metric: str = "val_smape_agg",
    include_test_metrics: bool = True,
    apply_label_map: bool = True,
    topn_per_architecture: bool = True,
    prefer_metric_for_order: str = "test_smape_agg",
    fallback_metric_for_order: str = "val_smape_agg",
    dominant_threshold: float = 0.80,
) -> Dict[str, Any]:
    """
    Multi-campaign runner:
      - runs N EXPS via run_final_suite_single (reuses existing core)
      - reads YAML splits from src/experiment_configs/hpo_campaigns/{EXP}/...
      - attaches split metadata to each table2
      - returns:
          tables1_by_exp: List[(exp, table1)]
          tables2_by_exp: List[(exp, table2_with_meta)]
          table2_unified: concatenated table2
      - writes a global unified CSV into:
          src/experiment_configs/hpo_campaigns/final_suite_outputs/table2_unified.csv
    """
    root = _repo_root_from_any(repo_root)

    tables1_by_exp: List[Tuple[str, pd.DataFrame]] = []
    tables2_by_exp: List[Tuple[str, pd.DataFrame]] = []
    skipped: List[Dict[str, str]] = []

    for exp in exps:
        try:
            res = run_final_suite_single(
                root,
                exp,
                top_n=top_n,
                sort_metric=sort_metric,
                include_test_metrics=include_test_metrics,
                apply_label_map=apply_label_map,
                topn_per_architecture=topn_per_architecture,
                prefer_metric_for_order=prefer_metric_for_order,
                fallback_metric_for_order=fallback_metric_for_order,
            )
        except Exception as e:
            log.warning("[run_final_suite_multi] Skipping EXP=%s due to error: %s", exp, e)
            skipped.append({"exp": exp, "error": str(e)})
            continue

        table1 = res["table1"]
        table2 = res["table2"]

        # Split metadata from YAMLs
        df_splits = read_campaign_splits(root, exp)
        split_info = reduce_campaign_split(df_splits, dominant_threshold=dominant_threshold)

        # Attach metadata to table2 (do not change how table2 is computed)
        t2 = table2.copy()

        # Insert meta columns at the front (stable, readable)
        def _insert_front(df: pd.DataFrame, name: str, value: Any) -> None:
            if name in df.columns:
                return
            df.insert(0, name, value)

        _insert_front(t2, "exp", exp)
        _insert_front(t2, "split_id", split_info.get("split_id", "unknown"))
        _insert_front(t2, "split_mode", split_info.get("split_mode", "unknown"))

        if split_info.get("split_mode") == "campaign":
            _insert_front(t2, "val_size", split_info.get("val_size", np.nan))
            _insert_front(t2, "test_size", split_info.get("test_size", np.nan))
        elif split_info.get("split_mode") == "family":
            # Ensure consistent schema (all families appear as columns)
            for fam in ("seq2", "arps", "darts"):
                _insert_front(t2, f"val_size_{fam}", split_info.get(f"val_size_{fam}", np.nan))
                _insert_front(t2, f"test_size_{fam}", split_info.get(f"test_size_{fam}", np.nan))
            log.warning(
                "[run_final_suite_multi] EXP=%s has mixed splits; using per-family columns (table2).", exp
            )
        else:
            log.warning("[run_final_suite_multi] EXP=%s split metadata not found; marking as unknown.", exp)

        tables1_by_exp.append((exp, table1))
        tables2_by_exp.append((exp, t2))

    if tables2_by_exp:
        table2_unified = pd.concat([t for _, t in tables2_by_exp], ignore_index=True)
    else:
        table2_unified = pd.DataFrame()

    # Global unified output (single location)
    global_out_dir = root / "src" / "experiment_configs" / "hpo_campaigns" / "final_suite_outputs"
    global_out_dir.mkdir(parents=True, exist_ok=True)
    unified_path = global_out_dir / "table2_unified.csv"
    table2_unified.to_csv(unified_path, index=False)
    log.info("Written unified table2: %s", str(unified_path))

    return {
        "tables1_by_exp": tables1_by_exp,
        "tables2_by_exp": tables2_by_exp,
        "table2_unified": table2_unified,
        "skipped_exps": skipped,
        "unified_csv_path": str(unified_path),
    }


import logging
import re
from typing import Any, Mapping, Sequence

def resolve_arps_pure_experiments_from_positions(
    campaign_splits: Mapping[str, Sequence[str]],
    pure_position_by_split: Mapping[str, int | None],
    *,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    """
    Resolve the experiment name that represents the ARPS Pure baseline for each split,
    using positional rules (e.g., "first experiment in split_0.40").

    Parameters
    ----------
    campaign_splits
        Mapping like {"split_0.40": ["HPO_100...", "HPO_54...", ...], ...}.
        IMPORTANT: pass only real split groups (do not include "all").
    pure_position_by_split
        Mapping like {"split_0.40": 0, "split_0.45": None, ...}.
        - int => index of the experiment in that split list that should be ARPS Pure
        - None => split currently has no ARPS Pure experiment
    logger
        Optional logger for warnings.

    Returns
    -------
    dict[str, str]
        Mapping split_key -> exp_name for ARPS Pure rows.
    """
    log = logger or logging.getLogger(__name__)
    resolved: dict[str, str] = {}

    for split_key, pos in pure_position_by_split.items():
        if pos is None:
            continue

        exp_list = campaign_splits.get(split_key)
        if not exp_list:
            log.warning("Split '%s' not found in campaign_splits. Skipping ARPS Pure resolution.", split_key)
            continue

        if not (0 <= pos < len(exp_list)):
            log.warning(
                "Invalid ARPS Pure position for split '%s': pos=%s (len=%d). Skipping.",
                split_key, pos, len(exp_list)
            )
            continue

        resolved[split_key] = exp_list[pos]

    return resolved


def build_experiment_story_table(
    table: pd.DataFrame,
    *,
    campaign_splits: Mapping[str, Sequence[str]],
    arps_pure_exp_by_split: Mapping[str, str] | None = None,
    exp_column: str = "exp",
    architecture_column: str = "Architecture",
    drop_columns: Sequence[str] = ("split_mode", "split_id"),
    rename_exp_to: str = "HPO Trials",
    arps_raw_label: str = "ARPS",
    pinn_raw_label: str = "PINN",
    arps_pure_label: str = "Arps Pure",
    arps_ensemble_label: str = "Arps Ensemble",
    pinn_analytical_label: str = "PINN Analytical",
    sort_rows: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Create a storytelling-ready version of the unified final-suite table without modifying
    the original input table.

    What this function does
    -----------------------
    1) Drops non-informative columns (e.g., split_mode, split_id)
    2) Converts `exp` -> `HPO Trials` (extracts the trials count from strings like HPO_150_...)
    3) Remaps `Architecture` labels:
       - PINN -> PINN Analytical
       - ARPS -> Arps Ensemble (default)
       - ARPS -> Arps Pure only when the experiment matches `arps_pure_exp_by_split[split]`
    4) Optionally sorts rows to improve readability / narrative flow

    Parameters
    ----------
    table
        Original DataFrame (e.g., `table2_unified` from run_final_suite_multi).
    campaign_splits
        Mapping split -> list of experiment names used to infer split context from `exp`.
        IMPORTANT: pass only real split groups (do not include "all").
    arps_pure_exp_by_split
        Optional mapping split -> experiment name that represents ARPS Pure.
        Any ARPS row not matched here is labeled as Arps Ensemble.
    exp_column
        Name of the experiment column in the input table.
    architecture_column
        Name of the architecture column in the input table.
    drop_columns
        Columns to remove if present.
    rename_exp_to
        New column name for the extracted HPO trial count.
    ...
    sort_rows
        If True, sort by test_size, val_size, HPO Trials, and architecture narrative order.
    logger
        Optional logger.

    Returns
    -------
    pd.DataFrame
        New transformed DataFrame. Original input is not modified.
    """
    log = logger or logging.getLogger(__name__)

    if not isinstance(table, pd.DataFrame):
        raise TypeError("`table` must be a pandas DataFrame.")

    required_cols = {exp_column, architecture_column}
    missing_required = [c for c in required_cols if c not in table.columns]
    if missing_required:
        raise KeyError(f"Missing required columns in table: {missing_required}")

    df = table.copy(deep=True)

    # ---------------------------------------------------------------------
    # Build experiment -> split lookup (ignore duplicated experiment names if any)
    # ---------------------------------------------------------------------
    exp_to_split: dict[str, str] = {}
    for split_key, exp_list in campaign_splits.items():
        for exp_name in exp_list:
            if exp_name in exp_to_split and exp_to_split[exp_name] != split_key:
                log.warning(
                    "Experiment '%s' appears in multiple splits (%s, %s). "
                    "Keeping first occurrence.",
                    exp_name, exp_to_split[exp_name], split_key
                )
                continue
            exp_to_split[exp_name] = split_key

    arps_pure_exp_by_split = dict(arps_pure_exp_by_split or {})

    # ---------------------------------------------------------------------
    # Helper columns: split context + parsed HPO trials
    # ---------------------------------------------------------------------
    df["_split_key"] = df[exp_column].map(exp_to_split)

    # Flexible parser: captures the number in "HPO_<n>_..."
    trials_extracted = (
        df[exp_column]
        .astype(str)
        .str.extract(r"HPO_(\d+)", expand=False)
    )
    df["_hpo_trials_num"] = pd.to_numeric(trials_extracted, errors="coerce")

    # Replace exp column with numeric HPO Trials when available, else keep original text
    df[rename_exp_to] = df["_hpo_trials_num"].where(df["_hpo_trials_num"].notna(), df[exp_column])

    # If fully numeric, cast to pandas nullable integer for cleaner display
    if df["_hpo_trials_num"].notna().all():
        df[rename_exp_to] = df["_hpo_trials_num"].astype("Int64")

    # ---------------------------------------------------------------------
    # Architecture remapping with split-aware ARPS Pure vs Ensemble logic
    # ---------------------------------------------------------------------
    def _map_architecture(row: pd.Series) -> str:
        raw_arch = str(row.get(architecture_column, "")).strip()
        exp_name = str(row.get(exp_column, ""))
        split_key = row.get("_split_key")

        if raw_arch == pinn_raw_label:
            return pinn_analytical_label

        if raw_arch == arps_raw_label:
            pure_exp_for_split = arps_pure_exp_by_split.get(split_key)
            if pure_exp_for_split is not None and exp_name == pure_exp_for_split:
                return arps_pure_label
            return arps_ensemble_label

        # Unknown / future architecture => preserve original label
        return raw_arch

    df[architecture_column] = df.apply(_map_architecture, axis=1)

    # ---------------------------------------------------------------------
    # Drop / rename columns
    # ---------------------------------------------------------------------
    columns_to_drop = [c for c in drop_columns if c in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # Drop original exp column after creating HPO Trials
    if exp_column in df.columns:
        df = df.drop(columns=[exp_column])

    # Insert HPO Trials near the front (where exp used to be conceptually)
    desired_front_order = []
    for candidate in ["test_size", "val_size", rename_exp_to, architecture_column]:
        if candidate in df.columns:
            desired_front_order.append(candidate)

    remaining_cols = [c for c in df.columns if c not in desired_front_order and not c.startswith("_")]
    df = df[desired_front_order + remaining_cols]

    # ---------------------------------------------------------------------
    # Narrative sorting (optional)
    # ---------------------------------------------------------------------
    if sort_rows:
        arch_order = {
            arps_pure_label: 0,
            arps_ensemble_label: 1,
            pinn_analytical_label: 2,
        }
        df["_arch_order"] = df[architecture_column].map(arch_order).fillna(99)

        sort_cols: list[str] = []
        for c in ["test_size", "val_size"]:
            if c in df.columns:
                sort_cols.append(c)

        # Use numeric helper for stable ordering even when display column may contain mixed types
        sort_cols.extend(["_hpo_trials_num", "_arch_order"])

        existing_sort_cols = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(existing_sort_cols, kind="stable").reset_index(drop=True)

        df = df.drop(columns=[c for c in ["_arch_order"] if c in df.columns])

    # Final cleanup of private helper columns
    df = df.drop(columns=[c for c in ["_split_key", "_hpo_trials_num"] if c in df.columns])

    return df