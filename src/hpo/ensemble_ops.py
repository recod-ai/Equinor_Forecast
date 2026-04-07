# src/hpo/ensemble_ops.py
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
log = logging.getLogger("ensemble")


# --- small helpers ------------------------------------------------------------
def _require_split_col(df: pd.DataFrame, df_name: str):
    if "split" not in df.columns:
        raise ValueError(f"[ensemble] '{df_name}' is missing the 'split' column. "
                         f"Check reader enrichment & boundaries manifest.")

def _arch_family_from_name(s: pd.Series) -> pd.Series:
    """
    Normaliza rótulos de arquitetura em famílias canônicas:
      - 'arps*'      → 'arps'
      - 'darts*'     → 'darts'
      - 'seq2*'      → 'seq2pin'  (colapsa seq2, seq2pin, seq2trend, seq2context, etc.)
    Mantém o índice original da Series.
    """
    x = s.astype(str).str.lower()

    # remove separadores/ruídos leves para facilitar matching (ex.: "seq2-pin", "arps_canonical")
    x_squash = x.str.replace(r"[\s\-_]+", "", regex=True)

    fam = np.where(
        x_squash.str.contains("arps"),
        "arps",
        np.where(
            x_squash.str.contains("darts?"),  # cobre 'dart'/'darts'
            "darts",
            np.where(
                x_squash.str.contains("seq2"),
                "seq2pin",  # decisão: colapsar todas as variantes em 'seq2pin'
                x  # fallback: devolve o texto original normalizado (lower), se nenhuma regra bater
            )
        )
    )
    return pd.Series(fam, index=s.index)


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")

def _weighted_mean_and_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """
    Compute weighted mean and weighted standard deviation (population version)
    for 1D arrays. Assumes weights are non-negative and sum to 1 (we normalize
    before using).
    """
    if values.size == 0:
        return np.nan, np.nan
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    if w.sum() <= 0 or v.size != w.size:
        return np.nan, np.nan
    w = w / w.sum()
    mean = np.sum(w * v)
    var = np.sum(w * (v - mean) ** 2)
    return float(mean), float(np.sqrt(var))


# --- PUBLIC API ---------------------------------------------------------------

def build_intra_family_ensemble(
    champions_df: pd.DataFrame,
    series_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each (well, family, split), compute a performance-weighted average forecast
    across champions (campaigns) belonging to that family. Also compute a weighted
    std at each horizon index to quantify intra-family uncertainty.

    Returns one row per (well, arch, split, idx) with:
      ['split','well','arch','t','idx','yhat_family_mean','std_family','ytrue','val_smape_agg_family_mean']
    """
    # Expected inputs
    _ensure_cols(champions_df, ["job_hash", "well", "val_smape_agg"])
    # champions_df can have 'architecture' or 'architecture_name'; we normalize to 'arch_family'
    arch_col_ch = (
        "arch" if "arch" in champions_df.columns else
        "architecture" if "architecture" in champions_df.columns else
        "architecture_name" if "architecture_name" in champions_df.columns else None
    )
    if arch_col_ch is None:
        raise KeyError("Missing arch/architecture column in champions_df")

    _ensure_cols(series_df, ["job_hash", "well", "arch", "idx", "yhat"])
    # Keep 't' and 'ytrue' if available
    if "t" not in series_df.columns:
        series_df = series_df.assign(t=lambda d: d["idx"])  # fallback
    if "ytrue" not in series_df.columns:
        series_df = series_df.assign(ytrue=np.nan)

    # Normalize architectures to families on both sides
    champs = champions_df.copy()
    champs["arch_family"] = _arch_family_from_name(champs[arch_col_ch])

    sr = series_df.copy()
    sr["arch_family"] = sr["arch"].astype(str).str.lower()
    # Common aliases
    sr["arch_family"] = sr["arch_family"].replace({
        "arps_canonical": "arps",
        "arpscanonical": "arps",
        "seq2pin": "seq2pin",
        "seq2trend": "seq2trend",
        "seq2context": "seq2context"
    })
    # Collapse seq2* families if desired; for now we keep 'seq2pin' as in your store.
    sr["arch_family"] = sr["arch_family"].replace({
        "arps": "arps",
        "seq2pin": "seq2pin",
        "darts": "darts"
    })

    # Prefer Stage-1 split from champions (series_df may have 'unknown')
    split_col = "split" if "split" in champs.columns else None
    if split_col is None:
        champs["split"] = "T_current"  # safe default
        split_col = "split"

    # Merge by job_hash (most robust join)
    merged = sr.merge(
        champs[["job_hash", "well", "val_smape_agg", "arch_family", split_col]],
        on="job_hash",
        how="inner",
        suffixes=("", "_ch")
    )

    _require_split_col(merged, "series_df×champions (merged)")

    if merged.empty:
        # Return schema-correct empty frame
        return pd.DataFrame(columns=[
            "split","well","arch","t","idx","yhat_family_mean","std_family","ytrue",
            "val_smape_agg_family_mean"
        ])

    # Raw weights (higher = better)
    merged["_w_raw"] = 1.0 / (merged["val_smape_agg"].astype(float) + 1e-9)

    # Normalize weights WITHIN (well, arch_family, split) – so per split we get sum=1
    grp_w = merged.groupby(["well", "arch_family", split_col], sort=False)["_w_raw"].transform("sum")
    merged["_w"] = np.where(grp_w > 0, merged["_w_raw"] / grp_w, 0.0)

    # Aggregate to intra-family mean & std per (well, arch, split, idx)
    def _agg_block(g: pd.DataFrame) -> pd.Series:
        m, s = _weighted_mean_and_std(g["yhat"].to_numpy(), g["_w"].to_numpy())
        # ytrue should be identical across members; take first non-null
        ytrue = g["ytrue"].dropna().iloc[0] if g["ytrue"].notna().any() else np.nan
        # take any representative t (idx-aligned)
        t = g["t"].iloc[0]
        return pd.Series({
            "t": t,
            "yhat_family_mean": m,
            "std_family": s,
            "ytrue": ytrue
        })

    agg = (
        merged
        .groupby(["well", "arch_family", split_col, "idx"], sort=False, as_index=False)
        .apply(_agg_block)
        .reset_index(drop=True)
    )

    # Add a per-(well, arch_family, split) family quality score (mean of val_smape_agg)
    fam_quality = (
        merged
        .drop_duplicates(subset=["job_hash"])  # each champion appears many times across idx
        .groupby(["well", "arch_family", split_col], sort=False)["val_smape_agg"]
        .mean()
        .reset_index()
        .rename(columns={"val_smape_agg": "val_smape_agg_family_mean"})
    )

    out = agg.merge(fam_quality, on=["well", "arch_family", split_col], how="left")
    # Final tidy columns
    out = out.rename(columns={
        "arch_family": "arch",
        split_col: "split"
    })[["split","well","arch","t","idx","yhat_family_mean","std_family","ytrue","val_smape_agg_family_mean"]]

    try:
        rows_val  = int(out.loc[out["split"].astype(str).str.lower().eq("val")].shape[0])
        rows_test = int(out.loc[out["split"].astype(str).str.lower().eq("test")].shape[0])
        n_arch    = int(out["arch"].nunique())
        log.info(f"✅ [ensemble] intra: val={rows_val}, test={rows_test}, families={n_arch}")
    except Exception:
        pass

    return out


def build_inter_family_ensemble(
    intra_family_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combina os forecasts por família em um meta-ensemble por (well, split, idx).
    Alphas ∝ 1 / val_smape_agg_family_mean, normalizados por (well, split, idx).
    Var total = E[Var intra] + Var(E inter).

    Saída (uma linha por (well, split, idx)):
      ['split','well','t','idx','yhat_final_mean','std_final','ytrue']
    """
    _ensure_cols(intra_family_df, [
        "split","well","arch","idx","t","yhat_family_mean","std_family","val_smape_agg_family_mean"
    ])

    df = intra_family_df.copy()
    _require_split_col(df, "intra_family_df")

    # 0) Dedup mínimo por (well, split, arch, idx) para evitar alphas duplicados no mesmo ponto
    df = (
        df.sort_values(["well","split","arch","idx"])  # estável
          .drop_duplicates(subset=["well","split","arch","idx"], keep="last")
          .reset_index(drop=True)
    )

    # 1) Pesos "alpha" ∝ 1 / smape (robusto a NaN/inf)
    smape = pd.to_numeric(df["val_smape_agg_family_mean"], errors="coerce")
    smape = smape.replace([np.inf, -np.inf], np.nan)

    # Evita zero/negativo: desloca mínimo para garantir > 0 onde possível
    # (só afeta casos degenerados; em dados normais não muda nada)
    eps = 1e-9
    smape_safe = smape.copy()
    if smape_safe.notna().any():
        mpos = smape_safe[smape_safe > 0].min()
        if pd.notna(mpos):
            smape_safe = smape_safe.fillna(mpos)  # se algum NaN, usa o menor positivo observado
            smape_safe = np.where(smape_safe <= 0, mpos, smape_safe)
        else:
            # se tudo <= 0 / NaN (patológico), cai para 1.0
            smape_safe = smape_safe.fillna(1.0)
            smape_safe = np.where(pd.Series(smape_safe) <= 0, 1.0, smape_safe)
    else:
        smape_safe = np.ones(len(df), dtype=float)

    df["_alpha_raw"] = 1.0 / (np.asarray(smape_safe, dtype=float) + eps)

    # 2) Normaliza por (well, split, idx)
    grp_sum = df.groupby(["well","split","idx"], sort=False)["_alpha_raw"].transform("sum")
    df["_alpha"] = np.where(grp_sum > 0, df["_alpha_raw"] / grp_sum, 0.0)

    # 3) Agrega para meta-ensemble
    def _agg_meta(g: pd.DataFrame) -> pd.Series:
        a  = g["_alpha"].to_numpy(dtype=float)
        mu = g["yhat_family_mean"].to_numpy(dtype=float)
        s  = g["std_family"].to_numpy(dtype=float)
        sf2 = s * s

        asum = a.sum()
        if asum <= 0 or not np.isfinite(asum):
            # caso degenerado: pesos nulos — use média simples
            a = np.ones_like(mu, dtype=float)
            asum = a.sum()
        a = a / asum

        yhat_mean = float(np.sum(a * mu))
        within = float(np.sum(a * sf2))                     # E[Var_intra]
        between = float(np.sum(a * (mu - yhat_mean) ** 2))  # Var(E_inter)
        total_var = within + between
        total_var = max(total_var, 0.0)                     # robustez numérica
        total_std = float(np.sqrt(total_var))

        # ytrue: pega o primeiro não-nulo; se múltiplos e diferentes, manter o primeiro é suficiente p/plot
        ytrue = g["ytrue"].dropna().iloc[0] if g["ytrue"].notna().any() else np.nan
        t = g["t"].iloc[0]
        return pd.Series({
            "t": t,
            "yhat_final_mean": yhat_mean,
            "std_final": total_std,
            "ytrue": ytrue
        })

    out = (
        df.groupby(["well","split","idx"], sort=False, as_index=False)
          .apply(_agg_meta)
          .reset_index(drop=True)
          .sort_values(["well","split","idx"])
    )

    try:
        rows_val  = int(out.loc[out["split"].astype(str).str.lower().eq("val")].shape[0])
        rows_test = int(out.loc[out["split"].astype(str).str.lower().eq("test")].shape[0])
        log.info(f"✅ [ensemble] inter: final rows val={rows_val}, test={rows_test}")
    except Exception:
        pass

    return out[["split","well","t","idx","yhat_final_mean","std_final","ytrue"]]

