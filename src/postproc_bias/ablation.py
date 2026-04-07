from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Public runner API
# -----------------------------------------------------------------------------

def run_anchor_policy_ablation(
    *,
    cfg: Any,
    policy_names: Sequence[str],
    stage2_bundle: Optional[Mapping[str, Any]] = None,
    save_dir: Optional[str | Path] = None,
    selection_metric: str = "val_metric_inter",
    tie_break_metric: str = "val_risk_q90",
    prefer_lower_tie_break: bool = True,
) -> Dict[str, Any]:
    """
    Main notebook-facing ablation runner.

    Flow:
      1) load Stage 2 once (unless stage2_bundle was provided)
      2) replay Phase 4 for each anchor policy
      3) consolidate well × policy table
      4) choose selected policy per well using validation only
      5) optionally save artifacts

    Returns:
      {
        "stage2_bundle": ...,
        "comparison_df": ...,
        "selected_df": ...,
        "artifacts_by_policy": ...,
      }
    """
    from common.phase_orchestrator import (
        run_phase4_until_stage2,
        run_phase4_from_loaded_stage2,
    )

    if stage2_bundle is None:
        stage2_bundle = run_phase4_until_stage2(cfg)

    champs = stage2_bundle["champions_df"]
    series_df = stage2_bundle["series_df"]
    boundaries_df = stage2_bundle["boundaries_df"]
    full_history_by_well = stage2_bundle["full_history_by_well"]
    score_col = stage2_bundle.get("score_col")

    artifacts_by_policy: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for policy_name in policy_names:
        art = run_phase4_from_loaded_stage2(
            cfg=cfg,
            champions_df=champs,
            series_df=series_df,
            boundaries_df=boundaries_df,
            full_history_by_well=full_history_by_well,
            anchor_config=str(policy_name),
            score_col=score_col,
        )
        artifacts_by_policy[str(policy_name)] = art

        policy_rows = _build_policy_rows(
            artifacts=art,
            policy_name=str(policy_name),
            cfg=cfg,
        )
        rows.extend(policy_rows)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        selected_df = pd.DataFrame(
            columns=[
                "well",
                "selected_policy_name",
                "selection_metric",
                "selection_value",
                "selection_tie_break_metric",
                "selection_tie_break_value",
                "campaign_group",
                "split_tag",
                "selection_reason",
            ]
        )
    else:
        selected_df = select_best_policy_per_well(
            comparison_df=comparison_df,
            selection_metric=selection_metric,
            tie_break_metric=tie_break_metric,
            prefer_lower_tie_break=prefer_lower_tie_break,
            campaign_group=_infer_campaign_group(cfg),
            split_tag=_infer_split_tag(cfg),
        )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(save_dir / "anchor_policy_comparison.csv", index=False)
        selected_df.to_csv(save_dir / "selected_anchor_policies.csv", index=False)

    return dict(
        stage2_bundle=stage2_bundle,
        comparison_df=comparison_df,
        selected_df=selected_df,
        artifacts_by_policy=artifacts_by_policy,
    )


# -----------------------------------------------------------------------------
# Selection
# -----------------------------------------------------------------------------

def select_best_policy_per_well(
    *,
    comparison_df: pd.DataFrame,
    selection_metric: str = "val_metric_inter",
    tie_break_metric: str = "val_risk_q90",
    prefer_lower_tie_break: bool = True,
    campaign_group: Optional[str] = None,
    split_tag: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validation-only selection:
      primary: selection_metric
      tie-break: tie_break_metric
      test columns are ignored for selection
    """
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame()

    df = comparison_df.copy()

    required = ["well", "policy_name", selection_metric]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required selection columns: {missing}")

    df["_sel"] = pd.to_numeric(df[selection_metric], errors="coerce")
    df["_tie"] = pd.to_numeric(df.get(tie_break_metric, np.nan), errors="coerce")

    # lower is better for both by default
    tie_sign = 1.0 if prefer_lower_tie_break else -1.0

    picks = []
    for well, sub in df.groupby("well", dropna=False):
        local = sub.copy()

        # primary sort: smaller selection metric is better
        # secondary sort: smaller tie-break metric is better
        local = local.sort_values(
            by=["_sel", "_tie", "policy_name"],
            ascending=[True, True if tie_sign > 0 else False, True],
            kind="stable",
        )

        best = local.iloc[0].copy()
        picks.append(
            dict(
                well=str(best["well"]),
                selected_policy_name=str(best["policy_name"]),
                selection_metric=str(selection_metric),
                selection_value=_safe_float(best["_sel"]),
                selection_tie_break_metric=str(tie_break_metric),
                selection_tie_break_value=_safe_float(best["_tie"]),
                campaign_group=str(campaign_group) if campaign_group is not None else None,
                split_tag=str(split_tag) if split_tag is not None else None,
                selection_reason=(
                    f"best {selection_metric} on validation; "
                    f"tie-break={tie_break_metric}; "
                    f"test used only for audit"
                ),
            )
        )

    out = pd.DataFrame(picks).sort_values("well", kind="stable").reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Comparison table builder
# -----------------------------------------------------------------------------

def _build_policy_rows(
    *,
    artifacts: Mapping[str, Any],
    policy_name: str,
    cfg: Any,
) -> List[Dict[str, Any]]:
    """
    Build a compact comparison table with one row per well × policy.

    Expected columns:
      well
      policy_name
      val_metric_inter
      val_metric_intra_best
      val_risk_q50
      val_risk_q90
      test_metric_inter
      test_risk_q50
      test_risk_q90
    """
    final_ensemble_df = _ensure_df(artifacts.get("final_ensemble_df"))
    intra_family_df = _ensure_df(artifacts.get("intra_family_df"))
    risk_df = _ensure_df(artifacts.get("risk_df"))

    inter_metrics = _extract_inter_metrics_by_well(final_ensemble_df)
    intra_metrics = _extract_best_intra_metrics_by_well(intra_family_df)
    risk_metrics = _extract_risk_metrics_by_well(risk_df)

    wells = sorted(
        set(inter_metrics.keys()) |
        set(intra_metrics.keys()) |
        set(risk_metrics.keys())
    )

    rows: List[Dict[str, Any]] = []
    for well in wells:
        inter = inter_metrics.get(well, {})
        intra = intra_metrics.get(well, {})
        risk = risk_metrics.get(well, {})

        rows.append(
            dict(
                well=str(well),
                policy_name=str(policy_name),
                val_metric_inter=_safe_float(inter.get("val_metric")),
                test_metric_inter=_safe_float(inter.get("test_metric")),
                val_metric_intra_best=_safe_float(intra.get("val_metric_best")),
                test_metric_intra_best=_safe_float(intra.get("test_metric_of_best_val")),
                val_risk_q50=_safe_float(risk.get("val_risk_q50")),
                val_risk_q90=_safe_float(risk.get("val_risk_q90")),
                test_risk_q50=_safe_float(risk.get("test_risk_q50")),
                test_risk_q90=_safe_float(risk.get("test_risk_q90")),
                n_members_total=_safe_int(inter.get("n_members_total")),
                n_families_seen=_safe_int(risk.get("n_families_seen")),
            )
        )

    return rows


# -----------------------------------------------------------------------------
# Metrics extractors
# -----------------------------------------------------------------------------

def _series_smape(yhat: Any, ytrue: Any, eps: float = 1e-8) -> float:
    yhat = pd.to_numeric(pd.Series(yhat), errors="coerce").to_numpy(dtype=float)
    ytrue = pd.to_numeric(pd.Series(ytrue), errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(yhat) & np.isfinite(ytrue)
    if not mask.any():
        return np.nan

    yhat = yhat[mask]
    ytrue = ytrue[mask]

    denom = np.abs(yhat) + np.abs(ytrue)
    denom = np.maximum(denom, eps)

    smape = 200.0 * np.mean(np.abs(yhat - ytrue) / denom)
    return float(smape)


def _series_mae(yhat: Any, ytrue: Any) -> float:
    yhat = pd.to_numeric(pd.Series(yhat), errors="coerce").to_numpy(dtype=float)
    ytrue = pd.to_numeric(pd.Series(ytrue), errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(yhat) & np.isfinite(ytrue)
    if not mask.any():
        return np.nan

    return float(np.mean(np.abs(yhat[mask] - ytrue[mask])))

def _extract_inter_metrics_by_well(final_ensemble_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute post-ensemble metrics directly from the final ensemble series.

    Expected final schema in current Phase 4:
      - split
      - well
      - yhat_final_mean
      - ytrue
      - n_members_total (or n_members)
    """
    if final_ensemble_df is None or final_ensemble_df.empty or "well" not in final_ensemble_df.columns:
        return {}

    yhat_col = "yhat_final_mean" if "yhat_final_mean" in final_ensemble_df.columns else None
    if yhat_col is None or "ytrue" not in final_ensemble_df.columns or "split" not in final_ensemble_df.columns:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for well, sub in final_ensemble_df.groupby("well", dropna=False):
        val_sub = sub.loc[sub["split"].astype(str).str.lower().eq("val")].copy()
        test_sub = sub.loc[sub["split"].astype(str).str.lower().eq("test")].copy()

        out[str(well)] = dict(
            val_metric=_series_smape(val_sub[yhat_col], val_sub["ytrue"]) if not val_sub.empty else np.nan,
            test_metric=_series_smape(test_sub[yhat_col], test_sub["ytrue"]) if not test_sub.empty else np.nan,
            n_members_total=_infer_members_total(sub),
        )
    return out


def _extract_best_intra_metrics_by_well(intra_family_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    For each well, pick the best family using validation-only metrics computed
    from yhat_family_mean vs ytrue.
    """
    if intra_family_df is None or intra_family_df.empty or "well" not in intra_family_df.columns:
        return {}

    yhat_col = "yhat_family_mean" if "yhat_family_mean" in intra_family_df.columns else None
    if yhat_col is None or "ytrue" not in intra_family_df.columns or "split" not in intra_family_df.columns:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for well, sub in intra_family_df.groupby("well", dropna=False):
        family_rows = []
        for arch, fam in sub.groupby("arch", dropna=False):
            val_sub = fam.loc[fam["split"].astype(str).str.lower().eq("val")].copy()
            test_sub = fam.loc[fam["split"].astype(str).str.lower().eq("test")].copy()

            family_rows.append(
                dict(
                    arch=str(arch),
                    val_metric=_series_smape(val_sub[yhat_col], val_sub["ytrue"]) if not val_sub.empty else np.nan,
                    test_metric=_series_smape(test_sub[yhat_col], test_sub["ytrue"]) if not test_sub.empty else np.nan,
                )
            )

        fam_df = pd.DataFrame(family_rows)
        fam_df = fam_df.sort_values(["val_metric", "arch"], ascending=[True, True], kind="stable")

        if fam_df.empty:
            continue

        best = fam_df.iloc[0]
        out[str(well)] = dict(
            val_metric_best=_safe_float(best.get("val_metric")),
            test_metric_of_best_val=_safe_float(best.get("test_metric")),
            best_family=str(best.get("arch")) if "arch" in best.index else None,
        )
    return out


def _extract_risk_metrics_by_well(risk_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Current risk in Phase 4 is family-level and simple:
      - VAL = full
      - TEST = head(H)
    We aggregate to one row per well for ablation ranking.

    Heuristic used in v1:
      - validation risk: take the largest available q50/q90 across family rows
      - test risk: same, prioritizing the longest finite horizon if present,
        otherwise use whatever exists
    This is conservative and avoids over-selling the current risk layer.
    """
    if risk_df is None or risk_df.empty or "well" not in risk_df.columns:
        return {}

    df = risk_df.copy()
    if "split_selector" in df.columns:
        df["split_selector"] = df["split_selector"].astype(str).str.lower()

    out: Dict[str, Dict[str, Any]] = {}
    for well, sub in df.groupby("well", dropna=False):
        val_sub = _subset_risk_for_selector(sub, preferred=("val", "validation", "val+test"))
        test_sub = _subset_risk_for_selector(sub, preferred=("test", "val+test"))

        test_sub = _prefer_longest_horizon(test_sub)

        out[str(well)] = dict(
            val_risk_q50=_safe_float(_colmax(val_sub, "q50")),
            val_risk_q90=_safe_float(_colmax(val_sub, "q90")),
            test_risk_q50=_safe_float(_colmax(test_sub, "q50")),
            test_risk_q90=_safe_float(_colmax(test_sub, "q90")),
            n_families_seen=_safe_int(test_sub["arch"].nunique()) if "arch" in test_sub.columns and not test_sub.empty else np.nan,
        )
    return out


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _ensure_df(obj: Any) -> pd.DataFrame:
    return obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame()


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _safe_int(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return np.nan
        return int(x)
    except Exception:
        return np.nan


def _colmax(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if len(vals) else np.nan


def _pick_first_existing_pair(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
) -> Tuple[Optional[str], Optional[str]]:
    cols = set(df.columns)
    for a, b in pairs:
        if a in cols:
            return a, b if b in cols else None
    return None, None


def _infer_members_total(df: pd.DataFrame) -> float:
    for col in ("n_members_total", "n_members"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                return int(vals.iloc[0])
    return np.nan


def _subset_risk_for_selector(df: pd.DataFrame, preferred: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "split_selector" not in df.columns:
        return df.copy()

    local = df.copy()
    selector = local["split_selector"].astype(str).str.lower()

    for name in preferred:
        mask = selector.eq(str(name).lower())
        if mask.any():
            return local.loc[mask].copy()

    # fallback: contains
    for name in preferred:
        mask = selector.str.contains(str(name).lower(), regex=False, na=False)
        if mask.any():
            return local.loc[mask].copy()

    return local.copy()


def _prefer_longest_horizon(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "horizon_days" not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    local = df.copy()
    h = pd.to_numeric(local["horizon_days"], errors="coerce")

    finite = local.loc[np.isfinite(h) & (h >= 0)].copy()
    if not finite.empty:
        max_h = float(pd.to_numeric(finite["horizon_days"], errors="coerce").max())
        return finite.loc[pd.to_numeric(finite["horizon_days"], errors="coerce").eq(max_h)].copy()

    # keep all if only "-1" or malformed values exist
    return local


def _infer_campaign_group(cfg: Any) -> Optional[str]:
    mp = getattr(cfg, "campaigns_to_ensemble", None)
    if isinstance(mp, dict) and len(mp):
        return str(next(iter(mp.values())))
    return None


def _infer_split_tag(cfg: Any) -> Optional[str]:
    mp = getattr(cfg, "campaigns_to_ensemble", None)
    if isinstance(mp, dict) and len(mp):
        return str(next(iter(mp.keys())))
    return None
