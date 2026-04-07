# src/postproc_bias/anchoring.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Stage 2.6 — History Anchoring
# =============================================================================

@dataclass(frozen=True)
class HistoryAnchorConfig:
    """
    Config for the new Stage 2.6 replay anchoring.

    Methodological rule:
      - choose policy on validation
      - audit on test
      - on test, keep the same policy but recompute the anchor with the
        causal history available up to the test boundary
    """
    policy_name: str = "baseline_default"
    enabled: bool = True

    # Where the replay correction is applied
    target_splits: tuple[str, ...] = ("val", "test")
    correction_mode: str = "offset"   # offset | offset_decay | scale
    decay_halflife: float = 45.0

    # IO contract
    yhat_col: str = "yhat"
    split_col: str = "split"
    well_col: str = "well"
    idx_col: str = "idx"

    # Prediction reference at split start
    prediction_ref_mode: str = "head_median"   # first | head_median
    prediction_ref_points: int = 7

    # History estimator controls
    min_history_points: int = 15
    q0_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Safety
    clip_min: Optional[float] = 0.0
    clip_scale_range: tuple[float, float] = (0.25, 4.0)
    preserve_train: bool = True
    attach_metadata: bool = True


# =============================================================================
# Policy registry
# =============================================================================

def make_history_anchor_registry() -> Dict[str, HistoryAnchorConfig]:
    """
    Explicit registry for notebook ablations and later replay.
    All q0_kwargs below are compatible with forecast_pipeline.arps_offline.compute_history_q0_phys.
    """
    registry = {
        "baseline_default": HistoryAnchorConfig(
            policy_name="baseline_default",
            correction_mode="offset",
            prediction_ref_mode="head_median",
            prediction_ref_points=7,
            q0_kwargs=dict(
                window=21,
                detect_window=60,
                trend_window=60,
                kind="median",
                use_log_space=False,
            ),
        ),
        "short_window_spike_sensitive": HistoryAnchorConfig(
            policy_name="short_window_spike_sensitive",
            correction_mode="offset",
            prediction_ref_mode="head_median",
            prediction_ref_points=5,
            q0_kwargs=dict(
                window=10,
                detect_window=30,
                trend_window=30,
                kind="median",
                use_log_space=False,
                clip_sigma_high=1.0,
            ),
        ),
        "long_window_smoother": HistoryAnchorConfig(
            policy_name="long_window_smoother",
            correction_mode="offset_decay",
            decay_halflife=60.0,
            prediction_ref_mode="head_median",
            prediction_ref_points=9,
            q0_kwargs=dict(
                window=45,
                detect_window=80,
                trend_window=80,
                kind="median",
                use_log_space=False,
            ),
        ),
        "trend_strict": HistoryAnchorConfig(
            policy_name="trend_strict",
            correction_mode="offset_decay",
            decay_halflife=30.0,
            prediction_ref_mode="first",
            prediction_ref_points=1,
            q0_kwargs=dict(
                window=21,
                detect_window=60,
                trend_window=60,
                kind="last",
                use_log_space=False,
                clip_sigma_high=0.9,
                weight_floor=0.03,
            ),
        ),
        "log_space_robust": HistoryAnchorConfig(
            policy_name="log_space_robust",
            correction_mode="scale",
            prediction_ref_mode="head_median",
            prediction_ref_points=7,
            q0_kwargs=dict(
                window=21,
                detect_window=60,
                trend_window=60,
                kind="median",
                use_log_space=True,
            ),
            clip_scale_range=(0.50, 2.00),
        ),
    }
    return registry


def resolve_history_anchor_config(
    policy_name: str,
    overrides: Optional[Mapping[str, Any]] = None,
) -> HistoryAnchorConfig:
    registry = make_history_anchor_registry()
    if policy_name not in registry:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown anchor policy '{policy_name}'. Known: {known}")

    cfg = registry[policy_name]
    if not overrides:
        return cfg
    return replace(cfg, **dict(overrides))


# =============================================================================
# Public API
# =============================================================================

def apply_history_anchoring(
    series_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    full_history_by_well: Mapping[str, pd.DataFrame],
    config: HistoryAnchorConfig,
    q0_estimator: Optional[Callable[..., float]] = None,
) -> pd.DataFrame:
    """
    Apply Stage 2.6 history anchoring on top of Stage-2-loaded champion series.

    Design goals:
    - works on the already-loaded series_df from Phase 4 Stage 2
    - uses history-only information per split
    - does not touch train unless explicitly requested
    - attaches lightweight metadata for auditability
    """
    if series_df is None or series_df.empty or not config.enabled:
        return series_df.copy() if isinstance(series_df, pd.DataFrame) else pd.DataFrame()

    _validate_required_columns(series_df, [config.well_col, config.split_col, config.yhat_col])

    out = series_df.copy()
    out[config.split_col] = out[config.split_col].astype(str).str.lower()

    if q0_estimator is None:
        q0_estimator = _resolve_q0_estimator()

    history_stats: list[dict[str, Any]] = []
    target_splits = {str(s).lower() for s in config.target_splits}

    # process well × split independently so val and test get different causal anchors
    for well in sorted(out[config.well_col].dropna().astype(str).unique()):
        hist_df = _lookup_history_df(full_history_by_well, well)
        if hist_df is None or hist_df.empty:
            logger.warning("[Stage 2.6] skipping well=%s: no history found", well)
            continue

        hist_idx_col = _infer_history_idx_col(hist_df)
        hist_val_col = _infer_history_value_col(hist_df)
        if hist_idx_col is None or hist_val_col is None:
            logger.warning(
                "[Stage 2.6] skipping well=%s: could not infer history columns (idx=%s, value=%s)",
                well, hist_idx_col, hist_val_col,
            )
            continue

        bounds = _lookup_boundaries(boundaries_df, well)
        well_mask = out[config.well_col].astype(str).eq(str(well))

        for split_name in sorted(target_splits):
            split_mask = well_mask & out[config.split_col].eq(split_name)
            if not bool(split_mask.any()):
                continue

            subset = out.loc[split_mask].copy()
            subset = _sort_subset(subset, config)
            if subset.empty:
                continue

            cutoff_idx = _resolve_causal_cutoff_idx(
                split_name=split_name,
                subset=subset,
                bounds=bounds,
                idx_col=config.idx_col,
            )
            if cutoff_idx is None:
                logger.warning(
                    "[Stage 2.6] skipping well=%s split=%s: could not resolve cutoff_idx",
                    well, split_name,
                )
                continue

            hist_causal = hist_df.loc[hist_df[hist_idx_col] <= cutoff_idx].copy()
            if len(hist_causal) < int(config.min_history_points):
                logger.warning(
                    "[Stage 2.6] skipping well=%s split=%s: insufficient causal history (%d < %d)",
                    well, split_name, len(hist_causal), config.min_history_points,
                )
                continue

            anchor_q0 = _estimate_history_q0(
                hist_causal=hist_causal,
                hist_value_col=hist_val_col,
                estimator=q0_estimator,
                q0_kwargs=dict(config.q0_kwargs or {}),
            )
            if not np.isfinite(anchor_q0):
                logger.warning(
                    "[Stage 2.6] skipping well=%s split=%s: non-finite anchor_q0",
                    well, split_name,
                )
                continue

            pred_ref = _estimate_prediction_reference(subset, config)
            if not np.isfinite(pred_ref):
                logger.warning(
                    "[Stage 2.6] skipping well=%s split=%s: non-finite pred_ref",
                    well, split_name,
                )
                continue

            anchored = _apply_split_correction(
                subset=subset,
                config=config,
                anchor_q0=float(anchor_q0),
                pred_ref=float(pred_ref),
            )

            if config.attach_metadata:
                anchored["anchor_policy_name"] = str(config.policy_name)
                anchored["anchor_correction_mode"] = str(config.correction_mode)
                anchored["anchor_q0"] = float(anchor_q0)
                anchored["anchor_pred_ref"] = float(pred_ref)
                anchored["anchor_history_n"] = int(len(hist_causal))
                anchored["anchor_history_cutoff_idx"] = float(cutoff_idx)
                anchored["anchor_history_value_col"] = str(hist_val_col)

            out.loc[anchored.index, :] = anchored

            history_stats.append(
                dict(
                    well=str(well),
                    split=str(split_name),
                    cutoff_idx=float(cutoff_idx),
                    history_n=int(len(hist_causal)),
                    anchor_q0=float(anchor_q0),
                    pred_ref=float(pred_ref),
                )
            )

    if history_stats:
        stats_df = pd.DataFrame(history_stats)
        logger.info(
            "[Stage 2.6] applied history anchoring: wells=%d, well×split=%d, policy=%s",
            stats_df["well"].nunique(),
            len(stats_df),
            config.policy_name,
        )
    else:
        logger.warning("[Stage 2.6] no split was anchored.")

    return out


# =============================================================================
# Internals
# =============================================================================

def _validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _resolve_q0_estimator() -> Callable[..., float]:
    """
    Preferred project entrypoint:
      src/forecast_pipeline/arps_offline.py -> compute_history_q0_phys
    """
    try:
        from forecast_pipeline.arps_offline import compute_history_q0_phys  # type: ignore
        return compute_history_q0_phys
    except Exception:
        pass

    # optional fallback if project path exposes the module at top level
    try:
        from arps_offline import compute_history_q0_phys  # type: ignore
        return compute_history_q0_phys
    except Exception:
        pass

    # conservative internal fallback for notebook/prototyping
    return _fallback_history_q0_phys


def _lookup_history_df(full_history_by_well: Mapping[str, pd.DataFrame], well: str) -> Optional[pd.DataFrame]:
    for key in _well_aliases(well):
        df = full_history_by_well.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    return None


def _well_aliases(well: Any) -> list[str]:
    if well is None:
        return []
    s = str(well).strip()
    if not s:
        return []
    out = {s}
    if "/" in s:
        out.add(s.replace("/", "-"))
    if "-" in s and s.startswith("15-9-"):
        out.add(s.replace("-", "/", 1))
    return list(out)


def _infer_history_idx_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("idx", "global_idx", "t_idx", "time_idx", "t"):
        if col in df.columns:
            return col
    return None


def _infer_history_value_col(df: pd.DataFrame) -> Optional[str]:
    preferred = (
        "y", "ytrue", "rate", "qo", "q", "oil_rate", "y_obs", "target"
    )
    for col in preferred:
        if col in df.columns:
            return col
    blacklist = {"idx", "global_idx", "t", "ds", "timestamp"}
    for col in df.columns:
        if col in blacklist:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return str(col)
    return None


def _lookup_boundaries(boundaries_df: pd.DataFrame, well: str) -> Optional[dict[str, Any]]:
    if boundaries_df is None or boundaries_df.empty or "well" not in boundaries_df.columns:
        return None

    for key in _well_aliases(well):
        sub = boundaries_df.loc[boundaries_df["well"].astype(str).eq(str(key))].copy()
        if not sub.empty:
            return sub.iloc[0].to_dict()
    return None


def _sort_subset(subset: pd.DataFrame, config: HistoryAnchorConfig) -> pd.DataFrame:
    if config.idx_col in subset.columns:
        return subset.sort_values(config.idx_col, kind="stable")
    return subset.sort_index(kind="stable")


def _resolve_causal_cutoff_idx(
    split_name: str,
    subset: pd.DataFrame,
    bounds: Optional[Mapping[str, Any]],
    idx_col: str,
) -> Optional[float]:
    split_name = str(split_name).lower()

    if bounds:
        if split_name in {"val", "validation"} and bounds.get("train_end") is not None:
            return float(bounds["train_end"])
        if split_name == "test" and bounds.get("val_end") is not None:
            return float(bounds["val_end"])

    if idx_col in subset.columns and not subset[idx_col].dropna().empty:
        return float(subset[idx_col].min()) - 1.0

    # fallback for cases where subset uses only "t"
    if "t" in subset.columns and not subset["t"].dropna().empty:
        return float(pd.to_numeric(subset["t"], errors="coerce").min()) - 1.0

    return None


def _estimate_history_q0(
    hist_causal: pd.DataFrame,
    hist_value_col: str,
    estimator: Callable[..., float],
    q0_kwargs: Mapping[str, Any],
) -> float:
    values = hist_causal[hist_value_col].astype(float).to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    # Preferred path for forecast_pipeline.arps_offline.compute_history_q0_phys:
    # it expects X_any=... and internally extracts the first window.
    # We wrap the causal tail as shape (1, L, 1), with no scaler inversion.
    try:
        x_any = values.reshape(1, -1, 1)
        out = estimator(
            X_any=x_any,
            scaler_X=None,
            scaler_target=None,
            channel=-1,
            x_in_scaler_space=False,
            logger=logger,
            debug=False,
            **dict(q0_kwargs),
        )

        # project function returns (q0, meta)
        if isinstance(out, tuple) and len(out) >= 1:
            q0 = out[0]
        else:
            q0 = out

        if q0 is not None and np.isfinite(float(q0)):
            return float(q0)
    except TypeError:
        pass
    except Exception:
        pass

    # Secondary permissive attempts for simpler estimators/fallbacks
    call_variants = [
        lambda: estimator(values, **q0_kwargs),
        lambda: estimator(history_values=values, **q0_kwargs),
        lambda: estimator(series=values, **q0_kwargs),
        lambda: estimator(hist_causal, value_col=hist_value_col, **q0_kwargs),
        lambda: estimator(df=hist_causal, value_col=hist_value_col, **q0_kwargs),
    ]
    for fn in call_variants:
        try:
            out = fn()
            if isinstance(out, tuple) and len(out) >= 1:
                out = out[0]
            if np.isscalar(out) and np.isfinite(float(out)):
                return float(out)
        except TypeError:
            continue
        except Exception:
            continue

    return float(_fallback_history_q0_phys(values, **_clean_fallback_kwargs(q0_kwargs)))


def _clean_fallback_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {"window"}
    return {k: v for k, v in dict(kwargs).items() if k in allowed}

def _estimate_prediction_reference(subset: pd.DataFrame, config: HistoryAnchorConfig) -> float:
    vals = subset[config.yhat_col].astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")

    k = int(max(1, config.prediction_ref_points))
    head = vals[:k]

    if config.prediction_ref_mode == "first":
        return float(head[0])
    return float(np.median(head))


def _apply_split_correction(
    subset: pd.DataFrame,
    config: HistoryAnchorConfig,
    anchor_q0: float,
    pred_ref: float,
) -> pd.DataFrame:
    out = subset.copy()
    vals = out[config.yhat_col].astype(float).to_numpy()

    if config.correction_mode == "offset":
        delta = float(anchor_q0 - pred_ref)
        vals = vals + delta
        if config.attach_metadata:
            out["anchor_delta"] = delta
            out["anchor_scale"] = 1.0

    elif config.correction_mode == "offset_decay":
        delta = float(anchor_q0 - pred_ref)
        step = _relative_steps(out, config)
        decay = np.exp(-np.log(2.0) * step / max(1e-9, float(config.decay_halflife)))
        vals = vals + (delta * decay)
        if config.attach_metadata:
            out["anchor_delta"] = delta
            out["anchor_scale"] = 1.0
            out["anchor_decay_halflife"] = float(config.decay_halflife)

    elif config.correction_mode == "scale":
        denom = pred_ref if abs(pred_ref) > 1e-12 else np.nan
        scale = float(anchor_q0 / denom) if np.isfinite(denom) else 1.0
        lo, hi = config.clip_scale_range
        scale = float(np.clip(scale, lo, hi))
        vals = vals * scale
        if config.attach_metadata:
            out["anchor_delta"] = 0.0
            out["anchor_scale"] = scale

    else:
        raise ValueError(f"Unknown correction_mode='{config.correction_mode}'")

    if config.clip_min is not None:
        vals = np.maximum(vals, float(config.clip_min))

    out[config.yhat_col] = vals
    return out


def _relative_steps(df: pd.DataFrame, config: HistoryAnchorConfig) -> np.ndarray:
    if config.idx_col in df.columns:
        idx = pd.to_numeric(df[config.idx_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(idx).any():
            idx0 = np.nanmin(idx)
            steps = idx - idx0
            steps[~np.isfinite(steps)] = 0.0
            return np.maximum(steps, 0.0)
    return np.arange(len(df), dtype=float)


def _to_1d_numeric(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    if isinstance(values, pd.DataFrame):
        for col in values.columns:
            if pd.api.types.is_numeric_dtype(values[col]):
                return pd.to_numeric(values[col], errors="coerce").to_numpy(dtype=float)
        return np.array([], dtype=float)
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False).ravel()
    if isinstance(values, (list, tuple)):
        return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return np.array([float(values)], dtype=float)



from pathlib import Path


# =============================================================================
# Notebook / reporting helpers
# =============================================================================

def build_anchor_registry_table(
    registry: Optional[Mapping[str, HistoryAnchorConfig]] = None,
) -> pd.DataFrame:
    """
    Compact, notebook-friendly view of the anchor policy registry.
    """
    if registry is None:
        registry = make_history_anchor_registry()

    rows = []
    for name, cfg in registry.items():
        rows.append(
            dict(
                policy_name=name,
                correction_mode=cfg.correction_mode,
                prediction_ref_mode=cfg.prediction_ref_mode,
                prediction_ref_points=cfg.prediction_ref_points,
                decay_halflife=cfg.decay_halflife,
                target_splits=",".join(cfg.target_splits),
                q0_kwargs=str(cfg.q0_kwargs),
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "policy_name",
                "correction_mode",
                "prediction_ref_mode",
                "prediction_ref_points",
                "decay_halflife",
                "target_splits",
                "q0_kwargs",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("policy_name", kind="stable")
        .reset_index(drop=True)
    )


def build_anchor_run_summary(
    comparison_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One-row executive summary for the ablation run.
    """
    comparison_df = comparison_df.copy() if isinstance(comparison_df, pd.DataFrame) else pd.DataFrame()
    selected_df = selected_df.copy() if isinstance(selected_df, pd.DataFrame) else pd.DataFrame()

    risk_available = False
    if "val_risk_q90" in comparison_df.columns:
        risk_available = bool(pd.to_numeric(comparison_df["val_risk_q90"], errors="coerce").notna().any())

    return pd.DataFrame(
        [
            dict(
                n_rows=int(len(comparison_df)),
                n_wells=int(comparison_df["well"].nunique()) if "well" in comparison_df.columns else 0,
                n_policies=int(comparison_df["policy_name"].nunique()) if "policy_name" in comparison_df.columns else 0,
                n_selected_wells=int(selected_df["well"].nunique()) if "well" in selected_df.columns else 0,
                risk_available=bool(risk_available),
            )
        ]
    )


def build_anchor_comparison_view(
    comparison_df: pd.DataFrame,
    *,
    sort_metric: str = "val_metric_inter",
) -> pd.DataFrame:
    """
    Main compact table for well × policy comparison.
    """
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame(
            columns=[
                "well",
                "policy_name",
                "val_metric_inter",
                "val_metric_intra_best",
                "val_risk_q50",
                "val_risk_q90",
                "test_metric_inter",
                "test_metric_intra_best",
                "test_risk_q50",
                "test_risk_q90",
                "n_members_total",
                "n_families_seen",
            ]
        )

    cols = [
        "well",
        "policy_name",
        "val_metric_inter",
        "val_metric_intra_best",
        "val_risk_q50",
        "val_risk_q90",
        "test_metric_inter",
        "test_metric_intra_best",
        "test_risk_q50",
        "test_risk_q90",
        "n_members_total",
        "n_families_seen",
    ]
    cols = [c for c in cols if c in comparison_df.columns]

    sort_cols = ["well", sort_metric, "policy_name"]
    sort_cols = [c for c in sort_cols if c in comparison_df.columns]

    out = comparison_df[cols].copy()
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable")

    return out.reset_index(drop=True)


def build_anchor_selection_view(
    selected_df: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge selected policy per well with the corresponding comparison metrics,
    so the decision table already supports a conclusion.
    """
    if selected_df is None or selected_df.empty:
        return pd.DataFrame(
            columns=[
                "well",
                "selected_policy_name",
                "selection_metric",
                "selection_value",
                "selection_tie_break_metric",
                "selection_tie_break_value",
                "val_metric_inter",
                "val_metric_intra_best",
                "val_risk_q90",
                "test_metric_inter",
                "test_metric_intra_best",
                "test_risk_q90",
                "selection_reason",
            ]
        )

    out = selected_df.copy()

    if comparison_df is not None and not comparison_df.empty:
        keep = [
            "well",
            "policy_name",
            "val_metric_inter",
            "val_metric_intra_best",
            "val_risk_q90",
            "test_metric_inter",
            "test_metric_intra_best",
            "test_risk_q90",
            "n_members_total",
            "n_families_seen",
        ]
        keep = [c for c in keep if c in comparison_df.columns]

        comp = comparison_df[keep].copy()
        comp = comp.rename(columns={"policy_name": "selected_policy_name"})

        join_keys = [c for c in ["well", "selected_policy_name"] if c in out.columns and c in comp.columns]
        if join_keys:
            out = out.merge(comp, on=join_keys, how="left")

    ordered = [
        "well",
        "selected_policy_name",
        "selection_metric",
        "selection_value",
        "selection_tie_break_metric",
        "selection_tie_break_value",
        "val_metric_inter",
        "val_metric_intra_best",
        "val_risk_q90",
        "test_metric_inter",
        "test_metric_intra_best",
        "test_risk_q90",
        "n_members_total",
        "n_families_seen",
        "selection_reason",
    ]
    ordered = [c for c in ordered if c in out.columns]

    return (
        out[ordered]
        .sort_values(["well"], kind="stable")
        .reset_index(drop=True)
    )


def build_anchor_storytelling_views(
    comparison_df: pd.DataFrame,
    *,
    baseline_policy: str = "baseline_default",
) -> Dict[str, pd.DataFrame]:
    """
    Compact storytelling views:
      - validation pivot
      - delta vs baseline
      - optional validation risk pivot
    """
    views: Dict[str, pd.DataFrame] = {}

    if comparison_df is None or comparison_df.empty:
        return views

    if {"well", "policy_name", "val_metric_inter"}.issubset(comparison_df.columns):
        pivot_val = comparison_df.pivot_table(
            index="well",
            columns="policy_name",
            values="val_metric_inter",
            aggfunc="first",
        ).sort_index()
        views["pivot_val_metric"] = pivot_val

        if baseline_policy in pivot_val.columns:
            delta = pivot_val.subtract(pivot_val[baseline_policy], axis=0)
            views["delta_vs_baseline"] = delta.sort_index()

    if {"well", "policy_name", "val_risk_q90"}.issubset(comparison_df.columns):
        risk_vals = pd.to_numeric(comparison_df["val_risk_q90"], errors="coerce")
        if risk_vals.notna().any():
            pivot_risk = comparison_df.pivot_table(
                index="well",
                columns="policy_name",
                values="val_risk_q90",
                aggfunc="first",
            ).sort_index()
            views["pivot_val_risk_q90"] = pivot_risk

    return views


def build_anchor_artifact_status(save_dir: str | Path) -> pd.DataFrame:
    """
    Compact artifact-status view for the notebook footer.
    """
    save_dir = Path(save_dir)
    files = [
        "anchor_policy_comparison.csv",
        "selected_anchor_policies.csv",
    ]

    rows = []
    for name in files:
        path = save_dir / name
        exists = path.exists()
        n_rows = np.nan
        if exists:
            try:
                n_rows = int(len(pd.read_csv(path)))
            except Exception:
                n_rows = np.nan

        rows.append(
            dict(
                artifact=name,
                exists=bool(exists),
                rows=n_rows,
                path=str(path),
            )
        )

    return pd.DataFrame(rows)


# =============================================================================
# Editorial views for notebook / report
# =============================================================================

def _drop_all_nan_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            if not vals.notna().any():
                out = out.drop(columns=[col])
    return out


def build_anchor_selected_editorial_table(
    selected_df: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame] = None,
    *,
    include_test_metrics: bool = True,
    include_risk_if_available: bool = True,
) -> pd.DataFrame:
    """
    Editorial table:
      one row per well,
      selected policy from validation,
      plus test audit columns.
    """
    raw = build_anchor_selection_view(selected_df, comparison_df)
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=[
                "Well",
                "Selected Policy",
                "Validation Error (%)",
                "Test Audit (%)",
                "Best Intra Val (%)",
                "Best Intra Test (%)",
                "Members",
            ]
        )

    out = raw.copy()

    # if selection_value is the same thing as val_metric_inter, prefer the explicit metric column
    if "val_metric_inter" not in out.columns and "selection_value" in out.columns:
        out["val_metric_inter"] = out["selection_value"]

    keep = [
        "well",
        "selected_policy_name",
        "val_metric_inter",
        "test_metric_inter",
        "val_metric_intra_best",
        "test_metric_intra_best",
        "val_risk_q90",
        "test_risk_q90",
        "n_members_total",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    rename = {
        "well": "Well",
        "selected_policy_name": "Selected Policy",
        "val_metric_inter": "Validation Error (%)",
        "test_metric_inter": "Test Audit (%)",
        "val_metric_intra_best": "Best Intra Val (%)",
        "test_metric_intra_best": "Best Intra Test (%)",
        "val_risk_q90": "Validation Risk q90",
        "test_risk_q90": "Test Risk q90",
        "n_members_total": "Members",
    }
    out = out.rename(columns=rename)

    if not include_test_metrics:
        for col in ["Test Audit (%)", "Best Intra Test (%)", "Test Risk q90"]:
            if col in out.columns:
                out = out.drop(columns=[col])

    if not include_risk_if_available:
        for col in ["Validation Risk q90", "Test Risk q90"]:
            if col in out.columns:
                out = out.drop(columns=[col])

    out = _drop_all_nan_columns(out, ["Validation Risk q90", "Test Risk q90"])

    preferred = [
        "Well",
        "Selected Policy",
        "Validation Error (%)",
        "Test Audit (%)",
        "Best Intra Val (%)",
        "Best Intra Test (%)",
        "Validation Risk q90",
        "Test Risk q90",
        "Members",
    ]
    preferred = [c for c in preferred if c in out.columns]

    return (
        out[preferred]
        .sort_values(["Well"], kind="stable")
        .reset_index(drop=True)
    )


def build_anchor_test_audit_table(
    comparison_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    *,
    near_best_tol: float = 0.25,
) -> pd.DataFrame:
    """
    Audit table:
    Was the validation-selected policy also good on test?

    One row per well:
      - selected policy
      - selected test metric
      - best policy on test
      - best test metric
      - regret vs best test
      - concise verdict
    """
    if comparison_df is None or comparison_df.empty or selected_df is None or selected_df.empty:
        return pd.DataFrame(
            columns=[
                "Well",
                "Selected Policy",
                "Best Test Policy",
                "Verdict",
                "Test Audit (%)",
                "Best Test (%)",
                "Test Regret (%)",
            ]
        )

    need = {"well", "policy_name", "test_metric_inter"}
    if not need.issubset(comparison_df.columns):
        return pd.DataFrame()

    selected_map = (
        selected_df[["well", "selected_policy_name"]]
        .drop_duplicates()
        .copy()
    )

    rows = []
    for well, sub in comparison_df.groupby("well", dropna=False):
        local = sub.copy()
        local["test_metric_inter"] = pd.to_numeric(local["test_metric_inter"], errors="coerce")
        local = local.loc[local["test_metric_inter"].notna()].copy()
        if local.empty:
            continue

        sel_row = selected_map.loc[selected_map["well"].astype(str).eq(str(well))]
        if sel_row.empty:
            continue

        selected_policy = str(sel_row.iloc[0]["selected_policy_name"])
        selected_sub = local.loc[local["policy_name"].astype(str).eq(selected_policy)].copy()
        if selected_sub.empty:
            continue

        selected_test = float(selected_sub.iloc[0]["test_metric_inter"])

        ranked = local.sort_values(
            ["test_metric_inter", "policy_name"],
            ascending=[True, True],
            kind="stable",
        ).reset_index(drop=True)

        best = ranked.iloc[0]
        best_test_policy = str(best["policy_name"])
        best_test = float(best["test_metric_inter"])
        regret = float(selected_test - best_test)

        if selected_policy == best_test_policy:
            verdict = "supports"
        elif regret <= near_best_tol:
            verdict = "near-best"
        else:
            verdict = "diverges"

        rows.append(
            dict(
                **{
                    "Well": str(well),
                    "Selected Policy": selected_policy,
                    "Best Test Policy": best_test_policy,
                    "Verdict": verdict,
                    "Test Audit (%)": selected_test,
                    "Best Test (%)": best_test,
                    "Test Regret (%)": regret,
                }
            )
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["Well"], kind="stable")
        .reset_index(drop=True)
    )


def render_anchor_editorial_report(
    comparison_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    *,
    include_risk_if_available: bool = True,
    show_full_comparison: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Rich-rendered editorial bundle for the notebook.

    Focus:
      1) selected policy per well, with test audit beside it
      2) concise audit table answering whether validation held on test
    """
    from common.log_table import render_rich_table

    selection_table = build_anchor_selected_editorial_table(
        selected_df=selected_df,
        comparison_df=comparison_df,
        include_test_metrics=True,
        include_risk_if_available=include_risk_if_available,
    )

    audit_table = build_anchor_test_audit_table(
        comparison_df=comparison_df,
        selected_df=selected_df,
    )

    if not selection_table.empty:
        metric_cols = [
            c for c in [
                "Validation Error (%)",
                "Test Audit (%)",
                "Best Intra Val (%)",
                "Best Intra Test (%)",
            ]
            if c in selection_table.columns
        ]
        render_rich_table(
            selection_table,
            title="Anchor Selection — validation rule, test audit",
            metric_columns=metric_cols,
            show_lines=True,
            decimals=2,
            bar_width=12,
            bar_track=False,
            expand=False,
            column_options={
                "Well": {"no_wrap": True, "max_width": 12},
                "Selected Policy": {"overflow": "fold", "max_width": 18},
                "Members": {"no_wrap": True, "max_width": 8},
            },
        )

    if not audit_table.empty:
        metric_cols = [
            c for c in [
                "Test Audit (%)",
                "Best Test (%)",
                "Test Regret (%)",
            ]
            if c in audit_table.columns
        ]
        render_rich_table(
            audit_table,
            title="Test Audit — did the validation choice hold?",
            metric_columns=metric_cols,
            show_lines=True,
            decimals=2,
            bar_width=10,
            bar_track=False,
            expand=False,
            column_options={
                "Well": {"no_wrap": True, "max_width": 12},
                "Selected Policy": {"overflow": "fold", "max_width": 18},
                "Best Test Policy": {"overflow": "fold", "max_width": 18},
                "Verdict": {"no_wrap": True, "max_width": 10},
            },
        )

    if show_full_comparison:
        full_table = build_anchor_comparison_view(comparison_df)
        full_table = _drop_all_nan_columns(
            full_table,
            ["val_risk_q50", "val_risk_q90", "test_risk_q50", "test_risk_q90"],
        )
        if not full_table.empty:
            full_table = full_table.rename(
                columns={
                    "well": "Well",
                    "policy_name": "Policy",
                    "val_metric_inter": "Validation Error (%)",
                    "val_metric_intra_best": "Best Intra Val (%)",
                    "test_metric_inter": "Test Audit (%)",
                    "test_metric_intra_best": "Best Intra Test (%)",
                    "n_members_total": "Members",
                    "n_families_seen": "Families",
                }
            )
            metric_cols = [
                c for c in [
                    "Validation Error (%)",
                    "Best Intra Val (%)",
                    "Test Audit (%)",
                    "Best Intra Test (%)",
                ]
                if c in full_table.columns
            ]
            render_rich_table(
                full_table,
                title="Anchor Comparison — full table (debug)",
                metric_columns=metric_cols,
                show_lines=False,
                decimals=2,
                bar_width=10,
                bar_track=False,
                expand=False,
                column_options={
                    "Well": {"no_wrap": True, "max_width": 12},
                    "Policy": {"overflow": "fold", "max_width": 18},
                },
            )
    else:
        full_table = pd.DataFrame()

    return {
        "selection_table": selection_table,
        "audit_table": audit_table,
        "full_table": full_table,
    }


def load_selected_anchor_policies(
    selected_policy_source: Any,
) -> pd.DataFrame:
    """
    Load selected anchor policies from:
      - DataFrame
      - CSV path
      - Parquet path

    Expected columns:
      - well
      - selected_policy_name
    """
    if isinstance(selected_policy_source, pd.DataFrame):
        df = selected_policy_source.copy()
    else:
        path = Path(selected_policy_source)
        if not path.exists():
            raise FileNotFoundError(f"Selected policy file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError(
                f"Unsupported selected policy file format: {path.suffix}. "
                "Use .csv or .parquet"
            )

    required = {"well", "selected_policy_name"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Selected policy artifact missing columns: {sorted(missing)}")

    out = df[["well", "selected_policy_name"]].copy()
    out["well"] = out["well"].astype(str).str.strip()
    out["selected_policy_name"] = out["selected_policy_name"].astype(str).str.strip()
    out = out.dropna(subset=["well", "selected_policy_name"])
    out = out.drop_duplicates(subset=["well"], keep="first").reset_index(drop=True)
    return out


def _build_selected_policy_lookup(selected_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build alias-aware lookup:
      selected well name -> policy
      plus common aliases (e.g. 15/9-F-12 <-> 15-9-F-12)
    """
    lookup: Dict[str, str] = {}

    if selected_df is None or selected_df.empty:
        return lookup

    for _, row in selected_df.iterrows():
        well = str(row["well"]).strip()
        policy = str(row["selected_policy_name"]).strip()
        if not well or not policy:
            continue

        for alias in _well_aliases(well):
            lookup[str(alias)] = policy

    return lookup


def _resolve_policy_name_for_well(
    well: Any,
    lookup: Mapping[str, str],
    default_policy_name: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve selected policy for a well using alias-aware matching.
    """
    for alias in _well_aliases(well):
        if alias in lookup:
            return str(lookup[alias])

    return default_policy_name


def apply_history_anchoring_by_selected_policy(
    series_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    full_history_by_well: Mapping[str, pd.DataFrame],
    selected_policy_source: Any,
    *,
    policy_overrides_by_name: Optional[Mapping[str, Mapping[str, Any]]] = None,
    default_policy_name: Optional[str] = None,
    well_col: str = "well",
) -> pd.DataFrame:
    """
    Apply Stage 2.6 anchoring using a selected policy per well.

    Workflow:
      - read selected policy artifact
      - resolve one policy per well
      - apply the corresponding HistoryAnchorConfig to each well subset
      - concatenate back preserving original index order

    This is the bridge between:
      notebook A (policy selection) -> notebook B (operational replay)
    """
    if series_df is None or series_df.empty:
        return pd.DataFrame() if not isinstance(series_df, pd.DataFrame) else series_df.copy()

    if well_col not in series_df.columns:
        raise KeyError(f"series_df is missing required well column: '{well_col}'")

    selected_df = load_selected_anchor_policies(selected_policy_source)
    lookup = _build_selected_policy_lookup(selected_df)
    overrides_by_name = dict(policy_overrides_by_name or {})

    parts = []
    audit_rows = []

    for well, sub in series_df.groupby(well_col, dropna=False, sort=False):
        well_str = str(well)
        policy_name = _resolve_policy_name_for_well(
            well=well_str,
            lookup=lookup,
            default_policy_name=default_policy_name,
        )

        if policy_name is None:
            local = sub.copy()
            local["anchor_policy_name"] = local.get("anchor_policy_name", pd.Series(index=local.index, dtype=object))
            local["anchor_policy_name"] = local["anchor_policy_name"].fillna("legacy_no_replay")
            local["anchor_policy_source"] = "selected_map_missing"
            parts.append(local)

            audit_rows.append(
                dict(
                    well=well_str,
                    selected_policy_name=None,
                    status="missing_policy",
                )
            )
            continue

        overrides = overrides_by_name.get(policy_name)
        cfg = resolve_history_anchor_config(policy_name, overrides=overrides)

        local = apply_history_anchoring(
            series_df=sub,
            boundaries_df=boundaries_df,
            full_history_by_well=full_history_by_well,
            config=cfg,
        )

        local["anchor_policy_source"] = "selected_map"
        parts.append(local)

        audit_rows.append(
            dict(
                well=well_str,
                selected_policy_name=policy_name,
                status="applied",
            )
        )

    if not parts:
        out = series_df.copy()
    else:
        out = pd.concat(parts, axis=0).sort_index(kind="stable")

    if audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        logger.info(
            "[Stage 2.6] selected-policy replay: applied=%d, missing=%d, wells=%d",
            int((audit_df["status"] == "applied").sum()),
            int((audit_df["status"] == "missing_policy").sum()),
            int(audit_df["well"].nunique()),
        )

    return out