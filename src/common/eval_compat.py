# src/common/eval_compat.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, Sequence, Iterable, Union
import logging
import numpy as np

log = logging.getLogger(__name__)

def _build_forecast_agg_with_train(
    series_artifacts: Optional[Dict[str, Any]],
    y_train_original: Optional[Any],
) -> Dict[str, Any]:
    """
    Ensure forecast_agg has a 'train' block so the Series Store can write TRAIN rows.

    - Uses y_train_original as both ytrue and (for now) yhat, i.e. a perfect
      reconstruction over the train window.
    - If 'train' already exists in series_artifacts, it is left untouched.
    - If y_train_original is missing, returns the original series_artifacts.
    """
    # Start from existing artifacts (val/test etc.)
    fa: Dict[str, Any] = dict(series_artifacts or {})

    # If someone upstream already populated 'train', respect it.
    if "train" in fa:
        return fa

    # Convert training target to a 1D float array
    y_train_1d = _to_1d_float(y_train_original)
    if y_train_1d is None or y_train_1d.size == 0:
        return fa

    # Simple 0..N-1 time axis for train, consistent with full_history
    t_train = np.arange(y_train_1d.size, dtype=float)

    fa["train"] = {
        # Train region in physical domain, aligned with full_history indices
        "t": t_train.tolist(),
        # For ARPS we don't currently have a separate train yhat stored, so we
        # use ytrue as a "perfect fit" reconstruction. If you later compute a
        # true train prediction curve, you can swap it in here.
        "yhat": y_train_1d.tolist(),
        "ytrue": y_train_1d.tolist(),
    }
    return fa


def unpack_eval7(ret: Sequence[Any]) -> Tuple[Any, Any, Any, Any, Any, Any, Optional[Dict[str, Any]]]:
    """
    Accepts either a 6-item (legacy) or 7-item (new) evaluation return.
    Returns a 7-tuple where the 7th item is None if the input had 6 items.
    """
    if not isinstance(ret, (list, tuple)):
        raise RuntimeError("Evaluator did not return a tuple/list.")
    n = len(ret)
    if n == 7:
        return ret  # type: ignore[return-value]
    if n == 6:
        a, b, c, d, e, f = ret
        return a, b, c, d, e, f, None
    raise RuntimeError(f"Unexpected evaluator return length={n}.")


# ------------------------------ helpers ---------------------------------

def _to_1d_float(arr_like: Any) -> Optional[np.ndarray]:
    """
    Best-effort conversion to a 1D float numpy array.
    Accepts: list/tuple/np.ndarray (possibly nested windows). Returns None if not possible.
    """
    try:
        if arr_like is None:
            return None
        # Flatten aggressively (handles windowed lists)
        a = np.asarray(arr_like, dtype=float).reshape(-1)
        if a.size == 0:
            return None
        return a
    except Exception:
        return None


def _build_full_history_payload(
    y_train_original: Optional[Any],
    series_artifacts: Optional[Dict[str, Any]],
) -> Optional[Dict[str, list]]:
    """
    Compose a continuous ground-truth timeline from:
      - training target (physical domain) -> train segment
      - validation ytrue (if present)     -> val segment
      - test ytrue (if present)           -> test segment
    Returns {"t": [...], "ytrue": [...]} or None when insufficient data.
    """
    # Training segment
    y_train_1d = _to_1d_float(y_train_original)

    # Validation/test segments (optional)
    val_block  = (series_artifacts or {}).get("val")  if isinstance(series_artifacts, dict) else None
    test_block = (series_artifacts or {}).get("test") if isinstance(series_artifacts, dict) else None

    val_ytrue  = _to_1d_float((val_block or {}).get("ytrue")) if isinstance(val_block, dict) else None
    test_ytrue = _to_1d_float((test_block or {}).get("ytrue")) if isinstance(test_block, dict) else None

    # Nothing? return None so caller can skip
    if y_train_1d is None and val_ytrue is None and test_ytrue is None:
        return None

    pieces: list[np.ndarray] = []
    if y_train_1d is not None:
        pieces.append(y_train_1d)
    if val_ytrue is not None:
        pieces.append(val_ytrue)
    if test_ytrue is not None:
        pieces.append(test_ytrue)

    ytrue_full = np.concatenate(pieces) if pieces else np.asarray([], dtype=float)
    t_full = np.arange(ytrue_full.size, dtype=float)

    return {
        "t":     t_full.tolist(),
        "ytrue": ytrue_full.tolist(),
    }


# ------------------------------ public API ---------------------------------

def attach_series_context(result_dict: Dict[str, Any],
                          scaler_target: Any,
                          series_artifacts: Optional[Dict[str, Any]],
                          y_train_original: Optional[Any] = None) -> None:
    """
    Embed self-heal plotting context into the final result dict.
    """
    try:
        split_ctx = getattr(scaler_target, "_split_ctx", {}) or {}
        boundaries = (split_ctx.get("boundaries") or None) if isinstance(split_ctx, dict) else None

        # ✅ NEW: extend val/test artifacts with a synthetic TRAIN block
        forecast_agg = _build_forecast_agg_with_train(series_artifacts, y_train_original)

        # Use the extended forecast_agg when building full_history
        full_history = _build_full_history_payload(y_train_original, forecast_agg)
        if full_history:
            log.info("✅ [context] full_history attached: total_gt=%d", len(full_history["ytrue"]))
        else:
            log.info("⚠️  [context] full_history missing (no train/val/test GT available)")

        result_dict["series_context"] = {
            "boundaries": boundaries,
            "warmup_len_val": split_ctx.get("warmup_len_val") if isinstance(split_ctx, dict) else None,
            "warmup_len_test": split_ctx.get("warmup_len_test") if isinstance(split_ctx, dict) else None,
            # ✅ store train+val+test artifacts here
            "forecast_agg": forecast_agg,
            "full_history": full_history,
        }
    except Exception as e:
        log.debug("[series_store] Failed to embed series_context: %s", e)



def ensure_job_meta(result_dict: Dict[str, Any],
                    params: Dict[str, Any],
                    ds: Dict[str, Any],
                    well: str,
                    job_hash: Optional[str] = None) -> None:
    """
    Fill a compact job_meta block, used by the Series Store writer for partitioning.
    Ensures group/campaign are populated from ds/params as fallbacks.
    """
    jm = result_dict.setdefault("job_meta", {})

    def _first(*vals: Union[str, None]) -> Optional[str]:
        for v in vals:
            if isinstance(v, str) and v.strip():
                return v
        return None

    group = _first(
        jm.get("group"),
        ds.get("campaign_group"),
        params.get("campaign_group"),
        ds.get("group"),
        params.get("group"),
    )
    campaign = _first(
        jm.get("campaign"),
        ds.get("campaign_name"),
        params.get("campaign_name"),
        ds.get("campaign"),
        params.get("campaign"),
    )
    arch = _first(jm.get("arch"), params.get("architecture_name"))
    dataset = _first(jm.get("dataset"), ds.get("dataset_name"), ds.get("name"))

    jm["group"] = group or "unknown_group"
    jm["campaign"] = campaign or "unknown_campaign"
    jm["arch"] = arch or jm.get("arch") or "unknown_arch"
    jm["job_hash"] = job_hash or jm.get("job_hash")
    jm["dataset"] = dataset
    jm["well"] = well

    log.info("✅ [meta] job_meta resolved: group=%s campaign=%s arch=%s dataset=%s",
             jm["group"], jm["campaign"], jm["arch"], jm.get("dataset"))


def maybe_persist_series(result_dict: Dict[str, Any],
                         series_store_cfg: Dict[str, Any],
                         logger: Optional[logging.Logger] = None) -> None:
    """
    Non-fatal Parquet write side-effect. No-ops when disabled.
    Requires forecast_agg to avoid empty champion-series writes.
    Also persists:
      - meta/boundaries.parquet (idempotent upsert)
      - history/well=<well>/history.parquet (idempotent, keep longest)
    """
    _log = logger or logging.getLogger(__name__)
    try:
        if not series_store_cfg or not series_store_cfg.get("enabled"):
            return

        # Champion series: require forecast_agg present so we don't write empty parquet
        sc = result_dict.get("series_context") or {}
        if not sc or not sc.get("forecast_agg"):
            _log.info("[series_store] enabled but forecast_agg missing; skip series write.")
            return

        from common.series_store_writer import (
            build_series_record,
            persist_series,
            persist_boundaries_manifest,
            persist_full_history,
        )

        record, diag = build_series_record(result_dict)
        _log.info("ℹ️  [series_store] series extraction diag: %s", " | ".join(diag) if diag else "(no diag)")

        # 1) champion series parquet (per job_hash)
        persist_series(record, series_store_cfg)

        # 2) boundaries manifest (global/meta)
        persist_boundaries_manifest(record, series_store_cfg)

        # 3) full history (per well)
        persist_full_history(record, series_store_cfg)

        _log.info("✅ Successfully persisted forecast series and meta artifacts to Series Store.")
    except Exception as e:
        _log.error("Failed to persist series/meta to Series Store: %s", e)

