from __future__ import annotations
"""forecast_pipeline.data_cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi‑level caching helpers for *prepared* data used by training jobs.

Public entry‑point
------------------
``cached_prepare_job_data(job)`` – a drop‑in replacement for the original
``prepare_job_data`` that transparently applies:

* **Per‑process RAM cache** via ``functools.lru_cache``
* **Cross‑run disk cache** via ``joblib.Memory``

Usage
-----
```python
from forecast_pipeline.data_cache import cached_prepare_job_data as prepare_job_data
```
No other code changes are required.
"""
import os
import json
import logging
import hashlib
from functools import lru_cache
from typing import Any, Dict, Tuple

from joblib import Memory

# ─── Local imports ──────────────────────────────────────────────────────────
from common.config_wells import DATA_SOURCES
from forecast_pipeline.config import (
    DEFAULT_EXP_PARAMS,
    SEQ2SEQ_ARCHS,
    STRATEGY_OPTIONS,
    EXTRACTOR_OPTIONS,
    FUSER_OPTIONS,
)
from forecast_pipeline.experiments.seq2context import ExperimentSeq2Context
from forecast_pipeline.experiments.seq2value import ExperimentSeq2Value

# ---------------------------------------------------------------------------
# 1.  Disk‑level cache (shared across runs)                                   
# ---------------------------------------------------------------------------
from pathlib import Path

CACHE_VERSION = "v1"  # bump to invalidate *all* previous artefacts

# ---------------------------------------------------------------------------
# 1½.  Robust cache root – absolute & user‑overridable                       
# ---------------------------------------------------------------------------
_default_root = Path.home() / ".fp_cache"
_cache_root = Path(os.getenv("FP_CACHE_DIR", _default_root)).expanduser().resolve()
_cache_root.mkdir(parents=True, exist_ok=True)

# ``store_function_code=False`` avoids race conditions where multiple
# workers attempt to rewrite the cached source-code file at the same time –
# something we just ran into (FileNotFoundError on func_code.py).
#
# Trade‑off: Changes inside _disk_load **will not** automatically invalidate
# old cache entries, so if you refactor the preprocessing logic you must bump
# ``CACHE_VERSION`` or clear the cache directory manually.

disk_cache: Memory = Memory(
    location=str(_cache_root),
    compress=0,           # no gzip
    verbose=0,
)

# ---------------------------------------------------------------------------
# 2.  Utility helpers                                                        
# ---------------------------------------------------------------------------

from functools import lru_cache


def _stable_json(obj: Any) -> str:
    """Dump *any* JSON-serialisable object deterministically."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _resolve_data_source(name: str) -> Dict[str, Any]:
    """Return the *original* data‑source dict given its unique name."""
    for ds in DATA_SOURCES:
        if ds["name"] == name:
            return ds
    raise ValueError(f"Unknown data‑source '{name}'")

# ---- data‑relevant keys (extend if needed) ---------------------------------
_DATA_KEYS = {
    "lag_window",
    "horizon",
    "feature_kind",
    "test_size",
    "input_variables",
    "target_variable",
}


def _data_signature(params: Dict[str, Any]) -> str:
    """Return JSON for *only* the parameters that influence data prep."""
    slim = {k: v for k, v in params.items() if k in _DATA_KEYS}
    return _stable_json(slim)

# ---------------------------------------------------------------------------  Utility helpers                                                         
# ---------------------------------------------------------------------------

def _stable_json(obj: Any) -> str:
    """Dump *any* JSON‑serialisable object deterministically."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

# ---- data‑relevant keys (extend if needed) ---------------------------------
_DATA_KEYS = {
    "lag_window",
    "horizon",
    "feature_kind",
    "test_size",
    "input_variables",
    "target_variable",
}


def _data_signature(params: Dict[str, Any]) -> str:
    """Return JSON for *only* the parameters that influence data prep."""
    slim = {k: v for k, v in params.items() if k in _DATA_KEYS}
    return _stable_json(slim)

# ---------------------------------------------------------------------------
# 3.  Heavy worker – wrapped by joblib.Memory                                 
# ---------------------------------------------------------------------------
@disk_cache.cache
def _disk_load(
    ds_name: str,
    well: str,
    arch_name: str,
    params_json: str,
    cache_ver: str,
):
    """Perform the expensive ``exp.load_and_prepare`` and persist the result."""
    logging.info(
        "[DiskCache] MISS – loading %s/%s (arch=%s, ver=%s)",
        ds_name,
        well,
        arch_name,
        cache_ver,
    )

    ds = _resolve_data_source(ds_name)

    # ------------------------------------------------------------------
    # Build **full** parameter dict expected by Experiment classes.
    # Only a *subset* influences the prepared data, but missing keys like
    # 'strategy_config' would otherwise trigger KeyError down the line.
    # We therefore merge:
    #   * project defaults         (DEFAULT_EXP_PARAMS)
    #   * safe placeholders         (first element of *_OPTIONS lists)
    #   * data‑relevant overrides   (from params_json)
    # ------------------------------------------------------------------
    p_from_signature: Dict[str, Any] = json.loads(params_json)

    full_params: Dict[str, Any] = {
        **DEFAULT_EXP_PARAMS,
        "strategy_config":  STRATEGY_OPTIONS[0],
        "extractor_config": EXTRACTOR_OPTIONS[0],
        "fuser_config":     FUSER_OPTIONS[0],
        **p_from_signature,
    }

    # Choose experiment class ------------------------------------------------
    if arch_name in SEQ2SEQ_ARCHS:
        exp_cls = ExperimentSeq2Context
    elif arch_name == "Seq2Value":
        exp_cls = ExperimentSeq2Value
    else:
        raise ValueError(f"Unknown architecture '{arch_name}'")

    exp = exp_cls(ds, well, full_params, 0)  # job_id positional
    return exp.load_and_prepare()

# ---------------------------------------------------------------------------
# 4.  Per‑process RAM cache – extremely cheap                                 
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _mem_load(
    ds_name: str,
    well: str,
    arch_name: str,
    params_json: str,
    cache_ver: str,
):
    logging.debug("[MemCache] MISS – %s/%s in PID %d", ds_name, well, os.getpid())
    return _disk_load(ds_name, well, arch_name, params_json, cache_ver)

# ---------------------------------------------------------------------------
# 5.  Public API                                                              
# ---------------------------------------------------------------------------

def cached_prepare_job_data(job: Tuple[Dict[str, Any], str, Dict[str, Any], int]):
    """Replacement for the original ``prepare_job_data`` with caching."""
    ds, well, params, job_id = job
    arch = params.get("architecture_name", DEFAULT_EXP_PARAMS["architecture_name"])

    # Serialise params once so they become a hashable cache key --------------
    params_json = _data_signature(params)

    (
        train_kwargs,
        prediction_input,
        y_test,
        scaler_target,
        y_train_original,
    ) = _mem_load(ds["name"], well, arch, params_json, CACHE_VERSION)

    return (
        train_kwargs,
        prediction_input,
        y_test,
        scaler_target,
        y_train_original,
        params,
        ds,
        well,
        job_id,
    )
