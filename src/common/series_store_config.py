# src/common/series_store_config.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import logging

_SUPPORTED_FORMATS = {"parquet"}
_SUPPORTED_CODECS = {"zstd", "snappy", "gzip"}

_DEFAULTS = {
    "enabled": False,
    "format": "parquet",
    "compress": "zstd",
    "schema_version": 1,
    "self_heal": True,
    # base_dir resolves relative to config.infra.experiments_output_dir if not absolute
    "base_dir": None,  # will be resolved later
}

def _as_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):  return True
        if s in ("0", "false", "no", "n", "off"): return False
    return default

def _pick(value: Any, allowed: set[str], default: str, label: str) -> str:
    v = str(value).strip().lower() if value is not None else default
    if v not in allowed:
        logging.warning(
            "[series_store] Unsupported %s '%s' (allowed: %s). "
            "Falling back to '%s'.", label, v, ", ".join(sorted(allowed)), default
        )
        return default
    return v

def _resolve_base_dir(base_dir: Any, experiments_output_dir: str) -> Path:
    """
    If base_dir is:
      - None: use <experiments_output_dir> (we keep it simple for Step 0)
      - relative: resolve relative to experiments_output_dir
      - absolute: keep as-is
    """
    exp = Path(experiments_output_dir).resolve()
    if base_dir in (None, "", False):
        return exp  # Step 0: keep series under experiments_output_dir tree
    p = Path(str(base_dir))
    return p if p.is_absolute() else (exp / p).resolve()

def normalize_and_validate_series_store(
    config_obj: Any,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Reads config.series_store (if present), applies defaults, validates fields,
    resolves base_dir, and returns:
      - normalized dict suitable to keep in config (Box-friendly)
      - a dict of human-readable 'effective' values for logging
    This function is side-effect-free (Step 0 only logs later).
    """
    # Defensive reads (works with Box or plain dicts)
    infra = getattr(config_obj, "infra", {}) or {}
    experiments_output_dir = str(getattr(infra, "experiments_output_dir", "") or infra.get("experiments_output_dir", ""))
    if not experiments_output_dir:
        logging.warning("[series_store] 'infra.experiments_output_dir' not set; base_dir resolution may be wrong.")

    raw = getattr(config_obj, "series_store", {}) or {}
    # If Box, convert to dict-like access without breaking
    try:
        raw = raw.to_dict()  # type: ignore[attr-defined]
    except Exception:
        pass

    # Start from defaults
    out: Dict[str, Any] = dict(_DEFAULTS)

    # Merge raw → defaults
    out.update({k: raw.get(k, out[k]) for k in _DEFAULTS.keys()})

    # Coerce types / validate
    out["enabled"] = _as_bool(out.get("enabled"), _DEFAULTS["enabled"])
    out["self_heal"] = _as_bool(out.get("self_heal"), _DEFAULTS["self_heal"])
    out["schema_version"] = int(out.get("schema_version") or _DEFAULTS["schema_version"])

    out["format"] = _pick(out.get("format"), _SUPPORTED_FORMATS, _DEFAULTS["format"], "format")
    out["compress"] = _pick(out.get("compress"), _SUPPORTED_CODECS, _DEFAULTS["compress"], "compress")

    base_dir = _resolve_base_dir(out.get("base_dir"), experiments_output_dir)
    out["base_dir"] = str(base_dir)

    # Prepare human-readable log lines
    readable = {
        "enabled": str(out["enabled"]),
        "format": out["format"],
        "compress": out["compress"],
        "schema_version": str(out["schema_version"]),
        "self_heal": str(out["self_heal"]),
        "base_dir": out["base_dir"],
    }

    return out, readable
