# src/utils/debug_dumps.py
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Any

def _json_default(o: Any):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
    except Exception:
        pass
    try:
        return str(o)
    except Exception:
        return None

def dump_debug(tag: str, payload: dict, base_dir: str | os.PathLike = "./debug_dumps") -> str:
    """
    Writes a single-line JSON file to ./debug_dumps/<timestamp>_<tag>.json
    Safe for NumPy/Pandas types and most nested dicts.
    Returns the written path as a string.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{tag}.json"
    out = base / fname
    # enrich payload with standard fields
    payload = dict(payload)
    payload.setdefault("tag", tag)
    payload.setdefault("timestamp", ts)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=_json_default)
    return str(out)
