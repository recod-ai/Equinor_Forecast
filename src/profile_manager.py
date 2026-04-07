# src/profile_manager.py
from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError


from pydantic import BaseModel, ValidationError, Field
from typing import Literal

# ---------------------------
# Schema family discrimination
# ---------------------------

ProfileFamily = Literal["trainable", "arps"]

class _BaseExperimentSchema(BaseModel):
    """
    Common minimal fields across all families.
    Keep this light to avoid forcing ARPS to look like a trainable model.
    """
    architecture_name: str
    lag_window: int = Field(gt=0)
    horizon: int = Field(gt=0)
    seed: int

    # Optional fields for analysis & tracking
    experiment_id: Optional[str] = None
    profile_group: Optional[str] = "default_group"

    class Config:
        extra = "allow"


class TrainableExperimentSchema(_BaseExperimentSchema):
    """
    Seq2/Darts-like trainable configs (physics-in-loss, deep learning, etc.)
    """
    physics_strategy: str
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    data_sample: float = Field(gt=0, le=1.0)


class ArpsExperimentSchema(_BaseExperimentSchema):
    """
    ARPS / analytic configs: do NOT require epochs/lr/batch/etc.
    Only enforce what ARPS truly needs.
    """
    # optional for tracking only
    physics_strategy: Optional[str] = None

    # Core ARPS knobs (optional but validated if present)
    variant: Optional[str] = None
    solver: Optional[str] = None
    loss: Optional[str] = None
    weighting: Optional[str] = None

    # Grid-specific knobs (validated if present; conditional checks done in validator function)
    b_grid_kind: Optional[str] = None
    b_min: Optional[float] = None
    b_max: Optional[float] = None
    b_grid_size: Optional[int] = None



# # --- 1. Define the canonical schema for an experiment profile's content.
# # This schema acts as a contract for the fields coming from the profile file.
# # It ensures type safety and presence of required fields.
# class ExperimentSchema(BaseModel):
#     architecture_profile: Optional[str] = None
#     physics_strategy: str
#     lag_window: int = Field(gt=0)
#     epochs: int = Field(gt=0)
#     learning_rate: float = Field(gt=0)
#     batch_size: int = Field(gt=0)
#     data_sample: float = Field(gt=0, le=1.0)
#     seed: int

#     # Optional fields for analysis & tracking
#     experiment_id: Optional[str] = None
#     profile_group: Optional[str] = "default_group"

#     class Config:
#         extra = 'allow'  # Allows other parameters to pass through without validation.

class ExperimentSchema(BaseModel):
    """
    Backward-compatible public schema.
    Many parts of the codebase import ExperimentSchema from profile_manager.
    This wrapper delegates validation to the appropriate family schema.
    """
    class Config:
        extra = "allow"

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        family = _detect_profile_family(config)
        cfg = _normalize_for_schema(config, family=family)

        if family == "arps":
            _validate_arps_conditionals(cfg)
            model = ArpsExperimentSchema(**cfg)
        else:
            model = TrainableExperimentSchema(**cfg)

        # Return validated dict while preserving extra fields
        return {**cfg, **model.dict()}


def _detect_profile_family(c: Dict[str, Any]) -> ProfileFamily:
    """
    Decide which schema should validate this row.
    Primary signal: architecture_name.
    Fallback: presence of ARPS-like fields (variant/solver/b_grid_kind).
    """
    arch = str(c.get("architecture_name", "")).strip().lower()

    # strong signals
    if "arps" in arch:
        return "arps"

    # fallback signals (useful if someone names arch differently)
    if any(k in c for k in ("variant", "solver", "b_grid_kind", "b_min", "b_max", "b_grid_size")):
        return "arps"

    return "trainable"


def _apply_family_defaults(c: Dict[str, Any], family: ProfileFamily) -> Dict[str, Any]:
    """
    Add minimal safe defaults per family without polluting configs.
    """
    out = dict(c)

    if family == "arps":
        # Not required, but helps downstream filtering/logging if you expect it sometimes
        if _is_missing(out.get("physics_strategy")):
            out["physics_strategy"] = "arps"
        return out

    # trainable defaults
    if _is_missing(out.get("data_sample")):
        out["data_sample"] = 0.0001

    return out


def _validate_arps_conditionals(c: Dict[str, Any]) -> None:
    """
    Extra conditional checks beyond pydantic basic typing.
    Keep this tiny and explicit.
    """
    solver = str(c.get("solver", "")).strip().lower()

    if solver == "grid":
        missing = []
        for k in ("b_grid_kind", "b_grid_size", "b_min", "b_max"):
            if _is_missing(c.get(k)):
                missing.append(k)
        if missing:
            raise ValueError(f"ARPS solver='grid' requires fields: {missing}")

        # optional sanity checks
        try:
            bmin = float(c["b_min"])
            bmax = float(c["b_max"])
            if not (bmax > bmin):
                raise ValueError("ARPS requires b_max > b_min")
        except Exception:
            raise ValueError("ARPS grid requires numeric b_min and b_max")



# --- Global cache for architecture configs to avoid re-reading the file ---
_ARCH_CONFIG_CACHE: Optional[Dict[str, Any]] = None

def _load_arch_configs(path: str | os.PathLike) -> Dict[str, Any]:
    """Loads and caches architecture definitions from a YAML file."""
    global _ARCH_CONFIG_CACHE
    if _ARCH_CONFIG_CACHE is None:
        arch_path = Path(path)
        if not arch_path.exists():
            raise FileNotFoundError(f"Architecture definition file not found: {arch_path}")
        with open(arch_path, 'r', encoding='utf-8') as f:
            _ARCH_CONFIG_CACHE = yaml.safe_load(f)
            print(f"INFO: Cached {len(_ARCH_CONFIG_CACHE)} architecture profiles from {arch_path}")
    return _ARCH_CONFIG_CACHE


_ARCH_CONFIG_CACHE = None

def _is_na(x):
    return (
        x is None
        or (isinstance(x, float) and math.isnan(x))
        or (isinstance(x, str) and x.strip() == "")
    )

def _read_profile_rows(profile_path: str) -> list[dict]:
    p = Path(profile_path)
    if p.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_csv(p, sep="\t")
        return df.to_dict(orient="records")
    elif p.suffix.lower() in {".yml", ".yaml"}:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        return data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"Unsupported profile format: {p.suffix}")

def _normalize_profile_row(row: dict) -> dict:
    """Unify column names across Seq2/Darts; drop NaNs."""
    r = {k: v for k, v in row.items() if not _is_na(v)}

    # architecture column: allow 'architecture' or 'architecture_name'
    if "architecture" in r and "architecture_name" not in r:
        r["architecture_name"] = r.pop("architecture")

    # profile column: allow 'architecture_profile' or 'profile'
    if "architecture_profile" in r and "profile" not in r:
        r["profile"] = r["architecture_profile"]

    return r

def _load_arch_configs(path: str | None) -> dict:
    """Return {} when path is None or missing (pass-through mode for Darts validation)."""
    global _ARCH_CONFIG_CACHE
    if path is None:
        logging.info("No arch_defs_path provided; skipping architecture expansion.")
        return {}
    p = Path(path)
    if not p.exists():
        logging.warning(f"Architecture definition file not found: {p} – proceeding without expansion.")
        return {}
    if _ARCH_CONFIG_CACHE is not None:
        return _ARCH_CONFIG_CACHE
    with open(p, "r", encoding="utf-8") as f:
        _ARCH_CONFIG_CACHE = yaml.safe_load(f) or {}
    return _ARCH_CONFIG_CACHE

def load_and_expand_profile(profile_path: str, arch_defs_path: str | None):
    """
    Load a 'rich' profile and expand rows using YAML when available.
    If no YAML exists, treat rows as already-expanded (Darts-friendly).
    """
    base_rows = [_normalize_profile_row(r) for r in _read_profile_rows(profile_path)]
    arch_defs = _load_arch_configs(arch_defs_path)

    # No definitions → pass-through (this is what we want for Darts validation)
    if not arch_defs:
        logging.info("No architecture definitions loaded; treating profile as already-expanded.")
        return base_rows

    # Otherwise (Seq2 path): expand rows by (architecture_name, profile)
    expanded = []
    for r in base_rows:
        prof = r.get("profile")
        arch = r.get("architecture_name")
        if _is_na(prof) or _is_na(arch):
            # row already explicit or not expandable → keep as-is
            expanded.append(r)
            continue

        arch_bucket = arch_defs.get(arch, {})
        prof_dict = arch_bucket.get(prof)
        if not prof_dict:
            logging.warning(f"Profile '{prof}' not found for architecture '{arch}'. Keeping row as-is.")
            expanded.append(r)
            continue

        rr = dict(r)
        # Merge profile defaults; row values win
        for k, v in prof_dict.items():
            rr.setdefault(k, v)
        expanded.append(rr)

    return _validate_and_prepare(expanded)


def generate_job_hash(config: Dict[str, Any]) -> str:
    """Creates a deterministic MD5 hash from a configuration dictionary."""
    # Exclude non-essential keys from hash to ensure reproducibility if only core params change
    core_config = {k: v for k, v in config.items() if k not in ['experiment_id', 'profile_group']}
    payload = json.dumps(core_config, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _coerce_list(val, of=float):
    """
    Try to coerce a CSV-loaded value into a list of the desired type.
    Accepts list/tuple directly or a string like "[0.25, 0.5, 0.75]".
    """
    if isinstance(val, (list, tuple)):
        try:
            return [of(x) for x in val]
        except Exception:
            return list(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return [of(x) for x in parsed]
        except Exception:
            pass
    return None

def _is_missing(v) -> bool:
    # Treat None, NaN and empty strings as missing
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False

def _normalize_for_schema(config: Dict[str, Any], family: ProfileFamily = "trainable") -> Dict[str, Any]:
    c = dict(config)

    # Apply family defaults FIRST (so later coercions see consistent fields)
    c = _apply_family_defaults(c, family)

    # ---- trainable-only normalizations ----
    if family == "trainable":
        # epochs: map Darts 'n_epochs' -> 'epochs' if epochs missing/NaN
        if _is_missing(c.get("epochs")) and not _is_missing(c.get("n_epochs")):
            try:
                c["epochs"] = int(float(c["n_epochs"]))
            except Exception:
                pass

        # data_sample: schema requires >0.0 (gt=0)
        if _is_missing(c.get("data_sample")):
            c["data_sample"] = 0.0001

    # ---- shared normalization (applies to both) ----

    # aggregation_quantiles: coerce from string if needed; set default if missing
    aq = c.get("aggregation_quantiles")
    coerced = _coerce_list(aq, float)
    if coerced is not None:
        c["aggregation_quantiles"] = coerced
    elif _is_missing(aq):
        c["aggregation_quantiles"] = [0.25, 0.5, 0.75]

    # 4) Common booleans that can arrive as strings
    for b in ("use_past_covariates", "plot", "evaluate_by_slice",
              "use_known_good", "show_components"):
        if b in c and isinstance(c[b], str):
            c[b] = c[b].strip().lower() in ("1", "true", "yes", "y", "t")

    # 5) Int-like fields as strings
    for k in ("lag_window", "horizon", "batch_size",
              "input_chunk_length", "output_chunk_length", "patience"):
        if isinstance(c.get(k), str):
            try:
                c[k] = int(float(c[k]))
            except Exception:
                pass

    # 6) Floats
    if isinstance(c.get("learning_rate"), str):
        try:
            c["learning_rate"] = float(c["learning_rate"])
        except Exception:
            pass

    # 7) Darts: coerce "scalar-int" hyperparams that sometimes arrive as float or singleton list/tuple
    def _to_int_if_scalar(x):
        # leave non-integer floats as-is; only coerce 256.0 -> 256, "32" -> 32, ("64",) -> 64
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            try:
                return int(x) if float(x).is_integer() else x
            except Exception:
                return x
        if isinstance(x, str):
            try:
                xf = float(x)
                return int(xf) if xf.is_integer() else x
            except Exception:
                try:
                    return int(x)
                except Exception:
                    return x
        return x

    def _scalarize(v):
        # convert [64] or (64,) -> 64
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return v[0]
        return v

    # TiDE / deep models common knobs that must be ints
    for k in (
        "hidden_size",                # TiDE
        "temporal_width",             # TiDE
        "temporal_decoder_hidden_size",  # TiDE
        "kernel_size",                # TiDE, TCN, etc.
        "num_encoder_layers",         # TiDE/Transformer
        "num_decoder_layers",         # TiDE/Transformer
        "n_heads", "attention_heads", # Transformer families (future-proofing)
        "num_stacks", "num_blocks", "num_layers",  # N-HiTS variants & others
        "decoder_output_dim", "temporal_width_past", "temporal_width_future"
    ):
        if k in c and c[k] is not None:
            c[k] = _scalarize(c[k])
            c[k] = _to_int_if_scalar(c[k])

    # N-HiTS: layer_widths can be an int or list[int]; ensure entries are ints if list
    lw = c.get("layer_widths")
    if lw is not None:
        if isinstance(lw, (list, tuple)):
            new_lw = []
            for v in lw:
                v2 = _to_int_if_scalar(v)
                new_lw.append(v2 if isinstance(v2, int) else v)
            c["layer_widths"] = new_lw
        else:
            v2 = _to_int_if_scalar(lw)
            c["layer_widths"] = v2 if isinstance(v2, int) else lw

    # 8) LinearRegression: ensure lags arguments are int or list[int]
    def _coerce_lags(val):
        if val is None:
            return val
        # parse stringified lists: "[1, 2, 3]"
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                val = parsed
            except Exception:
                pass
        if isinstance(val, (list, tuple)):
            out = []
            for v in val:
                v2 = _to_int_if_scalar(v)
                if isinstance(v2, int) and v2 > 0:
                    out.append(v2)
            return out if out else None
        else:
            v2 = _to_int_if_scalar(val)
            return v2 if isinstance(v2, int) and v2 > 0 else None

    for lk in ("lags", "lags_past_covariates", "lags_future_covariates"):
        if lk in c:
            coerced = _coerce_lags(c[lk])
            if coerced is not None:
                c[lk] = coerced
            else:
                # if bad/zero value came in, drop it to let model defaults apply
                c.pop(lk, None)

    # 9) ARIMA: keep the prior seasonal sanitation (already added earlier)
    arch_lower = str(c.get("architecture_name", "")).lower()
    if "darts_arima" in arch_lower or c.get("physics_strategy", "").lower() == "arima":
        c.pop("m", None)  # statsmodels.ARIMA doesn't use 'm'
        so = c.get("seasonal_order")
        if isinstance(so, str):
            try:
                c["seasonal_order"] = ast.literal_eval(so)
                so = c["seasonal_order"]
            except Exception:
                pass
        if not (isinstance(so, (list, tuple)) and len(so) == 4):
            c.pop("seasonal_order", None)
            if c.get("seasonal", False):
                c["seasonal"] = False

    c = _normalize_aggregation_fields(c)
    return c


def _coerce_str_list(val) -> Optional[List[str]]:
    """
    Coerce val to list[str] if possible.
    Accepts list/tuple directly or a string like "['a','b']".
    """
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val if str(x).strip() != ""]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed if str(x).strip() != ""]
        except Exception:
            return None
    return None


def _normalize_aggregation_fields(c: Dict[str, Any]) -> Dict[str, Any]:
    """
    Plug-and-play guardrails for aggregation settings.

    Goals:
      - aggregation_method must be a short string (never a list-literal).
      - aggregation_candidates becomes list[str] when provided (stringified or list).
      - When method is corrupted (e.g., "['a','b']"), convert it to candidates
        and set method='auto' (or a safe default).
      - Optionally set aggregation_sweep=True when candidates exist.
    """
    out = dict(c)

    def _is_list_literal_str(x: Any) -> bool:
        return isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]")

    # 1) Parse candidates if present (string -> list[str])
    cand_raw = out.get("aggregation_candidates")
    cand_list = _coerce_str_list(cand_raw)
    if cand_list is not None:
        out["aggregation_candidates"] = cand_list

    # 2) Fix corrupted aggregation_method that is actually a list-literal string
    meth_raw = out.get("aggregation_method")

    if _is_list_literal_str(meth_raw):
        parsed = _coerce_str_list(meth_raw)
        if parsed:
            # treat it as candidates
            out["aggregation_candidates"] = parsed
        # choose safe behavior: auto sweep/pick later
        out["aggregation_method"] = "auto"

    # 3) If aggregation_method is missing/blank and candidates exist, set auto
    meth = out.get("aggregation_method")
    if (meth is None) or (isinstance(meth, str) and meth.strip() == ""):
        if isinstance(out.get("aggregation_candidates"), list) and out["aggregation_candidates"]:
            out["aggregation_method"] = "auto"

    # 4) If candidates exist, make sweep explicit (non-breaking; your eval checks aggregation_sweep)
    if isinstance(out.get("aggregation_candidates"), list) and out["aggregation_candidates"]:
        if "aggregation_sweep" not in out:
            out["aggregation_sweep"] = True
        # also normalize common auto tokens if user put "all/sweep" etc
        if isinstance(out.get("aggregation_method"), str):
            m = out["aggregation_method"].strip().lower()
            if m in {"all", "sweep"}:
                out["aggregation_method"] = "auto"

    return out


# def _validate_and_prepare(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Validates configs against the Pydantic schema and auto-fills IDs."""
#     validated_configs = []
#     for i, config in enumerate(configs):
#         try:
#             # Pydantic will check for required fields and correct types
#             config = _normalize_for_schema(config)
#             model = ExperimentSchema(**config)
#             # Re-create the dictionary from the validated model, including any extra fields
#             validated_config = {**config, **model.dict()}
#         except ValidationError as e:
#             raise RuntimeError(f"Profile validation failed on row {i+1} for config:\n{config}\nError:\n{e}")
            
#         # If experiment_id was not provided in the profile, generate one.
#         if not validated_config.get("experiment_id"):
#             validated_config["experiment_id"] = f'job_{generate_job_hash(validated_config)[:10]}'
#         validated_configs.append(validated_config)
#     return validated_configs

def _validate_and_prepare(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated_configs: List[Dict[str, Any]] = []

    for i, raw in enumerate(configs):
        try:
            validated_config = ExperimentSchema.validate_config(raw)
        except Exception as e:
            raise RuntimeError(
                f"Profile validation failed on row {i+1}.\n"
                f"Config:\n{raw}\n\nError:\n{e}"
            )

        if not validated_config.get("experiment_id"):
            validated_config["experiment_id"] = f'job_{generate_job_hash(validated_config)[:10]}'

        validated_configs.append(validated_config)

    return validated_configs
