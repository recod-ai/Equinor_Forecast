# src/profile_manager.py
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError, Field

# --- 1. Define the canonical schema for an experiment profile's content.
# This schema acts as a contract for the fields coming from the profile file.
# It ensures type safety and presence of required fields.
class ExperimentSchema(BaseModel):
    architecture_profile: Optional[str] = None
    physics_strategy: str
    lag_window: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    data_sample: float = Field(gt=0, le=1.0)
    seed: int

    # Optional fields for analysis & tracking
    experiment_id: Optional[str] = None
    profile_group: Optional[str] = "default_group"

    class Config:
        extra = 'allow'  # Allows other parameters to pass through without validation.

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

def load_and_expand_profile(
    profile_path: str | os.PathLike, 
    arch_defs_path: str | os.PathLike = '../../src/experiment_configs/architectures_PINNs.yaml'
) -> List[Dict[str, Any]]:
    """
    Loads an experiment profile, expands architecture references, and validates the result.

    This is the main entry point for this module.

    Args:
        profile_path: Path to the main experiment profile (CSV, XLSX, or YAML).
        arch_defs_path: Path to the YAML file containing architecture definitions.

    Returns:
        A list of fully expanded and validated experiment configuration dictionaries.
    """
    p = Path(profile_path)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found at: {p}")
    
    # Step 1: Read the base profile file into a list of dictionaries
    if p.suffix.lower() == ".csv":
        base_configs: List[Dict] = pd.read_csv(p).to_dict(orient="records")
    elif p.suffix.lower() in {".xlsx", ".xls"}:
        base_configs = pd.read_excel(p).to_dict(orient="records")
    elif p.suffix.lower() in {".yml", ".yaml"}:
        with open(p, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
        # Handle both single-list YAML and multi-document (---) YAML
        base_configs = docs[0] if isinstance(docs[0], list) else docs
    else:
        raise ValueError(f"Unsupported profile format: {p.suffix}")
        
    # Step 2: Expand architecture references
    arch_definitions = _load_arch_configs(arch_defs_path)
    expanded_configs: List[Dict[str, Any]] = []
    for config in base_configs:

        # Create a new dictionary excluding keys where the value is NaN.
        cleaned_config = {k: v for k, v in config.items() if pd.notna(v)}
        arch_id = config.get("architecture_profile")
        
        # Only expand if arch_id is a valid, non-empty string.
        if arch_id and pd.notna(arch_id):
            if arch_id not in arch_definitions:
                # This error is still correct if the ID is present but not found.
                raise KeyError(f"Architecture profile '{arch_id}' not found in definitions.")
            
            # The expansion logic: Start with the definition, then update with profile values.
            final_config = arch_definitions[arch_id].copy()
            final_config.update(config)
            expanded_configs.append(final_config)
        else:
            # If `architecture_profile` is missing or NaN (like for Seq2Trend),
            # just use the config from the profile as is.
            expanded_configs.append(cleaned_config)

    return _validate_and_prepare(expanded_configs)

def generate_job_hash(config: Dict[str, Any]) -> str:
    """Creates a deterministic MD5 hash from a configuration dictionary."""
    # Exclude non-essential keys from hash to ensure reproducibility if only core params change
    core_config = {k: v for k, v in config.items() if k not in ['experiment_id', 'profile_group']}
    payload = json.dumps(core_config, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()



def _validate_and_prepare(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validates configs against the Pydantic schema and auto-fills IDs."""
    validated_configs = []
    for i, config in enumerate(configs):
        try:
            # Pydantic will check for required fields and correct types
            model = ExperimentSchema(**config)
            # Re-create the dictionary from the validated model, including any extra fields
            validated_config = {**config, **model.dict()}
        except ValidationError as e:
            raise RuntimeError(f"Profile validation failed on row {i+1} for config:\n{config}\nError:\n{e}")
            
        # If experiment_id was not provided in the profile, generate one.
        if not validated_config.get("experiment_id"):
            validated_config["experiment_id"] = f'job_{generate_job_hash(validated_config)[:10]}'
        validated_configs.append(validated_config)
    return validated_configs