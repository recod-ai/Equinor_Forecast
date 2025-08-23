# File: src/config_loader.py

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from box import Box

# =========================================================================
# === EXISTING CODE (UNCHANGED) ===========================================
# =========================================================================
# Your existing models and functions are preserved to maintain compatibility.

class BlockConfig(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

class ArchitectureConfig(BaseModel):
    architecture_id: str
    summary: str
    builder: str
    blocks: Optional[List[BlockConfig]] = None
    params: Optional[Dict[str, Any]] = None

class HyperparameterProfile(BaseModel):
    hyperparam_id: str
    summary: str
    overrides: Dict[str, Any] = Field(default_factory=dict)

def load_yaml_config(file_path: Path) -> List[Dict]:
    """Loads and parses a YAML configuration file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_experiment_configs(config_dir: str | Path = "experiment_configs"):
    """
    Loads and validates all architecture and hyperparameter configurations.
    """
    base_path = Path(__file__).resolve().parent / config_dir
    
    arch_data = load_yaml_config(base_path / "architectures.yaml")
    architectures = [ArchitectureConfig(**item) for item in arch_data]
    
    hp_data = load_yaml_config(base_path / "hyperparameters.yaml")
    hyperparams = [HyperparameterProfile(**item) for item in hp_data]
    
    print(f"INFO: Loaded {len(architectures)} architectures and {len(hyperparams)} hyperparameter profiles.")
    
    return {
        "architectures": architectures,
        "hyperparameters": hyperparams
    }

# =========================================================================
# === NEW FUNCTIONALITY (ADDED) ===========================================
# =========================================================================

def load_campaign_config(path: Union[str, Path]) -> Box:
    """
    Loads a generic YAML campaign configuration file from the given path.

    This function is designed to load files like 'f14_context_volve.yaml'
    and provides easy, attribute-style access to its contents.

    Args:
        path (Union[str, Path]): The path to the YAML campaign file.

    Returns:
        Box: A Box object that allows accessing config values with dot notation
             (e.g., config.run_scope.wells).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Campaign configuration file not found: {path}")
        
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert the dictionary to a Box object for convenient access.
    # default_box=True ensures that nested dictionaries also become Box objects.
    return Box(config_dict, default_box=True)