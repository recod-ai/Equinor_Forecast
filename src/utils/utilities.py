# src/utils/utilities.py
from __future__ import annotations

import inspect
import json
import re
from typing import Any, Dict, Optional

import numpy as np

import os
import shutil

from pathlib import Path

import glob
import logging


# ---------------------------
# Generic signature helpers
# ---------------------------

def _func_accepts_var_kwargs(func) -> bool:
    """Return True if the function accepts **kwargs."""
    try:
        sig = inspect.signature(func)
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        # Be permissive when we cannot inspect (keeps runtime robust)
        return True


def _filter_kwargs_for(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pass only kwargs that the function can accept (unless it has **kwargs).
    This prevents 'got an unexpected keyword argument' errors when APIs evolve.
    """
    if _func_accepts_var_kwargs(func):
        return kwargs
    try:
        allowed = set(inspect.signature(func).parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


# ---------------------------
# ARPS-specific adapters
# ---------------------------

_SOLVER_ALIASES = {
    "wls": "grid",              # legacy label for analytic WLS → 'grid'
    "nelder": "nelder-mead",
    "nelder_mead": "nelder-mead",
    "continuous": "lbfgs",      # older name meaning "continuous optimization"
}

def _normalize_solver(raw: Optional[str]) -> str:
    """
    Normalize solver/method labels to the canonical set accepted by ARPS fit.
    """
    s = (raw or "grid").strip().lower()
    return _SOLVER_ALIASES.get(s, s)


def _parse_b_grid_repr(text: Any) -> Optional[np.ndarray]:
    """
    Parse a human-friendly 'b_grid' representation into a numpy array.

    Accepted formats:
      - "linspace(a,b,n)"
      - "logspace(a,b,n)"   (a,b are exponents, e.g., logspace(-2, 0, 5))
      - JSON list string, e.g., "[0.1, 0.2, 0.5]"
    Returns None if parsing fails.
    """
    if not isinstance(text, str):
        return None

    m = re.match(r"\s*linspace\(\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*,\s*(\d+)\s*\)\s*$", text)
    if m:
        a, b, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
        return np.linspace(a, b, max(3, n))

    m = re.match(r"\s*logspace\(\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*,\s*(\d+)\s*\)\s*$", text)
    if m:
        a, b, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
        return np.logspace(a, b, max(3, n))

    if text.strip().startswith("[") and text.strip().endswith("]"):
        try:
            arr = json.loads(text)
            return np.asarray(arr, dtype=float)
        except Exception:
            return None

    return None


def _adapt_arps_kwargs_for_fit(fit_func, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert heterogeneous/legacy ARPS params into kwargs compatible with the current
    `fit_arps_canonical` signature. Also materializes `b_grid` when solver='grid'
    (using b_min/b_max/b_grid_size/b_grid_kind), and clamps safe ranges.

    - Accepts both flat and legacy names:
        variant / arps_variant
        weighting / arps_weighting
        solver / method
        piecewise / allow_regime_change
        piecewise_min_delta_bic / cp_search
        loss / loss_scale (→ loss_delta)
        b_grid (str/list/np.ndarray) OR b_min/b_max/b_grid_size/b_grid_kind
    - If loss='quantile' and quantile_tau missing, default to 0.5.
    - burn_in_fraction is clamped to [0.0, 0.2].

    Returns a filtered dict that can be splatted into `fit_arps_canonical`.
    """
    k: Dict[str, Any] = dict(params or {})

    # Unify solver/method
    raw_solver = k.pop("solver", None)
    raw_method = k.pop("method", None)
    chosen_solver = raw_solver if raw_solver is not None else raw_method
    k["solver"] = _normalize_solver(chosen_solver)

    # Variant & weighting (keep given values; fallback to legacy names)
    if "variant" not in k and "arps_variant" in k:
        k["variant"] = k.pop("arps_variant")

    if "weighting" not in k and "arps_weighting" in k:
        k["weighting"] = k.pop("arps_weighting")

    # Loss scale alias
    if "loss_delta" not in k and "loss_scale" in k:
        try:
            k["loss_delta"] = float(k.pop("loss_scale"))
        except Exception:
            k.pop("loss_scale", None)

    # Piecewise aliases
    if "piecewise" not in k and "allow_regime_change" in k:
        k["piecewise"] = bool(k.pop("allow_regime_change"))

    if "piecewise_min_delta_bic" not in k and "cp_search" in k:
        try:
            k["piecewise_min_delta_bic"] = float(k.pop("cp_search"))
        except Exception:
            # Remove unusable alias value and keep default
            k.pop("cp_search", None)

    # Quantile tau default
    if str(k.get("loss", "wls")).lower() == "quantile" and "quantile_tau" not in k:
        k["quantile_tau"] = 0.5

    # Clamp burn-in
    if "burn_in_fraction" in k:
        try:
            b = float(k["burn_in_fraction"])
        except Exception:
            b = 0.0
        k["burn_in_fraction"] = float(max(0.0, min(0.2, b)))

    # b_grid handling
    # 1) Try direct b_grid (supports str/list/np.ndarray)
    if "b_grid" in k and k["b_grid"] is not None:
        if isinstance(k["b_grid"], str):
            parsed = _parse_b_grid_repr(k["b_grid"])
            if parsed is not None:
                k["b_grid"] = parsed
            else:
                k.pop("b_grid", None)
        elif not isinstance(k["b_grid"], np.ndarray):
            try:
                k["b_grid"] = np.asarray(k["b_grid"], dtype=float)
            except Exception:
                k.pop("b_grid", None)

    # 2) If solver='grid' and b_grid still missing, build it from bounds + size + kind
    if str(k.get("solver", "grid")).lower() == "grid" and k.get("b_grid") is None:
        try:
            b_min = float(k.pop("b_min", 0.1) or 0.1)
        except Exception:
            b_min = 0.1
        try:
            b_max = float(k.pop("b_max", 2.0) or 2.0)
        except Exception:
            b_max = 2.0
        try:
            size = int(k.pop("b_grid_size", 21) or 21)
        except Exception:
            size = 21
        kind = str(k.pop("b_grid_kind", "lin")).lower()

        size = max(3, size)
        if kind.startswith("log"):
            b_min = max(1e-6, b_min)
            b_max = max(b_min * 1.001, b_max)
            k["b_grid"] = np.exp(np.linspace(np.log(b_min), np.log(b_max), size))
        else:
            k["b_grid"] = np.linspace(b_min, max(b_min + 1e-6, b_max), size)

    # Strip helper-only keys if still present (not consumed by grid construction)
    for helper in ("b_min", "b_max", "b_grid_size", "b_grid_kind"):
        k.pop(helper, None)

    # Final filtering against the fit function signature
    return _filter_kwargs_for(fit_func, k)

# =============================================================================
# Helper Functions (General)
# =============================================================================

def clean_checkpoint_files(checkpoint_dir="model_checkpoint"):
    """Remove all files in the checkpoint directory."""
    for file in glob.glob(os.path.join(checkpoint_dir, "*")):
        os.remove(file)
    logging.info("Checkpoint directory cleaned.")
    



def clean_checkpoint_folder(checkpoint_folder='model_checkpoint', max_models=100) -> None:
    """
    Cleans the checkpoint folder by removing all its contents if it contains more than max_models items.

    Parameters:
        checkpoint_folder (str): Path to the folder containing model checkpoints.
        max_models (int): Maximum allowed number of model checkpoints before cleaning.
    """
    folder = Path(checkpoint_folder)
    if not folder.exists():
        return  # Nothing to clean if the folder doesn't exist

    items = list(folder.iterdir())
    if len(items) > max_models:
        for item in items:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

# Example usage: at the beginning of your program
if __name__ == '__main__':
    checkpoint_path = 'model_checkpoint'
    max_allowed_models = 10  # Set your threshold here
    clean_checkpoint_folder(checkpoint_path, max_allowed_models)



def generate_jobs(data_sources, wells, experiments):
    """Generate a list of experiment jobs with a unique counter."""
    jobs = []
    exp_counter = 1
    for config in data_sources:
        for well in wells:
            for exp_params in experiments:
                jobs.append((config, well, exp_params, exp_counter))
                exp_counter += 1
    return jobs


def delete_all_files_in_folder(folder_path):
    """
    Apaga todos os arquivos e subdiretórios dentro da pasta especificada.
    
    Args:
        folder_path (str): Caminho para a pasta onde os arquivos devem ser apagados.
    
    Returns:
        None
    """
    # Verifica se o diretório existe e é uma pasta
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Itera sobre todos os arquivos e subdiretórios na pasta
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Verifica se é um arquivo ou um link simbólico e apaga
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    print(f"Arquivo {file_path} apagado com sucesso.")
                # Se for um diretório, apaga recursivamente
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Diretório {file_path} apagado com sucesso.")
            except Exception as e:
                print(f"Erro ao apagar {file_path}. Motivo: {e}")
    else:
        print(f"Pasta {folder_path} não encontrada ou não é um diretório.")

        
from rich.console import Console
from rich.text import Text
import pyfiglet
        
def print_style(text):
    # Cria um console rich para exibir o texto estilizado
    console = Console()

    # Gera o texto em ASCII art com pyfiglet
    ascii_art = pyfiglet.figlet_format(text, font="standard")

    # Cria um objeto Text para adicionar cor e estilo
    styled_text = Text(ascii_art, style="italic blue")


    # Imprime o texto estilizado
    console.print(styled_text)
    
    
    
from typing import Any, Callable, Dict, List

def apply_filter_to_predictions(
    y_pred_list: List[List[float]],
    filter_function: Callable
) -> List[List[float]]:
    """
    Applies a filter function to the prediction lists for each well.

    Parameters:
    - y_pred_list (List[List[float]]): List of prediction lists per well.
    - filter_function (Callable): The filtering function to apply.

    Returns:
    - List[List[float]]: The filtered prediction lists.
    """
    y_pred_list_filter = []
    for current_data in y_pred_list:
        if current_data:
            filtered_data = filter_function(current_data)
            y_pred_list_filter.append(filtered_data.tolist())
        else:
            y_pred_list_filter.append([])
    print("Applied Filter")
    return y_pred_list_filter



import numpy as np
import pandas as pd

def check_data(*datasets):
    all_ok = True
    
    for i, data in enumerate(datasets):
        name = f'dataset_{i+1}'
        issues_found = False
        
        print(f'Checking {name}...')
        
        if isinstance(data, np.ndarray):
            if np.isnan(data).any():
                nan_positions = np.argwhere(np.isnan(data))
                print(f'  ❌ NaN detected in {name}, count: {nan_positions.shape[0]}')
                print(f'  NaN positions (first 10 shown):\n {nan_positions[:10]}')
                issues_found = True
            if np.isinf(data).any():
                inf_positions = np.argwhere(np.isinf(data))
                print(f'  ❌ Infinite values detected in {name}, count: {inf_positions.shape[0]}')
                print(f'  Infinite positions (first 10 shown):\n {inf_positions[:10]}')
                issues_found = True
            if np.abs(data).max() > 1e6:
                print(f'  ⚠️ Potential outliers detected in {name}')
                issues_found = True
        
        elif isinstance(data, pd.DataFrame):
            nan_count = data.isna().sum().sum()
            if nan_count > 0:
                nan_positions = data.isna()
                print(f'  ❌ NaN detected in {name}, count: {nan_count}')
                print(f'  NaN positions:\n{nan_positions[nan_positions].stack().index.tolist()[:10]}')
                issues_found = True
            if np.isinf(data.to_numpy()).sum() > 0:
                inf_positions = np.isinf(data.to_numpy())
                print(f'  ❌ Infinite values detected in {name}')
                print(f'  Infinite positions:\n{list(zip(*np.where(inf_positions)))[:10]}')
                issues_found = True
            if ((data.dtypes != 'object') & (data.abs() > 1e6).any()).any():
                print(f'  ⚠️ Potential outliers detected in {name}')
                issues_found = True
        
        if not issues_found:
            print(f'  ✅ No issues detected in {name}')
        else:
            all_ok = False
        
    if all_ok:
        print("🎉 All datasets passed the check with no issues!")
    else:
        print("⚠️ Some issues were found in the datasets.")

import joblib
import os
import warnings
import logging

"""Suppress TensorFlow and addon warnings for a cleaner console."""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='tensorflow_addons'
)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# =====================================================
# Utility Functions for Scaling Inversion
# =====================================================
def load_scaler(filepath: str):
    """
    Loads a scaler from a pickle file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scaler file not found at {filepath}.")
    scaler = joblib.load(filepath)
    tf.print(f"Scaler successfully loaded from {filepath}.")
    return scaler

def invert_feature_scaling(feature_scaled, scaler_X_mean, scaler_X_std, feature_idx: int):
    """
    Inverts the scaling of a feature given its index in the scaler arrays.
    
    Parameters:
      - feature_scaled: Scaled feature tensor.
      - scaler_X_mean: Mean values from the input scaler.
      - scaler_X_std: Standard deviation values from the input scaler.
      - feature_idx: Index of the feature in the scaler arrays.
      
    Returns:
      - The original (unscaled) feature.
    """
    feature_mean = scaler_X_mean[feature_idx]
    feature_std = scaler_X_std[feature_idx]
    return feature_scaled * feature_std + feature_mean

def get_center_and_scale(scaler, as_tf=True, dtype=tf.float32):
    """
    Devolve (center, scale) para qualquer scaler compatível com scikit-learn.
    Se as_tf=True, retorna tf.Tensor; caso contrário, retorna NumPy ndarray.
    center = média (Standard) ou mediana (Robust) ou quantil mediano (QuantileTransformer)  
    scale  = desvio-padrão (Standard) ou IQR (Robust) ou escala do QuantileTransformer  
    """
    # 1) Extrair arrays NumPy
    if hasattr(scaler, "mean_"):          # StandardScaler
        center_np = scaler.mean_
    elif hasattr(scaler, "center_"):      # RobustScaler
        center_np = scaler.center_
    elif hasattr(scaler, "quantiles_"):   # QuantileTransformer
        # para QuantileTransformer, o "quantiles_" é um array onde o elemento do meio é a mediana
        n = len(scaler.quantiles_) // 2
        center_np = scaler.quantiles_[n]
    else:
        raise AttributeError("Scaler sem atributo de centro conhecido")

    if not hasattr(scaler, "scale_"):
        raise AttributeError("Scaler sem atributo scale_")
    scale_np = scaler.scale_

    if as_tf:
        center_tf = tf.constant(center_np, dtype=dtype)
        scale_tf  = tf.constant(scale_np,  dtype=dtype)
        return center_tf, scale_tf
    else:
        return center_np, scale_np


import numpy as np
import tensorflow as tf
from typing import Tuple, Any, Union

def get_center_and_scale(
    scaler: Any, 
    as_tf: bool = True, 
    dtype: tf.DType = tf.float32
) -> Union[Tuple[tf.Tensor, tf.Tensor], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (center, scale) for any scikit-learn compatible scaler.

    If as_tf=True, returns tf.Tensor; otherwise, returns NumPy ndarray.
    This function intelligently extracts the equivalent of a location and scale
    parameter for various scaler types.

    Scaler Logic:
    - StandardScaler: center=mean_, scale=scale_ (std dev)
    - RobustScaler: center=center_ (median), scale=scale_ (IQR)
    - QuantileTransformer: center=quantiles_[median_idx], scale=scale_ (IQR, approx.)
    - MinMaxScaler: center=min_, scale=(max_ - min_)
    - MaxAbsScaler: center=0, scale=max_abs_
    - PowerTransformer: Returns the stats of its internal Standard/RobustScaler.
    """
    # --- 1. Handle PowerTransformer ---
    # PowerTransformer contains another scaler internally. We analyze that one.
    if hasattr(scaler, 'lambdas_'): # Heuristic to identify PowerTransformer
        # After fitting, PowerTransformer stores a final scaler (usually StandardScaler)
        if hasattr(scaler, '_scaler'): 
            return get_center_and_scale(scaler._scaler, as_tf=as_tf, dtype=dtype)
        else:
            # If not fitted or structure is unexpected, return identity
            center_np = np.array([0.0])
            scale_np = np.array([1.0])
    
    # --- 2. Extract Center Parameter ---
    elif hasattr(scaler, "mean_"):      # StandardScaler
        center_np = scaler.mean_
    elif hasattr(scaler, "center_"):    # RobustScaler
        center_np = scaler.center_
    elif hasattr(scaler, "min_"):       # MinMaxScaler
        center_np = scaler.min_
    elif hasattr(scaler, "quantiles_"): # QuantileTransformer
        # For QuantileTransformer, the median is the center of the learned quantiles
        median_index = scaler.quantiles_.shape[0] // 2
        center_np = scaler.quantiles_[median_index]
    elif hasattr(scaler, "max_abs_"):   # MaxAbsScaler
        # MaxAbsScaler centers at 0 by definition
        # We need to determine the number of features to create a correctly shaped array
        num_features = scaler.max_abs_.shape[0]
        center_np = np.zeros(num_features)
    else:
        raise AttributeError(f"Scaler of type {type(scaler).__name__} does not have a known center attribute.")

    # --- 3. Extract Scale Parameter ---
    if hasattr(scaler, "scale_"):       # StandardScaler, RobustScaler, QuantileTransformer
        scale_np = scaler.scale_
    elif hasattr(scaler, "min_") and hasattr(scaler, "data_max_"): # MinMaxScaler
        # The scale for MinMaxScaler is the range (max - min)
        scale_np = scaler.data_max_ - scaler.min_
    elif hasattr(scaler, "max_abs_"):   # MaxAbsScaler
        scale_np = scaler.max_abs_
    elif hasattr(scaler, '_scaler') and hasattr(scaler._scaler, "scale_"): # PowerTransformer check
        scale_np = scaler._scaler.scale_
    else:
        raise AttributeError(f"Scaler of type {type(scaler).__name__} does not have a known scale attribute.")
        
    # Ensure scale is never zero to avoid division issues
    scale_np[scale_np == 0] = 1.0

    # --- 4. Convert to TensorFlow Tensor if requested ---
    if as_tf:
        center_tf = tf.constant(center_np.astype(np.float32), dtype=dtype)
        scale_tf = tf.constant(scale_np.astype(np.float32), dtype=dtype)
        return center_tf, scale_tf
    else:
        return center_np, scale_np

# =============================================================================
# X. FEW-SHOT JOBS
# =============================================================================
from pathlib import Path
import shutil
import subprocess
import sys
import time
import os
import psutil
import pandas as pd
from itertools import product
from typing import Any, Dict, List, Tuple

from common.config_wells import get_data_sources
from forecast_pipeline.config import ExecutionMode


def prompt_and_clean_workspace(
    clean_dir: Path,
    manifest_path: Path,
    generator_script: Path
) -> bool:
    """
    Prompts user to confirm destructive cleanup of output and regenerates manifest.
    Returns True if cleanup and regeneration succeed.
    """
    print("=" * 60)
    print("🧹 CLEANUP MODE ENABLED (START_FRESH = True)")
    print("=" * 60)
    print(f"WARNING: This action is DESTRUCTIVE and will delete all contents of:")
    print(f"  - Output Directory: '{clean_dir}'")
    print(f"  - Old Manifest: '{manifest_path}'")
    confirm = input("\n> To continue, type 'CONFIRM' and press Enter: ")
    if confirm != "CONFIRM":
        print("\nOperation cancelled by user. No files were changed.")
        return False

    # Remove and recreate output directory
    if clean_dir.exists():
        try:
            shutil.rmtree(clean_dir)
            print(f"  - Directory '{clean_dir}' successfully deleted.")
        except Exception as e:
            print(f"  - ERROR deleting directory '{clean_dir}': {e}")
            return False
    clean_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - Directory '{clean_dir}' recreated.")

    # Remove old manifest
    if manifest_path.exists():
        try:
            manifest_path.unlink()
            print(f"  - Old manifest '{manifest_path}' deleted.")
        except Exception as e:
            print(f"  - ERROR deleting manifest '{manifest_path}': {e}")
            return False

    # Generate new manifest
    print("\nGenerating new manifest...")
    try:
        result = subprocess.run(
            [sys.executable, str(generator_script), "--output-path", str(manifest_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print("Manifest generator output:")
        print(result.stdout)
        print("✅ New manifest successfully generated.")
        return True
    except subprocess.CalledProcessError as e:
        print("ERROR generating the new manifest!")
        print(e.stderr)
        return False


def generate_job_filenames(
    job: Dict[str, Any],
    mode: ExecutionMode,
    output_dir: Path
) -> Tuple[str, Path, Path]:
    """
    Generates a unique job identifier and corresponding notebook & log file paths.
    """
    well_id = job.get('well', '').replace('/', '_')
    if mode == ExecutionMode.MANIFEST:
        job_id = job['job_id']
    elif mode == ExecutionMode.SENSITIVITY:
        job_id = f"sens_{job['dataset']}_{well_id}_win{job['window_size']}_hor{job['forecast_horizon']}"
    else:
        job_id = f"{job['dataset']}_{well_id}"

    notebook_path = output_dir / f"output_{job_id}.ipynb"
    log_path = output_dir / f"log_{job_id}.txt"
    return job_id, notebook_path, log_path


def create_sensitivity_job_queue(
    datasets_filter: List[str],
    window_sizes: List[int],
    forecast_horizons: List[int]
) -> List[Dict[str, Any]]:
    """
    Builds job queue for sensitivity mode via cartesian product of parameters.
    """
    queue: List[Dict[str, Any]] = []
    print("INFO: Creating job queue in 'sensitivity' mode...")
    for dataset in datasets_filter:
        wells: List[str]
        if dataset == "OPSD":
            wells = ['solar', 'wind', 'load']
        else:
            sources = get_data_sources()
            src = next((s for s in sources if s["name"] == dataset), None)
            wells = src["wells"] if src else []
        for window, horizon in product(window_sizes, forecast_horizons):
            for well in wells:
                queue.append({
                    "dataset": dataset,
                    "well": well,
                    "window_size": window,
                    "forecast_horizon": horizon
                })
    return queue


def create_simple_job_queue(
    datasets_filter: List[str]
) -> List[Dict[str, Any]]:
    """
    Builds job queue for simple mode, using provided dataset filters.
    """
    queue: List[Dict[str, Any]] = []
    print("INFO: Creating job queue in 'simple' mode...")
    filters = datasets_filter or []
    if not filters:
        all_sources = get_data_sources() + get_data_sources(opsd_type='solar')
        filters = sorted({s["name"] for s in all_sources})
        print(f"INFO: No filters provided. Running all datasets: {filters}")

    for dataset in filters:
        if dataset == "OPSD":
            for t in ['solar', 'wind', 'load']:
                queue.append({"dataset": dataset, "well": t})
        else:
            sources = get_data_sources()
            src = next((s for s in sources if s["name"] == dataset), None)
            if src:
                for well in src["wells"]:
                    queue.append({"dataset": dataset, "well": well})
    return queue


def create_manifest_job_queue(manifest_path: Path) -> List[Dict[str, Any]]:
    """
    Builds job queue from CSV manifest, selecting only 'pending' jobs.
    """
    print(f"INFO: Creating job queue in 'manifest' mode from: {manifest_path}")
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}")
        return []
    df = pd.read_csv(manifest_path)
    pending = df[df['status'] == 'pending']
    if pending.empty:
        print("INFO: No jobs with status 'pending' found in the manifest.")
    return pending.to_dict('records')


from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import re
import uuid
import nbformat
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import yaml

import plotly.graph_objects as go
import yaml

from forecast_pipeline.config import (
    AggregationMode,
)

# -----------------------------------------------------------------------------
# THEMES
# -----------------------------------------------------------------------------
themes = {
    "minimal": {"text": "#333333", "bg": "#FFFFFF", "accent": "#4CAF50", "grid": "#DDDDDD"},
    "dark":    {"text": "#F0F0F0", "bg": "#2C2C2C", "accent": "#76B947", "grid": "#555555"},
}

# -----------------------------------------------------------------------------
# FEW SHOT UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def is_uuid(text: str) -> bool:
    """Return True if text is a valid UUID string."""
    try:
        uuid.UUID(text)
        return True
    except (ValueError, TypeError):
        return False


def find_notebooks(base_dir: Path, mode: AggregationMode) -> List[Path]:
    """Find output notebooks in base_dir filtered by aggregation mode."""
    if not base_dir.is_dir():
        print(f"WARNING: Directory '{base_dir}' does not exist.")
        return []
    notebooks = [p for p in base_dir.rglob("output_*.ipynb") if "-checkpoint" not in p.name]
    if mode == AggregationMode.SIMPLE:
        return [
            p for p in notebooks
            if not is_uuid(p.stem.split("_")[-1]) and not p.name.startswith("output_sens_")
        ]
    if mode == AggregationMode.MANIFEST:
        return [p for p in notebooks if is_uuid(p.stem.split("_")[-1])]
    if mode == AggregationMode.SENSITIVITY:
        return [p for p in notebooks if p.name.startswith("output_sens_")]
    return []


def find_sensitivity_notebooks(base_dir: Path) -> List[Path]:
    """Find only sensitivity analysis notebooks in base_dir."""
    if not base_dir.is_dir():
        return []
    return sorted(p for p in base_dir.rglob("output_sens_*.ipynb") if "-checkpoint" not in p.name)


def extract_metadata(nb_path: Path, mode: AggregationMode, manifest_df: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    """Extract dataset and well metadata from notebook filename or manifest."""
    stem = nb_path.stem.replace("output_", "")
    if mode == AggregationMode.SIMPLE:
        parts = stem.split("_")
        if len(parts) < 2:
            return None
        return {"Dataset": parts[0], "Well": "_".join(parts[1:])}
    if mode == AggregationMode.MANIFEST and manifest_df is not None:
        row = manifest_df[manifest_df["job_id"] == stem]
        if not row.empty:
            return row.iloc[0].to_dict()
    return None


def extract_sensitivity_metadata(nb_path: Path) -> Optional[Dict[str, Any]]:
    """Extract dataset, well, window size, and horizon from sensitivity notebook filename."""
    stem = nb_path.stem.replace("output_sens_", "")
    pattern = re.compile(r"^(?P<dataset>.+?)_(?P<well>.+?)_win(?P<window>\d+)_hor(?P<horizon>\d+)$")
    match = pattern.match(stem)
    if not match:
        print(f"WARNING: Cannot parse sensitivity metadata from '{nb_path.name}'")
        return None
    data = match.groupdict()
    return {
        "Dataset": data["dataset"],
        "Well": data["well"].replace("_", "/"),
        "Window Size": int(data["window"]),
        "Forecast Horizon": int(data["horizon"]),
    }


def parse_notebook_for_text(nb_path: Path) -> str:
    """Read a notebook and concatenate all text output streams."""
    try:
        with nb_path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception:
        return ""
    texts = []
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                texts.append(output.get("text", ""))
    return "\n".join(texts)


def parse_all_metrics_from_text(text: str) -> Dict[str, Dict[str, float]]:
    """Extract SMAPE and MAE metrics for each method from given text."""
    results: Dict[str, Dict[str, float]] = {}
    metric_pattern = re.compile(r"SMAPE on the test set:\s*([\d\.]+)%\s*MAE on the test set:\s*([\d\.]+)", re.DOTALL)
    header_pattern = re.compile(r"Metrics per Well \((.*?)\):")
    metric_matches = list(metric_pattern.finditer(text))
    header_matches = list(header_pattern.finditer(text))
    for m in metric_matches:
        for h in header_matches:
            if h.start() > m.end():
                method = h.group(1).strip()
                results[method] = {"SMAPE": float(m.group(1)), "MAE": float(m.group(2))}
                break
    return results


def display_styled_dataframe(df: pd.DataFrame, title: str, theme_name: str = "dark") -> None:
    """Style and display DataFrame with title using the selected theme."""
    if df.empty:
        display(HTML(f"<h3>{title}</h3><p>No data to display.</p>"))
        return
    theme = themes.get(theme_name, themes["minimal"])
    formatters = {col: fmt for col, fmt in {
        "SMAPE": "{:.2f}%", "MAE": "{:,.2f}", "Average SMAPE": "{:.3f}%", "Rank": "<b>#{}</b>"
    }.items() if col in df.columns}
    gradient_cols = [c for c in ["SMAPE", "Average SMAPE"] if c in df.columns]
    styled = (
        df.style.format(formatters, na_rep="-")
        .set_properties(**{
            "font-size": "11pt", "text-align": "center",
            "color": theme["text"], "background-color": theme["bg"]
        })
        .background_gradient(cmap="viridis_r", subset=gradient_cols)
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", theme["accent"]), ("color", "white"),
                ("font-size", "12pt"), ("font-weight", "bold"),
                ("text-transform", "uppercase"), ("padding", "8px 12px")
            ]},
            {"selector": "td", "props": [
                ("border", f"1px solid {theme['grid']}"), ("padding", "8px")
            ]},
            {"selector": "tr:hover", "props": [
                ("background-color", theme["accent"] + "40")
            ]},
        ])
        .hide(axis="index")
    )
    display(HTML(f"<h3>{title}</h3>"))
    display(styled)


# -----------------------------------------------------------------------------
# AGGREGATION FUNCTIONS
# -----------------------------------------------------------------------------
def run_sensitivity_analysis(base_dir: Path) -> Optional[pd.DataFrame]:
    """
    Generate a heatmap DataFrame of average SMAPE vs. window size and horizon.
    Returns the pivoted DataFrame or None if no data.
    """
    nb_paths = find_sensitivity_notebooks(base_dir)
    if not nb_paths:
        print("No sensitivity notebooks found.")
        return None
    print(f"Found {len(nb_paths)} sensitivity notebooks.")
    manifest = []
    for p in nb_paths:
        meta = extract_sensitivity_metadata(p)
        if not meta:
            continue
        text = parse_notebook_for_text(p)
        metrics = parse_all_metrics_from_text(text).get("Kalman")
        if metrics:
            entry = {**meta, **metrics}
            manifest.append(entry)
    if not manifest:
        print("No 'Kalman' metrics extracted.")
        return None
    df = pd.DataFrame(manifest)
    pivot = df.groupby(["Window Size", "Forecast Horizon"])["SMAPE"].mean().reset_index()
    heatmap = pivot.pivot(index="Window Size", columns="Forecast Horizon", values="SMAPE")
    heatmap.index.name = "Window Size (Days)"
    heatmap.columns.name = "Forecast Horizon (Days)"
    return heatmap


def run_simple_aggregation(base_dir: Path, dataset_order: List[str], exclude: List[str], theme: str) -> None:
    """
    Aggregate simple run notebooks, display and save metrics per method and well.
    """
    nb_paths = find_notebooks(base_dir, AggregationMode.SIMPLE)
    if not nb_paths:
        print("No simple mode notebooks found.")
        return
    records = []
    for p in nb_paths:
        meta = extract_metadata(p, AggregationMode.SIMPLE)
        if not meta:
            continue
        text = parse_notebook_for_text(p)
        for method, mts in parse_all_metrics_from_text(text).items():
            entry = {**meta, "Method": method, **mts}
            records.append(entry)
    if not records:
        print("No metrics extracted.")
        return
    df = pd.DataFrame(records)
    if exclude:
        print(f"Excluding datasets: {exclude}")
        df = df[~df["Dataset"].isin(exclude)]
    if dataset_order:
        present = [d for d in dataset_order if d in df["Dataset"].unique()]
        df["Dataset"] = pd.Categorical(df["Dataset"], categories=present, ordered=True)
    df = df.sort_values(["Dataset", "Well"])
    for method in ["Kalman", "No Filter"]:
        mdf = df[df["Method"] == method]
        if mdf.empty:
            continue
        means = mdf[["MAE", "SMAPE"]].mean().to_dict()
        title = (
            f"{method} Detailed Results<br>"
            f"<small>Overall Mean — MAE: <b>{means['MAE']:,.2f}</b> | "
            f"SMAPE: <b>{means['SMAPE']:.2f}%</b></small>"
        )
        cols = ["Dataset", "Well", "MAE", "SMAPE"]
        display_styled_dataframe(mdf[cols], title, theme_name=theme)
        csv = f"analysis_results/results_{method.lower().replace(' ', '_')}_simple.csv"
        mdf[cols].to_csv(csv, index=False)
        print(f"Saved detailed results for '{method}' to '{csv}'")


def run_manifest_aggregation(
    base_dir: Path,
    manifest_path: Path,
    config_dir: Path,
    top_n: int,
    exclude: List[str],
    theme: str
) -> Optional[pd.DataFrame]:
    """
    Build and display leaderboard from manifest experiments and save top configurations.
    Returns pivoted leaderboard DataFrame or None.
    """
    try:
        manifest_df = pd.read_csv(manifest_path)
    except FileNotFoundError:
        print(f"Manifest file not found at '{manifest_path}'.")
        return None
    nb_paths = find_notebooks(base_dir, AggregationMode.MANIFEST)
    if not nb_paths:
        print("No manifest mode notebooks found.")
        return None
    records = []
    for p in nb_paths:
        job_id = p.stem.replace("output_", "")
        if not is_uuid(job_id):
            continue
        text = parse_notebook_for_text(p)
        metrics = parse_all_metrics_from_text(text).get("Kalman")
        if metrics:
            records.append({"job_id": job_id, **metrics})
    if not records:
        print("No 'Kalman' metrics extracted.")
        return None
    results_df = pd.DataFrame(records)
    merged = manifest_df.merge(results_df, on="job_id", how="inner")
    if exclude:
        print(f"Excluding datasets: {exclude}")
        merged = merged[~merged["dataset"].isin(exclude)]
    if merged.empty:
        print("No matching results after exclusion.")
        return None
    lb = (
        merged.groupby(["architecture_id", "hyperparam_id"])["SMAPE"]
        .mean().reset_index(name="Average SMAPE")
    )
    lb = lb.sort_values("Average SMAPE").reset_index(drop=True)
    lb["Rank"] = lb.index + 1
    try:
        archs = {c["architecture_id"]: c["summary"] for c in yaml.safe_load((config_dir / "architectures.yaml").read_text())}
        hps = {c["hyperparam_id"]: c["summary"] for c in yaml.safe_load((config_dir / "hyperparameters.yaml").read_text())}
        lb["Architecture"] = lb["architecture_id"].map(archs)
        lb["Hyperparameters"] = lb["hyperparam_id"].map(hps)
    except Exception:
        print("WARNING: YAML config not found or invalid.")
        lb["Architecture"] = "N/A"
        lb["Hyperparameters"] = "N/A"
    cols = ["Rank", "Average SMAPE", "Architecture", "Hyperparameters", "architecture_id", "hyperparam_id"]
    display_styled_dataframe(lb[cols], "Experiment Leaderboard (Average SMAPE)", theme_name=theme)
    csv_path = manifest_path.parent / "experiment_leaderboard.csv"
    lb.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved leaderboard to '{csv_path}'")
    print(f"\nTop {top_n} configuration details:")
    for _, row in lb.head(top_n).iterrows():
        detail = merged[
            (merged["architecture_id"] == row["architecture_id"]) &
            (merged["hyperparam_id"] == row["hyperparam_id"])
        ]
        mean_verified = detail["SMAPE"].mean()
        title = (
            f"Rank #{int(row['Rank'])}: {row['architecture_id']} / {row['hyperparam_id']}<br>"
            f"<small>Leaderboard SMAPE: {row['Average SMAPE']:.3f}% | Verified SMAPE: <b>{mean_verified:.3f}%</b></small>"
        )
        detail = detail.rename(columns={"dataset": "Dataset", "well": "Well"})
        display_styled_dataframe(detail[["Dataset", "Well", "MAE", "SMAPE"]], title, theme_name=theme)

    return lb.pivot(index="hyperparam_id", columns="architecture_id", values="Average SMAPE")

# -----------------------------------------------------------------------------
# SENSITIVITY DRILL-DOWN
# -----------------------------------------------------------------------------
def run_sensitivity_drilldown_analysis(base_dir: Path, exclude_datasets: List[str]) -> None:
    """
    For each (window, horizon) combination, display detailed metrics per well.
    """
    print("INFO: Starting sensitivity drill-down analysis...")
    notebooks = find_notebooks(base_dir, AggregationMode.SENSITIVITY)
    if not notebooks:
        print("No sensitivity notebooks found.")
        return

    records: List[Dict[str, Any]] = []
    for nb in notebooks:
        meta = extract_sensitivity_metadata(nb)
        if not meta:
            continue
        text = parse_notebook_for_text(nb)
        metrics = parse_all_metrics_from_text(text).get("Kalman")
        if metrics:
            records.append({**meta, **metrics})

    if not records:
        print("No 'Kalman' metrics extracted from sensitivity notebooks.")
        return

    df = pd.DataFrame(records)
    if exclude_datasets:
        df = df[~df["Dataset"].isin(exclude_datasets)]

    if df.empty:
        print("No data remaining after excluding datasets.")
        return

    combos = (
        df[["Window Size", "Forecast Horizon"]]
        .drop_duplicates()
        .sort_values(["Window Size", "Forecast Horizon"])
    )
    print(f"Found {len(combos)} unique window/horizon combinations.")
    for _, combo in combos.iterrows():
        w, h = combo["Window Size"], combo["Forecast Horizon"]
        subset = df[(df["Window Size"] == w) & (df["Forecast Horizon"] == h)].copy()
        if subset.empty:
            continue
        means = subset[["MAE", "SMAPE"]].mean().to_dict()
        title = (
            f"Window={w}d, Horizon={h}d<br>"
            f"<small>Mean — MAE: <b>{means['MAE']:.2f}</b> | "
            f"SMAPE: <b>{means['SMAPE']:.2f}%</b></small>"
        )
        subset = subset.sort_values(["Dataset", "Well"])
        display_styled_dataframe(
            subset[["Dataset", "Well", "MAE", "SMAPE"]],
            title,
            theme_name='dark'
        )


# -----------------------------------------------------------------------------
# PERFORMANCE DASHBOARD
# -----------------------------------------------------------------------------
def create_performance_dashboard_from_pivot(pivot_df: pd.DataFrame, title: str):
    """
    Função de plotagem genérica que cria um heatmap estilizado a partir de um DataFrame 
    já pivotado, com controle preciso sobre os eixos.
    """
    z_values = pivot_df.values
    x_labels = pivot_df.columns
    y_labels = pivot_df.index
    
    if np.all(np.isnan(z_values)):
        print(f"AVISO: Nenhum dado para plotar para '{title}'.")
        return
        
    # Encontra o melhor performer (menor SMAPE)
    min_val = np.nanmin(z_values)
    min_pos = np.where(z_values == min_val)
    # Garante que não haja erro se a matriz estiver vazia ou toda NaN
    min_row_idx, min_col_idx = (min_pos[0][0], min_pos[1][0]) if min_pos[0].size > 0 else (0,0)
    
    # Cria o texto para cada célula, destacando o melhor
    text_labels = np.array([f"{val:.2f}%" if not pd.isna(val) else "" for val in z_values.flatten()]).reshape(z_values.shape)
    # text_labels[min_row_idx, min_col_idx] = f"🏆<br><b>{min_val:.2f}%</b>"

    # Define cores de texto para contraste
    color_threshold = np.nanmedian(z_values)
    text_colors = np.where(z_values <= color_threshold, "black", "white")
    text_colors[min_row_idx, min_col_idx] = "#1E5631"
    
    # Cria a figura do heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn_r',
        text=text_labels,
        texttemplate="%{text}",
        textfont=dict(size=24),
        hovertemplate="Window: %{y} days<br>Horizon: %{x} days<br>Mean SMAPE: %{z:.2f}%<extra></extra>",
        colorbar=dict(
            title="<b>Mean SMAPE (%)</b>",
            tickfont=dict(size=18),
            lenmode="fraction", len=0.8, thickness=20,
        )
    ))
    
    # ★★★ LÓGICA DE LAYOUT E EIXOS APRIMORADA ★★★
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, y=0.95, font=dict(size=28)),
        xaxis_title="<b>Forecast Horizon (Days)</b>",
        yaxis_title="<b>Window Size (Days)</b>",
        width=1000, # Aumentado para melhor espaçamento
        height=900,
        paper_bgcolor='white', 
        plot_bgcolor='#f9f9f9'
    )
    
    # Aplica o controle preciso sobre os eixos
    fig.update_xaxes(
        tickmode='array',
        title_font=dict(size=24),
        tickfont=dict(size=22),
        tickvals=x_labels, # Usa os valores exatos das colunas como ticks
        tickangle=-30,
        constrain='domain'
    )
    
    fig.update_yaxes(
        # A ordem do índice já deve estar correta (ex: 7, 14, 21), então não invertemos
        tickmode='array',
        title_font=dict(size=24),
        tickfont=dict(size=22),
        tickvals=y_labels, # Usa os valores exatos do índice como ticks
    )
    
    fig.show()

    fig.write_image(
                "analysis_results/Horizon_Window.jpeg",
                format='jpeg',
                width=1000,
                height=900,
                scale=3
            )


# -----------------------------------------------------------------------------
# LEADERBOARD HEATMAP
# -----------------------------------------------------------------------------
def load_and_prepare_data(leaderboard_path: Path, ARCHITECTURE_ALIASES = Dict[str, str], HYPERPARAM_ALIASES= Dict[str, str]) -> Optional[pd.DataFrame]:
    """
    Load leaderboard CSV and map IDs to readable labels.
    Returns DataFrame or None on error.
    """
    try:
        df = pd.read_csv(leaderboard_path)
    except FileNotFoundError:
        print(f"ERROR: Leaderboard file not found at '{leaderboard_path}'.")
        return None

    required = {"architecture_id", "hyperparam_id", "Average SMAPE"}
    if not required.issubset(df.columns):
        print(f"ERROR: Leaderboard CSV missing columns: {required}.")
        return None

    df["Architecture"] = df["architecture_id"].map(ARCHITECTURE_ALIASES)
    df["HP Profile"] = df["hyperparam_id"].map(HYPERPARAM_ALIASES)
    df.dropna(subset=["Architecture", "HP Profile"], inplace=True)
    return df



def create_annotated_heatmap(df: pd.DataFrame, ARCHITECTURE_ALIASES = Dict[str, str], HYPERPARAM_DESCRIPTIONS= Dict[str, str]) -> None:
    FONT_FAMILY = "Lato, sans-serif"
    PLOT_TITLE = "<b>Averege SMAPE (%): Architectures vs Hyperparameter</b>"
    # ---------------------------------------------------------------------
    # 1. Pivot + ordenação
    # ---------------------------------------------------------------------
    pivot = (
        df.pivot(index="HP Profile", columns="Architecture", values="Average SMAPE")
        .reindex(columns=sorted(ARCHITECTURE_ALIASES.values()))
    )
    
    z = pivot.values
    x_lab, y_lab = pivot.columns, pivot.index

    # ---------------------------------------------------------------------
    # 2. Paleta e faixa explícita (continua igual)
    # ---------------------------------------------------------------------
    color_scale = [
        [0.0, "#2ECC71"],   # verde
        [0.5, "#F4D03F"],   # amarelo
        [1.0, "#E74C3C"]    # vermelho
    ]
    
    zmin=np.nanmin(z) 
    zmax=np.nanmax(z)

    # ---------------------------------------------------------------------
    # 3. Texto nas células — ordem COLUNA-major       ★ correção aqui
    # ---------------------------------------------------------------------
    color_thr = np.nanmedian(z)
    flat_vals   = z.T.flatten()                    # ← gira a matriz
    text_labels = np.char.mod('%.3f%%', flat_vals)
    text_colors = np.where(flat_vals <= color_thr, "black", "white")

    scatter_text = go.Scatter(
        x=np.repeat(x_lab, len(y_lab)),           # A, A, A, … B, B, …
        y=np.tile (y_lab, len(x_lab)),            # hp1,hp2,hp3, … hp1,hp2…
        mode='text',
        text=text_labels,
        textfont=dict(family=FONT_FAMILY, size=22, color=text_colors),
        hoverinfo='skip',
        showlegend=False
    )

    # ---------------------------------------------------------------------
    # 4. Heatmap de fundo (inalterado, só removi código duplicado)
    # ---------------------------------------------------------------------
    heatmap = go.Heatmap(
        z=z, x=x_lab, y=y_lab, colorscale='RdYlGn_r',
        zmin=zmin, zmax=zmax, showscale=False,
        colorbar=dict(title="<b>Mean SMAPE (%)</b>", tickfont=dict(size=18)),
        hovertemplate="<b>Architecture:</b> %{x}<br>"
                      "<b>HP Profile:</b> %{y}<br>"
                      "<b>Mean SMAPE:</b> %{z:.3f}%<extra></extra>"
    )

    # ---------------------------------------------------------------------
    # 5. Troféu na menor célula (mesma lógica, mas agora limpa o texto)
    # ---------------------------------------------------------------------
    best_idx           = np.unravel_index(np.nanargmin(z), z.shape)
    best_row, best_col = best_idx
    best_x , best_y    = x_lab[best_col], y_lab[best_row]
    best_val           = z[best_idx]

    cols, rows  = len(x_lab), len(y_lab)
    fig_w, fig_h = 1600, 780                     #  ↞  mesmo width/height do layout
    hm_dom      = (0, 0.65)                      #  ↞  domain do eixo X
    
    cell_w_px   = (hm_dom[1]-hm_dom[0]) * fig_w / cols
    cell_h_px   =              fig_h            / rows

    # apaga o número que apareceria nessa célula para não sobrepor o troféu
    text_labels[best_col*len(y_lab) + best_row] = ""

    trophy = dict(
        x=best_x, y=best_y, xref="x", yref="y",
        text=f"🏆",
        showarrow=False,
        font=dict(size=20, family="Arial Black", color="#145A32"),
        bgcolor="rgba(255,255,255,0.5)",
        bordercolor="#FF5733", borderwidth=0, borderpad=4,
        # 👉 posicionamento no canto superior-direito da célula
    xanchor="right", yanchor="top",
    xshift= cell_w_px/4,     # 4 px de “acolchoamento” interno
    yshift=-cell_h_px/9,
    )

    # ---------------------------------------------------------------------
    # 6. Layout e guia lateral (como antes)
    # ---------------------------------------------------------------------
    guide_lines = ["<b>Hyperparameter Profile Guide:</b><br>"]
    for label in y_lab:
        guide_lines.append(f"• <b>{label}</b>: {HYPERPARAM_DESCRIPTIONS[label]}<br>")
    guide = dict(
        showarrow=False, text="".join(guide_lines), align="left",
        xref="paper", yref="paper", x=1.37, y=0.96,
        font=dict(size=18, family=FONT_FAMILY),
        bordercolor="#5A6B7F", borderwidth=2, borderpad=6,
        bgcolor="#f8f9f9", opacity=0.9,
    )

    fig = go.Figure(data=[heatmap, scatter_text])
    fig.update_layout(
        title=dict(text=PLOT_TITLE, x=0.1, y=0.95,
                   font=dict(size=28, family=FONT_FAMILY)),
        xaxis=dict(title="<b>Model Architecture</b>", tickangle=-30,
                   title_font=dict(size=24),
                    tickfont=dict(size=22), showgrid=False, domain=[0, 0.65]),
        yaxis=dict(title="<b>Hyperparameter Profile</b>",
                   title_font=dict(size=24),
                    tickfont=dict(size=22), showgrid=False,
                   autorange="reversed"),
        annotations=[trophy],
        width=1600, height=780, paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=180, r=450, b=120, t=100),
        font=dict(family=FONT_FAMILY, size=24),
    )

    OUTPUT_IMAGE_PATH = "analysis_results/Manifest_Heatmap.jpeg"
    fig.write_image(OUTPUT_IMAGE_PATH, scale=3, format='jpeg',
                width=1600,
                height=780)
    fig.show()
    print(f"✅ Plot saved in: {OUTPUT_IMAGE_PATH}")

import numpy as np

def _inverse_transform_1d(scaler, arr_1d: np.ndarray) -> np.ndarray:
    if arr_1d.ndim != 1:
        raise ValueError(f"_inverse_transform_1d expects 1D array, got shape={arr_1d.shape}")
    return scaler.inverse_transform(arr_1d.reshape(-1, 1)).ravel()

def _inverse_transform_2d(scaler, arr_2d: np.ndarray) -> np.ndarray:
    if arr_2d.ndim != 2:
        raise ValueError(f"_inverse_transform_2d expects 2D array, got shape={arr_2d.shape}")
    flat = arr_2d.reshape(-1, 1)
    inv  = scaler.inverse_transform(flat).reshape(arr_2d.shape[0], arr_2d.shape[1])
    return inv

def _looks_scaled(arr: np.ndarray) -> bool:
    """
    Heuristically checks if an array appears to be scaled by examining its
    magnitude and standard deviation.

    This version is more robust by focusing on two primary conditions:
    1. The absolute maximum value is within a typical range for scaled data (e.g., under 10).
    2. The standard deviation is neither negligibly small (like a constant) nor
       excessively large (like unscaled raw data).

    This avoids overly strict assumptions about the data's mean or its exact range.
    """
    if arr is None or arr.size < 2:
        return False
        
    a = arr[np.isfinite(arr)]
    if a.size < 2:
        return False

    # --- Calculate key statistics ---
    std_val = np.nanstd(a)
    abs_max = np.nanmax(np.abs(a))

    # --- Core Logic ---
    # 1. Is the standard deviation meaningful (i.e., not a near-constant array)?
    has_meaningful_std = std_val > 0.01
    
    # 2. Are the values contained within a "small" numeric range?
    #    This is the most reliable check to distinguish from raw, unscaled data
    #    which often has values in the hundreds, thousands, or more.
    has_small_magnitude = abs_max < 10.0 # Um pouco mais flexível que 6.0

    return has_meaningful_std and has_small_magnitude




def _detect_family_for_df(df: pd.DataFrame) -> str:
    """Heurística simples e robusta: 'seq2' | 'arps' | 'darts'."""
    txt = (
        df.get("architecture", pd.Series([], dtype=str)).fillna("").astype(str) + " " +
        df.get("architecture_name", pd.Series([], dtype=str)).fillna("").astype(str)
    ).str.lower()
    joined = " ".join(txt.tolist())

    # Sinais fortes por coluna
    if "variant" in df.columns:
        return "arps"
    if "profile" in df.columns:
        return "darts"

    # Sinais por nome
    if "arps" in joined:
        return "arps"
    if "darts" in joined:
        return "darts"

    # Seq2 é o fallback mais comum quando há physics_strategy/aggregation_method
    if "physics_strategy" in df.columns or "aggregation_method" in df.columns:
        return "seq2"

    return "seq2"  # fallback seguro

# In src/utils/utilities.py (or wherever the function is located)

def _detect_family_for_df(df: pd.DataFrame) -> str:
    """Robust heuristic to detect family: 'seq2' | 'arps' | 'darts'."""
    # Combine relevant text columns into a single series, handling missing data
    text_cols = []
    if "architecture" in df.columns:
        text_cols.append(df["architecture"].fillna("").astype(str))
    if "architecture_name" in df.columns:
        text_cols.append(df["architecture_name"].fillna("").astype(str))

    if not text_cols:
        # If no architecture columns exist, rely on other signals
        joined = ""
    else:
        # Concatenate all text columns into one, then join all rows into a single string
        full_text_series = pd.concat(text_cols, axis=1).apply(lambda row: ' '.join(row), axis=1)
        joined = " ".join(full_text_series.str.lower().tolist())

    # --- Strong signals first (based on column presence) ---
    if "variant" in df.columns:
        return "arps"
    # Check for 'profile' and if it has any non-null values
    if "profile" in df.columns and df["profile"].notna().any():
        return "darts"

    # --- Weaker signals (based on text content) ---
    if "arps" in joined:
        return "arps"
    if "darts" in joined:
        return "darts"

    # Seq2 is a common fallback if physics_strategy or aggregation_method exist
    if "physics_strategy" in df.columns or "aggregation_method" in df.columns:
        return "seq2"
    
    # Final fallback if architecture names contain "Seq2"
    if "seq2" in joined:
        return "seq2"

    return "generic" # A safer fallback than "seq2"


def _normalize_for_champions(df: pd.DataFrame) -> pd.DataFrame:
    """Harmoniza campos mínimos sem efeitos colaterais (epochs ← n_epochs)."""
    d = df.copy()
    if "epochs" not in d.columns and "n_epochs" in d.columns:
        d["epochs"] = d["n_epochs"]
    return d


def resolve_champions_columns_minimal(df: pd.DataFrame, metric: str = "val_smape_agg") -> list[str]:
    """
    Escolhe um subconjunto curto e informativo por família.
    Só retorna colunas que EXISTEM no df. Mantém 'well' e métrica no fim.
    """
    fam = _detect_family_for_df(df)

    by_family = {
        "seq2": [
            "well",
            "physics_strategy", metric, "aggregation_method",
            "epochs", "batch_size", "learning_rate", "data_sample", "lag_window",
            
        ],
        "arps": [
            "well",
            "variant", metric, "solver", "weighting", "loss", "piecewise",
            "loss_delta", "quantile_tau", "burn_in_fraction",
            # diagnósticos úteis se existirem:
            "piecewise_min_delta_bic", "b_min", "b_max",

        ],
        "darts": [
            "well",
            "profile", metric,               # principal “modelo” dentro de Darts
            "epochs", "batch_size", "learning_rate",
            "input_chunk_length", "output_chunk_length", "lag_window",
            
        ],
    }

    wanted = by_family.get(fam, by_family["seq2"])
    present = [c for c in wanted if c in df.columns]

    # garantir presença de chaves úteis
    if "well" in df.columns and "well" not in present:
        present.insert(0, "well")
    if metric in df.columns and metric not in present:
        present.append(metric)

    return present


# Minimal fallbacks to keep the notebook self-contained.
def check_store_health(artifacts, series_store_root=None, max_show=10):
    series_df = artifacts.get("series_df", pd.DataFrame())
    bounds    = artifacts.get("boundaries_df", pd.DataFrame())
    hist_map  = artifacts.get("full_history_by_well", {}) or {}
    print("=== Series health ===")
    print(f"rows={len(series_df)}  "
          f"wells={series_df['well'].nunique() if 'well' in series_df else 0}  "
          f"jobs={series_df['job_hash'].nunique() if 'job_hash' in series_df else 0}")
    print(f"has columns: {sorted(set(series_df.columns) & {'t','ytrue','split','idx','arch'})}")

    print("\n=== Boundaries manifest ===")
    if isinstance(bounds, pd.DataFrame) and not bounds.empty:
        print(bounds.head(min(max_show, len(bounds))))
    else:
        print("NOT FOUND (plots will use fallback unless you persist /meta/boundaries.parquet)")

    print("\n=== Full history presence (from reader extras) ===")
    wells = sorted(set(series_df["well"].dropna().astype(str))) if "well" in series_df.columns else []
    ok = [w for w in wells if w in hist_map and not hist_map[w].empty]
    missing = [w for w in wells if w not in hist_map or hist_map[w].empty]
    print(f"ok={len(ok)}  missing={len(missing)}")
    print("sample ok:", ok[:max_show])
    print("sample missing:", missing[:max_show])

    if series_store_root:
        root = Path(series_store_root).resolve()
        print("\n=== Files on disk (meta/history) ===")
        meta = root / "meta" / "boundaries.parquet"
        print("boundaries.parquet:", "EXISTS" if meta.exists() else "missing", meta)
        for w in wells[:max_show]:
            hp = root / "history" / f"well={w}" / "history.parquet"
            print(f"history[{w}]:", "EXISTS" if hp.exists() else "missing", hp)

def quick_risk_panel(
    *,
    artifacts: Dict[str, pd.DataFrame],
    well: str,
    arch_list: tuple[str, ...] = ("seq2", "arps"),
    selector: str = "val+test",
    horizons: tuple[int, ...] = (300, 600, 900, -1),
    weighting: str = "uniform",
    temp: float = 0.5,
    palette: str = "default",
    show: bool = True,
):
    """
    Lightweight panel to render Risk CDFs for one well across arches/horizons.
    Uses plot_risk_cdf_for under the hood (no I/O).
    """
    try:
        from plotting.risk_plots import plot_risk_cdf_for
    except Exception as e:
        print(f"[quick_risk_panel] risk plotting not available: {e}")
        return

    series_df = artifacts.get("series_df", pd.DataFrame())
    final_ensemble_df = artifacts.get("final_ensemble_df", pd.DataFrame())
    boundaries_df = artifacts.get("boundaries_df", pd.DataFrame())
    full_history_by_well = artifacts.get("full_history_by_well", {}) or {}

    for arch in arch_list:
        for H in horizons:
            _ = plot_risk_cdf_for(
                series_df=series_df,
                final_ensemble_df=final_ensemble_df,
                boundaries_df=boundaries_df,
                full_history_by_well=full_history_by_well,
                well=well,
                arch=arch,
                selector=selector,
                horizon_days=H,
                weighting=weighting,
                temp=temp,
                palette=palette,
                title_prefix="Risk Curve",
                show=show,
            )
