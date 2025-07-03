import os
import shutil

from pathlib import Path

import glob
import logging


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


