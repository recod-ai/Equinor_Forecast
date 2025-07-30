import os
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from pathlib import Path 

def generate_experiment_name(datasets, arch_name, ensemble_size):
    dataset_part = "-".join(datasets)
    return f"{dataset_part}_{arch_name}_ens{ensemble_size}"


def truncate_sheet_name(variant: str, metric: str, counter: int) -> str:
    """Ensure Excel sheet names are unique and <= 31 chars."""
    base = f"{variant}__{metric}".replace(" ", "_")
    truncated = base[:27]  # Leave space for counter suffix
    return f"{truncated}_{counter}"


def save_experiment_to_excel(configs: Dict[str, Any],
                              results: Dict[str, Dict[str, pd.DataFrame]],
                              experiment_name: str,
                             datasets_to_run: [str],
                             num_ensemble_models: int, 
                              base_dir: str = "experiments") -> str:
    """
    Save experiment configuration and organized_results in a structured Excel format
    with clear and meaningful sheet names (e.g., global_metrics, aggregated_metrics).
    """

    # Carimba a data
    date_stamp = pd.Timestamp.now().strftime('%Y%m%d')

    # 1) Encontra o project root (duas pastas acima deste arquivo)
    project_root = Path(__file__).resolve().parents[2]

    # 2) Monta o diretório final: PROJECT_ROOT/experiments/{experiment_name}_{date_stamp}
    exp_dir = project_root / base_dir / f"{experiment_name}_{date_stamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save YAML config
    full_config = {
        "DEFAULT_EXP_PARAMS": configs,
        "datasets_to_run": datasets_to_run,
        "num_ensemble_models": num_ensemble_models
    }
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(full_config, f)

    # Save all DataFrames in a structured Excel file
    excel_path = os.path.join(exp_dir, "results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for variant, metrics_dict in results.items():
            for metric_name, df in metrics_dict.items():
                # Use the metric name directly as sheet name
                sheet_name = metric_name[:31]  # Ensure it's within Excel's limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    return exp_dir

# In src/forecast_pipeline/io_utils.py

import json
import os
import tempfile
import datetime
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any

def atomic_write_json(data: Dict[str, Any], dest: str | os.PathLike):
    """
    Atomically writes a dictionary to a JSON file.

    It first writes to a temporary file in the same directory, then
    renames it. This is a safe, atomic operation on most systems and
    prevents file corruption if the process is interrupted.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to a temporary file in the same directory.
    # The 'delete=False' is crucial on Windows.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dest_path.parent, delete=False) as tmp_file:
        json.dump(data, tmp_file, indent=2, default=str)
        tmp_path = tmp_file.name # Get the temp file path
    
    # The atomic operation: rename the completed temp file to the final destination.
    # On POSIX, this is an atomic move. On Windows, it's a safe replace.
    os.replace(tmp_path, dest_path)


def build_run_metadata() -> Dict[str, Any]:
    """
    Gathers key environment and execution metadata for reproducibility.
    """
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
        
    return {
        "run_timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "git_commit_hash": git_hash,
    }

import logging
from colorlog import ColoredFormatter
from forecast_pipeline.config import LOG_LEVEL
def configure_logging():
    """Set up a colored logger respecting the global LOG_LEVEL."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)-7s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'blue',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            }
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    # Map 0→WARNING, 1→INFO, 2→DEBUG
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    root.setLevel(level.get(LOG_LEVEL, logging.INFO))