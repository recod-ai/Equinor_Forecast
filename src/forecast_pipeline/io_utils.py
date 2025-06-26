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