# hpo/optuna_utils.py
import logging
import os
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Union

import optuna
import pandas as pd
import optuna.visualization as vis
from IPython.display import display
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler  # v4: TPESampler also handles MO

from forecast_pipeline.config import HPO_STUDIES_DIR
from profile_manager import ExperimentSchema  # if you actually use it


def _make_sampler(_: Optional[List[str]] = None) -> TPESampler:
    return TPESampler(
        n_startup_trials=15,      # (antes: 30) aprende mais cedo
        multivariate=True,
        group=True,
        seed=42,
        constant_liar=True,       # ✅ MUITO importante no seu fluxo batch ask()
    )


import numpy as np
import optuna

def _select_diverse_pareto(pareto_trials, n, directions):
    # values: shape (m, k)
    V = np.array([t.values for t in pareto_trials], dtype=float)

    # Converte MAXIMIZE para MINIMIZE (para ficar consistente)
    # MINIMIZE -> +1, MAXIMIZE -> -1
    sign = np.array([1.0 if d == optuna.study.StudyDirection.MINIMIZE else -1.0 for d in directions])
    V = V * sign

    # Normaliza [0,1] por objetivo (evita escala dominar)
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    denom = np.maximum(vmax - vmin, 1e-12)
    X = (V - vmin) / denom

    selected = []

    # 1) extremos: melhor em cada objetivo
    for j in range(X.shape[1]):
        idx = int(np.argmin(X[:, j]))
        selected.append(idx)

    selected = list(dict.fromkeys(selected))  # unique, preserva ordem

    # 2) completa por diversidade (maximin distance)
    if len(selected) >= n:
        return [pareto_trials[i] for i in selected[:n]]

    remaining = [i for i in range(len(pareto_trials)) if i not in selected]

    while remaining and len(selected) < n:
        S = X[selected]  # (s, k)
        R = X[remaining] # (r, k)
        # dist de cada candidato ao conjunto selecionado = min dist
        dists = np.sqrt(((R[:, None, :] - S[None, :, :]) ** 2).sum(axis=2))
        score = dists.min(axis=1)  # maximin
        pick_pos = int(np.argmax(score))
        pick_idx = remaining[pick_pos]
        selected.append(pick_idx)
        remaining.pop(pick_pos)

    return [pareto_trials[i] for i in selected]





def _generate_trials_base(
    study_name: str,
    storage_url: str,
    n_trials: int,
    output_profile_path: Path,
    search_space_func: Callable[[optuna.trial.Trial], Dict[str, Any]],
    directions: Optional[List[str]] = None,
    **fixed_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    The core logic for generating HPO trials.
    Supports both single-objective and multi-objective studies.
    """
    sampler = _make_sampler(directions)
    if directions:
        study_params = {
            "directions": directions,
            "study_name": study_name,
            "storage": storage_url,
            "load_if_exists": True,
            "pruner": HyperbandPruner(),
            "sampler": sampler,
        }
        print(f"Creating/loading MULTI-OBJECTIVE study with {len(directions)} objectives.")
    else:
        study_params = {
            "direction": "minimize",
            "study_name": study_name,
            "storage": storage_url,
            "load_if_exists": True,
            "pruner": HyperbandPruner(),
            "sampler": sampler,
        }
        print("Creating/loading SINGLE-OBJECTIVE study.")

    study = optuna.create_study(**study_params)
    
    logging.info(f"Study '{study_name}' at '{storage_url}' has {len(study.trials)} existing trials.")
    
    records = []
    for _ in range(n_trials):
        trial = study.ask()
        params_core = search_space_func(trial)
        params = {**(fixed_params or {}), **params_core}
        params["optuna_trial_number"] = trial.number
        params["experiment_id"] = f"{study_name}_trial_{trial.number}"
        records.append(params)
        
    df = pd.DataFrame(records)
    
    output_profile_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_profile_path, index=False)
    
    logging.info(f"Successfully generated {n_trials} new trials and saved to '{output_profile_path}'")
    return df

def generate_trials_from_study(
    study_name: str,
    n_trials: int,
    output_file: str | Path,
    search_space_func: Callable[[optuna.trial.Trial], Dict[str, Any]],
    storage_url: Optional[str] = None,
    directions: Optional[List[str]] = None,
    **fixed_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generates HPO trials, acting as a backward-compatible adapter.
    It resolves paths and passes all relevant parameters to the core `_generate_trials_base`.
    """
    # --- Compatibility Logic for paths ---
    if storage_url is None:
        logging.debug("Running generate_trials_from_study in LEGACY path mode.")
        storage_url = f"sqlite:///{HPO_STUDIES_DIR / study_name}.db"
        final_output_path = Path(output_file).resolve()
    else:
        logging.debug("Running generate_trials_from_study in NEW config-driven path mode.")
        final_output_path = Path(output_file)

    # --- Call the core function, passing ALL parameters through ---
    return _generate_trials_base(
        study_name=study_name,
        storage_url=storage_url,
        n_trials=n_trials,
        output_profile_path=final_output_path,
        search_space_func=search_space_func,
        directions=directions,  # Pass the new parameter down
        **fixed_params
    )





# def generate_profile_from_top_trials(
#     study_name: str,
#     storage_url: str,
#     n_top_trials: int,
#     output_profile_path: str | Path,
#     fixed_params: Optional[Dict[str, Any]] = None,
#     add_rank_to_experiment_id: bool = True
# ) -> Optional[pd.DataFrame]:
#     """
#     Loads a completed Optuna study, selects the top N unique best-performing
#     configurations, and saves them to a clean CSV profile.
#     This version is compatible with older versions of Optuna.
#     """
#     try:
#         study = optuna.load_study(study_name=study_name, storage=storage_url)
#     except (KeyError, ValueError):
#         logging.error(f"Study '{study_name}' not found at '{storage_url}'.")
#         return None

#     # --- BACKWARD-COMPATIBLE METHOD to get completed trials ---
#     all_trials_df = study.trials_dataframe()
#     if all_trials_df.empty:
#         logging.warning(f"Study '{study_name}' has no trials at all.")
#         return None

#     # Check if the 'state' column exists before filtering
#     if 'state' not in all_trials_df.columns:
#         logging.error("Could not find 'state' column in study.trials_dataframe(). Cannot filter for completed trials.")
#         return None

#     completed_trials_df = all_trials_df[all_trials_df['state'] == 'COMPLETE'].copy()
#     # --- END OF COMPATIBILITY FIX ---

#     if completed_trials_df.empty:
#         logging.warning(f"Study '{study_name}' has no completed trials to analyze.")
#         return None
        
#     param_columns = [col for col in completed_trials_df.columns if col.startswith('params_')]
#     unique_trials_df = completed_trials_df.drop_duplicates(subset=param_columns, keep='first')
    
#     # Use sort_values, which is more robust than nsmallest if the direction might change
#     is_minimizing = study.direction == optuna.study.StudyDirection.MINIMIZE
#     top_n_df = unique_trials_df.sort_values(by='value', ascending=is_minimizing).head(n_top_trials)
    
#     logging.info(f"Selected {len(top_n_df)} unique best trials for validation profile.")

#     records = []
#     for rank, (_, trial_row) in enumerate(top_n_df.iterrows(), 1):
#         params = {col.replace('params_', ''): trial_row[col] for col in param_columns}
#         params.update(fixed_params or {})

#         params["optuna_trial_number"] = trial_row['number']
        
#         if add_rank_to_experiment_id:
#             params["experiment_id"] = f"validation_{study_name}_rank_{rank:02d}"
            
#         records.append(params)
        
#     final_df = pd.DataFrame(records)
#     output_path = Path(output_profile_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     final_df.to_csv(output_path, index=False)
    
#     logging.info(f"✅ Successfully generated validation profile at '{output_path}'")
#     return final_df


def generate_profile_from_top_trials(
    study_name: str,
    storage_url: str,
    n_top_trials: int,
    output_profile_path: str | Path,
    fixed_params: Optional[Dict[str, Any]] = None,
    add_rank_to_experiment_id: bool = True
) -> Optional[pd.DataFrame]:
    """
    Export the best configurations to CSV.
    - Single-objective (SO): sort by 'value' obeying study.direction (as before).
    - Multi-objective (MO): take the Pareto front via study.best_trials and cut top-N
      using a simple lexicographic order on the objectives (no aggregation).
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except (KeyError, ValueError):
        logging.error(f"Study '{study_name}' not found at '{storage_url}'.")
        return None

    is_multi = hasattr(study, "directions") and len(study.directions) > 1

    if is_multi:
        # --- MO path: use the Pareto front ---
        pareto = [t for t in study.best_trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not pareto:
            logging.warning(f"Study '{study_name}' has no complete Pareto-optimal trials.")
            return None

        selected = _select_diverse_pareto(pareto, n_top_trials, study.directions)

        records = []
        for rank, t in enumerate(selected, 1):
            params = dict(t.params)
            params.update(fixed_params or {})
            params["optuna_trial_number"] = t.number
            if add_rank_to_experiment_id:
                params["experiment_id"] = f"validation_{study_name}_rank_{rank:02d}"
            records.append(params)

        final_df = pd.DataFrame(records)
        output_path = Path(output_profile_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        logging.info(f"✅ Saved {len(final_df)} Pareto trials to '{output_path}'")
        return final_df

    # --- SO path: keep your original behavior ---
    all_trials_df = study.trials_dataframe()
    if all_trials_df.empty:
        logging.warning(f"Study '{study_name}' has no trials at all.")
        return None

    if 'state' not in all_trials_df.columns:
        logging.error("Missing 'state' column in trials DataFrame. Cannot filter completed trials.")
        return None

    completed_trials_df = all_trials_df[all_trials_df['state'] == 'COMPLETE'].copy()
    if completed_trials_df.empty:
        logging.warning(f"Study '{study_name}' has no completed trials to analyze.")
        return None

    param_columns = [col for col in completed_trials_df.columns if col.startswith('params_')]
    unique_trials_df = completed_trials_df.drop_duplicates(subset=param_columns, keep='first')

    is_minimizing = study.direction == optuna.study.StudyDirection.MINIMIZE
    top_n_df = unique_trials_df.sort_values(by='value', ascending=is_minimizing).head(n_top_trials)

    records = []
    for rank, (_, trial_row) in enumerate(top_n_df.iterrows(), 1):
        params = {col.replace('params_', ''): trial_row[col] for col in param_columns}
        params.update(fixed_params or {})
        params["optuna_trial_number"] = trial_row['number']
        if add_rank_to_experiment_id:
            params["experiment_id"] = f"validation_{study_name}_rank_{rank:02d}"
        records.append(params)

    final_df = pd.DataFrame(records)
    output_path = Path(output_profile_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logging.info(f"✅ Successfully generated validation profile at '{output_path}'")
    return final_df



def _report_results_to_study_base(
    study_name: str,
    storage_url: str,
    leaderboard_df: pd.DataFrame,
    objective_keys: Union[str, List[str]],
) -> int:
    """
    The core logic for reporting results back to an Optuna study.
    Supports both single and multi-objective reporting.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except (KeyError, ValueError):
        logging.error(f"Error: Study '{study_name}' not found at '{storage_url}'.")
        return 0

    # --- START OF MODIFICATION ---
    is_multi_objective = isinstance(objective_keys, list)

    if is_multi_objective:
        # Check if all objective columns exist in the dataframe
        missing_keys = [key for key in objective_keys if key not in leaderboard_df.columns]
        if missing_keys:
            logging.error(f"Error: Objective metrics {missing_keys} not in leaderboard columns.")
            return 0
        logging.info(f"Reporting {len(leaderboard_df)} results to MULTI-OBJECTIVE study '{study_name}'...")
    else:
        # Single-objective mode check
        if objective_keys not in leaderboard_df.columns:
            logging.error(f"Error: Metric '{objective_keys}' not in leaderboard columns.")
            return 0
        logging.info(f"Reporting {len(leaderboard_df)} results to SINGLE-OBJECTIVE study '{study_name}'...")

    reported_count = 0
    for _, row in leaderboard_df.iterrows():
        try:
            trial_number = int(row["optuna_trial_number"])
            
            # Check if trial exists and is in a reportable state
            if trial_number >= len(study.trials):
                continue
            trial = study.trials[trial_number]
            if trial.state not in [optuna.trial.TrialState.WAITING, optuna.trial.TrialState.RUNNING]:
                continue

            # Extract either a single value or a list of values
            if is_multi_objective:
                values = [row[key] for key in objective_keys]
                study.tell(trial_number, values)
            else:
                value = row[objective_keys]
                study.tell(trial_number, value)
            
            reported_count += 1
            
        except Exception as e:
            logging.warning(f"Could not report result for trial {row.get('optuna_trial_number', 'N/A')}: {e}")
    # --- END OF MODIFICATION ---
            
    logging.info(f"Finished. Reported {reported_count} new results to study '{study_name}'.")
    return reported_count

def generate_profile_from_dataframe(
    candidates_df: pd.DataFrame,
    output_profile_path: str | Path,
    fixed_params: Optional[Dict[str, Any]] = None
) -> Optional[pd.DataFrame]:
    """
    STRICT REPLAY (pass-through):
    Gera um CSV de perfil preservando TODAS as colunas presentes em candidates_df,
    apenas garantindo as mínimas para o runner e aplicando overrides fixos (ex.: seed).
    """
    if candidates_df is None or candidates_df.empty:
        logging.error("Input candidates_df is empty. Cannot generate profile.")
        return None

    df = candidates_df.copy()

    # 1) Garantias mínimas
    required_min = {"dataset", "well"}
    missing = [c for c in required_min if c not in df.columns]
    if missing:
        logging.error("Cannot generate profile. Missing required columns: %s", missing)
        return None

    # 2) Arquitetura: aceitar 'architecture_name' ou 'architecture'
    if "architecture_name" not in df.columns:
        if "architecture" in df.columns:
            # espelho estável: gerar 'architecture_name' sem perder 'architecture'
            df["architecture_name"] = df["architecture"]
        else:
            logging.error("Profile requires 'architecture_name' or 'architecture' column.")
            return None

    # 3) Normalizações leves e seguras (não sobrescrevem se já existirem)
    #    - manter compatibilidade de nomes para Darts/Seq2 sem alterar valores
    if "input_chunk_length" not in df.columns and "lag_window" in df.columns:
        df["input_chunk_length"] = df["lag_window"]
    if "output_chunk_length" not in df.columns and "horizon" in df.columns:
        df["output_chunk_length"] = df["horizon"]
    if "epochs" not in df.columns and "n_epochs" in df.columns:
        df["epochs"] = df["n_epochs"]

    # 4) Overrides fixos (ex.: seed novo) — aplicados como colunas/replace
    if fixed_params:
        for k, v in fixed_params.items():
            df[k] = v

    # 5) Persistir exatamente como está (pass-through)
    output_path = Path(output_profile_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logging.info(
        "✅ Strict replay profile saved with %d rows and %d columns at '%s'",
        len(df), df.shape[1], output_path
    )
    return df

def get_study(study_name: str, hpo_studies_dir: str | Path) -> Optional[optuna.Study]:
    """
    Safely loads an Optuna study from a given directory.

    Args:
        study_name (str): The name of the campaign/study.
        hpo_studies_dir (str | Path): Path to the directory containing .db files.

    Returns:
        Optional[optuna.Study]: The loaded study object, or None if not found.
    """
    storage_path = f"sqlite:///{Path(hpo_studies_dir) / study_name}.db"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_path)
        print(f"✅ Study '{study_name}' loaded successfully from '{storage_path}'.")
        return study
    except KeyError:
        print(f"❌ ERROR: Study '{study_name}' not found at '{storage_path}'.")
        return None

def report_results_to_study(
    study_name: str,
    leaderboard_df: pd.DataFrame, 
    *, # Makes all subsequent arguments keyword-only
    storage_url: Optional[str] = None,
    metric_to_optimize: Optional[str] = None,          # For single-objective (legacy)
    objective_keys: Optional[Union[str, List[str]]] = None # For both modes (new)
) -> int:
    """
    Reads a leaderboard and reports results to an Optuna study.
    This adapter is fully backward-compatible and handles both single and multi-objective modes.
    """
    if storage_url is None:
        logging.debug("Running report_results_to_study in LEGACY path mode.")
        final_storage_url = f"sqlite:///{HPO_STUDIES_DIR / study_name}.db"
    else:
        logging.debug("Running report_results_to_study in NEW config-driven path mode.")
        final_storage_url = storage_url

    # --- START OF MODIFICATION ---
    # Prioritize the new, more explicit 'objective_keys' parameter if provided.
    # Otherwise, fall back to the legacy 'metric_to_optimize' for backward compatibility.
    final_objective_keys = objective_keys if objective_keys is not None else metric_to_optimize
    if final_objective_keys is None:
        raise ValueError("Must provide either 'objective_keys' or 'metric_to_optimize'.")
    # --- END OF MODIFICATION ---

    return _report_results_to_study_base(
        study_name=study_name,
        storage_url=final_storage_url,
        leaderboard_df=leaderboard_df,
        objective_keys=final_objective_keys
    )

# ==============================================================================
# 1. CORE UTILITY FUNCTIONS (Extracted Logic)
# ==============================================================================

def get_hyperparameter_columns(study: optuna.Study) -> List[str]:
    """
    Extracts the names of the hyperparameters from a study's completed trials.
    """
    # Find the first completed trial to inspect its params
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.params:
            return list(trial.params.keys())
    print("⚠️ WARNING: No completed trials with parameters found in the study.")
    return []

def get_top_n_configs(df: pd.DataFrame, metric: str, hyper_cols: List[str], n: int = 10) -> pd.DataFrame:
    """
    Returns the top-N unique best hyperparameter configurations from a leaderboard.
    """
    if metric not in df.columns:
        print(f"❌ ERROR: Metric '{metric}' not found in DataFrame columns.")
        return pd.DataFrame()
    
    # Sort by the optimization metric (assuming lower is better for now)
    # A more robust version could take a direction argument.
    sorted_df = df.sort_values(by=metric, ascending=True)
    
    # Drop duplicates based on hyperparameter columns to find unique configs
    unique_df = sorted_df.drop_duplicates(subset=hyper_cols, keep='first')
    
    return unique_df.head(n)

# no topo do arquivo (se ainda não estiverem):
import os

def visualize_study(study: optuna.Study, metric_to_optimize: str):
    """
    Generates and displays the standard Optuna visualizations for a study.
    """
    if not study.trials:
        print("Study has no trials to visualize.")
        return
        
    print("\n--- 📊 HPO Visualizations 📊 ---")
    
    # Check for completed trials before plotting
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) < 2:
        print("Not enough completed trials for meaningful visualizations.")
        return

    # 1. Optimization History
    print("\n--- Optimization History ---")
    display(vis.plot_optimization_history(study, target_name=metric_to_optimize))
    
    # 2. Hyperparameter Importances
    try:
        print("\n--- Hyperparameter Importances ---")
        display(vis.plot_param_importances(study))
    except Exception as e:
        print(f"Could not generate parameter importances plot: {e}")


# ==============================================================================
# 2. HIGH-LEVEL WORKFLOW FUNCTION
# ==============================================================================

def analyze_single_campaign(
    campaign_name: str,
    config_path: str | Path # Path to a campaign YAML to get infra paths
):
    """
    Performs a full, standardized analysis of a single completed HPO campaign.
    """
    from config_loader import load_campaign_config
    
    print(f"\n{'='*20} 🔬 Analyzing Campaign: {campaign_name} {'='*20}")
    
    # 1. Load configuration to get paths and parameters
    try:
        config = load_campaign_config(config_path)
        hpo_studies_dir = config.infra.hpo_studies_dir
        results_dir = config.infra.experiments_output_dir
        metric_to_optimize = config.hpo_params.metric_to_optimize
    except Exception as e:
        print(f"❌ ERROR: Could not load or parse config file '{config_path}': {e}")
        return

    # 2. Load the Optuna study and the aggregated results
    study = get_study(campaign_name, hpo_studies_dir)
    master_leaderboard = load_master_leaderboard(campaign_name, results_dir)

    if study is None or master_leaderboard is None or master_leaderboard.empty:
        print("--- Analysis aborted due to missing data. ---")
        return

    # 3. Identify hyperparameters and show the top unique configurations
    hyper_cols = get_hyperparameter_columns(study)
    if hyper_cols:
        print(f"\n--- 🏆 Top 10 Unique Configurations (by '{metric_to_optimize}') ---")
        top_configs = get_top_n_configs(master_leaderboard, metric_to_optimize, hyper_cols, n=10)
        
        # Define columns to display for clarity
        display_cols = hyper_cols + [metric_to_optimize, 'val_smape_cum', 'val_smape_agg']
        available_cols = [c for c in display_cols if c in top_configs.columns]
        display(top_configs[available_cols])
    else:
        print("Could not determine hyperparameters, showing raw top 10.")
        display(master_leaderboard.nsmallest(10, metric_to_optimize))
        
    # 4. Generate and display visualizations
    visualize_study(study, metric_to_optimize)
    
    print(f"\n{'='*20} ✅ Analysis Complete: {campaign_name} {'='*20}")
