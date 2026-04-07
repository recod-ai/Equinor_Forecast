# In src/hpo/validation.py

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict
from box import Box
from IPython.display import display

# Import from our existing toolkit
from .optuna_utils import generate_profile_from_top_trials
from .hpo_runner import run_campaign_from_config_object
from .optuna_utils import get_study
from .posthoc_filtering import load_master_leaderboard
from config_loader import load_campaign_config

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict
from IPython.display import HTML # Ensure this is imported

# ==============================================================================
# 1. HELPER FUNCTIONS (Internal to this module)
# ==============================================================================

def _prepare_validation_paths_and_config(
    project_root: Path,
    source_campaign_name: str,
    n_top: int,
    validation_seed: int,
    plot_results: bool = True,
    ensemble_size: int = 1
) -> Dict:
    """
    Prepares all paths and generates the validation config, exactly replicating
    the logic from the working notebook.
    """
    # 1a. Load source config
    source_config_path = project_root / f"src/experiment_configs/hpo_campaigns/{source_campaign_name.lower()}.yaml"
    source_config = load_campaign_config(source_config_path)

    # 1b. Define names
    validation_campaign_name = f"validation_{source_campaign_name}_top_{n_top}"
    
    # --- 1c. REPLICATE THE WORKING PATH LOGIC ---
    # This uses the exact pathing hacks that you validated.
    hpo_studies_dir = (project_root / source_config.infra.hpo_studies_dir).resolve()
    # Your [6:] slice to handle the `../../` prefix
    profiles_dir = (project_root / source_config.infra.profiles_dir[6:]).resolve() 
    
    # 1d. Construct all necessary paths
    paths = {
        "validation_campaign_name": validation_campaign_name,
        "validation_profile_path": profiles_dir / f"{validation_campaign_name}.csv",
        "validation_config_path": project_root / f"src/experiment_configs/hpo_campaigns/{validation_campaign_name.lower()}.yaml",
        "hpo_storage_url": f"sqlite:///{project_root}/{'src/experiment_configs/studies'}/{source_campaign_name}.db",
        "hpo_results_dir": (project_root / source_config.infra.experiments_output_dir[6:]).resolve(),
        "hpo_studies_dir": (project_root / source_config.infra.hpo_studies_dir[6:]).resolve(),
    }
    # --- END OF PATH LOGIC REPLICATION ---

    # 2. Generate and save the validation YAML
    logging.info(f"Generating validation campaign YAML at: {paths['validation_config_path']}")
    validation_config_dict = source_config.to_dict()
    validation_config_dict['campaign_name'] = paths['validation_campaign_name']
    if 'hpo_params' in validation_config_dict: del validation_config_dict['hpo_params']
    validation_config_dict['job_defaults']['seed'] = validation_seed
    validation_config_dict['job_defaults']['plot'] = plot_results
    validation_config_dict['run_params']['ensemble_size'] = ensemble_size
    
    with open(paths['validation_config_path'], 'w') as f:
        yaml.dump(validation_config_dict, f, sort_keys=False, default_flow_style=None, indent=2)
        
    # 3. Load and resolve paths within the config object for execution
    validation_config = load_campaign_config(paths['validation_config_path'])
    if 'infra' in validation_config and validation_config.infra:
        for key, value in validation_config.infra.items():
            if isinstance(value, str) and not Path(value).is_absolute():
                validation_config.infra[key] = str((project_root / value[6:]).resolve())

    paths['validation_config_object'] = validation_config
    return paths

# ==============================================================================
# 2. THE MAIN ORCHESTRATOR
# ==============================================================================

def run_full_validation_workflow(
    project_root: Path,
    source_campaign_name: str,
    n_top_models: int,
    validation_seed: int,
    plot_results: bool = True,
    ensemble_size: int = 1
):
    """Orchestrates the entire validation workflow from start to finish."""
    
    # --- 1. SETUP ---
    logging.info(f"--- 1. Starting Validation Workflow for '{source_campaign_name}' ---")
    paths = _prepare_validation_paths_and_config(project_root, source_campaign_name, n_top_models, validation_seed, plot_results, ensemble_size)

    # --- 2. GENERATE PROFILE ---
    logging.info(f"\n--- 2. Generating validation profile from DB: {paths['hpo_storage_url']} ---")
    top_trials_df = generate_profile_from_top_trials(
        study_name=source_campaign_name, storage_url=paths['hpo_storage_url'],
        n_top_trials=n_top_models, output_profile_path=paths['validation_profile_path'],
        fixed_params={"seed": validation_seed}
    )
    if top_trials_df is None: return
    display(top_trials_df)
    
    # --- 3. EXECUTE ---
    logging.info(f"\n--- 3. Executing validation campaign: {paths['validation_campaign_name']} ---")
    validation_leaderboard = run_campaign_from_config_object(config=paths['validation_config_object'])
    if validation_leaderboard is None or validation_leaderboard.empty:
        logging.error("Validation run FAILED or produced no results.")
        return
    display(validation_leaderboard)
    
    # --- 4. ANALYZE & COMPARE ---
    logging.info(f"\n--- 4. Generating final comparison report ---")
    hpo_leaderboard = load_master_leaderboard(source_campaign_name, paths['hpo_results_dir'])
    hpo_study = get_study(source_campaign_name, paths['hpo_studies_dir'])

    if hpo_leaderboard is not None and hpo_study is not None:
        run_validation_comparison(hpo_leaderboard, validation_leaderboard, hpo_study)
    else:
        logging.error("Could not load original HPO data for comparison.")



def create_validation_report(
    hpo_leaderboard: pd.DataFrame,
    validation_leaderboard: pd.DataFrame,
    hyperparameter_cols: List[str]
) -> pd.DataFrame:
    """
    Compara HPO vs Validation. Aceita hpo_leaderboard com OU sem sufixos.
    Se vier sem sufixo, a função agrega/mescla e cria *_hpo e *_validation.
    """
    # Detecta se já está sufixado
    need_merge = not {"val_smape_cum_hpo","val_smape_agg_hpo"}.issubset(hpo_leaderboard.columns)

    if need_merge:
        # HPO (sem sufixo): reduz a uma linha por configuração (melhor métrica)
        hpo_reduced = (
            hpo_leaderboard
            .sort_values(["val_smape_cum","val_smape_agg"], ascending=[True, True])
            .drop_duplicates(subset=hyperparameter_cols, keep="first")
            [hyperparameter_cols + ["val_smape_cum", "val_smape_agg"]]
            .rename(columns={
                "val_smape_cum": "val_smape_cum_hpo",
                "val_smape_agg": "val_smape_agg_hpo",
            })
        )
    else:
        # Já vem com *_hpo
        hpo_reduced = hpo_leaderboard[hyperparameter_cols + ["val_smape_cum_hpo","val_smape_agg_hpo"]]

    # Validation: média por configuração
    val_agg = (
        validation_leaderboard
        .groupby(hyperparameter_cols, as_index=False)
        .agg(val_smape_cum_validation=("val_smape_cum","mean"),
             val_smape_agg_validation=("val_smape_agg","mean"))
    )

    # Merge
    comparison_df = pd.merge(hpo_reduced, val_agg, on=hyperparameter_cols, how="inner")

    # Deltas relativos (%)
    comparison_df['Δ SMAPE Cum (%)'] = (
        (comparison_df['val_smape_cum_validation'] - comparison_df['val_smape_cum_hpo'])
        / comparison_df['val_smape_cum_hpo'] * 100
    )
    comparison_df['Δ SMAPE Agg (%)'] = (
        (comparison_df['val_smape_agg_validation'] - comparison_df['val_smape_agg_hpo'])
        / comparison_df['val_smape_agg_hpo'] * 100
    )

    # Equivalência (tolerância simples, mantenho seu critério original)
    comparison_df['Equivalent?'] = comparison_df['Δ SMAPE Cum (%)'].abs().le(20.0)

    ideal_cols = [
        'architecture_profile','batch_size','data_sample','epochs','lag_window',
        'learning_rate','physics_strategy',
        'val_smape_cum_hpo','val_smape_agg_hpo',
        'val_smape_cum_validation','val_smape_agg_validation',
        'Δ SMAPE Cum (%)','Δ SMAPE Agg (%)','Equivalent?'
    ]
    final_cols = [c if c in comparison_df.columns else None for c in ideal_cols]
    final_cols = [c for c in final_cols if c is not None]

    report_df = comparison_df[final_cols].copy().rename(columns={
        "val_smape_cum_hpo": "SMAPE Cum (HPO)",
        "val_smape_cum_validation": "SMAPE Cum (Validation)",
        "val_smape_agg_hpo": "SMAPE Agg (HPO)",
        "val_smape_agg_validation": "SMAPE Agg (Validation)",
    })
    return report_df.sort_values(by='SMAPE Cum (Validation)').reset_index(drop=True)


def style_validation_report(report_df: pd.DataFrame):
    """Applies beautiful styling to the validation report for presentation."""
    return report_df.style.format({
        'SMAPE Cum (HPO)': '{:.4f}',
        'SMAPE Cum (Validation)': '{:.4f}',
        'Δ SMAPE Cum (%)': '{:+.2f}%',
        'SMAPE Agg (HPO)': '{:.2f}',
        'SMAPE Agg (Validation)': '{:.2f}',
        'Δ SMAPE Agg (%)': '{:+.2f}%',
    }).background_gradient(
        cmap='coolwarm',
        subset=['Δ SMAPE Cum (%)', 'Δ SMAPE Agg (%)'],
        vmin=-25, vmax=25
    ).map(
        lambda is_equiv: 'color: #155724; background-color: #d4edda; font-weight: bold;' if is_equiv else 'color: #721c24; background-color: #f8d7da;',
        subset=['Equivalent?']
    ).set_caption(
        "HPO vs. Validation Performance Comparison"
    ).set_properties(**{'text-align': 'center'})

# ==============================================================================
# HIGH-LEVEL VALIDATION WORKFLOW
# ==============================================================================
# In src/hpo/analysis.py

def _format_pp(x: float, decimals: int = 3) -> str:
    # x is already in percent units; Δ in percentage points (pp)
    return f"{x:.{decimals}f}"

def _style_equivalence(val: bool) -> str:
    return "background-color:#DCFCE7;color:#065F46;font-weight:bold;" if val else "background-color:#FEE2E2;color:#7F1D1D;font-weight:bold;"

def build_validation_delta_table(
    joined_df: pd.DataFrame,
    *,
    cum_pp_tolerance: float = 0.25,  # tolerance in percentage points for SMAPE Cum
    agg_pp_tolerance: float = 1.00,  # tolerance in percentage points for SMAPE Agg
    show_relative_columns: bool = False,
    hyperparameter_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute absolute percentage-point deltas between HPO and Validation.
    Uses pp bands to decide 'Equivalent?' (equivalence testing mindset).

    Assumptions:
      - `joined_df` contains suffix columns from a merge:
        'val_smape_cum_hpo', 'val_smape_cum_validation',
        'val_smape_agg_hpo', 'val_smape_agg_validation'
      - Values are *percent* (e.g., 0.54 means 0.54%).
    """
    req = [
        "val_smape_cum_hpo", "val_smape_cum_validation",
        "val_smape_agg_hpo", "val_smape_agg_validation",
    ]
    missing = [c for c in req if c not in joined_df.columns]
    if missing:
        raise KeyError(f"Joined dataframe missing required columns: {missing}")

    df = joined_df.copy()

    # Compute percentage-point deltas
    df["delta_cum_pp"] = df["val_smape_cum_validation"] - df["val_smape_cum_hpo"]
    df["delta_agg_pp"] = df["val_smape_agg_validation"] - df["val_smape_agg_hpo"]

    # Optional: still compute relative deltas for context (NOT used for equivalence)
    if show_relative_columns:
        eps = 1e-12
        df["delta_cum_rel_pct"] = 100.0 * df["delta_cum_pp"] / df["val_smape_cum_hpo"].clip(lower=eps)
        df["delta_agg_rel_pct"] = 100.0 * df["delta_agg_pp"] / df["val_smape_agg_hpo"].clip(lower=eps)

    # Equivalence via absolute pp bands
    df["Equivalent?"] = (df["delta_cum_pp"].abs() <= cum_pp_tolerance) & \
                        (df["delta_agg_pp"].abs() <= agg_pp_tolerance)

    # Build a concise, readable table
    hp_cols = hyperparameter_cols or []
    keep_cols = [
        # Hyperparameters (if present, keep order)
        *[c for c in hp_cols if c in df.columns],

        # Metrics
        "val_smape_cum_hpo", "val_smape_agg_hpo",
        "val_smape_cum_validation", "val_smape_agg_validation",
        "delta_cum_pp", "delta_agg_pp", "Equivalent?",
    ]
    if show_relative_columns:
        keep_cols += ["delta_cum_rel_pct", "delta_agg_rel_pct"]

    # Always include a few context cols if present and not already in hp_cols
    context_cols = [c for c in ["well", "architecture_name", "physics_strategy"] if c in df.columns and c not in keep_cols and c not in hp_cols]
    keep_cols = context_cols + keep_cols

    out = df[[c for c in keep_cols if c in df.columns]].copy()

    # Rename for display
    rename_map = {
        "val_smape_cum_hpo": "SMAPE Cum (HPO)",
        "val_smape_agg_hpo": "SMAPE Agg (HPO)",
        "val_smape_cum_validation": "SMAPE Cum (Validation)",
        "val_smape_agg_validation": "SMAPE Agg (Validation)",
        "delta_cum_pp": "Δ Cum (pp)",
        "delta_agg_pp": "Δ Agg (pp)",
        "delta_cum_rel_pct": "Δ Cum (rel %)",
        "delta_agg_rel_pct": "Δ Agg (rel %)",
    }
    out.rename(columns=rename_map, inplace=True)
    return out


def run_validation_comparison(
    hpo_leaderboard: pd.DataFrame,
    validation_leaderboard: Optional[pd.DataFrame] = None,
    hyperparameter_cols: Optional[List[str]] = None,
    *,
    # NEW: equivalence bands in percentage points
    cum_pp_tolerance: float = 0.25,
    agg_pp_tolerance: float = 1.00,
    # Whether to also show old relative % deltas (informational only)
    show_relative_columns: bool = False,
    # Return options
    return_df: bool = False,
):
    """
    Backward-compatible entry point. If `hpo_leaderboard` already contains merged
    *_hpo and *_validation columns, we use it directly; otherwise we try to merge
    using `hyperparameter_cols` with `validation_leaderboard`.

    Adds absolute pp deltas and an 'Equivalent?' flag by tolerance bands.
    """
    df_joined = hpo_leaderboard.copy()

    # Detect if already merged
    required_merged = {"val_smape_cum_hpo", "val_smape_cum_validation", "val_smape_agg_hpo", "val_smape_agg_validation"}
    if not required_merged.issubset(df_joined.columns):
        if validation_leaderboard is None:
            raise ValueError(
                "hpo_leaderboard is not a merged table and no validation_leaderboard provided."
            )
        # Try merging on provided keys
        if not hyperparameter_cols:
            raise ValueError("To auto-merge, hyperparameter_cols must be provided.")
        keys = ["well", "architecture_name"] + list(hyperparameter_cols)
        keys = [k for k in keys if k in hpo_leaderboard.columns and k in validation_leaderboard.columns]

        df_joined = pd.merge(
            validation_leaderboard, hpo_leaderboard,
            on=keys, how="left", suffixes=("_validation", "_hpo")
        )

    table = build_validation_delta_table(
        df_joined,
        cum_pp_tolerance=cum_pp_tolerance,
        agg_pp_tolerance=agg_pp_tolerance,
        show_relative_columns=show_relative_columns,
        hyperparameter_cols=hyperparameter_cols,
    )

    # Styler for readability
    format_map = {
        "SMAPE Cum (HPO)": "{:.3f}",
        "SMAPE Cum (Validation)": "{:.3f}",
        "Δ Cum (pp)": "{:.3f}",
        "SMAPE Agg (HPO)": "{:.2f}",
        "SMAPE Agg (Validation)": "{:.2f}",
        "Δ Agg (pp)": "{:.2f}",
        "Δ Cum (rel %)": "{:+.1f}%",
        "Δ Agg (rel %)": "{:+.1f}%",
    }
    present_cols = [c for c in format_map if c in table.columns]

    styler = (
        table.style
             .format({c: fmt for c, fmt in format_map.items() if c in table.columns})
             .apply(lambda s: ["font-weight:bold;" if s.name in ("SMAPE Cum (HPO)", "SMAPE Cum (Validation)",
                                                                 "SMAPE Agg (HPO)", "SMAPE Agg (Validation)") else ""],
                    axis=0)
             .apply(lambda col: [_style_equivalence(v) if col.name == "Equivalent?" else "" for v in col], axis=0)
             .set_properties(subset=present_cols, **{"white-space": "nowrap"})
    )

    if return_df:
        return table
    try:
        from IPython.display import display
        display(styler)
    except Exception:
        # Fallback: return raw df if not in a notebook
        return table