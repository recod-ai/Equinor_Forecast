# In src/hpo/search_space.py

import yaml
from pathlib import Path
import optuna
from typing import Dict, Any
import logging

def _get_architecture_profiles(yaml_path: str | Path) -> list:
    """
    Private helper to load architecture profile names from a given YAML file path.
    This is kept as an internal detail of the module.
    """
    # Added a check for a more robust function
    path = Path(yaml_path)
    if not path.is_file():
        # If the file doesn't exist, it's better to return an empty list
        # than to crash, especially if some search spaces don't need it.
        return []
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    # Ensure data is a dictionary before calling .keys()
    return list(data.keys()) if isinstance(data, dict) else []

# ==============================================================================
# 1. INDEPENDENTLY DEFINED SEARCH SPACE FUNCTIONS
# ==============================================================================
# Each function is now self-contained and ready to be used.

def define_seq2context_space(trial: optuna.trial.Trial, architecture_yaml_path: str) -> Dict[str, Any]:
    """
    Search space specifically for Seq2Context models.
    It now explicitly requires the path to the architecture definitions.
    """

    logging.info(Path.cwd())
    logging.info(architecture_yaml_path)
    architecture_profiles = _get_architecture_profiles(architecture_yaml_path)
    if not architecture_profiles:
        raise ValueError(f"Could not find architecture profiles at {architecture_yaml_path} for Seq2Context search.")

    return {
        "architecture_profile": trial.suggest_categorical("architecture_profile", architecture_profiles),
        # "lag_window": trial.suggest_categorical("lag_window", [15, 30]),
        "epochs": trial.suggest_categorical("epochs", [20, 50, 100, 200]),
        "learning_rate": trial.suggest_categorical("learning_rate", [5e-2, 1e-2, 1e-3, 5e-3]),
        "data_sample": trial.suggest_categorical("data_sample", [0.01, 0.25, 0.5, 0.99]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "physics_strategy": trial.suggest_categorical("physics_strategy", [
            "pressure_ensemble", "arps", "combined_exp_arps", "exponential", "static"
        ]),
    }

def define_seq2pin_family_space(trial: optuna.trial.Trial, **kwargs) -> Dict[str, Any]:
    """
    Seq2PIN/Seq2Trend family:
      - learning_rate: agora contínuo log-uniforme (sem grid + jitter).
      - aggregation é post-hoc (sweep): fixamos 'AUTO' e habilitamos o sweep flag.
    """
    # 1) Standard choices
    physics_strategy = trial.suggest_categorical(
        "physics_strategy",
        ["pressure_ensemble", "arps", "combined_exp_arps", "exponential", "static"]
    )
    epochs = trial.suggest_categorical("epochs", [20, 50, 100, 200, 250, 300])

    # 2) Batch size
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # 3) Learning rate contínuo em log10 (1e-4 a 1e-1, ajuste se quiser)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    lr = float(round(lr, 6))  # reprodutibilidade no CSV

    # 4) Data sample (continua em log-float arredondado)
    data_sample = _suggest_rounded_float(trial, "data_sample", 0.001, 0.5, log=True, decimals=4)

    return {
        "physics_strategy": physics_strategy,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "data_sample": data_sample,

        # Delegate filter choice to the post-hoc evaluation sweep
        "aggregation_method": "AUTO",
        "aggregation_sweep": True,
    }




from typing import Dict, Any

# --- NEW: Helper function to suggest floats with controlled precision ---
def _suggest_rounded_float(
    trial, 
    name: str, 
    low: float, 
    high: float, 
    decimals: int = 6, 
    log: bool = False
) -> float:
    """
    A wrapper around trial.suggest_float that rounds the suggested value.
    This ensures that the hyperparameters have a reasonable, serializable precision,
    preventing reproducibility issues caused by floating-point inaccuracies when
    saving and reloading results.

    Args:
        trial: The Optuna trial object.
        name: The name of the hyperparameter.
        low: The lower bound of the range.
        high: The upper bound of the range.
        decimals: The number of decimal places to round to.
        log: Whether to sample from a log-uniform distribution.

    Returns:
        The rounded floating-point value.
    """
    value = trial.suggest_float(name, low, high, log=log)
    return round(value, decimals)


def define_arps_space(trial, **overrides) -> Dict[str, Any]:
    """
    Optuna search space for the canonical ARPS backend.

    This version uses a helper to suggest floats with controlled precision,
    ensuring reproducibility between HPO and validation runs.
    """
    # --- Core model knobs ---
    variant = trial.suggest_categorical("variant", ["hyperbolic", "harmonic", "exponential"])
    solver  = trial.suggest_categorical("solver",  ["grid", "lbfgs"])
    weighting = trial.suggest_categorical("weighting", ["none", "1_over_q2", "time_decay"])
    loss = trial.suggest_categorical("loss", ["wls", "huber", "cauchy", "quantile"])

    # --- Use the new rounded float suggestion for all continuous parameters ---
    loss_delta = _suggest_rounded_float(trial, "loss_delta", 1e-3, 5.0, log=True, decimals=4)
    
    quantile_tau = 0.5
    if loss == "quantile":
        quantile_tau = _suggest_rounded_float(trial, "quantile_tau", 0.1, 0.9, decimals=4)

    burn_in_fraction = _suggest_rounded_float(trial, "burn_in_fraction", 0.0, 0.2, decimals=4)

    piecewise = trial.suggest_categorical("piecewise", [False, True])
    piecewise_min_delta_bic = _suggest_rounded_float(trial, "piecewise_min_delta_bic", 0.0, 15.0, decimals=4)

    # --- b search strategy: grid vs bounds (exclusive) ---
    params: Dict[str, Any]

    if solver == "grid":
        b_grid_kind = trial.suggest_categorical("b_grid_kind", ["lin", "log"])
        b_min = _suggest_rounded_float(trial, "b_min", 0.05, 1.2, decimals=4)
        # Ensure b_max is always greater than b_min
        b_max_low = max(b_min + 0.01, 0.10) 
        b_max = _suggest_rounded_float(trial, "b_max", b_max_low, 2.5, decimals=4)
        b_grid_size = trial.suggest_int("b_grid_size", 10, 50)

        params = dict(
            variant=variant, solver=solver, weighting=weighting, loss=loss,
            loss_delta=loss_delta, quantile_tau=quantile_tau,
            burn_in_fraction=burn_in_fraction, piecewise=piecewise,
            piecewise_min_delta_bic=piecewise_min_delta_bic,
            b_grid_kind=b_grid_kind, b_min=b_min, b_max=b_max, b_grid_size=b_grid_size,
        )
    else: # solver == "lbfgs"
        b_min = _suggest_rounded_float(trial, "b_min", 0.05, 1.2, decimals=4)
        b_max_low = max(b_min + 0.01, 0.10)
        b_max = _suggest_rounded_float(trial, "b_max", b_max_low, 2.5, decimals=4)

        params = dict(
            variant=variant, solver=solver, weighting=weighting, loss=loss,
            loss_delta=loss_delta, quantile_tau=quantile_tau,
            burn_in_fraction=burn_in_fraction, piecewise=piecewise,
            piecewise_min_delta_bic=piecewise_min_delta_bic,
            b_min=b_min, b_max=b_max,
        )

    # Allow upstream to inject fixed values or narrow ranges
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})

    return params

def define_darts_space(trial: optuna.trial.Trial, **overrides) -> Dict[str, Any]:
    """
    Unified Darts search space using 'snap-to-grid + small log-jitter' for learning_rate,
    mirroring the Seq2 family behavior. Still selects a model (physics_strategy alias),
    then a compact per-model profile grid, and finally adds universal training knobs.

    Configuration via YAML (hpo_params.search_space_overrides), e.g.:
      search_space_overrides:
        allowed_models: ["TiDE","TiDE_RIN","NHiTS","NLinear","LinearRegression","ARIMA","AutoARIMA"]
        exclude_models: []
        profile_limit_per_model: 5

        # Universal training knobs (sampled here):
        epochs_choices: [20, 50, 100, 200]
        batch_size_choices: [8, 16, 32]

        # Learning rate setup (snap grid + jitter just like Seq2):
        learning_rate_grid: [0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001]
        learning_rate_jitter_log10_range: [-0.25, 0.25]
        round_lr_decimals: 6

        # Optional base dims for some grids (LinearRegression lags, etc.):
        lag_window: 100
        horizon: 300
        # Or directly:
        input_chunk_length: 100
        output_chunk_length: 300
    """
    # ---------------------------------------------------------
    # 1) Select the algorithm (model family = physics_strategy)
    # ---------------------------------------------------------
    all_models = list(GRID_BUILDERS.keys())  # from grids_darts.py
    include = overrides.get("allowed_models", all_models) or all_models
    exclude = set(overrides.get("exclude_models", []) or [])
    models = [m for m in include if m in GRID_BUILDERS and m not in exclude]
    if not models:
        models = all_models

    model_key = trial.suggest_categorical("darts_model", models)  # alias of physics_strategy

    # ---------------------------------------------------------
    # 2) Base dims for grids that use them (harmless passthrough)
    # ---------------------------------------------------------
    base = {
        "input_chunk_length": overrides.get("input_chunk_length", overrides.get("lag_window", 100)),
        "output_chunk_length": overrides.get("output_chunk_length", overrides.get("horizon", 100)),
    }

    # ---------------------------------------------------------
    # 3) Build per-model profile grid (small, curated list)
    #    and optionally limit it to N profiles per model
    # ---------------------------------------------------------
    grid = make_search_grid(model_key, base)  # returns a list of dicts, each with a "profile" key
    limit = int(overrides.get("profile_limit_per_model", overrides.get("darts_profile_limit", 5)) or 5)
    grid = grid[:max(1, min(limit, len(grid)))]

    # Stable Optuna param name per model so the search space doesn’t “change”
    slug = model_key.lower().replace("+", "plus").replace(" ", "_")
    profile_names = [g["profile"] for g in grid]
    chosen_profile = trial.suggest_categorical(f"profile__{slug}", profile_names)
    prof = next(g for g in grid if g["profile"] == chosen_profile)

    # Ensure JSON serializable payloads for CSVs / configs
    if "seasonal_order" in prof and isinstance(prof["seasonal_order"], tuple):
        prof = {**prof, "seasonal_order": list(prof["seasonal_order"])}

    # ---------------------------------------------------------
    # 4) Universal training knobs (epochs, batch_size, LR)
    #    LR = snap-to-grid + small log10 jitter (Seq2-style)
    # ---------------------------------------------------------
    epochs_choices = overrides.get("epochs_choices", [20, 50, 100, 200])
    batch_choices  = overrides.get("batch_size_choices", [8, 16, 32])

    n_epochs = trial.suggest_categorical("n_epochs", epochs_choices)
    batch_size = trial.suggest_categorical("batch_size", batch_choices)

    # Snap-to-grid for LR (same idea as Seq2) + small jitter
    lr_grid = overrides.get("learning_rate_grid",
                            [1e-1, 5e-2, 3e-2, 1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4])
    lr_base = trial.suggest_categorical("learning_rate_grid", lr_grid)

    jitter_min, jitter_max = overrides.get("learning_rate_jitter_log10_range", [-0.25, 0.25])
    jitter_log10 = trial.suggest_float("learning_rate_jitter_log10", float(jitter_min), float(jitter_max))
    learning_rate = float(lr_base * (10.0 ** jitter_log10))

    # Round for reproducibility in configs/CSVs
    lr_decimals = int(overrides.get("round_lr_decimals", 6))
    learning_rate = float(round(learning_rate, lr_decimals))

    # ---------------------------------------------------------
    # 5) Compose final param dict (trainer-compatible)
    # ---------------------------------------------------------
    params: Dict[str, Any] = {
        "architecture_name": MODEL_TO_ARCH[model_key],  # e.g., "Darts_TiDE"
        "physics_strategy": model_key,                  # alias used by analytics (Seq2-like behavior)
        "profile": chosen_profile,

        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,

        # Keep fields surfaced in leaderboards (non-breaking)
        "expand_darts_profiles": True,
        "darts_profile_limit": limit,
    }

    # Pass base dims (harmless if trainer overrides)
    if base.get("input_chunk_length") is not None:
        params["input_chunk_length"] = int(base["input_chunk_length"])
    if base.get("output_chunk_length") is not None:
        params["output_chunk_length"] = int(base["output_chunk_length"])

    # Merge the chosen profile’s hyperparams (excluding the 'profile' key itself)
    params.update({k: v for k, v in prof.items() if k != "profile"})

    return params



from hpo.grids_darts import GRID_BUILDERS, make_search_grid
# Map human-friendly model keys to the trainer's architecture_name
MODEL_TO_ARCH = {
    "TiDE":              "Darts_TiDE",
    "TiDE_RIN":          "Darts_TiDE_RIN",             # same builder, RIN-tilted profiles
    "NHiTS":             "Darts_NHiTS",
    "NLinear":           "Darts_NLinear",
    "LinearRegression":  "Darts_LinearRegression",
    "ARIMA":             "Darts_ARIMA",
    "AutoARIMA":         "Darts_AutoARIMA",
}



# ==============================================================================
# 2. THE DISPATCHER MAPPING (For the new system)
# ==============================================================================
# This dictionary will be used by our new `hpo_runner.py` to select the
# correct search space function based on the config YAML.
SEARCH_SPACE_MAPPING = {
    "Seq2Context":     define_seq2context_space,
    "Seq2Trend":       define_seq2pin_family_space,
    "Seq2PIN":         define_seq2pin_family_space,
    "Darts":           define_darts_space,
    "Arps_Canonical":  define_arps_space
}