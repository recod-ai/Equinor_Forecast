# File: src/forecast_pipeline/jobs.py

# 1. Future imports
from __future__ import annotations

# 2. Standard library imports
import gc
import inspect
import logging
import math
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from typing import Any, Dict
import ast

# 3. Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from box import Box

# 4. Local application imports (absolute paths)
from common.config_wells import get_data_sources
from evaluation.evaluation import (
    compute_metrics_to_df,
    compute_metrics_to_df_seq,
    evaluate,
    evaluate_and_plot,
    evaluate_cumulative,
    evaluate_cumulative_seq,
    evaluate_model,
    evaluate_model_seq,
)
from forecast_pipeline.analytics import (
    _reconstruct_train_series_phys,
    evaluate_job,
    scenario_curve,
)
from forecast_pipeline.analytics_darts import evaluate_job_from_darts
from forecast_pipeline.arps_canonical import (
    ArpsParams,
    fit_arps_canonical,
    forecast_canonical_from_train,
    build_arps_ensemble_out_from_theta_hat
)
from forecast_pipeline.ensemble_output import EnsembleOutput, to_ensemble_output
from forecast_pipeline.io_utils import atomic_write_json, build_run_metadata
from forecast_pipeline.logging_utils import get_logger, log_context, phase
from hpo.grids_darts import make_search_grid
from profile_manager import generate_job_hash, load_and_expand_profile
from training.train_darts import main_train_darts_model
from training.train_models import main_train_model
from training.train_utils import analyze_contributions, train_predict_chunk
from utils.utilities import _adapt_arps_kwargs_for_fit

# 5. Local application imports (relative paths)
from .config import (
    DEFAULT_EXP_PARAMS,
    EXPERIMENT_CONFIGURATIONS,
    EXTRACTOR_OPTIONS,
    FUSER_OPTIONS,
    SEQ2SEQ_ARCHS,
    STRATEGY_OPTIONS,
    get_experiment_base_config,
)
from .experiments.arps_experiment import ExperimentArps
from .experiments.darts_experiment import ExperimentDarts
from .experiments.seq2context import ExperimentSeq2Context
from .experiments.seq2value import ExperimentSeq2Value
from .plotting import plot_integrated_view, plot_predictions_wrapper


def _legacy_generate_jobs(
    data_sources: List[Dict[str, Any]],
    default_exp: Dict[str, Any]
) -> List[Tuple[Dict[str, Any], str, Dict[str, Any], int]]:
    """
    Build the list of experiment jobs from data sources and default parameters.
    """
    jobs: List[Tuple[Dict[str, Any], str, Dict[str, Any], int]] = []
    exp_id = 1
    for ds in data_sources:
        ds_name = ds["name"]
        base_configs = EXPERIMENT_CONFIGURATIONS.get(ds_name, [])
        if not base_configs:
            logging.debug(f"Skipping {ds_name}: no experiment configurations defined.")
            continue
        for well in ds.get("wells", []):
            for config in base_configs:
                for strat in STRATEGY_OPTIONS:
                    for ext in EXTRACTOR_OPTIONS:
                        for fus in FUSER_OPTIONS:
                            params = {
                                **default_exp,
                                **config,
                                "strategy_config": strat,
                                "extractor_config": ext,
                                "fuser_config": fus
                            }
                            jobs.append((ds, well, params, exp_id))
                            exp_id += 1
    return jobs


# --- NEW imports at top of the module (or keep near existing imports) ---
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    # Curated Darts profiles
    from hpo.grids_darts import make_search_grid
except Exception:
    make_search_grid = None  # Graceful degradation if not available

from common.config_wells import get_data_sources



def _derive_model_key_from_arch(arch_name: str) -> Optional[str]:
    """'Darts_TiDE' -> 'TiDE'. Returns None if not a Darts architecture."""
    if not isinstance(arch_name, str):
        return None
    if arch_name.startswith("Darts_"):
        return arch_name.split("Darts_", 1)[1]
    return None


def _expand_darts_family_for_ds(
    ds: Dict[str, Any],
    wells: List[str],
    base_params: Dict[str, Any],
    model_keys: List[str],
    limit: Optional[int],
) -> List[Tuple[Dict[str, Any], str, Dict[str, Any], str]]:
    """
    Build (ds_config, well, params, exp_id) tuples for Darts models using curated grids.
    """
    if make_search_grid is None:
        raise RuntimeError("hpo.grids_darts.make_search_grid not available; cannot expand Darts family.")

    jobs: List[Tuple[Dict[str, Any], str, Dict[str, Any], str]] = []
    ds_name = ds["name"]

    for well in wells:
        for mk in model_keys:
            arch_name = f"Darts_{mk}"
            # rule-based features / base knobs per dataset + arch
            base_cfg = get_experiment_base_config(ds_name, arch_name)

            grid = make_search_grid(mk, base_params)
            if limit is not None:
                grid = grid[:limit]
            logging.info("Darts grid for %s (%s): %d profiles", arch_name, well, len(grid))

            for i, prof in enumerate(grid, start=1):
                exp_id = prof.get("experiment_id") or f"darts_{mk}_{i:03d}"
                final_params = {
                    **base_params,
                    **base_cfg,
                    **prof,
                    "architecture_name": arch_name,
                    "physics_strategy": mk,
                    "target_column": ds.get("target_column", "BORE_OIL_VOL"),
                }
                jobs.append((ds, well, final_params, exp_id))
    return jobs


# --- add these small helpers somewhere above generate_jobs ---

def _is_darts_request(config_or_defaults: Dict[str, Any]) -> bool:
    """Detects when caller intends to run Darts (no physics)."""
    arch = str(config_or_defaults.get("architecture_name", "") or "")
    return bool(
        config_or_defaults.get("darts_model_keys") or
        config_or_defaults.get("expand_darts_profiles") or
        arch.startswith("Darts_")
    )

def _build_darts_jobs(
    data_sources: List[Dict[str, Any]],
    params: Dict[str, Any],
    profile_limit: Optional[int] = None,
    model_keys: Optional[List[str]] = None,
) -> List[Tuple[Dict[str, Any], str, Dict[str, Any], str]]:
    """
    Expand Darts families into curated profiles.
    - Sets architecture_name = f"Darts_{family}"
    - Sets physics_strategy = family (label only; no physics)
    - Carries over JOB_DEFAULTS (e.g., epochs, batch_size, splits)
    """
    jobs: List[Tuple[Dict[str, Any], str, Dict[str, Any], str]] = []

    # Determine the families to expand
    families: List[str] = []
    if model_keys:
        families = model_keys
    elif params.get("darts_model_keys"):
        families = list(params["darts_model_keys"])
    elif params.get("expand_darts_profiles"):
        # infer from architecture_name, e.g., "Darts_TiDE" -> ["TiDE"]
        arch = params.get("architecture_name", "Darts_TiDE")
        if arch.startswith("Darts_"):
            families = [arch.split("Darts_", 1)[1]]
    if not families:
        return jobs  # nothing to do

    # Expand per dataset / well / family / curated profile
    for ds in data_sources:
        ds_name = ds["name"]
        wells = ds.get("wells", [])
        for family in families:
            arch_name = f"Darts_{family}"
            base = {
                **params,
                "architecture_name": arch_name,
                "physics_strategy": family,   # label used by downstream views
            }
            grid = make_search_grid(family, base)  # curated 5 profiles
            if profile_limit is not None:
                grid = grid[:profile_limit]
            for i, profile in enumerate(grid, start=1):
                cfg = {**base, **profile}
                job_id = f"darts_{family}_{i:03d}"
                for well in wells:
                    jobs.append((ds, well, cfg, job_id))
    return jobs



# ==============================================================================
# --- Funções Auxiliares (coloque-as antes da sua generate_jobs) ---
# ==============================================================================

# PATCH em hpo/... (onde está _as_dict_box)

def _as_dict_box(obj) -> Dict[str, Any]:
    job_defaults: Dict[str, Any] = {}
    darts_model_keys = None
    darts_profile_limit = None
    expand_darts_profiles = False

    ensemble_size = None
    rp_seeds = None  # <-- novo

    from box import Box
    if isinstance(obj, Box) and hasattr(obj, "job_defaults") and hasattr(obj, "run_params"):
        job_defaults = obj.job_defaults.to_dict()
        ensemble_size = getattr(obj.run_params, "ensemble_size", None)
        rp_seeds = getattr(obj.run_params, "seeds", None)  # <-- novo
        hp = getattr(obj, "hpo_params", None)
        darts_model_keys = getattr(hp, "darts_model_keys", None) if hp else None
        darts_profile_limit = getattr(hp, "darts_profile_limit", None) if hp else None
        expand_darts_profiles = bool(getattr(obj.job_defaults, "expand_darts_profiles", False))

    elif isinstance(obj, dict) and "job_defaults" in obj and isinstance(obj["job_defaults"], dict):
        jd = dict(obj["job_defaults"])
        job_defaults = jd
        # tentar run_params em dict tb
        rp = obj.get("run_params", {}) or {}
        ensemble_size = rp.get("ensemble_size", obj.get("ensemble_models"))
        rp_seeds = rp.get("seeds")  # <-- novo
        darts_model_keys = obj.get("darts_model_keys")
        darts_profile_limit = obj.get("darts_profile_limit")
        expand_darts_profiles = bool(obj.get("expand_darts_profiles", False) or jd.get("expand_darts_profiles", False))

    else:
        job_defaults = dict(obj) if isinstance(obj, dict) else {}
        ensemble_size = job_defaults.get("ensemble_models")
        rp_seeds = job_defaults.get("seeds")  # <-- novo
        darts_model_keys = job_defaults.get("darts_model_keys")
        darts_profile_limit = job_defaults.get("darts_profile_limit")
        expand_darts_profiles = bool(job_defaults.get("expand_darts_profiles", False))

    # --- NORMALIZAÇÃO DE ENSEMBLE (aliases no job_defaults) ---
    if ensemble_size is not None:
        esz = int(ensemble_size)
        job_defaults["ensemble_size"] = esz        # alias 1
        job_defaults["ensemble_models"] = esz      # alias 2

    if rp_seeds is not None:
        job_defaults["seeds"] = list(rp_seeds)     # deixa explícito para o job

    return {
        "job_defaults": job_defaults,
        "darts_model_keys": darts_model_keys,
        "darts_profile_limit": darts_profile_limit,
        "expand_darts_profiles": expand_darts_profiles,
    }




def _derive_model_key_from_arch(arch_name: str) -> Optional[str]:
    """Converte 'Darts_TiDE' -> 'TiDE'. Retorna None se não for uma arquitetura Darts."""
    if isinstance(arch_name, str) and arch_name.startswith("Darts_"):
        return arch_name.split("Darts_", 1)[1]
    return None


def _expand_darts_family_for_ds(
    ds: Dict[str, Any],
    wells: List[str],
    base_params: Dict[str, Any],
    model_keys: List[str],
    limit: Optional[int],
) -> List[Tuple[Dict[str, Any], str, Dict[str, Any], str]]:
    """Cria tuplas de jobs para modelos Darts usando os grids pré-selecionados."""
    if make_search_grid is None:
        raise RuntimeError("A função 'hpo.grids_darts.make_search_grid' não está disponível.")

    jobs: List[Tuple[Dict[str, Any], str, Dict[str, Any], str]] = []
    ds_name = ds["name"]

    for well in wells:
        for mk in model_keys:
            arch_name = f"Darts_{mk}"
            base_cfg = get_experiment_base_config(ds_name, arch_name)  # Aplica regras de negócio
            grid = make_search_grid(mk, base_params)
            if limit is not None:
                grid = grid[:limit]
            
            logging.info("Grid Darts para %s (%s): %d perfis", arch_name, well, len(grid))

            for i, prof in enumerate(grid, start=1):
                exp_id = prof.get("experiment_id") or f"darts_{mk}_{i:03d}"
                final_params = {
                    **base_params,
                    **base_cfg,
                    **prof,
                    "architecture_name": arch_name,
                    "physics_strategy": mk,
                }
                jobs.append((ds, well, final_params, exp_id))
    return jobs

# ==============================================================================
# --- VERSÃO FINAL E CORRETA DA SUA FUNÇÃO ---
# ==============================================================================

def _resolve_arch_defs_path(cfg_or_defaults) -> str:
    try:
        from pathlib import Path
        # Box → config.infra.architecture_yaml_path
        if isinstance(cfg_or_defaults, Box):
            p = cfg_or_defaults.infra.architecture_yaml_path
            return str(p)
        # dict → ["infra"]["architecture_yaml_path"]
        if isinstance(cfg_or_defaults, dict):
            p = cfg_or_defaults.get("infra", {}).get("architecture_yaml_path")
            if p:
                return str(p)
    except Exception:
        pass
    # Last-resort fallback (shouldn’t be used if ExperimentContext is wired)
    return str((Path.cwd() / "src" / "experiment_configs" / "architectures_PINNs.yaml").resolve())


def _get_runparam_ensemble_size(cfg) -> int | None:
    try:
        # Box
        if hasattr(cfg, "run_params"):
            v = getattr(cfg.run_params, "ensemble_size", None)
            return int(v) if v is not None else None
    except Exception:
        pass
    # dict
    if isinstance(cfg, dict):
        rp = (cfg.get("run_params") or {})
        v = rp.get("ensemble_size", None)
        return int(v) if v is not None else None
    return None


from forecast_pipeline.logging_utils import get_logger, log_context, phase

def generate_jobs(
    data_sources: List[Dict[str, Any]],
    config_or_defaults: Union[Box, Dict[str, Any]],
    profile_path: Optional[str] = None
) -> List[Tuple[Dict[str, Any], str, Dict[str, Any], str]]:
    """
    Generates experiment jobs with full backward compatibility and
    optional expansion of the Darts model family.

    Paths (checked in order):
      D) Rich Profile CSV
      A/B) Classic Profile (Box or dict)
      E) Darts Family Expansion
      C) Legacy Fallback
    """
    logger = get_logger(__name__)
    profile_tag = (str(profile_path) if profile_path else None)

    try:
        profile_df = pd.read_csv(profile_path) if profile_path else None
        is_rich_profile = bool(profile_df is not None and {"well", "dataset"} <= set(profile_df.columns))
    except Exception:
        is_rich_profile = False

    # Normalize knobs once so all paths see defaults
    knobs = _as_dict_box(config_or_defaults)
    job_defaults = knobs["job_defaults"]
    arch_defs_path = _resolve_arch_defs_path(config_or_defaults)

    with log_context(profile=profile_tag, sources=len(data_sources)):
        with phase(logger, "generate_jobs"):
            # --- (D) Rich Profile ---
            if is_rich_profile:
                logger.info("decision=path kind=rich_profile")
                expanded = load_and_expand_profile(profile_path, arch_defs_path=arch_defs_path)
                jobs: List[Tuple[Dict[str, Any], str, Dict[str, Any], str]] = []
                all_data_sources = get_data_sources()

                for idx, row_cfg in enumerate(expanded):
                    target_dataset = row_cfg["dataset"]
                    target_well = row_cfg["well"]
                    ds_full = next((d for d in all_data_sources if d["name"] == target_dataset), None)
                    if not ds_full:
                        logger.warning("skip reason=dataset_not_found dataset=%s row=%d", target_dataset, idx)
                        continue

                    if target_dataset == "VOLVE" and isinstance(target_well, str):
                        target_well = target_well.replace("-", "/", 1)

                    ds_for_job = dict(ds_full)
                    ds_for_job["wells"] = [target_well]

                    arch_name_default = job_defaults.get("architecture_name")
                    arch_name_effective = row_cfg.get("architecture_name", arch_name_default)
                    base_cfg = get_experiment_base_config(target_dataset, arch_name_effective)

                    final_params = {**job_defaults, **base_cfg, **row_cfg}

                    ens = final_params.get("ensemble_size")
                    if ens:
                        esz = int(ens)
                        final_params["ensemble_models"] = esz
                        final_params["ensemble_size"] = esz

                    exp_id = final_params.get("experiment_id", f"validation_trial_{idx}")
                    jobs.append((ds_for_job, target_well, final_params, exp_id))

                logger.info("generated count=%d path=rich_profile", len(jobs))
                return jobs

            # --- (A/B) Classic Profile-Driven ---
            if profile_path:
                logger.info("decision=path kind=profile file=%s", profile_tag)
                profile_configs = load_and_expand_profile(profile_path, arch_defs_path=arch_defs_path)
                jobs = []
                for ds in data_sources:
                    ds_name = ds["name"]

                    if isinstance(config_or_defaults, Box):
                        arch_name_default = config_or_defaults.job_defaults.architecture_name
                    else:
                        arch_name_default = job_defaults.get("architecture_name")

                    for well in ds.get("wells", []):
                        for prof_cfg in profile_configs:
                            arch_name_effective = prof_cfg.get("architecture_name", arch_name_default)
                            base_cfg = get_experiment_base_config(ds_name, arch_name_effective)

                            params = {**job_defaults, **base_cfg, **prof_cfg}
                            exp_id = params.get("experiment_id", "unknown_id")
                            jobs.append((ds, well, params, exp_id))

                num_wells = sum(len(d.get("wells", [])) for d in data_sources)
                logger.info("generated count=%d path=profile configs=%d wells=%d",
                            len(jobs), len(profile_configs), num_wells)
                return jobs

            # --- (E) Darts Family Expansion ---
            darts_model_keys = knobs["darts_model_keys"]
            expand_darts_profiles = knobs["expand_darts_profiles"]
            arch_name_default = job_defaults.get("architecture_name", "")
            single_model_key = _derive_model_key_from_arch(arch_name_default)

            model_keys_to_expand: List[str] = []
            if isinstance(darts_model_keys, list) and darts_model_keys:
                model_keys_to_expand = darts_model_keys
            elif expand_darts_profiles and single_model_key:
                model_keys_to_expand = [single_model_key]

            if model_keys_to_expand:
                logger.info("decision=path kind=darts_expand families=%s limit=%s",
                            model_keys_to_expand, knobs["darts_profile_limit"])
                jobs = []
                for ds in data_sources:
                    if not ds.get("wells"):
                        continue
                    jobs.extend(
                        _expand_darts_family_for_ds(
                            ds=ds,
                            wells=ds["wells"],
                            base_params=job_defaults,
                            model_keys=model_keys_to_expand,
                            limit=knobs["darts_profile_limit"],
                        )
                    )
                logger.info("generated count=%d path=darts_expand", len(jobs))
                return jobs

            # --- (C) Legacy Fallback ---
            logger.warning("decision=path kind=legacy reason=no_profile_and_no_darts")
            legacy_jobs = _legacy_generate_jobs(data_sources, job_defaults)
            out = [(ds, well, params, str(job_id)) for ds, well, params, job_id in legacy_jobs]
            logger.info("generated count=%d path=legacy", len(out))
            return out




def prepare_job_data(job):
    """
    Loads and prepares data for a single job.
    This function is now a dynamic dispatcher that routes jobs based on the
    'architecture_name' specified within the job's own parameters.
    """
    ds, well, params, job_id = job
    arch = params.get("architecture_name")
    
    # Fallback to the global default ONLY if the key is missing in the job's params.
    # This maintains compatibility with very old legacy calls.
    if arch is None:
        logging.warning("'architecture_name' not found in job parameters. Falling back to global DEFAULT_EXP_PARAMS.")
        arch = DEFAULT_EXP_PARAMS.get("architecture_name")

    # --- The Dispatcher Logic ---
    if arch.startswith("Darts_"):
        # NEW RULE: Route all Darts models to a dedicated Experiment class.
        exp_cls = ExperimentDarts
    elif arch in SEQ2SEQ_ARCHS:
        # Existing rule for your hybrid models.
        exp_cls = ExperimentSeq2Context
    elif arch == "Seq2Value":
        # Existing rule for Seq2Value.
        exp_cls = ExperimentSeq2Value
    else:
        raise ValueError(f"Unknown architecture: '{arch}'. No matching Experiment class found.")

    # The rest of the function is now universal and works for all families.
    exp = exp_cls(ds, well, params, job_id)
    
    # The return signature from load_and_prepare will be different for Darts.
    # We will handle this in the next step when we create ExperimentDarts.
    # For now, we assume the same signature.
    (train_kwargs, prediction_input, y_test, 
     scaler_X, scaler_target, y_train_original) = exp.load_and_prepare()

    # This enrichment is still crucial for the hybrid models.
    # We will adapt it for the Darts models as well.
    if scaler_X and scaler_target: # Only add if they are not None
        train_kwargs['scaler_X'] = scaler_X
        train_kwargs['scaler_target'] = scaler_target
    train_kwargs['dataset_name'] = ds['name']

    return (
        train_kwargs,
        prediction_input,
        y_test,
        scaler_X,
        scaler_target,
        y_train_original,
        params,
        ds,
        well,
        job_id
    )

def _analyze_ensemble_stability(
    individual_run_outputs: List[Dict[str, Any]],
    y_val_scaled: np.ndarray,
    scaler_target: Any,
    params: Dict[str, Any]
) -> None:
    """
    Estabilidade no VAL aplicando a MESMA policy definida em params['aggregation_method']
    quando ela não é AUTO; caso seja AUTO/sweep, faz um sweep SEGURO (sem reconstruct_warm_*)
    e escolhe a melhor pela métrica (default: SMAPE). Loga mean/std e a policy escolhida.
    """

    from evaluation.evaluation import evaluate  # (R², SMAPE, MAE)
    from common.seq_preprocessing import aggregate_predictions, reconstruct_true_series

    logger = logging.getLogger(__name__)

    # --- 0) Guard-clause: sem membros não há o que medir ---
    if not individual_run_outputs:
        logger.info("stability_analysis=skip reason=no_members")
        return

    # --- 1) Inverse-transform do VAL (GT + predições por membro) ---
    def _inv(x):
        try:
            return scaler_target.inverse_transform(x)
        except Exception:
            return x

    y_val_inv = _inv(y_val_scaled)                    # (K, H) ou (N,H)
    gt_1d     = reconstruct_true_series(y_val_inv)    # (H,)

    member_preds_inv: List[np.ndarray] = []
    for run in individual_run_outputs:
        p = run.get("pred_val")
        if p is None:
            continue
        member_preds_inv.append(_inv(p))
    if not member_preds_inv:
        logger.info("stability_analysis=skip reason=no_member_preds")
        return

    # stack & média (K,H) preservando contrato
    # (para estabilidade, usamos a média simples das ribbons por membro)
    mean_pred_inv = np.mean(member_preds_inv, axis=0)
    H_pred = int(mean_pred_inv.shape[1])

    # --- 2) LEFT opcional (não obrigatório para hp_*_warm) ---
    warm_left_inv = None
    try:
        # preferir LEFT por membro quando houver
        lefts = []
        for run in individual_run_outputs:
            l = run.get("pred_val_left")
            if l is not None:
                lefts.append(_inv(l))
        if lefts:
            warm_left_inv = np.mean(lefts, axis=0)
        else:
            # sidecar do scaler (construído na etapa de prep)
            ctx = getattr(scaler_target, "_split_ctx", {}) or {}
            warm_left_scaled = ctx.get("X_val_left_scaled")
            if warm_left_scaled is not None:
                # extrai canal-alvo se vier (K,L,F) ou (K,L)
                if warm_left_scaled.ndim == 3:
                    target = warm_left_scaled[:, :, -1]
                else:
                    target = warm_left_scaled
                warm_left_inv = _inv(target)
    except Exception:
        warm_left_inv = None

    L_left = int(warm_left_inv.shape[1]) if (isinstance(warm_left_inv, np.ndarray) and warm_left_inv.ndim >= 2) else None

    # --- 3) Política alvo / modo de seleção ---
    policy_in  = str(params.get("aggregation_method", "reconstruct")).strip()
    sel_metric = str(params.get("aggregation_selection_metric", "SMAPE")).upper()

    def _score(r2, smape, mae) -> float:
        if sel_metric == "R²" or sel_metric == "R2":
            return -float(r2)   # maximiza R²
        if sel_metric == "MAE":
            return float(mae)
        return float(smape)     # default: SMAPE

    # Políticas que exigem LEFT com H colunas (incompatíveis com L_left=100)
    WARM_RECON_POLICIES = {"reconstruct_warm_raw", "reconstruct_warm_ewma", "reconstruct_warm_holt", "reconstruct_warm_hp"}

    # Conjunto seguro para estabilidade (funcionam sem LEFT=H)
    SAFE_CANDIDATES = ["hp_raw_warm", "hp_hist_warm", "hp_raw", "hp_hist", "reconstruct"]

    def _try_policy(pol: str) -> Tuple[float, Dict[str, float]]:
        """Retorna (score, métricas) — lança somente em erros inesperados."""
        args = {"policy": pol, "hp_lambda": float(params.get("hp_lambda", 8000.0))}
        if pol in WARM_RECON_POLICIES:
            # Skip se não há LEFT compatível (L_left==H)
            if warm_left_inv is None or L_left != H_pred:
                raise RuntimeError("skip_left_mismatch")
            args["warm_left_windows_2d"] = warm_left_inv
        elif pol.endswith("_warm") and (warm_left_inv is not None):
            # hp_*_warm aceitam LEFT opcional; se existir, passamos
            args["warm_left_windows_2d"] = warm_left_inv

        agg_pred = aggregate_predictions(mean_pred_inv, **args)
        r2, smape, mae = evaluate(gt_1d, agg_pred)
        return _score(r2, smape, mae), {"R²": r2, "SMAPE": smape, "MAE": mae}

    rows = []

    # --- 4A) Caminho FIXO: policy explícita (não AUTO) -------------------------
    if policy_in and policy_in.upper() != "AUTO":
        try:
            s, m = _try_policy(policy_in.lower())
            rows.append((policy_in, s, m))
            chosen = policy_in
        except RuntimeError as e:
            if str(e) == "skip_left_mismatch":
                logger.warning("[Stability] explicit_policy_incompatible policy=%s reason=LEFT/H mismatch (L_left=%s, H=%s)",
                               policy_in, L_left, H_pred)
                # fallback suave: tenta SAFE_CANDIDATES
                best = None
                for pol in SAFE_CANDIDATES:
                    try:
                        ss, mm = _try_policy(pol)
                        rows.append((pol, ss, mm))
                        if (best is None) or (ss < best[0]):
                            best = (ss, pol, mm)
                    except RuntimeError as e2:
                        if str(e2) == "skip_left_mismatch":
                            continue
                        logger.exception("[Stability] policy_failed policy=%s", pol)
                if best is None:
                    # fallback final: reconstruct (sem erro)
                    pol = "reconstruct"
                    ss, mm = _try_policy(pol)
                    rows.append((pol, ss, mm))
                    chosen = pol
                else:
                    chosen = best[1]
            else:
                logger.exception("[Stability] explicit_policy_failed policy=%s", policy_in)
                return
        # chosen já definido
    # --- 4B) Caminho AUTO/sweep seguro -----------------------------------------
    else:
        best = None
        for pol in SAFE_CANDIDATES:
            try:
                ss, mm = _try_policy(pol)
                rows.append((pol, ss, mm))
                if (best is None) or (ss < best[0]):
                    best = (ss, pol, mm)
            except RuntimeError as e:
                if str(e) == "skip_left_mismatch":
                    logger.warning("[Stability] skip policy=%s reason=LEFT/H mismatch (L_left=%s, H=%s)", pol, L_left, H_pred)
                    continue
                logger.exception("[Stability] sweep_failed policy=%s", pol)
        if best is None:
            # fallback final e determinístico
            pol = "reconstruct"
            ss, mm = _try_policy(pol)
            rows.append((pol, ss, mm))
            best = (ss, pol, mm)
        chosen = best[1]

    # --- 5) Estatísticas (mean/std por membro) usando a policy escolhida -------
    # Observação: quando members=1, std=NaN (esperado)
    member_metrics = []
    for run in individual_run_outputs:
        pred = run.get("pred_val")
        if pred is None:
            continue
        pred_inv = _inv(pred)
        try:
            # reaplica somente a policy escolhida
            args = {"policy": chosen, "hp_lambda": float(params.get("hp_lambda", 8000.0))}
            if chosen in WARM_RECON_POLICIES:
                if warm_left_inv is None or L_left != H_pred:
                    raise RuntimeError("skip_left_mismatch")
                args["warm_left_windows_2d"] = warm_left_inv
            elif chosen.endswith("_warm") and (warm_left_inv is not None):
                args["warm_left_windows_2d"] = warm_left_inv

            agg_member = aggregate_predictions(pred_inv, **args)
            r2, smape, mae = evaluate(gt_1d, agg_member)
            member_metrics.append({"R²": r2, "SMAPE": smape, "MAE": mae})
        except RuntimeError as e:
            if str(e) == "skip_left_mismatch":
                # não deveria acontecer, pois já tratamos na escolha; apenas registra
                logger.warning("[Stability] member_skip reason=LEFT/H mismatch policy=%s", chosen)
            else:
                logger.exception("[Stability] member_eval_failed policy=%s", chosen)

    if not member_metrics:
        # se algo deu muito errado, ainda mostramos uma linha com a média da predição
        r2, smape, mae = evaluate(gt_1d, aggregate_predictions(mean_pred_inv, policy="reconstruct"))
        member_metrics = [{"R²": r2, "SMAPE": smape, "MAE": mae}]

    metrics_df = pd.DataFrame(member_metrics)
    stability_stats = metrics_df[["SMAPE", "MAE", "R²"]].agg(['mean', 'std'])

    # --- 6) Logging bonito + grava policy escolhida (não muda aggregation_method) ---
    logger.info("=" * 100)
    logger.info("INTERNAL STABILITY ANALYSIS REPORT (Validation Set)")
    logger.info("Based on %d independent runs within the ensemble.", len(individual_run_outputs))
    logger.info("Aggregation policy (stability): %s", chosen)
    logger.info("-" * 80)
    for line in stability_stats.to_string(float_format="%.4f").splitlines():
        logger.info(line)
    logger.info("=" * 100)

    # expõe para quem persistir o params (JSON): visibilidade + auditoria
    params.setdefault("aggregation_method_effective", chosen)
    params.setdefault("_stability", {})["chosen_aggregation_method_stability"] = chosen



# def process_chunks(train_kwargs: dict, data_inputs: dict, params: dict, scaler_target) -> dict:
#     """
#     Aggregate ensemble predictions across batches. Also aggregates *_left when provided
#     by the worker. Returns a dict compatible with `to_ensemble_output` (extras go to `.meta`).

#     NOTE:
#       - This version is latent-aware but remains behaviorally identical when
#         latent_mode == "off" (default). Any latent configuration is only logged
#         and propagated to the output dict (ends up in EnsembleOutput.meta).
#     """
#     import math
#     import numpy as np

#     logger = get_logger(__name__)
#     total = int(params["ensemble_models"])
#     chunk = min(1, total)  # keep legacy behavior
#     snaps = int(params.get("with_snapshots", 5))
#     retries, skip = 2, True

#     epochs      = int(params["epochs"])
#     batch_size  = int(params["batch_size"])
#     patience    = int(params["patience"])
#     learning_rate = float(params["learning_rate"])

#     arch = str(params.get("architecture_name", ""))
#     kind = str(params.get("feature_kind", ""))

#     # -------- latent / extrapolation context (scaffolding only) --------
#     latent_cfg = train_kwargs.get("latent_cfg") or {}
#     latent_mode = str(latent_cfg.get("mode", latent_cfg.get("latent_mode", "off"))).lower()

#     split_recon_lengths = train_kwargs.get("split_recon_lengths") or {}
#     if split_recon_lengths:
#         logger.info(
#             "latent_ctx(process_chunks) mode=%s split_recon_lengths val=%s test=%s",
#             latent_mode,
#             split_recon_lengths.get("val"),
#             split_recon_lengths.get("test"),
#         )
#     else:
#         logger.info("latent_ctx(process_chunks) mode=%s split_recon_lengths=None", latent_mode)

#     # accumulators --------------------------------------------------------------
#     sum_test_preds = sum_val_preds = None
#     sum_val_left = sum_test_left = None
#     sum_qs = sum_res = None
#     sum_sigma_test = sum_sigma_val = None
#     sum_sigma_val_left = sum_sigma_test_left = None
#     sum_alpha = 0.0
#     total_models = 0
#     can_analyze = True
#     individual_run_outputs = []

#     with log_context(method=arch, kind=kind, total=total, chunk=chunk, snaps=snaps):
#         with phase(logger, "process_chunks", epochs=epochs, batch=batch_size, lr=learning_rate):
#             num_batches = math.ceil(total / chunk)
#             logger.info("plan batches=%d", num_batches)

#             for b in range(num_batches):
#                 size = min(chunk, total - b * chunk)
#                 with phase(logger, "batch", idx=b + 1, of=num_batches, size=size):
#                     chunk_out = train_predict_chunk(
#                         main_train_model,
#                         params["architecture_name"],
#                         params["feature_kind"],
#                         train_kwargs,
#                         data_inputs,
#                         size,
#                         snaps,
#                         epochs,
#                         batch_size,
#                         patience,
#                         learning_rate,
#                         retries,
#                         skip,
#                     )

#                     n = int(chunk_out.get("successful_models", 0))
#                     if n == 0:
#                         logger.warning("skip reason=no_successful_models")
#                         continue

#                     total_models += n

#                     # keep only what we need for stability analysis (val + optional LEFT)
#                     individual_run_outputs.append({
#                         "pred_val": chunk_out.get("pred_val"),
#                         "pred_val_left": chunk_out.get("pred_val_left"),  # <-- novo (se existir)
#                     })

#                     # initialize sums lazily (using first successful shapes)
#                     if sum_test_preds is None:
#                         sum_test_preds = np.zeros_like(chunk_out["pred_test"], dtype=np.float64)
#                         sum_val_preds  = np.zeros_like(chunk_out["pred_val"],  dtype=np.float64)

#                     # main predictions
#                     sum_test_preds += chunk_out["pred_test"] * n
#                     sum_val_preds  += chunk_out["pred_val"]  * n

#                     # *_left ribbons (optional)
#                     if chunk_out.get("pred_val_left") is not None:
#                         if sum_val_left is None:
#                             sum_val_left = np.zeros_like(chunk_out["pred_val_left"], dtype=np.float64)
#                         sum_val_left += chunk_out["pred_val_left"] * n

#                     if chunk_out.get("pred_test_left") is not None:
#                         if sum_test_left is None:
#                             sum_test_left = np.zeros_like(chunk_out["pred_test_left"], dtype=np.float64)
#                         sum_test_left += chunk_out["pred_test_left"] * n

#                     # contributions / residuals / alpha (optional)
#                     if chunk_out.get("q_phys") is not None:
#                         if sum_qs is None:
#                             sum_qs = np.zeros_like(chunk_out["q_phys"], dtype=np.float64)
#                         sum_qs += chunk_out["q_phys"] * n
#                     else:
#                         can_analyze = False

#                     if chunk_out.get("res_val") is not None:
#                         if sum_res is None:
#                             sum_res = np.zeros_like(chunk_out["res_val"], dtype=np.float64)
#                         sum_res += chunk_out["res_val"] * n
#                     else:
#                         can_analyze = False

#                     if chunk_out.get("alpha_val") is not None:
#                         sum_alpha += float(chunk_out["alpha_val"]) * n
#                     else:
#                         can_analyze = False

#                     # uncertainty ribbons (optional)
#                     if "sigma_test" in chunk_out:
#                         if sum_sigma_test is None:
#                             sum_sigma_test = np.zeros_like(chunk_out["sigma_test"], dtype=np.float64)
#                             sum_sigma_val  = np.zeros_like(chunk_out["sigma_val"],  dtype=np.float64)
#                         sum_sigma_test += chunk_out["sigma_test"] * n
#                         sum_sigma_val  += chunk_out["sigma_val"]  * n

#                     if "sigma_val_left" in chunk_out:
#                         if sum_sigma_val_left is None:
#                             sum_sigma_val_left = np.zeros_like(chunk_out["sigma_val_left"], dtype=np.float64)
#                         sum_sigma_val_left += chunk_out["sigma_val_left"] * n

#                     if "sigma_test_left" in chunk_out:
#                         if sum_sigma_test_left is None:
#                             sum_sigma_test_left = np.zeros_like(chunk_out["sigma_test_left"], dtype=np.float64)
#                         sum_sigma_test_left += chunk_out["sigma_test_left"] * n

#                     logger.info("progress total_models=%d/%d", total_models, total)

#             if total_models == 0:
#                 raise RuntimeError("No models processed in any chunk")

#             # Stability analysis (validation-only) --------------------------------
#             y_val_scaled = train_kwargs.get("y_val")
#             # if y_val_scaled is not None:
#             #     logger.info("stability_analysis=run members=%d", len(individual_run_outputs))
#             #     _analyze_ensemble_stability(
#             #         individual_run_outputs=individual_run_outputs,
#             #         y_val_scaled=y_val_scaled,
#             #         scaler_target=scaler_target,
#             #         params=params,
#             #     )
#             # else:
#             #     logger.info("stability_analysis=skip reason=missing_y_val")

#             # Build outputs (averages) --------------------------------------------
#             out_dict = {
#                 "pred_test": (sum_test_preds / total_models),
#                 "pred_val":  (sum_val_preds  / total_models),
#             }
#             if sum_val_left  is not None: out_dict["pred_val_left"]  = (sum_val_left  / total_models)
#             if sum_test_left is not None: out_dict["pred_test_left"] = (sum_test_left / total_models)
#             if sum_sigma_test is not None:
#                 out_dict["sigma_test"] = (sum_sigma_test / total_models)
#                 out_dict["sigma_val"]  = (sum_sigma_val  / total_models)
#             if sum_sigma_val_left  is not None: out_dict["sigma_val_left"]  = (sum_sigma_val_left  / total_models)
#             if sum_sigma_test_left is not None: out_dict["sigma_test_left"] = (sum_sigma_test_left / total_models)

#             # Optional contribution analysis (test only) --------------------------
#             if can_analyze:
#                 final_qs    = (sum_qs    / total_models)
#                 final_res   = (sum_res   / total_models)
#                 final_alpha = (sum_alpha / total_models)
#                 if arch.startswith("Seq2"):
#                     L = int(params.get("horizon", final_qs.shape[1] if final_qs is not None else 0))
#                     out_dict["pred_test"] = out_dict["pred_test"][:, :L]
#                     final_qs  = final_qs[:, :L]
#                     final_res = final_res[:, :L]
#                 logger.info("contrib_analysis=run")
#                 analyze_contributions(Qs=final_qs, res=final_res, alpha=final_alpha, scaler_target=scaler_target)
#             else:
#                 logger.info("contrib_analysis=skip")

#             # -------- propagate latent context to EnsembleOutput.meta ------------
#             if latent_cfg:
#                 out_dict["latent_cfg"] = latent_cfg
#             if split_recon_lengths:
#                 out_dict["split_recon_lengths"] = split_recon_lengths

#             # Closing note (keys returned) ----------------------------------------
#             logger.info("return keys=%s", sorted(list(out_dict.keys())))
#             return out_dict



def process_chunks(train_kwargs: dict, data_inputs: dict, params: dict, scaler_target) -> dict:
    """
    Aggregate ensemble predictions across batches. Also aggregates *_left when provided
    by the worker. Returns a dict compatible with `to_ensemble_output` (extras go to `.meta`).

    Plug-and-play refactor:
      - Keeps legacy behavior for preds/left/sigma/contribs.
      - Promotes integrated-view spaghetti members ONLY when they are explicitly *_scaled,
        preventing "double inverse_transform" downstream.
      - Still searches both top-level and out["meta"] for members/meta.
      - Avoids numpy truthiness bugs (never uses `a or b` when `a` may be an array).
    """
    import math
    import numpy as np

    logger = get_logger(__name__)

    # Reuse your compact UI
    from common.log_utils import (
        stage_banner,
        log_kv_block,
        effective_log_width,
        is_compact_logging,
        arr_fingerprint,
        info_v,
    )

    width = effective_log_width(None, fallback=100)
    compact = is_compact_logging(None)

    total = int(params["ensemble_models"])
    chunk = min(1, total)  # keep legacy behavior
    snaps = int(params.get("with_snapshots", 5))
    retries, skip = 2, True

    epochs = int(params["epochs"])
    batch_size = int(params["batch_size"])
    patience = int(params["patience"])
    learning_rate = float(params["learning_rate"])

    arch = str(params.get("architecture_name", ""))
    kind = str(params.get("feature_kind", ""))

    # -------- latent / extrapolation context (scaffolding only) --------
    latent_cfg = train_kwargs.get("latent_cfg") or {}
    latent_mode = str(latent_cfg.get("mode", latent_cfg.get("latent_mode", "off"))).lower()

    split_recon_lengths = train_kwargs.get("split_recon_lengths") or {}

    # One banner per call
    stage_banner(
        "AGG",
        "process_chunks",
        f"arch={arch} kind={kind} total_models={total} chunk={chunk} snapshots={snaps}",
        width=width,
    )

    # Compact enter block
    enter_kv = {
        "latent_mode": latent_mode,
        "split_recon_lengths": {
            "val": split_recon_lengths.get("val"),
            "test": split_recon_lengths.get("test"),
        } if split_recon_lengths else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "learning_rate": learning_rate,
        "with_snapshots": snaps,
    }
    log_kv_block("Process Chunks — Enter", enter_kv, width=width)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _first_not_none(*xs):
        for x in xs:
            if x is not None:
                return x
        return None

    def _as_meta(d: dict) -> dict:
        m = d.get("meta", None) if isinstance(d, dict) else None
        return m if isinstance(m, dict) else {}

    def _pick_members_scaled_anywhere(d: dict, split: str):
        """
        split: "val" or "test"
        IMPORTANT: returns ONLY *scaled* members. Do NOT accept keys without '_scaled'.
        This prevents downstream inverse_transform from being applied twice.
        """
        meta = _as_meta(d)

        if split == "test":
            return _first_not_none(
                d.get("integrated_view_test_members_scaled"),
                d.get("pred_test_members_scaled"),          # optional/future
                d.get("pred_members_scaled"),               # generic scaled
                meta.get("integrated_view_test_members_scaled"),
                meta.get("pred_test_members_scaled"),
                meta.get("pred_members_scaled"),
            )

        # split == "val"
        return _first_not_none(
            d.get("integrated_view_val_members_scaled"),
            d.get("pred_val_members_scaled"),             # optional/future
            d.get("pred_members_val_scaled"),             # legacy-but-explicit scaled
            d.get("pred_members_scaled"),                 # last-resort generic scaled
            meta.get("integrated_view_val_members_scaled"),
            meta.get("pred_val_members_scaled"),
            meta.get("pred_members_val_scaled"),
            meta.get("pred_members_scaled"),
        )

    def _pick_members_meta_anywhere(d: dict, split: str) -> dict:
        meta = _as_meta(d)

        if split == "test":
            m = _first_not_none(
                d.get("integrated_view_test_members_meta"),
                d.get("pred_test_members_meta"),
                d.get("pred_members_meta"),
                meta.get("integrated_view_test_members_meta"),
                meta.get("pred_test_members_meta"),
                meta.get("pred_members_meta"),
            )
            return dict(m or {})

        m = _first_not_none(
            d.get("integrated_view_val_members_meta"),
            d.get("pred_val_members_meta"),
            d.get("pred_members_val_meta"),
            meta.get("integrated_view_val_members_meta"),
            meta.get("pred_val_members_meta"),
            meta.get("pred_members_val_meta"),
        )
        return dict(m or {})

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    sum_test_preds = sum_val_preds = None
    sum_val_left = sum_test_left = None
    sum_qs = sum_res = None
    sum_sigma_test = sum_sigma_val = None
    sum_sigma_val_left = sum_sigma_test_left = None
    sum_alpha = 0.0
    total_models = 0
    can_analyze = True
    individual_run_outputs = []

    # integrated-view spaghetti payload (visual-only)
    iv_val_members_scaled = None
    iv_test_members_scaled = None
    iv_val_members_meta = None
    iv_test_members_meta = None

    with log_context(method=arch, kind=kind, total=total, chunk=chunk, snaps=snaps):
        with phase(logger, "process_chunks", epochs=epochs, batch=batch_size, lr=learning_rate):
            num_batches = math.ceil(total / chunk)

            # Keep plan info, but make it compact-friendly
            log_kv_block("Batch Plan", {"num_batches": num_batches, "chunk_size": chunk, "total_models": total}, width=width)

            for b in range(num_batches):
                size = min(chunk, total - b * chunk)
                with phase(logger, "batch", idx=b + 1, of=num_batches, size=size):
                    chunk_out = train_predict_chunk(
                        main_train_model,
                        params["architecture_name"],
                        params["feature_kind"],
                        train_kwargs,
                        data_inputs,
                        size,
                        snaps,
                        epochs,
                        batch_size,
                        patience,
                        learning_rate,
                        retries,
                        skip,
                    )

                    n = int(chunk_out.get("successful_models", 0))
                    if n == 0:
                        logger.warning("batch_skip reason=no_successful_models batch=%d/%d", b + 1, num_batches)
                        continue

                    total_models += n

                    # Capture integrated-view spaghetti members from worker (first non-empty)
                    if iv_test_members_scaled is None:
                        iv_test_members_scaled = _pick_members_scaled_anywhere(chunk_out, "test")
                        iv_test_members_meta = _pick_members_meta_anywhere(chunk_out, "test")

                    if iv_val_members_scaled is None:
                        iv_val_members_scaled = _pick_members_scaled_anywhere(chunk_out, "val")
                        iv_val_members_meta = _pick_members_meta_anywhere(chunk_out, "val")

                    # keep only what we need for stability analysis (val + optional LEFT)
                    individual_run_outputs.append(
                        {
                            "pred_val": chunk_out.get("pred_val"),
                            "pred_val_left": chunk_out.get("pred_val_left"),
                        }
                    )

                    # initialize sums lazily (using first successful shapes)
                    if sum_test_preds is None:
                        sum_test_preds = np.zeros_like(chunk_out["pred_test"], dtype=np.float64)
                        sum_val_preds = np.zeros_like(chunk_out["pred_val"], dtype=np.float64)

                    # main predictions
                    sum_test_preds += chunk_out["pred_test"] * n
                    sum_val_preds += chunk_out["pred_val"] * n

                    # *_left ribbons (optional)
                    if chunk_out.get("pred_val_left") is not None:
                        if sum_val_left is None:
                            sum_val_left = np.zeros_like(chunk_out["pred_val_left"], dtype=np.float64)
                        sum_val_left += chunk_out["pred_val_left"] * n

                    if chunk_out.get("pred_test_left") is not None:
                        if sum_test_left is None:
                            sum_test_left = np.zeros_like(chunk_out["pred_test_left"], dtype=np.float64)
                        sum_test_left += chunk_out["pred_test_left"] * n

                    # contributions / residuals / alpha (optional)
                    if chunk_out.get("q_phys") is not None:
                        if sum_qs is None:
                            sum_qs = np.zeros_like(chunk_out["q_phys"], dtype=np.float64)
                        sum_qs += chunk_out["q_phys"] * n
                    else:
                        can_analyze = False

                    if chunk_out.get("res_val") is not None:
                        if sum_res is None:
                            sum_res = np.zeros_like(chunk_out["res_val"], dtype=np.float64)
                        sum_res += chunk_out["res_val"] * n
                    else:
                        can_analyze = False

                    if chunk_out.get("alpha_val") is not None:
                        sum_alpha += float(chunk_out["alpha_val"]) * n
                    else:
                        can_analyze = False

                    # uncertainty ribbons (optional)
                    if "sigma_test" in chunk_out:
                        if sum_sigma_test is None:
                            sum_sigma_test = np.zeros_like(chunk_out["sigma_test"], dtype=np.float64)
                            sum_sigma_val = np.zeros_like(chunk_out["sigma_val"], dtype=np.float64)
                        sum_sigma_test += chunk_out["sigma_test"] * n
                        sum_sigma_val += chunk_out["sigma_val"] * n

                    if "sigma_val_left" in chunk_out:
                        if sum_sigma_val_left is None:
                            sum_sigma_val_left = np.zeros_like(chunk_out["sigma_val_left"], dtype=np.float64)
                        sum_sigma_val_left += chunk_out["sigma_val_left"] * n

                    if "sigma_test_left" in chunk_out:
                        if sum_sigma_test_left is None:
                            sum_sigma_test_left = np.zeros_like(chunk_out["sigma_test_left"], dtype=np.float64)
                        sum_sigma_test_left += chunk_out["sigma_test_left"] * n

                    # Keep progress line (but make it compact-friendly)
                    if compact:
                        logger.info("progress models=%d/%d (batch=%d/%d)", total_models, total, b + 1, num_batches)
                    else:
                        logger.info("progress total_models=%d/%d", total_models, total)

            if total_models == 0:
                raise RuntimeError("No models processed in any chunk")

            # Stability analysis (validation-only) --------------------------------
            # y_val_scaled = train_kwargs.get("y_val")
            # if y_val_scaled is not None:
            #     logger.info("stability_analysis=run members=%d", len(individual_run_outputs))
            #     _analyze_ensemble_stability(
            #         individual_run_outputs=individual_run_outputs,
            #         y_val_scaled=y_val_scaled,
            #         scaler_target=scaler_target,
            #         params=params,
            #     )
            # else:
            #     logger.info("stability_analysis=skip reason=missing_y_val")

            # Build outputs (averages) --------------------------------------------
            out_dict = {
                "pred_test": (sum_test_preds / total_models),
                "pred_val": (sum_val_preds / total_models),
            }
            if sum_val_left is not None:
                out_dict["pred_val_left"] = (sum_val_left / total_models)
            if sum_test_left is not None:
                out_dict["pred_test_left"] = (sum_test_left / total_models)
            if sum_sigma_test is not None:
                out_dict["sigma_test"] = (sum_sigma_test / total_models)
                out_dict["sigma_val"] = (sum_sigma_val / total_models)
            if sum_sigma_val_left is not None:
                out_dict["sigma_val_left"] = (sum_sigma_val_left / total_models)
            if sum_sigma_test_left is not None:
                out_dict["sigma_test_left"] = (sum_sigma_test_left / total_models)

            # Optional contribution analysis (test only) --------------------------
            if can_analyze:
                final_qs = (sum_qs / total_models)
                final_res = (sum_res / total_models)
                final_alpha = (sum_alpha / total_models)
                if arch.startswith("Seq2"):
                    L = int(params.get("horizon", final_qs.shape[1] if final_qs is not None else 0))
                    out_dict["pred_test"] = out_dict["pred_test"][:, :L]
                    final_qs = final_qs[:, :L]
                    final_res = final_res[:, :L]
                logger.info("contrib_analysis=run")
                analyze_contributions(Qs=final_qs, res=final_res, alpha=final_alpha, scaler_target=scaler_target)
            else:
                logger.info("contrib_analysis=skip")

            # Promote integrated-view members (top-level payload for downstream plotting)
            if iv_test_members_scaled is not None:
                out_dict["integrated_view_test_members_scaled"] = iv_test_members_scaled
                if iv_test_members_meta is not None:
                    out_dict["integrated_view_test_members_meta"] = iv_test_members_meta

            if iv_val_members_scaled is not None:
                out_dict["integrated_view_val_members_scaled"] = iv_val_members_scaled
                if iv_val_members_meta is not None:
                    out_dict["integrated_view_val_members_meta"] = iv_val_members_meta

            # -------- propagate latent context to EnsembleOutput.meta ------------
            if latent_cfg:
                out_dict["latent_cfg"] = latent_cfg
            if split_recon_lengths:
                out_dict["split_recon_lengths"] = split_recon_lengths

            # Compact exit summary (instead of "return keys=..." spam)
            promoted_shapes = {
                "val_members": None if iv_val_members_scaled is None else tuple(np.asarray(iv_val_members_scaled).shape),
                "test_members": None if iv_test_members_scaled is None else tuple(np.asarray(iv_test_members_scaled).shape),
            }
            exit_kv = {
                "successful_models": int(total_models),
                "pred_val": arr_fingerprint(out_dict.get("pred_val")),
                "pred_test": arr_fingerprint(out_dict.get("pred_test")),
                "has_val_left": bool(out_dict.get("pred_val_left") is not None),
                "has_test_left": bool(out_dict.get("pred_test_left") is not None),
                "promoted_members": promoted_shapes,
                "latent_mode": latent_mode,
                "split_recon_lengths": {
                    "val": split_recon_lengths.get("val"),
                    "test": split_recon_lengths.get("test"),
                } if split_recon_lengths else None,
            }
            log_kv_block("Process Chunks — Summary", exit_kv, width=width)

            # Keep keys available, but only in verbose (or debug)
            info_v("return keys=%s", sorted(list(out_dict.keys())))

            return out_dict





def normalize_job_parameters(initial_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes a job's parameter dictionary and ensures it has the final,
    unified structure needed by the entire pipeline.
    
    This function is now fully ARCHITECTURE-AWARE.
    """
    params = initial_params.copy()

    if "architecture_name" not in params:
        params["architecture_name"] = "Seq2Context" 
        logging.warning("'architecture_name' not in params, falling back to 'Seq2Context'.")
    
    arch_name = params["architecture_name"]

    # This check is ONLY valid for the Seq2* family.
    if arch_name in ["Seq2Context", "Seq2Fuser"]:
        # Ensure `strategy_config` exists for hybrid models.
        if "strategy_config" not in params:
            if "physics_strategy" not in params:
                raise KeyError(f"Job parameters for '{arch_name}' must contain 'physics_strategy'.")
            params["strategy_config"] = {"strategy_name": params["physics_strategy"]}

        # Conditionally ensure `extractor_config` and `fuser_config` exist.
        if arch_name in ["Seq2Context", "Seq2Fuser"]:
            if "extractor_config" not in params:
                raise KeyError(f"Job parameters for '{arch_name}' are missing 'extractor_config'.")
            if "fuser_config" not in params:
                raise KeyError(f"Job parameters for '{arch_name}' are missing 'fuser_config'.")

    elif arch_name.startswith("Darts_"):
        # Darts models do not need any of these specific checks.
        # We can add Darts-specific normalization here in the future if needed.
        pass # It's a valid Darts job, so no error.
        
    else:
        pass
    
    return params


def _maybe_parse_dictlike(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return {}
    # tenta python-literal primeiro (perfil CSV costuma ter "{'a': 1}")
    try:
        obj = ast.literal_eval(s)
        return obj
    except Exception:
        pass
    # tenta YAML/JSON como fallback (aceita "{a: 1}" ou '{"a":1}')
    try:
        import yaml
        obj = yaml.safe_load(s)
        return obj
    except Exception:
        return v  # mantém como string mesmo


def _coerce_known_nested_blocks(params: Dict[str, Any]) -> Dict[str, Any]:
    # apenas chaves “perigosas” que podem virar string no CSV
    DICTLIKE_KEYS = {
        "arps_ensemble",
        "latent_cfg",
        "arps",
        "theta_sampling_clip",
        "series_store",
    }
    out = dict(params)
    for k in DICTLIKE_KEYS:
        if k in out:
            parsed = _maybe_parse_dictlike(out[k])
            if isinstance(parsed, dict):
                out[k] = parsed
    return out


def run_single_job(job: Tuple, persist_result: bool = False) -> Dict[str, Any]:
    """
    Orchestrate a single job (Seq2* or Darts). If `persist_result=True`, write JSON
    and return a compact status dict (legacy-compatible).
    """
    ds, well, params, experiment_id = job
    # ✅ normalize params ONCE for ALL kinds (safe, because it only fixes dict-like strings)
    if isinstance(params, dict):
        params = _coerce_known_nested_blocks(params)
        job = (ds, well, params, experiment_id)
    arch = params.get("architecture_name", "")
    job_hash = generate_job_hash(params)
    logger = get_logger(__name__)

    with log_context(well=well, exp_id=experiment_id, arch=arch, job_hash=job_hash, persist=persist_result):
        with phase(logger, "run_single_job"):
            if isinstance(arch, str) and arch.startswith("Arps_"):
                kind = "arps"
            elif isinstance(arch, str) and arch.startswith("Darts_"):
                kind = "darts"
            else:
                kind = "seq2"
            logger.info("decision=path kind=%s", kind)

            if kind == "arps":
                result = _run_single_job_arps(job)
            elif kind == "darts":
                result = _run_single_job_darts(job)
            else:
                result = _run_single_job_seq(job)

            if persist_result:
                return _persist_if_requested(
                    result=result,
                    params=params,
                    job_hash=job_hash,
                    well=well,
                    experiment_id=experiment_id,
                )
            return result

    try:
        gc.collect()
    finally:
        return _failure_result(well=well, experiment_id=experiment_id)




def _inject_left_sidecar(train_kwargs: dict, params: dict, scaler_target) -> None:
    """
    Prepare LEFT windows (already scaled) and inject into `scaler_target._split_ctx`.

    - Validation: LEFT = last K windows from X_train
    - Test:       LEFT = last K windows from X_val
    - K = min(horizon, available_windows)
    """
    import numpy as np

    logger = get_logger(__name__)

    X_train = train_kwargs.get("X_train")
    X_val   = train_kwargs.get("X_val")

    if scaler_target is None:
        logger.info("left_sidecar=skip reason=scaler_target_none")
        return

    H = int(params.get("horizon", 1))

    with log_context(horizon=H):
        with phase(logger, "inject_left_sidecar"):
            # ensure sidecar dict exists
            ctx = getattr(scaler_target, "_split_ctx", None)
            if ctx is None:
                ctx = {}
                setattr(scaler_target, "_split_ctx", ctx)

            # Validation LEFT
            if isinstance(X_train, np.ndarray) and X_train.size > 0:
                k_val = min(H, X_train.shape[0])
                ctx["X_val_left_scaled"] = X_train[-k_val:].copy() if k_val > 0 else None
                val_len = int(k_val if k_val > 0 else 0)
            else:
                ctx["X_val_left_scaled"] = None
                val_len = 0

            # Test LEFT
            if isinstance(X_val, np.ndarray) and X_val.size > 0:
                k_test = min(H, X_val.shape[0])
                ctx["X_test_left_scaled"] = X_val[-k_test:].copy() if k_test > 0 else None
                test_len = int(k_test if k_test > 0 else 0)
            else:
                ctx["X_test_left_scaled"] = None
                test_len = 0

            logger.info(
                "left_sidecar prepared val_len=%d test_len=%d "
                "X_train_shape=%s X_val_shape=%s",
                val_len,
                test_len,
                getattr(X_train, "shape", None),
                getattr(X_val, "shape", None),
            )


# Reuse the shared glue (tolerant unpack, self-heal context, job_meta, non-fatal writer)
from common.eval_compat import (
    unpack_eval7,
    attach_series_context,
    ensure_job_meta,
    maybe_persist_series,
)

# ---------------------------------------------------------------------
# ARPS runner
# ---------------------------------------------------------------------


def _run_single_job_arps(job: Tuple) -> Dict[str, Any]:
    """
    Canonical ARPS path (compatible with new evaluate_job 7-item return; tolerant to legacy 6).

    Plug-and-play version with pragmatic, boundary-correct q0 anchoring for ARPS spaghetti:
      - DOES NOT modify compute_history_q0_phys
      - Feeds ONLY the LAST y_train window as X_any=(H,1) so compute_history_q0_phys anchors at train boundary
      - Uses scaler_target as scaler_X (pragmatic hack that matches your working experiment)
      - Keeps ensemble_out optional and non-breaking

    Notes:
      - This MUST NOT affect seq2 offline_analytic coupling flows.
      - Ensemble logic is triggered only if params["arps_ensemble"]["enabled"]=True (or legacy shorthand).
    """
    ds, well, params, experiment_id = job
    logger = get_logger(__name__, context={"well": well, "method": params.get("architecture_name", "Arps")})

    from common.seq_preprocessing import reconstruct_true_series

    # ---------------------------
    # 0) Resolve ARPS ensemble config (pure-ARPS only)
    # ---------------------------
    arps_ens_cfg: Dict[str, Any] = {}
    try:
        arps_ens_cfg = (params.get("arps_ensemble") or params.get("ensemble_arps") or {}) if isinstance(params, dict) else {}
    except Exception:
        arps_ens_cfg = {}

    # allow shorthand: params["arps_ensemble"] = True
    if isinstance(arps_ens_cfg, bool):
        arps_ens_cfg = {"enabled": bool(arps_ens_cfg)}

    enabled_ens = bool(arps_ens_cfg.get("enabled", False))
    emit_members = bool(arps_ens_cfg.get("emit_members", params.get("plot", False)))

    # ---------------------------
    # 1) Load & prepare
    # ---------------------------
    with phase(logger, "ARPS_Canonical.load_and_prepare"):
        exp = ExperimentArps(config=ds, well=well, params=params, exp_id=experiment_id)
        (train_kwargs, X_test, y_test_scaled_win, scaler_X, scaler_target, y_train_original) = exp.load_and_prepare()

        # build x_ctx in PHYSICAL units (target channel from X_train)
        x_ctx = _extract_unscaled_target_windows(
            X_train_scaled=train_kwargs["X_train"],
            y_train_windows=y_train_original,
            scaler_target=scaler_target,
        )

        # Point-forecast contract expects 1D arrays for val/test
        y_val_scaled_win = train_kwargs["y_val"]
        y_val_scaled_1d  = reconstruct_true_series(y_val_scaled_win).astype(float).reshape(-1)
        y_test_scaled_1d = reconstruct_true_series(y_test_scaled_win).astype(float).reshape(-1)

        # Physical-domain train series for ARPS fitting
        train_series_phys = _reconstruct_train_series_phys(y_train_original, scaler_target).reshape(-1)
        L_train = int(train_series_phys.size)
        L_val   = int(y_val_scaled_1d.size)
        L_test  = int(y_test_scaled_1d.size)

        if L_train <= 0 or L_val <= 0 or L_test <= 0:
            raise ValueError(f"ARPS_Canonical invalid lengths: L_train={L_train} L_val={L_val} L_test={L_test}")

    # ---------------------------
    # 2) Fit ARPS (build kwargs, log digest)
    # ---------------------------
    with phase(logger, "ARPS_Canonical.fit"):
        arps_subcfg = params.get("arps", {}) or {}
        raw_kwargs  = {**params, **arps_subcfg}
        arps_kwargs = _adapt_arps_kwargs_for_fit(fit_arps_canonical, raw_kwargs)

        def _digest(kwargs: Dict[str, Any]) -> Dict[str, Any]:
            d = dict(kwargs)
            bg = d.pop("b_grid", None)
            if isinstance(bg, np.ndarray):
                d["b_grid_digest"] = {"len": int(bg.size), "min": float(bg.min()), "max": float(bg.max())}
            return d

        logger.info("ARPS.fit kwargs=%s", _digest(arps_kwargs))

        theta: ArpsParams = fit_arps_canonical(q_train_phys=train_series_phys, **arps_kwargs)
        logger.info(
            "arps_params variant=%s qi=%.6f D=%.6e b=%.4f piecewise=%s cp_index=%s",
            getattr(theta, "variant", "—"),
            getattr(theta, "qi", float("nan")),
            getattr(theta, "D", float("nan")),
            getattr(theta, "b", float("nan")),
            getattr(theta, "piecewise", False),
            getattr(theta, "cp_index", None),
        )

    # ---------------------------
    # 3) Forecast (single curve) & optional ensemble (members/bands) + rescale
    # ---------------------------
    with phase(logger, "ARPS_Canonical.forecast"):
        # ---- base point forecast (physical)
        yhat_val_phys, yhat_test_phys = forecast_canonical_from_train(theta, L_train, L_val, L_test)

        def _scale_1d(scaler, arr_1d):
            return scaler.transform(np.asarray(arr_1d, dtype=float).reshape(-1, 1)).reshape(-1)

        y_val_pred_scaled  = _scale_1d(scaler_target, yhat_val_phys).reshape(-1)
        y_test_pred_scaled = _scale_1d(scaler_target, yhat_test_phys).reshape(-1)

        # ---- optional ensemble generation (pure ARPS path)
        ensemble_out: Optional[Dict[str, Any]] = None
        if enabled_ens:
            try:
                # ----------------------------
                # PRAGMATIC FIX:
                # Feed ONLY the LAST y_train window as (H,1), so compute_history_q0_phys
                # (which uses extract_first_window) anchors at the TRAIN boundary.
                # Also, use scaler_target as scaler_X (matches your working experiment).
                # ----------------------------
                ytr = train_kwargs.get("y_train", None)
                X_q0 = None
                if ytr is not None:
                    ytr = np.asarray(ytr)
                    if ytr.ndim == 2 and ytr.shape[0] >= 1:
                        X_q0 = ytr[-1].reshape(-1, 1)  # (H,1)
                    elif ytr.ndim == 1:
                        X_q0 = ytr.reshape(-1, 1)

                ensemble_out = build_arps_ensemble_out_from_theta_hat(
                    theta_hat=theta,
                    train_len=int(L_train),
                    val_len=int(L_val),
                    test_len=int(L_test),
                    scaler_target=scaler_target,
                    cfg=arps_ens_cfg,
                    logger=logger,
                    X_any=X_q0,
                    scaler_X=scaler_target,
                )

                # If requested, override point forecast with ensemble aggregate (p50/median)
                # But keep it non-breaking: only if keys exist.
                if isinstance(ensemble_out, dict):
                    val_agg = ensemble_out.get("val", {}).get("agg_scaled", None)
                    test_agg = ensemble_out.get("test", {}).get("agg_scaled", None)
                    if val_agg is not None:
                        y_val_pred_scaled = np.asarray(val_agg, dtype=float).reshape(-1)
                    if test_agg is not None:
                        y_test_pred_scaled = np.asarray(test_agg, dtype=float).reshape(-1)

                    # optionally drop members unless explicitly requested
                    if not emit_members:
                        for sp in ("val", "test"):
                            if isinstance(ensemble_out.get(sp), dict):
                                ensemble_out[sp].pop("members_scaled", None)

            except Exception as ex:
                logger.exception("ARPS ensemble build failed; falling back to single-curve. (%s)", str(ex))
                ensemble_out = None

        # Defensive: lengths must match targets for evaluate_job
        if int(y_val_pred_scaled.size) != int(L_val):
            raise ValueError(f"ARPS_Canonical: y_val_pred_scaled.size={y_val_pred_scaled.size} != L_val={L_val}")
        if int(y_test_pred_scaled.size) != int(L_test):
            raise ValueError(f"ARPS_Canonical: y_test_pred_scaled.size={y_test_pred_scaled.size} != L_test={L_test}")

    # ---------------------------
    # 4) Evaluate (tolerant to (6|7) items)
    # ---------------------------
    with phase(logger, "ARPS_Canonical.evaluate"):
        eval_ret = evaluate_job(
            y_test_scaled=y_test_scaled_1d,
            y_test_pred=y_test_pred_scaled,
            y_val_scaled=y_val_scaled_1d,
            y_val_pred=y_val_pred_scaled,
            scaler_target=scaler_target,
            y_train_original=y_train_original,
            params=params,
            config=ds,
            well=well,
            plot=params.get("plot", False),
            ensemble_out=ensemble_out,  # may be None
            x_train_main_windows=train_kwargs["X_train"],
        )
        (agg_test_df, cum_test_df, gm_test,
         agg_val_df,  cum_val_df,  gm_val,
         series_artifacts) = unpack_eval7(eval_ret)

    # ---------------------------
    # 5) Assemble result + embed context + optional write
    # ---------------------------
    result_dict = _assemble_result_dict(
        well=well,
        experiment_id=experiment_id,
        agg_test_df=agg_test_df, cum_test_df=cum_test_df, gm_test=gm_test,
        agg_val_df=agg_val_df,   cum_val_df=cum_val_df,   gm_val=gm_val,
        extra_meta={
            "arps_params": {
                "variant": getattr(theta, "variant", None),
                "qi": getattr(theta, "qi", None),
                "D": getattr(theta, "D", None),
                "b": getattr(theta, "b", None),
                "piecewise": getattr(theta, "piecewise", False),
                "cp_index": getattr(theta, "cp_index", None),
            },
            "arps_fit_kwargs_effective": {k: v for k, v in _digest(arps_kwargs).items()},
            "arps_ensemble_cfg_effective": (dict(arps_ens_cfg) if isinstance(arps_ens_cfg, dict) else None),
        },
    )

    y_train_full = reconstruct_train_series_with_X_prefix(y_train_original, x_ctx)

    attach_series_context(
        result_dict,
        scaler_target,
        series_artifacts,
        y_train_original=y_train_full,
    )
    ensure_job_meta(result_dict, params, ds, well, job_hash=generate_job_hash(params))
    maybe_persist_series(result_dict, params.get("series_store") or {}, logger)

    return result_dict



# ---------------------------------------------------------------------
# Seq2* runner
# ---------------------------------------------------------------------

def _build_latent_cfg_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a stable latent configuration WITHOUT clobbering preset knobs.

    Rules:
      1) Start from params["latent_cfg"] if present (preset-driven).
      2) If params["latent_mode"] is provided, it becomes the authoritative cfg["mode"].
      3) Preserve any existing ARPS/offline_analytic knobs (e.g., arps_coupling_mode).
      4) Attach conservative options only if provided, without overwriting other fields.
    """
    # 1) Start from preset-driven latent_cfg (if any)
    from copy import deepcopy
    cfg: Dict[str, Any] = deepcopy(params.get("latent_cfg") or {})

    # 2) Determine mode (high-level switch wins if present)
    raw_mode = params.get("latent_mode", None)
    if raw_mode is not None and str(raw_mode).strip() != "":
        cfg["mode"] = str(raw_mode).strip().lower()
    else:
        # Backward-compat: allow latent_mode inside cfg
        if "mode" not in cfg and "latent_mode" in cfg:
            cfg["mode"] = str(cfg.get("latent_mode") or "off").strip().lower()

    cfg.setdefault("mode", "off")

    # 3) Optional conservative block (non-destructive)
    cons_in_cfg = cfg.get("conservative")
    conservative: Dict[str, Any] = dict(cons_in_cfg) if isinstance(cons_in_cfg, dict) else {}

    if "latent_conservative_horizon" in params:
        conservative["horizon"] = params.get("latent_conservative_horizon", None)

    if "latent_conservative_factor" in params:
        try:
            conservative["factor"] = float(params.get("latent_conservative_factor", 1.0))
        except Exception:
            conservative["factor"] = 1.0

    if conservative:
        cfg["conservative"] = conservative

    return cfg

import ast
from copy import deepcopy
from typing import Any, Dict, List

def _parse_py_literal(x: Any):
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return x
    try:
        return ast.literal_eval(s)   # suporta "{'a':1}", "['x']", "(1.0,)"
    except Exception:
        return x

def _coerce_dict(x: Any) -> Dict[str, Any]:
    x = _parse_py_literal(x)
    if x is None:
        return {}
    if isinstance(x, dict):
        return deepcopy(x)
    if isinstance(x, str):
        # shorthand tipo "off" / "offline_analytic"
        return {"mode": x.strip().lower()} if x.strip() else {}
    raise TypeError(f"latent_cfg inválido: {type(x).__name__} {x!r}")

def _coerce_list(x: Any) -> List[Any]:
    x = _parse_py_literal(x)
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]  # fallback conservador

def coerce_param_types(params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params)

    # o que está te quebrando
    params["latent_cfg"] = _coerce_dict(params.get("latent_cfg"))

    return params

def reconstruct_train_series_with_X_prefix(
    y_train_windows_unscaled: np.ndarray,   # shape (N, H), no domínio físico
    x_target_windows_unscaled: np.ndarray,  # shape (N, L), no domínio físico (target no X)
) -> np.ndarray:
    """
    Retorna série 1D completa do treino: [X_target_prefix(L)] + [reconstruct(y)(N+H-1)].
    """
    from common.seq_preprocessing import reconstruct_true_series
    x_prefix = np.asarray(x_target_windows_unscaled)[0, :].reshape(-1)     # L
    y_future = reconstruct_true_series(np.asarray(y_train_windows_unscaled))  # N+H-1
    return np.concatenate([x_prefix, y_future]).astype(float)



def _run_single_job_seq(job: Tuple) -> Dict[str, Any]:
    """
    Full pipeline for a single Seq2* job.

    Responsibilities:
      - Normalize params (merging DefaultExperimentParams + job overrides)
      - Build and attach latent_cfg (non-intrusive; defaults to 'off')
      - Train / predict via process_chunks
      - Evaluate via evaluate_job (tolerant to 6- or 7-tuple returns)
      - Attach series context and optional Series Store persistence
    """
    from common.seq_preprocessing import aggregate_predictions, reconstruct_true_series
    from forecast_pipeline.logging_utils import get_logger

    logger = get_logger(__name__)

    ds, well, params, experiment_id = job
    (
        train_kwargs,
        X_test,
        y_test_scaled,
        scaler_X,
        scaler_target,
        y_train_original,
        _,
        _,
        _,
        _,
    ) = prepare_job_data(job)

    # ------------------------------------------------------------------
    # 1) Normalize params (legacy behaviour preserved)
    # ------------------------------------------------------------------
    params = {**DEFAULT_EXP_PARAMS, **params}
    params = coerce_param_types(params)
    params = normalize_job_parameters(params)

    # ------------------------------------------------------------------
    # 2) NEW: build and attach latent_cfg (read-only configuration)
    # ------------------------------------------------------------------

    latent_cfg = _build_latent_cfg_from_params(params)
    params["latent_cfg"] = latent_cfg
    # Make it visible to the training pipeline as well (workers / executors)
    train_kwargs["latent_cfg"] = latent_cfg

    logger.info(
        "latent_cfg(attached) mode=%s arps_coupling_mode=%s keys=%s",
        latent_cfg.get("mode"),
        latent_cfg.get("arps_coupling_mode"),
        sorted(list(latent_cfg.keys())),
    )


    # ------------------------------------------------------------------
    # 3) Training context for integrated plots (unchanged)
    # ------------------------------------------------------------------
    x_ctx = _extract_unscaled_target_windows(
        X_train_scaled=train_kwargs["X_train"],
        y_train_windows=y_train_original,
        scaler_target=train_kwargs.get("scaler_target", scaler_target),
    )

    # Validation inputs
    X_val, y_val_scaled = train_kwargs["X_val"], train_kwargs["y_val"]

    # Ensure scaler_target available to downstream code (LEFT sidecar / _split_ctx)
    data_inputs = {
        "X_test": X_test,
        "X_val": X_val,
        "scaler_target": scaler_target,
    }

    # ------------------------------------------------------------------
    # 4) Train / predict chunks (unchanged)
    # ------------------------------------------------------------------
    ensemble_raw = process_chunks(train_kwargs, data_inputs, params, scaler_target)
    ensemble = to_ensemble_output(ensemble_raw)

    # ------------------------------------------------------------------
    # 5) Evaluation (unchanged)
    # ------------------------------------------------------------------
    eval_ret = evaluate_job(
        y_test_scaled=y_test_scaled,
        y_test_pred=ensemble.pred_test,
        y_val_scaled=y_val_scaled,
        y_val_pred=ensemble.pred_val,
        scaler_target=scaler_target,
        y_train_original=y_train_original,
        params=params,
        config=ds,
        well=well,
        plot=params.get("plot", False),
        ensemble_out=ensemble,
        x_train_main_windows=x_ctx,
    )
    (
        agg_test_df,
        cum_test_df,
        gm_test,
        agg_val_df,
        cum_val_df,
        gm_val,
        series_artifacts,
    ) = unpack_eval7(eval_ret)

    # ------------------------------------------------------------------
    # 6) Assemble canonical result dict (unchanged)
    # ------------------------------------------------------------------
    result_dict = _assemble_result_dict(
        well=well,
        experiment_id=experiment_id,
        agg_test_df=agg_test_df,
        cum_test_df=cum_test_df,
        gm_test=gm_test,
        agg_val_df=agg_val_df,
        cum_val_df=cum_val_df,
        gm_val=gm_val,
    )

    # ------------------------------------------------------------------
    # 7) Self-heal context + optional Series Store (unchanged)
    # ------------------------------------------------------------------
    y_train_full = reconstruct_train_series_with_X_prefix(y_train_original, x_ctx)

    attach_series_context(
        result_dict,
        scaler_target,
        series_artifacts,
        y_train_original=y_train_full,
    )
    ensure_job_meta(result_dict, params, ds, well, job_hash=generate_job_hash(params))
    maybe_persist_series(result_dict, params.get("series_store") or {}, logger)

    return result_dict


# ---------------------------------------------------------------------
# Darts runner
# ---------------------------------------------------------------------


def _run_single_job_darts(job: Tuple) -> Dict[str, Any]:
    """
    Executes a single training and evaluation job for a Darts family model.
    Tolerant to 6|7 item eval returns, embeds series_context, optional non-fatal Series Store write.

    Plug-and-play patch:
      - Avoid undefined x_ctx / SEQ2-only "prefix" logic.
      - Attach series_context using canonical train block (y_train_original) + eval artifacts.
      - Sanity-log boundaries if available via scaler_target._split_ctx.
    """
    logger = get_logger(__name__)
    ds, well, params, experiment_id = job

    context_info = {"well": well, "exp_id": experiment_id, "arch": params.get("architecture_name", "")}
    with log_context(**context_info):

        # 1) Prepare
        with phase(logger, "prepare_data"):
            prep_results       = prepare_job_data(job)
            train_kwargs       = prep_results[0]
            prediction_in      = prep_results[1]
            y_test_u           = prep_results[2]   # (unused here, but keep if prepare returns it)
            scaler_X           = prep_results[3]
            scaler_target      = prep_results[4]
            y_train_original   = prep_results[5]
            final_params       = prep_results[6]

            # ---- (Optional) sanity log: boundaries if present (canonical split contract) ----
            try:
                split_ctx = getattr(scaler_target, "_split_ctx", {}) or {}
                boundaries = split_ctx.get("boundaries") if isinstance(split_ctx, dict) else None
                if boundaries:
                    train_end  = boundaries.get("train_end")
                    val_start  = boundaries.get("val_start")
                    val_end    = boundaries.get("val_end")
                    test_start = boundaries.get("test_start")
                    total_len  = boundaries.get("total_len")
                    logger.info(
                        "✅ [darts|split_ctx] boundaries: train_end=%s | val=[%s,%s) | test_start=%s | total_len=%s",
                        train_end, val_start, val_end, test_start, total_len
                    )
                else:
                    logger.info("⚠️  [darts|split_ctx] no boundaries found in scaler_target._split_ctx")
            except Exception:
                logger.exception("⚠️  [darts|split_ctx] failed to log boundaries")

        # 2) Train
        train_hyperparams = {
            "epochs":        final_params.get("n_epochs") or final_params.get("epochs"),
            "batch_size":    final_params.get("batch_size"),
            "patience":      final_params.get("patience"),
            "learning_rate": final_params.get("learning_rate"),
        }
        with phase(logger, "train_darts", **train_hyperparams):
            model, _hist, pred_test_np, pred_val_np = main_train_darts_model(
                architecture_name=final_params["architecture_name"],
                train_kwargs=train_kwargs,
                data_inputs=prediction_in,
                **train_hyperparams,
            )

        # 3) Evaluate
        with phase(logger, "evaluate_darts", plot=bool(final_params.get("plot", False))):
            eval_ret = evaluate_job_from_darts(
                train_kwargs=train_kwargs,
                prediction_input=prediction_in,
                pred_val_ribbons=pred_val_np,
                pred_test_ribbons=pred_test_np,
                y_train_original=y_train_original,
                params=final_params,
                well=well,
                config=ds,
                plot=bool(final_params.get("plot", False)),
            )
            (agg_test_df, cum_test_df, gm_test,
             agg_val_df,  cum_val_df,  gm_val,
             series_artifacts) = unpack_eval7(eval_ret)

        # 4) Assemble + context + optional write
        with phase(logger, "assemble_result"):
            result_dict = _assemble_result_dict(
                well=well,
                experiment_id=experiment_id,
                agg_test_df=agg_test_df,
                cum_test_df=cum_test_df,
                gm_test=gm_test,
                agg_val_df=agg_val_df,
                cum_val_df=cum_val_df,
                gm_val=gm_val,
            )

            # ✅ IMPORTANT: For Darts, do NOT try to reconstruct "train with X prefix" here.
            # That SEQ2-only logic requires x_ctx and can reintroduce misalignment.
            # Instead, pass the canonical TRAIN GT series as-is.
            attach_series_context(
                result_dict,
                scaler_target,
                series_artifacts,
                y_train_original=y_train_original,
            )

            ensure_job_meta(result_dict, final_params, ds, well, job_hash=generate_job_hash(final_params))
            maybe_persist_series(result_dict, final_params.get("series_store") or {}, logger)

            return result_dict



def _extract_unscaled_target_windows(
    *,
    X_train_scaled: np.ndarray,  # Shape: (N_total, L, F)
    y_train_windows: np.ndarray, # Shape: (N_real, H)
    scaler_target,               # Objeto scaler (ex: StandardScaler)
) -> np.ndarray:
    """
    Extracts the target-channel windows, returning them in physical units.

    This function slices the input windows to match the number of original 
    (non-augmented) samples and then inverse-transforms the target channel.

    Args:
        X_train_scaled: The complete set of scaled input windows, which may
                        include synthetic samples from data augmentation.
        y_train_windows: The set of target windows, used to determine the
                         number of real (non-synthetic) samples.
        scaler_target: The scaler object previously fitted on the target variable.

    Returns:
        A numpy array of shape (N_real, L) containing the unscaled target
        values for each of the original input windows.
    """
    # 1. Determine the number of real (non-synthetic) windows
    num_real_windows = y_train_windows.shape[0]
    window_length = X_train_scaled.shape[1]

    # 2. Slice X to keep only the real windows, discarding any synthetic ones
    X_real_scaled = X_train_scaled[:num_real_windows]

    # 3. Isolate the target channel (assumed to be the last one)
    # The shape becomes (N_real, L, 1)
    target_channel_scaled = X_real_scaled[:, :, -1:]

    # 4. Reshape for the scaler's inverse_transform method, which expects 2D input
    # The shape becomes (N_real * L, 1)
    target_flat_scaled = target_channel_scaled.reshape(-1, 1)

    # 5. Inverse-transform to get back to physical units
    target_flat_unscaled = scaler_target.inverse_transform(target_flat_scaled)

    # 6. Reshape back into the windowed format (N_real, L)
    target_windows_unscaled = target_flat_unscaled.reshape(num_real_windows, window_length)

    return target_windows_unscaled.astype(float)


# forecast_pipeline/jobs.py

from typing import Any, Dict, Optional
import pandas as pd

def _assemble_result_dict(
    *,
    well: str,
    experiment_id: str,
    agg_test_df: pd.DataFrame, cum_test_df: pd.DataFrame, gm_test: Dict[str, Any],
    agg_val_df: pd.DataFrame,  cum_val_df: pd.DataFrame,  gm_val: Dict[str, Any],
    extra_meta: Optional[Dict[str, Any]] = None,   # <-- NEW (optional)
) -> Dict[str, Any]:
    """Build the legacy-compatible return dictionary (with optional metadata)."""
    # Slices (kept for backward compatibility)
    slice_agg_test, slice_cum_test, slice_glob_test = [], [], []
    slice_agg_val,  slice_cum_val,  slice_glob_val  = [], [], []

    out = {
        "status": "success",
        "aggregated_metrics_test": agg_test_df,
        "cumulative_metrics_test": cum_test_df,
        "global_metrics_test": gm_test,
        "aggregated_metrics_val": agg_val_df,
        "cumulative_metrics_val": cum_val_df,
        "global_metrics_val": gm_val,
        "slice_agg_test": slice_agg_test, "slice_cum_test": slice_cum_test, "slice_glob_test": slice_glob_test,
        "slice_agg_val":  slice_agg_val,  "slice_cum_val":  slice_cum_val,  "slice_glob_val":  slice_glob_val,
        "well": well, "experiment_id": experiment_id,
    }
    if extra_meta is not None:
        out["extra_meta"] = extra_meta
    return out



def _persist_if_requested(
    *,
    result: Dict[str, Any],
    params: Dict[str, Any],
    job_hash: str,
    well: str,
    experiment_id: str,
) -> Dict[str, Any]:
    # --- NOVO: consolidar aggregation_method efetivo ---
    def _lower(x):
        return str(x).strip().lower() if x is not None else ""

    agg_in = _lower(params.get("aggregation_method", ""))
    best = None

    # 1) tentar pegar do global_metrics_val (dict)
    try:
        gm_val = result.get("global_metrics_val") or {}
        best = gm_val.get("aggregation_method") or best
    except Exception:
        pass

    # 2) fallback: primeira linha de aggregated_metrics_val (DF)
    if not best:
        try:
            df_val = result.get("aggregated_metrics_val")
            if df_val is not None and hasattr(df_val, "empty") and not df_val.empty:
                if "aggregation_method" in df_val.columns:
                    best = df_val["aggregation_method"].iloc[0]
        except Exception:
            pass

    # 3) fallback: meta do sweep (se evaluate_job preencheu)
    if not best:
        best = (params.get("_aggregation_sweep_info") or {}).get("chosen_filter")

    # Se encontramos o vencedor, reflita no params antes de persistir
    if best:
        params["aggregation_method_effective"] = best  # mantém um campo explícito
        if agg_in in {"auto", "sweep", "all"} or params.get("aggregation_sweep", False):
            params["aggregation_method"] = best
            params["aggregation_sweep"] = True
            # opcional: carregue mais metadados do sweep, se existirem
            asi = params.get("_aggregation_sweep_info") or {}
            if asi.get("selection_metric"):
                params["aggregation_selection_metric"] = asi["selection_metric"]
            if asi.get("explored_filters"):
                params["aggregation_explored"] = asi["explored_filters"]
    # --- FIM DO NOVO BLOCO ---

    agg_val_df: pd.DataFrame = result["aggregated_metrics_val"]
    cum_val_df: pd.DataFrame = result["cumulative_metrics_val"]

    key_metrics = {
        "val_smape_agg": agg_val_df["SMAPE"].iloc[0] if not agg_val_df.empty else None,
        "val_smape_cum": cum_val_df["SMAPE"].iloc[0] if not cum_val_df.empty else None,
    }

    # Convert to serializable for JSON
    serializable = {
        k: (v.to_dict(orient="records") if hasattr(v, "to_dict") else v)
        for k, v in result.items()
    }
    # keep slice lists shape
    for key in ["slice_agg_test", "slice_cum_test", "slice_agg_val", "slice_cum_val"]:
        serializable[key] = [df.to_dict(orient="records") for df in serializable[key]]

    out_dir = params.get("run_output_dir")
    if not out_dir:
        raise ValueError("'run_output_dir' is required when persist_result=True.")

    payload = {
        "status": "success",
        "job_hash": job_hash,
        "experiment_id": experiment_id,
        "well": well,
        "key_metrics": key_metrics,
        "config": params,
        "metadata": build_run_metadata(),
        "results": serializable,
    }
    out_path = Path(out_dir) / f"{job_hash}.json"
    atomic_write_json(payload, out_path)

    return {
        "status": "success",
        "job_hash": job_hash,
        "well": well,
        "experiment_id": experiment_id,
        "output_path": str(out_path),
        **key_metrics,
    }


def _failure_result(*, well: str, experiment_id: str) -> Dict[str, Any]:
    return {
        "status": "failure",
        "error": "see logs",
        "well": well,
        "experiment_id": experiment_id,
    }



def select_data_sources(
    all_sources: List[Dict[str, Any]],
    selected_names: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """
    Return data sources matching `selected_names`, or all if None.
    """
    if selected_names is None:
        logging.info("No specific data sources selected; using all available.")
        return all_sources
    matched = [ds for ds in all_sources if ds["name"] in selected_names]
    if not matched:
        logging.warning("No data sources match the selection criteria.")
    return matched


def create_filter_configurations(
    filter_methods: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """
    Generate filter configurations based on provided methods.
    If `filter_methods` is None or empty, use only the no-filter case.
    """
    if filter_methods:
        return [
            {"apply_adaptive_filtering": True, "filter_method": m, "filter_kwargs": {"smoothing_level": 0.2}}
            for m in filter_methods
        ]
    return [{"apply_adaptive_filtering": False, "filter_method": None, "filter_kwargs": {}}]

