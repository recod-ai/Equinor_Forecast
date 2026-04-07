# src/hpo/hpo_runner.py

# 1. Standard library imports
import inspect
import logging
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# 2. Third-party imports
import pandas as pd
from box import Box

# 3. Local application imports
from common.config_wells import get_data_sources
from common.context_integration import apply_context_to_config
from config_loader import load_campaign_config
from forecast_pipeline.jobs import generate_jobs
from forecast_pipeline.metrics import collate_robust_results
from forecast_pipeline.runner import execute_jobs_robust
from hpo.posthoc_filtering import add_weighted_score
import hpo.search_space as search_space_module
from hpo.optuna_utils import generate_trials_from_study, report_results_to_study
from common.series_store_config import normalize_and_validate_series_store


# Project root for path resolution if needed
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==============================================================================
#                      CORE LOGIC (DESIGNED TO BE IMPORTED)
# ==============================================================================


def _sync_results_to_intended(legacy_run_dir: Path, intended_run_dir: Path) -> None:
    """
    Merge/copy artifacts from a legacy run directory (returned by the runner)
    into the intended (configured) run directory. Idempotent.
    Copies *.json under results/, plus common top-level run artifacts if present.
    """
    try:
        legacy_run_dir = legacy_run_dir.resolve()
        intended_run_dir = intended_run_dir.resolve()
        legacy_results = legacy_run_dir / "results"
        intended_results = intended_run_dir / "results"
        intended_results.mkdir(parents=True, exist_ok=True)

        # 1) Copy atomic JSON result files
        if legacy_results.is_dir():
            for jf in legacy_results.glob("*.json"):
                dst = intended_results / jf.name
                try:
                    shutil.copy2(jf, dst)
                except Exception as e:
                    logging.warning(f"[sync] Could not copy {jf} → {dst}: {e}")

        # 2) Copy common summary artifacts if present
        for name in ("run_summary.csv", "leaderboard.csv"):
            srcf = legacy_run_dir / name
            if srcf.exists():
                dstf = intended_run_dir / name
                try:
                    shutil.copy2(srcf, dstf)
                except Exception as e:
                    logging.warning(f"[sync] Could not copy {srcf} → {dstf}: {e}")

        # 3) Leave a small pointer/readme in the legacy folder
        try:
            readme = legacy_run_dir / "_SYNCED_TO.txt"
            readme.write_text(
                f"This run's canonical artifacts were synced to:\n{intended_run_dir}\n",
                encoding="utf-8",
            )
        except Exception as e:
            logging.debug(f"[sync] Could not write pointer file: {e}")

        logging.info(f"[sync] Merged artifacts into intended folder: {intended_run_dir}")
    except Exception as e:
        logging.warning(f"[sync] Skipped sync due to error: {e}")


def run_robust_pipeline(config: Box, profile_path: str) -> Optional[str]:
    logging.info(f"--- Starting pipeline for profile: {profile_path} ---")

    # Log effective series_store flags (Step 0: visibility only)
    try:
        series_norm, series_log = normalize_and_validate_series_store(config)
        # re-stash in case object was modified upstream or missing
        try:
            config.series_store = series_norm
        except Exception:
            pass
        print(
            "[series_store] enabled=%s | format=%s | compress=%s | schema_version=%s | self_heal=%s | base_dir=%s",
            series_log.get("enabled"),
            series_log.get("format"),
            series_log.get("compress"),
            series_log.get("schema_version"),
            series_log.get("self_heal"),
            series_log.get("base_dir"),
        )
    except Exception as e:
        print("[series_store] could not validate/log config: %s", e)


    # 0) Resolve & enforce the configured results root
    try:
        results_root = Path(config.infra.experiments_output_dir).resolve()
        results_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create/resolve experiments_output_dir: {e}")
        return None
    logging.info(f"[pipeline] configured results root: {results_root}")

    # 0.1) Best-effort monkey patch of legacy module-level defaults
    try:
        import forecast_pipeline.config as fp_cfg  # noqa
        for attr in (
            "EXPERIMENTS_OUTPUT_DIR", "RESULTS_DIR",
            "EXPERIMENTS_DIR", "EXPERIMENTS_OUTPUT_PATH",
        ):
            if hasattr(fp_cfg, attr):
                old = getattr(fp_cfg, attr)
                setattr(fp_cfg, attr, str(results_root))
                logging.info(f"[pipeline] patched forecast_pipeline.config.{attr}: "
                             f"{old} → {results_root}")
    except Exception as e:
        logging.debug(f"[pipeline] config monkey-patch skipped ({e})")

    # 1) Profile sniffing: rich vs lean
    try:
        profile_df = pd.read_csv(profile_path)
        is_rich_profile = 'well' in profile_df.columns and 'dataset' in profile_df.columns
    except Exception:
        is_rich_profile = False

    if is_rich_profile:
        sources = []
        logging.info("Rich profile detected. Bypassing initial data source filtering.")
    else:
        all_sources = get_data_sources()
        dataset_name = config.run_scope.dataset_name
        selected_wells = config.run_scope.wells

        target_ds_config = next((ds.copy() for ds in all_sources if ds['name'] == dataset_name), None)
        if not target_ds_config:
            logging.error(f"Dataset '{dataset_name}' from config not found.")
            return None
        if selected_wells != "all":
            target_ds_config['wells'] = selected_wells
        sources = [target_ds_config]

    # 2) Generate jobs (works for both rich & lean)
    jobs = generate_jobs(sources, config, profile_path=profile_path)
    if not jobs:
        logging.warning("No jobs generated from profile. Exiting.")
        return None

    # 3) Compute the intended per-run folders and inject into every job
    run_name = Path(profile_path).stem  # e.g., UNISIM_IV_P12_Seq2PIN_cycle_1
    intended_run_dir = results_root / run_name
    intended_results_dir = intended_run_dir / "results"
    intended_results_dir.mkdir(parents=True, exist_ok=True)

    try:
        for j in jobs:
            if isinstance(j, dict) and "params" in j:
                j["params"]["run_output_dir"] = str(intended_results_dir)
            elif isinstance(j, (list, tuple)) and len(j) >= 3 and isinstance(j[2], dict):
                j[2]["run_output_dir"] = str(intended_results_dir)
        sample_params = (
            jobs[0]["params"] if isinstance(jobs[0], dict) and "params" in jobs[0]
            else (jobs[0][2] if isinstance(jobs[0], (list, tuple)) and len(jobs[0]) >= 3 else {})
        )
        logging.info(f"[pipeline] injected run_output_dir: {sample_params.get('run_output_dir')}")
    except Exception as e:
        logging.warning(f"[pipeline] could not inject run_output_dir into jobs: {e}")

    # 4) Execute (runner may also compute its own run dir; we sync afterward)
    legacy_run_output_dir = execute_jobs_robust(
        jobs=jobs,
        base_output_dir=str(results_root),  # honored by new runner; ignored by some legacy paths
        run_name=run_name,
        max_workers=config.run_params.max_workers
    )
    if not legacy_run_output_dir:
        logging.error("execute_jobs_robust returned no run_output_dir.")
        return None

    legacy_run_output_dir = Path(legacy_run_output_dir).resolve()
    logging.info(f"[pipeline] runner reported run_output_dir: {legacy_run_output_dir}")
    logging.info(f"[pipeline] intended run_output_dir:        {intended_run_dir}")

    # 5) If the runner wrote somewhere else, consolidate artifacts into intended
    if legacy_run_output_dir != intended_run_dir:
        logging.warning(
            "[pipeline] ⚠ mismatch: runner output dir != intended dir. "
            "Consolidating artifacts into the configured location…"
        )
        _sync_results_to_intended(legacy_run_output_dir, intended_run_dir)
        effective_run_dir = intended_run_dir
    else:
        effective_run_dir = legacy_run_output_dir

    # 6) Marker for quick discovery (at configured root)
    try:
        marker = results_root / f"{run_name}.path"
        marker.write_text(str(effective_run_dir), encoding="utf-8")
    except Exception as e:
        logging.debug(f"Could not write marker file: {e}")

    # 7) Collate and save leaderboard in the canonical location
    leaderboard_df = collate_robust_results(effective_run_dir)
    if leaderboard_df is not None and not leaderboard_df.empty:
        leaderboard_path = effective_run_dir / "leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False)
        logging.info(f"Leaderboard for run '{run_name}' saved to: {leaderboard_path}")
    else:
        logging.warning("No successful jobs found to collate into a leaderboard.")

    return str(effective_run_dir)

def _normalize_chunk_lengths_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Garantir input_chunk_length=lag_window e output_chunk_length=horizon."""
    out = dict(d) if d else {}
    lag = out.get("lag_window", out.get("input_chunk_length"))
    hor = out.get("horizon", out.get("output_chunk_length"))
    try:
        if lag is not None:
            lag = int(lag)
    except Exception:
        pass
    try:
        if hor is not None:
            hor = int(hor)
    except Exception:
        pass
    if lag is not None:
        out["input_chunk_length"] = lag
    if hor is not None:
        out["output_chunk_length"] = hor
    return out

def _sanitize_profile_csv(profile_csv_path: Path) -> None:
    """
    Regrava o CSV de perfil garantindo:
      input_chunk_length = lag_window
      output_chunk_length = horizon
    (se as colunas existirem)
    """
    try:
        import pandas as pd
        df = pd.read_csv(profile_csv_path)
        cols = set(df.columns.str.lower())

        # nomes como estão no seu CSV
        if "lag_window" in cols and "input_chunk_length" in cols:
            df["input_chunk_length"] = df["lag_window"].astype(int)
        if "horizon" in cols and "output_chunk_length" in cols:
            df["output_chunk_length"] = df["horizon"].astype(int)

        # salva só se algo mudou
        df.to_csv(profile_csv_path, index=False)
        logging.info(f"[profile] sanitized chunk lengths in {profile_csv_path}")
    except Exception as e:
        logging.warning(f"[profile] could not sanitize profile csv: {e}")





# def run_hpo_campaign_loop(
#     config: Box,
#     search_space_func: Callable,
#     pipeline_runner_func: Callable,
# ) -> pd.DataFrame:
#     """
#     Runs a full, automated HPO campaign in cycles with result feedback.
#     This is the core loop, driven entirely by the config object.
#     """
#     # --- 1. Extract all necessary parameters from the config object ---
#     study_name = config.campaign_name
#     trials_schedule = config.hpo_params.trials_per_cycle_schedule
#     metric_weights = config.hpo_params.metric_weights.to_dict()
#     lower_is_better = config.hpo_params.lower_is_better.to_dict()
#     metric_to_optimize = config.hpo_params.metric_to_optimize
#     fixed_params = config.job_defaults.to_dict()
    
#     # Paths are now also read from the config
#     profiles_dir = Path(config.infra.profiles_dir)
#     hpo_studies_dir = Path(config.infra.hpo_studies_dir)
#     storage_url = f"sqlite:///{hpo_studies_dir}/{study_name}.db"
    
#     # Create necessary directories if they don't exist
#     profiles_dir.mkdir(parents=True, exist_ok=True)
#     hpo_studies_dir.mkdir(parents=True, exist_ok=True)

#     # --- 2. Initialize state for the campaign ---
#     all_cycle_leaderboards = []
#     total_cycles = len(trials_schedule)
#     start_time = time.time()

#     # --- 3. The main HPO cycle loop ---
#     for i, num_trials in enumerate(trials_schedule):
#         cycle_num = i + 1
#         logging.info(f"\n{'='*25} HPO CYCLE {cycle_num}/{total_cycles} {'='*25}")

#         # Define the path for the trial profile CSV for this cycle
#         profile_path = profiles_dir / f"{study_name}_cycle_{cycle_num}.csv"

#         logging.info(f"Generating {num_trials} new trials for study '{study_name}'...")
#         # Generate new trials using Optuna
#         generate_trials_from_study(
#             study_name=study_name,
#             storage_url=storage_url, # Pass the explicit storage URL
#             n_trials=num_trials,
#             output_file=profile_path,
#             search_space_func=search_space_func,
#             **fixed_params
#         )

#         _sanitize_profile_csv(profile_path)
#         logging.info(f"Executing pipeline for profile: {profile_path}")
#         # The pipeline runner wrapper already has the config, so it only needs the profile path
#         run_output_dir = pipeline_runner_func(profile_path=str(profile_path))
        
#         if not run_output_dir:
#             logging.warning(f"Pipeline run for cycle {cycle_num} failed. Skipping.")
#             continue

#         logging.info("Collating results and reporting back to Optuna...")
#         cycle_leaderboard_df = collate_robust_results(run_output_dir)
        
#         if cycle_leaderboard_df.empty:
#             logging.warning("No successful jobs in this cycle to report.")
#             continue

#         # Save the leaderboard for this specific cycle
#         leaderboard_path = Path(run_output_dir) / "leaderboard.csv"
#         cycle_leaderboard_df.to_csv(leaderboard_path, index=False)

#         # Add the weighted score for optimization
#         leaderboard_with_score_df = add_weighted_score(
#             cycle_leaderboard_df, metric_weights, lower_is_better
#         )
#         all_cycle_leaderboards.append(leaderboard_with_score_df)

#         # Report the results back to the Optuna study
#         report_results_to_study(
#             study_name=study_name,
#             storage_url=storage_url, # Pass the explicit storage URL
#             leaderboard_df=leaderboard_with_score_df,
#             metric_to_optimize=metric_to_optimize
#         )
#         logging.info(f"--- End of Cycle {cycle_num} ---")

#     # --- 4. Finalize and return results ---
#     elapsed_time = time.time() - start_time
#     logging.info(f"\nTotal HPO campaign runtime: {elapsed_time:.2f} seconds")
#     logging.info("🎉 HPO Campaign Complete!")

#     if all_cycle_leaderboards:
#         # Concatenate results from all cycles into one master leaderboard
#         return pd.concat(all_cycle_leaderboards, ignore_index=True)
    
#     # Return an empty DataFrame if no results were generated
#     return pd.DataFrame()


def run_hpo_campaign_loop(
    config: Box,
    search_space_func: Callable,
    pipeline_runner_func: Callable,
) -> pd.DataFrame:
    """
    Runs a full HPO campaign in cycles, supporting both single and multi-objective
    optimization and reporting.
    """
    # --- 1. Extract parameters from config ---
    study_name = config.campaign_name
    hpo_cfg = config.hpo_params
    trials_schedule = hpo_cfg.trials_per_cycle_schedule
    
    profiles_dir = Path(config.infra.profiles_dir)
    hpo_studies_dir = Path(config.infra.hpo_studies_dir)
    storage_url = f"sqlite:///{hpo_studies_dir}/{study_name}.db"
    
    profiles_dir.mkdir(parents=True, exist_ok=True)
    hpo_studies_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Initialize state ---
    all_cycle_leaderboards = []
    total_cycles = len(trials_schedule)
    start_time = time.time()

    # --- 3. The main HPO cycle loop ---
    for i, num_trials in enumerate(trials_schedule):
        cycle_num = i + 1
        logging.info(f"\n{'='*25} HPO CYCLE {cycle_num}/{total_cycles} {'='*25}")

        profile_path = profiles_dir / f"{study_name}_cycle_{cycle_num}.csv"
        logging.info(f"Generating {num_trials} new trials for study '{study_name}'...")
        
        trial_gen_kwargs = {
            "study_name": study_name, "storage_url": storage_url, "n_trials": num_trials,
            "output_file": profile_path, "search_space_func": search_space_func,
        }
        
        is_multi_objective = hpo_cfg.get("mode") == "multi-objective"
        
        if is_multi_objective:
            objectives_dict = hpo_cfg.get("objectives", {})
            if objectives_dict:
                trial_gen_kwargs["directions"] = list(objectives_dict.values())
        
        trial_gen_kwargs.update(config.job_defaults.to_dict())
        generate_trials_from_study(**trial_gen_kwargs)
        
        _sanitize_profile_csv(profile_path)
        logging.info(f"Executing pipeline for profile: {profile_path}")
        
        run_output_dir = pipeline_runner_func(profile_path=str(profile_path))
        
        if not run_output_dir:
            logging.warning(f"Pipeline run for cycle {cycle_num} failed. Skipping.")
            continue

        logging.info("Collating results and reporting back to Optuna...")
        cycle_leaderboard_df = collate_robust_results(run_output_dir)
        
        if cycle_leaderboard_df is None or cycle_leaderboard_df.empty:
            logging.warning("No successful jobs in this cycle to report.")
            continue

        leaderboard_path = Path(run_output_dir) / "leaderboard.csv"
        cycle_leaderboard_df.to_csv(leaderboard_path, index=False)

        # --- START OF MODIFICATION ---

        report_kwargs = {
            "study_name": study_name,
            "storage_url": storage_url,
            "leaderboard_df": cycle_leaderboard_df
        }

        if is_multi_objective:
            # For multi-objective, we don't calculate a weighted score.
            # We pass the list of objective names to the reporting function.
            objective_names = list(hpo_cfg.get("objectives", {}).keys())
            report_kwargs["objective_keys"] = objective_names
            leaderboard_to_report = cycle_leaderboard_df
        else:
            # For single-objective, we calculate the weighted score as before.
            leaderboard_to_report = add_weighted_score(
                cycle_leaderboard_df, 
                hpo_cfg.metric_weights.to_dict(), 
                hpo_cfg.lower_is_better.to_dict()
            )
            report_kwargs["objective_keys"] = hpo_cfg.metric_to_optimize
        
        all_cycle_leaderboards.append(leaderboard_to_report)

        # Call the reporting function with the appropriate arguments for the mode.
        report_results_to_study(**report_kwargs)
        
        # --- END OF MODIFICATION ---

        logging.info(f"--- End of Cycle {cycle_num} ---")

    # --- 4. Finalize ---
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal HPO campaign runtime: {elapsed_time:.2f} seconds")
    logging.info("🎉 HPO Campaign Complete!")

    if all_cycle_leaderboards:
        return pd.concat(all_cycle_leaderboards, ignore_index=True)
    
    return pd.DataFrame()


def cleanup_old_study_data(config: Box):
    # --- The cleanup function, also from the previous step ---
    study_name = config.campaign_name
    hpo_studies_dir = Path(config.infra.hpo_studies_dir)
    experiments_output_dir = Path(config.infra.experiments_output_dir)
    
    hpo_studies_dir.mkdir(parents=True, exist_ok=True)
    experiments_output_dir.mkdir(parents=True, exist_ok=True)

    db_path = hpo_studies_dir / f"{study_name}.db"
    if db_path.exists():
        print(f"--- Removing old study database: {db_path} ---")
        db_path.unlink()

    for old_run_dir in experiments_output_dir.glob(f"{study_name}_cycle_*"):
        if old_run_dir.is_dir():
            print(f"--- Removing old experiment directory: {old_run_dir} ---")
            shutil.rmtree(old_run_dir)

def _to_plain_dict(obj):
    try:
        from box import Box
        if isinstance(obj, Box):
            return obj.to_dict()
    except Exception:
        pass
    return dict(obj) if isinstance(obj, dict) else {}

def _func_accepts_var_kwargs(func) -> bool:
    try:
        sig = inspect.signature(func)
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        return True  # be permissive if we cannot inspect

def _filter_kwargs_for(func, kwargs: dict) -> dict:
    """Only pass kwargs that the function can accept (unless it has **kwargs)."""
    if _func_accepts_var_kwargs(func):
        return kwargs
    try:
        sig = inspect.signature(func)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs

def _prepare_search_space_partial(search_space_func, config: Box):
    """
    Build a partial(search_space_func, **extra_kwargs) safely:
    - includes architecture_yaml_path only if present and accepted
    - includes optional hpo_params.search_space_overrides (if any)
    """
    extra = {}
    arch_yaml = getattr(config, "infra", {}).get("architecture_yaml_path")
    if arch_yaml:
        extra["architecture_yaml_path"] = arch_yaml

    overrides = {}
    if hasattr(config, "hpo_params") and config.hpo_params:
        raw = getattr(config.hpo_params, "search_space_overrides", {}) or {}
        overrides = _to_plain_dict(raw)

    merged = {**extra, **overrides}
    filtered = _filter_kwargs_for(search_space_func, merged)
    from functools import partial as _partial
    return _partial(search_space_func, **filtered)

def _has_hpo(config: Box) -> bool:
    return hasattr(config, "hpo_params") and bool(config.hpo_params)

def _is_darts_micro_campaign(config: Box) -> bool:
    """
    Micro-campaign → curated Darts grids without an HPO study/profile.
    Triggered only when:
      - architecture_name starts with 'Darts'
      - no hpo_params present
    """
    arch = str(getattr(getattr(config, "job_defaults", {}), "architecture_name", "") or "").lower()
    return arch.startswith("darts") and not _has_hpo(config)




def run_campaign_from_config_object(config: Box, profile_override_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Executes a full campaign from a pre-loaded and pre-processed config object.
    Backward compatible:
      - HPO mode (legacy/new)
      - Straight-run (validation)
      - Darts micro-campaign (no profile; curated grids)
    """
    logging.info(f"Starting campaign '{config.campaign_name}' from config object.")

    # 1) Clean up old artifacts for this specific campaign
    cleanup_old_study_data(config)

    # 2) Branches
    if _has_hpo(config):
        # --- HPO EXECUTION PATH ---
        logging.info("Executing in HPO mode.")

        base_search_func_name = config.hpo_params.search_space_func_name
        if base_search_func_name == "define_fast_search_space":
            search_space_func = search_space_module.define_fast_search_space
        else:
            arch_name = config.job_defaults.architecture_name
            if arch_name not in search_space_module.SEARCH_SPACE_MAPPING:
                logging.error(f"No search space for architecture '{arch_name}' in SEARCH_SPACE_MAPPING.")
                return None
            search_space_func = search_space_module.SEARCH_SPACE_MAPPING[arch_name]

        # Safe/compatible partial with optional overrides + yaml_path
        final_search_space_func = _prepare_search_space_partial(search_space_func, config)

        # Wrapper for robust pipeline
        def pipeline_runner_wrapper(profile_path: str):
            return run_robust_pipeline(config, profile_path)

        # Run the HPO loop
        master_leaderboard = run_hpo_campaign_loop(
            config=config,
            search_space_func=final_search_space_func,
            pipeline_runner_func=pipeline_runner_wrapper,
        )

    elif _is_darts_micro_campaign(config):
        # --- DARTS MICRO-CAMPAIGN (no profile CSV) ---
        logging.info("Executing Darts micro-campaign (curated grids, no profile file).")

        # Build sources from run_scope
        all_sources = get_data_sources()
        ds_name = config.run_scope.dataset_name
        ds = next((d.copy() for d in all_sources if d["name"] == ds_name), None)
        if not ds:
            logging.error(f"Dataset '{ds_name}' from config not found.")
            return None

        wells = config.run_scope.wells
        if wells != "all":
            ds["wells"] = wells

        # Generate curated Darts jobs (generate_jobs must support profile_path=None for Darts)
        jobs = generate_jobs([ds], config, profile_path=None)
        if not jobs:
            logging.warning("No jobs generated for Darts micro-campaign.")
            return pd.DataFrame()

        run_output_dir = execute_jobs_robust(
            jobs=jobs,
            base_output_dir=config.infra.experiments_output_dir,
            run_name=config.campaign_name,
            max_workers=config.run_params.max_workers,
        )

        if not run_output_dir:
            logging.error("Darts micro-campaign failed — no output directory.")
            return pd.DataFrame()

        leaderboard_df = collate_robust_results(run_output_dir)
        if leaderboard_df is not None and not leaderboard_df.empty:
            lb_path = Path(run_output_dir) / "leaderboard.csv"
            leaderboard_df.to_csv(lb_path, index=False)
            logging.info(f"Leaderboard saved to: {lb_path}")
        else:
            logging.warning("No successful jobs to collate for this Darts micro-campaign.")

        master_leaderboard = leaderboard_df if leaderboard_df is not None else pd.DataFrame()

    else:
        # --- STRAIGHT EXECUTION PATH (e.g., Validation) ---
        logging.info("Executing in straight run mode (not HPO).")

        if profile_override_path:
            profile_path = Path(profile_override_path)
        else:
            profile_path = Path(config.infra.profiles_dir) / f"{config.campaign_name}.csv"

        if not profile_path.exists():
            logging.error(f"Profile for straight run not found at {profile_path}")
            return None

        run_output_dir = run_robust_pipeline(config, str(profile_path))

        if run_output_dir:
            master_leaderboard = collate_robust_results(run_output_dir)
            leaderboard_path = Path(run_output_dir) / "leaderboard.csv"
            master_leaderboard.to_csv(leaderboard_path, index=False)
        else:
            master_leaderboard = pd.DataFrame()

    return master_leaderboard



def run_campaign_from_config_file(config_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load a campaign config, apply ExperimentContext defaults (group/arch), resolve
    relative path fields, then run the campaign.
    """
    try:
        config_file = Path(config_path).resolve(strict=True)
        config = load_campaign_config(config_file)

        # --- Step 0: normalize and validate series_store flags ---
        series_norm, series_log = normalize_and_validate_series_store(config)
        # Stash normalized block back into the config (Box-friendly)
        try:
            config.series_store = series_norm  # Box allows attribute assign
        except Exception:
            # Fallback for plain dict-like
            if hasattr(config, "update"):
                config.update({"series_store": series_norm})  # type: ignore

        
    except FileNotFoundError:
        logging.error(f"Configuration file not found at '{config_path}'")
        return None

    # Inject context-driven defaults (preserves absolute paths)
    config = apply_context_to_config(config)

    base_dir = config_file.parent

    def resolve_str_path(s: str) -> str:
        p = Path(s)
        resolved = p if p.is_absolute() else (base_dir / p).resolve()
        text = str(resolved)
        while "/src/src/" in text:
            text = text.replace("/src/src/", "/src/")
        return text

    infra = getattr(config, "infra", None) or {}
    if infra:
        for key, val in infra.items():
            if isinstance(val, str) and any(tok in key.lower() for tok in ("path", "dir")):
                config.infra[key] = resolve_str_path(val)

    for attr, val in vars(config).items():
        if isinstance(val, str) and any(tok in attr.lower() for tok in ("path", "dir")):
            setattr(config, attr, resolve_str_path(val))

    return run_campaign_from_config_object(config)



# ==============================================================================
# Runner (no edits typically needed below)
# ==============================================================================
from typing import Any, Dict, List, Optional, Mapping
from dataclasses import dataclass
from forecast_pipeline.plotting import render_champions_view_auto

@dataclass(frozen=True)
class ModePreset:
    name: str
    selector_mode: str
    scoring_strategy: str
    metric_to_optimize: str
    neighborhood_overrides: Dict[str, Any]
    posthoc_overrides: Dict[str, Any]
    help: str

def _resolve_champions_columns(df: pd.DataFrame, primary_metric: str, score_col: str) -> List[str]:
    if df is None or df.empty:
        return []
    def _has(c: str) -> bool: return c in df.columns

    family_cols = []
    if _has("variant") and _has("solver"):
        family_cols = ["variant", "solver", "weighting", "loss"]
    elif _has("profile") and _has("n_epochs"):
        family_cols = ["architecture_name", "physics_strategy", "profile", "n_epochs", "batch_size", "learning_rate"]
    elif _has("physics_strategy") and _has("epochs"):
        family_cols = ["architecture_name", "physics_strategy", "aggregation_method", "data_sample", "epochs", "learning_rate", "batch_size"]

    base = ["rank", "well", primary_metric, score_col]
    if score_col == "robust_score":
        base += ["_neighbor_scope", "_neighbor_local_mad", "_neighbor_rank_gap"]  # neighborhood diagnostics if present
    cols = list(dict.fromkeys([c for c in (base + family_cols) if c in df.columns]))
    return cols

def _print_mode_help(p: ModePreset) -> None:
    print("\n" + "=" * 90)
    print(f"MODE: {p.name}")
    print(p.help)
    print(f"selector_mode     : {p.selector_mode}")
    print(f"scoring_strategy  : {p.scoring_strategy}")
    print(f"metric_to_optimize: {p.metric_to_optimize}")
    if p.selector_mode.startswith("NEIGHBOR_"):
        nm = p.neighborhood_overrides.get("pool_method", "unknown")
        cfg = p.neighborhood_overrides.get("pool_cfg", {})
        print(f"neighborhood pool : {nm} | pool_cfg={cfg}")
    print("=" * 90 + "\n")

def render_result(result: Mapping[str, Any], preset: ModePreset) -> None:
    top = result.get("top_performers")
    meta = result.get("meta", {}) or {}
    summary = result.get("summary")

    print("\n--- OUTPUT -----------------------------")
    print(f"leaderboard_path      : {result.get('leaderboard_path')}")
    print(f"validation_profile    : {result.get('validation_profile_path')}")
    print(f"top_performers rows   : {len(top) if isinstance(top, pd.DataFrame) else 0}")
    print(f"summary rows          : {len(summary) if isinstance(summary, pd.DataFrame) else 0}")
    print(f"meta.selection_path   : {meta.get('selection_path')}")
    print(f"meta.selector_mode    : {meta.get('selector_mode')}")
    print(f"meta.selection_col    : {meta.get('selection_col')}")
    print(f"meta.pool_method      : {meta.get('pool_method')}")
    print(f"meta.score_direction  : {meta.get('score_direction')}")

    # --------------------------
    # Champions view
    # --------------------------
    if isinstance(top, pd.DataFrame) and not top.empty:
        primary_metric = "val_smape_agg"
        score_col = str(meta.get("selection_col") or preset.metric_to_optimize)
        cols = _resolve_champions_columns(top, primary_metric, score_col)

        print("\n--- 🏆 Champions View 🏆 ---")
        styled = render_champions_view_auto(
            df=top[cols] if cols else top,
            per_well_k=2,
            metric=primary_metric,
            lower_is_better=True,
        )
        display(styled)
    else:
        print("\nNo champions were selected (or mode not implemented).")

    # --------------------------
    # Regret table (compact)
    # --------------------------
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        print("\n--- 📉 Regret (TEST audit-only) ---")

        # 1. Select Columns (Snake case logic)
        wanted = [
            "dataset", "well", "architecture",
            "chosen_val_smape_agg",
            "chosen_test_smape_agg",
            "pool_best_test_smape_agg",
            "regret_test",
            "ratio_test",
            "val_test_spearman",
            "chosen_test_percentile",
        ]
        cols = [c for c in wanted if c in summary.columns]
        out = summary[cols].copy()

        # 2. Process Values (Numeric conversion + Rounding)
        # We do this BEFORE renaming to keep logic simple
        num_cols = [c for c in out.columns if c not in {"dataset", "well", "architecture"}]
        for c in num_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        out = out.round(2)

        # 3. Sort (Stable sort on keys)
        sort_cols = [c for c in ["dataset", "well", "architecture"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, kind="mergesort")
        
        # 4. Rename for Display (Compact Names)
        column_mapping = {
            "dataset": "Dataset",
            "well": "Well",
            "architecture": "Arch",
            "chosen_val_smape_agg": "Chosen Val",
            "chosen_test_smape_agg": "Chosen Test",
            "pool_best_test_smape_agg": "Best Test",
            "regret_test": "Regret Test",
            "ratio_test": "Ratio Test",
            "val_test_spearman": "Spearman",
            "chosen_test_percentile": "Test Pctl"
        }
        
        display(out.rename(columns=column_mapping))

        # Helpful hint if Spearman isn't present yet
        if "val_test_spearman" not in summary.columns:
            print("NOTE: 'val_test_spearman' is not in summary yet. Add it in build_canonical_summary.")

        summary = result.get("summary")
        render_quick_audit(summary)


def render_quick_audit(summary):
    import numpy as np
    import pandas as pd

    if summary is None or not isinstance(summary, pd.DataFrame) or summary.empty:
        return

    def q(x: np.ndarray, p: float) -> float:
        x = x[np.isfinite(x)]
        return float(np.quantile(x, p)) if len(x) else np.nan

    regret = pd.to_numeric(summary.get("regret_test", np.nan), errors="coerce").to_numpy(dtype=float)
    ratio = pd.to_numeric(summary.get("ratio_test", np.nan), errors="coerce").to_numpy(dtype=float)
    spearman = pd.to_numeric(summary.get("val_test_spearman", np.nan), errors="coerce").to_numpy(dtype=float)

    print("\n=== Quick audit (campaign-wide; TEST is audit-only) ===")
    print("Spearman(VAL,TEST) is a rank correlation (range [-1, +1]). +1 means VAL ranking matches TEST ranking (good proxy); 0 means weak/no relationship; -1 means VAL ranking is inverted vs TEST (risky proxy).")
    print("TEST regret = chosen_TEST - best_TEST (absolute gap; 0 is perfect; lower is better). TEST ratio = chosen_TEST / best_TEST (relative gap; 1.0 is perfect; e.g., 1.21 means the chosen model is ~21% worse than the best TEST for that group).")
    print("Quantiles: median (p50) is the typical case; p90 means 90% of groups are at or below that value (so it's a 'near-worst-case' summary).\n")

    print(f"- TEST regret: median={q(regret, 0.5):.4g} | p90={q(regret, 0.9):.4g} | max={np.nanmax(regret) if np.isfinite(regret).any() else np.nan:.6g}")
    print(f"- TEST ratio : median={q(ratio, 0.5):.4g} | p90={q(ratio, 0.9):.4g} | max={np.nanmax(ratio) if np.isfinite(ratio).any() else np.nan:.6g}")
    print(f"- Spearman(VAL,TEST): median={q(spearman, 0.5):.4g} | min={np.nanmin(spearman) if np.isfinite(spearman).any() else np.nan:.6g} | max={np.nanmax(spearman) if np.isfinite(spearman).any() else np.nan:.6g}")

    n_bad = int(np.sum(np.isfinite(spearman) & (spearman < 0)))
    if n_bad:
        print(f"\n⚠️  Note: {n_bad}/{len(summary)} groups have negative Spearman. In those groups, VAL ranking may be a poor proxy for TEST ranking.")
