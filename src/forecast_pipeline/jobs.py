from itertools import product
from typing import Dict, Any, List, Tuple


import logging 
from itertools import product 
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .config import (SEQ2SEQ_ARCHS, STRATEGY_OPTIONS, EXTRACTOR_OPTIONS, FUSER_OPTIONS, EXPERIMENT_CONFIGURATIONS)

# ─── Standard library imports ────────────────────────────────────────────────────
import gc                                          # garbage collection
import logging                                     # logging utilities
import math                                        # math functions

# ─── Third-party imports ─────────────────────────────────────────────────────────
import numpy as np                                 # array computing
import pandas as pd
import matplotlib.pyplot as plt

# ─── Local application imports ───────────────────────────────────────────────────
from .experiments.seq2context import ExperimentSeq2Context
from .experiments.seq2value import ExperimentSeq2Value
from training.train_models import main_train_model
from training.train_utils import train_predict_chunk, analyze_contributions
from .config import DEFAULT_EXP_PARAMS

from common.seq_preprocessing import reconstruct_true_series


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


def generate_jobs(
    data_sources: List[Dict[str, Any]],
    default_exp_params: Dict[str, Any],
    profile_path: Optional[str] = None
) -> List[Tuple[Dict[str, Any], str, Dict[str, Any], str]]:
    """
    Generates a list of experiment jobs to be executed.
    
    If a `profile_path` is provided, it loads experiments from the profile file.
    Otherwise, it falls back to the legacy Cartesian product generation.
    """
    # --- Path A: New Profile-Driven Logic ---
    if profile_path:
        logging.info(f"Loading experiment configurations from profile: {profile_path}")
        # Import locally to prevent potential circular dependency issues.
        from profile_manager import load_and_expand_profile 
        
        # This function does all the heavy lifting we tested in the notebook.
        profile_configs = load_and_expand_profile(profile_path)
        
        jobs = []
        for ds in data_sources:
            ds_name = ds["name"]
            
            # 2. Get the "base" configuration for this data source (e.g., the feature list).
            base_configs = EXPERIMENT_CONFIGURATIONS.get(ds_name, [])
            if not base_configs:
                logging.warning(f"No base configurations in EXPERIMENT_CONFIGURATIONS for data source '{ds_name}'. Skipping.")
                continue
            # In most cases, there's only one base config per data source (e.g., one feature list).
            base_config = base_configs[0]
    
            for well in ds.get("wells", []):
                for profile_config in profile_configs:
                    final_params = {
                        **default_exp_params, 
                        **base_config,
                        **profile_config
                    }
                    
                    exp_id = final_params.get("experiment_id", "unknown_id")
                    jobs.append((ds, well, final_params, exp_id))
        
        logging.info(f"Generated {len(jobs)} jobs from profile ({len(profile_configs)} configs x {len(data_sources)*len(ds.get('wells',[]))} wells).")
        return jobs
    # --- Path B: Legacy Fallback Logic ---
    else:
        logging.warning("No profile path provided. Falling back to legacy job generation.")
        # Note: The old function returns an integer ID, but the new one returns a string.
        # This is a minor type difference that the rest of the pipeline should handle gracefully.
        # We can cast it to a string for consistency if needed.
        legacy_jobs = _legacy_generate_jobs(data_sources, default_exp_params)
        # Convert integer job_id to string for type consistency
        return [(ds, well, params, str(job_id)) for ds, well, params, job_id in legacy_jobs]


def prepare_job_data(job):
    """
    Load & prepare data once; dispatch to Seq2Context or Seq2Value based on DEFAULT_EXP_PARAMS.

    Returns:
      train_kwargs: dict with X_train, y_train, X_val, y_val, and configs
      prediction_input: test inputs for inference
      y_test: test targets (scaled)
      scaler_target: scaler for the target variable
      y_train_original: original, unscaled training targets
      params, ds, well, job_id: context for the experiment
    """
    ds, well, params, job_id = job
    arch = DEFAULT_EXP_PARAMS.get("architecture_name")

    if arch in SEQ2SEQ_ARCHS:
        exp_cls = ExperimentSeq2Context
    elif arch == "Seq2Value":
        exp_cls = ExperimentSeq2Value
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    exp = exp_cls(ds, well, params, job_id)
    train_kwargs, prediction_input, y_test, scaler_target, y_train_original = exp.load_and_prepare()

    return (
        train_kwargs,
        prediction_input,
        y_test,
        scaler_target,
        y_train_original,
        params,
        ds,
        well,
        job_id
    )



def process_chunks(
    train_kwargs: dict,
    data_inputs: dict,
    params: dict,
    scaler_target
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train and predict an ensemble in parallel chunks, aggregating both test and validation predictions.

    Returns:
      final_test_pred: np.ndarray, aggregated test predictions
      final_val_pred:  np.ndarray, aggregated validation predictions
    """
    total = params["ensemble_models"]
    chunk = min(1, total)
    snaps = params.get("with_snapshots", 5)
    retries = 2
    skip = True

    epochs     = params["epochs"]
    batch_size = params["batch_size"]
    patience   = params["patience"]
    learning_rate = params["learning_rate"]

    X_test = data_inputs["X_test"]
    X_val  = data_inputs["X_val"]

    logging.info(f"→ Beginning process_chunks: total={total}, chunk={chunk}")
    sum_test_preds = None
    sum_val_preds  = None
    sum_qs         = None
    sum_res        = None
    sum_sigma_test = sum_sigma_val = None
    sum_alpha      = 0.0
    total_models   = 0
    can_analyze    = True

    num_batches = math.ceil(total / chunk)
    for b in range(num_batches):
        size = min(chunk, total - b * chunk)
        logging.info(f"  → Batch {b+1}/{num_batches} (size={size})")
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
            skip
        )
        logging.info(f"  → {chunk_out.keys()}")
        n = chunk_out["successful_models"]
        if n == 0:
            continue

        total_models += n

        # initialize sums
        if sum_test_preds is None:
            sum_test_preds = np.zeros_like(chunk_out["pred_test"], dtype=np.float64)
            sum_val_preds  = np.zeros_like(chunk_out["pred_val"],   dtype=np.float64)
        sum_test_preds += chunk_out["pred_test"] * n
        sum_val_preds  += chunk_out["pred_val"]  * n

        # contributions on test
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
            sum_alpha += chunk_out["alpha_val"] * n
        else:
            can_analyze = False

        if "sigma_test" in chunk_out:
            if sum_sigma_test is None:
                sum_sigma_test = np.zeros_like(chunk_out["sigma_test"], dtype=np.float64)
                sum_sigma_val  = np.zeros_like(chunk_out["sigma_val"],  dtype=np.float64)
            sum_sigma_test += chunk_out["sigma_test"] * n
            sum_sigma_val  += chunk_out["sigma_val"]  * n

        logging.info(f"  ← Batch {b+1} done, total models so far: {total_models}/{total}")

    if total_models == 0:
        raise RuntimeError("No models processed in any chunk")

    final_test_pred = sum_test_preds / total_models
    final_val_pred  = sum_val_preds  / total_models

    
    out_dict = {
        "pred_test": final_test_pred,
        "pred_val":  final_val_pred,
    }

    if sum_sigma_test is not None:
        out_dict["sigma_test"] = sum_sigma_test / total_models
        out_dict["sigma_val"]  = sum_sigma_val  / total_models

    # contribution analysis on test only
    if can_analyze:
        final_qs    = sum_qs    / total_models
        final_res   = sum_res   / total_models
        final_alpha = sum_alpha / total_models

        if params["architecture_name"].startswith('Seq2'):
            L = params.get("horizon")
            final_test_pred = final_test_pred[:, :L]
            final_qs        = final_qs[:,   :L]
            final_res       = final_res[:,  :L]

        logging.info("← Running contribution analysis on aggregated test results")
        analyze_contributions(
            Qs=final_qs,
            res=final_res,
            alpha=final_alpha,
            scaler_target=scaler_target
        )
    else:
        logging.info("← RePINN Contribution analysis skipped")

    logging.info("← process_chunks complete, returning test and validation predictions")
    return out_dict


from forecast_pipeline.ensemble_output import to_ensemble_output, EnsembleOutput
from forecast_pipeline.analytics import scenario_curve
from forecast_pipeline.plotting import plot_predictions_wrapper

from evaluation.evaluation import (
    evaluate_model, evaluate_cumulative, compute_metrics_to_df,
    evaluate_model_seq, evaluate_cumulative_seq, compute_metrics_to_df_seq, evaluate
)


# -------------------------------------------------------------
# evaluate_job refatorado
# -------------------------------------------------------------

# def evaluate_job(
#     y_test_scaled: np.ndarray,
#     y_test_pred:  np.ndarray,
#     y_val_scaled: np.ndarray,
#     y_val_pred:   np.ndarray,
#     scaler_target,
#     y_train_original: np.ndarray,
#     params: Dict[str, Any],
#     config: Dict[str, Any],
#     well: str,
#     plot: bool = True,
#     *,
#     ensemble_out: Optional["EnsembleOutput"] = None,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any],
#            pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
#     """Avalia e plota, devolvendo métricas para Teste e Validação."""

#     # -------- tags comuns p/ DataFrames e global_metrics -------------
#     base_tags = {
#         "Method":    params["architecture_name"],
#         "Well":      well,
#         "strategy":  params["strategy_config"]["strategy_name"],
#         "extractor": params["extractor_config"]["type"],
#         "fuser":     params["fuser_config"]["type"],
#     }

#     arch  = params["architecture_name"]
#     if arch == "Seq2Context":
#         label = (
#             f"Well {well} │ PINN: "
#             f"{base_tags['strategy'].replace('_',' ').title()} │ <br> "
#             f"Data-Driven: {base_tags['extractor'].upper()} & {base_tags['fuser'].capitalize()}"
#         )
#     elif arch == "Seq2PIN":
#         label = (
#             f"Well {well} │ PINN: "
#             f"{base_tags['strategy'].replace('_',' ').title()} "
#         )
#     elif arch == "Seq2Trend":
#         label = (
#             f"Well {well} │ PINN + Trend: "
#             f"{base_tags['strategy'].replace('_',' ').title()} "
#         )
#     else:
#         label = (
#             f"Well {well} │ NONE: "
#             f"{base_tags['strategy'].replace('_',' ').title()} "
#         )
        

#     # ------------------------- PLOT HELPER ----------------------------
#     def _plot_seq(
#         truth: np.ndarray,
#         pred:  np.ndarray,
#         title_suffix: str,
#         *,
#         is_cum: bool = False,
#         window_size: int | None = None,
#         steps: int | None = None,
#         pct_split: float | None = None,
#         split: str = "test"
#     ) -> None:
#         """Encapsula toda a lógica de plot (novo ou legado)."""
#         if not plot:
#             return

#         # --- métricas rápidas p/ anotar ------------------------------
#         r2, smape, mae = evaluate(truth, pred)

#         # -------------------------------------------------------------
#         # Decide de onde vem µ e σ   (test  vs  val)
#         # -------------------------------------------------------------
#         if split == "val":
#             mu_stack   = ensemble_out.pred_val
#             sigma_stack = ensemble_out.sigma_val
#         else:                             # default = "test"
#             mu_stack   = ensemble_out.pred_test
#             sigma_stack = ensemble_out.sigma_test

#         # --- novo plot probabilístico (se ensemble_out passado) ------
#         if ensemble_out is not None:
#             kind = params.get("__plot_kind__", "P50")
#             band = params.get("band")
#             show_comp = params.get("show_components", False)

#             # monta kwargs comuns (legenda, métricas…)
#             common_kw = dict(
#                 scaler=scaler_target,
#                 smape=smape, mae=mae,
#                 window_size=window_size, forecast_steps=steps,
#                 percentage_split=pct_split,
#                 show_components=show_comp,
#                 title=label,
#             )

#             # ---------------------------------------------------------
#             # Calcula on-the-fly
#             # ---------------------------------------------------------
#             manual_env: Optional[Tuple[np.ndarray, np.ndarray]] = None
#             if is_cum and band and sigma_stack is not None:
#                 # ------------------------------------------------------------------
#                 # 1) converte (B,H) ➜ série longa (L,) p/ µ e σ
#                 # ------------------------------------------------------------------
#                 mu_rate  = reconstruct_true_series(mu_stack)      # (L,)
#                 sig_rate = reconstruct_true_series(sigma_stack)   # (L,)
            
#                 # 2) desscala:    µ  = shift+scale     σ = somente scale
#                 mu_rate_phys = scaler_target.inverse_transform(mu_rate.reshape(-1, 1)).ravel()
            
#                 scale = getattr(scaler_target, "scale_", [1.0])[0]   # Safe fallback
#                 sig_rate_phys = sig_rate * scale                     # sem shift!
            
#                 # 3) curva P10/P90 por passo
#                 plo, phi = band
#                 low_rate  = scenario_curve(mu_rate_phys, sig_rate_phys, plo)
#                 high_rate = scenario_curve(mu_rate_phys, sig_rate_phys, phi)
            
#                 # 4) acumula e alinha ao primeiro ponto da série cumulativa
#                 adj = truth[0] - mu_rate_phys[0]
#                 low_cum  = np.cumsum(low_rate)  + adj
#                 high_cum = np.cumsum(high_rate) + adj
#                 manual_env = (low_cum, high_cum)

#             # wrapper decide o resto
#             plot_predictions_wrapper(
#                 ensemble_out,
#                 truth=truth,
#                 kind=kind,
#                 well=well,
#                 band=band,
#                 mean_override=pred if not is_cum else pred,  # mantém curva média
#                 manual_envelope=manual_env,
#                 is_cum = is_cum,
#                 **common_kw,
#             )
#         # --- fallback legado ----------------------------------------
#         else:
#             from evaluation.evaluation import evaluate_and_plot
#             evaluate_and_plot(
#                 y_true=truth,
#                 y_pred=pred,
#                 title=f"{label} – {title_suffix}",
#                 well=well,
#                 set_name=title_suffix,
#                 additional_params=dict(
#                     window_size=window_size,
#                     forecast_steps=steps,
#                     percentage_split=pct_split,
#                 ),
#             )

#     # ======================= AVA. SEQ-TO-SEQ =========================
#     def _eval_seq(y_true, y_pred, y_train, split):
#         res = evaluate_model_seq(
#             y_true, y_pred, scaler_target,
#             params["lag_window"], params["horizon"],
#             1 - params["test_size"], config,
#             eval_title="Seq-to-Seq", set_name=label,
#             aggregation_method=params.get("aggregation_method"),
#             quantiles=params.get("aggregation_quantiles"),
#             plot=False,
#         )
#         agg_y, agg_pred, gm = res["agg_y_test"], res["agg_y_pred"], res["global_metrics"]

#         # ---------- plots (já implementados) -------------------------
#         _plot_seq(agg_y, agg_pred, "Aggregated",
#                   window_size=params["lag_window"], steps=params["horizon"],
#                   pct_split=1 - params["test_size"], is_cum=False, split=split)

#         cum = evaluate_cumulative_seq(
#             agg_y, agg_pred, y_train, scaler_target,
#             params["lag_window"], params["horizon"],
#             config, plot=False,
#         )
#         _plot_seq(cum["y_test_cumsum"], cum["y_pred_cumsum"], "Cumulative",
#                   window_size=params["lag_window"], steps=params["horizon"],
#                   pct_split=None, is_cum=True, split=split)

#         # ---------- DataFrames com tags ------------------------------
#         df_agg = pd.DataFrame([
#             compute_metrics_to_df_seq(agg_y, agg_pred, well, arch, "Aggregated")
#         ]).assign(**base_tags)
#         df_agg["Kind"] = "Aggregated_Window"

#         df_cum = pd.DataFrame([
#             compute_metrics_to_df_seq(cum["y_test_cumsum"], cum["y_pred_cumsum"], well, arch, "Cumulative")
#         ]).assign(**base_tags)
#         df_cum["Kind"] = "Cumulative_Sum"

#         gm_full = {**base_tags, **gm, "Category": "Global", "Kind": "Overall"}
#         return df_agg, df_cum, gm_full

#     # ======================= AVA. SEQ-TO-VALUE =======================
#     def _eval_value(y_true, y_pred, y_train, split):
#         r2, smape, mae = evaluate_model(
#             y_true, y_pred, scaler_target,
#             params["lag_window"], params["horizon"],
#             1 - params["test_size"], config,
#             eval_title="", set_name=label,
#         )
#         _plot_seq(y_true, y_pred, "Point Forecast",
#                   window_size=params["lag_window"], steps=params["horizon"],
#                   pct_split=1 - params["test_size"], is_cum=False, split=split)

#         y_inv_cum, y_pred_inv_cum = evaluate_cumulative(
#             y_true, y_pred, y_train,
#             params["lag_window"], params["horizon"],
#             config, set_name=label,
#         )
#         _plot_seq(y_inv_cum, y_pred_inv_cum, "Cumulative",
#                   window_size=params["lag_window"], steps=params["horizon"],
#                   pct_split=None, is_cum=True, split=split)

#         df_reg = pd.DataFrame([
#             compute_metrics_to_df(y_true, y_pred, well, arch, "Series")
#         ]).assign(**base_tags)
#         df_cum = pd.DataFrame([
#             compute_metrics_to_df(y_inv_cum, y_pred_inv_cum, well, arch, "Series")
#         ]).assign(**base_tags)

#         gm_full = {**base_tags, "R²": r2, "SMAPE": smape, "MAE": mae,
#                    "Category": "Global", "Kind": "Overall"}
#         return df_reg, df_cum, gm_full

#     # ======================= DISPATCHER ==============================
#     if arch in SEQ2SEQ_ARCHS:
#         agg_test, cum_test, gm_test = _eval_seq(y_test_scaled, y_test_pred, y_train_original, split="test")
#         agg_val,  cum_val,  gm_val  = _eval_seq(y_val_scaled,  y_val_pred,  y_train_original, split="val")
#     else:
#         agg_test, cum_test, gm_test = _eval_value(y_test_scaled, y_test_pred, y_train_original, split="test")
#         agg_val,  cum_val,  gm_val  = _eval_value(y_val_scaled,  y_val_pred,  y_train_original, split="val")

#     return agg_test, cum_test, gm_test, agg_val, cum_val, gm_val


# NEW: Add this import to bring in the new plotting function
from .plotting import plot_integrated_view


def evaluate_job(
    y_test_scaled: np.ndarray,
    y_test_pred:  np.ndarray,
    y_val_scaled: np.ndarray,
    y_val_pred:   np.ndarray,
    scaler_target,
    y_train_original: np.ndarray,
    params: Dict[str, Any],
    config: Dict[str, Any],
    well: str,
    plot: bool = True,
    *,
    ensemble_out: Optional["EnsembleOutput"] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any],
           pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates and plots model performance, returning metrics for Test and Validation.

    This refactored version includes:
    1. All original per-split plots (Aggregated and Cumulative for Test/Val).
    2. A new, integrated plot showing Train, Validation, and Test data together.
    3. A new, integrated cumulative plot.
    """
    # -------- tags comuns p/ DataFrames e global_metrics -------------
    arch = params.get("architecture_name")

    # Start with tags that are common to all architectures
    base_tags = {
        "Method": arch,
        "Well": well,
    }

    # Add strategy tag, which is also common
    if "strategy_config" in params:
        base_tags["strategy"] = params["strategy_config"].get("strategy_name", "N/A")
    else:
        base_tags["strategy"] = "N/A" # Fallback

    # Conditionally add tags for architectures that have extractor/fuser
    if arch in ["Seq2Context", "Seq2Fuser"]:
        # These architectures MUST have these configs
        base_tags["extractor"] = params["extractor_config"]["type"]
        base_tags["fuser"] = params["fuser_config"]["type"]
    else:
        # For Seq2PIN, Seq2Trend, etc., we use placeholder values
        base_tags["extractor"] = "none"
        base_tags["fuser"] = "none"
        
    if arch == "Seq2Context":
        label = (
            f"Well {well} │ PINN: "
            f"{base_tags['strategy'].replace('_',' ').title()} │ <br> "
            f"Data-Driven: {base_tags['extractor'].upper()} & {base_tags['fuser'].capitalize()}"
        )
    elif arch == "Seq2PIN":
        label = (f"Well {well} │ PINN: {base_tags['strategy'].replace('_',' ').title()} ")
    elif arch == "Seq2Trend":
        label = (f"Well {well} │ PINN + Trend: {base_tags['strategy'].replace('_',' ').title()} ")
    else:
        label = (f"Well {well} │ NONE: {base_tags['strategy'].replace('_',' ').title()} ")

    # ------------------------- PLOT HELPER (Unchanged) ----------------------------
    def _plot_seq(
        truth: np.ndarray,
        pred:  np.ndarray,
        title_suffix: str,
        *,
        is_cum: bool = False,
        window_size: int | None = None,
        steps: int | None = None,
        pct_split: float | None = None,
        split: str = "test"
    ) -> None:
        """Encapsula toda a lógica de plot (novo ou legado)."""
        if not plot:
            return
        r2, smape, mae = evaluate(truth, pred)
        if split == "val":
            mu_stack, sigma_stack = ensemble_out.pred_val, ensemble_out.sigma_val
        else:
            mu_stack, sigma_stack = ensemble_out.pred_test, ensemble_out.sigma_test

        if ensemble_out is not None:
            kind = params.get("__plot_kind__", "P50")
            band = params.get("band")
            show_comp = params.get("show_components", False)

            # monta kwargs comuns (legenda, métricas…)
            common_kw = dict(
                scaler=scaler_target,
                smape=smape, mae=mae,
                window_size=params["lag_window"], forecast_steps= params["horizon"],
                percentage_split=1 - params["test_size"] - params["val_size"],
                show_components=show_comp,
                title=label,
            )

            # ---------------------------------------------------------
            # Calcula on-the-fly
            # ---------------------------------------------------------
            manual_env: Optional[Tuple[np.ndarray, np.ndarray]] = None
            if is_cum and band and sigma_stack is not None:
                # ------------------------------------------------------------------
                # 1) converte (B,H) ➜ série longa (L,) p/ µ e σ
                # ------------------------------------------------------------------
                mu_rate  = reconstruct_true_series(mu_stack)      # (L,)
                sig_rate = reconstruct_true_series(sigma_stack)   # (L,)
            
                # 2) desscala:    µ  = shift+scale     σ = somente scale
                mu_rate_phys = scaler_target.inverse_transform(mu_rate.reshape(-1, 1)).ravel()
            
                scale = getattr(scaler_target, "scale_", [1.0])[0]   # Safe fallback
                sig_rate_phys = sig_rate * scale                     # sem shift!
            
                # 3) curva P10/P90 por passo
                plo, phi = band
                low_rate  = scenario_curve(mu_rate_phys, sig_rate_phys, plo)
                high_rate = scenario_curve(mu_rate_phys, sig_rate_phys, phi)
            
                # 4) acumula e alinha ao primeiro ponto da série cumulativa
                adj = truth[0] - mu_rate_phys[0]
                low_cum  = np.cumsum(low_rate)  + adj
                high_cum = np.cumsum(high_rate) + adj
                manual_env = (low_cum, high_cum)

            # wrapper decide o resto
            plot_predictions_wrapper(
                ensemble_out,
                truth=truth,
                kind=kind,
                well=well,
                band=band,
                mean_override=pred if not is_cum else pred,  # mantém curva média
                manual_envelope=manual_env,
                is_cum = is_cum,
                **common_kw,
            )
        # --- fallback legado ----------------------------------------
        else:
            from evaluation.evaluation import evaluate_and_plot
            evaluate_and_plot(
                y_true=truth,
                y_pred=pred,
                title=f"{label} – {title_suffix}",
                well=well,
                set_name=title_suffix,
                additional_params=dict(
                    window_size=window_size,
                    forecast_steps=steps,
                    percentage_split=pct_split,
                ),
            )

    # ======================= EVALUATION HELPERS (Modified) =========================
    # These helpers are slightly modified to return the raw result dictionaries,
    # which contain the unscaled predictions needed for the integrated plot.

    def _eval_seq(y_true, y_pred, y_train, split):
        res = evaluate_model_seq(
            y_true, y_pred, scaler_target,
            params["lag_window"], params["horizon"],
            1 - params["test_size"], config,
            eval_title="Seq-to-Seq", set_name=label,
            aggregation_method=params.get("aggregation_method"),
            quantiles=params.get("aggregation_quantiles"),
            plot=False,
        )
        agg_y, agg_pred, gm = res["agg_y_test"], res["agg_y_pred"], res["global_metrics"]
        _plot_seq(agg_y, agg_pred, "Aggregated", split=split)
        cum = evaluate_cumulative_seq(agg_y, agg_pred, y_train, scaler_target, params["lag_window"], params["horizon"], config=config, plot=False)
        _plot_seq(cum["y_test_cumsum"], cum["y_pred_cumsum"], "Cumulative", is_cum=True, split=split)

        df_agg = pd.DataFrame([compute_metrics_to_df_seq(agg_y, agg_pred, well, arch, "Aggregated")]).assign(**base_tags)
        df_agg["Kind"] = "Aggregated_Window"
        df_cum = pd.DataFrame([compute_metrics_to_df_seq(cum["y_test_cumsum"], cum["y_pred_cumsum"], well, arch, "Cumulative")]).assign(**base_tags)
        df_cum["Kind"] = "Cumulative_Sum"
        gm_full = {**base_tags, **gm, "Category": "Global", "Kind": "Overall"}
        
        # MODIFICATION: Return `res` to provide access to unscaled predictions
        return df_agg, df_cum, gm_full, res

    def _eval_value(y_true, y_pred, y_train, split):
        r2, smape, mae = evaluate_model(y_true, y_pred, scaler_target, params["lag_window"], params["horizon"], 1 - params["test_size"], set_name=label)
        _plot_seq(y_true, y_pred, "Point Forecast", split=split)
        y_inv_cum, y_pred_inv_cum = evaluate_cumulative(y_true, y_pred, y_train, config=config, set_name=label)
        _plot_seq(y_inv_cum, y_pred_inv_cum, "Cumulative", is_cum=True, split=split)
        
        df_reg = pd.DataFrame([compute_metrics_to_df(y_true, y_pred, well, arch, "Series")]).assign(**base_tags)
        df_cum = pd.DataFrame([compute_metrics_to_df(y_inv_cum, y_pred_inv_cum, well, arch, "Series")]).assign(**base_tags)
        gm_full = {**base_tags, "R²": r2, "SMAPE": smape, "MAE": mae, "Category": "Global", "Kind": "Overall"}

        # MODIFICATION: Create and return a result dict to standardize output
        y_true_unscaled = scaler_target.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred_unscaled = scaler_target.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        res = {"agg_y_test": y_true_unscaled, "agg_y_pred": y_pred_unscaled}
        return df_reg, df_cum, gm_full, res

    # ======================= DISPATCHER (Modified) ==============================
    # Captures the raw result dictionaries (`res_test`, `res_val`) for later use.
    if arch in SEQ2SEQ_ARCHS:
        agg_test, cum_test, gm_test, res_test = _eval_seq(y_test_scaled, y_test_pred, y_train_original, split="test")
        agg_val,  cum_val,  gm_val, res_val   = _eval_seq(y_val_scaled,  y_val_pred,  y_train_original, split="val")
    else:
        agg_test, cum_test, gm_test, res_test = _eval_value(y_test_scaled, y_test_pred, y_train_original, split="test")
        agg_val,  cum_val,  gm_val, res_val   = _eval_value(y_val_scaled,  y_val_pred,  y_train_original, split="val")

    # =================================================================================
    ### NEW SECTION: INTEGRATED PLOTTING HELPER ###
    # This new, self-contained helper function prepares data for and calls the
    # integrated plotting function without cluttering the main logic.
    # =================================================================================
    def _prepare_and_plot_integrated_view():
        """Assembles data and calls the new integrated plot functions."""
        if not plot:
            return

        # 1. Extract unscaled predictions and ground truths from the captured results
        y_val_pred_unscaled = res_val["agg_y_pred"]
        y_test_pred_unscaled = res_test["agg_y_pred"]
        y_val_actual_unscaled = res_val["agg_y_test"]
        y_test_actual_unscaled = res_test["agg_y_test"]

        # 2. Define split boundaries for the full timeline
        len_train = len(y_train_original)
        len_val = len(y_val_actual_unscaled)
        len_test = len(y_test_actual_unscaled)
        total_len = len_train + len_val + len_test
        split_indices = {'train_end': len_train, 'val_end': len_train + len_val}
        x_axis = np.arange(total_len)

        # 3. Assemble full ground truth series (Train + Val + Test)
        print('SHAPE')
        y_actual_full = np.concatenate([y_train_original[:, -1], y_val_actual_unscaled, y_test_actual_unscaled])

        # 4. Assemble prediction series with NaN padding for plotting gaps
        y_pred_val_full = np.full(total_len, np.nan)
        y_pred_val_full[split_indices['train_end']:split_indices['val_end']] = y_val_pred_unscaled
        y_pred_test_full = np.full(total_len, np.nan)
        y_pred_test_full[split_indices['val_end']:] = y_test_pred_unscaled
        
        # 5. Prepare metrics dictionaries for plot annotations
        metrics_for_val_plot = {"SMAPE": gm_val.get("SMAPE"), "MAE": gm_val.get("MAE")}
        metrics_for_test_plot = {"SMAPE": gm_test.get("SMAPE"), "MAE": gm_test.get("MAE")}
        
        # 6. --- PLOT 1: Integrated Series View ---
        plot_integrated_view(
            x_axis=x_axis, y_actual=y_actual_full, y_pred_val=y_pred_val_full, y_pred_test=y_pred_test_full,
            split_indices=split_indices, metrics_val=metrics_for_val_plot, metrics_test=metrics_for_test_plot,
            title=label, yaxis_title="Rate", well=well
        )
        
        # 7. Prepare data for the cumulative plot, anchoring predictions correctly
        y_actual_full_cum = np.cumsum(y_actual_full)
        
        # Anchor validation cumulative sum to the end of the training actuals
        y_pred_val_cum = np.full(total_len, np.nan)
        val_anchor_point = y_actual_full_cum[split_indices['train_end'] - 1] if len_train > 0 else 0
        val_pred_cum_segment = val_anchor_point + np.cumsum(y_val_pred_unscaled)
        y_pred_val_cum[split_indices['train_end']:split_indices['val_end']] = val_pred_cum_segment
        
        # Anchor test cumulative sum to the end of the validation actuals
        y_pred_test_cum = np.full(total_len, np.nan)
        test_anchor_point = y_actual_full_cum[split_indices['val_end'] - 1] if len_val > 0 else val_anchor_point
        test_pred_cum_segment = test_anchor_point + np.cumsum(y_test_pred_unscaled)
        y_pred_test_cum[split_indices['val_end']:] = test_pred_cum_segment

        metrics_cum_val = cum_val[["SMAPE", "MAE"]].iloc[0].to_dict()
        metrics_cum_test = cum_test[["SMAPE", "MAE"]].iloc[0].to_dict()

        # 8. --- PLOT 2: Integrated Cumulative View ---
        plot_integrated_view(
            x_axis=x_axis, y_actual=y_actual_full_cum, y_pred_val=y_pred_val_cum, y_pred_test=y_pred_test_cum,
            split_indices=split_indices, 
            # MODIFICAÇÃO: Passa os dicionários de métricas para o plot
            metrics_val=metrics_cum_val, 
            metrics_test=metrics_cum_test,
            title=label, # Este título será atualizado no próximo patch
            yaxis_title="Cumulative Sum", well=well
        )

    # Final call to the new plotting logic before returning results
    _prepare_and_plot_integrated_view()
    
    # The original return statement remains unchanged
    return agg_test, cum_test, gm_test, agg_val, cum_val, gm_val




def evaluate_slices(
    y_test_full: np.ndarray,
    y_pred_test_full: np.ndarray,
    y_val_full: np.ndarray,
    y_pred_val_full: np.ndarray,
    scaler_target,
    y_train_original: np.ndarray,
    params: dict,
    ds_config: dict,
    well: str
) -> tuple[
    list, list, list,
    list, list, list
]:
    """
    Compute slice-based metrics on both test and validation sets.

    Returns six lists:
      slice_agg_test, slice_cum_test, slice_glob_test,
      slice_agg_val,  slice_cum_val,  slice_glob_val
    """
    from math import ceil
    import matplotlib.pyplot as plt

    slice_agg_test, slice_cum_test, slice_glob_test = [], [], []
    slice_agg_val,  slice_cum_val,  slice_glob_val  = [], [], []
    total = len(y_test_full)

    for q in params.get("slice_ratios", []):
        n = int(ceil(total * q))
        if n <= 0:
            continue

        # Test slice
        y_test_slice = y_test_full[:n]
        pred_test_slice = y_pred_test_full[:n]

        # Validation slice
        y_val_slice = y_val_full[:n]
        pred_val_slice = y_pred_val_full[:n]

        # Evaluate slices using evaluate_job
        (
            agg_test_df, cum_test_df, gm_test,
            agg_val_df, cum_val_df, gm_val
        ) = evaluate_job(
            y_test_slice,
            pred_test_slice,
            y_val_slice,
            pred_val_slice,
            scaler_target,
            y_train_original,
            params,
            ds_config,
            well,
            plot=False
        )

        tag = f"{int(q * 100)}%"
        # Annotate categories
        gm_test['Category'] = f"Global {tag}"
        agg_test_df['Category'] = f"Aggregated {tag}"
        cum_test_df['Category'] = f"Cumulative {tag}"  

        gm_val['Category'] = f"Global {tag}"
        agg_val_df['Category'] = f"Aggregated {tag}"
        cum_val_df['Category'] = f"Cumulative {tag}"  

        # Append to lists
        slice_glob_test.append(gm_test)
        slice_agg_test.append(agg_test_df)
        slice_cum_test.append(cum_test_df)

        slice_glob_val.append(gm_val)
        slice_agg_val.append(agg_val_df)
        slice_cum_val.append(cum_val_df)

    # Close any figures
    if params.get('plot', False):
        plt.close('all')

    return (
        slice_agg_test, slice_cum_test, slice_glob_test,
        slice_agg_val,  slice_cum_val,  slice_glob_val
    )

# def run_single_job(job):

#     """Orquestra um job completo, agora compatível com cenários/ band."""
#     try:
#         # 1. preparação -------------------------------------------------
#         (
#             train_kwargs, X_test, y_test_scaled, scaler_target,
#             y_train_original, params, ds, well, job_id
#         ) = prepare_job_data(job)            # type: ignore

#         params = {**DEFAULT_EXP_PARAMS, **params}   # garante chaves novas
#         X_val, y_val_scaled = train_kwargs["X_val"], train_kwargs["y_val"]
#         data_inputs = {"X_test": X_test, "X_val": X_val}

#         # 2. inferência ensemble ---------------------------------------
#         ensemble_raw = process_chunks(train_kwargs, data_inputs, params, scaler_target)  # type: ignore
#         ensemble = to_ensemble_output(ensemble_raw)

#         # 2½. curva p/ métricas (P50)  +  cenário p/ visual -------------
#         plot_kind = params.get("scenario", "P50").upper()
#         if plot_kind in ("P90", "P10", "BAND") and ensemble.sigma_test is None:
#             logging.warning("Sigma indisponível; degradando cenário %s para P50", plot_kind)
#             plot_kind = "P50"

#         params["__plot_kind__"] = plot_kind                # usado no _maybe_plot
#         final_test_pred = ensemble.pred_test               # métricas sempre na média
#         final_val_pred  = ensemble.pred_val

#         # 3. avaliação --------------------------------------------------
#         (
#             agg_test_df, cum_test_df, gm_test,
#             agg_val_df,  cum_val_df,  gm_val,
#         ) = evaluate_job(
#             y_test_scaled, final_test_pred,
#             y_val_scaled,  final_val_pred,
#             scaler_target, y_train_original,
#             params, ds, well,
#             plot=params.get("plot", False),
#             ensemble_out=ensemble,
#         )

#         # 4. métricas por slice ----------------------------------------
#         slice_agg_test, slice_cum_test, slice_glob_test = [], [], []
#         slice_agg_val,  slice_cum_val,  slice_glob_val  = [], [], []
#         if params.get("evaluate_by_slice", False):
#             (
#                 slice_agg_test, slice_cum_test, slice_glob_test,
#                 slice_agg_val,  slice_cum_val,  slice_glob_val
#             ) = evaluate_slices(  # type: ignore
#                 y_test_scaled, final_test_pred,
#                 y_val_scaled,  final_val_pred,
#                 scaler_target, y_train_original,
#                 params, ds, well,
#             )

#         print("cumulative_metrics_val", cum_val_df,)
#         print("aggregated_metrics_val", agg_val_df,)

#         # 5. resultado --------------------------------------------------
#         result = {
#             "status": "success",
#             "aggregated_metrics_test": agg_test_df,
#             "cumulative_metrics_test": cum_test_df,
#             "global_metrics_test": gm_test,
#             "aggregated_metrics_val": agg_val_df,
#             "cumulative_metrics_val": cum_val_df,
#             "global_metrics_val": gm_val,
#             "slice_agg_test": slice_agg_test,
#             "slice_cum_test": slice_cum_test,
#             "slice_glob_test": slice_glob_test,
#             "slice_agg_val":  slice_agg_val,
#             "slice_cum_val":  slice_cum_val,
#             "slice_glob_val": slice_glob_val,
#             "well": well,
#             "experiment_id": job_id,
#             # "ensemble_outputs": ensemble,
#         }

#     except Exception as e:
#         logging.exception("Job failed")
#         result = {
#             "status": "failure",
#             "error": str(e),
#             "well": job[1],
#             "experiment_id": job[3],
#         }
#     finally:
#         import gc; gc.collect()

#     return result


from profile_manager import generate_job_hash 
from forecast_pipeline.io_utils import atomic_write_json, build_run_metadata


# In src/forecast_pipeline/jobs.py

def normalize_job_parameters(initial_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes a job's parameter dictionary and ensures it has the final,
    unified structure needed by the entire pipeline.
    
    This function is now ARCHITECTURE-AWARE.
    """
    params = initial_params.copy()

    # --- 1. Ensure `architecture_name` exists ---
    # This is the most critical key, as it drives all other logic.
    if "architecture_name" not in params:
        # Fallback to a default if it's somehow missing
        params["architecture_name"] = "Seq2Context" 
        logging.warning("'architecture_name' not in params, falling back to 'Seq2Context'.")
    
    arch_name = params["architecture_name"]

    # --- 2. Ensure `strategy_config` exists ---
    # This is required by all Seq2* models.
    if "strategy_config" not in params:
        if "physics_strategy" not in params:
            raise KeyError("Job parameters must contain either 'strategy_config' or 'physics_strategy'.")
        params["strategy_config"] = {"strategy_name": params["physics_strategy"]}

    # --- 3. Conditionally ensure `extractor_config` and `fuser_config` exist ---
    # This is the key fix. We only check for these keys if the architecture requires them.
    if arch_name in ["Seq2Context", "Seq2Fuser"]:
        if "extractor_config" not in params:
            raise KeyError(
                f"Job parameters for architecture '{arch_name}' are missing 'extractor_config'."
            )
        if "fuser_config" not in params:
            raise KeyError(
                f"Job parameters for architecture '{arch_name}' are missing 'fuser_config'."
            )
    
    return params

def run_single_job(
    job: Tuple, 
    persist_result: bool = False  # <-- NEW: Flag to control behavior
) -> Dict[str, Any]:
    """
    Orchestrates a complete job and evaluates it.

    If `persist_result` is True, it saves the full result to a unique JSON file
    and returns a lightweight summary dictionary.

    If `persist_result` is False (default), it returns the original, large result
    dictionary with DataFrames for full backward compatibility with the legacy pipeline.
    """
    ds, well, params, experiment_id_from_profile = job
    
    # Generate a hash for logging and potential filename
    job_hash = generate_job_hash(params)
    
    try:
        # --- Steps 1 & 2: Your core logic (unchanged) ---
        (train_kwargs, X_test, y_test_scaled, scaler_target, y_train_original, _, _, _, _) = prepare_job_data(job)
        params = {**DEFAULT_EXP_PARAMS, **params}
        params = normalize_job_parameters(params)
        X_val, y_val_scaled = train_kwargs["X_val"], train_kwargs["y_val"]
        data_inputs = {"X_test": X_test, "X_val": X_val}
        ensemble_raw = process_chunks(train_kwargs, data_inputs, params, scaler_target)
        ensemble = to_ensemble_output(ensemble_raw)
        final_test_pred = ensemble.pred_test
        final_val_pred  = ensemble.pred_val

        # --- Step 3: Evaluation (Fixed) ---
        # The call to evaluate_job now has all its arguments correctly passed.
        (
            agg_test_df, cum_test_df, gm_test,
            agg_val_df,  cum_val_df,  gm_val,
        ) = evaluate_job(
            y_test_scaled=y_test_scaled, 
            y_test_pred=final_test_pred,
            y_val_scaled=y_val_scaled,  
            y_val_pred=final_val_pred,
            scaler_target=scaler_target, 
            y_train_original=y_train_original,
            params=params, 
            config=ds, # Pass the data source config `ds` here
            well=well,
            plot=params.get("plot", False),
            ensemble_out=ensemble,
        )

        # --- Step 4: Slice Metrics (unchanged) ---
        if params.get("evaluate_by_slice", False):
            (
                slice_agg_test, slice_cum_test, slice_glob_test,
                slice_agg_val,  slice_cum_val,  slice_glob_val
            ) = evaluate_slices(  # type: ignore
                y_test_scaled, final_test_pred,
                y_val_scaled,  final_val_pred,
                scaler_target, y_train_original,
                params, ds, well,
            )
        else:
            slice_agg_test, slice_cum_test, slice_glob_test = [], [], []
            slice_agg_val,  slice_cum_val,  slice_glob_val  = [], [], []
        
        # --- Step 5: Assemble the original result dictionary ---
        # This is the format your legacy pipeline expects.
        original_result_dict = {
            "status": "success",
            "aggregated_metrics_test": agg_test_df,
            "cumulative_metrics_test": cum_test_df,
            "global_metrics_test": gm_test,
            "aggregated_metrics_val": agg_val_df,
            "cumulative_metrics_val": cum_val_df,
            "global_metrics_val": gm_val,
            "slice_agg_test": slice_agg_test, "slice_cum_test": slice_cum_test, "slice_glob_test": slice_glob_test,
            "slice_agg_val":  slice_agg_val,  "slice_cum_val":  slice_cum_val,  "slice_glob_val": slice_glob_val,
            "well": well, "experiment_id": experiment_id_from_profile,
        }

        # --- Step 6: Decide what to do based on the 'persist_result' flag ---
        if persist_result:
            # --- NEW PATH: Save to disk and return summary ---
            key_metrics = {
                "val_smape_agg": agg_val_df['SMAPE'].iloc[0] if not agg_val_df.empty else None,
                "val_smape_cum": cum_val_df['SMAPE'].iloc[0] if not cum_val_df.empty else None,
            }
            
            # Convert DataFrames to dicts for JSON serialization
            serializable_results = {k: (v.to_dict(orient='records') if hasattr(v, 'to_dict') else v) for k, v in original_result_dict.items()}
            # Handle list of DFs in slices
            for key in ["slice_agg_test", "slice_cum_test", "slice_agg_val", "slice_cum_val"]:
                serializable_results[key] = [df.to_dict(orient='records') for df in serializable_results[key]]

            persistence_dict = {
                "status": "success", "job_hash": job_hash, "experiment_id": experiment_id_from_profile,
                "well": well, "key_metrics": key_metrics, "config": params,
                "metadata": build_run_metadata(), "results": serializable_results
            }
            
            output_dir = params.get("run_output_dir")
            if not output_dir: raise ValueError("'run_output_dir' is required when persist_result=True.")
            
            output_file = Path(output_dir) / f"{job_hash}.json"
            atomic_write_json(persistence_dict, output_file)

            return {
                "status": "success", "job_hash": job_hash, "well": well, 
                "experiment_id": experiment_id_from_profile, "output_path": str(output_file), **key_metrics
            }
        else:
            return original_result_dict

    except Exception as e:
        logging.exception(f"Job with hash {job_hash} for well {well} failed.")
        # This failure dictionary is compatible with both old and new pipelines.
        failure_report = {
            "status": "failure",
            "error": str(e),
            "well": well,
            "experiment_id": experiment_id_from_profile,
        }
        # If in persistence mode, also try to save a failure report to disk.
        if persist_result:
            output_dir = params.get("run_output_dir")
            if output_dir:
                failure_report.update({"job_hash": job_hash, "config": params, "metadata": build_run_metadata()})
                output_file = Path(output_dir) / f"{job_hash}.json"
                atomic_write_json(failure_report, output_file)
        
        return failure_report
        
    finally:
        gc.collect()



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
    logging.info("No filter methods provided; running with no adaptive filtering.")
    return [{"apply_adaptive_filtering": False, "filter_method": None, "filter_kwargs": {}}]
