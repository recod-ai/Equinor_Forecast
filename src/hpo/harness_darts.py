# hpo/harness_darts.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import logging, numpy as np, pandas as pd
from evaluation.evaluation import smape

from common.config_wells import get_data_sources
from forecast_pipeline.jobs import prepare_job_data
from training.train_darts import main_train_darts_model
from forecast_pipeline.plotting import plot_darts_integrated
from common.eval_darts import latest_from_ribbons, _inv_ts_1d, _inv_ribbons_2d, log_split_lengths, summarize_ribbons_shape
from common.log_table import log_grid_table
from hpo.grids_darts import make_search_grid
from typing import Sequence, Mapping

def base_feature_config(dataset_name: str, architecture_name: str) -> Dict[str, Any]:
    if architecture_name.startswith("Darts_"):
        return {"selected_features": ["PI", "AVG_DOWNHOLE_PRESSURE", "Tempo_Inicio_Prod", "BORE_OIL_VOL"]}
    return {}
    
def run_one_job(
    *,
    dataset_name: str,
    well_name: str,
    architecture_name: str,   # e.g., "Darts_TiDE"
    base_params: Dict[str, Any],
    profile_params: Dict[str, Any],
    job_id: str = "adhoc",
    do_plot: bool = True,
) -> Dict[str, Any]:
    # Resolve datasource
    ds_config = next(d for d in get_data_sources() if d["name"] == dataset_name)

    # Merge params
    cfg = {
        **base_params,
        **base_feature_config(dataset_name, architecture_name),
        **profile_params,
        "architecture_name": architecture_name,
        "target_column": ds_config.get("target_column", "BORE_OIL_VOL"),
    }

    # Load data via the unified gateway (ExperimentDarts behind the scenes)
    prep = prepare_job_data((ds_config, well_name, cfg, job_id))
    train_kwargs, prediction_input, y_test_u, scaler_X, scaler_target, y_train_u, params, *_rest = \
        (prep + (None,)*10)[:10]  # tolerate older returns; we need first 7

    # sanity: ensure scalers live in train_kwargs (jobs.py already does this)
    train_kwargs.setdefault("scaler_target", scaler_target)
    train_kwargs.setdefault("scaler_X", scaler_X)

    log_split_lengths(train_kwargs, prediction_input, title="SPLIT LENGTHS (after prep)")

    # Train
    model, _hist, pred_test_np, pred_val_np = main_train_darts_model(
        architecture_name=architecture_name,
        train_kwargs=train_kwargs,
        data_inputs=prediction_input,
        epochs=cfg.get("n_epochs"),
        batch_size=cfg.get("batch_size"),
        patience=cfg.get("patience"),
        learning_rate=cfg.get("learning_rate"),
    )

    # Metrics on original units (mirror the integrated plot)
    main_col = train_kwargs["main_col"]
    scaler   = train_kwargs["scaler_target"]
    y_train  = _inv_ts_1d(train_kwargs["X_train"][main_col], scaler)
    y_val    = _inv_ts_1d(train_kwargs["X_val"][main_col],   scaler)
    y_test   = _inv_ts_1d(prediction_input["ts_test"][main_col], scaler)

    val_rib_u  = _inv_ribbons_2d(pred_val_np,  scaler)
    test_rib_u = _inv_ribbons_2d(pred_test_np, scaler)

    yhat_val   = latest_from_ribbons(val_rib_u)[: len(y_val)]
    yhat_test  = latest_from_ribbons(test_rib_u)[: len(y_test)]

    smape_val  = smape(y_val,  yhat_val)
    smape_test = smape(y_test, yhat_test)

    logging.info("RIBBONS: " + summarize_ribbons_shape(pred_val_np, pred_test_np))
    logging.info(f"sMAPE — val: {smape_val:.2f}% | test: {smape_test:.2f}%")

    if do_plot:
        plot_darts_integrated(
            train_kwargs=train_kwargs,
            prediction_input=prediction_input,
            pred_val_ribbons=pred_val_np,
            pred_test_ribbons=pred_test_np,
            title_prefix=f"{architecture_name} — {well_name} ({profile_params.get('profile','unnamed')})",
        )

    # Collect artifacts
    art = {
        "job_id": job_id,
        "dataset": dataset_name,
        "well": well_name,
        "architecture": architecture_name,
        "profile": profile_params.get("profile", ""),
        "status": "ok",
        "smape_val": smape_val,
        "smape_test": smape_test,
        "pred_val": np.asarray(pred_val_np),
        "pred_test": np.asarray(pred_test_np),
        "train_kwargs": train_kwargs,
        "prediction_input": prediction_input,
        "params": cfg,
        "model": model,
    }
    return art

def run_grid(
    *,
    dataset_name: str,
    well_name: str,
    model_key: str,            # e.g., "TiDE"
    base_params: Dict[str, Any],
    limit: Optional[int] = None,
    do_plot_each: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Runs the curated grid for a single Darts model key (e.g. 'TiDE') and returns
    a results DataFrame (sMAPE metrics, status) plus an artifacts dict keyed by job_id.
    """
    architecture_name = f"Darts_{model_key}"
    grid = make_search_grid(model_key, base_params)
    if limit is not None:
        grid = grid[:limit]
    log_grid_table(f"Search grid for {architecture_name}", grid)

    rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, Dict[str, Any]] = {}

    for i, profile in enumerate(grid, start=1):
        job_id = f"grid_{model_key}_{i:03d}"
        try:
            art = run_one_job(
                dataset_name=dataset_name,
                well_name=well_name,
                architecture_name=architecture_name,
                base_params=base_params,
                profile_params=profile,
                job_id=job_id,
                do_plot=do_plot_each,
            )
            artifacts[job_id] = art
            rows.append({
                "job_id": job_id,
                "dataset": dataset_name,
                "well": well_name,
                "architecture": architecture_name,
                "profile": art["profile"],
                "smape_val": round(art["smape_val"], 4),
                "smape_test": round(art["smape_test"], 4),
                "status": "ok",
            })
        except Exception as e:
            logging.exception("Job failed: %s", job_id)
            rows.append({
                "job_id": job_id,
                "dataset": dataset_name,
                "well": well_name,
                "architecture": architecture_name,
                "profile": profile.get("profile", ""),
                "smape_val": np.nan,
                "smape_test": np.nan,
                "status": "error",
                "error": repr(e),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["status", "smape_test"], ascending=[True, True], na_position="last").reset_index(drop=True)
    return df, artifacts

def run_model_suite(
    *,
    dataset_name: str,
    well_name: str,
    model_keys: Sequence[str],
    base_params: Dict[str, Any],
    per_model_limit: Optional[Mapping[str, Optional[int]]] = None,
    do_plot_each: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows, arts_ns = [], {}
    for key in model_keys:
        lim = (per_model_limit or {}).get(key)  # can be None
        logging.info(f"=== Running Darts_{key} grid (limit={lim}) ===")
        df, arts = run_grid(
            dataset_name=dataset_name,
            well_name=well_name,
            model_key=key,
            base_params=base_params,
            limit=lim,
            do_plot_each=do_plot_each,
        )
        df["model_key"] = key
        rows.append(df)
        arts_ns.update({f"{key}:{k}": v for k, v in arts.items()})
    combined = (pd.concat(rows, ignore_index=True)
                  .sort_values(["status", "model_key", "smape_test"], ascending=[True, True, True])
                  .reset_index(drop=True))
    return combined, arts_ns
