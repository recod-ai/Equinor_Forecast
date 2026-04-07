# src/forecast_pipeline/metrics.py
import pandas as pd
import numpy as np

import logging 
from typing import List, Dict, Any, Tuple 
from .config import ( _COLS_TO_DROP_ALWAYS, _COLS_TO_DROP_FILTER, _METRIC_COLS_ORDER, _BASE_ORDER, _FILTER_ORDER )
import json
from pathlib import Path
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

def _process_dataframe(
    df: pd.DataFrame,
    name: str,
    is_slice: bool,
    filter_tags: Dict[str, Any],
    remove_filter_cols: bool
) -> pd.DataFrame:
    """
    Process, clean, and reorder a single Raw DataFrame.
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()

    if not remove_filter_cols:
        df = df.assign(**filter_tags)

    if not is_slice and 'Category' not in df.columns:
        df['Category'] = name.capitalize()

    cols_to_drop = _COLS_TO_DROP_ALWAYS + (_COLS_TO_DROP_FILTER if remove_filter_cols else [])
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    order_template = _BASE_ORDER if remove_filter_cols else _FILTER_ORDER
    target_order   = order_template + _METRIC_COLS_ORDER

    current_cols = list(df.columns)
    ordered      = [c for c in target_order if c in current_cols]
    remaining    = sorted([c for c in current_cols if c not in ordered])

    return df[ordered + remaining]


def collate_metrics(
    raw_results: List[Dict[str, Any]]
) -> Dict[str, pd.DataFrame]:
    """
    Collate raw result dicts into merged DataFrames that include both test and validation metrics.
    """
    import logging
    import pandas as pd

    # Collect per-model metrics
    g_test, g_val = [], []
    a_test, a_val = [], []
    c_test, c_val = [], []
    slice_g_test, slice_a_test, slice_c_test = [], [], []

    for r in raw_results:
        if r.get("status") == "success":
            if "global_metrics_test" in r:
                g_test.append(r["global_metrics_test"])
            if "global_metrics_val" in r:
                g_val.append(r["global_metrics_val"])
            if "aggregated_metrics_test" in r:
                a_test.append(r["aggregated_metrics_test"])
            if "aggregated_metrics_val" in r:
                a_val.append(r["aggregated_metrics_val"])
            if "cumulative_metrics_test" in r:
                c_test.append(r["cumulative_metrics_test"])
            if "cumulative_metrics_val" in r:
                c_val.append(r["cumulative_metrics_val"])
            # slice-level (test only)
            slice_g_test.extend(r.get("slice_glob_test", []))
            slice_a_test.extend(r.get("slice_agg_test", []))
            slice_c_test.extend(r.get("slice_cum_test", []))
        else:
            if r.get("status") == "failure":
                logging.warning(f"Skipping failed job {r.get('experiment_id')} for well {r.get('well')}")
            else:
                logging.warning(f"Unexpected result format: {r}")

    # Build DataFrames
    df_g_test = pd.DataFrame(g_test) if g_test else pd.DataFrame()
    df_g_val  = pd.DataFrame(g_val)  if g_val  else pd.DataFrame()
    df_a_test = pd.concat(a_test, ignore_index=True) if a_test else pd.DataFrame()
    df_a_val  = pd.concat(a_val,  ignore_index=True) if a_val  else pd.DataFrame()
    df_c_test = pd.concat(c_test, ignore_index=True) if c_test else pd.DataFrame()
    df_c_val  = pd.concat(c_val,  ignore_index=True) if c_val  else pd.DataFrame()

    # Helper to merge test/val
    def _merge(val_df, test_df):
        if test_df.empty or val_df.empty:
            return test_df if not test_df.empty else val_df
        common = set(test_df.columns).intersection(val_df.columns)
        metric_cols = {"R²", "SMAPE", "MAE"}
        join_keys = [c for c in common if c not in metric_cols and c != "Set"]
        return val_df.merge(
            test_df,
            on=join_keys,
            suffixes=("_VAL", "_TEST"),
            how="outer"
        )

    df_global = _merge(df_g_val, df_g_test)
    df_agg    = _merge(df_a_val, df_a_test)
    df_cum    = _merge(df_c_val, df_c_test)

    # slice-level DataFrames
    df_slice_global = pd.DataFrame(slice_g_test) if slice_g_test else pd.DataFrame()
    df_slice_agg    = pd.concat(slice_a_test, ignore_index=True) if slice_a_test else pd.DataFrame()
    df_slice_cum    = pd.concat(slice_c_test, ignore_index=True) if slice_c_test else pd.DataFrame()

    return {
        'df_global': df_global,
        'df_agg':    df_agg,
        'df_cum':    df_cum,
        'df_slice_global': df_slice_global,
        'df_slice_agg':    df_slice_agg,
        'df_slice_cum':    df_slice_cum
    }


def clean_and_structure_results(
    df_global: pd.DataFrame,
    df_agg: pd.DataFrame,
    df_cum: pd.DataFrame,
    df_slice_global: pd.DataFrame,
    df_slice_agg: pd.DataFrame,
    df_slice_cum: pd.DataFrame,
    filter_tags: Dict[str, Any],
    remove_cols: bool
) -> Dict[str, pd.DataFrame]:
    """
    Organize and clean combined result DataFrames into a structured format.
    """
    return {
        'global_metrics': _process_dataframe(df_global, 'global', False, filter_tags, remove_cols),
        'aggregated_metrics': _process_dataframe(df_agg, 'aggregated', False, filter_tags, remove_cols),
        'cumulative_metrics': _process_dataframe(df_cum, 'cumulative', False, filter_tags, remove_cols),
        'global_quantiles': _process_dataframe(df_slice_global, 'global', True, filter_tags, remove_cols),
        'aggregated_quantiles': _process_dataframe(df_slice_agg, 'aggregated', True, filter_tags, remove_cols),
        'cumulative_quantiles': _process_dataframe(df_slice_cum, 'cumulative', True, filter_tags, remove_cols)
    }




def collate_robust_results(run_output_dir: str | Path) -> pd.DataFrame:
    """
    Scans a run directory for all atomic JSON result files, aggregates them
    into a single "leaderboard" DataFrame.

    This function is the cornerstone of post-experiment analysis in the new pipeline.

    Args:
        run_output_dir: The path to the main timestamped directory for a specific run
                        (e.g., 'experiments/2025-07-15_10-08-54_...').

    Returns:
        A pandas DataFrame containing a leaderboard of all successful experiments.
        Returns an empty DataFrame if no successful results are found.
    """
    run_path = Path(run_output_dir)
    results_dir = run_path / "results"
    
    if not results_dir.is_dir():
        logging.warning(f"Results directory not found at the expected path: {results_dir}")
        return pd.DataFrame()

    all_results: List[Dict[str, Any]] = []
    
    json_files = list(results_dir.glob("*.json"))
    logging.info(f"Found {len(json_files)} result files in {results_dir}. Processing...")

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from file: {json_file}. Skipping.")
                continue
        
        # Only process results from successfully completed jobs
        if data.get("status") != "success":
            continue
        
        # --- Flatten the nested JSON into a single row for the DataFrame ---
        # Start with the top-level identifying information
        config_to_add = data.get("config", {})
        flat_row = {
            "experiment_id": data.get("experiment_id"),
            "well": data.get("well"),
            "job_hash": data.get("job_hash"),
            "optuna_trial_number": config_to_add.get("optuna_trial_number"),
        }

        print('data.get("key_metrics", {})', data.get("key_metrics", {}))
        
        # Add the easily accessible key metrics for sorting and quick analysis
        flat_row.update(data.get("key_metrics", {}))
        
        # Add the full experiment configuration, excluding redundant or overly complex keys
        config_to_add = data.get("config", {})
        # We can exclude very large or nested items from the main leaderboard for clarity
        excluded_keys = ["selected_features", "extractor_config", "fuser_config", "run_output_dir"]
        for key, value in config_to_add.items():
            if key not in excluded_keys and not isinstance(value, (dict, list)):
                flat_row[key] = value

        all_results.append(flat_row)
            
    if not all_results:
        logging.warning("No successful jobs found to create a leaderboard.")
        return pd.DataFrame()
        
    # Convert the list of dictionaries to a DataFrame
    leaderboard_df = pd.DataFrame(all_results)

        # --- Canonicalize column names so Seq2* and Darts look the same downstream ---
    # epochs
    if "epochs" not in leaderboard_df.columns and "n_epochs" in leaderboard_df.columns:
        leaderboard_df["epochs"] = leaderboard_df["n_epochs"]

    # architecture (stable for analysis/plots)
    if "architecture" not in leaderboard_df.columns and "architecture_name" in leaderboard_df.columns:
        leaderboard_df["architecture"] = leaderboard_df["architecture_name"]

    # lag_window / horizon (map from Darts' input/output chunk lengths)
    if "lag_window" not in leaderboard_df.columns and "input_chunk_length" in leaderboard_df.columns:
        leaderboard_df["lag_window"] = leaderboard_df["input_chunk_length"]

    if "horizon" not in leaderboard_df.columns and "output_chunk_length" in leaderboard_df.columns:
        leaderboard_df["horizon"] = leaderboard_df["output_chunk_length"]

    
    return leaderboard_df


# forecast_pipeline/analytics.py
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def _smape_from_global(obj: Any) -> float | None:
    """Extracts SMAPE from a 'global_metrics_*' block (dict)."""
    try:
        if isinstance(obj, dict) and "SMAPE" in obj:
            return float(obj["SMAPE"])
    except Exception:
        pass
    return None

def _smape_from_list_or_df(obj: Any) -> float | None:
    """
    Extracts SMAPE from an 'aggregated/cumulative_metrics_*' block, which can be:
      - a list[dict] with a single row
      - a serialized dict (e.g., {'SMAPE':[...]} or {'SMAPE': value})
      - a serialized DataFrame (uncommon, but handled defensively)
    """
    try:
        # list of dicts
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            if "SMAPE" in obj[0]:
                return float(obj[0]["SMAPE"])
            return None
        # dict
        if isinstance(obj, dict):
            if "SMAPE" in obj:
                v = obj["SMAPE"]
                if isinstance(v, (list, tuple)) and v:
                    return float(v[0])
                return float(v)
            # something like {'data': [{'SMAPE': ...}]}
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], dict) and "SMAPE" in v[0]:
                    return float(v[0]["SMAPE"])
        # DataFrame (if already in memory as a DF)
        if isinstance(obj, pd.DataFrame) and not obj.empty and "SMAPE" in obj.columns:
            return float(obj["SMAPE"].iloc[0])
    except Exception:
        return None
    return None

def _pick_test_val_metrics(data: dict) -> dict:
    """
    Reads aggregated/cumulative SMAPE for TEST and VAL from within 'results' (preferred),
    with a fallback to root-level blocks if they exist.
    """
    out = {}

    res = data.get("results") or {}

    # ---- VALIDATION ----
    # 1) key_metrics already provides val_smape_* -> do not overwrite if it exists
    # 2) fallback to global/cumulative metrics within 'results'
    val_agg = _smape_from_global(res.get("global_metrics_val"))
    if val_agg is None:
        val_agg = _smape_from_list_or_df(res.get("aggregated_metrics_val"))
    out["val_smape_agg_fallback"] = val_agg

    val_cum = _smape_from_list_or_df(res.get("cumulative_metrics_val"))
    if val_cum is None:
        # sometimes only 'global' exists; as a last resort, reuse it
        val_cum = _smape_from_global(res.get("global_metrics_val"))
    out["val_smape_cum_fallback"] = val_cum

    # ---- TEST ----
    test_agg = _smape_from_global(res.get("global_metrics_test"))
    if test_agg is None:
        test_agg = _smape_from_list_or_df(res.get("aggregated_metrics_test"))
    out["test_smape_agg"] = test_agg

    test_cum = _smape_from_list_or_df(res.get("cumulative_metrics_test"))
    if test_cum is None:
        # last resort: use global
        test_cum = _smape_from_global(res.get("global_metrics_test"))
    out["test_smape_cum"] = test_cum

    return out

def collate_robust_results(
    run_output_dir: str | Path,
    *,
    required_objectives: tuple[str, ...] = ("val_smape_agg", "val_smape_cum"),
    allow_missing_objectives: bool = False,
) -> pd.DataFrame:
    """
    Collate robust results from run_output_dir/results/*.json into a leaderboard DataFrame.

    Guarantees:
      - only includes status == "success"
      - optuna_trial_number coerced to int (rows without it are dropped)
      - objectives are numeric + finite (unless allow_missing_objectives=True)
      - safe against malformed JSON and missing keys
    """
    run_path = Path(run_output_dir)
    results_dir = run_path / "results"

    if not results_dir.is_dir():
        logging.warning("Results directory not found at: %s", results_dir)
        return pd.DataFrame()

    files = sorted(results_dir.glob("*.json"))
    logging.info("Found %d result files in %s. Processing...", len(files), results_dir)

    rows: List[Dict[str, Any]] = []

    # keys we never want overwritten by key_metrics
    _protected_keys = {"experiment_id", "well", "job_hash", "optuna_trial_number"}

    # config keys we should skip (heavy / nested / irrelevant)
    excluded_cfg = {"selected_features", "extractor_config", "fuser_config", "run_output_dir"}

    for jf in files:
        try:
            raw = jf.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            logging.warning("Could not decode JSON '%s': %s", jf, e)
            continue

        if not isinstance(data, dict):
            continue
        if data.get("status") != "success":
            continue

        cfg = data.get("config") or {}
        if not isinstance(cfg, dict):
            cfg = {}

        # ---- optuna_trial_number: strong coercion to int ----
        tn = cfg.get("optuna_trial_number", None)
        try:
            tn = int(tn) if tn is not None else None
        except Exception:
            tn = None

        row: Dict[str, Any] = {
            "experiment_id": data.get("experiment_id"),
            "well": data.get("well"),
            "job_hash": data.get("job_hash"),
            "optuna_trial_number": tn,
        }

        # 1) validation metrics from key_metrics (safe merge; don't overwrite protected fields)
        km = data.get("key_metrics") or {}
        if isinstance(km, dict):
            for k, v in km.items():
                if k in _protected_keys:
                    continue
                row[k] = v

        # 2) fallbacks for metrics via _pick_test_val_metrics (if you have it)
        try:
            falls = _pick_test_val_metrics(data)  # type: ignore[name-defined]
            if not isinstance(falls, dict):
                falls = {}
        except Exception as e:
            logging.warning("Fallback metric picker failed for '%s': %s", jf, e)
            falls = {}

        if row.get("val_smape_agg") is None:
            row["val_smape_agg"] = falls.get("val_smape_agg_fallback")
        if row.get("val_smape_cum") is None:
            row["val_smape_cum"] = falls.get("val_smape_cum_fallback")

        # always attach test metrics if present (ok if None)
        row["test_smape_agg"] = falls.get("test_smape_agg")
        row["test_smape_cum"] = falls.get("test_smape_cum")

        # 3) shallow hyperparameters from cfg
        for k, v in cfg.items():
            if k in excluded_cfg:
                continue
            if isinstance(v, (dict, list)):
                continue
            # don't overwrite protected keys coming from top-level row
            if k in _protected_keys:
                continue
            row[k] = v

        rows.append(row)

    if not rows:
        logging.warning("No successful jobs found to create a leaderboard.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ---- Drop rows without trial_number (cannot report to Optuna) ----
    if "optuna_trial_number" not in df.columns:
        logging.warning("Missing optuna_trial_number column after collation. Returning empty df.")
        return pd.DataFrame()

    df = df[df["optuna_trial_number"].notna()].copy()
    if df.empty:
        logging.warning("All rows missing optuna_trial_number. Returning empty df.")
        return pd.DataFrame()

    # ensure int dtype
    df["optuna_trial_number"] = pd.to_numeric(df["optuna_trial_number"], errors="coerce")
    df = df[df["optuna_trial_number"].notna()].copy()
    df["optuna_trial_number"] = df["optuna_trial_number"].astype(int)

    # ---- Standardize column names ----
    if "epochs" not in df.columns and "n_epochs" in df.columns:
        df["epochs"] = df["n_epochs"]
    if "architecture" not in df.columns and "architecture_name" in df.columns:
        df["architecture"] = df["architecture_name"]
    if "lag_window" not in df.columns and "input_chunk_length" in df.columns:
        df["lag_window"] = df["input_chunk_length"]
    if "horizon" not in df.columns and "output_chunk_length" in df.columns:
        df["horizon"] = df["output_chunk_length"]

    # ---- Ensure numeric dtypes for metrics columns ----
    metric_cols = ("val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum")
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Enforce finite objectives (to avoid tell() with NaN/inf) ----
    for obj in required_objectives:
        if obj not in df.columns:
            df[obj] = np.nan

    if not allow_missing_objectives:
        mask = np.isfinite(df[list(required_objectives)].to_numpy(dtype=float)).all(axis=1)
        df = df[mask].copy()

    # Optional: stable ordering
    df = df.sort_values("optuna_trial_number").reset_index(drop=True)

    return df


