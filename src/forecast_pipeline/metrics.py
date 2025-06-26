import pandas as pd
import numpy as np

import logging 
from typing import List, Dict, Any, Tuple 
from .config import ( _COLS_TO_DROP_ALWAYS, _COLS_TO_DROP_FILTER, _METRIC_COLS_ORDER, _BASE_ORDER, _FILTER_ORDER )

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
