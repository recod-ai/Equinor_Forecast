# src/hpo/sort_utils.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from hpo.score_ordering import Ordering, resolve_ordering

logger = logging.getLogger(__name__)


DEFAULT_TIEBREAK_CANDIDATES: List[str] = [
    # Most deterministic identifiers first
    "optuna_trial_number",
    "trial",
    "trial_id",
    "number",
    "job_hash",
    "trial_hash",
    "experiment_id",
]


def _present_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def infer_tiebreak_cols(df: pd.DataFrame, extra: Optional[Sequence[str]] = None) -> List[str]:
    """
    Choose deterministic columns to break ties.
    Uses common HPO identifiers when present; may include extra provided.
    """
    cols = _present_cols(df, DEFAULT_TIEBREAK_CANDIDATES)
    if extra:
        cols = cols + [c for c in extra if c in df.columns and c not in cols]
    return cols


def stable_sort(
    df: pd.DataFrame,
    metric: str,
    ordering: Ordering,
    *,
    tiebreak_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Stable sort putting best rows first, with deterministic tie-break.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    if metric not in df.columns:
        raise KeyError(f"[sort] metric='{metric}' not in df.columns")

    tb = list(tiebreak_cols) if tiebreak_cols is not None else infer_tiebreak_cols(df)

    sort_cols = [metric] + tb
    ascending = [ordering.ascending] + [True] * len(tb)  # tie-break always ascending

    out = df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
    return out


def topk(
    df: pd.DataFrame,
    metric: str,
    k: int,
    *,
    lower_is_better: Optional[dict] = None,
    default_ascending: bool = True,
    tiebreak_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Stable top-k according to resolve_ordering(metric).
    """
    if df is None or df.empty or k <= 0:
        return pd.DataFrame() if df is None else df.head(0).copy()

    ordering = resolve_ordering(metric, lower_is_better=lower_is_better, default_ascending=default_ascending)
    return stable_sort(df, metric, ordering, tiebreak_cols=tiebreak_cols).head(int(k)).copy()


def best_idx_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric: str,
    *,
    lower_is_better: Optional[dict] = None,
    default_ascending: bool = True,
    tiebreak_cols: Optional[Sequence[str]] = None,
) -> pd.Index:
    """
    Returns the index of the best row per group with stable tie-break.
    Avoids nondeterminism of raw idxmin/idxmax under ties.
    """
    if df is None or df.empty:
        return pd.Index([])

    for c in group_cols:
        if c not in df.columns:
            raise KeyError(f"[sort] group col '{c}' missing")

    ordering = resolve_ordering(metric, lower_is_better=lower_is_better, default_ascending=default_ascending)
    tb = list(tiebreak_cols) if tiebreak_cols is not None else infer_tiebreak_cols(df)

    sort_cols = [metric] + tb
    ascending = [ordering.ascending] + [True] * len(tb)

    # Stable order then take first of each group
    sorted_df = df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
    idx = (
        sorted_df.groupby(list(group_cols), sort=False, dropna=False)
        .head(1)
        .index
    )
    return idx
