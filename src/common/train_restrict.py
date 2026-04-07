# ============================================
# Train restriction helpers (plug-and-play, instrumented)
# ============================================
from __future__ import annotations
from typing import Dict
import logging
import pandas as pd

log = logging.getLogger("train_restrict")


def _compute_primary_per_well(
    champions_df: pd.DataFrame,
    score_col: str = "weighted_score",
) -> Dict[str, str]:
    if champions_df is None or champions_df.empty:
        log.warning("[train_restrict] _compute_primary_per_well: champions_df is empty.")
        return {}

    if "well" not in champions_df.columns or "job_hash" not in champions_df.columns:
        log.error("[train_restrict] _compute_primary_per_well: missing 'well' or 'job_hash'.")
        return {}

    use_col = score_col if score_col in champions_df.columns else "val_smape_agg"
    if use_col != score_col:
        log.warning("[train_restrict] score_col '%s' not found, falling back to '%s'.", score_col, use_col)

    # menor é melhor → ascending=True
    tmp = (
        champions_df
        .copy()
        .assign(_score=pd.to_numeric(champions_df[use_col], errors="coerce"))
        .sort_values(["well", "_score", "job_hash"], ascending=[True, True, True])
    )

    # manter wells NaN (se sua versão suportar); fallback sem dropna se não
    try:
        idx = tmp.groupby("well", dropna=False, as_index=False).head(1).index
    except TypeError:
        idx = tmp.groupby("well", as_index=False).head(1).index

    primary = (
        tmp.loc[idx, ["well", "job_hash"]]
        .set_index("well")["job_hash"]
        .astype(str)
        .to_dict()
    )

    log.info("[train_restrict] selected %d primary jobs for %d wells.",
             len(primary), champions_df["well"].nunique(dropna=False))
    if primary:
        log.info("[train_restrict] sample primary_by_well=%s", list(primary.items())[:8])
    return primary



def _restrict_train_to_primary(
    series_df: pd.DataFrame,
    primary_by_well: Dict[str, str],
) -> pd.DataFrame:
    """
    Keep only the primary job's rows inside the TRAIN split for each well.
    Do not touch VAL/TEST.
    Also adds 'is_primary' boolean column for convenience.

    Logging:
      - incoming shape and split distribution
      - how many rows flagged as primary
      - rows removed/kept in TRAIN
    """
    if series_df is None or series_df.empty:
        log.warning("[train_restrict] _restrict_train_to_primary: series_df is empty.")
        return series_df

    missing_cols = [c for c in ("well", "job_hash") if c not in series_df.columns]
    if missing_cols:
        log.error(
            "[train_restrict] _restrict_train_to_primary: missing required columns %s.",
            missing_cols,
        )
        return series_df

    log.info(
        "[train_restrict] _restrict_train_to_primary: series_df shape=%s, columns=%s",
        series_df.shape, list(series_df.columns)
    )
    if "split" in series_df.columns:
        split_counts = series_df["split"].astype(str).value_counts(dropna=False)
        log.info(
            "[train_restrict] _restrict_train_to_primary: split distribution before:\n%s",
            split_counts.to_string()
        )
    else:
        log.info(
            "[train_restrict] _restrict_train_to_primary: no 'split' column; "
            "function will be a no-op (just marks 'is_primary')."
        )

    if not primary_by_well:
        log.warning(
            "[train_restrict] _restrict_train_to_primary: primary_by_well is empty; "
            "no row will be marked as primary."
        )

    out = series_df.copy()

    def _mark(r):
        w = r.get("well")
        j = r.get("job_hash")
        return (
            pd.notna(w)
            and pd.notna(j)
            and str(j) == str(primary_by_well.get(str(w)))
        )

    out["is_primary"] = out.apply(_mark, axis=1)

    primary_rows = int(out["is_primary"].sum())
    log.info(
        "[train_restrict] _restrict_train_to_primary: %d rows flagged as primary.",
        primary_rows,
    )

    # If no split info, we only add the column
    if "split" not in out.columns:
        log.info(
            "[train_restrict] _restrict_train_to_primary: returning with 'is_primary' only; no split filtering."
        )
        return out

    is_train = out["split"].astype(str).str.lower().eq("train")
    before_train = int(is_train.sum())
    before_total = len(out)

    keep_mask = ~is_train | (is_train & out["is_primary"])
    out_filtered = out.loc[keep_mask].reset_index(drop=True)

    after_train = int(
        out_filtered["split"].astype(str).str.lower().eq("train").sum()
    )
    after_total = len(out_filtered)

    log.info(
        "[train_restrict] _restrict_train_to_primary: TRAIN rows %d → %d (Δ=%d).",
        before_train, after_train, after_train - before_train
    )
    log.info(
        "[train_restrict] _restrict_train_to_primary: TOTAL rows %d → %d (Δ=%d).",
        before_total, after_total, after_total - before_total
    )

    if "split" in out_filtered.columns:
        split_counts_after = out_filtered["split"].astype(str).value_counts(dropna=False)
        log.info(
            "[train_restrict] _restrict_train_to_primary: split distribution after:\n%s",
            split_counts_after.to_string()
        )

    return out_filtered
