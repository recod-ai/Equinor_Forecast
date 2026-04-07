#hpo/neighborhood_selection.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Defaults (safe)
# -------------------------
DEFAULT_POOL_CFG = dict(
    top_pct=0.20,        # for top_pct
    drop=0.10,           # for val_band
    take=0.40,           # for val_band
    min_candidates=20,
)

DEFAULT_ROBUST = dict(
    k=10,
    min_strat=25,
    alpha=0.65,
    beta=0.03,
    gamma=0.35,
    luck_q=0.25,
    w_lr=2.0,
    w_ep=0.25,
    w_bs=0.50,
)

DEFAULT_CONS = dict(
    use=True,
    topk_per_cycle=10,
    min_hits=2,
    min_rate=None,
    stab_lambda=0.10,
)

# Column map (adapt if needed)
C = dict(
    val="val_smape_agg",
    val_cum="val_smape_cum",
    test="test_smape_agg",            # audit-only
    strat="physics_strategy",
    trial="optuna_trial_number",
    ep="epochs",
    bs="batch_size",
    lr="learning_rate",
    cycle="cycle",
    dataset="dataset",
    well="well",
    arch="architecture",
)


# -------------------------
# Helpers
# -------------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def _rank_gap(df: pd.DataFrame) -> np.ndarray:
    if C["val_cum"] not in df.columns:
        return np.zeros(len(df), dtype=float)
    a = _to_num(df[C["val"]])
    c = _to_num(df[C["val_cum"]])
    ra = a.rank(method="average", ascending=True)
    rc = c.rank(method="average", ascending=True)
    return (ra - rc).abs().to_numpy(dtype=float)


def _prep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (C["val"], C["val_cum"], C["test"], C["ep"], C["bs"], C["lr"], C["trial"]):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if C["strat"] not in out.columns:
        out[C["strat"]] = "unknown"
    if C["trial"] not in out.columns:
        out[C["trial"]] = np.nan
    return out


def _rob_norm_10_90(x: np.ndarray) -> np.ndarray:
    x2 = x[np.isfinite(x)]
    if len(x2) == 0:
        return np.full_like(x, np.nan)
    lo, hi = np.percentile(x2, [10, 90])
    span = (hi - lo) if hi > lo else (np.std(x2) if np.std(x2) > 0 else 1.0)
    return (x - lo) / (span + 1e-12)


def _features(df: pd.DataFrame, robust: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    need = (C["lr"], C["ep"], C["bs"])
    if not all(c in df.columns for c in need):
        return None, None

    lr = df[C["lr"]].to_numpy(dtype=float)
    lr = np.where(lr > 0, lr, np.nan)
    log_lr = np.log10(lr)  # keep as in original

    ep = _rob_norm_10_90(df[C["ep"]].to_numpy(dtype=float))
    bs = _rob_norm_10_90(df[C["bs"]].to_numpy(dtype=float))

    feats = np.column_stack([log_lr, ep, bs])
    w = np.array([robust["w_lr"], robust["w_ep"], robust["w_bs"]], dtype=float)
    return feats, w


def _l1_weighted(feats: np.ndarray, q: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(feats - q[None, :]) * w[None, :], axis=1)


def _cand_key(row: pd.Series) -> str:
    tr = row.get(C["trial"], np.nan)
    if pd.notna(tr) and np.isfinite(float(tr)):
        return f"trial:{int(float(tr))}"
    if "job_hash" in row and pd.notna(row["job_hash"]):
        return f"job:{str(row['job_hash'])}"
    if "experiment_id" in row and pd.notna(row["experiment_id"]):
        return f"exp:{str(row['experiment_id'])}"
    s = str(row.get(C["strat"], "unknown"))
    lr = row.get(C["lr"], np.nan)
    ep = row.get(C["ep"], np.nan)
    bs = row.get(C["bs"], np.nan)
    return f"hp:{s}|lr={lr}|ep={ep}|bs={bs}"


# -------------------------
# Pool (VAL-only)
# -------------------------
def pick_pool_idx(df: pd.DataFrame, *, pool_method: str, pool_cfg: Dict[str, Any]) -> np.ndarray:
    pool_method = str(pool_method or "top_pct").lower()
    cfg = {**DEFAULT_POOL_CFG, **(pool_cfg or {})}

    v = _to_num(df[C["val"]])
    ok = np.isfinite(v.to_numpy())
    idx_all = df.index[ok]
    if len(idx_all) == 0:
        return np.array([], dtype=int)

    idx_sorted = v.loc[idx_all].sort_values(ascending=True).index.to_numpy()
    n = len(idx_sorted)

    if pool_method == "top_pct":
        k = min(n, max(int(cfg["min_candidates"]), int(math.ceil(float(cfg["top_pct"]) * n))))
        return idx_sorted[:k].astype(int)

    if pool_method == "val_band":
        drop_n = int(math.floor(float(cfg["drop"]) * n))
        take_n = max(int(cfg["min_candidates"]), int(math.ceil(float(cfg["take"]) * n)))
        start = min(max(0, drop_n), n)
        end = min(start + take_n, n)
        if end <= start:
            start, end = 0, min(int(cfg["min_candidates"]), n)
        return idx_sorted[start:end].astype(int)

    # fallback
    k = min(n, max(int(cfg["min_candidates"]), int(math.ceil(0.2 * n))))
    return idx_sorted[:k].astype(int)


# -------------------------
# Robust scoring (VAL-only)
# -------------------------
def compute_robust_scores(
    df_in: pd.DataFrame,
    cand_idx: np.ndarray,
    *,
    robust: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = _prep_numeric(df_in)
    df = df.loc[np.isfinite(df[C["val"]].to_numpy(dtype=float))].reset_index(drop=True)
    n = len(df)

    scores = pd.Series(np.nan, index=df.index, dtype=float)
    scopes = pd.Series("", index=df.index, dtype=object)
    local_mads = pd.Series(np.nan, index=df.index, dtype=float)
    rgs = pd.Series(np.nan, index=df.index, dtype=float)

    if n == 0 or len(cand_idx) == 0:
        return scores, scopes, local_mads, rgs

    rg = _rank_gap(df)
    feats, w = _features(df, robust)

    def neighbors(i: int) -> Tuple[np.ndarray, str]:
        strat = df.loc[i, C["strat"]]
        pool = df.index[df[C["strat"]] == strat].to_numpy()
        scope = "within_strategy"
        if len(pool) < int(robust["min_strat"]):
            pool = df.index.to_numpy()
            scope = "global"

        pool = pool[pool != i]
        if len(pool) == 0:
            return np.array([], dtype=int), scope

        if feats is None:
            d = np.abs(df.loc[pool, C["val"]].to_numpy(dtype=float) - float(df.loc[i, C["val"]]))
        else:
            q = feats[i, :]
            if not np.all(np.isfinite(q)):
                d = np.abs(df.loc[pool, C["val"]].to_numpy(dtype=float) - float(df.loc[i, C["val"]]))
            else:
                pool_ok = pool[np.all(np.isfinite(feats[pool]), axis=1)]
                if len(pool_ok) < max(3, min(int(robust["k"]), len(pool))):
                    if scope == "within_strategy":
                        pool2 = df.index.to_numpy()
                        pool2 = pool2[pool2 != i]
                        pool_ok = pool2[np.all(np.isfinite(feats[pool2]), axis=1)]
                        scope = "global"
                if len(pool_ok) == 0:
                    d = np.abs(df.loc[pool, C["val"]].to_numpy(dtype=float) - float(df.loc[i, C["val"]]))
                else:
                    pool = pool_ok
                    d = _l1_weighted(feats[pool], q, w)

        k = min(int(robust["k"]), len(pool))
        nn = pool[np.argsort(d)[:k]]
        return nn.astype(int), scope

    for i in cand_idx.astype(int):
        if i < 0 or i >= n:
            continue

        nn, scope = neighbors(int(i))
        vals_nn = (
            np.concatenate([[float(df.loc[i, C["val"]])], df.loc[nn, C["val"]].to_numpy(dtype=float)])
            if len(nn)
            else np.array([float(df.loc[i, C["val"]])], dtype=float)
        )

        loc_mad = _mad(vals_nn)
        if not np.isfinite(loc_mad) or loc_mad < 1e-12:
            loc_mad = float(np.std(vals_nn, ddof=1)) if len(vals_nn) > 1 else 0.0

        rg_i = float(rg[i]) if np.isfinite(rg[i]) else 0.0

        # luckiness penalty
        if len(nn) >= 3:
            neigh_vals = df.loc[nn, C["val"]].to_numpy(dtype=float)
            baseline = float(np.quantile(neigh_vals, float(robust["luck_q"])))
            spread = _mad(neigh_vals)
            if not np.isfinite(spread) or spread < 1e-12:
                spread = float(np.std(neigh_vals, ddof=1)) if len(neigh_vals) > 1 else 1.0
            z = (baseline - float(df.loc[i, C["val"]])) / (spread + 1e-12)
            luck = float(max(0.0, z))
        else:
            luck = 0.0

        score = float(np.mean(vals_nn)) \
                + float(robust["alpha"]) * float(loc_mad) \
                + float(robust["beta"]) * float(rg_i) \
                + float(robust["gamma"]) * float(luck)

        scores.iloc[i] = score
        scopes.iloc[i] = scope
        local_mads.iloc[i] = float(loc_mad)
        rgs.iloc[i] = float(rg_i)

    return scores, scopes, local_mads, rgs


@dataclass
class Selection:
    row: pd.Series
    pool_idx: np.ndarray
    robust_score: float
    scope: str
    local_mad: float = np.nan
    rank_gap: float = np.nan
    chosen_key: str = ""
    cycles_seen: str = ""


def _basic_select(
    df_group: pd.DataFrame,
    *,
    pool_method: str,
    pool_cfg: Dict[str, Any],
    robust: Dict[str, Any],
) -> Selection:
    df = _prep_numeric(df_group)
    df = df.loc[np.isfinite(df[C["val"]].to_numpy(dtype=float))].reset_index(drop=True)

    if len(df) < 5:
        i = int(df[C["val"]].idxmin())
        r = df.loc[i]
        return Selection(row=r, pool_idx=np.array([], dtype=int), robust_score=float(r[C["val"]]), scope="none", chosen_key=_cand_key(r))

    pool = pick_pool_idx(df, pool_method=pool_method, pool_cfg=pool_cfg)
    if len(pool) == 0:
        i = int(df[C["val"]].idxmin())
        r = df.loc[i]
        return Selection(row=r, pool_idx=np.array([], dtype=int), robust_score=float(r[C["val"]]), scope="none", chosen_key=_cand_key(r))

    scores, scopes, mads, rgs = compute_robust_scores(df, pool, robust=robust)
    cand_scores = scores.loc[pool]
    if cand_scores.isna().all():
        i = int(df[C["val"]].idxmin())
        r = df.loc[i]
        return Selection(row=r, pool_idx=pool, robust_score=float(r[C["val"]]), scope="none", chosen_key=_cand_key(r))

    best_i = int(cand_scores.idxmin())
    r = df.loc[best_i]
    return Selection(
        row=r,
        pool_idx=pool,
        robust_score=float(scores.loc[best_i]),
        scope=str(scopes.loc[best_i]) or "within_strategy",
        local_mad=float(mads.loc[best_i]) if pd.notna(mads.loc[best_i]) else np.nan,
        rank_gap=float(rgs.loc[best_i]) if pd.notna(rgs.loc[best_i]) else np.nan,
        chosen_key=_cand_key(r),
    )

def run_neighborhood_selection(
    audit_df: pd.DataFrame,
    *,
    pool_method: str,
    pool_cfg: Optional[Dict[str, Any]] = None,
    robust_cfg: Optional[Dict[str, Any]] = None,
    group_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - champions_df: up to TOP_STRATEGIES_PER_GROUP rows per (dataset, well, architecture),
        selected by VAL-only robust selector, with test_* reattached.
      - diagnostics_df: diagnostics (audit-only) including pool size + TEST regret stats.

    Anti-leak:
      - Selection logic uses VAL-only view (drops test_*).
      - Audit/regret uses test_* but does NOT affect selection.
    """

    from forecast_pipeline._plotting_core import _get_color_palette, plot_error_distributions_story

    # -----------------------------
    # Hardcoded policy switches
    # -----------------------------
    TOP_STRATEGIES_PER_GROUP = 2  # top-2 per group
    PICK_DISTINCT_STRATEGIES = False
    # DEFAULT: pick top-2 overall, even if same physics_strategy
    # If True: pick 1 per physics_strategy, then keep top-2 across strategies (previous behavior)

    # -----------------------------
    # Hardcoded plotting switches
    # -----------------------------
    PLOT_STORY = False          # <- set False to disable
    PLOT_MAX_GROUPS = 999999   # safety cap
    PLOT_SHOW = True           # plotly fig.show()

    if audit_df is None or audit_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = group_cols or [C["dataset"], C["well"], C["arch"]]
    pool_cfg = {**DEFAULT_POOL_CFG, **(pool_cfg or {})}
    robust_cfg = {**DEFAULT_ROBUST, **(robust_cfg or {})}
    pool_method = str(pool_method or "top_pct").lower()

    # selection_df is VAL-only view (anti-leak)
    test_cols = [c for c in audit_df.columns if str(c).startswith("test_")]
    selection_df = audit_df.drop(columns=test_cols, errors="ignore") if test_cols else audit_df

    # Ensure strategy/family column exists
    family_col = C["strat"]
    if family_col not in selection_df.columns:
        selection_df = selection_df.copy()
        selection_df[family_col] = "unknown"

    def _pct_rank_lower_better(x: float, arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0 or not np.isfinite(x):
            return float("nan")
        if len(arr) == 1:
            return 0.0
        return float(np.mean(arr <= x))

    rows: List[pd.Series] = []
    diag_rows: List[Dict[str, Any]] = []

    # -----------------------------
    # Helper: run selection on one slice (g_sel, g_audit)
    # -----------------------------
    def _select_one(
        g_sel: pd.DataFrame,
        g_audit: pd.DataFrame,
        *,
        dataset: str = "",
        well: str = "",
        architecture: str = "",
        plot_counter: Optional[List[int]] = None,  # mutable int holder
    ) -> Optional[pd.Series]:
        # Preserve group-row mapping via _row_id (position inside this slice)
        g_sel2 = g_sel.copy()
        g_sel2["_row_id"] = np.arange(len(g_sel2), dtype=int)

        g_clean = _prep_numeric(g_sel2)
        g_clean = g_clean.loc[np.isfinite(g_clean[C["val"]].to_numpy(dtype=float))].reset_index(drop=True)
        if g_clean.empty:
            return None

        sel = _basic_select(g_clean, pool_method=pool_method, pool_cfg=pool_cfg, robust=robust_cfg)

        # Map chosen and pool indices back to original slice row positions
        chosen_row_id = None
        try:
            chosen_row_id = int(sel.row.get("_row_id"))
        except Exception:
            chosen_row_id = None

        pool_row_ids = np.array([], dtype=int)
        if sel.pool_idx is not None and len(sel.pool_idx) > 0 and "_row_id" in g_clean.columns:
            pool_row_ids = g_clean.loc[sel.pool_idx, "_row_id"].to_numpy(dtype=int)

        pool_audit = g_audit.iloc[pool_row_ids] if len(pool_row_ids) > 0 else g_audit.iloc[[]]
        chosen_audit = (
            g_audit.iloc[[chosen_row_id]]
            if (chosen_row_id is not None and 0 <= chosen_row_id < len(g_audit))
            else g_audit.iloc[[]]
        )

        # Audit regret on TEST metric (lower is better)
        test_metric = C["test"]
        chosen_test = float("nan")
        best_test = float("nan")
        regret = float("nan")
        ratio = float("nan")
        chosen_test_pct = float("nan")

        if test_metric in g_audit.columns:
            chosen_test_arr = pd.to_numeric(
                chosen_audit.get(test_metric, pd.Series([], dtype=float)), errors="coerce"
            ).to_numpy(dtype=float)
            chosen_test = float(chosen_test_arr[0]) if len(chosen_test_arr) else float("nan")

            pool_test = pd.to_numeric(
                pool_audit.get(test_metric, pd.Series([], dtype=float)), errors="coerce"
            ).to_numpy(dtype=float)

            if np.isfinite(pool_test).any():
                best_test = float(np.nanmin(pool_test))
                if np.isfinite(chosen_test) and np.isfinite(best_test):
                    regret = float(chosen_test - best_test)
                    ratio = float(chosen_test / (best_test + 1e-12))
                    chosen_test_pct = _pct_rank_lower_better(chosen_test, pool_test)

        # -----------------------------
        # PLOT STORY (hardcoded)
        # -----------------------------
        if PLOT_STORY and plot_error_distributions_story is not None:
            if plot_counter is not None:
                if plot_counter[0] >= int(PLOT_MAX_GROUPS):
                    pass
                else:
                    plot_counter[0] += 1

                    extra = f"robust={float(sel.robust_score):.3g} | scope={sel.scope} | pool={pool_method.lower()}"
                    if pool_method.lower() == "val_band":
                        extra += f" | band={pool_cfg['drop']:.2f}-{(pool_cfg['drop'] + pool_cfg['take']):.2f}"

                    palette = _get_color_palette("default") if _get_color_palette is not None else None

                    # Prefer audit-chosen row (has test), else fallback to sel.row
                    chosen_plot_row = chosen_audit.iloc[0] if len(chosen_audit) else sel.row

                    try:
                        plot_error_distributions_story(
                            g_audit,  # distribution includes test_* when available
                            chosen_row=chosen_plot_row,
                            title=str(extra),
                            dataset=str(dataset),
                            well=str(well),
                            architecture=str(architecture),
                            chosen_strategy=str(sel.row.get(C["strat"], "")),
                            chosen_trial=sel.row.get(C["trial"], None),
                            best_test=best_test,
                            regret_test=regret,
                            ratio_test=ratio,
                            spearman_val_test=None,  # you can wire later if you compute it
                            pool_chosen_test_percentile=chosen_test_pct,
                            palette=palette,
                            show=bool(PLOT_SHOW),
                        )
                    except Exception:
                        # Never break selection because of plotting
                        pass

        # Diagnostics row (identity is stamped outside)
        diag_rows.append({
            "pool_method": pool_method,
            "pool_size": int(len(sel.pool_idx)) if sel.pool_idx is not None else 0,
            "pool_best_val_smape_agg": float(
                np.nanmin(
                    pd.to_numeric(
                        pool_audit.get(C["val"], pd.Series([], dtype=float)), errors="coerce"
                    ).to_numpy(dtype=float)
                )
            ) if (C["val"] in pool_audit.columns and len(pool_audit)) else float("nan"),
            "pool_best_test_smape_agg": best_test,
            "regret_test": regret,
            "ratio_test": ratio,
            "chosen_test_percentile": chosen_test_pct,
            "notes": "",
            "_neighbor_scope": sel.scope,
            "_neighbor_local_mad": float(sel.local_mad) if np.isfinite(sel.local_mad) else float("nan"),
            "_neighbor_rank_gap": float(sel.rank_gap) if np.isfinite(sel.rank_gap) else float("nan"),
            "_chosen_key": sel.chosen_key,
        })

        # Champion row: start from VAL row, attach test_* by stable in-slice row position
        r = sel.row.copy()
        row_id = None
        try:
            row_id = int(r.get("_row_id"))
        except Exception:
            row_id = None

        r["robust_score"] = float(sel.robust_score)
        r["_neighbor_scope"] = sel.scope
        r["_neighbor_local_mad"] = sel.local_mad
        r["_neighbor_rank_gap"] = sel.rank_gap
        r["_chosen_key"] = sel.chosen_key

        if row_id is not None and 0 <= row_id < len(g_audit) and test_cols:
            for tc in test_cols:
                r[tc] = g_audit.iloc[row_id].get(tc, np.nan)

        if "_row_id" in r.index:
            r = r.drop(labels=["_row_id"])

        return r

    plot_counter = [0]  # mutable counter for plot cap

    # -----------------------------
    # Mode A (DEFAULT): pick top-2 overall per group (strategy can repeat)
    # -----------------------------
    if not PICK_DISTINCT_STRATEGIES:
        for gkey, g_sel in selection_df.groupby(group_cols, dropna=False, sort=False):
            g_audit = audit_df.loc[g_sel.index]

            gkey_tuple = gkey if isinstance(gkey, tuple) else (gkey,)
            dataset = str(gkey_tuple[0]) if len(gkey_tuple) > 0 else ""
            well = str(gkey_tuple[1]) if len(gkey_tuple) > 1 else ""
            arch = str(gkey_tuple[2]) if len(gkey_tuple) > 2 else ""

            # First pick
            r1 = _select_one(
                g_sel, g_audit,
                dataset=dataset, well=well, architecture=arch,
                plot_counter=plot_counter,
            )
            if r1 is None:
                continue

            id_map = {group_cols[i]: gkey_tuple[i] for i in range(len(group_cols))}
            for k, v in id_map.items():
                r1[k] = v
            diag_rows[-1].update(id_map)
            rows.append(r1)

            if TOP_STRATEGIES_PER_GROUP <= 1:
                continue

            # Remove chosen row (best-effort) and pick again
            key_col = None
            for cand in ["job_hash", "trial_hash", "experiment_id", C["trial"]]:
                if cand in g_sel.columns:
                    key_col = cand
                    break

            if key_col is not None and key_col in r1.index:
                chosen_key_val = r1.get(key_col, None)
                g_sel2 = g_sel[g_sel[key_col] != chosen_key_val]
                g_audit2 = g_audit.loc[g_sel2.index]
            else:
                g_sel2 = g_sel.copy()
                g_audit2 = g_audit.copy()

            r2 = _select_one(
                g_sel2, g_audit2,
                dataset=dataset, well=well, architecture=arch,
                plot_counter=plot_counter,
            )
            if r2 is None:
                continue

            for k, v in id_map.items():
                r2[k] = v
            diag_rows[-1].update(id_map)
            rows.append(r2)

        champions_df = pd.DataFrame(rows).reset_index(drop=True)
        diagnostics_df = pd.DataFrame(diag_rows).reset_index(drop=True)
        return champions_df, diagnostics_df

    # -----------------------------
    # Mode B: pick 1 per strategy, then keep top-2 strategies per group
    # -----------------------------
    strat_group_cols = list(group_cols) + [family_col]

    tmp_rows: List[pd.Series] = []
    tmp_diag_rows: List[Dict[str, Any]] = []

    # Local diag sink for this mode
    diag_rows_local = tmp_diag_rows

    def _select_one_local(*args, **kwargs):
        # temporarily redirect diag_rows writes
        nonlocal diag_rows
        old = diag_rows
        diag_rows = diag_rows_local
        try:
            return _select_one(*args, **kwargs)
        finally:
            diag_rows = old

    for gkey, g_sel in selection_df.groupby(strat_group_cols, dropna=False, sort=False):
        g_audit = audit_df.loc[g_sel.index]

        gkey_tuple = gkey if isinstance(gkey, tuple) else (gkey,)
        dataset = str(gkey_tuple[0]) if len(gkey_tuple) > 0 else ""
        well = str(gkey_tuple[1]) if len(gkey_tuple) > 1 else ""
        arch = str(gkey_tuple[2]) if len(gkey_tuple) > 2 else ""

        r = _select_one_local(
            g_sel, g_audit,
            dataset=dataset, well=well, architecture=arch,
            plot_counter=plot_counter,
        )
        if r is None:
            continue

        id_map = {strat_group_cols[i]: gkey_tuple[i] for i in range(len(strat_group_cols))}
        for k, v in id_map.items():
            r[k] = v
        diag_rows_local[-1].update(id_map)

        tmp_rows.append(r)

    champions_all = pd.DataFrame(tmp_rows).reset_index(drop=True)
    diagnostics_df = pd.DataFrame(tmp_diag_rows).reset_index(drop=True)

    if champions_all.empty:
        return champions_all, diagnostics_df

    # Keep only top-2 strategies per group (rank by VAL, fallback robust)
    rank_col = C["val"] if C["val"] in champions_all.columns else "robust_score"
    champions_all = champions_all.copy()
    champions_all[rank_col] = pd.to_numeric(champions_all.get(rank_col, np.nan), errors="coerce")

    sort_cols = [c for c in group_cols if c in champions_all.columns] + [rank_col]
    champions_all = champions_all.sort_values(sort_cols, ascending=True, kind="mergesort", na_position="last")

    champions_df = (
        champions_all.groupby([c for c in group_cols if c in champions_all.columns], dropna=False, sort=False)
        .head(int(TOP_STRATEGIES_PER_GROUP))
        .reset_index(drop=True)
    )

    return champions_df, diagnostics_df






