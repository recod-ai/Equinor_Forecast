# src/hpo/pri_enrichment.py
from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ==============================================================================
# Core helpers (VAL-only)
# ==============================================================================

def _is_test_col(c: str) -> bool:
    return str(c).startswith("test_")

def _drop_test_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    test_cols = [c for c in df.columns if _is_test_col(c)]
    return df.drop(columns=test_cols, errors="ignore")

def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Canonical SMAPE (mean):
      200 * |y - yhat| / (|y| + |yhat| + eps)
    Returns NaN if inputs are invalid or empty.
    """
    if y_true is None or y_pred is None:
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return float("nan")
    den = np.abs(y_true) + np.abs(y_pred) + eps
    num = np.abs(y_true - y_pred)
    out = 200.0 * np.mean(num / den)
    return float(out)

def _split_halves(n: int) -> Tuple[slice, slice]:
    """
    Deterministic halves split by index:
      mid = n // 2
      first  = [0:mid]
      second = [mid:n]
    If odd, second gets 1 more element.
    """
    mid = n // 2
    return slice(0, mid), slice(mid, n)

def _deep_get(d: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return None
        cur = cur[k]
    return cur

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _extract_val_series(j: Mapping[str, Any]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Extract VAL-only ytrue/yhat from known canonical paths.
    Extend/adjust fallback paths here if you discover variants.
    """
    # primary
    ytrue = _deep_get(j, ["results", "series_context", "forecast_agg", "val", "ytrue"])
    yhat  = _deep_get(j, ["results", "series_context", "forecast_agg", "val", "yhat"])

    # fallback A (if some jsons store forecast_agg under series_context directly)
    if ytrue is None or yhat is None:
        ytrue = _deep_get(j, ["series_context", "forecast_agg", "val", "ytrue"])
        yhat  = _deep_get(j, ["series_context", "forecast_agg", "val", "yhat"])

    # fallback B (if val arrays stored under forecast_agg.val_true/val_pred)
    if ytrue is None or yhat is None:
        ytrue = _deep_get(j, ["results", "series_context", "forecast_agg", "val", "val_true"])
        yhat  = _deep_get(j, ["results", "series_context", "forecast_agg", "val", "val_pred"])

    if not isinstance(ytrue, list) or not isinstance(yhat, list):
        return None, None
    return ytrue, yhat


# ==============================================================================
# Artifacts + metadata
# ==============================================================================

@dataclass(frozen=True)
class PriThresholds:
    # Stable if ALL conditions pass
    t_stability: float = 0.60   # Spearman >=
    t_gap_rel: float = 0.02     # gap_k_rel >= (2% separation)
    t_pool_eps: float = 0.02    # pool within +2% of best (relative)
    t_pool_max_frac: float = 0.30  # pool size <= 30% of candidates
    t_drift_p90: float = 3.0    # drift p90 <= (absolute SMAPE points)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_stability": self.t_stability,
            "t_gap_rel": self.t_gap_rel,
            "t_pool_eps": self.t_pool_eps,
            "t_pool_max_frac": self.t_pool_max_frac,
            "t_drift_p90": self.t_drift_p90,
        }

def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _hash_df(df: pd.DataFrame) -> str:
    # stable-ish hash for audit: schema + values (csv bytes)
    b = df.to_csv(index=False).encode("utf-8", errors="ignore")
    return hashlib.sha1(b).hexdigest()


# ==============================================================================
# M1 — Enrichment 1:1 (job_hash -> results/{job_hash}.json)
# ==============================================================================
def build_jobhash_json_index(results_dir: Path) -> Dict[str, Path]:
    """
    Build an index: {job_hash (stem) -> json_path}, searching recursively.

    Expected layout examples:
      results_dir/**/results/<job_hash>.json
      results_dir/**/<job_hash>.json   (fallback)

    If duplicates exist, we keep the first deterministic path (sorted).
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return {}

    candidates: List[Path] = []

    # Prefer the canonical subfolder name 'results'
    candidates.extend(results_dir.rglob("results/*.json"))

    # Fallback: any json under results_dir (rare, but safe)
    if not candidates:
        candidates.extend(results_dir.rglob("*.json"))

    # deterministic ordering
    candidates = sorted([p for p in candidates if p.is_file()], key=lambda p: str(p))

    idx: Dict[str, Path] = {}
    for p in candidates:
        stem = p.stem  # filename without suffix = job_hash
        if stem and stem not in idx:
            idx[stem] = p
    return idx


def _resolve_json_path(job_hash: str, results_dir: Path, json_index: Optional[Dict[str, Path]]) -> Optional[Path]:
    """
    Resolve job_hash -> json Path using index if provided, else fallback to direct path.
    """
    if not job_hash or job_hash.lower() in {"nan", "none"}:
        return None

    if json_index is not None:
        return json_index.get(job_hash)

    # legacy fallback (old layout)
    jp = Path(results_dir) / f"{job_hash}.json"
    return jp if jp.exists() else None


def enrich_leaderboard_with_val_halves(
    leaderboard_df: pd.DataFrame,
    results_dir: Path,
    *,
    key_col: str = "job_hash",
    val_smape_col_for_sanity: str = "val_smape_agg",
    enable_sanity_check: bool = True,
    sanity_tol: float = 1e-3,
    json_index: Optional[Dict[str, Path]] = None,   # <<< mantém
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    VAL-only enrichment:
      - loads results/{job_hash}.json
      - extracts val.ytrue/yhat
      - splits val into halves and computes:
          val_smape_first, val_smape_second, val_drift
      - optional sanity-check vs leaderboard[val_smape_agg] (log stats only)

    Returns:
      enriched_df (NO test_* columns added)
      stats dict for audit
    """
    if leaderboard_df is None or leaderboard_df.empty:
        return leaderboard_df, {
            "n_rows": 0,
            "coverage": 0.0,
            "missing_json": 0,
            "missing_key": 0,
            "missing_val_series": 0,
            "sanity_mismatches": 0,
        }

    df = leaderboard_df.copy()
    df = _drop_test_cols(df)  # hard anti-leak

    if key_col not in df.columns:
        raise ValueError(
            f"Expected key_col='{key_col}' in leaderboard_df.columns, got={list(df.columns)[:30]}..."
        )

    n_rows = int(len(df))

    # Pré-aloca arrays (muito mais rápido que df.at em loop)
    val_smape_first = np.full(n_rows, np.nan, dtype=float)
    val_smape_second = np.full(n_rows, np.nan, dtype=float)
    val_drift = np.full(n_rows, np.nan, dtype=float)

    missing_key = 0
    missing_json = 0
    missing_val_series = 0

    sanity_mismatches = 0
    sanity_examples: List[Dict[str, Any]] = []

    # Acesso rápido às colunas
    job_hashes = df[key_col].astype(str).to_numpy()

    has_sanity_col = enable_sanity_check and (val_smape_col_for_sanity in df.columns)
    logged_vals = None
    if has_sanity_col:
        # converte 1x; no loop só indexa
        logged_vals = pd.to_numeric(df[val_smape_col_for_sanity], errors="coerce").to_numpy(dtype=float)

    results_dir = Path(results_dir)

    for i in range(n_rows):
        job_hash = job_hashes[i]
        if (not job_hash) or (job_hash.lower() in {"nan", "none"}):
            missing_key += 1
            continue

        jp = _resolve_json_path(job_hash, results_dir, json_index)
        if jp is None:
            missing_json += 1
            continue

        j = _load_json(jp)
        if j is None:
            missing_json += 1
            continue

        ytrue, yhat = _extract_val_series(j)
        if ytrue is None or yhat is None:
            missing_val_series += 1
            continue

        # Conversão numpy 1x
        ytrue_arr = np.asarray(ytrue, dtype=float)
        yhat_arr = np.asarray(yhat, dtype=float)

        if ytrue_arr.size == 0 or yhat_arr.size == 0 or ytrue_arr.size != yhat_arr.size:
            missing_val_series += 1
            continue

        s1, s2 = _split_halves(int(ytrue_arr.size))
        sm1 = smape(ytrue_arr[s1], yhat_arr[s1])
        sm2 = smape(ytrue_arr[s2], yhat_arr[s2])

        val_smape_first[i] = sm1
        val_smape_second[i] = sm2
        val_drift[i] = (sm2 - sm1)

        if has_sanity_col:
            sm_total = smape(ytrue_arr, yhat_arr)
            logged = float(logged_vals[i]) if logged_vals is not None else float("nan")

            if np.isfinite(sm_total) and np.isfinite(logged):
                diff = abs(sm_total - logged)
                if diff > sanity_tol:
                    sanity_mismatches += 1
                    if len(sanity_examples) < 10:
                        sanity_examples.append({
                            "job_hash": job_hash,
                            "smape_total_recalc": float(sm_total),
                            "smape_logged": float(logged),
                            "abs_diff": float(diff),
                        })

    # Escreve de volta em 1 shot (rápido)
    df["val_smape_first"] = val_smape_first
    df["val_smape_second"] = val_smape_second
    df["val_drift"] = val_drift

    n_ok = int(np.sum(np.isfinite(val_drift)))
    coverage = float(n_ok / n_rows) if n_rows else 0.0

    stats: Dict[str, Any] = {
        "n_rows": n_rows,
        "coverage": coverage,
        "missing_key": int(missing_key),
        "missing_json": int(missing_json),
        "missing_val_series": int(missing_val_series),
        "sanity_mismatches": int(sanity_mismatches),
        "sanity_examples_top10": sanity_examples,
        "anti_leak": {
            "dropped_test_cols_from_input": True,
            "output_contains_test_cols": bool(any(_is_test_col(c) for c in df.columns)),
        },
    }
    return df, stats



def write_leaderboard_enriched(
    enriched_df: pd.DataFrame,
    out_path: Path,
    *,
    stats: Optional[Dict[str, Any]] = None,
    stats_path: Optional[Path] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df = _drop_test_cols(enriched_df)
    if out_path.suffix.lower() in {".parquet", ".pq"}:
        enriched_df.to_parquet(out_path, index=False)
    else:
        enriched_df.to_csv(out_path, index=False)

    if stats is not None and stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)


# ==============================================================================
# M2 — PRI report by group (VAL-only)
# ==============================================================================

def _spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman correlation using rank arrays (no scipy dependency).
    Returns NaN if invalid.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if a.size < 3:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    # Pearson corr of ranks
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2)))
    if den <= 0:
        return float("nan")
    return float(np.sum(ra * rb) / den)

def _gap_k_relative(values_sorted: np.ndarray, k: int) -> float:
    """
    values_sorted is ascending (lower is better).
    gap_rel = (v_k - v_1) / max(|v_1|, eps)
    """
    eps = 1e-12
    if values_sorted.size == 0:
        return float("nan")
    if values_sorted.size < k:
        k = int(values_sorted.size)
    v1 = float(values_sorted[0])
    vk = float(values_sorted[k - 1])
    return float((vk - v1) / max(abs(v1), eps))

def compute_pri_report(
    enriched_df: pd.DataFrame,
    *,
    group_cols: List[str],
    val_metric_col: str,
    k_gap: int = 5,
    pool_eps_rel: float = 0.02,
    thresholds: PriThresholds = PriThresholds(),
) -> pd.DataFrame:
    """
    Computes per-group PRI signals (VAL-only):
      - val_gap_k_rel
      - val_pool_eps_frac
      - pri_rank_stability (Spearman between val_smape_first and val_smape_second rankings)
      - pri_drift_median, pri_drift_p90
      - pri_regime_label (deterministic thresholds)
    """
    if enriched_df is None or enriched_df.empty:
        return pd.DataFrame()

    df = enriched_df.copy()
    df = _drop_test_cols(df)

    # validate columns
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"Missing group_col='{c}' in enriched_df. Available={list(df.columns)[:40]}...")

    required = ["val_smape_first", "val_smape_second", "val_drift", val_metric_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required PRI columns={missing}. Did you run M1 enrichment first?")

    rows: List[Dict[str, Any]] = []
    gobj = df.groupby(group_cols, dropna=False, sort=False)

    for gkey, g in gobj:
        g = g.copy()

        # numeric coercion
        m = pd.to_numeric(g[val_metric_col], errors="coerce").to_numpy(dtype=float)
        m = m[np.isfinite(m)]
        n = int(len(m))
        if n == 0:
            continue

        m_sorted = np.sort(m)  # ascending (lower is better)
        gap_rel = _gap_k_relative(m_sorted, k=k_gap)

        best = float(m_sorted[0])
        band = best * (1.0 + float(pool_eps_rel))
        pool_count = int(np.sum(m_sorted <= band))
        pool_frac = float(pool_count / max(n, 1))

        # Rank stability between halves (Spearman of ranks)
        a = pd.to_numeric(g["val_smape_first"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(g["val_smape_second"], errors="coerce").to_numpy(dtype=float)
        stab = _spearman_rank_corr(a, b)

        drift = pd.to_numeric(g["val_drift"], errors="coerce").to_numpy(dtype=float)
        drift = drift[np.isfinite(drift)]
        drift_med = float(np.median(drift)) if drift.size else float("nan")
        drift_p90 = float(np.quantile(drift, 0.90)) if drift.size else float("nan")

        # deterministic regime label
        is_stable = (
            (np.isfinite(stab) and stab >= thresholds.t_stability) and
            (np.isfinite(gap_rel) and gap_rel >= thresholds.t_gap_rel) and
            (np.isfinite(pool_frac) and pool_frac <= thresholds.t_pool_max_frac) and
            (np.isfinite(drift_p90) and drift_p90 <= thresholds.t_drift_p90)
        )
        label = "stable" if is_stable else "unstable"

        rec: Dict[str, Any] = {}
        if isinstance(gkey, tuple):
            for k, v in zip(group_cols, gkey):
                rec[k] = v
        else:
            rec[group_cols[0]] = gkey

        rec.update({
            "n_candidates": n,
            "val_metric_col": val_metric_col,
            "val_gap_k_rel": gap_rel,
            "val_pool_eps_rel": float(pool_eps_rel),
            "val_pool_eps_count": pool_count,
            "val_pool_eps_frac": pool_frac,
            "pri_rank_stability": stab,
            "pri_drift_median": drift_med,
            "pri_drift_p90": drift_p90,
            "pri_regime_label": label,
        })
        rows.append(rec)

    out = pd.DataFrame(rows)
    out = _drop_test_cols(out)
    return out


def write_pri_report(pri_report_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pri_report_df = _drop_test_cols(pri_report_df)
    pri_report_df.to_csv(out_path, index=False)


# ==============================================================================
# M3 — Policy builder (external meta-selector)
# ==============================================================================

def build_selection_policy(
    pri_report_df: pd.DataFrame,
    *,
    group_cols: List[str],
    stable_mode: str,
    unstable_mode: str,
    thresholds: PriThresholds = PriThresholds(),
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds a deterministic policy mapping group -> mode.
    Output is JSON-serializable and VAL-only.
    """
    if pri_report_df is None or pri_report_df.empty:
        return {"meta": meta or {}, "thresholds": thresholds.to_dict(), "policy": []}

    df = pri_report_df.copy()
    df = _drop_test_cols(df)

    # validate group cols
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"Missing group_col='{c}' in pri_report_df")

    policy_rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        label = str(r.get("pri_regime_label", "unstable"))
        chosen = stable_mode if label == "stable" else unstable_mode

        item = {c: r.get(c) for c in group_cols}
        item.update({
            "pri_regime_label": label,
            "pri_policy_mode": chosen,
            "why": {
                "pri_rank_stability": _safe_float(r.get("pri_rank_stability")),
                "val_gap_k_rel": _safe_float(r.get("val_gap_k_rel")),
                "val_pool_eps_frac": _safe_float(r.get("val_pool_eps_frac")),
                "pri_drift_p90": _safe_float(r.get("pri_drift_p90")),
            }
        })
        policy_rows.append(item)

    out = {
        "meta": meta or {},
        "thresholds": thresholds.to_dict(),
        "modes": {"stable": stable_mode, "unstable": unstable_mode},
        "group_cols": group_cols,
        "policy": policy_rows,
        "anti_leak": {"contains_test_cols": False},
    }
    return out


def write_selection_policy(policy: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # hard anti-leak check
    s = json.dumps(policy)
    if '"test_' in s:
        raise ValueError("Anti-leak violation: selection_policy contains 'test_' fields")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, sort_keys=True)


def _find_candidate_leaderboards(results_dir: Path, pattern: str) -> List[Path]:
    paths = sorted([p for p in results_dir.glob(pattern) if p.is_file()], key=lambda p: str(p))
    return paths

def _pick_best_leaderboard(paths: List[Path]) -> Optional[Path]:
    """
    Deterministic choice:
      - prefer a file that has 'master' in name
      - else prefer shortest path (often closer to root)
      - else first lexicographic
    """
    if not paths:
        return None
    scored = []
    for p in paths:
        name = p.name.lower()
        score = 0
        if "master" in name:
            score += 10
        if "hpo" in name:
            score += 3
        scored.append(( -score, len(str(p)), str(p), p))
    scored.sort()
    return scored[0][-1]

# ==============================================================================
# Orchestrator — PRI offline (M1 + M2 + M3)
# ==============================================================================

def _resolve_pri_artifact_paths(
    *,
    reports_dir: Path,
    results_dir: Path,
    leaderboard_path: Path,
) -> Dict[str, Path]:
    """
    Flat layout (NO subfolders):
      artifacts are written directly into reports_dir.

    To avoid collisions across different leaderboards (master vs cycle),
    we prefix filenames with a deterministic tag derived from leaderboard stem.
    """
    base = Path(reports_dir if reports_dir is not None else results_dir)
    base.mkdir(parents=True, exist_ok=True)

    # deterministic tag (keeps filenames stable, avoids deep paths)
    stem = Path(leaderboard_path).stem  # e.g., "hpo_master_leaderboard" or "leaderboard"
    tag = f"pri__{stem}"

    return {
        "leaderboard_in": Path(leaderboard_path),
        "leaderboard_enriched": base / f"{tag}__leaderboard_enriched.parquet",
        "leaderboard_enriched_stats": base / f"{tag}__leaderboard_enriched_stats.json",
        "pri_report": base / f"{tag}__pri_report.csv",
        "selection_policy": base / f"{tag}__selection_policy.json",
    }




def _read_leaderboard(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _maybe_load_existing(
    enriched_path: Path,
    pri_report_path: Path,
    policy_path: Path,
    *,
    force: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    If artifacts exist and not force -> load and return them.
    Policy is optional (may not exist in early MVP).
    """
    if force:
        return None, None, None

    enriched_df = None
    pri_df = None
    policy = None

    if enriched_path.exists():
        try:
            enriched_df = pd.read_parquet(enriched_path) if enriched_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(enriched_path)
        except Exception:
            enriched_df = None

    if pri_report_path.exists():
        try:
            pri_df = pd.read_csv(pri_report_path)
        except Exception:
            pri_df = None

    if policy_path.exists():
        try:
            with policy_path.open("r", encoding="utf-8") as f:
                policy = json.load(f)
        except Exception:
            policy = None

    return enriched_df, pri_df, policy


def _resolve_leaderboard_path(
    results_dir: Path,
    *,
    leaderboard_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Deterministic leaderboard discovery.
    Prefer explicit leaderboard_path; else search common patterns in results_dir.

    NOTE: Returns None if not found (caller decides to skip).
    """
    if leaderboard_path is not None:
        lp = Path(leaderboard_path)
        return lp if lp.exists() else None

    patterns = [
        "leaderboard.csv",
        "leaderboard.parquet",
        "**/leaderboard.csv",
        "**/leaderboard.parquet",
        "**/*leaderboard*.csv",
        "**/*leaderboard*.parquet",
    ]

    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(_find_candidate_leaderboards(results_dir, pat))

    picked = _pick_best_leaderboard(candidates)
    return picked


def run_pri_offline(
    *,
    results_dir: Path,
    reports_dir: Path,
    group_cols: List[str],
    val_metric_col: str,
    k_gap: int,
    pool_eps: float,
    rank_method: str,  # (mantido por compat; não usado aqui)
    thresholds: Mapping[str, Any],
    stable_mode: str,
    unstable_mode: str,
    leaderboard_path: Optional[Path] = None,
    force: bool = False,
    key_col: str = "job_hash",
    enable_sanity_check: bool = True,
    campaign_name: Optional[str] = None,          # <<< NOVO: injeta campaign = GROUP
    architecture_value: Optional[str] = None,     # <<< OPCIONAL: se quiser fixar arch (quando roda 1 por vez)
) -> Dict[str, Any]:
    """
    Offline PRI pipeline (VAL-only), plug-and-play.

    Semântica correta do seu pipeline:
      - campaign = GROUP (metadata do contexto, não coluna do leaderboard)
      - architecture/architecture_name são equivalentes (normaliza)
      - se não há leaderboard: SKIP (não quebra fluxo legado)
    """
    results_dir = Path(results_dir)
    reports_dir = Path(reports_dir)
    if not results_dir.exists():
        # manter hard fail: results_dir inválido é erro real de chamada
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Resolve leaderboard input (NÃO-FATAL)
    lb_path = _resolve_leaderboard_path(results_dir, leaderboard_path=leaderboard_path)
    if lb_path is None:
        return {
            "skipped": True,
            "reason": "leaderboard_not_found",
            "results_dir": str(results_dir),
            "reports_dir": str(reports_dir),
            "leaderboard_path_arg": str(leaderboard_path) if leaderboard_path is not None else None,
            "meta": {
                "campaign_name": campaign_name,
                "architecture_value": architecture_value,
                "group_cols": list(group_cols),
            },
        }

    paths = _resolve_pri_artifact_paths(reports_dir=reports_dir, results_dir=results_dir, leaderboard_path=lb_path)

    # 2) Fast path (load if exists)
    existing_enriched, existing_pri, existing_policy = _maybe_load_existing(
        paths["leaderboard_enriched"], paths["pri_report"], paths["selection_policy"], force=force
    )

    thr = PriThresholds(
        t_stability=float(thresholds.get("T_stab", thresholds.get("t_stability", 0.60))),
        t_gap_rel=float(thresholds.get("T_gap", thresholds.get("t_gap_rel", 0.02))),
        t_pool_eps=float(thresholds.get("T_pool_eps", thresholds.get("t_pool_eps", 0.02))),
        t_pool_max_frac=float(thresholds.get("T_pool_max_frac", thresholds.get("t_pool_max_frac", 0.30))),
        t_drift_p90=float(thresholds.get("T_drift_p90", thresholds.get("t_drift_p90", 3.0))),
    )

    meta: Dict[str, Any] = {
        "results_dir": str(results_dir),
        "reports_dir": str(reports_dir),
        "leaderboard_in": str(lb_path),
        "group_cols": list(group_cols),
        "val_metric_col": str(val_metric_col),
        "k_gap": int(k_gap),
        "pool_eps_rel": float(pool_eps),
        "rank_method": str(rank_method),
        "thresholds": thr.to_dict(),
        "modes": {"stable": str(stable_mode), "unstable": str(unstable_mode)},
        "force": bool(force),
        "campaign_name": campaign_name,
        "architecture_value": architecture_value,
    }

    # cache hit
    if existing_enriched is not None and existing_pri is not None and existing_policy is not None:
        existing_enriched = _drop_test_cols(existing_enriched)
        existing_pri = _drop_test_cols(existing_pri)

        return {
            "skipped": False,
            "loaded_from_cache": True,
            "leaderboard_enriched_path": str(paths["leaderboard_enriched"]),
            "pri_report_path": str(paths["pri_report"]),
            "selection_policy_path": str(paths["selection_policy"]),
            "hashes": {
                "leaderboard_in": _hash_file(lb_path) if lb_path.exists() else None,
                "leaderboard_enriched": _hash_file(paths["leaderboard_enriched"]) if paths["leaderboard_enriched"].exists() else None,
                "pri_report": _hash_file(paths["pri_report"]) if paths["pri_report"].exists() else None,
                "selection_policy": _hash_file(paths["selection_policy"]) if paths["selection_policy"].exists() else None,
                "enriched_df": _hash_df(existing_enriched) if isinstance(existing_enriched, pd.DataFrame) else None,
                "pri_df": _hash_df(existing_pri) if isinstance(existing_pri, pd.DataFrame) else None,
            },
            "meta": meta,
        }

    # 3) Read leaderboard (VAL-only)
    lb = _read_leaderboard(lb_path)
    lb = _drop_test_cols(lb)

    # --- Semântica: injeções/normalizações mínimas (antes do M1/M2) ---
    # campaign = GROUP (metadata)
    if campaign_name and ("campaign" in group_cols) and ("campaign" not in lb.columns):
        lb["campaign"] = str(campaign_name)

    # normaliza architecture
    # - se você roda 1 arch por vez, pode fixar via architecture_value
    if architecture_value:
        lb["architecture"] = str(architecture_value)
    else:
        # tenta derivar de architecture_name se faltar
        if ("architecture" in group_cols) and ("architecture" not in lb.columns) and ("architecture_name" in lb.columns):
            lb["architecture"] = lb["architecture_name"]

    # 4) M1 — Enrich leaderboard (usa results_dir p/ achar {job_hash}.json)
    json_index = build_jobhash_json_index(results_dir)

    enriched_df, stats = enrich_leaderboard_with_val_halves(
        lb,
        results_dir,
        key_col=key_col,
        val_smape_col_for_sanity="val_smape_agg",
        enable_sanity_check=bool(enable_sanity_check),
        json_index=json_index,   # <<< NOVO
    )

    # garante que campaign/architecture persistam no enriched_df também (caso tenham sido injetados)
    if campaign_name and ("campaign" in group_cols) and ("campaign" not in enriched_df.columns):
        enriched_df["campaign"] = str(campaign_name)
    if architecture_value and ("architecture" in group_cols) and ("architecture" not in enriched_df.columns):
        enriched_df["architecture"] = str(architecture_value)
    if ("architecture" in group_cols) and ("architecture" not in enriched_df.columns) and ("architecture_name" in enriched_df.columns):
        enriched_df["architecture"] = enriched_df["architecture_name"]

    # Persist M1
    write_leaderboard_enriched(
        enriched_df,
        paths["leaderboard_enriched"],
        stats=stats,
        stats_path=paths["leaderboard_enriched_stats"],
    )

    # 5) M2 — PRI report
    pri_df = compute_pri_report(
        enriched_df,
        group_cols=list(group_cols),
        val_metric_col=str(val_metric_col),
        k_gap=int(k_gap),
        pool_eps_rel=float(pool_eps),
        thresholds=thr,
    )
    write_pri_report(pri_df, paths["pri_report"])

    # 6) M3 — Policy JSON
    policy = build_selection_policy(
        pri_df,
        group_cols=list(group_cols),
        stable_mode=str(stable_mode),
        unstable_mode=str(unstable_mode),
        thresholds=thr,
        meta=dict(meta),
    )
    write_selection_policy(policy, paths["selection_policy"])

    enriched_df = _drop_test_cols(enriched_df)
    pri_df = _drop_test_cols(pri_df)

    return {
        "skipped": False,
        "loaded_from_cache": False,
        "leaderboard_enriched_path": str(paths["leaderboard_enriched"]),
        "leaderboard_enriched_stats_path": str(paths["leaderboard_enriched_stats"]),
        "pri_report_path": str(paths["pri_report"]),
        "selection_policy_path": str(paths["selection_policy"]),
        "coverage": float(stats.get("coverage", float("nan"))) if isinstance(stats, dict) else float("nan"),
        "missing_json": int(stats.get("missing_json", -1)) if isinstance(stats, dict) else None,
        "missing_val_series": int(stats.get("missing_val_series", -1)) if isinstance(stats, dict) else None,
        "hashes": {
            "leaderboard_in": _hash_file(lb_path) if lb_path.exists() else None,
            "leaderboard_enriched": _hash_file(paths["leaderboard_enriched"]) if paths["leaderboard_enriched"].exists() else None,
            "pri_report": _hash_file(paths["pri_report"]) if paths["pri_report"].exists() else None,
            "selection_policy": _hash_file(paths["selection_policy"]) if paths["selection_policy"].exists() else None,
            "enriched_df": _hash_df(enriched_df) if isinstance(enriched_df, pd.DataFrame) else None,
            "pri_df": _hash_df(pri_df) if isinstance(pri_df, pd.DataFrame) else None,
        },
        "stats": stats,
        "meta": meta,
    }


