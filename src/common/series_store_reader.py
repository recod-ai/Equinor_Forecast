from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable, Tuple
import numpy as np
import logging
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)



def _normalize_well_key(well: Any) -> Optional[str]:
    """
    Normalize a well identifier to the filesystem-safe form used on disk.

    Examples
    --------
    - '15/9-F-14'  -> '15-9-F-14'
    - '15-9-F-14'  -> '15-9-F-14'
    - 'P11'        -> 'P11'
    """
    if well is None:
        return None
    s = str(well).strip()
    if not s:
        return None
    return s.replace("/", "-")



def _well_to_path_parts(well: str) -> List[Path]:
    """
    '15/9-F-14' -> [Path('well=15'), Path('9-F-14')]
    'P11'       -> [Path('well=P11')]
    """
    if well is None or str(well).strip() == "":
        return []
    w = str(well).strip()
    segs = w.split("/")
    parts: List[Path] = [Path(f"well={segs[0]}")]
    for extra in segs[1:]:
        if extra:  # skip empty segments just in case
            parts.append(Path(extra))
    return parts



def _load_full_history_for_well(series_store_root: Path, well: str) -> Optional[pd.DataFrame]:
    """
    Load history/well=<well>/history.parquet if present. Returns None if missing.
    """
    path = (Path(series_store_root).resolve() / "history" / f"well={well}" / "history.parquet")
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        # ensure minimal schema
        if "t" not in df.columns or "ytrue" not in df.columns:
            return None
        return df[["t", "ytrue"]].copy()
    except Exception as e:
        logging.getLogger("series_reader").warning(
            "[series_reader] failed to read full_history for well=%s: %s", well, e
        )
        return None


def _pick_arch_base(row: pd.Series) -> Optional[str]:
    """Best-effort canonical arch base from champion row."""
    # explicit
    if "arch" in row and pd.notna(row["arch"]) and str(row["arch"]).strip():
        return str(row["arch"]).strip().lower()

    for col in ("architecture", "architecture_name"):
        if col in row and pd.notna(row[col]):
            txt = str(row[col]).strip().lower()
            # normalize common families
            if "arps" in txt:
                return "arps"
            if "darts" in txt:
                return "darts"
            if "seq2" in txt:
                # base family; alias handling is done later
                return "seq2pin" if "pin" in txt else "seq2"
            return txt.replace(" ", "_")
    return None


def _arch_aliases(arch_base: str) -> List[str]:
    """
    Return arch partition candidates to try, in order.
    Keeps compatibility with Phase 3 (e.g., seq2pin often stored under 'seq2').
    """
    a = arch_base.lower().strip()
    if a == "seq2pin":
        return ["seq2pin", "seq2"]
    if a == "seq2":
        return ["seq2", "seq2pin"]
    # others are usually exact
    return [a]


def _well_aliases(well: Optional[str]) -> List[Optional[str]]:
    """
    Generate candidate well identifiers for filesystem lookups.

    We try, in order:
      1) The raw value from champions_df (e.g. '15/9-F-14')
      2) The normalized FS value (slashes -> dashes), e.g. '15-9-F-14'
      3) A conservative "reverse" alias (first '-' back to '/') for rare legacy layouts.

    This keeps champions_df semantics intact while aligning FS lookups
    with the backfill/persist_series layout.
    """
    if well is None or str(well).strip() == "":
        return [None]

    raw = str(well).strip()
    aliases: List[Optional[str]] = [raw]

    # FS-normalized form: matches backfill layout (15/9-F-14 -> 15-9-F-14)
    norm = _normalize_well_key(raw)
    if norm and norm not in aliases:
        aliases.append(norm)

    # Very conservative inverse alias (for odd legacy layouts, if any)
    if "-" in raw:
        back_once = raw.replace("-", "/", 1)
        if back_once not in aliases:
            aliases.append(back_once)

    return aliases



def _series_parquet_path(
    series_store_root: Path,
    *,
    group: str,
    arch: str,
    campaign: str,
    job_hash: str,
    dataset: Optional[str],
    well: Optional[str],
) -> Path:
    # default (flat) path
    parts: List[Path] = [
        series_store_root,
        Path(f"group={group}"),
        Path(f"arch={arch}"),
    ]
    if dataset:
        parts.append(Path(f"dataset={dataset}"))
    if well:
        parts.append(Path(f"well={well}"))  # flat form
    parts.append(Path(f"campaign={campaign}"))
    return Path(*parts) / f"job={job_hash}.parquet"

def _series_parquet_path_nested(
    series_store_root: Path,
    *,
    group: str,
    arch: str,
    campaign: str,
    job_hash: str,
    dataset: Optional[str],
    well: Optional[str],
) -> Path:
    # nested form for wells with slashes
    parts: List[Path] = [
        series_store_root,
        Path(f"group={group}"),
        Path(f"arch={arch}"),
    ]
    if dataset:
        parts.append(Path(f"dataset={dataset}"))
    if well:
        parts.extend(_well_to_path_parts(well))
    parts.append(Path(f"campaign={campaign}"))
    return Path(*parts) / f"job={job_hash}.parquet"


def _first_existing_path(
    series_store_root: Path,
    *,
    group: str,
    arch_candidates: Iterable[str],
    well_candidates: Iterable[Optional[str]],
    campaign: str,
    job_hash: str,
    dataset: Optional[str],
) -> Tuple[Optional[Path], List[Path]]:
    """
    Locate the first existing Parquet path for a given champion.

    Supports BOTH layouts:

      Legacy (dataset-partitioned)
      ----------------------------
      group=.../arch=.../dataset=<dataset>/well=<well>/campaign=<campaign>/job=<job_hash>.parquet

      New layout (backfill / persist_series)
      --------------------------------------
      group=.../arch=.../well=<well>/campaign=<campaign>/job=<job_hash>.parquet

    Returns
    -------
    (path_or_None, tried_paths)
      - path_or_None: first path that exists, or None if nothing found.
      - tried_paths : list of direct paths attempted (for logging).
    """
    tried: List[Path] = []
    root = Path(series_store_root).resolve()

    # 1) Direct attempts: legacy (with dataset) AND new layout (no dataset)
    for arch in arch_candidates:
        arch_dir = root / f"group={group}" / f"arch={arch}"

        for well in well_candidates:
            # 1.a) Legacy layout: with dataset partition (if present)
            if dataset:
                if well:
                    p_legacy = (
                        arch_dir
                        / f"dataset={dataset}"
                        / f"well={well}"
                        / f"campaign={campaign}"
                        / f"job={job_hash}.parquet"
                    )
                else:
                    p_legacy = (
                        arch_dir
                        / f"dataset={dataset}"
                        / f"campaign={campaign}"
                        / f"job={job_hash}.parquet"
                    )
                tried.append(p_legacy)
                if p_legacy.is_file():
                    return p_legacy, tried

            # 1.b) New layout: no dataset partition (matches backfill / _manual_hive_path)
            if well:
                p_new = (
                    arch_dir
                    / f"well={well}"
                    / f"campaign={campaign}"
                    / f"job={job_hash}.parquet"
                )
            else:
                p_new = (
                    arch_dir
                    / f"campaign={campaign}"
                    / f"job={job_hash}.parquet"
                )
            tried.append(p_new)
            if p_new.is_file():
                return p_new, tried

    # 2) Fallback: search by (campaign, job_hash) under each arch dir
    best: Optional[Path] = None
    best_key: Tuple[int, float] = (10**9, -1.0)  # (depth, -mtime)

    for arch in arch_candidates:
        base = root / f"group={group}" / f"arch={arch}"
        if not base.exists():
            continue

        hits = list(base.glob(f"**/campaign={campaign}/job={job_hash}.parquet"))
        for h in hits:
            depth = len(h.relative_to(base).parts)
            try:
                mtime = h.stat().st_mtime
            except OSError:
                mtime = -1.0
            key = (depth, -mtime)
            if key < best_key:
                best_key = key
                best = h

    # 3) Last resort: search by job_hash only (ignore campaign)
    if best is None:
        for arch in arch_candidates:
            base = root / f"group={group}" / f"arch={arch}"
            if not base.exists():
                continue

            hits = list(base.glob(f"**/job={job_hash}.parquet"))
            for h in hits:
                depth = len(h.relative_to(base).parts)
                try:
                    mtime = h.stat().st_mtime
                except OSError:
                    mtime = -1.0
                key = (depth, -mtime)
                if key < best_key:
                    best_key = key
                    best = h

    if best is not None:
        return best, tried

    return None, tried




def _best_from_columns(row: pd.Series, candidates: List[str]) -> Optional[Any]:
    """Return the first non-empty candidate value from the row."""
    for c in candidates:
        if c in row and pd.notna(row[c]):
            val = row[c]
            # treat empty strings as missing
            if isinstance(val, str) and not val.strip():
                continue
            return val
    return None


def _build_meta_maps_from_champions(champions_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build per-job_hash maps for H, L, dataset, split from champions_df,
    using a robust set of candidate column names.
    """
    # Candidate columns to pull from champions_df
    H_CANDS = ["H", "horizon", "output_chunk_length", "forecast_horizon"]
    L_CANDS = ["L", "lag_window", "input_chunk_length", "lookback", "window", "lag"]
    DATASET_CANDS = ["dataset", "dataset_name"]
    SPLIT_CANDS = ["split"]

    maps = {"H": {}, "L": {}, "dataset": {}, "split": {}}

    # Ensure we have job_hash to key maps
    if "job_hash" not in champions_df.columns:
        return maps

    for _, r in champions_df.iterrows():
        jh = r.get("job_hash")
        if pd.isna(jh):
            continue
        jh = str(jh)

        h_val = _best_from_columns(r, H_CANDS)
        l_val = _best_from_columns(r, L_CANDS)
        d_val = _best_from_columns(r, DATASET_CANDS)
        s_val = _best_from_columns(r, SPLIT_CANDS)

        if h_val is not None:
            maps["H"][jh] = h_val
        if l_val is not None:
            maps["L"][jh] = l_val
        if d_val is not None:
            maps["dataset"][jh] = d_val
        if s_val is not None:
            maps["split"][jh] = s_val

    return maps


def _enrich_meta_from_champions(series_df: pd.DataFrame,
                                champions_df: pd.DataFrame,
                                log: logging.Logger) -> pd.DataFrame:
    """
    Fill missing H, L, dataset, split in series_df using champions_df.
    If H is still missing, infer per job_hash as max(idx)+1.
    """
    if series_df.empty or "job_hash" not in series_df.columns:
        return series_df

    maps = _build_meta_maps_from_champions(champions_df)

    # Ensure columns exist
    for col in ("H", "L", "dataset", "split"):
        if col not in series_df.columns:
            series_df[col] = None

    before_H_null = int(series_df["H"].isna().sum())
    before_L_null = int(series_df["L"].isna().sum())
    before_D_null = int(series_df["dataset"].isna().sum())
    before_S_null = int(series_df["split"].isna().sum())

    # Map fills (job_hash-aligned)
    if maps["H"]:
        series_df["H"] = series_df["H"].fillna(series_df["job_hash"].map(maps["H"]))
    if maps["L"]:
        series_df["L"] = series_df["L"].fillna(series_df["job_hash"].map(maps["L"]))
    if maps["dataset"]:
        series_df["dataset"] = series_df["dataset"].fillna(series_df["job_hash"].map(maps["dataset"]))
    if maps["split"]:
        series_df["split"] = series_df["split"].fillna(series_df["job_hash"].map(maps["split"]))

    

    # Infer H from idx if still missing (per job_hash)
    if "idx" in series_df.columns:
        missing_H_hashes = series_df.loc[series_df["H"].isna(), "job_hash"].unique()
        inferred = 0
        for jh in missing_H_hashes:
            sub = series_df[series_df["job_hash"] == jh]
            if sub.empty:
                continue
            try:
                # horizon length = number of rows in this series block (since rows are forecast_agg only)
                # safer: (max idx + 1), works even if a few rows are filtered out
                H_val = int(sub["idx"].max()) + 1
                series_df.loc[series_df["job_hash"] == jh, "H"] = H_val
                inferred += 1
            except Exception:
                pass
        if inferred:
            log.info("[series_reader] Inferred H for %d job(s) from idx.", inferred)

    # Final diagnostics
    after_H_null = int(series_df["H"].isna().sum())
    after_L_null = int(series_df["L"].isna().sum())
    after_D_null = int(series_df["dataset"].isna().sum())
    after_S_null = int(series_df["split"].isna().sum())

    if before_H_null > after_H_null:
        log.info("[series_reader] Filled H for %d rows.", before_H_null - after_H_null)
    if before_L_null > after_L_null:
        log.info("[series_reader] Filled L for %d rows.", before_L_null - after_L_null)
    if before_D_null > after_D_null:
        log.info("[series_reader] Filled dataset for %d rows.", before_D_null - after_D_null)
    if before_S_null > after_S_null:
        log.info("[series_reader] Filled split for %d rows.", before_S_null - after_S_null)

    return series_df



# =========================
# Step-4 helpers (NEW)
# =========================

def _load_boundaries_df(series_store_root: Path, log: logging.Logger) -> pd.DataFrame:
    """
    Load boundaries manifest from series_store/meta/boundaries.parquet.
    Returns an empty DataFrame if missing.
    Expected columns: ['well','group','dataset','train_end','val_end','test_start','source']
    """
    p = (series_store_root / "meta" / "boundaries.parquet").resolve()
    if not p.exists():
        log.warning("⚠️  [series_reader] boundaries manifest not found at %s (plots may fallback).", p)
        return pd.DataFrame(columns=["well","group","dataset","train_end","val_end","test_start","source"])
    try:
        df = pd.read_parquet(p)
        # defensive: normalize schema & dedupe keys
        need = ["well","group","dataset","train_end","val_end","test_start"]
        for c in need:
            if c not in df.columns:
                df[c] = np.nan
        df = df[["well","group","dataset","train_end","val_end","test_start"]].drop_duplicates(
            subset=["well","group","dataset"], keep="last"
        )
        return df
    except Exception as e:
        log.warning("⚠️  [series_reader] failed to read boundaries (%s); plots may fallback.", e)
        return pd.DataFrame(columns=["well","group","dataset","train_end","val_end","test_start"])


def _load_history_by_well(series_store_root: Path,
                          wells: Iterable[str],
                          log: logging.Logger) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    base = (series_store_root / "history").resolve()

    for w in sorted({str(x) for x in wells if isinstance(x, str) and x.strip()}):
        # a) flat
        p_flat = base / f"well={w}" / "history.parquet"
        candidates = [p_flat]

        # b) nested when the well contains '/'
        if "/" in w:
            nested = base / Path("well=" + w.split("/")[0]) / Path("/".join(w.split("/")[1:])) / "history.parquet"
            candidates.append(nested)

        # c) common alias: slash->dash (flat)
        if "/" in w:
            p_dash = base / f"well={w.replace('/', '-')}" / "history.parquet"
            candidates.append(p_dash)

        found = None
        for p in candidates:
            if p.exists():
                found = p; break

        if not found:
            continue

        try:
            df = pd.read_parquet(found)
            cols = [c for c in ["t","ytrue"] if c in df.columns]
            if cols:
                out[w] = df[cols].copy()
        except Exception as e:
            log.warning("⚠️  [series_reader] failed to read full_history for well=%s: %s", w, e)

    return out



# =========================
# PUBLIC API (UPDATED)
# =========================

def read_series_by_champions(
    series_store_root: Path,
    champions_df: pd.DataFrame,
    *,
    return_extras: bool = False
):
    """
    Fast, exact, and robust reader for Series Store Parquet by champion rows.

    Step-4 extras:
      - Merges boundaries manifest on (well, group, dataset) into series_df
      - Loads full-history GT per well into a dict
      - Logs concise summary and missing items (fallback allowed)

    If return_extras=True: returns (series_df, {"boundaries_df": ..., "full_history_by_well": ...})
    Otherwise: returns series_df.
    """
    log = logging.getLogger("series_reader")
    series_store_root = Path(series_store_root).resolve()

    # ----------------------------
    # small helpers (local scope)
    # ----------------------------
    def _is_missing_scalar(v: Any) -> bool:
        # treat None, NaN, and empty-string as missing
        if v is None:
            return True
        if isinstance(v, float) and pd.isna(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    def _ensure_object_before_fill(s: pd.Series) -> pd.Series:
        # make sure the column can hold strings/objects before fillna
        return s.astype(object) if s.dtype.kind in "biufc" else s

    def _norm_key_series(s: pd.Series) -> pd.Series:
        # normalize merge keys to 'object' and None for missing
        s = s.astype(object, copy=False)
        return s.where(pd.notna(s) & (s.astype(str).str.strip() != ""), None)

    # ----------------------------
    # guards
    # ----------------------------
    if champions_df is None or champions_df.empty:
        log.warning("[series_reader] champions_df is empty. Nothing to load.")
        empty_ret = (pd.DataFrame(),
                     {"boundaries_df": pd.DataFrame(), "full_history_by_well": {}})
        return empty_ret if return_extras else empty_ret[0]

    must_have = {"group", "campaign", "job_hash"}
    missing = must_have - set(champions_df.columns)
    if missing:
        log.error("[series_reader] champions_df missing required columns: %s", sorted(missing))
        empty_ret = (pd.DataFrame(),
                     {"boundaries_df": pd.DataFrame(), "full_history_by_well": {}})
        return empty_ret if return_extras else empty_ret[0]

    loaded: List[pd.DataFrame] = []
    n = len(champions_df)

    # ----------------------------
    # main loop: locate and read
    # ----------------------------
    for i, row in champions_df.iterrows():
        group   = str(row.get("group") or "").strip()
        campaign = str(row.get("campaign") or "").strip()
        job_hash = str(row.get("job_hash") or "").strip()

        dataset = row.get("dataset")
        dataset = str(dataset).strip() if pd.notna(dataset) and str(dataset).strip() else None

        well = row.get("well")
        well = str(well).strip() if pd.notna(well) and str(well).strip() else None

        if not group or not campaign or not job_hash:
            log.warning("[series_reader] Skipping row %s/%s (missing group/campaign/job_hash).", i + 1, n)
            continue

        arch_base = _pick_arch_base(row)
        if not arch_base:
            log.warning("[series_reader] Skipping row %s/%s (could not infer architecture).", i + 1, n)
            continue

        arch_cands = _arch_aliases(arch_base)
        well_cands = _well_aliases(well)

        p, tried = _first_existing_path(
            series_store_root,
            group=group,
            arch_candidates=arch_cands,
            well_candidates=well_cands,
            campaign=campaign,
            job_hash=job_hash,
            dataset=dataset,
        )
        if p is None:
            if tried:
                log.warning(
                    "[series_reader] Missing Parquet for champion (skipping). Tried %d path(s). First: %s  Last: %s",
                    len(tried), tried[0], tried[-1]
                )
            else:
                log.warning("[series_reader] Missing Parquet for champion (skipping): group=%s arch=%s campaign=%s job=%s",
                            group, arch_base, campaign, job_hash)
            continue

        try:
            df = pd.read_parquet(p)
        except Exception as e:
            log.warning("[series_reader] Failed to read %s: %s", p, e)
            continue

        # propagate key identifiers from champion row when missing
        for col in ("split", "group", "campaign", "job_hash", "dataset", "well"):
            val = row.get(col)
            if col not in df.columns:
                # create the column (single scalar broadcast)
                df[col] = None if _is_missing_scalar(val) else val
                continue

            # only fill NaNs if we truly have a scalar value; never call fillna(None)
            if _is_missing_scalar(val):
                # nothing to fill
                continue

            df[col] = _ensure_object_before_fill(df[col])
            df[col] = df[col].fillna(val)

        loaded.append(df)

    if not loaded:
        log.warning("[series_reader] No Parquet files loaded (0/%d champions).", n)
        empty_ret = (pd.DataFrame(),
                     {"boundaries_df": pd.DataFrame(), "full_history_by_well": {}})
        return empty_ret if return_extras else empty_ret[0]

    # Concatenate; light de-dup if needed
    series_df = pd.concat(loaded, ignore_index=True)
    if "job_hash" in series_df.columns and "idx" in series_df.columns:
        before = len(series_df)
        series_df = series_df.drop_duplicates(subset=["job_hash", "idx", "split"], keep="last")
        after = len(series_df)
        if after < before:
            log.info("[series_reader] Dropped %d duplicate rows by (job_hash, idx, split).", before - after)

    # Enrich H/L/dataset/split from champions where missing
    series_df = _enrich_meta_from_champions(series_df, champions_df, log)

    log.info(
        "[series_reader] Loaded %d/%d champion file(s): %d rows, %d columns.",
        len(loaded), n, len(series_df), len(series_df.columns)
    )
    if "job_hash" in series_df.columns:
        log.info("[series_reader] Unique job_hash in series_df: %d", series_df["job_hash"].nunique(dropna=True))

    # =========================
    # Step-4: merge boundaries + load full_history
    # =========================
    boundaries_df = _load_boundaries_df(series_store_root, log)

    # Normalize join keys as strings/None on BOTH sides to avoid dtype/name drift
    for k in ("well", "group", "dataset"):
        if k not in series_df.columns:
            series_df[k] = None
        series_df[k] = _norm_key_series(series_df[k])

    if not boundaries_df.empty:
        # Normalize keys
        if "well" not in boundaries_df.columns:
            boundaries_df["well"] = None
        boundaries_df["well"] = _norm_key_series(boundaries_df["well"])

        pre_wells = series_df["well"].nunique(dropna=True)

        # Only keep the columns we actually need for plots
        bnd_cols = [c for c in ["well", "train_end", "val_end", "test_start"] if c in boundaries_df.columns]
        bnd = boundaries_df[bnd_cols].drop_duplicates(subset=["well"], keep="last")

        series_df = series_df.merge(bnd, on="well", how="left")
        merged_wells = series_df.loc[series_df["train_end"].notna(), "well"].nunique(dropna=True)
        log.info("✅ [series_reader] merged boundaries for %d/%d wells", merged_wells, pre_wells)


    # Load full-history per well
    wells = series_df["well"].dropna().astype(str).unique().tolist() if "well" in series_df.columns else []
    full_history_by_well = _load_history_by_well(series_store_root, wells, log)

    if full_history_by_well:
        lens = [len(df) for df in full_history_by_well.values() if not df.empty]
        med_len = int(np.median(lens)) if lens else 0
        log.info("✅ [series_reader] loaded full_history for %d wells (median len=%s)",
                 len(full_history_by_well), med_len)
        missing_hist = sorted(set(wells) - set(full_history_by_well.keys()))
        if missing_hist:
            log.warning("⚠️  [series_reader] missing full_history for wells: %s "
                        "(plots will fallback on GT from JSON if possible)", missing_hist)
    else:
        if wells:
            log.warning("⚠️  [series_reader] no full_history found for any well; plots will fallback.")
        else:
            log.warning("⚠️  [series_reader] no wells detected in series_df; nothing to plot.")

    if return_extras:
        return series_df, {
            "boundaries_df": boundaries_df if isinstance(boundaries_df, pd.DataFrame) else pd.DataFrame(),
            "full_history_by_well": full_history_by_well if isinstance(full_history_by_well, dict) else {},
        }
    return series_df
