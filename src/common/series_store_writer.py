from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

JsonLike = Union[Dict[str, Any], "Box"]  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de lock + escrita atômica (drop-in)
# ─────────────────────────────────────────────────────────────────────────────
import os, tempfile
from contextlib import contextmanager

try:
    import fcntl as _fcntl  # POSIX
except Exception:  # Windows sem fcntl
    _fcntl = None

@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    try:
        if _fcntl is not None:
            _fcntl.flock(fh, _fcntl.LOCK_EX)
        yield
    finally:
        if _fcntl is not None:
            try: _fcntl.flock(fh, _fcntl.LOCK_UN)
            except Exception: pass
        try: fh.close()
        except Exception: pass

def _atomic_to_parquet(df: pd.DataFrame, out_path: Path, compression: Optional[str] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # escreve em arquivo temporário na mesma pasta e depois renomeia
    with tempfile.NamedTemporaryFile(delete=False, dir=str(out_path.parent), suffix=".parquet") as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False, compression=compression)
        os.replace(tmp_path, out_path)  # atômico na maioria dos FS
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass



# =========================
# Settings & path helpers
# =========================

def _resolve_series_context(result_dict: JsonLike) -> Dict[str, Any]:
    """
    Robustly fetch the series_context dict from either:
      - top-level result_dict['series_context'], or
      - nested result_dict['results']['series_context'].

    Always returns a dict (possibly empty).
    """
    try:
        if isinstance(result_dict, dict):
            res = result_dict.get("results") or {}
            if isinstance(res, dict):
                ctx = res.get("series_context") or result_dict.get("series_context") or {}
            else:
                ctx = result_dict.get("series_context") or {}
        else:
            # Box-like object
            res = _get(result_dict, "results", {}) or {}
            ctx = _get(res, "series_context") or _get(result_dict, "series_context") or {}
        return ctx if isinstance(ctx, dict) else {}
    except Exception:
        return {}



def _normalize_settings(
    cfg: Union[SeriesStoreSettings, Dict[str, Any]]
) -> SeriesStoreSettings:
    """
    Accept either a SeriesStoreSettings instance or a plain dict
    (like the Hydra/OMEGACONF config used in the main pipeline),
    and always return a SeriesStoreSettings.

    This is the key to making persist_* plug-and-play with both
    the backfill tool and the live eval pipeline.
    """
    if isinstance(cfg, SeriesStoreSettings):
        return cfg

    if isinstance(cfg, dict):
        base_dir = cfg.get("base_dir")
        if not base_dir:
            raise ValueError("[series_store] 'base_dir' is required in series_store_cfg")

        return SeriesStoreSettings(
            enabled=bool(cfg.get("enabled", True)),
            format=str(cfg.get("format", "parquet")),
            compress=str(cfg.get("compress", "zstd")),
            schema_version=int(cfg.get("schema_version", 1)),
            self_heal=bool(cfg.get("self_heal", False)),
            base_dir=str(base_dir),
        )

    raise TypeError(
        f"[series_store] Expected SeriesStoreSettings or dict for settings, "
        f"got {type(cfg).__name__}"
    )


@dataclass(frozen=True)
class SeriesStoreSettings:
    enabled: bool
    format: str         # "parquet"
    compress: str       # "zstd"
    schema_version: int
    self_heal: bool
    base_dir: str       # absolute or relative to project root


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def derive_series_path(
    *,
    settings: SeriesStoreSettings,
    group: Optional[str],
    arch: Optional[str],
    campaign: Optional[str],
    job_hash: Optional[str],
    dataset: Optional[str] = None,
    well: Optional[str] = None,
    ext: str = "parquet",
) -> Path:
    """
    <base_dir>/group=G/arch=A[/dataset=D][/well=W]/campaign=C/job=HASH.parquet
    """
    base = Path(settings.base_dir).resolve()
    group = group or "unknown_group"
    arch = arch or "unknown_arch"
    campaign = campaign or "unknown_campaign"
    job_hash = job_hash or "unknown_job"

    parts: List[Path] = [base, Path(f"group={group}"), Path(f"arch={arch}")]
    if dataset:
        parts.append(Path(f"dataset={dataset}"))
    if well:
        parts.append(Path(f"well={well}"))
    parts.append(Path(f"campaign={campaign}"))
    return Path(*parts) / f"job={job_hash}.{ext}"


# =========================
# Metadata coalescing
# =========================

def _coalesce_meta(result_dict: JsonLike) -> Dict[str, Any]:
    """
    Make job metadata robust across legacy and new shapes.
    Prefer job_meta, but fall back to top-level when missing.
    """
    meta = dict(_get(result_dict, "job_meta", {}) or {})
    # Common identifiers
    meta.setdefault("group", _get(result_dict, "group"))
    meta.setdefault("arch", _get(result_dict, "arch"))
    meta.setdefault("campaign", _get(result_dict, "campaign"))
    meta.setdefault("job_hash", _get(result_dict, "job_hash"))
    # Critical partitions
    meta.setdefault("dataset", _get(result_dict, "dataset"))
    meta.setdefault("well", _get(result_dict, "well"))
    return meta


# =========================
# Series extraction (robust)
# =========================

def _emit_rows(
    rows: List[Dict[str, Any]],
    split: str,
    block: Dict[str, Any],
    where: str,
    diag: List[str],
) -> None:
    # Accept multiple legacy spellings
    t = block.get("t") or block.get("time") or block.get("timestamp") or []
    yhat = block.get("yhat") or block.get("y_hat") or block.get("y_pred") or []
    ytrue = block.get("ytrue") or block.get("y_true") or block.get("target")

    # Normalize shapes: lists only
    t = list(t) if isinstance(t, (list, tuple)) else []
    yhat = list(yhat) if isinstance(yhat, (list, tuple)) else []

    # If ytrue missing, mirror t-length (not yhat-length) to keep timeline consistent
    if isinstance(ytrue, (list, tuple)):
        ytrue = list(ytrue)
    else:
        ytrue = [None] * len(t)

    lt, lyh, lyt = len(t), len(yhat), len(ytrue)

    # Quick reject
    if lt == 0 or lyh == 0:
        diag.append(
            f"[emit_rows] reject {where}::{split} — empty "
            f"(len(t)={lt}, len(yhat)={lyh}, len(ytrue)={lyt})"
        )
        return

    n = min(lt, lyh, lyt)
    if n == 0:
        diag.append(
            f"[emit_rows] reject {where}::{split} — n=0 "
            f"(len(t)={lt}, len(yhat)={lyh}, len(ytrue)={lyt})"
        )
        return

    # --- Align by tail (critical when yhat is horizon-only but t/ytrue include context) ---
    aligned = (lt == n and lyh == n and lyt == n)
    t2, yhat2, ytrue2 = t[-n:], yhat[-n:], ytrue[-n:]

    # Lightweight range/shape diagnostics
    def _safe_num(x):
        try:
            return float(x)
        except Exception:
            return None

    t_head = _safe_num(t2[0]) if t2 else None
    t_tail = _safe_num(t2[-1]) if t2 else None

    # Detect obviously bad time direction (only if numeric)
    monotone_warn = ""
    if t_head is not None and t_tail is not None and t_tail < t_head:
        monotone_warn = " (WARNING: t_tail < t_head)"

    diag.append(
        f"[emit_rows] {where}::{split} lens t/yhat/ytrue={lt}/{lyh}/{lyt} "
        f"-> n={n} aligned={'no' if aligned else 'TAIL'} "
        f"t_range=({t_head},{t_tail}){monotone_warn}"
    )

    # Optional: catch the classic “off by ~100”
    if lt != lyh or lyt != lyh:
        diag.append(
            f"[emit_rows] mismatch detail {where}::{split}: "
            f"Δ(t-yhat)={lt-lyh}, Δ(ytrue-yhat)={lyt-lyh}"
        )

    for i in range(n):
        rows.append({"split": split, "t": t2[i], "idx": i, "yhat": yhat2[i], "ytrue": ytrue2[i]})

    diag.append(f"[emit_rows] accepted {where}::{split} — {n} rows")



def _find_series_payload(result_dict: JsonLike, diag: List[str]) -> List[Dict[str, Any]]:
    """
    Try multiple legacy/new shapes and collect a row-wise payload.
    The diagnostic list accumulates where we searched and why we accepted/rejected.
    """
    rows: List[Dict[str, Any]] = []

    # NEW: resolve series_context from results.series_context OR top-level
    ctx = _resolve_series_context(result_dict)
    ctx_keys = list(ctx.keys()) if isinstance(ctx, dict) else []


    # A) canonical: series_context.forecast_agg.{val|test} or flat
    fa = ctx.get("forecast_agg") if isinstance(ctx, dict) else None
    if isinstance(fa, dict):
        if isinstance(fa.get("val"), dict) or isinstance(fa.get("test"), dict):
            if isinstance(fa.get("val"), dict):
                _emit_rows(rows, "val", fa["val"], "series_context.forecast_agg", diag)
            if isinstance(fa.get("test"), dict):
                _emit_rows(rows, "test", fa["test"], "series_context.forecast_agg", diag)
            if rows:
                diag.append(f"series_context keys: {ctx_keys}")
                return rows
        else:
            # flat form
            _emit_rows(rows, "unknown", fa, "series_context.forecast_agg(flat)", diag)
            if rows:
                diag.append(f"series_context keys: {ctx_keys}")
                return rows

    # B) legacy: series_context.series_artifacts.{val|test}
    sa = ctx.get("series_artifacts") if isinstance(ctx, dict) else None
    if isinstance(sa, dict):
        if isinstance(sa.get("val"), dict):
            _emit_rows(rows, "val", sa["val"], "series_context.series_artifacts", diag)
        if isinstance(sa.get("test"), dict):
            _emit_rows(rows, "test", sa["test"], "series_context.series_artifacts", diag)
        if rows:
            diag.append(f"series_context keys: {ctx_keys}")
            return rows

    # C) legacy: series_context.{val|test}
    if isinstance(ctx, dict) and (isinstance(ctx.get("val"), dict) or isinstance(ctx.get("test"), dict)):
        if isinstance(ctx.get("val"), dict):
            _emit_rows(rows, "val", ctx["val"], "series_context", diag)
        if isinstance(ctx.get("test"), dict):
            _emit_rows(rows, "test", ctx["test"], "series_context", diag)
        if rows:
            diag.append(f"series_context keys: {ctx_keys}")
            return rows

    # D) top-level: result_dict['series'] (flat or with val/test)
    top_series = _get(result_dict, "series", {})
    if isinstance(top_series, dict):
        if isinstance(top_series.get("val"), dict) or isinstance(top_series.get("test"), dict):
            if isinstance(top_series.get("val"), dict):
                _emit_rows(rows, "val", top_series["val"], "top.series", diag)
            if isinstance(top_series.get("test"), dict):
                _emit_rows(rows, "test", top_series["test"], "top.series", diag)
            if rows:
                return rows
        else:
            _emit_rows(rows, "unknown", top_series, "top.series(flat)", diag)
            if rows:
                return rows

    # E) sometimes 'forecast_series' exists
    fs = _get(result_dict, "forecast_series", {})
    if isinstance(fs, dict):
        if isinstance(fs.get("val"), dict):
            _emit_rows(rows, "val", fs["val"], "top.forecast_series", diag)
        if isinstance(fs.get("test"), dict):
            _emit_rows(rows, "test", fs["test"], "top.forecast_series", diag)
        if rows:
            return rows
        _emit_rows(rows, "unknown", fs, "top.forecast_series(flat)", diag)
        if rows:
            return rows

    # F) fallback: recursive hunt for any dict with plausible {t, yhat}
    def _hunt(obj: Any, trail: str = "root") -> None:
        if isinstance(obj, dict):
            t = obj.get("t") or obj.get("time") or obj.get("timestamp")
            y = obj.get("yhat") or obj.get("y_hat") or obj.get("y_pred")
            if isinstance(t, (list, tuple)) and isinstance(y, (list, tuple)) and len(t) and len(t) == len(y):
                _emit_rows(rows, "unknown", obj, f"hunt[{trail}]", diag)
            for k, v in obj.items():
                _hunt(v, f"{trail}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _hunt(v, f"{trail}[{i}]")

    _hunt(result_dict)
    if rows:
        return rows

    # No match — provide helpful diagnostics
    diag.append(f"series_context keys: {ctx_keys}")
    diag.append("No compatible series structure found in any known location.")
    return rows


# =========================
# Record building
# =========================

import numpy as np
from typing import Any, Dict, List

def _maybe_extend_with_train_from_context(
    rows: List[Dict[str, Any]],
    ctx: Any,
    diag: List[str],
) -> List[Dict[str, Any]]:
    if not isinstance(ctx, dict):
        diag.append("[attach_train] ctx not dict → skip")
        return rows

    forecast_agg = ctx.get("forecast_agg")
    if not isinstance(forecast_agg, dict):
        diag.append("[attach_train] ctx.forecast_agg not dict → skip")
        return rows

    train_block = forecast_agg.get("train")
    if not isinstance(train_block, dict):
        diag.append("[attach_train] forecast_agg.train missing/not dict → skip")
        return rows

    t_raw     = train_block.get("t")
    yhat_raw  = train_block.get("yhat")
    ytrue_raw = train_block.get("ytrue")

    # Pre-coercion lens
    def _lenish(x):
        return len(x) if isinstance(x, (list, tuple)) else None

    diag.append(
        f"[attach_train] raw lens: t={_lenish(t_raw)} yhat={_lenish(yhat_raw)} ytrue={_lenish(ytrue_raw)}"
    )

    try:
        t_arr     = np.asarray(t_raw, dtype=float).reshape(-1)
        yhat_arr  = np.asarray(yhat_raw, dtype=float).reshape(-1)
        ytrue_arr = np.asarray(ytrue_raw, dtype=float).reshape(-1)
    except Exception as e:
        diag.append(f"[attach_train] FAILED coerce to 1D float arrays: {e}")
        return rows

    n = min(t_arr.size, yhat_arr.size, ytrue_arr.size)
    if n == 0:
        diag.append("[attach_train] train block empty after coercion → skip")
        return rows

    # Range diagnostics
    t0 = float(t_arr[0]) if t_arr.size else None
    t1 = float(t_arr[-1]) if t_arr.size else None
    diag.append(f"[attach_train] coerced sizes t/yhat/ytrue={t_arr.size}/{yhat_arr.size}/{ytrue_arr.size} n={n} t_range=({t0},{t1})")

    # Infer schema
    if rows:
        template_row = dict(rows[0])
        base_keys = set(template_row.keys())
    else:
        base_keys = {"split", "t", "idx", "yhat", "ytrue"}
        template_row = {k: None for k in base_keys}

    has_idx = "idx" in base_keys

    # IMPORTANT: always start idx at 0 for train block (so plots that use idx won't shift)
    # If some consumer expects global idx, they should sort by t.
    start_idx = 0

    # Append
    for offset in range(n):
        r = dict(template_row)
        r["split"] = "train"
        r["t"]     = float(t_arr[offset])
        r["yhat"]  = float(yhat_arr[offset])
        r["ytrue"] = float(ytrue_arr[offset])
        if has_idx:
            r["idx"] = start_idx + offset
        rows.append(r)

    diag.append(f"[attach_train] attached train rows={n} (idx starts at {start_idx})")
    return rows



def build_series_record(result_dict: JsonLike) -> Tuple[Dict[str, Any], List[str]]:
    """
    Return (record, diagnostics).

    Robustly extracts:
      - series rows (train/val/test) from multiple legacy/new shapes
      - context.boundaries & context.full_history from *either*
        top-level 'series_context' OR 'results.series_context'

    Notes
    -----
    - Backwards compatible:
        * Legacy JSONs (no 'forecast_agg.train') behave exactly as before.
        * New JSONs with 'forecast_agg.train' will get an extra 'train' split
          appended to 'series_rows'.
    - If context is missing, persistence of meta (boundaries/history) will NOOP/skip.
    """
    diag: List[str] = []

    # ---- robust meta ----
    meta = _coalesce_meta(result_dict)

    # ---- robust series rows (legacy path) ----
    rows = _find_series_payload(result_dict, diag)

    # ---- robust context (reuses helper) ----
    ctx = _resolve_series_context(result_dict)
    boundaries   = ctx.get("boundaries")   if isinstance(ctx, dict) else None
    full_history = ctx.get("full_history") if isinstance(ctx, dict) else None

    # ---- NEW: opportunistically attach TRAIN from forecast_agg ----
    rows = _maybe_extend_with_train_from_context(rows, ctx, diag)

    record: Dict[str, Any] = {
        "schema_version": 1,
        "meta": {
            "group": meta.get("group"),
            "arch": meta.get("arch"),
            "campaign": meta.get("campaign"),
            "job_hash": meta.get("job_hash"),
            "dataset": meta.get("dataset"),
            "well": meta.get("well"),
            "H": _get(result_dict, "horizon"),
            "L": _get(result_dict, "lag_window"),
        },
        "series_rows": rows,
        "metrics_head": (result_dict.get("metrics") or {}) if isinstance(result_dict, dict) else {},
        "context": {
            "boundaries": boundaries,
            "full_history": full_history,
        },
    }
    return record, diag


# =========================
# Persistence (champion series parquet)
# =========================

def _build_dataframe_from_record(record: Dict[str, Any]) -> pd.DataFrame:
    meta = record.get("meta", {}) or {}
    rows = record.get("series_rows", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for k in ("group", "arch", "dataset", "well", "campaign", "job_hash"):
        df[k] = meta.get(k)
    df["H"] = meta.get("H")
    df["L"] = meta.get("L")
    return df


def persist_series(record: Dict[str, Any],
                   settings: Union["SeriesStoreSettings", Dict[str, Any]]) -> Path:
    """
    Real Parquet writer (idempotent). Raises informative errors on empty series.

    Plug-and-play: accepts either a SeriesStoreSettings instance or a
    plain dict config (as used by the main pipeline).
    """
    settings = _normalize_settings(settings)

    if not settings.enabled:
        raise RuntimeError("SeriesStore is disabled in settings")
    if settings.format != "parquet":
        raise ValueError(f"[series_store] unsupported format: {settings.format}")

    meta = record.get("meta", {}) or {}
    series_rows = record.get("series_rows", []) or []
    if not series_rows:
        raise ValueError("[series_store] empty series; nothing to write")

    out_path = derive_series_path(
        settings=settings,
        group=meta.get("group"),
        arch=meta.get("arch"),
        campaign=meta.get("campaign"),
        dataset=meta.get("dataset"),
        well=meta.get("well"),
        job_hash=meta.get("job_hash"),
        ext="parquet",
    )

    if out_path.exists():
        logging.info("[series_store] SKIP (exists): %s", out_path)
        return out_path

    df = _build_dataframe_from_record(record)
    if df.empty:
        raise ValueError("[series_store] dataframe empty after normalization")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression=settings.compress)

    logging.info("[series_store] WROTE %d rows → %s (codec=%s, schema=%s)",
                 len(df), out_path, settings.compress, settings.schema_version)
    return out_path


# =========================
# NEW: Meta writers
# =========================

def _meta_dir(base_dir: str | Path) -> Path:
    p = Path(base_dir).resolve() / "meta"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _history_dir(base_dir: str | Path, well: str) -> Path:
    p = Path(base_dir).resolve() / "history" / f"well={well}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def persist_boundaries_manifest(
    record: Dict[str, Any],
    settings: Union["SeriesStoreSettings", Dict[str, Any]],
) -> Optional[Path]:
    """
    Upsert idempotente de fronteiras em {base}/meta/boundaries.parquet
    Chave: (well, group, dataset)  — tolerante a None/NaN.
    Evita corridas (lock) e usa escrita atômica.
    """
    settings = _normalize_settings(settings)
    if not settings.enabled or settings.format != "parquet":
        return None

    meta = record.get("meta", {}) or {}
    ctx  = record.get("context", {}) or {}
    bnd  = (ctx.get("boundaries") or {}) if isinstance(ctx, dict) else {}

    well       = meta.get("well")
    group      = meta.get("group")
    dataset    = meta.get("dataset")
    train_end  = bnd.get("train_end")
    val_end    = bnd.get("val_end")
    test_start = bnd.get("test_start")

    # Requer apenas well + os 3 índices; group/dataset podem ser None
    if any(v is None for v in (well, train_end, val_end, test_start)):
        return None

    # Normaliza tipos simples
    def _s(x): 
        return None if x is None else (str(x).strip() if isinstance(x, str) else x)

    well    = _s(well)
    group   = _s(group)
    dataset = _s(dataset)

    path = _meta_dir(settings.base_dir) / "boundaries.parquet"
    lock = path.with_suffix(".lock")

    # Schema estável
    cols = ["well","group","dataset","train_end","val_end","test_start","source"]
    new_row = pd.DataFrame([{
        "well": well, "group": group, "dataset": dataset,
        "train_end": train_end, "val_end": val_end, "test_start": test_start,
        "source": "metadata",
    }])[cols]

    with _file_lock(lock):
        # lê existente (se quebrado, recomeça)
        if path.exists():
            try:
                df = pd.read_parquet(path)
            except Exception:
                df = pd.DataFrame(columns=cols)
        else:
            df = pd.DataFrame(columns=cols)

        # garante colunas
        for c in cols:
            if c not in df.columns:
                df[c] = pd.Series(dtype=new_row[c].dtype)

        # Comparação tolerante a None/NaN
        def _eq(a, b):
            return (pd.isna(a) and pd.isna(b)) or (a == b)

        mask = (
            df["well"].astype(object).map(lambda x: x if x != "" else None).fillna(pd.NA).eq(well).fillna(well is None) &
            df["group"].astype(object).map(lambda x: x if x != "" else None).fillna(pd.NA).eq(group).fillna(group is None) &
            df["dataset"].astype(object).map(lambda x: x if x != "" else None).fillna(pd.NA).eq(dataset).fillna(dataset is None)
        )

        if mask.any():
            cur = df.loc[mask, ["train_end","val_end","test_start"]].iloc[-1].to_dict()
            if (cur.get("train_end") == train_end and
                cur.get("val_end")   == val_end   and
                cur.get("test_start")== test_start):
                logging.info("⚠️  [series_store] boundaries unchanged (noop) well=%s", well)
                return path
            df.loc[mask, ["train_end","val_end","test_start","source"]] = [train_end, val_end, test_start, "metadata"]
        else:
            df = pd.concat([df, new_row], ignore_index=True)

        # dedupe por chave, keep='last'
        df = df.sort_index()
        df = df.drop_duplicates(subset=["well","group","dataset"], keep="last")

        _atomic_to_parquet(df[cols], path, compression=settings.compress)

    logging.info("✅ [series_store] upserted boundaries: well=%s train_end=%s val_end=%s test_start=%s",
                 well, train_end, val_end, test_start)
    return path



def persist_full_history(
    record: Dict[str, Any],
    settings: Union["SeriesStoreSettings", Dict[str, Any]],
) -> Optional[Path]:
    """
    Persiste GT longo por poço em {base}/history/well=<well>/history.parquet
    Regras:
      - com lock + escrita atômica
      - se já existir e o novo for mais curto: mantém antigo
      - se novo for mais longo: substitui
      - se igual: noop
    """
    settings = _normalize_settings(settings)
    if not settings.enabled or settings.format != "parquet":
        return None

    meta = record.get("meta", {}) or {}
    ctx  = record.get("context", {}) or {}
    fh   = (ctx.get("full_history") or {}) if isinstance(ctx, dict) else {}

    well = meta.get("well")
    group = meta.get("group")
    dataset = meta.get("dataset")

    t = fh.get("t")
    y = fh.get("ytrue")
    if well is None or t is None or y is None:
        return None

    df_new = pd.DataFrame({"t": list(t), "ytrue": list(y)})
    df_new["group"] = None if group is None else str(group)
    df_new["dataset"] = None if dataset is None else str(dataset)

    out_dir = _history_dir(settings.base_dir, well=str(well))
    out_path = out_dir / "history.parquet"
    lock = out_path.with_suffix(".lock")

    with _file_lock(lock):
        if out_path.exists():
            try:
                df_old = pd.read_parquet(out_path)
                n_old, n_new = len(df_old), len(df_new)
                if n_new == n_old:
                    logging.info("⚠️  [series_store] full_history exists (noop): well=%s points=%d", well, n_old)
                    return out_path
                if n_old >= n_new:
                    logging.warning("⚠️  [series_store] full_history exists; kept longer (old=%d, new=%d) well=%s",
                                    n_old, n_new, well)
                    return out_path
                _atomic_to_parquet(df_new, out_path, compression=settings.compress)
                logging.info("✅ [series_store] wrote full_history (replaced shorter): well=%s points=%d", well, n_new)
                return out_path
            except Exception:
                # se leitura falhar, escreve fresh
                _atomic_to_parquet(df_new, out_path, compression=settings.compress)
                logging.info("✅ [series_store] wrote full_history (fresh after read error): well=%s points=%d", well, len(df_new))
                return out_path

        _atomic_to_parquet(df_new, out_path, compression=settings.compress)
        logging.info("✅ [series_store] wrote full_history: well=%s points=%d", well, len(df_new))
        return out_path

