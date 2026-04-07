from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from common.series_store_writer import (
    build_series_record,
    persist_series,
    persist_boundaries_manifest,
    persist_full_history,
    SeriesStoreSettings,
)

# ─────────────────────────────────────────────────────────────────────────────
# Pretty reporter (self-contained, no external deps além de logging)
# ─────────────────────────────────────────────────────────────────────────────

import os, sys
from collections import Counter

def _detect_ansi() -> bool:
    """Respeita NO_COLOR/FORCE_COLOR e TTY."""
    if os.environ.get("NO_COLOR"):     # qualquer valor desativa
        return False
    if os.environ.get("FORCE_COLOR"):  # qualquer valor ativa
        return True
    return sys.stdout.isatty()

_ANSI = _detect_ansi()

def _c(code: str, text: str) -> str:
    if not _ANSI: return text
    return f"\033[{code}m{text}\033[0m"

def _bold(s: str) -> str:   return _c("1", s)
def _dim(s: str) -> str:    return _c("2", s)
def _green(s: str) -> str:  return _c("32", s)
def _blue(s: str) -> str:   return _c("34", s)
def _red(s: str) -> str:    return _c("31", s)
def _yellow(s: str) -> str: return _c("33", s)
def _cyan(s: str) -> str:   return _c("36", s)

def _rule(ch: str = "─", n: int = 92) -> str:
    return ch * n

def _left(text: str, width: int) -> str:
    if len(text) <= width: return text + " " * (width - len(text))
    return text[: max(0, width-1)] + "…"


def _guess_kind_from_reason_or_path(reason: str, path: Path | None) -> str:
    rs = (reason or "").upper()
    if "BOUNDARIES" in rs: return "BOUNDARIES"
    if "HISTORY" in rs:    return "HISTORY"
    p = (str(path) if path else "").lower()
    if "/meta/" in p:     return "BOUNDARIES"
    if "/history/" in p:  return "HISTORY"
    return "PARQUET"

class _BackfillReporter:
    WIDTH = 92
    def __init__(self, run_name: str, series_root: Path, dry_run: bool, workers: int) -> None:
        self.run_name   = run_name
        self.series_root= series_root
        self.dry_run    = dry_run
        self.workers    = workers
        self._results: list[BackfillResult] = []

    def banner(self) -> None:
        line = _rule("═", self.WIDTH)
        logging.info("\n%s\n  🚀  %s\n%s", line, _bold(f"Series Store Backfill — {self.run_name}"), line)
        logging.info("      %s  %s  %s",
                     _dim("root=")+str(self.series_root),
                     _dim("dry_run=")+str(self.dry_run),
                     _dim("workers=")+str(self.workers))

    def tasks_line(self, parquet_n: int, meta_n: int) -> None:
        logging.info("%s %s  %s", _cyan("Tasks"), _dim("parquet=")+str(parquet_n), _dim("meta=")+str(meta_n))

    def ingest_many(self, rows: list[BackfillResult]) -> None:
        self._results.extend(rows)

    def render_summary(self) -> None:
        if not self._results:
            logging.info(_dim("Nothing to summarize."))
            return
        by_action = Counter(r.action for r in self._results)
        by_kind   = Counter(_guess_kind_from_reason_or_path(r.reason, r.parquet_path) for r in self._results)

        logging.info("\n%s %s", _blue("▸"), _bold("Summary"))
        logging.info("  %s  %s  %s  %s  %s  %s",
                     _dim("created=")+_green(str(by_action.get("CREATED", 0))),
                     _dim("updated=")+_blue(str(by_action.get("UPDATED", 0))),
                     _dim("skipped=")+_dim(str(by_action.get("SKIPPED", 0))),
                     _dim("failed=")+_red(str(by_action.get("FAILED", 0))),
                     _dim("dry=")+_yellow(str(by_action.get("DRY_RUN", 0))),
                     _dim("total=")+_bold(str(sum(by_action.values()))),
        )
        logging.info("  %s  %s  %s",
                     _dim("parquet=")+str(by_kind.get("PARQUET", 0)),
                     _dim("history=")+str(by_kind.get("HISTORY", 0)),
                     _dim("boundaries=")+str(by_kind.get("BOUNDARIES", 0)),
        )

        # Optional: show a couple of sample paths per kind
        samples: dict[str, str] = {}
        for r in self._results:
            k = _guess_kind_from_reason_or_path(r.reason, r.parquet_path)
            if k not in samples and r.parquet_path:
                samples[k] = str(r.parquet_path)
            if len(samples) == 3:
                break
        if samples:
            logging.info("\n%s", _dim("  samples"))
            for k in ("PARQUET", "HISTORY", "BOUNDARIES"):
                if k in samples:
                    logging.info("    %-11s %s", _dim(k.lower()+":"), _left(samples[k], self.WIDTH-18))

        # Optional: show top failure reasons
        fails = Counter(r.reason for r in self._results if r.action == "FAILED").most_common(5)
        if fails:
            logging.info("\n%s", _dim("  top failures"))
            for msg, n in fails:
                logging.info("    %s ×%d", _red(_left(msg, self.WIDTH-10)), n)

        logging.info(_rule("─", self.WIDTH))



# =========================
# Data contracts
# =========================

@dataclass(frozen=True)
class MissingArtifact:
    job_hash: str
    json_path: Path
    parquet_path: Path
    job_meta: Dict[str, Any]

@dataclass(frozen=True)
class MissingMeta:
    job_hash: str
    json_path: Path
    kind: str             # "BOUNDARIES" | "HISTORY"
    target_path: Path
    reason: str

@dataclass(frozen=True)
class BackfillResult:
    job_hash: str
    parquet_path: Path
    action: str           # CREATED | UPDATED | NOOP | SKIPPED | FAILED | DRY_RUN
    reason: str


class MetaStatus(str, Enum):
    CREATED = "CREATED"
    UPDATED = "UPDATED"
    NOOP    = "NOOP"
    SKIPPED = "SKIPPED"
    FAILED  = "FAILED"


# =========================
# Log squelch (hide noisy INFO lines from downstream writers)
# =========================

class _SubstringLevelFilter(logging.Filter):
    """Hide INFO records containing any of the given substrings."""
    def __init__(self, substrings: List[str], min_level: int = logging.WARNING) -> None:
        super().__init__()
        self.substrings = tuple(substrings)
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if record.levelno < self.min_level and any(s in msg for s in self.substrings):
            return False
        return True

class _squelch:
    """Context manager to temporarily hide chatty INFO logs."""
    def __init__(self, substrings: List[str]) -> None:
        self._filter = _SubstringLevelFilter(substrings)
        self._installed_on: List[logging.Logger] = []

    def __enter__(self):
        targets = [logging.getLogger()]  # root logger
        for lg in targets:
            lg.addFilter(self._filter)
            self._installed_on.append(lg)
        return self

    def __exit__(self, exc_type, exc, tb):
        for lg in self._installed_on:
            try:
                lg.removeFilter(self._filter)
            except Exception:
                pass
        self._installed_on.clear()


# =========================
# JSON + path helpers
# =========================
def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logging.warning("Failed to read JSON %s: %s", path, e)
        return None



def _fs_keys_from_run_dir(run_base_dir: Path) -> Dict[str, str]:
    # .../src/experiment_configs/{GROUP}/results/{ARCH}/{CAMPAIGN}
    campaign = run_base_dir.name
    arch = run_base_dir.parent.name
    results_dir = run_base_dir.parent.parent
    group = results_dir.parent.name
    return {"group": group, "arch": arch, "campaign": campaign}

def _normalize_partition_value(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize a hive-style partition value for filesystem safety.

    For now we only replace '/' with '-' so that wells like '15/9-F-14'
    become '15-9-F-14' on disk, while still keeping the original label
    around if the caller wants to log it.
    """
    if value is None:
        return None, None
    raw = str(value)
    normalized = raw.replace("/", "-")
    return raw, normalized


def _normalize_well_in_meta(meta: Dict[str, Any], *, context: str) -> None:
    """
    Ensure the 'well' used for on-disk partitions is filesystem-safe.

    - Preserves the original semantic label in 'well_raw'
    - Stores the filesystem-safe value in 'well'
    - Logs the normalization once per record/context
    """
    raw = meta.get("well")
    raw_str, fs_str = _normalize_partition_value(raw)
    if not fs_str or raw_str == fs_str:
        # Nothing to normalize or already safe
        return

    # Preserve original label
    meta.setdefault("well_raw", raw_str)
    meta["well"] = fs_str

    # logging.info(
    #     "[backfill] Normalized well for filesystem (%s): raw=%s -> fs=%s",
    #     context,
    #     raw_str,
    #     fs_str,
    # )


def _manual_hive_path(
    series_store_root: Path,
    *,
    group: str,
    arch: str,
    campaign: str,
    job_hash: str,
    dataset: Optional[str],
    well: Optional[str],
) -> Path:
    """
    Build the expected series_store path for a given job.

    NOTE: the on-disk layout does NOT use a dataset=... partition, and
    well IDs may contain '/', which would otherwise create nested
    directories (e.g. well=15/9-F-14 → well=15/9/F-14).

    We therefore normalise the well for filesystem usage
    (e.g. '15/9-F-14' → '15-9-F-14') but keep the semantic label
    untouched in the JSON/job_meta.
    """
    parts = [series_store_root, Path(f"group={group}"), Path(f"arch={arch}")]
    if well:
        raw_well, well_fs = _normalize_partition_value(well)
        well_fs = well_fs or str(well)
        parts.append(Path(f"well={well_fs}"))
    parts.append(Path(f"campaign={campaign}"))
    return Path(*parts) / f"job={job_hash}.parquet"



def _boundaries_manifest_path(series_store_root: Path) -> Path:
    return (series_store_root / "meta" / "boundaries.parquet").resolve()

def _history_path(series_store_root: Path, well: str) -> Path:
    """
    Build the expected history path, normalizing the well for
    filesystem safety so we do not create nested 'well=15/9/...' trees.
    """
    raw_well, well_fs = _normalize_partition_value(well)
    well_fs = well_fs or str(well)
    return (series_store_root / "history" / f"well={well_fs}" / "history.parquet").resolve()


def _extract_well_from_history_path(p: Path) -> Optional[str]:
    """Extract well=... from a partitioned history path."""
    for part in p.parts:
        if part.startswith("well="):
            return part.split("well=", 1)[1]
    return None


# =========================
# Discovery
# =========================

def _iter_result_jsons(json_results_dir: Path) -> Iterator[Path]:
    yield from sorted(json_results_dir.glob("*.json"))

def _coalesce_job_meta(data: dict, fs_meta: Dict[str, str], job_hash: str) -> Dict[str, Any]:
    jm = data.get("job_meta") or data.get("results", {}).get("job_meta") or {}
    out = dict(jm)
    out.setdefault("group", fs_meta["group"])
    out.setdefault("arch", fs_meta["arch"])
    out.setdefault("campaign", fs_meta["campaign"])
    out.setdefault("job_hash", job_hash)
    out.setdefault("dataset", jm.get("dataset") or data.get("dataset") or data.get("results", {}).get("job_meta", {}).get("dataset"))
    out.setdefault("well", jm.get("well") or data.get("well") or data.get("results", {}).get("job_meta", {}).get("well"))
    return out

def _extract_ctx(data: dict) -> Dict[str, Any]:
    return data.get("results", {}).get("series_context") or data.get("series_context") or {}

def discover_missing_artifacts(
    run_base_dir: Path,
    series_store_root: Path,
) -> Tuple[List[MissingArtifact], List[MissingMeta]]:
    json_results_dir = run_base_dir / "results"
    missing_parquets: List[MissingArtifact] = []
    missing_meta: List[MissingMeta] = []

    if not json_results_dir.is_dir():
        return missing_parquets, missing_meta

    fs = _fs_keys_from_run_dir(run_base_dir)
    group, arch, campaign = fs["group"], fs["arch"], fs["campaign"]

    boundaries_manifest = _boundaries_manifest_path(series_store_root)
    have_boundaries_manifest = boundaries_manifest.exists()

    for json_path in _iter_result_jsons(json_results_dir):
        data = _safe_read_json(json_path)
        if not data or data.get("status") != "success":
            continue

        job_hash = json_path.stem
        jm = _coalesce_job_meta(data, fs, job_hash)
        dataset = jm.get("dataset")
        well = jm.get("well")

        # Champion parquet discovery (mesma regra do original)
        parquet_path = _manual_hive_path(
            series_store_root,
            group=group, arch=arch, campaign=campaign, job_hash=job_hash,
            dataset=dataset, well=well,
        )
        if not parquet_path.exists():
            missing_parquets.append(MissingArtifact(
                job_hash=job_hash,
                json_path=json_path,
                parquet_path=parquet_path,
                job_meta=jm,
            ))

        # Meta discovery (opportunístico) — mantém comportamento original
        ctx = _extract_ctx(data)

        # Boundaries → agenda upsert idempotente (sempre que presente no JSON)
        if isinstance(ctx.get("boundaries"), dict):
            missing_meta.append(MissingMeta(
                job_hash=job_hash,
                json_path=json_path,
                kind="BOUNDARIES",
                target_path=boundaries_manifest,
                reason=("manifest_missing" if not have_boundaries_manifest else "upsert"),
            ))

        # Full history → agenda write/extend por well
        fh = ctx.get("full_history")
        if isinstance(fh, dict) and well:
            hist_path = _history_path(series_store_root, str(well))
            missing_meta.append(MissingMeta(
                job_hash=job_hash,
                json_path=json_path,
                kind="HISTORY",
                target_path=hist_path,
                reason=("missing" if not hist_path.exists() else "maybe_update"),
            ))

    return missing_parquets, missing_meta


# =========================
# Meta de-duplication
# =========================

def _dedupe_meta(tasks: List[MissingMeta]) -> List[MissingMeta]:
    """
    Mantém no máximo:
      • 1 HISTORY por well
      • 1 BOUNDARIES por (well, group, dataset)  ← igual ao original
    Emite WARNING se boundaries para o mesmo well divergirem.
    """
    seen_hist: set[Tuple[str]] = set()
    seen_bnd: set[Tuple[Optional[str], Optional[str], Optional[str]]] = set()
    deduped: List[MissingMeta] = []

    # leitura leve dos valores de boundaries para logs de conflito
    def _read_bnd_values(p: Path) -> Optional[Tuple[int, int, int]]:
        data = _safe_read_json(p) or {}
        ctx  = (data.get("results", {}) or {}).get("series_context") or data.get("series_context") or {}
        bnd  = ctx.get("boundaries") or {}
        try:
            return int(bnd["train_end"]), int(bnd["val_end"]), int(bnd["test_start"])
        except Exception:
            return None

    kept_by_well: Dict[str, Tuple[int, int, int]] = {}

    for t in tasks:
        if t.kind == "HISTORY":
            well = _extract_well_from_history_path(t.target_path)
            key = (well or "",)
            if key not in seen_hist:
                seen_hist.add(key)
                deduped.append(t)
        elif t.kind == "BOUNDARIES":
            data = _safe_read_json(t.json_path) or {}
            jm   = (data.get("job_meta")
                    or data.get("results", {}).get("job_meta")
                    or {})
            well     = str(jm.get("well")) if jm.get("well") is not None else None
            group    = str(jm.get("group")) if jm.get("group") is not None else None
            dataset  = str(jm.get("dataset")) if jm.get("dataset") is not None else None
            key = (well, group, dataset)
            if key not in seen_bnd:
                seen_bnd.add(key)
                deduped.append(t)
                # guarda valores por well para detectar divergências entre datasets/grupos
                if well:
                    vals = _read_bnd_values(t.json_path)
                    if vals and well not in kept_by_well:
                        kept_by_well[well] = vals
            else:
                # conflito potencial: para o mesmo well, valores diferentes
                if well and well in kept_by_well:
                    cur = _read_bnd_values(t.json_path)
                    if cur and cur != kept_by_well[well]:
                        logging.warning(
                            "BOUNDARIES conflict for well=%s across (group,dataset): kept=%s, skipped=%s",
                            well, kept_by_well[well], cur
                        )
        else:
            deduped.append(t)

    return deduped


# =========================
# Tiny file lock (cross-platform)
# =========================

# Imports opcionais para portabilidade
try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None  # sentinel

try:
    import msvcrt  # type: ignore
except Exception:
    msvcrt = None  # sentinel


class _FileLock:
    """
    Lock exclusivo de arquivo:
      • POSIX: fcntl.flock
      • Windows: msvcrt.locking
      • Fallback: no-op (ainda cria o arquivo para coordenar diretórios)
    """
    def __init__(self, path: Path):
        self.path = path
        self._fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # abrir sempre em modo append para coexistir com múltiplos processos
        self._fd = open(self.path, "a+b")
        if fcntl is not None:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        elif msvcrt is not None:
            try:
                msvcrt.locking(self._fd.fileno(), msvcrt.LK_LOCK, 1)
            except Exception:
                pass  # último recurso: prossegue sem lock duro
        # fallback: no-op lock
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if fcntl is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            elif msvcrt is not None:
                try:
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
        finally:
            try:
                self._fd.close()
            except Exception:
                pass



# =========================
# Meta validations (soft — apenas warnings, não bloqueiam escrita)
# =========================

def _soft_check_boundaries(bnd: Any, *, well: Any) -> None:
    try:
        if not isinstance(bnd, dict):
            logging.warning("Boundaries not a dict for well=%s", well); return
        te, ve, ts = int(bnd["train_end"]), int(bnd["val_end"]), int(bnd["test_start"])
        if min(te, ve, ts) < 0:
            logging.warning("Boundaries negative index for well=%s: %s", well, (te, ve, ts))
        if not (te <= ve < ts):
            logging.warning("Boundaries non-increasing for well=%s: %s", well, (te, ve, ts))
    except Exception as e:
        logging.warning("Boundaries malformed for well=%s: %s", well, e)

def _soft_check_full_history(fh: Any, *, well: Any) -> None:
    try:
        if not isinstance(fh, dict):
            logging.warning("full_history not a dict for well=%s", well); return
        t, y = fh.get("t"), fh.get("ytrue")
        if not (isinstance(t, list) and isinstance(y, list)):
            logging.warning("full_history t/ytrue not lists for well=%s", well); return
        if len(t) != len(y) or len(t) == 0:
            logging.warning("full_history length mismatch/empty for well=%s (t=%d,y=%d)", well, len(t), len(y))
        # checagem leve de monotonia
        last = None
        for i, tv in enumerate(t):
            try:
                v = float(tv)
            except Exception:
                logging.warning("full_history non-numeric t at pos %d for well=%s", i, well); break
            if last is not None and not (v > last):
                logging.warning("full_history t not strictly increasing at pos %d for well=%s", i, well); break
            last = v
    except Exception as e:
        logging.warning("full_history malformed for well=%s: %s", well, e)


# =========================
# Meta write with on-disk verification
# =========================

def _persist_meta_checked(
    *,
    kind: str,                    # "BOUNDARIES" | "HISTORY"
    record: Dict,
    settings: SeriesStoreSettings,
    series_store_root: Path,
) -> Tuple[MetaStatus, Optional[Path], str]:
    meta = (record or {}).get("meta", {}) or {}
    ctx  = (record or {}).get("context", {}) or {}

    # Normalise well once here so both the writer and our expected paths
    # see the same partition value.
    _normalize_well_in_meta(meta, context=f"meta kind={kind} job={meta.get('job_hash', '?')}")

    def _bnd_path() -> Path:
        return (series_store_root / "meta" / "boundaries.parquet").resolve()

    def _hist_path(well: str) -> Path:
        return _history_path(series_store_root, well)

    if kind == "BOUNDARIES":
        bnd = ctx.get("boundaries") or {}
        if not isinstance(bnd, dict) or not all(k in bnd for k in ("train_end", "val_end", "test_start")):
            return MetaStatus.SKIPPED, None, "no boundaries in JSON"
        # soft validations (do not block writes)
        _soft_check_boundaries(bnd, well=meta.get("well_raw") or meta.get("well"))
        path = _bnd_path()
        existed_before = path.exists()
        try:
            # serialize writes to the single manifest file
            lockfile = series_store_root / "meta" / ".boundaries.lock"
            with _FileLock(lockfile):
                persist_boundaries_manifest(record, settings)
        except Exception as e:
            return MetaStatus.FAILED, path, f"writer error: {e}"
        if not path.exists():
            return MetaStatus.FAILED, path, "writer returned but file missing"

        # logging.info(
        #     "[backfill] BOUNDARIES manifest updated: path=%s (existed_before=%s)",
        #     path,
        #     existed_before,
        # )
        return (MetaStatus.UPDATED if existed_before else MetaStatus.CREATED), path, "upserted"

    if kind == "HISTORY":
        well = meta.get("well")
        fh   = ctx.get("full_history") or {}
        if not well:
            return MetaStatus.SKIPPED, None, "missing well id"
        if not isinstance(fh, dict) or not fh.get("t") or not fh.get("ytrue"):
            return MetaStatus.SKIPPED, None, "no full_history in JSON"
        # soft validations (do not block writes)
        _soft_check_full_history(fh, well=meta.get("well_raw") or well)
        path = _hist_path(str(well))
        existed_before = path.exists()
        try:
            # per-well lock to avoid races — safe and compatible
            lockfile = path.parent / ".history.lock"
            with _FileLock(lockfile):
                persist_full_history(record, settings)
        except Exception as e:
            return MetaStatus.FAILED, path, f"writer error: {e}"
        if not path.exists():
            return MetaStatus.FAILED, path, "writer returned but file missing"

        # logging.info(
        #     "[backfill] HISTORY path for well: raw=%s fs=%s -> %s (existed_before=%s)",
        #     meta.get("well_raw") or well,
        #     well,
        #     path,
        #     existed_before,
        # )
        return (MetaStatus.UPDATED if existed_before else MetaStatus.CREATED), path, "written"

    return MetaStatus.FAILED, None, f"unknown meta kind: {kind}"



# =========================
# Backfill Orchestrator
# =========================

def run_backfill(
    results_root: Path,
    series_store_root: Path,
    *,
    concurrency: int = 8,
    dry_run: bool = True,
) -> List[BackfillResult]:
    results_root = results_root.resolve()
    series_store_root = series_store_root.resolve()

    # clamp concorrência (evita 0/negativo virem de CLI/env)
    workers = max(1, int(concurrency))

    # Pretty banner (replaces _banner)
    _reporter = _BackfillReporter(
        run_name=results_root.name,
        series_root=series_store_root,
        dry_run=dry_run,
        workers=workers,
    )
    _reporter.banner()

    # Discover work
    missing_parquets, missing_meta = discover_missing_artifacts(
        run_base_dir=results_root,
        series_store_root=series_store_root,
    )
    missing_meta = _dedupe_meta(missing_meta)
    _reporter.tasks_line(len(missing_parquets), len(missing_meta))

    if not missing_parquets and not missing_meta:
        logging.info(_dim("Nothing to do."))
        return []

    # Settings (unchanged)
    settings = SeriesStoreSettings(
        enabled=not dry_run,
        base_dir=str(series_store_root),
        format="parquet",
        compress="zstd",
        schema_version=1,
        self_heal=False,
    )

    results: List[BackfillResult] = []

    # Writers (kept, but with well-normalization)
    def _write_parquet(art: MissingArtifact) -> BackfillResult:
        if dry_run:
            return BackfillResult(
                art.job_hash,
                art.parquet_path,
                "DRY_RUN",
                f"Would create {art.parquet_path}",
            )

        data = _safe_read_json(art.json_path)
        if not data:
            return BackfillResult(
                art.job_hash,
                art.parquet_path,
                "FAILED",
                "read JSON",
            )

        fs_meta = _fs_keys_from_run_dir(results_root)
        data.setdefault("job_meta", {})
        data["job_meta"].setdefault("group", fs_meta["group"])
        data["job_meta"].setdefault("arch", fs_meta["arch"])
        data["job_meta"].setdefault("campaign", fs_meta["campaign"])
        data["job_meta"].setdefault("job_hash", art.job_hash)

        record, _ = build_series_record(data)
        if not record.get("series_rows"):
            return BackfillResult(
                art.job_hash,
                art.parquet_path,
                "SKIPPED",
                "empty series",
            )

        # Normalise well in meta so the on-disk layout does not create
        # nested 'well=15/9/...' directories.
        meta = record.get("meta") or {}
        _normalize_well_in_meta(meta, context=f"series job={art.job_hash}")
        record["meta"] = meta

        try:
            out_path = persist_series(record, settings)
            return BackfillResult(
                art.job_hash,
                out_path,
                "CREATED",
                "series",
            )
        except Exception as e:
            return BackfillResult(
                art.job_hash,
                art.parquet_path,
                "FAILED",
                f"series writer: {e}",
            )

    def _write_meta(task: MissingMeta) -> BackfillResult:
        if dry_run:
            return BackfillResult(
                task.job_hash,
                task.target_path,
                "DRY_RUN",
                f"{task.kind} planned ({task.reason})",
            )

        data = _safe_read_json(task.json_path)
        if not data:
            return BackfillResult(
                task.job_hash,
                task.target_path,
                "FAILED",
                f"{task.kind}: read JSON",
            )

        record, _ = build_series_record(data)
        status, out_path, detail = _persist_meta_checked(
            kind=task.kind,
            record=record,
            settings=settings,
            series_store_root=series_store_root,
        )
        return BackfillResult(
            task.job_hash,
            out_path or task.target_path,
            status.value,
            f"{task.kind}: {detail}",
        )

    noisy_markers = [
        "full_history exists (noop)",
        "boundaries unchanged (noop)",
        "⚠️  [series_store] full_history exists",
        "⚠️  [series_store] boundaries unchanged",
        "[series_store] WROTE",
        " wrote full_history",
        " upserted boundaries",
        " SKIP (exists): ",
        "SKIP (exists): /",
    ]

    with _squelch(noisy_markers):
        # 1) Parquet champions in parallel
        if missing_parquets:
            try:
                from tqdm import tqdm  # optional
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results.extend(
                        list(
                            tqdm(
                                pool.map(_write_parquet, missing_parquets),
                                total=len(missing_parquets),
                                desc=f"Parquet {results_root.name}",
                            )
                        )
                    )
            except Exception:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results.extend(list(pool.map(_write_parquet, missing_parquets)))

        # 2) Meta tasks: HISTORY in parallel; BOUNDARIES serial (single lock)
        bnd_tasks = [t for t in missing_meta if t.kind == "BOUNDARIES"]
        hist_tasks = [t for t in missing_meta if t.kind == "HISTORY"]

        if hist_tasks:
            try:
                from tqdm import tqdm
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results.extend(
                        list(
                            tqdm(
                                pool.map(_write_meta, hist_tasks),
                                total=len(hist_tasks),
                                desc=f"Meta(HISTORY) {results_root.name}",
                            )
                        )
                    )
            except Exception:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results.extend(list(pool.map(_write_meta, hist_tasks)))

        if bnd_tasks:
            try:
                from tqdm import tqdm
                for t in tqdm(
                    bnd_tasks,
                    total=len(bnd_tasks),
                    desc=f"Meta(BOUNDARIES) {results_root.name}",
                ):
                    results.append(_write_meta(t))
            except Exception:
                for t in bnd_tasks:
                    results.append(_write_meta(t))

    # Filesystem post-check (kept)
    bnd = _boundaries_manifest_path(series_store_root)
    sample_hist = _history_path(series_store_root, "P11")
    if not bnd.exists() or not sample_hist.exists():
        logging.warning(
            "Post-check: meta missing — boundaries=%s history(P11)=%s",
            "OK" if bnd.exists() else "MISSING",
            "OK" if sample_hist.exists() else "MISSING",
        )

    # Pretty summary
    _reporter.ingest_many(results)
    _reporter.render_summary()
    return results




# =========================
# Quick diagnostics 
# =========================

def quick_meta_diagnostics(results_dir: Path, series_store_root: Path) -> None:
    """
    Fast reality check:
      - Do JSONs contain boundaries/full_history?
      - Do expected meta files exist?
    """
    jsons = sorted((results_dir / "results").glob("*.json"))
    have_bnd = have_hist = 0
    sample_well = None

    for p in jsons:
        data = _safe_read_json(p)
        if not data:
            continue
        ctx = (data.get("results", {}) or {}).get("series_context") or data.get("series_context") or {}
        if isinstance(ctx.get("boundaries"), dict) and all(k in ctx["boundaries"] for k in ("train_end", "val_end", "test_start")):
            have_bnd += 1
        fh = ctx.get("full_history")
        if isinstance(fh, dict) and fh.get("t") and fh.get("ytrue"):
            have_hist += 1
            if not sample_well:
                jm = (data.get("job_meta") or data.get("results", {}).get("job_meta") or {})
                sample_well = jm.get("well") or data.get("well")

    bnd_path = (series_store_root / "meta" / "boundaries.parquet").resolve()
    hist_path = (series_store_root / "history" / f"well={sample_well}" / "history.parquet").resolve() if sample_well else None

    logging.info("Diag: JSONs with boundaries=%d / %d", have_bnd, len(jsons))
    logging.info("Diag: JSONs with full_history=%d / %d", have_hist, len(jsons))
    logging.info("Diag: boundaries.parquet exists? %s", "YES" if bnd_path.exists() else "NO")
    if sample_well:
        logging.info("Diag: history for well=%s exists? %s", sample_well, "YES" if hist_path.exists() else "NO")
    else:
        logging.info("Diag: no sample_well with full_history found in JSONs")
