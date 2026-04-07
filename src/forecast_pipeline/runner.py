# src/forecast_pipeline/runner.py

# ─── Standard library imports ────────────────────────────────────────────────────
from __future__ import annotations
import threading, os            # já existem? adiciona se faltar
_SUBPROC_ENV = os.environ.copy()
_SUBPROC_ENV["MPLBACKEND"] = "Agg"
import logging                                     # logging utilities
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple         # tipos para annotations
import datetime
from pathlib import Path 

# ─── Third-party imports ─────────────────────────────────────────────────────────
from more_itertools import chunked                 # advanced iteration
from tqdm.auto import tqdm                        # progress bars
from IPython.utils import io                       # capture output
import pandas as pd

# ─── Local application imports ───────────────────────────────────────────────────
from .config import DEFAULT_EXP_PARAMS, LOG_LEVEL, MAX_WORKERS, EXPERIMENTS_OUTPUT_DIR
from .jobs import generate_jobs, run_single_job
from .metrics import collate_metrics, clean_and_structure_results
from contextlib import nullcontext
from profile_manager import generate_job_hash
from .config import EXPERIMENTS_OUTPUT_DIR

# NEW: unified logging utilities
from .logging_utils import (
    get_logger,
    phase,
    log_context,
    get_process_pool_initializer,
)

#All logging calls can use the logger (with automatic context).
logger = get_logger("runner")


JobTuple = Tuple[Dict[str, Any], str, Dict[str, Any], int]     # alias

# ---------------------------------------------------------------------------
# Module-level helper so it is picklable by multiprocessing
# ---------------------------------------------------------------------------
_JOBS_REF: list[JobTuple] | None = None        # will be set inside execute_jobs

def _run_from_index(idx: int):
    """Look up the job in the global list and execute it."""
    from .jobs import run_single_job           # local import avoids circularity
    assert _JOBS_REF is not None, "Job list not initialised"
    return run_single_job(_JOBS_REF[idx])



# ────────────────────────────────────────────────────────────────────────────
#  NEW execute_jobs – persistent pool, tiny pickles, safe interrupt
# ────────────────────────────────────────────────────────────────────────────
def execute_jobs(jobs: List[JobTuple]) -> List[Dict[str, Any]]:
    """
    Execute all jobs in a single ProcessPoolExecutor instance.

    Improvements over naive versions:
      1) Single pool for all jobs (lower spin-up overhead).
      2) 'Index trick': workers receive small ints instead of large pickles.
      3) Chunked submission for bounded in-flight futures.
      4) Graceful Ctrl-C with pool shutdown.
    """
    if not jobs:
        return []

    # Single-job fast path (avoid pool startup cost)
    if len(jobs) == 1:
        logging.info("Only one job: running sequentially.")
        return [run_single_job(jobs[0])]

    max_workers = min(MAX_WORKERS, len(jobs))
    pbar = tqdm(total=len(jobs), desc="Jobs", unit="job")
    results: List[Dict[str, Any]] = []

    def _handle_future(fut, idx: int) -> None:
        """Collect result / error for a finished future."""
        _, well, _, job_id = jobs[idx]
        try:
            results.append(fut.result())
        except Exception as e:  # noqa: BLE001
            logging.error(
                "Job %s (%s) failed: %s", job_id, well, e, exc_info=(LOG_LEVEL >= 2)
            )
            results.append(
                {"status": "failure", "error": str(e), "well": well, "experiment_id": job_id}
            )
        finally:
            pbar.update(1)

    # ---- Process pool setup (unified initializer) ----
    init_fn, init_args = get_process_pool_initializer(LOG_LEVEL)
    CHUNK = 1_000  # max in-flight futures

    # Make jobs visible to workers through a module-level ref
    global _JOBS_REF
    _JOBS_REF = jobs

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_fn,
            initargs=init_args,
        ) as pool:
            # Choose capture context once (reduces nested 'with' noise)
            if DEFAULT_EXP_PARAMS.get("plot", False) or LOG_LEVEL >= 2:
                cap_ctx = nullcontext()
            elif LOG_LEVEL == 1:
                cap_ctx = io.capture_output(stdout=True, stderr=False)
            else:
                cap_ctx = io.capture_output(stdout=True, stderr=True)

            with cap_ctx:
                for start in range(0, len(jobs), CHUNK):
                    batch = range(start, min(start + CHUNK, len(jobs)))
                    fut_to_idx = {pool.submit(_run_from_index, i): i for i in batch}

                    for fut in as_completed(fut_to_idx):
                        _handle_future(fut, fut_to_idx[fut])

    except KeyboardInterrupt:
        # pool object exists here; ensure we try to cancel remaining work
        logging.warning("Interrupted by user – shutting down pool…", exc_info=False)
        try:
            pool.shutdown(cancel_futures=True)
        except Exception:
            pass
        raise
    finally:
        pbar.close()

    return results


def execute_jobs_robust(
    jobs: List[Tuple],
    base_output_dir: Optional[str],
    run_name: str,
    max_workers: int | None = None,
) -> str:
    """
    Executes jobs robustly and idempotently, persisting each job result atomically.

    Side effects:
      - Creates <base_output_dir>/<run_name>/results
      - Writes per-job JSONs to .../results/<job_hash>.json
      - Writes run_summary.csv to .../<run_name>/run_summary.csv
    """
    # 1) Resolve the intended root (honor base_output_dir if provided)
    try:
        base_root = Path(base_output_dir).resolve() if base_output_dir else Path(EXPERIMENTS_OUTPUT_DIR).resolve()
    except Exception:
        base_root = Path(EXPERIMENTS_OUTPUT_DIR).resolve()

    base_root.mkdir(parents=True, exist_ok=True)
    run_dir = base_root / run_name
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2) Determine which jobs still need to run (skip already-persisted)
    jobs_to_run: list[Tuple] = []
    for job in jobs:
        _, _, params, _ = job
        job_hash = generate_job_hash(params)  # compute before mutating params
        expected_file = results_dir / f"{job_hash}.json"

        if expected_file.exists():
            logging.debug("Skipping job %s: result already exists.", job_hash[:8])
            continue

        # Force canonical results dir into params so workers persist in the right place
        params["run_output_dir"] = str(results_dir)
        jobs_to_run.append(job)

    total_jobs = len(jobs)
    completed = total_jobs - len(jobs_to_run)
    logging.info("Found %d completed jobs. Submitting %d new jobs.", completed, len(jobs_to_run))

    if not jobs_to_run:
        logging.info("All jobs for this run are already complete.")
        (run_dir / "run_summary.csv").write_text("", encoding="utf-8")
        return str(run_dir)

    # 3) Parallel configuration
    if max_workers is None:
        max_workers = min(MAX_WORKERS, len(jobs_to_run))

    pbar = tqdm(total=len(jobs_to_run), desc="Jobs", unit="job")

    # Capture context for cleaner logs
    if DEFAULT_EXP_PARAMS.get("plot", False) or LOG_LEVEL >= 2:
        cap_ctx = nullcontext()
    elif LOG_LEVEL == 1:
        cap_ctx = io.capture_output(stdout=True, stderr=False)
    else:
        cap_ctx = io.capture_output(stdout=True, stderr=True)

    summaries: list[dict] = []

    # Unified initializer
    init_fn, init_args = get_process_pool_initializer(LOG_LEVEL)

    # 4) Execute with robustness
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_fn,          # <-- fixed: use unified initializer
            initargs=init_args,
        ) as executor, cap_ctx:

            future_to_job = {executor.submit(run_single_job, job, True): job for job in jobs_to_run}

            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                _, well, _, job_id = job
                try:
                    summary = fut.result()
                    summaries.append(summary)
                    if summary.get("status") == "failure":
                        logging.warning("Job for well %s failed: %s", summary.get("well"), summary.get("error"))
                except Exception as e:  # noqa: BLE001
                    logging.error(
                        "Job %s (%s) failed: %s", job_id, well, e, exc_info=(LOG_LEVEL >= 2)
                    )
                    summaries.append(
                        {"status": "failure", "error": str(e), "well": well, "experiment_id": job_id}
                    )
                finally:
                    pbar.update(1)

    except KeyboardInterrupt:
        logging.warning("Interrupted by user – shutting down pool…", exc_info=False)
        try:
            executor.shutdown(cancel_futures=True)
        except Exception:
            pass
        raise
    finally:
        pbar.close()

    # 5) Persist a compact run summary (for quick inspection)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(run_dir / "run_summary.csv", index=False)
    logging.info("Robust execution run finished. Summary saved to run_summary.csv")

    return str(run_dir)



def run_experiments_for_config(
    filter_cfg: Dict[str, Any],
    data_sources: List[Dict[str, Any]],
    ensemble_size: int,
    profile_path: Optional[str] = None,
):
    """
    Full pipeline: generate → execute → collate → clean.

    Pass Darts knobs inside `filter_cfg` to trigger family expansion:
      - filter_cfg['darts_model_keys'] = ["TiDE","ARIMA",...]
      - filter_cfg['darts_profile_limit'] = 2
      - filter_cfg['expand_darts_profiles'] = True  # when architecture_name="Darts_*"
    """
    key = f"adaptive_{filter_cfg.get('apply_adaptive_filtering', False)}_method_{filter_cfg.get('filter_method','None')}"
    params = {**DEFAULT_EXP_PARAMS, "ensemble_models": ensemble_size, **filter_cfg}

    jobs = generate_jobs(data_sources, params, profile_path=profile_path)
    logging.info("Dispatching %d jobs", len(jobs))

    raw = execute_jobs(jobs)  # unchanged; uses run_single_job internally
    dfs = collate_metrics(raw)

    structured = clean_and_structure_results(
        dfs["df_global"], dfs["df_agg"], dfs["df_cum"],
        dfs["df_slice_global"], dfs["df_slice_agg"], dfs["df_slice_cum"],
        {
            "adaptive_filter": filter_cfg.get("apply_adaptive_filtering", False),
            "filter_method": filter_cfg.get("filter_method", "None"),
        },
        remove_cols=not filter_cfg.get("apply_adaptive_filtering", False),
    )
    return {key: structured}



