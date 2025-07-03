# ─── Standard library imports ────────────────────────────────────────────────────
import logging                                     # logging utilities
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple         # tipos para annotations

# ─── Third-party imports ─────────────────────────────────────────────────────────
from more_itertools import chunked                 # advanced iteration
from tqdm.auto import tqdm                        # progress bars
from IPython.utils import io                       # capture output

# ─── Local application imports ───────────────────────────────────────────────────
from .config import DEFAULT_EXP_PARAMS, LOG_LEVEL, MAX_WORKERS
from .jobs import generate_jobs, run_single_job
from .metrics import collate_metrics, clean_and_structure_results


# def execute_jobs(
#     jobs: List[Tuple[Dict[str, Any], str, Dict[str, Any], int]]
# ) -> List[Dict[str, Any]]:
#     """
#     Run experiments (sequential or parallel) with configurable stdout/stderr capture.
#     """
#     results: List[Dict[str, Any]] = []

#     # single-job shortcut
#     if len(jobs) == 1:
#         logging.info("Only one job: running sequentially")
#         return [run_single_job(jobs[0])]

#     max_workers = min(MAX_WORKERS, len(jobs))
#     pbar = tqdm(total=len(jobs), desc="Jobs completed", unit="job")

#     def _dispatch_batch(batch):
#         with ProcessPoolExecutor(max_workers=max_workers) as exec:
#             futures = {exec.submit(run_single_job, j): j for j in batch}
#             for fut in as_completed(futures):
#                 _, well, _, job_id = futures[fut]
#                 try:
#                     results.append(fut.result())
#                 except Exception as e:
#                     logging.error(f"Job {job_id} failed in pool: {e}", exc_info=(LOG_LEVEL >= 2))
#                     results.append({"status":"failure","error":str(e),"well":well,"experiment_id":job_id})
#                 finally:
#                     pbar.update(1)

#     def _run_all():
#         for batch in chunked(jobs, max_workers):
#             _dispatch_batch(batch)

#     plot_on = DEFAULT_EXP_PARAMS.get("plot", False)
#     if plot_on or LOG_LEVEL >= 2:
#         _run_all()
#     elif LOG_LEVEL == 1:
#         with io.capture_output(stdout=True, stderr=False):
#             _run_all()
#     else:  # LOG_LEVEL == 0
#         with io.capture_output(stdout=True, stderr=True):
#             _run_all()

#     pbar.close()
#     return results


# ─── runner.py ──────────────────────────────────────────────────────────────
from contextlib import nullcontext  


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
    Execute all jobs in a *single* ProcessPoolExecutor instance.

    Key improvements vs. the old version
    ------------------------------------
    1. **One pool for all jobs** – processes are started only once.
    2. **Index trick** – we pass an *int* to each worker instead of pickling
       the whole job tuple (which can be large).
    3. **Chunked submission** – keeps the outstanding-future list bounded.
    4. **Graceful Ctrl-C** – cancels remaining futures and closes the pool.
    """
    if not jobs:
        return []

    # Single-job fast-path stays sequential (avoids pool spin-up cost)
    if len(jobs) == 1:
        logging.info("Only one job: running sequentially.")
        return [run_single_job(jobs[0])]

    max_workers = min(MAX_WORKERS, len(jobs))
    pbar = tqdm(total=len(jobs), desc="Jobs", unit="job")
    results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    def _worker_init(level: int):
        """Set log level once per process (runs *inside* each worker)."""
        logging.getLogger().setLevel(level)

    def _run(idx: int):
        """Lookup the job in the shared list and execute it."""
        return run_single_job(jobs[idx])

    def _handle_future(fut, idx: int):
        """Collect result / error for a finished future."""
        _, well, _, job_id = jobs[idx]
        try:
            results.append(fut.result())
        except Exception as e:               # noqa: BLE001
            logging.error("Job %d (%s) failed: %s", job_id, well, e,
                          exc_info=(LOG_LEVEL >= 2))
            results.append(
                {"status": "failure", "error": str(e),
                 "well": well, "experiment_id": job_id}
            )
        finally:
            pbar.update(1)

    # ------------------------------------------------------------------ #
    # Pool execution
    # ------------------------------------------------------------------ #
    CHUNK = 1_000            # how many futures to keep in flight at once
    global _JOBS_REF
    _JOBS_REF = jobs 
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(LOG_LEVEL,),
        ) as pool:

            # choose capture context once to avoid nested `with` each loop
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

    except KeyboardInterrupt:                     # graceful ^C handling
        logging.warning("Interrupted by user – shutting down pool…", exc_info=False)
        pool.shutdown(cancel_futures=True)
        raise

    finally:
        pbar.close()

    return results



def run_experiments_for_config(filter_cfg, data_sources, ensemble_size):
    """Full pipeline: generate → execute → collate → clean."""
    key = f"adaptive_{filter_cfg['apply_adaptive_filtering']}_method_{filter_cfg.get('filter_method','None')}"
    params = {**DEFAULT_EXP_PARAMS, "ensemble_models": ensemble_size, **filter_cfg}
    jobs = generate_jobs(data_sources, params)
    logging.info(f"Dispatching {len(jobs)} jobs")
    raw = execute_jobs(jobs)
    dfs = collate_metrics(raw)
    
    structured = clean_and_structure_results(
        dfs["df_global"], dfs["df_agg"], dfs["df_cum"],
        dfs["df_slice_global"], dfs["df_slice_agg"], dfs["df_slice_cum"],
        {"adaptive_filter": filter_cfg["apply_adaptive_filtering"],
         "filter_method": filter_cfg.get("filter_method","None")},
        remove_cols=not filter_cfg["apply_adaptive_filtering"]
    )
    

    return {key: structured}




