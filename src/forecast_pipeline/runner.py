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


def execute_jobs(
    jobs: List[Tuple[Dict[str, Any], str, Dict[str, Any], int]]
) -> List[Dict[str, Any]]:
    """
    Run experiments (sequential or parallel) with configurable stdout/stderr capture.
    """
    results: List[Dict[str, Any]] = []

    # single-job shortcut
    if len(jobs) == 1:
        logging.info("Only one job: running sequentially")
        return [run_single_job(jobs[0])]

    max_workers = min(MAX_WORKERS, len(jobs))
    pbar = tqdm(total=len(jobs), desc="Jobs completed", unit="job")

    def _dispatch_batch(batch):
        with ProcessPoolExecutor(max_workers=max_workers) as exec:
            futures = {exec.submit(run_single_job, j): j for j in batch}
            for fut in as_completed(futures):
                _, well, _, job_id = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    logging.error(f"Job {job_id} failed in pool: {e}", exc_info=(LOG_LEVEL >= 2))
                    results.append({"status":"failure","error":str(e),"well":well,"experiment_id":job_id})
                finally:
                    pbar.update(1)

    def _run_all():
        for batch in chunked(jobs, max_workers):
            _dispatch_batch(batch)

    plot_on = DEFAULT_EXP_PARAMS.get("plot", False)
    if plot_on or LOG_LEVEL >= 2:
        _run_all()
    elif LOG_LEVEL == 1:
        with io.capture_output(stdout=True, stderr=False):
            _run_all()
    else:  # LOG_LEVEL == 0
        with io.capture_output(stdout=True, stderr=True):
            _run_all()

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




