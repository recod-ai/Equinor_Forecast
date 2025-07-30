# forecast_pipeline/runner_external.py
# runner_external.py  (topo do arquivo)
import os, random, numpy as np, tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"        # força kernels determinísticos
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"      # idem CuDNN
os.environ["TF_GPU_THREAD_MODE"] = "single"     # 1 thread por kernel
os.environ["OMP_NUM_THREADS"] = "1"             # BLAS 1 thread
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# … só depois importe o resto do seu código/modelos

"""
Executa **um** JobTuple serializado em JSON.

Uso:
    python -m forecast_pipeline.scripts.runner_external <job.json> <run_name> <exp_out_dir>
"""
import json, sys, pathlib, logging
from forecast_pipeline.runner import execute_jobs_robust

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")

if __name__ == "__main__":
    job_file, run_name, exp_dir = map(pathlib.Path, sys.argv[1:4])

    # carrega JobTuple único
    job = json.loads(job_file.read_text())

    # executa de forma determinística (máx. 1 worker interno)
    execute_jobs_robust([job],
                        base_output_dir=str(exp_dir),
                        run_name=run_name.name,   # mesma convenção que antes
                        max_workers=1)            # ← garante determinismo



# # forecast_pipeline/runner.py   (ou onde você mantiver essa função)
# from __future__ import annotations
# import json, shutil, subprocess, sys, tempfile, logging, inspect
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import Optional


# # --------------------------------------------------------------------- #
# # Helper notebook-safe para achar o runner externo
# # --------------------------------------------------------------------- #
# def _locate_runner() -> Path:
#     """
#     Resolve forecast_pipeline/runner_external.py independentemente de
#     estarmos num script (onde __file__ existe) ou num notebook.
#     """
#     # 1) Mesmo diretório deste arquivo (funciona em script normal)
#     try:
#         base = Path(__file__).resolve().parent
#         p = base / "runner_external.py"
#         if p.exists():
#             return p
#     except NameError:
#         # Estamos em notebook
#         pass

#     # 2) Diretório onde esta função foi definida
#     base = Path(inspect.getfile(_locate_runner)).resolve().parent
#     p = base / "runner_external.py"
#     if p.exists():
#         return p

#     # 3) Fallback: current working dir
#     root = Path.cwd().parents[1]  # /home/gabriel/Documentos/Equinor
#     p = root / "src/forecast_pipeline/runner_external.py"
#     if p.exists():
#         return p

#     raise FileNotFoundError("runner_external.py não encontrado; "
#                             "coloque-o dentro de forecast_pipeline ou ao lado do notebook.")

# # --------------------------------------------------------------------- #
# # Função principal
# # --------------------------------------------------------------------- #
# def run_robust_pipeline(profile_path: str, ensemble_size: int) -> Optional[str]:
#     """
#     Executa uma lista de *jobs* de forma determinística, despachando **um
#     processo Python por job**.  Compatível com o pipeline original.
#     """
#     logging.info("— Pipeline ROBUST para profile: %s —", profile_path)

#     # 1. Gerar jobs a partir do profile ------------------------------------------------
#     sources = select_data_sources(DATA_SOURCES, DEFAULT_DATASET)
#     default_params = {**vars(DefaultExperimentParams()),
#                       "ensemble_models": ensemble_size}

#     jobs = generate_jobs(sources, default_params, profile_path=profile_path)
#     if not jobs:
#         logging.warning("No jobs generated from profile. Exiting.")
#         return None

#     # 2. Pastas de saída ---------------------------------------------------------------
#     run_name  = Path(profile_path).stem          # ex.: F12_RePIN_cycle_1
#     run_dir   = EXPERIMENTS_OUTPUT_DIR / run_name
#     results_dir = run_dir / "results"
#     results_dir.mkdir(parents=True, exist_ok=True)

#     # 3. Serializa cada job num tmpdir -------------------------------------------------
#     tmp_dir = Path(tempfile.mkdtemp(prefix="jobs_"))
#     job_files = []
#     for job in jobs:
#         fp = tmp_dir / f"{job[3]}.json"          # job[3] é experiment_id
#         fp.write_text(json.dumps(job, default=str))
#         job_files.append(fp)

#     # 4. Localiza runner externo (resolve notebook vs script) -------------------------
#     script_path = _locate_runner()

#     # 5. Dispara subprocessos independentes -------------------------------------------
#     max_procs = min(MAX_WORKERS or 1, len(job_files)) or 1
#     logging.info("Launching %d subprocesses (max %d parallel)…",
#                  len(job_files), max_procs)

#     def _launch(fp: Path):
#         cmd = [
#             sys.executable, str(script_path),
#             str(fp),                     # job-file
#             str(run_name),               # sub-folder name
#             str(EXPERIMENTS_OUTPUT_DIR)  # root output dir
#         ]
#         return subprocess.run(cmd, capture_output=True, check=True)

#     with ThreadPoolExecutor(max_workers=max_procs) as pool:
#         fut2file = {pool.submit(_launch, f): f for f in job_files}
#         for fut in as_completed(fut2file):
#             f = fut2file[fut]
#             try:
#                 fut.result()
#             except subprocess.CalledProcessError as e:
#                 logging.error("Job %s failed:\n%s", f.name,
#                               e.stderr.decode(errors='ignore')[:500])
#             finally:
#                 f.unlink(missing_ok=True)   # remove json

#     shutil.rmtree(tmp_dir, ignore_errors=True)

#     # 6. Pós-processamento ------------------------------------------------------------
#     run_output_dir = str(run_dir)
#     logging.info("Run complete. Collating results from: %s", run_output_dir)

#     leaderboard_df = collate_robust_results(run_output_dir)
#     if not leaderboard_df.empty:
#         lb_path = Path(run_output_dir) / "leaderboard.csv"
#         leaderboard_df.to_csv(lb_path, index=False)
#         logging.info("Leaderboard salvo em %s", lb_path)
#         try:
#             print(leaderboard_df.sort_values("val_smape_cum").head(5))
#         except KeyError:
#             print(leaderboard_df.head(5))
#     else:
#         logging.warning("Nenhum job bem-sucedido para gerar leaderboard.")

#     return run_output_dir
