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
