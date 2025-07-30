# -*- coding: utf-8 -*-
"""
Sub-runner “à prova de Jupyter”.

Uso:  python -m forecast_pipeline.cli_subprocess <job.json>

stdout ► uma única linha:  __RESULT__ <absolute-path-to-pickle>
stderr ► tudo que for print, logging, warnings, display()
"""

from __future__ import annotations
# ─── CONFIGURAÇÃO ESSENCIAL (NOVO) ──────────────────────────────────────
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Subprocess:%(process)d] - %(levelname)s - %(message)s',
    stream=sys.stderr  # Garante que vá para o stderr
)

import contextlib, json, os, pickle, sys, tempfile, warnings
from pathlib import Path

# ─── inibir janelas / abas ────────────────────────────────────────────────
os.environ["MPLBACKEND"] = "Agg"
os.environ["PLOTLY_RENDERER"] = "json"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt  # noqa
_plt.show = lambda *a, **k: None  # noqa: E731

import shap, plotly.io as pio
shap.initjs = lambda *a, **k: None
pio.show = lambda *a, **k: None




# bloqueia qualquer tentativa de abrir navegador
import webbrowser  # noqa
webbrowser.open = lambda *a, **k: False  # type: ignore

# bloqueia IPython.display.display
try:
    import IPython.display as _ip_disp  # noqa

    def _nop(*a, **k):  # noqa: D401
        """no-op display"""
        return None

    _ip_disp.display = _nop  # type: ignore[attr-defined]
except ImportError:
    pass

warnings.filterwarnings("ignore", module="matplotlib")

# ─── deps locais após blindagem de display ───────────────────────────────
from forecast_pipeline.jobs import run_single_job  # noqa: E402
import pandas as pd, numpy as np  # noqa: E402


def _json_safe(obj):
    """Fallback minimal para coisas que forem parar no log em JSON."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return f"<{obj.__class__.__name__} shape={obj.shape}>"
    if isinstance(obj, (np.ndarray, np.integer, np.floating)):
        return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
    return str(obj)


def main(job_json_path: str) -> None:  # noqa: D401
    # ------------------------------------------------------------------ #
    # 0. carrega job
    job = json.loads(Path(job_json_path).read_text())
    print(">>> SUBPROCESS: Boot OK – backend set to Agg", file=sys.stderr)

    # 1. prepara arquivo temporário onde o summary completo será guardado
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pkl", prefix="summary_")
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    # 2. captura TUDO que for para stdout/stderr durante run_single_job
    with contextlib.redirect_stdout(sys.stderr):  # prints → stderr
        print(f">>> SUBPROCESS: about to run_single_job id={job[-1]}", file=sys.stderr)
        summary = run_single_job(job, persist_result=True)
        print(">>> SUBPROCESS: run_single_job finished", file=sys.stderr)

    # 3. grava summary intacto (DataFrames, numpy, etc.)
    with tmp_path.open("wb") as fh:
        pickle.dump(summary, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f">>> SUBPROCESS: summary pickled @ {tmp_path}", file=sys.stderr)
    # 4. emite a “marcação” para o pai
    sys.stdout.write(f"__RESULT__ {tmp_path}\n")
    sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1])
