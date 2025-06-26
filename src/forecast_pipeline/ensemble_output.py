"""Camada de dados para outputs do ensemble.

Este módulo define:
  * EnsembleOutput – dataclass canônica usada em todo o pipeline;
  * to_ensemble_output – adaptador para formatos legados (tuple/list).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class EnsembleOutput:
    """Estrutura-padrão que carrega todas as curvas agregadas pelo ensemble.

    Qualquer campo ausente deve vir como ``None``.
    Campos adicionais podem ser colocados em ``meta`` sem quebrar contratos.
    """

    pred_test: np.ndarray
    pred_val: np.ndarray

    # opcionais ---------------------------------------------------------
    q_phys: Optional[np.ndarray] = None
    res_test: Optional[np.ndarray] = None
    res_val: Optional[np.ndarray] = None
    sigma_test: Optional[np.ndarray] = None
    sigma_val: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None

    # armazenamento para futuras extensões
    meta: Dict[str, Any] = field(default_factory=dict)

    # utilidade rápida para verificar se σ existe
    def has_sigma(self) -> bool:
        return self.sigma_test is not None and self.sigma_val is not None

    # representação amigável para debug
    def __repr__(self) -> str:
        fields = [
            f"pred_test shape={tuple(self.pred_test.shape)}",
            f"pred_val shape={tuple(self.pred_val.shape)}",
        ]
        if self.q_phys is not None:
            fields.append("q_phys ✓")
        if self.res_test is not None:
            fields.append("res ✓")
        if self.has_sigma():
            fields.append("sigma ✓")
        if self.alpha is not None:
            fields.append("alpha ✓")
        if self.meta:
            fields.append(f"meta keys={list(self.meta)}")
        return f"EnsembleOutput({', '.join(fields)})"


# ---------------------------------------------------------------------
# Adaptador de compatibilidade ----------------------------------------
# ---------------------------------------------------------------------

def to_ensemble_output(raw: Any) -> EnsembleOutput:
    """Converte saída de ``process_chunks`` para :class:`EnsembleOutput`.

    Aceita:
      * :class:`EnsembleOutput` (retorna como está)
      * ``(pred_test, pred_val)`` em tuple/list (formato legado)
    """

    if isinstance(raw, EnsembleOutput):
        return raw

    if isinstance(raw, dict):
        return EnsembleOutput(
            pred_test = np.asarray(raw["pred_test"]),
            pred_val  = np.asarray(raw["pred_val"]),
            q_phys    = raw.get("q_phys"),
            res_test  = raw.get("res_test"),
            res_val   = raw.get("res_val"),
            sigma_test= raw.get("sigma_test"),
            sigma_val = raw.get("sigma_val"),
            alpha     = raw.get("alpha"),
            meta      ={k:v for k,v in raw.items()
                        if k not in {"pred_test","pred_val","q_phys",
                                     "res_test","res_val","sigma_test",
                                     "sigma_val","alpha"}},
        )


    raise TypeError("Unsupported output format for EnsembleOutput conversion")
