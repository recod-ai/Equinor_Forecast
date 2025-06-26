# forecast_pipeline/experiments/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseExperiment(ABC):
    """Interface mínima para qualquer experimento."""

    def __init__(self, ds, well: str, params: Dict[str, Any], exp_id: int):
        self.ds = ds
        self.well = well
        self.params = params
        self.id = exp_id

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Executa o experimento e retorna um dicionário de resultados."""
        ...
