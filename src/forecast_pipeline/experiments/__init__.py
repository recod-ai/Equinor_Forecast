from importlib import import_module
from typing import Type
from .base import BaseExperiment

_EXPERIMENTS: dict[str, str] = {
    "Seq2Context": "forecast_pipeline.experiments.seq2context:ExperimentSeq2Context",
    "Seq2Value":   "forecast_pipeline.experiments.seq2value:ExperimentSeq2Value",
}

def get_experiment_cls(name: str) -> Type[BaseExperiment]:
    if name not in _EXPERIMENTS:
        raise KeyError(f"Experiment '{name}' não registrado.")
    module_path, cls_name = _EXPERIMENTS[name].split(":")
    return getattr(import_module(module_path), cls_name)

__all__ = ["get_experiment_cls", "BaseExperiment"]
