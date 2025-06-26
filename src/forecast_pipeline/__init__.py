_EXPERIMENTS = {
    "Seq2Context": "forecast_pipeline.experiments.seq2context:ExperimentSeq2Context",
    "Seq2Value":   "forecast_pipeline.experiments.seq2value:ExperimentSeq2Value",
}

def get_experiment_cls(arch_name: str):
    path = _EXPERIMENTS[arch_name]
    module_path, cls_name = path.split(":")
    return getattr(import_module(module_path), cls_name)
