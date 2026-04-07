# File: src/common/context_integration.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from box import Box

from .experiment_context import ExperimentContext


def arch_to_token(architecture_name: str | None) -> str:
    name = (architecture_name or "").strip().lower()
    if name.startswith("darts"):
        return "darts"
    return "seq2"


def _is_abs(p: str | None) -> bool:
    try:
        return bool(p) and Path(p).is_absolute()
    except Exception:
        return False


def apply_context_to_config(cfg: Box) -> Box:
    """
    Idempotently injects ExperimentContext paths into cfg.infra based on:
      - cfg.campaign_group
      - cfg.job_defaults.architecture_name
    Absolute paths already present are preserved.
    """
    group = str(getattr(cfg, "campaign_group", "DEMO_1"))
    arch_token = arch_to_token(getattr(getattr(cfg, "job_defaults", {}), "architecture_name", None))

    ctx = ExperimentContext(group=group, arch=arch_token, ensure_dirs=True)

    if not hasattr(cfg, "infra") or cfg.infra is None:
        cfg.infra = Box()

    # Only fill when missing or relative
    if not _is_abs(cfg.infra.get("profiles_dir")):
        cfg.infra["profiles_dir"] = str(ctx.profiles_dir)
    if not _is_abs(cfg.infra.get("experiments_output_dir")):
        cfg.infra["experiments_output_dir"] = str(ctx.results_dir)
    if not _is_abs(cfg.infra.get("hpo_studies_dir")):
        cfg.infra["hpo_studies_dir"] = str(ctx.studies_dir)
    if not _is_abs(cfg.infra.get("architecture_yaml_path")):
        cfg.infra["architecture_yaml_path"] = str(ctx.architecture_yaml_path)

    # Auxiliary (not required by runner, but handy for UIs/notebooks)
    cfg.infra["campaigns_dir"] = str(ctx.campaigns_dir)
    cfg.infra["reports_dir"] = str(ctx.reports_dir)  # group-level

    return cfg
