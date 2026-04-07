# File: src/common/experiment_context.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# Try to reuse your existing project root; fall back safely if not importable.
try:
    from forecast_pipeline.config import PROJECT_ROOT as _PRJ
except Exception:
    _PRJ = Path(__file__).resolve().parents[2]  # repo root fallback

@dataclass(frozen=True)
class ExperimentContext:
    """
    Campaign-aware filesystem context.

    Directory layout produced (defaults shown with group="DEMO_1", arch="seq2"):

      src/experiment_configs/
        ├── DEMO_1/
        │   ├── profiles/
        │   │   └── seq2/
        │   ├── results/
        │   │   └── seq2/
        │   ├── studies/
        │   │   └── seq2/
        │   └── reports/
        └── hpo_campaigns/
            └── DEMO_1/
                └── seq2/

    Notes
    -----
    - `reports_dir` is *group-level* to keep analysis CSVs centralized.
      If you prefer arch-specific reports, use `reports_dir_arch`.
    - All dirs are created on init when `ensure_dirs=True`.
    """
    # Required campaign knobs
    group: str = "DEMO_1"
    arch: Optional[str] = "seq2"  # "seq2" | "darts" | None/"" for no arch subfolder

    # Optional knobs
    project_root: Path = _PRJ
    ensure_dirs: bool = True

    # Derived core roots (filled in __post_init__)
    exp_cfg_root: Path = field(init=False)
    group_root: Path = field(init=False)

    # Public paths (filled in __post_init__)
    profiles_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    studies_dir: Path = field(init=False)
    campaigns_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)       # group-level reports
    reports_dir_arch: Path = field(init=False)  # optional arch-level reports

    # Common single-file resource
    architecture_yaml_path: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exp_cfg_root", self.project_root / "src" / "experiment_configs")
        object.__setattr__(self, "group_root", self.exp_cfg_root / self.group)

        arch_parts = [self.arch] if (self.arch or "").strip() else []

        # Per-arch trees under the group root
        profiles = self.group_root / "profiles" / Path(*arch_parts)
        results  = self.group_root / "results"  / Path(*arch_parts)
        studies  = self.group_root / "studies"  / Path(*arch_parts)

        object.__setattr__(self, "profiles_dir", profiles)
        object.__setattr__(self, "results_dir",  results)
        object.__setattr__(self, "studies_dir",  studies)

        # HPO campaign YAMLs live under global hpo_campaigns/<group>/<arch?>
        campaigns = self.exp_cfg_root / "hpo_campaigns" / self.group / Path(*arch_parts)
        object.__setattr__(self, "campaigns_dir", campaigns)

        # Reports: group-level (default) + optional arch-level view
        reports = self.group_root / "reports"
        object.__setattr__(self, "reports_dir", reports)
        object.__setattr__(self, "reports_dir_arch", reports / Path(*arch_parts) if arch_parts else reports)

        # Architecture catalog (shared)
        object.__setattr__(self, "architecture_yaml_path", self.exp_cfg_root / "architectures_PINNs.yaml")

        if self.ensure_dirs:
            self.ensure()

    # -----------------------
    # Convenience operations
    # -----------------------
    def ensure(self) -> None:
        """Create all relevant directories if they don't exist."""
        for p in (
            self.profiles_dir,
            self.results_dir,
            self.studies_dir,
            self.campaigns_dir,
            self.reports_dir,
            self.reports_dir_arch,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def to_infra_dict(self, *, log_level: str = "INFO") -> Dict[str, str]:
        """
        Helper for composing HPO/validation YAML `infra` blocks.
        """
        return {
            "profiles_dir": str(self.profiles_dir),
            "experiments_output_dir": str(self.results_dir),
            "hpo_studies_dir": str(self.studies_dir),
            "architecture_yaml_path": str(self.architecture_yaml_path),
            "log_level": log_level,
        }

    def with_arch(self, arch: Optional[str]) -> "ExperimentContext":
        """Clone this context with a different architecture subtree."""
        return ExperimentContext(
            group=self.group,
            arch=arch,
            project_root=self.project_root,
            ensure_dirs=self.ensure_dirs,
        )

    # ---------
    # Builders
    # ---------
    def profile_csv(self, name: str) -> Path:
        """Path for a profile CSV under the campaign profiles directory."""
        return self.profiles_dir / f"{name}.csv"

    def campaign_yaml(self, name: str) -> Path:
        """Path for a campaign YAML under hpo_campaigns/<group>/<arch>."""
        return self.campaigns_dir / f"{name}.yaml"

    def run_dir(self, name: str) -> Path:
        """Canonical run directory under results/<arch>/<name>."""
        return self.results_dir / name

    # ----------------
    # Introspection
    # ----------------
    def pathmap(self) -> Dict[str, str]:
        """Return a dict of the important paths (as strings)."""
        return {
            "project_root": str(self.project_root),
            "exp_cfg_root": str(self.exp_cfg_root),
            "group_root": str(self.group_root),
            "profiles_dir": str(self.profiles_dir),
            "results_dir": str(self.results_dir),
            "studies_dir": str(self.studies_dir),
            "campaigns_dir": str(self.campaigns_dir),
            "reports_dir": str(self.reports_dir),
            "reports_dir_arch": str(self.reports_dir_arch),
            "architecture_yaml_path": str(self.architecture_yaml_path),
        }
