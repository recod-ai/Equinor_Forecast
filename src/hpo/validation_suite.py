#src/hpo/validation_suite.py
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd



# -------------------------------
# Datamodel for a single experiment
# -------------------------------
@dataclass
class ValidationExperiment:
    """
    One validation experiment configuration.
    - policy: one of:
        'reconstruct', 'reconstruct_warm',
        'hp_hist', 'hp_raw', 'hp_hist_warm', 'hp_raw_warm'
    - reconstruct_warm_filter: only used when policy == 'reconstruct_warm'
        choices: 'hp' | 'none' | 'ewma' | 'holt'
    - name_suffix: optional suffix in run_name/csv_name beyond policy/filter
    - overrides: optional shallow overrides to job_defaults (e.g., {'lag_window': 120})
    """
    policy: str
    reconstruct_warm_filter: Optional[str] = None
    name_suffix: Optional[str] = None
    overrides: Dict[str, object] = field(default_factory=dict)

    def csv_stem(self) -> str:
        # e.g. "validation_reconstruct", "validation_reconstruct_warm_hp", "validation_hp_raw_warm"
        parts = ["validation", self.policy]
        if self.policy == "reconstruct_warm" and self.reconstruct_warm_filter:
            parts.append(self.reconstruct_warm_filter)
        if self.name_suffix:
            parts.append(self.name_suffix)
        return "_".join(parts).lower().replace("__", "_")

    def run_name(self, base: str) -> str:
        # e.g. base="validation_seq2" -> "validation_seq2__reconstruct_warm_hp"
        parts = [base, self.policy]
        if self.policy == "reconstruct_warm" and self.reconstruct_warm_filter:
            parts.append(self.reconstruct_warm_filter)
        if self.name_suffix:
            parts.append(self.name_suffix)
        return "__".join([p for p in parts if p]).lower().replace(" ", "-")

    def to_overrides(self, template_overrides: Dict) -> Dict:
        """
        Create a deep-copied CONFIG_OVERRIDES adjusted for this experiment.
        - Sets job_defaults.aggregation_method to `policy`
        - Injects job_defaults.reconstruct_warm_filter if relevant
        - Applies any per-experiment 'overrides' passed at construction
        """
        ov = copy.deepcopy(template_overrides or {})
        jd = ov.setdefault("job_defaults", {})

        # required aggregation method
        jd["aggregation_method"] = self.policy

        # optional filter for reconstruct_warm
        if self.policy == "reconstruct_warm" and self.reconstruct_warm_filter:
            jd["reconstruct_warm_filter"] = self.reconstruct_warm_filter

        # apply any user-provided overrides (e.g., lag_window=120)
        for k, v in (self.overrides or {}).items():
            jd[k] = v

        return ov


# -------------------------------
# Minimal CSV saver
# -------------------------------
def save_minimal_csv(
    leaderboard_df: pd.DataFrame,
    out_dir: Path,
    csv_stem: str,
    extra_cols: Optional[Sequence[str]] = None,
) -> Path:
    """
    Save a minimal CSV with the required columns.
    Ensures English column names via format_final_leaderboard (architecture rename).
    """
    from hpo.pivot_validation import format_final_leaderboard
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = format_final_leaderboard(leaderboard_df)
    minimal = ["well", "architecture", "val_smape_agg", "val_smape_cum", "test_smape_agg", "test_smape_cum", "physics_strategy"]
    if extra_cols:
        for c in extra_cols:
            if c not in minimal:
                minimal.append(c)
    existing = [c for c in minimal if c in fmt.columns]

    out_path = out_dir / f"{csv_stem}.csv"
    fmt[existing].to_csv(out_path, index=False)

    return out_path


# -------------------------------
# Suite runner
# -------------------------------
def run_validation_suite(
    *,
    project_root: Path,
    ctx,  # ExperimentContext (opaque here, just needs .reports_dir_arch)
    base_run_name: str,
    master_profile_filename: str,
    filters: Dict,
    template_overrides: Dict,
    experiments: Sequence[ValidationExperiment],
    execution_mode: str = "interactive",
    ensemble_size: int = 1,
) -> pd.DataFrame:
    """
    Run one or multiple validation experiments, each with its own aggregation method
    and optional reconstruct_warm_filter, saving a minimal CSV per experiment.

    Returns a small summary DataFrame with run_name, csv_path, rows, policy, filter.
    """
    from hpo.pivot_validation import run_pivot_validation
    rows: List[Dict] = []

    for exp in experiments:
        logging.info("Running validation for policy='%s' filter='%s'", exp.policy, exp.reconstruct_warm_filter or "-")

        # Build run-specific overrides and names
        ov = exp.to_overrides(template_overrides)
        run_name = exp.run_name(base_run_name)
        csv_stem = exp.csv_stem()

        # Execute the existing validation pipeline
        val_df = run_pivot_validation(
            project_root=project_root,
            master_profile_filename=master_profile_filename,
            run_name=run_name,
            filters=filters,
            execution_mode=execution_mode,
            ensemble_size=ensemble_size,
            config_overrides=ov,
            delete_previous_results=True,
            reports_dir=ctx.reports_dir_arch,
            ctx=ctx,
        )

        if val_df is None or val_df.empty:
            logging.warning("Validation produced no results for run_name=%s", run_name)
            rows.append(
                dict(
                    run_name=run_name,
                    csv_path=None,
                    n_rows=0,
                    policy=exp.policy,
                    filter=exp.reconstruct_warm_filter or "",
                )
            )
            continue

        # Persist the minimal CSV
        out_path = save_minimal_csv(
            leaderboard_df=val_df,
            out_dir=ctx.reports_dir_arch,
            csv_stem=csv_stem,
            # optionally add helpful context columns if they exist
            # extra_cols=["val_smape_cum", "epochs", "data_sample", "learning_rate", "lag_window", "batch_size"],
        )

        rows.append(
            dict(
                run_name=run_name,
                csv_path=str(out_path),
                n_rows=len(val_df),
                policy=exp.policy,
                filter=exp.reconstruct_warm_filter or "",
            )
        )

    summary = pd.DataFrame(rows)
    return summary
