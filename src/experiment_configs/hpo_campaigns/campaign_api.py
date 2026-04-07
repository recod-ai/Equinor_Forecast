# File: src/experiment_configs/hpo_campaigns/campaign_api.py
# Plug-and-play replacement (drop-in)

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import copy
import yaml
from common.config_wells import get_data_sources

# ---------------------------
# Helpers
# ---------------------------

def _family_slug(family: str) -> str:
    """Human → folder slug."""
    s = family.strip().lower()
    if s == "seq2":
        return "seq2"
    if s == "darts":
        return "darts"
    if s == "arps":
        return "arps"
    return s

def _build_infra_paths(
    base_root: Path,
    *,
    camp_name: Optional[str],
    family: str,
) -> Dict[str, str]:
    """
    Builds infra paths with the layout:
      <base_root>/<CAMP_NAME>/{profiles,results,studies}/<family_slug>
    If camp_name is None, omits the <CAMP_NAME> level.
    """
    fam = _family_slug(family)
    if camp_name:
        group_root = base_root / camp_name
        profiles_dir   = group_root / "profiles" / fam
        results_dir    = group_root / "results" / fam
        studies_dir    = group_root / "studies" / fam
    else:
        profiles_dir   = base_root / "profiles" / fam
        results_dir    = base_root / "results" / fam
        studies_dir    = base_root / "studies" / fam

    # Ensure folders exist
    for p in (profiles_dir, results_dir, studies_dir):
        p.mkdir(parents=True, exist_ok=True)

    return {
        "profiles_dir": str(profiles_dir.resolve()),
        "experiments_output_dir": str(results_dir.resolve()),
        "hpo_studies_dir": str(studies_dir.resolve()),
    }

def _campaign_yaml_dir(
    hpo_root: Path,
    *,
    camp_name: Optional[str],
    family: str,
) -> Path:
    """
    Where to write YAMLs:
      <hpo_root>/<CAMP_NAME>/<family_slug>/*.yaml
    or
      <hpo_root>/<family_slug>/*.yaml   (if no camp_name)
    """
    fam = _family_slug(family)
    if camp_name:
        out = hpo_root / camp_name / fam
    else:
        out = hpo_root / fam
    out.mkdir(parents=True, exist_ok=True)
    return out

def _normalize_wells_list(wells):
    """Handle 'w1, w2, w3' style entries from config_wells."""
    if isinstance(wells, list) and len(wells) == 1 and isinstance(wells[0], str) and ("," in wells[0]):
        return [w.strip() for w in wells[0].split(",")]
    return wells


def build_wells_by_dataset(datasets: List[str], dataset_well_filters: Dict[str, List[str]] | None = None) -> Dict[str, List[str]]:
    all_sources = get_data_sources()
    out: Dict[str, List[str]] = {}
    for ds in datasets:
        src = next((x for x in all_sources if x["name"] == ds), None)
        if not src:
            continue
        wells = _normalize_wells_list(src.get("wells", [])) or []
        if dataset_well_filters and dataset_well_filters.get(ds):
            allowed = set(dataset_well_filters[ds])
            wells = [w for w in wells if w in allowed]
        out[ds] = wells
    return out

# ---------------------------
# Public API
# ---------------------------

def generate_campaign_files(
    *,
    base_template: Dict,
    datasets: List[str],
    wells_by_dataset: Dict[str, List[str]],
    family: str,                             # "Seq2", "Darts" or "Arps"
    architectures: List[str],                # e.g., ["Seq2PIN"], ["Darts"], ["Arps_Canonical"]
    output_dir: Path,                        # usually: src/experiment_configs/hpo_campaigns
    trials_per_arch: Dict[str, List[int]],
    darts_overrides: Optional[Dict] = None,  # only for family="Darts"
    arps_overrides: Optional[Dict] = None,   # only for family="Arps"
    camp_name: Optional[str] = None,         # e.g., "DEMO_1"
) -> List[Path]:
    """
    Create one YAML per (dataset, well, architecture) with organized infra paths.

    Layout used inside the YAML:
      infra.profiles_dir           → <src/experiment_configs>/<CAMP>/profiles/<family>
      infra.experiments_output_dir → <src/experiment_configs>/<CAMP>/results/<family>
      infra.hpo_studies_dir        → <src/experiment_configs>/<CAMP>/studies/<family>

    YAML location on disk:
      <src/experiment_configs>/hpo_campaigns/<CAMP>/<family>/*.yaml
    """
    created: List[Path] = []

    # Roots
    base_root = Path(__file__).resolve().parents[2] / "experiment_configs"
    base_root.mkdir(parents=True, exist_ok=True)

    # Where to write YAMLs
    yaml_dir = _campaign_yaml_dir(output_dir, camp_name=camp_name, family=family)

    # Shared infra (per camp/family)
    infra_paths = _build_infra_paths(base_root, camp_name=camp_name, family=family)

    for ds in datasets:
        wells = _normalize_wells_list(wells_by_dataset.get(ds, [])) or []
        for well in wells:
            for arch in architectures:
                cfg = copy.deepcopy(base_template)

                # campaign name
                base_name = f"{ds}_{str(well).replace('/', '-')}_{arch}"

                # job_defaults / family wiring
                cfg["campaign_name"] = base_name
                cfg["run_scope"]["dataset_name"] = ds
                cfg["run_scope"]["wells"] = [well]
                cfg["job_defaults"]["architecture_name"] = arch

                # architecture-aware search schedule
                schedule = trials_per_arch.get(arch, trials_per_arch.get("default", [10, 5]))
                cfg["hpo_params"]["trials_per_cycle_schedule"] = schedule

                # -----------------------
                # Family-specific options
                # -----------------------
                fam = family.strip().lower()

                if fam == "seq2":
                    # For Seq2, only Seq2Context needs the YAML path
                    if cfg["job_defaults"]["architecture_name"] != "Seq2Context":
                        cfg["infra"].pop("architecture_yaml_path", None)

                elif fam == "darts":
                    # For Darts, we never need architecture_yaml_path
                    cfg["infra"].pop("architecture_yaml_path", None)
                    if darts_overrides:
                        cfg["job_defaults"].update(darts_overrides or {})

                elif fam == "arps":
                    # ARPS does not use an external architecture YAML
                    cfg["infra"].pop("architecture_yaml_path", None)
                    # Optional per-family overrides (e.g., default plot=False)
                    if arps_overrides:
                        cfg["job_defaults"].update(arps_overrides or {})
                else:
                    # Unknown family: keep as-is (future-proof)
                    cfg["infra"].pop("architecture_yaml_path", None)

                # --- Root normalization (always) ---
                jd = cfg["job_defaults"]
                # input_chunk_length follows lag_window (if present)
                lag = jd.get("lag_window", jd.get("input_chunk_length", 100))
                # output_chunk_length follows horizon (if present)
                hor = jd.get("horizon", jd.get("output_chunk_length", 1))
                try:
                    lag = int(lag)
                except Exception:
                    lag = 100
                try:
                    hor = int(hor)
                except Exception:
                    hor = 1
                jd["input_chunk_length"] = lag
                jd["output_chunk_length"] = hor
                cfg["job_defaults"] = jd
                # --- end normalization ---

                # inject organized infra
                cfg["infra"].update(infra_paths)

                # optional campaign_group
                if camp_name:
                    cfg["campaign_group"] = str(camp_name)

                # write YAML
                out_path = yaml_dir / f"{base_name}.yaml"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    yaml.dump(cfg, f, sort_keys=False, default_flow_style=None, indent=2)
                created.append(out_path)

    return created
