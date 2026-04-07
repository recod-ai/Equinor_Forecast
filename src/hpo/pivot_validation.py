# src/hpo/pivot_validation.py

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import yaml

# Optional nice console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Project imports (runtime-level, no circulars)
from config_loader import load_campaign_config
from hpo.hpo_runner import run_campaign_from_config_object
from hpo.posthoc_filtering import load_master_leaderboard

from hpo.analysis import run_validation_comparison
from hpo.validation_suite import ValidationExperiment, run_validation_suite, save_minimal_csv

# =============================================================================
# Helpers — small, focused, testable
# =============================================================================

# --- plug-and-play: validation generalization helpers -------------------------
from typing import List, Iterable, Tuple
import numpy as np
import pandas as pd

def detect_family(df: pd.DataFrame) -> str:
    """Detecta 'seq2', 'darts', 'arps' ou 'generic' a partir de colunas/conteúdo."""
    if df is None or df.empty:
        return "generic"
    def _has(c: str) -> bool: return c in df.columns
    a  = df.get("architecture", pd.Series([], dtype=str)).astype(str)
    an = df.get("architecture_name", pd.Series([], dtype=str)).astype(str)
    if _has("variant") or a.str.contains("Arps", na=False).any() or an.str.contains("Arps", na=False).any():
        return "arps"
    if _has("profile") or _has("n_epochs") or an.str.startswith("Darts_", na=False).any():
        return "darts"
    if _has("physics_strategy") or _has("aggregation_method") or a.str.contains("Seq2", na=False).any() or an.str.contains("Seq2", na=False).any():
        return "seq2"
    return "generic"

def arch_col(df: pd.DataFrame, candidates: Iterable[str] = ("architecture_name","architecture")) -> str:
    """Escolhe a melhor coluna de arquitetura para contexto/merge."""
    for c in candidates:
        if c in df.columns:
            return c
    return candidates[0]

def plan_columns_for(df: pd.DataFrame) -> List[str]:
    """
    Quais colunas exibir no RESUMO DO PLANO, conforme a família detectada.
    Apenas retorna a preferência — a função de impressão filtrará pelo que existir.
    """
    fam = detect_family(df)
    if fam == "seq2":
        return ["dataset","well","architecture_name","physics_strategy", "aggregation_method", 
                "epochs","data_sample","learning_rate","lag_window","batch_size"]
    if fam == "darts":
        return ["dataset","well","architecture_name","physics_strategy","profile",
                "n_epochs","batch_size","learning_rate","input_chunk_length","output_chunk_length"]
    if fam == "arps":
        return ["dataset","well","architecture_name","variant","solver","loss",
                "burn_in_fraction","piecewise"]
    # genérico
    return ["dataset","well","architecture_name","physics_strategy","epochs","batch_size","learning_rate"]

def fingerprint_priority_for(family: str) -> List[str]:
    """
    Ordem de prioridade das chaves de HPO (para o MERGE HPO vs Validation).
    Usaremos só as que existirem em ambos os DFs (até 5).
    """
    if family == "seq2":
        return ["physics_strategy","aggregation_method","data_sample",
                "learning_rate","lag_window","batch_size","epochs"]
    if family == "darts":
        return ["profile","n_epochs","batch_size","learning_rate",
                "input_chunk_length","output_chunk_length"]
    if family == "arps":
        return ["variant","solver","loss","burn_in_fraction","piecewise",
                "weighting","quantile_tau"]
    return ["physics_strategy","learning_rate","batch_size","epochs"]

def pick_fingerprint(hpo_df: pd.DataFrame, val_df: pd.DataFrame, family: str, k: int = 5) -> List[str]:
    """Seleciona até k chaves que existam em AMBOS os DFs, respeitando a prioridade da família."""
    pref = fingerprint_priority_for(family)
    both = [c for c in pref if (c in hpo_df.columns and c in val_df.columns)]
    # evita usar 'architecture_name' aqui (vai como contexto à parte)
    return both[:max(1, min(k, len(both)))]

# --- normalização leve para chaves numéricas (estável e minimalista) ----------
_FLOAT_KEYS = {"learning_rate","data_sample","burn_in_fraction","quantile_tau"}
_INT_KEYS   = {"epochs","batch_size","lag_window","n_epochs","input_chunk_length","output_chunk_length"}

def _cast_float_series(s: pd.Series, decimals: int = 6) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.round(decimals)

def _cast_int_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    # se todos muito próximos de inteiros, arredonda e usa Int64
    if x.dropna().empty:
        return x
    tol = 1e-9
    if ((x.dropna() - x.dropna().round()).abs() <= tol).all():
        try:
            return x.round().astype("Int64")
        except Exception:
            return x.round()
    return x

import re
import pandas as pd

def _canon_well(x: object) -> object:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    s = str(x).strip()

    # Caso específico muito comum no VOLVE: "15/9-F-14" -> "15/9/F-14"
    # Regra: troca o primeiro '-' após algo como "15/9" por '/'
    # (mantém "15-9-F-14" intacto, porque não tem "/")
    s = re.sub(r"^(\d+/\d+)-", r"\1/", s)

    return s

_FLOAT_KEYS = {"learning_rate","data_sample","burn_in_fraction","quantile_tau"}
_INT_KEYS   = {"epochs","batch_size","lag_window","n_epochs","input_chunk_length","output_chunk_length"}

def normalize_merge_keys(df: pd.DataFrame, keys) -> pd.DataFrame:
    out = df.copy()

    for k in keys:
        if k not in out.columns:
            continue

        if k == "well":
            out[k] = out[k].map(_canon_well)
            continue

        if k in _FLOAT_KEYS:
            out[k] = pd.to_numeric(out[k], errors="coerce").round(6)
            continue

        if k in _INT_KEYS:
            x = pd.to_numeric(out[k], errors="coerce")
            # arredonda e usa Int64 quando fizer sentido
            out[k] = x.round().astype("Int64")
            continue

        # strings gerais (ex.: physics_strategy)
        if out[k].dtype == "object":
            out[k] = out[k].astype(str).str.strip()

    return out




def resolve_experiments_for_arch(arch: str, default_suite: List[ValidationExperiment]) -> List[ValidationExperiment]:
    """
    Enforce arch-specific policy:
      - For 'darts': allow only ValidationExperiment(policy="reconstruct").
      - Otherwise: use the provided default suite.
    """
    if arch.lower() == "darts":
        return [ValidationExperiment(policy="reconstruct")]
    return default_suite

def find_existing_validation_csvs(reports_dir: Path, stem_prefix: str = "validation_") -> List[Path]:
    """
    Return any previously saved validation CSVs so we can guard against accidental overwrite.
    """
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob(f"{stem_prefix}*.csv"))

def confirm_or_abort_overwrite(existing_csvs: List[Path], *, force: bool = False) -> bool:
    """
    If CSVs exist and force=False, ask the user before proceeding.
    Returns True to proceed (and delete), False to abort.
    """
    if not existing_csvs:
        return True
    if force:
        return True

    print("⚠️ Found previously saved validation CSVs:")
    for p in existing_csvs:
        print("   •", p.name)
    ans = input("Proceed and delete these files? [y/N] ").strip().lower()
    return ans == "y"

def delete_files(paths: List[Path]) -> None:
    for p in paths:
        try:
            p.unlink()
        except Exception as e:
            print(f"[warn] Could not delete {p.name}: {e}")


# ============================================================================
# Small utilities
# ============================================================================

def _deep_update(base: dict, override: dict) -> dict:
    """Recursive dict update (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


# ============================================================================
# 1) Load & Filter champions
# ============================================================================

def load_and_filter_champions(
    reports_dir: Path,
    profile_filename: str,
    filters: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Loads the champions profile CSV in `reports_dir` and applies simple equality / IN filters.
    filters may include keys like 'dataset', 'well', 'architecture_name', etc.
    """
    profile_path = reports_dir / profile_filename
    if not profile_path.exists():
        logging.error("Master validation profile not found at: %s", profile_path)
        return None

    df = pd.read_csv(profile_path)

    for key, value in (filters or {}).items():
        if value is None or key not in df.columns:
            continue
        df = df[df[key].isin(value)] if isinstance(value, (list, tuple, set)) else df[df[key] == value]

    if df.empty:
        logging.warning("No candidates found after filters: %s", filters)
        return None

    logging.info("Filtered down to %d candidates for this validation run.", len(df))
    return df


# ============================================================================
# 2) Config builder for validation runs
# ============================================================================

def build_validation_config(
    run_name: str,
    ensemble_size: int,
    experiments_output_dir: str,
    config_overrides: Optional[dict] = None,
    *,
    infra_from_ctx: Optional[dict] = None,
) -> dict:
    """
    Minimal validation-run campaign config.

    Main knobs (keep minimal, stable defaults):
      - latent_mode: "off" | "offline_analytic" | "full_sequence"
          High-level switch for post-processing prediction ribbons.

      - latent_cfg (optional dict):
          Extra knobs for the selected latent_mode. For offline_analytic:
            * mode: should match latent_mode (kept for backward-compat).
            * arps_fit_window: int (fixed default=80)
            * analytic_fit_region: "head" | "tail"   (fixed default="head")
            * analytic_anchor_kind: "median" | "mean" | "last" (fixed default="median")
            * analytic_anchor_window: int (fixed default=10)
            * analytic_use_only_first_window: bool (default=True; legacy behavior)
            * analytic_window_index: int (default=0; set to -1 to test "most recent window")
            * arps_coupling_mode: "none" | "val_only" | "val_plus_test" (default="val_only")

      - eval_mode: "seq" | "fullseq_k"
          Controls which evaluation function is used:
            * "seq"       -> evaluate_model_seq(...)
            * "fullseq_k" -> evaluate_fullseq_k_mode(...)

      - fullseq_mode: kept for compatibility; when LEFT is not used, keep "deploy_split_k".
      - fullseq_k: number of last windows used by fullseq_k evaluation (default=1).
    """
    base_infra = infra_from_ctx or {
        "profiles_dir": "src/experiment_configs/profiles",
        "experiments_output_dir": experiments_output_dir,
        "hpo_studies_dir": "src/experiment_configs/studies",
        "architecture_yaml_path": "src/experiment_configs/architectures_PINNs.yaml",
        "log_level": "INFO",
    }

    base = {
        "campaign_name": run_name,
        "run_scope": {"dataset_name": "FILTERED", "wells": ["FILTERED"]},
        "job_defaults": {
            "seed": 42,
            "plot": True,
            "feature_kind": "Normal",
            "use_known_good": False,
            "lag_window": 100,
            "horizon": 300,
            "patience": 50,
            "test_size": 0.7,
            "val_size": 0.1,

            # --- Aggregation / classic evaluation ---
            "aggregation_method": "reconstruct",
            "evaluate_by_slice": True,
            "slice_ratios": [1.0],
            "aggregation_quantiles": [0.25, 0.5, 0.75],

            # --- Latent/analytic post-processing (high-level knob) ---
            "latent_mode": "off",  # "off" | "offline_analytic" | "full_sequence"

            # --- Minimal evaluation routing knob ---
            "eval_mode": "seq",    # "seq" | "fullseq_k"

            # --- Full-seq eval knobs (keep for compat, even if LEFT is not used) ---
            "fullseq_mode": "deploy_split_k",  # keep fixed when not using LEFT
            "fullseq_k": 1,

            # --- Optional: latent_cfg is the real config carrier ---
            "latent_cfg": {
                "mode": "off",
                # offline_analytic defaults (invariants can stay here or in executor defaults)
                "arps_fit_window": 80,
                "analytic_use_only_first_window": True,
                # experimental knob: 0 (first window) vs -1 (last window)
                "analytic_window_index": 0,
                "arps_coupling_mode": "val_only",
                "arps_anchor_window": 15,
            },
        },
        "run_params": {"ensemble_size": ensemble_size, "max_workers": 1},
        "infra": base_infra,
    }
    return _deep_update(base, config_overrides or {})




# ============================================================================
# 3) Artifact preparation (CSV + YAML)
# ============================================================================

from typing import Optional
# …
def prepare_run_artifacts(
    project_root: Path,
    run_name: str,
    candidate_df: pd.DataFrame,
    ensemble_size: int,
    config_overrides: Optional[dict] = None,
    *,
    ctx: "ExperimentContext" | None = None,   # <— NEW
) -> Dict[str, Path]:
    """
    Writes the filtered profile CSV and a YAML config for the run.
    Uses ExperimentContext when provided to place everything under the proper group/family.
    """
    if ctx is not None:
        profiles_dir = ctx.profiles_dir
        campaigns_dir = ctx.campaigns_dir
        experiments_output_dir = str(ctx.results_dir)
        infra = ctx.to_infra_dict()
    else:
        profiles_dir = (project_root / "src/experiment_configs/profiles").resolve()
        campaigns_dir = (project_root / "src/experiment_configs/hpo_campaigns").resolve()
        experiments_output_dir = str(project_root / "src/experiment_configs/results")
        infra = None

    profiles_dir.mkdir(parents=True, exist_ok=True)
    campaigns_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = sanitize_profile_for_execution(candidate_df)

    temp_profile_path = profiles_dir / f"{run_name}.csv"
    candidate_df.to_csv(temp_profile_path, index=False)

    cfg_dict = build_validation_config(
        run_name=run_name,
        ensemble_size=ensemble_size,
        experiments_output_dir=experiments_output_dir,
        config_overrides=config_overrides,
        infra_from_ctx=infra,
    )

    _force_run_params_overrides(cfg_dict, config_overrides)

    config_path = campaigns_dir / f"{run_name}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, indent=2)

    run_dir = (Path(cfg_dict["infra"]["experiments_output_dir"]) / run_name).resolve()

    return {
        "profile_path": temp_profile_path,
        "config_path": config_path,
        "run_dir": run_dir,
    }



# ============================================================================
# 4) Optional cleanup
# ============================================================================

def clear_previous_run_artifacts(run_dir: Path) -> bool:
    """
    Deletes the run results directory (e.g., results/<run_name>) if it exists.
    Returns True if something was removed.
    """
    if run_dir.exists():
        shutil.rmtree(run_dir)
        logging.info("Deleted previous results folder: %s", run_dir)
        return True
    logging.info("No previous results folder to delete: %s", run_dir)
    return False


# ============================================================================
# 5) Pretty plan (Rich fallback)
# ============================================================================

def display_run_plan(run_name: str, candidate_df: pd.DataFrame, filters: Dict[str, Any]) -> None:
    cols_pref = plan_columns_for(candidate_df)
    cols_exist = [c for c in cols_pref if c in candidate_df.columns]

    if not RICH_AVAILABLE:
        print(f"--- Validation Plan: {run_name} ---")
        print(f"Filters applied: {filters}")
        print(f"Models to validate: {len(candidate_df)}")
        print(candidate_df[cols_exist].head())
        return

    console = Console()
    filter_str = "\n".join([f"[magenta]{k}:[/] [green]{v}[/]" for k, v in filters.items() if v is not None])
    console.print(Panel(
        f"[bold]Filters Applied:[/]\n{filter_str}\n\n[bold]Models to Validate:[/] {len(candidate_df)}",
        title=f"📋 Pivot Validation Plan: [yellow]{run_name}[/]",
        border_style="blue",
    ))

    # Cabeçalho amigável (só para colunas existentes)
    header_names = {
        "dataset": "Dataset", "well": "Well", "architecture_name": "Arch",
        "aggregation_method": "Filter",
        "physics_strategy": "Physics", "architecture_profile": "Profile",
        "epochs": "Epochs", "data_sample": "Data %", "learning_rate": "LR",
        "lag_window": "Lag", "batch_size": "Batch", "trend_degree": "Trend",
        "profile": "Profile", "n_epochs": "Epochs",
        "input_chunk_length": "In", "output_chunk_length": "Out",
        "variant": "Variant", "solver": "Solver", "loss": "Loss",
        "burn_in_fraction": "Burn-in", "piecewise": "Piecewise",
    }

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    for col in cols_exist:
        header = header_names.get(col, col)
        justify = "right" if pd.api.types.is_numeric_dtype(candidate_df[col]) else "left"
        table.add_column(header, justify=justify)

    for _, row in candidate_df[cols_exist].head(20).iterrows():
        cells = []
        for col in cols_exist:
            val = row.get(col)
            if isinstance(val, float):
                fmt = "{:.5f}" if col in ("learning_rate",) else "{:.2f}"
                cells.append(fmt.format(val))
            else:
                cells.append(str(val))
        table.add_row(*cells)

    console.print(table)



# ============================================================================
# 6) Final leaderboard formatter
# ============================================================================

def format_final_leaderboard(leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and orders key columns for a clean summary view.
    Now also exposes test_* metrics if they exist (non-breaking).
    """
    desired = [
        "well",
        "architecture_name",
        "weighted_score",
        "val_smape_agg",
        "val_smape_cum",
        # --- new (optional) test metrics for visibility in display ---
        "test_smape_agg",
        "test_smape_cum",
        # ------------------------------------------------------------
        "trend_degree",
        "physics_strategy",
        "epochs",
        "data_sample",
        "learning_rate",
        "lag_window",
        "batch_size",
    ]
    cols = [c for c in desired if c in leaderboard_df.columns]
    out = leaderboard_df[cols].rename(columns={"architecture_name": "architecture"})
    return out




from typing import Dict, Iterable, Optional
import pandas as pd

# --- ajuste 2: FORÇAR override de run_params antes de escrever o YAML ---
def _force_run_params_overrides(cfg_dict: dict, config_overrides: Optional[dict]) -> None:
    """
    Aplica apenas overrides do bloco run_params ao cfg_dict (in-place).
    Garante que ensemble_size/max_workers do YAML reflitam CONFIG_OVERRIDES.
    """
    if not config_overrides:
        return
    rp = (config_overrides or {}).get("run_params") or {}
    if not rp:
        return
    cfg_dict.setdefault("run_params", {})
    # só sobrescreve chaves presentes em overrides
    for k, v in rp.items():
        cfg_dict["run_params"][k] = v


def apply_profile_overrides(
    df: pd.DataFrame,
    config_overrides: Optional[Dict] = None,
    *,
    # quais chaves de job_defaults podem ser criadas se não existirem no df
    allow_create: Iterable[str] = (
        "aggregation_method",
        "plot",
        "lag_window",
        "horizon",
        "test_size",
        "val_size",
    ),
) -> pd.DataFrame:
    """
    Minimalista: pega CONFIG_OVERRIDES['job_defaults'] e sobrescreve colunas no df.
    - Se a coluna existir: substitui por valor único (broadcast).
    - Se não existir: cria só se estiver em allow_create.
    Não mexe na estrutura do df (sem groupby/concat). Retorna uma cópia.
    """
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        return df

    jd = (config_overrides or {}).get("job_defaults", {}) or {}
    if not jd:
        return df

    out = df.copy()

    for key, value in jd.items():
        if key in out.columns:
            out[key] = value
        elif key in allow_create:
            out[key] = value
        # senão: ignoramos silenciosamente (não criamos colunas inesperadas)

    return out



import re

_METRIC_COL_PATTERNS = [
    r"^weighted_score$",
    r"^(val|train|test)_(smape|mae|mse|rmse|r2).*",
    r"^(val|train|test)_loss.*",        # <-- scoped to metric columns only
    r"^smape.*", r"^mae.*", r"^mse.*", r"^rmse.*", r"^r2.*",
    r".*_metric$",
]

def sanitize_profile_for_execution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas de métricas e QUALQUER coluna que possa contaminar execução:
    - métricas/resultados (val_*, train_*, weighted_score, *_norm)
    - ensemble_size / ensemble_models / seeds / num_seeds
    """
    if df is None or df.empty:
        return df

    drop_cols = set()

    # 1) padrões de métricas/resultados
    for c in df.columns:
        for pat in _METRIC_COL_PATTERNS:
            if re.match(pat, str(c), flags=re.IGNORECASE):
                drop_cols.add(c)
                break

    # 2) colunas que SEMPRE devem sair (fonte de contaminação)
    drop_cols.update({"ensemble_size", "ensemble_models", "seeds", "num_seeds"})

    # Proteções mínimas para não derrubar chaves de join
    keep_keys = {"well", "architecture_name", "physics_strategy", "dataset"}
    drop_cols = [c for c in drop_cols if c not in keep_keys]

    return df.drop(columns=drop_cols, errors="ignore")


def run_single_validation_entry(
    *,
    project_root: Path,
    ctx: ExperimentContext,
    master_profile_filename: str,
    run_name: str,
    filters: Dict[str, object],
    execution_mode: str,
    ensemble_size: int,
    config_overrides: Dict[str, object],
    delete_previous_results: bool,
    compare_with_hpo: bool,
) -> Optional[pd.DataFrame]:
    """
    Single-run entry point:
      - Runs exactly the provided CONFIG_OVERRIDES (no experiment sweep).
      - Does NOT save any CSVs (explicitly required to avoid overwriting suite outputs).
    """
    print("▶ Running SINGLE validation (no CSV will be written)…")
    df = run_pivot_validation(
        project_root=project_root,
        master_profile_filename=master_profile_filename,
        run_name=run_name,
        filters=filters,
        execution_mode=execution_mode,
        ensemble_size=ensemble_size,
        config_overrides=config_overrides,
        delete_previous_results=delete_previous_results,
        reports_dir=ctx.reports_dir_arch,
        ctx=ctx,
    )
    if df is None or df.empty:
        print("❌ Validation produced no results.")
        return None

    print(f"✅ Validation finished with {len(df)} rows.")
    display(format_final_leaderboard(df).head(30))

    if compare_with_hpo:
        _compare_with_hpo(df, ctx=ctx)

    return df

def run_suite_validation_entry(
    *,
    project_root: Path,
    ctx: ExperimentContext,
    base_run_name: str,
    master_profile_filename: str,
    filters: Dict[str, object],
    template_overrides: Dict[str, object],
    experiments: List[ValidationExperiment],
    execution_mode: str,
    ensemble_size: int,
    force_overwrite: bool,
) -> Optional[pd.DataFrame]:
    """
    Suite-run entry point:
      - If previously saved CSVs exist under reports_dir_arch, ask for confirmation (or honor force_overwrite).
      - On confirmation, delete those CSVs and run the full suite grid.
      - Each experiment writes its minimal CSV via save_minimal_csv (inside run_validation_suite).
    """
    reports_dir = ctx.reports_dir_arch
    existing_csvs = find_existing_validation_csvs(reports_dir, stem_prefix="validation_")

    if existing_csvs:
        proceed = confirm_or_abort_overwrite(existing_csvs, force=force_overwrite)
        if not proceed:
            print("⏹ Aborted by user — keeping existing CSVs.")
            return None
        # Clean up old CSVs explicitly before running the suite
        delete_files(existing_csvs)
        print("🧹 Removed old validation CSVs.")

    # Run the suite
    print("▶ Running SUITE validation…")
    suite_summary = run_validation_suite(
        project_root=project_root,
        ctx=ctx,
        base_run_name=base_run_name,
        master_profile_filename=master_profile_filename,
        filters=filters,
        template_overrides=template_overrides,
        experiments=experiments,
        execution_mode=execution_mode,
        ensemble_size=ensemble_size,
    )

    if suite_summary is None or suite_summary.empty:
        print("❌ Suite produced no results.")
        return None

    print("✅ Suite finished. Summary:")
    display(suite_summary)
    return suite_summary

def _compare_with_hpo(val_df: pd.DataFrame, *, ctx: ExperimentContext) -> None:
    """
    Agnóstico: detecta família, escolhe fingerprint, normaliza chaves, faz merge simples
    (com um único fallback: retirar 'data_sample' se o merge vier vazio) e renderiza o relatório.
    """
    hpo_csv = ctx.reports_dir_arch / "hpo_master_leaderboard.csv"
    if not hpo_csv.exists():
        print("⚠️ HPO master leaderboard not found at:", hpo_csv)
        return

    hpo_master_df = pd.read_csv(hpo_csv)
    if val_df is None or val_df.empty or hpo_master_df is None or hpo_master_df.empty:
        print("⚠️ Empty validation or HPO dataframe; skipping comparison.")
        return

    family = detect_family(val_df)
    arch_c_val = arch_col(val_df)
    arch_c_hpo = arch_col(hpo_master_df)
    use_arch = arch_c_val if (arch_c_val in val_df.columns and arch_c_val in hpo_master_df.columns) else None

    # fingerprint: até 5 chaves que existam em ambos
    fp = pick_fingerprint(hpo_master_df, val_df, family, k=5)

    # merge keys = well + (arch?) + fingerprint
    merge_keys = ["well"] + ([use_arch] if use_arch else []) + fp
    if len(merge_keys) < 2:
        print("⚠️ Not enough common keys to compare with HPO; skipping.")
        return

    # normalização leve das chaves
    val_n = normalize_merge_keys(val_df, merge_keys)
    hpo_n = normalize_merge_keys(hpo_master_df, merge_keys)

    # 1º merge: se HPO já vier sufixado, ok; senão, reduziremos depois
    joined = pd.merge(
        val_n, hpo_n,
        on=merge_keys, how="left", suffixes=("_validation", "_hpo")
    )

    def _has_hpo_metrics(df: pd.DataFrame) -> bool:
        return {"val_smape_cum_hpo","val_smape_agg_hpo"}.issubset(df.columns)

    # Se ainda não há *_hpo, criar visão reduzida do HPO e mesclar de novo
    if not _has_hpo_metrics(joined):
        cols_needed = {"val_smape_cum","val_smape_agg"}
        if cols_needed.issubset(hpo_n.columns):
            hpo_reduced = (
                hpo_n
                .sort_values(["val_smape_cum","val_smape_agg"], ascending=[True, True])
                .drop_duplicates(subset=merge_keys, keep="first")
                [merge_keys + ["val_smape_cum","val_smape_agg"]]
                .rename(columns={"val_smape_cum":"val_smape_cum_hpo","val_smape_agg":"val_smape_agg_hpo"})
            )
            joined = pd.merge(val_n, hpo_reduced, on=merge_keys, how="left")

    # Fallback único: retirar data_sample se estiver nas chaves e o merge falhou (tudo NaN)
    if (("data_sample" in merge_keys) and
        (joined[["val_smape_cum_hpo","val_smape_agg_hpo"]].isna().all().all())):
        mk2 = [k for k in merge_keys if k != "data_sample"]
        if len(mk2) >= 2:
            hpo_reduced2 = (
                hpo_n
                .sort_values(["val_smape_cum","val_smape_agg"], ascending=[True, True])
                .drop_duplicates(subset=mk2, keep="first")
            )
            if {"val_smape_cum","val_smape_agg"}.issubset(hpo_reduced2.columns):
                hpo_reduced2 = (
                    hpo_reduced2[mk2 + ["val_smape_cum","val_smape_agg"]]
                    .rename(columns={"val_smape_cum":"val_smape_cum_hpo","val_smape_agg":"val_smape_agg_hpo"})
                )
                joined = pd.merge(val_n, hpo_reduced2, on=mk2, how="left")
                # atualiza hyperparameter_cols efetivas (sem data_sample)
                fp = [c for c in fp if c != "data_sample"]
                merge_keys = mk2

    if joined.empty or (joined[["val_smape_cum_hpo","val_smape_agg_hpo"]].isna().all().all()):
        print("⚠️ No matching HPO rows for validated models.")
        return

    # Renderização final (usa sua função existente)
    run_validation_comparison(
        hpo_leaderboard=joined,               # já contém colunas *_hpo
        validation_leaderboard=val_n,        # passado apenas para compatibilidade
        hyperparameter_cols=fp,              # fingerprint efetivo usado no merge
        cum_pp_tolerance=0.30,
        agg_pp_tolerance=1.00,
        show_relative_columns=False,
    )


# ============================================================================
# 7) Main orchestrator
# ============================================================================
def run_pivot_validation(
    project_root: Path,
    master_profile_filename: str,
    run_name: str,
    filters: Dict[str, Any],
    *,
    execution_mode: str = "interactive",
    ensemble_size: int = 1,
    config_overrides: Optional[dict] = None,
    delete_previous_results: bool = True,
    reports_dir: Optional[Path] = None,         # <— NEW
    ctx: "ExperimentContext" | None = None,     # <— NEW
) -> Optional[pd.DataFrame]:
    """
    High-level API to run a validation pivot using a champions CSV.
    If `reports_dir` is provided, look for the champions there (recommended).
    If `ctx` is provided, artifacts are placed under ctx.{profiles,results,campaigns}.
    """
    # Where to read the champions CSV
    if reports_dir is None:
        reports_dir = (project_root / "src/experiment_configs/reports").resolve()

    logging.info("Planning validation run '%s'", run_name)
    candidate_df = load_and_filter_champions(reports_dir, master_profile_filename, filters)

    # print('master_profile_filename', master_profile_filename)
    # print('reports_dir', reports_dir)
    # print(candidate_df)


    candidate_df = apply_profile_overrides(
        candidate_df,
        config_overrides,
    )
    
    if candidate_df is None:
        return None

    display_run_plan(run_name, candidate_df, filters)

    logging.info("Preparing artifacts for '%s'", run_name)
    arts = prepare_run_artifacts(
        project_root=project_root,
        run_name=run_name,
        candidate_df=candidate_df,
        ensemble_size=ensemble_size,
        config_overrides=config_overrides,
        ctx=ctx,  # <— pass through
    )

    if delete_previous_results:
        clear_previous_run_artifacts(arts["run_dir"])

    logging.info("Executing validation in '%s' mode", execution_mode)

    if execution_mode == "interactive":
        cfg = load_campaign_config(arts["config_path"])

        rp_ov = (config_overrides or {}).get("run_params", {}) or {}
        if "ensemble_size" in rp_ov:
            cfg.run_params.ensemble_size = int(rp_ov["ensemble_size"])
        
        # (opcional) alguns executores só ativam ensemble se existir uma lista de seeds:
        if getattr(cfg.run_params, "ensemble_size", 1) > 1:
            base_seed = int(getattr(cfg.job_defaults, "seed", 42))
            esz = int(cfg.run_params.ensemble_size)
            cfg.run_params.seeds = list(range(base_seed, base_seed + esz))
            # se seu executor usa outro nome, deixe esse alias também:
            cfg.run_params.num_seeds = esz
        
        logging.info("Effective run_params at runtime: %s", dict(cfg.run_params))
        
        val_df = run_campaign_from_config_object(config=cfg, profile_override_path=str(arts["profile_path"]))
    elif execution_mode == "subprocess":
        run_script = project_root / "run_campaign.py"
        cmd = ["python", str(run_script), "--config", str(arts["config_path"]), "--profile", str(arts["profile_path"])]
        proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(project_root))
        if proc.returncode != 0:
            logging.error("Subprocess failed.\nSTDOUT:\n%s\nSTDERR:\n%s", proc.stdout, proc.stderr)
            return None
        results_root = Path(cfg.infra.experiments_output_dir) if 'cfg' in locals() else (project_root / "src/experiment_configs/results")
        from hpo.analysis import load_master_leaderboard
        val_df = load_master_leaderboard(run_name, results_root)
    else:
        raise ValueError(f"Unknown execution_mode: {execution_mode!r}. Use 'interactive' or 'subprocess'.")

    if val_df is None or val_df.empty:
        logging.error("Validation produced no results.")
        return None

    logging.info("✅ Validation run completed successfully.")
    return val_df

