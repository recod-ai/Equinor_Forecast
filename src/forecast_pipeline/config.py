# -*- coding: utf-8 -*-
"""
Settings and constants for Seq2Context, Seq2PIN_Trend, Seq2Fuser, and Seq2Value experiments.
The code is organized into logical sections to facilitate future maintenance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Set to True to enable deterministic behavior for reproducibility.
# When enabled, this will set random seeds for Python, NumPy, and TensorFlow,
# ensuring that model training produces the same results across runs for the
# same inputs and configurations. The default setting is True.
ENABLE_DETERMINISM = True

# =============================================================================
# I. PATHS AND DATASETS
# =============================================================================

# Base directory to store experiment results
BASE_DIR: Path = Path(__file__).resolve().parent.parent / "experiments"

# Default dataset
DEFAULT_DATASET: str = "UNISIM_IV"  # Possible alternatives: "UNISIM_IV", "VOLVE"

# =============================================================================
# II. DEFAULT EXPERIMENT PARAMETERS
# =============================================================================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# -------------------------------------------------------------------
# I. ARCHITECTURE SELECTION
# -------------------------------------------------------------------
# Change this to switch architecture:
ARCH: str = "Seq2PIN"

# Which of the family of seq2seq models we support
SEQ2SEQ_ARCHS: List[str] = ["Seq2Context", "Seq2PIN", "Seq2Trend", "Seq2Fuser"]

# -------------------------------------------------------------------
# II. DEFAULT EXPERIMENT PARAMETERS
# -------------------------------------------------------------------
@dataclass(frozen=True)
class DefaultExperimentParams:
    """
    Default parameters for all Seq2* architectures except Seq2Value.
    """
    forecast_policy            = "Hybrid PINN" # PINN Original, original_pinn
    architecture_name: str     = ARCH
    feature_kind: str          = "Normal"
    use_known_good: bool       = False
    data_sample: float         = 0.1
    lag_window: int            = 100
    horizon: int               = 300
    epochs: int                = 100
    patience: int              = 50
    batch_size: int            = 8
    learning_rate:float        = 5e-2
    test_size: float           = 0.45
    val_size: float            = 0.2
    aggregation_method: str    = "hp_hist" # "reconstruct", "reconstruct_warm_raw", "reconstruct_warm_hp", "reconstruct_warm_ewma"
    evaluate_by_slice: bool    = True
    slice_ratios: List[float]  = (1.0,)
    aggregation_quantiles: List[float] = (0.25, 0.5, 0.75)
    plot: bool                 = True
    # New parameters:
    scenario: str              = "P50"              # Options: P50, P90, P10, BAND
    band: Optional[List[float]] = None              # e.g. [0.1, 0.9] if scenario == "BAND"
    show_components: bool      = False

    # ------------------------------------------------------------------
    # NEW: latent / extrapolation configuration (Seq2* only)
    # ------------------------------------------------------------------
    # Controls how the latent-physics extrapolation behaves.
    # "off"          -> completely disabled (legacy behavior)
    # "full_sequence"-> also produce full-length 1D sequences per split
    latent_mode: str = "full_sequence"
    fullseq_mode: Optional[int] = "deploy_split_k" # Or use deploy_split_k
    fullseq_k: float = 1

# Create a plain dict for downstream code
DEFAULT_EXP_PARAMS: Dict[str, Any] = DefaultExperimentParams().__dict__


# -------------------------------------------------------------------
# III. LOG LEVEL AND PARALLELISM
# -------------------------------------------------------------------
LOG_LEVEL: int    = 1    # 0 → progress bar only; 1 → adds logging.info; 2 → all detailed outputs
MAX_WORKERS: int  = 1    # Maximum number of workers for parallelism


# -------------------------------------------------------------------
# IV. SWEEP OPTIONS: STRATEGIES, EXTRACTORS, AND FUSERS
# -------------------------------------------------------------------
STRATEGY_OPTIONS: List[Dict[str, str]] = [
    {"strategy_name": "pressure_ensemble"},
    {"strategy_name": "arps"},
    {"strategy_name": "combined_exp_arps"},
    {"strategy_name": "exponential"},
    {"strategy_name": "static"},
]

# Conditionally override extractor and fuser options
if ARCH in {"Seq2PIN", "Seq2Trend"}:
    EXTRACTOR_OPTIONS = [{"type": "None"}]
    FUSER_OPTIONS    = [{"type": "None"}]
else:
    EXTRACTOR_OPTIONS: List[Dict[str, str]] = [
        # {"type": "tcn"},
        # {"type": "rnn"},
        # {"type": "cnn"},
        {"type": "aggregate"},
        # {"type": "identity"},
    ]
    FUSER_OPTIONS: List[Dict[str, str]] = [
        # {"type": "film"},
        {"type": "bias_scale"},
        # {"type": "prob_bias"},
        # {"type": "None"},
    ]


# If the architecture is Seq2Value, use this alternative set:
# DEFAULT_EXP_PARAMS_SEQ2VALUE: Dict[str, Any] = {
#     "architecture_name": "Seq2Value",
#     "feature_kind": "Normal",
#     "use_known_good": False,
#     "lag_window": 7,
#     "horizon": 30,
#     "epochs": 100,
#     "batch_size": 64,
#     "patience": 100,
#     "test_size": 0.6,
#     "aggregation_method": "median",
#     "evaluate_by_slice": True,
#     "slice_ratios": [1.0],
#     "aggregation_quantiles": [0.25, 0.5, 0.75],
#     "plot": False,
# }

from pathlib import Path

# =============================================================================
# I. CORE PROJECT PATHS
# =============================================================================

# Define the absolute root of the project
# This assumes config.py is in src/forecast_pipeline/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Centralized directory for all experiment input files
EXPERIMENT_CONFIG_DIR = PROJECT_ROOT / "src" / "experiment_configs"

# Directory for architecture definitions
ARCHITECTURE_YAML_PATH = EXPERIMENT_CONFIG_DIR / "architectures_PINNs.yaml"

# NEW: Centralized directory for all generated profiles (grid search, HPO, etc.)
PROFILES_DIR = EXPERIMENT_CONFIG_DIR / "profiles"

# NEW: Centralized root directory for all experiment outputs (results, plots, etc.)
EXPERIMENTS_OUTPUT_DIR = EXPERIMENT_CONFIG_DIR / "results"

HPO_STUDIES_DIR = EXPERIMENT_CONFIG_DIR / "studies"


# =============================================================================
# V. VARIABLE MAPPING AND CANONICAL FEATURES
# =============================================================================

# Dictionary for column renaming in raw reads
VARIABLE_MAPPING: Dict[str, str] = {
    "Well pressure": "PWFO",
    "Oil flow": "QOOB",
    "Water flow": "QWOB",
    "Liquid flow (oil + water)": "QLOB",
    "Gas flow": "QGOB",
}

# List of canonical features that can be used in different experiments
CANON_FEATURES: List[str] = [
    "PI",
    "CE",
    "BORE_GAS_VOL",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_WHP_P",
    "Tempo_Inicio_Prod",
    "Taxa_Declinio",
    "BORE_OIL_VOL",
]

# Specific mapping for UNISIM-IV → renames column names to canonical columns
_UNISIM_IV_MAP: Dict[str, str] = {
    "Gas Rate SC":               "BORE_GAS_VOL",
    "Oil Rate SC":               "BORE_OIL_VOL",
    "Well Bottom-hole Pressure": "AVG_DOWNHOLE_PRESSURE",
    # There is no “AVG_WHP_P” in UNISIM-IV → column will be created empty later
    "delta_P":                   "delta_P",  # Not part of CANON_FEATURES
    "PI":                        "PI",
    "Tempo_Inicio_Prod":         "Tempo_Inicio_Prod",
    "Taxa_Declinio":             "Taxa_Declinio",
    # CE does not exist in UNISIM-IV → will be filled with NaN in preprocessing
}

# =============================================================================
# VI. UNIT CONSTANTS AND CONVERSION FUNCTIONS
# =============================================================================

# Unit conversion factors
KPA_TO_PSI: float       = 0.145037738   # 1 kPa → psi
BAR_TO_PSI: float       = 14.5038       # 1 bar → psi
M3DAY_TO_STBDAY: float  = 6.2898        # 1 m³/d → stb/d
M3DAY_TO_SCFDAY = 35.3147               # 1 m³/d → scf/d

def kpa2psi(p_kpa: float) -> float:
    return p_kpa * KPA_TO_PSI

def psi2kpa(p_psi: float) -> float:
    return p_psi / KPA_TO_PSI

def bar2psi(p_bar: float) -> float:
    return p_bar * BAR_TO_PSI

def m3d2stbd(q_m3d: float) -> float:
    return q_m3d * M3DAY_TO_STBDAY

def stbd2m3d(q_stb: float) -> float:
    return q_stb / M3DAY_TO_STBDAY

def m3d2scfd(q_m3d: float) -> float:
    return q_m3d * M3DAY_TO_SCFDAY

# =============================================================================
# VII. INITIAL PRESSURES BY DATASET
# =============================================================================

# Initial pressure for each case
INITIAL_PRESSURE: Dict[str, float] = {
    "UNISIM-IV-2024": 63_000.0,
    "VOLVE": 310.0,
}

# =============================================================================
# VIII. METRICS CONFIGURATIONS AND RESULT COLUMNS
# =============================================================================

# Columns that should always be dropped from result DataFrames
_COLS_TO_DROP_ALWAYS: List[str] = ["Method", "Kind"]
_COLS_TO_DROP_FILTER: List[str] = ["adaptive_filter", "filter_method"]

# Names of main metrics
_METRIC_NAMES: List[str] = ["R²", "MAE", "SMAPE"]

# Result column order: validation values first, then test values
_METRIC_COLS_ORDER: List[str] = [f"{m}_VAL" for m in _METRIC_NAMES] + [f"{m}_TEST" for m in _METRIC_NAMES]

# Default column order in aggregated tables:
_BASE_ORDER: List[str] = ["Well", "Category", "strategy", "extractor", "fuser"]
_FILTER_ORDER: List[str] = ["Well", "Category", "adaptive_filter", "filter_method", "strategy", "extractor", "fuser"]

# =============================================================================
# IX. EXPERIMENT CONFIGURATIONS (BY DATASET AND ARCHITECTURE)
# =============================================================================

# Determines which configuration set to use according to the selected architecture
architecture_name: str = DEFAULT_EXP_PARAMS.get("architecture_name")

if architecture_name in ("Seq2Context", "Seq2PIN", "Seq2Trend", "Seq2Fuser"):
    EXPERIMENT_CONFIGURATIONS: Dict[str, List[Dict[str, Any]]] = {
        "VOLVE": [
            {"selected_features": CANON_FEATURES},
        ],
        "UNISIM_IV": [
            {"selected_features": CANON_FEATURES},
        ],
    }
elif architecture_name == "Seq2Value":
    EXPERIMENT_CONFIGURATIONS: Dict[str, List[Dict[str, Any]]] = {
        "VOLVE": [
            {"selected_features": ["PI", "BORE_GAS_VOL", "BORE_OIL_VOL"]},
        ],
    }
else:
    # If new architectures need to be added, include them here
    EXPERIMENT_CONFIGURATIONS: Dict[str, List[Dict[str, Any]]] = {}

# Additional configuration for generic UNISIM experiments
EXPERIMENT_CONFIGURATIONS_3: Dict[str, List[Dict[str, Any]]] = {
    "UNISIM": [
        {"selected_features": ["QLOB", "QGOB", "QOOB"]},
    ]
}

# Additional configuration for OPSD experiments
EXPERIMENT_CONFIGURATIONS_4: Dict[str, List[Dict[str, Any]]] = {
    "OPSD": [
        {"selected_features": ["GB_GBN_wind_generation_tax", "GB_GBN_wind_generation_actual"]},
    ]
}

import logging
from typing import Dict, Any, List

def get_experiment_base_config(dataset_name: str, architecture_name: str) -> Dict[str, Any]:
    """
    Returns the rule-based base configuration (e.g., feature list) for a given
    dataset and architecture. Preserves existing behavior and adds ARPS support.
    """
    arch = str(architecture_name or "").strip()

    # --- 1) Seq2 family (preserve current canonical behavior) ---
    if arch in ("Seq2Context", "Seq2PIN", "Seq2Trend", "Seq2Fuser"):
        if dataset_name == "VOLVE":
            return {"selected_features": CANON_FEATURES}
        elif dataset_name == "UNISIM_IV":
            return {"selected_features": CANON_FEATURES}

    # --- 2) Darts family (preserve lean covariates) ---
    if arch.startswith("Darts"):
        darts_features: List[str] = [
            "PI",                    # covariate
            "AVG_DOWNHOLE_PRESSURE", # covariate
            "Tempo_Inicio_Prod",     # time
            "BORE_OIL_VOL",          # target (must be included)
        ]
        # Ensure target is present (defensive)
        if "BORE_OIL_VOL" not in darts_features:
            darts_features.append("BORE_OIL_VOL")
        return {"selected_features": darts_features}

    # --- 3) Seq2Value (preserve current rule) ---
    if arch == "Seq2Value" and dataset_name == "VOLVE":
        return {"selected_features": ["PI", "BORE_GAS_VOL", "BORE_OIL_VOL"]}

    # --- 4) NEW: ARPS canonical backend (minimal & safe) ---
    # ARPS trains on the physical series; require time + target only.
    if arch == "Arps_Canonical" or arch.startswith("Arps_"):
        arps_feats: List[str] = ["Tempo_Inicio_Prod", "BORE_OIL_VOL"]
        # Defensive: guarantee uniqueness and target presence
        if "BORE_OIL_VOL" not in arps_feats:
            arps_feats.append("BORE_OIL_VOL")
        # De-duplicate preserving order
        seen, uniq = set(), []
        for f in arps_feats:
            if f not in seen and f is not None:
                uniq.append(f); seen.add(f)
        return {"selected_features": uniq}

    # --- 5) Additional dataset-specific fallbacks (preserve your originals) ---
    if dataset_name == "UNISIM":
        return EXPERIMENT_CONFIGURATIONS_3.get("UNISIM", [{}])[0]
    if dataset_name == "OPSD":
        return EXPERIMENT_CONFIGURATIONS_4.get("OPSD", [{}])[0]

    # --- 6) Final fallback: keep running with a minimal safe set ---
    # Prefer not to warn noisily; provide a minimal default that works in most loaders.
    logging.info(
        "No specific base config found for %s/%s. Falling back to minimal features.",
        dataset_name, architecture_name
    )
    return {"selected_features": ["Tempo_Inicio_Prod", "BORE_OIL_VOL"]}


# =============================================================================
# X. FEW-SHOT PREDICTION SETTINGS
# =============================================================================
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import List

SHAP_ANALYSIS = False

class ExecutionMode(Enum):
    """Available execution modes."""
    SIMPLE = 'simple'
    MANIFEST = 'manifest'
    SENSITIVITY = 'sensitivity'

@dataclass(frozen=True)
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    window_sizes: List[int] = field(default_factory=lambda: [3, 7, 14, 21])
    forecast_horizons: List[int] = field(default_factory=lambda: [14, 28, 56, 70, 94, 112])
    datasets_filter: List[str] = field(default_factory=lambda: ["UNISIM", "VOLVE", "OPSD"])
    architecture: str = 'Generic'

@dataclass(frozen=True)
class SimpleConfig:
    """Configuration for simple execution mode."""
    datasets_filter: List[str] = field(default_factory=lambda: ["UNISIM", "VOLVE", "OPSD"])

@dataclass(frozen=True)
class ManifestConfig:
    """Configuration for manifest execution mode."""
    path: Path = field(default_factory=lambda: Path.cwd().parent.parent / "output_manifest" / "manifest.csv")
    output_notebooks_dir: Path = field(default_factory=lambda: Path.cwd().parent.parent / "output_notebooks")

@dataclass(frozen=True)
class Config:
    """Top-level execution control configuration."""
    start_fresh: bool = False
    project_root: Path = field(default_factory=lambda: Path.cwd().parent.parent)
    execution_mode: ExecutionMode = ExecutionMode.SIMPLE
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    simple: SimpleConfig = field(default_factory=SimpleConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)
    template_notebook: str = "base_pipeline.ipynb"
    max_concurrent_jobs: int = 12
    generate_manifest_script: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'generate_manifest_script',
            self.project_root / "src" / "generate_manifest.py"
        )

# --- Theme Definition ---
themes = {
    "minimal": {"text": "#333333", "bg": "#FFFFFF", "accent": "#4CAF50", "grid": "#DDDDDD"},
    "dark": {"text": "#F0F0F0", "bg": "#2C2C2C", "accent": "#76B947", "grid": "#555555"}
}

class AggregationMode(Enum):
    SIMPLE = 'simple'
    MANIFEST = 'manifest'
    SENSITIVITY = 'sensitivity'

@dataclass(frozen=True)
class AnalysisConfig:
    aggregation_mode: AggregationMode = AggregationMode.SIMPLE
    exclude_datasets: List[str] = field(default_factory=lambda: ["UNISIM_IV"])
    top_n_configs: int = 3
    project_root: Path = Path.cwd().parent.parent
    output_notebooks_dir: Path = field(init=False)
    manifest_path: Path = field(init=False)
    config_dir: Path = field(init=False)
    custom_dataset_order: List[str] = field(default_factory=lambda: ["VOLVE", "UNISIM", "OPSD"])
    table_theme: str = "dark"

    def __post_init__(self) -> None:
        object.__setattr__(self, 'output_notebooks_dir', self.project_root / "notebooks/output_notebooks")
        object.__setattr__(self, 'manifest_path', self.project_root / "output_manifest" / "manifest.csv")
        object.__setattr__(self, 'config_dir', self.project_root / "src" / "experiment_configs")

config = AnalysisConfig()

if config.aggregation_mode == AggregationMode.MANIFEST:
    print(f"Showing top {config.top_n_configs} configurations details")

if config.aggregation_mode == AggregationMode.MANIFEST:
    print(f"Checking manifest at: '{config.manifest_path}'")
    if not config.manifest_path.exists():
        print("WARNING: Manifest file not found. Run 'generate_manifest.py' first.")
    print(f"Checking config directory at: '{config.config_dir}'")
    if not config.config_dir.exists():
        print("WARNING: Config directory not found.")

def validate_config(config: Config) -> None:
    """
    Validates execution mode and existence of manifest when required.
    Raises:
        ValueError: If execution mode is invalid.
    """
    if config.execution_mode not in ExecutionMode:
        raise ValueError("EXECUTION_MODE must be 'simple', 'manifest', or 'sensitivity'.")
    if config.execution_mode == ExecutionMode.MANIFEST and not config.manifest.path.exists():
        print(f"WARNING: Manifest file not found at '{config.manifest.path}'.")
        print("Please run the script 'src/generate_manifest.py' first.")
