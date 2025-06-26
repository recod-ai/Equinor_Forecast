# -*- coding: utf-8 -*-
"""
Configurações e constantes para experimentos de Seq2Context, Seq2PIN_Trend, Seq2Fuser e Seq2Value.
O código foi reorganizado em seções lógicas para facilitar manutenções futuras.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

# =============================================================================
# I. PATHS E DATASETS
# =============================================================================

# Diretório base para armazenar resultados de experimentos
BASE_DIR: Path = Path(__file__).resolve().parent.parent / "experiments"

# Dataset padrão
DEFAULT_DATASET: str = "UNISIM_IV"  # Alternativa possível: "UNISIM_IV", "VOLVE"


# =============================================================================
# II. PARÂMETROS PADRÃO DE EXPERIMENTO
# =============================================================================

SEQ2SEQ_ARCHS = ["Seq2Context", "Seq2PIN", "Seq2Trend", "Seq2Fuser"]

@dataclass(frozen=True)
class DefaultExperimentParams:
    """
    Parâmetros padrão que se aplicam a arquiteturas Seq2Context, Seq2PIN/Seq2Trend e Seq2Fuser.
    Para Seq2Value, será definido seu próprio dicionário logo abaixo.
    """
    architecture_name: str = "Seq2Context"
    feature_kind: str = "Normal"
    use_known_good: bool = False
    lag_window: int = 30
    horizon: int = 30
    epochs: int = 250
    batch_size: int = 8
    patience: int = 100
    test_size: float = 0.6
    aggregation_method: str = "median"
    evaluate_by_slice: bool = True
    slice_ratios: List[float] = (1.0,)
    aggregation_quantiles: List[float] = (0.25, 0.5, 0.75)
    plot: bool = False
    # Novos parâmetros:
    scenario: str = "P50"             # Opções: P50, P90, P10, BAND
    band: Optional[List[float]] = None  # Ex: [0.1, 0.9] se scenario == "BAND"
    show_components: bool = False

# Instância de parâmetros que será usada no restante do código
DEFAULT_EXP_PARAMS: Dict[str, Any] = DefaultExperimentParams().__dict__


# Se a arquitetura for Seq2Value, usamos este conjunto alternativo:
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


# =============================================================================
# III. NÍVEL DE LOG E PARALELISMO
# =============================================================================

# 0 → somente progress bar
# 1 → adiciona logging.info
# 2 → todos os outputs detalhados
LOG_LEVEL: int = 1

# Número máximo de workers para paralelismo em múltiplos jobs
MAX_WORKERS: int = 6


# =============================================================================
# IV. OPÇÕES DE SWEEP: ESTRATÉGIAS, EXTRATORES E FUSERS
# =============================================================================

STRATEGY_OPTIONS: List[Dict[str, str]] = [
    {"strategy_name": "pressure_ensemble"},
    {"strategy_name": "arps"},
    {"strategy_name": "combined_exp_arps"},
    {"strategy_name": "exponential"},
    {"strategy_name": "static"},
]
# {"strategy_name": "diffusivity_decay"},  # Desativada por padrão

EXTRACTOR_OPTIONS: List[Dict[str, str]] = [
    {"type": "tcn"},
    {"type": "rnn"},
    {"type": "cnn"},
    {"type": "aggregate"},
    {"type": "identity"},
    # {"type": "None"},
]

FUSER_OPTIONS: List[Dict[str, str]] = [
    {"type": "film"},
    {"type": "bias_scale"},
    # {"type": "prob_bias"},
    # {"type": "None"},
    # {"type": "None"},
]


# =============================================================================
# V. MAPEAMENTO DE VARIÁVEIS E FEATURES CANÔNICAS
# =============================================================================

# Dicionário para renomeação de colunas nas leituras brutas
VARIABLE_MAPPING: Dict[str, str] = {
    "Well pressure": "PWFO",
    "Oil flow": "QOOB",
    "Water flow": "QWOB",
    "Liquid flow (oil + water)": "QLOB",
    "Gas flow": "QGOB",
}

# Lista de features canônicas que podem ser usadas em diferentes experimentos
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

# Mapeamento específico para UNISIM-IV → renomeia nomes de coluna para colunas canônicas
_UNISIM_IV_MAP: Dict[str, str] = {
    "Gas Rate SC":               "BORE_GAS_VOL",
    "Oil Rate SC":               "BORE_OIL_VOL",
    "Well Bottom-hole Pressure": "AVG_DOWNHOLE_PRESSURE",
    # Não há “AVG_WHP_P” em UNISIM-IV → coluna será criada vazia posteriormente
    "delta_P":                   "delta_P",  # Não faz parte de CANON_FEATURES
    "PI":                        "PI",
    "Tempo_Inicio_Prod":         "Tempo_Inicio_Prod",
    "Taxa_Declinio":             "Taxa_Declinio",
    # CE não existe em UNISIM-IV → será preenchido com NaN no preprocessamento
}


# =============================================================================
# VI. CONSTANTES DE UNIDADES E FUNÇÕES DE CONVERSÃO
# =============================================================================

# Fatores de conversão de unidades
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
# VII. PRESSÕES INICIAIS POR DATASET
# =============================================================================

# Pressão inicial para cada case
INITIAL_PRESSURE: Dict[str, float] = {
    "UNISIM-IV-2024": 63_000.0,
    "VOLVE": 310.0,
}


# =============================================================================
# VIII. CONFIGURAÇÕES DE MÉTRICAS E COLUNAS PARA RESULTADOS
# =============================================================================

# Colunas que sempre devem ser descartadas em DataFrames de resultados
_COLS_TO_DROP_ALWAYS: List[str] = ["Method", "Kind"]
_COLS_TO_DROP_FILTER: List[str] = ["adaptive_filter", "filter_method"]

# Nominação das métricas principais
_METRIC_NAMES: List[str] = ["R²", "MAE", "SMAPE"]

# Ordem das colunas dos resultados: valores de validação primeiro, depois de teste
_METRIC_COLS_ORDER: List[str] = [f"{m}_VAL" for m in _METRIC_NAMES] + [f"{m}_TEST" for m in _METRIC_NAMES]

# Ordem padrão de colunas em tabelas agregadas:
_BASE_ORDER: List[str] = ["Well", "Category", "strategy", "extractor", "fuser"]
_FILTER_ORDER: List[str] = ["Well", "Category", "adaptive_filter", "filter_method", "strategy", "extractor", "fuser"]


# =============================================================================
# IX. CONFIGURAÇÕES DE EXPERIMENTOS (POR DATASET E ARQUITETURA)
# =============================================================================

# Determina qual conjunto de configuracões usar conforme a arquitetura selecionada
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
    # Caso seja necessário adicionar novas arquiteturas, incluir aqui
    EXPERIMENT_CONFIGURATIONS: Dict[str, List[Dict[str, Any]]] = {}


# Configuração adicional para experimentos UNISIM genérico
EXPERIMENT_CONFIGURATIONS_3: Dict[str, List[Dict[str, Any]]] = {
    "UNISIM": [
        {"selected_features": ["QLOB", "QGOB", "QOOB"]},
    ]
}

# Configuração adicional para experimentos OPSD
EXPERIMENT_CONFIGURATIONS_4: Dict[str, List[Dict[str, Any]]] = {
    "OPSD": [
        {"selected_features": ["GB_GBN_wind_generation_tax", "GB_GBN_wind_generation_actual"]},
    ]
}
