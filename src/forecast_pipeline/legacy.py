# def engineer_features(
#     df: pd.DataFrame,
#     cum_sum: bool = False,
#     well: str = None
# ) -> pd.DataFrame:
#     """
#     Realiza o feature engineering no DataFrame, adicionando novas features e
#     opcionalmente adicionando features físicas.

#     Args:
#         df (pd.DataFrame): DataFrame processado.
#         cum_sum (bool, optional): Indica se as colunas devem ser acumuladas. Defaults to False.
#         add_physical_features (bool, optional): Indica se features físicas devem ser adicionadas. Defaults to False.

#     Returns:
#         pd.DataFrame: DataFrame com as novas features adicionadas.
#     """
    
#     # Calcula a taxa de declínio
#     # df['Taxa_Declinio'] = df['BORE_OIL_VOL'].pct_change()
#     df['Taxa_Declinio'] = -np.log(df['BORE_OIL_VOL'] / df['BORE_OIL_VOL'].shift(1))


#     # Adiciona a coluna 'Tempo_Inicio_Prod' iniciando em 1
#     df['Tempo_Inicio_Prod'] = (df.index + 1)

#     # Cálculo de índices adicionais
#     # 1. Gradiente de Pressão (ΔP)
#     df['delta_P'] = df['AVG_DOWNHOLE_PRESSURE'] - df['AVG_WHP_P']

#     # 7. Índice de Produtividade (PI)
#     df['PI'] = df['BORE_OIL_VOL'] / df['delta_P'].replace(0, np.nan)

#     # 8. Eficiência de Choke (CE)
#     df['CE'] = df['BORE_OIL_VOL'] / df['AVG_CHOKE_SIZE_P'].replace(0, np.nan)
    
#     # plot_time_series(df['BORE_OIL_VOL'], 'BORE_OIL_VOL', well)

#     if cum_sum:
#         df['BORE_OIL_VOL'] = df['BORE_OIL_VOL'].cumsum()
#         df['BORE_GAS_VOL'] = df['BORE_GAS_VOL'].cumsum()
#         df['BORE_WI_VOL_15_9_F_4'] = df['BORE_WI_VOL_15_9_F_4'].cumsum()
        
#         selected_features = ['BORE_GAS_VOL', 'BORE_WAT_VOL', 'BORE_WI_VOL_15_9_F_4', 'CE', 'BORE_OIL_VOL']    
#         df = df[selected_features].copy()

#     return df

# ----------------------------------------------------------------------
# engineer_features ­– versão “Volve ready” (mesma assinatura solicitada)


Preciso refatorar o código abaixo, de forma mais organizada, setorizando as coisas. O código já funciona, então a lógica deve ser mantida. 


from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent / "experiments"

DEFAULT_DATASET = "VOLVE" # "UNISIM_IV"

# Default experiment parameters Seq2Context or Seq2PIN_Trend
DEFAULT_EXP_PARAMS: Dict[str, Any] = {
    "architecture_name": "Seq2PIN_Trend",
    "feature_kind": "Normal",
    "use_known_good": False,
    "lag_window": 30,
    "horizon": 30,
    "epochs": 100,
    "batch_size": 8,
    "patience": 50,
    "test_size": 0.6,
    "aggregation_method": "median",
    "evaluate_by_slice": True,
    "slice_ratios": [1.0],
    "aggregation_quantiles": [0.25, 0.5, 0.75],
    "plot": True,
    # NOVOS ---------------------------------------------------------
    "scenario": "P50",           # P50, P90, P10, BAND
    "band": None,                # ex. [0.1,0.9] se BAND
    "show_components": False,
}

# Default experiment parameters Seq2Value
# DEFAULT_EXP_PARAMS: Dict[str, Any] = {
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
#     # "slice_ratios": [0.1, 0.25, 0.5, 0.75, 1.0],
#      "slice_ratios": [1.0],
#     "aggregation_quantiles": [0.25, 0.5, 0.75],
#     "plot": True
# }
    

# 0 - only the progress bar
# 1 - add logging.info,
# 2 - All outputs

LOG_LEVEL = 1

# Paralelismo para múltiplos jobs
MAX_WORKERS = 12

# -----------------------------------------------------------------------------
# Sweep configurations for physics strategies, context extractors, and fusers.
# -----------------------------------------------------------------------------
STRATEGY_OPTIONS = [
    {"strategy_name": "pressure_ensemble"},
    {"strategy_name": "arps"},
    {"strategy_name": "combined_exp_arps"},
    {"strategy_name": "exponential"},
    {"strategy_name": "static"},
    
]
# {"strategy_name": "diffusivity_decay"},

EXTRACTOR_OPTIONS = [
    {"type": "tcn"},
    {"type": "rnn"},
    {"type": "cnn"},
    {"type": "aggregate"},
    {"type": "identity"},
    {"type": "None"},
]

FUSER_OPTIONS = [
    {"type": "film"},
    {"type": "bias_scale"},
    {"type": "prob_bias"},
     {"type": "None"},
]

# Mapping dictionary for variable names
VARIABLE_MAPPING = {
    'Well pressure': 'PWFO',
    'Oil flow': 'QOOB',
    'Water flow': 'QWOB',
    'Liquid flow (oil + water)': 'QLOB',
    'Gas flow': 'QGOB',
}


CANON_FEATURES = [
    "PI",
    "CE",
    "BORE_GAS_VOL",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_WHP_P",
    "Tempo_Inicio_Prod",
    "Taxa_Declinio",
    "BORE_OIL_VOL",
]


_UNISIM_IV_MAP = {
    "Gas Rate SC":              "BORE_GAS_VOL",
    "Oil Rate SC":              "BORE_OIL_VOL",
    "Well Bottom-hole Pressure": "AVG_DOWNHOLE_PRESSURE",
    # não há “AVG_WHP_P” aqui → vamos criar coluna vazia
    "delta_P":                  "delta_P",  # mas não faz parte das CANON_FEATURES
    "PI":                       "PI",
    "Tempo_Inicio_Prod":        "Tempo_Inicio_Prod",
    "Taxa_Declinio":            "Taxa_Declinio",
    # CE não existe em UNISIM-IV → vai ser NaN
}

# units -----------------------------------------------------------
KPA_TO_PSI = 0.145037738    # 1 kPa  ➜ psi
M3DAY_TO_STBDAY = 6.2898    # 1 m³/d ➜ stb/d
BAR_TO_PSI      = 14.5038          # 1 bar  → psi

def kpa2psi(p_kpa):    return p_kpa * KPA_TO_PSI
def psi2kpa(p_psi):    return p_psi / KPA_TO_PSI
def m3d2stbd(q_m3d):    return q_m3d * M3DAY_TO_STBDAY
def stbd2m3d(q_stb):    return q_stb / M3DAY_TO_STBDAY
def bar2psi(p_bar):   return p_bar * BAR_TO_PSI
def m3d2scfd(q_m3d):  return q_m3d * M3DAY_TO_SCFDAY

INITIAL_PRESSURE = {
    "UNISIM-IV-2024": 63000.0,
    "VOLVE": 310,
}


# --- Constants and Configuration ---
_COLS_TO_DROP_ALWAYS = ["Method", "Kind"]
_COLS_TO_DROP_FILTER = ["adaptive_filter", "filter_method"]
_METRIC_NAMES      = ["R²", "MAE", "SMAPE"]
_METRIC_COLS_ORDER = [f"{m}_VAL" for m in _METRIC_NAMES] + [f"{m}_TEST" for m in _METRIC_NAMES]
_BASE_ORDER = ["Well", "Category", "strategy", "extractor", "fuser"]
_FILTER_ORDER = ["Well", "Category", "adaptive_filter", "filter_method", "strategy", "extractor", "fuser"]

CANON_FEATURES = [
    "PI",
    "CE",
    "BORE_GAS_VOL",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_WHP_P",
    "Tempo_Inicio_Prod",
    "Taxa_Declinio",
    "BORE_OIL_VOL",
]


name = DEFAULT_EXP_PARAMS.get('architecture_name')
if name in ('Seq2Context', 'Seq2PIN_Trend', 'Seq2Fuser'):
    EXPERIMENT_CONFIGURATIONS: Dict[str, List[Dict[str, Any]]] = {
            "VOLVE": [{ "selected_features": CANON_FEATURES }],
            "UNISIM_IV": [{ "selected_features": CANON_FEATURES }],
    }

if DEFAULT_EXP_PARAMS['architecture_name'] == 'Seq2Value':
    EXPERIMENT_CONFIGURATIONS: Dict[str, List[Dict[str, Any]]] = {
        "VOLVE": [
            {
                "selected_features": [
                    "PI",
                    "BORE_GAS_VOL",
                    "BORE_OIL_VOL",
                ]
            }
        ]
    }

# 3) unisim
EXPERIMENT_CONFIGURATIONS_3: Dict[str, List[Dict[str, Any]]] = {
    "UNISIM": [
        {
            "selected_features": [
                "QLOB",
                "QGOB",
                "QOOB",
            ]
        }
    ]
}

# 4) opsd
EXPERIMENT_CONFIGURATIONS_4: Dict[str, List[Dict[str, Any]]] = {
    "OPSD": [
        {
            "selected_features": [
                "GB_GBN_wind_generation_tax",
                "GB_GBN_wind_generation_actual",
            ]
        }
    ]
}



def create_model(
    input_shape,
    horizon,
    scaler_X_path='scalers/scaler_X.pkl',
    scaler_target_path='scalers/scaler_target.pkl',
    strategy_config={"strategy_name": "pressure_ensemble"},
    trend_degree=2,
    phase="balanced",
    freeze_trend=False,
    freeze_physics=False,
    fusion_type="pin",
    extractor_config=None,
    fuser_config=None,
    name='Seq2PIN'
) -> Model:
    """
    Constructs the oil production forecasting model with four possible modes.
    
    Parameters:
      input_shape: Input data shape.
      horizon: Forecast horizon (number of timesteps).
      scaler_X_path, scaler_target_path: Paths for scaling.
      strategy_config: exponential, arps, static, weighted_ensemble, combined_exp_arps, pressure_ensemble, diffusivity_decay, diffusivity_pressure.
      trend_degree: Degree parameter for the TrendBlock.
      phase: A phase label (for logging).
      freeze_trend: If True, freeze the Trend branch.
      freeze_physics: If True, freeze the PIN (physics) branch.
      fusion_type: One of "trend", "pin", "average", or "concat_dense".
    
    Returns:
      A compiled Keras model.
    """
    timesteps, total_features = input_shape.shape[1], input_shape.shape[2]
    inputs = Input(shape=(timesteps, total_features), name='all_features')

    # Feature extraction indices
    DATA_INDICES = [0, 2, 7]
    PHYSICS_INDICES = [0, 3, 5]
    
    data_features = Lambda(lambda x: tf.gather(x, DATA_INDICES, axis=-1), name='data_features')(inputs)
    physics_features = Lambda(lambda x: tf.gather(x, PHYSICS_INDICES, axis=-1), name='physics_features')(inputs)
    

    # -------- Trend Branch --------
    pooled_data = GlobalAveragePooling1D(name='avg_pool_trend')(data_features)
    trend_block = TrendBlock(degree=trend_degree, forecast_horizon=horizon, name='trend_block')
    trend_forecast = trend_block(pooled_data)
    
    
    # -------- PIN (Physics) Branch --------
    physics_horizon = Lambda(lambda x: x[:, -horizon:, :], name='physics_horizon')(physics_features)
    physics_block = DarcyPhysicsLayer(
        scaler_X_path=scaler_X_path,
        scaler_target_path=scaler_target_path,
        strategy_config=strategy_config,
        name='physics_block'
    )
    physics_forecast = physics_block(physics_horizon)
    
    # Apply freeze settings
    trend_block.trainable = not freeze_trend
    physics_block.trainable = not freeze_physics

    # --- For pure branch models based on fusion_type ---
    if fusion_type.lower() == "trend":
        model = Model(inputs, trend_forecast, name="trend_block")
        tf.print("Created Trend-only model.")
        return model

    if name == "Seq2PIN":
        model = Model(inputs, physics_forecast, name="physics_block")
        tf.print("Created PIN-only model.")
        return model

    # # --- Hybrid Models ---
    # if fusion_type.lower() == "concat_dense":
    #     combined = Concatenate(name='fusion_concat')([trend_forecast, physics_forecast])
    #     final_forecast = Dense(horizon, activation='linear', name=fusion_type)(combined)
    # elif fusion_type.lower() == "average":
    #     final_forecast = Lambda(average_outputs, name='fusion_avg')([trend_forecast, physics_forecast])
    # else:
    #     raise ValueError(f"Invalid fusion_type: {fusion_type}")
    
    # model = Model(inputs, final_forecast, name=fusion_type)
    # return model

# class TrendBlock(tf.keras.layers.Layer):
#     """
#     Polynomial trend of fixed degree.
#     • inicialização determinística (zeros) ― arranca sempre da linha reta
#     • L2 forte + max-norm 1.0 ― evita explosão dos coeficientes
#     """
#     def __init__(self, degree: int = 3, forecast_horizon: int = 1, **kwargs):
#         super().__init__(**kwargs)
#         self.degree = degree
#         self.h      = forecast_horizon     # abreviação interna

#     # ------------------------------------------------------------ build
#     def build(self, _):
#         self.theta_layer = Dense(
#             self.degree + 1,
#             kernel_initializer=HeNormal(),          # <<< (1) init neutro
#             bias_initializer=HeNormal(),
#             kernel_regularizer=L2(1e-3),         # <<< (2) L2 forte
#             kernel_constraint=max_norm(1.0)      # <<< (3) coef ≤ 1
#         )

#         t = np.linspace(0, 1, self.h)
#         basis = np.vstack([t**i for i in range(self.degree + 1)]).T   # (H, D+1)
#         self.basis = tf.constant(basis, dtype=tf.float32)

#     # ------------------------------------------------------------ call
#     def call(self, inputs):
#         theta = self.theta_layer(inputs)                 # (B, D+1)
#         return tf.matmul(theta, self.basis, transpose_b=True)  # (B, H)

#     # ------------------------------------------------------------ config
#     def get_config(self):
#         cfg = super().get_config()
#         cfg.update({"degree": self.degree,
#                     "forecast_horizon": self.h})
#         return cfg
