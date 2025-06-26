import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from models.PINNs import DynamicEnsembleStrategy, ExponentialDecayStrategy, ArpsDeclineStrategy, WeightedEnsembleStrategy, CombinedExpArpsStrategy, StaticPressureStrategy
from tensorflow.keras.initializers import HeNormal

from utils.utilities import load_scaler, invert_feature_scaling, get_center_and_scale
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm

# =====================================================
# A. TrendBlock: Data-Driven Residual Trend Module
# =====================================================
class TrendBlock(tf.keras.layers.Layer):
    """
    Captures residual trends not explained by the physics.
    It predicts polynomial coefficients to model smooth trends such as gradual reservoir depletion.
    """
    def __init__(self, degree: int = 3, forecast_horizon: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.forecast_horizon = forecast_horizon

    def build(self, input_shape):
        # Fully connected layer to predict polynomial coefficients.
        self.theta_layer = Dense(self.degree + 1, kernel_initializer=HeNormal())
        # Create a fixed polynomial basis normalized over the forecast horizon.
        t = np.linspace(0, 1, self.forecast_horizon)
        basis = np.vstack([t**i for i in range(self.degree + 1)]).T  # Shape: (forecast_horizon, degree+1)
        self.basis = tf.constant(basis, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        theta = self.theta_layer(inputs)  # (batch, degree+1)
        # Compute forecast by multiplying polynomial coefficients with the basis.
        forecast = tf.matmul(theta, self.basis, transpose_b=True)  # (batch, forecast_horizon)
        return forecast

    def get_config(self):
        config = super().get_config()
        config.update({
            "degree": self.degree,
            "forecast_horizon": self.forecast_horizon,
        })
        return config

# =============================================================================
# 3. Physics Strategy Factory
# =============================================================================
def physics_strategy_factory(strategy_name, params):
    if strategy_name == 'exponential':
        return ExponentialDecayStrategy(params['P_reservoir'], params['decay_rate'])
    elif strategy_name == 'arps':
        return ArpsDeclineStrategy(params['P_reservoir'], params['decay_rate'], params['b_factor'])
    elif strategy_name == 'static':
        return StaticPressureStrategy(params['P_reservoir'])
    elif strategy_name == 'weighted_ensemble':
        return WeightedEnsembleStrategy(params['P_reservoir'], params['decay_rate'], params.get('alpha_init', 0.5))
    elif strategy_name == 'combined_exp_arps':
        return CombinedExpArpsStrategy(params['P_reservoir'], params['decay_rate'], params['b_factor'])
    elif strategy_name == 'pressure_ensemble':
        return DynamicEnsembleStrategy(params['P_reservoir'], params['decay_rate'], params['absolute_value'])
    elif strategy_name == 'diffusivity_decay':
        return DiffusivityDecayStrategy(params['P_reservoir'], kappa = 0.1)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# =============================================================================
# 4. DarcyPhysicsLayer 
# =============================================================================
from forecast_pipeline.config import DEFAULT_DATASET, INITIAL_PRESSURE, kpa2psi
class DarcyPhysicsLayer(tf.keras.layers.Layer):
    """
    Physics layer that delegates physics-based computation to a selected strategy.
    """
    def __init__(self, scaler_X_path, scaler_target_path, strategy_config, **kwargs):
        super().__init__(**kwargs)
        self.scaler_X_path = scaler_X_path
        self.scaler_target_path = scaler_target_path
        self.strategy_config = strategy_config
        self.iteration = tf.Variable(0, trainable=False, dtype=tf.int32)

    def build(self, input_shape):
        # Load scalers (assumed implemented elsewhere)
        self.scaler_X = load_scaler(self.scaler_X_path)
        self.scaler_target = load_scaler(self.scaler_target_path)
        
        # 2) Extrai centro e escala em NumPy (para inicialização de pesos)
        self.scaler_X_center_np, self.scaler_X_scale_np = get_center_and_scale(self.scaler_X, as_tf=False)
        self.oil_center_np,      self.oil_scale_np      = get_center_and_scale(self.scaler_target, as_tf=False)

        # 3) Cria também versões TensorFlow para uso em call()
        self.scaler_X_mean, self.scaler_X_std = get_center_and_scale(self.scaler_X, as_tf=True, dtype=tf.float32)
        self.oil_mean,      self.oil_std      = get_center_and_scale(self.scaler_target, as_tf=True, dtype=tf.float32)

        # Verify indices: PI index = 0, AVG_DOWNHOLE_PRESSURE index = 3.
        self.PI_idx = 0
        self.P_wf_idx = 3
        
        self._init_physics()
        super().build(input_shape)
        
    def _init_physics(self):
        """Create physics weights and instantiate the chosen strategy using add_weight."""
        # 1) Read chosen strategy
        strat = self.strategy_config["strategy_name"]

        # 2) For each strategy, declare which weights are trainable
        train_map = {
            'exponential':       {'P_reservoir': True, 'decay_rate': True,  'b_factor': True},
            'arps':              {'P_reservoir': True, 'decay_rate': True,  'b_factor': True },
            'static':            {'P_reservoir': False,'decay_rate': False, 'b_factor': False },
            'combined_exp_arps': {'P_reservoir': False, 'decay_rate': True,  'b_factor': True },
            'pressure_ensemble': {'P_reservoir': True, 'decay_rate': True,  'b_factor': False},
            'diffusivity_decay': {'P_reservoir': True, 'decay_rate': False, 'b_factor': False},
        }
        flags = train_map[strat]

        if DEFAULT_DATASET == "VOLVE":
            init_P = float(self.scaler_X_center_np[self.P_wf_idx])*1.1
        if DEFAULT_DATASET == "UNISIM_IV":
            init_P = kpa2psi(INITIAL_PRESSURE["UNISIM-IV-2024"])

        init_d   = 0.01
        init_b   = 0.5

        # 4) Define all weight specs in one place
        param_defs = {
            'P_reservoir': {
                'initializer': tf.keras.initializers.Constant(init_P),
                'constraint':  None,
                'trainable':   flags['P_reservoir']
            },
            'decay_rate': {
                'initializer': tf.keras.initializers.Constant(init_d),
                'constraint':  lambda x: tf.clip_by_value(x, 0.001, 5),
                'trainable':   flags['decay_rate']
            },
            'b_factor': {
                'initializer': tf.keras.initializers.Constant(init_b),
                'constraint':  lambda x: tf.clip_by_value(x, 0.001, 5),
                'trainable':   flags['b_factor']
            },
        }

        # 5) Create each weight via add_weight in a single loop
        for var_name, spec in param_defs.items():
            w = self.add_weight(
                name=var_name,
                shape=(),
                initializer=spec['initializer'],
                trainable=spec['trainable'],
                constraint=spec['constraint']
            )
            setattr(self, var_name, w)

        # 6) Build params dict for strategy factory
        params = {
            "P_reservoir": self.P_reservoir,
            "decay_rate":  self.decay_rate,
            "b_factor":    self.b_factor,
            "absolute_value": True
        }

        # 7) Instantiate the physics strategy
        self.physics_strategy = physics_strategy_factory(strat, params)

        
    def diagnostic(self, PI_measured, P_wf, Q_phys_base, t_feature, loss=None):
        """Imprime estatísticas da iteração + parâmetros físicos."""
        P_res_t = self.P_reservoir * tf.exp(-self.decay_rate * t_feature)
    
        tf.print("\n=== DarcyPhysicsLayer Diagnostics @ iter", self.iteration, "===")
        tf.print("Strategy: ", self.strategy_config)
        tf.print(" P_reservoir         :", self.P_reservoir)
        tf.print(" decay_rate          :", self.decay_rate)
        if hasattr(self, "b_factor"):
            tf.print(" b_factor            :", self.b_factor)
        tf.print(" ΔP mean             :", tf.reduce_mean(P_res_t - P_wf))
        tf.print(" PI_measured mean    :", tf.reduce_mean(PI_measured))
        tf.print(" P_wf mean           :", tf.reduce_mean(P_wf))
        tf.print(" Q_phys_base mean    :", tf.reduce_mean(Q_phys_base))
        if loss is not None:
            tf.print(" batch loss          :", loss)
        tf.print("=== End ===\n")


        
    def call(self, inputs, training=False):
        
        # Expected input shape: (batch, horizon, 4) with features [PI, P_scaled, t_feature, ...]
        PI_scaled = inputs[..., 0]
        P_scaled = inputs[..., 1]
        t_feature = inputs[..., 2]

        # Invert scaling using your defined utility (assumed available)
        PI_measured = invert_feature_scaling(PI_scaled, self.scaler_X_mean, self.scaler_X_std, self.PI_idx)
        P_wf = invert_feature_scaling(P_scaled, self.scaler_X_mean, self.scaler_X_std, self.P_wf_idx)

        # Delegate physics computation to the chosen strategy
        Q_phys_base = self.physics_strategy.compute_Q_phys(PI_measured, P_wf, t_feature)

        # Scale the physics forecast back to normalized values (using oil scaler)
        physics_prediction_scaled = (Q_phys_base - self.oil_mean) / self.oil_std
        
        if tf.equal(tf.math.mod(self.iteration, 5000), 0):
            self.diagnostic(PI_measured, P_wf, Q_phys_base, t_feature)
        self.iteration.assign_add(1)
        
        return physics_prediction_scaled

    def get_config(self):
        config = super().get_config()
        config.update({
            "scaler_X_path": self.scaler_X_path,
            "scaler_target_path": self.scaler_target_path,
            "strategy_config": self.strategy_config,
        })
        return config


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

    if fusion_type.lower() == 'pin':
        model = Model(inputs, physics_forecast, name="physics_block")
        tf.print("Created PIN-only model.")
        return model

    # -- Hybrid Models ---
    if fusion_type.lower() == "concat_dense":
        combined = Concatenate(name='fusion_concat')([trend_forecast, physics_forecast])
        final_forecast = Dense(horizon, activation='linear', name=fusion_type)(combined)
    model = Model(inputs, final_forecast, name=fusion_type)
    return model