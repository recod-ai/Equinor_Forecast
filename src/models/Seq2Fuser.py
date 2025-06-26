# Standard library imports
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import joblib  # or pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, initializers, Input, layers, Model, models
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    GlobalAveragePooling1D,
    GRU,
    LSTM,
    Lambda,
    TimeDistributed,
)

# Local application imports
from forecast_pipeline.config import DEFAULT_DATASET
from models.PINNs import (
    AggregateContextExtractor,
    ArpsDeclineStrategy,
    BiasScaleContextFuser,
    CNNContextExtractor,
    CombinedExpArpsStrategy,
    ConcatContextFuser,
    DynamicEnsembleStrategy,
    ExponentialDecayStrategy,
    FiLMContextFuser,
    GatingContextFuser,
    IdentityContextExtractor,
    ProbabilisticBiasScaleFuser,
    RNNContextExtractor,
    StaticPressureStrategy,
    TCNContextExtractor,
    WeightedEnsembleStrategy,
)
from models.filter_layers import PolynomialSmoothingLayer, WaveletDenoiseLayer
from utils.utilities import get_center_and_scale, invert_feature_scaling, load_scaler



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



def create_context_extractor(config: dict):
    """Factory to create context extractor based on config."""
    cfg = config.copy()
    extractor_type = cfg.pop('type', '').lower()
    name = cfg.get('name', f"{extractor_type}_context_extractor")
    
    if extractor_type == 'tcn':
        defaults = {
            'tcn_filters': 32, 'tcn_kernel_size': 3, 'tcn_dilations': (1, 2, 4, 8),
            'tcn_stacks': 1, 'tcn_activation': 'relu', 'tcn_normalization': 'layer',
            'dropout_rate': 0.1
        }
        params = {k: cfg.get(k, v) for k, v in defaults.items()}
        return TCNContextExtractor(name=name, **params)

    if extractor_type == 'cnn':
        defaults = {
            'cnn_filters': 32, 'cnn_kernel_size': 3, 'cnn_layers': 1,
            'cnn_activation': 'relu', 'cnn_pooling': 'global_avg',
            'dropout_rate': 0.1
        }
        params = {k: cfg.get(k, v) for k, v in defaults.items()}
        return CNNContextExtractor(name=name, **params)

    if extractor_type == 'rnn':
        defaults = {
            'rnn_units': 32, 'rnn_layers': 1, 'rnn_type': 'lstm',
            'bidirectional': False, 'dropout_rate': 0.1
        }
        params = {k: cfg.get(k, v) for k, v in defaults.items()}
        return RNNContextExtractor(name=name, **params)

    if extractor_type == 'aggregate':
        indices = tuple(cfg.get('feature_indices', (0, 1)))
        return AggregateContextExtractor(name=name, feature_indices=indices)

    if extractor_type == 'identity':
        mode = cfg.get('mode', 'last_step')
        return IdentityContextExtractor(name=name, mode=mode)

    raise ValueError(f"Unknown extractor type: '{extractor_type}'")


def create_context_fuser(config: dict, forecast_horizon: int):
    """Factory to create context fuser based on config."""
    cfg = config.copy()
    fuser_type = cfg.pop('type', '').lower()
    name = cfg.get('name', f"{fuser_type}_context_fuser")

    # Shared defaults
    defaults = {
        'residual_hidden_units': 64,
        'dropout_rate': 0.1
    }
    shared = {k: cfg.get(k, v) for k, v in defaults.items()}

    if fuser_type == 'film':
        return FiLMContextFuser(name=name, forecast_horizon=forecast_horizon, **shared)

    if fuser_type == 'gating':
        gate_act = cfg.get('gate_activation', 'sigmoid')
        return GatingContextFuser(
            name=name, forecast_horizon=forecast_horizon,
            gate_activation=gate_act, **shared
        )

    if fuser_type == 'bias_scale':
        scale_act = cfg.get('scale_activation', 'linear')
        bias_act = cfg.get('bias_activation', 'linear')
        return BiasScaleContextFuser(
            name=name, forecast_horizon=forecast_horizon,
            scale_activation=scale_act, bias_activation=bias_act, **shared
        )

    if fuser_type == 'concat':
        return ConcatContextFuser(
            name=name, forecast_horizon=forecast_horizon, **shared
        )

    if fuser_type in {'none', 'no_context'}:
        return NoContextFuser(
            name=name, forecast_horizon=forecast_horizon, **shared
        )

    raise ValueError(f"Unknown fuser type: '{fuser_type}'")


# =============================================================================
# 1. PhysicsBaselineLayer 
# =============================================================================
class PhysicsBaselineLayer(tf.keras.layers.Layer):
    def __init__(self,
                 scaler_x_path,
                 scaler_y_path,
                 strategy_cfg, # Será copiado
                 horizon,
                 local_pi_idx, local_p0_idx, local_t_idx,
                 global_pi_idx, global_p0_idx, global_t_idx,
                 global_pwf_scaler_idx,
                 name='physics_baseline_block', **kwargs):
        super().__init__(name=name, **kwargs)
        self.scaler_x_path = str(scaler_x_path)
        self.scaler_y_path = str(scaler_y_path)
        # 1. Guarde uma cópia de strategy_cfg
        self.strategy_cfg = dict(strategy_cfg) if strategy_cfg else {"strategy_name": "static"}
        self.horizon = horizon

        self.local_pi_idx = local_pi_idx
        self.local_p0_idx = local_p0_idx
        self.local_t_idx = local_t_idx

        self.global_pi_idx = global_pi_idx
        self.global_p0_idx = global_p0_idx
        self.global_t_idx = global_t_idx
        
        self.global_pwf_scaler_idx = global_pwf_scaler_idx

        self._load_scalers()
        self._init_physics_parameters()

    def _load_scalers(self): # Sem alterações aqui
        sx = load_scaler(self.scaler_x_path)
        sy = load_scaler(self.scaler_y_path)
        self.x_mean = tf.constant(sx.mean_, tf.float32)
        self.x_std = tf.constant(sx.scale_, tf.float32)
        self.y_mean = tf.constant(sy.mean_[0], tf.float32) 
        self.y_std = tf.constant(sy.scale_[0], tf.float32)
        if self.global_pwf_scaler_idx >= sx.mean_.shape[0]:
            raise ValueError(f"global_pwf_scaler_idx {self.global_pwf_scaler_idx} out of range for scaler X.")

    def _init_physics_parameters(self): # Sem alterações aqui, mas ponto 2 da tabela é sobre o factory
        strat_name = self.strategy_cfg.get("strategy_name", "static")
        # ... (resto do código de _init_physics_parameters igual)
        train_map = {
            'exponential':       {'p_reservoir': False, 'decay': True,  'b': False, 'kappa': False},
            'arps':              {'p_reservoir': False, 'decay': True,  'b': True,  'kappa': False},
            'static':            {'p_reservoir': True,  'decay': False, 'b': False, 'kappa': False},
            'weighted_ensemble': {'p_reservoir': False, 'decay': True,  'b': False, 'kappa': False},
            'combined_exp_arps': {'p_reservoir': False, 'decay': True,  'b': True,  'kappa': False},
            'pressure_ensemble': {'p_reservoir': False, 'decay': True,  'b': False, 'kappa': False},
            'diffusivity_decay': {'p_reservoir': False, 'decay': False, 'b': False, 'kappa': True},
        }
        flags = train_map.get(strat_name)
        if flags is None: flags = train_map['static']

        init_p_res = float(self.x_mean[self.global_pwf_scaler_idx])
        init_decay = self.strategy_cfg.get("initial_decay_rate", 0.01)
        init_b = self.strategy_cfg.get("initial_b_factor", 0.5)
        init_kappa = self.strategy_cfg.get("initial_kappa", 0.1)

        param_defs = {
            'p_reservoir_phys': {'init': init_p_res, 'train': flags['p_reservoir'], 'constr': None},
            'decay_rate_phys': {'init': init_decay, 'train': flags['decay'], 'constr': lambda x: tf.clip_by_value(x, 0.0001, 5.0)},
            'b_factor_phys': {'init': init_b, 'train': flags['b'], 'constr': lambda x: tf.clip_by_value(x, 0.01, 2.0)},
            'kappa_phys': {'init': init_kappa, 'train': flags['kappa'], 'constr': lambda x: tf.clip_by_value(x, 0.0001, 5.0)}
        }
        
        for param_name, spec in param_defs.items():
            setattr(self, param_name, self.add_weight(
                name=param_name, shape=(), initializer=tf.keras.initializers.Constant(spec['init']),
                trainable=spec['train'], constraint=spec['constr']
            ))

        strategy_params = { # Estes são todos os params que podem ser usados
            "P_reservoir": self.p_reservoir_phys,
            "decay_rate": self.decay_rate_phys,
            "b_factor": self.b_factor_phys,
            "kappa": self.kappa_phys,
            "absolute_value": self.strategy_cfg.get("absolute_value", True), # Para DynamicEnsemble
            "alpha_init": self.strategy_cfg.get("alpha_init", 0.5) # Para WeightedEnsemble
        }
        # 2. O factory precisa ignorar chaves extras. Se o __init__ das strategies aceitar **kwargs, está ok.
        self.strategy = physics_strategy_factory(strat_name, strategy_params)


    def call(self, x_phys_features): # Ponto 3 e 4 da tabela
        last_step_features = x_phys_features[:, -1, :]

        pi_scaled = last_step_features[:, self.local_pi_idx : self.local_pi_idx + 1]
        p0_scaled = last_step_features[:, self.local_p0_idx : self.local_p0_idx + 1]
        t0_scaled = last_step_features[:, self.local_t_idx : self.local_t_idx + 1]

        PI_unscaled = invert_feature_scaling(pi_scaled, self.x_mean, self.x_std, self.global_pi_idx)
        P0_unscaled = invert_feature_scaling(p0_scaled, self.x_mean, self.x_std, self.global_p0_idx)
        t0_unscaled_val = invert_feature_scaling(t0_scaled, self.x_mean, self.x_std, self.global_t_idx)
        
        # 4. Opcional: tf.round() para t0_unscaled_val se for tempo em dias inteiros, por exemplo.
        # t0_unscaled_val = tf.round(t0_unscaled_val) # Exemplo

        time_steps_for_forecast = tf.range(1, self.horizon + 1, dtype=tf.float32) # Shape (H,)
        # 3. Broadcast em compute_Q_phys:
        #    PI_unscaled (B,1), P0_unscaled (B,1), time_steps_for_forecast (H,)
        #    A estratégia deve lidar com isso para produzir (B,H).
        #    Ex: Se P_res_t = self.P_reservoir * tf.exp(-self.decay_rate * time_steps_for_forecast)
        #        P_res_t terá shape (H,). Então (P_res_t - P0_unscaled) será (B,H) por broadcast.
        #        PI_unscaled * (P_res_t - P0_unscaled) -> (B,1) * (B,H) -> (B,H)
        #    Isso parece correto e é o comportamento padrão do TensorFlow/NumPy.
        q_phys_base = self.strategy.compute_Q_phys(PI_unscaled, P0_unscaled, time_steps_for_forecast)
        
        # Checagem de shape para q_phys_base (deve ser (B, H))
        # tf.debugging.assert_shapes([(q_phys_base, (None, self.horizon))], message="q_phys_base shape error")

        forecast_times_abs = t0_unscaled_val + tf.expand_dims(time_steps_for_forecast, 0)
        q_phys_scaled = (q_phys_base - self.y_mean) / self.y_std

        per_step_info = {
            'q_phys_scaled': q_phys_scaled,
            'forecast_times': forecast_times_abs,
            'pi_last_measured': PI_unscaled,
            'p_wf_last': P0_unscaled,
            't_last': t0_unscaled_val # Passando o valor (possivelmente arredondado)
        }
        return q_phys_scaled, per_step_info

    def get_config(self): # Sem alterações aqui
        config = super().get_config()
        config.update({
            'scaler_x_path': self.scaler_x_path,
            'scaler_y_path': self.scaler_y_path,
            'strategy_cfg': self.strategy_cfg,
            'horizon': self.horizon,
            'local_pi_idx': self.local_pi_idx, 'local_p0_idx': self.local_p0_idx, 'local_t_idx': self.local_t_idx,
            'global_pi_idx': self.global_pi_idx, 'global_p0_idx': self.global_p0_idx, 'global_t_idx': self.global_t_idx,
            'global_pwf_scaler_idx': self.global_pwf_scaler_idx,
        })
        return config

# =============================================================================
# 1.5. CombinationLayerAlpha (COM AJUSTE DE NOME)
# =============================================================================
class CombinationLayerAlpha(tf.keras.layers.Layer):
    # Sugestão: Renomear para CombinerBlock ou similar se for consistência com outros *_block
    def __init__(self, horizon, min_alpha=0.3, max_alpha=0.7, name='combiner_block', **kwargs): # Nome ajustado
        super().__init__(name=name, **kwargs)
        self.horizon = horizon
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        # Ajuste para inicializar alpha_raw para que o sigmoid(alpha_raw) seja 0.5,
        # resultando em alpha = (min_alpha + max_alpha) / 2
        # Se sigmoid(target) = (target_alpha - min_alpha) / (max_alpha - min_alpha)
        # Para target_alpha = (min_alpha + max_alpha) / 2:
        # sigmoid_target = [((min_alpha + max_alpha) / 2) - min_alpha] / (max_alpha - min_alpha)
        #                = [(max_alpha - min_alpha) / 2] / (max_alpha - min_alpha)
        #                = 0.5
        # Para sigmoid(alpha_raw_init_value) = 0.5, alpha_raw_init_value deve ser 0.
        self.alpha_raw_init_value = 0.0

    def build(self, input_shape): # Sem alterações aqui
        self.alpha_raw = self.add_weight(
            name=f"{self.name}_alpha_raw_weight", # Nome do peso mais específico
            shape=(self.horizon,),
            initializer=tf.keras.initializers.Constant(self.alpha_raw_init_value),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs): # Sem alterações aqui
        q_phys_scaled, residual_scaled = inputs
        range_alpha = self.max_alpha - self.min_alpha
        alpha_vec = range_alpha * tf.sigmoid(self.alpha_raw) + self.min_alpha
        alpha_vec = tf.reshape(alpha_vec, (1, self.horizon)) # Garante broadcast (B,H)
        
        # Checagem de shapes para garantir broadcast correto
        # q_phys_scaled: (B,H), residual_scaled: (B,H), alpha_vec: (1,H)
        # (1,H) * (B,H) -> (B,H)
        # (1,H) * (B,H) -> (B,H)
        y_hat = alpha_vec * q_phys_scaled + (1.0 - alpha_vec) * residual_scaled
        return y_hat, alpha_vec

    def get_config(self): # Sem alterações aqui
        config = super().get_config()
        config.update({'horizon': self.horizon, 'min_alpha': self.min_alpha, 'max_alpha': self.max_alpha})
        return config

# =============================================================================
# 2. build_three_stage_model
# =============================================================================
from pathlib import Path
from typing import Dict, List, Union

import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.layers import Lambda

def _get_local_idx(global_idx: int, indices: List[int], name: str) -> int:
    try:
        return indices.index(global_idx)
    except ValueError:
        raise ValueError(
            f"Global index {global_idx} for '{name}' not found in {indices}"
        )

def create_model(
    input_shape: tuple,  # (num_samples, timesteps, features)
    horizon: int,
    strategy_config: dict,
    extractor_config: dict,
    fuser_config: dict,
    scaler_X_path: Union[str, Path] = 'scalers/scaler_X.pkl',
    scaler_Y_path: Union[str, Path] = 'scalers/scaler_target.pkl',
    architecture_name: str = 'HybridThreeStageModel',
    output_mode: str = 'single_with_metrics',  # 'dict' or 'single_with_metrics'
    global_pi_idx: int = 0,
    global_pwf_idx: int = 3,
    global_time_idx: int = 5,
    data_features_indices: List[int] = [0, 2, 7],
    physics_features_indices: List[int] = [0, 3, 5, 6],
    add_extractor_head: bool = False,
) -> Model:
    timesteps, feats = input_shape.shape[1], input_shape.shape[2]
    inputs = Input(shape=(timesteps, feats), name='all_features')

    def gather(x, idx, name):
        return Lambda(lambda t: tf.gather(t, idx, axis=-1), name=name)(x)

    data_x = gather(inputs, data_features_indices, 'data_features')
    phys_x = gather(inputs, physics_features_indices, 'physics_features')

    # find the local positions of global indices
    local_pi = _get_local_idx(global_pi_idx, physics_features_indices, 'pi')
    local_p0 = _get_local_idx(global_pwf_idx, physics_features_indices, 'pwf_for_p0')
    local_t  = _get_local_idx(global_time_idx, physics_features_indices, 'time')

    # physics branch
    phys_layer = PhysicsBaselineLayer(
        scaler_X_path, scaler_Y_path, strategy_config, horizon,
        local_pi_idx=local_pi, global_pi_idx=global_pi_idx,
        local_p0_idx=local_p0,   global_p0_idx=global_pwf_idx,
        local_t_idx=local_t,      global_t_idx=global_time_idx,
        global_pwf_scaler_idx=global_pwf_idx,
        name='physics_block'
    )
    q_phys_s, per_step = phys_layer(phys_x)
    q_phys_s = layers.Activation('linear', name='physics_scaled_output')(q_phys_s)

    # extractor branch
    extractor = create_context_extractor(extractor_config)
    if not extractor.name.endswith('_block'):
        extractor._name = f"{extractor.name or 'context_extractor'}_block"
    ctx_vec = extractor(data_x)
    ctx_vec = layers.Activation('linear', name='context_vector_output')(ctx_vec)

    # fuser branch
    fuser = create_context_fuser(fuser_config, horizon)
    if not fuser.name.endswith('_block'):
        fuser._name = f"{fuser.name or 'context_fuser'}_block"
    residual = fuser(ctx_vec, per_step)
    residual = layers.Activation('linear', name='residual_scaled_output')(residual)

    # combine physics + residual
    combiner = CombinationLayerAlpha(horizon, min_alpha=0.3, max_alpha=0.7)
    y_hat, alpha_v = combiner([q_phys_s, residual])
    y_hat   = layers.Activation('linear', name='final_forecast_output')(y_hat)
    alpha_v = layers.Activation('linear', name='alpha_values_output')(alpha_v)

    # build outputs
    if output_mode == 'dict':
        outputs = {
            'final_forecast':    y_hat,
            'physics_scaled':    q_phys_s,
            'residual_scaled':   residual,
            'context_vector':    ctx_vec,
            'alpha_values':      alpha_v,
        }
        if add_extractor_head:
            head = layers.Dense(horizon, name='extractor_head_dense_block')
            outputs['extractor_head_output'] = head(ctx_vec)
    elif output_mode == 'single_with_metrics':
        outputs = y_hat
    else:
        raise ValueError("output_mode must be 'dict' or 'single_with_metrics'")

    model = Model(inputs=inputs, outputs=outputs, name=architecture_name)

    # attach blocks for easy access
    model.physics_block   = phys_layer
    model.extractor_block = extractor
    model.fuser_block     = fuser
    model.combiner_block  = combiner
    if add_extractor_head and output_mode == 'dict':
        model.extractor_head_dense_block = model.get_layer('extractor_head_dense_block')

    # optional metrics on single-output mode
    if output_mode == 'single_with_metrics':
        model.add_metric(q_phys_s, name='metric_physics_scaled', aggregation='mean')
        model.add_metric(residual, name='metric_residual_scaled', aggregation='mean')
        model.add_metric(alpha_v, name='metric_alpha_values_mean', aggregation='mean')

    return model










