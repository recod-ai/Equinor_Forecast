import os
import joblib
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras.layers import Input, Lambda, Dense, GlobalAveragePooling1D, Concatenate, BatchNormalization, LSTM, GRU, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from models.filter_layers import WaveletDenoiseLayer, PolynomialSmoothingLayer

from tensorflow.keras import layers, activations, initializers, models
import joblib # Or pickle


from typing import Any, Dict, Optional, Union, Tuple
from pathlib import Path

from utils.utilities import load_scaler, invert_feature_scaling, get_center_and_scale

from models.PINNs import (
    DynamicEnsembleStrategy,
    ExponentialDecayStrategy,
    ArpsDeclineStrategy,
    WeightedEnsembleStrategy,
    CombinedExpArpsStrategy,
    StaticPressureStrategy,
    FiLMContextFuser,
    ProbabilisticBiasScaleFuser,
    BiasScaleContextFuser,
    GatingContextFuser,
    ConcatContextFuser,
    TCNContextExtractor,
    AggregateContextExtractor,
    CNNContextExtractor,
    RNNContextExtractor,
    IdentityContextExtractor,
)


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
            'tcn_filters': 64, 'tcn_kernel_size': 3, 'tcn_dilations': (1, 2, 4),
            'tcn_stacks': 1, 'tcn_activation': 'relu', 'tcn_normalization': 'layer',
            'dropout_rate': 0
        }
        params = {k: cfg.get(k, v) for k, v in defaults.items()}
        return TCNContextExtractor(name=name, **params)

    if extractor_type == 'cnn':
        defaults = {
            'cnn_filters': 64, 'cnn_kernel_size': 3, 'cnn_layers': 3,
            'cnn_activation': 'relu', 'cnn_pooling': 'global_avg',
            'dropout_rate': 0
        }
        params = {k: cfg.get(k, v) for k, v in defaults.items()}
        return CNNContextExtractor(name=name, **params)

    if extractor_type == 'rnn':
        defaults = {
            'rnn_units': 64, 'rnn_layers': 3, 'rnn_type': 'lstm',
            'bidirectional': False, 'dropout_rate': 0
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

    if fuser_type == 'prob_bias':
        return ProbabilisticBiasScaleFuser(
            name=name, forecast_horizon=forecast_horizon, **shared
        )

    if fuser_type == 'concat':
        return ConcatContextFuser(
            name=name, forecast_horizon=forecast_horizon, **shared
        )

    if fuser_type == 'trend_context':
        return TrendContextFuser(
            forecast_horizon=forecast_horizon,
            **shared
        )
    
    raise ValueError(f"Unknown fuser type: '{fuser_type}'")



# ---------------------------------------
# utils.py
def gaussian_kernel(window, sigma=1.0):
    t = tf.range(-(window//2), window//2 + 1, dtype=tf.float32)
    k = tf.exp(-0.5 * (t/sigma)**2)
    k /= tf.reduce_sum(k)
    return k  # shape (w,)

# ---------------------------------------
# layers.py
class ResidualSmoother(layers.Layer):
    def __init__(self, ksize=5, sigma=1.2, **kw):
        super().__init__(**kw)
        k = gaussian_kernel(ksize, sigma)              # (ksize,)
        #  ➜ (ksize, 1, 1, 1):  H_k × W_k × in_ch × ch_mult
        self.kernel = k[:, None, None, None]           # <- FOCO
        self.ksize = ksize

    def call(self, res):                               # res (B,H)
        r = tf.expand_dims(res, -1)                    # (B,H,1)
        r = tf.expand_dims(r, 1)                       # (B,1,H,1)  NHWC
        r = tf.nn.depthwise_conv2d(
            r, self.kernel,
            strides=[1, 1, 1, 1], padding="SAME")
        return tf.squeeze(r, [1, 3])                   # (B,H)



from tensorflow.keras.constraints import MinMaxNorm
from scipy.stats import norm
from forecast_pipeline.config import DEFAULT_DATASET, kpa2psi, INITIAL_PRESSURE
# =============================================================================
# 5. Main Modular Forecaster Layer
# =============================================================================
class PhysicsInformedForecaster(layers.Layer):
    """
    Physics-Informed forecaster: projects physics baseline and learns residuals.
    """
    def __init__(
        self,
        scaler_x_path,
        scaler_y_path,
        strategy_cfg,
        fuser_cfg: dict,
        horizon: int = 30,
        pi_idx: int = 0,
        p_idx: int = 1,
        t_idx: int = 2,
        pwf_idx: int = 3,
        return_branches: bool = True,
        name: str = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        # config
        self.scaler_x_path = scaler_x_path
        self.scaler_y_path = scaler_y_path
        self.strategy_cfg = strategy_cfg
        self.fuser_cfg = fuser_cfg
        self.horizon = horizon
        self.return_branches = return_branches
        # feature indices
        self.pi_idx = pi_idx
        self.p_idx = p_idx
        self.t_idx = t_idx
        self.pwf_idx = pwf_idx

        self.residual_magnitude_weight = 0.1
        self.residual_smoothness_weight = 0.01
        
        tf.print('strategy_config', self.strategy_cfg)
        tf.print('self.fuser_config', self.fuser_cfg)

        self._load_scalers()
        self.iter = tf.Variable(0, trainable=False, dtype=tf.int64)

    def _load_scalers(self):
        """Load scalers and store mean/std (center/scale) both as NumPy and as tf.Tensor."""
        try:
            sx = load_scaler(self.scaler_x_path)
            sy = load_scaler(self.scaler_y_path)

            # 1) Extract NumPy center/scale for scaler_x and scaler_y
            self.scaler_x_center_np, self.scaler_x_scale_np = get_center_and_scale(sx, as_tf=False)
            self.scaler_y_center_np, self.scaler_y_scale_np = get_center_and_scale(sy, as_tf=False)

            # 2) Extract TF versions of center/scale
            self.x_mean, self.x_std = get_center_and_scale(sx, as_tf=True, dtype=tf.float32)
            self.y_mean, self.y_std = get_center_and_scale(sy, as_tf=True, dtype=tf.float32)

            # 3) Ensure pwf_idx is valid
            if self.pwf_idx >= self.scaler_x_center_np.shape[0]:
                raise ValueError(f"pwf_idx {self.pwf_idx} out of range (got {self.pwf_idx}, "
                                 f"but scaler_x has dimension {self.scaler_x_center_np.shape[0]})")
        except Exception as e:
            raise RuntimeError(f"Scaler init error: {e}")

    def build(self, input_shape):
        """Initialize physics weights and fusion module."""
        if not (isinstance(input_shape, (list, tuple)) and len(input_shape) == 2):
            raise ValueError("build expects [phys_shape, ctx_shape]")
        phys_shape, _ = input_shape
        if max(self.pi_idx, self.p_idx, self.t_idx) >= phys_shape[-1]:
            raise ValueError("feature index out of bounds")

        self._init_physics()
        self.fuser = create_context_fuser(self.fuser_cfg.copy(), self.horizon)
        

        # 1) Camada que gera um vetor α[horizon] a partir das mesmas variáveis per-step
        self.alpha_raw = self.add_weight(
            "alpha_raw", shape=(self.horizon,),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True
        )
        
        
        super().build(phys_shape)

    def _init_physics(self):
        """Create physics weights and instantiate the chosen strategy."""
        strat = self.strategy_cfg.get("strategy_name", "darcy")

        # 1) Map each strategy to which weights should be trainable
        train_map = {
            'exponential':       {'p_reservoir': True, 'decay': True,  'b': True},
            'arps':              {'p_reservoir': True, 'decay': True,  'b': True },
            'static':            {'p_reservoir': False,'decay': False, 'b': False },
            'combined_exp_arps': {'p_reservoir': False,'decay': True,  'b': True },
            'pressure_ensemble': {'p_reservoir': True, 'decay': True,  'b': False},
            'diffusivity_decay': {'p_reservoir': True, 'decay': False, 'b': False},
        }
        flags = train_map[strat]

        # 2) Prepare initial values
        if DEFAULT_DATASET == "VOLVE":
            init_P = float(self.scaler_x_center_np[self.pwf_idx])*1.1
        if DEFAULT_DATASET == "UNISIM_IV":
            init_P = kpa2psi(INITIAL_PRESSURE["UNISIM-IV-2024"])
        init_decay = 0.01
        init_b = 0.5

        # 3) Define each weight in one place: (init, constraint_fn, trainable_flag)
        param_defs = {
            'p_reservoir': {
                'initializer': initializers.Constant(init_P),
                'constraint': None,
                'trainable': flags['p_reservoir']
            },
            'decay': {
                'initializer': initializers.Constant(init_decay),
                'constraint': lambda x: tf.clip_by_value(x, 0.001, 10),
                'trainable': flags['decay']
            },
            'b': {
                'initializer': initializers.Constant(init_b),
                'constraint': lambda x: tf.clip_by_value(x, 0.001, 5),
                'trainable': flags['b']
            },
        }

        # 4) Loop over param_defs to call add_weight only once
        for name, spec in param_defs.items():
            w = self.add_weight(
                name,
                shape=(),
                initializer=spec['initializer'],
                trainable=spec['trainable'],
                constraint=spec['constraint']
            )
            setattr(self, name, w)

        # 5) Build params dict and factory the strategy
        params = {
            "P_reservoir": self.p_reservoir,
            "decay_rate":  self.decay,
            "b_factor":    self.b,
            "absolute_value": True
        }
        self.strategy = physics_strategy_factory(strat, params)

    def _project_baseline(self, x):
        """Compute scaled baseline and vars for fusion."""
        batch = tf.shape(x)[0]
        last = x[:, -1, :]
        pi = last[:, self.pi_idx:self.pi_idx + 1]
        p = last[:, self.p_idx:self.p_idx + 1]
        t0 = last[:, self.t_idx:self.t_idx + 1]

        PI = invert_feature_scaling(pi, self.x_mean, self.x_std, self.pi_idx)
        P0 = invert_feature_scaling(p, self.x_mean, self.x_std, self.pwf_idx)

        # time offsets for forecast horizon
        steps = tf.range(1, self.horizon + 1, dtype=tf.float32)
        # broadcast adds offsets to each batch row without extra tiling
        times = t0 + tf.expand_dims(steps, 0)

        # physics output
        Q_base = self.strategy.compute_Q_phys(PI, P0, steps)
        Q_scaled = (Q_base - self.y_mean) / self.y_std
        return Q_scaled, Q_base, times, PI, P0, t0

    def _add_residual_regularization(self, residual_tensor):
        """Calculates and adds regularization losses based on the residual tensor."""
        # A loss só é adicionada se o peso for maior que zero, evitando cálculos desnecessários.
        if self.residual_magnitude_weight > 0:
            # Penaliza a magnitude L2 do resíduo para incentivá-lo a ser pequeno.
            magnitude_loss = tf.reduce_mean(tf.square(residual_tensor))
            self.add_loss(self.residual_magnitude_weight * magnitude_loss)
            self.add_metric(magnitude_loss, name='residual_magnitude_loss') # Opcional: para monitorar

        if self.residual_smoothness_weight > 0:
            # Penaliza a diferença entre passos de tempo adjacentes para forçar a suavidade.
            # Esta é a regularização mais importante para estabilidade.
            smoothness_loss = tf.reduce_mean(tf.square(
                residual_tensor[:, 1:] - residual_tensor[:, :-1]
            ))
            self.add_loss(self.residual_smoothness_weight * smoothness_loss)
            self.add_metric(smoothness_loss, name='residual_smoothness_loss') # Opcional: para monitorar

    def call(self, inputs, training=False):
        """Forward pass returning forecast and optional branches."""
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 2):
            raise ValueError("call expects [phys_inputs, context]")
        x, ctx = inputs

        
        Qs, Qb, times, PI, P0, t0 = self._project_baseline(x)
        # times_norm = (times - tf.reduce_mean(times, axis=1, keepdims=True)) / tf.math.reduce_std(times, axis=1, keepdims=True)
        per_step = {
            'q_phys_scaled': Qs,
            'forecast_times': times,
            'pi_last_measured': PI,
            'p_wf_last': P0,
            't_last': t0
        }

        # -----------------------------------------------
        #  fluxo probabilístico (prob_bias)
        # -----------------------------------------------
        ftype = self.fuser_cfg.get("type", "").lower()
        if ftype == "prob_bias":
        
            # 1) fuser devolve [mu_raw , log_sigma_raw]
            fused = self.fuser(ctx, per_step, training=training)
            mu_raw, log_sigma_raw = tf.split(fused, 2, axis=-1)
            mu_raw        = tf.squeeze(mu_raw,      -1)      # (B,H)
            log_sigma_raw = tf.squeeze(log_sigma_raw, -1)    # (B,H)
        
            # 2) mistura com o baseline físico
            alpha_vec = 0.7 * tf.sigmoid(self.alpha_raw) + 0.3  # (H,)
            alpha_vec = tf.reshape(alpha_vec, (1, self.horizon))
            mu = alpha_vec * Qs + (1.0 - alpha_vec) * mu_raw    # curva final P50
        
            # 3) σ sempre positivo
            sigma = tf.nn.softplus(log_sigma_raw) + 1e-6
        
            # 4) distribuição normal — out será P50
            dist = tfd.Normal(loc=mu, scale=sigma)
            out = dist.mean()                                   # idem fluxo “else”

            # out = out + norm.ppf(0.90) * sigma
        
            # ------------- RETORNO -------------
            # ordem precisa casar com _OUT_SCHEMAS[5]
            # ("pred", "q_phys", "res", "sigma", "alpha")
            res = mu_raw      # ← mesmo conceito de resíduo que o deterministic usa
            return out, Qs, res, sigma, alpha_vec

        else:
            res = self.fuser(ctx, per_step, training=training)
            self._add_residual_regularization(res)
            alpha_vec = 0.8 * tf.sigmoid(self.alpha_raw) + 0.2   # (H,)
            # alpha_vec = tf.reshape(alpha_vec, (1, self.horizon))  # broadcast (1,H)
            out = alpha_vec * Qs + (1.0 - alpha_vec) * res
            
            return (out, Qs, res) 
            
        if training:
            if tf.equal(self.iter % 8000, 0) and self.iter > 0:
                self.diagnostic(PI, P0, Qb, t0)
            self.iter.assign_add(1)

        

    def diagnostic(self, PI_last, P_wf_last, Q_base, t_last):
        """Print detailed physics diagnostics."""
        P_res_t = self.p_reservoir * tf.exp(-self.decay * 0.0)
        tf.print("\n=== Diagnostics at Iter", self.iter, "===")
        tf.print("Strategy: ", self.strategy_cfg)
        tf.print("P_reservoir:", self.p_reservoir)
        tf.print("decay_rate:", self.decay)
        tf.print("b_factor:", self.b)
        tf.print("ΔP mean:",tf.reduce_mean(P_res_t - P_wf_last)        )
        tf.print("PI_last mean:", tf.reduce_mean(PI_last))
        tf.print("P_wf_last mean:", tf.reduce_mean(P_wf_last))
        tf.print("Q_phys_base mean:", tf.reduce_mean(Q_base))
        tf.print("=== End ===\n")

    def get_config(self):
        """Serialize layer config."""
        cfg = super().get_config()
        cfg.update({
            'scaler_x_path': self.scaler_x_path,
            'scaler_y_path': self.scaler_y_path,
            'strategy_cfg': self.strategy_cfg,
            'fuser_cfg': self.fuser_cfg,
            'horizon': self.horizon,
            'pi_idx': self.pi_idx,
            'p_idx': self.p_idx,
            't_idx': self.t_idx,
            'pwf_idx': self.pwf_idx,
            'residual_magnitude_weight': self.residual_magnitude_weight,
            'residual_smoothness_weight': self.residual_smoothness_weight
        })
        return cfg





import logging
def create_model(
    input_shape: tf.TensorShape,
    horizon: int,
    scaler_X_path: Union[str, Path] = 'scalers/scaler_X.pkl',
    scaler_target_path: Union[str, Path] = 'scalers/scaler_target.pkl',
    strategy_config: Optional[Dict[str, Any]] = None,
    extractor_config: Optional[Dict[str, Any]] = None,
    fuser_config: Optional[Dict[str, Any]] = None,
    trend_degree: int = 2,
    phase: str = "balanced",
    freeze_trend: bool = False,
    freeze_physics: bool = False,
    fusion_type: str = "pin",
    name: str = "HybridModel",
) -> tf.keras.Model:
    """
    Build a hybrid forecasting model.

    Parameters:
        input_shape: shape of input tensor (batch, lag, features).
        horizon: forecast horizon.
        scaler_X_path: path to input scaler.
        scaler_target_path: path to target scaler.
        strategy_config: physics strategy (e.g. 'exponential', 'arps', 'static', 
            'weighted_ensemble', 'combined_exp_arps', 'pressure_ensemble', 
            'diffusivity_decay', 'diffusivity_pressure').
        extractor_config: 'tcn', 'cnn', 'rnn', 'aggregate', 'identity'.
        fuser_config: film', 'gating', 'bias_scale', 'concat', 'none'.
        trend_degree: polynomial degree for trend block.
        phase: training phase, e.g. 'balanced'.
        freeze_trend: if True, freeze trend branch weights.
        freeze_physics: if True, freeze physics branch weights.
        fusion_type: one of 'pin', 'trend', 'concat_dense', 'average'.
        name: model name.

    Returns:
        A Keras Model instance.
    """
    # defaults
    strategy_config = strategy_config or {"strategy_name": "exponential"}
    extractor_config = extractor_config or {"type": "cnn"}
    fuser_config = fuser_config or {"type": "gating"}
    fusion_type = fusion_type.lower()

    logging.info(f'Strategy: {strategy_config}')
    logging.info(f'Extractor: {extractor_config}')
    logging.info(f'Fuser: {fuser_config}')

    # feature indices
    DATA_IDX = [0, 2, 7]
    PHYS_IDX = [0, 3, 5, 6]

    # helper to slice features
    def gather(x, idx, name):
        return Lambda(lambda t: tf.gather(t, idx, axis=-1), name=name)(x)

    # inputs
    timesteps, feats = input_shape.shape[1], input_shape.shape[2]
    inputs = Input(shape=(timesteps, feats), name='all_features')
    data_feats = gather(inputs, DATA_IDX, 'data_features')
    phys_feats = gather(inputs, PHYS_IDX, 'physics_features')

    # compute relative physics indices
    rel_pi, rel_p, rel_t = (PHYS_IDX.index(i) for i in (0, 3, 5))

    # context extractor
    ctxt_extractor = create_context_extractor(dict(extractor_config))
    ctxt_vec = ctxt_extractor(data_feats)

    # physics branch
    phys_block = PhysicsInformedForecaster(
        scaler_x_path=scaler_X_path,
        scaler_y_path=scaler_target_path,
        strategy_cfg=strategy_config,
        fuser_cfg=fuser_config,
        horizon=horizon,
        pi_idx=rel_pi,
        p_idx=rel_p,
        t_idx=rel_t,
        pwf_idx=3,
        name='physics_block'
    )
    phys_out = phys_block([phys_feats, ctxt_vec])
    phys_block.trainable = not freeze_physics

    # fusion strategies mapping
    def require_trend():
        raise ValueError(f"Trend branch required for fusion '{fusion_type}'")

    fusion_map = {
    'pin':   lambda: phys_out,
    }

    if fusion_type not in fusion_map:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

    outputs = fusion_map[fusion_type]()

    return Model(inputs=inputs, outputs=outputs, name=name)


