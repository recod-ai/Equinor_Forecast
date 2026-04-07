# models/darcy_components.py

"""
Contains all reusable Keras Layer components for the Darcy-based model.
This includes the core physics layers, context encoders, driver forecasters,
and the trend block.
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Type

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, Concatenate, GlobalAveragePooling1D,
    Flatten, Dropout, Conv1D, LSTM, GRU
)
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm

# Project-specific module imports
from models.PINNs import (
    DynamicEnsembleStrategy, ExponentialDecayStrategy, ArpsDeclineStrategy,
    WeightedEnsembleStrategy, CombinedExpArpsStrategy, StaticPressureStrategy,
    DiffusivityDecayStrategy
)
from utils.utilities import invert_feature_scaling, get_center_and_scale
from forecast_pipeline.config import INITIAL_PRESSURE, kpa2psi



# =============================================================================
# 1. PhysicsParameterMixin to share initialization logic
# =============================================================================
class PhysicsParameterMixin:
    """
    Shared logic for initializing physics parameters and the strategy.

    Host layer must define before calling _init_physics():
      - self.strategy_config (Dict)
      - self.scaler_X_center_np (np.ndarray)
      - self.P_wf_idx (int)
      - self._physics_clip_upper_bound (float)
    """

    def _init_physics(self):
        strat = self.strategy_config.get("strategy_name", "pressure_ensemble")

        # Default trainable flags per strategy (unchanged legacy behaviour).
        default_train_map = {
            'exponential':       {'P_reservoir': True,  'decay_rate': True,  'b_factor': True},
            'arps':              {'P_reservoir': True,  'decay_rate': True,  'b_factor': True},
            'static':            {'P_reservoir': False, 'decay_rate': False, 'b_factor': False},
            'combined_exp_arps': {'P_reservoir': False, 'decay_rate': True,  'b_factor': True},
            'pressure_ensemble': {'P_reservoir': True,  'decay_rate': True,  'b_factor': False},
            'diffusivity_decay': {'P_reservoir': True,  'decay_rate': False, 'b_factor': False},
        }

        # Optional override: strategy_config["physics_train_flags"]
        override_flags = self.strategy_config.get("physics_train_flags", None)
        train_map = dict(default_train_map)
        if override_flags is not None and strat in train_map:
            # Merge overrides on top of defaults for the current strategy only.
            merged = dict(train_map[strat])
            merged.update(override_flags)
            train_map[strat] = merged

        flags = train_map[strat]

        # Dataset-specific P_reservoir init (legacy behaviour kept).
        dataset_name = self.strategy_config.get("dataset_name")
        if dataset_name == "VOLVE":
            init_P = float(self.scaler_X_center_np[self.P_wf_idx]) * 1.1
        elif dataset_name == "UNISIM_IV":
            init_P = kpa2psi(INITIAL_PRESSURE["UNISIM-IV-2024"])
        else:
            raise ValueError(f"Unknown dataset_name '{dataset_name}' for physics initialization.")

        init_d = 0.01
        init_b = 0.5

        param_defs = {
            'P_reservoir': {
                'initializer': tf.keras.initializers.Constant(init_P),
                'constraint':  None,
                'trainable':   flags['P_reservoir'],
            },
            'decay_rate': {
                'initializer': tf.keras.initializers.Constant(init_d),
                'constraint':  lambda x: tf.clip_by_value(x, 0.001, self._physics_clip_upper_bound),
                'trainable':   flags['decay_rate'],
            },
            'b_factor': {
                'initializer': tf.keras.initializers.Constant(init_b),
                'constraint':  lambda x: tf.clip_by_value(x, 0.001, self._physics_clip_upper_bound),
                'trainable':   flags['b_factor'],
            },
        }

        # Create layer weights and attach them as attributes (legacy API).
        for var_name, spec in param_defs.items():
            w = self.add_weight(
                name=var_name,
                shape=(),
                initializer=spec['initializer'],
                trainable=spec['trainable'],
                constraint=spec['constraint'],
            )
            setattr(self, var_name, w)

        # Base parameters dict (used by both legacy and latent paths).
        # Note: these are layer-level scalars; encoder can use them as priors.
        self.base_params = {
            "P_reservoir": self.P_reservoir,
            "decay_rate":  self.decay_rate,
            "b_factor":    self.b_factor,
        }

        # Physics strategy instance (used by legacy/context layers).
        params_for_strategy = {
            **self.base_params,
            "absolute_value": True,
        }
        self.physics_strategy = physics_strategy_factory(strat, params_for_strategy)



# =============================================================================
# Time grid utility
# =============================================================================
def infer_t_grid(t_hist, steps, mode="relative", dt_scale=1.0, start_at_t0=True, dt_scope="last_M"):
    diffs = t_hist[:, 1:] - t_hist[:, :-1]
    # Use last segment to estimate dt if requested
    diffs_used = diffs if dt_scope != "last_M" else diffs[:, -tf.shape(diffs)[-1]:]
    
    mean_dt = tf.reduce_mean(diffs_used, axis=-1, keepdims=True)
    # Fallback for safe division/multiplication
    one = tf.ones_like(mean_dt)
    dt = tf.where(mean_dt > 1e-8, mean_dt, one) * dt_scale

    k0, k1 = (0, steps) if start_at_t0 else (1, steps + 1)
    k = tf.cast(tf.range(k0, k1), t_hist.dtype)[tf.newaxis, :]
    t_rel = k * dt
    t_abs = t_hist[:, -1:] + t_rel

    return t_rel if mode == "relative" else t_abs

# =============================================================================
# Latent parameter encoder (history -> physical parameters)
# =============================================================================


class PhysicsParameterEncoder(tf.keras.layers.Layer, PhysicsParameterMixin):
    """
    Encoder that maps history/context to physics parameters
    (P_reservoir, decay_rate, b_factor).

    Default behaviour (no extra wiring in create_model):

    - Uses PhysicsParameterMixin to create *physical priors*:
        self.base_params = {
            "P_reservoir": <scalar>,
            "decay_rate":  <scalar>,
            "b_factor":    <scalar>,
        }
      initialized exactly like the legacy physics layers.

    - A GRU + Dense head predicts small residuals in [-1, 1] (via tanh),
      and we combine them with the priors as:

        P_res = P_base * (1 + α_P * ΔP̂)
        decay = d_base * (1 + α_d * Δd̂)
        b     = b_base + α_b * Δb̂

      where α_* are residual_scales (small factors like 0.1–0.5).

    So the encoder is *always* a residual model around physical priors, and
    you do NOT need to pass base_params explicitly from create_model.
    """

    def __init__(
        self,
        scaler_X,
        scaler_target,
        strategy_config: Dict,
        rnn_units: int = 64,
        activations: Optional[Dict[str, str]] = None,  # kept for BC, currently unused
        residual_scales: Optional[Dict[str, float]] = None,
        clip_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        diag_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scaler_X = scaler_X
        self.scaler_target = scaler_target
        self.strategy_config = dict(strategy_config or {})
        self.rnn_units = int(rnn_units)

        # Residual configuration (how far we can move from the prior).
        default_scales = {
            "P_reservoir": 0.5,
            "decay_rate": 0.5,
            "b_factor": 0.5,
        }
        self.residual_scales = {**default_scales, **(residual_scales or {})}

        # Optional physical clipping per parameter: {"name": (min, max)}
        self.clip_bounds = clip_bounds or {}

        # Diagnostics (optional).
        diag_cfg = dict(diag_config or self.strategy_config.get("latent_diag", {}))
        self._diag_enabled = bool(diag_cfg.get("enabled", False))
        self._diag_period = int(diag_cfg.get("period", 500))
        self._iter = tf.Variable(0, trainable=False, dtype=tf.int32)

        # Core layers.
        self.rnn = GRU(self.rnn_units, name="latent_param_gru")
        # 3 params: [P_reservoir, decay_rate, b_factor]
        self.head = Dense(3, name="latent_param_head")

    def build(self, input_shape):
        # Needed by PhysicsParameterMixin._init_physics()
        self.scaler_X_center_np, _ = get_center_and_scale(
            self.scaler_X, as_tf=False
        )
        # Consistent with the rest of the code: Pwf index in X scaler.
        self.P_wf_idx = 3

        # Layer-specific upper bound for physics params.
        self._physics_clip_upper_bound = float(
            self.strategy_config.get("physics_clip_upper_bound", 0.99)
        )

        # This creates:
        #   - self.P_reservoir, self.decay_rate, self.b_factor (tf.Variables)
        #   - self.base_params = {"P_reservoir", "decay_rate", "b_factor"}
        #   - self.physics_strategy (not used here, but harmless)
        self._init_physics()

        super().build(input_shape)

    def _combine_with_residual(
        self,
        base: tf.Tensor,
        delta_hat: tf.Tensor,
        name: str,
        mode: str = "mul",
    ) -> tf.Tensor:
        """
        Combine a base parameter with a small residual.

        Parameters
        ----------
        base : tf.Tensor
            Scalar or (B, 1) tensor (physical base value).
        delta_hat : tf.Tensor
            (B, 1) tensor in [-1, 1] coming from tanh.
        name : str
            Parameter name for scale / clipping lookup.
        mode : {"mul", "add"}, optional
            - "mul": base * (1 + α * Δ)
            - "add": base + α * Δ

        Returns
        -------
        tf.Tensor
            Combined parameter with optional clipping applied.
        """
        base = tf.convert_to_tensor(base)
        delta_hat = tf.convert_to_tensor(delta_hat)

        base = tf.cast(base, delta_hat.dtype)
        scale = float(self.residual_scales.get(name, 0.1))
        scale_t = tf.cast(scale, delta_hat.dtype)

        if mode == "add":
            param = base + scale_t * delta_hat
        else:  # "mul" by default
            param = base * (1.0 + scale_t * delta_hat)

        # Optional physical clipping.
        lo, hi = self.clip_bounds.get(name, (None, None))
        if lo is not None or hi is not None:
            lo_t = tf.cast(-np.inf if lo is None else lo, param.dtype)
            hi_t = tf.cast(np.inf if hi is None else hi, param.dtype)
            param = tf.clip_by_value(param, lo_t, hi_t)

        return param

    def _maybe_log(
        self,
        p_res: tf.Tensor,
        d_res: tf.Tensor,
        b_res: tf.Tensor,
    ):
        if not self._diag_enabled:
            return

        cond = tf.equal(tf.math.mod(self._iter, self._diag_period), 0)

        def _do_log():
            tf.print("\n[latent_diag] PhysicsParameterEncoder iter =", self._iter)
            log_tensor_stats("enc.param.P_reservoir", p_res)
            log_tensor_stats("enc.param.decay_rate", d_res)
            log_tensor_stats("enc.param.b_factor", b_res)
            return 0

        tf.cond(cond, _do_log, lambda: 0)

    def call(self, inputs: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        """
        Parameters
        ----------
        inputs : tf.Tensor
            History/context with shape (B, M, F).

        Returns
        -------
        Dict[str, tf.Tensor]
            {
                "P_reservoir": (B, 1),
                "decay_rate":  (B, 1),
                "b_factor":    (B, 1),
            }

        Behaviour: always predicts residuals around `self.base_params` created
        via PhysicsParameterMixin.
        """
        # Shared backbone.
        h = self.rnn(inputs, training=training)  # (B, rnn_units)
        raw = self.head(h)                       # (B, 3)

        # Split raw outputs.
        p_raw = raw[..., 0][..., tf.newaxis]  # (B, 1)
        d_raw = raw[..., 1][..., tf.newaxis]
        b_raw = raw[..., 2][..., tf.newaxis]

        # Residuals in [-1, 1].
        p_delta = tf.nn.tanh(p_raw)
        d_delta = tf.nn.tanh(d_raw)
        b_delta = tf.nn.tanh(b_raw)

        # Base values from PhysicsParameterMixin.
        p_base = self.base_params["P_reservoir"]
        d_base = self.base_params["decay_rate"]
        b_base = self.base_params["b_factor"]

        # Combine with small residuals.
        p_res = self._combine_with_residual(
            base=p_base,
            delta_hat=p_delta,
            name="P_reservoir",
            mode="mul",
        )
        d_res = self._combine_with_residual(
            base=d_base,
            delta_hat=d_delta,
            name="decay_rate",
            mode="mul",
        )
        b_res = self._combine_with_residual(
            base=b_base,
            delta_hat=b_delta,
            name="b_factor",
            mode="add",  # more natural for b-factor
        )

        self._maybe_log(p_res, d_res, b_res)
        self._iter.assign_add(1)

        return {
            "P_reservoir": p_res,
            "decay_rate": d_res,
            "b_factor": b_res,
        }

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "rnn_units": self.rnn_units,
                "residual_scales": self.residual_scales,
                "clip_bounds": self.clip_bounds,
                "strategy_config": self.strategy_config,
            }
        )
        return config


# =============================================================================
# Stateless Physics Decoder (params + drivers + time -> rate)
# =============================================================================


# =============================================================================
# Stateless Physics Decoder (params + drivers + time -> normalized rate)
# =============================================================================

class PhysicsDecoder(tf.keras.layers.Layer):
    """
    Stateless decoder that maps:
        params (dict of (B,1)),
        PI_future (B, N),
        P_wf_future (B, N),
        t_grid (B, N)
    into a normalized rate (B, N), using the same target scaler as other paths.

    Optional diagnostics (for debugging latent_params):
        diag_config = {"enabled": bool, "period": int}
    """

    def __init__(
        self,
        scaler_target,
        strategy_name: str,
        diag_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scaler_target = scaler_target
        self.strategy_name = strategy_name

        diag_cfg = dict(diag_config or {})
        self._diag_enabled = bool(diag_cfg.get("enabled", False))
        self._diag_period = int(diag_cfg.get("period", 1000))
        self._iter = tf.Variable(0, trainable=False, dtype=tf.int32)

        # Filled in build()
        self.oil_mean = None
        self.oil_std = None

        # Affine guard weights
        self.a_affine = None
        self.b_affine = None

        # Scalar std value for constraint (set in build)
        self._oil_std_scalar = None

    def build(self, input_shape):
        # Tensor versions used for normalization in call()
        self.oil_mean, self.oil_std = get_center_and_scale(
            self.scaler_target, as_tf=True, dtype=tf.float32
        )

        # Also get a scalar std (numpy) to define a shape-compatible constraint
        oil_mean_np, oil_std_np = get_center_and_scale(
            self.scaler_target, as_tf=False
        )
        # Use average std in case scaler has multiple targets
        self._oil_std_scalar = float(np.mean(oil_std_np))

        # Sanity: std > 0
        with tf.control_dependencies([
            tf.debugging.assert_greater(
                self.oil_std, 1e-12, message="oil_std ~ 0 (invalid scaler_target?)"
            )
        ]):
            pass

        # -------------------------
        # Affine guard (level + slope)
        # -------------------------
        # a_affine ≈ 1, allow ~20% slope variation
        self.a_affine = self.add_weight(
            "a_affine",
            shape=(),
            initializer="ones",
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0.8, 1.2),
        )

        # b_affine in PHYSICAL UNITS: allow a few stds of shift
        bias_limit = 5.0 * self._oil_std_scalar  # python float → scalar
        self.b_affine = self.add_weight(
            "b_affine",
            shape=(),
            initializer="zeros",
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, -bias_limit, bias_limit),
        )

        super().build(input_shape)

    def _maybe_log(
        self,
        params: Dict[str, tf.Tensor],
        PI_future: tf.Tensor,
        P_wf_future: tf.Tensor,
        t_grid: tf.Tensor,
        Q_phys_corr: tf.Tensor,
        y_scaled: tf.Tensor,
    ):
        if not self._diag_enabled:
            return

        cond = tf.equal(tf.math.mod(self._iter, self._diag_period), 0)

        def _do_log():
            tf.print("\n[latent_diag] PhysicsDecoder iter =", self._iter,
                     "| strategy =", self.strategy_name)
            for k, v in params.items():
                log_tensor_stats(f"dec.param.{k}", v)
            log_tensor_stats("dec.PI_future", PI_future)
            log_tensor_stats("dec.P_wf_future", P_wf_future)
            log_tensor_stats("dec.t_grid", t_grid)
            log_tensor_stats("dec.Q_phys", Q_phys_corr)
            log_tensor_stats("dec.y_scaled", y_scaled)
            return 0

        tf.cond(cond, _do_log, lambda: 0)

    def call(
        self,
        params: Dict[str, tf.Tensor],
        PI_future: tf.Tensor,
        P_wf_future: tf.Tensor,
        t_grid: tf.Tensor,
    ) -> tf.Tensor:
        """
        params     : dict of tensors, typically (B,1).
        PI_future  : (B, N) in physical units.
        P_wf_future: (B, N) in physical units.
        t_grid     : (B, N) time grid.

        Returns
        -------
        y_scaled : (B, N) in target scale.
        """
        # 1) Raw physical curve from chosen strategy
        Q_phys = compute_Q_phys_functional(
            strategy_name=self.strategy_name,
            params=params,
            PI_future=PI_future,
            P_wf_future=P_wf_future,
            t_grid=t_grid,
        )

        tf.debugging.assert_all_finite(Q_phys, "Q_phys has NaN/Inf")

        # 2) Affine guard: learn small global slope + level correction
        Q_phys_corr = self.a_affine * Q_phys + self.b_affine

        # 3) Normalize back to target scale for the loss
        y_scaled = (Q_phys_corr - self.oil_mean) / self.oil_std

        tf.debugging.assert_all_finite(y_scaled, "y_scaled has NaN/Inf")

        self._maybe_log(params, PI_future, P_wf_future, t_grid, Q_phys_corr, y_scaled)
        self._iter.assign_add(1)

        return y_scaled

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "strategy_name": self.strategy_name,
            "diag_config": {
                "enabled": self._diag_enabled,
                "period": int(self._diag_period),
            },
        })
        return config

# # =============================================================================
# # Latent parameter encoder (history -> physical parameters)
# # =============================================================================

# class PhysicsParameterEncoder(tf.keras.layers.Layer, PhysicsParameterMixin):
#     """
#     History -> physics parameters (P_reservoir, decay_rate, b_factor).

#     Default behaviour (no extra wiring in create_model):

#     - Uses PhysicsParameterMixin to create *physical priors*:
#         self.base_params = {
#             "P_reservoir": <scalar>,
#             "decay_rate":  <scalar>,
#             "b_factor":    <scalar>,
#         }
#       initialized exactly like the legacy physics layers.

#     - The GRU + Dense head predicts small residuals in [-1, 1] (tanh),
#       and we combine them with the priors as:

#         P_res = P_base * (1 + α_P * ΔP̂)
#         decay = d_base * (1 + α_d * Δd̂)
#         b     = b_base + α_b * Δb̂

#       where α_* are residual_scales (small factors like 0.1–0.3).

#     So the encoder is *always* a residual model around physical priors,
#     and you do NOT need to pass base_params from create_model.
#     """

#     def __init__(
#         self,
#         scaler_X,
#         scaler_target,
#         strategy_config: Dict,
#         rnn_units: int = 64,
#         activations: Optional[Dict[str, str]] = None,   # kept for BC, currently unused
#         residual_scales: Optional[Dict[str, float]] = None,
#         clip_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
#         diag_config: Optional[Dict] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.scaler_X = scaler_X
#         self.scaler_target = scaler_target
#         self.strategy_config = dict(strategy_config or {})
#         self.rnn_units = int(rnn_units)

#         # Residual configuration (how far we can move from the prior).
#         default_scales = {
#             "P_reservoir": 0.5,
#             "decay_rate":  0.5,
#             "b_factor":    0.5,
#         }
#         self.residual_scales = {**default_scales, **(residual_scales or {})}

#         # Optional physical clipping per parameter: {"name": (min, max)}
#         self.clip_bounds = clip_bounds or {}

#         # Diagnostics (optional)
#         diag_cfg = dict(diag_config or self.strategy_config.get("latent_diag", {}))
#         self._diag_enabled = bool(diag_cfg.get("enabled", False))
#         self._diag_period = int(diag_cfg.get("period", 500))
#         self._iter = tf.Variable(0, trainable=False, dtype=tf.int32)

#         # Core layers
#         self.rnn = GRU(self.rnn_units, name="latent_param_gru")
#         # 3 params: [P_reservoir, decay_rate, b_factor]
#         self.head = Dense(3, name="latent_param_head")

#     def build(self, input_shape):
#         # Needed by PhysicsParameterMixin._init_physics()
#         self.scaler_X_center_np, _ = get_center_and_scale(self.scaler_X, as_tf=False)
#         # Consistent with the rest of the code: Pwf index in X scaler
#         self.P_wf_idx = 3

#         # Layer-specific clip upper bound for physics params
#         self._physics_clip_upper_bound = float(
#             self.strategy_config.get("physics_clip_upper_bound", 0.99)
#         )

#         # This creates:
#         #   - self.P_reservoir, self.decay_rate, self.b_factor (tf.Variables)
#         #   - self.base_params = {"P_reservoir", "decay_rate", "b_factor"}
#         #   - self.physics_strategy (not used here, but harmless)
#         self._init_physics()

#         super().build(input_shape)

#     def _combine_with_residual(
#         self,
#         base: tf.Tensor,
#         delta_hat: tf.Tensor,
#         name: str,
#         mode: str = "mul",
#     ) -> tf.Tensor:
#         """
#         Combine base parameter with a small residual.

#         base      : scalar or (B,1) tensor (physical base value).
#         delta_hat : (B,1) in [-1, 1] from tanh.
#         name      : parameter name for scale / clipping lookup.
#         mode      : "mul" (base * (1 + α * Δ)) or "add" (base + α * Δ).
#         """
#         base = tf.convert_to_tensor(base)
#         delta_hat = tf.convert_to_tensor(delta_hat)

#         base = tf.cast(base, delta_hat.dtype)
#         scale = float(self.residual_scales.get(name, 0.1))
#         scale_t = tf.cast(scale, delta_hat.dtype)

#         if mode == "add":
#             param = base + scale_t * delta_hat
#         else:  # "mul" by default
#             param = base * (1.0 + scale_t * delta_hat)

#         # Optional physical clipping
#         lo, hi = self.clip_bounds.get(name, (None, None))
#         if lo is not None or hi is not None:
#             lo_t = tf.cast(-np.inf if lo is None else lo, param.dtype)
#             hi_t = tf.cast(np.inf if hi is None else hi, param.dtype)
#             param = tf.clip_by_value(param, lo_t, hi_t)

#         return param

#     def _maybe_log(
#         self,
#         p_res: tf.Tensor,
#         d_res: tf.Tensor,
#         b_res: tf.Tensor,
#     ):
#         if not self._diag_enabled:
#             return

#         cond = tf.equal(tf.math.mod(self._iter, self._diag_period), 0)

#         def _do_log():
#             tf.print(
#                 "\n[latent_diag] PhysicsParameterEncoder iter =", self._iter
#             )
#             log_tensor_stats("enc.param.P_reservoir", p_res)
#             log_tensor_stats("enc.param.decay_rate", d_res)
#             log_tensor_stats("enc.param.b_factor", b_res)
#             return 0

#         tf.cond(cond, _do_log, lambda: 0)

#     def call(self, inputs: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
#         """
#         inputs : (B, M, F) history/context.

#         Returns
#         -------
#         params : dict
#             {
#                 "P_reservoir": (B, 1),
#                 "decay_rate":  (B, 1),
#                 "b_factor":    (B, 1),
#             }

#         Behaviour: always residual around self.base_params created
#         via PhysicsParameterMixin.
#         """
#         # Shared backbone
#         h = self.rnn(inputs, training=training)  # (B, rnn_units)
#         raw = self.head(h)                      # (B, 3)

#         # Split raw outputs
#         p_raw = raw[..., 0][..., tf.newaxis]  # (B, 1)
#         d_raw = raw[..., 1][..., tf.newaxis]
#         b_raw = raw[..., 2][..., tf.newaxis]

#         # Residuals in [-1, 1]
#         p_delta = tf.nn.tanh(p_raw)
#         d_delta = tf.nn.tanh(d_raw)
#         b_delta = tf.nn.tanh(b_raw)

#         # Base values from PhysicsParameterMixin
#         p_base = self.base_params["P_reservoir"]
#         d_base = self.base_params["decay_rate"]
#         b_base = self.base_params["b_factor"]

#         # Combine with small residuals
#         p_res = self._combine_with_residual(
#             base=p_base,
#             delta_hat=p_delta,
#             name="P_reservoir",
#             mode="mul",
#         )
#         d_res = self._combine_with_residual(
#             base=d_base,
#             delta_hat=d_delta,
#             name="decay_rate",
#             mode="mul",
#         )
#         b_res = self._combine_with_residual(
#             base=b_base,
#             delta_hat=b_delta,
#             name="b_factor",
#             mode="add",  # more natural for b-factor
#         )

#         self._maybe_log(p_res, d_res, b_res)
#         self._iter.assign_add(1)

#         return {
#             "P_reservoir": p_res,
#             "decay_rate":  d_res,
#             "b_factor":    b_res,
#         }

#     def get_config(self) -> Dict:
#         config = super().get_config()
#         config.update({
#             "rnn_units": self.rnn_units,
#             "residual_scales": self.residual_scales,
#             "clip_bounds": self.clip_bounds,
#             "strategy_config": self.strategy_config,
#         })
#         return config


# # =============================================================================
# # Stateless Physics Decoder (params + drivers + time -> normalized rate)
# # =============================================================================

# class PhysicsDecoder(tf.keras.layers.Layer):
#     """
#     Stateless decoder that maps:
#         params (dict of (B,1)),
#         PI_future (B, N),
#         P_wf_future (B, N),
#         t_grid (B, N)
#     into a normalized rate (B, N), using the same target scaler as other paths.

#     Optional diagnostics (for debugging latent_params):
#         diag_config = {"enabled": bool, "period": int}
#     """

#     def __init__(
#         self,
#         scaler_target,
#         strategy_name: str,
#         diag_config: Optional[Dict] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.scaler_target = scaler_target
#         self.strategy_name = strategy_name

#         diag_cfg = dict(diag_config or {})
#         self._diag_enabled = bool(diag_cfg.get("enabled", False))
#         self._diag_period = int(diag_cfg.get("period", 1000))
#         self._iter = tf.Variable(0, trainable=False, dtype=tf.int32)

#         # These will be populated in build()
#         self.oil_mean = None
#         self.oil_std = None

#     def build(self, input_shape):
        
        
#         # Reuse existing utility to get center and scale
#         self.oil_mean, self.oil_std = get_center_and_scale(
#             self.scaler_target, as_tf=True, dtype=tf.float32
#         )

#         # Sanity: std > 0
#         with tf.control_dependencies([
#             tf.debugging.assert_greater(self.oil_std, 1e-12, message="oil_std ~ 0 (scaler_target inválido?)")
#         ]):
#             pass

        
#         super().build(input_shape)

#     def _maybe_log(
#         self,
#         params: Dict[str, tf.Tensor],
#         PI_future: tf.Tensor,
#         P_wf_future: tf.Tensor,
#         t_grid: tf.Tensor,
#         Q_phys: tf.Tensor,
#         y_scaled: tf.Tensor,
#     ):
#         if not self._diag_enabled:
#             return

#         cond = tf.equal(tf.math.mod(self._iter, self._diag_period), 0)

#         def _do_log():
#             tf.print("\n[latent_diag] PhysicsDecoder iter =", self._iter,
#                      "| strategy =", self.strategy_name)
#             for k, v in params.items():
#                 log_tensor_stats(f"dec.param.{k}", v)
#             log_tensor_stats("dec.PI_future", PI_future)
#             log_tensor_stats("dec.P_wf_future", P_wf_future)
#             log_tensor_stats("dec.t_grid", t_grid)
#             log_tensor_stats("dec.Q_phys", Q_phys)
#             log_tensor_stats("dec.y_scaled", y_scaled)
#             return 0

#         tf.cond(cond, _do_log, lambda: 0)

#     def call(
#         self,
#         params: Dict[str, tf.Tensor],
#         PI_future: tf.Tensor,
#         P_wf_future: tf.Tensor,
#         t_grid: tf.Tensor,
#     ) -> tf.Tensor:
#         """
#         params     : dict of tensors, typically (B,1).
#         PI_future  : (B, N) in physical units.
#         P_wf_future: (B, N) in physical units.
#         t_grid     : (B, N) time grid.

#         Returns
#         -------
#         y_scaled : (B, N) in target scale.
#         """
#         Q_phys = compute_Q_phys_functional(
#             strategy_name=self.strategy_name,
#             params=params,
#             PI_future=PI_future,
#             P_wf_future=P_wf_future,
#             t_grid=t_grid,
#         )

#         y_scaled = (Q_phys - self.oil_mean) / self.oil_std

#         self._maybe_log(params, PI_future, P_wf_future, t_grid, Q_phys, y_scaled)
#         self._iter.assign_add(1)

#         # Guard rails: ranges plausíveis
#         tf.debugging.assert_all_finite(Q_phys, "Q_phys has NaN/Inf")
#         tf.debugging.assert_all_finite(y_scaled, "y_scaled has NaN/Inf")

#         # Dentro do PhysicsDecoder.call, logo antes de retornar:
#         if getattr(self, "_affine_guard", None) is None:
#             self._affine_guard = {
#                 "a": self.add_weight("a_affine", shape=(), initializer="ones", trainable=True,
#                                      constraint=lambda x: tf.clip_by_value(x, 0.9, 1.1)),
#                 "b": self.add_weight("b_affine", shape=(), initializer="zeros", trainable=True,
#                                      constraint=lambda x: tf.clip_by_value(x, -0.1, 0.1)),
#             }
#         Q_phys_corr = self._affine_guard["a"] * Q_phys + self._affine_guard["b"]
#         y_scaled = (Q_phys_corr - self.oil_mean) / self.oil_std


#         return y_scaled

#     def get_config(self) -> Dict:
#         config = super().get_config()
#         config.update({
#             "strategy_name": self.strategy_name,
#             "diag_config": {
#                 "enabled": self._diag_enabled,
#                 "period": int(self._diag_period),
#             },
#         })
#         return config




# =============================================================================
# 2. Legacy Physics: DarcyPhysicsLayer 
# =============================================================================
class DarcyPhysicsLayer(tf.keras.layers.Layer, PhysicsParameterMixin):
    """
    Original physics layer that *consumes drivers you provide* (no driver generation).
    Input shape: (B, N, 3) -> [PI_scaled, P_scaled, t_feature]
    Output     : (B, N)    -> normalized oil rate
    """
    def __init__(self, scaler_X, scaler_target, strategy_config, **kwargs):
        super().__init__(**kwargs)
        self.scaler_X = scaler_X
        self.scaler_target = scaler_target
        self.strategy_config = strategy_config
        self.iteration = tf.Variable(0, trainable=False, dtype=tf.int32)

    def build(self, input_shape):
        # --- diagnostics config ---
        diag_cfg = dict(self.strategy_config.get("diagnostics", {}))
        self._diag_enabled = bool(diag_cfg.get("enabled", False))
        self._diag_period  = int(diag_cfg.get("period", 500))
        
        self.scaler_X_center_np, self.scaler_X_scale_np = get_center_and_scale(self.scaler_X, as_tf=False)
        self.oil_center_np,      self.oil_scale_np      = get_center_and_scale(self.scaler_target, as_tf=False)
        self.scaler_X_mean, self.scaler_X_std = get_center_and_scale(self.scaler_X, as_tf=True, dtype=tf.float32)
        self.oil_mean,      self.oil_std      = get_center_and_scale(self.scaler_target, as_tf=True, dtype=tf.float32)
        self.PI_idx = 0
        self.P_wf_idx = 3

        # Set the specific clipping value for this layer before calling the mixin method
        self._physics_clip_upper_bound = 5.0
        self._init_physics() # This now calls the method from PhysicsParameterMixin
        
        super().build(input_shape)

    # The _init_physics method has been removed from this class and now lives in the Mixin.

    def call(self, inputs, training: bool = False):
        PI_scaled = inputs[..., 0]
        P_scaled  = inputs[..., 1]
        t_feature = inputs[..., 2]
        PI_measured = invert_feature_scaling(PI_scaled, self.scaler_X_mean, self.scaler_X_std, self.PI_idx)
        P_wf       = invert_feature_scaling(P_scaled,  self.scaler_X_mean, self.scaler_X_std, self.P_wf_idx)
        Q_phys_base = self.physics_strategy.compute_Q_phys(PI_measured, P_wf, t_feature)
        physics_prediction_scaled = (Q_phys_base - self.oil_mean) / self.oil_std

        # diagnostics
        if self._diag_enabled and tf.equal(tf.math.mod(self.iteration, self._diag_period), 0):
            self.diagnostic(PI_measured, P_wf, Q_phys_base, t_feature, loss=None)
            
        self.iteration.assign_add(1)
        return physics_prediction_scaled

    def diagnostic(self, PI_measured, P_wf, Q_phys_base, t_feature, loss=None):
        """Print compact iteration statistics + physics parameters (legacy path)."""
        P_res_t = self.P_reservoir * tf.exp(-self.decay_rate * t_feature)
        tf.print("\n=== DarcyPhysicsLayer Diagnostics @ iter", self.iteration, "===")
        tf.print(" strategy_config:", self.strategy_config)
        tf.print(" P_reservoir    :", self.P_reservoir)
        tf.print(" decay_rate     :", self.decay_rate)
        if hasattr(self, "b_factor"):
            tf.print(" b_factor       :", self.b_factor)
        tf.print(" ΔP mean        :", tf.reduce_mean(P_res_t - P_wf))
        tf.print(" PI mean        :", tf.reduce_mean(PI_measured))
        tf.print(" P_wf mean      :", tf.reduce_mean(P_wf))
        tf.print(" Q_phys mean    :", tf.reduce_mean(Q_phys_base))
        if loss is not None:
            tf.print(" batch loss     :", loss)
        tf.print("=== End ===\n")

    def get_config(self):
        config = super().get_config()
        config.update({
            "strategy_config": self.strategy_config,
        })
        return config


# =============================================================================
# 3. Context-aware Physics: DarcyTimeDecoderLayer (now uses the Mixin)
# =============================================================================
# class DarcyTimeDecoderLayer(tf.keras.layers.Layer, PhysicsParameterMixin):
#     def __init__(self, scaler_X, scaler_target, strategy_config: Dict, **kwargs):
#         super().__init__(**kwargs)
#         self.scaler_X = scaler_X
#         self.scaler_target = scaler_target
#         self.strategy_config = dict(strategy_config or {})
#         self.iteration = tf.Variable(0, trainable=False, dtype=tf.int32)

#     def build(self, input_shape):
#         # --- diagnostics ---
#         diag_cfg = dict(self.strategy_config.get("diagnostics", {}))
#         self._diag_enabled = bool(diag_cfg.get("enabled", False))
#         self._diag_period  = int(diag_cfg.get("period", 500))

#         self._gen_horizon = int(self.strategy_config.get("forecast_horizon", 0)) or None
#         self._time_mode   = self.strategy_config.get("time_mode", "relative")
#         self._dt_scale    = float(self.strategy_config.get("dt_scale", 1.0))
#         self._debug       = bool(self.strategy_config.get("debug", False))

#         self.scaler_X_center_np, self.scaler_X_scale_np = get_center_and_scale(self.scaler_X, as_tf=False)
#         self.oil_center_np,      self.oil_scale_np      = get_center_and_scale(self.scaler_target, as_tf=False)
#         self.scaler_X_mean, self.scaler_X_std = get_center_and_scale(self.scaler_X, as_tf=True, dtype=tf.float32)
#         self.oil_mean,      self.oil_std      = get_center_and_scale(self.scaler_target, as_tf=True, dtype=tf.float32)
#         self.PI_idx   = 0
#         self.P_wf_idx = 3
        
#         # Set the specific clipping value for this layer before calling the mixin method
#         self._physics_clip_upper_bound = 0.99
#         self._init_physics() # This now calls the method from PhysicsParameterMixin

#         # ---- Context Encoder & Forecaster setup (unique to this layer) ----
#         self._use_ctx = bool(self.strategy_config.get("use_context_encoder", False))
#         self._ctx_src = self.strategy_config.get("context_source", "inputs")
#         if self._use_ctx:
#             ctx_cfg = dict(self.strategy_config.get("context_encoder", {}))
#             self._ctx = make_context_encoder(ctx_cfg)
#             self._ctx.build(input_shape)

#         policy = self.strategy_config.get("forecast_policy", "hold_last")
#         hist_len = input_shape[1]
#         if hist_len is None:
#             raise ValueError("DarcyTimeDecoderLayer requires a fixed history length M.")
#         steps = self._gen_horizon or int(hist_len)

#         forecaster_cfg = dict(self.strategy_config.get("forecaster_config", {}))
#         self.forecaster = make_forecaster(policy, steps=steps, config=forecaster_cfg)
#         self.forecaster.build((input_shape[0], input_shape[1], 2))
        
#         super().build(input_shape)

#     # The _init_physics method has been removed from this class and now lives in the Mixin.

#     def _infer_dt_and_future_times(self, t_hist: tf.Tensor, steps: int) -> tf.Tensor:
#         diffs   = t_hist[:, 1:] - t_hist[:, :-1]
#         mean_dt = tf.reduce_mean(diffs, axis=-1, keepdims=True)
#         one     = tf.ones_like(mean_dt)
#         dt      = tf.where(mean_dt > 1e-8, mean_dt, one) * self._dt_scale
#         k     = tf.cast(tf.range(1, steps + 1), t_hist.dtype)[tf.newaxis, :]
#         t_rel = k * dt
#         t_abs = t_hist[:, -1:] + t_rel
#         return t_rel if (self._time_mode == "relative") else t_abs

#     def call(self, inputs, training: bool = False):
#         PI_scaled = inputs[..., 0]
#         P_scaled  = inputs[..., 1]
#         t_hist    = inputs[..., 2]

#         steps = self._gen_horizon or tf.shape(t_hist)[1]
#         t_grid = self._infer_dt_and_future_times(t_hist, steps)

#         last_pi_s = PI_scaled[:, -1:]
#         last_p_s  = P_scaled[:,  -1:]
#         hist_s    = tf.stack([PI_scaled, P_scaled], axis=-1)
#         last_s    = tf.stack([last_pi_s, last_p_s], axis=-1)
#         centered  = hist_s - last_s

#         if self._use_ctx:
#             if self._ctx_src == "inputs":
#                 ctx_in = inputs
#             elif self._ctx_src == "centered":
#                 t_exp = tf.expand_dims(t_hist, axis=-1)
#                 ctx_in = tf.concat([centered, t_exp], axis=-1)
#             else: # "hist"
#                 t_exp = tf.expand_dims(t_hist, axis=-1)
#                 ctx_in = tf.concat([hist_s, t_exp], axis=-1)
#             context_vector = self._ctx(ctx_in, training=training)
#             forecaster_input = (centered, context_vector)
#         else:
#             forecaster_input = centered

#         PI_trend_s, P_trend_s = self.forecaster(forecaster_input, training=training)

#         PI_base = invert_feature_scaling(last_pi_s, self.scaler_X_mean, self.scaler_X_std, self.PI_idx)
#         P_base  = invert_feature_scaling(last_p_s,  self.scaler_X_mean, self.scaler_X_std, self.P_wf_idx)
#         PI_trend = PI_trend_s * self.scaler_X_std[self.PI_idx]
#         P_trend  = P_trend_s  * self.scaler_X_std[self.P_wf_idx]
#         PI_future = PI_base + PI_trend
#         P_future  = P_base  + P_trend

#         Q_phys = self.physics_strategy.compute_Q_phys(PI_future, P_future, t_grid)
#         oil_scaled = (Q_phys - self.oil_mean) / self.oil_std

#         if self._diag_enabled and tf.equal(tf.math.mod(self.iteration, self._diag_period), 0):
#             # The 'diagnostic' method is not defined here, but would be needed for diagnostics
#             pass

#         self.iteration.assign_add(1)
#         return oil_scaled

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict

class DarcyTimeDecoderLayer(tf.keras.layers.Layer, PhysicsParameterMixin):
    """
    Darcy time decoder with Boundary Continuity (Level Anchor).

    Anchor level is computed from the *driver history* (Q) in target scale,
    using either the mean or the median of a tail window.

    - anchor_stat: "mean" or "median"  (default: "mean")
    - anchor_window: number of last timesteps to use; if <= 0, use full history
    - warm_steps: number of forecast steps where the anchor correction is active
    - gate_kind: "exp" (default) or "linear"

    Diagnostic knobs:
    - diagnostics.enabled / diagnostics.period: global on/off + frequency
    - anchor_diag.enabled / anchor_diag.period: specific to level anchor
    - anchor_trace_steps: how many forecast steps to print (default: 10)
    - anchor_trace_batch_index: which batch element to trace (default: 0)
    """

    def __init__(self, scaler_X, scaler_target, strategy_config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.scaler_X = scaler_X
        self.scaler_target = scaler_target
        self.strategy_config = dict(strategy_config or {})
        self.iteration = tf.Variable(0, trainable=False, dtype=tf.int32)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, input_shape):
        # -------- General configs --------
        self._gen_horizon = int(self.strategy_config.get("forecast_horizon", 0)) or None
        self._time_mode   = self.strategy_config.get("time_mode", "relative")
        self._dt_scale    = float(self.strategy_config.get("dt_scale", 1.0))

        # -------- Diagnostics --------
        diag_cfg = dict(self.strategy_config.get("diagnostics", {}))
        self._diag_enabled = bool(diag_cfg.get("enabled", False))
        self._diag_period  = int(diag_cfg.get("period", 2000))

        anchor_diag_cfg = dict(self.strategy_config.get("anchor_diag", {}))
        self._anchor_diag_enabled = bool(anchor_diag_cfg.get("enabled", False) or self._diag_enabled)
        self._anchor_diag_period  = int(anchor_diag_cfg.get("period", self._diag_period))

        # Extra trace knobs (optional)
        self._anchor_trace_steps = int(self.strategy_config.get("anchor_trace_steps", 10))
        self._anchor_trace_batch_index = int(self.strategy_config.get("anchor_trace_batch_index", 0))

        # -------- Level Anchor knobs --------
        self._start_at_t0      = bool(self.strategy_config.get("start_at_t0", True))
        self._use_level_anchor = bool(self.strategy_config.get("use_level_anchor", True))
        self._warm_steps       = int(self.strategy_config.get("warm_steps", 10))
        self._gate_kind        = str(self.strategy_config.get("gate_kind", "exp"))

        # New, simplified anchor: statistic over tail of driver history
        self._anchor_stat    = str(self.strategy_config.get("anchor_stat", "mean")).lower()  # "mean"|"median"
        self._anchor_window  = int(self.strategy_config.get("anchor_window", 0))  # <=0 → full history

        # -------- Driver (Q) location in context slice --------
        # physics_features slice layout assumed e.g. [PI, Pwf, time, BORE_OIL_VOL]
        self._driver_chan_ctx   = int(self.strategy_config.get("driver_channel_in_context", 3))
        self._driver_feat_in_X  = int(self.strategy_config.get("driver_feature_index_in_X", -1))
        self._driver_is_tgt     = bool(self.strategy_config.get("driver_is_target_scaled", True))

        # -------- Scaling stats --------
        self.scaler_X_center_np, self.scaler_X_scale_np = get_center_and_scale(self.scaler_X, as_tf=False)
        self.oil_center_np,      self.oil_scale_np      = get_center_and_scale(self.scaler_target, as_tf=False)

        self.scaler_X_mean, self.scaler_X_std = get_center_and_scale(self.scaler_X, as_tf=True, dtype=tf.float32)
        self.oil_mean,      self.oil_std      = get_center_and_scale(self.scaler_target, as_tf=True, dtype=tf.float32)

        # Indices in scaler_X for PI and Pwf
        self.PI_idx   = 0
        self.P_wf_idx = 3

        # -------- Physics init --------
        self._physics_clip_upper_bound = 0.99
        self._init_physics()

        # -------- Context encoder + forecaster --------
        self._use_ctx = bool(self.strategy_config.get("use_context_encoder", False))
        self._ctx_src = self.strategy_config.get("context_source", "inputs")

        if self._use_ctx:
            ctx_cfg = dict(self.strategy_config.get("context_encoder", {}))
            self._ctx = make_context_encoder(ctx_cfg)

        policy   = self.strategy_config.get("forecast_policy", "hold_last")
        hist_len = input_shape[1]
        steps    = self._gen_horizon or int(hist_len)

        forecaster_cfg = dict(self.strategy_config.get("forecaster_config", {}))
        self.forecaster = make_forecaster(policy, steps=steps, config=forecaster_cfg)

        try:
            # Forecaster sees centered [PI_s, Pwf_s] of shape (B, T, 2)
            self.forecaster.build((input_shape[0], hist_len, 2))
        except Exception:
            pass

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Time grid
    # ------------------------------------------------------------------
    def _infer_dt_and_future_times(self, t_hist: tf.Tensor, steps: int) -> tf.Tensor:
        diffs   = t_hist[:, 1:] - t_hist[:, :-1]
        mean_dt = tf.reduce_mean(diffs, axis=-1, keepdims=True)
        one     = tf.ones_like(mean_dt)
        dt      = tf.where(mean_dt > 1e-8, mean_dt, one) * self._dt_scale

        if self._start_at_t0:
            k = tf.cast(tf.range(0, steps), t_hist.dtype)[tf.newaxis, :]
        else:
            k = tf.cast(tf.range(1, steps + 1), t_hist.dtype)[tf.newaxis, :]

        t_rel = k * dt
        t_abs = t_hist[:, -1:] + t_rel
        return t_rel if (self._time_mode == "relative") else t_abs

    # ------------------------------------------------------------------
    # Driver history in target scale
    # ------------------------------------------------------------------
    def _driver_hist_in_target_space(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Returns full history of the driver (Q) in target scale.
        inputs: (B, T, F_ctx). driver_channel_in_context points to Q within this slice.
        """
        drv_s = inputs[..., self._driver_chan_ctx]  # (B, T), scaled as either X or target

        if self._driver_is_tgt:
            # Driver is already on target scale (same scaler as y)
            return drv_s

        # Defensive path: if index in scaler_X is invalid, assume already scaled as target
        n_X_feats = int(self.scaler_X_center_np.shape[0])
        if self._driver_feat_in_X < 0 or self._driver_feat_in_X >= n_X_feats:
            return drv_s

        j = self._driver_feat_in_X
        drv_phys = drv_s * self.scaler_X_std[j] + self.scaler_X_mean[j]  # back to physical
        drv_tgt  = (drv_phys - self.oil_mean) / self.oil_std             # to target scale
        return drv_tgt

    # ------------------------------------------------------------------
    # Anchor level: mean/median over tail of driver history
    # ------------------------------------------------------------------
    def _compute_anchor_level(self, driver_hist_tgt: tf.Tensor) -> tf.Tensor:
        """
        driver_hist_tgt: (B, T) in target scale.
        Returns anchor_level: (B, 1) in target scale.
        """
        B = tf.shape(driver_hist_tgt)[0]
        T = tf.shape(driver_hist_tgt)[1]

        # Tail window [T-W, T)
        if self._anchor_window > 0:
            W = tf.minimum(tf.cast(self._anchor_window, tf.int32), T)
            tail = driver_hist_tgt[:, -W:]  # (B, W)
        else:
            tail = driver_hist_tgt         # full history

        if self._anchor_stat == "median":
            anchor = tf.experimental.numpy.median(tail, axis=-1, keepdims=True)
        else:
            # default: "mean"
            anchor = tf.reduce_mean(tail, axis=-1, keepdims=True)  # (B,1)

        return anchor

    # ------------------------------------------------------------------
    # Temporal gate for the offset
    # ------------------------------------------------------------------
    def _make_level_gate(self, steps_tensor, dtype) -> tf.Tensor:
        if self._warm_steps <= 0:
            return tf.zeros((1, steps_tensor), dtype=dtype)

        k    = tf.cast(tf.range(steps_tensor)[tf.newaxis, :], dtype)
        warm = tf.cast(self._warm_steps, dtype)

        if self._gate_kind == "linear":
            gate = 1.0 - k / tf.maximum(warm, 1.0)
            gate = tf.clip_by_value(gate, 0.0, 1.0)
        else:
            # Exponential decay: ~1 at t=0, ~0 around warm_steps
            ln100 = tf.math.log(tf.constant(100.0, dtype=dtype))
            tau   = tf.maximum(warm / ln100, tf.constant(1e-6, dtype=dtype))
            gate  = tf.exp(-k / tau)

        gate = tf.where(k < warm, gate, tf.zeros_like(gate))
        return gate  # (1, H)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def call(self, inputs, training: bool = False):
        # inputs: (B, T, F_ctx), assumed layout [PI_s, Pwf_s, t, Q_s, ...]
        PI_scaled = inputs[..., 0]
        P_scaled  = inputs[..., 1]
        t_hist    = inputs[..., 2]

        steps  = self._gen_horizon or tf.shape(t_hist)[1]
        t_grid = self._infer_dt_and_future_times(t_hist, steps)

        # Centered history for the forecaster
        last_pi_s = PI_scaled[:, -1:]
        last_p_s  = P_scaled[:, -1:]
        hist_s    = tf.stack([PI_scaled, P_scaled], axis=-1)  # (B,T,2)
        last_s    = tf.stack([last_pi_s, last_p_s], axis=-1)  # (B,1,2)
        centered  = hist_s - last_s                           # (B,T,2)

        if self._use_ctx:
            if self._ctx_src == "hist":
                t_exp = tf.expand_dims(t_hist, axis=-1)
                ctx_in = tf.concat([hist_s, t_exp], axis=-1)
            else:
                ctx_in = inputs
            context_vector   = self._ctx(ctx_in, training=training)
            forecaster_input = (centered, context_vector)
        else:
            forecaster_input = centered

        # 1) Forecast drivers in scaled space (centered)
        PI_trend_s, P_trend_s = self.forecaster(forecaster_input, training=training)  # (B,H)

        # 2) Reconstruct physical PI, Pwf
        PI_base = invert_feature_scaling(last_pi_s, self.scaler_X_mean, self.scaler_X_std, self.PI_idx)  # (B,1)
        P_base  = invert_feature_scaling(last_p_s,  self.scaler_X_mean, self.scaler_X_std, self.P_wf_idx)

        PI_trend  = PI_trend_s * self.scaler_X_std[self.PI_idx]
        P_trend   = P_trend_s  * self.scaler_X_std[self.P_wf_idx]
        PI_future = PI_base + PI_trend
        P_future  = P_base  + P_trend

        # 3) Physics (Darcy) → physical Q
        Q_phys = self.physics_strategy.compute_Q_phys(PI_future, P_future, t_grid)  # (B,H)

        # 4) Map to target scale
        y_base = (Q_phys - self.oil_mean) / self.oil_std  # (B,H) in target scale

        # ------------------------------------------------------------------
        # Level Anchor
        # ------------------------------------------------------------------
        if self._use_level_anchor and self._warm_steps > 0:
            # Driver history in target scale
            driver_hist_tgt = self._driver_hist_in_target_space(inputs)  # (B,T)

            # Anchor level: statistic over tail window
            anchor_level = self._compute_anchor_level(driver_hist_tgt)   # (B,1)

            # Offset at t=0 and gate over horizon
            offset0 = anchor_level - y_base[:, :1]                       # (B,1)
            H       = tf.shape(y_base)[1]
            gate    = self._make_level_gate(H, y_base.dtype)             # (1,H)
            gate    = tf.broadcast_to(gate, tf.shape(y_base))            # (B,H)

            y_out = y_base + gate * offset0                              # (B,H)

            # ---------------- Detailed diagnostics ----------------
            if self._anchor_diag_enabled and self._anchor_diag_period > 0:
                cond = tf.equal(tf.math.mod(self.iteration, self._anchor_diag_period), 0)

                def _print_diag():
                    # Choose batch element and number of steps to trace
                    B = tf.shape(y_base)[0]
                    H = tf.shape(y_base)[1]
                    b0 = tf.minimum(tf.cast(self._anchor_trace_batch_index, tf.int32), B - 1)
                    K  = tf.minimum(tf.cast(self._anchor_trace_steps, tf.int32), H)

                    base_b0 = y_base[b0, :K]
                    out_b0  = y_out[b0, :K]
                    gate_b0 = gate[b0, :K]
                    anchor0 = anchor_level[b0, 0]

                    gap0   = tf.abs(out_b0[0] - anchor0)
                    off_val = tf.reduce_mean(offset0)

                    # Physical units
                    oil_mean0 = self.oil_mean[0]
                    oil_std0  = self.oil_std[0]
                    base_phys = base_b0 * oil_std0 + oil_mean0
                    out_phys  = out_b0  * oil_std0 + oil_mean0
                    anchor_phys0 = anchor0 * oil_std0 + oil_mean0

                    tf.print(
                        "\n[AnchorDiag] Iter:", self.iteration,
                        "| stat:", self._anchor_stat,
                        "| Gap@Step0:", gap0,
                        "| AvgOffset:", off_val,
                        "| WarmSteps:", self._warm_steps,
                        "| H:", H,
                        "| GateStart:", gate_b0[0],
                        "| GateEnd:", gate_b0[-1]
                    )
                    tf.print(
                        "[AnchorDiag] sample", b0,
                        "(scaled) anchor:", anchor0,
                        " first", K, "y_base:", base_b0,
                        " first", K, "y_out:", out_b0,
                        " gate:", gate_b0
                    )
                    tf.print(
                        "[AnchorDiag] sample", b0,
                        "(phys) anchor:", anchor_phys0,
                        " first", K, "y_base_phys:", base_phys,
                        " first", K, "y_out_phys:", out_phys
                    )
                    return 0

                # tf.cond(cond, _print_diag, lambda: 0)

            oil_scaled = y_out
        else:
            oil_scaled = y_base

        self.iteration.assign_add(1)
        return oil_scaled

# =============================================================================
# Diagnostics helpers
# =============================================================================

def log_tensor_stats(name: str, x: tf.Tensor):
    """Lightweight tf.print stats for debugging (mean, std, min, max)."""
    tf.print(
        "[latent_diag]", name,
        "shape=", tf.shape(x),
        "mean=", tf.reduce_mean(x),
        "std=", tf.math.reduce_std(x),
        "min=", tf.reduce_min(x),
        "max=", tf.reduce_max(x),
    )


# =====================================================
# 4. TrendBlock: Data-Driven Residual Trend Module
# =====================================================
class TrendBlock(tf.keras.layers.Layer):
    def __init__(self, degree: int = 2, forecast_horizon: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.forecast_horizon = forecast_horizon

    def build(self, input_shape):
        self.theta_layer = Dense(
            self.degree + 1,
            kernel_initializer=HeNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="trend_theta",
        )
        t = np.linspace(0, 1, self.forecast_horizon)
        basis = np.vstack([t**i for i in range(self.degree + 1)]).T
        self.basis = tf.constant(basis, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        theta = self.theta_layer(inputs)
        forecast = tf.matmul(theta, self.basis, transpose_b=True)
        return forecast

    def get_config(self):
        config = super().get_config()
        config.update({
            "degree": self.degree,
            "forecast_horizon": self.forecast_horizon,
        })
        return config

# =============================================================================
# Functional physics (stateless) for latent-parameter path
# =============================================================================

def compute_Q_phys_functional(
    strategy_name: str,
    params: Dict[str, tf.Tensor],
    PI_future: tf.Tensor,
    P_wf_future: tf.Tensor,
    t_grid: tf.Tensor,
) -> tf.Tensor:
    """
    Stateless physics call that reuses the existing strategy implementations.

    Parameters
    ----------
    strategy_name : name of the physics strategy (e.g. 'pressure_ensemble').
    params        : dict with tensors, typically (B,1) each:
                    {"P_reservoir", "decay_rate", "b_factor", ...}
    PI_future     : (B, N) physical PI.
    P_wf_future   : (B, N) physical Pwf.
    t_grid        : (B, N) time grid.

    Returns
    -------
    Q_phys : (B, N) physical rate.
    """
    # Ensure we pass the "absolute_value" flag like in the legacy layer
    strategy_params = dict(params)
    strategy_params.setdefault("absolute_value", True)

    strategy = physics_strategy_factory(strategy_name, strategy_params)
    return strategy.compute_Q_phys(PI_future, P_wf_future, t_grid)



# =============================================================================
# 5. Physics Strategy Factory (kept)
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
        return DiffusivityDecayStrategy(params['P_reservoir'], kappa=0.1)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")



# =============================================================================
# 5. Strategy Pattern for Driver Forecasting (new)
# =============================================================================
class BaseDriverForecaster(tf.keras.layers.Layer):
    """Returns residual/trend (in input scaling) for (PI, P_wf)."""
    def __init__(self, steps: int, config: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.steps = int(steps)
        self.config = dict(config or {})

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError


class OriginalPINNForecaster(BaseDriverForecaster):
    """Sentinel forecaster for legacy path; not used in context mode.
    Returns zeros if called (same residual as hold-last)."""
    def call(self, inputs, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        b = tf.shape(inputs)[0]
        z = tf.zeros([b, self.steps], dtype=inputs.dtype)
        return z, z


class HoldLastForecaster(BaseDriverForecaster):
    def call(self, inputs, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        b = tf.shape(inputs)[0]
        z = tf.zeros([b, self.steps], dtype=inputs.dtype)
        return z, z


# =============================================================================
# 6.1 Context Encoder Factory (LSTM/GRU/TCN) -> embedding vetorial
# =============================================================================
class ContextEncoder(tf.keras.layers.Layer):
    """
    Encoder temporal leve que produz um vetor de contexto a partir de uma série (B, M, C).
    Suporta: 'lstm', 'gru', 'tcn'. Usa defaults seguros e é estateless entre chamadas.
    """
    def __init__(self, enc_type: str = "lstm", units: int = 64, layers: int = 1,
                 dropout: float = 0.0, kernel_size: int = 5, dilations=None, **kwargs):
        super().__init__(**kwargs)
        self.enc_type   = enc_type.lower()
        self.units      = int(units)
        self.layers     = int(layers)
        self.dropout    = float(dropout)
        self.kernel_size= int(kernel_size)
        self.dilations  = list(dilations or [1, 2, 4, 8])

        self._built = False
        self._stack = []

    def build(self, input_shape):
        if self._built:
            return
        if self.enc_type in ("lstm", "gru"):
            RNN = tf.keras.layers.LSTM if self.enc_type == "lstm" else tf.keras.layers.GRU
            for i in range(self.layers):
                # return_sequences=True nas camadas intermediárias
                self._stack.append(
                    RNN(self.units,
                        return_sequences=(i < self.layers - 1),
                        dropout=self.dropout,
                        name=f"{self.enc_type}_{i}")
                )
        elif self.enc_type == "tcn":
            # pilha causal + GAP -> Dense(units)
            self._in   = tf.keras.layers.Conv1D(filters=self.units, kernel_size=1, name="ctx_tcn_in")
            self._blocks = []
            for i, d in enumerate(self.dilations):
                self._blocks.append({
                    "conv": tf.keras.layers.Conv1D(
                        filters=self.units, kernel_size=self.kernel_size,
                        dilation_rate=int(d), padding="causal",
                        activation="relu", name=f"ctx_tcn_conv_{i}"
                    ),
                    "drop": tf.keras.layers.Dropout(self.dropout, name=f"ctx_tcn_drop_{i}"),
                    "proj": tf.keras.layers.Conv1D(filters=self.units, kernel_size=1, name=f"ctx_tcn_proj_{i}"),
                })
            self._gap  = tf.keras.layers.GlobalAveragePooling1D(name="ctx_tcn_gap")
            self._head = tf.keras.layers.Dense(self.units, activation=None, name="ctx_tcn_head")
        else:
            raise ValueError(f"Unknown context encoder type: {self.enc_type}")

        self._built = True
        super().build(input_shape)

    def call(self, x, training=False):
        # x: (B, M, C)
        if self.enc_type in ("lstm", "gru"):
            h = x
            for layer in self._stack:
                h = layer(h, training=training)
            return h  # (B, units)
        else:
            h = self._in(x)
            for b in self._blocks:
                res = h
                h = b["conv"](h, training=training)
                h = b["drop"](h, training=training)
                if res.shape[-1] != h.shape[-1]:
                    res = b["proj"](res)
                h = h + res
            h = self._gap(h)
            h = self._head(h)
            return h  # (B, units)

def make_context_encoder(cfg: dict) -> ContextEncoder:
    cfg = dict(cfg or {})
    enc_type = cfg.get("type", "lstm")
    units    = int(cfg.get("units", 64))
    layers   = int(cfg.get("layers", 1))
    dropout  = float(cfg.get("dropout", 0.0))
    kernel   = int(cfg.get("kernel_size", 5))
    dil      = list(cfg.get("dilations", [1, 2, 4, 8]))
    return ContextEncoder(enc_type=enc_type, units=units, layers=layers,
                          dropout=dropout, kernel_size=kernel, dilations=dil, name="context_encoder")


class TCNResidualForecaster(BaseDriverForecaster):
    def build(self, input_shape):
        cfg = self.config
        filters = int(cfg.get("filters", 32))
        kernel_size = int(cfg.get("kernel_size", 5))
        dilations = list(cfg.get("dilations", [2, 4, 8, 16, 32, 64]))
        dropout = float(cfg.get("dropout", 0.0))

        self._in = tf.keras.layers.Conv1D(filters, 1, name="tcn_in")
        self._blocks = []
        for i, d in enumerate(dilations):
            self._blocks.append({
                "conv": tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                               dilation_rate=int(d), padding="causal",
                                               activation="relu", name=f"tcn_conv_{i}"),
                "drop": tf.keras.layers.Dropout(dropout, name=f"tcn_drop_{i}"),
                "proj": tf.keras.layers.Conv1D(filters=filters, kernel_size=1, name=f"tcn_proj_{i}"),
            })
        self._head_series = tf.keras.layers.Conv1D(2, 1, name="tcn_head_series")
        self._flat        = tf.keras.layers.Flatten(name="tcn_flat")
        # head final projeta (features_series + context) -> steps*2
        self._concat      = tf.keras.layers.Concatenate(name="tcn_concat_ctx")
        self._time        = tf.keras.layers.Dense(self.steps * 2, name="tcn_time")
        super().build(input_shape)

    def call(self, inputs, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        if isinstance(inputs, (list, tuple)):
            series, context = inputs
        else:
            series, context = inputs, None

        x = self._in(series)
        for b in self._blocks:
            res = x
            x = b["conv"](x, training=training)
            x = b["drop"](x, training=training)
            if res.shape[-1] != x.shape[-1]:
                res = b["proj"](res)
            x = x + res
        x = self._head_series(x)
        x = self._flat(x)

        if context is not None:
            x = self._concat([x, context])

        x = self._time(x)
        bs = tf.shape(x)[0]
        x = tf.reshape(x, [bs, self.steps, 2])
        return x[..., 0], x[..., 1]


class MLPResidualForecaster(BaseDriverForecaster):
    def build(self, input_shape):
        cfg = self.config
        hidden = list(cfg.get("hidden_units", [128, 64]))
        dropout = float(cfg.get("dropout", 0.1))
        act = cfg.get("activation", "relu")
        l2_reg = float(cfg.get("l2_reg", 1e-4))
        max_norm_val = float(cfg.get("max_norm", 3.0))

        self._flat   = tf.keras.layers.Flatten(name="mlp_flat")
        self._concat = tf.keras.layers.Concatenate(name="mlp_concat_ctx")

        self._layers = [
            tf.keras.layers.Dense(h, activation=act,
                                  kernel_initializer=HeNormal(),
                                  kernel_regularizer=L2(l2_reg),
                                  activity_regularizer=L2(l2_reg),
                                  kernel_constraint=max_norm(max_norm_val),
                                  name=f"mlp_{i}")
            for i, h in enumerate(hidden)
        ]
        self._drop = tf.keras.layers.Dropout(dropout, name="mlp_drop")
        self._head = tf.keras.layers.Dense(self.steps * 2, name="mlp_head")
        super().build(input_shape)

    def call(self, inputs, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        if isinstance(inputs, (list, tuple)):
            series, context = inputs
        else:
            series, context = inputs, None

        x = self._flat(series)
        if context is not None:
            x = self._concat([x, context])

        for layer in self._layers:
            x = layer(x)
        x = self._drop(x, training=training)
        x = self._head(x)

        bs = tf.shape(x)[0]
        x = tf.reshape(x, [bs, self.steps, 2])
        return x[..., 0], x[..., 1]

# =============================================================================
# 7. Forecaster registry
# =============================================================================
FORECASTER_REGISTRY: Dict[str, Type[BaseDriverForecaster]] = {}

def register_forecaster(name: str):
    def _decorator(cls: Type[BaseDriverForecaster]):
        FORECASTER_REGISTRY[name] = cls
        return cls
    return _decorator


register_forecaster("original_pinn")(OriginalPINNForecaster)
register_forecaster("hold_last")(HoldLastForecaster)
register_forecaster("tcn_residual_drivernet")(TCNResidualForecaster)
register_forecaster("mlp_residual_drivernet")(MLPResidualForecaster)


def make_forecaster(name: str, steps: int, config: Optional[Dict] = None) -> BaseDriverForecaster:
    try:
        cls = FORECASTER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown forecaster '{name}'. Registered: {list(FORECASTER_REGISTRY)}") from e
    return cls(steps=steps, config=config)






