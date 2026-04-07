# """
# Contains the high-level builder function 'create_model' for constructing
# the complete PINN model, including data-driven and physics-informed paths.
# """
# from typing import Optional, Tuple, Dict
# import tensorflow as tf
# from tensorflow.keras import Model, Input
# from tensorflow.keras.layers import Lambda, Dense, Concatenate, GlobalAveragePooling1D

# # Project utilities
# from utils.utilities import invert_feature_scaling, get_center_and_scale

# # Import all necessary components from the local components file
# from .darcy_components import (
#     TrendBlock,
#     DarcyPhysicsLayer,
#     DarcyTimeDecoderLayer,
#     PhysicsParameterEncoder,
#     PhysicsDecoder,
#     infer_t_grid,
# )


# # =============================================================================
# # 1. Builder helper: choose legacy or context physics by policy
# # =============================================================================

# def build_physics_block(
#     scaler_X,
#     scaler_target,
#     policy: str,
#     strategy_config: Dict,
#     name: str = "physics_block",
# ) -> tf.keras.layers.Layer:
#     if policy == "original_pinn":  # LEGACY path
#         return DarcyPhysicsLayer(
#             scaler_X=scaler_X,
#             scaler_target=scaler_target,
#             strategy_config=strategy_config,
#             name=name,
#         )
#     # CONTEXT path (driver generation inside the layer)
#     return DarcyTimeDecoderLayer(
#         scaler_X=scaler_X,
#         scaler_target=scaler_target,
#         strategy_config=strategy_config,
#         name=name,
#     )


# def create_model(
#     input_shape: Tuple[int, int],
#     horizon: int,
#     scaler_X,
#     scaler_target,
#     strategy_config: Dict = {"strategy_name": "pressure_ensemble"},
#     trend_degree: int = 2,
#     phase: str = "balanced",
#     freeze_trend: bool = False,
#     freeze_physics: bool = False,
#     fusion_type: str = "pin",
#     extractor_config: Optional[Dict] = None,
#     fuser_config: Optional[Dict] = None,
#     name: str = "Seq2PIN",
# ) -> Model:
#     """Builds a model with selectable physics path and fusion.

#     Choices (strategy_config):
#       - forecast_policy:
#           'original_pinn' (legacy DarcyPhysicsLayer)
#           'hold_last' | 'tcn_residual_drivernet' | 'mlp_residual_drivernet' (context)
#           'latent_params' (encoder + stateless physics decoder)
#       - context_length: int (only for context/latent paths; default = horizon)

#     Fusion types:
#       - 'trend' | 'pin' | 'average' | 'concat_dense'
#     """
#     timesteps, feats = input_shape
#     inputs = Input(shape=(timesteps, feats), name="all_features")

#     # Indices for features (keep original mapping)
#     DATA_INDICES = [0, 2, 7]
#     PHYSICS_INDICES = [0, 3, 5, 7]  # [PI_s, Pwf_s, t_feat, Q_s]

#     data_features = Lambda(
#         lambda x: tf.gather(x, DATA_INDICES, axis=-1),
#         name="data_features",
#     )(inputs)
#     physics_features = Lambda(
#         lambda x: tf.gather(x, PHYSICS_INDICES, axis=-1),
#         name="physics_features",
#     )(inputs)

#     # ---------------- Trend Branch ----------------
#     summary_vector = GlobalAveragePooling1D(name="avg_pool_trend")(data_features)
#     trend_block = TrendBlock(
#         degree=trend_degree,
#         forecast_horizon=horizon,
#         name="trend_block",
#     )
#     trend_forecast = trend_block(summary_vector)

#     # ---------------- Physics Branch ----------------
#     context_len = int(strategy_config.get("context_length", horizon))
#     physics_context = Lambda(
#         lambda x: x[:, -context_len:, :],
#         name="physics_context",
#     )(physics_features)
#     physics_horizon = Lambda(
#         lambda x: x[:, -horizon:, :],
#         name="physics_horizon",
#     )(physics_features)

#     # Strategy config (copy to avoid side-effects)
#     strat_cfg: Dict = dict(strategy_config)
#     strat_cfg.setdefault("forecast_horizon", horizon)

#     # Determine physics policy, preserving user choice if present
#     physics_policy = strat_cfg.get("forecast_policy")
#     physics_policy = "latent_params"
#     if physics_policy is None:
#         # Default: context path with MLP forecaster (current behavior)
#         physics_policy = "mlp_residual_drivernet"
#         strat_cfg["forecast_policy"] = physics_policy

#     # Context-specific defaults (only if using context policies)
#     if physics_policy in {"hold_last", "tcn_residual_drivernet", "mlp_residual_drivernet"}:
#         strat_cfg.setdefault("use_context_encoder", True)
#         strat_cfg.setdefault(
#             "context_encoder",
#             {"type": "gru", "units": 32, "layers": 1, "dropout": 0.1},
#         )
#         strat_cfg.setdefault("context_source", "hist")  # uses [PI_s, Pwf_s, t] scaled
#         strat_cfg.setdefault(
#             "forecaster_config",
#             {
#                 "hidden_units": [128, 64],
#                 "dropout": 0.1,
#                 "l2_reg": 1e-4,
#                 "max_norm": 3.0,
#             },
#         )

#         strat_cfg.setdefault("use_level_anchor", True)
#         strat_cfg.setdefault("start_at_t0", True)
#         strat_cfg.setdefault("warm_steps", 10)
#         strat_cfg.setdefault("gate_kind", "exp")

#         # driver location in context slice
#         strat_cfg.setdefault("driver_channel_in_context", 3)
#         strat_cfg.setdefault("driver_feature_index_in_X", -1)  # Q already in target scale
#         strat_cfg.setdefault("driver_is_target_scaled", True)

#         # anchor knobs
#         strat_cfg.setdefault("anchor_stat", "mean")
#         strat_cfg.setdefault("anchor_hp_lambda", 160000.0)
#         strat_cfg.setdefault("anchor_diag", {"enabled": True, "period": 500})

#     # ---------------- Physics path selection ----------------

#     # Precompute scalers for latent path (PI/Pwf in physical units)
#     scaler_X_mean, scaler_X_std = get_center_and_scale(
#         scaler_X, as_tf=True, dtype=tf.float32
#     )
#     PI_IDX = 0   # index in original X scaler for PI
#     PWF_IDX = 3  # index in original X scaler for Pwf

#     if physics_policy == "latent_params":
#         # ----- Latent-parameter physics path -----
#         # physics_context layout: [PI_s, Pwf_s, t_feat, Q_s, ...]
#         pi_hist_s = Lambda(
#             lambda x: x[..., 0],
#             name="latent_pi_hist_s",
#         )(physics_context)
#         p_hist_s = Lambda(
#             lambda x: x[..., 1],
#             name="latent_p_hist_s",
#         )(physics_context)
#         t_hist = Lambda(
#             lambda x: x[..., 2],
#             name="latent_t_hist",
#         )(physics_context)

#         # Future time grid from history
#         def _latent_t_grid(th):
#             return infer_t_grid(
#                 t_hist=th,
#                 steps=horizon,
#                 mode=strat_cfg.get("time_mode", "relative"),
#                 dt_scale=float(strat_cfg.get("dt_scale", 1.0)),
#                 start_at_t0=bool(strat_cfg.get("start_at_t0", True)),
#             )

#         t_grid = Lambda(_latent_t_grid, name="latent_t_grid")(t_hist)

#         # Hold-last drivers in physical units, tiled over horizon
#         def _pi_future(x):
#             last_pi_s = x[:, -1:]  # (B,1)
#             pi_base = invert_feature_scaling(
#                 last_pi_s, scaler_X_mean, scaler_X_std, PI_IDX
#             )  # physical
#             return tf.tile(pi_base, [1, horizon])  # (B,H)

#         def _p_future(x):
#             last_p_s = x[:, -1:]  # (B,1)
#             p_base = invert_feature_scaling(
#                 last_p_s, scaler_X_mean, scaler_X_std, PWF_IDX
#             )  # physical
#             return tf.tile(p_base, [1, horizon])  # (B,H)

#         PI_future = Lambda(_pi_future, name="latent_PI_future")(pi_hist_s)
#         P_future = Lambda(_p_future, name="latent_P_future")(p_hist_s)

#         # Physics parameter encoder on context
#         latent_encoder = PhysicsParameterEncoder(
#             scaler_X=scaler_X,
#             scaler_target=scaler_target,
#             strategy_config=strat_cfg,
#             rnn_units=int(strat_cfg.get("latent_units", 64)),
#             name="latent_param_encoder",
#             diag_config={"enabled": True, "period": 500},
#         )
#         latent_params = latent_encoder(physics_context)

#         # Stateless physics decoder (same strategy_name as existing paths)
#         physics_decoder = PhysicsDecoder(
#             scaler_target=scaler_target,
#             strategy_name=strat_cfg.get("strategy_name", "pressure_ensemble"),
#             name="latent_physics_decoder",
#             diag_config={"enabled": True, "period": 500}
#         )
#         physics_forecast = physics_decoder(
#             latent_params, PI_future, P_future, t_grid
#         )

#         # For consistency with later freezing logic
#         physics_block = None

#     else:
#         # ----- Existing legacy/context physics paths -----
#         if physics_policy == "original_pinn":
#             physics_block_in = physics_horizon  # legacy expects N-step inputs already present
#         else:
#             physics_block_in = physics_context  # context mode will generate N steps from context

#         physics_block = build_physics_block(
#             scaler_X=scaler_X,
#             scaler_target=scaler_target,
#             policy=physics_policy,
#             strategy_config=strat_cfg,
#             name="physics_block",
#         )
#         physics_forecast = physics_block(physics_block_in)

#     # ---------------- Freezing ----------------
#     trend_block.trainable = not freeze_trend
#     if physics_policy == "latent_params":
#         latent_encoder.trainable = not freeze_physics
#         physics_decoder.trainable = not freeze_physics
#     else:
#         physics_block.trainable = not freeze_physics

#     # ---------------- Fusion ----------------
#     ft = fusion_type.lower()
#     if ft == "pin":
#         outputs = physics_forecast
#         model_name = "physics_block"
#     elif ft == "average":
#         outputs = Lambda(
#             lambda x: 0.5 * (x[0] + x[1]),
#             name="avg_fusion",
#         )([trend_forecast, physics_forecast])
#         model_name = "avg_fusion"
#     elif ft == "concat_dense":
#         combined = Concatenate(name="fusion_concat")(
#             [trend_forecast, physics_forecast]
#         )
#         hidden1 = Dense(32, activation="relu", name="fusion_dense_1")(combined)
#         outputs = Dense(
#             horizon,
#             activation="linear",
#             name="concat_dense",
#         )(hidden1)
#         model_name = "concat_dense"
#     else:
#         raise ValueError(f"Unknown fusion_type: {fusion_type}")

#     model = Model(inputs, outputs, name=model_name)
#     return model



# def make_latent_extrapolation_model(
#     trained_model: Model,
#     scaler_X,
#     scaler_target,
#     strategy_config: Dict,
#     steps: int,
#     name: str = "Seq2PIN_latent_extrap",
# ) -> Model:
#     """
#     Extrapolador latente que:

#       - REUTILIZA o mesmo PhysicsDecoder treinado ('latent_physics_decoder'),
#         incluindo o affine_guard (a, b).
#       - REUTILIZA a mesma strategy_config do encoder (time_mode, dt_scale,
#         start_at_t0, context_length, etc.).
#       - Usa infer_t_grid exatamente como no caminho de treino, apenas
#         mudando o número de steps.

#     Objetivo: se steps == H, as previsões extrapoladas devem ser o mais
#     próximas possível do caminho de treino; qualquer diferença passa a ser
#     atribuída sobretudo à forma de stitching / reconstrução, não a uma
#     física diferente.
#     """
#     import tensorflow as tf
#     from tensorflow.keras.layers import Lambda, Input
#     from tensorflow.keras import Model
#     from utils.utilities import invert_feature_scaling, get_center_and_scale
#     from .darcy_components import PhysicsDecoder, infer_t_grid  # noqa: F401

#     if steps <= 0:
#         raise ValueError(f"`steps` must be positive, got {steps}")

#     # ------------------------------------------------------------------
#     # 1) Encoder latente + strategy_config herdada do treino
#     # ------------------------------------------------------------------
#     try:
#         latent_encoder = trained_model.get_layer("latent_param_encoder")
#     except ValueError as e:
#         raise ValueError(
#             "Expected a layer 'latent_param_encoder' (physics_policy='latent_params')."
#         ) from e

#     # strategy_config ORIGINAL usada no treino do encoder
#     cfg_from_model = {}
#     if hasattr(latent_encoder, "strategy_config"):
#         try:
#             cfg_from_model = dict(latent_encoder.strategy_config or {})
#         except Exception:
#             cfg_from_model = {}

#     # strategy_config efetiva = original do modelo + overrides opcionais
#     strat_cfg = dict(cfg_from_model)
#     strat_cfg.update(strategy_config or {})

#     # ------------------------------------------------------------------
#     # 1b) PhysicsDecoder treinado (mesmo usado no caminho de treino/val)
#     # ------------------------------------------------------------------
#     try:
#         physics_decoder_ref = trained_model.get_layer("latent_physics_decoder")
#     except ValueError as e:
#         raise ValueError(
#             "Expected a layer 'latent_physics_decoder' in trained_model "
#             "(needed to reuse the same PhysicsDecoder in extrapolation)."
#         ) from e

#     inferred_strategy = getattr(physics_decoder_ref, "strategy_name", None)
#     if inferred_strategy is None:
#         try:
#             cfg_dec = physics_decoder_ref.get_config()
#             inferred_strategy = cfg_dec.get("strategy_name", None)
#         except Exception:
#             inferred_strategy = None
#     strategy_name = inferred_strategy or strat_cfg.get("strategy_name") or "arps"

#     # ------------------------------------------------------------------
#     # 2) Entrada (B, T, F) - igual ao modelo treinado
#     # ------------------------------------------------------------------
#     if not isinstance(trained_model.input_shape, tuple) or len(trained_model.input_shape) != 3:
#         raise ValueError(f"Expected input (batch, T, F). Got {trained_model.input_shape!r}")
#     _, timesteps, n_feats = trained_model.input_shape

#     inputs = Input(shape=(timesteps, n_feats), name="all_features_extrap")

#     # ------------------------------------------------------------------
#     # 3) Fatiamento de features de física e contexto
#     # ------------------------------------------------------------------
#     # Mantém mesma convenção usada em create_model:
#     #   PHYSICS_INDICES = [0, 3, 5, 7] = [PI_s, Pwf_s, t_feat, Q_s]
#     PHYSICS_INDICES = [0, 3, 5, 7]
#     PI_IDX  = 0
#     PWF_IDX = 3

#     physics_features = Lambda(
#         lambda x: tf.gather(x, PHYSICS_INDICES, axis=-1),
#         name="physics_features_extrap",
#     )(inputs)

#     # Mesmo context_length da config de treino (ou timesteps se não houver)
#     context_len = int(strat_cfg.get("context_length", timesteps))
#     physics_context = Lambda(
#         lambda x: x[:, -context_len:, :],
#         name="physics_context_extrap",
#     )(physics_features)

#     # ------------------------------------------------------------------
#     # 4) Históricos em escala NORMALIZADA (igual ao modelo base)
#     # ------------------------------------------------------------------
#     pi_hist_s = Lambda(lambda x: x[..., 0], name="latent_pi_hist_s_extrap")(physics_context)
#     p_hist_s  = Lambda(lambda x: x[..., 1], name="latent_p_hist_s_extrap")(physics_context)
#     t_hist_s  = Lambda(lambda x: x[..., 2], name="latent_t_hist_s_extrap")(physics_context)

#     # ------------------------------------------------------------------
#     # 5) Grade temporal via infer_t_grid (MESMO utilitário do treino)
#     # ------------------------------------------------------------------
#     time_mode   = strat_cfg.get("time_mode", "relative")
#     dt_scale    = float(strat_cfg.get("dt_scale", 1.0))
#     start_at_t0 = bool(strat_cfg.get("start_at_t0", True))

#     def _latent_t_grid(th_scaled):
#         """
#         Usa infer_t_grid sobre o t_feat na mesma escala usada no treino.
#         Apenas muda 'steps' para estender o horizonte.
#         """
#         return infer_t_grid(
#             t_hist=th_scaled,
#             steps=steps,
#             mode=time_mode,
#             dt_scale=dt_scale,
#             start_at_t0=start_at_t0,
#         )

#     t_grid = Lambda(_latent_t_grid, name=f"latent_t_grid_{steps}")(t_hist_s)

#     # ------------------------------------------------------------------
#     # 6) Drivers hold-last em UNIDADES FÍSICAS (PI, Pwf)
#     # ------------------------------------------------------------------
#     scaler_X_mean, scaler_X_std = get_center_and_scale(
#         scaler_X, as_tf=True, dtype=tf.float32
#     )

#     def _last_phys_from_scaled(x_s, feat_idx):
#         last_s = x_s[:, -1:]  # (B,1)
#         return invert_feature_scaling(last_s, scaler_X_mean, scaler_X_std, feat_idx)

#     PI_last  = Lambda(lambda x: _last_phys_from_scaled(x, PI_IDX),  name="dbg_PI_last_phys")(pi_hist_s)
#     PWF_last = Lambda(lambda x: _last_phys_from_scaled(x, PWF_IDX), name="dbg_PWF_last_phys")(p_hist_s)

#     PI_future = Lambda(
#         lambda x: tf.tile(x, [1, steps]),
#         name=f"latent_PI_future_{steps}"
#     )(PI_last)

#     P_future = Lambda(
#         lambda x: tf.tile(x, [1, steps]),
#         name=f"latent_P_future_{steps}"
#     )(PWF_last)

#     # ------------------------------------------------------------------
#     # 7) Encoder latente + MESMO decoder físico treinado
#     # ------------------------------------------------------------------
#     latent_params = latent_encoder(physics_context, training=False)
#     physics_forecast = physics_decoder_ref(
#         latent_params, PI_future, P_future, t_grid
#     )

#     # ------------------------------------------------------------------
#     # 8) Logs de sanidade do tempo (debug leve)
#     # ------------------------------------------------------------------
#     def _dbg(th_s, tg, pi, pwf):
#         dt = th_s[:, 1:] - th_s[:, :-1]
#         dtm = tf.reduce_mean(dt)
#         tmax = tf.reduce_mean(tg[:, -1])
#         return tf.print(
#             "[latent_extrap] strategy=", strategy_name,
#             " time_mode=", time_mode,
#             " dt_scale=", dt_scale,
#             " start_at_t0=", start_at_t0,
#             " dt_mean_scaled=", dtm,
#             " steps=", steps,
#             " t_max≈", tmax,
#             " PI_last≈", tf.reduce_mean(pi),
#             " Pwf_last≈", tf.reduce_mean(pwf),
#             summarize=-1
#         )

#     _ = Lambda(
#         lambda xs: (_dbg(xs[0], xs[1], xs[2], xs[3]), xs[0])[1],
#         name="dbg_latent_time"
#     )([t_hist_s, t_grid, PI_last, PWF_last])

#     return Model(inputs, physics_forecast, name=name)


"""
Contains the high-level builder function 'create_model' for constructing
the complete PINN model, including data-driven and physics-informed paths.
"""
from typing import Optional, Tuple, Dict
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Lambda, Dense, Concatenate, GlobalAveragePooling1D

# Project utilities
from utils.utilities import invert_feature_scaling, get_center_and_scale

# Import all necessary components from the local components file
from .darcy_components import (
    TrendBlock,
    DarcyPhysicsLayer,
    DarcyTimeDecoderLayer,
    PhysicsParameterEncoder,
    PhysicsDecoder,
    infer_t_grid,
)


# =============================================================================
# 1. Builder helper: choose legacy or context physics by policy
# =============================================================================


def build_physics_block(
    scaler_X,
    scaler_target,
    policy: str,
    strategy_config: Dict,
    name: str = "physics_block",
) -> tf.keras.layers.Layer:
    """
    Factory that returns the appropriate physics block given a policy.

    Parameters
    ----------
    scaler_X, scaler_target : scalers
        Feature and target scalers used for (de)normalization.
    policy : str
        One of:
          - "original_pinn"           -> DarcyPhysicsLayer (legacy path)
          - "hold_last"               -> DarcyTimeDecoderLayer (context)
          - "tcn_residual_drivernet"  -> DarcyTimeDecoderLayer (context)
          - "mlp_residual_drivernet"  -> DarcyTimeDecoderLayer (context)
    strategy_config : dict
        Physics/forecast configuration.
    name : str
        Layer name.

    Returns
    -------
    tf.keras.layers.Layer
        Configured physics layer.
    """
    if policy == "original_pinn":  # LEGACY path
        return DarcyPhysicsLayer(
            scaler_X=scaler_X,
            scaler_target=scaler_target,
            strategy_config=strategy_config,
            name=name,
        )

    # CONTEXT path (driver generation inside the layer)
    return DarcyTimeDecoderLayer(
        scaler_X=scaler_X,
        scaler_target=scaler_target,
        strategy_config=strategy_config,
        name=name,
    )


# =============================================================================
# 2. Main builder
# =============================================================================


def create_model(
    input_shape: Tuple[int, int],
    horizon: int,
    scaler_X,
    scaler_target,
    strategy_config: Dict = None,
    trend_degree: int = 2,
    phase: str = "balanced",
    freeze_trend: bool = False,
    freeze_physics: bool = False,
    fusion_type: str = "pin",
    extractor_config: Optional[Dict] = None,
    fuser_config: Optional[Dict] = None,
    name: str = "Seq2PIN",
) -> Model:
    """
    Build a model with selectable physics path and fusion.

    The physics path is controlled by `strategy_config["forecast_policy"]`:

      - "original_pinn"
          Uses DarcyPhysicsLayer (legacy PINN implementation).

      - "hold_last" | "tcn_residual_drivernet" | "mlp_residual_drivernet"
          Use DarcyTimeDecoderLayer (context-based driver forecaster).

      - "latent_params"
          Uses PhysicsParameterEncoder + PhysicsDecoder to infer latent
          physical parameters from the context and decode a full rate curve.

    If `strategy_config` is None or does not define "forecast_policy",
    the behaviour is **identical to the legacy pipeline**:
      default policy = "mlp_residual_drivernet".

    Fusion types (data/physics):
      - "pin"          -> use physics output only
      - "average"      -> simple average between trend and physics
      - "concat_dense" -> concatenate and fuse with a small MLP
    """
    if strategy_config is None:
        strategy_config = {"strategy_name": "pressure_ensemble"}

    timesteps, feats = input_shape
    inputs = Input(shape=(timesteps, feats), name="all_features")

    # ---------------------------------------------------------------------
    # Feature slicing (keeps original mapping)
    # ---------------------------------------------------------------------
    DATA_INDICES = [0, 2, 7]
    PHYSICS_INDICES = [0, 3, 5, 7]  # [PI_s, Pwf_s, t_feat, Q_s]

    data_features = Lambda(
        lambda x: tf.gather(x, DATA_INDICES, axis=-1),
        name="data_features",
    )(inputs)
    physics_features = Lambda(
        lambda x: tf.gather(x, PHYSICS_INDICES, axis=-1),
        name="physics_features",
    )(inputs)

    # ---------------------------------------------------------------------
    # Trend branch (purely data-driven)
    # ---------------------------------------------------------------------
    summary_vector = GlobalAveragePooling1D(name="avg_pool_trend")(data_features)
    trend_block = TrendBlock(
        degree=trend_degree,
        forecast_horizon=horizon,
        name="trend_block",
    )
    trend_forecast = trend_block(summary_vector)

    # ---------------------------------------------------------------------
    # Physics branch
    # ---------------------------------------------------------------------
    # Context used by any physics policy (tail of the physics features).
    strat_cfg: Dict = dict(strategy_config or {})
    strat_cfg.setdefault("strategy_name", "pressure_ensemble")
    strat_cfg.setdefault("forecast_horizon", horizon)

    context_len = int(strat_cfg.get("context_length", horizon))
    physics_context = Lambda(
        lambda x: x[:, -context_len:, :],
        name="physics_context",
    )(physics_features)

    physics_horizon = Lambda(
        lambda x: x[:, -horizon:, :],
        name="physics_horizon",
    )(physics_features)

    # Determine physics policy, preserving legacy behaviour when unset.
    physics_policy = strat_cfg.get("forecast_policy")
    # physics_policy = "latent_params"
    if physics_policy is None:
        # LEGACY DEFAULT: context path with MLP forecaster
        physics_policy = "mlp_residual_drivernet"
        strat_cfg["forecast_policy"] = physics_policy

    # Context-specific defaults (only for context policies, not latent)
    if physics_policy in {"hold_last", "tcn_residual_drivernet", "mlp_residual_drivernet"}:
        strat_cfg.setdefault("use_context_encoder", True)
        strat_cfg.setdefault(
            "context_encoder",
            {"type": "gru", "units": 32, "layers": 1, "dropout": 0.1},
        )
        strat_cfg.setdefault("context_source", "hist")  # uses [PI_s, Pwf_s, t] scaled
        strat_cfg.setdefault(
            "forecaster_config",
            {
                "hidden_units": [128, 64],
                "dropout": 0.1,
                "l2_reg": 1e-4,
                "max_norm": 3.0,
            },
        )

        strat_cfg.setdefault("use_level_anchor", True)
        strat_cfg.setdefault("start_at_t0", True)
        strat_cfg.setdefault("warm_steps", 10)
        strat_cfg.setdefault("gate_kind", "exp")

        # driver location in context slice
        strat_cfg.setdefault("driver_channel_in_context", 3)
        strat_cfg.setdefault("driver_feature_index_in_X", -1)  # Q already in target scale
        strat_cfg.setdefault("driver_is_target_scaled", True)

        # anchor knobs
        strat_cfg.setdefault("anchor_stat", "mean")
        strat_cfg.setdefault("anchor_hp_lambda", 160000.0)
        strat_cfg.setdefault("anchor_diag", {"enabled": True, "period": 500})

    # ---------------------------------------------------------------------
    # Physics path selection
    # ---------------------------------------------------------------------

    # Precompute scalers for latent path (PI/Pwf in physical units).
    # This is cheap and safe even if we end up not using the latent policy.
    scaler_X_mean, scaler_X_std = get_center_and_scale(
        scaler_X, as_tf=True, dtype=tf.float32
    )
    PI_IDX = 0   # index in original X scaler for PI
    PWF_IDX = 3  # index in original X scaler for Pwf

    if physics_policy == "latent_params":
        # ==============================================================
        # 2.1 Latent-parameter physics path
        # ==============================================================

        # physics_context layout: [PI_s, Pwf_s, t_feat, Q_s]
        pi_hist_s = Lambda(
            lambda x: x[..., 0],
            name="latent_pi_hist_s",
        )(physics_context)
        p_hist_s = Lambda(
            lambda x: x[..., 1],
            name="latent_p_hist_s",
        )(physics_context)
        t_hist = Lambda(
            lambda x: x[..., 2],
            name="latent_t_hist",
        )(physics_context)

        # --- Time grid from history (same utility as extrapolation) ---
        def _latent_t_grid(th):
            return infer_t_grid(
                t_hist=th,
                steps=horizon,
                mode=strat_cfg.get("time_mode", "relative"),
                dt_scale=float(strat_cfg.get("dt_scale", 1.0)),
                start_at_t0=bool(strat_cfg.get("start_at_t0", True)),
            )

        t_grid = Lambda(_latent_t_grid, name="latent_t_grid")(t_hist)

        # --- Hold-last drivers in physical units, tiled over horizon ---
        def _pi_future(x):
            last_pi_s = x[:, -1:]  # (B, 1)
            pi_base = invert_feature_scaling(
                last_pi_s, scaler_X_mean, scaler_X_std, PI_IDX
            )  # physical PI
            return tf.tile(pi_base, [1, horizon])  # (B, H)

        def _p_future(x):
            last_p_s = x[:, -1:]  # (B, 1)
            p_base = invert_feature_scaling(
                last_p_s, scaler_X_mean, scaler_X_std, PWF_IDX
            )  # physical Pwf
            return tf.tile(p_base, [1, horizon])  # (B, H)

        PI_future = Lambda(_pi_future, name="latent_PI_future")(pi_hist_s)
        P_future = Lambda(_p_future, name="latent_P_future")(p_hist_s)

        # --- Physics parameter encoder on context ---
        latent_encoder = PhysicsParameterEncoder(
            scaler_X=scaler_X,
            scaler_target=scaler_target,
            strategy_config=strat_cfg,
            rnn_units=int(strat_cfg.get("latent_units", 64)),
            name="latent_param_encoder",
            diag_config={"enabled": True, "period": 500},
        )
        latent_params = latent_encoder(physics_context)

        # --- Stateless physics decoder (same strategy_name as other paths) ---
        physics_decoder = PhysicsDecoder(
            scaler_target=scaler_target,
            strategy_name=strat_cfg.get("strategy_name", "pressure_ensemble"),
            name="latent_physics_decoder",
            diag_config={"enabled": True, "period": 500},
        )

        # Default call: returns target-scaled rate (backward compatible)
        physics_forecast = physics_decoder(
            latent_params, PI_future, P_future, t_grid
        )

        # For consistency with later freezing logic
        physics_block = None

    else:
        # ==============================================================
        # 2.2 Existing legacy/context physics paths
        # ==============================================================

        if physics_policy == "original_pinn":
            # Legacy DarcyPhysicsLayer expects horizon features already present.
            physics_block_in = physics_horizon
        else:
            # Context-based policies consume a shorter context window.
            physics_block_in = physics_context

        physics_block = build_physics_block(
            scaler_X=scaler_X,
            scaler_target=scaler_target,
            policy=physics_policy,
            strategy_config=strat_cfg,
            name="physics_block",
        )
        physics_forecast = physics_block(physics_block_in)

    # ---------------------------------------------------------------------
    # Freezing knobs
    # ---------------------------------------------------------------------
    trend_block.trainable = not freeze_trend
    if physics_policy == "latent_params":
        latent_encoder.trainable = not freeze_physics
        physics_decoder.trainable = not freeze_physics
    else:
        physics_block.trainable = not freeze_physics

    # ---------------------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------------------
    ft = fusion_type.lower()
    if ft == "pin":
        outputs = physics_forecast
        model_name = "physics_block"
    elif ft == "average":
        outputs = Lambda(
            lambda x: 0.5 * (x[0] + x[1]),
            name="avg_fusion",
        )([trend_forecast, physics_forecast])
        model_name = "avg_fusion"
    elif ft == "concat_dense":
        combined = Concatenate(name="fusion_concat")(
            [trend_forecast, physics_forecast]
        )
        hidden1 = Dense(32, activation="relu", name="fusion_dense_1")(combined)
        outputs = Dense(
            horizon,
            activation="linear",
            name="concat_dense",
        )(hidden1)
        model_name = "concat_dense"
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

    model = Model(inputs, outputs, name=model_name)
    return model


# =============================================================================
# 3. Latent extrapolation model
# =============================================================================


def make_latent_extrapolation_model(
    trained_model: Model,
    scaler_X,
    scaler_target,
    strategy_config: Dict,
    steps: int,
    name: str = "Seq2PIN_latent_extrap",
    anchor_mode: str = "mean_tail",   # "last", "mean_tail", "median_tail"
    anchor_window: int = 20,          # quantos dias usar na cauda (<=0 -> usa tudo)
    warm_steps: int = 20,             # quantos steps o offset fica ativo
    gate_kind: str = "exp",           # "exp" ou "linear"
) -> Model:
    """
    Latent extrapolator com Instance-Level Anchoring na borda t=0.

    - Treino continua idêntico (H_train = 300).
    - Na inferência full_sequence:
        1) Reusa latent_param_encoder + latent_physics_decoder treinados.
        2) Gera Q_phys e aplica affine guard (a_affine, b_affine) normalmente.
        3) Corrige o NÍVEL da curva para respeitar a condição de contorno:

               y_pred(0) ≈ anchor(histórico)

           onde 'anchor(histórico)' pode ser:
             - último valor de Q_s ("last"),
             - média da cauda ("mean_tail"),
             - mediana da cauda ("median_tail").

        4) O offset decai nos primeiros `warm_steps` via gate temporal.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Lambda, Input
    from tensorflow.keras import Model
    from utils.utilities import invert_feature_scaling, get_center_and_scale
    from .darcy_components import infer_t_grid  # usar a mesma função utilitária

    if steps <= 0:
        raise ValueError(f"`steps` must be positive, got {steps}")

    # ------------------------------------------------------------------
    # 1) Recuperar encoder e decoder treinados
    # ------------------------------------------------------------------
    try:
        latent_encoder = trained_model.get_layer("latent_param_encoder")
    except ValueError as e:
        raise ValueError(
            "Expected a layer named 'latent_param_encoder' "
            "(physics_policy='latent_params')."
        ) from e

    try:
        physics_decoder_ref = trained_model.get_layer("latent_physics_decoder")
    except ValueError as e:
        raise ValueError(
            "Expected a layer 'latent_physics_decoder' in trained_model "
            "(needed to reuse the same PhysicsDecoder in extrapolation)."
        ) from e

    # Recuperar strategy_config originalmente usada pelo encoder e permitir override
    cfg_from_model = {}
    if hasattr(latent_encoder, "strategy_config"):
        try:
            cfg_from_model = dict(latent_encoder.strategy_config or {})
        except Exception:
            cfg_from_model = {}
    strat_cfg = dict(cfg_from_model)
    strat_cfg.update(strategy_config or {})

    # ------------------------------------------------------------------
    # 2) Input (B, T, F) - mesmo shape do modelo treinado
    # ------------------------------------------------------------------
    if (
        not isinstance(trained_model.input_shape, tuple)
        or len(trained_model.input_shape) != 3
    ):
        raise ValueError(
            f"Expected input (batch, T, F). Got {trained_model.input_shape!r}"
        )
    _, timesteps, n_feats = trained_model.input_shape

    inputs = Input(shape=(timesteps, n_feats), name="all_features_extrap")

    # ------------------------------------------------------------------
    # 3) Physics features e contexto (mesmo slicing do create_model)
    # ------------------------------------------------------------------
    # PHYSICS_INDICES = [0, 3, 5, 7] = [PI_s, Pwf_s, t_feat, Q_s]
    PHYSICS_INDICES = [0, 3, 5, 7]
    PI_IDX = 0
    PWF_IDX = 3

    physics_features = Lambda(
        lambda x: tf.gather(x, PHYSICS_INDICES, axis=-1),
        name="physics_features_extrap",
    )(inputs)

    context_len = int(strat_cfg.get("context_length", timesteps))
    physics_context = Lambda(
        lambda x: x[:, -context_len:, :],
        name="physics_context_extrap",
    )(physics_features)

    # Históricos escalados (como no treino)
    pi_hist_s = Lambda(lambda x: x[..., 0], name="latent_pi_hist_s_extrap")(physics_context)
    p_hist_s  = Lambda(lambda x: x[..., 1], name="latent_p_hist_s_extrap")(physics_context)
    t_hist_s  = Lambda(lambda x: x[..., 2], name="latent_t_hist_s_extrap")(physics_context)
    q_hist_s  = Lambda(lambda x: x[..., 3], name="latent_q_hist_s_extrap")(physics_context)  # Q_s (target-scaled)

    # ------------------------------------------------------------------
    # 4) Grade de tempo via infer_t_grid (mesmo modo que no treino)
    # ------------------------------------------------------------------
    time_mode   = strat_cfg.get("time_mode", "relative")
    dt_scale    = float(strat_cfg.get("dt_scale", 1.0))
    start_at_t0 = bool(strat_cfg.get("start_at_t0", True))

    def _latent_t_grid(th_scaled):
        return infer_t_grid(
            t_hist=th_scaled,
            steps=steps,
            mode=time_mode,
            dt_scale=dt_scale,
            start_at_t0=start_at_t0,
        )

    t_grid = Lambda(_latent_t_grid, name=f"latent_t_grid_{steps}")(t_hist_s)

    # ------------------------------------------------------------------
    # 5) Drivers futuros em unidades físicas (hold-last)
    # ------------------------------------------------------------------
    scaler_X_mean, scaler_X_std = get_center_and_scale(
        scaler_X, as_tf=True, dtype=tf.float32
    )

    def _last_phys_from_scaled(x_s, feat_idx):
        last_s = x_s[:, -1:]  # (B, 1)
        return invert_feature_scaling(last_s, scaler_X_mean, scaler_X_std, feat_idx)

    PI_last = Lambda(
        lambda x: _last_phys_from_scaled(x, PI_IDX),
        name="dbg_PI_last_phys",
    )(pi_hist_s)
    PWF_last = Lambda(
        lambda x: _last_phys_from_scaled(x, PWF_IDX),
        name="dbg_PWF_last_phys",
    )(p_hist_s)

    PI_future = Lambda(
        lambda x: tf.tile(x, [1, steps]),
        name=f"latent_PI_future_{steps}",
    )(PI_last)

    P_future = Lambda(
        lambda x: tf.tile(x, [1, steps]),
        name=f"latent_P_future_{steps}",
    )(PWF_last)

    # ------------------------------------------------------------------
    # 6) Core latent physics: mesmo encoder + mesmo decoder do treino
    # ------------------------------------------------------------------
    latent_params = latent_encoder(physics_context, training=False)
    y_base = physics_decoder_ref(latent_params, PI_future, P_future, t_grid)  # (B, steps), target-scaled

    # ------------------------------------------------------------------
    # 7) Instance-Level Anchoring com gate temporal
    # ------------------------------------------------------------------
    def _apply_anchor(args):
        """
        args = [y_base, q_hist_s]
        y_base : (B, steps)   -> previsão em escala de target
        q_hist_s : (B, M)     -> histórico de Q_s em escala de target
        """
        y_b, q_h = args  # (B, H), (B, M) em target scale
        B = tf.shape(y_b)[0]
        H = tf.shape(y_b)[1]
        T = tf.shape(q_h)[1]

        # ----- 7.1 Anchor a partir da cauda do histórico -----
        if anchor_mode == "last":
            # último valor da série
            anchor = q_h[:, -1:]  # (B,1)
        else:
            # janela de cauda
            if anchor_window > 0:
                W = tf.minimum(tf.cast(anchor_window, tf.int32), T)
                tail = q_h[:, -W:]  # (B, W)
            else:
                tail = q_h

            if anchor_mode == "median_tail":
                anchor = tf.experimental.numpy.median(tail, axis=-1, keepdims=True)
            else:
                # default: "mean_tail"
                anchor = tf.reduce_mean(tail, axis=-1, keepdims=True)  # (B,1)

        # ----- 7.2 Offset em t=0 -----
        y0 = y_b[:, :1]              # (B,1)
        offset0 = anchor - y0        # (B,1)

        # ----- 7.3 Gate temporal -----
        if warm_steps <= 0:
            # Sem gate => não aplica correção
            return y_b

        k = tf.cast(tf.range(H)[tf.newaxis, :], y_b.dtype)  # (1, H)
        warm = tf.cast(warm_steps, y_b.dtype)

        if gate_kind == "linear":
            gate = 1.0 - k / tf.maximum(warm, 1.0)
            gate = tf.clip_by_value(gate, 0.0, 1.0)
        else:
            # Exponencial: ~1 em t=0, ~0 em warm_steps
            ln100 = tf.math.log(tf.constant(100.0, dtype=y_b.dtype))
            tau   = tf.maximum(warm / ln100, tf.constant(1e-6, dtype=y_b.dtype))
            gate  = tf.exp(-k / tau)
            gate  = tf.where(k < warm, gate, tf.zeros_like(gate))

        # gate: (1,H) -> (B,H)
        gate = tf.tile(gate, [B, 1])

        # ----- 7.4 Aplicar correção -----
        y_out = y_b + gate * offset0  # broadcast offset0 (B,1) ao longo de H

        return y_out

    y_corrected = Lambda(
        _apply_anchor,
        name=f"latent_anchor_{anchor_mode}_warm{warm_steps}",
    )([y_base, q_hist_s])

    # ------------------------------------------------------------------
    # 8) (Opcional) Debug leve de tempo / drivers (pode comentar se quiser)
    # ------------------------------------------------------------------
    def _dbg(xs):
        th_s, tg, pi, pwf = xs
        dt   = th_s[:, 1:] - th_s[:, :-1]
        dtm  = tf.reduce_mean(dt)
        tmax = tf.reduce_mean(tg[:, -1])
        return tf.print(
            "[latent_extrap] steps=",
            steps,
            " dt_mean_scaled=",
            dtm,
            " t_max≈",
            tmax,
            " PI_last≈",
            tf.reduce_mean(pi),
            " Pwf_last≈",
            tf.reduce_mean(pwf),
            summarize=-1,
        )

    _ = Lambda(
        lambda xs: (_dbg(xs), xs[0])[1],
        name="dbg_latent_time_extrap",
    )([t_hist_s, t_grid, PI_last, PWF_last])

    return Model(inputs, y_corrected, name=name)


