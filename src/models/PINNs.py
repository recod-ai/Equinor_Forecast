from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple

import tensorflow as tf

from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import layers, initializers, activations


import tensorflow_probability as tfp
tfd = tfp.distributions

# =============================================================================
# 1. Base Interface for Physics Strategies
# =============================================================================
class PhysicsStrategy(ABC):
    @abstractmethod
    def compute_Q_phys(self, PI_measured, P_wf, t_feature, absolute_value):
        """
        Compute the physics-based forecast (Q_phys).
        """
        pass

# =============================================================================
# 2. Concrete Physics Strategy Implementations
# =============================================================================

class DiffusivityDecayStrategy(PhysicsStrategy):
    """Implements a pressure diffusion decay model."""
    def __init__(self, P_reservoir, kappa):
        self.P_reservoir = P_reservoir
        self.kappa = tf.Variable(
            kappa,
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0.0001, 5.0),
            name='kappa_diffusivity'
        )

    def compute_Q_phys(self, PI_measured, P_wf, t_feature):
        decay_factor = 1.0 - tf.exp(-self.kappa * t_feature)
        Q_phys = tf.abs(PI_measured * (self.P_reservoir - P_wf) * decay_factor)
        return Q_phys


class ExponentialDecayStrategy(PhysicsStrategy):
    """
    Assumes reservoir pressure decreases exponentially over time, modeling transient flow conditions.
    The exponential decay is characterized by a constant decline rate.
    """
    def __init__(self, P_reservoir, decay_rate):
        self.P_reservoir = P_reservoir
        self.decay_rate = decay_rate
        
    def compute_Q_phys(self, PI_measured, P_wf, t_feature):
        P_res_t = self.P_reservoir * tf.exp(-self.decay_rate * t_feature)
        return tf.abs(PI_measured * (P_res_t - P_wf))

class ArpsDeclineStrategy(PhysicsStrategy):
    """
    Implements Arps hyperbolic decline to represent reservoir pressure during boundary-dominated flow regimes.
    Suitable for late-time reservoir behavior, assuming hyperbolic pressure decline.
    """
    def __init__(self, P_reservoir, decay_rate, b_factor):
        self.P_reservoir = P_reservoir
        self.decay_rate = decay_rate
        self.b_factor = b_factor

    def compute_Q_phys(self, PI_measured, P_wf, t_feature):
        P_res_t = self.P_reservoir / (1 + self.b_factor * self.decay_rate * t_feature)**(1/self.b_factor)
        return tf.abs(PI_measured * (P_res_t - P_wf))

class WeightedEnsembleStrategy(PhysicsStrategy):
    """
    Combines two distinct PhysicsStrategies through a learnable weighted sum.
    Allows the model to dynamically determine the importance of each underlying physical assumption,
    providing improved flexibility but potentially introducing gradient optimization challenges.
    """
    def __init__(self, P_reservoir, decay_rate, alpha_init=0.5):
        self.strategy1 = DynamicEnsembleStrategy(P_reservoir, decay_rate)
        self.strategy2 = ExponentialDecayStrategy(P_reservoir, decay_rate)
        self.alpha = tf.Variable(alpha_init, trainable=True, 
                                 constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0),
                                 name='ensemble_alpha')

    def compute_Q_phys(self, PI_measured, P_wf, t_feature):
        Q1 = self.strategy1.compute_Q_phys(PI_measured, P_wf, t_feature)
        Q2 = self.strategy2.compute_Q_phys(PI_measured, P_wf, t_feature)
        return self.alpha * Q1 + (1.0 - self.alpha) * Q2

class CombinedExpArpsStrategy(PhysicsStrategy):
    def __init__(self, P_reservoir, decay_rate, b_factor):
        self.exp_strategy = ExponentialDecayStrategy(P_reservoir, decay_rate)
        self.arps_strategy = ArpsDeclineStrategy(P_reservoir, decay_rate, b_factor)

    def compute_Q_phys(self, PI_measured, P_wf, t_feature):
        Q_exp = self.exp_strategy.compute_Q_phys(PI_measured, P_wf, t_feature)
        Q_arps = self.arps_strategy.compute_Q_phys(PI_measured, P_wf, t_feature)
        return (Q_exp + Q_arps) / 2



class StaticPressureStrategy(PhysicsStrategy):
    """
    Assumes reservoir pressure remains constant over the forecast horizon.
    Represents an idealized scenario ignoring depletion, useful as a lower-bound approximation.
    """
    def __init__(self, P_reservoir):
        self.P_reservoir = P_reservoir

    def compute_Q_phys(self, PI_measured, P_wf, t_feature):
        Q_phys_base = tf.abs(PI_measured * (self.P_reservoir - P_wf)) 
        #   Q_phys_base (shape [batch,1] ou [batch]) broadcasta para [batch,30]
        Q_phys = Q_phys_base * tf.ones_like(t_feature)

        # Agora Q_phys tem shape [batch_size, 30], compatível com t_feature
        return Q_phys
    
class DynamicEnsembleStrategy(PhysicsStrategy):

    """
        Computes a physics-based oil production forecast by dynamically combining two distinct 
        reservoir pressure assumptions through a simple averaging ensemble:
    
        Q_phys(t) = [| PI_measured × (P_res(t) - P_wf) | + | PI_measured × (P_reservoir - P_wf) |] / 2
    
        where:
        - P_res(t) = P_reservoir × exp(-decay_rate × t_feature), representing a transient exponential pressure decline.
        - P_reservoir represents the static (initial) reservoir pressure assumption.

        This strategy explicitly integrates two competing physical assumptions to approximate realistic reservoir behavior:

        1. Exponential Decay Term:*
           - Captures early-time transient flow regimes characterized by rapid reservoir pressure depletion.
           - Often results in slight overestimation due to overly rapid pressure decay, especially if the reservoir undergoes pressure stabilization 
           from external sources (e.g., water influx, well management).

        2. Static Pressure Term:
           - Represents a constant reservoir pressure scenario, ignoring reservoir depletion entirely.
           - Tends to systematically underestimate production, especially in reservoirs actively undergoing depletion.

        Why Ensemble These Two Scenarios?
           The dynamic averaging approach leverages empirical data-driven corrections implicitly, creating a hybrid scenario where:
           - Early-time reservoir dynamics dominated by transient flow (captured by exponential decay) are balanced by late-time boundary-dominated 
           flow dynamics (partially captured by static pressure, representing an idealized lower-bound scenario).
           - The averaging method implicitly incorporates reservoir complexities such as varying fluid mobility, heterogeneity, 
           intermittent well operations, or partial pressure support from aquifers.

        Physical Interpretation and Acceptability:
           Although averaging two idealized physics models reduces strict interpretability of parameters (initial pressure and decay rate), 
           it remains physically plausible due to:
           - Real reservoir behavior often transitioning between flow regimes (transient to boundary-dominated), naturally leading to intermediate dynamics 
           not accurately represented by either pure exponential or static assumptions individually.
           - Serving as a pragmatic empirical regularization, stabilizing model predictions, and mitigating systematic biases inherent to each individual physics model.

        Technical Benefits (Gradient and Optimization):
           From a machine learning (Physics-Informed Neural Network, PINN) standpoint:
           - The averaging approach significantly reduces gradient instability compared to weighted ensembles (like learnable alpha parameters), 
           providing smoother gradient updates that facilitate stable optimization.
           - It effectively acts as implicit regularization, limiting overfitting to overly simplified physics assumptions, 
           and thus enhancing generalization and robustness.

        When is This Model Appropriate?
           - Ideal for reservoirs exhibiting moderate pressure support, intermediate transient-boundary behavior, 
           or complex, heterogeneous flow dynamics not strictly captured by single, simple reservoir pressure models.
           - Suitable when the primary goal is empirical prediction accuracy over strict parameter interpretability, 
           and when gradients or optimization stability is a concern.
    """

    def __init__(self, P_reservoir, decay_rate, absolute_value=True):
        self.P_reservoir = P_reservoir
        self.decay_rate = decay_rate
        self.absolute_value = absolute_value

    def compute_Q_phys(self, PI_measured, P_wf, t_feature):
        # Exponential decay of reservoir pressure
        P_res_t = self.P_reservoir * tf.exp(-self.decay_rate * t_feature)
        # Combined pressure ensemble using decayed and static reservoir pressure
        
        if self.absolute_value:
            Q_phys_base_1 = tf.abs(PI_measured * (P_res_t - P_wf))
            Q_phys_base_2 = tf.abs(PI_measured * (self.P_reservoir - P_wf))
        else:
            Q_phys_base_1 = PI_measured * (P_res_t - P_wf)
            Q_phys_base_2 = PI_measured * (self.P_reservoir - P_wf)
        Q_phys_base = (Q_phys_base_1 + Q_phys_base_2) / 2
        return Q_phys_base


# =============================================================================
# 1. Base Classes (Interfaces)
# =============================================================================
class BaseContextExtractor(layers.Layer, ABC):
    """Abstract Base Class for history context extractors."""
    @abstractmethod
    def call(self, inputs, training=False):
        """
        Processes historical input sequence.
        Args:
            inputs: Tensor shape (batch, lag_window, features).
            training: Boolean flag.
        Returns:
            context_vector: Tensor shape (batch, context_dim).
        """
        pass

class BaseContextFuser(layers.Layer, ABC):
    """Abstract Base Class for fusing context with per-step info for residuals."""
    def __init__(self, forecast_horizon, **kwargs):
        super().__init__(**kwargs)
        self.forecast_horizon = forecast_horizon # Needed by many fusers

    @abstractmethod
    def call(self, context_vector, per_step_inputs, training=False):
        """
        Computes the residual forecast based on context and per-step info.
        Args:
            context_vector: Tensor shape (batch, context_dim).
            per_step_inputs: A dictionary containing tensors needed for residual calculation,
                             typically including 'q_phys_scaled' (batch, horizon),
                             'forecast_times' (batch, horizon), 'pi_last_measured' (batch, 1), etc.
                             Each fuser will define what it expects in this dict.
            training: Boolean flag.
        Returns:
            residual_scaled: Tensor shape (batch, forecast_horizon).
        """
        pass


# =============================================================================
# 2. Concrete Context Extractor Implementation: TCN
# =============================================================================

class TCNResidualBlock(layers.Layer):
    """Temporal Convolutional Network residual block with optional normalization and constrained initializers."""
    def __init__(self, dilation_rate, nb_filters, kernel_size, padding,
                 dropout_rate=0.01, activation='relu', normalization_type='layer',
                 kernel_initializer=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # Configuration
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.activation_fn = activations.get(activation)
        self.normalization_type = normalization_type.lower() if normalization_type else None
        # Initializer
        self.kernel_initializer = kernel_initializer
        # Layers: conv1 -> norm -> act -> drop
        self.conv1 = layers.Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_conv1"
        )
        self.norm1 = self._make_norm(name=f"{self.name}_norm1")
        self.act1 = layers.Activation(self.activation_fn, name=f"{self.name}_act1")
        self.drop1 = layers.Dropout(dropout_rate, name=f"{self.name}_drop1")
        # conv2 -> norm -> act -> drop
        self.conv2 = layers.Conv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_conv2"
        )
        self.norm2 = self._make_norm(name=f"{self.name}_norm2")
        self.act2 = layers.Activation(self.activation_fn, name=f"{self.name}_act2")
        self.drop2 = layers.Dropout(dropout_rate, name=f"{self.name}_drop2")
        # Placeholder for residual projector
        self.downsample = None
        self.act_final = layers.Activation(self.activation_fn, name=f"{self.name}_act_final")

    def _make_norm(self, name):
        if self.normalization_type == 'layer':
            return layers.LayerNormalization(name=name)
        if self.normalization_type == 'batch':
            return layers.BatchNormalization(name=name)
        return None

    def build(self, input_shape):
        # Project input channels if needed
        if input_shape[-1] != self.nb_filters:
            self.downsample = layers.Conv1D(
                filters=self.nb_filters,
                kernel_size=1,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                name=f"{self.name}_downsample"
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        # First conv block
        x = self.conv1(x)
        if self.norm1:
            kwargs = {'training': training} if isinstance(self.norm1, layers.BatchNormalization) else {}
            x = self.norm1(x, **kwargs)
        x = self.act1(x)
        x = self.drop1(x, training=training)
        # Second conv block
        x = self.conv2(x)
        if self.norm2:
            kwargs = {'training': training} if isinstance(self.norm2, layers.BatchNormalization) else {}
            x = self.norm2(x, **kwargs)
        x = self.act2(x)
        x = self.drop2(x, training=training)
        # Residual add
        residual = self.downsample(inputs) if self.downsample else inputs
        x = layers.add([residual, x])
        return self.act_final(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'dilation_rate': self.dilation_rate,
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation_fn),
            'normalization_type': self.normalization_type,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config


class TCNContextExtractor(BaseContextExtractor):
    """Extracts context using a stack of TCN residual blocks with constrained initializers."""
    def __init__(self,
                 tcn_filters=128,
                 tcn_kernel_size=3,
                 tcn_dilations=(1, 2, 4),
                 tcn_stacks=1,
                 tcn_activation='relu',
                 tcn_normalization='layer',
                 dropout_rate=0.01,
                 initializer=None,
                 name='tcn_context_extractor',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        # Configuration
        self.tcn_filters = tcn_filters
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dilations = tuple(tcn_dilations)
        self.tcn_stacks = tcn_stacks
        self.tcn_activation = tcn_activation
        self.tcn_normalization = tcn_normalization
        self.dropout_rate = dropout_rate
        # Use provided initializer or a conservative default (GlorotUniform)
        self.kernel_initializer = initializer
        # Layers built later
        self.tcn_initial_conv = None
        self.tcn_blocks = []

    def build(self, input_shape):
        # Project to tcn_filters if needed
        if input_shape[-1] != self.tcn_filters:
            self.tcn_initial_conv = layers.Conv1D(
                filters=self.tcn_filters,
                kernel_size=1,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                name=f'{self.name}_initial_proj'
            )
        # Reset and build TCN blocks
        self.tcn_blocks = []
        for s in range(self.tcn_stacks):
            for d in self.tcn_dilations:
                self.tcn_blocks.append(
                    TCNResidualBlock(
                        dilation_rate=d,
                        nb_filters=self.tcn_filters,
                        kernel_size=self.tcn_kernel_size,
                        padding='causal',
                        dropout_rate=self.dropout_rate,
                        activation=self.tcn_activation,
                        normalization_type=self.tcn_normalization,
                        kernel_initializer=self.kernel_initializer,
                        name=f'{self.name}_stack_{s}_dilation_{d}'
                    )
                )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        if self.tcn_initial_conv is not None:
            x = self.tcn_initial_conv(x)
        for block in self.tcn_blocks:
            x = block(x, training=training)
        # Return last time step as context vector
        return x[:, -1, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            'tcn_filters': self.tcn_filters,
            'tcn_kernel_size': self.tcn_kernel_size,
            'tcn_dilations': list(self.tcn_dilations),
            'tcn_stacks': self.tcn_stacks,
            'tcn_activation': self.tcn_activation,
            'tcn_normalization': self.tcn_normalization,
            'dropout_rate': self.dropout_rate,
            'initializer': self.kernel_initializer
        })
        return config



class CNNContextExtractor(BaseContextExtractor):
    """Extracts context using CNN layers with constrained initializers."""
    def __init__(self,
                 cnn_filters=16,
                 cnn_kernel_size=3,
                 cnn_layers=2,
                 cnn_activation='relu',
                 cnn_pooling='global_avg',
                 dropout_rate=0.01,
                 initializer=None,
                 name='cnn_context_extractor',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_layers = cnn_layers
        self.cnn_activation = cnn_activation
        self.cnn_pooling = cnn_pooling.lower()
        self.dropout_rate = dropout_rate
        # Constrain kernel init (He) unless overridden
        self.kernel_initializer = initializer
        self.conv_layers = []
        self.pooling_layer = None
        self.dropout_layer = layers.Dropout(self.dropout_rate, name=f"{self.name}_dropout")

    def build(self, input_shape):
        self.conv_layers = []
        for i in range(self.cnn_layers):
            self.conv_layers.append(
                layers.Conv1D(
                    filters=self.cnn_filters,
                    kernel_size=self.cnn_kernel_size,
                    activation=self.cnn_activation,
                    padding='causal',
                    kernel_initializer=self.kernel_initializer,
                    name=f"{self.name}_conv1d_{i+1}"
                )
            )
        if self.cnn_pooling == 'global_max':
            self.pooling_layer = layers.GlobalMaxPooling1D(name=f"{self.name}_global_max_pool")
        else:
            self.pooling_layer = layers.GlobalAveragePooling1D(name=f"{self.name}_global_avg_pool")
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
        context = self.pooling_layer(x)
        return self.dropout_layer(context, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'cnn_filters': self.cnn_filters,
            'cnn_kernel_size': self.cnn_kernel_size,
            'cnn_layers': self.cnn_layers,
            'cnn_activation': self.cnn_activation,
            'cnn_pooling': self.cnn_pooling,
            'dropout_rate': self.dropout_rate,
            'initializer': self.kernel_initializer
        })
        return config

class RNNContextExtractor(BaseContextExtractor):
    """Extracts context using LSTM or GRU layers with constrained initializers."""
    def __init__(self,
                 rnn_units=16,
                 rnn_layers=2,
                 rnn_type='lstm',
                 bidirectional=False,
                 dropout_rate=0.01,
                 initializer=None,
                 name='rnn_context_extractor',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializer
        self.recurrent_initializer = initializer
        self.rnn_layer_instances = []
        self.dropout_layer = layers.Dropout(self.dropout_rate, name=f"{self.name}_dropout")
        self._wrap = layers.Bidirectional if self.bidirectional else lambda x: x

    def build(self, input_shape):
        RNNCell = layers.LSTM if self.rnn_type == 'lstm' else layers.GRU
        self.rnn_layer_instances = []
        for i in range(self.rnn_layers):
            is_last = (i == self.rnn_layers - 1)
            rnn = RNNCell(
                self.rnn_units,
                return_sequences=not is_last,
                return_state=is_last,
                dropout=self.dropout_rate if not is_last else 0.0,
                kernel_initializer=self.kernel_initializer,
                recurrent_initializer=self.recurrent_initializer,
                name=f"{self.name}_{self.rnn_type}_{i+1}"
            )
            self.rnn_layer_instances.append(self._wrap(rnn))
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = inputs
        final_h = None
        for idx, layer in enumerate(self.rnn_layer_instances):
            is_last = (idx == len(self.rnn_layer_instances) - 1)
            if is_last:
                outputs = layer(x, training=training)
                if self.bidirectional:
                    if self.rnn_type == 'lstm':
                        _, fwd_h, _, bwd_h, _ = outputs
                    else:
                        _, fwd_h, bwd_h = outputs
                    final_h = tf.concat([fwd_h, bwd_h], axis=-1)
                else:
                    if self.rnn_type == 'lstm':
                        _, h, _ = outputs
                    else:
                        _, h = outputs
                    final_h = h
            else:
                x = layer(x, training=training)
        if final_h is None:
            raise RuntimeError("Failed to retrieve final RNN state.")
        return self.dropout_layer(final_h, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rnn_units': self.rnn_units,
            'rnn_layers': self.rnn_layers,
            'rnn_type': self.rnn_type,
            'bidirectional': self.bidirectional,
            'dropout_rate': self.dropout_rate,
            'initializer': self.kernel_initializer
        })
        return config

class IdentityContextExtractor(BaseContextExtractor):
    """
    Pass-through context extractor:
    - 'last_step': returns last time step features
    - 'flatten': flattens entire window
    - 'none': returns zeros
    """

    def __init__(self, mode='last_step', name='identity_context', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mode = mode.lower()
        if self.mode not in {'last_step', 'flatten', 'none'}:
            raise ValueError("mode must be 'last_step', 'flatten', or 'none'")

        # Will be set in build()
        self.output_dim = None

    def build(self, input_shape):
        """Determine output dimension based on mode."""
        _, time_len, feat_dim = input_shape
        if self.mode == 'last_step':
            self.output_dim = feat_dim
        elif self.mode == 'flatten':
            self.output_dim = time_len * feat_dim
        else:  # 'none'
            self.output_dim = 1
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Return context according to the selected mode."""
        batch = tf.shape(inputs)[0]

        if self.mode == 'last_step':
            # shape: (batch, features)
            return inputs[:, -1, :]

        if self.mode == 'flatten':
            # shape: (batch, time*features)
            return tf.reshape(inputs, (batch, -1))

        # 'none' → shape: (batch, 1)
        return tf.zeros((batch, 1), dtype=inputs.dtype)

    def compute_output_shape(self, input_shape):
        """Report output shape for Keras compatibility."""
        return tf.TensorShape([input_shape[0], self.output_dim])

    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            "mode": self.mode,
        })
        return config

class AggregateContextExtractor(BaseContextExtractor):
    """
    Extrai um vetor de contexto fazendo agregação simples (mean e/ou last step)
    sobre as features selecionadas.
    """
    def __init__(self, feature_indices: Tuple[int, ...], name: str = "simple_aggregate_extractor"):
        super().__init__(name=name)
        self.feature_indices = feature_indices

    def call(self, inputs, training=False):
        # inputs: (batch, timesteps, features)
        # seleciona apenas as features de interesse
        x = tf.gather(inputs, self.feature_indices, axis=-1)    # (B, T, K)
        # 1) média ao longo do tempo
        mean = tf.reduce_mean(x, axis=1)                        # (B, K)
        # 2) último valor
        last = x[:, -1, :]                                       # (B, K)
        # concatena para formar um vetor de contexto de tamanho 2·K
        ctx = tf.concat([mean, last], axis=-1)                  # (B, 2K)
        return ctx

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "feature_indices": self.feature_indices
        })
        return cfg



# =============================================================================
# 3. Concrete Context Fuser Implementations
# =============================================================================
class FiLMContextFuser(BaseContextFuser):
    """Fuses context using FiLM with constrained initializers."""
    def __init__(self,
                 forecast_horizon,
                 residual_hidden_units=16,
                 dropout_rate=0.01,
                 initializer=None,
                 name='film_context_fuser',
                 **kwargs):
        super().__init__(forecast_horizon=forecast_horizon, name=name, **kwargs)
        self.residual_hidden_units = residual_hidden_units
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializer
        self.bias_initializer = initializers.Constant(0.0)
        self.film_gamma_dense = None
        self.film_beta_dense = None
        self.residual_dense1 = layers.Dense(
            self.residual_hidden_units,
            activation='relu',
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp_dense1"
        )
        self.residual_dropout = layers.Dropout(self.dropout_rate, name=f"{self.name}_mlp_drop")
        self.residual_dense2 = layers.Dense(
            1,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp_dense2"
        )
        self.per_step_feature_dim = 3

    def call(self, context_vector, per_step_inputs, training=False):
        if self.film_gamma_dense is None:
            self.film_gamma_dense = layers.Dense(
                self.per_step_feature_dim,
                activation='linear',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=initializers.Constant(1.0),
                name=f"{self.name}_film_gamma"
            )
        if self.film_beta_dense is None:
            self.film_beta_dense = layers.Dense(
                self.per_step_feature_dim,
                activation='linear',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                name=f"{self.name}_film_beta"
            )
        q_phys_scaled = per_step_inputs['q_phys_scaled']
        forecast_times = per_step_inputs['forecast_times']
        pi_last_measured = per_step_inputs['pi_last_measured']
        per_step_features = tf.concat([
            tf.expand_dims(q_phys_scaled, -1),
            tf.expand_dims(forecast_times, -1),
            tf.tile(tf.expand_dims(pi_last_measured, 1), [1, self.forecast_horizon, 1])
        ], -1)
        gamma_raw = self.film_gamma_dense(context_vector)
        beta_raw = self.film_beta_dense(context_vector)
        gamma = tf.tile(tf.expand_dims(gamma_raw, 1), [1, self.forecast_horizon, 1])
        beta = tf.tile(tf.expand_dims(beta_raw, 1), [1, self.forecast_horizon, 1])
        modulated = per_step_features * gamma + beta
        hidden = self.residual_dense1(modulated)
        hidden = self.residual_dropout(hidden, training=training)
        residual = self.residual_dense2(hidden)
        return tf.squeeze(residual, -1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'residual_hidden_units': self.residual_hidden_units,
            'dropout_rate': self.dropout_rate,
            'initializer': self.kernel_initializer
        })
        return config

class GatingContextFuser(BaseContextFuser):
    """Fuses context by learning a gate with constrained initializers."""
    def __init__(self,
                 forecast_horizon,
                 residual_hidden_units=16,
                 dropout_rate=0.01,
                 gate_activation='sigmoid',
                 initializer=None,
                 name='gating_context_fuser',
                 **kwargs):
        super().__init__(forecast_horizon=forecast_horizon, name=name, **kwargs)
        self.residual_hidden_units = residual_hidden_units
        self.dropout_rate = dropout_rate
        self.gate_activation = gate_activation
        self.kernel_initializer = initializer
        self.residual_mlp_dense1 = layers.Dense(
            residual_hidden_units,
            activation='relu',
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp_dense1"
        )
        self.residual_mlp_dropout = layers.Dropout(dropout_rate, name=f"{self.name}_mlp_drop")
        self.residual_mlp_dense2 = layers.Dense(
            1,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp_dense2"
        )
        self.gate_dense = layers.Dense(
            1,
            activation=gate_activation,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_gate_dense"
        )

    def call(self, context_vector, per_step_inputs, training=False):
        q_phys = per_step_inputs['q_phys_scaled']
        times = per_step_inputs['forecast_times']
        pi_last = per_step_inputs['pi_last_measured']
        features = tf.concat([
            tf.expand_dims(q_phys, -1),
            tf.expand_dims(times, -1),
            tf.tile(tf.expand_dims(pi_last, 1), [1, self.forecast_horizon, 1])
        ], -1)
        hidden = self.residual_mlp_dense1(features)
        hidden = self.residual_mlp_dropout(hidden, training=training)
        prelim = self.residual_mlp_dense2(hidden)
        gate_raw = self.gate_dense(context_vector)
        gate = tf.tile(tf.expand_dims(gate_raw, 1), [1, self.forecast_horizon, 1])
        gated = prelim * gate
        return tf.squeeze(gated, -1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'residual_hidden_units': self.residual_hidden_units,
            'dropout_rate': self.dropout_rate,
            'gate_activation': self.gate_activation,
            'initializer': self.kernel_initializer
        })
        return config

class ConcatContextFuser(BaseContextFuser):
    """Fuses context by concatenating context and per-step inputs with constrained initializers."""
    def __init__(self,
                 forecast_horizon,
                 residual_hidden_units=16,
                 dropout_rate=0.01,
                 initializer=None,
                 name='concat_context_fuser',
                 **kwargs):
        super().__init__(forecast_horizon=forecast_horizon, name=name, **kwargs)
        self.residual_hidden_units = residual_hidden_units
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializer
        self.residual_dense1 = layers.Dense(
            residual_hidden_units,
            activation='relu',
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp_dense1"
        )
        self.residual_dropout = layers.Dropout(dropout_rate, name=f"{self.name}_mlp_drop")
        self.residual_dense2 = layers.Dense(
            1,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp_dense2"
        )

    def call(self, context_vector, per_step_inputs, training=False):
        q_phys = per_step_inputs['q_phys_scaled']
        times = per_step_inputs['forecast_times']
        pi_last = per_step_inputs['pi_last_measured']
        q_exp = tf.expand_dims(q_phys, -1)
        t_exp = tf.expand_dims(times, -1)
        pi_tiled = tf.tile(tf.expand_dims(pi_last, 1), [1, self.forecast_horizon, 1])
        ctx_tiled = tf.tile(tf.expand_dims(context_vector, 1), [1, self.forecast_horizon, 1])
        inp = tf.concat([ctx_tiled, q_exp, t_exp, pi_tiled], -1)
        hidden = self.residual_dense1(inp)
        hidden = self.residual_dropout(hidden, training=training)
        out = self.residual_dense2(hidden)
        return tf.squeeze(out, -1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'residual_hidden_units': self.residual_hidden_units,
            'dropout_rate': self.dropout_rate,
            'initializer': self.kernel_initializer
        })
        return config


class BiasScaleContextFuser(BaseContextFuser):
    """Fuses context by learning scale and bias factors with constrained initializers."""
    def __init__(self,
                 forecast_horizon,
                 residual_hidden_units=16,
                 dropout_rate=0.01,
                 scale_activation='linear',
                 bias_activation='linear',
                 initializer=None,
                 name='bias_scale_context_fuser',
                 **kwargs):
        super().__init__(forecast_horizon=forecast_horizon, name=name, **kwargs)
        self.residual_hidden_units = residual_hidden_units
        self.dropout_rate = dropout_rate
        self.scale_activation = scale_activation
        self.bias_activation = bias_activation
        self.kernel_initializer = initializer
        self.bias_const = initializers.Constant(0.0)
        self.residual_mlp_dense1 = layers.Dense(
            residual_hidden_units,
            activation='relu',
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp_dense1"
        )
        # self.norm_context = tf.keras.layers.LayerNormalization(name=f"{self.name}_context_norm")
        self.residual_mlp_dropout = layers.Dropout(dropout_rate, name=f"{self.name}_mlp_drop")
        self.residual_mlp_dense2 = layers.Dense(
            1,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_mlp_dense2",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.scale_dense = layers.Dense(
            1,
            activation=scale_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=initializers.Constant(1.0),
            name=f"{self.name}_scale_dense",
            # kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.bias_dense = layers.Dense(
            1,
            activation=bias_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_const,
            name=f"{self.name}_bias_dense",
            # kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

    def call(self, context_vector, per_step_inputs, training=False):
        q_phys = per_step_inputs['q_phys_scaled']
        times = per_step_inputs['forecast_times']
        pi_last = per_step_inputs['pi_last_measured']
        features = tf.concat([
            tf.expand_dims(q_phys, -1),
            tf.expand_dims(times, -1),
            tf.tile(tf.expand_dims(pi_last, 1), [1, self.forecast_horizon, 1])
        ], -1)
        # context_vector = self.norm_context(context_vector)
        hidden = self.residual_mlp_dense1(features)
        hidden = self.residual_mlp_dropout(hidden, training=training)
        prelim = self.residual_mlp_dense2(hidden)
        scale_raw = self.scale_dense(context_vector)
        bias_raw = self.bias_dense(context_vector)
        scale = tf.tile(tf.expand_dims(scale_raw, 1), [1, self.forecast_horizon, 1])
        bias = tf.tile(tf.expand_dims(bias_raw, 1), [1, self.forecast_horizon, 1])
        adjusted = prelim * scale + bias
        return tf.squeeze(adjusted, -1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'residual_hidden_units': self.residual_hidden_units,
            'dropout_rate': self.dropout_rate,
            'scale_activation': self.scale_activation,
            'bias_activation': self.bias_activation,
            'initializer': self.kernel_initializer
        })
        return config




class ProbabilisticBiasScaleFuser(BaseContextFuser):
    """
    Fuser probabilístico que ajusta o baseline físico via α 
    e prevê distribuição Normal (μ e σ) condicionada ao contexto + per-step features.
    """
    def __init__(
        self,
        forecast_horizon: int,
        residual_hidden_units: int = 64,
        dropout_rate: float = 0.1,
        initializer: Union[str, initializers.Initializer] = "he_normal",
        name: str = "prob_bias_context_fuser",
        **kwargs
    ):
        super().__init__(forecast_horizon=forecast_horizon, name=name, **kwargs)
        self.residual_hidden_units = residual_hidden_units
        self.dropout_rate = dropout_rate
        self.initializer = initializers.get(initializer)

        # MLP para gerar mu_raw e log_sigma_raw
        self.dense1 = layers.Dense(
            residual_hidden_units,
            activation="relu",
            kernel_initializer=self.initializer,
            name=f"{self.name}_dense1"
        )
        self.dropout = layers.Dropout(dropout_rate, name=f"{self.name}_dropout")
        # cabeça μ
        self.mu_dense = layers.Dense(
            1,
            activation=None,
            kernel_initializer=self.initializer,
            name=f"{self.name}_mu_dense"
        )
        # cabeça log σ
        self.log_sigma_dense = layers.Dense(
            1,
            activation=None,
            kernel_initializer=self.initializer,
            name=f"{self.name}_logsigma_dense"
        )
        # bias treinável para logsigma (inicia em 0 → σ≈1)
        self.logsigma_bias = self.add_weight(
            name=f"{self.name}_logsigma_bias",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True
        )

    def call(self, context_vector, per_step_inputs, training=False):
        # per-step features
        q_phys = per_step_inputs["q_phys_scaled"][..., None]       # (B,H,1)
        times  = per_step_inputs["forecast_times"][..., None]     # (B,H,1)
        pi_last = per_step_inputs["pi_last_measured"]             # (B,1)
        # expand pi_last para (B,H,1)
        pi_rep = tf.tile(pi_last[:, None, :], [1, self.forecast_horizon, 1])

        # expande contexto para cada passo: (B,H,C)
        ctx_rep = tf.tile(context_vector[:, None, :], [1, self.forecast_horizon, 1])

        # concatena tudo: contexto + q_phys + time + pi_last
        x = tf.concat([ctx_rep, q_phys, times, pi_rep], axis=-1)  # (B,H,C+3)

        # MLP
        x = self.dense1(x)
        x = self.dropout(x, training=training)

        # previsões brutas
        mu_raw        = self.mu_dense(x)                       # (B,H,1)
        log_sigma_raw = self.log_sigma_dense(x) + self.logsigma_bias  # (B,H,1)

        # achata para (B,H)
        mu_raw        = tf.reshape(mu_raw,       (-1, self.forecast_horizon))
        log_sigma_raw = tf.reshape(log_sigma_raw,(-1, self.forecast_horizon))

        # empilha mu e log_sigma para o forecaster
        params = tf.stack([mu_raw, log_sigma_raw], axis=-1)   # (B,H,2)
        return params

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "residual_hidden_units": self.residual_hidden_units,
            "dropout_rate": self.dropout_rate,
            "initializer": self.initializer,
        })
        return cfg


'''
\section{Explainability and Interpretability of the \textit{TrendBlock} Architecture}

The \textit{TrendBlock}, integrated into our hybrid oil and gas forecasting model, represents a thoughtful intersection between classical statistical modeling and machine learning (ML). At its core, the TrendBlock estimates residual trends through a polynomial basis expansion:

\begin{equation}
    f_{\text{trend}}(t) = \sum_{i=0}^{d} c_i \, t^{i},
\end{equation}

where $t$ denotes the normalized time horizon and $c_i$ are polynomial coefficients learned via a dense neural layer. This design provokes an important theoretical question concerning model interpretability and explainability—particularly critical in industrial settings where transparency directly influences stakeholder trust.

\subsection{Interpretability of the TrendBlock}

Unlike black-box architectures (e.g., Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs), or Transformer-based attention mechanisms), the TrendBlock is inherently transparent due to its explicit polynomial structure. The interpretability arises from the following design considerations:

\begin{enumerate}
    \item \textbf{Fixed Polynomial Basis:} The TrendBlock employs a predefined polynomial basis ($\{1, t, t^2, \ldots, t^d\}$), meaning each term has clear and mathematically interpretable behavior. Stakeholders can directly associate polynomial terms with recognizable trend patterns, such as linear drift ($t$), quadratic curvature ($t^2$), or higher-order trends. Thus, the polynomial terms themselves are explicitly interpretable.

    \item \textbf{Explicit Coefficients as Learned Parameters:} The dense neural layer learns polynomial coefficients $c_i$ from historical data. While this introduces some complexity, the resulting parameters remain transparent, analogous to classical linear regression coefficients. These coefficients can be directly inspected, plotted, and related explicitly to physical or operational interpretations. This direct parameter inspection sharply contrasts with internal parameters in CNN or LSTM architectures, which are typically uninterpretable individually.

\end{enumerate}

Thus, the TrendBlock maintains interpretability under the assumption that stakeholders trust regression-like modeling. The learned polynomial coefficients directly map the model's predictions back to transparent, intuitive terms.

\subsection{Comparative Transparency with Black-Box Models}

The transparency of TrendBlock significantly surpasses traditional deep learning layers:

\begin{itemize}
    \item \textbf{CNNs} operate via learned convolutional filters, making their internal mechanics opaque; filter parameters cannot be easily translated into physical insights.
    
    \item \textbf{LSTMs} encode temporal dependencies in hidden states and nonlinear gating mechanisms, making it difficult or impossible to assign direct physical meanings to learned parameters.

    \item \textbf{Attention mechanisms}, while offering interpretability in terms of "importance," lack explicit parametric interpretability—they cannot straightforwardly map attention weights to meaningful physical trends.

\end{itemize}

In contrast, the TrendBlock's polynomial coefficients explicitly correspond to meaningful trend components. Thus, stakeholders familiar with polynomial fitting or regression methods gain immediate interpretability advantages.

\subsection{Implications for Stakeholder Trust}

In the oil and gas industry, where strategic decisions frequently involve significant financial and operational risk, interpretability translates directly into stakeholder confidence. The TrendBlock facilitates trust through the following mechanisms:

\begin{itemize}
    \item \textbf{Explicit parameter inspection:} Stakeholders—such as reservoir engineers, field operators, and financial managers—can directly interrogate and reason about polynomial parameters, building confidence through transparency.
    
    \item \textbf{Physical consistency:} Polynomial trends naturally reflect common production phenomena (e.g., exponential decline curves), aligning model outputs with domain knowledge and experience, thereby reinforcing stakeholder acceptance.
    
    \item \textbf{Diagnostic capability:} Deviations from expected polynomial forms can prompt insightful discussions, guiding stakeholders toward deeper physical or operational investigations.

\end{itemize}

\subsection{Broader Relevance as a Hybrid Philosophy}

The TrendBlock exemplifies a "hybrid philosophy," strategically combining the flexibility and predictive power of ML with classical statistical modeling rigor. By blending explicit parametric forms (polynomial trends) with ML adaptability (learned coefficients), it bridges the gap between purely data-driven approaches and classical domain expertise. 

Therefore, TrendBlock's design philosophy aligns strongly with the industrial need for transparent, interpretable, and physically grounded predictions, enhancing both stakeholder confidence and practical utility in oil and gas forecasting applications.

'''