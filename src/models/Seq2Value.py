"""
- Adds learned positional encoding to the input sequence.
- Extracts local features with a two‐layer Conv1D block and residual connection.
- Applies stacked transformer encoder blocks to capture long-range dependencies.
- Uses global average pooling to aggregate features into a fixed-length vector.
- Feeds the pooled representation to dense layers for regression.
- Designed for seq_to_value regression tasks.
"""



import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, BatchNormalization, Activation, Add, Dropout,
    GlobalAveragePooling1D, Concatenate, LayerNormalization, Embedding, LeakyReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from typing import List, Optional

from models.filter_layers import WaveletDenoiseLayer


# --- Positional Encoding Layer with get_config() ---
class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        # Learnable positional embeddings
        self.pos_embedding = Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, inputs):
        # Create position indices for the sequence
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        pos_encoding = self.pos_embedding(positions)
        # Expand dims so that the encoding can be added to the batch of inputs
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        return inputs + pos_encoding

    def get_config(self):
        config = super(PositionalEncodingLayer, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model
        })
        return config

# --- Transformer Encoder Block ---
def transformer_encoder_block(inputs, num_heads=4, key_dim=64, ff_dim=256, dropout_rate=0.1):
    # Self-attention layer
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, kernel_initializer=GlorotUniform()
    )(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)
    
    # Feed-forward network
    ffn = Dense(ff_dim, activation='relu', kernel_initializer=HeNormal())(out1)
    ffn = Dense(inputs.shape[-1], kernel_initializer=HeNormal())(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    out2 = Add()([out1, ffn])
    return LayerNormalization(epsilon=1e-6)(out2)

# --- Model 3 Branch Builder (Nonlinear branch) ---
def create_model_3(input_shape, 
                     num_transformer_blocks=1, 
                     conv_filters=32, 
                     kernel_size=3, 
                     dropout_rate=0.1):
    """
    Builds the nonlinear branch for time series regression using:
      - Learned positional encoding,
      - A convolutional block with residual connection,
      - Stacked transformer encoder blocks,
      - Global average pooling.
    
    Parameters:
      input_shape: Tuple (timesteps, features)
      num_transformer_blocks: Number of transformer blocks.
      conv_filters: Number of filters in Conv1D layers.
      kernel_size: Kernel size for Conv1D layers.
      dropout_rate: Dropout rate.
    
    Returns:
      A Keras Model representing this branch.
    """
    timesteps, features = input_shape
    branch_input = Input(shape=input_shape)

    # Learned positional encoding
    x = PositionalEncodingLayer(sequence_length=timesteps, d_model=features)(branch_input)

    # --- Convolutional Block with Residual Connection ---
    conv1 = Conv1D(filters=conv_filters, kernel_size=kernel_size,
                   padding='same', kernel_initializer=HeNormal())(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    conv2 = Conv1D(filters=conv_filters, kernel_size=kernel_size,
                   padding='same', kernel_initializer=HeNormal())(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    # Residual connection (project x if needed)
    if x.shape[-1] != conv_filters:
        x_proj = Conv1D(filters=conv_filters, kernel_size=1, padding='same')(x)
    else:
        x_proj = x
    x_res = Add()([x_proj, conv2])
    x_res = Activation('relu')(x_res)

    # --- Stacked Transformer Encoder Blocks ---
    for _ in range(num_transformer_blocks):
        x_res = transformer_encoder_block(x_res, num_heads=4, key_dim=conv_filters,
                                           ff_dim=conv_filters * 4, dropout_rate=dropout_rate)
    
    # Global average pooling to obtain a fixed-length vector
    x_pool = GlobalAveragePooling1D()(x_res)
    
    branch_model = Model(inputs=branch_input, outputs=x_pool)
    return branch_model

# --- TrendBlock ---
class TrendBlock(tf.keras.layers.Layer):
    """
    A trend block inspired by N-BEATS that explicitly models the trend component
    by predicting coefficients for a fixed polynomial basis.
    
    Parameters:
      degree: Degree of the polynomial (e.g., degree=2 for quadratic trend).
      forecast_horizon: Number of forecast steps (typically 1 for scalar regression).
    """
    def __init__(self, degree: int = 3, forecast_horizon: int = 1, **kwargs):
        super(TrendBlock, self).__init__(**kwargs)
        self.degree = degree
        self.forecast_horizon = forecast_horizon

    def build(self, input_shape):
        # Fully connected layer to compute theta (polynomial coefficients)
        self.theta_layer = Dense(self.degree + 1, kernel_initializer=HeNormal())
        # Create fixed polynomial basis.
        # Normalize time in [0,1] over the forecast horizon.
        t = np.linspace(0, 1, self.forecast_horizon)
        # Create basis matrix of shape (forecast_horizon, degree+1)
        basis = np.vstack([t**i for i in range(self.degree + 1)]).T  # shape: (H, degree+1)
        self.basis = tf.constant(basis, dtype=tf.float32)  # fixed basis

    def call(self, inputs):
        # inputs: a global summary of the raw input (e.g., after GlobalAveragePooling1D)
        theta = self.theta_layer(inputs)  # shape: (batch, degree+1)
        # Multiply by basis to get forecast; result shape: (batch, forecast_horizon)
        forecast = tf.matmul(theta, self.basis, transpose_b=True)
        return forecast

    def get_config(self):
        config = super(TrendBlock, self).get_config()
        config.update({
            "degree": self.degree,
            "forecast_horizon": self.forecast_horizon,
        })
        return config

# --- Full Model with Advanced Extrapolation Features ---
def create_model(
    input_shape: Optional[tuple] = None,
    input_shapes: Optional[List[tuple]] = None,
    trend_degree: int = 3  # degree for the polynomial trend block
) -> Model:
    """
    Creates a hybrid model that combines a nonlinear branch (with CNNs and transformers)
    and an explicit trend block (TrendBlock) to model trend extrapolation reliably.
    
    The design is inspired by hybrid forecasting models and N‑BEATS, which decompose the 
    signal into interpretable components.
    
    Parameters:
        input_shape: Tuple representing the input shape (timesteps, features) for single-input models.
        input_shapes: List of tuples for multi-branch models.
        trend_degree: Degree of the polynomial trend to be modeled.
    
    Returns:
        A compiled Keras Model.
    
    Raises:
        ValueError if neither 'input_shape' nor 'input_shapes' is provided.
    """
    if input_shapes is not None:
        # Multi-branch case:
        branch_inputs = []
        branch_nonlinear_outputs = []
        branch_trend_outputs = []
        for shape in input_shapes:
            # Build the nonlinear branch
            branch_model = create_model_3(shape)
            branch_inputs.append(branch_model.input)
            branch_nonlinear_outputs.append(branch_model.output)
            # Build the trend branch using a global average pooling then TrendBlock.
            pooled = GlobalAveragePooling1D()(branch_model.input)
            branch_trend_outputs.append(TrendBlock(degree=trend_degree, forecast_horizon=1)(pooled))
        
        # Combine outputs across branches.
        if len(branch_nonlinear_outputs) > 1:
            combined_nonlinear = Concatenate()(branch_nonlinear_outputs)
            combined_trend = Concatenate()(branch_trend_outputs)
            
        else:
            combined_nonlinear = branch_nonlinear_outputs[0]
            combined_trend = branch_trend_outputs[0]
        
        # Additional nonlinear processing:
        x = Dense(32, kernel_initializer='he_normal')(combined_nonlinear)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = Dense(16, kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        nonlinear_output = Dense(1, kernel_initializer='he_normal')(x)
        
        combined_trend = tf.reduce_mean(combined_trend, axis=1, keepdims=True)
        
        # Final prediction: sum the nonlinear forecast and the explicit trend forecast.
        outputs = Add()([nonlinear_output, combined_trend])
        model = Model(inputs=branch_inputs, outputs=outputs)
        return model

    elif input_shape is not None:
        # Single-input case:
        timesteps, features = input_shape.shape[1], input_shape.shape[2]
        inputs = Input(shape=(timesteps, features))  # Ignore batch size
        
        # Nonlinear branch (CNNs + transformers)
        nonlinear_branch = create_model_3((timesteps, features))
        x_nonlinear = nonlinear_branch(inputs)
        
        # Additional nonlinear processing
        x = Dense(8, kernel_initializer='he_normal')(x_nonlinear)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        nonlinear_output = Dense(1, kernel_initializer='he_normal')(x)
        
        # Explicit trend branch: pool the raw input and feed through TrendBlock.
        pooled = GlobalAveragePooling1D()(inputs)
        trend_output = TrendBlock(degree=trend_degree, forecast_horizon=1)(pooled)
        
        # ---------------- Fusion: Combine Both Forecasts ----------------
        combined_forecast = Concatenate()([nonlinear_output, trend_output])
        final_forecast = Dense(1, activation='linear')(combined_forecast)
        
        model = Model(inputs=inputs, outputs=final_forecast)
        # model.summary()
        return model
    else:
        raise ValueError("Either 'input_shape' or 'input_shapes' must be provided.")
