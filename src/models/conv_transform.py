import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Flatten, LayerNormalization, BatchNormalization,
    Activation, Dropout, LeakyReLU, Embedding, Add, MultiHeadAttention, Concatenate,
    RepeatVector, TimeDistributed, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.optimizers import Adam
from itertools import combinations
import numpy as np
from typing import Any, List, Dict, Tuple, Optional, Union

"""
- Applies embedding-based positional encoding by summing an index embedding with inputs.
- Processes the sequence via two Conv1D layers with a residual connection.
- Enhances features using a single transformer encoder block.
- Flattens the output to preserve the detailed spatial arrangement.
- Connects flattened features to dense layers for regression.
- Implements a seq_to_value approach for regression.
"""


def create_branch(input_shape: tuple) -> Model:
    """
    Creates a branch model for a given input shape, including positional encoding,
    convolutional layers, and a transformer encoder block.
    
    Parameters:
        input_shape: Tuple representing the shape (timesteps, features).
    
    Returns:
        A Keras Model representing the branch.
    """
    
    branch_input = Input(shape=input_shape)
    
    
    # Positional Encoding: use a simple embedding-based positional encoding.
    position_indices = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embedding = Embedding(
        input_dim=input_shape[0], output_dim=input_shape[1]
    )(position_indices)
    x = branch_input + position_embedding
    
    

    # Convolutional layers for local feature extraction.
    conv1 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    

    # Residual connection
    conv2 = Add()([conv1, conv2])
    
    

    # Transformer Encoder block.
    def transformer_encoder(inputs, num_heads=4, key_dim=64, dropout=0.1):
        attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, kernel_initializer=GlorotUniform()
        )(inputs, inputs)
        attn_output = Dropout(dropout)(attn_output)
        out1 = Add()([inputs, attn_output])
        out1 = LayerNormalization(epsilon=1e-6)(out1)
        ffn = Dense(key_dim * 4, activation='relu')(out1)
        ffn = Dense(inputs.shape[-1])(ffn)
        ffn = Dropout(dropout)(ffn)
        out2 = Add()([out1, ffn])
        return LayerNormalization(epsilon=1e-6)(out2)
    
    x = transformer_encoder(conv2)
    x = Flatten()(x)

    branch_model = Model(inputs=branch_input, outputs=x)
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

def create_model(
    input_shape: Optional[tuple] = None,
    input_shapes: Optional[List[tuple]] = None,
    trend_degree: int = 3  # degree for the polynomial trend block
) -> Model:
    """
    Creates a model that combines a nonlinear branch (using CNNs and a transformer encoder)
    with an explicit trend branch (TrendBlock). For the single-input case, the input is 
    defined as:
    
        inputs = Input(shape=(input_shape.shape[1], input_shape.shape[2]))
        x = create_branch((input_shape.shape[1], input_shape.shape[2]))(inputs)
    
    The nonlinear branch is further processed through a few Dense layers, and an explicit 
    trend forecast is computed using a TrendBlock fed by a global average pooling of the raw input.
    The final output is given by:
    
        outputs = Add()([nonlinear_output, trend_output])
    
    Parameters:
        input_shape: Tuple representing the input shape (timesteps, features) for single-input models.
        input_shapes: List of tuples representing input shapes for multi-branch models.
        trend_degree: Degree of the polynomial trend to be modeled.
    
    Returns:
        A compiled Keras Model.
    """
    if input_shapes is not None:
        # Multi-branch case: (this part can be adapted similarly to the single-input case)
        branch_inputs = []
        branch_nonlinear_outputs = []
        branch_trend_outputs = []
        for shape in input_shapes:
            # Build the nonlinear branch using create_branch (or your alternative branch builder)
            branch_model = create_branch(shape)
            branch_inputs.append(branch_model.input)
            branch_nonlinear_outputs.append(branch_model.output)
            # Build the trend branch: apply global average pooling to the raw input then TrendBlock.
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
        x = Dropout(0.2)(x)
        x = Dense(16, kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        nonlinear_output = Dense(1, kernel_initializer='he_normal')(x)
        
        combined_trend = tf.reduce_mean(combined_trend, axis=1, keepdims=True)
        
        # Final prediction: add the nonlinear output and the trend forecast.
        outputs = Add()([nonlinear_output, combined_trend])
        model = Model(inputs=branch_inputs, outputs=outputs)
        return model

    elif input_shape is not None:
       
        # Single-input case:
        timesteps, features = input_shape.shape[1], input_shape.shape[2]
        # Preserve the input definition.
        inputs = Input(shape=(timesteps, features))  # Ignore batch size
        # Nonlinear branch via create_branch.
        branch_out = create_branch((timesteps, features))(inputs)
        
        
        # Additional nonlinear processing.
        x = Dense(32, kernel_initializer='he_normal')(branch_out)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(16, kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        nonlinear_output = Dense(1, kernel_initializer='he_normal')(x)
        

        # Explicit trend branch:
        # Use GlobalAveragePooling1D on the raw input and feed through TrendBlock.
        pooled = GlobalAveragePooling1D()(inputs)
        trend_output = TrendBlock(degree=trend_degree, forecast_horizon=1)(pooled)
        
        # ---------------- Fusion: Combine Both Forecasts ----------------
        # Here you can implement any fusion logic; we use a simple concatenation followed by a Dense layer.
        combined_forecast = Concatenate()([trend_output, nonlinear_output])
        final_forecast = Dense(1, activation='linear')(combined_forecast)
        

        # Final prediction: sum of nonlinear output and trend output.
        # outputs = Add()([nonlinear_output, final_forecast])
        model = Model(inputs=inputs, outputs=final_forecast, name='Trend_Value')
        return model
    else:
        raise ValueError("Either 'input_shape' or 'input_shapes' must be provided.")
