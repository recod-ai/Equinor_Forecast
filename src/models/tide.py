# Standard library imports
import os

# Third-party imports
import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential, layers
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    GRU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable



@register_keras_serializable()
class MonteCarloDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
    
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class ResidualBlock(layers.Layer):
    def __init__(self, input_dim, output_dim, hidden_size, dropout, use_layer_norm, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        self.dense = tf.keras.Sequential([
            layers.Dense(hidden_size, activation='relu'),
            layers.Dense(output_dim),
            MonteCarloDropout(dropout)
        ])
        
        self.skip = layers.Dense(output_dim)
        self.layer_norm = layers.LayerNormalization() if use_layer_norm else None

    def call(self, inputs, training=None):
        residual = self.skip(inputs)
        x = self.dense(inputs)
        x += residual
        
        if self.layer_norm:
            x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
            'use_layer_norm': self.use_layer_norm
        })
        return config

@register_keras_serializable()
class TideModule(Model):
    def __init__(self, input_dim, output_dim, future_cov_dim, static_cov_dim, nr_params,
                 num_encoder_layers, num_decoder_layers, decoder_output_dim, hidden_size,
                 temporal_decoder_hidden, temporal_width_past, temporal_width_future,
                 use_layer_norm, dropout, temporal_hidden_size_past=None,
                 temporal_hidden_size_future=None, input_chunk_length=None,
                 output_chunk_length=None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = input_dim - output_dim - future_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.temporal_width_past = temporal_width_past
        self.temporal_width_future = temporal_width_future
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.temporal_hidden_size_past = temporal_hidden_size_past or hidden_size
        self.temporal_hidden_size_future = temporal_hidden_size_future or hidden_size

        # Covariate projections
        self.past_cov_proj = self._build_projection(self.past_cov_dim, temporal_width_past,
                                                  self.temporal_hidden_size_past)
        self.future_cov_proj = self._build_projection(future_cov_dim, temporal_width_future,
                                                    self.temporal_hidden_size_future)

        # Calculate encoder dimension
        encoder_dim = self._calculate_encoder_dim()
        
        # Encoder/Decoder stacks
        self.encoders = [ResidualBlock(
            encoder_dim if i==0 else hidden_size, hidden_size, hidden_size,
            dropout, use_layer_norm
        ) for i in range(num_encoder_layers)]

        self.decoders = [ResidualBlock(
            hidden_size, 
            (decoder_output_dim * output_chunk_length * nr_params) if i==num_decoder_layers-1 else hidden_size,
            hidden_size, dropout, use_layer_norm
        ) for i in range(num_decoder_layers)]

        # Temporal decoder
        decoder_input_dim = decoder_output_dim * nr_params
        if temporal_width_future and future_cov_dim:
            decoder_input_dim += temporal_width_future
        self.temp_decoder = ResidualBlock(
            decoder_input_dim, output_dim * nr_params,
            temporal_decoder_hidden, dropout, use_layer_norm
        )

        # Skip connection
        self.lookback_skip = layers.Dense(output_chunk_length * nr_params)

    def _build_projection(self, input_dim, output_dim, hidden_size):
        if input_dim and output_dim:
            return ResidualBlock(input_dim, output_dim, hidden_size,
                               self.dropout, self.use_layer_norm)
        return None

    def _calculate_encoder_dim(self):
        past_cov_dim = (self.input_chunk_length * self.temporal_width_past 
                       if self.past_cov_proj else 0)
        future_cov_dim = ((self.input_chunk_length + self.output_chunk_length) *
                         self.temporal_width_future if self.future_cov_proj else 0)
        return (self.input_chunk_length * self.output_dim +
               past_cov_dim + future_cov_dim + self.static_cov_dim)

    def call(self, inputs, training=None):
        x, future_cov, static_cov = inputs
        x_lookback = x[:, :, :self.output_dim]

        # Process covariates
        past_cov = self._process_past_cov(x)
        future_cov_proj = self._process_future_cov(x, future_cov)
        static_expanded = self._process_static_cov(static_cov)

        # Encode features
        encoded = self._encode_features(x_lookback, past_cov, future_cov_proj, static_expanded)
        
        # Process through encoder/decoder
        for enc in self.encoders:
            encoded = enc(encoded)
        decoded = encoded
        for dec in self.decoders:
            decoded = dec(decoded)
            
        # Temporal decoding
        temporal_out = self._temporal_decode(decoded, future_cov_proj)
        
        # Skip connection
        skip = self.lookback_skip(tf.transpose(x_lookback, [0,2,1]))
        skip = tf.reshape(skip, [-1, self.output_chunk_length, self.nr_params])
        
        return temporal_out + skip

    def _process_past_cov(self, x):
        if self.past_cov_proj:
            return self.past_cov_proj(x[:, :, self.output_dim:self.output_dim+self.past_cov_dim])
        return None

    def _process_future_cov(self, x, future_cov):
        if self.future_cov_proj:
            future_cat = tf.concat([x[:, :, -self.future_cov_dim:], future_cov], axis=1)
            return self.future_cov_proj(future_cat)
        return None

    def _process_static_cov(self, static_cov):
        if self.static_cov_dim:
            return tf.repeat(tf.expand_dims(static_cov, 1), self.input_chunk_length, axis=1)
        return None

    def _encode_features(self, lookback, past_cov, future_cov, static_cov):
        parts = []
        for part in [lookback, past_cov, future_cov, static_cov]:
            if part is not None:
                parts.append(tf.reshape(part, [-1, tf.reduce_prod(part.shape[1:])]))
        return tf.concat(parts, axis=1)

    def _temporal_decode(self, decoded, future_cov_proj):
        decoded = tf.reshape(decoded, [-1, self.output_chunk_length, self.decoder_output_dim * self.nr_params])
        if future_cov_proj is not None:
            decoded = tf.concat([decoded, future_cov_proj[:, -self.output_chunk_length:,:]], axis=-1)
        return self.temp_decoder(decoded)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'future_cov_dim': self.future_cov_dim,
            'static_cov_dim': self.static_cov_dim,
            'nr_params': self.nr_params,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'decoder_output_dim': self.decoder_output_dim,
            'hidden_size': self.hidden_size,
            'temporal_decoder_hidden': self.temporal_decoder_hidden,
            'temporal_width_past': self.temporal_width_past,
            'temporal_width_future': self.temporal_width_future,
            'use_layer_norm': self.use_layer_norm,
            'dropout': self.dropout,
            'temporal_hidden_size_past': self.temporal_hidden_size_past,
            'temporal_hidden_size_future': self.temporal_hidden_size_future,
            'input_chunk_length': self.input_chunk_length,
            'output_chunk_length': self.output_chunk_length
        })
        return config

@register_keras_serializable()
class TiDEModel(Model):
    def __init__(self, input_chunk_length, output_chunk_length, output_chunk_shift=0,
                 num_encoder_layers=1, num_decoder_layers=1, decoder_output_dim=16,
                 hidden_size=128, temporal_width_past=4, temporal_width_future=4,
                 temporal_hidden_size_past=None, temporal_hidden_size_future=None,
                 temporal_decoder_hidden=32, use_layer_norm=False, dropout=0.1,
                 use_static_covariates=True, **kwargs):
        super().__init__(**kwargs)
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.temporal_hidden_size_past = temporal_hidden_size_past or hidden_size
        self.temporal_hidden_size_future = temporal_hidden_size_future or hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_static_covariates = use_static_covariates
        
        self.core_model = TideModule(
            input_dim=None,  # Will be set in build
            output_dim=None,
            future_cov_dim=0,
            static_cov_dim=0,
            nr_params=1,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            decoder_output_dim=decoder_output_dim,
            hidden_size=hidden_size,
            temporal_decoder_hidden=temporal_decoder_hidden,
            temporal_width_past=temporal_width_past,
            temporal_width_future=temporal_width_future,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            temporal_hidden_size_past=temporal_hidden_size_past,
            temporal_hidden_size_future=temporal_hidden_size_future,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.core_model.input_dim = input_dim
        self.core_model.output_dim = 1  # Modify based on your target shape
        self.core_model.future_cov_dim = 0  # Set based on your covariates
        self.core_model.static_cov_dim = 0  # Set based on your static features
        super().build(input_shape)

    def call(self, inputs, training=None):
        main_input, future_cov, static_cov = inputs
        return self.core_model((main_input, future_cov, static_cov))

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_chunk_length': self.input_chunk_length,
            'output_chunk_length': self.output_chunk_length,
            'num_encoder_layers': self.core_model.num_encoder_layers,
            'num_decoder_layers': self.core_model.num_decoder_layers,
            'decoder_output_dim': self.core_model.decoder_output_dim,
            'hidden_size': self.core_model.hidden_size,
            'temporal_width_past': self.core_model.temporal_width_past,
            'temporal_width_future': self.core_model.temporal_width_future,
            'temporal_hidden_size_past': self.temporal_hidden_size_past,
            'temporal_hidden_size_future': self.temporal_hidden_size_future,
            'temporal_decoder_hidden': self.temporal_decoder_hidden,
            'use_layer_norm': self.core_model.use_layer_norm,
            'dropout': self.core_model.dropout,
            'use_static_covariates': self.use_static_covariates
        })
        return config
    

def create_model(input_shape, use_lstm=False, use_conv=False, use_transformer=False):
    # TiDE configuration (adjust parameters as needed)
    input_chunk_length = input_shape.shape[1]
    output_chunk_length = input_shape.shape[1]  # Predict same length as input
    output_dim = 1  # Regression output dimension
    input_dim = input_shape.shape[2]  # Total features in input
    past_cov_dim = input_dim - output_dim  # Assume first feature is target
    
    # Dummy future and static cov dimensions for compatibility
    future_cov_dim = 0
    static_cov_dim = 0
    
    # Define the main input layer
    main_input = Input(shape=(input_chunk_length, input_dim), name='main_input')
    
    # Dummy future covariates (shape: [batch, output_chunk_length, 0])
    future_cov = tf.zeros(shape=(tf.shape(main_input)[0], output_chunk_length, 0))
    # Dummy static covariates (shape: [batch, 0])
    static_cov = tf.zeros(shape=(tf.shape(main_input)[0], 0))
    
    # Initialize TiDE module
    tide_module = TideModule(
        input_dim=input_dim,
        output_dim=output_dim,
        future_cov_dim=future_cov_dim,
        static_cov_dim=static_cov_dim,
        nr_params=1,  # MAE loss (no likelihood)
        num_encoder_layers=1,
        num_decoder_layers=1,
        decoder_output_dim=16,
        hidden_size=128,
        temporal_decoder_hidden=32,
        temporal_width_past=4,
        temporal_width_future=4,
        use_layer_norm=False,
        dropout=0.1,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length
    )
    
    # Get predictions (auto-handles dummy covariates)
    tide_output = tide_module((main_input, future_cov, static_cov))
    # Squeeze unnecessary dimensions (output_dim=1, nr_params=1)
    predictions = tf.squeeze(tide_output, axis=[-1, -2])
    
    # Create and compile model
    model = Model(inputs=main_input, outputs=predictions)
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(), metrics=['mae'])
    
    return model