# Standard library imports
import os

# Third-party imports
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Add,
    Activation,
    Bidirectional,
    Conv1D,
    Dense,
    LayerNormalization,
    LeakyReLU,
    ReLU,
    LSTM,
    MultiHeadAttention,
    Dropout
)
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics


def positional_encoding(inputs):
    # Número de posições e dimensão do modelo
    positions = tf.cast(tf.range(start=0, limit=inputs.shape[1], delta=1), dtype=tf.float32)
    d_model = tf.cast(inputs.shape[2], dtype=tf.float32)

    # Calcular os ângulos com todas as operações em float32
    angles = positions[:, tf.newaxis] / tf.math.pow(10000.0, (2 * (tf.cast(tf.range(int(d_model)) // 2, dtype=tf.float32))) / d_model)

    # Concatenar seno e cosseno para gerar o encoding
    encoding = tf.concat([tf.math.sin(angles), tf.math.cos(angles)], axis=-1)
    
    # Ajustar a dimensão de `encoding` para corresponder a `inputs` se necessário
    encoding = encoding[:, :int(d_model)]  # Garante que o encoding tenha a mesma dimensão que inputs

    # Expandir `encoding` para ter a mesma dimensão de batch que `inputs` e somar
    encoding = tf.expand_dims(encoding, axis=0)
    return inputs + encoding


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    """Encoder do Transformer para capturar padrões complexos em séries temporais."""
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, kernel_initializer=GlorotUniform())(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization()(out1)

    ffn_output = Dense(ff_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=HeNormal())(out1)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1], kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=HeNormal())(ffn_output)
    out2 = Add()([out1, ffn_output])
    return LayerNormalization()(out2)



# def create_model(input_shape, use_lstm=False, use_conv=True, use_transformer=False):
#     inputs = Input(shape=(input_shape.shape[1], input_shape.shape[2]))
    
#     x = inputs
    
#     if use_transformer:
#         x = positional_encoding(x)
#         x = transformer_encoder(x, head_size=16, num_heads=4, ff_dim=128)
        
#     if use_conv:
#         x = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer=HeNormal())(x)
#         x = LayerNormalization()(x)
#         x = Activation('relu')(x)

#         x = Conv1D(filters=32, kernel_size=3, padding='same', kernel_initializer=HeNormal())(x)
#         x = LayerNormalization()(x)
#         x = Activation('relu')(x)
        
#     if use_lstm:
#         x = Bidirectional(LSTM(units=32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2, kernel_initializer=GlorotUniform()))(x)
#         x = Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.3, recurrent_dropout=0.2, kernel_initializer=GlorotUniform()))(x)
#         x = LayerNormalization()(x)
    


#     # x = Dense(units=16, kernel_initializer=HeNormal())(x)
#     # x = LeakyReLU(alpha=0.01)(x)
#     x = Dense(units=16, kernel_initializer=HeNormal())(x)
#     x = LeakyReLU(alpha=0.01)(x)
#     outputs = Dense(units=1, kernel_initializer=HeNormal())(x)
    
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(loss='mae', optimizer=Adam(), metrics=[metrics.mean_absolute_error])
#     return model

def create_model(input_shape, use_lstm=False, use_conv=True, use_transformer=False):
    """
    Cria um modelo Keras que representa a arquitetura Model6_Transformer_Conv_BiLSTM
    com o perfil de hiperparâmetros hp_large_robust.

    A sequência é: Transformer -> Conv -> BiLSTM -> Dense -> Output.
    Os argumentos use_... são mantidos para compatibilidade, mas os padrões foram 
    ajustados para ativar todos os blocos necessários.
    """
    # A correção do input_shape que discutimos anteriormente
    inputs = Input(shape=(input_shape.shape[1], input_shape.shape[2]))
    x = inputs
    
    # --- Bloco Transformer ---
    # Ativado por padrão e com os parâmetros 'large_robust'
    if use_transformer:
        x = positional_encoding(x)
        x = transformer_encoder(
            x, 
            head_size=64,             # Padrão da arquitetura
            num_heads=8,              # ★ ALTERADO (de 4 para 8)
            ff_dim=256,               # ★ ALTERADO (de 128 para 256)
            dropout=0.4               # ★ ALTERADO (de 0.2 para 0.4)
        )
        
    # --- Bloco Convolucional ---
    # Ativado por padrão e com os parâmetros 'large_robust'
    if use_conv:
        # A configuração pedia apenas uma camada convolucional, vamos ajustar para isso.
        x = Conv1D(
            filters=64,              
            kernel_size=3,            
            padding='same', 
            kernel_initializer=HeNormal()
        )(x)
        x = LayerNormalization()(x)
        x = Activation('relu')(x)

        x = Conv1D(
            filters=32,              
            kernel_size=3,            
            padding='same', 
            kernel_initializer=HeNormal()
        )(x)
        x = LayerNormalization()(x)
        x = Activation('relu')(x)
        
    # --- Bloco BiLSTM ---
    # Ativado por padrão e com os parâmetros 'large_robust'
    if use_lstm:
        # A configuração pedia uma única camada BiLSTM.
        # `return_sequences` deve ser False porque a próxima camada é Dense.
        x = Bidirectional(LSTM(
            units=128,                # ★ ALTERADO (de 32 para 128)
            return_sequences=False,   # ★ ALTERADO (de True para False, crucial!)
            dropout=0.4,              # ★ ALTERADO (de 0.3 para 0.4)
            recurrent_dropout=0.4,    # ★ ALTERADO (de 0.2 para 0.4)
            kernel_initializer=GlorotUniform()
        ))(x)
        x = LayerNormalization()(x)

    
    # --- Blocos Finais (Dense/Output) ---
    # Mantidos como estavam na sua função original, pois correspondem à configuração.
    x = Dense(units=16, kernel_initializer=HeNormal())(x)
    x = LeakyReLU(alpha=0.01)(x)
    outputs = Dense(units=1, kernel_initializer=HeNormal())(x)
    
    # --- Compilação do Modelo ---
    model = Model(inputs=inputs, outputs=outputs)
    
    # Usa a taxa de aprendizado do perfil 'large_robust'
    optimizer = Adam()
    model.compile(loss='mae', optimizer=optimizer, metrics=[metrics.mean_absolute_error])
    
    return model