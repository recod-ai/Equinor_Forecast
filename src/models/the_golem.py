# Standard library imports
import os

# Third-party imports
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    GRU,
    LSTM,
    TimeDistributed
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics



#THE GOLEM
def create_model(
    input_shape,
    model_type='GRU',            # 'GRU', 'BI_GRU', 'LSTM', 'BI_LSTM'
    distributed=True,           # Se True, usa TimeDistributed nos Dense
    dense_layers=1,              # Número de camadas densas
    future_step=1,               # Quantos passos preditivos no output
    dropout_rate=0.5,            # Dropout padrão p/ GRU, BI_GRU, BI_LSTM
    rnn_units=128,              # Tamanho padrão do snippet
    seed=42,                     # Fixar semente para reprodutibilidade
):
    """
    Adaptação do seu modelo Sequencial para a API Funcional, seguindo estrutura 'canônica'.
    """
    # 1) Semente para reprodutibilidade
    tf.random.set_seed(seed)
    
    # 2) Definir camada de entrada
    #    Supondo que input_shape seja do tipo (batch, timesteps, n_var) ou similar
    #    e que input_shape[1] = histórico (params['historic_step'])
    #    e input_shape[2] = n_var
    
    inputs = Input(shape=(input_shape.shape[1], input_shape.shape[2]))
    x = inputs

    # ---------------------------------------------------------------------
    # 3) RNN de acordo com model_type
    # ---------------------------------------------------------------------
    if model_type == 'GRU':
        # 3 camadas GRU unidirecionais
        x = GRU(rnn_units, return_sequences=True)(x)   # 1ª
        x = Dropout(dropout_rate)(x)
        
        x = GRU(rnn_units, return_sequences=True)(x)   # 2ª
        x = Dropout(dropout_rate)(x)
        
        # 3ª depende de 'distributed'
        if distributed:
            x = GRU(rnn_units, return_sequences=True)(x)
        else:
            x = GRU(rnn_units, return_sequences=False)(x)
        x = Dropout(dropout_rate)(x)

    elif model_type == 'BI_GRU':
        # 2 camadas GRU bidirecionais
        x = Bidirectional(
                GRU(rnn_units, return_sequences=True)
            )(x)
        x = Dropout(dropout_rate)(x)

        # 2ª camada depende de 'distributed'
        if distributed:
            x = Bidirectional(GRU(rnn_units, return_sequences=True))(x)
        else:
            x = Bidirectional(GRU(rnn_units, return_sequences=False))(x)

    elif model_type == 'LSTM':
        # 2 camadas LSTM unidirecionais, com dropout 0.3 na primeira
        x = LSTM(rnn_units, return_sequences=True)(x)  # 1ª
        x = Dropout(0.3)(x)  # fixo em 0.3, conforme seu snippet original

        # 2ª depende de 'distributed'
        if distributed:
            x = LSTM(rnn_units, return_sequences=True)(x)
        else:
            x = LSTM(rnn_units, return_sequences=False)(x)

    elif model_type == 'BI_LSTM':
        # 2 camadas LSTM bidirecionais
        x = Bidirectional(
                LSTM(rnn_units, return_sequences=True)
            )(x)
        x = Dropout(dropout_rate)(x)

        # 2ª depende de 'distributed'
        if distributed:
            x = Bidirectional(LSTM(rnn_units, return_sequences=True))(x)
        else:
            x = Bidirectional(LSTM(rnn_units, return_sequences=False))(x)
    else:
        raise ValueError("Escolha 'GRU', 'BI_GRU', 'LSTM' ou 'BI_LSTM'.")

    # ---------------------------------------------------------------------
    # 4) Camadas densas (TimeDistributed ou não)
    #    Mesmo cálculo do snippet original
    # ---------------------------------------------------------------------
    interval = (rnn_units - future_step) // dense_layers

    if distributed:
        # Quando distributed=True, aplicamos TimeDistributed
        for i in range(1, dense_layers):
            n_units = rnn_units - (i * interval)
            x = TimeDistributed(Dense(n_units))(x)

        outputs = TimeDistributed(Dense(future_step))(x)

    else:
        # denso "normal"
        for i in range(1, dense_layers):
            n_units = rnn_units - (i * interval)
            x = Dense(n_units)(x)

        outputs = Dense(future_step)(x)

    # ---------------------------------------------------------------------
    # 5) Definir e compilar o modelo
    # ---------------------------------------------------------------------
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(),
        loss=Huber(delta=1.0),
        metrics=[
            metrics.mean_squared_error,
            metrics.mean_absolute_error,
            metrics.categorical_accuracy
        ]
    )

    return model












