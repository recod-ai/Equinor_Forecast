# Arquivo: src/architecture_builder.py

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D, LSTM, Bidirectional, Dense, LayerNormalization, 
    Activation, LeakyReLU, Add, Dropout, MultiHeadAttention
)
from tensorflow.keras.initializers import GlorotUniform, HeNormal

# Importa os blocos de construção existentes
from models.generic_configurable import positional_encoding, transformer_encoder

class ArchitectureBuilder:
    """
    Constrói um modelo Keras dinamicamente a partir de uma configuração
    de arquitetura baseada em blocos.
    """
    def __init__(self, arch_config: dict, hp_config: dict = None):
        """
        Inicializa o builder com as configurações da arquitetura e hiperparâmetros.

        Args:
            arch_config (dict): A configuração da arquitetura (um item da lista de architectures.yaml).
            hp_config (dict, optional): O perfil de hiperparâmetros para sobrescrever os padrões.
        """
        self.arch_config = arch_config
        self.hp_config = hp_config or {}
        
        # Mapeia os tipos de bloco para os métodos de construção
        self.block_registry = {
            "conv": self._add_conv_block,
            "bilstm": self._add_bilstm_block,
            "transformer": self._add_transformer_block,
            "dense": self._add_dense_block,
            "output": self._add_output_block,
        }

    def _get_params(self, block_type: str, default_params: dict) -> dict:
        """
        Obtém os parâmetros para um bloco, mesclando padrões com overrides de hiperparâmetros.
        A ordem de precedência é: hp_config -> default_params
        """
        params = default_params.copy()
        hp_overrides = self.hp_config.get("overrides", {})
        
        # Aplica overrides globais de HP (ex: dropout geral)
        if 'dropout' in hp_overrides:
            if 'dropout' in params:
                params['dropout'] = hp_overrides['dropout']

        # Aplica overrides específicos do bloco (ex: hp_config.overrides.conv.filters)
        if block_type in hp_overrides:
            params.update(hp_overrides[block_type])
            
        return params

    def build(self, input_shape: tuple) -> Model:
        """
        Constrói e compila o modelo Keras.

        Args:
            input_shape (tuple): O formato dos dados de entrada (ex: (timesteps, features)).

        Returns:
            Um modelo Keras compilado.
        """
        inputs = Input(shape=(input_shape.shape[1], input_shape.shape[2]))
        x = inputs

        for block in self.arch_config["blocks"]:
            block_type = block["type"]
            if block_type not in self.block_registry:
                raise ValueError(f"Tipo de bloco desconhecido: '{block_type}'")
            
            # Chama o método de construção correspondente
            build_fn = self.block_registry[block_type]
            x = build_fn(x, block["params"])

        model = Model(inputs=inputs, outputs=x)
        
        # Compila o modelo usando a taxa de aprendizado dos hiperparâmetros, se disponível
        learning_rate = self.hp_config.get("overrides", {}).get("learning_rate", 0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mae', optimizer=optimizer, metrics=[tf.keras.metrics.mean_absolute_error])
        
        return model

    # --- Métodos Privados para Construção de Blocos ---

    def _add_conv_block(self, x, params: dict):
        p = self._get_params("conv", params)
        x = Conv1D(filters=p['filters'], kernel_size=p['kernel_size'], padding=p['padding'], kernel_initializer=HeNormal())(x)
        x = LayerNormalization()(x)
        x = Activation('relu')(x)
        return x

    def _add_bilstm_block(self, x, params: dict):
        p = self._get_params("bilstm", params)
        # return_sequences deve ser True se houver outra camada recorrente/conv depois
        # A configuração YAML deve cuidar disso.
        x = Bidirectional(LSTM(
            units=p['units'], 
            return_sequences=p.get('return_sequences', False), # Padrão para False
            dropout=p.get('dropout', 0.0),
            recurrent_dropout=p.get('recurrent_dropout', 0.0),
            kernel_initializer=GlorotUniform()
        ))(x)
        return x

    def _add_transformer_block(self, x, params: dict):
        p = self._get_params("transformer", params)
        x = positional_encoding(x)
        x = transformer_encoder(x, head_size=p['head_size'], num_heads=p['num_heads'], ff_dim=p['ff_dim'], dropout=p.get('dropout', 0.0))
        return x

    def _add_dense_block(self, x, params: dict):
        p = self._get_params("dense", params)
        x = Dense(units=p['units'], kernel_initializer=HeNormal())(x)
        x = LeakyReLU(alpha=0.01)(x)
        return x

    def _add_output_block(self, x, params: dict):
        p = self._get_params("output", params)
        # A camada de saída geralmente não é afetada por overrides de HP, mas mantemos o padrão
        x = Dense(units=p['units'], kernel_initializer=HeNormal())(x)
        return x